import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.keras.engine import training
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
import glob
import time
import pickle as pkl
import argparse
import copy
import random 
import importlib
import sys
import functools
import json
import imageio
from PIL import Image
import larq as lq
import larq_zoo as lqz
import larq_compute_engine as lce
import random
from sklearn.utils import shuffle
from scipy import interpolate

module_path = os.path.abspath('C:\Proj\phd\core50_evaluation')
if module_path not in sys.path:
    sys.path.append(module_path)
    
import BNF as bnf
import Utilities as uti
import VAL_Quantization as quant

#tf.random.set_seed(42)
tf.random.set_seed(105)

from sklearn.model_selection import train_test_split

def GetIndexesClass(labels, cls_idx):
    return np.where(labels == cls_idx)[0]

def SplitDatasetInExperienceNC(images, labels, n_experiences):
    
    values, counts = np.unique(labels, return_counts=True)
    
    indexes_for_class = []
    
    for item in values:
        indexes_for_class.append(GetIndexesClass(labels, item))
        
    samples_class_per_experience = int(np.min(counts))
    
    experiences = []
    experiences_gt = []
    
    np.random.seed(42)
    
    # Shuffle images belonging to the same class
    indexes_per_class_shuffled = []
    for idx, item in enumerate(indexes_for_class):
        indexes_per_class_shuffled.append(np.random.choice(item, size=item.shape[0], replace=False))
    
    # Select which classes have to be selected for each experience
    classes_per_experience = np.array_split(np.random.choice(np.arange(len(counts)), size=10, replace=False), n_experiences)

    for idx in range(len(classes_per_experience)):
        
        curr_exp = np.empty(shape=[0,])
        curr_exp_gt = np.empty(shape=[0,])

        # loop on current experience
        for idx2 in range(len(classes_per_experience[idx])):

            curr_class = classes_per_experience[idx][idx2]

            curr_exp = np.concatenate((curr_exp, indexes_per_class_shuffled[curr_class]), axis=0)

            curr_gt = np.full((samples_class_per_experience,), curr_class)
            curr_exp_gt = np.concatenate((curr_exp_gt, curr_gt))

        experiences.append(images[curr_exp.astype(int),:])
        experiences_gt.append(curr_exp_gt.astype(int))
        
    return experiences, experiences_gt

from sklearn.utils import shuffle

def SplitDatasetInExperienceNI(images, labels, n_experiences):
    
    values, counts = np.unique(labels, return_counts=True)
    
    indexes_for_class = []
    
    for item in values:
        indexes_for_class.append(GetIndexesClass(labels, item))
        
    samples_class_per_experience = int(np.min(counts) / n_experiences)
    
    experiences = []
    experiences_gt = []
    
    #np.random.seed(42)
    np.random.seed(105)
    
    for i in range(n_experiences):
        
        curr_exp = np.empty(shape=[0,])
        curr_exp_gt = np.empty(shape=[0,])
        
        for idx in range(len(indexes_for_class)):
        
            curr_indexes = np.random.choice(indexes_for_class[idx], samples_class_per_experience, replace=False)
            curr_exp = np.concatenate((curr_exp, curr_indexes), axis=0)
            indexes_for_class[idx] = np.array([i for i in indexes_for_class[idx] if i not in curr_indexes])
            
            curr_gt = np.full((samples_class_per_experience,), idx)
            curr_exp_gt = np.concatenate((curr_exp_gt, curr_gt))
        
        curr_exp, curr_exp_gt = shuffle(curr_exp, curr_exp_gt, random_state=42)
        
        experiences.append(images[curr_exp.astype(int),:])
        experiences_gt.append(curr_exp_gt.astype(int))
        
    return experiences, experiences_gt

def GetNIDataset(num_exp=10, batch_size=50):
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
    test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")
    
    exp_images, exp_gt = SplitDatasetInExperienceNI(train_images, train_labels, num_exp)
    
    train_ds = []
    
    #test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    val_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    def normalization(images, labels):
    
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        images = (images / 127.5) - 1
        return images, labels

    def augmentation(images, labels):

        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        images = tf.image.resize_with_crop_or_pad(images, 40, 40)
        images = tf.image.random_crop(images, [32, 32, 3])
        images = tf.image.random_flip_left_right(images)
        images = (images / 127.5) - 1.0
        return images, labels
    
    for i in range(num_exp):
        #curr_gt = tf.keras.utils.to_categorical(exp_gt[i], 10)
        curr_gt = exp_gt[i]
        train_ds.append(tf.data.Dataset.from_tensor_slices((exp_images[i], curr_gt)).shuffle(exp_gt[i].shape[0], reshuffle_each_iteration=True))
        
        train_ds[i] = train_ds[i].map(lambda x,y: tf.numpy_function(func=augmentation, inp=[x, y], Tout=[tf.float32, tf.float32]))
        train_ds[i] = train_ds[i].batch(batch_size, drop_remainder=True)
        train_ds[i] = train_ds[i].prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
    val_ds = val_ds.map(normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)    
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_ds, val_ds

def GetNCDataset(num_exp=5, batch_size=50):
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
    test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")
    
    exp_images, exp_gt = SplitDatasetInExperienceNC(train_images, train_labels, num_exp)
    
    train_ds = []
    
    #test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    val_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    def normalization(images, labels):
    
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        images = (images / 127.5) - 1
        return images, labels

    def augmentation(images, labels):

        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        images = tf.image.resize_with_crop_or_pad(images, 40, 40)
        images = tf.image.random_crop(images, [32, 32, 3])
        images = tf.image.random_flip_left_right(images)
        images = (images / 127.5) - 1.0
        return images, labels
    
    for i in range(num_exp):
        #curr_gt = tf.keras.utils.to_categorical(exp_gt[i], 10)
        curr_gt = exp_gt[i]
        train_ds.append(tf.data.Dataset.from_tensor_slices((exp_images[i], curr_gt)).shuffle(exp_gt[i].shape[0], reshuffle_each_iteration=True))
        
        train_ds[i] = train_ds[i].map(lambda x,y: tf.numpy_function(func=augmentation, inp=[x, y], Tout=[tf.float32, tf.float32]))
        train_ds[i] = train_ds[i].batch(batch_size, drop_remainder=True)
        train_ds[i] = train_ds[i].prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
    val_ds = val_ds.map(normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)    
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_ds, val_ds

from collections import namedtuple

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point', 'num_bits'])

# x tensor if in float representation
def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    qmin = tf.constant(0.)
    qmax = tf.cast(tf.pow(tf.constant(2), tf.convert_to_tensor(num_bits)) - tf.constant(1), tf.float32)
    
    if (min_val is None) and (max_val is None): 
        min_val, max_val = tf.math.reduce_min(x, axis=None), tf.math.reduce_max(x, axis=None)

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = tf.clip_by_value(tf.math.round(qmin - min_val / scale), qmin, qmax)

    q_x = tf.round(tf.clip_by_value(zero_point + x / scale, qmin, qmax))
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point, num_bits=num_bits)


def dequantize_tensor(q_x):
    return q_x.scale * (tf.cast(q_x.tensor, dtype=tf.float32) - q_x.zero_point)

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
    
    # Calc Scale and zero point of next 
    qmin = tf.constant(0.)
    qmax = tf.cast(tf.pow(tf.constant(2), tf.convert_to_tensor(num_bits)) - tf.constant(1), tf.float32)

    scale_next = (max_val - min_val) / (qmax - qmin)
    zero_point_next = tf.clip_by_value(tf.math.round(qmin - min_val / scale_next), qmin, qmax)

    return scale_next, zero_point_next

def multiply_quantized_tensors(input1, input2, stat_output, transpose1=True):
    
    assert(input1.num_bits == input2.num_bits)
    
    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat_output['min'], max_val=stat_output['max'])
  
    output = (input1.scale*input2.scale/scale_next)*tf.linalg.matmul(input1.tensor-input1.zero_point, input2.tensor-input2.zero_point, transpose_a=transpose1) + zero_point_next
    return QTensor(tensor=output, scale=scale_next, zero_point=zero_point_next, num_bits=input1.num_bits)

def quantizeLayer(x, layer, stat, scale_x, zp_x, num_bits=8):
    
    index_for_weights = 0
    W = tf.Variable(layer.weights[index_for_weights])
    #W = layer.weights[index_for_weights]
    
    if (layer.use_bias == True):
        #B = layer.weights[1]
        B = tf.Variable(layer.weights[1])

    tmp = tf.reduce_min(layer.weights[index_for_weights])
    tmp2 = tf.reduce_max(layer.weights[index_for_weights])
    if(tmp == tmp2) and (tmp == 0.0):
        w = QTensor(tensor=layer.weights[index_for_weights], scale=1.0, zero_point=0.0, num_bits=8)
    else:
        w = quantize_tensor(layer.weights[index_for_weights], num_bits=num_bits) 
    
    if (layer.use_bias == True):
        b = quantize_tensor(layer.weights[1], num_bits=num_bits)

    layer.weights[index_for_weights].assign(tf.cast(w.tensor, dtype=tf.float32))
    
    if (layer.use_bias == True):
        layer.weights[1].assign(tf.cast(b.tensor, dtype=tf.float32))

    ####################################################################
    # This is Quantisation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    scale_w = w.scale
    zp_w = w.zero_point
    
    if (layer.use_bias == True):
        scale_b = b.scale
        zp_b = b.zero_point
    

    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

    # Preparing input by shifting
    X = tf.cast(x, dtype=tf.float32) - zp_x
    layer.weights[index_for_weights].assign((scale_x * scale_w/scale_next)*(layer.weights[index_for_weights] - zp_w))
    if (layer.use_bias == True):
        layer.weights[1].assign((scale_b/scale_next)*(layer.weights[1] + zp_b))
        
    # All int
    x_inter = layer(X) + zero_point_next
    x = tf.keras.layers.Softmax()(x_inter)

    # Reset
    layer.weights[0].assign(W)
    if (layer.use_bias == True):
        layer.weights[1].assign(B)
    
    return x, scale_next, zero_point_next

def updateStats(x, stats, key):
    batch_size = tf.cast(tf.shape(x)[0], dtype=tf.float32)
    
    max_val = tf.math.reduce_max(x, axis=None)
    min_val = tf.math.reduce_min(x, axis=None)
    
    
    if key not in stats:
        stats[key] = {"max": max_val, "min": min_val, "total": batch_size}
    else:
        stats[key]['max'] = tf.math.maximum(max_val, stats[key]['max'])
        stats[key]['min'] = tf.math.minimum(min_val, stats[key]['min'])
        stats[key]['total'] += batch_size
    
    return stats

from tensorflow.python.ops.gen_math_ops import mod

def get_train_step_function():
    
    @tf.function
    def train_step_func(model, images, labels, loss_object, optimizer, train_loss, train_loss_additional, train_accuracy, training_all, weight_update_quant_bits, all_other_quant_bits, gradient_error_metric):
        with tf.GradientTape() as tape:
            predictions = model(images, training=training_all)
            loss = loss_object(labels, predictions[0])
            main_loss = loss
            additional_loss = sum(model.losses)

            # Add custom losses of each layer
            loss += additional_loss
            #loss += tf.math.reduce_sum(tf.where(tf.math.is_nan(model.losses), tf.zeros_like(model.losses), model.losses))
        gradients = tape.gradient(loss, model.trainable_variables)
        #optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        optimizer.apply_gradients((grad, var) 
                                   for (grad, var) in zip(gradients, model.trainable_variables) 
                                   if grad is not None)

        # Apply constraints
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))

        train_loss(main_loss)
        train_loss_additional(additional_loss)
        for i in range(len(train_accuracy)):
            train_accuracy[i].update_state(labels, predictions[0])
    return train_step_func

def get_train_step_function_exp_bigger_1(quantized = True):
    
    @tf.function
    def quantize(x, min_val=None, max_val=None, num_bits=8):
        if min_val == None:
            min_val = tf.reduce_min(x)
        if max_val == None:
            max_val = tf.reduce_max(x)
        # In NC scenario the weights of CWR are always zero at the beginning
        if (min_val == max_val) and (min_val == 0.0):
            return x
        return bnf.standard_pow2_add_quantization_noise(x, min_val, max_val, quant_bits=num_bits)
    
    #@tf.function
    def stable_softmax(X):
        return tf.exp(X) / tf.reduce_sum(tf.exp(X), axis=-1, keepdims=True)
    
    @tf.function
    def stable_softmax_q(X):
        return quantize(tf.exp(quantize(X))) / quantize(tf.reduce_sum(tf.exp(quantize(X)), axis=-1, keepdims=True))

    # Labels require to be sparse
    #@tf.function
    def derivative_cross_entropy_softmax(logits, labels):
        batch_size = tf.shape(labels)[0]
        probs = stable_softmax(logits)
        labels = tf.cast(labels, tf.int32)
        repre = tf.cast(tf.one_hot(labels, depth=10), dtype=tf.float32)
        diff = tf.math.subtract(probs, repre)
        return tf.math.divide(diff, tf.cast(batch_size, tf.float32))
    
    @tf.function
    def derivative_cross_entropy_softmax_q(logits, labels):
        batch_size = tf.shape(labels)[0]
        probs = stable_softmax_q(logits)
        labels = tf.cast(labels, tf.int32)
        repre = tf.cast(tf.one_hot(labels, depth=10), dtype=tf.float32)
        diff = quantize(tf.math.subtract(probs, repre))
        return quantize(tf.math.divide(diff, tf.cast(batch_size, tf.float32)))
    
    #@tf.function
    def derivative_cross_entropy_softmax_q_total(logits, labels, quantized_probs):
        
        categorical_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=10, dtype=tf.float32)
        diff = tf.math.subtract(quantized_probs, categorical_labels)
        quantized_der = tf.math.divide(diff, tf.cast(tf.shape(labels)[0], tf.float32))
        return quantized_der

    #@tf.function
    def train_step_func_cwr(model, images, labels, loss_object, optimizer, train_loss, train_loss_additional, train_accuracy, statistics, training_all, weight_update_quant_bits, all_other_quant_bits, gradient_error_metric):
        
        predictions = model(images, training=training_all)
        loss = loss_object(labels, predictions[0])
        main_loss = loss

        derivative_softmax = derivative_cross_entropy_softmax(predictions[2], labels)

        gradients = tf.linalg.matmul(predictions[1], derivative_softmax, transpose_a=True)
        
        optimizer.apply_gradients((grad, var) 
                                   for (grad, var) in zip([gradients], model.trainable_variables) 
                                   if grad is not None)

        train_loss(main_loss)
        for i in range(len(train_accuracy)):
            train_accuracy[i].update_state(labels, predictions[0])

    #@tf.function
    def train_step_func_cwr_q(model, images, labels, loss_object, optimizer, train_loss, train_loss_additional, train_accuracy, statistics, training_all, weight_update_quant_bits, all_other_quant_bits, gradient_error_metric):
        
        # for NC scenario 16 bits are necessary
        num_quantization_bits = all_other_quant_bits#8
        
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions[0])
        
        # float computation
        derivative_softmax_f = derivative_cross_entropy_softmax(predictions[2], labels)
        gradients_f = tf.linalg.matmul(predictions[1], derivative_softmax_f, transpose_a=True)

        #derivative_softmax = quantize(derivative_cross_entropy_softmax_q(predictions[2], labels))
        #gradients = tf.linalg.matmul(quantize(predictions[1]), derivative_softmax, transpose_a=True)
        
        batch_size = tf.cast(tf.shape(labels)[0], tf.float32)
        lr = tf.keras.backend.get_value(optimizer.lr)
        categorical_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=10, dtype=tf.float32)
        
        # Necessary to quantized output activations
        #cwr_input = quantize_tensor(predictions[1], min_val=statistics['cwr_input']['min'], max_val=statistics['cwr_input']['max'], num_bits=num_quantization_bits)
        #quantized_probs, scale_next, zero_point_next = quantizeLayer(cwr_input.tensor, model.get_layer('cwr'), statistics['output'], cwr_input.scale, cwr_input.zero_point, num_bits=num_quantization_bits)
        
        w_min = tf.math.reduce_min(model.get_layer('cwr').weights[0], axis=None)
        w_max = tf.math.reduce_max(model.get_layer('cwr').weights[0], axis=None)
        margin = (w_max-w_min)*0.02
        curr_weights_fixed = quantize(model.get_layer('cwr').weights[0], w_min-margin, w_max+margin, num_bits=num_quantization_bits)
        
        # cwr quantized implementation
        cwr_input_q = quantize(predictions[1], min_val=statistics['cwr_input']['min'], max_val=statistics['cwr_input']['max'], num_bits=num_quantization_bits)
        temp = tf.linalg.matmul(cwr_input_q, curr_weights_fixed, transpose_a=False)
        tmp2 = quantize(tf.keras.layers.Softmax()(temp), 0.0, 1.0, num_bits=num_quantization_bits)
        
        probs_fixed_point = quantize(predictions[0], 0.0, 1.0, num_bits=num_quantization_bits)
        #quantized_derivative = tf.math.subtract(probs_fixed_point, categorical_labels) / batch_size
        #quantized_derivative = tf.math.subtract(quantized_probs, categorical_labels) / batch_size
        quantized_derivative = tf.math.subtract(tmp2, categorical_labels) / batch_size
        
        cwr_input_fixed = quantize(predictions[1], statistics['cwr_input']['min'], statistics['cwr_input']['max'], num_bits=num_quantization_bits)
        derivative_fixed = quantize(quantized_derivative, statistics['derivata_logits']['min'], statistics['derivata_logits']['max'], num_bits=num_quantization_bits)
                
        gradients = tf.linalg.matmul(cwr_input_fixed, derivative_fixed, transpose_a=True)
        new_cwr_weights = curr_weights_fixed - (lr*gradients)
        
        gradient_error_metric.update_state(gradients_f, gradients)
        
        #optimizer.apply_gradients((grad, var) 
        #                           for (grad, var) in zip([gradients], model.trainable_variables) 
        #                           if grad is not None)
        model.get_layer('cwr').weights[0].assign(quantize(new_cwr_weights, num_bits=weight_update_quant_bits))

        train_loss(loss)
        for i in range(len(train_accuracy)):
            train_accuracy[i].update_state(labels, predictions[0])
    
    if (quantized == True):
        # Final selection
        return train_step_func_cwr_q
    else:
        return train_step_func_cwr

def get_val_step_function():
    @tf.function
    def val_step_func(model, images, labels, loss_object, test_loss, test_loss_additional, test_accuracy):
        predictions = model(images)
        t_loss = loss_object(labels, predictions[0])
        t_loss_additional = sum(model.losses)

        test_loss(t_loss)
        test_loss_additional(t_loss_additional)
        for i in range(len(test_accuracy)):
            test_accuracy[i].update_state(labels, predictions[0])
    return val_step_func

def get_val_cwr_step_function():
    
    #@tf.function
    def val_step_func_quantized(model, images, labels, loss_object, test_loss, test_loss_additional, test_accuracy, statistics):
        
        predictions = model(images)
        
        cwr_input = quantize_tensor(predictions[1], min_val=statistics['cwr_input']['min'], max_val=statistics['cwr_input']['max'])
        quantized_probs, scale_next, zero_point_next = quantizeLayer(cwr_input.tensor, model.get_layer('cwr'), statistics['output'], cwr_input.scale, cwr_input.zero_point)
        
        t_loss = loss_object(labels, scale_next*(quantized_probs-zero_point_next))

        test_loss(t_loss)
        for i in range(len(test_accuracy)):
            test_accuracy[i].update_state(labels, scale_next*(quantized_probs-zero_point_next))
    return val_step_func_quantized

def get_run_logdir(add_info = None):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    if(add_info != None):
        run_id += add_info
        # os.path.realpath(__file__)
    return os.path.join(os.path.dirname(os.path.abspath('')) + os.path.sep + "tf_logs", run_id)

def tf_model_fit(model, train_ds, val_ds, epochs, 
                 train_metrics, val_metrics, 
                 loss_object, optimizer,
                 validation_frequency = 1,
                 train_all = True,
                 learning_rate_scheduler = None,
                 train_steps_epoch=-1,
                 val_steps_epoch=-1,
                 on_epoch_begin_callback=None,
                 add_tensorboard_info=False,
                 epochs_for_cwr_experiences=40,
                 is_cwr_quantized=True,
                 add_tensorboard_layer_inputs_outputs=False,
                 accuracy_to_monitor=None,
                 weight_update_quant_bits=16,
                 all_other_quant_bits=8,
                 before_training_experience=None,
                 gradient_error_metric=tf.keras.metrics.MeanAbsolutePercentageError(),
                 after_training_experience=None):
    
    experiences_training_loss = []
    experiences_validation_loss = []
    experiences_validation_accuracy = []
    experiences_training_accuracy = []
    per_experience_validation_accuracy = []
    experiences_gradient_error_metric = []

    if(val_steps_epoch == -1):
        num_val_steps_per_epoch = tf.data.experimental.cardinality(val_ds).numpy()
        num_val_samples = 0
        for idx, item in enumerate(val_ds):
            num_val_samples += 1
        num_val_steps_per_epoch = num_val_samples
    else:
        num_val_steps_per_epoch=val_steps_epoch

    train_func = get_train_step_function()
    val_func = get_val_step_function()
    val_func_cwr = get_val_cwr_step_function()
    train_func_cwr = get_train_step_function_exp_bigger_1(is_cwr_quantized)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_additional = tf.keras.metrics.Mean(name='train_loss_additional')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_loss_additional = tf.keras.metrics.Mean(name='val_loss_additional')

    train_accuracy = train_metrics
    val_accuracy = val_metrics

    # Create tensorboard
    tf.summary.trace_on()
    tb_dir = get_run_logdir()
    embeddings_ckpt = tb_dir + 'train/'

    if not os.path.exists(embeddings_ckpt):
        os.makedirs(embeddings_ckpt)
    embeddings_ckpt += 'model_best_weights.ckpt'
    
    num_experiences = len(train_ds)
    
    initial_lr = float(tf.keras.backend.get_value(optimizer.lr))
    
    # Variable that keeps the activation ranges used for quantization
    training_stats_activation = {}
    
    for idx_exp in range(num_experiences):
        
        # Reset learning rate at each experience
        tf.keras.backend.set_value(optimizer.lr, initial_lr)
        
        num_train_samples = 0
        for idx, item in enumerate(train_ds[idx_exp]):
            num_train_samples += 1
        num_train_steps_per_epoch = num_train_samples
        
        if before_training_experience is not None:
            before_training_experience(idx_exp, model, train_ds[idx_exp])

        if idx_exp == 0:
            num_epochs = epochs
        else:
            optimizer = tf.keras.optimizers.SGD(0.0025)
            num_epochs = epochs_for_cwr_experiences
            
        epochs_training_loss = []
        epochs_validation_loss = []
        epochs_validation_accuracy = []
        epochs_training_accuracy = []
        epochs_gradient_error_metric = []
        
        if 'best_accuracy_value' in locals():
            del best_accuracy_value
        
        for epoch in range(num_epochs):

            if add_tensorboard_info == True:
                tensorboard_writer = tf.summary.create_file_writer(tb_dir)

            tf.summary.experimental.set_step(epoch)

            if on_epoch_begin_callback is not None:
                on_epoch_begin_callback(epoch)

            if learning_rate_scheduler is not None:
                lr = float(tf.keras.backend.get_value(optimizer.lr))
                new_lr = learning_rate_scheduler(epoch, lr)
                if new_lr != lr:
                    print('Updated lr {:0.4f} at epoch {}'.format(new_lr, epoch))
                    tf.keras.backend.set_value(optimizer.lr, new_lr)

            # Reset the metrics for the next epoch
            train_loss.reset_states()
            train_loss_additional.reset_states()
            val_loss.reset_states()
            val_loss_additional.reset_states()
            
            gradient_error_metric.reset_states()

            for i in range(len(train_accuracy)):
                train_accuracy[i].reset_states()
            for i in range(len(val_accuracy)):
                val_accuracy[i].reset_states()

            print("Epoch %d/%d" %(epoch+1, num_epochs))
            # Training loop
            t_start = time.perf_counter()
            for idx, (images, labels) in enumerate(train_ds[idx_exp]):
                
                if idx_exp == 0:
                    train_func(model, images, labels, loss_object, optimizer, train_loss, train_loss_additional, train_accuracy, train_all, weight_update_quant_bits, all_other_quant_bits, gradient_error_metric)
                else:
                    train_func_cwr(model, images, labels, loss_object, optimizer, train_loss, train_loss_additional, train_accuracy, training_stats_activation, train_all, weight_update_quant_bits, all_other_quant_bits, gradient_error_metric)

                temp_str = "{}--{}/{} - train loss= {:0.4f}".format(idx_exp+1, idx+1, num_train_steps_per_epoch, train_loss.result())
                if len(model.losses) > 0:
                    temp_str = temp_str + ' reg loss= {:0.4f}'.format(train_loss_additional.result())
                for i in range(len(train_accuracy)):
                    temp_str = temp_str + ' - {}= {:0.4f}'.format(train_accuracy[i].name, train_accuracy[i].result())
                if(idx < (num_train_steps_per_epoch-1)):
                    temp_str = temp_str + '\r'
                    print(temp_str, end='', flush=True)
                else:
                    print(temp_str, flush=True)

            if add_tensorboard_info == True:
                with tensorboard_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result())

                    for i in range(len(train_accuracy)):
                        tf.summary.scalar(train_accuracy[i].name, train_accuracy[i].result())

            from tensorflow.python.ops import array_ops

            t_stop = time.perf_counter()
            #print("\nTrain epoch time %.2f seconds" %( t_stop-t_start), flush=True)

            if mod((epoch+1), validation_frequency) == 0:
                # Validation loop
                for idx, (test_images, test_labels) in enumerate(val_ds):
                    if (idx_exp == 0):
                        val_func(model, test_images, test_labels, loss_object, val_loss, val_loss_additional, val_accuracy)
                    else:
                        val_func_cwr(model, test_images, test_labels, loss_object, val_loss, val_loss_additional, val_accuracy, training_stats_activation)
                if len(model.losses) == 0:
                    print("--%d/%d - val loss= %.4f" %(idx+1, num_val_steps_per_epoch, val_loss.result()), end="", flush=True)
                else:
                    print("--%d/%d - val loss= %.4f - val reg loss= %.4f" %(idx+1, num_val_steps_per_epoch, val_loss.result(), val_loss_additional.result()), end="", flush=True)
        
                for i in range(len(val_accuracy)):
                    print(" - " + str(val_accuracy[i].name + "= %.4f" %(val_accuracy[i].result())), end="")
                print("", flush=True)
                                
                #'''
                # After each experience, delete best_accuracy_value from locals
                if 'best_accuracy_value' in locals():
                    curr_accuracy = val_loss.result()
                    if curr_accuracy < best_accuracy_value:
                        model.save_weights(embeddings_ckpt, overwrite=True)
                        best_accuracy_value = curr_accuracy
                else:
                    best_accuracy_value = val_loss.result()
                    model.save_weights(embeddings_ckpt, overwrite=True)
                #'''

            print(" ")

            epochs_training_loss.append(float(train_loss.result().numpy()))
            epochs_validation_loss.append(float(val_loss.result().numpy()))
            epoch_train_res = []
            epoch_val_res = []
            for i in range(len(train_accuracy)):
                epoch_train_res.append(float(train_accuracy[i].result().numpy() * 100))
            for i in range(len(val_accuracy)):
                epoch_val_res.append(float(val_accuracy[i].result().numpy() * 100))
            epochs_training_accuracy.append(epoch_train_res)
            epochs_validation_accuracy.append(epoch_val_res)
            
            epochs_gradient_error_metric.append(float(gradient_error_metric.result().numpy()))

            train_metrics_names = []
            valid_metrics_names = []
            for i in train_metrics:
                train_metrics_names.append(i.name)
            for i in val_metrics:
                valid_metrics_names.append(i.name)
             
        if (idx_exp == 0):
            for idx, (images, labels) in enumerate(train_ds[0]):
                preds = model(images)
                updateStats(preds[1], training_stats_activation, 'cwr_input')
                updateStats(preds[0], training_stats_activation, 'output')
                
                categorical_labels = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), depth=10, dtype=tf.float32))
                derivata = (preds[0] - categorical_labels)/tf.cast(tf.shape(labels)[0], tf.float32)
                updateStats(derivata, training_stats_activation, 'derivata_logits')
                gradients_f = tf.linalg.matmul(preds[1], derivata, transpose_a=True)
                updateStats(gradients_f, training_stats_activation, 'derivata_weights')
                
        experiences_training_loss.append(epochs_training_loss)
        experiences_validation_loss.append(epochs_validation_loss)
        experiences_training_accuracy.append(epochs_training_accuracy)
        experiences_validation_accuracy.append(epochs_validation_accuracy)
        
        experiences_gradient_error_metric.append(epochs_gradient_error_metric)
        
        # load best loss result
        model.load_weights(embeddings_ckpt)
                
        if after_training_experience is not None:
            after_training_experience(idx_exp, model)
            
        print('Validation after experience:')
        
        for i in range(len(val_accuracy)):
            val_accuracy[i].reset_states()
        val_loss.reset_states()
        val_loss_additional.reset_states()
        
        for idx, (test_images, test_labels) in enumerate(val_ds):
            val_func(model, test_images, test_labels, loss_object, val_loss, val_loss_additional, val_accuracy)  
        print("--%d/%d - val loss= %.4f" %(idx+1, num_val_steps_per_epoch, val_loss.result()), end="", flush=True)

        for i in range(len(val_accuracy)):
            print(" - " + str(val_accuracy[i].name + "= %.4f" %(val_accuracy[i].result())), end="")
        print("", flush=True)
        
        if gradient_error_metric!= None:
            #print("Gradient error quantization= %.4f"%(gradient_error_metric.result()))
            print("Gradient error quantization= %.4f"%(np.mean(epochs_gradient_error_metric)))
            
        per_experience_validation_accuracy.append(float(val_accuracy[0].result().numpy()))

        print(" ")

    return {'epochs_training_loss' : experiences_training_loss,
            'epochs_training_accuracy' : experiences_training_accuracy,
            'epochs_validation_loss' : experiences_validation_loss,
            'epochs_validation_accuracy' : experiences_validation_accuracy,
            'per_experience_validation_accuracy':per_experience_validation_accuracy,
            'experiences_gradient_error_metric': experiences_gradient_error_metric}

def tf_model_evaluate(model, dataset, 
                      evaluation_metrics, 
                      loss_object):

    average_loss = tf.keras.metrics.Mean(name='average_loss')
    average_loss_additional = tf.keras.metrics.Mean(name='average_loss_additional')

    for i in range(len(evaluation_metrics)):
        evaluation_metrics[i].reset_states()

    for idx, (images, labels) in enumerate(dataset):

        predictions = model(images)
        t_loss = loss_object(labels, predictions[0])
        if len(model.losses) > 0:
            t_loss_additional = sum(model.losses)
            average_loss_additional(t_loss_additional)

        average_loss(t_loss)

        for i in range(len(evaluation_metrics)):
            evaluation_metrics[i](labels, predictions[0])

    res = {'average_loss' : average_loss.result().numpy()}
    if len(model.losses) > 0:
        res['average_loss_additional'] = average_loss_additional.result().numpy()
    for i in range(len(evaluation_metrics)):
        res[evaluation_metrics[i].name] = evaluation_metrics[i].result().numpy()

    return res

class LearnableBias(tf.keras.layers.Layer):
    def __init__(self, out_chn):
        super().__init__()
        self.learnable_bias = tf.Variable(tf.zeros([1, 1, 1, out_chn]), name="learnable_bias_"+str(tf.keras.backend.get_uid("learnable_bias")))

    def call(self, inputs):
        return tf.add(inputs, self.learnable_bias)

    def get_config(self):
        return {**super().get_config(), "learnable_bias": self.learnable_bias.numpy()}

def Birealnet18(input_shape, num_classes):
    
    def AddSE(x, to_mul, num_out_channels):
        
        temp2 = tf.keras.layers.GlobalAveragePooling2D()(x)
        temp2 = tf.keras.layers.Dense(num_out_channels/4, activation=None, use_bias=True)(temp2)
        temp2 = tf.keras.layers.LeakyReLU()(temp2)
        temp2 = tf.keras.layers.Dense(num_out_channels, activation=None, use_bias=True)(temp2)
        temp2 = bnf.SymmetricLinearLeaky(1.25, 0.02)(temp2)
        temp2 = bnf.MovingAverage(decay=0.9)(temp2)
        
        temp2 = tf.keras.layers.Multiply()([temp2, to_mul])
        temp2 = tf.keras.layers.BatchNormalization(center=True, scale=False)(temp2)
        return temp2
    
    def block(x, double_filters=False, stride=1):

        # compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = in_filters if not double_filters else 2 * in_filters

        shortcut = x
        if (stride != 1) or (in_filters != out_filters):
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="valid")(shortcut)
            shortcut = bnf.QuantConv2DMixed(out_filters, (1,1), strides=(1, 1), padding="same", pad_values=1.0,
                               kernel_quantizer=lq.quantizers.MagnitudeAwareSign(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.9)(shortcut)

        x = LearnableBias(in_filters)(x)
        x = bnf.QuantConv2DMixed(out_filters, (3,3), strides=(stride, stride), padding="same", pad_values=1.0,
                               input_quantizer=lq.quantizers.ApproxSign(), 
                               kernel_quantizer=lq.quantizers.MagnitudeAwareSign(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        
        #x = AddSE(shortcut, x, out_filters)
        
        x = tf.keras.layers.add([x, shortcut])
        
        x = LearnableBias(out_filters)(x)
        x = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25), shared_axes=[1,2])(x)
        #x = bnf.DPReLU()(x)
        return LearnableBias(out_filters)(x)

    #img_input, orig_input = CreateBinaryInput(input_shape, num_filters=22, use_pointwise=False)
    img_input = tf.keras.layers.Input(shape=input_shape)

    # layer 1
    out = tf.keras.layers.Conv2D(
        64,
        3,
        strides=1,
        kernel_initializer="glorot_normal",
        padding="same",
        use_bias=False
    )(img_input)
    out = tf.keras.layers.BatchNormalization(momentum=0.9)(out)

    # 64 channels
    out = block(out, double_filters=False, stride=1)
    for _ in range(1,4):
        out = block(out, double_filters=False, stride=1)

    # 128 channels
    out = block(out, double_filters=True, stride=2)
    for _ in range(1,4):
        out = block(out, double_filters=False)
        
    # 256 channels
    out = block(out, double_filters=True, stride=2)
    for _ in range(1,4):
        out = block(out, double_filters=False)
        
    # 512 channels
    out = block(out, double_filters=True, stride=2)
    for _ in range(1,4):
        out = block(out, double_filters=False)

    # Final dense layer
    out = tf.keras.layers.GlobalAvgPool2D()(out)
    #cwr_logits = tf.keras.layers.Dense(num_classes, activation=None, use_bias=False, name='cwr')(cwr_input)
    #out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)

    #return tf.keras.Model(inputs=img_input, outputs=[out, cwr_input, cwr_logits])
    
    out = tf.keras.layers.Dense(num_classes, activation="softmax", use_bias=True)(out)

    return tf.keras.Model(inputs=img_input, outputs=out)

def Birealnet(input_shape, num_classes):
    
    def block(x, double_filters=False, stride=1):

        # compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = in_filters if not double_filters else 2 * in_filters

        shortcut = x
        if (stride != 1) or (in_filters != out_filters):
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="valid")(shortcut)
            shortcut = bnf.QuantConv2DMixed(out_filters, (1,1), strides=(1, 1), padding="same", pad_values=1.0,
                               kernel_quantizer=bnf.StdBinaryQuantXTh(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.9)(shortcut)

        x = bnf.QuantConv2DMixed(out_filters, (3,3), strides=(stride, stride), padding="same", pad_values=1.0,
                               input_quantizer=bnf.StdBinaryQuantXTh(), 
                               kernel_quantizer=bnf.StdBinaryQuantXTh(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.add([x, shortcut])
        
        x = bnf.DPReLU()(x)
        return x

    img_input = tf.keras.layers.Input(shape=input_shape)

    # layer 1
    out = tf.keras.layers.Conv2D(
        64,
        3,
        strides=1,
        kernel_initializer="glorot_normal",
        padding="same",
        use_bias=False,
    )(img_input)
    out = tf.keras.layers.BatchNormalization(momentum=0.9)(out)

    # 64 channels
    out = block(out, double_filters=False, stride=1)
    for _ in range(1,4):
        out = block(out, double_filters=False, stride=1)

    # 128 channels
    out = block(out, double_filters=True, stride=2)
    for _ in range(1,4):
        out = block(out, double_filters=False)
        
    # 256 channels
    out = block(out, double_filters=True, stride=2)
    for _ in range(1,4):
        out = block(out, double_filters=False)
        
    # 512 channels
    out = block(out, double_filters=True, stride=2)
    for _ in range(1,4):
        out = block(out, double_filters=False)

    # Final dense layer
    out = tf.keras.layers.GlobalAvgPool2D()(out)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", use_bias=True)(out)

    return tf.keras.Model(inputs=img_input, outputs=out)

def BinaryMobilenet(input_shape, num_classes):
    
    def block(x, double_filters=False, stride=1):

        # compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = in_filters if not double_filters else 2 * in_filters

        out1 = LearnableBias(in_filters)(x)
        out1 = bnf.QuantConv2DMixed(in_filters, (3,3), strides=(stride, stride), padding="same", pad_values=1.0,
                               input_quantizer=lq.quantizers.ApproxSign(), 
                               kernel_quantizer=lq.quantizers.MagnitudeAwareSign(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(out1)
        out1 = tf.keras.layers.BatchNormalization(momentum=0.9)(out1)
        
        if(stride == 2):
            x = tf.keras.layers.AvgPool2D(2, strides=2, padding="valid")(x)
            
        out1 = tf.keras.layers.add([x, out1])
        
        out1 = LearnableBias(in_filters)(out1)
        out1 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25), shared_axes=[1,2])(out1)
        out1 = LearnableBias(in_filters)(out1)
        
        out2 = LearnableBias(in_filters)(out1)
        out2 = bnf.StdBinaryQuant()(out2)
        
        if (in_filters == out_filters):
            out2 = bnf.QuantConv2DMixed(out_filters, (1,1), strides=(1, 1), padding="same", pad_values=1.0,
                               kernel_quantizer=lq.quantizers.MagnitudeAwareSign(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(out2)
            out2 = tf.keras.layers.BatchNormalization(momentum=0.9)(out2)
            out2 = tf.keras.layers.add([out1, out2])
        
        else:
            
            out2_1 = bnf.QuantConv2DMixed(in_filters, (1,1), strides=(1, 1), padding="same", pad_values=1.0,
                               kernel_quantizer=lq.quantizers.MagnitudeAwareSign(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(out2)
            out2_2 = bnf.QuantConv2DMixed(in_filters, (1,1), strides=(1, 1), padding="same", pad_values=1.0,
                               kernel_quantizer=lq.quantizers.MagnitudeAwareSign(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(out2)
            
            out2_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(out2_1)
            out2_2 = tf.keras.layers.BatchNormalization(momentum=0.9)(out2_2)
            
            out2_1 = tf.keras.layers.add([out1, out2_1])
            out2_2 = tf.keras.layers.add([out1, out2_2])
            
            out2 = tf.keras.layers.Concatenate(axis=-1)([out2_1, out2_2])
            
        
        out2 = LearnableBias(out_filters)(out2)
        out2 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25), shared_axes=[1,2])(out2)
        return LearnableBias(out_filters)(out2)
    
    img_input = tf.keras.layers.Input(shape=input_shape)

    # layer 1
    out = tf.keras.layers.Conv2D(
        32,
        3,
        strides=2,
        kernel_initializer="glorot_normal",
        padding="same",
        use_bias=False,
    )(img_input)
    out = tf.keras.layers.BatchNormalization(momentum=0.9)(out)
    
    # 64 channels
    out = block(out, double_filters=True, stride=1)
    
    # 128 channels
    out = block(out, double_filters=True, stride=2)
    out = block(out, double_filters=False, stride=1)
    
    # 256 channels
    out = block(out, double_filters=True, stride=2)
    out = block(out, double_filters=False, stride=1)
    
    # 512 channels
    out = block(out, double_filters=True, stride=2)
    out = block(out, double_filters=False, stride=1)
    #out = block(out, double_filters=False, stride=1)
    #out = block(out, double_filters=False, stride=1)
    #out = block(out, double_filters=False, stride=1)
    #out = block(out, double_filters=False, stride=1)
    
    # 1024 channels
    #out = block(out, double_filters=True, stride=2)
    #out = block(out, double_filters=False, stride=1)
    
    #out = tf.keras.layers.GlobalAvgPool2D()(out)
    #out = tf.keras.layers.Dense(num_classes, activation="softmax", use_bias=True)(out)

    #return tf.keras.Model(inputs=img_input, outputs=out)    
    
    cwr_input = tf.keras.layers.GlobalAvgPool2D()(out)
    cwr_logits = tf.keras.layers.Dense(num_classes, activation=None, use_bias=False, name='cwr')(cwr_input)
    #cwr_logits = bnf.QuantDense(num_classes, activation=None, use_bias=False, 
    #                     kernel_quantizer=bnf.FakeQuant_pow2(), 
    #                     input_quantizer=bnf.FakeQuant_pow2(), 
    #                     binarization_delay_calls=6000, name='cwr')(cwr_input)
    out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)

    return tf.keras.Model(inputs=img_input, outputs=[out, cwr_input, cwr_logits])

if __name__ == "__main__":
    
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    
    parser = argparse.ArgumentParser(description='Train Continual on CIFAR10 dataset')
    parser.add_argument('-o', type=str, help='output model folder')
    parser.add_argument('-s', type=str, help='select scenario NC or NI', default="NC")
    parser.add_argument('-c', type=str, help='select to quantize or not cwr (float/quantized)', default="float")
    parser.add_argument('-q', type=int, help='quantization level for weights', default=16)
    parser.add_argument('-w', type=int, help='quantization level for others', default=8)
    parser.add_argument('-n', type=str, help='Select network type (quicknet/quicknetLarge)', default="quicknet")

    args = parser.parse_args()

    if (args.__dict__['o'] is None):
        parser.print_help()
        sys.exit(0)
        
    output_folder = os.path.dirname(args.__dict__['o'])
    
    print("Command line options:")
    print("Output folder: " + output_folder)
    print("Network type: " + args.__dict__['n'])
    print("Continual scenario: " + args.__dict__['s'])
    print("Gradient computation type: " + args.__dict__['c'])
    print("Quantization level for weights: " + str(args.__dict__['q']))
    print("Quantization level for others: " + str(args.__dict__['w']))
    
    if (args.__dict__['n'] == "reactnet-pretrained"):
        reactnet_model = Birealnet18((32, 32, 3), 200)
        reactnet_model.load_weights('C:/Proj/phd/continual_learning/reactnet_pretrained_tiny_imagenet.h5')

        cwr_input = reactnet_model.layers[-2].output
        cwr_logits = tf.keras.layers.Dense(10, activation=None, use_bias=False, name='cwr')(cwr_input)
        out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)
        reactnet_model = tf.keras.Model(inputs=reactnet_model.input, outputs=[out, cwr_input, cwr_logits])
        
    elif (args.__dict__['n'] == "bi-realnet-pretrained"):
        reactnet_model = Birealnet((32, 32, 3), 200)
        reactnet_model.load_weights('C:/Proj/phd/continual_learning/birealnet_pretrained_tiny_imagenet.h5')

        cwr_input = reactnet_model.layers[-2].output
        cwr_logits = tf.keras.layers.Dense(10, activation=None, use_bias=False, name='cwr')(cwr_input)
        out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)
        reactnet_model = tf.keras.Model(inputs=reactnet_model.input, outputs=[out, cwr_input, cwr_logits])
        
    elif (args.__dict__['n'] == "reactnet"):
        reactnet_model = Birealnet18((32, 32, 3), 10)

        cwr_input = reactnet_model.layers[-2].output
        cwr_logits = tf.keras.layers.Dense(10, activation=None, use_bias=False, name='cwr')(cwr_input)
        out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)
        reactnet_model = tf.keras.Model(inputs=reactnet_model.input, outputs=[out, cwr_input, cwr_logits])
        
    elif (args.__dict__['n'] == "bi-realnet"):
        reactnet_model = Birealnet((32, 32, 3), 10)

        cwr_input = reactnet_model.layers[-2].output
        cwr_logits = tf.keras.layers.Dense(10, activation=None, use_bias=False, name='cwr')(cwr_input)
        out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)
        reactnet_model = tf.keras.Model(inputs=reactnet_model.input, outputs=[out, cwr_input, cwr_logits])
        
    else:
        print("Wrong pre-trained model")
        
    reactnet_model.summary()
    
    if (args.__dict__['s'] == "NC"):
        train_ds, val_ds = GetNCDataset()
    elif (args.__dict__['s'] == "NI"):
        train_ds, val_ds = GetNIDataset()
    else:
        print("Wrong continual scenario!")
        sys.exit(0) 
        
    from collections import defaultdict

    samples_per_class_in_experience = defaultdict(int)
    past_sample_per_classes = defaultdict(int)

    classes_in_experience = None
    saved_cwr_weights = {}

    def quantize_np(x):
        range_ = [np.amin(x), np.amax(x)]
        diff = range_[1] - range_[0]
        scale = 255.0 / diff
        return np.around((x - range_[0])*scale)/scale + range_[0]

    def example_per_class(train_ds):

        tot_label = np.empty(shape=(0))
        for (img, label) in train_ds.as_numpy_iterator():
            tot_label = np.concatenate((tot_label, label))
        classes, idx_classes, count_classes = tf.unique_with_counts(tot_label)

        sorted_idx = tf.argsort(classes)
        classes = tf.gather(classes, tf.argsort(sorted_idx))
        count_classes = tf.gather(count_classes, tf.argsort(sorted_idx))

        result_classes = defaultdict(int)
        result_count_classes = defaultdict(int)
        for idx, item in enumerate(classes):
            result_classes[int(item)]=count_classes[idx]
            result_count_classes[int(item)]=count_classes[idx]
        return result_classes, result_count_classes
        #return classes, count_classes

    def reset_weights(model):

        global saved_cwr_weights

        cwr_layer = model.get_layer('cwr')
        cwr_layer.weights[0].assign(tf.zeros_like(cwr_layer.weights[0]))
        n_input_cwr = cwr_layer.weights[0].numpy().shape[0]

        for c, w in saved_cwr_weights.items():

            c = int(c)
            if int(c) in classes_in_experience:

                indices = tf.concat([tf.reshape(tf.range(0, n_input_cwr), (n_input_cwr, 1)), tf.constant(c, shape=(n_input_cwr,1))], axis=1)
                updates = tf.transpose(saved_cwr_weights[c])
                new_weights = tf.tensor_scatter_nd_update(cwr_layer.weights[0], indices, updates)
                cwr_layer.weights[0].assign(new_weights)

    def set_consolidate_weights(model):

        global saved_cwr_weights

        cwr_layer = model.get_layer('cwr')
        n_input_cwr = cwr_layer.weights[0].numpy().shape[0]

        layer_w = cwr_layer.weights[0].numpy().T
        for c, w in saved_cwr_weights.items():
            layer_w[c,:] = saved_cwr_weights[c]

        cwr_layer.weights[0].assign(layer_w.T)

        #for c, w in saved_cwr_weights.items():
        #    
        #    c = int(c)
        #        
        #    indices = tf.concat([tf.reshape(tf.range(0, n_input_cwr), (n_input_cwr, 1)), tf.constant(c, shape=(n_input_cwr,1))], axis=1)
        #    updates = tf.transpose(saved_cwr_weights[c])#cwr_layer.weights[0][:,c]
        #    new_weights = tf.tensor_scatter_nd_update(cwr_layer.weights[0], indices, updates)
        #    cwr_layer.weights[0].assign(new_weights)

    def consolidate_weights(model):

        global classes_in_experience
        global past_sample_per_classes
        global samples_per_class_in_experience
        global saved_cwr_weights

        cwr_layer = model.get_layer('cwr')

        #weights_q = quantize_np(cwr_layer.weights[0].numpy())
        globavg = tf.math.reduce_mean(tf.transpose(cwr_layer.weights[0]).numpy()[list(classes_in_experience.keys())])
        #globavg = quantize_np(tf.math.reduce_mean(tf.transpose(weights_q).numpy()[classes_in_experience.numpy().astype(int)]).numpy())

        for c in classes_in_experience.keys():

            c = int(c)

            curr_w = cwr_layer.weights[0].numpy().T[c]
            #curr_w = weights_q.T[c]

            new_w = curr_w - globavg

            #if (samples_per_class_in_experience[c] == 0):
            #    continue

            if c in saved_cwr_weights.keys():

                w_pastj = tf.cast(tf.math.sqrt(tf.math.divide_no_nan(tf.cast(past_sample_per_classes[c], tf.float32), tf.cast(samples_per_class_in_experience[c], tf.float32) )), tf.float32)
                #w_pastj = quantize_np(tf.cast(tf.math.sqrt(past_sample_per_classes[c] / samples_per_class_in_experience[c]), tf.float32))

                # consolidation
                saved_cwr_weights[c] = tf.math.divide_no_nan(saved_cwr_weights[c] * w_pastj + new_w, (w_pastj+1))
                #saved_cwr_weights[c] = quantize_np((saved_cwr_weights[c] * w_pastj + new_w) / (w_pastj+1))
                past_sample_per_classes[c] += samples_per_class_in_experience[c]

            else:
                #saved_cwr_weights[c] = quantize_np(new_w)
                saved_cwr_weights[c] = new_w
                past_sample_per_classes[c] = samples_per_class_in_experience[c]

    def before_training_experience_callback(idx_exp, model, curr_exp_train_ds):

        global classes_in_experience
        global samples_per_class_in_experience

        print('Starting experience {}'.format(idx_exp+1))

        if idx_exp >= 1:
            # Freeze all layers
            for layer in model.layers:
                if layer.name != 'cwr':
                    layer.trainable = False

        classes_in_experience, samples_per_class_in_experience = example_per_class(curr_exp_train_ds)

        # reset weights
        reset_weights(model)

    def after_training_experience_callback(idx_exp, model):

        print('Finished experience {}'.format(idx_exp+1))
        consolidate_weights(model)
        set_consolidate_weights(model)
        
    learning_rate = 0.0025
    lr_decay = 1e-6
    lr_drop = 20
    #num_epochs = 10 # NC scenario
    num_epochs = 10 # NI scenario

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    #optimizer = tf.keras.optimizers.SGD(learning_rate)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    if (args.__dict__['c'] == "float"):
        quantize_cwr = False
    else:
        quantize_cwr = True


    history = tf_model_fit(reactnet_model, train_ds, val_ds, num_epochs,
                           [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='train_accuracy')],
                           [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='val_accuracy')],
                           loss_obj, optimizer,
                           epochs_for_cwr_experiences=5,
                           is_cwr_quantized=quantize_cwr,
                           validation_frequency=1,
                           weight_update_quant_bits=args.__dict__['q'],
                           all_other_quant_bits=args.__dict__['w'],
                           gradient_error_metric=tf.keras.metrics.MeanAbsolutePercentageError(),
                           before_training_experience=before_training_experience_callback,
                           after_training_experience=after_training_experience_callback)
    
    if (quantize_cwr == True):
        output_file_name = output_folder + '/' + args.__dict__['n'] + '_' + args.__dict__['s'] + '_' + 'gradient_quantized_quant_weights_' + str(args.__dict__['q']) + '_quant_others_' + str(args.__dict__['w']) + '.json'
    else:
        output_file_name = output_folder + '/' + args.__dict__['n'] + '_' + args.__dict__['s'] + '_' + 'gradient_float' + '.json'
    
    print("Saved output file: " + output_file_name)
    
    with open(output_file_name, 'w') as fp:
        json.dump(history, fp, indent=4)