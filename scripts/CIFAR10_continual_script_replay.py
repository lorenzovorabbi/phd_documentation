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
import Backpropagation_impl as back_prop
import CWR_replay_buffer as cwr

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

if __name__ == "__main__":
    
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    
    parser = argparse.ArgumentParser(description='Train Continual on CIFAR10 dataset')
    parser.add_argument('-o', type=str, help='output model folder')
    parser.add_argument('-s', type=str, help='select scenario NC or NI', default="NC")
    parser.add_argument('-c', type=str, help='select to quantize or not cwr (float/quantized)', default="float")
    parser.add_argument('-q', type=int, help='quantization level for weights', default=16)
    parser.add_argument('-w', type=int, help='quantization level for others', default=8)
    parser.add_argument('-b', type=int, help='quantization level for binary weights', default=8)
    parser.add_argument('-n', type=str, help='Select network type (quicknet/quicknetLarge)', default="quicknet")

    args = parser.parse_args()

    if (args.__dict__['o'] is None):
        parser.print_help()
        sys.exit(0)
        
    output_folder = os.path.dirname(args.__dict__['o'])
    quantization_for_binary_weights = args.__dict__['b'] 
    
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
        reactnet_model = tf.keras.Model(inputs=reactnet_model.input, outputs=[out])
        
    elif (args.__dict__['n'] == "bi-realnet-pretrained"):
        reactnet_model = Birealnet((32, 32, 3), 200)
        reactnet_model.load_weights('C:/Proj/phd/continual_learning/birealnet_pretrained_tiny_imagenet.h5')

        cwr_input = reactnet_model.layers[-2].output
        cwr_logits = tf.keras.layers.Dense(10, activation=None, use_bias=False, name='cwr')(cwr_input)
        out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)
        reactnet_model = tf.keras.Model(inputs=reactnet_model.input, outputs=[out])
        
    elif (args.__dict__['n'] == "reactnet"):
        reactnet_model = Birealnet18((32, 32, 3), 10)

        cwr_input = reactnet_model.layers[-2].output
        cwr_logits = tf.keras.layers.Dense(10, activation=None, use_bias=False, name='cwr')(cwr_input)
        out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)
        reactnet_model = tf.keras.Model(inputs=reactnet_model.input, outputs=[out])
        
    elif (args.__dict__['n'] == "bi-realnet"):
        reactnet_model = Birealnet((32, 32, 3), 10)

        cwr_input = reactnet_model.layers[-2].output
        cwr_logits = tf.keras.layers.Dense(10, activation=None, use_bias=False, name='cwr')(cwr_input)
        out = tf.keras.layers.Softmax(name='softmax')(cwr_logits)
        reactnet_model = tf.keras.Model(inputs=reactnet_model.input, outputs=[out])
        
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
        
    def find_layer_in_model(model, layer_name):
        for layer in model.layers:
            if layer.name == layer_name:
                return layer
        return None

    def get_layer_by_index(model, idx):
        return model.layers[idx].output

    def get_layer_by_index2(model, idx):
        return model.layers[idx]


    # Change this function to remove bin quant weights
    def split_model_for_online_training(model, num_fw_bits=None, num_quant_bits_bw=None):

        global quantization_for_binary_weights
        global model_selected

        layers_aggregated = []

        layers_aggregated.append(back_prop.ReactnetBlock_BinConv_BN_Bias_PRelu(get_layer_by_index2(model, -23),
                                                                              get_layer_by_index2(model, -22),
                                                                              get_layer_by_index2(model, -24),
                                                                              get_layer_by_index2(model, -20),
                                                                              get_layer_by_index2(model, -19),
                                                                              get_layer_by_index2(model, -18),
                                                                              num_quant_bits_fw=num_fw_bits,
                                                                              num_quant_bits_bw=num_quant_bits_bw,
                                                                              bin_weights_quant=quantization_for_binary_weights))

        layers_aggregated.append(back_prop.ReactnetBlock_BinConv_BN_Bias_PRelu(get_layer_by_index2(model, -16),
                                                                              get_layer_by_index2(model, -15),
                                                                              get_layer_by_index2(model, -17),
                                                                              get_layer_by_index2(model, -13),
                                                                              get_layer_by_index2(model, -12),
                                                                              get_layer_by_index2(model, -11),
                                                                              num_quant_bits_fw=num_fw_bits,
                                                                              num_quant_bits_bw=num_quant_bits_bw,
                                                                              bin_weights_quant=quantization_for_binary_weights))

        layers_aggregated.append(back_prop.ReactnetBlock_BinConv_BN_Bias_PRelu(get_layer_by_index2(model, -9),
                                                                              get_layer_by_index2(model, -8),
                                                                              get_layer_by_index2(model, -10),
                                                                              get_layer_by_index2(model, -6),
                                                                              get_layer_by_index2(model, -5),
                                                                              get_layer_by_index2(model, -4),
                                                                              num_quant_bits_fw=num_fw_bits,
                                                                              num_quant_bits_bw=num_quant_bits_bw,
                                                                              bin_weights_quant=quantization_for_binary_weights))

        layers_aggregated.append(back_prop.GlobalAVGPooling_layer(num_quant_bits_fw=num_fw_bits, num_quant_bits_bw=num_quant_bits_bw))
        layers_aggregated.append(back_prop.CWR_layer(get_layer_by_index2(model, -2), num_quant_bits_fw=num_fw_bits, num_quant_bits_bw=num_quant_bits_bw))

        # Extract model with frozen layers
        reactnet_model_reduced = tf.keras.Model(inputs=reactnet_model.input, outputs=get_layer_by_index(model, -25))
        for layer in reactnet_model_reduced.layers:
            layer.trainable = False


        output = dict()
        output['frozen_model'] = reactnet_model_reduced
        output['trainable_model'] = layers_aggregated

        return output
    
    def split_model_birealnet_for_online_training(model, num_fw_bits=None, num_quant_bits_bw=None):

        global quantization_for_binary_weights
        global model_selected

        layers_aggregated = []

        layers_aggregated.append(back_prop.BirealnetBlock_BinConv_Conv_BN_DPRelu(get_layer_by_index2(model, -15),
                                                                              get_layer_by_index2(model, -14),
                                                                              get_layer_by_index2(model, -12),
                                                                              num_quant_bits_fw=num_fw_bits,
                                                                              num_quant_bits_bw=num_quant_bits_bw,
                                                                              bin_weights_quant=quantization_for_binary_weights))

        layers_aggregated.append(back_prop.BirealnetBlock_BinConv_Conv_BN_DPRelu(get_layer_by_index2(model, -11),
                                                                              get_layer_by_index2(model, -10),
                                                                              get_layer_by_index2(model, -8),
                                                                              num_quant_bits_fw=num_fw_bits,
                                                                              num_quant_bits_bw=num_quant_bits_bw,
                                                                              bin_weights_quant=quantization_for_binary_weights))

        layers_aggregated.append(back_prop.BirealnetBlock_BinConv_Conv_BN_DPRelu(get_layer_by_index2(model, -7),
                                                                              get_layer_by_index2(model, -6),
                                                                              get_layer_by_index2(model, -4),
                                                                              num_quant_bits_fw=num_fw_bits,
                                                                              num_quant_bits_bw=num_quant_bits_bw,
                                                                              bin_weights_quant=quantization_for_binary_weights))

        layers_aggregated.append(back_prop.GlobalAVGPooling_layer(num_quant_bits_fw=num_fw_bits, num_quant_bits_bw=num_quant_bits_bw))
        layers_aggregated.append(back_prop.CWR_layer(get_layer_by_index2(model, -2), num_quant_bits_fw=num_fw_bits, num_quant_bits_bw=num_quant_bits_bw))

        # Extract model with frozen layers
        reactnet_model_reduced = tf.keras.Model(inputs=reactnet_model.input, outputs=get_layer_by_index(model, -16))
        for layer in reactnet_model_reduced.layers:
            layer.trainable = False


        output = dict()
        output['frozen_model'] = reactnet_model_reduced
        output['trainable_model'] = layers_aggregated

        return output

    replay_buffer = cwr.ClassBalancedBuffer(300, (4, 4, 512), 10)
    cwr_obj = cwr.CWR(replay_buffer=replay_buffer)
        
    learning_rate = 0.0025
    lr_decay = 1e-6
    lr_drop = 20
    #num_epochs = 10 # NC scenario
    num_epochs = 15 # NI scenario

    #optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    if (args.__dict__['c'] == "float"):
        quantize_cwr = False
        quantization_for_binary_weights = None
    else:
        quantize_cwr = True
    
    history = cwr.tf_model_fit(reactnet_model, train_ds, val_ds, 
                           num_epochs,
                           [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='train_accuracy')],
                           [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='val_accuracy')],
                           tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           tf.keras.optimizers.SGD(0.0025),
                           epochs_for_cwr_experiences=10,
                           is_cwr_quantized=quantize_cwr,
                           validation_frequency=5,
                           weight_update_quant_bits=args.__dict__['q'],
                           all_other_quant_bits=args.__dict__['w'],
                           gradient_error_metric=tf.keras.metrics.MeanAbsolutePercentageError(),
                           cwr_obj=cwr_obj,
                           split_model_online_training_callback=split_model_birealnet_for_online_training)
    
    if (quantize_cwr == True):
        output_file_name = output_folder + '/' + args.__dict__['n'] + '_' + args.__dict__['s'] + '_' + 'gradient_quantized_quant_weights_' + str(args.__dict__['q']) + '_quant_others_' + str(args.__dict__['w']) + '_quant_binary_' + str(quantization_for_binary_weights) + '.json'
    else:
        output_file_name = output_folder + '/' + args.__dict__['n'] + '_' + args.__dict__['s'] + '_' + 'gradient_float' + '.json'
    
    print("Saved output file: " + output_file_name)
    
    with open(output_file_name, 'w') as fp:
        json.dump(history, fp, indent=4)