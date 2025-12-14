#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from typing import Dict
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import partial
import glob
import time
import pickle as pkl
import argparse
import random 
import importlib
import sys
import functools
import Utilities as uti
import larq as lq
import larq_zoo as lqz
import larq_compute_engine as lce
import BNF as bnf

if importlib.util.find_spec("tensorflow_model_optimization") is None:
    print("Cannot find tensorflow optimization package, please install it!")
    sys.exit(0)

import tensorflow_model_optimization as tfmot
import MyBinaryNet as Mybinary

MBNetV2_sizeX = 224
MBNetV2_sizeY = 224

COR250_num_classes = 50
CORe50_num_sequences = 11
COR350_num_sequences_validation = 3
CORe50_batch_size = 64#128#64


def get_run_logdir(add_info = None):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    if(add_info != None):
        run_id += add_info
    return os.path.join(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "tf_logs", run_id)

def MBNetV2_preprocess(image, label):
    image = tf.image.resize(image, [MBNetV2_sizeX, MBNetV2_sizeY])
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    #image = keras.applications.efficientnet.preprocess_input(image)
    return image, label

def QuickNet_preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    #image = lqz.preprocess_input(image)
    image = tf.image.per_image_standardization(image)
    #image = uti.NormalizeImagenetDataset(image)
    return image, label

def CustomBinary_pre_process_augmentation(image, label):

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tfa.image.rotate(image, tf.random.uniform(shape=[], minval=0, maxval=45, dtype=tf.float32))
    #image = tf.image.resize_with_pad(image, 112, 112)
    image = tf.image.resize(image, [112, 112], tf.image.ResizeMethod.BICUBIC)
    image = tf.image.per_image_standardization(image)
    return image, label

def CustomBinary_pre_process_validation(image, label):

    #image = tf.image.resize_with_pad(image, 112, 112)
    image = tf.image.resize(image, [112, 112], tf.image.ResizeMethod.BICUBIC)
    image = tf.image.per_image_standardization(image)
    return image, label

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  class_id = tf.strings.substr(parts[-2], 1, -1)
  res = tf.strings.to_number(class_id, out_type=tf.dtypes.int32)-tf.constant(1)
  return res

def load_img(img_path):
  img = tf.io.read_file(img_path)
  img = tf.io.decode_png(img, channels=3)
  #img = tf.image.convert_image_dtype(img, tf.dtypes.float32)
  img = tf.image.resize(img, [MBNetV2_sizeX, MBNetV2_sizeY])
  #return keras.applications.mobilenet_v2.preprocess_input(img)
  return keras.applications.efficientnet.preprocess_input(img)
  #return img

def process_path(file_path):
  label = get_label(file_path)
  img = load_img(file_path)
  return img, label

def configure_for_performance(ds : tf.data.Dataset,
                              size : int,
                              batch_size : int) -> tf.data.Dataset:
  #ds = ds.cache()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds

@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)



def CORe50_dataset(image_file, paths_file):

    pkl_file = open(paths_file, 'rb') 
    paths = pkl.load(pkl_file)
    num_total_samples = len(paths)

    #classes_for_validation =  sorted(random.sample(range(1, CORe50_num_sequences), COR350_num_sequences_validation))
    classes_for_validation = [3, 7, 10]
    val_set = []
    train_set = []

    for idx, x in enumerate(paths):
       seq_id = int(x.split('/')[-1].split('_')[-3])
       class_id = int(x.split('/')[-1].split('_')[-2]) - 1
       if seq_id in classes_for_validation:
           val_set.append((idx, seq_id, class_id))
       else:
           train_set.append((idx, seq_id, class_id))

    val_ranges = []
    init_range = val_set[0][0]
    prev_idx = init_range
    for i in range(1, len(val_set)):
        if(val_set[i][0] - prev_idx == 1):
            prev_idx = val_set[i][0]
            continue
        else:
            val_ranges.append((init_range, prev_idx))
            init_range = val_set[i][0]
            prev_idx = init_range

    val_ranges.append((init_range, prev_idx))

    train_ranges = []
    init_range = train_set[0][0]
    prev_idx = init_range
    for i in range(1, len(train_set)):
        if(train_set[i][0] - prev_idx == 1):
            prev_idx = train_set[i][0]
            continue
        else:
            train_ranges.append((init_range, prev_idx))
            init_range = train_set[i][0]
            prev_idx = init_range

    train_ranges.append((init_range, prev_idx))

    val_labels = [x[2] for x in val_set]
    train_labels = [x[2] for x in train_set]

    val_array_labels = np.array(val_labels)
    train_array_labels = np.array(train_labels)

    print("Loading images...", end='', flush=True)
    imgs = np.load(image_file)['x']
    print(" loaded!")

    val_images = [imgs[x[0]:x[1]+1] for x in val_ranges]
    val_images = np.concatenate( val_images, axis=0 )
    train_images = [imgs[x[0]:x[1]+1] for x in train_ranges]
    train_images = np.concatenate( train_images, axis=0 )

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_array_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_array_labels))

    train_ds = train_ds.shuffle(len(train_images))

    #train_ds = train_ds.map(MBNetV2_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #val_ds = val_ds.map(MBNetV2_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #train_ds = train_ds.map(QuickNet_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #val_ds = val_ds.map(QuickNet_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.repeat().map(CustomBinary_pre_process_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(CustomBinary_pre_process_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = configure_for_performance(train_ds, len(train_images), CORe50_batch_size)
    val_ds = configure_for_performance(val_ds, len(val_images), CORe50_batch_size)

    return train_ds, val_ds

def LoadCORe50_dataset_from_images(base_folder_path):

    def SelectImagesForTrainValidation(base_folder_path, sequence_id):
        num_elements = 0
        dataset = None

        for i in range(COR250_num_classes):
                if(i == 0):
                    dir = base_folder_path + "s" + str(sequence_id) + os.path.sep + "o" + str(i+1) + os.path.sep
                    dataset = tf.data.Dataset.list_files(dir + "*.png", shuffle=True)
                    num_elements += len(list(glob.glob(dir + "*.png")))
                else:
                    dir = base_folder_path + "s" + str(sequence_id) + os.path.sep + "o" + str(i+1) + os.path.sep
                    dataset = tf.data.Dataset.list_files(dir + "*.png", shuffle=True).concatenate(dataset)
                    num_elements += len(list(glob.glob(dir + "*.png")))

        return dataset, num_elements

    num_train_samples = 0
    num_val_samples = 0

    for i in range(CORe50_num_sequences):
    
        print("Loading sequence id: " + str(i+1) + "...", flush=True)

        if((i+1) in [3, 7, 10]):
            if (i+1) == 3:
                val_ds, num_val_samples = SelectImagesForTrainValidation(base_folder_path, i+1)
            else:
                val_ds_tmp, num_val_samples_tmp = SelectImagesForTrainValidation(base_folder_path, i+1)
                val_ds = val_ds.concatenate(val_ds_tmp)
                num_val_samples += num_val_samples_tmp
        else:
            if (i+1) == 1:
                train_ds, num_train_samples = SelectImagesForTrainValidation(base_folder_path, i+1)
            else:
                train_ds_tmp, num_train_tmp = SelectImagesForTrainValidation(base_folder_path, i+1)
                train_ds = train_ds.concatenate(train_ds_tmp)
                num_train_samples += num_train_tmp

    train_ds = train_ds.shuffle(num_train_samples)

    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = configure_for_performance(train_ds, num_train_samples, CORe50_batch_size)
    val_ds = configure_for_performance(val_ds, num_val_samples, CORe50_batch_size)

    return train_ds, val_ds

def prune_model(model, train_ds, val_ds, out_file, initial_sparse=0.1, final_sparse=0.5, epochs=3):

    end_step = tf.data.experimental.cardinality(train_ds).numpy().astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparse,
                                                                    final_sparsity=final_sparse,
                                                                    begin_step=0,
                                                                    end_step=end_step)
    }

    def apply_pruning_strategy(layer):
        if (isinstance(layer, tf.keras.layers.experimental.preprocessing.Rescaling) or 
            isinstance(layer, tf.keras.layers.experimental.preprocessing.Normalization)):
                return layer
        else:
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    model_to_prune = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_strategy,
    )

    #model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_to_prune.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model_to_prune.summary()

    prune_log_dir = get_run_logdir("pruned")
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=prune_log_dir, 
                                                 histogram_freq = 1,
                                                 write_images = False)
    history = model_to_prune.fit(train_ds,
                                    validation_data=val_ds,
                                    callbacks=[tfmot.sparsity.keras.UpdatePruningStep(), 
                                    tfmot.sparsity.keras.PruningSummaries(log_dir=get_run_logdir())],
                                    epochs=epochs)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_to_prune)
    tf.keras.models.save_model(model_for_export, out_file, include_optimizer=False)

    return model_for_export


def cluster_weights(model, train_ds, val_ds, num_clusters, optimizer, out_file):

    clustering_params_last = {
    'number_of_clusters': 8,
    'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.DENSITY_BASED
    }

    clustering_params_previous = {
    'number_of_clusters': 512,
    'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.DENSITY_BASED
    }

    clustering_params_first = {
    'number_of_clusters': 16,
    'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.DENSITY_BASED
    }

    num_layers = 0
    num_conv2d = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            num_conv2d += 1
    
    def apply_clustering_to_conv(layer):
        nonlocal num_layers
        nonlocal num_conv2d
        if isinstance(layer, tf.keras.layers.Conv2D):
            num_layers += 1
            if (num_layers == (num_conv2d)):
                return tfmot.clustering.keras.cluster_weights(layer, **clustering_params_last)
            #elif (num_layers == 1):
            #    return tfmot.clustering.keras.cluster_weights(layer, **clustering_params_first)
            else:
                return layer
        return layer


    # Use `tf.keras.models.clone_model` to apply `apply_clustering_to_dense` 
    # to the layers of the model.
    clustered_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_clustering_to_conv,
    )

    clustered_model.compile(
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            optimizer=optimizer,
                            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    clustered_model.summary()

    
    model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)
    model_for_export.compile(
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            optimizer=optimizer,
                            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model_for_export.summary()
    tf.keras.models.save_model(model_for_export, out_file, include_optimizer=False)
    print("zipped size of pruned+weigth clustering model: " + str(uti.get_gzipped_model_size(out_file)))
    model_for_export.evaluate(val_ds)

    '''
    history = clustered_model.fit(train_ds,
                                  validation_data=val_ds,
                                  epochs=1)
    '''

    return model_for_export



def train_core50_model(model, train_ds, val_ds, optimizer, epochs, out_file_name):
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    checkpoint_cb = keras.callbacks.ModelCheckpoint(out_file_name, save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=get_run_logdir(), 
                                                     histogram_freq = 1,
                                                     write_images = False)
    

    '''
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    t_loss, t_accu, v_loss, v_accu = uti.tf_model_fit(model, train_ds, val_ds, epochs, train_loss, train_accuracy, valid_loss, valid_accuracy, loss_object, optimizer)
    
    history = {'train_loss' : t_loss,
               'train_accuracy' : t_accu,
               'val_loss' : v_loss,
               'val_accuracy' : v_accu}
    '''
    
    start = time.perf_counter()
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        steps_per_epoch=120000 // CORe50_batch_size,
                        callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb],
                        epochs=epochs,
                        validation_freq=1)
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds during training.' % elapsed)
    
    
    model.save(out_file_name)

    return model, history

def quantize_model(model, train_ds, val_ds, out_model_file, quant_type, shuffle_length=100, optimize=True, force_all_quantize = True, dataset_provided=True):

    val_ds = val_ds.shuffle(shuffle_length)

    def dataset_generator():
        for input_value in val_ds.batch(1).take(100):
            yield [input_value[0]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if(optimize == True):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if dataset_provided==True:
        converter.representative_dataset = dataset_generator

    if force_all_quantize == True:
        if quant_type == "u8":
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        elif quant_type == "f16":
            converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    open(out_model_file, "wb").write(tflite_model)

    return tflite_model

def InitQuickNet(num_labels, model_path=None):

    model_quick = lqz.sota.QuickNet(input_shape=(224, 224, 3), weights="imagenet", include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(model_quick.output)
    output = keras.layers.Dense(num_labels, activation="softmax")(x)
    model = keras.models.Model(inputs=model_quick.input, outputs=output)
    #model.load_weights("/home/lvorabbi/Proj/Phd/state_art/core50_models/QuickNet/QuickNet_CORe50.h5")

    for layer in model.layers:
        if isinstance(layer, lq.layers.QuantConv2D):
            curr_weights = layer.get_weights()
            if(len(curr_weights) > 0):
                temp = np.ravel(curr_weights)
                if isinstance(layer.kernel_constraint, lq.constraints.WeightClip):
                    layer.kernel_constraint = keras.constraints.max_norm(1.)
                #for item in temp:

    return model

def InitCustomBinary(num_labels, model_path=None):

    #model_img_net = Mybinary.CreateBinaryNetwork((112, 112, 3), 1000)
    model_img_net = Mybinary.CreateBNFBinaryNetwork((112, 112, 3), 1000)
    #model_img_net.load_weights("/home/lvorabbi/Proj/Phd/state_art/Imagenet/MyBinary/fine_tuning_model_without_prelu.h5")

    x = keras.layers.Dense(num_labels,
                        activation='softmax',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        name='probs')(model_img_net.layers[-2].output)
    
    out_model = keras.Model(inputs=model_img_net.input, outputs=x)
    if(model_path != None):
        out_model.load_weights(model_path)
    return out_model

def main(image_file, path_file, output_model_path, enable_traing_added_layers, train_quant_aware, weight_pruning, num_fine_epochs):

    tf_gpus = tf.config.list_physical_devices('GPU')
    if(len(tf_gpus) > 0):
        
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs available: ", len(tf_gpus))
        
        for gpu in tf_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU is not available!")

    #model = InitQuickNet(COR250_num_classes)
    model = InitCustomBinary(COR250_num_classes)

    train_ds, val_ds = CORe50_dataset(image_file, path_file)

    '''
    base_model = tf.keras.applications.MobileNetV2(input_shape=(MBNetV2_sizeX, MBNetV2_sizeY, 3),
                                                        weights="imagenet",
                                                        include_top=False)
    '''

    '''
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(MBNetV2_sizeX, MBNetV2_sizeY, 3),
                                                        weights="imagenet",
                                                        include_top=False)

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(COR250_num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=base_model.input, outputs=output)
    '''
    
    #model = InitQuickNet(COR250_num_classes)

    model.summary()

    print("\n********** Transfer learning using MobileNetV2! **********\n")

    '''
    if(enable_traing_added_layers == "True"):

        for layer in base_model.layers:
            layer.trainable = False
        base_model.trainable = False

        model, history = train_core50_model(model, train_ds, val_ds, 
                                   keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01),
                                   2, 
                                   output_model_path + os.path.sep + "MobileNetV2_CORe50_checkpoint_coarse_2.h5")
    #else:
        #model = keras.models.load_model(output_model_path + os.path.sep + "MobileNetV2_CORe50_checkpoint_coarse_2.h5")
    '''

    print("\n********** Fine tuning training! **********\n")

    '''
    for layer in base_model.layers:
        layer.trainable = True

    base_model.trainable = True
    '''

    model.trainable = True

    model.save("C:\\Proj\\phd\\temp_binary_bnf.h5")
    model, history = train_core50_model(model, train_ds, val_ds, 
                                #tf.keras.optimizers.Nadam(0.001),
                                tf.keras.optimizers.Adam(learning_rate=0.0025),
                                #keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True),
                                #keras.optimizers.SGD(learning_rate=0.05, momentum=0.9, nesterov=True),
                                num_fine_epochs, 
                                output_model_path + os.path.sep + "fine_tuning_no_prelu.h5")
    #uti.PlotHistory(history)

    if(weight_pruning == "True"):
        model_pruned = prune_model(model, train_ds, val_ds, output_model_path + os.path.sep + "MobileNetV2_CORe50_pruned.h5")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train MobileNetV2 on CORe50 dataset')
    parser.add_argument('-f', type=str, help='path to the img cropped 128x128 .npz file')
    parser.add_argument('-p', type=str, help='path to the pickle path file')
    parser.add_argument('-o', type=str, help='output model path .h5 format')
    parser.add_argument('-a', type=str, help='enable training freezing the layers of pre-trained model')
    parser.add_argument('-q', type=str, help='enable quantization aware training')
    parser.add_argument('-w', type=str, help='enable weight pruning after fine training')
    parser.add_argument('-e', type=int, help='num epochs of fine training')

    args = parser.parse_args()

    tf.keras.metrics.Precision()

    if (args.__dict__['f'] is None) or (args.__dict__['p'] is None):
        parser.print_help()
        sys.exit(0)

    if(args.o == None):
        args.o = os.path.dirname(os.path.realpath(__file__))

    if(args.a == None):
        args.a = "True"

    if(args.q == None):
        args.q = "True"

    if(args.w == None):
        args.w = "True"

    if(args.e == None):
        args.e = 60

    main(args.f, args.p, args.o, args.a, args.q, args.w, args.e)