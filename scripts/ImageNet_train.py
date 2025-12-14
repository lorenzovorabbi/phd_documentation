import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib
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
import json
import imageio
from PIL import Image
import larq as lq
import larq_zoo as lqz
import larq_compute_engine as lce

#import tkinter as tk
#matplotlib.use('TKAgg', force=True) 

import Utilities as uti
import tensorflow_model_optimization as tfmot

import EfficientNetB0_impl as efficient

import MyBinaryNet as Mybinary

batch_size = 128#64

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(16)

TARGET_SIZE = 112

def _normalize(image, mean_rgb=MEAN_RGB, stddev_rgb=STDDEV_RGB):
    """Normalizes images to variance 1 and mean 0 over the whole dataset"""

    image -= tf.broadcast_to(mean_rgb, tf.shape(image))
    image /= tf.broadcast_to(stddev_rgb, tf.shape(image))

    return image

def rotate(image : tf.Tensor, 
           label : tf.Tensor) -> (tf.Tensor, tf.Tensor):

    return tfa.image.rotate(image, tf.random.uniform(shape=[], minval=0, maxval=359, dtype=tf.float32)), label

def augmentation_function(image : tf.Tensor, 
                         label : tf.Tensor) -> (tf.Tensor, tf.Tensor):

    image = tf.image.random_crop(image, size=[224, 224, 3])

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    #image = tf.image.random_brightness(image, max_delta=20.0)
    image = tfa.image.rotate(image, tf.random.uniform(shape=[], minval=0, maxval=45, dtype=tf.float32))
    image = tf.image.resize_with_pad(image, TARGET_SIZE, TARGET_SIZE)

    image = tf.image.per_image_standardization(image)
    #image -= tf.broadcast_to(uti.MEAN_RGB, tf.shape(image))
    #image /= tf.broadcast_to(uti.STDDEV_RGB, tf.shape(image))

    return image, label

def validation_function(image : tf.Tensor, 
                        label : tf.Tensor) -> (tf.Tensor, tf.Tensor):

    image = tf.image.random_crop(image, size=[224, 224, 3])
    image = tf.image.resize_with_pad(image, TARGET_SIZE, TARGET_SIZE)

    image = tf.image.per_image_standardization(image)

    #image -= tf.broadcast_to(uti.MEAN_RGB, tf.shape(image))
    #image /= tf.broadcast_to(uti.STDDEV_RGB, tf.shape(image))

    return image, label

def load_dataset(image_net_tfrecords_path):

    global batch_size

    '''
    train_ds = tf.keras.preprocessing.image_dataset_from_directory('/home/lvorabbi/Desktop/Disk2/ImageNet/ILSVRC2012_img_train', 
                                                                   image_size=(320, 320), seed=1234, 
                                                                   batch_size=batch_size, shuffle=True, label_mode='int')
    val_ds = tf.keras.preprocessing.image_dataset_from_directory('/home/lvorabbi/Desktop/Disk2/ImageNet/ILSVRC2012_img_val', 
                                                                   image_size=(320, 320), seed=1234, batch_size=batch_size, label_mode='int')
    '''

    train_ds = uti.Get_TfRecords_dataset(image_net_tfrecords_path, 'train', batch_size, 0)
    val_ds = uti.Get_TfRecords_dataset(image_net_tfrecords_path, 'validation', batch_size, 0)

    return train_ds, val_ds

def Augment(train_ds):

    random_crop_rotations = train_ds.map(augmentation_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return random_crop_rotations


def EfficientNet_preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    image = keras.applications.efficientnet.preprocess_input(image)
    return image, label

def MBNetV2_preprocess(image, label):
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def binary_preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    return image, label

def QuickNet_preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    image = lqz.preprocess_input(image)
    return image, label

def InitQuickNet(num_labels, model_path=None):

    model = lqz.sota.QuickNet(input_shape=(224, 224, 3),weights="imagenet", include_top=True)
    return model

'''
#model_no_se = efficient.EfficientNetB0(input_shape=(224, 224, 3), weights="imagenet", include_top=False)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights=None, include_top=False, alpha=1.0)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(1000, activation="softmax")(x)
model = keras.models.Model(inputs=base_model.input, outputs=output)
'''

model = Mybinary.CreateBinaryNetwork((TARGET_SIZE, TARGET_SIZE, 3), 1000)
#model = InitQuickNet(1000)

'''
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                weights=None,#"imagenet",
                                                include_top=True)
'''

train_ds, val_ds = load_dataset('/home/lvorabbi/Desktop/SSD/Images/ILSVRC2012/tfrecords/')
train_ds, val_ds = load_dataset('/mnt/data2/ILSVRC2012/tfrecords/')
train_ds = Augment(train_ds)   

val_ds = val_ds.map(validation_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = train_ds.batch(batch_size=batch_size)
train_ds = train_ds.prefetch(batch_size)
val_ds = val_ds.batch(batch_size=batch_size)
val_ds = val_ds.prefetch(batch_size)

#uti.ShowImagesFromDataset(train_ds, num_images=16)

#train_ds = train_ds.map(MBNetV2_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#val_ds = val_ds.map(MBNetV2_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#train_ds = train_ds.map(EfficientNet_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#val_ds = val_ds.map(EfficientNet_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

num_epochs_warmup = 5
num_epochs = 80
initial_lr = 0.01
final_lr = 0.0001
learning_rate_fn_warmup = uti.get_lr_func(num_epochs_warmup, 'linear_warmup', initial_lr, initial_lr)
learning_rate_fn = uti.get_lr_func(num_epochs, 'linear', initial_lr, final_lr)
#current_optimizer = keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9, nesterov=True)
#current_optimizer = tf.keras.optimizers.RMSprop(learning_rate=initial_lr)
#current_optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.99, nesterov=True)
#current_optimizer = keras.optimizers.Adam(learning_rate=0.0001)#initial_lr)
current_optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

model.trainable = True
for layer in model.layers:
    layer.trainable = True

#tf.keras.backend.set_learning_phase(True)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                optimizer=current_optimizer,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), 
                            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
#model.save("/home/lvorabbi/Proj/Phd/state_art/Imagenet/MyBinary/model_generated.h5")
lq.models.summary(model)


#checkpoint_warmup_cb = keras.callbacks.ModelCheckpoint("/home/lvorabbi/Proj/Phd/state_art/Imagenet/MyBinary/warmup_check_point.h5", save_best_only=True)
#checkpoint_cb = keras.callbacks.ModelCheckpoint("/home/lvorabbi/Proj/Phd/state_art/Imagenet/MyBinary/fine_tuning_check_point.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=uti.get_run_logdir(), 
                                                    histogram_freq = 1,
                                                    write_images = False)

# warmup training
'''
history = model.fit(train_ds,
                    validation_data=val_ds,
                    steps_per_epoch=1281167 // batch_size,
                    validation_steps=50000 // batch_size,
                    callbacks=[tensorboard_cb, learning_rate_fn_warmup, checkpoint_warmup_cb],
                    epochs=num_epochs_warmup,
                    validation_freq=1)
model.save("/home/lvorabbi/Proj/Phd/state_art/Imagenet/MyBinary/warmup.h5")
'''

'''
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

t_loss, t_accu, v_loss, v_accu = uti.tf_model_fit(model, train_ds, val_ds, num_epochs, 
                                                train_loss, train_accuracy, valid_loss, valid_accuracy, 
                                                loss_object, current_optimizer, 
                                                train_steps_epoch=1281167 // batch_size,
                                                val_steps_epoch=50000 // batch_size)
'''                                                
#'''
# callbacks=[checkpoint_cb, tensorboard_cb, early_stopping_cb, learning_rate_fn],
history = model.fit(train_ds,
                    validation_data=val_ds,
                    steps_per_epoch=1281167 // batch_size,
                    validation_steps=50000 // batch_size,
                    callbacks=[early_stopping_cb, tensorboard_cb],
                    epochs=num_epochs,
                    validation_freq=1)
#'''
#model.save("/home/lvorabbi/Proj/Phd/state_art/Imagenet/MyBinary/fine_tuning_model_without_prelu.h5")
#with open("/home/lvorabbi/Proj/Phd/state_art/Imagenet/MyBinary/fine_tuning_model_without_prelu.tflite", "wb") as flatbuffer_file:
#        flatbuffer_bytes = lce.convert_keras_model(model)
#        flatbuffer_file.write(flatbuffer_bytes)