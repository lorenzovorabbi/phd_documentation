import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.keras.engine import training
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
import copy
import random 
import importlib
import sys
import functools
import json
import imageio
from PIL import Image
import random
from sklearn.utils import shuffle
from scipy import interpolate
import larq as lq

module_path = os.path.abspath('C:/Proj/phd/core50_evaluation/')
if module_path not in sys.path:
    sys.path.append(module_path)
    
import BNF as bnf
import Utilities as uti
import VAL_Quantization as quant


from tensorflow.python.ops.gen_math_ops import mod

def LoadDataset(Batch_size=50):
    
    num_classes = 10

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
    test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(50000, reshuffle_each_iteration=True)
    val_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(50000, reshuffle_each_iteration=True)

    def normalization(images, labels):

        images = tf.cast(images, tf.float32)
        images = (images / 127.5) - 1
        return images, labels

    def augmentation(images, labels):

        images = tf.cast(images, tf.float32)
        images = tf.image.resize_with_crop_or_pad(images, 40, 40)
        images = tf.image.random_crop(images, [32, 32, 3])
        images = tf.image.random_flip_left_right(images)

        images = (images / 127.5) - 1.0
        return images, labels

    train_ds = train_ds.map(lambda x,y: tf.numpy_function(func=augmentation, inp=[x, y], Tout=[tf.float32, tf.float32]))
    #train_ds = train_ds.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_ds = val_ds.map(normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.batch(Batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(Batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_ds, val_ds

def CreateQuantizedBitsRepresentations(input):
    
    input_shape = tf.shape(input)  
    batch_size = input_shape[0]
    rows = input_shape[1]
    cols = input_shape[2]
    
    num_frac_bits = 7
    
    #quantized = tf.cast(input, dtype=tf.int32)
    quantized = tf.cast(tf.clip_by_value(tf.math.round(tf.cast(tf.math.pow(2, num_frac_bits), dtype=tf.float32)*(input)), -127.0, +127.0), dtype=tf.int32)
    quantized_bits = tf.cast(tf.math.floormod(tf.bitwise.right_shift(tf.expand_dims(quantized,4), tf.range(8)), 2), dtype=tf.int32)    

    # Channel 0
    ch0_bit0 = tf.slice(quantized_bits, [0, 0, 0, 0, 0], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch0_bit1 = tf.slice(quantized_bits, [0, 0, 0, 0, 1], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch0_bit2 = tf.slice(quantized_bits, [0, 0, 0, 0, 2], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch0_bit3 = tf.slice(quantized_bits, [0, 0, 0, 0, 3], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch0_bit4 = tf.slice(quantized_bits, [0, 0, 0, 0, 4], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch0_bit5 = tf.slice(quantized_bits, [0, 0, 0, 0, 5], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch0_bit6 = tf.slice(quantized_bits, [0, 0, 0, 0, 6], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch0_bit7 = tf.slice(quantized_bits, [0, 0, 0, 0, 7], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    
    # Channel 1
    ch1_bit0 = tf.slice(quantized_bits, [0, 0, 0, 1, 0], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch1_bit1 = tf.slice(quantized_bits, [0, 0, 0, 1, 1], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch1_bit2 = tf.slice(quantized_bits, [0, 0, 0, 1, 2], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch1_bit3 = tf.slice(quantized_bits, [0, 0, 0, 1, 3], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch1_bit4 = tf.slice(quantized_bits, [0, 0, 0, 1, 4], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch1_bit5 = tf.slice(quantized_bits, [0, 0, 0, 1, 5], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch1_bit6 = tf.slice(quantized_bits, [0, 0, 0, 1, 6], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch1_bit7 = tf.slice(quantized_bits, [0, 0, 0, 1, 7], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    
    # Channel 2
    ch2_bit0 = tf.slice(quantized_bits, [0, 0, 0, 2, 0], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch2_bit1 = tf.slice(quantized_bits, [0, 0, 0, 2, 1], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch2_bit2 = tf.slice(quantized_bits, [0, 0, 0, 2, 2], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch2_bit3 = tf.slice(quantized_bits, [0, 0, 0, 2, 3], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch2_bit4 = tf.slice(quantized_bits, [0, 0, 0, 2, 4], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch2_bit5 = tf.slice(quantized_bits, [0, 0, 0, 2, 5], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch2_bit6 = tf.slice(quantized_bits, [0, 0, 0, 2, 6], [input_shape[0], input_shape[1], input_shape[2], 1, 1])
    ch2_bit7 = tf.slice(quantized_bits, [0, 0, 0, 2, 7], [input_shape[0], input_shape[1], input_shape[2], 1, 1])

    ones = tf.ones_like(ch0_bit0, dtype=tf.float32)
    #ones = tf.constant(0.0, shape=tf.shape(ch0_bit0), dtype=tf.float32)
    less_one = tf.math.negative(tf.ones_like(ch0_bit0, dtype=tf.float32))
    #less_one = tf.constant(255.0, shape=tf.shape(ch0_bit0), dtype=tf.float32)

    # Convert bit representation to sign
    ch_0_bit0 = tf.where(ch0_bit0 == 1, ones, less_one)
    ch_0_bit1 = tf.where(ch0_bit1 == 1, ones, less_one)
    ch_0_bit2 = tf.where(ch0_bit2 == 1, ones, less_one)
    ch_0_bit3 = tf.where(ch0_bit3 == 1, ones, less_one)
    ch_0_bit4 = tf.where(ch0_bit4 == 1, ones, less_one)
    ch_0_bit5 = tf.where(ch0_bit5 == 1, ones, less_one)
    ch_0_bit6 = tf.where(ch0_bit6 == 1, ones, less_one)
    ch_0_bit7 = tf.where(ch0_bit7 == 1, ones, less_one)
    
    ch_1_bit0 = tf.where(ch1_bit0 == 1, ones, less_one)
    ch_1_bit1 = tf.where(ch1_bit1 == 1, ones, less_one)
    ch_1_bit2 = tf.where(ch1_bit2 == 1, ones, less_one)
    ch_1_bit3 = tf.where(ch1_bit3 == 1, ones, less_one)
    ch_1_bit4 = tf.where(ch1_bit4 == 1, ones, less_one)
    ch_1_bit5 = tf.where(ch1_bit5 == 1, ones, less_one)
    ch_1_bit6 = tf.where(ch1_bit6 == 1, ones, less_one)
    ch_1_bit7 = tf.where(ch1_bit7 == 1, ones, less_one)
    
    ch_2_bit0 = tf.where(ch2_bit0 == 1, ones, less_one)
    ch_2_bit1 = tf.where(ch2_bit1 == 1, ones, less_one)
    ch_2_bit2 = tf.where(ch2_bit2 == 1, ones, less_one)
    ch_2_bit3 = tf.where(ch2_bit3 == 1, ones, less_one)
    ch_2_bit4 = tf.where(ch2_bit4 == 1, ones, less_one)
    ch_2_bit5 = tf.where(ch2_bit5 == 1, ones, less_one)
    ch_2_bit6 = tf.where(ch2_bit6 == 1, ones, less_one)
    ch_2_bit7 = tf.where(ch2_bit7 == 1, ones, less_one)

    return [ch_0_bit0, ch_0_bit1, ch_0_bit2, ch_0_bit3, ch_0_bit4, ch_0_bit5, ch_0_bit6, ch_0_bit7], [ch_1_bit0, ch_1_bit1, ch_1_bit2, ch_1_bit3, ch_1_bit4, ch_1_bit5, ch_1_bit6, ch_1_bit7], [ch_2_bit0, ch_2_bit1, ch_2_bit2, ch_2_bit3, ch_2_bit4, ch_2_bit5, ch_2_bit6, ch_2_bit7]

def CreateBinaryInput(input_shape, num_filters=32, starting_index_slices=4, use_pointwise=False):
    
    input = tf.keras.layers.Input(shape=input_shape)
    
    bit_slices0, bit_slices1, bit_slices2 = CreateQuantizedBitsRepresentations(input)
    
    slice_shape = (input_shape[0], input_shape[1], 1)
    
    out_slices0 = []
    out_slices1 = []
    out_slices2 = []
    
    multipliers = [tf.constant(1.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32),
                   tf.constant(2.0, dtype=tf.float32), tf.constant(2.0, dtype=tf.float32),
                   tf.constant(4.0, dtype=tf.float32), tf.constant(4.0, dtype=tf.float32),
                   tf.constant(8.0, dtype=tf.float32), tf.constant(8.0, dtype=tf.float32)]
    
    momentum_value = 0.9
    
    kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
    
    # Channel 0
    for idx, item in enumerate(bit_slices0):
    
        if (idx < starting_index_slices):
        	continue
        
        x = tf.keras.layers.Reshape(slice_shape)(item)
        x = bnf.QuantConv2DMixed(num_filters, kernel_sizes[idx], activation=None, use_bias=False, 
                                input_quantizer=bnf.StdBinaryQuantXTh(), 
                                kernel_quantizer=bnf.StdBinaryQuant(),
                                bias_quantizer=bnf.StdBinaryQuant(),
                                padding='same', pad_values=1.0,
                                use_as_FP=False)(x)
        x = tf.keras.layers.BatchNormalization(center=True, scale=False)(x)
        x = bnf.DPReLU()(x)
        
        x = tf.math.multiply(x, multipliers[idx])
        
        out_slices0.append(x)
        
    # Channel 1
    for idx, item in enumerate(bit_slices1):
    
        if (idx < starting_index_slices):
        	continue
        
        x = tf.keras.layers.Reshape(slice_shape)(item)
        x = bnf.QuantConv2DMixed(num_filters, kernel_sizes[idx], activation=None, use_bias=False, 
                                input_quantizer=bnf.StdBinaryQuantXTh(), 
                                kernel_quantizer=bnf.StdBinaryQuant(),
                                bias_quantizer=bnf.StdBinaryQuant(),
                                padding='same', pad_values=1.0,
                                use_as_FP=False)(x)
        x = tf.keras.layers.BatchNormalization(center=True, scale=False)(x)
        x = bnf.DPReLU()(x)
        
        x = tf.math.multiply(x, multipliers[idx])
        
        out_slices1.append(x)
        
    # Channel 2
    for idx, item in enumerate(bit_slices2):
    
        if (idx < starting_index_slices):
        	continue
        
        x = tf.keras.layers.Reshape(slice_shape)(item)
        x = bnf.QuantConv2DMixed(num_filters, kernel_sizes[idx], activation=None, use_bias=False, 
                                input_quantizer=bnf.StdBinaryQuantXTh(), 
                                kernel_quantizer=bnf.StdBinaryQuant(),
                                bias_quantizer=bnf.StdBinaryQuant(),
                                padding='same', 
                                pad_values=1.0,
                                use_as_FP=False)(x)
        x = tf.keras.layers.BatchNormalization(center=True, scale=False)(x)
        x = bnf.DPReLU()(x)
        
        x = tf.math.multiply(x, multipliers[idx])
        
        out_slices2.append(x)
        
    ch0 = tf.keras.layers.Add()(out_slices0)
    ch0 = tf.keras.layers.BatchNormalization(center=True)(ch0)
    
    ch1 = tf.keras.layers.Add()(out_slices1)
    ch1 = tf.keras.layers.BatchNormalization(center=True)(ch1)
    
    ch2 = tf.keras.layers.Add()(out_slices2)
    ch2 = tf.keras.layers.BatchNormalization(center=True)(ch2)

    bin_input = tf.keras.layers.Concatenate(axis=-1)([ch0, ch1, ch2])
    
    # *****************************************************************************************************
    # Point-wise convolution used to correlate different features maps altogether
    # It must be 8-bit quantized
    if (use_pointwise == True):
        filters_num = num_filters*3
        #bin_input = tf.keras.layers.Conv2D(num_filters*3, 1, strides=(1, 1), activation=None, padding="same", use_bias=False)(bin_input)
        #bin_input = bnf.SymmetricLinearLeaky(1.25, 0.02)(bin_input)
        #bin_input = tf.keras.layers.BatchNormalization(scale=True, center=True, momentum=momentum_value)(bin_input)
        
        shortcut = bin_input
        x = LearnableBias(filters_num)(bin_input)
        x = bnf.QuantConv2DMixed(filters_num, (1,1), strides=(1, 1), padding="same", 
                               pad_values=1.0,
                               input_quantizer=lq.quantizers.ApproxSign(), 
                               kernel_quantizer=lq.quantizers.MagnitudeAwareSign(),
                               kernel_initializer="glorot_normal", 
                               kernel_constraint=lq.constraints.WeightClip(1.0),
                               use_bias=False, use_as_FP=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        bin_input = tf.keras.layers.add([x, shortcut])
    # *****************************************************************************************************
    
    return bin_input, input

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

    img_input, orig_input = CreateBinaryInput(input_shape, num_filters=32, starting_index_slices=0, use_pointwise=False)

    # layer 1
    #out = tf.keras.layers.Conv2D(
    #    64,
    #    3,
    #    strides=1,
    #    kernel_initializer="glorot_normal",
    #    padding="same",
    #    use_bias=False,
    #)(img_input)
    #out = tf.keras.layers.BatchNormalization(momentum=0.9)(out)

    # 64 channels
    out = block(img_input, double_filters=False, stride=1)
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

    return tf.keras.Model(inputs=orig_input, outputs=out)

if __name__ == "__main__":
    
    num_epochs = 140
    
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    
    train_ds, val_ds = LoadDataset()
    
    with strategy.scope():
        reactnet_model = Birealnet18((32, 32, 3), 10)
        reactnet_model.summary()
    
    learning_rate = 0.0025
    lr_decay = 1e-6
    lr_drop = 20

    with strategy.scope():

        def decayed_learning_rate(epoch, curr_lr):
            decay_step = 20#30
            decay_rate = 0.25
            return learning_rate * (decay_rate ** (epoch // decay_step))

        #optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # tf.keras.callbacks.LearningRateScheduler(decayed_learning_rate)
        reactnet_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=optimizer, metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy')])
        historyVGGSmall_2 = reactnet_model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[])
        
        hist_df = pd.DataFrame(historyVGGSmall_2.history) 

        # save to json:  
        hist_json_file = 'C:/Proj/phd/core50_evaluation/input_binarization/reactnet-18-history-standard-input-binary.json' 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)