import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.keras.engine import training
import tensorflow_datasets as tfds
#import tensorflow_addons as tfa
import numpy as np
#import pandas as pd
#import matplotlib
#import matplotlib.pyplot as plt
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
#import imageio
#from PIL import Image
import larq as lq
import larq_zoo as lqz
import random
#from sklearn.utils import shuffle
#from scipy import interpolate

module_path = os.path.abspath('C:\Proj\phd\core50_evaluation')
if module_path not in sys.path:
    sys.path.append(module_path)
    
import BNF as bnf
#import Utilities as uti

def binary_quantization(x):
    
    @tf.custom_gradient
    def _call2(x):

        def grad(dy):
            return dy

        zero = tf.zeros_like(x)
        one = tf.ones_like(x)
        neg_one = -tf.ones_like(x)
        output = tf.where(tf.math.greater_equal(x, zero), one, neg_one)
        
        return output, grad
    return _call2(x)

def ternary_quantization(x, th=0.05):
    
    @tf.custom_gradient
    def _call2(x):

        def grad(dy):
            return dy

        zero = tf.zeros_like(x)
        one = tf.ones_like(x)
        neg_one = -tf.ones_like(x)
        mask = tf.math.logical_and(tf.math.greater_equal(x, -th), tf.math.less_equal(x, th))
        mask1 = tf.math.less(x, -th)
        mask2 = tf.math.greater(x, th)
        output = tf.where(mask, zero, x)
        
        final_result = tf.where(mask, zero, tf.where(mask1, neg_one, tf.where(mask2, one, zero)))
        
        return final_result, grad
    return _call2(x)

@tf.function
def my_quantize2(x, min_val=None, max_val=None, num_bits=8):
    if num_bits == None:
        return x
    if min_val == None:
        min_val = tf.reduce_min(x)
    if max_val == None:
        max_val = tf.reduce_max(x)
    # In NC scenario the weights of CWR are always zero at the beginning
    if (min_val == max_val) and (min_val == 0.0):
        return x
    if (num_bits == 2):
        return ternary_quantization(x)
    elif (num_bits == 1):
        return binary_quantization(x)
    else:
        return bnf.standard_pow2_add_quantization_noise(x, min_val, max_val, quant_bits=num_bits)

@tf.function
def my_quantize(x, min_val=None, max_val=None, num_bits=8):
    if num_bits == None:
        return x
    if min_val == None:
        min_val = tf.reduce_min(x)
    if max_val == None:
        max_val = tf.reduce_max(x)
    # In NC scenario the weights of CWR are always zero at the beginning
    if (min_val == max_val) and (min_val == 0.0):
        return x
    return bnf.standard_pow2_add_quantization_noise(x, min_val, max_val, quant_bits=num_bits)

'''
def conv_backward_weights_gradients(input_data, inc_gradient, padding='VALID', padding_value=0.0, kernel_size=3, num_bits=None):
    
    input_f = tf.cast(input_data, tf.float32)
    
    tf.debugging.assert_greater(len(tf.shape(input_f)), 3, message='Input tensor must have at least 3 dimensions!')
    
    if kernel_size == 3:
        pad_dim = 1
    elif kernel_size == 1:
        pad_dim = 0
    else:
        tf.debugging.assert_equal(1, 3, message='Conv kernel size must be 3 or 1!')
    
    if len(tf.shape(input_f)) < 4:
        in_d = tf.expand_dims(input_f, axis=0) # To add batch dimension to input tensor
    else:
        in_d = input_f
    
    if padding == 'SAME':
        in_d = tf.pad(in_d, [[0, 0], [pad_dim, pad_dim], [pad_dim, pad_dim], [0, 0]], mode='CONSTANT', constant_values=padding_value)
        
    # Add batch dimension for conv3d
    in_d = tf.expand_dims(in_d, axis=0)
    
    filter_d = tf.expand_dims(inc_gradient, axis=3) # To add input_ch to filter tensor
    
    outputs = []
    for in_ch_idx in range(tf.shape(in_d)[-1].numpy()):
        res = tf.nn.conv3d(tf.expand_dims(in_d[:,:,:,:,in_ch_idx], axis=4), filter_d, strides=[1, 1, 1, 1, 1], padding='VALID')
        outputs.append(res)
    grad_weights = tf.transpose(tf.squeeze(tf.concat(outputs, axis=1), axis=0), [1, 2, 0, 3])
    
    return grad_weights
'''
def conv_backward_weights_gradients(input_data, inc_gradient, padding='VALID', padding_value=0.0, kernel_size=3, stride=1, num_bits=None, kernel_shape=None):

    if kernel_size == 3:
        pad_dim = 1
    elif kernel_size == 1:
        pad_dim = 0
    else:
        tf.debugging.assert_equal(1, 3, message='Conv kernel size must be 3 or 1!')
    
    if padding == 'SAME':
        input_data = tf.pad(input_data, [[0, 0], [pad_dim, pad_dim], [pad_dim, pad_dim], [0, 0]], mode='CONSTANT', constant_values=padding_value)
        
    if(num_bits != None):
        input_data = my_quantize(input_data, None, None, num_bits)
    
    grad_weights = tf.raw_ops.Conv2DBackpropFilter(input=input_data, filter_sizes=kernel_shape, out_backprop=inc_gradient,
                                strides=[1, stride, stride, 1], padding='VALID')
                                
    if(num_bits != None):
        grad_weights = my_quantize(grad_weights, None, None, num_bits)
        
    return grad_weights

'''
def conv_backwards_gradient_propagation(inc_gradient, W, padding='VALID', padding_value=0.0, kernel_size=3, num_bits=None):
    
    num_batches = tf.shape(inc_gradient)[0]
    filter_d = tf.expand_dims(inc_gradient, axis=3)
    
    if kernel_size == 3:
        pad_dim = 1
        pad_dim2 = 2
    elif kernel_size == 1:
        pad_dim = 0
        pad_dim2 = 0
    else:
        tf.debugging.assert_equal(1, 3, message='Conv kernel size Muste be 3 or 1!')
    
    if padding == 'SAME':
        filter_d = tf.pad(filter_d, [[0, 0], [pad_dim, pad_dim], [pad_dim, pad_dim], [0, 0], [0, 0]], mode='CONSTANT', constant_values=padding_value)
    else:
        filter_d = tf.pad(filter_d, [[0, 0], [pad_dim2, pad_dim2], [pad_dim2, pad_dim2], [0, 0], [0, 0]], mode='CONSTANT', constant_values=padding_value)
    grad_shape = tf.shape(filter_d)

    kernel_size =  tf.shape(W)[0]
    pad_dim = kernel_size // 2
    num_out_channels = tf.shape(W)[-1].numpy()

    grads_for_batch = []

    for batch_id in range(num_batches):
        grads_vs_input = []
        for in_ch_idx in range(tf.shape(W)[-2]):

            # rotate -180°
            W3 = tf.reshape(tf.experimental.numpy.rot90(W[:,:,in_ch_idx,:], -2), [1, kernel_size, kernel_size, 1, num_out_channels])

            grad_to_conv = tf.reshape(tf.expand_dims(filter_d[batch_id,:,:,:,:], axis=0), (1, 1, grad_shape[1], grad_shape[2], grad_shape[4]))
            res = tf.nn.conv3d(grad_to_conv, W3, strides=[1, 1, 1, 1, 1], padding='VALID')

            res2 = tf.math.reduce_sum(tf.squeeze(res, axis=0), axis=-1, keepdims=True)
            grads_vs_input.append(res2)

        grads_for_batch.append(tf.concat(grads_vs_input, axis=-1))

    grads_to_propagate = tf.concat(grads_for_batch, axis=0)
    return grads_to_propagate
'''

def conv_backwards_gradient_propagation(inc_gradient, W, padding='VALID', padding_value=0.0, kernel_size=3, stride=1, num_bits=None, input_shape=None):
    
    if(num_bits != None):
        W = my_quantize(W, None, None, num_bits)
    
    grads_to_propagate = tf.raw_ops.Conv2DBackpropInput(input_sizes=input_shape, filter=W, out_backprop=inc_gradient,
                               strides=[1, stride, stride, 1], padding=padding)
    
    
    return grads_to_propagate

def conv_backward(inc_grads, input_data, weights, padding='VALID', padding_value=0.0, kernel_size=3, num_bits=None):
    
    W = weights[0]
    
    padding = padding.upper()
    
    if(num_bits != None):
        inc_grads = my_quantize(inc_grads, None, None, num_bits)
    
    grad_w = conv_backward_weights_gradients(input_data, inc_grads, padding=padding, padding_value=padding_value, 
                                             kernel_size=kernel_size, num_bits=num_bits, kernel_shape=tf.shape(W))
    grad_propagate = conv_backwards_gradient_propagation(inc_grads, W, 
                                                         padding=padding, 
                                                         padding_value=padding_value, 
                                                         kernel_size=kernel_size, num_bits=num_bits,
                                                         input_shape=tf.shape(input_data))
    if(num_bits != None):
        grad_propagate = my_quantize(grad_propagate, None, None, num_bits)
    
    if len(weights) == 2:
        grad_bias = tf.math.reduce_sum(inc_grads, axis=[0, 1, 2])
        
        if(num_bits != None):
            grad_bias = my_quantize(grad_bias, None, None, num_bits)
        
        return grad_propagate, [grad_w, grad_bias]
    else:
        return grad_propagate, [grad_w]
        
def depthwiseconv_backwards_gradient_propagation(inc_gradient, W, padding='VALID', stride=2, num_bits=None, input_shape=None):
    
    if(num_bits != None):
        W = my_quantize(W, None, None, num_bits)
                               
                               
    grads_to_propagate = tf.raw_ops.DepthwiseConv2dNativeBackpropInput(input_sizes=input_shape, 
                                              filter=W,
                                              out_backprop=inc_gradient,
                                              strides=[1, stride, stride, 1],
                                              padding=padding)
    
    return grads_to_propagate
    
def depthwiseconv_backward_weights_gradients(input_data, inc_gradient, padding='VALID', padding_value=0.0, stride=2, num_bits=None, kernel_shape=None):

    if kernel_shape[0] == 3:
        pad_dim = 1
    elif kernel_shape[0] == 1:
        pad_dim = 0
    else:
        tf.debugging.assert_equal(1, 3, message='Conv kernel size must be 3 or 1!')
    
    if padding == 'SAME':
        input_data = tf.pad(input_data, [[0, 0], [pad_dim, pad_dim], [pad_dim, pad_dim], [0, 0]], mode='CONSTANT', constant_values=padding_value)
        
    if(num_bits != None):
        input_data = my_quantize(input_data, None, None, num_bits)
    
    grad_weights = tf.raw_ops.DepthwiseConv2dNativeBackpropFilter(input=input_data,
                                               filter_sizes=kernel_shape,
                                               out_backprop=inc_gradient,
                                               strides=[1, stride, stride, 1],
                                               padding='VALID')
                                
    if(num_bits != None):
        grad_weights = my_quantize(grad_weights, None, None, num_bits)
        
    return grad_weights
        
def depthwiseconv_backward(inc_grads, input_data, weights, padding='SAME', padding_value=0.0, stride=2, num_bits=None):

    W = weights[0]
    
    padding = padding.upper()
    
    if(num_bits != None):
        inc_grads = my_quantize(inc_grads, None, None, num_bits)
    
    grad_w = depthwiseconv_backward_weights_gradients(input_data, inc_grads, padding=padding, padding_value=padding_value, 
                                                      stride=stride, num_bits=num_bits, kernel_shape=tf.shape(W))
    grad_propagate = depthwiseconv_backwards_gradient_propagation(inc_grads, W, 
                                                         padding=padding, 
                                                         stride=stride, num_bits=num_bits,
                                                         input_shape=tf.shape(input_data))
    if(num_bits != None):
        grad_propagate = my_quantize(grad_propagate, None, None, num_bits)
    
    if len(weights) == 2:
        grad_bias = tf.math.reduce_sum(inc_grads, axis=[0, 1, 2])
        
        if(num_bits != None):
            grad_bias = my_quantize(grad_bias, None, None, num_bits)
        
        return grad_propagate, [grad_w, grad_bias]
    else:
        return grad_propagate, [grad_w]
    
    
def ste_backward(inc_grads, input_not_binary, clip_value):
    
    mask = tf.math.less_equal(tf.math.abs(input_not_binary), clip_value)
    return tf.where(mask, inc_grads, tf.zeros_like(inc_grads))

def approxsign_backward(inc_grads, input_not_binary, clip_value=1.0):
    abs_input = tf.math.abs(input_not_binary)
    zeros = tf.zeros_like(inc_grads)
    mask = tf.math.less_equal(abs_input, clip_value)
    return tf.where(mask, (clip_value - abs_input) * 2 * inc_grads, zeros)

def magnitude_aware_sign_backward(inc_grads, input_not_binary, scale_factor, clip_value=1.0):
    return ste_backward(inc_grads, input_not_binary, clip_value)*scale_factor

def conv_backward_binary_ste(inc_grads, input_data, weights, padding='VALID', 
                             kernel_bin='ste', kernel_clip_value=1.0, 
                             input_bin='ste', input_clip_value=1.0,
                             padding_value=1.0,
                             kernel_size=3,
                             num_bits=None,
                             num_bits_binary=8):
    
    W = weights[0]
    
    padding = padding.upper()
    
    if input_bin == 'ste':
        input_data_binarized = bnf.StdBinaryQuant(input_clip_value)(input_data)
    else:
        input_data_binarized = input_data
        
    if kernel_bin == 'ste':
        kernel_binarized = bnf.StdBinaryQuant(kernel_clip_value)(W)
    else:
        kernel_binarized = W
    
    # !!! In forward padding is applied as set by user, in backward it's used as 0!!!
    grad_w = conv_backward_weights_gradients(input_data_binarized, inc_grads, padding=padding, 
                                             padding_value=1.0, kernel_size=kernel_size, num_bits=num_bits, 
                                             kernel_shape=tf.shape(W))
    grad_propagate = conv_backwards_gradient_propagation(inc_grads, kernel_binarized, 
                                                         padding=padding, 
                                                         padding_value=padding_value, 
                                                         kernel_size=kernel_size, num_bits=num_bits,
                                                         input_shape=tf.shape(input_data))
    
    if kernel_bin == 'ste':
        grad_w = ste_backward(grad_w, W, kernel_clip_value)
        
    if input_bin == 'ste':
        grad_propagate = ste_backward(grad_propagate, input_data, input_clip_value)
        
    if(num_bits != None):
        grad_propagate = my_quantize(grad_propagate, None, None, num_bits)
    
    if len(weights) == 2:
        grad_bias = tf.math.reduce_sum(inc_grads, axis=[0, 1, 2])
        
        if(num_bits != None):
            grad_bias = my_quantize(grad_bias, None, None, num_bits)
        
        return grad_propagate, [grad_w, grad_bias]
    else:
        return grad_propagate, [grad_w]
    
def conv_backward_binary_reactnet(inc_grads, input_data, weights, padding='SAME', 
                             kernel_bin='magnitudeawaresign', kernel_clip_value=1.0, 
                             input_bin='approxsign', input_clip_value=1.0,
                             stride=1,
                             padding_value=1.0,
                             kernel_size=3,
                             num_bits=None,
                             num_bits_binary=8):
    
    W = weights[0]
    
    padding = padding.upper()

    # Magnitude aware binarization
    scale_factor = tf.reduce_mean(tf.abs(W), axis=list(range(len(W.shape) - 1)))
    
    if (input_bin == 'approxsign') or (input_bin == 'ste'):
        input_data_binarized = bnf.StdBinaryQuant(input_clip_value)(input_data)
    else:
        input_data_binarized = input_data
        
    if kernel_bin == 'magnitudeawaresign':
        kernel_binarized = bnf.StdBinaryQuant(kernel_clip_value)(W)*scale_factor
    else:
        kernel_binarized = W
    
    # !!! In forward padding is applied as set by user, in backward it's used as 0!!!
    grad_w = conv_backward_weights_gradients(input_data_binarized, inc_grads, padding=padding, 
                                             padding_value=0.0, kernel_size=kernel_size, num_bits=num_bits_binary, 
                                             kernel_shape=tf.shape(W), stride=stride)
    grad_propagate = conv_backwards_gradient_propagation(inc_grads, kernel_binarized, 
                                                     padding=padding, 
                                                     padding_value=padding_value, 
                                                     kernel_size=kernel_size, num_bits=num_bits,
                                                     input_shape=tf.shape(input_data), stride=stride)
    
    if kernel_bin == 'magnitudeawaresign':
        grad_w = magnitude_aware_sign_backward(grad_w, W, scale_factor, kernel_clip_value)
        
    if input_bin == 'approxsign':
        grad_propagate = approxsign_backward(grad_propagate, input_data, input_clip_value)
    elif input_bin == 'ste':
        grad_propagate = ste_backward(grad_propagate, input_data, input_clip_value)
        
    if(num_bits != None):
        grad_propagate = my_quantize(grad_propagate, None, None, num_bits)
    
    if len(weights) == 2:
        grad_bias = tf.math.reduce_sum(inc_grads, axis=[0, 1, 2])
        
        if(num_bits != None):
            grad_bias = my_quantize(grad_bias, None, None, num_bits_binary)
        
        return grad_propagate, [grad_w, grad_bias]
    else:
        return grad_propagate, [grad_w]
    
def conv_backward_binary_birealnet(inc_grads, input_data, weights, padding='SAME', 
                             kernel_bin='x-th', kernel_clip_value=1.0, x_offset_kernel=0.0,
                             input_bin='x-th', input_clip_value=1.0, x_offset_input=0.0,
                             stride=1,
                             padding_value=1.0,
                             kernel_size=3,
                             num_bits=None,
                             num_bits_binary=8):
    
    W = weights[0]
    
    padding = padding.upper()

    # Magnitude aware binarization
    scale_factor = tf.reduce_mean(tf.abs(W), axis=list(range(len(W.shape) - 1)))
    
    if input_bin == 'x-th':
        input_data_binarized = bnf.StdBinaryQuantXTh(clip_value=input_clip_value, x_offset=x_offset_input)(input_data)
    else:
        input_data_binarized = input_data
        
    if kernel_bin == 'x-th':
        kernel_binarized = bnf.StdBinaryQuantXTh(clip_value=input_clip_value, x_offset=x_offset_kernel)(W)
    else:
        kernel_binarized = W
    
    # !!! In forward padding is applied as set by user, in backward it's used as 0!!!
    grad_w = conv_backward_weights_gradients(input_data_binarized, inc_grads, padding=padding, 
                                             padding_value=1.0, kernel_size=kernel_size, num_bits=num_bits_binary, 
                                             kernel_shape=tf.shape(W), stride=stride)
    grad_propagate = conv_backwards_gradient_propagation(inc_grads, kernel_binarized, 
                                                     padding=padding, 
                                                     padding_value=padding_value, 
                                                     kernel_size=kernel_size, num_bits=num_bits,
                                                     input_shape=tf.shape(input_data), stride=stride)
    
    if kernel_bin == 'x-th':
        grad_w = ste_backward(grad_w, W+x_offset_kernel, kernel_clip_value)
        
    if input_bin == 'x-th':
        grad_propagate = ste_backward(grad_propagate, input_data+x_offset_input, input_clip_value)
        
    if(num_bits != None):
        grad_propagate = my_quantize(grad_propagate, None, None, num_bits)
        
    grad_offset_kernel = tf.math.reduce_sum(grad_w, axis=[0, 1, 2])
    grad_offset_input = tf.math.reduce_sum(grad_propagate, axis=[0, 1, 2])
    
    if len(weights) == 2:
        grad_bias = tf.math.reduce_sum(inc_grads, axis=[0, 1, 2])
        
        if(num_bits != None):
            grad_bias = my_quantize(grad_bias, None, None, num_bits_binary)
        
        return grad_propagate, [grad_w, grad_bias, grad_offset_kernel, grad_offset_input]
    else:
        return grad_propagate, [grad_w, grad_offset_kernel, grad_offset_input]
    
def max_pool2d_backward(input_maxpool, output_maxpool, inc_gradients, pool_size=2, stride=2, padding="VALID", num_bits=None):
    
    out_grad = tf.raw_ops.MaxPoolGrad(orig_input=input_maxpool, orig_output=output_maxpool, grad=inc_gradients, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding=padding)
    
    # quantize gradient
    if(num_bits != None):
        out_grad = my_quantize(out_grad, None, None, num_bits)
    
    return out_grad

def global_avg_pooling(input_to_pooling, inc_gradients, num_bits=None):
    
    in_shape = tf.shape(input_to_pooling)
    # Adjust to 4-D the dimensions of global pooling gradients
    pool_grad_out = tf.expand_dims(tf.expand_dims(inc_gradients, axis=2), axis=3)
    grad_to_prop = tf.raw_ops.AvgPoolGrad(orig_input_shape=in_shape, grad=pool_grad_out, ksize=[1, in_shape[1], in_shape[2], 1], strides=[1, in_shape[1], in_shape[2], 1], padding='VALID')
    
    # quantize gradient
    if(num_bits != None):
        grad_to_prop = my_quantize(grad_to_prop, None, None, num_bits)
    
    return grad_to_prop

def avg_pooling(input_to_pooling, inc_gradients, layer, num_bits=None):
    
    in_shape = tf.shape(input_to_pooling)
    
    grad_to_prop = tf.raw_ops.AvgPoolGrad(orig_input_shape=in_shape, grad=inc_gradients, ksize=[1, layer.pool_size[0], layer.pool_size[1], 1], strides=[1, layer.strides[0], layer.strides[1], 1], padding='VALID')
    
    # quantize gradient
    if(num_bits != None):
        grad_to_prop = my_quantize(grad_to_prop, None, None, num_bits)
    
    return grad_to_prop

def batch_norm_forward(layer, input_data, is_training=True):
    
    in_shape = tf.shape(input_data)
    
    axes_for_bn = list(range(0,len(tf.shape(input_data))-1))
    
    #param_shape = [axes_for_bn.index(i) if i in axes_for_bn else 1 for i in range(len(tf.shape(input_data)))]
    
    eps = layer.epsilon
    
    if is_training == True:
        
        in_mean = tf.math.reduce_mean(input_data, axis=axes_for_bn, keepdims=True)
        in_var = tf.math.reduce_variance(input_data, axis=axes_for_bn, keepdims=True)
        
        input_standardized = (input_data - in_mean) * tf.math.rsqrt(in_var + eps)
        
        cache = (in_mean, in_var, input_standardized, axes_for_bn)
        
        # Update moving_average
        layer.moving_mean.assign(layer.moving_mean * layer.momentum + tf.math.reduce_mean(input_data, axis=axes_for_bn, keepdims=False) * (1 - layer.momentum))

        # Update moving variance
        layer.moving_variance.assign(layer.moving_variance * layer.momentum + tf.math.reduce_variance(input_data, axis=axes_for_bn, keepdims=False) * (1 - layer.momentum))
        
    else:
        
        input_standardized = (input_data - layer.moving_mean) * tf.math.rsqrt(layer.moving_variance + eps)
        
        cache = None
        
    output = layer.gamma * input_standardized + layer.beta
        
    return output, cache

def batch_norm_backward(layer, inc_grads, cache, num_bits=None):
    
    (in_mean, in_var, input_standardized, axes_for_bn) = cache
    
    grad_gamma = tf.math.reduce_sum(input_standardized * inc_grads, axis=axes_for_bn)
    
    grad_beta = tf.math.reduce_sum(inc_grads, axis=axes_for_bn)
    
    N = 1.0
    for axes_id in axes_for_bn:
        N *= tf.cast(tf.shape(input_standardized)[axes_id], tf.float32)

    grad_to_bn_input = 1.0/N * layer.gamma * tf.math.rsqrt(in_var + layer.epsilon) * (-grad_gamma*input_standardized + (N*inc_grads) -(grad_beta))    
    
    out_grads = [grad_to_bn_input]
    
    if layer.scale == True:
        out_grads.append(grad_gamma)
        
    if layer.center == True:
        out_grads.append(grad_beta)
    
    return out_grads


def relu_backward(inc_grads, input_data, num_bits=None):
    return tf.where(tf.math.less_equal(input_data, 0.0), tf.zeros_like(inc_grads), inc_grads)

class Shortcut_BinConv_BN(tf.keras.layers.Layer):
    
    def __init__(self, conv_layer, bn_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, bin_weights_quant=None, **kwargs):
        super().__init__(**kwargs)
        
        self.conv = conv_layer
        self.bn = bn_layer
        self.relu = tf.keras.layers.ReLU()
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs = None
        self.out_conv = None
        self.out_relu = None
        self.bn_cache = None
        self.bn_grads = None
        self.conv_grads = None
        self.bin_weights_quant = bin_weights_quant
        
    def call(self, inputs, training=None):
    
        if(self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
            self.conv.weights[0].assign(my_quantize(self.conv.weights[0], None, None, self.bin_weights_quant))
        
        self.inputs = inputs
        self.out_conv = self.conv(self.inputs)
        
        if(self.quant_fw != None):
            self.out_conv = my_quantize(self.out_conv, None, None, self.quant_fw)
        
        self.out_relu = self.relu(self.out_conv)
        out_bn, self.bn_cache = batch_norm_forward(self.bn, self.out_relu, is_training=training)
        shortcut_out = out_bn + self.inputs
        
        if(self.quant_fw != None):
            shortcut_out = my_quantize(shortcut_out, None, None, self.quant_fw)
        
        return shortcut_out
    
    def backward(self, inc_grads):
        
        self.bn_grads = batch_norm_backward(self.bn, inc_grads, self.bn_cache)
        grad_relu_input = relu_backward(self.bn_grads[0], self.out_conv)
        input_conv_grad, self.conv_grads = conv_backward_binary_ste(grad_relu_input, self.inputs, 
                                                                    self.conv.weights, padding=self.conv.padding,
                                                                    padding_value=0.0,
                                                                    kernel_clip_value=self.conv.kernel_quantizer.clip_value,
                                                                    input_clip_value=self.conv.input_quantizer.clip_value,
                                                                    kernel_size=self.conv.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
        
        return input_conv_grad + inc_grads
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        # Update weights
        if self.bn.scale == True:
            self.bn.gamma.assign(self.bn.gamma - curr_lr*self.bn_grads[1])
        
        if self.bn.center == True:
            self.bn.beta.assign(self.bn.beta - curr_lr*self.bn_grads[-1])
            
        if(self.bin_weights_quant != None):
            curr_w = my_quantize2(self.conv.weights[0], None, None, self.bin_weights_quant)
            self.conv.weights[0].assign(my_quantize2(curr_w - curr_lr*self.conv_grads[0], None, None, self.bin_weights_quant))
        else:
            self.conv.weights[0].assign(self.conv.weights[0] - curr_lr*self.conv_grads[0])
            
        if self.conv.use_bias == True:
            if(self.bin_weights_quant != None):
                curr_b = my_quantize2(self.conv.weights[1], None, None, self.bin_weights_quant)
                self.conv.weights[1].assign(my_quantize2(curr_b - curr_lr*self.conv_grads[1], None, None, self.bin_weights_quant))
            else:
                self.conv.weights[1].assign(self.conv.weights[1] - curr_lr*self.conv_grads[1])
            
        # Clip weights after gradient update
        clip_val = self.conv.kernel_quantizer.clip_value
        self.conv.weights[0].assign(tf.clip_by_value(self.conv.weights[0], -clip_val, clip_val))
        
class ReactnetBlock_BinConv_BN_Bias_PRelu(tf.keras.layers.Layer):
    
    def __init__(self, conv_layer, bn_layer, bias1, bias2, prelu, bias3, num_quant_bits_fw=None, num_quant_bits_bw=None, bin_weights_quant=None, **kwargs):
        super().__init__(**kwargs)
        
        self.conv = conv_layer
        self.bn = BN_layer(bn_layer, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.prelu = PReLU_layer(prelu, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias1 = LearnableBias_layer(bias1, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias2 = LearnableBias_layer(bias2, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias3 = LearnableBias_layer(bias3, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.bin_weights_quant = bin_weights_quant
        
        self.inputs = None
        self.out_bias1 = None
        self.out_conv = None
        self.out_bn = None
        self.out_add = None
        self.out_bias2 = None
        self.out_prelu = None
        self.out_bias3 = None
        
        self.conv_grads = None        
        
    def call(self, inputs, training=None):
    
        if(self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
        
        self.inputs = inputs
        
        self.out_bias1 = self.bias1(self.inputs)
        
        self.out_conv = self.conv(self.out_bias1)
        if(self.quant_fw != None):
            self.out_conv = my_quantize(self.out_conv, None, None, self.quant_fw)
        
        self.out_bn = self.bn(self.out_conv, training=training)
        
        self.out_add = self.out_bn + self.inputs
        
        self.out_bias2 = self.bias2(self.out_add)
            
        self.out_prelu = self.prelu(self.out_bias2)
        
        self.out_bias3 = self.bias3(self.out_prelu)
        
        return self.out_bias3
    
    def backward(self, inc_grads):
        
        grad_to_prop = self.bias3.backward(inc_grads)
        grad_to_prop = self.prelu.backward(grad_to_prop)
        grad_to_prop = self.bias2.backward(grad_to_prop)
        grad_before_shortcut = grad_to_prop
        grad_to_prop = self.bn.backward(grad_to_prop)
        
        grad_to_prop, self.conv_grads = conv_backward_binary_reactnet(grad_to_prop, self.out_bias1, 
                                                                    self.conv.weights, padding=self.conv.padding,
                                                                    padding_value=0.0,
                                                                    stride=self.conv.strides[0],
                                                                    kernel_clip_value=self.conv.kernel_quantizer.clip_value,
                                                                    input_clip_value=1.0,
                                                                    kernel_size=self.conv.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
        grad_to_prop = self.bias1.backward(grad_to_prop)
        
        return grad_before_shortcut + grad_to_prop
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        self.bn.update_weights(optimizer)
        self.bias3.update_weights(optimizer)
        self.bias2.update_weights(optimizer)
        self.bias1.update_weights(optimizer)
        self.prelu.update_weights(optimizer)
            
        if(self.bin_weights_quant != None):
            curr_w = my_quantize2(self.conv.weights[0], None, None, 1)
            self.conv.weights[0].assign(my_quantize2(curr_w - curr_lr*self.conv_grads[0], None, None, self.bin_weights_quant))
        else:
            self.conv.weights[0].assign(self.conv.weights[0] - curr_lr*self.conv_grads[0])
            
        if self.conv.use_bias == True:
            if(self.bin_weights_quant != None):
                curr_b = my_quantize2(self.conv.weights[1], None, None, self.bin_weights_quant)
                self.conv.weights[1].assign(my_quantize2(curr_b - curr_lr*self.conv_grads[1], None, None, self.bin_weights_quant))
            else:
                self.conv.weights[1].assign(self.conv.weights[1] - curr_lr*self.conv_grads[1])
            
        # Clip weights after gradient update
        clip_val = self.conv.kernel_quantizer.clip_value
        self.conv.weights[0].assign(tf.clip_by_value(self.conv.weights[0], -clip_val, clip_val))
        
class ReactnetBlock_BinConv_BN_Bias_PRelu_Avg_pooling(tf.keras.layers.Layer):
    
    def __init__(self, conv_layer, bn_layer, bias1, bias2, prelu, bias3, avg_layer, avg_conv_layer, avg_bn_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, bin_weights_quant=None, **kwargs):
        super().__init__(**kwargs)
        
        self.conv = conv_layer
        self.bn = BN_layer(bn_layer, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.prelu = PReLU_layer(prelu, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias1 = LearnableBias_layer(bias1, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias2 = LearnableBias_layer(bias2, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias3 = LearnableBias_layer(bias3, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.avg_layer = avg_layer
        self.avg_conv_layer = avg_conv_layer
        self.avg_bn_layer = BN_layer(avg_bn_layer, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.bin_weights_quant = bin_weights_quant
        
        self.inputs = None
        self.out_bias1 = None
        self.out_conv = None
        self.out_bn = None
        self.out_add = None
        self.out_bias2 = None
        self.out_prelu = None
        self.out_bias3 = None
        
        self.out_avg = None
        self.out_avg_conv = None
        self.out_avg_bn = None
        
        self.conv_grads = None        
        self.avg_conv_grads = None
        
    def call(self, inputs, training=None):
    
        if(self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
        
        self.inputs = inputs
        
        self.out_bias1 = self.bias1(self.inputs)
        
        self.out_conv = self.conv(self.out_bias1)
        if(self.quant_fw != None):
            self.out_conv = my_quantize(self.out_conv, None, None, self.quant_fw)
        
        self.out_bn = self.bn(self.out_conv, training=training)
        
        # avg branch
        self.out_avg = self.avg_layer(self.inputs)
        self.out_avg_conv = self.avg_conv_layer(self.out_avg)
        self.out_avg_bn = self.avg_bn_layer(self.out_avg_conv, training=training)
        
        self.out_add = self.out_bn + self.out_avg_bn
        
        self.out_bias2 = self.bias2(self.out_add)
            
        self.out_prelu = self.prelu(self.out_bias2)
        
        self.out_bias3 = self.bias3(self.out_prelu)
        
        return self.out_bias3
    
    def backward(self, inc_grads):
        
        grad_to_prop = self.bias3.backward(inc_grads)
        grad_to_prop = self.prelu.backward(grad_to_prop)
        grad_to_prop = self.bias2.backward(grad_to_prop)
        grad_before_shortcut = grad_to_prop
        self.prova = grad_to_prop
        grad_to_prop = self.bn.backward(grad_before_shortcut)
        
        grad_to_prop, self.conv_grads = conv_backward_binary_reactnet(grad_to_prop, self.out_bias1, 
                                                                    self.conv.weights, padding=self.conv.padding,
                                                                    padding_value=0.0,
                                                                    kernel_clip_value=self.conv.kernel_quantizer.clip_value,
                                                                    input_clip_value=1.0,
                                                                    stride=self.conv.strides[0],
                                                                    kernel_size=self.conv.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
        grad_to_prop = self.bias1.backward(grad_to_prop)
        
        grad_to_prop_avg = self.avg_bn_layer.backward(grad_before_shortcut)
        grad_to_prop_avg, self.avg_conv_grads = conv_backward_binary_reactnet(grad_to_prop_avg, self.out_avg, 
                                                                    self.avg_conv_layer.weights, padding=self.avg_conv_layer.padding,
                                                                    padding_value=0.0,
                                                                    stride=self.avg_conv_layer.strides[0],
                                                                    input_bin=None,
                                                                    kernel_clip_value=self.avg_conv_layer.kernel_quantizer.clip_value,
                                                                    kernel_size=self.avg_conv_layer.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
        grad_to_prop_avg = avg_pooling(self.inputs, grad_to_prop_avg, self.avg_layer, num_bits=self.quant_bw)
        
        return grad_to_prop_avg + grad_to_prop
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        self.bn.update_weights(optimizer)
        self.bias3.update_weights(optimizer)
        self.bias2.update_weights(optimizer)
        self.bias1.update_weights(optimizer)
        self.prelu.update_weights(optimizer)
        
        self.avg_bn_layer.update_weights(optimizer)
            
        if(self.bin_weights_quant != None):
            curr_w = my_quantize2(self.conv.weights[0], None, None, self.bin_weights_quant)
            self.conv.weights[0].assign(my_quantize2(curr_w - curr_lr*self.conv_grads[0], None, None, self.bin_weights_quant))
            
            curr_w = my_quantize2(self.avg_conv_layer.weights[0], None, None, self.bin_weights_quant)
            self.avg_conv_layer.weights[0].assign(my_quantize2(curr_w - curr_lr*self.avg_conv_grads[0], None, None, self.bin_weights_quant))
        else:
            self.conv.weights[0].assign(self.conv.weights[0] - curr_lr*self.conv_grads[0])
            self.avg_conv_layer.weights[0].assign(self.avg_conv_layer.weights[0] - curr_lr*self.avg_conv_grads[0])
            
        if self.conv.use_bias == True:
            if(self.bin_weights_quant != None):
                curr_b = my_quantize2(self.conv.weights[1], None, None, self.bin_weights_quant)
                self.conv.weights[1].assign(my_quantize2(curr_b - curr_lr*self.conv_grads[1], None, None, self.bin_weights_quant))
                
                curr_b = my_quantize2(self.avg_conv_layer.weights[1], None, None, self.bin_weights_quant)
                self.avg_conv_layer.weights[1].assign(my_quantize2(curr_b - curr_lr*self.avg_conv_grads[1], None, None, self.bin_weights_quant))
            else:
                self.conv.weights[1].assign(self.conv.weights[1] - curr_lr*self.conv_grads[1])
                self.avg_conv_layer.weights[1].assign(self.avg_conv_layer.weights[1] - curr_lr*self.avg_conv_grads[1])
            
        # Clip weights after gradient update
        clip_val = self.conv.kernel_quantizer.clip_value
        self.conv.weights[0].assign(tf.clip_by_value(self.conv.weights[0], -clip_val, clip_val))
        
        clip_val = self.avg_conv_layer.kernel_quantizer.clip_value
        self.avg_conv_layer.weights[0].assign(tf.clip_by_value(self.avg_conv_layer.weights[0], -clip_val, clip_val))
        
class RepBNNBlock(tf.keras.layers.Layer):
    
    def __init__(self, conv_layer1, conv_layer2, conv_layer3, conv_layer4, bias1, bias2, bias3, bias4, bias5, bias6, 
                 repeat_factor, bn_layer1, bn_layer2, bn_layer3, bn_layer4, avg_layer, 
                 prelu1, prelu2, double_filters, override_stride, 
                 num_quant_bits_fw=None, num_quant_bits_bw=None, bin_weights_quant=None, **kwargs):
        super().__init__(**kwargs)
        
        self.conv1 = conv_layer1
        self.conv2 = conv_layer2
        self.conv3 = conv_layer3
        self.conv4 = conv_layer4
        
        self.bias1 = LearnableBias_layer(bias1, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias2 = LearnableBias_layer(bias2, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias3 = LearnableBias_layer(bias3, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias4 = LearnableBias_layer(bias4, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias5 = LearnableBias_layer(bias5, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bias6 = LearnableBias_layer(bias6, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        
        self.repeat1 = Repeat_layer(repeat_factor, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.repeat2 = Repeat_layer(repeat_factor, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.repeat3 = Repeat_layer(repeat_factor, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.repeat4 = Repeat_layer(repeat_factor, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        
        self.bn1 = BN_layer(bn_layer1, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bn2 = BN_layer(bn_layer2, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bn3 = BN_layer(bn_layer3, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.bn4 = BN_layer(bn_layer4, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        
        self.avg_layer = avg_layer
        
        self.prelu1 = PReLU_layer(prelu1, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.prelu2 = PReLU_layer(prelu2, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        
        self.double_filters = double_filters
        self.override_stride = override_stride
        
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.bin_weights_quant = bin_weights_quant
        
        self.inputs = None
        self.shortcut = None
        self.in_filters = None
        self.out_filters = None
        self.stride = None
        self.out_bias1 = None
        self.out_conv1 = None
        self.out_final_shape = None
        
        self.out_bn1 = None
        self.out_add1 = None
        self.out_before_branch = None
        self.out_before_branch_2 = None
        self.out_branch = None
        
        self.out = None
        
        # Gradients
        self.conv_grads1 = None
        self.conv_grads2 = None
        self.conv_grads3 = None
        self.conv_grads4 = None

        self.grad_before_shortcut1 = None
        self.grad_before_shortcut21 = None
        self.grad_before_shortcut22 = None
        
    def call(self, inputs, training=None):
    
        if(self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
            
        self.in_filters = inputs.get_shape()[-1]
        self.out_filters = self.in_filters if not self.double_filters else 2 * self.in_filters
        self.stride = 1 if (self.override_stride or not self.double_filters) else 2
        
        self.inputs = inputs
        self.shortcut = inputs
        
        if(self.stride == 2):
            self.shortcut = self.avg_layer(self.shortcut)
        
        self.out_bias1 = self.bias1(self.inputs)
        self.out_conv1 = self.conv1(self.out_bias1)
        if(self.quant_fw != None):
            self.out_conv1 = my_quantize(self.out_conv1, None, None, self.quant_fw)
            
        x = self.repeat1(self.out_conv1)
        self.out_bn1 = self.bn1(x, training=training)
        
        self.out_add1 = self.out_bn1 + self.shortcut
        x = self.bias2(self.out_add1)
        x = self.prelu1(x)
        self.out_before_branch_2 = self.bias3(x)
        
        self.out_before_branch = self.bias4(self.out_before_branch_2)
        
        if(self.in_filters == self.out_filters):
            x = self.conv2(self.out_before_branch)
            x = self.repeat2(x)
            x = self.bn2(x, training=training)
            self.out_branch = self.out_before_branch_2 + x
        else:
            x21 = self.conv3(self.out_before_branch)
            x22 = self.conv4(self.out_before_branch)
            x21 = self.repeat3(x21)
            x22 = self.repeat3(x22)
            x21 = self.bn3(x21, training=training)
            x22 = self.bn4(x22, training=training)
            x21 = self.out_before_branch_2 + x21
            x22 = self.out_before_branch_2 + x22
            self.out_branch = tf.concat([x21, x22], axis=-1)
            
        out = self.bias5(self.out_branch)
        out = self.prelu2(out)
        self.out = self.bias6(out)
        self.out_final_shape = self.out.get_shape()
        return self.out
    
    def backward(self, inc_grads):

        # Reshape in case next layer is a global average layer
        grad_to_prop = tf.reshape(inc_grads, self.out_final_shape)

        grad_to_prop = self.bias6.backward(grad_to_prop)
        grad_to_prop = self.prelu2.backward(grad_to_prop)
        grad_to_prop = self.bias5.backward(grad_to_prop)
        #grad_x2 = None
        
        if(self.in_filters == self.out_filters):
            self.grad_before_shortcut1 = grad_to_prop
            grad_to_prop = self.bn2.backward(self.grad_before_shortcut1)
            grad_to_prop = self.repeat2.backward(grad_to_prop)
            grad_to_prop, self.conv_grads2 = conv_backward_binary_reactnet(grad_to_prop, self.out_before_branch, 
                                                                    self.conv2.weights, padding=self.conv2.padding,
                                                                    padding_value=0.0,
                                                                    input_bin='ste',
                                                                    kernel_clip_value=self.conv2.kernel_quantizer.clip_value,
                                                                    input_clip_value=1.0,
                                                                    stride=self.conv2.strides[0],
                                                                    kernel_size=self.conv2.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
            grad_to_prop = self.bias4.backward(grad_to_prop)
            
        else:
            half_channel = int(grad_to_prop.get_shape()[-1]/2)
            grad_x21 = grad_to_prop[:,:,:,:half_channel]
            grad_x22 = grad_to_prop[:,:,:,half_channel:]
            
            self.grad_before_shortcut21 = grad_x21
            self.grad_before_shortcut22 = grad_x22
            
            grad_x21 = self.bn3.backward(grad_x21)
            grad_x22 = self.bn4.backward(grad_x22)
            
            grad_x21 = self.repeat3.backward(grad_x21)
            grad_x22 = self.repeat3.backward(grad_x22)
            
            grad_x21, self.conv_grads3 = conv_backward_binary_reactnet(grad_x21, self.out_before_branch, 
                                                                    self.conv3.weights, padding=self.conv3.padding,
                                                                    padding_value=0.0,
                                                                    input_bin='ste',
                                                                    kernel_clip_value=self.conv3.kernel_quantizer.clip_value,
                                                                    input_clip_value=1.0,
                                                                    stride=self.conv3.strides[0],
                                                                    kernel_size=self.conv3.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
            grad_x22, self.conv_grads4 = conv_backward_binary_reactnet(grad_x22, self.out_before_branch, 
                                                                    self.conv4.weights, padding=self.conv4.padding,
                                                                    padding_value=0.0,
                                                                    input_bin='ste',
                                                                    kernel_clip_value=self.conv4.kernel_quantizer.clip_value,
                                                                    input_clip_value=1.0,
                                                                    stride=self.conv4.strides[0],
                                                                    kernel_size=self.conv4.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
            
            # Concatenation operation
            grad_x2 = tf.math.add(grad_x21, grad_x22) 
            grad_to_prop = self.bias4.backward(grad_x2)
            
        if(self.in_filters == self.out_filters):
            grad_to_prop = tf.math.add(grad_to_prop, self.grad_before_shortcut1)                  
        else:
            #grad_to_prop = grad_to_prop + self.grad_before_shortcut21 + self.grad_before_shortcut22
            grad_to_prop = tf.math.add(grad_to_prop, self.grad_before_shortcut21) 
            grad_to_prop = tf.math.add(grad_to_prop, self.grad_before_shortcut22) 
        
        grad_to_prop = self.bias3.backward(grad_to_prop)
        grad_to_prop = self.prelu1.backward(grad_to_prop)
        grad_to_prop = self.bias2.backward(grad_to_prop)        
        grad_before_shortcut2 = grad_to_prop


        grad_to_prop = self.bn1.backward(grad_before_shortcut2)
        grad_to_prop = self.repeat1.backward(grad_to_prop)
        grad_to_prop, self.conv_grads1 = conv_backward_binary_reactnet(grad_to_prop, self.out_bias1, 
                                                                    self.conv1.weights, padding=self.conv1.padding,
                                                                    padding_value=1.0,
                                                                    input_bin='ste',
                                                                    kernel_clip_value=self.conv1.kernel_quantizer.clip_value,
                                                                    input_clip_value=1.0,
                                                                    stride=self.conv1.strides[0],
                                                                    kernel_size=self.conv1.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
        grad_to_prop_r = self.bias1.backward(grad_to_prop) 
        
        if(self.stride == 2):
            grad_to_prop = avg_pooling(self.inputs, grad_before_shortcut2, self.avg_layer, num_bits=self.quant_bw)

            #grad_to_prop = grad_to_prop_r + grad_to_prop
            grad_to_prop = tf.math.add(grad_to_prop_r, grad_to_prop) 
        else:
            grad_to_prop = tf.math.add(grad_to_prop_r, grad_before_shortcut2)

        return grad_to_prop
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        self.bias1.update_weights(optimizer)
        self.repeat1.update_weights(optimizer)
        self.bn1.update_weights(optimizer)
        self.bias2.update_weights(optimizer)
        self.prelu1.update_weights(optimizer)
        self.bias3.update_weights(optimizer)
        self.bias4.update_weights(optimizer)

        if(self.in_filters == self.out_filters):
            self.repeat2.update_weights(optimizer)
            self.bn2.update_weights(optimizer)
        else:
            self.repeat3.update_weights(optimizer)
            self.bn3.update_weights(optimizer)
            self.bn4.update_weights(optimizer)

        self.bias5.update_weights(optimizer)
        self.prelu2.update_weights(optimizer)
        self.bias6.update_weights(optimizer)
            
        if(self.bin_weights_quant != None):
            curr_w = my_quantize2(self.conv1.weights[0], None, None, self.bin_weights_quant)
            self.conv1.weights[0].assign(my_quantize2(curr_w - curr_lr*self.conv_grads1[0], None, None, self.bin_weights_quant))
            
            if(self.in_filters == self.out_filters):
                curr_w = my_quantize2(self.conv2.weights[0], None, None, self.bin_weights_quant)
                self.conv2.weights[0].assign(my_quantize2(curr_w - curr_lr*self.conv_grads2[0], None, None, self.bin_weights_quant))
            else:
                curr_w = my_quantize2(self.conv3.weights[0], None, None, self.bin_weights_quant)
                self.conv3.weights[0].assign(my_quantize2(curr_w - curr_lr*self.conv_grads3[0], None, None, self.bin_weights_quant))
                curr_w = my_quantize2(self.conv4.weights[0], None, None, self.bin_weights_quant)
                self.conv4.weights[0].assign(my_quantize2(curr_w - curr_lr*self.conv_grads4[0], None, None, self.bin_weights_quant))
        else:
            self.conv1.weights[0].assign(self.conv1.weights[0] - curr_lr*self.conv_grads1[0])

            if(self.in_filters == self.out_filters):
                self.conv2.weights[0].assign(self.conv2.weights[0] - curr_lr*self.conv_grads2[0])
            else:
                self.conv3.weights[0].assign(self.conv3.weights[0] - curr_lr*self.conv_grads3[0])
                self.conv4.weights[0].assign(self.conv4.weights[0] - curr_lr*self.conv_grads4[0])
            
        if self.conv1.use_bias == True:
            if(self.bin_weights_quant != None):
                curr_b = my_quantize2(self.conv1.weights[1], None, None, self.bin_weights_quant)
                self.conv1.weights[1].assign(my_quantize2(curr_b - curr_lr*self.conv_grads1[1], None, None, self.bin_weights_quant))
                
                if(self.in_filters == self.out_filters):
                    curr_w = my_quantize2(self.conv2.weights[1], None, None, self.bin_weights_quant)
                    self.conv2.weights[1].assign(my_quantize2(curr_w - curr_lr*self.conv_grads2[1], None, None, self.bin_weights_quant))
                else:
                    curr_w = my_quantize2(self.conv3.weights[1], None, None, self.bin_weights_quant)
                    self.conv3.weights[1].assign(my_quantize2(curr_w - curr_lr*self.conv_grads3[1], None, None, self.bin_weights_quant))
                    curr_w = my_quantize2(self.conv4.weights[1], None, None, self.bin_weights_quant)
                    self.conv4.weights[1].assign(my_quantize2(curr_w - curr_lr*self.conv_grads4[1], None, None, self.bin_weights_quant))
                
            else:
                self.conv1.weights[1].assign(self.conv1.weights[1] - curr_lr*self.conv_grads1[1])

                if(self.in_filters == self.out_filters):
                    self.conv2.weights[1].assign(self.conv2.weights[1] - curr_lr*self.conv_grads2[1])
                else:
                    self.conv3.weights[1].assign(self.conv3.weights[1] - curr_lr*self.conv_grads3[1])
                    self.conv4.weights[1].assign(self.conv4.weights[1] - curr_lr*self.conv_grads4[1])
            
        # Clip weights after gradient update
        clip_val = self.conv1.kernel_quantizer.clip_value
        self.conv1.weights[0].assign(tf.clip_by_value(self.conv1.weights[0], -clip_val, clip_val))

        if(self.in_filters == self.out_filters):
            clip_val = self.conv2.kernel_quantizer.clip_value
            self.conv2.weights[0].assign(tf.clip_by_value(self.conv2.weights[0], -clip_val, clip_val))
        else:
            clip_val = self.conv3.kernel_quantizer.clip_value
            self.conv3.weights[0].assign(tf.clip_by_value(self.conv3.weights[0], -clip_val, clip_val))
            clip_val = self.conv4.kernel_quantizer.clip_value
            self.conv4.weights[0].assign(tf.clip_by_value(self.conv4.weights[0], -clip_val, clip_val))
        
class BirealnetBlock_BinConv_Conv_BN_DPRelu(tf.keras.layers.Layer):
    
    def __init__(self, conv_layer, bn_layer, dprelu, num_quant_bits_fw=None, num_quant_bits_bw=None, bin_weights_quant=None, **kwargs):
        super().__init__(**kwargs)
        
        self.conv = conv_layer
        self.bn = BN_layer(bn_layer, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.prelu = DPReLU_layer(dprelu, num_quant_bits_fw=num_quant_bits_fw, num_quant_bits_bw=num_quant_bits_bw)
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.bin_weights_quant = bin_weights_quant
        
        self.inputs = None
        self.out_conv = None
        self.out_bn = None
        self.out_add = None
        self.out_prelu = None
        
        self.conv_grads = None        
        
    def call(self, inputs, training=None):
    
        if(self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
        
        self.inputs = inputs
        
        self.out_conv = self.conv(self.inputs)
        if(self.quant_fw != None):
            self.out_conv = my_quantize(self.out_conv, None, None, self.quant_fw)
        
        self.out_bn = self.bn(self.out_conv, training=training)
        
        self.out_add = self.out_bn + self.inputs
            
        self.out_prelu = self.prelu(self.out_add)
        
        return self.out_prelu
    
    def backward(self, inc_grads):
        
        grad_to_prop = self.prelu.backward(inc_grads)
        grad_before_shortcut = grad_to_prop
        grad_to_prop = self.bn.backward(grad_to_prop)
        
        grad_to_prop, self.conv_grads = conv_backward_binary_birealnet(grad_to_prop, self.inputs, 
                                                                    self.conv.weights, padding=self.conv.padding,
                                                                    padding_value=0.0,
                                                                    stride=self.conv.strides[0],
                                                                    kernel_clip_value=self.conv.kernel_quantizer.clip_value,
                                                                    x_offset_kernel=self.conv.kernel_quantizer.xoff,
                                                                    input_clip_value=1.0,
                                                                    x_offset_input=self.conv.input_quantizer.xoff,
                                                                    kernel_size=self.conv.kernel_size[0],
                                                                    num_bits=self.quant_bw,
                                                                    num_bits_binary=self.bin_weights_quant)
        
        return grad_before_shortcut + grad_to_prop
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        self.bn.update_weights(optimizer)
        self.prelu.update_weights(optimizer)
            
        if(self.bin_weights_quant != None):
            curr_w = my_quantize2(self.conv.weights[0], None, None, 1)
            self.conv.weights[0].assign(my_quantize2(curr_w - curr_lr*self.conv_grads[0], None, None, self.bin_weights_quant))
        else:
            self.conv.weights[0].assign(self.conv.weights[0] - curr_lr*self.conv_grads[0])
            
        if self.conv.use_bias == True:
            if(self.bin_weights_quant != None):
                curr_b = my_quantize2(self.conv.weights[1], None, None, self.bin_weights_quant)
                self.conv.weights[1].assign(my_quantize2(curr_b - curr_lr*self.conv_grads[1], None, None, self.bin_weights_quant))
                self.conv.kernel_quantizer.xoff.assign(my_quantize2(self.conv.kernel_quantizer.xoff - curr_lr*self.conv_grads[2], None, None, self.bin_weights_quant))
                self.conv.input_quantizer.xoff.assign(my_quantize2(self.conv.input_quantizer.xoff - curr_lr*self.conv_grads[3], None, None, self.bin_weights_quant))
            else:
                self.conv.weights[1].assign(self.conv.weights[1] - curr_lr*self.conv_grads[1])
                self.conv.kernel_quantizer.xoff.assign(self.conv.kernel_quantizer.xoff - curr_lr*self.conv_grads[2])
                self.conv.input_quantizer.xoff.assign(self.conv.input_quantizer.xoff - curr_lr*self.conv_grads[3])
        else:
            if(self.bin_weights_quant != None):
                self.conv.kernel_quantizer.xoff.assign(my_quantize2(self.conv.kernel_quantizer.xoff - curr_lr*self.conv_grads[1], None, None, self.bin_weights_quant))
                self.conv.input_quantizer.xoff.assign(my_quantize2(self.conv.input_quantizer.xoff - curr_lr*self.conv_grads[2], None, None, self.bin_weights_quant))
            else:
                self.conv.kernel_quantizer.xoff.assign(self.conv.kernel_quantizer.xoff - curr_lr*self.conv_grads[1])
                self.conv.input_quantizer.xoff.assign(self.conv.input_quantizer.xoff - curr_lr*self.conv_grads[2])
            
        # Clip weights after gradient update
        clip_val = self.conv.kernel_quantizer.clip_value
        self.conv.weights[0].assign(tf.clip_by_value(self.conv.weights[0], -clip_val, clip_val))
        
class BN_layer(tf.keras.layers.Layer):
    
    def __init__(self, bn_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.bn = bn_layer
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.bn_cache = None
        self.bn_grads = None
        self.bn_cache = None
        
    def call(self, inputs, training=None):
        
        out_bn, self.bn_cache = batch_norm_forward(self.bn, inputs, is_training=training)
        #out_bn2 = self.bn(inputs, training=training)
        return out_bn
    
    def backward(self, inc_grads):
        
        self.bn_grads = batch_norm_backward(self.bn, inc_grads, self.bn_cache)        
        return self.bn_grads[0]
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        # Update weights
        if self.bn.scale == True:
            self.bn.gamma.assign(self.bn.gamma - curr_lr*self.bn_grads[1])
        
        if self.bn.center == True:
            self.bn.beta.assign(self.bn.beta - curr_lr*self.bn_grads[-1])
            
class Conv_layer(tf.keras.layers.Layer):
    
    def __init__(self, conv_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.conv = conv_layer
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs = None
        self.conv_grads = None
        
    def call(self, inputs, training=None):
        
        if (self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
            self.conv.weights[0].assign(my_quantize(self.conv.weights[0], None, None, self.quant_fw))
        
        self.inputs = inputs
        out_conv = self.conv(self.inputs)
        
        if (self.quant_fw != None):
            out_conv = my_quantize(out_conv, None, None, self.quant_fw)
            
        return out_conv
    
    def backward(self, inc_grads):

        input_conv_grad, self.conv_grads = conv_backward(inc_grads, self.inputs, self.conv.weights, 
                                                         padding=self.conv.padding, kernel_size=self.conv.kernel_size[0],
                                                         num_bits=self.quant_bw)
        return input_conv_grad
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        # Update weights            
        if(self.quant_bw != None):
            self.conv.weights[0].assign(my_quantize(self.conv.weights[0] - curr_lr*self.conv_grads[0], None, None, self.quant_bw))
        else:
            self.conv.weights[0].assign(self.conv.weights[0] - curr_lr*self.conv_grads[0])
            
        if self.conv.use_bias == True:
            if(self.quant_bw != None):
                self.conv.weights[1].assign(my_quantize(self.conv.weights[1] - curr_lr*self.conv_grads[1], None, None, self.quant_bw))
            else:
                self.conv.weights[1].assign(self.conv.weights[1] - curr_lr*self.conv_grads[1])
                
class DepthWiseConv_layer(tf.keras.layers.Layer):

    def __init__(self, depthwiseconv_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.depthwiseconv = depthwiseconv_layer
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs = None
        self.conv_grads = None
        
    def call(self, inputs, training=None):
        
        if (self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
            self.depthwiseconv.weights[0].assign(my_quantize(self.depthwiseconv.weights[0], None, None, self.quant_fw))
        
        self.inputs = inputs
        out_conv = self.depthwiseconv(self.inputs)
        
        if (self.quant_fw != None):
            out_conv = my_quantize(out_conv, None, None, self.quant_fw)
            
        return out_conv
    
    def backward(self, inc_grads):

        input_conv_grad, self.conv_grads = depthwiseconv_backward(inc_grads, self.inputs, self.depthwiseconv.weights, 
                                                         padding=self.depthwiseconv.padding, stride=self.depthwiseconv.strides[0],
                                                         num_bits=self.quant_bw)
        return input_conv_grad
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        # Update weights            
        if(self.quant_bw != None):
            self.depthwiseconv.weights[0].assign(my_quantize(self.depthwiseconv.weights[0] - curr_lr*self.conv_grads[0], None, None, self.quant_bw))
        else:
            self.depthwiseconv.weights[0].assign(self.depthwiseconv.weights[0] - curr_lr*self.conv_grads[0])
            
        if self.depthwiseconv.use_bias == True:
            if(self.quant_bw != None):
                self.depthwiseconv.weights[1].assign(my_quantize(self.depthwiseconv.weights[1] - curr_lr*self.conv_grads[1], None, None, self.quant_bw))
            else:
                self.depthwiseconv.weights[1].assign(self.depthwiseconv.weights[1] - curr_lr*self.conv_grads[1])
            
class GlobalAVGPooling_layer(tf.keras.layers.Layer):
    
    def __init__(self, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.global_avg = tf.keras.layers.GlobalAveragePooling2D()
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs = None
        
    def call(self, inputs, training=None):
    
        if (self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
        
        self.inputs = inputs
        out_pool = self.global_avg(self.inputs)
        
        if (self.quant_fw != None):
            out_pool = my_quantize(out_pool, None, None, self.quant_fw)
        
        return out_pool
    
    def backward(self, inc_grads):

        out_grad = global_avg_pooling(self.inputs, inc_grads, num_bits=self.quant_bw)        
        return out_grad
    
    def update_weights(self, optimizer):
        pass
    
class Repeat_layer(tf.keras.layers.Layer):
    
    def __init__(self, channel_mul_factor, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.channel_mul_factor = int(channel_mul_factor)
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs_shape = None
        
    def call(self, inputs, training=None):
    
        if (self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
        
        self.inputs_shape = inputs.get_shape().as_list()
        x = tf.repeat(inputs, [self.channel_mul_factor], axis=3)
        out = tf.keras.layers.Reshape((self.inputs_shape[1], self.inputs_shape[2], self.inputs_shape[3]*self.channel_mul_factor))(x)
        
        if (self.quant_fw != None):
            out = my_quantize(out, None, None, self.quant_fw)
        
        return out
    
    def backward(self, inc_grads):

        #out_grad = tf.nn.conv2d(inc_grads, tf.ones([1, 1, self.inputs_shape[3], self.inputs_shape[3]]), 
        #                           strides=[1, 1, 1, 1], 
        #                           padding="VALID",
        #                           data_format='NHWC', 
        #                           dilations=[1, 1, 1, 1])
        list_elements = [None]*self.inputs_shape[-1]
        for i in tf.range(self.inputs_shape[-1]):
            vx = tf.expand_dims(tf.math.reduce_sum(tf.slice(inc_grads, [0, 0, 0, i*self.channel_mul_factor], 
                                                            [self.inputs_shape[0], self.inputs_shape[1], self.inputs_shape[2], self.channel_mul_factor]), axis=-1), axis=3)
            list_elements[i] = vx

        out_grad = tf.concat(list_elements, axis=-1)
        
        if (self.quant_bw != None):
            out_grad = my_quantize(out_grad, None, None, self.quant_bw)
        return out_grad
    
    def update_weights(self, optimizer):
        pass
        
class ReLUMaxPool_layer(tf.keras.layers.Layer):

    def __init__(self, max_pool_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.relu = tf.keras.layers.ReLU()
        self.pool = max_pool_layer
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs = None
        self.relu_out = None
        self.out_pool = None
        
    def call(self, inputs, training=None):
    
        if (self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
            
        self.inputs = inputs
        
        self.relu_out = self.relu(inputs)
        
        self.out_pool = self.pool(self.relu_out)
        
        return self.out_pool
        
    def backward(self, inc_grads):

        pool_grads = max_pool2d_backward(self.relu_out, self.out_pool, inc_grads, pool_size=self.pool.pool_size[0], 
                                         stride=self.pool.strides[0], padding=self.pool.padding.upper(), num_bits=self.quant_bw)
        out_grad = relu_backward(pool_grads, self.inputs, num_bits=self.quant_bw)    
        return out_grad
        
    def update_weights(self, optimizer):
        pass
        
class PReLU_layer(tf.keras.layers.Layer):
    
    def __init__(self, prelu_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.prelu_layer = prelu_layer
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs = None
        self.alpha_grad = None
        
    def call(self, inputs, training=None):
        
        if (self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
            
        self.inputs = inputs
        
        relu_out = self.prelu_layer(self.inputs)
        
        if (self.quant_fw != None):
            relu_out = my_quantize(relu_out, None, None, self.quant_fw)
        
        return relu_out
    
    def backward(self, inc_grads):
        
        out_grad = tf.where(self.inputs < 0.0, self.prelu_layer.weights[0], 1.0)*inc_grads
        if(self.quant_bw != None):
            out_grad = my_quantize(out_grad, None, None, self.quant_bw)
            
        neg_values = tf.where(self.inputs < 0.0, inc_grads, tf.zeros_like(inc_grads))
        self.alpha_grad = tf.math.reduce_sum(inc_grads*neg_values, axis=[0, 1, 2])
        if(self.quant_bw != None):
            self.alpha_grad = my_quantize(self.alpha_grad, None, None, self.quant_bw)
        
        return out_grad
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        if(self.quant_bw != None):
                self.prelu_layer.weights[0].assign(my_quantize(self.prelu_layer.weights[0] - curr_lr*self.alpha_grad, None, None, self.quant_bw))
        else:
            self.prelu_layer.weights[0].assign(self.prelu_layer.weights[0] - curr_lr*self.alpha_grad)
            
class DPReLU_layer(tf.keras.layers.Layer):
    
    def __init__(self, dprelu_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.dprelu_layer = dprelu_layer
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs = None
        self.grad_etha = None
        self.grad_gamma = None
        self.grad_beta = None
        self.grad_alpha = None
        
    def call(self, inputs, training=None):
        
        if (self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
            
        self.inputs = inputs
        
        relu_out = self.dprelu_layer(self.inputs)
        
        if (self.quant_fw != None):
            relu_out = my_quantize(relu_out, None, None, self.quant_fw)
        
        return relu_out
    
    def backward(self, inc_grads):
        
        # Grad to input
        out_grad = tf.where(self.inputs >= 0.0, self.dprelu_layer._etha, self.dprelu_layer._gamma)*inc_grads
        
        if(self.quant_bw != None):
            out_grad = my_quantize(out_grad, None, None, self.quant_bw)
        
        # Compute grad etha
        grad_etha = tf.where((self.inputs-self.dprelu_layer._alpha) >= 0.0, (self.inputs-self.dprelu_layer._alpha), 0.0)*inc_grads
        self.grad_etha = tf.math.reduce_sum(grad_etha, axis=[0, 1, 2])
        
        # Compute grad gamma
        grad_gamma = tf.where((self.inputs-self.dprelu_layer._alpha) < 0.0, (self.inputs-self.dprelu_layer._alpha), 0.0)*inc_grads
        self.grad_gamma = tf.math.reduce_sum(grad_gamma, axis=[0, 1, 2])
        
        # grad to beta
        self.grad_beta = tf.math.reduce_sum(-inc_grads, axis=[0, 1, 2])
        
        # grad alpha
        grad_alpha = tf.where((self.inputs-self.dprelu_layer._alpha) >= 0.0, (-self.dprelu_layer._etha), -self.dprelu_layer._gamma)*inc_grads
        self.grad_alpha = tf.math.reduce_sum(grad_alpha, axis=[0, 1, 2])
        
        if(self.quant_bw != None):
            self.grad_etha = my_quantize(self.grad_etha, None, None, self.quant_bw)
            self.grad_gamma = my_quantize(self.grad_gamma, None, None, self.quant_bw)
            self.grad_beta = my_quantize(self.grad_beta, None, None, self.quant_bw)
            self.grad_alpha = my_quantize(self.grad_alpha, None, None, self.quant_bw)
        
        return out_grad
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        # weights order is alpha, beta, etha, gamma
        
        if(self.quant_bw != None):
                self.dprelu_layer.weights[0].assign(my_quantize(self.dprelu_layer.weights[0] - curr_lr*self.grad_alpha, None, None, self.quant_bw))
                self.dprelu_layer.weights[1].assign(my_quantize(self.dprelu_layer.weights[1] - curr_lr*self.grad_beta, None, None, self.quant_bw))
                self.dprelu_layer.weights[2].assign(my_quantize(self.dprelu_layer.weights[2] - curr_lr*self.grad_etha, None, None, self.quant_bw))
                self.dprelu_layer.weights[3].assign(my_quantize(self.dprelu_layer.weights[3] - curr_lr*self.grad_gamma, None, None, self.quant_bw))
        else:
            self.dprelu_layer.weights[0].assign(self.dprelu_layer.weights[0] - curr_lr*self.grad_alpha)
            self.dprelu_layer.weights[1].assign(self.dprelu_layer.weights[1] - curr_lr*self.grad_beta)
            self.dprelu_layer.weights[2].assign(self.dprelu_layer.weights[2] - curr_lr*self.grad_etha)
            self.dprelu_layer.weights[3].assign(self.dprelu_layer.weights[3] - curr_lr*self.grad_gamma)
            
class LearnableBias_layer(tf.keras.layers.Layer):
    
    def __init__(self, learnablebias_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.learnablebias_layer = learnablebias_layer
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.alpha_grad = None
        
    def call(self, inputs, training=None):
        
        if (self.quant_fw != None):
            inputs = my_quantize(inputs, None, None, self.quant_fw)
        
        relu_out = self.learnablebias_layer(inputs)
        
        if (self.quant_fw != None):
            relu_out = my_quantize(relu_out, None, None, self.quant_fw)
        
        return relu_out
    
    def backward(self, inc_grads):
            
        self.alpha_grad = tf.math.reduce_sum(inc_grads, axis=[0, 1, 2])
        if(self.quant_bw != None):
            self.alpha_grad = my_quantize(self.alpha_grad, None, None, self.quant_bw)
        
        return inc_grads
    
    def update_weights(self, optimizer):
        
        curr_lr = K.get_value(optimizer.lr)
        
        if(self.quant_bw != None):
            self.learnablebias_layer.learnable_bias.assign(my_quantize(self.learnablebias_layer.learnable_bias - curr_lr*self.alpha_grad, None, None, self.quant_bw))
        else:
            self.learnablebias_layer.learnable_bias.assign(self.learnablebias_layer.learnable_bias - curr_lr*self.alpha_grad)

class CWR_layer(tf.keras.layers.Layer):
    
    def __init__(self, dense_layer, num_quant_bits_fw=None, num_quant_bits_bw=None, **kwargs):
        super().__init__(**kwargs)
        
        self.dense = dense_layer
        self.quant_fw = num_quant_bits_fw
        self.quant_bw = num_quant_bits_bw
        self.inputs = None
        self.dense_grads = None
        self.classes_seen = None
        
    def call(self, inputs, training=None):
        
        if self.quant_fw != None:
            self.inputs = my_quantize(inputs, num_bits=self.quant_fw)
            weights = my_quantize(self.dense.weights[0], num_bits=self.quant_fw)
        else:
            self.inputs = inputs
            weights = self.dense.weights[0]
        
        out_mat_mul = tf.linalg.matmul(self.inputs, weights, transpose_a=False)
        
        if self.quant_fw != None:
            predictions = my_quantize(tf.keras.layers.Softmax()(out_mat_mul), 0.0, 1.0, num_bits=self.quant_fw)
        else:
            predictions = tf.keras.layers.Softmax()(out_mat_mul)
        return predictions
    
    def compute_loss_value(self, labels, predictions):
        
        #labels = tf.cast(labels, tf.float32)
        w_shape = tf.shape(self.dense.weights[0])
        num_labels = w_shape[1]
        
        batch_size = tf.cast(tf.shape(labels)[0], tf.float32)
        
        if (tf.shape(labels)[1] != num_labels):
            categorical_labels = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), depth=num_labels, dtype=tf.float32))
        else:
            categorical_labels = tf.cast(labels, tf.float32)
            
        gradient_to_logits = tf.math.subtract(predictions, categorical_labels) / batch_size
        
        if self.quant_bw != None:
            gradient_to_logits = my_quantize(gradient_to_logits, num_bits=self.quant_bw)
        
        return gradient_to_logits
    
    def backward(self, inc_grads):
        
        self.gradients_to_weights = tf.linalg.matmul(self.inputs, inc_grads, transpose_a=True)
        
        if self.quant_bw != None:
            self.gradients_to_weights = my_quantize(self.gradients_to_weights, num_bits=self.quant_bw)
        
        grad_to_input = tf.linalg.matmul(inc_grads, self.dense.weights[0], transpose_a=False, transpose_b=True)
        
        if self.quant_bw != None:
            grad_to_input = my_quantize(grad_to_input, num_bits=self.quant_bw)
 
        return grad_to_input
    
    def update_weights(self, optimizer):
    
        if self.classes_seen == None:
        
            curr_lr = K.get_value(optimizer.lr)
            
            new_w = self.dense.weights[0] - curr_lr*self.gradients_to_weights
            
            if self.quant_bw != None:
                new_w = my_quantize(new_w, num_bits=self.quant_bw)
            
            # Update weights            
            self.dense.weights[0].assign(new_w)
            
        else:
        
            # new weights considering all neurons
            new_w = self.dense.weights[0] - K.get_value(optimizer.lr)*self.gradients_to_weights
            self.dense.weights[0].assign(new_w)
        
            curr_lr = 0.0005
            
            n_input_cwr = self.dense.weights[0].numpy().shape[0]
            
            for c, w in self.classes_seen.items():
            
                grad_of_neuron = self.gradients_to_weights[:,c]
                weights_of_neuron = self.dense.weights[0][:,c]
                updates = weights_of_neuron - curr_lr*grad_of_neuron
                
                indices = tf.concat([tf.reshape(tf.range(0, n_input_cwr), (n_input_cwr, 1)), tf.constant(c, shape=(n_input_cwr,1))], axis=1)
                
                new_w = tf.tensor_scatter_nd_update(self.dense.weights[0], indices, updates)
                self.dense.weights[0].assign(new_w)
        
def Execute_custom_layers_forward(input_data, layers, training=None, final_idx=None):
    
    if(final_idx==None)or(final_idx>=len(layers)):
        final_idx = len(layers)
    
    for idx in range(0, final_idx):
        output = layers[idx](input_data, training=training)
        input_data = output
    return output

def Execute_custom_layers_backward(predictions, labels, layers, final_idx=-1):
    
    gradient_to_prop = None
        
    for idx in range(len(layers)-1, final_idx, -1):
        print("Back_prop {} - {}".format(idx ,layers[idx]))
        if(idx == (len(layers)-1)):
            if getattr(layers[idx], "compute_loss_value", None) != None:
               gradient_to_prop = layers[idx].compute_loss_value(labels, predictions) 
               gradient_to_prop = layers[idx].backward(gradient_to_prop)
        else:
            gradient_to_prop = layers[idx].backward(gradient_to_prop)
    return gradient_to_prop

def Update_weights(layers, optimizer, final_idx=-1):
    
    for idx in range(len(layers)-1, final_idx, -1):
        #print("Weights update {} - {}".format(idx ,layers[idx]))
        if getattr(layers[idx], "update_weights", None) != None:
               layers[idx].update_weights(optimizer)