import tensorflow as tf
from typing import Any, Mapping
from contextlib import contextmanager
from typing import Callable, Union, Optional
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.ops.array_ops import ones
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
import types
import math

physical_devices = tf.config.list_physical_devices('GPU') 
if (physical_devices != None) and (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

################################################################################################################
################################################################################################################
###############################         Utils
################################################################################################################
################################################################################################################

def register_keras_custom_object(cls):
    """See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/keras_utils.py#L25"""
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls

def register_alias(name: str):
    """A decorator to register a custom keras object under a given alias.
    !!! example
        ```python
        @utils.register_alias("degeneration")
        class Degeneration(tf.keras.metrics.Metric):
            pass
        ```
    """

    def register_func(cls):
        tf.keras.utils.get_custom_objects()[name] = cls
        return cls

    return register_func

@contextmanager
def patch_object(object, name, value):
    """Temporarily overwrite attribute on object"""
    old_value = getattr(object, name)
    setattr(object, name, value)
    yield
    setattr(object, name, old_value)

################################################################################################################
################################################################################################################

class BaseQuantizer(tf.keras.layers.Layer):
    """Common base class for defining quantizers.

    # Attributes
        precision: An integer defining the precision of the output. This value will be
            used by `lq.models.summary()` for improved logging.
    """

    num_quantization_bits = None

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        super().build(input_shape)

    @property
    def non_trainable_weights(self):
        return [self.num_quantization_bits]

@register_alias("no_quant")
@register_keras_custom_object
class NoOpQuantize(BaseQuantizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def _clipped_gradient(x, dy, clip_value):
    """Calculate `clipped_gradent * dy`."""

    if clip_value is None:
        return dy

    zeros = tf.zeros_like(dy)
    #grad_over_th_set = tf.constant(grad_over_th, dtype=dy.dtype, shape=dy.get_shape())
    mask = tf.math.less_equal(tf.math.abs(x), clip_value)
    #return tf.where(mask, dy, grad_over_th_set)
    return tf.where(mask, dy, zeros)


def ste_sign(x: tf.Tensor, clip_value: float = 1.0, back_prop_grad_over_th = 0.0) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy, variables=None):
            return _clipped_gradient(x, dy, clip_value)

        ones = tf.ones_like(x)
        less_one = tf.math.negative(ones)
        mask = tf.math.greater_equal(x, tf.constant(0, dtype=x.dtype))
        return tf.where(mask, ones, less_one), grad
        #return tf.math.sign(x), grad

    return _call(x)

def ste_sign_normalized(x: tf.Tensor, clip_value: float = 1.0) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            in_mean = tf.math.reduce_mean(x)
            in_std = tf.math.reduce_std(x)
            inputs_mod = (x - in_mean)/in_std
            return _clipped_gradient(inputs_mod, dy, clip_value)

        in_mean = tf.math.reduce_mean(x)
        in_std = tf.math.reduce_std(x)
        inputs_mod = (x - in_mean)/in_std
        ones = tf.ones_like(inputs_mod)
        less_one = tf.math.negative(ones)
        mask = tf.math.greater_equal(inputs_mod, tf.constant(0, dtype=x.dtype))
        return tf.where(mask, ones, less_one), grad

    return _call(x)


def ste_sign_normalized_enhanced(x: tf.Tensor, t_Value, k_value) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            in_mean = tf.math.reduce_mean(x)
            in_std = tf.math.reduce_std(x)
            inputs_mod = (x - in_mean)/in_std

            return dy*(t_Value*k_value*(1.0 - (tf.math.tanh(t_Value*inputs_mod)*tf.math.tanh(t_Value*inputs_mod))))

        in_mean = tf.math.reduce_mean(x)
        in_std = tf.math.reduce_std(x)
        inputs_mod = (x - in_mean)/in_std
        ones = tf.ones_like(inputs_mod)
        less_one = tf.math.negative(ones)
        mask = tf.math.greater_equal(inputs_mod, tf.constant(0, dtype=x.dtype))
        return tf.where(mask, ones, less_one), grad

    return _call(x)

def ste_sign_normalized_enhanced_all_in(x: tf.Tensor, t_Value, k_value, clip_value, selector) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            in_mean = tf.math.reduce_mean(x)
            in_std = tf.math.reduce_std(x)
            inputs_mod = (x - in_mean)/in_std

            if selector == 1:
                return dy*(t_Value*k_value*(1.0 - (tf.math.tanh(t_Value*inputs_mod)*tf.math.tanh(t_Value*inputs_mod))))
            else:
                return _clipped_gradient(inputs_mod, dy, clip_value)

        in_mean = tf.math.reduce_mean(x)
        in_std = tf.math.reduce_std(x)
        inputs_mod = (x - in_mean)/in_std
        ones = tf.ones_like(inputs_mod)
        less_one = tf.math.negative(ones)
        mask = tf.math.greater_equal(inputs_mod, tf.constant(0, dtype=x.dtype))
        return tf.where(mask, ones, less_one), grad

    return _call(x)

def binarization_with_sign_approx(x: tf.Tensor, clip_value: float = 1.0) -> tf.Tensor:
    @tf.custom_gradient
    def _call2(x):
        def grad(dy):
            abs_x = tf.math.abs(x)
            zeros = tf.zeros_like(dy)
            mask = tf.math.less_equal(abs_x, clip_value)
            return tf.where(mask, (clip_value - abs_x) * 2 * dy, zeros)

        ones = tf.ones_like(x)
        less_one = tf.math.negative(ones)
        mask = tf.math.greater_equal(x, tf.constant(0, dtype=x.dtype))
        return tf.where(mask, ones, less_one), grad

    return _call2(x)

def standard_pow2_quantization(x : tf.Tensor, quant_bits: int = 8) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):

        def grad(dy):
            return dy

        quantized_range = tf.stop_gradient(tf.pow(tf.constant(2), tf.constant(quant_bits)) - tf.constant(1))
        #offset_quantized = tf.stop_gradient(tf.pow(tf.constant(2), tf.constant(quant_bits-1)))
        max_val = tf.math.reduce_max(x)
        min_val = tf.math.reduce_min(x)
        diff = tf.stop_gradient(max_val - min_val)
        diff = tf.where(tf.math.equal(diff, 0), tf.keras.backend.epsilon(), diff)
        func_to_map = lambda x: tf.round(((x - tf.stop_gradient(min_val))*tf.cast(quantized_range, x.dtype))/(diff)) - tf.constant(128.0)
        final_result = tf.map_fn(func_to_map, x)
        return final_result, grad
    return _call(x)

def standard_pow2_add_quantization_noise(x : tf.Tensor, min_range, max_range, quant_bits: int = 8) -> tf.Tensor:
    @tf.custom_gradient
    def _call2(x):

        def grad(dy):
            return dy

        curr_range = [min_range, max_range]

        quantized_range = tf.pow(tf.constant(2), tf.convert_to_tensor(quant_bits)) - tf.constant(1)
        diff = curr_range[1] - curr_range[0]
        scale = tf.cast(quantized_range, x.dtype) / diff
        x = tf.clip_by_value(x, curr_range[0], curr_range[1])

        func_to_map = lambda x: (tf.round((x - curr_range[0])*scale)/scale + curr_range[0])
        final_result = tf.map_fn(func_to_map, x)
        return final_result, grad
    return _call2(x)

def fixed_point_quantization_zero_centered(x : tf.Tensor, 
                                           num_total_bits,
                                           num_fractional_bits,
                                           is_signed) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return dy

        if (is_signed == True):
            num_for_range = 2**(num_total_bits-1)
        else:
            num_for_range = 2**(num_total_bits)
         
        multiplier = 2.0**num_fractional_bits#tf.math.pow(tf.constant(2.0), tf.constant(num_fractional_bits, dtype=tf.float32))
        res = tf.clip_by_value(tf.math.round(multiplier * x), -num_for_range+1.0, num_for_range+1.0)/multiplier
        return res, grad
    return _call(x)

@register_alias("none_quant")
@register_keras_custom_object
class NoneQuantizer(tf.keras.layers.Layer):

    num_quantization_bits = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return inputs

    def get_config(self):
        return {**super().get_config()}

@register_alias("bin_quant")
@register_keras_custom_object
class StdBinaryQuant(tf.keras.layers.Layer):

    num_quantization_bits = 1

    def __init__(self, clip_value: float = 1.0, back_prop_grad_values=0.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value
        self.back_prop_grad_values = back_prop_grad_values

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        outputs = ste_sign(inputs, clip_value=self.clip_value, back_prop_grad_over_th=self.back_prop_grad_values)
        return outputs

    def get_config(self):
        return {**super().get_config(), 
                "clip_value": self.clip_value,
                "back_prop_grad_values": self.back_prop_grad_values}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_alias("bin_quant_x_th")
@register_keras_custom_object
class StdBinaryQuantXTh(tf.keras.layers.Layer):

    num_quantization_bits = 1

    def __init__(self, clip_value: float = 1.0, x_offset=0.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value
        self.x_offset_val = x_offset

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        
        self.xoff = self.add_weight(
          name='x-offset',
          shape=(input_shape[-1]),
          initializer=tf.keras.initializers.Constant(self.x_offset_val),
          dtype=tf.float32,
          trainable=True)

    def call(self, inputs):
        x = tf.math.add(inputs, self.xoff)
        outputs = ste_sign(x, clip_value=self.clip_value, back_prop_grad_over_th=0.0)
        return outputs

    def get_config(self):
        return {**super().get_config(), 
                "clip_value": self.clip_value}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_alias("bin_quant_x_th")
@register_keras_custom_object
class StdBinaryQuantXYTh(tf.keras.layers.Layer):

    num_quantization_bits = 1

    def __init__(self, clip_value: float = 1.0, x_offset=0.0, y_offset=0.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value
        self.x_offset_val = x_offset
        self.y_offset_val = y_offset

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        
        self.xoff = self.add_weight(
          name='x-offset',
          shape=(input_shape[-1]),
          initializer=tf.keras.initializers.Constant(self.x_offset_val),
          dtype=tf.float32,
          trainable=True)

        self.yoff = self.add_weight(
          name='y-offset',
          shape=(input_shape[-1]),
          initializer=tf.keras.initializers.Constant(self.y_offset_val),
          dtype=tf.float32,
          trainable=True)

    def call(self, inputs):
        x = tf.math.add(inputs, self.xoff)
        outputs = ste_sign(x, clip_value=self.clip_value, back_prop_grad_over_th=0.0)
        outputs = tf.math.add(outputs, self.yoff)
        return outputs

    def get_config(self):
        return {**super().get_config(), 
                "clip_value": self.clip_value}
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_alias("bin_quant_norm")
@register_keras_custom_object
class StdBinaryQuantNormalized(tf.keras.layers.Layer):

    num_quantization_bits = 1

    def __init__(self, clip_value: float = 1.0,  **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        outputs = ste_sign_normalized(inputs, clip_value=self.clip_value)
        return outputs

    def get_config(self):
        return {**super().get_config(), "clip_value": self.clip_value}

@register_keras_custom_object
class StdBinaryQuantNormalizedEnhancedBackprop(tf.keras.layers.Layer):

    num_quantization_bits = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.k_value = tf.Variable(initial_value=max(1/0.1, 1.0), trainable=False, dtype=tf.float32, name='kvalue')#k_value
        #self.t_value = tf.Variable(initial_value=0.1, trainable=False, dtype=tf.float32, name='tvalue')#t_value

    def build(self, input_shape):

        super().build(input_shape)
        self.k_value = self.add_weight(initializer=tf.keras.initializers.Constant(max(1/0.1, 1.0)), trainable=False, dtype=tf.float32, name='kvalue')
        self.t_value = self.add_weight(initializer=tf.keras.initializers.Constant(0.1), trainable=False, dtype=tf.float32, name='tvalue')
        self.selection = tf.Variable(1, trainable=False, name='selection')

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        #outputs = tf.cond(self.selection == 1, lambda: ste_sign_normalized_enhanced(inputs, self.t_value.value(), self.k_value.value()), lambda: ste_sign_normalized(inputs, clip_value=1.25, back_prop_grad_over_th=0.0))
        #if self.selection == 1:
        #    return ste_sign_normalized_enhanced(inputs, self.t_value.value(), self.k_value.value())
        #else:
            #return ste_sign_normalized(inputs, clip_value=1.25, back_prop_grad_over_th=0.0)
        #return outputs
        return ste_sign_normalized_enhanced_all_in(inputs, self.t_value.value(), self.k_value.value(), 1.25, self.selection.value())

    def get_config(self):
        return {**super().get_config()}

def multi_thresholds_binarization(x: tf.Tensor, tL1, tH1, tL2, tH2) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            
            mask = tf.math.logical_and(tf.math.greater_equal(x, tL1), tf.math.less_equal(x, tH2))
            return tf.where(mask, dy, tf.constant(0, dtype=x.dtype))

        ones = tf.ones_like(x)
        less_one = tf.math.negative(ones)
        
        x = tf.clip_by_value(x, clip_value_min=tL1, clip_value_max=tH2)
        
        flip_to_positive_mask = tf.math.logical_and(tf.math.greater(x, tH1), tf.math.less(x, tf.constant(0, dtype=x.dtype)))
        flip_to_negative_mask = tf.math.logical_and(tf.math.greater(x, tf.constant(0, dtype=x.dtype)), tf.math.less(x, tL2))
        
        x = tf.where(flip_to_positive_mask, ones, x)
        x = tf.where(flip_to_negative_mask, less_one, x)
        
        return x, grad

    return _call(x)

@register_alias("bin_quant_multi_thresholds")
@register_keras_custom_object
class StdBinaryQuantMultiThresholds(tf.keras.layers.Layer):

    num_quantization_bits = 1

    def __init__(self, tL1=-1.25, tH1=-0.75, tL2=0.75, tH2=1.25, **kwargs):
        super().__init__(**kwargs)
        self.tL1 = tL1
        self.tH1 = tH1
        self.tL2 = tL2
        self.tH2 = tH2

    def build(self, input_shape):

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return multi_thresholds_binarization(inputs, self.tL1, self.tH1, self.tL2, self.tH2)

    def get_config(self):
        return {**super().get_config()}

@register_alias("pow_2bits_quant_sim")
@register_keras_custom_object
class FakeQuant_pow2(tf.keras.layers.Layer):

    def __init__(self, num_quant_bits = 8, **kwargs):
        super().__init__(**kwargs)
        self.num_quant_bits = num_quant_bits

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, min_val=None, max_val=None):
        if(min_val == None) and (max_val == None):
            min_val = tf.math.reduce_min(inputs)
            max_val = tf.math.reduce_max(inputs)  
        outputs = standard_pow2_add_quantization_noise(inputs, min_val, max_val, self.num_quant_bits)
        return outputs

    def get_config(self):
        ret_data = {**super().get_config(), "num_quant_bits": self.num_quant_bits}
        return ret_data

@register_alias("fixed_point_quant_sim")
@register_keras_custom_object
class FixedPoinQuantNoise(tf.keras.layers.Layer):

    def __init__(self, num_quant_bits = 8, 
                       num_bits_fractional_part = 4,
                       is_signed = True,
                        **kwargs):
        super().__init__(**kwargs)
        self.num_quant_bits = num_quant_bits
        self.num_bits_fractional_part = num_bits_fractional_part
        self.is_signed = is_signed

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, min_val=None, max_val=None):
        outputs = fixed_point_quantization_zero_centered(inputs, 
                                                         self.self.num_quant_bits,
                                                         self.num_bits_fractional_part,
                                                         self.is_signed)
        return outputs

    def get_config(self):
        ret_data = {**super().get_config(), "num_bits_fractional_part": self.num_bits_fractional_part}
        return ret_data


def hard_tanh_activation(x: tf.Tensor, clip_value: float = 1.0) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            zeros = tf.zeros_like(dy)
            mask = tf.math.less_equal(tf.math.abs(x), clip_value)
            return tf.where(mask, dy*clip_value, zeros)

        res = tf.clip_by_value(x, -clip_value, clip_value)
        return res, grad

    return _call(x)

def hard_tanh_activation_x_shift(x: tf.Tensor, clip_value: float = 1.0, x_shift: float = 0.0) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            zeros = tf.zeros_like(dy)
            if x_shift != 0.0:
                mask = tf.math.logical_and(tf.math.less_equal(x, clip_value+x_shift), tf.math.greater_equal(x, -clip_value+x_shift))
            else:
                mask = tf.math.less_equal(tf.math.abs(x), clip_value)
            return tf.where(mask, dy*clip_value, zeros)

        res = tf.clip_by_value(x-x_shift, -clip_value, clip_value)
        return res, grad
    return _call(x)

def symmetric_activation_leaky(x: tf.Tensor, clip_value, alpha) -> tf.Tensor:
    #@tf.custom_gradient
    def _call(x):
        def grad(dy):
            alpha_val = tf.constant(alpha, dtype=tf.float32, shape=x.get_shape())
            mask = tf.math.less_equal(tf.math.abs(x), clip_value)
            return tf.where(mask, dy*clip_value, alpha_val)

        return (
            tf.clip_by_value(x, -clip_value, clip_value)
            + (tf.math.maximum(x, clip_value) - clip_value) * alpha
            + (tf.math.minimum(x, -clip_value) + clip_value) * alpha
        )#, grad

    return _call(x)
@register_alias("sym_linear")
@register_keras_custom_object
class SymmetricLinear(tf.keras.layers.Layer):

    def __init__(self, clip_value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        outputs = hard_tanh_activation(inputs, clip_value=self.clip_value)
        #outputs = hard_tanh_activation_x_shift(inputs, clip_value=self.clip_value, x_shift=0.5)
        return outputs

    def get_config(self):
        return {**super().get_config(), "clip_value": self.clip_value}

@register_keras_custom_object
def hard_tanh_variable(x: tf.Tensor, clip_value: float = 1.0):

    return hard_tanh_activation(x, clip_value=clip_value)

@register_alias("sym_linear_leaky")
@register_keras_custom_object
class SymmetricLinearLeaky(tf.keras.layers.Layer):

    def __init__(self, clip_value: float = 1.0, alpha = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value
        self.alpha = alpha

    def build(self, input_shape):

        self.leaky_param = self.add_weight(name='leaky_slope', 
                                        dtype = tf.float32, 
                                        trainable=True)
        self.leaky_param = self.alpha

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        outputs = symmetric_activation_leaky(inputs, clip_value=self.clip_value, alpha=self.leaky_param)
        return outputs

    def get_config(self):
        return {**super().get_config(), 
                "clip_value": self.clip_value,
                'alpha' : self.alpha,
                'leaky_slope' : self.leaky_param}


import math as m

def activation_inverted_leaky(x: tf.Tensor, clip_value: float = 1.0, alpha: float = 0.1) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            mask = tf.math.less_equal(tf.math.abs(x), clip_value)
            return tf.where(mask, clip_value*dy, -alpha*dy)

        #res = tf.where(x < -clip_value, tf.math.minimum((-alpha*(x+clip_value)-clip_value), clip_value), \
        #               tf.where(x > clip_value, tf.math.maximum((-alpha*(x-clip_value)+clip_value), -clip_value), \
        #                        tf.clip_by_value(x, -clip_value, clip_value)))
        res = tf.where(x < -clip_value, (-alpha*(x+clip_value)-clip_value), \
                       tf.where(x > clip_value, (-alpha*(x-clip_value)+clip_value), \
                                tf.clip_by_value(x, -clip_value, clip_value)))
        return res, grad

    return _call(x)

class InvertedLeaky(tf.keras.layers.Layer):

    def __init__(self, clip_value: float = 1.0, alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value
        self.alpha = alpha
        self.slope = self.add_weight(initializer=tf.keras.initializers.Constant(self.alpha), trainable=False, dtype=tf.float32, name=self.name + '_inverted_leaky_slope')

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        outputs = activation_inverted_leaky(inputs, self.clip_value, self.slope)
        return outputs

    def get_config(self):
        return {**super().get_config(), 
                "clip_value": self.clip_value,
                "slope": self.slope.value().numpy()}


class Addition_with_Clipping(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        return input_shape

    #def build(self, input_shape):
    #    super().build(input_shape)

    def call(self, inputs):
        
        '''
        if len(inputs) == 2:
            output = tf.keras.layers.Add()([inputs[0], inputs[1]])
            output = tf.clip_by_value(output, -127.0, +127.0)
            return output
        else:
            return None
        '''

        output = tf.keras.layers.Add()([inputs[0], inputs[1]])
        output = tf.clip_by_value(output, -127.0, +127.0)
        return output

    def get_config(self):
        return {**super().get_config()}


def serialize_quant(quantizer: tf.keras.layers.Layer):
    return tf.keras.utils.serialize_keras_object(quantizer)


def deserialize_quant(name, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="quantization function",
    )

def get_quantizer(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize_quant(identifier)
    if isinstance(identifier, str):
        return deserialize_quant(str(identifier))
    if callable(identifier):
        return identifier
    raise ValueError(
        f"Could not interpret quantization function identifier: {identifier}"
    )


@register_keras_custom_object
class WeightClip(tf.keras.constraints.Constraint):
    """Weight Clip constraint

    Constrains the weights incident to each hidden unit
    to be between `[-clip_value, clip_value]`.

    # Arguments
        clip_value: The value to clip incoming weights.
    """

    def __init__(self, clip_value: float = 1):
        self.clip_value = clip_value

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(x, -self.clip_value, self.clip_value)

    def get_config(self) -> Mapping[str, Any]:
        return {"clip_value": self.clip_value}


# Aliases
@register_alias("weight_clip_constraint")
@register_keras_custom_object
class weight_clip(WeightClip):
    pass

@register_alias("weight_constraint_binary")
@register_keras_custom_object
class WeightForceBinary(tf.keras.constraints.Constraint):

    def __init__(self, tL1=-1.25, tH1=-0.75, tL2=0.75, tH2=1.25):
        self.tL1 = tL1
        self.tH1 = tH1
        self.tL2 = tL2
        self.tH2 = tH2

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        
        ones = tf.ones_like(x)
        less_one = tf.math.negative(ones)
        
        x = tf.clip_by_value(x, clip_value_min=self.tL1, clip_value_max=self.tH2)
        
        flip_to_positive_mask = tf.math.logical_and(tf.math.greater(x, self.tH1), tf.math.less(x, tf.constant(0, dtype=x.dtype)))
        flip_to_negative_mask = tf.math.logical_and(tf.math.greater(x, tf.constant(0, dtype=x.dtype)), tf.math.less(x, self.tL2))
        
        x = tf.where(flip_to_positive_mask, ones, x)
        x = tf.where(flip_to_negative_mask, less_one, x)
        
        return x

    def get_config(self):
        return {**super().get_config(),
                "tL1": self.tL1,
                "tH1": self.tH1,
                "tL2": self.tL2,
                "tH2": self.tH2}

@register_alias("binary_initializer")
@register_keras_custom_object
class BinaryInitializer(tf.keras.initializers.Initializer):

    def __init__(self, stddev=0.5):
        super().__init__()
        self.stddev = stddev

    def __call__(self, shape, dtype=tf.float32, **kwargs):
        x = tf.random.normal(shape, 0.0, self.stddev)
        ones = tf.ones(shape)
        less_one = tf.math.negative(ones)
        mask = tf.math.greater_equal(x, tf.constant(0, dtype=x.dtype))
        return tf.where(mask, ones, less_one)

    def get_config(self):
        return {**super().get_config()}

QuantizerType = Union[BaseQuantizer, Callable[[tf.Tensor], tf.Tensor]]

def moving_average_custom(input : tf.Tensor, _aver_ema : tf.Tensor, decay) -> tf.Tensor:
    @tf.function
    def _call2(input, _aver_ema, decay):
        _aver_ema.assign_sub((1.0-decay)*(_aver_ema - input))
        return _aver_ema        
    return _call2(input, _aver_ema, decay)

@tf.function
def exponential_moving_average(input, _aver_ema, decay):
    _aver_ema -= (1.0-decay)*(_aver_ema - input)
    return tf.stop_gradient(_aver_ema)        

@register_keras_custom_object
class MovingAverage(tf.keras.layers.Layer):

    def __init__(self, decay=0.998, **kwargs) -> None:
        super().__init__(**kwargs)
        self.decay = decay

    def build(self, input_shape):
        if input_shape.ndims > 4:
            raise ValueError('Input has undefined rank:', input_shape)
        else:
            self.ema = self.add_weight(initializer=tf.keras.initializers.Constant(0.0), 
                                       name='ema_mov_average',
                                       shape=input_shape[-1], 
                                       trainable=False)

        self.num_calls = tf.Variable(0, trainable=False)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.shape(self.ema)

    def call(self, inputs, training=None):

        if (training is not None) and (training == True):

            mean = tf.math.reduce_mean(inputs, 0)

            #if self.num_calls == 0:
            #    self.ema.assign(mean)
            #else:
            #    self.ema.assign_sub((1.0-self.decay)*(self.ema - mean))

            #self.num_calls.assign_add(1)   
            self.ema.assign_sub((1.0-self.decay)*(self.ema - mean))

            return tf.expand_dims(mean, axis=0)
        else:
            return tf.expand_dims(self.ema, axis=0)

    def get_config(self):
        return {**super().get_config(),
                'decay': self.decay}

@register_alias("quant_regularizer")
@register_keras_custom_object
class QuantizationRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, factor, penalize_mean=True) -> None:
        super().__init__()
        self.factor = factor
        self.penalize_mean = penalize_mean

    def get_config(self):
        return {**super().get_config(),
                "factor":self.factor,
                "penalize_mean":self.penalize_mean}

    def __call__(self, x):

        if self.penalize_mean == True:
            neg_aver = tf.abs(tf.reduce_mean(tf.gather_nd(x, tf.where(x < 0.))))
            pos_aver = tf.reduce_mean(tf.gather_nd(x, tf.where(x > 0.)))
            return self.factor*tf.abs(pos_aver-neg_aver)
        else:
            neg_aver = tf.abs(tf.reduce_min(tf.gather_nd(x, tf.where(x < 0.))))
            pos_aver = tf.reduce_max(tf.gather_nd(x, tf.where(x > 0.)))
            return self.factor*tf.abs(pos_aver-neg_aver)

@register_alias("quant_penalize_over_th")
@register_keras_custom_object
class RegularizerPenalizeOverValue(tf.keras.regularizers.Regularizer):

    def __init__(self, factor, threshold) -> None:
        super().__init__()
        self.factor = factor
        self.threshold = threshold

    def get_config(self):
        return {"factor":self.factor,
                "threshold":self.threshold}

    @tf.function
    def __call__(self, x):

            above_pos_th = tf.gather_nd(x, tf.where(x > self.threshold))
            below_neg_th = tf.gather_nd(x, tf.where(x < -self.threshold))
            percent_above = tf.cast(tf.size(above_pos_th), tf.float32)/tf.cast(tf.size(x), tf.float32)
            percent_below = tf.cast(tf.size(below_neg_th), tf.float32)/tf.cast(tf.size(x), tf.float32)

            if (tf.equal(tf.size(above_pos_th), 0) == True) and (tf.equal(tf.size(below_neg_th), 0) == True):
                return 0.0
            elif (tf.equal(tf.size(above_pos_th), 0) == True):
                return self.factor*((tf.abs(tf.reduce_mean(below_neg_th)-self.threshold))*percent_below)
            elif (tf.equal(tf.size(below_neg_th), 0) == True):
                return self.factor*((tf.reduce_mean(above_pos_th)-self.threshold)*percent_above)
            else:
                return self.factor*((tf.reduce_mean(above_pos_th)-self.threshold)*percent_above + \
                                    (tf.abs(tf.reduce_mean(below_neg_th)-self.threshold))*percent_below)

###########################################################################################################
###########################################################################################################
#######################             Base Layers     ##################################
###########################################################################################################
###########################################################################################################

def _compute_padded_size(stride, dilation_rate, input_size, filter_size):
    if input_size is None:
        return None
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    output_size = (input_size + stride - 1) // stride
    padded_size = (output_size - 1) * stride + effective_filter_size
    if tf.is_tensor(input_size):
        return tf.math.maximum(padded_size, input_size)
    return max(padded_size, input_size)


def _compute_padding(stride, dilation_rate, input_size, filter_size):
    padded_size = _compute_padded_size(stride, dilation_rate, input_size, filter_size)
    total_padding = padded_size - input_size
    padding = total_padding // 2
    return padding, padding + (total_padding % 2)

class BaseLayer(tf.keras.layers.Layer):
    """Base class for defining quantized layers.

    `input_quantizer` is the element-wise quantization functions to use.
    If `input_quantizer=None` this layer is equivalent to `tf.keras.layers.Layer`.
    """

    def __init__(self, *args, input_quantizer=None, **kwargs):
        self.input_quantizer = get_quantizer(input_quantizer)
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        return super().call(inputs)

    def get_config(self):
        return {
            **super().get_config(),
            "input_quantizer": serialize_quant(self.input_quantizer),
        }

class QuantizerBase(BaseLayer):
    """Base class for defining quantized layers with a single kernel.

    `kernel_quantizer` is the element-wise quantization functions to use.
    If `kernel_quantizer=None` this layer is equivalent to `BaseLayer`.
    """

    def __init__(self, *args, kernel_quantizer=None, **kwargs):
        self.kernel_quantizer = get_quantizer(kernel_quantizer)
        super().__init__(*args, **kwargs)

    def get_config(self):
        return {
            **super().get_config(),
            "kernel_quantizer": serialize_quant(self.kernel_quantizer),
        }

class QuantizerBaseConv(tf.keras.layers.Layer):
    """Base class for defining quantized conv layers"""

    def __init__(self, *args, pad_values=0.0, **kwargs):
        self.pad_values = pad_values
        super().__init__(*args, **kwargs)
        is_zero_padding = not tf.is_tensor(self.pad_values) and self.pad_values == 0.0
        self._is_native_padding = self.padding != "same" or is_zero_padding
        if self.padding == "causal" and not is_zero_padding:
            raise ValueError("Causal padding with `pad_values != 0` is not supported.")

    def _get_spatial_padding_same(self, shape):
        return [
            _compute_padding(stride, dilation_rate, shape[i], filter_size)
            for i, (stride, dilation_rate, filter_size) in enumerate(
                zip(self.strides, self.dilation_rate, self.kernel_size)
            )
        ]

    def _get_spatial_shape(self, input_shape):
        return (
            input_shape[1:-1]
            if self.data_format == "channels_last"
            else input_shape[2:]
        )

    def _get_padding_same(self, inputs):
        input_shape = inputs.shape
        if not input_shape[1:].is_fully_defined():
            input_shape = tf.shape(inputs)
        padding = self._get_spatial_padding_same(self._get_spatial_shape(input_shape))
        return (
            [[0, 0], *padding, [0, 0]]
            if self.data_format == "channels_last"
            else [[0, 0], [0, 0], *padding]
        )

    def _get_padding_same_shape(self, input_shape):
        spatial_input_shape = self._get_spatial_shape(input_shape)
        spatial_shape = [
            _compute_padded_size(stride, dilation, size, filter_size)
            for size, stride, dilation, filter_size in zip(
                spatial_input_shape,
                self.strides,
                self.dilation_rate,
                self.kernel_size,
            )
        ]
        if self.data_format == "channels_last":
            return tf.TensorShape([input_shape[0], *spatial_shape, input_shape[-1]])
        return tf.TensorShape([*input_shape[:2], *spatial_shape])

    def get_config(self):
        return {
            **super().get_config(),
            "pad_values": tf.keras.backend.get_value(self.pad_values),
        }

###########################################################################################################
###########################################################################################################

@register_keras_custom_object
class QuantDense(tf.keras.layers.Dense):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        bias_quantizer=None,
        activation_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        binarization_delay_calls=0,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.input_quantizer = get_quantizer(input_quantizer)
        self.kernel_quantizer = get_quantizer(kernel_quantizer)
        self.bias_quantizer = get_quantizer(bias_quantizer)
        self.activation_quantizer = get_quantizer(activation_quantizer)
        self.binarization_delay_calls = binarization_delay_calls

    def build(self, input_shape):
        super().build(input_shape)

        self.kernel_copy = self.add_weight(name='kernel_copy', 
                                           shape = self.kernel.get_shape(), 
                                           dtype = self.kernel.dtype, 
                                           trainable=False)
        self.kernel_copy.assign(self.kernel)
        if(self.use_bias == True):
            self.bias_copy = self.add_weight(name='bias_copy', 
                                           shape = self.bias.get_shape(), 
                                           dtype = self.bias.dtype, 
                                           trainable=False)
            self.bias_copy.assign(self.bias)

        self.num_calls = self.add_weight(name='num_calls', 
                                        dtype = tf.int32, 
                                        trainable=False)
        self.num_calls_to_check = self.add_weight(name='num_calls_to_check', 
                                        dtype = tf.int32, 
                                        trainable=False)
        self.num_calls_to_check = self.binarization_delay_calls

        '''
        self.pre_activation_scaling = self.add_weight(name='pre-activation_scaling_factor', 
                                        dtype = tf.float32, 
                                        trainable=True,
                                        initializer=tf.keras.initializers.Constant(1.0)
        '''

    def call(self, inputs, training=None):

        #outputs_float = super().call(inputs)

        if (self.num_calls >= self.num_calls_to_check):
            if(self.num_calls == self.num_calls_to_check):
                tf.print("Switched layer " + self.name + ' to binarization mode!')
                
            if (self.input_quantizer is not None):
                inputs_mod = self.input_quantizer(inputs)
            else:
                inputs_mod = inputs

            if (self.kernel_quantizer is not None):
                self.kernel_copy.assign(self.kernel)
                self.kernel.assign(self.kernel_quantizer(self.kernel_copy))
            
            if(self.use_bias == True) and (self.bias_quantizer is not None):
                self.bias_copy.assign(self.bias)
                self.bias.assign(self.bias_quantizer(self.bias_copy))

            outputs = super().call(inputs_mod)

            #if self.activation is None:
            #    outputs = outputs * self.pre_activation_scaling

            if (self.activation_quantizer is not None):
                outputs = self.activation_quantizer(outputs)

            if (self.kernel_quantizer is not None):
                self.kernel.assign(self.kernel_copy)

            if(self.use_bias == True) and (self.bias_quantizer is not None):
                self.bias.assign(self.bias_copy)
        else:
            outputs = super().call(inputs)

        if training == True:
            self.num_calls.assign_add(1)
        
        return outputs

    def get_config(self):
        ret_val = {
            **super().get_config()
        }
        if (hasattr(self, 'input_quantizer') == True):
            ret_val["input_quantizer"] = serialize_quant(self.input_quantizer)
        if (hasattr(self, 'kernel_quantizer') == True):
            ret_val["kernel_quantizer"] = serialize_quant(self.kernel_quantizer)
        if (hasattr(self, 'activation_quantizer') == True):
            ret_val["activation_quantizer"] = serialize_quant(self.activation_quantizer)
        if (hasattr(self, 'bias_quantizer') == True):
            ret_val["bias_quantizer"] = serialize_quant(self.bias_quantizer)
        return ret_val     

@register_keras_custom_object
class QuantConv2D(QuantizerBaseConv, tf.keras.layers.Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        pad_values=0.0,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        bias_quantizer=None,
        activation_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            pad_values=pad_values,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        if input_quantizer is not None:
            self.input_quantizer = get_quantizer(input_quantizer)
        if kernel_quantizer is not None: 
            self.kernel_quantizer = get_quantizer(kernel_quantizer)
        if bias_quantizer is not None:
            self.bias_quantizer = get_quantizer(bias_quantizer)
        if activation_quantizer is not None:
            self.activation_quantizer = get_quantizer(activation_quantizer)

    def build(self, input_shape):

        if(self._is_native_padding == True):
            super().build(input_shape)
        elif(self.padding == 'valid'):
            super().build(input_shape)
        elif(self.padding == 'same'):
            with patch_object(self, "padding", "valid"):
                super().build(self._get_padding_same_shape(input_shape))

        self.kernel_copy = self.add_weight(name='kernel_copy', 
                                           shape = self.kernel.get_shape(), 
                                           dtype = self.kernel.dtype, 
                                           trainable=False)
        self.kernel_copy.assign(self.kernel)
        if(self.use_bias == True):
            self.bias_copy = self.add_weight(name='bias_copy', 
                                           shape = self.bias.get_shape(), 
                                           dtype = self.bias.dtype, 
                                           trainable=False)
            self.bias_copy.assign(self.bias)

    def call(self, inputs):

        if(self.padding == 'same') and (not self._is_native_padding):
            inputs = tf.pad(inputs, self._get_padding_same(inputs), constant_values=self.pad_values)

        if (hasattr(self, 'input_quantizer') == True) and (self.input_quantizer is not None):
            input_mods = self.input_quantizer(inputs)
        else:
            input_mods = inputs

        if (hasattr(self, 'kernel_quantizer') == True) and (self.kernel_quantizer is not None):
            self.kernel_copy.assign(self.kernel)
            self.kernel.assign(self.kernel_quantizer(self.kernel_copy))
        
        if(self.use_bias == True) and (hasattr(self, 'bias_quantizer') == True) and (self.bias_quantizer is not None):
            self.bias_copy.assign(self.bias)
            self.bias.assign(self.bias_quantizer(self.bias_copy))

        if(self.padding == 'same') and (not self._is_native_padding):
            with patch_object(self, "padding", "valid"):
                outputs = super().call(input_mods)
        else:
            outputs = super().call(input_mods)

        if (hasattr(self, 'activation_quantizer') == True) and (self.activation_quantizer is not None):
            outputs = self.activation_quantizer(outputs)

        if (hasattr(self, 'kernel_quantizer') == True) and (self.kernel_quantizer is not None):
            self.kernel.assign(self.kernel_copy)

        if(self.use_bias == True) and (hasattr(self, 'bias_quantizer') == True) and (self.bias_quantizer is not None):
            self.bias.assign(self.bias_copy)
        
        return outputs

    def get_config(self):
        ret_val = {
            **super().get_config()
        }
        if (hasattr(self, 'input_quantizer') == True):
            ret_val["input_quantizer"] = serialize_quant(self.input_quantizer)
        if (hasattr(self, 'kernel_quantizer') == True):
            ret_val["kernel_quantizer"] = serialize_quant(self.kernel_quantizer)
        if (hasattr(self, 'activation_quantizer') == True):
            ret_val["activation_quantizer"] = serialize_quant(self.activation_quantizer)
        if (hasattr(self, 'bias_quantizer') == True):
            ret_val["bias_quantizer"] = serialize_quant(self.bias_quantizer)
        return ret_val

@register_keras_custom_object
class QuantDepthwiseConv2D(QuantizerBaseConv, tf.keras.layers.DepthwiseConv2D):

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        pad_values=0.0,
        depth_multiplier=1,
        data_format=None,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        depthwise_quantizer=None,
        bias_quantizer=None,
        activation_quantizer=None,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            pad_values=pad_values,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        if input_quantizer is not None:
            self.input_quantizer = get_quantizer(input_quantizer)
        if depthwise_quantizer is not None:
            self.depthwise_quantizer = get_quantizer(depthwise_quantizer)
        if bias_quantizer is not None:
            self.bias_quantizer = get_quantizer(bias_quantizer)
        if activation_quantizer is not None:
            self.activation_quantizer = get_quantizer(activation_quantizer)

    def build(self, input_shape):

        if(self._is_native_padding == True):
            super().build(input_shape)
        elif(self.padding == 'valid'):
            super().build(input_shape)
        elif(self.padding == 'same'):
            with patch_object(self, "padding", "valid"):
                super().build(self._get_padding_same_shape(input_shape))

        self.kernel_copy = self.add_weight(name='kernel_copy', 
                                           shape = self.depthwise_kernel.get_shape(), 
                                           dtype = self.depthwise_kernel.dtype, 
                                           trainable=False)
        self.kernel_copy.assign(self.depthwise_kernel)
        if(self.use_bias == True):
            self.bias_copy = self.add_weight(name='bias_copy', 
                                           shape = self.bias.get_shape(), 
                                           dtype = self.bias.dtype, 
                                           trainable=False)
            self.bias_copy.assign(self.bias)

    def call(self, inputs):

        if(self.padding == 'same') and (not self._is_native_padding):
            inputs = tf.pad(inputs, self._get_padding_same(inputs), constant_values=self.pad_values)

        if (hasattr(self, 'input_quantizer') == True) and (self.input_quantizer is not None):
            input_mods = self.input_quantizer(inputs)
        else:
            input_mods = inputs

        if (hasattr(self, 'depthwise_quantizer') == True) and (self.depthwise_quantizer is not None):
           self.kernel_copy.assign(self.depthwise_kernel)
           self.depthwise_kernel.assign(self.depthwise_quantizer(self.kernel_copy))                
        
        if(self.use_bias == True) and (hasattr(self, 'bias_quantizer') == True) and (self.bias_quantizer is not None):
            self.bias_copy.assign(self.bias)
            self.bias.assign(self.bias_quantizer(self.bias_copy)) 

        if(self.padding == 'same') and (not self._is_native_padding):
            with patch_object(self, "padding", "valid"):
                outputs = super().call(input_mods)
        else:
            outputs = super().call(input_mods)

        if (hasattr(self, 'activation_quantizer') == True) and (self.activation_quantizer is not None):
            outputs = self.activation_quantizer(outputs)

        if (hasattr(self, 'depthwise_quantizer') == True) and (self.depthwise_quantizer is not None):
           self.depthwise_kernel.assign(self.kernel_copy)

        if(self.use_bias == True) and (hasattr(self, 'bias_quantizer') == True) and (self.bias_quantizer is not None):
            self.bias.assign(self.bias_copy)
        
        return outputs

    def get_config(self):
        ret_val = {
            **super().get_config()
        }
        if (hasattr(self, 'input_quantizer') == True):
            ret_val["input_quantizer"] = serialize_quant(self.input_quantizer)
        if (hasattr(self, 'depthwise_quantizer') == True):
            ret_val["depthwise_quantizer"] = serialize_quant(self.depthwise_quantizer)
        if (hasattr(self, 'activation_quantizer') == True):
            ret_val["activation_quantizer"] = serialize_quant(self.activation_quantizer)
        if (hasattr(self, 'bias_quantizer') == True):
            ret_val["bias_quantizer"] = serialize_quant(self.bias_quantizer)
        return ret_val


@register_keras_custom_object
class QuantFakeQuantizationDense(tf.keras.layers.Dense):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        fake_quantizer=None,
        fake_quantizer_input=None,
        add_fake_quantization_to_activation=False,
        exponential_decay=0.9998,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        if fake_quantizer is not None:
            self.fake_quantizer = get_quantizer(fake_quantizer)
            self.add_fake_quantization_to_activation = add_fake_quantization_to_activation

            if self.add_fake_quantization_to_activation == True:
                self.ema_activation = MovingAverage(decay=exponential_decay)
                self.ema_acc = 0.0
                self.decay = exponential_decay
        else:
            tf.print("Fake quantization is not applied in FakeQuantizationDense!")
            self.add_fake_quantization_to_activation = False

        self.fake_quantizer_input = get_quantizer(fake_quantizer_input)

    def build(self, input_shape):
        super().build(input_shape)

        self.kernel_copy = self.add_weight(name='kernel_copy', 
                                           shape = self.kernel.get_shape(), 
                                           dtype = self.kernel.dtype, 
                                           trainable=False)
        self.kernel_copy.assign(self.kernel)
        if(self.use_bias == True):
            self.bias_copy = self.add_weight(name='bias_copy', 
                                           shape = self.bias.get_shape(), 
                                           dtype = self.bias.dtype, 
                                           trainable=False)
            self.bias_copy.assign(self.bias)

    def call(self, inputs, training=None):

        if(self.fake_quantizer_input is not None):
            inputs = self.fake_quantizer_input(inputs)

        if self.fake_quantizer is not None:
            self.kernel_copy.assign(self.kernel)
            self.kernel.assign(self.fake_quantizer(self.kernel_copy))            
        
            if(self.use_bias == True):
                self.bias_copy.assign(self.bias)
                self.bias.assign(self.fake_quantizer(self.bias_copy))

            outputs = super().call(inputs)

            self.kernel.assign(self.kernel_copy)

            if(self.use_bias == True):
                self.bias.assign(self.bias_copy)

            if (self.add_fake_quantization_to_activation == True) and (self.activation is not None) and \
               (self.activation is not 'linear'):
                curr_average = self.ema_activation(outputs)
                outputs = self.fake_quantizer(outputs, tf.math.reduce_min(curr_average), tf.math.reduce_max(curr_average))
        else:
            outputs = super().call(inputs)

        return outputs

    def get_config(self):
        ret_val = {
            **super().get_config()
        }
        if (hasattr(self, 'fake_quantizer') == True):
            ret_val["fake_quantizer"] = serialize_quant(self.fake_quantizer)
        ret_val["add_fake_quantization_to_activation"] = self.add_fake_quantization_to_activation
        if(self.fake_quantizer_input is not None):
            ret_val["fake_quantizer_input"] = serialize_quant(self.fake_quantizer_input)
        return ret_val

    def get_quantized_weights(self):

        output = []
        if self.fake_quantizer is not None:
            output.append(self.fake_quantizer(self.kernel))

            if self.use_bias == True:
                output.append(self.fake_quantizer(self.bias))
        else:
            output.append(self.kernel)
            output.append(self.bias)
        
        return output


@register_keras_custom_object
class QuantFakeQuantizationConv2D(QuantizerBaseConv, tf.keras.layers.Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        pad_values=0.0,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        fake_quantizer=None,
        add_fake_quantization_to_activation=False,
        exponential_decay=0.9998,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            pad_values=pad_values,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        if fake_quantizer is not None:
            self.fake_quantizer = get_quantizer(fake_quantizer)
            self.add_fake_quantization_to_activation = add_fake_quantization_to_activation

            if self.add_fake_quantization_to_activation == True:
                self.ema_activation = MovingAverage(decay=exponential_decay)
                self.ema_acc = 0.0
                self.decay = exponential_decay
        else:
            tf.print("Fake quantization is not applied FakeQuantizationConv2D!")
            self.add_fake_quantization_to_activation = False

    def build(self, input_shape):

        if(self._is_native_padding == True):
            super().build(input_shape)
        elif(self.padding == 'valid'):
            super().build(input_shape)
        elif(self.padding == 'same'):
            with patch_object(self, "padding", "valid"):
                super().build(self._get_padding_same_shape(input_shape))

        self.kernel_copy = self.add_weight(name='kernel_copy', 
                                           shape = self.kernel.get_shape(), 
                                           dtype = self.kernel.dtype, 
                                           trainable=False)
        self.kernel_copy.assign(self.kernel)
        if(self.use_bias == True):
            self.bias_copy = self.add_weight(name='bias_copy', 
                                           shape = self.bias.get_shape(), 
                                           dtype = self.bias.dtype, 
                                           trainable=False)
            self.bias_copy.assign(self.bias)

    def call(self, inputs):

        if(self.padding == 'same') and (not self._is_native_padding):
            inputs = tf.pad(inputs, self._get_padding_same(inputs), constant_values=self.pad_values)

        if self.fake_quantizer is not None:
            self.kernel_copy.assign(self.kernel)
            self.kernel.assign(self.fake_quantizer(self.kernel_copy))            
        
            if(self.use_bias == True):
                self.bias_copy.assign(self.bias)
                self.bias.assign(self.fake_quantizer(self.bias_copy))

            if(self.padding == 'same') and (not self._is_native_padding):
                with patch_object(self, "padding", "valid"):
                    outputs = super().call(inputs)
            else:
                outputs = super().call(inputs)

            self.kernel.assign(self.kernel_copy)

            if(self.use_bias == True):
                self.bias.assign(self.bias_copy)

            if (self.add_fake_quantization_to_activation == True) and (self.activation is not None) and \
               (self.activation is not 'linear'):
                curr_average = self.ema_activation(outputs)
                #curr_average = exponential_moving_average(outputs, self.ema_acc, self.decay)
                outputs = self.fake_quantizer(outputs, tf.math.reduce_min(curr_average), tf.math.reduce_max(curr_average))
        else:
            outputs = super().call(inputs)

        return outputs

    def get_config(self):
        ret_val = {
            **super().get_config()
        }
        if (hasattr(self, 'fake_quantizer') == True):
            ret_val["fake_quantizer"] = serialize_quant(self.fake_quantizer)
        ret_val["add_fake_quantization_to_activation"] = self.add_fake_quantization_to_activation
        return ret_val


@register_keras_custom_object
class QuantFakeQuantizationDepthwiseConv2D(QuantizerBaseConv, tf.keras.layers.DepthwiseConv2D):

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        pad_values=0.0,
        depth_multiplier=1,
        data_format=None,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        fake_quantizer=None,
        add_fake_quantization_to_activation=False,
        exponential_decay=0.9998,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            pad_values=pad_values,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        if fake_quantizer is not None:
            self.fake_quantizer = get_quantizer(fake_quantizer)
            self.add_fake_quantization_to_activation = add_fake_quantization_to_activation

            if self.add_fake_quantization_to_activation == True:
                self.ema_activation = MovingAverage(decay=exponential_decay)
                self.ema_acc = 0.0
                self.decay = exponential_decay
        else:
            tf.print("Fake quantization is not applied in FakeDepthWiseConv!")
            self.add_fake_quantization_to_activation = False

    def build(self, input_shape):

        if(self._is_native_padding == True):
            super().build(input_shape)
        elif(self.padding == 'valid'):
            super().build(input_shape)
        elif(self.padding == 'same'):
            with patch_object(self, "padding", "valid"):
                super().build(self._get_padding_same_shape(input_shape))

        self.kernel_copy = self.add_weight(name='kernel_copy', 
                                           shape = self.depthwise_kernel.get_shape(), 
                                           dtype = self.depthwise_kernel.dtype, 
                                           trainable=False)
        self.kernel_copy.assign(self.depthwise_kernel)
        if(self.use_bias == True):
            self.bias_copy = self.add_weight(name='bias_copy', 
                                           shape = self.bias.get_shape(), 
                                           dtype = self.bias.dtype, 
                                           trainable=False)
            self.bias_copy.assign(self.bias)

    def call(self, inputs):

        if(self.padding == 'same') and (not self._is_native_padding):
            inputs = tf.pad(inputs, self._get_padding_same(inputs), constant_values=self.pad_values)

        if self.fake_quantizer is not None:
            self.kernel_copy.assign(self.depthwise_kernel)
            self.depthwise_kernel.assign(self.fake_quantizer(self.kernel_copy))            
        
            if(self.use_bias == True):
                self.bias_copy.assign(self.bias)
                self.bias.assign(self.fake_quantizer(self.bias_copy))

            if(self.padding == 'same') and (not self._is_native_padding):
                with patch_object(self, "padding", "valid"):
                    outputs = super().call(inputs)
            else:
                outputs = super().call(inputs)

            self.depthwise_kernel.assign(self.kernel_copy)

            if(self.use_bias == True):
                self.bias.assign(self.bias_copy)

            if (self.add_fake_quantization_to_activation == True) and (self.activation is not None) and \
               (self.activation is not 'linear'):
                curr_average = self.ema_activation(outputs)
                #curr_average = exponential_moving_average(outputs, self.ema_acc, self.decay)
                outputs = self.fake_quantizer(outputs, tf.math.reduce_min(curr_average), tf.math.reduce_max(curr_average))
        else:
            outputs = super().call(inputs)

        return outputs

    def get_config(self):
        ret_val = {
            **super().get_config()
        }
        if (hasattr(self, 'fake_quantizer') == True):
            ret_val["fake_quantizer"] = serialize_quant(self.fake_quantizer)
        ret_val["add_fake_quantization_to_activation"] = self.add_fake_quantization_to_activation
        return ret_val

@register_keras_custom_object
class QuantFakeQuantizationAddLayer(tf.keras.layers.Layer):

    def __init__(self,
                 fake_quantizer=None,
                 exponential_decay=0.998,
                 regularization_factor=0.001,
                 add_quantization_to_operands=False,
                 init_delay=0,
                 **kwargs):
        super().__init__(**kwargs)

        self.regularization_factor = regularization_factor
        self.add_quantization_to_operands = add_quantization_to_operands
        if fake_quantizer is not None:
            self.fake_quantizer = get_quantizer(fake_quantizer)
            self.ema = MovingAverage(decay=exponential_decay, init_delay=init_delay)
        else:
            self.fake_quantizer = None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):

        x1, x2 = inputs

        if self.add_quantization_to_operands == True:
            if self.fake_quantizer is not None:
                x1 = self.fake_quantizer(x1, tf.math.reduce_min(x1), tf.math.reduce_max(x1))
                x2 = self.fake_quantizer(x2, tf.math.reduce_min(x2), tf.math.reduce_max(x2))

        output = tf.math.add(x1, x2)

        if self.fake_quantizer is not None:
            quantized = self.ema(output, training)
            output = self.fake_quantizer(output, tf.math.reduce_min(quantized), tf.math.reduce_max(quantized))
        
        return output

    def get_config(self):
        ret_val = {
                **super().get_config()
            }
        if self.fake_quantizer is not None:
            ret_val["fake_quantizer"] = self.fake_quantizer.get_config()
        if self.ema is not None:
            ret_val["exponential_average"] = self.ema.get_config()
        return ret_val

class QuantConv2DMixed(tf.keras.layers.Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        pad_values=0.0,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        lamba_regularization=None,
        clip_xnor_quant=False,
        bias_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_as_FP = True,
        **kwargs,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.pad_values = pad_values
        self.use_as_FP = use_as_FP
        self.input_quantizer = input_quantizer
        self.kernel_quantizer = kernel_quantizer
        self.bias_quantizer = bias_quantizer
        self.lamba_regularization = lamba_regularization
        self.clip_xnor_quant = clip_xnor_quant

    def build(self, input_shape):

        super().build(input_shape)
        
        if len(input_shape) == 3:
            in_h = input_shape[0]
            in_w = input_shape[1]
        elif len(input_shape) == 4:
            in_h = input_shape[1]
            in_w = input_shape[2]
            
        out_height = math.ceil(in_h / self.strides[0])
        out_width = math.ceil(in_w / self.strides[1])
        
        if (in_h % self.strides[0] == 0):
          pad_along_height = max(self.kernel_size[0] - self.strides[0], 0)
        else:
          pad_along_height = max(self.kernel_size[0] - (in_h % self.strides[0]), 0)
        
        if (in_w % self.strides[1] == 0):
          pad_along_width = max(self.kernel_size[1] - self.strides[1], 0)
        else:
          pad_along_width = max(self.kernel_size[1] - (in_w % self.strides[1]), 0)
        
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        
        self.paddings_values = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

    def call(self, inputs):
        
        if self.use_as_FP == False:

            if(self.padding == 'same'):            
                input_mods = tf.pad(inputs, self.paddings_values, mode='CONSTANT', constant_values=self.pad_values)
            else:
                input_mods = inputs
                
            # quantize input
            if self.input_quantizer is not None:
                input_mods = self.input_quantizer(input_mods)

            # quantize kernel
            if self.kernel_quantizer is not None:
                w_bin = self.kernel_quantizer(self.kernel)
            else:
                w_bin = self.kernel

            # Convolution
            outputs = tf.nn.conv2d(input_mods, w_bin, 
                                   strides=[1, self.strides[0], self.strides[1], 1], 
                                   padding="VALID",
                                   data_format='NHWC', 
                                   dilations=[1, self.dilation_rate[0], self.dilation_rate[1], 1])
            
            if (self.lamba_regularization is not None):
                self.add_loss(tf.math.reduce_std(outputs)*self.lamba_regularization)

            if(self.use_bias == True):
                if self.bias_quantizer is not None:
                    bias_bin = self.bias_quantizer(self.bias)
                else:
                    bias_bin = self.bias
                outputs = tf.nn.bias_add(outputs, bias_bin)

            if(self.clip_xnor_quant == True):
                outputs = tf.clip_by_value(outputs, -127.0, 127.0)

            if (self.activation is not None):
                outputs = self.activation(outputs)
        
        else:
            
            if(self.padding == 'same'):            
                input_mods = tf.pad(inputs, self.paddings_values, mode='CONSTANT', constant_values=self.pad_values)
            else:
                input_mods = inputs
                
            # Convolution
            outputs = tf.nn.conv2d(input_mods, self.kernel, 
                                   strides=[1, self.strides[0], self.strides[1], 1], 
                                   padding="VALID",
                                   data_format='NHWC', 
                                   dilations=[1, self.dilation_rate[0], self.dilation_rate[1], 1])

            if(self.use_bias == True):
                outputs = tf.nn.bias_add(outputs, self.bias)

            if(self.clip_xnor_quant == True):
                outputs = tf.clip_by_value(outputs, -127.0, 127.0)

            if (self.activation is not None):
                outputs = self.activation(outputs)
        
        return outputs

    def get_config(self):
        ret_val = {
            **super().get_config(),
            'use_as_FP': self.use_as_FP,
            'pad_values': self.pad_values,
            'clip_xnor_quant': self.clip_xnor_quant,
            'input_quantizer' : self.input_quantizer,
            'kernel_quantizer' : self.kernel_quantizer,
            'bias_quantizer' : self.bias_quantizer
        }
        return ret_val
    
class QuantDenseMixed(tf.keras.layers.Dense):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        bias_quantizer=None,
        clip_xnor_quant=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_as_FP = True,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.input_quantizer = input_quantizer
        self.kernel_quantizer = kernel_quantizer
        self.bias_quantizer = bias_quantizer
        self.use_as_FP = use_as_FP
        self.clip_xnor_quant = clip_xnor_quant

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        
        if self.use_as_FP == True:
            
            outputs = super().call(inputs)

            #if(self.clip_xnor_quant == True):
            #    outputs = tf.clip_by_value(outputs, -1.0, +1.0)
            #outputs = tf.clip_by_value(outputs, -1.0, +1.0)
            
            return outputs
            
        else:
            
            if self.input_quantizer is not None:
                inputs_mod = self.input_quantizer(inputs)
            else:
                inputs_mod = inputs
            
            if self.kernel_quantizer is not None:
                w_bin = self.kernel_quantizer(self.kernel)
            else:
                w_bin = self.kernel
            
            outputs = tf.linalg.matmul(inputs_mod, w_bin)
            
            if(self.use_bias == True):
                if self.bias_quantizer is not None:
                    bias_bin = self.bias_quantizer(self.bias)
                else:
                    bias_bin = self.bias
                outputs = tf.nn.bias_add(outputs, bias_bin)

            if(self.clip_xnor_quant == True):
                outputs = tf.clip_by_value(outputs, -127.0, +127.0)
                
            if (self.activation is not None):
                outputs = self.activation(outputs)
        
            return outputs

    def get_config(self):
        ret_val = {
            **super().get_config()
        }
        ret_val['use_as_FP'] = self.use_as_FP
        ret_val['clip_xnor_quant'] = self.clip_xnor_quant
        ret_val['input_quantizer'] = self.input_quantizer
        ret_val['kernel_quantizer'] = self.kernel_quantizer
        ret_val['bias_quantizer'] = self.bias_quantizer
        return ret_val

def use_model_as_FP(model, val, count=-1):
    num = 0
    for layer in model.layers:
        if (isinstance(layer, QuantConv2DMixed) == True) or (isinstance(layer, QuantDenseMixed) == True):
            layer.use_as_FP = val
            num += 1
            if(count > 0) and (num >= count):
                break

def SetAttributeToLayers(model, attribute_name, attribute_value):
    for layer in model.layers:
        if hasattr(layer, attribute_name) == True:
            setattr(layer, attribute_name, attribute_value)

class DoubleLeakyReLU(tf.keras.layers.Layer):

    def __init__(self, alpha=0.02, max_value=1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.max_value = max_value
        self.slope = tf.Variable(initial_value=alpha, trainable=False, dtype=tf.float32, name=self.name + '_slope_activation')
        self.is_custom_activation_function = True

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config()}

    def call(self, inputs, training=None):

        if training:
            result = tf.where(inputs < 0.0, self.slope*inputs, tf.where(inputs > self.max_value, (inputs*self.slope)+(self.max_value-self.slope), inputs))
        else:
            result = tf.where(inputs < 0.0, 0.0, tf.where(inputs > self.max_value, self.max_value, inputs))
        return result

class DPReLU(tf.keras.layers.Layer):

    def __init__(self, alpha=0.0, betha=0.0, etha=1.0, gamma=0.25, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self._alpha = alpha
        self._betha = betha
        self._etha = etha
        self._gamma = gamma

    def build(self, input_shape):
        super().build(input_shape)
        
        input_shape = tf.TensorShape(input_shape)
        self.alpha = self.add_weight(name='alpha', shape=(input_shape[-1]),
                                    initializer=tf.keras.initializers.Constant(self._alpha),
                                    dtype=tf.float32, trainable=True)
        self.betha = self.add_weight(name='betha', shape=(input_shape[-1]),
                                    initializer=tf.keras.initializers.Constant(self._betha),
                                    dtype=tf.float32, trainable=True)
        self.etha = self.add_weight(name='etha', shape=(input_shape[-1]),
                                    initializer=tf.keras.initializers.Constant(self._etha),
                                    dtype=tf.float32, trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1]),
                                    initializer=tf.keras.initializers.Constant(self._gamma),
                                    dtype=tf.float32, trainable=True)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config()}

    def call(self, inputs, training=None):        
        
        result = tf.where(inputs-self.alpha >= 0.0, 
                          self.etha*(inputs-self.alpha)-self.betha, 
                          self.gamma*(inputs-self.alpha)-self.betha)
        
        return result

    def get_tau_1(self):
        return (self.alpha + (self.betha/self.etha))
    
    def get_tau_2(self):
        return (self.alpha + (self.betha/self.gamma))
                