import numpy as np
import tensorflow as tf
from tensorflow import keras
import BNF as bnf


class QuantClip(tf.keras.layers.Layer):
    """Custom Layer to Implement Clipping logic

    This layer simply clips its input relative to the passed
    q_range value and rounds value to closes quantized value based
    on q_scale

    Args:
      * q_scale (int): sets the quantization scale of the output
      * q_range (int): sets the range of the output

    Returns:
      * quantized clipped input value

    """
    def __init__(self, q_scale, q_range, **kwargs):
        super().__init__(**kwargs)
        self.q_scale = q_scale
        self.q_range = q_range

    def call(self, x):
        quantized = tf.math.floor(2**self.q_scale * x)
        saturated = tf.clip_by_value(quantized, -2**self.q_range,
                                     2**self.q_range - 1)
        return saturated / (2**self.q_scale)

    def get_config(self):
        return {
            **super().get_config(), 'q_scale': self.q_scale,
            'q_range': self.q_range
        }


def _Insert_layer2Sequential(model, to_insert, pos):
    """Used to add new layers to existing Sequential Model

    Args:
      * model: existing model to which new layer will be inserted
      * to_insert: the new layer to be inserted
      * pos: the position of the new layer

    Returns:
      * New model with the inserted layer
    """
    layers = model.layers
    layers.insert(pos, to_insert)
    #new_model = keras.Sequential(layers=layers)
    new_model = tf.keras.Model(inputs=model.input, outputs=model.output)
    return new_model


def _KeraShape2Np(shape):
    """Converts a keras tensor shape into a numpy shape

    Namely it replaces all 'None' shape paramters to '-1' to be compliant with numpy shape.

    Args:
      * shape: Keras or other shape

    Returns:
      * original shape passed with all 'None's converted to '-1's
    """

    return tuple(-1 if d is None else d for d in shape)


def _layer_predict(layer, input_data):
    """Simplifed method to get output of a model layer given a specified input

    Function reshapes input to expected converts expected tensor shape, converts the
    reshaped data into a tensor, inputs this tensor into the layer, and then converts the output
    back to a numpy array

    Args:
      * layer: layer to be called
      * input_data: numpy array of inputs to layer

    Returns:
      * numpy array of layer outputs.
    """
    x = input_data.reshape(_KeraShape2Np(layer.input.shape))
    # no layer methods for batch-split processing, upgrade to model
    return tf.keras.Sequential([layer]).predict(x)

def _subnet_predict(model, layer_idx, data):
    """Computes output of a subsection of a sequential model

    Args:
      * model: full model from which subnet will be produced
      * layer_idx: layer index of the model from which output should be extracted
      * data: input data into the model

    Returns:
      * The output of a sequential model at layer_idx

    """
    '''
    layers = model.layers[:layer_idx]
    #subnet = keras.Sequential(layers=layers)
    subnet = tf.keras.Model(inputs=model.input, outputs=[layers[-1].output])
    return subnet.predict(data)
    '''
    new_model_output = tf.keras.Model(inputs=model.input, outputs=model.layers[layer_idx].output)
    return new_model_output.predict(data)


def _check_if_output(model, layer):
    """Checks if layer provided is an output layer

    This is important for functional models with multiple output heads and
    for settig the "Is_Output" section of the Quantization Dict

    Args:
      * model: model containing passed layer
      * layer: layer to check

    Returns:
      * Boolean of whether passed layer is an output layer

    ISSUES:
      * If seperate activation layers are used last dense layer won't be flagged
        as output with current set up.
    """
    if isinstance(model.output, tuple) or isinstance(model.output, list):
        for output in model.output:
                if output.name.split('/')[0] == layer.name:
                    return True
    else:
        if model.output.name.split('/')[0] == layer.name:
            return True

    return False


def _getSDQs8(_min, _max):
    """Provides SDQ from passed min, max pair for signed 8bit range

    SDQ stands for (Sign, Dynamic range, Quantization scale)
    This Function is used to find the SDQ of inputs, weights, and layer outputs

    Args:
      * _min: mininum of value range
      * _max: maximum of value range

    Returns:
      * SDQ of provided range
    """
    SDQ = {'S': 1, 'D': 0, 'Q': 0}
    SDQ['D'] = int(np.clip(np.ceil(np.log2(max(abs(_min), abs(_max)))), 0, 7))
    SDQ['Q'] = 7 - SDQ['D']
    return SDQ

def _getSDQs16(_min, _max):
    """Provides SDQ from passed min, max pair for signed 8bit range

    SDQ stands for (Sign, Dynamic range, Quantization scale)
    This Function is used to find the SDQ of inputs, weights, and layer outputs

    Args:
      * _min: mininum of value range
      * _max: maximum of value range

    Returns:
      * SDQ of provided range
    """
    SDQ = {'S': 1, 'D': 0, 'Q': 0}
    SDQ['D'] = int(np.clip(np.ceil(np.log2(max(abs(_min), abs(_max)))), 0, 15))
    SDQ['Q'] = 15 - SDQ['D']
    return SDQ


def _getSDQs16s8xs8(SDQ1, SDQ2):
    """Provides SDQ in signed 16bit range of two signed 8bit SDQs

    SDQ stands for (Sign, Dynamic range, Quantization scale)
    This function is used to create the SDQ of the network activations

    Args:
      * SDQ1 and SDQ2: SDQ of signed 8bit

    Returns:
      * signed 16 bit SDQ of two signed 8bit SDQs
    """
    return {
        'S': 1,
        'D': 15 - SDQ1['Q'] - SDQ2['Q'],
        'Q': SDQ1['Q'] + SDQ2['Q']
    }


def _Float_Layer(layer, inputs, is_output_layer):
    """Used in Quantize_Network function when a layer is not flagged to be quantized

    Simply sets layer as untrainable and passes expected dict back to Quantize Network

    Args:
      * layer: layer to be left floating point
      * inputs: output of previous layer
      * is_output_layer: boolean describing if current layer is an output layer

    Returns:
      * Dict containing layer quantization information
    """

    layer.trainable = False

    return {
        "WeightsQ": layer.weights[0].numpy(),
        "BiasesQ": layer.weights[1].numpy(),
        "Shift": 0,
        "InputSDQ": {
            'S': 0,
            'D': 0,
            'Q': 0
        },
        "WeightsSDQ": {
            'S': 0,
            'D': 0,
            'Q': 0
        },
        "OutputSDQ": {
            'S': 0,
            'D': 0,
            'Q': 0
        },
        #Generate New Output using quantized parameters
        "Output": _layer_predict(layer, inputs),
        "Is_Output_Layer": int(is_output_layer)
    }


def _Quantize_Layer(layer, inputs, inputSDQ, quantiles, is_output_layer,
                    quant_next_layer):
    """Quantizes the passed layer per the parameters passed

    This is done in the following steps:

      * Concatenating the weights and biases and determining their SDQ (sign, dynamic range, quanization step) values
      * Determining the SDQ of the output of the layer, cliped by provided quantile range
      * Determining the SDQ of the Activation (weights * inputs)
      * Ensuring that the dynamic range of the Activation is at least equal to that of the Output
        * There is some question as to the usefullness of this check, assuming all is correct there
          should be no circumstance in which this is needed
      * Calculate the floating point equivalent of quantized weights and biases
      * Reassign the weights and biases of this layer these new parameters and set trainable attribute to False
      * Generate output of layer based on quantization flag of next layer:
        * if next layer is quantized: clip to int8 using output quanztization scale
        * if next layer is not quantized: clip to int16 using weightsXinput quantization scale
      * Create Dict contraining layer quantization information

    Args:
      * layer: layer to be quantized (keras.layers.Dense)
      * inputs: inputs to this layer (numpy array)
      * inputSDQ: SDQ of inputs to this layer (Dict from _getSDQs8)
      * quantiles: quantiles of layer output to be used for activation quantization (two element list)
      * is_output_layer: boolean flagging current layer as an output layer (Boolean)
      * quant_next_layer: booleanin flaggin if next layer will be quantized (Boolean)

    Returns:
      * Dict containing layer quantization information

    Improvements:
      * Determine if weight biases concatentation needed...

    """

    #concatenate weights and biases
    if len(layer.weights) > 1:
        w_b = np.append(layer.weights[0].numpy(), layer.weights[1].numpy())
    else:
        w_b = layer.weights[0].numpy()

    #Determine Sign, Dynamic Range, and Quantization Step of inputs, weights/biases, and output.
    weightsSDQ = _getSDQs8(np.min(w_b), np.max(w_b))
    quantiles = np.quantile(_layer_predict(layer, inputs),
                            quantiles,
                            interpolation='nearest')
    print("Layer Output Quantiles: " + str(quantiles))
    outputSDQ = _getSDQs8(quantiles[0], quantiles[1])
    weightsXinputSDQ = _getSDQs16s8xs8(inputSDQ, weightsSDQ)

    #Ensure Activation Dynamic Range At Least Equal to Output Rnage
    if quant_next_layer:
        while outputSDQ['D'] > weightsXinputSDQ['D']:
            weightsSDQ['D'] += 1
            weightsSDQ['Q'] -= 1
            weightsXinputSDQ = _getSDQs16s8xs8(inputSDQ, weightsSDQ)

    #Generate New Quantized Parameters
    if len(layer.weights) > 1:
        
        # Round-to-nearest version
        #weights = 1.0 / 2**weightsSDQ['Q'] * np.clip(np.round(2**weightsSDQ['Q'] * layer.weights[0].numpy()), -2**7 + 1, 2**7 - 1)
        
        # Round-to-zero version
        tmp = 2**weightsSDQ['Q'] * layer.weights[0].numpy()
        weights = 1.0 / 2**weightsSDQ['Q'] * np.clip(np.where(tmp >= 0.0, np.round(tmp), np.ceil(tmp)), -2**7 + 1, 2**7 - 1)
        
        #potential bug in Wolfram, range should be -2^15-1->2^15-1
        # Round-to-nearest version
        #biases = 1.0 / 2**weightsXinputSDQ['Q'] * np.clip(np.round(2**weightsXinputSDQ['Q'] * layer.weights[1].numpy()), -2**15, 2**15 - 1)
        
        # Round-to-zero version
        tmp = 2**weightsXinputSDQ['Q'] * layer.weights[1].numpy()
        biases = 1.0 / 2**weightsXinputSDQ['Q'] * np.clip(np.where(tmp >= 0.0, np.round(tmp), np.ceil(tmp)), -2**15 + 1, 2**15 - 1)
        
        #Layer with Quantized Parameters
        layer.weights[0].assign(weights)
        layer.weights[1].assign(biases)
    else:
        # Round-to-nearest version
        #weights = 1.0 / 2**weightsSDQ['Q'] * np.clip(np.round(2**weightsSDQ['Q'] * layer.weights[0].numpy()), -2**7 + 1,  2**7 - 1)
        
        # Round-to-zero version
        tmp = 2**weightsSDQ['Q'] * layer.weights[0].numpy()
        weights = 1.0 / 2**weightsSDQ['Q'] * np.clip(np.where(tmp >= 0.0, np.round(tmp), np.ceil(tmp)), -2**7 + 1, 2**7 - 1)
        
        layer.weights[0].assign(weights)
    layer.trainable = False

    #Generate New Output using quantized parameters
    if quant_next_layer:
        layer_output = 1.0 / 2**outputSDQ['Q'] * np.clip(
            np.floor(2**outputSDQ['Q'] * _layer_predict(layer, inputs)), -2**7,
            2**7 - 1)
    else:
        layer_output = 1.0 / 2**weightsXinputSDQ['Q'] * np.clip(
            np.floor(2**weightsXinputSDQ['Q'] * _layer_predict(layer, inputs)),
            -2**15, 2**15 - 1)

    return {
        "WeightsQ": np.round(2**weightsSDQ['Q'] * weights).astype(np.int8),
        "BiasesQ":
        np.round(2**weightsXinputSDQ['Q'] * biases).astype(np.int16) if len(layer.weights) > 1 else None,
        "Shift": weightsXinputSDQ['Q'] - outputSDQ['Q'],
        "InputSDQ": inputSDQ,
        "WeightsSDQ": weightsSDQ,
        "OutputSDQ": outputSDQ,
        "Output": layer_output,
        "Is_Output_Layer": int(is_output_layer)
    }


def Quantize_Network(model_path, train_set, train_labels, quantiles,
                     batch_size, epochs, learning_rate, to_quant):
    """Quantizes model saved at model_path into a VAL compliant quantized model

    This functions sets through a Sequential model and quantizes each Dense layer per the to_quant list.
    For each layer the weights and biases are set to discrete quantized value, their trainable attribute is set to false,
    and a lambda layer is inserted to restrict the clip the output of the layer to within the quantizable range.
    The q_model is then recompiled (to apply the trainable attribute change) and retrained with the train_set and labels
    to adjust the network to the quantized layer. This is repeated for each Dense layer in the model.

    Args:
      * model_path (str): path for the saved model to be quantized
      * train_set (np.array, dtype=np.float32): data set used to determine quantization ranges
          and retrain the network after layer quantization
      * train_lables (np.array): labels for the passed train_set
      * quantiles (two element float list): quantiles used to determine layer output ranges for quantization.
          expected is [_min, _max]
      * batch_size (int): batch_size used during retraining
      * epochs (int): number of epochs to use during retraining
      * learning_rate (float): learning rate to be using during retraining
      * to_quant (boolean list): flags for each dense layer specifing if it is to be quantized

    Returns:
      * q_model (keras.models.Sequential): quantized model with inserted lambda layers to clip the output of each layer
      * Quantization_result (list of dicts): list of all the quantization information for each layer. To be used with
          CreateJsonDict to produced VAL json file.

    Improvements:
      * Add support for final output to be float
      * Add support for functional models with multiple output heads
      * Add support for float->int8 conversions
      * Consider adding deminishing learning_rate with each quantized layer or option for list of learning_rates

    """
    q_model = keras.models.load_model(model_path)

    #Create InputSDQ, Should it be quantiled?
    inputSDQ = _getSDQs8(np.min(train_set), np.max(train_set))
    inputs = train_set
    hidden_layer_idx = 0
    quant_idx = 0
    layer_idx = 1
    Quantization_results = []

    #check if current layer flaged to be quantized
    for layer in q_model.layers:
        #check if layer is a Dense layer
        if (not isinstance(layer, keras.layers.Dense)) or (isinstance(layer, bnf.QuantDense)):
            layer_idx += 1
            layer.trainable = False
            continue

        is_output = _check_if_output(q_model, layer)

        if to_quant[quant_idx]:
            try:
                quant_next_layer = to_quant[quant_idx + 1]
            except IndexError:
                #add float out option here later
                quant_next_layer = True
            quant_res = _Quantize_Layer(layer, inputs, inputSDQ, quantiles,
                                        is_output, quant_next_layer)
            Quantization_results.append(quant_res)

            if quant_next_layer:
                q_out = quant_res['OutputSDQ']['Q']
                q_model = _Insert_layer2Sequential(q_model,
                                                   QuantClip(q_out,
                                                             7), layer_idx)
            else:
                q_out = quant_res['Shift'] + quant_res['OutputSDQ']['Q']
                q_model = _Insert_layer2Sequential(q_model,
                                                   QuantClip(q_out,
                                                             15), layer_idx)
            layer_idx += 1

            #consider reducing learning rate with every iteration (i.e. learning_rate /= 2)
            q_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse')
            if not is_output:
                inputs = _subnet_predict(q_model, layer_idx, train_set)
                inputSDQ = quant_res['OutputSDQ']
                hidden_layer_idx += 1

                q_model.fit(train_set,
                            train_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0)
        else:
            quant_res = _Float_Layer(layer, inputs, is_output)
            Quantization_results.append(quant_res)
            #consider reducing learning rate with every iteration (i.e. learning_rate /= 2)
            q_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse')
            if not is_output:
                inputs = _subnet_predict(q_model, layer_idx, train_set)
                inputSDQ = quant_res['OutputSDQ']
                hidden_layer_idx += 1
        layer_idx += 1
        quant_idx += 1
        print("Dense Layer " + str(quant_idx) + " Quantized")

    return q_model, Quantization_results

def Quantize_NetworkCustomLoop(init_q_model, train_samples, quantiles, train_loop_callback,
                               epochs, optimizer, to_quant, quantized_model_path):
    """Quantizes model saved at model_path into a VAL compliant quantized model

    This functions sets through a Sequential model and quantizes each Dense layer per the to_quant list.
    For each layer the weights and biases are set to discrete quantized value, their trainable attribute is set to false,
    and a lambda layer is inserted to restrict the clip the output of the layer to within the quantizable range.
    The q_model is then recompiled (to apply the trainable attribute change) and retrained with the train_samples and labels
    to adjust the network to the quantized layer. This is repeated for each Dense layer in the model.

    Args:
      * model_path (str): path for the saved model to be quantized
      * train_samples (np.array, dtype=np.float32): data set used to determine quantization ranges
          and retrain the network after layer quantization
      * train_lables (np.array): labels for the passed train_samples
      * quantiles (two element float list): quantiles used to determine layer output ranges for quantization.
          expected is [_min, _max]
      * batch_size (int): batch_size used during retraining
      * epochs (int): number of epochs to use during retraining
      * learning_rate (float): learning rate to be using during retraining
      * to_quant (boolean list): flags for each dense layer specifing if it is to be quantized

    Returns:
      * q_model (keras.models.Sequential): quantized model with inserted lambda layers to clip the output of each layer
      * Quantization_result (list of dicts): list of all the quantization information for each layer. To be used with
          CreateJsonDict to produced VAL json file.

    Improvements:
      * Add support for final output to be float
      * Add support for functional models with multiple output heads
      * Add support for float->int8 conversions
      * Consider adding deminishing learning_rate with each quantized layer or option for list of learning_rates

    """
    #q_model = tf.keras.models.clone_model(init_q_model)
    q_model = init_q_model

    #Create InputSDQ, Should it be quantiled?
    inputSDQ = _getSDQs8(np.min(train_samples), np.max(train_samples))
    inputs = train_samples
    hidden_layer_idx = 0
    quant_idx = 0
    layer_idx = 1
    Quantization_results = []

    #check if current layer flaged to be quantized
    for layer in q_model.layers:
        #check if layer is a Dense layer
        if (isinstance(layer, keras.layers.Dense) == False) or (isinstance(layer, bnf.QuantDense) == True):
            layer_idx += 1
            layer.trainable = False
            continue

        is_output = _check_if_output(q_model, layer)
        print('Layer to be quantized: ' + layer.name)

        if to_quant[quant_idx]:
            try:
                quant_next_layer = to_quant[quant_idx + 1]
            except IndexError:
                #add float out option here later
                quant_next_layer = True
            quant_res = _Quantize_Layer(layer, inputs, inputSDQ, quantiles,
                                        is_output, quant_next_layer)
            Quantization_results.append(quant_res)

            if quant_next_layer:
                q_out = quant_res['OutputSDQ']['Q']
                q_model = _Insert_layer2Sequential(q_model,
                                                   QuantClip(q_out,
                                                             7), layer_idx)
            else:
                q_out = quant_res['Shift'] + quant_res['OutputSDQ']['Q']
                q_model = _Insert_layer2Sequential(q_model,
                                                   QuantClip(q_out,
                                                             15), layer_idx)
            layer_idx += 1

            if not is_output:
                inputs = _subnet_predict(q_model, layer_idx, train_samples)
                inputSDQ = quant_res['OutputSDQ']
                hidden_layer_idx += 1
                
                if train_loop_callback is not None:
                    q_model = train_loop_callback(q_model, optimizer, epochs, quantized_model_path)
        else:
            quant_res = _Float_Layer(layer, inputs, is_output)
            Quantization_results.append(quant_res)
            
            
            
            if not is_output:
                inputs = _subnet_predict(q_model, layer_idx, train_samples)
                inputSDQ = quant_res['OutputSDQ']
                hidden_layer_idx += 1
        layer_idx += 1
        quant_idx += 1
        print("Dense Layer " + str(quant_idx) + " Quantized")

    return q_model, Quantization_results

def Quantize_NetworkCustomTrainingLoop_ApplyClipping(model, train_samples, quantiles, train_loop_callback, layer_to_quantize, epochs, optimizer, quantized_model_path):
    
    clipping_layers = []
    index_layers_quantized = []
    quantization_results = []
    
    num_dense_layers_processed = 0
    
    inputs = train_samples
    inputSDQ = _getSDQs8(np.min(inputs), np.max(inputs))
    
    for idx_layer, layer in enumerate(model.layers):
        
        if (isinstance(layer, tf.keras.layers.Dense) == False) and (isinstance(layer, tf.keras.layers.Conv2D) == False):
            continue
            
        index_layers_quantized.append(idx_layer)
        
        if layer_to_quantize[num_dense_layers_processed] == True:
        
            if _check_if_output(model, layer) == True:

                print("Quantizing output layer: " + layer.name)

                quant_res = _Quantize_Layer(layer, inputs, inputSDQ, quantiles, True, False)
                quantization_results.append(quant_res)

                q_out = quant_res['Shift'] + quant_res['OutputSDQ']['Q']
                #clipping_layers.append({'idx_layer': idx_layer, 'clip': QuantClip(q_out, 15)})

            else:

                print("Quantizing layer: " + layer.name + " with index " + str(idx_layer))

                quant_res = _Quantize_Layer(layer, inputs, inputSDQ, quantiles, False, True)
                quantization_results.append(quant_res)

                q_out = quant_res['OutputSDQ']['Q']
                inputs = _subnet_predict(model, idx_layer, train_samples)

                if train_loop_callback is not None:
                        model = train_loop_callback(model, optimizer, epochs, quantized_model_path)

                # Apply clipping values
                inputs = QuantClip(q_out, 7)(inputs).numpy()
                inputSDQ = quant_res['OutputSDQ']
                #clipping_layers.append({'idx_layer': idx_layer, 'clip': QuantClip(q_out, 7)})
        else:
            
            is_output = _check_if_output(model, layer)
            
            quant_res = _Float_Layer(layer, inputs, is_output)
            quantization_results.append(quant_res)
            
            if is_output == False:
                inputs = _subnet_predict(model, idx_layer, train_samples)
                inputSDQ = quant_res['OutputSDQ']
                
        num_dense_layers_processed += 1
    
    return model, quantization_results


def _GenJsonLayer(in_type, accum_type, out_type, activation, quant_params):
    """Creates a dict based off inputed values

    Args:
      * in_type (str): input type of layer. Accepted values are: "int8_t", "float"
      * accum_type (str): accumulator type of layer. Accepted values are: "int16_t", "float"
      * out_type (str): output type of layer. Accepted values are: "int8_t", "float"
      * activation (str): activation type of layer. Accepted values are: "Double ReLU", "ReLU", "None"
      * quant_params (dict): dict element produced by Quantize_Network

    Returns:
      * VAL_dict (dict): a dict that contains all the information needed to create json object
          describing a Fully Connected layer

    """
    VAL_dict = {
        'Input Type': in_type,
        'Accumulator Type': accum_type,
        'Output Type': out_type,
        'Activation': activation,
        'Quantization Scale': quant_params['Shift'],
        'Fractional Bits': quant_params['OutputSDQ']['Q'],
        'Neurons Row Major': 1,
        'Is Output': quant_params['Is_Output_Layer'],
        'Weights': quant_params['WeightsQ'].tolist(),
        'Biases': quant_params['BiasesQ'].tolist()
    }
    return VAL_dict


#add check for activations is list
#add check that len of quant == acts
#add check that activation is valid
def CreateJsonDict(name, quant_params, activations, output_type):
    """Creates a dict in the VAL json format

    This function takes the Quantization_results output from the Quantize_Network function
    and converts it into the format expected by the VAL json interpreter

    Args:
      * name (str): string name for the network (will be used as the namespace name when converted to C++)
      * quant_params (dict): result from the Quantize_Network function
      * activations (str list): list describing the activation function for each layer. Accepted values are:
          ["Double ReLU", "ReLU", "None"]
      * output_type (str): specify the final output type of the network. Accepted values are ["float", "int8_t"]

    Returns:
      * dicts (list of dicts): list of dicts that can be passed to json.dumps to create the json file used by
          CreateFCHeader.py script.

    Improvements:
      * Add explict checking of activations and output_type to ensure the comply with specifed accepted values

    """
    dicts = {name: []}
    for i in range(len(quant_params)):
        if quant_params[i]['WeightsQ'].dtype == np.int8:
            in_type = "int8_t"
            accum_type = "int16_t"
        else:
            in_type = "float"
            accum_type = "float"

        if quant_params[i]['Is_Output_Layer'] != 1:
            if quant_params[i + 1]['WeightsQ'].dtype == np.int8:
                out_type = "int8_t"
            else:
                out_type = "float"
        else:
            out_type = output_type

        dicts[name].append(
            _GenJsonLayer(in_type, accum_type, out_type, activations[i],
                          quant_params[i]))

    return dicts
