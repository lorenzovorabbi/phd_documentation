#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import partial
import glob
import pickle as pkl
import argparse
import random 
import time

import CORE50_training_script as core50
import Utilities as uti

def RunTFLiteModel(tflite_model, test_ds, num_test_samples, is_path = False):

    if(is_path == True):
        interpreter = tf.lite.Interpreter(model_path=tflite_model)
    else:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_type = interpreter.get_input_details()[0]['dtype']
    print('quantization input type: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('quantization output type: ', output_type)


    num_corrected = 0

    for i, test_image_instance in enumerate(test_ds.take(num_test_samples)):
        test_image = test_image_instance[0]
        test_label = test_image_instance[1]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
          input_scale, input_zero_point = input_details["quantization"]
          test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predicted = output.argmax()
        #print(str(i+1) + ") GT label is " + str(test_label.numpy()) + " - predicted: " + str(predicted), end='')
        if(predicted == test_label.numpy()):
            num_corrected += 1
            #print(" -> OK")
        #else:
        #    print(" -> KO")
        print("\r %d/%d Accuracy= %.2f" %(i, num_test_samples, (num_corrected/num_test_samples)*100), end='')
    print(" ")

    return num_corrected

def evaluate_model(tflite_model, test_ds, num_test_samples, is_path = False):

  num_corrected = RunTFLiteModel(tflite_model, test_ds, num_test_samples, is_path)

  accuracy = (num_corrected * 100) / num_test_samples

  print('model accuracy is %.4f%% (Number of test samples=%d)' % (accuracy, num_test_samples))

  return accuracy

def MeasureTfliteInference(tflite_model_file_name, test_ds, is_path = False):
      
    if(is_path == True):
        interpreter = tf.lite.Interpreter(model_path=tflite_model_file_name)
    else:
        interpreter = tf.lite.Interpreter(model_content=tflite_model_file_name)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    best_nanos = 2 ** 32

    for i, test_image_instance in enumerate(test_ds.take(1)):
        test_image = test_image_instance[0]
        test_label = test_image_instance[1]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
          input_scale, input_zero_point = input_details["quantization"]
          test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)

        #best_nanos = uti.MeasureTimeBest(100, lambda : interpreter.invoke())
        best_nanos = 1000000000000000000
        for i in range(100):
            t_start = time.perf_counter_ns()
            interpreter.invoke()
            end_time = (time.perf_counter_ns() - t_start)
            if end_time < best_nanos:
                best_nanos = end_time
        
        output = interpreter.get_tensor(output_details["index"])[0]

    return best_nanos

@tf.function
def test_step(model, images):
  predictions = model(images)

def MeasureTFModel(model, val_ds, num_reps):
      
    elapsed = 0
    for i, test_image_instance in enumerate(val_ds.take(1)):
          
      t_start = time.perf_counter_ns()
      for i in range(num_reps):
          test_step(model, test_image_instance)
          #model.predict(test_image_instance)

      elapsed = (time.perf_counter_ns() - t_start)/num_reps
    
    return elapsed

if __name__ == "__main__":
      
    if(tf.test.is_gpu_available() == True):
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs available: ", len(gpu_devices))
    else:
        print("GPU is not available!")

    directory_name = "/home/lvorabbi/Proj/Phd/state_art/"

    CORe50_model_path = "/home/lvorabbi/Proj/Phd/state_art/core50_models/MobileNetV2_CORe50_checkpoint_fine_2.h5"


    pruned_name = "/home/lvorabbi/Proj/Phd/state_art/core50_models/MobileNetV2_CORe50_pruned_accur_88_7_prune_50.h5"
    pruned_name_65 = "/home/lvorabbi/Proj/Phd/state_art/core50_models/MobileNetV2_CORe50_pruned_accur_84_8_prune_65.h5"
    pruned_name_70 = "/home/lvorabbi/Proj/Phd/state_art/core50_models/MobileNetV2_CORe50_pruned_accur_83_prune_70.h5"
    pruned_weight_cluster_name_50 = directory_name + "MobileNetV2_CORe50_pruned_50_and_weight_cluster.h5"
    pruned_weight_cluster_name = directory_name + "MobileNetV2_CORe50_pruned_65_and_weight_cluster.h5"
    pruned_weight_cluster_name_70 = directory_name + "MobileNetV2_CORe50_pruned_70_and_weight_cluster.h5"
    quantization_u8_name = directory_name + "MobileNetV2_CORe50_quantized_u8.tflite"
    quantization_f16_name = directory_name + "MobileNetV2_CORe50_quantized_f16.tflite"
    
    pruned_50_cluster_quantization_u8_name = directory_name + "MobileNetV2_CORe50_pruned_50_cluster_quantized_u8.tflite"
    pruned_50_clusterquantization_f16_name = directory_name + "MobileNetV2_CORe50_pruned_50_cluster_quantized_f16.tflite"
    pruned_65_clusterquantization_u8_name_65 = directory_name + "MobileNetV2_CORe50_pruned_65_cluster_quantized_u8_65.tflite"
    pruned_65_clusterquantization_f16_name_65 = directory_name + "MobileNetV2_CORe50_pruned_65_cluster_quantized_f16_65.tflite"
    pruned_70_clusterquantization_u8_name_70 = directory_name + "MobileNetV2_CORe50_pruned_70_cluster_quantized_u8_70.tflite"
    pruned_70_clusterquantization_f16_name_70 = directory_name + "MobileNetV2_CORe50_pruned_70_cluster_quantized_f16_70.tflite"
    
    pruned_weight_cluster_quantization_u8_name = directory_name + "MobileNetV2_CORe50_pruned_weight_clustering_quantized_u8.tflite"
    pruned_weight_cluster_quantization_f16_name = directory_name + "MobileNetV2_CORe50_pruned_weight_clustering_quantized_f16.tflite"

    '''
    model_loaded = keras.models.load_model(CORe50_model_path)
    model_pruned = keras.models.load_model(pruned_name)
    model_pruned_65 = keras.models.load_model(pruned_name_65)
    model_pruned_70 = keras.models.load_model(pruned_name_70)

    model_loaded.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer=keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    model_pruned.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer=keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    model_pruned_65.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer=keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model_pruned_70.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer=keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    '''

    '''
    for layer in model_loaded.layers:
            curr_weights = layer.get_weights()
            if(len(curr_weights) > 0):
                total = 0
                tot_non_zero = 0
                temp = np.ravel(curr_weights)
                for item in temp:
                    if abs(item) <= 1e-3:
                        tot_non_zero += 1
                total += temp.size
                print("Non zero weights: %.2f %%" %((tot_non_zero*100)/total))
    '''

    '''
    for idx, layer in enumerate(model_loaded.layers):
        if idx == 0
        print(layer)
    '''

    '''
    print("zipped size of original model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_f32.tflite")))
    print("zipped size of pruned 70 model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70.h5")))
    print("zipped size of pruned 70 + weight cluster model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70_clustering.h5")))
    print("zipped size of quantized u8 model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_quantized_u8.tflite")))
    print("zipped size of quantized f16 model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_quantized_f16.tflite")))
    print("zipped size of pruned 70 quantized u8 model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70_quantized_u8.tflite")))
    print("zipped size of pruned 70 quantized f16 model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70_quantized_f16.tflite")))
    print("zipped size of pruned 70 + clustering + quantized u8 model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70_clustering_quantized_u8.tflite")))
    print("zipped size of pruned 70 + clustering + quantized f16 model: " + str(uti.get_gzipped_model_size("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70_clustering_quantized_f16.tflite")))
    '''
    
    #model_loaded = keras.models.load_model("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70_clustering.h5")
    model_loaded = keras.models.load_model("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/EfficientB0_CORe50_checkpoint_fine_2.h5")

    #inital_non_zero = uti.GetModelNonZeroWeightsPercentage(model_loaded)

    train_ds, val_ds = core50.CORe50_dataset("/home/lvorabbi/Desktop/Disk2/Core50/core50_imgs.npz", 
                                             "/home/lvorabbi/Desktop/Disk2/Core50/paths.pkl")
    val_for_quantization = val_ds.unbatch()

    #core50.prune_model(model_loaded, train_ds, val_ds, "/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70.h5", 1.0-(inital_non_zero/100.0), 0.7)

    #u8_model = core50.quantize_model(model_loaded, train_ds, val_for_quantization, "/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70_clustering_quantized_u8.tflite", "u8")
    #f16_model = core50.quantize_model(model_loaded, train_ds, val_for_quantization, "/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_pruned_70_clustering_quantized_f16.tflite", "f16")

    print("Evauation quantized u8: " + str(evaluate_model("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_quantized_u8.tflite", val_for_quantization.shuffle(10000), 10000, is_path=True)))
    print("Evauation quantized f16: " + str(evaluate_model("/home/lvorabbi/Proj/Phd/state_art/core50_models/EfficientNet/Model_quantized_f16.tflite", val_for_quantization.shuffle(10000), 10000, is_path=True)))

    #quantized_model_pruned_u8 = core50.quantize_model(model_loaded, train_ds, val_for_quantization, "/home/lvorabbi/Proj/Phd/state_art/core50_models/MobileNetV2_CORe50_f32.tflite", "f16", 1000, optimize=False, force_all_quantize=False, dataset_provided=False)

    print("Evauation pruned + cluster 50 quantized f16: " + str(evaluate_model("/home/lvorabbi/Proj/Phd/state_art/core50_models/quantization/MobileNetV2_CORe50_pruned_50_cluster_quantized_u8.tflite", val_for_quantization.shuffle(10000), 1000, is_path=True)))
    print("Evauation pruned + cluster 65 quantized f16: " + str(evaluate_model("/home/lvorabbi/Proj/Phd/state_art/core50_models/quantization/MobileNetV2_CORe50_pruned_65_cluster_quantized_u8_65.tflite", val_for_quantization.shuffle(10000), 1000, is_path=True)))
    print("Evauation pruned + cluster 70 quantized f16: " + str(evaluate_model("/home/lvorabbi/Proj/Phd/state_art/core50_models/quantization/MobileNetV2_CORe50_pruned_70_cluster_quantized_u8_70.tflite", val_for_quantization.shuffle(10000), 1000, is_path=True)))

    '''
    quantized_model_pruned_u8 = core50.quantize_model(model_pruned, train_ds, val_for_quantization, pruned_quantization_u8_name, "u8", 1000)
    quantized_model_pruned_f16 = core50.quantize_model(model_pruned, train_ds, val_for_quantization, pruned_quantization_f16_name, "f16", 1000)
    quantized_model_pruned_u8_65 = core50.quantize_model(model_pruned, train_ds, val_for_quantization, pruned_quantization_u8_name_65, "u8", 1000)
    quantized_model_pruned_f16_65 = core50.quantize_model(model_pruned, train_ds, val_for_quantization, pruned_quantization_f16_name_65, "f16", 1000)
    quantized_model_pruned_u8_70 = core50.quantize_model(model_pruned, train_ds, val_for_quantization, pruned_quantization_u8_name_70, "u8", 1000)
    quantized_model_pruned_f16_70 = core50.quantize_model(model_pruned, train_ds, val_for_quantization, pruned_quantization_f16_name_70, "f16", 1000)
    '''

    #pruned_weight_cluster = keras.models.load_model(pruned_weight_cluster_name)

    #model_pruned = core50.prune_model(model_loaded, train_ds, val_ds, pruned_name)
    #model_pruned = core50.prune_model(model_pruned, train_ds, val_ds, pruned_name)
    
    '''
    pruned_weight_cluster_50 = core50.cluster_weights(model_pruned, train_ds, val_ds, 16, 
                                                   keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True), 
                                                   pruned_weight_cluster_name_50)
    pruned_weight_cluster_65 = core50.cluster_weights(model_pruned_65, train_ds, val_ds, 16, 
                                                   keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True), 
                                                   pruned_weight_cluster_name)
    pruned_weight_cluster_70 = core50.cluster_weights(model_pruned_70, train_ds, val_ds, 16, 
                                                   keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True), 
                                                   pruned_weight_cluster_name_70)
    '''
    
    print("zipped size of pruned 50 + clutering: " + str(uti.get_gzipped_model_size(pruned_weight_cluster_name_50)))
    print("zipped size of pruned 65 + clutering: " + str(uti.get_gzipped_model_size(pruned_weight_cluster_name)))
    print("zipped size of pruned 70 + clustering model: " + str(uti.get_gzipped_model_size(pruned_weight_cluster_name_70)))

    quantized_model_pruned_50_cluster_u8 = core50.quantize_model(pruned_weight_cluster_50, train_ds, val_for_quantization, pruned_50_cluster_quantization_u8_name, "u8", 1000)
    quantized_model_pruned_50_cluster_f16 = core50.quantize_model(pruned_weight_cluster_50, train_ds, val_for_quantization, pruned_50_clusterquantization_f16_name, "f16", 1000)

    quantized_model_pruned_65_cluster_u8 = core50.quantize_model(pruned_weight_cluster_65, train_ds, val_for_quantization, pruned_65_clusterquantization_u8_name_65, "u8", 1000)
    quantized_model_pruned_65_cluster_f16 = core50.quantize_model(pruned_weight_cluster_65, train_ds, val_for_quantization, pruned_65_clusterquantization_f16_name_65, "f16", 1000)

    quantized_model_pruned_70_cluster_u8 = core50.quantize_model(pruned_weight_cluster_70, train_ds, val_for_quantization, pruned_70_clusterquantization_u8_name_70, "u8", 1000)
    quantized_model_pruned_70_cluster_f16 = core50.quantize_model(pruned_weight_cluster_70, train_ds, val_for_quantization, pruned_70_clusterquantization_f16_name_70, "f16", 1000)

    print("zipped size of pruned 50 + clutering + quantization u8: " + str(uti.get_gzipped_model_size(pruned_50_cluster_quantization_u8_name)))
    print("zipped size of pruned 50 + clutering + quantization f16: " + str(uti.get_gzipped_model_size(pruned_50_clusterquantization_f16_name)))
    print("zipped size of pruned 65 + clutering + quantization u8: " + str(uti.get_gzipped_model_size(pruned_65_clusterquantization_u8_name_65)))
    print("zipped size of pruned 65 + clutering + quantization f16: " + str(uti.get_gzipped_model_size(pruned_65_clusterquantization_f16_name_65)))
    print("zipped size of pruned 70 + clutering + quantization u8: " + str(uti.get_gzipped_model_size(pruned_70_clusterquantization_u8_name_70)))
    print("zipped size of pruned 70 + clutering + quantization f16: " + str(uti.get_gzipped_model_size(pruned_70_clusterquantization_f16_name_70)))

    '''
    pruned_weight_cluster.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer=keras.optimizers.SGD(learning_rate=0.0025, momentum=0.9, nesterov=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    '''              
    
    #print("Evaluation original model: ", end='')
    #print(model_loaded.evaluate(val_ds))
    #print("Evaluation model pruned: ", end='')
    #print(model_pruned.evaluate(val_ds))
    #print("Evaluation model pruned + weight clustering: ", end='')
    #print(pruned_weight_cluster.evaluate(val_ds))

    #quantized_model_u8 = core50.quantize_model(model_loaded, train_ds, val_for_quantization, quantization_u8_name, "u8", 1000)
    #quantized_model_f16 = core50.quantize_model(model_loaded, train_ds, val_for_quantization, quantization_f16_name, "f16", 1000)

    quantized_model_pruned_u8 = core50.quantize_model(model_pruned, train_ds, val_for_quantization, pruned_quantization_u8_name, "u8", 1000)
    quantized_model_pruned_f16 = core50.quantize_model(model_pruned, train_ds, val_for_quantization, pruned_quantization_f16_name, "f16", 1000)

    #quantized_model_pruned_weight_cluster_u8 = core50.quantize_model(pruned_weight_cluster, train_ds, val_for_quantization, pruned_weight_cluster_quantization_u8_name, "u8", 1000)
    #quantized_model_pruned_weight_cluster_f16 = core50.quantize_model(pruned_weight_cluster, train_ds, val_for_quantization, pruned_weight_cluster_quantization_f16_name, "f16", 1000)
    
    #print("Evauation quantized u8: " + str(evaluate_model(quantized_model_u8, val_for_quantization, 100)))
    #print("Evauation quantized f16: " + str(evaluate_model(quantized_model_f16, val_for_quantization, 100)))
    print("Evauation pruned quantized u8: " + str(evaluate_model(quantized_model_pruned_u8, val_for_quantization, 100)))
    print("Evauation pruned quantized f16: " + str(evaluate_model(quantized_model_pruned_f16, val_for_quantization, 100)))
    #print("Evauation pruned weight cluster quantized u8: " + str(evaluate_model(quantized_model_pruned_weight_cluster_u8, val_for_quantization, 10)))
    #print("Evauation pruned weight cluster quantized f16: " + str(evaluate_model(quantized_model_pruned_weight_cluster_f16, val_for_quantization, 10)))

    print("zipped size of original model: " + str(uti.get_gzipped_model_size(CORe50_model_path)))
    print("zipped size of pruned model: " + str(uti.get_gzipped_model_size(pruned_name)))
    print("zipped size of pruned + weight cluster model: " + str(uti.get_gzipped_model_size(pruned_weight_cluster_name)))
    print("zipped size of quantized u8 model: " + str(uti.get_gzipped_model_size(quantization_u8_name)))
    print("zipped size of quantized f16 model: " + str(uti.get_gzipped_model_size(quantization_f16_name)))
    print("zipped size of pruned quantized u8 model: " + str(uti.get_gzipped_model_size(pruned_quantization_u8_name)))
    print("zipped size of pruned quantized f16 model: " + str(uti.get_gzipped_model_size(pruned_quantization_f16_name)))
    #print("zipped size of pruned + weight cluster quantized u8 model: " + str(uti.get_gzipped_model_size(pruned_weight_cluster_quantization_u8_name)))
    #print("zipped size of pruned + weight cluster quantized f16 model: " + str(uti.get_gzipped_model_size(pruned_weight_cluster_quantization_f16_name)))

    
    '''
    print("Inference best original model f32 (batch size 64): " + str((MeasureTFModel(model_loaded, val_ds, 100)/64)/1e6) + " ms")
    print("Inference best model pruned f32 (batch size 64): " + str((MeasureTFModel(model_pruned, val_ds, 100)/64)/1e6) + " ms")
    print("Inference best model pruned + weight cluster f32 (batch size 64): " + str((MeasureTFModel(pruned_weight_cluster, val_ds, 100)/64)/1e6) + " ms")

    print("Inference best original model f32 (batch size 1): " + str(MeasureTFModel(model_loaded, val_for_quantization.batch(1), 100)/1e6) + " ms")
    print("Inference best model pruned f32 (batch size 1): " + str(MeasureTFModel(model_pruned, val_for_quantization.batch(1), 100)/1e6) + " ms")
    print("Inference best model pruned + weight cluster f32 (batch size 1): " + str(MeasureTFModel(pruned_weight_cluster, val_for_quantization.batch(1), 100)/1e6) + " ms")
  
    print("Inference best quantization u8: " + str(MeasureTfliteInference(quantization_u8_name, val_for_quantization)/1e6) + " ms")
    print("Inference best quantization f16: " + str(MeasureTfliteInference(quantization_f16_name, val_for_quantization)/1e6) + " ms")
    print("Inference best quantization pruned u8: " + str(MeasureTfliteInference(pruned_quantization_u8_name, val_for_quantization)/1e6) + " ms")
    print("Inference best quantization pruned f16: " + str(MeasureTfliteInference(pruned_quantization_f16_name, val_for_quantization)/1e6) + " ms")
    print("Inference best quantization pruned weight clustering u8: " + str(MeasureTfliteInference(pruned_weight_cluster_quantization_u8_name, val_for_quantization)/1e6) + " ms")
    print("Inference best quantization pruned weight clustering f16: " + str(MeasureTfliteInference(pruned_weight_cluster_quantization_f16_name, val_for_quantization)/1e6) + " ms")
    '''