import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import random 
import sys
import json
import random
import time
from sklearn.utils import shuffle
from collections import defaultdict
import Backpropagation_impl as back_prop

class ReservoirSamplingBuffer:
    
    def __init__(self, max_size, input_shape):
        
        self.max_size = max_size
        self.buffer = tf.zeros((0,)+input_shape)
        self.buffer_weights = tf.zeros((0))

    def update(self, new_dataset):

        len_dt = 0
        if isinstance(new_dataset, tf.data.Dataset) == True:
            for idx, item in new_dataset.enumerate():
                len_dt += 1
        else:
            len_dt = tf.shape(new_dataset)[0]

        new_weights = tf.random.uniform((len_dt,))
        cat_weights = tf.concat([new_weights, self.buffer_weights], axis=0)
        cat_dataset = tf.concat([new_dataset, self.buffer], axis=0)

        sorted_idxs = tf.argsort(cat_weights, direction='DESCENDING')
        sorted_weights = tf.gather(cat_weights, sorted_idxs)

        buffer_idxs = sorted_idxs[: self.max_size]

        self.buffer = tf.gather(cat_dataset, buffer_idxs)
        self.buffer_weights = sorted_weights[: self.max_size]
        
    def num_elements(self):
        return tf.shape(self.buffer)[0]
    
    def get_buffer(self):
        return self.buffer
        
class ClassBalancedBuffer:
    
    def __init__(self, max_size, input_shape, total_num_classes,
                 enable_debug_print = False):
        
        self.max_size = max_size
        self.input_shape = input_shape
        self.total_num_classes = total_num_classes
        self.buffer_size = self.max_size // self.total_num_classes
        self.enable_debug_print = enable_debug_print
        
        # buffers of elements saved per class
        self.buffers = [tf.zeros((0,)+self.input_shape)]*self.total_num_classes
        
        self.output_buffer = None
        self.output_gt = None
        self.buffers_length = [None]*self.total_num_classes
        
        self.out_buffer = tf.zeros((0,)+self.input_shape)
        self.output_gt = tf.zeros((0,))
        self.total_dataset = None
        self.classes = None
        self.current_index = 0
        
    def update(self, new_dataset):
        
        tot_label = tf.zeros(shape=(0))
        classes = None
        count_classes = None
        idx_classes = None
        if isinstance(new_dataset, tf.data.Dataset) == True:
            for (img, label) in new_dataset.as_numpy_iterator():
                tot_label = tf.concat([tot_label, label], axis=0)
            classes, idx_classes, count_classes = tf.unique_with_counts(tot_label)
        else:
            tf.debugging.assert_equal(0, 1, message='Accept only tf.data.Dataset')
            
        current_classes = tf.cast(tf.gather(classes, tf.argsort(classes)), tf.int32)        
        total_new_samples = tf.math.reduce_sum(count_classes)
        
        tot_items, tot_classes, classes_already_in_memory = self.get_total_num_items()
        
        if(self.enable_debug_print == True):
            print('Total items stored in buffer ', tot_items)
            print('Num new items to add ', total_new_samples)
        
        if(self.max_size > tot_items + total_new_samples):
        
            # Add all new elements
            for class_id in current_classes:
            
                tf.debugging.assert_less(tf.cast(class_id, tf.int32), self.total_num_classes, message='Invalid class ID')

                # Select the samples of the specific class
                curr_ds = new_dataset.unbatch().filter(lambda x,y: tf.reduce_all(tf.cast(y, tf.int32) == class_id)).batch(1)

                # Update buffer with new elements
                for (img, label) in curr_ds.as_numpy_iterator():
                    self.buffers[class_id] = tf.concat([img, self.buffers[class_id]], axis=0)

                self.buffers_length[class_id] = tf.shape(self.buffers[class_id])[0]
            
        else:
            
            # Replace some elements
            num_elements_to_remove = tot_items + total_new_samples - self.max_size
            
            if(self.enable_debug_print==True):
                print('Elements to remove: ', num_elements_to_remove)
            
            # Compute the new number of classes
            total_classes_seen, _, _ = tf.unique_with_counts(tf.concat([current_classes, classes_already_in_memory], axis=0))
            
            # Compute new number of elements per class
            new_num_elements_per_class = self.max_size // tf.shape(total_classes_seen)[0]
            
            if(self.enable_debug_print==True):
                print('Num elements per class ', new_num_elements_per_class)
        
            for class_id in current_classes:

                tf.debugging.assert_less(tf.cast(class_id, tf.int32), self.total_num_classes, message='Invalid class ID')

                # Select the samples of the specific class
                curr_ds = new_dataset.unbatch().filter(lambda x,y: tf.reduce_all(tf.cast(y, tf.int32) == class_id)).batch(1)

                # Extract elements of current class                
                cur_buffer = tf.zeros((0,)+self.input_shape)
                for (img, label) in curr_ds.as_numpy_iterator():
                    cur_buffer = tf.concat([img, cur_buffer], axis=0)
                    
                # concatenate new elements with those currently stored
                concat_buffer = tf.concat([self.buffers[class_id], cur_buffer], axis=0)

                tmp_buffer = ReservoirSamplingBuffer(new_num_elements_per_class, self.input_shape)
                tmp_buffer.update(concat_buffer)
                
                # Update buffer of current class
                self.buffers[class_id] = tmp_buffer.get_buffer()
                self.buffers_length[class_id] = tf.shape(self.buffers[class_id])[0]
        
        # Compose output buffer
        self.out_buffer = tf.zeros((0,)+self.input_shape)
        self.output_gt = tf.zeros((0,))
        for idx, buff in enumerate(self.buffers):
            
            if(tf.shape(buff)[0] > 0):
                self.out_buffer = tf.concat([self.out_buffer, self.buffers[idx]], axis=0)
                # GT value
                curr_gt = tf.constant(idx, shape=(tf.shape(self.buffers[idx])[0],), dtype=tf.float32)
                self.output_gt = tf.concat([self.output_gt, curr_gt], axis=0)
                
        self._shuffle_output()
        self.total_dataset = tf.data.Dataset.from_tensors((self.out_buffer, self.output_gt))
        self.current_index = 0
        
    def _shuffle_output(self):
        
        indices = tf.range(start=0, limit=tf.shape(self.out_buffer)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        self.out_buffer = tf.gather(self.out_buffer, shuffled_indices)
        self.output_gt = tf.gather(self.output_gt, shuffled_indices)
        
    def get_total_num_items(self):
        count=0
        num_classes_seen=0
        classes = []
        for idx, val in enumerate(self.buffers):
            nelems = tf.shape(val)[0]
            count += nelems
            if nelems > 0:
                num_classes_seen+=1
                classes.append(idx)
        return count, num_classes_seen, classes
    
    def get_buffer(self, target_size = 128):
        
        num_batches = int(tf.shape(self.out_buffer)[0] / target_size)
        
        for idx in tf.range(self.current_index,num_batches):
            out = (self.out_buffer[idx*target_size:idx*target_size+target_size, :, :, :], self.output_gt[idx*target_size:idx*target_size+target_size])
            self.current_index += 1
            if (self.current_index == num_batches):
                self._shuffle_output()
                self.current_index = 0
            return out
        
    def get_total_dataset(self):
        
        if(tf.shape(self.out_buffer)[0] == 0):
            output_buffer = tf.zeros((0,)+self.input_shape)
            output_gt = tf.zeros((0,))
            return tf.data.Dataset.from_tensors((output_buffer, output_gt))
        else:
            return self.total_dataset
        
class CWR:
    
    def __init__(self, replay_buffer=None, **kwargs):

        self.samples_per_class_in_experience = defaultdict(int)
        self.past_sample_per_classes = defaultdict(int)

        self.classes_in_experience = None
        self.saved_cwr_weights = {}
        self.classes_in_experiences_so_far = {}
        self.replay_buffer = replay_buffer
        
    def _example_per_class(self, train_ds):

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
    
    def _reset_weights(self, model, is_splitted=False):

        if is_splitted==False:
            cwr_layer = model.get_layer('cwr')
        else:
            cwr_layer = model['trainable_model'][-1].dense
            #cwr_layer = model['trainable_model'].get_layer('cwr')
            
        cwr_layer.weights[0].assign(tf.zeros_like(cwr_layer.weights[0]))
        n_input_cwr = cwr_layer.weights[0].numpy().shape[0]

        for c, w in self.saved_cwr_weights.items():

            c = int(c)
            
            if int(c) in self.classes_in_experience:

                indices = tf.concat([tf.reshape(tf.range(0, n_input_cwr), (n_input_cwr, 1)), tf.constant(c, shape=(n_input_cwr,1))], axis=1)
                updates = tf.transpose(self.saved_cwr_weights[c])
                new_weights = tf.tensor_scatter_nd_update(cwr_layer.weights[0], indices, updates)
                cwr_layer.weights[0].assign(new_weights)
                
    def _set_consolidate_weights(self, model, is_splitted=False):

        if is_splitted==False:
            cwr_layer = model.get_layer('cwr')
        else:
            cwr_layer = model['trainable_model'][-1].dense
            
        n_input_cwr = cwr_layer.weights[0].numpy().shape[0]

        layer_w = cwr_layer.weights[0].numpy().T
        for c, w in self.saved_cwr_weights.items():
            layer_w[c,:] = self.saved_cwr_weights[c]

        cwr_layer.weights[0].assign(layer_w.T)
        
    def _consolidate_weights(self, model, is_splitted=False):

        if is_splitted==False:
            cwr_layer = model.get_layer('cwr')
        else:
            cwr_layer = model['trainable_model'][-1].dense
            
        merged_dict = {**self.classes_in_experiences_so_far}
        for key, val in self.classes_in_experience.items():
            if key in self.classes_in_experiences_so_far:
                merged_dict[key] += val
            else:
                merged_dict[key] = val

        #globavg = tf.math.reduce_mean(tf.transpose(cwr_layer.weights[0]).numpy()[list(merged_dict.keys())])
        globavg = tf.math.reduce_mean(tf.transpose(cwr_layer.weights[0]).numpy()[list(self.classes_in_experience.keys())])

        for c in self.classes_in_experience.keys():

            c = int(c)

            curr_w = cwr_layer.weights[0].numpy().T[c]

            new_w = curr_w - globavg

            if c in self.saved_cwr_weights.keys():

                w_pastj = tf.cast(tf.math.sqrt(tf.math.divide_no_nan(tf.cast(self.past_sample_per_classes[c], tf.float32), tf.cast(self.samples_per_class_in_experience[c], tf.float32) )), tf.float32)
                
                # consolidation
                self.saved_cwr_weights[c] = tf.math.divide_no_nan(self.saved_cwr_weights[c] * w_pastj + new_w, (w_pastj+1))
                self.past_sample_per_classes[c] += self.samples_per_class_in_experience[c]

            else:
                self.saved_cwr_weights[c] = new_w
                self.past_sample_per_classes[c] = self.samples_per_class_in_experience[c]
                
    def before_training_experience(self, idx_exp, model, curr_exp_train_ds):

        print('Starting experience {}'.format(idx_exp+1))
        
        if idx_exp == 0:
            for layer in model.layers:
                layer.trainable = True
        else:
            if self.replay_buffer == None:
                for layer in model.layers:
                    if layer.name != 'cwr':
                        layer.trainable = False
            else:
                for layer in model['frozen_model'].layers:
                    layer.trainable = False
        
        if self.replay_buffer == None:
            self.classes_in_experience, self.samples_per_class_in_experience = self._example_per_class(curr_exp_train_ds)
        else:
            if idx_exp > 0:
                rm_ds = self.replay_buffer.get_total_dataset()
                tmp_ds = curr_exp_train_ds.concatenate(rm_ds)
                self.classes_in_experience, self.samples_per_class_in_experience = self._example_per_class(tmp_ds)
            else:
                self.classes_in_experience, self.samples_per_class_in_experience = self._example_per_class(curr_exp_train_ds)

        # reset weights
        if self.replay_buffer == None:
            self._reset_weights(model, False)
        else:
            if idx_exp==0:
                self._reset_weights(model, False)
            else:
                self._reset_weights(model, True)
            
    
    def after_training_experience(self, idx_exp, model, curr_exp_train_ds):

        print('Finished experience {}'.format(idx_exp+1))
        
        if self.replay_buffer == None:
            self._consolidate_weights(model, False)
            self._set_consolidate_weights(model, False)
        else:
            self._consolidate_weights(model, True)
            self._set_consolidate_weights(model, True)
        
        # Update the list of classes seen so far
        for key, val in self.classes_in_experience.items():
            if key in self.classes_in_experiences_so_far:
                self.classes_in_experiences_so_far[key] += val
            else:
                self.classes_in_experiences_so_far[key] = val
            
        # update replay buffer at the end of the experience
        output_buffer = tf.zeros((0,)+self.replay_buffer.input_shape)
        output_gt = tf.zeros((0,))
        for idx, (images, labels) in enumerate(curr_exp_train_ds):
        
            predictions_1 = model['frozen_model'](images)
            
            output_buffer = tf.concat([output_buffer, predictions_1], axis=0)
            output_gt = tf.concat([output_gt, labels], axis=0)
        
        update_ds = tf.data.Dataset.from_tensors((output_buffer, output_gt))        
        self.replay_buffer.update(update_ds)
        
        
from tensorflow.python.ops.gen_math_ops import mod

def get_train_step_function():
    
    def train_step_func_cwr_1_epoch(model, images, labels, loss_object, optimizer, train_loss, train_loss_additional, train_accuracy, statistics, training_all, weight_update_quant_bits, all_other_quant_bits, gradient_error_metric):

        with tf.GradientTape() as tape:
            predictions = model(images, training=training_all)
            loss = loss_object(labels, predictions)
            main_loss = loss

        gradients_auto = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients((grad, var) 
                                   for (grad, var) in zip(gradients_auto, model.trainable_variables) 
                                   if grad is not None)

        # Apply constraints
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))

        train_loss(main_loss)
        for i in range(len(train_accuracy)):
            train_accuracy[i].update_state(labels, predictions)
    return train_step_func_cwr_1_epoch

def get_train_step_function_exp_bigger_1(quantized = True):

    #@tf.function
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
    def train_step_func_cwr(model, images, labels, loss_object, optimizer, train_loss, train_loss_additional, 
                            train_accuracy, statistics, training_all, weight_update_quant_bits, all_other_quant_bits, 
                            gradient_error_metric, replay_buffer, idx_exp):

        predictions_1 = model['frozen_model'](images, training=True)

        replay_buffer_data, replay_buffer_gt = replay_buffer.get_buffer(target_size=128)
        if replay_buffer_gt != None:
            predictions_1 = tf.concat([predictions_1, replay_buffer_data], axis=0)
            labels = tf.concat([labels, replay_buffer_gt], axis=0)

        predictions = back_prop.Execute_custom_layers_forward(predictions_1, model['trainable_model'], training=True)

        main_loss = loss_object(labels, predictions)

        batch_size = tf.shape(predictions)[0]
        back_prop.Execute_custom_layers_backward(predictions, tf.reshape(labels, (batch_size, 1)), model['trainable_model'], -1)

        # Update weights
        back_prop.Update_weights(model['trainable_model'], optimizer)

        train_loss(main_loss)
        for i in range(len(train_accuracy)):
            train_accuracy[i].update_state(labels, predictions)

    return train_step_func_cwr

def get_val_step_function():
    #@tf.function
    def val_step_func(model, images, labels, loss_object, test_loss, test_loss_additional, test_accuracy):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        t_loss_additional = sum(model.losses)

        test_loss(t_loss)
        test_loss_additional(t_loss_additional)
        for i in range(len(test_accuracy)):
            test_accuracy[i].update_state(labels, predictions)
    return val_step_func

def get_val_cwr_step_function():
    
    #@tf.function
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
    def val_step_func_quantized(model, images, labels, loss_object, test_loss, test_loss_additional, test_accuracy, statistics):
        
        num_quantization_bits = 8
        
        predictions_1 = model['frozen_model'](images, training=False)
        
        # Execute on trainable model with quantized backprop
        predictions_2 = back_prop.Execute_custom_layers_forward(predictions_1, model['trainable_model'], training=False)
        #predictions_2 = model['trainable_model'](predictions_1, training=False)
        
        # Compute loss with concat data
        t_loss = loss_object(labels, predictions_2)
        
        test_loss(t_loss)
        for i in range(len(test_accuracy)):
            test_accuracy[i].update_state(labels, predictions_2)
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
                 weight_update_quant_bits=16,
                 all_other_quant_bits=8,
                 accuracy_to_monitor=None,
                 gradient_error_metric=None,
                 before_training_experience=None,
                 after_training_experience=None,
                 cwr_obj=None,
                 replay_buffer=None,
                 split_model_online_training_callback=None):
    
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
    
    model_splitted = None
    
    for idx_exp in range(num_experiences):
        
        # Reset learning rate at each experience
        tf.keras.backend.set_value(optimizer.lr, initial_lr)
        
        num_train_samples = 0
        for idx, item in enumerate(train_ds[idx_exp]):
            num_train_samples += 1
        num_train_steps_per_epoch = num_train_samples
        
        if (cwr_obj != None):
            cwr_obj.before_training_experience(idx_exp, model if idx_exp==0 else model_splitted, train_ds[idx_exp])
        elif before_training_experience is not None:
            before_training_experience(idx_exp, model if idx_exp==0 else model_splitted, train_ds[idx_exp])

        if idx_exp == 0:
            num_epochs = epochs
        else:
            #optimizer = tf.keras.optimizers.SGD(0.0025)
            #optimizer = tf.keras.optimizers.SGD(0.005)
            optimizer = tf.keras.optimizers.SGD(0.009)
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
                    train_func(model, images, labels, loss_object, optimizer, train_loss, train_loss_additional, train_accuracy, training_stats_activation, train_all, weight_update_quant_bits, all_other_quant_bits, gradient_error_metric)
                else:
                    train_func_cwr(model_splitted, images, labels, loss_object, optimizer, train_loss, train_loss_additional, train_accuracy, training_stats_activation, train_all, weight_update_quant_bits, all_other_quant_bits, gradient_error_metric, cwr_obj.replay_buffer, idx_exp)

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
                    
                #if after_training_experience is not None:
                    #after_training_experience(idx_exp, model)

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
                        val_func_cwr(model_splitted, test_images, test_labels, loss_object, val_loss, val_loss_additional, val_accuracy, training_stats_activation)
                    
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
                
        experiences_training_loss.append(epochs_training_loss)
        experiences_validation_loss.append(epochs_validation_loss)
        experiences_training_accuracy.append(epochs_training_accuracy)
        experiences_validation_accuracy.append(epochs_validation_accuracy)
        
        experiences_gradient_error_metric.append(epochs_gradient_error_metric)
        
        # load best loss result
        #model.load_weights(embeddings_ckpt)
        
        if (idx_exp == 0):
            if split_model_online_training_callback!= None:
                if(is_cwr_quantized == True):
                    model_splitted = split_model_online_training_callback(model, num_fw_bits=all_other_quant_bits, num_quant_bits_bw=weight_update_quant_bits)
                else:
                    model_splitted = split_model_online_training_callback(model, num_fw_bits=None, num_quant_bits_bw=None)
                
        if (cwr_obj != None):
            cwr_obj.after_training_experience(idx_exp, model_splitted, train_ds[idx_exp])
        elif after_training_experience is not None:
            after_training_experience(idx_exp, model_splitted, train_ds[idx_exp])
            
        print('Validation after experience:')
        
        for i in range(len(val_accuracy)):
            val_accuracy[i].reset_states()
        val_loss.reset_states()
        val_loss_additional.reset_states()
        
        for idx, (test_images, test_labels) in enumerate(val_ds):
            if (idx_exp == 0):
                val_func(model, test_images, test_labels, loss_object, val_loss, val_loss_additional, val_accuracy)
            else:
                val_func_cwr(model_splitted, test_images, test_labels, loss_object, val_loss, val_loss_additional, val_accuracy, training_stats_activation)
            #val_func(model, test_images, test_labels, loss_object, val_loss, val_loss_additional, val_accuracy)  
        print("--%d/%d - val loss= %.4f" %(idx+1, num_val_steps_per_epoch, val_loss.result()), end="", flush=True)

        for i in range(len(val_accuracy)):
            print(" - " + str(val_accuracy[i].name + "= %.4f" %(val_accuracy[i].result())), end="")
        print("", flush=True)
        
        if gradient_error_metric!= None:
            print("Gradient error quantization= %.4f"%(np.mean(epochs_gradient_error_metric)))
            
        per_experience_validation_accuracy.append(float(val_accuracy[0].result().numpy()))

        print(" ")

    return {'epochs_training_loss' : experiences_training_loss,
            'epochs_training_accuracy' : experiences_training_accuracy,
            'epochs_validation_loss' : experiences_validation_loss,
            'epochs_validation_accuracy' : experiences_validation_accuracy,
            'per_experience_validation_accuracy':per_experience_validation_accuracy,
            'experiences_gradient_error_metric': experiences_gradient_error_metric}