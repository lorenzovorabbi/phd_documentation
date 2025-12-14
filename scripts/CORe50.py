import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import partial
import glob
import time
import pickle as pkl
import random 

MBNetV2_sizeX = 224
MBNetV2_sizeY = 224

#CORe50_folder_images = 'F:\\Core50\\core50_128x128\\'
CORe50_folder_images = 'C:\\CORe50\\core50_128x128\\'
COR250_num_classes = 50
CORe50_num_sequences = 11
COR350_num_sequences_validation = 3
CORe50_label_names = ["o"+str(i) for i in range(1, COR250_num_classes+1)]

CORe50_batch_size = 64

root_logdir = os.path.join("E:\\Proj\PhD\\temp\\", "tf_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

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
  return tf.image.resize(img, [MBNetV2_sizeX, MBNetV2_sizeY])
  #return img

def normalize_img(image, label):
  final_image = keras.applications.mobilenet_v2.preprocess_input(image)
  return final_image, label

def MBNetV2_preprocess(image, label):
    image = tf.image.resize(image, [MBNetV2_sizeX, MBNetV2_sizeY])
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def process_path(file_path, normalize=True):
  label = get_label(file_path)
  img = load_img(file_path)
  if(normalize == True):
    img, label = normalize_img(img, label)
  return img, label

def configure_for_performance(ds : tf.data.Dataset,
                              size : int,
                              batch_size : int) -> tf.data.Dataset:
  #ds = ds.cache()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  #ds = ds.prefetch(buffer_size=3)
  return ds

def rotate(image : tf.Tensor, 
           label : tf.Tensor) -> (tf.Tensor, tf.Tensor):

    return tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), label

def SelectImagesForTrainValidation(sequence_id : int,
                                   training_perc : float,
                                   shuffle : bool = True) -> (tf.data.Dataset, tf.data.Dataset, int, int):
    num_elements = 0
    dataset = None

    for i in range(COR250_num_classes):
        if(i == 0):
            dir = CORe50_folder_images + "s" + str(sequence_id) + "\\o" + str(i+1) + "\\"
            dataset = tf.data.Dataset.list_files(dir + "*.png", shuffle=shuffle)
            num_elements += len(list(glob.glob(dir + "*.png")))
            #num_elements += tf.data.experimental.cardinality(dataset).numpy()
        else:
            dir = CORe50_folder_images + "s" + str(sequence_id) + "\\o" + str(i+1) + "\\"
            dataset = tf.data.Dataset.list_files(dir + "*.png", shuffle=shuffle).concatenate(dataset)
            num_elements += len(list(glob.glob(dir + "*.png")))
            #num_elements += tf.data.experimental.cardinality(dataset).numpy()

    val_size = int(num_elements * (1.0 - training_perc))
    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)

    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds, num_elements-val_size, val_size

def AugmentDataset(train_set : tf.data.Dataset,
                   num_elements : int) -> (tf.data.Dataset, int):

    train_set_to_rotate = train_set.repeat(2)
    train_set_to_rotate = train_set_to_rotate.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_set = train_set.concatenate(train_set_to_rotate)
    train_set = train_set.shuffle(num_elements, reshuffle_each_iteration=False)
    return train_set, num_elements*3

def VisualizeImages(dataset : tf.data.Dataset,
                    num_elements : int,
                    num_rows : int,
                    num_cols : int):

    random_data = dataset.shuffle(num_elements)
    plt.figure(figsize=(10, 10))
    for i in range(num_rows*num_cols):
      ax = plt.subplot(num_rows, num_cols, i + 1)
      image_batch, label_batch = next(iter(random_data))
      plt.imshow(image_batch.numpy().astype("float32"))
      plt.title(CORe50_label_names[label_batch-1])
      plt.axis("off")

def CreateCORe50Dataset(augment : bool = False,
                        enable_performance : bool = True) -> (tf.data.Dataset, tf.data.Dataset):

    for i in range(CORe50_num_sequences-COR350_num_sequences_validation):

        print("Loading sequence id: " + str(i+1) + "...", end='')

        if(i == 0):
            train_ds, val_ds, num_train_samples, num_val_samples = SelectImagesForTrainValidation(i+1, 1.0)
            print(str(num_train_samples) + "/" + str(num_val_samples) + " training/validation samples loaded")
        else:
            train_ds_tmp, val_ds_tmp, num_train_tmp, num_val_tmp = SelectImagesForTrainValidation(i+1, 1.0)
            print(str(num_train_tmp) + "/" + str(num_val_tmp) + " training/validation samples loaded")
            
            train_ds = train_ds.concatenate(train_ds_tmp)
            val_ds = val_ds.concatenate(val_ds_tmp)
            num_train_samples += num_train_tmp
            num_val_samples += num_val_tmp

    for i in range(CORe50_num_sequences-COR350_num_sequences_validation, CORe50_num_sequences):

        print("Loading sequence id: " + str(i+1) + "...", end='')

        train_ds_tmp, val_ds_tmp, num_train_tmp, num_val_tmp = SelectImagesForTrainValidation(i+1, 0.0)
        print(str(num_train_tmp) + "/" + str(num_val_tmp) + " training/validation samples loaded")
            
        train_ds = train_ds.concatenate(train_ds_tmp)
        val_ds = val_ds.concatenate(val_ds_tmp)
        num_train_samples += num_train_tmp
        num_val_samples += num_val_tmp

    print("Num training samples without augmentation: " + str(tf.data.experimental.cardinality(train_ds).numpy()))
    print("Num validation samples: " + str(tf.data.experimental.cardinality(val_ds).numpy()))

    if(augment == True):
        train_ds, num_train_samples = AugmentDataset(train_ds, num_train_samples)

    print("Num training samples with augmentation: " + str(tf.data.experimental.cardinality(train_ds).numpy()))

    if(enable_performance == True):
        train_ds = configure_for_performance(train_ds, num_train_samples, CORe50_batch_size)
        val_ds = configure_for_performance(val_ds, num_val_samples, CORe50_batch_size)

    return train_ds, val_ds

def PlotHistory(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

if __name__ == "__main__":

    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(gpu_devices))
    cpu_devices = tf.config.list_physical_devices('CPU')
    print("Num CPUs:", len(cpu_devices))

    pkl_file = open('F:\\Core50\\paths.pkl', 'rb') 
    paths = pkl.load(pkl_file)

    classes_for_validation =  sorted(random.sample(range(1, 11), 3))
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

    imgs = np.load('F:\\Core50\\core50_imgs.npz')['x']

    val_images = [imgs[x[0]:x[1]+1] for x in val_ranges]
    val_images = np.concatenate( val_images, axis=0 )
    train_images = [imgs[x[0]:x[1]+1] for x in train_ranges]
    train_images = np.concatenate( train_images, axis=0 )

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_array_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_array_labels))

    train_ds = train_ds.shuffle(len(train_images))

    train_ds = train_ds.map(MBNetV2_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(MBNetV2_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = configure_for_performance(train_ds, len(train_images), CORe50_batch_size)
    val_ds = configure_for_performance(val_ds, len(val_images), CORe50_batch_size)

    base_model = tf.keras.applications.MobileNetV2(input_shape=(MBNetV2_sizeX, MBNetV2_sizeY, 3),
                                                   weights="imagenet",
                                                   include_top=False)

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(COR250_num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=base_model.input, outputs=output)
    model.summary()

    # Create training and validation sets of CORe50
    #train_ds, val_ds = CreateCORe50Dataset(augment=False)

    tensorboard_cb_coarse = keras.callbacks.TensorBoard(log_dir=get_run_logdir(), 
                                                 histogram_freq = 1,
                                                 write_images = False)
    print("")
    print("********** Transfer learning using MobileNetV2! **********")
    print("")


    '''
    for layer in base_model.layers:
        layer.trainable = False


    #optimizer=tf.keras.optimizers.Adam(0.1)
    optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer=optimizer,
                  metrics=["accuracy"])
    checkpoint_cb = keras.callbacks.ModelCheckpoint("E:\\Proj\\PhD\\temp\\MobileNetV2_CORe50_checkpoint_coarse.h5", save_best_only=True)
    start = time.perf_counter()
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        callbacks=[tensorboard_cb_coarse, checkpoint_cb],
                        epochs=10,
                        validation_freq=5)
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds during coarse training.' % elapsed)
    model.save("E:\\Proj\\PhD\\temp\\MobileNetV2_CORe50_coarse_ultimate.h5")

    res_coarse_eval = model.evaluate(val_ds)
    print("Coarse training result: ", end='')
    print(res_coarse_eval)
    '''

    print("")
    print("********** Fine tuning training! **********")
    print("")

    for layer in base_model.layers:
        layer.trainable = True

    #optimizer = tf.keras.optimizers.Nadam(0.001)
    optimizer = tf.keras.optimizers.Nadam(0.001)
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer=optimizer,
                  metrics=["accuracy"])
    checkpoint_cb = keras.callbacks.ModelCheckpoint("E:\\Proj\\PhD\\temp\\MobileNetV2_CORe50_checkpoint_fine.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    tensorboard_cb_fine = keras.callbacks.TensorBoard(log_dir=get_run_logdir(), 
                                                 histogram_freq = 1,
                                                 write_images = False)
    start = time.perf_counter()
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        callbacks=[tensorboard_cb_fine, checkpoint_cb, early_stopping_cb],
                        epochs=60,
                        validation_freq=3)
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds during fine training.' % elapsed)

    model.save("E:\\Proj\\PhD\\temp\\MobileNetV2_CORe50.h5")
    #PlotHistory(history)

    '''
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    open("E:\\Proj\\PhD\\temp\\MobileNetV2_CORe50.tflite", "wb").write(tflite_model)

    # Quantize
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    '''