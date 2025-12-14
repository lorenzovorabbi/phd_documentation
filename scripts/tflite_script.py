#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
import imageio
import numpy as np
import argparse
import os
import sys
import time

def List_all_pngs(folder_path):

    filelist=os.listdir(folder_path)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    return filelist

def main(model_path_, image_path):
    interpreter = tflite.Interpreter(model_path=model_path_)

    detin = interpreter.get_input_details()
    detout = interpreter.get_output_details()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    im = imageio.imread(image_path).astype(input_details["dtype"])

    if input_details['dtype'] == np.uint8:
          #input_scale, input_zero_point = input_details["quantization"]
          #im = im.astype(np.float32)
          im = im.astype(np.uint8)
          #im = (im / 127.5)-1.0
          #im = im / input_scale + input_zero_point
    elif input_details['dtype'] == np.float32:
          #im = (im / 127.5)-1.0
          im = (im / 127.5)-1.0

    '''
    for index, val in np.ndenumerate(im):
        val = (val / 127.5)-1.0
        im[index] = val
    '''

    test_image = np.expand_dims(im, axis=0).astype(input_details["dtype"])
    interpreter.resize_tensor_input(input_details['index'], list(test_image.shape))
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    predicted = output.argmax()

    _, file = os.path.split(image_path)
    res = file.split("_")
    gt_val = int(res[2])-1

    print("Predicted vs. GT class " + str(predicted) + "/" + str(gt_val))

    if (gt_val == predicted):
        return 1
    else:
        return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train MobileNetV2 on CORe50 dataset')
    parser.add_argument('-m', type=str, help='path to the tflite model')
    parser.add_argument('-i', type=str, help='path to the image to test')
    parser.add_argument('-f', type=str, help='path to the folder containing all images')

    args = parser.parse_args()

    if (args.__dict__['m'] is None) or ((args.__dict__['f'] is None) and (args.__dict__['i'] is None)):
        parser.print_help()
        sys.exit(0)

    if args.__dict__['f'] is not None:
        images = List_all_pngs(args.f)

        num_corrects = 0
        for index, item in enumerate(images):
            print(str(index+1) + ") " + item + " - ", end='')
            num_corrects += main(args.m, args.f+item)
        print("Accuracy: " + str(num_corrects*100.0/len(images)) + "%")
    else:
        main(args.m, args.i)