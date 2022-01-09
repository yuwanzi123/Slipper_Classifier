import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

from PIL import Image
import os, sys

path = "/home/zz/Slipper/slippers_a/database/test_sample"
new_path = "/home/zz/Slipper/slippers_a/database/test_sample_rotation"
dirs = os.listdir( path )

#ImageDataGenerator rotation
#datagen = ImageDataGenerator(rotation_range = 30, fill_mode = 'nearest')
datagen = ImageDataGenerator(rotation_range = 30)

def rotation():
    for item in dirs:
        if os.path.isfile(path + item):
            img = Image.open(path + item)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            x = img_to_array(img)
            x = x.reshape((1, ) + x.shape)
            i = 0

            for batch in datagen.flow(x, batch_size = 1,
                                      save_to_dir = new_path,
                                      save_prefix = 'car' , save_format = 'jpeg'):
                i += 1
                if i > 5:
                    break

                


rotation()

                    
