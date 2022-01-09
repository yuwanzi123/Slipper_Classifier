import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

from PIL import Image
import os, sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
                
            f, e = os.path.splitext(path+item)
            r_image = rotate(img, angle = 45)
            r_image.save(new_path + item + 'rotated.jpg', 'JPEG', quality = 90)

                


rotation()

                    
