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

def rotation():
    for item in dirs:
        if os.path.isfile(path + item):
            img = Image.open(path + item)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                #ImageDataGenerator rotation
                #datagen = ImageDataGenerator(rotation_range = 30, fill_mode = 'nearest')
            datagen = ImageDataGenerator(rotation_range = 30)
            datagen.fit(img)
            i = 0
            for img_batch in datagen.flow(img, batch_size = 9):
                for img in img_batch:
                    plt.subplot(330 + 1 + i)
                    plt.imshow(img)
                    i = i + 1
                if i >= batch_size:
                    break


rotation()

                    
