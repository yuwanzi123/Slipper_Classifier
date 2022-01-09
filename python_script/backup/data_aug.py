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
                #Iterator
                aug_iter = datagen.flow(img, batch_size = 1)
                # generate samples and plot
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,15))
                #generate batch of images
                for i in range(4):
                    #convert to unsigned integers
                    image = next(aut_iter)[0].astype('unit8')
                    image.save(new_path + item + 'rotated.jpg', 'JPEG', quality = 90)

                    #plot image
                    ax[i].imshow(image)
                    ax[i].axis('off')

rotation()

                    
