import numpy as np
from scipy import ndimage
#import scipy.ndimage
import imageio
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

path = "/home/zz/Slipper/slippers_a/database/test_sample/0google.jpg"
new_path = "/home/zz/Slipper/slippers_a/database/test_sample_rotation"

datagen = ImageDataGenerator(rotation_range = 30) 
image = np.expand_dims(imageio.imread(path), 0)
datagen.fit(image)

for x, val in zip(datagen.flow(image,                    #image we chose
        save_to_dir = new_path,     #this is where we figure out where to save
        save_prefix = 'slipper',        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format = 'jpg'),
        range(1)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
    break

#这可能是jupeter里面的代码

