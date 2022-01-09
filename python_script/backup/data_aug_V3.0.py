import os, sys
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

path = "/home/zz/Slipper/slippers_a/database/test_sample"
new_path = "/home/zz/Slipper/slippers_a/database/test_sample_rotation"
dirs = os.listdir( path )

datagen = ImageDataGenerator(rotation_range = 30) 

for f in dirs:
    img = load_img(f)  
    x = img_to_array(img) 
    # Reshape the input image 
    x = x.reshape((1, ) + x.shape)  
    i = 0

    # generate 5 new augmented images 
    for batch in datagen.flow(x, batch_size = 1, 
                      save_to_dir = new_path,
                      save_prefix ='car', save_format ='jpeg'):
        i += 1
        if i > 5: 
            break
