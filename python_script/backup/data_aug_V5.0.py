from keras.preprocessing.image import ImageDataGenerator
import os, sys

path = "/home/zz/Slipper/slippers_a/database/test_sample/"
new_path = "/home/zz/Slipper/slippers_a/database/test_sample_rotation/"
dirs = os.listdir(path)

for item in dirs:
    datagen=ImageDataGenerator(rotation_range = 30)
    image = datagen.flow_from_directory(path + item,
                                        target_size=(50,50),
                                        save_to_dir=new_path,
                                        class_mode='binary',
                                        save_prefix='slipper',
                                        save_format='jpg',
                                        batch_size=2)
    image.next()
