#!/usr/bin/python
from PIL import Image
import os, sys

path = "/home/zz/Slipper/slippers_a/database/slippers_a_non_slipper_raw/tool/"
new_path = "/home/zz/Slipper/slippers_a/database/slippers_non_res/res_tool/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            if im.mode != 'RGB':
                im = im.convert('RGB')

            f, e = os.path.splitext(path+item)
            imResize = im.resize((128, 128), Image.ANTIALIAS)
            imResize.save(new_path + item + 'tool resized.jpg', 'JPEG', quality=90)

resize()
