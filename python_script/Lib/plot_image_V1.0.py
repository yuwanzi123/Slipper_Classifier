import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import time

from PIL import Image

#%matplotlib inline

PATH = "/home/zz/Slipper/slippers_a/database/test_sample/0google.jpg"
new_path = "/home/zz/Slipper/slippers_a/database/test_sample_rotation/"

for i in range(1,4):
    p = PATH.format(i)
    #print p
    image = mpimg.imread(p) # images are color images
    plt.gca().clear()
    plt.imshow(image);
    display.display(plt.gcf())
    display.clear_output(wait=True)
    #imResize = image.resize((128, 128), Image.ANTIALIAS)
    #imResize.save(new_path + item  + ' resized.jpg', 'JPEG', quality=90)
    time.sleep(1.0) # wait one second
