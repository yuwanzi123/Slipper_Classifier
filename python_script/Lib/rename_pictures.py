import os
os.getcwd()
collection = "/home/zz/Closet/database/raw_data/google_skirt_knee/"
for i, filename in enumerate(os.listdir(collection)):
    os.rename(collection + filename, collection + str(i) + ".jpg")
