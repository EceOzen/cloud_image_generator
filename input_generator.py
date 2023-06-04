# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
from numpy import asarray
from numpy import savez_compressed
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from numpy import asarray
from numpy import savez_compressed
import os

# path to the image directory
dir_data  = "train"
 
# setting image shape to 32x32
img_shape = (32, 32, 3)
 
# listing out all file names
nm_imgs   = np.sort(os.listdir(dir_data))

X_train = []
for file in nm_imgs:
    try:
        img = Image.open(dir_data+'/'+file)
        img = img.convert('RGB')
        img = img.resize((32,32))
        img = np.asarray(img)/255
        X_train.append(img)
    except:
        print("something went wrong")

print(X_train)

X_train = np.array(X_train)
X_train.shape

 
# save to npy file
savez_compressed('Downloads/cloud_kaggle_images_32x32.npz', X_train)
print("done")
