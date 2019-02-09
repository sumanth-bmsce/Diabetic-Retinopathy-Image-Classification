import numpy as np
import cv2
import os
import glob
import sys
import pandas as pd
import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

imgs = glob.glob('*.jpeg')
print('Found files:')
print(imgs)
folder = 'resize_images'
if not os.path.exists(folder):
    os.makedirs(folder)
cropx = 1890;
cropy = 1890;
for imgFile in imgs:
	img = cv2.imread(imgFile)
	y,x,channel = img.shape
	print (str(x) + "\t" + str(y)) 
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)
	print (str(startx) + "\t" + str(starty)) 
	img = img[starty:starty+cropy,startx:startx+cropx]
	img = cv2.resize(img, (224,224))
	dim1 = img.shape[0]
	dim2 = img.shape[1]
	dim3 = img.shape[2]
	print (str(dim1) + "\t" + str(dim2)  + "\t" + str(dim3))
	cv2.imwrite(folder+"/"+imgFile,img)

