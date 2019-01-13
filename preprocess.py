import pandas as pd
import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
import os

def createDir(file_path):
	"""
	create a new directory if file_path doesn't exist

	"""
	if not os.path.exists(file_path):
		os.makedirs(file_path)


def validateImages(file_path):
	"""
	checks whether image is completely black or not. Discard those images which are completely black
	create a new csv file with image name and validate/not validate status
	
	"""
	img_names = os.listdir(file_path)
	val_img = {}
	for img in img_names:
		if(np.mean(np.array(sk.io.imread(file_path+"/"+img))) != 0):
			val_img[img] = 1
		else:
			val_img[img] = 0
			os.remove(file_path + "/" + img)

	validated_img = pd.DataFrame(val_img.items(),columns = ["Image_name","validation_status"])
	validated_img.to_csv(file_path + "/image_validated_labels.csv",sep = ",", header = True,index = False)

def refine_labels(file_path,status_file_path):
	"""
	remove invalid images from labels file

	"""
	labels = pd.DataFrame(status_file_path)
	i = 0
	status = labels["validation_status"]
	for stat in status:
		if(stat == 0):
			status.drop(status.index[i])
		i+=1


def resizeImages(data_path,file_path):
	"""
	resizes image size to 256 x 256. Create a new folder and save resized images in that folder

	"""
	createDir(file_path)
	req_x = 1800
	req_y = 1800
	img_names = os.listdir(data_path)
	for img in img_names:
		img_arr = sk.io.imread(data_path + "/" + img)
		y,x,channel = img_arr.shape
		crop_y = y/2 - req_y/2
		crop_x = x/2 - req_x/2
		img_arr = img_arr[crop_y : crop_y+req_y,crop_x : crop_x+req_x]
		try:
			img_arr = sk.transform.resize(img_arr,(256,256))
		except RuntimeWarning:
			pass
		sk.io.imsave(file_path + "/" + img,img_arr)


#resizeImages("C:/Users/Nagaraj G/Desktop/DR_Reference/data/sample","C:/Users/Nagaraj G/Desktop/DR_Reference/data/resized_training_dataset")
#validateImages("C:/Users/Nagaraj G/Desktop/DR_Reference/data/sample")
