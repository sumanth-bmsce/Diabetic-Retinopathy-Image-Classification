import cv2
import numpy as np
from PIL import Image, ImageFilter

# Converting images into grayscale
def convert_to_grayscale(img_arr):
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	return gray_image

# Apply Equalize Histogram
def apply_equalize_histogram(img_arr):
	eq_image = cv2.equalizeHist(gray_image)
	return eq_image

# Apply CLAHE
def apply_CLAHE(img_arr):
	clahe = cv2.createCLAHE()
	cl_image = clahe.apply(gray_image)
	return cl_image

# Filtering Black Images : For black images mean pixel value is 0
# Under Development
def filter_black_images(img_arr):
	if(np.mean(img_arr)==0):
		return 1
	else:
		return 0

# Cropping black borders of image
def crop_image(img_arr, tot=0):
	gray_image = convert_to_grayscale(img_arr)
	_, thresh = cv2.threshold(gray_image,1,255,cv2.THRESH_BINARY)
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)
	#crop = img_arr[y:y+h,x:x+w]
	crop =  gray_image[y:y+h,x:x+w]
	return crop

# Normal Bilinear Rescaling
def normal_bilinear_rescaling(img_path_name):
	img = Image.open(img_path_name)
	w, h = img.size
	img = img.resize((w,h), resample = Image.BILINEAR)
	img.save('bilinear_image.jpeg')

# Normalizing the image pixels
def img_normalization(img_arr):
	mean = np.mean(img_arr)
	std = np.std(img_arr)
	print(std)
	img_arr -= int(mean)
	img_arr = np.true_divide(img_arr, int(std))
	return img_arr
	

#blur and unsharp the images
def blur_and_unsharp_image(img_path_name):
	img = Image.open(img_path_name)
	img_blur = img.filter(ImageFilter.GaussianBlur)
	img_unsharp = img_blur.filter(ImageFilter.UnsharpMask)
	img_unsharp.save('final_image.jpeg')
	

image = cv2.imread('13_left.jpeg')
# Step 1
gray_image = convert_to_grayscale(image)
# Step 2
cropped_image = crop_image(gray_image)
cv2.imwrite('cropped_image.jpeg', cropped_image)
# Step 3
# saved as bilinear_image.jpeg
normal_bilinear_rescaling('cropped_image.jpeg')
# Step 4
bilinear_image = cv2.imread('bilinear_image.jpeg')
# Step 5
clahe_image = apply_CLAHE(bilinear_image)
cv2.imwrite('clahe_image.jpeg',clahe_image)
# Step 6
blur_and_unsharp_image('clahe_image.jpeg')



