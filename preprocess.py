import cv2
import numpy as np
from PIL import Image, ImageFilter
import glob
import os


# Converting images into grayscale
def convert_to_grayscale(img_arr):
    gray_image = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return gray_image

# Apply Equalize Histogram
def equalize_hist(img_arr):

    for c in range(0, 2):
        img_arr[:,:,c] = cv2.equalizeHist(img_arr[:,:,c])
    return img_arr

# Cropping black borders of image
def crop_image(img_arr, tot=0):

    gray_image = convert_to_grayscale(img_arr)
    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    # crop = img_arr[y:y+h,x:x+w]
    crop = img_arr[y:y + h, x:x + w]
    return crop

# Normal Bilinear Rescaling
'''def normal_bilinear_rescaling(src_img_path, dest_img_path, image_name):
    img = Image.open(image_name)
    img = img.resize((500, 500), resample=Image.BILINEAR)
    img.save(dest_img_path + image_name)'''

def normal_bilinear_rescaling(img_arr):
    bilinear_image = cv2.resize(img_arr, (500,500), interpolation = cv2.INTER_LINEAR)
    return bilinear_image

def preprocess(img_path):
    imgs = glob.glob(img_path + "*.jpeg")
    img_count = 0
    names = [os.path.basename(x) for x in imgs]
    print(names)
    for imgfile in imgs:

        image = cv2.imread(imgfile)
        # Crop the image to remove the black borders
        cropped_image = crop_image(image)
        bilinear_image = normal_bilinear_rescaling(cropped_image)
        #eq_image = equalize_hist(bilinear_image)
        #print(imgfile)
        cv2.imwrite("E:\\DR\\datasets\\preprocessed_dataset\\train001\\" + names[img_count], bilinear_image)
        img_count += 1
        if(img_count % 1000 == 0):
            print(img_count)

img_path = "E:/DR/datasets/original_dataset/train001/"
preprocess(img_path)


