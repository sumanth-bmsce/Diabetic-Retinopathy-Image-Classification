import numpy as np
import keras
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import keras.applications.inception_v3
from keras.applications.vgg16 import VGG16
import keras.applications.vgg16
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def VGG16(img_path):
    x = image.load_img(img_path, target_size=(224,224,3))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.inception_v3.preprocess_input(x)
    model = VGG16(include_top = False, weights = 'imagenet',input_shape = (224,224,3))
    flatten_model = Sequential()
    flatten_model.add(model)
    #flatten the last max pool layer outputs (last CNN layer output)
    flatten_model.add(Flatten())
    print("Extracting features from VGG16")
    #extract features (convolutional autoencoder)
    features_input = flatten_model.predict(img_arr, verbose = 1,batch_size = 1)
    trained_model = load_model('/gdrive/My Drive/DiabeticRetinopathy_Detection/my_model_v3_train1234.h5')
    prediction = trained_model.predict(features_input, verbose = 1,batch_size = 1)
    #contains index of prediction result(class no) for ex for 2 img predicted_class = [2,1]
    predicted_class = prediction.indexnp.argmax(pred,-1)
    return predicted_class[0]         


def InceptionV3(img_path):
    x = image.load_img(img_path, target_size=(299,299,3))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.inception_v3.preprocess_input(x)
    #loading weights without fc and softmax layers for Transfer learning.
    model = InceptionV3(include_top = False, weights = 'imagenet',input_shape=(299,299,3),pooling="avg")
    flatten_model = Sequential()
    flatten_model.add(model)
    print("Extracting features from inception_v3")
    #extract features (convolutional autoencoder)
    features_input = flatten_model.predict(img_arr, verbose = 1,batch_size = 1)
    trained_model = load_model('/gdrive/My Drive/Inception_DR/inceptionv3_train.h5')
    prediction = trained_model.predict(features_input, verbose = 1,batch_size = 1)
    #contains index of prediction result(class no) for ex for 2 img predicted_class = [2,1]
    predicted_class = prediction.indexnp.argmax(pred,-1)    
    return predicted_class[0]

def getClass(class_no):
	labels = {
	0:"No",
	1:"Mild",
	2:"Moderate",
	3:"Severe",
	4:"Proliferative"
	}
	return labels[class_no]

img_path = ""
pred1 = VGG16(img_path)
if(pred1 != 0):
	pred2 = InceptionV3(img_path)
	result = getClass(max(pred1,pred2))
else:
	result = getClass(pred1)

result = result + " Diabetic Retinopathy"






