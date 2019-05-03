#Extraction of features from freezed (non-trainable) CNN layers

#import keras,numpy,os libraries

import numpy as np
import keras
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def VGG16Conv(path, initial, final):
    print("Loading Images .....")
    # reading images from specified path,converting it to numpy array with images size of (224,224,3)
    img_name = os.listdir(path)
    no_of_images = final - initial
    img_arr = np.zeros(shape=(no_of_images, 224, 224, 3))
    i = 0

    for img in img_name[initial:final]:
        x = image.load_img(path + img, target_size=(224, 224, 3))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_arr[i] = x
        i += 1
        if (i % 250 == 0):
            print(i)
    print("Loading Complete")
    print("Constructing VGG16 Model .....")
    model_vgg = VGG16(include_top = False, weights = 'imagenet',input_shape = (224,224,3))
    flatten_model_vgg = Sequential()
    flatten_model_vgg.add(model_vgg)
    flatten_model_vgg.add(Flatten())
    print("Predicting VGG16")
    features_input_vgg = flatten_model_vgg.predict(img_arr, verbose = 1,batch_size = 4)
    print("Extracted VGG16 features shape ",features_input_vgg.shape)
    print("VGG16 Convolutions Done and Returned")
    return features_input_vgg


def Inceptionv3Conv(path, initial, final):
    print("Loading Images .....")
    # reading images from specified path,converting it to numpy array with images size of (224,224,3)
    img_name = os.listdir(path)
    no_of_images = final - initial
    img_arr = np.zeros(shape=(no_of_images, 299, 299, 3))
    i = 0

    for img in img_name[initial:final]:
        x = image.load_img(path + img, target_size=(299, 299, 3))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_arr[i] = x
        i += 1
        if (i % 250 == 0):
            print(i)
    print("Loading Complete")
    # loading weights without fc and softmax layers for Transfer learning.
    print("Constructing Inception v3 Model .....")
    model_inception = InceptionV3(include_top = False, weights = 'imagenet',input_shape=(299,299,3),pooling="avg")
    flatten_model_inception = Sequential()
    flatten_model_inception.add(model_inception)
    print("Predicting Inception v3")
    features_input_inception = flatten_model_inception.predict(img_arr, verbose=1, batch_size=4)
    print("Extracted Inception v3 features shape ", features_input_inception.shape)
    print("Inception v3 Convolutions Done and Returned")
    return features_input_inception
