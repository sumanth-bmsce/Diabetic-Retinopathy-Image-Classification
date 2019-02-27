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
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#reading images from specified path,converting it to numpy array with images size of (224,224,3)
path = 'C:/Users/HarishChandra/Documents/FinalYearProject/Results and Images/sample/sample/train/'
img_name = os.listdir(path)
img_arr = np.zeros(shape=(6,290,290,3))
i = 0
for img in img_name[0:6]:
    x = image.load_img(path+img, target_size=(290,290,3))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_arr[i] = x
    i+=1
    if(i%100 == 0):
        print(i)

#loading weights without fc and softmax layers for Transfer learning.
model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
flatten_model = Sequential()
flatten_model.add(model)
#flatten the last max pool layer outputs (last CNN layer output)
flatten_model.add(Flatten())

print("predicting")
#extract features (convolutional autoencoder)
features_input = flatten_model.predict(img_arr, verbose = 1,batch_size = 4)
#saving CNN outputs in a local file on disk to load later
print("Extracted features shape ",features_input.shape)
np.save("imagedata_sample",features_input)