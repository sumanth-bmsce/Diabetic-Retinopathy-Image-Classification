import pandas as pd
import numpy as np
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

#original VGG16 network weight file
#model1 = VGG16(include_top = True, weights = 'imagenet', classes = 1000)  
#loading weights without fc and softmax layers for Transfer learning.
model = VGG16(include_top = False, weights = 'imagenet')
print(model.summary())
print(model.inputs)
print(model.outputs)


#flatten the last max pool layer outputs
model.add(Flatten())

#add a fc1 layer with 4096 output neurons and a dropout layer1
model.add(units = 4096, activation = 'relu', kernel_initializer='uniform')
keras.layers.Dropout(rate = 0.5, noise_shape=None, seed=None)

#add a fc2 layer with 4096 output neurons and a dropout layer2
model.add(units = 4096, activation = 'relu', kernel_initializer='uniform')
keras.layers.Dropout(rate = 0.5, noise_shape=None, seed=None)

#adding a softmax layer with 5 classes
model.add(Dense(units = 10))
model.add(keras.layers.Activation('softmax'))

#configures the model for training
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')

#train the model for given number of epochs
model.fit(x = , y = , batch_size = , epochs = , verbose = 2, validation_split = 0.3)

#return accuracy for the test data with trained weights
model.evaluate(x=, y=, batch_size=, verbose=1)
