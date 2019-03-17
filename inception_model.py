import datetime as dt
import pandas as pd
import numpy as np
import keras
import os
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

#reloading CNN outputs compressed image data  from file
features_input = np.load("C:/Users/HarishChandra/Documents/FinalYearProject/Implementations/imagedata_sample.npy")

#Converting labels to categorial one hot encoding
labels = pd.read_csv("C:/Users/HarishChandra/Documents/FinalYearProject/Results and Images/sample/sample/train/sample_labels.csv",header = None)
print(labels.head())
labels = labels.values
labels = labels[0:4000,1]
print(labels.shape)
print(labels[0])
labels = keras.utils.to_categorical(labels, num_classes=5)

#Adding classifier neural network (Dense + Softmax)
new_model = Sequential()
#add a fc1 layer with 256 output neurons and a dropout layer1
new_model.add(Dense(units = 256, activation = 'relu', kernel_initializer='uniform'))
new_model.add(Dropout(rate = 0.5, noise_shape=None, seed=None))
#add a fc2 layer with 256 output neurons and a dropout layer2
new_model.add(Dense(units = 256, activation = 'relu', kernel_initializer='uniform'))
new_model.add(Dropout(rate = 0.5, noise_shape=None, seed=None))
#adding a softmax layer with 5 classes
new_model.add(Dense(units = 5))
new_model.add(keras.layers.Activation('softmax'))
#configures the model for training
sgd = SGD(lr=0.001, decay=0.0, momentum=0.9,nesterov=True)
new_model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'] )

#train the model for given number of epochs
training_history = {}
print(dt.datetime.now())
try:
    training_history = new_model.fit(x = features_input, y = labels, batch_size = 32, epochs = 30, verbose = 2, validation_split = 0.2)
except Exception as error:
    print(error)
finally:
    print(training_history.history)  #validation accuracy and loss, training accuracy and loss
    new_model.save('inception_model_sample.h5')  # creates a HDF5 file 'my_model.h5'
    new_model.save_weights('inception_model_sample_weights.h5') #creates a HDF5 'my_model_weights.h5' weight file
#End of training and va;idation   
print(dt.datetime.now())
#plotting accuracy of training and validation
plt.plot(training_history.history['acc'])
plt.plot(training_history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plotting loss of training and validation
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()