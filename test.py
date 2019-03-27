import datetime as dt
import pandas as pd
import numpy as np
import keras
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from sklearn.metrics import roc_auc_score
from features_input import *
from stats import *


def test(img_path, initial, final, model_path, test_csv_path,stats):
    #reloading CNN outputs compressed image data  from file
    print("Load CNN outputs")
    features_input = np.load("E:\\DR\\trained_models\imagedata_testsample.npy")
    #features_input = VGG16Conv(img_path, initial, final)
    #Converting labels to categorial one hot encoding
    labels = pd.read_csv(test_csv_path,header = None)
    print(labels.head())
    labels = labels.values
    labels = labels[initial:final,1]
    #print(labels.shape)
    #print(labels[0])
    labels = keras.utils.to_categorical(labels, num_classes=5)

    new_model = load_model(model_path)
    print("Testing ...")
    evaluation_result = new_model.evaluate(x = features_input, y = labels, batch_size= 1, verbose=1)
    print("Result")
    print("Loss = " + str(evaluation_result[0]))
    print("Classification Accuracy = " + str(evaluation_result[1]*100) + " % ")
    #print("evaluation result = ",evaluation_result)   #loss,accuracy
    y_pred = new_model.predict_proba(features_input)
    y_pred_keras = new_model.predict(features_input)
    #print(y_pred) #Prints list of probabilities
    if(stats==1):
        # Display Confusion Matrix
        matrix = confusionmatrix(labels, y_pred)
        print("Confusion Matrix")
        print(matrix)
        print("ROC Value")
        print_roc(labels, y_pred)
        print("Classification Report")
        print_classification_report(labels, y_pred)


    #roc_auc_score(labels,y_pred)
    #print (cohen_kappa_score(labels, y_pred, labels=None, weights="quadratic", sample_weight=None))

test("E:\\DR\\datasets\\filtered_dataset\\train001\\", 0, 100,
     "E:\\DR\\trained_models\\oversample_model_train005_0_1427_512.h5", "E:\\DR\\labels\\train001_filter.csv", 1)