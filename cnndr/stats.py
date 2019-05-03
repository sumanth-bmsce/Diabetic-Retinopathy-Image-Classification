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


# Prints the confusion Matrix
def confusionmatrix(labels, y_pred):
    cf_matrix = confusion_matrix(labels.argmax(axis=1), y_pred.argmax(axis=1))
    return cf_matrix

def auroc(labels,y_pred):
    return tf.py_func(roc_auc_score, (labels,y_pred,tf.double))

# Prints the AUC value
def print_roc(labels,y_pred):
    print(roc_auc_score(labels, y_pred))

# Print classification_report with Precision and Recall
def print_classification_report(labels,y_pred):
    class_report = classification_report(y_true= labels.argmax(axis=1), y_pred= y_pred.argmax(axis=1))
    print(class_report)





