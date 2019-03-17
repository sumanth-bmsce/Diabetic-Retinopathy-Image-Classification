from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

#training data and labels
f_train = np.load("imagedata_train001.npy")
l_train = pd.read_csv("F:/Dataset/Diabetic retinopathy/train001Labels_filter.csv",header = None)
l_train = l_train.values
l_train = l_train[:,1]
l_train = l_train.astype('int')

#testing data and labels
f_test = np.load("imagedata_train005.npy")
l_test = pd.read_csv("F:/Dataset/Diabetic retinopathy/train005Labels_filter.csv",header = None)
l_test = l_test.values
l_test = l_test[:,1]
l_test = l_test.astype('int')

#training
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(f_train,l_train)
print("training accuracy = ",svm_model_linear.score(f_train,l_train)*100)

#testing
svm_predictions = svm_model_linear.predict(f_test)
cm = confusion_matrix(l_test, svm_predictions)
print("confusion_matrix = ",cm)
print("testing accuracy = ",svm_model_linear.score(f_test,l_test)*100)
