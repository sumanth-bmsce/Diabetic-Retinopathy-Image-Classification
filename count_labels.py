import pandas as pd
import numpy as np
#labels = pd.read_csv("F:/Dataset/Diabetic retinopathy/trainLabels_1234.csv",header = None)
labels = pd.read_csv("F:/Dataset/DR/training dataset/test.csv",header = None)
#labels = pd.read_csv("train_inception.csv",header = None)
labels = labels.values
unique, counts = np.unique(labels, return_counts=True)
labels_freq = np.asarray((unique, counts)).T
print(labels_freq[1:5])
print(len(labels))