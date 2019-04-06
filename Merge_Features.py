import numpy as np

"""
features_input = np.load("imagedata_train002_250_750.npy")
print(features_input.ndim) #2
print(features_input.shape) #(500,25088)
"""
no_of_images = 9316							
flatten_arr_dim = 2048
features_input = np.zeros(shape=(no_of_images,flatten_arr_dim),dtype = np.float32)

feature_file_index = [(0,2258),(2258,4415),(4415,6673),(6673,8906),(8906,9316)]
path = "C:/Users/Nagaraj G/Desktop/Final Sem Project/Code/"
#feature_file_names = ['imagedata_train001_filter_v3','imagedata_train002_filter_v3','imagedata_train003_filter_v3','imagedata_train004_filter_v3','imagedata_train005_filter_v3']

#feature_file_index = [(0,2258),(2258,3161),(3161,4176),(4176,5127),(5127,5307)]
feature_file_names = ['imagedata001_preprocessed_v3','imagedata002_preprocessed_v3','imagedata003_preprocessed_v3','imagedata004_preprocessed_v3','imagedata005_preprocessed_v3']
for index in range(len(feature_file_index)):
    name = path + feature_file_names[index]+".npy"
    length = feature_file_index[index]
    start = length[0]
    end = length[1]
    try:
    	flatten_features = np.load(name)
    except:
    	print name
    	exit()
    for i in range(0,end-start):
        features_input[start+i] = flatten_features[i]
    flatten_features = []

print("Saving features")
print(features_input.shape)
print(features_input.ndim)
np.save("inception_traindata",features_input)





