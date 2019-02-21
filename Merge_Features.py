import numpy as np

"""
features_input = np.load("imagedata_train002_250_750.npy")
print(features_input.ndim) #2
print(features_input.shape) #(500,25088)
"""
no_of_images = 4000
flatten_arr_dim = 25088
features_input = np.zeros(shape=(no_of_images,flatten_arr_dim),dtype = np.float32)

feature_file_index = [(0,250),(250,750),(750,1750),(1750,2750),(2750,4000)]
path = "C:/Users/Nagaraj G/Desktop/Final Sem Project/Code/"
feature_file_names = ['imagedata_train002.npy','imagedata_train002_250_750.npy','imagedata_train002_750_1750.npy','imagedata_train002_1750_2750.npy','imagedata_train002_2750_4000.npy']

for index in range(len(feature_file_index)):
    name = path + feature_file_names[index]
    length = feature_file_index[index]
    start = length[0]
    end = length[1]
    flatten_features = np.load(name)
    for i in range(0,end-start):
        features_input[start+i] = flatten_features[i]

print("Saving features")
print(features_input.shape)
print(features_input.ndim)
np.save("imagedata_train_0_4000",features_input)





