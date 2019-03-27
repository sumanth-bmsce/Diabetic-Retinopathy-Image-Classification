import numpy as np
import os
import cv2
import glob
import csv


def filter_black(image_file_path, dst_image_path):
    non_black_count = 0
    img_count = 0
    non_black_image_names = list()
    imgs = glob.glob(image_file_path + "*.jpeg")
    img_names = [os.path.basename(x) for x in imgs]
    print("Black Image Filtering in" + str(image_file_path))
    for imgfile in imgs:
        img_arr = cv2.imread(imgfile)
        if (_filter_black(img_arr)!=1):
            cv2.imwrite(dst_image_path + img_names[img_count], img_arr)
            non_black_image_names.append(img_names[img_count])
            non_black_count += 1
        img_count += 1
        if (img_count % 250 == 0):
            print("Image Count" + str(img_count))

    return non_black_image_names, non_black_count


def _filter_black(img_arr):
    if (np.mean(img_arr) < 28):
        return 1
    else:
        return 0


def filter_labels(src_image_file_path, dst_image_file_path, src_image_label_path, dst_image_label_path):
    print(src_image_file_path)
    print(dst_image_file_path)
    dict_labels_all = {}
    dict_non_black = {}
    parent_img_path = []
    non_black_img_names, non_black_count = filter_black(src_image_file_path, dst_image_file_path)
    print("Image filtering completed :: Label Filtering Started")
    #print(non_black_img_names)
    print("Non Black Image Count " + str(non_black_count))
    with open(src_image_label_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            dict_labels_all[row[0]] = row[1]
    for name in non_black_img_names:
        file = os.path.splitext(os.path.basename(name))[0]
        parent_img_path.append(file)

    for name in parent_img_path:
        #print(name)
        #print(dict_labels_all[name])
        dict_non_black[name] = dict_labels_all[name]


    with open(dst_image_label_path, "w") as csvfile:
        for name, label in dict_non_black.items():
            csvfile.write(name + "," + label + "\n")















#"E:/DR/datasets/original_dataset/train001"