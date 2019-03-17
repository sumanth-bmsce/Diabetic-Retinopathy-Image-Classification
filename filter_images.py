import csv 
import os
def generate_labels_from_file(parent_file_path, all_labels_file_path, output_file_path):
	train_image = os.listdir(parent_file_path)
	dict_labels_all = {}
	dict_labels_req_file = {}

	with open(all_labels_file_path,"r") as csvfile:
		reader = csv.reader(csvfile, delimiter = ",")
		for row in reader:
			dict_labels_all[row[0]] = row[1]

	for filename in train_image:
		filename = filename.replace(".jpeg","")
		if(int(dict_labels_all[filename]) == 0):
			dict_labels_req_file[filename] = dict_labels_all[filename]

	print "labels length = ",len(dict_labels_req_file)
	
	with open(output_file_path,"w") as csvfile:
		for name,label in dict_labels_req_file.items():
			csvfile.write(name + "," + label + "\n")
	

if __name__ == "__main__":
	generate_labels_from_file('F:/Dataset/DR/training dataset/train003/','G:/Diabetic Retinopathy/trainLabels.csv','F:/Dataset/Diabetic retinopathy/train003Labels_filter.csv')

