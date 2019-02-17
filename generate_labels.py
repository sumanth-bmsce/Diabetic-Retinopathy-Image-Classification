import glob
import csv 

def generate_labels_from_file(file_pattern, parent_file_path, all_labels_file_path, output_file_path):
	train_image=[]
	dict_labels_all = {}
	dict_labels_req_file = {}
	imgs = glob.glob(file_pattern) 

	for filename in imgs:
		filename = filename.replace(parent_file_path,"")
		filename = filename.replace(".jpeg","")
		train_image.append(filename)

	print (len(train_image))
	print (train_image[0])

	with open(all_labels_file_path,"r") as csvfile:
		reader = csv.reader(csvfile, delimiter = ",")
		for row in reader:
			dict_labels_all[row[0]] = row[1]

	for filename in train_image:
		dict_labels_req_file[filename] = dict_labels_all[filename]

	with open(output_file_path,"w") as csvfile:
		for name,label in dict_labels_req_file.items():
			csvfile.write(name + "," + label + "\n")
	

if __name__ == "__main__":
	generate_labels_from_file('E:\\DR\\train004\\train\\*.jpeg','E:\\DR\\train004\\train\\','E:\\DR\\tl.csv','E:\\DR\\tl004.csv')

	

