from zipfile import ZipFile
import sys

filename = sys.argv[1]

with ZipFile(filename,'r') as zip:
	# Printing the list of files extracted
	#zip.printdir()
	print('Extracting all the files') 
	zip.extractall(sys.argv[2]) 
	print('Done!') 
	
# cmd eg : python unzip.py srcfile_path destfile_path