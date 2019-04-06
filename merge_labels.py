import pandas as pd
root="F:/Dataset/DR/training dataset/"
filenames=['train001Labels_filter','train002Labels_filter','train003Labels_filter','train004Labels_filter','train005Labels_filter']
combined_csv=pd.concat([pd.read_csv(root+f+".csv",header=None) for f in filenames],sort=False)
combined_csv.to_csv("train_inception.csv", index=False,header=None)
print len(combined_csv)
print combined_csv[0:10]