import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20000
parser.add_argument("-init_file", dest = "filename", help="initial file you want to convert")

args = parser.parse_args()
filename = args.filename

print("Parsing: "+filename)

data = pd.read_csv(filename,header = None)

foldername = 'converted_'+filename.split('/')[1].split('c')[0]

os.mkdir(foldername)

for idx,row in data.iterrows():
    # print(row[1])
    f = open(foldername+"/output."+str(idx), "a")
    for index in range(1,41):
        f.write(str(row[index])+",")
    f.write(str(row[index+1])+"\n")
    f.close()