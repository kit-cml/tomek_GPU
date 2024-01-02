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

for indeks in range(10):
     f = open(foldername+"/output."+str(indeks), "a")
     f.write(str(data[0][42]*2000.00)+"\n")
     f.close()

for idx,row in data.iterrows():
    # print(row[1])
    # devided as core (10 cores..)
    f = open(foldername+"/output."+str(int(idx/200)), "a")
    for index in range(1,41):
        f.write(str(row[index])+",")
    f.write(str(row[index+1])+"\n")
    f.close()