import os
import argparse
import csv
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-init_file", dest = "filename", help="initial file you want to convert")

args = parser.parse_args()
filename = args.filename

core = 10
sample_size = 2000

print("Parsing: "+filename)

data = pd.read_csv(filename,header = None)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

foldername = 'converted_'+filename.split('/')[1].split('c')[0]

os.mkdir(foldername)
# cnt = 0
for indeks in range(core):
     f = open(foldername+"/output."+str(indeks), "w")
     f.write(str(data[42][0]*2000.00)+"\n")
     f.close()
        
with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        indeks = int(row[0])/(sample_size/core)
        f = open(foldername+"/output."+str(int(indeks)), "a")
        for index in range(1,41):
            f.write(str(row[index])+"0,")
        f.write(f"{row[index]}0\n")
        f.close()