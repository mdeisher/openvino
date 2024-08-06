import csv
import struct
import sys
import numpy as np
from kaldiio import WriteHelper

def make_ark(csvname, arkname):
    with open(csvname, 'r') as csvfile:
        data = []
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(i) for i in row])
    fspec = "ark:"+arkname
    with WriteHelper(fspec) as writer:
        uttname = "utt0"
        data = np.single(data)
        writer(uttname, data)

ark_name = ""
csv_name = ""

if len(sys.argv) == 1:
    print("Usage:  csvtoark.py csv_name <ark_name>")
    exit(-1)
if len(sys.argv) > 1:
  csv_name = sys.argv[1]
  ark_name = csv_name.replace(".csv",".ark")
  if ark_name == csv_name:
      print("Error:  attempt to overwrite source file!")
      exit(-1)
      
if len(sys.argv) > 2:
  ark_name = sys.argv[2]

make_ark(csv_name,ark_name)
