import csv
import struct
import sys
import numpy as np
from kaldiio import WriteHelper
import glob

if len(sys.argv) == 1:
    print("Usage:  csvtoark.py directory")
    exit(-1)

fspec = "ark:new_file.ark"
with WriteHelper(fspec) as writer:
    files = glob.glob(sys.argv[1]+"/*.csv")
    i = 0
    for f in files:
        print(f)
        uttname = "utt" + str(i)
        with open(f, 'r') as csvfile:
            data = []
            reader = csv.reader(csvfile)
            for row in reader:
                if row[len(row)-1] == '':  # fix trailing comma
                    row = row[:-1]
                data.append([float(i) for i in row])
        npdata = np.single(data)
        writer(uttname, npdata)
        i = i + 1
