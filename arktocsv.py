import csv
import struct
import sys
import kaldiio

def make_csv(arkname, csvname, index):
    arkdesc = "ark:" + arkname
    d = kaldiio.load_ark(arkname)
    i = 0;
    for key, data in d:
        if i == index:
            with open(csvname, 'w') as csvfile:
                writer = csv.writer(csvfile, lineterminator='\n')
                if len(data.shape) == 1:
                    writer.writerow(data)
                else:
                    writer.writerows(data)
                csvfile.close()
        i = i + 1


ark_name = ""
csv_name = ""

if len(sys.argv) == 1:
    print("Usage:  arktocsv.py ark_name <csv_name>")
    exit(-1)
if len(sys.argv) > 1:
  ark_name = sys.argv[1]
  csv_name = ark_name.replace(".ark",".csv")
if len(sys.argv) > 2:
  csv_name = sys.argv[2]
            
make_csv(ark_name,csv_name, 0)
