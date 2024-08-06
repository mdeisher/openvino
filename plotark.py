import struct
import os
import math
import matplotlib.pyplot as plt
import sys
from kaldiio import ReadHelper

def plot_arrays(name1, uttname1, data1, name2, uttname2, data2):
  if ((len(uttname1) > 0) and (len(uttname2) > 0)):
    for i in range(len(uttname1)):
      num_rows = data1[i].shape[0]
      for j in range(num_rows):
        num_cols = data1[i].shape[1]
        fig = plt.figure()
        title = "frame "+str(j)
        ref_array = data1[i][j]
        val_array = data2[i][j]
        ref_label = uttname1[i]+":"+name1
        val_label = uttname2[i]+":"+name2
        # work around matplotlib weirdness
        if ref_label[0] == "_":
          ref_label = ref_label.replace("_","",1)
        if val_label[0] == "_":
          val_label = val_label.replace("_","",1)
        fig.suptitle(title)
        plt.xlabel("sample")
        plt.ylabel("value")
        plt.plot(range(num_cols),ref_array,label=ref_label)
        plt.plot(range(num_cols),val_array,label=val_label)
        plt.legend(ncol=2,loc='lower right')
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()
        sum_squared_diff = 0
        max_abs_diff = 0
        for k in range(num_cols):
            ref_val = ref_array[k]
            val = val_array[k]
            sum_squared_diff += (val-ref_val)*(val-ref_val)
            abs_diff = abs(val-ref_val)
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff
        rmse = math.sqrt(sum_squared_diff/num_cols)
        print("Frame","%3d" % j,"RMSE =","%.3e" % rmse,"MaxAbsDiff","%.3e" % max_abs_diff)
  elif (len(uttname1) > 0):
    for i in range(len(uttname1)):
      num_rows = data1[i].shape[0]
      for j in range(num_rows):
        num_cols = data1[i].shape[1]
        fig = plt.figure()
        title = "frame "+str(j)
        val_label = uttname1[i]+name1
        fig.suptitle(title)
        plt.xlabel("sample")
        plt.ylabel("value")
        plt.plot(range(num_cols),data1[i][j], label=val_label)
        plt.legend(ncol=1,loc='lower right')
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()
    
ark1_name = ""
ark2_name = ""

if len(sys.argv) == 1:
  print("Usage:  plotark.py ark1_name <ark2_name>")
  exit(-1)
if len(sys.argv) > 1:
  ark1_name = sys.argv[1]
  ark1_spec = "ark:"+ark1_name
  with ReadHelper(ark1_spec) as reader:
    n1 = []
    d1 = []
    for key,numpy_array in reader:
      n1.append(key)
      d1.append(numpy_array)
if len(sys.argv) > 2:
  ark2_name = sys.argv[2]
  ark2_spec = "ark:"+ark2_name
  with ReadHelper(ark2_spec) as reader:
    n2 = []
    d2 = []
    for key,numpy_array in reader:
      n2.append(key)
      d2.append(numpy_array)
  f1 = os.path.splitext(os.path.basename(ark1_name))[0]
  f2 = os.path.splitext(os.path.basename(ark2_name))[0]
  plot_arrays(f1,n1,d1,f2,n2,d2)
else:
  f1 = os.path.splitext(os.path.basename(ark1_name))[0]
  plot_arrays(os.path.basename(ark1_name),n1,d1,"",[],[])
  
