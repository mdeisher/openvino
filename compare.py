import os
import math
import matplotlib.pyplot as plt
import sys

int16_dir = ".\\layers\\0.int16"
fp32_dir = ".\\layers\\0.fp32"

for root, dirs, files in os.walk(fp32_dir):
    for file in files:
        parts = file.split(".")
        basename = parts[0]
        parts = basename.split("_")
        if len(parts) > 2:
            layer_id = parts[0]
            description = parts[1]
            member = parts[2]
            if member != "input":
                parts = description.split("-")
                operation = parts[0]
                num_outputs = int(parts[1])
                num_inputs = int(parts[2])
                g = open(os.path.join(fp32_dir,file))
                h = open(os.path.join(int16_dir,file))
                #print("Comparing",file," - ",member)
                sum_squared_diff = 0
                max_abs_diff = 0
                ref_array = []
                val_array = []
                for i in range(num_outputs):
                    ref_val = float(g.readline().rstrip())
                    val = float(h.readline().rstrip())
                    ref_array.append(ref_val)
                    val_array.append(val)
                    sum_squared_diff += (val-ref_val)*(val-ref_val)
                    abs_diff = abs(val-ref_val)
                    if abs_diff > max_abs_diff:
                        max_abs_diff = abs_diff
                g.close()
                h.close()
                rmse = math.sqrt(sum_squared_diff/num_outputs)
                print("RMSE =","%.3e" % rmse,"MaxAbsDiff","%.3e" % max_abs_diff,":",file)
                if max_abs_diff > 0.1:
                    fig = plt.figure()
                    fig.suptitle(file)
                    plt.xlabel("sample")
                    plt.ylabel("value")
                    plt.plot(range(num_outputs),ref_array,label='fp32')
                    plt.plot(range(num_outputs),val_array,label='int16')
                    plt.legend(ncol=2,loc='lower right')
                    #plt.show()
                    plt.draw()
                    plt.waitforbuttonpress(0)
                    plt.close()
                    #print("Dumping file")
                    sum_squared_diff = 0
                    for i in range(num_outputs):
                        ref_val = ref_array[i]
                        val = val_array[i]
                        sum_squared_diff += (val-ref_val)*(val-ref_val)
                        #print(i,": val = ",val,"ref_val = ",ref_val,"sum_sq_diff",sum_squared_diff)
                    #input("Pausing...")
        
exit(0)
