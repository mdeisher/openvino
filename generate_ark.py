#*****************************************************************************
# 
# INTEL CONFIDENTIAL
# Copyright 2018-2019 Intel Corporation
# 
# The source code contained or described herein and all documents related 
# to the source code ("Material") are owned by Intel Corporation or its suppliers 
# or licensors. Title to the Material remains with Intel Corporation or its suppliers 
# and licensors. The Material contains trade secrets and proprietary 
# and confidential information of Intel or its suppliers and licensors.
# The Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified, 
# published, uploaded, posted, transmitted, distributed, or disclosed in any way 
# without Intel's prior express written permission.
# 
# No license under any patent, copyright, trade secret or other intellectual 
# property right is granted to or conferred upon you by disclosure or delivery 
# of the Materials, either expressly, by implication, inducement, estoppel 
# or otherwise. Any license under such intellectual property rights must 
# be express and approved by Intel in writing.
#*****************************************************************************
# generate_ark.py : generates ARK files with random data
#
import csv
import struct
import numpy as np
import copy
from kaldiio import WriteHelper
import sys

def make_random_ark(nummat,numrows, numcolumns, arkname, stdev):
    fspec = "ark:"+arkname
    with WriteHelper(fspec) as writer:
        for i in range(nummat):
            uttname = "utt"+str(i)
            data = np.random.normal(0,stdev,(numrows,numcolumns))
            data = np.single(data)
            writer(uttname, data)

def make_zero_ark(nummat,numrows, numcolumns, arkname):
    fspec = "ark:"+arkname
    with WriteHelper(fspec) as writer:
        for i in range(nummat):
            uttname = "utt"+str(i)
            data = np.zeros((numrows,numcolumns))
            data = np.single(data)
            writer(uttname, data)

def make_inc_ark(nummat,numrows, numcolumns, arkname):
    fspec = "ark:"+arkname
    with WriteHelper(fspec) as writer:
        for i in range(nummat):
            uttname = "utt"+str(i)
            data = np.zeros([numrows,numcolumns], dtype=np.float32)
            for j in range(numrows):
                data[j] = np.arange(1,numcolumns+1,1).astype(np.float32)
            data = np.single(data)
            writer(uttname, data)

testlen = 10
numtest = 1

if len(sys.argv) < 3:
  print("Usage:  generate_arkplotark.py ark_name size <test_length> <num_test>")
  exit(-1)
if len(sys.argv) > 2:
  ark_name = sys.argv[1]
  size = int(sys.argv[2])
if len(sys.argv) > 3:
  testlen = int(sys.argv[3])
if len(sys.argv) > 4:
  numtest = int(sys.argv[4])

make_random_ark(numtest,testlen,size,ark_name,5)
  
#make_random_ark(1,10,1*1*320,'layernorm_input.ark',5)
#make_inc_ark(1,1,16,'zinput.ark')

