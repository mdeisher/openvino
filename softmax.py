import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# This script demonstrates how to create an equivalent softmax operation using
# only integer operators that are supported on GNA.

in_batch = 1
in_channels = 1
in_height = 1
in_width = 128

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()

    def forward(self, x):
        # Kronecker kernel to simply make a copy of the input tensor.
        #    This is needed because GNA max pool operator is fused with
        #    the convolution operator and cannot be used apart from it.
        kernel = torch.Tensor([
            [[1, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 1, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 1, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 1, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 1]]])
        # This first sub-graph is to create a 128-element vector with the
        # maximum value of the input tensor in each element.
        y = torch.reshape(x, (1, 1, in_width))
        y = F.max_pool1d(F.conv1d(y, kernel, stride=8, padding=0), 4, 4, 0, 1)
        y = torch.reshape(y, (8, int(in_width/8/4)))
        y = torch.transpose(y, 0, 1)
        y = torch.reshape(y, (1, 1, int(in_width/4)))
        y = torch.cat((y, y, y, y), 2)
        y = F.max_pool1d(F.conv1d(y, kernel, stride=8,padding=0), 4, 4, 0, 1)
        y = torch.reshape(y, (1, 1, int(in_width/4)))
        y = torch.cat((y, y, y, y), 2)
        y = F.max_pool1d(F.conv1d(y, kernel, stride=8,padding=0), 4, 4, 0, 1)
        y = torch.reshape(y, (1, 1, int(in_width/4)))
        y = torch.cat((y, y, y, y), 2)
        y = F.max_pool1d(F.conv1d(y, kernel, stride=8,padding=0), 4, 4, 0, 1)
        y = torch.reshape(y, (1, int(in_width/4)))
        y = torch.cat((y, y, y, y), 1)
        # The rest of the softmax operation
        diff = torch.sub(x,y) # vector minus its maximum element achieves exp range (0,1]
        exp_diff = torch.reshape(torch.exp(diff), (in_width, 1))
        ones = torch.ones(in_width, in_width) # this should be split in two to save compute, also consider grouping
        sum_exp_diff = torch.matmul(ones,exp_diff) # produces sum of exp(x-c) elements
        log_sum_exp_diff = torch.log(sum_exp_diff)
        diff = torch.sub(diff,log_sum_exp_diff)
        x = torch.exp(diff)
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(1, in_batch*in_channels*in_height*in_width, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "softmax.onnx",            # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

