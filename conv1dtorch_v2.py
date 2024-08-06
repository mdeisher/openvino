import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

in_N = 1
in_C = 240
const_N = 1
const_C = 1
const_H = 4
const_W = 240
conv_N = 1
conv_C = 1
conv_H = 240
conv_W = 5
kernel_H = 1
kernel_W = 5
out_batch = 1
add_N = 1
add_C = 32
add_H = 1
add_W = 240

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv2d(conv_C, add_C, (kernel_H, kernel_W), (1, 1), (0, 0))
        self.relu = nn.ReLU(inplace=inplace)

        self._initialize_weights()

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        zeros = torch.zeros(const_N, const_C, const_H, const_W)
        x = torch.cat((zeros, x), 2)
        x = torch.transpose(x, 2, 3)
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.transpose(x, 1, 3)
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(in_N, in_C, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "conv1dtorch_v2.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

