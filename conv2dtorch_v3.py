import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

batch = 1
in_height = 32
in_width = 8
in_channels = 64

filter_height = 6
filter_width = 1  # width dimension moved/combined to input channel dimension
filter_in_channels = int(in_width*in_channels/4)  # split into 4 parts
out_channels = 64
convolution_stride = (1,1) # after flattening

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()

        self.conv0 = nn.Conv2d(filter_in_channels, out_channels, (filter_height, filter_width), (1, 1), (0, 0), bias=False)
        self.conv1 = nn.Conv2d(filter_in_channels, out_channels, (filter_height, filter_width), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(filter_in_channels, out_channels, (filter_height, filter_width), (1, 1), (0, 0), bias=False)
        self.conv3 = nn.Conv2d(filter_in_channels, out_channels, (filter_height, filter_width), (1, 1), (0, 0), bias=False)

        self._initialize_weights()

    def forward(self, x):
        x = torch.reshape(x, (batch, in_height, in_width, in_channels)) # input is really NHWC
        x = torch.reshape(x, (int(batch * in_height * in_width * in_channels / 4), 4)) # reshape to NHWC/4 x 4
        # split channelwise into channels (0,4,8,...,60), (1,5,9,...,61), (2,6,10,...,62), (3,7,11,...,63)
        x = torch.transpose(x, 0, 1)
        x0, x1, x2, x3 = torch.split(x, 1, 0)
        x0 = torch.reshape(x0, (batch, in_height, 1, int(in_channels*in_width/4)))
        x1 = torch.reshape(x1, (batch, in_height, 1, int(in_channels*in_width/4)))
        x2 = torch.reshape(x2, (batch, in_height, 1, int(in_channels*in_width/4)))
        x3 = torch.reshape(x3, (batch, in_height, 1, int(in_channels*in_width/4)))
        x0 = x0.permute(0,3,1,2)
        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3,1,2)
        x3 = x3.permute(0,3,1,2)
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x0 = x0.permute(0,2,3,1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x3 = x3.permute(0,2,3,1)
        y0 = torch.add(x0, x1)
        y1 = torch.add(x2, x3)
        y2 = torch.add(y0, y1)
        y = torch.sigmoid(y2)
        y = torch.reshape(y, (1, (in_height-filter_height+1)*out_channels))
        return y

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight)

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(1, batch*in_channels*in_height*in_width, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "conv2dtorch_v3.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

