import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

batch = 1
in_height = 5248
in_width = 1
in_channels = 16

filter_height = 11
filter_width = 1
out_channels = 16
cstride = (1,1)
cdilation = (5,1)
cpadding = (25,0)

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, (filter_height, filter_width), cstride, cpadding, cdilation, bias=False)

        self._initialize_weights()

    def forward(self, x):
        x = torch.reshape(x, (batch, in_channels, in_height, in_width))
        y = self.conv0(x)
        return y

    def _initialize_weights(self):
        init.orthogonal_(self.conv0.weight)

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
                  "conv_dilat.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

