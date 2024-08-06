import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

in_batch = 1
in_channels = 1
in_height = 1
in_width = 32*5*236
out_batch = 1
out_channels = 8
out_height = 5
out_width = 236

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(in_channels, 8, (1, 32), (1, 32), (0, 0))

        self._initialize_weights()

    def forward(self, x):
        x = torch.reshape(x, (in_batch,in_channels,in_height,in_width))
        x = self.relu(self.conv1(x))
        x = torch.reshape(x, (out_batch, out_channels, out_height, out_width))
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
                  "conv1dtorch.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

