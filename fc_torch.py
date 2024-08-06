import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

batch = 1
in_height = 1
in_width = 1024
in_channels = 1
n_cells = 512

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()
        self.weights = torch.randn((n_cells,in_width), dtype=torch.float32)
        self.bias = torch.randn(n_cells, dtype=torch.float32)

    def forward(self, x):
        x = torch.reshape(x,(in_width,1))
        x = torch.matmul(self.weights,x)
        x = torch.squeeze(x)
        x = torch.add(self.bias,x)
        y = torch.sigmoid(x)
        return y

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(batch*in_channels*in_height*in_width, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                  # model input (or a tuple for multiple inputs)
                  "fc_torch.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

