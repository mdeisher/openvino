import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

N = 1
C = 1
H = 4
W = 1024
epsilon = 1e-05

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()
        self.norm = nn.LayerNorm([W],eps=epsilon)

    def forward(self, x):
        x = torch.reshape(x, (N, C, H, W))
        x = self.norm(x)
        y = torch.reshape(x, (1, N*C*H*W))
        return y

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn((1, N*C*H*W), requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "layer_norm_torch.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

