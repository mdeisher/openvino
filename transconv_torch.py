import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

N = 1
C = 32
H = 8
W = 1
Co = 32
G = 1
pad = (1,0)
out_pad = (1,0)
kernel = (3,1)
stride = (3,1)

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()
        self.conv1 = nn.ConvTranspose2d(C, Co, kernel, stride, pad, out_pad, G, bias=False, dilation=1, padding_mode='zeros')
        self._initialize_weights()

    def forward(self, x):
        x = torch.reshape(x, (N, H, W, C))
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = x.permute(0,2,3,1)
        Q = H + (H-1) * (stride[0] - 1) + kernel[0]
        y = torch.reshape(x, (1, (Co * (Q - kernel[0] + 1))))
        return y

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight)

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.range(1, N*C*H*W, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "transconv_torch.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

