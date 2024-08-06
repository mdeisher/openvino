import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

N = 1
C = 1
H = 4
W = 1024
num_parts = 2  # when W>768 must split into equal parts for GNA
epsilon = 1e-05
normalize_variance = True

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()
        # constant tensors
        self.neg_avg_weights = torch.ones(8,int(W/num_parts),1,1) * (-1/W)
        self.avg_weights = torch.ones(8,int(W/num_parts),1,1) * (1/W)
        self.avg_broadcast = torch.zeros(W,8*num_parts,1,1)
        self.minus_half = torch.ones(1,H*W) * (-0.5)
        self.eps_tensor = torch.ones(1,H*W) * epsilon
        for i in range(W):
            self.avg_broadcast[i,0,0,0] = 1.0
            self.avg_broadcast[i,8,0,0] = 1.0

    def forward(self, x):
        x = torch.reshape(x, (N,H*num_parts,1,int(W/num_parts)))
        # mean subtraction graph
        input_2d = torch.reshape(x, (1,N*C*H*W))
        x = x.permute(0,3,1,2)
        x = F.conv2d(x, self.neg_avg_weights)
        x = x.permute(0,2,3,1)
        x = torch.reshape(x, (N,1,H,8*num_parts))
        x = x.permute(0,3,1,2)
        x = F.conv2d(x, self.avg_broadcast)
        x = x.permute(0,2,3,1)
        x = torch.reshape(x, (1,H*W))
        x_minus_mean = torch.add(x, input_2d)  # subtract mean
        y = torch.reshape(x_minus_mean, (1, N*C*H*W))

        if normalize_variance:
            x = torch.mul(x_minus_mean,x_minus_mean)  # (x-mean)^2
            x = torch.reshape(x, (N,H*num_parts,1,int(W/num_parts)))
            x = x.permute(0,3,1,2)
            x = F.conv2d(x, self.avg_weights)
            x = x.permute(0,2,3,1)
            x = torch.reshape(x, (N,1,H,8*num_parts))
            x = x.permute(0,3,1,2)
            x = F.conv2d(x, self.avg_broadcast)
            x = x.permute(0,2,3,1)
            x = torch.reshape(x, (1,H*W))
            x = torch.add(x, self.eps_tensor)
            x = torch.log(x)
            x = torch.mul(x, self.minus_half)
            x = torch.exp(x)
            x = torch.mul(x, x_minus_mean)
            y = torch.reshape(x, (1, N*C*H*W))
            
        return y

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight)

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(1, N*C*H*W, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "layer_norm_torch_gna.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

