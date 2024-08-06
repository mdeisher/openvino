import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

embed_dim = 68
num_heads = 4
source_seq_len = 64
target_seq_len = 64

class MyNet(nn.Module):
    def __init__(self, inplace=False):
        super(MyNet, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads) #,kdim=kdim,vdim=vdim)
        self._initialize_weights()

    def forward(self, QKV):
        qkv = torch.reshape(QKV, (target_seq_len, embed_dim))
        q = qkv
        k = qkv
        v = qkv
        y,w = self.mha(q,k,v)
        return (y ,w)

    def _initialize_weights(self):
        if self.mha._qkv_same_embed_dim:
            init.orthogonal_(self.mha.in_proj_weight)
        else:
            init.orthogonal_(self.mha.q_proj_weight)
            init.orthogonal_(self.mha.k_proj_weight)
            init.orthogonal_(self.mha.v_proj_weight)

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
QKV = torch.randn(1, target_seq_len * embed_dim, requires_grad=True)
(y,w) = torch_model(QKV)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  (QKV), # model input (or a tuple for multiple inputs)
                  "mhsa_torch.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['qkv'],   # input names
                  output_names = ['y','w'],   # output names
                  )

