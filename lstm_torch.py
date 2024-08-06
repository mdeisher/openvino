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
        self.lstm1 = nn.LSTMCell(in_width, n_cells, bias=True)
        self.lstm2 = nn.LSTMCell(n_cells, n_cells, bias=True)
        self.lstm3 = nn.LSTMCell(n_cells, n_cells, bias=True)
        self.lstm4 = nn.LSTMCell(n_cells, n_cells, bias=True)
        self.lstm5 = nn.LSTMCell(n_cells, n_cells, bias=True)
        self.lstm6 = nn.LSTMCell(n_cells, n_cells, bias=True)
        self._initialize_weights()

    def forward(self, x,y1,s1,y2,s2,y3,s3,y4,s4,y5,s5,y6,s6):
        x = torch.reshape(x, (batch, in_width))
        y1,s1 = self.lstm1(x,(y1,s1))
        y2,s2 = self.lstm2(y1,(y2,s2))
        y3,s3 = self.lstm3(y2,(y3,s3))
        y4,s4 = self.lstm4(y3,(y4,s4))
        y5,s5 = self.lstm5(y4,(y5,s5))
        y6,s6 = self.lstm6(y5,(y6,s6))
        return (y1,s1,y2,s2,y3,s3,y4,s4,y5,s5,y6,s6)

    def _initialize_weights(self):
        init.orthogonal_(self.lstm1.weight_ih)
        init.orthogonal_(self.lstm1.weight_hh)
        init.orthogonal_(self.lstm2.weight_ih)
        init.orthogonal_(self.lstm2.weight_hh)
        init.orthogonal_(self.lstm3.weight_ih)
        init.orthogonal_(self.lstm3.weight_hh)
        init.orthogonal_(self.lstm4.weight_ih)
        init.orthogonal_(self.lstm4.weight_hh)
        init.orthogonal_(self.lstm5.weight_ih)
        init.orthogonal_(self.lstm5.weight_hh)
        init.orthogonal_(self.lstm6.weight_ih)
        init.orthogonal_(self.lstm6.weight_hh)

# Create the model by using the above model definition.
torch_model = MyNet()

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(1, batch*in_channels*in_height*in_width, requires_grad=True)
y1 = torch.zeros(1, n_cells, requires_grad=True)
s1 = torch.zeros(1, n_cells, requires_grad=True)
y2 = torch.zeros(1, n_cells, requires_grad=True)
s2 = torch.zeros(1, n_cells, requires_grad=True)
y3 = torch.zeros(1, n_cells, requires_grad=True)
s3 = torch.zeros(1, n_cells, requires_grad=True)
y4 = torch.zeros(1, n_cells, requires_grad=True)
s4 = torch.zeros(1, n_cells, requires_grad=True)
y5 = torch.zeros(1, n_cells, requires_grad=True)
s5 = torch.zeros(1, n_cells, requires_grad=True)
y6 = torch.zeros(1, n_cells, requires_grad=True)
s6 = torch.zeros(1, n_cells, requires_grad=True)
(y1,s1,y2,s2,y3,s3,y4,s4,y5,s5,y6,s6) = torch_model(x,y1,s1,y2,s2,y3,s3,y4,s4,y5,s5,y6,s6)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  (x,y1,s1,y2,s2,y3,s3,y4,s4,y5,s5,y6,s6), # model input (or a tuple for multiple inputs)
                  "lstm_torch.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input','y1i','s1i','y2i','s2i','y3i','s3i','y4i','s4i','y5i','s5i','y6i','s6i'],   # input names
                  output_names = ['y1o','s1o','y2o','s2o','y3o','s3o','y4o','s4o','y5o','s5o','output','s6o'],   # output names
                  )

