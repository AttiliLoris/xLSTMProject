import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.mLSTMblock import mLSTMblock
from xLSTM.sLSTMblock import sLSTMblock

class xLSTM(nn.Module):
    def __init__(self, layers, x_example, kernel_size, depth=2, factor=2):
        super(xLSTM, self).__init__()
        self.activation = nn.ReLU()#
        self.layers = nn.ModuleList()
        self.device = x_example.device
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMblock(x_example, depth)
            elif layer_type == 'm':
                layer = mLSTMblock(x_example, factor, depth, kernel_size)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)
            self.linear = nn.Linear(x_example.shape[2], x_example.shape[2]-1)
    
    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
        
    def forward(self, x):
        x_original = x.clone()
        for l in self.layers:
            #projector = nn.Linear(x_original.shape[2], l(x).shape[2]).to(self.device)
            #x_original = projector(x_original)
            x = l(x) + x_original
        x = self.activation(x)
        x = self.linear(x[:,-1,:])
        self.activation(x)
        return x