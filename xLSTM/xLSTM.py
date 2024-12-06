import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.mLSTMblock import mLSTMblock
from xLSTM.sLSTMblock import sLSTMblock

class xLSTM(nn.Module):
    def __init__(self, layers, x_example, kernel_size, depth=9, factor=2):
        super(xLSTM, self).__init__()
        self.relu = nn.ReLU()#
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
            self.linear = nn.Linear(26, 26)
    
    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
        
    def forward(self, x):
        x_original = x.clone()
        for l in self.layers:
            one_hot = x_original[:, :, :-1]  # Primi C1 canali (one-hot encoding)
            timestamp = x_original[:, :, :11]  # Canali successivi (timestamp)
            one_hot_interpolated = F.interpolate(one_hot, size=x_original.shape[2] , mode='nearest')
            timestamp_interpolated = F.interpolate(timestamp, size=x_original.shape[2] , mode='linear', align_corners=True)
            x_original = torch.cat([one_hot_interpolated, timestamp_interpolated], dim=2)
            projector = nn.Linear(x_original.shape[2], l(x).shape[2]).to(self.device)
            x_original = projector(x_original)
            x = l(x) + x_original
        x = self.relu(x)
        x = self.linear(x[:,-1,:])
        self.relu(x)
        return x