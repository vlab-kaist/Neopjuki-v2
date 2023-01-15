import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(self,in_dim, mid_dim, out_dim):
        self.conv1 = nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, inp):
        out = self.conv2(self.relu1(self.conv1(inp)))
        return self.relu2(out+inp)
        




class stm(nn.Module):
    def __init__(self,block_num=5,):
        pass
    def forward(self, inp):
        pass
