import torch
import torch.nn as nn

# For importing the configuration from config.yaml


class block(nn.Module):
    def __init__(self, dims: tuple):
        self.conv1 = nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dims[1], dims[2], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, inp):
        out = self.conv2(self.relu1(self.conv1(inp)))
        return self.relu2(out+inp)
        


class stm(nn.Module):
    def __init__(self, block_num: int, block_dim: tuple):
        self.net = nn.ModuleList()
        for _ in range(block_num):
            self.net.append(block(block_dim))

        self.policy_head = None
        self.value_head = nn.Linear()
        
    def forward(self, inp):
        out = self.net(inp)
        pass
    
