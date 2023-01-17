import torch
import torch.nn as nn


# Should add the configuration file setting from hydra configuration

class block(nn.Module):
    def __init__(self, dims: tuple):
        super().__init__()
        self.conv1 = nn.Conv2d(dims[0], dims[1], kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(dims[1])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dims[1], dims[2], kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(dims[2])
        self.conv3 = nn.Conv2d(dims[0], dims[2], kernel_size=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, inp):
        out = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(inp)))))
        
        out = out + self.conv3(inp)
        out = self.relu2(out)
        return out


class stm(nn.Module):
    def __init__(self, input_shape, block_num: int,
                 block_dims: list, value_dim: int):
        super().__init__()
        self.net = nn.ModuleList()
        
        for i in range(block_num):
            self.net.append(block(block_dims[i]))

        self.policy_head = nn.Softmax(dim=1)
        self.value_head = nn.Sequential(
            nn.Linear(block_dims[-1][2]*input_shape[1]*input_shape[2], value_dim),
            nn.Linear(value_dim, 1),
            nn.Tanh()
            )
        
    def forward(self, inp):
        out = inp
        for net in self.net:
            out = net(inp)
        out_for_v = torch.flatten(out)
        
        return self.policy_head(out),self.value_head(out_for_v)
    


if __name__ == '__main__':
    device = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    print(f'device is: {device}')
    
    rand_t = torch.randn(1,1,19,19).to(device)
    ex_block_dim = (1,10,30)
    block_t = block(ex_block_dim).to(device)
    print(f'Input shape of randomly generated data is: {rand_t.size()}')
    print(f'Output shape of residual block has shape {ex_block_dim} is {block_t(rand_t).size()}')

    stm = stm((1,19,19), 1, [ex_block_dim], 256).to(device)
    print(stm(rand_t))
