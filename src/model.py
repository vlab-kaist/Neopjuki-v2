import torch
import torch.nn as nn
import torch.nn.functional as F



class block(nn.Module):
    def __init__(self, dims: tuple):
        super().__init__()
        self.conv1 = nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dims[1], dims[2], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, inp):
        out = self.conv2(self.relu1(self.conv1(inp)))
        return self.relu2(out+inp)
        


class stm(nn.Module):
    def __init__(self, block_num: int, block_dims: list[tuple]):
        self.net = nn.ModuleList()
        for i in range(block_num):
            self.net.append(block(block_dims[i]))

        self.policy_head = None
        self.value_head = nn.Linear()
        
    def forward(self, inp):
        out = self.net(inp)
        out_for_v = F.flatten(out)

        return self.policy_head(out), self.value_head(out_for_v) # Will not work until set
    


if __name__ == '__main__':
    device = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    print(f'device is: {device}')
    
    rand_t = torch.randn(1,1,19,19).to(device)
    ex_block_dim = (1,10,30)
    Block1 = block(ex_block_dim).to(device)
    print(f'Input shape of randomly generated data is: {rand_t.size()}')
    print(f'Output shape of residual block has shape {ex_block_dim} is {Block1(rand_t).size()}')
