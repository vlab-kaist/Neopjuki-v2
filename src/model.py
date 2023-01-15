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
    def __init__(self, input_shape: tuple, block_num: int,
                 block_dims: list[tuple], value_dim: int):
        self.net = nn.ModuleList()
        self.wall_eval = nn.Linear(2,1)
        for i in range(block_num):
            self.net.append(block(block_dims[i]))

        self.policy_head = nn.Softmax()
        self.value_head = nn.Sequential(
            nn.Linear(block_dims[-1][2], value_dim),
            nn.Linear(value_dim, 1)
            )
        
    def forward(self, inp, remaining_wall):
        out = self.net(inp)
        out_for_v = F.flatten(out)
        wall_eval = self.wall_eval(remaining_wall)
        
        return self.policy_head(wall_eval*out), self.value_head(wall_eval*out_for_v)
    


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
