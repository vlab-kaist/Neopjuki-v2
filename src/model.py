import torch
import torch.nn as nn


# Should add the configuration file setting from hydra configuration

class convblock(nn.Sequential):
    def __init__(self, input_channel, out_channels):
        super().__init__(
            nn.Conv2d(input_channel, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
class conv1x1block(nn.Sequential):
    def __init__(self, filters, p_out_channels):
        super().__init__(
            nn.Conv2d(filters, p_out_channels, kernel_size=1),
            nn.BatchNorm2d(p_out_channels),
            nn.ReLU()
        )
    
class resblock(nn.Module):
    def __init__(self, dims: tuple):
        super().__init__()
        self.conv1 = nn.Conv2d(dims[0], dims[1], kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(dims[1])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dims[1], dims[2], kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(dims[2])
        self.conv3 = nn.Conv2d(dims[2], dims[0], kernel_size=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, inp):
        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = inp + self.conv3(out)
        out = self.relu2(out)
        return out


class stm(nn.Module):
    def __init__(self, input_shape: tuple, input_channel: int, p_output_channel: int, filters: int, block_num: int, value_dim: int):
        super().__init__()

        self.net = nn.ModuleList()
        self.net.append(convblock(input_channel, filters))
        for _ in range(block_num):
            self.net.append(resblock((192, 192, 192)))

        self.net.append(conv1x1block(filters, p_output_channel))
        
        self.policy_head = nn.Softmax(dim=1)
        self.value_head = nn.Sequential(
            nn.Linear(p_output_channel*input_shape[1]*input_shape[2], value_dim),
            nn.Linear(value_dim, 1),
            nn.Tanh()
            )
        
    def forward(self, inp):
        out = inp
        for net in self.net:
            out = net(out)
        
        out_for_v = torch.flatten(out)
        
        return self.policy_head(out), self.value_head(out_for_v)
    


if __name__ == '__main__':
    device = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    print(f'device is: {device}')
    '''
    rand_t = torch.randn(1,1,19,19).to(device)
    ex_block_dim = (1,10,30)
    block_t = block(ex_block_dim).to(device)
    print(f'Input shape of randomly generated data is: {rand_t.size()}')
    print(f'Output shape of residual block has shape {ex_block_dim} is {block_t(rand_t).size()}')
    '''
    stm = stm(input_shape=(1,19,19), input_channel=1, p_output_channel=4, filters=192, block_num=5, value_dim=256).to(device)
    rand_stm = torch.randn(1,1,19,19).to(device)
    print(stm(rand_stm))
