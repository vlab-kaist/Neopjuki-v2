import os
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
from src.model import stm
from torch.optim import Adam
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.dataset import SupervisedDataset
from torch.utils.data import random_split

#wandb.init(project="pretrain_neopjuki-v2", entity="vlab-kaist")

conf = OmegaConf.load("config.yaml")

wandb.config = conf

pretraining_conf = conf['pretraining']

stm_conf = conf['stm']

ds = SupervisedDataset('data/')

train_size = int(0.8*len(ds))
test_size = len(ds) - train_size

train_ds, test_ds = random_split(ds, [train_size, test_size])

trainloader = DataLoader(train_ds, batch_size=pretraining_conf['batch_size'], shuffle=True, num_workers=8)
testloader = DataLoader(test_ds, batch_size=pretraining_conf['batch_size'], shuffle=False, num_workers=8)






device = torch.device("cuda:0")

stm = stm(input_shape=(stm_conf['input_channel'], conf['env']['board_size'], conf['env']['board_size']),
          input_channel=stm_conf['input_channel'], p_output_channel=stm_conf['output_channel'],
          filters=stm_conf['filters'], block_num=stm_conf['block_num'], value_dim=stm_conf['value_dim']).to(device)

optim = Adam(stm.parameters(), lr=pretraining_conf['lr'])
kl_loss = nn.KLDivLoss(reduction="batchmean")

epochs = pretraining_conf['epochs']


for epoch in range(epochs):
    valid_loss=0
    stm.train()
    for batch_num, (state, action) in enumerate(trainloader):
        optim.zero_grad()
        state = state.to(device)
        target = action.to(device)
        output = stm(state)
        
        loss = kl_loss(output[0], target)
        print(f' percentage:{(batch_num/len(trainloader))*100:.2f}%, loss:{loss.item()}')
        loss.backward()
        optim.step()

    for batch_num, (state, action) in tqdm(enumerate(testloader)):
        stm.eval()
        state = state.to(device)
        target = action.to(device)
        output = stm(state)
        
        loss = kl_loss(output[0], target)
        valid_loss += loss.item()
        print(f'percentage: {(batch_num/len(testloader))*100:.2f}%')
    
    print(f'average loss on epoch:{epoch}, loss:{valid_loss / len(trainloader)}')
        
        










