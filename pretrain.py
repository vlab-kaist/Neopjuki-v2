import wandb
import torch
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

ds = SupervisedDataset()

train_size = int(0.8*len(ds))
test_size = len(ds) - train_size

train_ds, test_ds = random_split(ds, [train_size, test_size])

trainloader = Dataloader(train_ds, batch_size=pretraining_conf['batch_size'], shuffle=True)
testloader = Dataloader(test_ds, batch_size=pretraining_conf['batch_size'], shuffle=False)

device = torch.device("cuda:0")

stm = stm(input_shape=(stm_conf['input_channel'], conf['env']['board_size'], conf['env']['board_size']),
          input_channel=stm_conf['input_channel'], p_output_channel=stm_conf['output_channel'],
          filters=stm_conf['filters'], block_num=stm_conf['block_num'], value_dim=stm_conf['value_dim']).to(device)

optim = Adam(stm.parameters(), lr=pretraining_conf['lr'])
kl_loss = nn.KLDivLoss(reduction="batchmean")

epochs = pretraining_conf['epochs']


for epoch in range(epochs):
    for batch_num, (state, action) in enumerate(trainloader):
        state = state.to(device)
        










