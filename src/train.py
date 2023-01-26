import os
import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from selfplay import load_model
from omegaconf import OmegaConf
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset import SelfplayDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel as DDP


conf = OmegaConf.load("../config.yaml")
ds = SelfplayDataset('../../plays/')
train_size = int(0.8*len(ds))
test_size = len(ds) - train_size
train_ds, test_ds = random_split(ds, [train_size, test_size])

run = wandb.init(project="train_neopjuki-v2", entity="vlab-kaist", group="block"+str(conf['stm']['block_num'])+"-version_1")

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size,))
    run.finish()


def main_worker(gpu_id, world_size):
    mp_context = mp.get_context('fork')
    
    num_worker = conf['hardware']['num_cpus']
    batch_size = conf['pretrain']['batch_size']
    
    num_worker_per_node = int(num_worker / world_size)
    batch_size_per_node = int(batch_size / world_size)

    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:8888',
        world_size=world_size,
        rank=gpu_id)

    stmp = load_model(conf, "../../pretrained.pt").to(f'cuda:{gpu_id}')

    torch.cuda.set_device(gpu_id)

    stmp = DDP(stmp, device_ids=[gpu_id], output_device=gpu_id).to(gpu_id)

    
    optim = Adam(stmp.parameters(), lr=conf['train']['lr'], weight_decay=1e-5)
    ce_loss = nn.CrossEntropyLoss().to(gpu_id)
    mse_loss = nn.MSELoss().to(gpu_id)

    for epoch in range(conf['train']['epochs']):
        valid_loss=0
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds).set_epoch(epoch)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
        trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_worker_per_node,
                                 sampler=train_sampler, pin_memory=True, multiprocessing_context=mp_context)
        testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_worker_per_node,
                                sampler=test_sampler, pin_memory=True, multiprocessing_context=mp_context)

        
        stmp.train()
        for batch_num, (state, policy, value) in enumerate(trainloader):
            optim.zero_grad()
            state = state.to(gpu_id)
            policy = policy.to(gpu_id)
            value = value.to(torch.float32).to(gpu_id)
            output, val = stmp(state)
            pol_loss = ce_loss(output, policy)
            run.log({'policy_loss':pol_loss.item()})
            val_loss = mse_loss(val, value)
            run.log({'value_loss': val_loss.item()})
            loss = pol_loss + 0.01*val_loss
            print(f' percentage:{(batch_num/len(trainloader))*100:.2f}%, loss:{loss.item()}')
            run.log({'batch_loss':loss.item()})
            loss.backward()
            optim.step()

        stmp.eval()
        for batch_num, (state, policy, value) in enumerate(testloader):
            state = state.to(gpu_id)
            policy = policy.to(gpu_id)
            value = value.to(gpu_id)
            output, val = stmp(state)
            pol_loss = ce_loss(output, policy)
            val_loss = mse_loss(val, value)
            loss = pol_loss + val_loss
            valid_loss += loss.item()
            print(f'percentage: {(batch_num/len(testloader))*100:.2f}%')

        print(f'average loss on epoch:{epoch}, loss:{valid_loss / len(testloader)}')
        run.log({'valid_loss':(valid_loss/len(testloader))})

        if gpu_id == 0:
            torch.save(stmp.state_dict(), conf['train']['saving_point']+"version1_"+str(epoch)+".pt")


if __name__ == '__main__':
    mp.freeze_support()
    main()
