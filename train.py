import os

import torch

import wandb

from omegaconf import OmegaConf


wandb_mode = "online"
wandb_api_key = ""


os.environ["WANDB_MODE"] = wandb_mode
os.environ["WANDB_API_KEY"] = wandb_api_key




class neopjuki_v2(object):
    def __init__(self):
        pass

    def train(self):
        pass

    def inference(self):
        pass
