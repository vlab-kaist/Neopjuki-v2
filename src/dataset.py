import os
import ray
import h5py
import numpy as np
from preprocess import generate_cache
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from fights.envs import PuoriborEnv


class SupervisedDataset(Dataset):
    def __init__(self, cpath="../data/"):

        h5 = "cache.h5"
        
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        if os.path.exists(cpath+h5):
            hf = h5py.File(cpath+h5, 'r')
            self.x_data = hf["xdata"]
            self.y_data = hf["ydata"]
        else:
            self.x_data, self.y_data = generate_cache(cpath)
            hf = h5py.File(cpath+h5, 'w')
            hf.create_dataset("xdata", shape=(0, 25, 9, 9), maxshape=(None, 25, 9, 9), dtype=np.float32, compression='lzf')
            hf.create_dataset("ydata", shape=(0, 324), maxshape=(None, 324), dtype=np.float32, compression='lzf')

            orig_length = hf["xdata"].shape[0]
            hf["xdata"].resize(orig_length + len(self.x_data), axis=0)
            hf["ydata"].resize(orig_length + len(self.y_data), axis=0)
            
            hf["xdata"][orig_length:orig_length + len(self.x_data)] = self.x_data
            hf["ydata"][orig_length:orig_length + len(self.y_data)] = self.y_data
            hf.close()
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

if __name__ == '__main__':
    pdataset = SupervisedDataset("../../")
    traindataloader = DataLoader(pdataset)
    for i, (state, action) in enumerate(traindataloader):
        print(state)
        print(action)
            
        
        breakpoint()
