import os
import numpy as np
from numba import jit
from tqdm import tqdm
from preprocess import preprocessor
from fights.envs.puoribor import PuoriborEnv

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

@jit(forceobj=True)
def remapping_action(x):
    x = list(x)
    if x[0] == 0: # move
        pass
    elif x[0] == 1: # place horizontal wall
        x[2] -= 1
    elif x[0] == 2: # place vertical wall
        x[1] -= 1
    elif x[0] == 3: # rotate section
        pass
    else:
        raise

    return tuple(x)

def generate_puribor(lines):
    game_list = list() 

    for (n, line) in enumerate(lines):
        if line[0] == "#": 
            continue

        state = PuoriborEnv().initialize_state()
        raw_actions = list(map(lambda x: tuple(map(lambda x: int(x), x.split(", "))), line[2:-3].split("), (")))
        actions = map(remapping_action, raw_actions)

        try:
            for (iter, action) in enumerate(actions):
                preprocessed = preprocessor(state, iter % 2)
                game_list.append((preprocessed, action))
                state = PuoriborEnv().step(state, iter % 2, action)

        except:
            print(f"#{n} is ignored.")
            continue

    return game_list


def data(path):
    file_list = os.listdir(path)
    xdata = []; ydata = []
    for fil in tqdm(file_list):
        dlist = generate_puribor(open(path+fil).readlines())
        for i, data in enumerate(dlist):
            xdata.append(data[0])
            ydata.append(data[1])
    return xdata, ydata

class PuriborDataset(Dataset):
    def __init__(self, filepath="../data/"):

        npz = "cache.npz"
        cpath = "../cache/"
        
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        if os.path.exists(cpath+npz):
            db = np.load(cpath+npz, allow_pickle=True)
            self.x_data = db["xdata"]
            self.y_data = db["ydata"]
        else:
            self.x_data, self.y_data = data(path=filepath)
            np.savez_compressed(cpath+npz, xdata=np.array(self.x_data), ydata=np.array(self.y_data))
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return list(x), list(y)

if __name__ == '__main__':
    PuriborDataset()
