import os
import h5py
import torch
import numpy as np
from model import stm
from mcts import MCTS
from omegaconf import OmegaConf
from fights.envs import PuoriborEnv
from preprocess import preprocessor
from collections import OrderedDict
from preprocess import hashing_state



def load_model(conf, path="../../"):
    stmp = stm(input_shape=(conf['stm']['input_channel'], conf['env']['board_size'], conf['env']['board_size']),
           input_channel=conf['stm']['input_channel'], p_output_channel=conf['stm']['output_channel'],
           filters=conf['stm']['filters'], block_num=conf['stm']['block_num'], value_dim=conf['stm']['value_dim'])
    
    state_dict = torch.load(path+"pretrained.pt")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    stmp.load_state_dict(new_state_dict)
    stmp.eval()

    return stmp


def load_hf(cache_name):
    if not os.path.exists(cache_name):
        hf = h5py.File(cache_name, 'w')
        hf.create_dataset("state", shape=(0, 25, 9, 9), maxshape=(None, 25, 9, 9), dtype=np.float32, compression='lzf')
        hf.create_dataset("policy", shape=(0, 324), maxshape=(None, 324), dtype=np.float32, compression='lzf')
        hf.create_dataset("value", shape=(0,1), maxshape=(None,1), dtype=np.int8, compression='lzf')
        return hf
    else:
        hf = h5py.File(cache_name, 'a')
        return hf


def selfplay(stmp, play_num, snum, cache_name):
        
    envs = PuoriborEnv()
    
    for i in range(play_num):
        hf = load_hf(cache_name)
        print(f'the game number is: {i}')
        state = envs.initialize_state()
        tr = MCTS(stmp, conf['mcts']['temp'])
        current = tr.root

        s_ts = []
        pi_ts = []

        while state.done == False:
            nodes = []
            for i in range(snum):
                leaf = tr.select()
                prep_states, nodes = tr.expand(leaf)
                for node in nodes:
                    if node.state.done == True:
                        tr.backpropagate(node, 1)
                    else:
                        _, val = tr.stm(torch.Tensor(preprocessor(node.state, (node.turn+1)%2)).unsqueeze(0).to(tr.dev))
                        if val >= 0:
                            tr.backpropagate(node, 1)
                        elif val < 0:
                            tr.backpropagate(node, -1)

            pi = np.zeros((324,))
            for addr in current.childs:
                node = current.childs[addr]
                pi[node.action[0]*81 + node.action[1]*9 + node.action[2]] += node.visits

            s_ts.append(preprocessor(state, current.turn))
            pi_ts.append(pi/current.visits)
            
            action = tr.decide(current)
            state = envs.step(state, current.turn, action)
            current = current.childs[hashing_state(state, (current.turn+1)%2)]
            tr.root = current
            if state.done == True:
                s_t = np.stack(s_ts, axis=0)
                pi_t = np.stack(pi_ts, axis=0)
                z_t = np.array([[(-1)**((current.turn+i)%2)] for i in range(len(s_ts))])
                orig_length = hf["state"].shape[0]
                hf["state"].resize(orig_length + s_t.shape[0], axis=0)
                hf["policy"].resize(orig_length + pi_t.shape[0], axis=0)
                hf["value"].resize(orig_length + z_t.shape[0], axis=0)

                hf["state"][orig_length:orig_length + s_t.shape[0]] = s_t
                hf["policy"][orig_length:orig_length + pi_t.shape[0]] = pi_t
                hf["value"][orig_length:orig_length + z_t.shape[0]] = z_t
                hf.close()
            



if __name__ == '__main__':
    conf = OmegaConf.load("../config.yaml")
    snum = conf['mcts']['sim_num']
    stmp = load_model(conf)

    states, pis, zs = selfplay(stmp, 2, snum, "../../plays/cache.h5")

    for state in states:
        print(state.shape)

    for pi in pis:
        print(pi.shape)

    for z in zs:
        print(z.shape)
