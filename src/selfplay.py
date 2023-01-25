import h5py
import torch
import numpy as np
from tqdm import tqdm
from model import stm
from mcts import MCTS
from omegaconf import OmegaConf
from fights.envs import PuoriborEnv
from preprocess import preprocessor
from preprocess import hashing_state
conf = OmegaConf.load("../config.yaml")

snum = conf['mcts']['sim_num']

stmp = stm(input_shape=(conf['stm']['input_channel'], conf['env']['board_size'], conf['env']['board_size']),
           input_channel=conf['stm']['input_channel'], p_output_channel=conf['stm']['output_channel'],
           filters=conf['stm']['filters'], block_num=conf['stm']['block_num'], value_dim=conf['stm']['value_dim'])


state_dict = torch.load("../../pretrained.pt")
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
        
stmp.load_state_dict(new_state_dict)


stmp.eval()




envs = PuoriborEnv()


state = envs.initialize_state()
tr = MCTS(stmp, conf['mcts']['temp'])
current = tr.root

t = 0
while state.done == False:
    nodes = []
    for i in tqdm(range(snum)):
        leaf = tr.select()
        prep_states, nodes = tr.expand(leaf)
        for node in nodes:
            _, val = tr.stm(torch.Tensor(preprocessor(node.state, (node.turn+1)%2)).unsqueeze(0))
            if val >= 0:
                tr.backpropagate(node, 1)
            elif val < 0:
                tr.backpropagate(node, -1)

    pi_t = np.zeros((324,))
    for addr in current.childs:
        node = current.childs[addr]
        pi_t[node.action[0]*81 + node.action[1]*9 + node.action[2]] += node.visits

    pi_t = pi_t/current.visits

    print(current.state)
    
    # Need to obtain whole pi and val (the target)
    action = tr.decide(current)
    
    
    state = envs.step(state, current.turn, action)
    
    current = current.childs[hashing_state(state, (current.turn+1)%2)]
    
    tr.root = current
    if sum(state.walls_remaining) == 0:
        t += 1
        if t > 50:
            break
