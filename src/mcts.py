import ray
import math
import torch
import random
from fights.envs import PuoriborEnv
from preprocess import preprocessor
from preprocess import hashing_state
from preprocess import generate_actions


class Node(object):
    def __init__(self, turn, state, action, parent):
        super().__init__()

        self.turn = turn
        self.state = state # this state should be original state.
        self.action = action
        self.address = hashing_state(state, turn)

        self.childs = {}
        self.parent = parent # Node, -1 for root
        
        self.pi = 0
        self.val = 0
        self.wins = 0
        self.visits = 1
        

class MCTS(object):
    def __init__(self, stm, temp, first=0, initial_state=None):
        super().__init__()

        
        self.stm = stm
        self.temp = temp
        self.env = PuoriborEnv()
        self.actions_cache = generate_actions((4,9,9))

        
        
        if initial_state != None:
            self.root = Node(first, initial_state, -1, -1)           
        else:
            self.root = Node(first, self.env.initialize_state(), -1, -1)

        
    def UCT(self, node):
        q_val = 0
        u_val = self.temp*node.pi*(math.sqrt(node.parent.visits)/(1+node.visits))

        if len(node.childs):
            q_val = (node.wins / node.visits)
                
        else:
            q_val = (node.parent.wins / node.parent.visits)

        return q_val + u_val

    
    def select(self):
        current = self.root
        while len(current.childs):
            keys = []
            uct_val = []
            for key, val in current.childs.items():
                uct_val.append(self.UCT(val))
                keys.append(key)
            current = current.childs[keys[torch.Tensor(uct_val).argmax().item()]]
            current.visits += 1
        return current
    
    def expand(self, node): # all possible states 
        assert len(node.childs) == 0 # Check this is the leaf node
        pol, val = self.stm(torch.Tensor(preprocessor(node.state, (node.turn+1)%2)).unsqueeze(0))

        pol = pol.flatten()
        pis = []
        vals = []
        nodes = []
        states = []
        actions = []

        prep_states = []
        
        for i, action in enumerate(self.actions_cache):
            try:
                state = self.env.step(node.state, node.turn, action)
                states.append(state)
                actions.append(action)
                pis.append(pol[i])
                _, val = self.stm(torch.Tensor(preprocessor(state, node.turn)).unsqueeze(0))
                if val.item() <= 0:
                    vals.append(-1)
                elif val.item() > 0:
                    vals.append(1)

            except ValueError:
                pass
        
        for i, state in enumerate(states):
            new_node = Node((node.turn+1)%2, state, actions[i], node)
            new_node.pi = pis[i]
            new_node.val = vals[i]
            node.childs[new_node.address] = new_node
            nodes.append(new_node)
            prep_states.append((state, new_node.turn))
        return prep_states, nodes

    def simulate(self, preps):
        starting_point = preps[1]
        state = preps[0]
        turn = preps[1]

        t=0
        while state.done == False:
            pol, val = self.stm(torch.Tensor(preprocessor(state, turn)).unsqueeze(0))
            pol = pol.squeeze().softmax(dim=0)
            
            index = int(torch.multinomial(torch.flatten(pol), 1))
            at = index//81
            xt = (index - (at*81))//9
            yt = index % 9 
            action = (at,xt,yt)

            print(state)
            print(action)
            try:
                state = self.env.step(state, turn, action)
                turn = (turn+1)%2
            except ValueError:
                if t > 10:
                    return 0
                t += 1
                
            
            
        if turn == starting_point:
            return -1
        elif turn != starting_point:
            return 1

    
    
    def backpropagate(self, node, win):
        assert node.visits == 1

        if win == 1:
            node.wins += win
        elif win == -1:
            pass
        absolute_turn = node.turn
        while node.parent == -1:
            if node.parent.turn != absolute_turn:
                if win == 1:
                    node.parent.wins += 1
                else:
                    pass
                
            elif node.parent.turn == absolute_turn:
                if win == -1:
                    node.parent.wins += 1
                else:
                    pass
            
            node = node.parent


    def decide(self, node):
        current = node
        keys = []
        uct_val = []
        for key, val in current.childs.items():
            uct_val.append(self.UCT(val))
            keys.append(key)
        current = current.childs[keys[torch.multinomial(torch.Tensor(uct_val).softmax(dim=0),1)]]
        current.visits += 1
        return current.action


@ray.remote
def simulate(env, stm, preps):
    
    starting_point = preps[1]
    state = preps[0]
    turn = preps[1]
    t = 0
    while state.done == False:
        pol, val = stm(torch.Tensor(preprocessor(state, turn)).unsqueeze(0))
        pol = pol.squeeze().softmax(dim=0)
            
        index = int(torch.multinomial(torch.flatten(pol), 1))
        at = index//81
        xt = (index - (at*81))//9
        yt = index % 9 
        action = (at,xt,yt)
        try:
            state = env.step(state, turn, action)
            turn = (turn+1)%2
        except ValueError:
            if t > 10:
                return 0
            t += 1
            
            
            
    if turn == starting_point:
        return -1
    elif turn != starting_point:
        return 1
    
    

if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from model import stm

    
    stm = stm((25, 9, 9), input_channel=25, p_output_channel=4, filters=192, block_num=7, value_dim=256)
    state_dict = torch.load("../../pretrained.pt")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
        

    stm.load_state_dict(new_state_dict)
    
    stm.eval()

    ray.init()
    
    env_id = ray.put(PuoriborEnv())
    stm_id = ray.put(stm)
    mcts = MCTS(stm, 1.4)
    leaf = mcts.select()
    prep_states, nodes = mcts.expand(leaf)

    mcts.simulate(prep_states[58])
    

    num_cpus = 8

    a = time.time()

    result_list = []
    leng = len(prep_states)//num_cpus
    result = ray.get([simulate.remote(env_id, stm_id, prep) for prep in prep_states])
    print(result)

    
    print(f'time: {time.time()-a}')
