import ray
import math
import torch
import random
from fights.envs import puoribor
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
        self.wins = 0
        self.visits = 1
        

class MCTS(object):
    def __init__(self, stm, temp, sim_num, max_iter=50, first=0, initial_state=None):
        super().__init__()

        
        self.stm = stm
        self.temp = temp
        self.max_iter=max_iter
        self.sim_num = sim_num
        self.env = puoribor.PuoriborEnv()
        self.actions_cache = generate_actions((4,9,9))

        
        
        if initial_state != None:
            self.root = Node(first, initial_state, -1, -1)           
        else:
            self.root = Node(first, self.env.initialize_state(), -1, -1)

        
    def UCT(self, node):
        q_val = 0
        u_val = self.temp*self.pi*(math.sqrt(node.parent.visits)/(1+node.visits))

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
            for key, val in current.childs.item():
                uct_val.append(UCT(val))
                keys.append(key)
            current = current.childs[keys[np.array(uct_val).argmax()]]
            current.visits += 1
        return current
    
    def expand(self, node): # all possible states 
        assert len(node.childs) == 0 # Check this is the leaf node
        pol, val = self.stm(torch.Tensor(preprocessor(node.state, (node.turn+1)%2)).unsqueeze(0))
        pol = pol.flatten()
        pis = []
        nodes = []
        states = []
        actions = []

        prep_states = []
        
        for i, action in enumerate(self.actions_cache):
            try:
                states.append(self.env.step(node.state, node.turn, action))
                actions.append(action)
                pis.append(pol[i])

            except ValueError:
                pass
        
        for i, state in enumerate(states):
            new_node = Node((node.turn+1)%2, state,actions[i], node)
            new_node.pi = pis[i]
            node.childs[new_node.address] = new_node
            nodes.append(new_node)
            prep_states.append((state, new_node.turn))
        return prep_states, nodes, val

    def simulate(self, preps):
        starting_point = preps[1]
        state = preps[0]

        turn = preps[1]

        while state.done == False:
            pol, val = self.stm(torch.Tensor(preprocessor(preps[0], preps[1])).unsqueeze(0).to(self.device))
            pol = pol.squeeze()
            index = int(torch.multinomial(torch.flatten(pol), 1))
            action = (index//81,index//9,index%9)
            
            try:
                state = self.env.step(state, turn, action)
                turn = (turn+1)%2
            except ValueError:
                pass
            
            
        if turn == starting_point:
            return -1
        elif turn != starting_point:
            return 1

    
    
    def backpropagate(self, node, win):
        assert node.visits == 1

        if win == -1:
            pass
        else:
            node.wins += win
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


@ray.remote
def simulate(env, stm, preps):
    
    starting_point = preps[1]
    state = preps[0]

    turn = preps[1]
    i = 0
    while state.done == False:
        pol, val = stm(torch.Tensor(preprocessor(preps[0], preps[1])).unsqueeze(0))
        pol = pol.squeeze()
        index = int(torch.multinomial(torch.flatten(pol), 1))
        action = (index//81,index//9,index%9)
    
        try:
            if(i > 1000):
                return 0
            state = env.step(state, turn, action)
            turn = (turn+1)%2
        except ValueError:
            i += 1
            pass
    if turn == starting_point:
        return -1
    
    elif turn != starting_point:
        return 1

    

if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from model import stm

    
    stm = stm((25, 9, 9), input_channel=25, p_output_channel=4, filters=192, block_num=5, value_dim=256)
    

    ray.init()
    
    env_id = ray.put(puoribor.PuoriborEnv())
    stm_id = ray.put(stm)
    mcts = MCTS(stm, 1.4, 10)
    leaf = mcts.select()
    prep_states, nodes, val = mcts.expand(leaf)
    print(len(prep_states))

    num_cpus = 8

    a = time.time()

    result_list = []
    leng = len(prep_states)//num_cpus
    for i in tqdm(range(leng+1)):
        if i == leng:
            result_list.append(ray.get([simulate.remote(env_id, stm_id, prep) for prep in prep_states[num_cpus*leng:]]))
        else:
            result_list.append(ray.get([simulate.remote(env_id, stm_id, prep) for prep in prep_states[num_cpus*i:num_cpus*(i+1)]]))
    print(result_list)
    
    
    
    print(f'time: {time.time()-a}')
