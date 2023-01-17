import ray
import math
import torch
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
    def __init__(self, stm, temp, sim_num, first=0, initial_state=None):
        super().__init__()
        
        self.stm = stm
        self.temp = temp
        self.sim_num = sim_num
        self.env = puoribor.PuoriborEnv()
        self.actions_cache = generate_actions((4,9,9))
        
        if initial_state != None:
            self.root = Node(first, initial_state, -1, -1)           
        else:
            self.root = Node(first, self.env.initialize_state(), -1, -1)

        
    def UCT(self, node):
        
        if len(node.childs):
            return (node.wins / node.visits) + self.temp*self.pi*(math.sqrt(node.parent.visits)/(1+node.visits))
        else:
            return node.parent.wins / node.parent.visits + self.temp*self.pi*(math.sqrt(node.parent.visits)/(1+node.visits))
        

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
        pol, val = self.stm(torch.Tensor(preprocessor(node.state, (node.turn+1)%2)).unsqueeze(0)) # should add matching cuda or cpu
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
        env = puoribor.PuoriborEnv()
        starting_point = preps[1]
        state = preps[0]

        turn = preps[1]

        while state.done == False:
            pol, val = self.stm(torch.Tensor(preprocessor(preps[0], preps[1])).unsqueeze(0))
            pol = pol.squeeze()
            index = int(torch.multinomial(torch.flatten(pol), 1))
            action = (index//81,index//9,index&9)
            
            try:
                state = env.step(state, turn, action)
                print(state)
                print(action)
                turn = (turn+1)%2
            except ValueError:
                pass
            
            
        if turn == starting_point:
            return -1
        elif turn != starting_point:
            return 1

    
    
    def backpropagate(self, node, win):
        assert node.visits == 1

        node.wins = win
        absolute_turn = node.turn
        while node.parent == -1:
            if node.parent.turn != absolute_turn:
                node.parent.wins += ((win+1) % 2)
            elif node.parent.turn == absolute_turn:
                node.parent.wins += (win % 2)
            
            node = node.parent



    

if __name__ == '__main__':
    import time
    from model import stm

    stm = stm((25, 9, 9), 1, [(25,32,4)], 256)
    mcts = MCTS(stm, 1.4, 10)

    leaf = mcts.select()
    
    prep_states, nodes, val = mcts.expand(leaf)
    print(val)
    print(nodes[68].state)
    
    mcts.simulate(prep_states[20])

