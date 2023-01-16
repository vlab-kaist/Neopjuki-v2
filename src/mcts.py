import torch
from fights.envs import puoribor
from preprocess import preprocessor
from preprocess import hashing_state


class Node(object):
    def __init__(self, state, parent):
        super().__init__()
        self.state = state # this state should be original state.
        self.parent = parent # Node, -1 for root
        self.address = hashing_state(state) 
        self.childs = {}
        self.visits = 1
        self.wins = 0
        


class MCTS(object):
    def __init__(self, stm, initial_state=None):
        super().__init__()
        
        self.stm = stm
        self.env = puoribor.PuoriborEnv()
        
        if initial_state != None:
            self.current = Node(initial_state, -1)
        else:
            self.current = Node(self.env.initialize_state(), -1)

    def UCT(self, ):
        pass

    def select(self, ):
        pass

    def expand(self, state):
        new_node = Node(state, self.current)
        self.current.child[new_node.address] = new_node
        self.current = new_node

    def simulate(self):
        pass

    def backpropagate():
        pass

    


