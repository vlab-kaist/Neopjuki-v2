import torch
from fights.envs import puoribor
from preprocess import preprocessor
from preprocess import hashing_state


class Node(object):
    def __init__(self, state, parent):
        super().__init__()
        self.state = state # this state should be original state.
        self.parent = parent # Node
        self.address = hashing_state(state) 
        self.childs = []
        self.visits = 1
        self.wins = 0
        


class MCTS(object):
    def __init__(self, stm, initial_state=None):
        super().__init__()
        device_name = 'cpu'
        if torch.cuda.is_available():
            device_name = "cuda:0"
        
        self.stm = stm.to(torch.device(device_name))
        self.env = puoribor.PuoriborEnv()
        
        if initial_state != None:
            self.startingNode = Node(initial_state, None)
        else:
            self.startingNode = Node(self.env.initialize_state(), None)

    def UCT(self, ):
        pass

    def select(self, ):
        pass

    def expand():
        pass

    def backpropagate():
        pass

    


