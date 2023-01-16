import torch
from preprocess import hashing_state


class Node(object):
    def __init__(self, state, parent):
        super().__init__()
        self.state = state
        self.parent = parent
        self.address = hashing_state(state)
        self.childs = []
        self.visits = 1
        self.wins = 0
        


class MCTS(object):
    def __init__(self, ):
        pass

    def selexpand():
        pass

    def backpropagate():
        pass

    


