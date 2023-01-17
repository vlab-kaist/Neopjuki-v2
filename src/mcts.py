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
    def __init__(self, stm, temp, initial_state=None):
        super().__init__()
        
        self.stm = stm
        self.temp = temp
        self.env = puoribor.PuoriborEnv()
        
        
        if initial_state != None:
            self.current = Node(initial_state, -1)
        else:
            self.current = Node(self.env.initialize_state(), -1)

    def UCT(self, states): # Calculate all possible scores from actions 
        self.temp*self.stm(preprocessor(self.current.state, ))

    def select(self):
        ## Should Implement from here ##
        

        
        if len(self.current.childs) == 0: # Check this is the leaf node
            return None
        else:
            pass

    def expand(self, states): # all possible states 
        assert len(self.current.childs) == 0 # Check this is the leaf node
        for state in states:
            new_node = Node(state, self.current)
            self.current.child[new_node.address] = new_node
        

    def simulate(self):
        assert len(self.current.childs) == 0 # Check this is the leaf node
        pass

    def backpropagate(self):
        pass

    


