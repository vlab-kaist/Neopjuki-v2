import ray
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
        self.address = hashing_state(state)

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
        self.actions_cache = generate_actions()
        
        if initial_state != None:
            self.root = Node(first, initial_state, -1, -1)
           
        else:
            self.root = Node(first, self.env.initialize_state(), -1, -1)

        
    def UCT(self): # Calculate all possible scores from actions
        pass

    def select(self):
        ## Should Implement from here ##
        pass
    
    def expand(self, node): # all possible states 
        assert len(node.childs) == 0 # Check this is the leaf node
        pol, val = self.stm(torch.Tensor(preprocessor(node.state, node.turn+1),device=self.stm.device))
        
        for state in states:
            new_node = Node(node.turn+1, state, node)
            node.childs[new_node.address] = new_node

    @ray.remote
    def simulate(self, state, turn):
        starting_point = turn
        while state.done == False:
            pol, val = self.stm(torch.Tensor(preprocessor(state, turn),device=self.stm.device))
            action = puoribor.PuoriborAction((pol==torch.max(pol)).nonzero().squeeze().detach().cpu().numpy()) # this part should be changed as randomly selected.
            state = self.env.step(state, turn, action)
            if turn == 0:
                turn = 1
            elif turn == 1:
                turn = 0
        if turn == starting_point:
            return -1
        elif turn != starting_point:
            return 1
        
    
    def backpropagate(self):
        pass

    


