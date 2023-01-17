import torch
import numpy as np

from joblib import hash as h

def preprocessor(inps, agent_id, max_walls=10):
    np_ones = np.ones(shape=inps.board[0].shape)
    np_zeros = np.zeros(shape=inps.board[0].shape)
    arrays = []
    if agent_id == 0:
        arrays = [np_zeros, inps.board[0], inps.board[1], inps.board[4], inps.board[5]]
        
    elif agent_id == 1:
        arrays = [np_ones, inps.board[1], inps.board[0], inps.board[4], inps.board[5]]
    
    for i in range(max_walls):
        if i+1 == inps.walls_remaining[0]:
            arrays.append(np_ones)
        else:
            arrays.append(np_zeros)

    for i in range(max_walls):
        if i+1 == inps.walls_remaining[1]:
            arrays.append(np_ones)
        else:
            arrays.append(np_zeros)
        
    board = np.stack(arrays, axis=0)
    
    return board

def hashing_state(inps, agent_id, max_walls=10):
    np_ones = np.ones(shape=inps.board[0].shape)
    np_zeros = np.zeros(shape=inps.board[0].shape)

    arrays = []
    if agent_id == 0:
        arrays = [np_zeros, inps.board[0], inps.board[1], inps.board[4], inps.board[5]]
        
    elif agent_id == 1:
        arrays = [np_ones, inps.board[1], inps.board[0], inps.board[4], inps.board[5]]
    
    for i in range(max_walls):
        if i+1 == inps.walls_remaining[0]:
            arrays.append(np_ones)
        else:
            arrays.append(np_zeros)

    for i in range(max_walls):
        if i+1 == inps.walls_remaining[1]:
            arrays.append(np_ones)
        else:
            arrays.append(np_zeros)

    board = np.stack(arrays, axis=0)
    
    return h(board)

def generate_actions(action_size):
    cartprod = np.stack(np.meshgrid(np.arange(action_size[0]), np.arange(action_size[1]), np.arange(action_size[2])),axis=-1).reshape(-1,3)
    return list(map(tuple, cartprod))

if __name__ == '__main__':

    from fights.envs import puoribor
    env = puoribor.PuoriborEnv()
    state = env.initialize_state()
    preprocessed_state = preprocessor(state,0)
    for i, val in enumerate(preprocessed_state):
        print(i+1, val, type(val))

    print(f'The result of hashing the state: {hashing_state(state,0)} at agent_id: {0}')

    preprocessed_state = preprocessor(state,1)
    for i, val in enumerate(preprocessed_state):
        print(i+1, val, type(val))
    print(f'The result of hashing the state: {hashing_state(state,1)} at agent_id: {1}')

    print(f'Testing the generating all actions in action space A: (4,9,9): {generate_actions((4,9,9))}')
