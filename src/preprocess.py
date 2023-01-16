import torch
import numpy as np

from joblib import hash as h

def preprocessor(inps, agent_id, max_walls=10):
    np_ones = np.ones(shape=inps.board[0].shape)
    np_zeros = np.zeros(shape=inps.board[0].shape)
    arrays = []
    if agent_id == 0:
        arrays = [inps.board[0],inps.board[1],inps.board[4],inps.board[5]]
        
    elif agent_id == 1:
        arrays = [np.rot90(inps.board[1],2), np.rot90(inps.board[0],2),
                  np.pad(np.rot90(inps.board[4], 2)[1:, 1:],((0, 1), (0, 1)), constant_values=0,),
                  np.pad(np.rot90(inps.board[5], 2)[1:, 1:],((0, 1), (0, 1)), constant_values=0,)]
        
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

def hashing_state(preprocessed_state):
    return h(preprocessed_state)



if __name__ == '__main__':

    from fights.envs import puoribor
    env = puoribor.PuoriborEnv()
    state = env.initialize_state()
    preprocessed_state = preprocessor(state,0)
    for i, val in enumerate(preprocessed_state):
        print(i, val)
    print(f'The result of hashing the state: {hashing_state(preprocessed_state)}')

    preprocessed_state = preprocessor(state,1)
    for i, val in enumerate(preprocessed_state):
        print(i, val)
    print(f'The result of hashing the state: {hashing_state(preprocessed_state)}')
