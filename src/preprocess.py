import torch
import numpy as np

def preprocessor(inps, max_walls=10):
    np_ones = np.ones(shape=inps.board[0].shape)
    np_zeros = np.zeros(shape=inps.board[0].shape)
    arrays = [inps.board[0],inps.board[1],inps.board[4],inps.board[5]]
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
        
    board = torch.Tensor(np.stack(arrays, axis=0))
    
    return board





if __name__ == '__main__':

    from fights.envs import puoribor
    env = puoribor.PuoriborEnv()
    state = env.initialize_state()
    for i, val in enumerate(preprocessor(state)):
        print(i, val)
