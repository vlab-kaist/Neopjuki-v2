import os
import ray
import itertools
import numpy as np
from numba import jit
from joblib import hash as h
from fights.envs import puoribor

def preprocessor(inps, agent_id, max_walls=10):
    np_ones = np.ones(shape=inps.board[0].shape)
    np_zeros = np.zeros(shape=inps.board[0].shape)
    arrays = []
    if agent_id == 0:
        arrays = [np_zeros, inps.board[0], inps.board[1], inps.board[4], inps.board[5]]

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
    
    elif agent_id == 1:
        arrays = [np_ones, inps.board[1], inps.board[0], inps.board[4], inps.board[5]]

        for i in range(max_walls):
            if i+1 == inps.walls_remaining[1]:
                arrays.append(np_ones)
            else:
                arrays.append(np_zeros)

        for i in range(max_walls):
            if i+1 == inps.walls_remaining[0]:
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


@jit(forceobj=True)
def remapping_action(x):
    x = list(x)
    if x[0] == 0: # move
        pass
    elif x[0] == 1: # place horizontal wall
        x[2] -= 1
    elif x[0] == 2: # place vertical wall
        x[1] -= 1
    elif x[0] == 3: # rotate section
        pass
    else:
        raise

    return tuple(x)


@ray.remote
def p_data(env_id, lines):
    
    x_list = list()
    y_list = list()

    for (n, line) in enumerate(lines):
        if line[0] == "#": 
            continue

        state = env_id.initialize_state()
        raw_actions = list(map(lambda x: tuple(map(lambda x: int(x), x.split(", "))), line[2:-3].split("), (")))
        actions = map(remapping_action, raw_actions)
        
        try:
            for (iter, action) in enumerate(actions):
                preprocessed = preprocessor(state, iter % 2)
                x_list.append(preprocessed)
                y_list.append(action)
                state = env_id.step(state, iter % 2, action)

        except:
            print(f"#{n} is ignored.")
            continue

    return x_list, y_list


def generate_cache(filepath):
    ray.init()
    env = puoribor.PuoriborEnv()
    env_id = ray.put(env)
    file_list = os.listdir(filepath)
    
    futures = [p_data.remote(env_id, open(filepath+fil).readlines()) for fil in file_list]
    returned = ray.get(futures)

    xdata = np.concatenate([returns[0] for returns in returned])
    ydata = np.array([list(i) for returns in returned for i in returns[1]])

    return xdata, ydata
    
    



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

    generate_cache('../data/')
