import os
import numpy as np
from numba import jit
from preprocess import preprocessor
from fights.envs.puoribor import PuoriborEnv



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

def generate_puribor(lines):
    game_list = list() 

    for (n, line) in enumerate(lines):
        if line[0] == "#": 
            continue

        state = PuoriborEnv().initialize_state()
        raw_actions = list(map(lambda x: tuple(map(lambda x: int(x), x.split(", "))), line[2:-3].split("), (")))
        actions = map(remapping_action, raw_actions)

        try:
            for (iter, action) in enumerate(actions):
                preprocessed = preprocessor(state, iter % 2)
                game_list.append((preprocessed, action))
                state = PuoriborEnv().step(state, iter % 2, action)

        except:
            print(f"#{n} is ignored.")
            continue

    return game_list


def data(path):
    file_list = os.listdir(path)

    for file in file_list:
        dlist = generate_puribor(open(filepath+file).readlines())
        
        



if __name__ == '__main__':
    data('../data/') 
