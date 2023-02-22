import re
import sys
import torch
import colorama
from msgpack import packb
from src.model import stm
from src.mcts import MCTS
from collections import deque
from omegaconf import OmegaConf
from fights.envs import puoribor
from colorama import Fore, Style
from fights.base import BaseAgent
from fights.envs.puoribor import PuoriborEnv


parser = argparse.ArgumentParser(description='Neopjuki')
parser.add_argument('--io', type=bool, help='STDIO for server', default=False)
parser.add_argument('--p2e', type=bool, help='Player vs AI Mode', default=False)


DEBUG = True
IO = args.io
p2e = args.p2e
SIM_NUM = 10
EnvX = PuoriborEnv()
conf = OmegaConf.load("../config.yaml")

def load_model(conf, path="../checkpoints/selfplayed/"):
    stmp = stm(input_shape=(conf['stm']['input_channel'], conf['env']['board_size'], conf['env']['board_size']),
           input_channel=conf['stm']['input_channel'], p_output_channel=conf['stm']['output_channel'],
           filters=conf['stm']['filters'], block_num=conf['stm']['block_num'], value_dim=conf['stm']['value_dim'])
    
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    stmp.load_state_dict(new_state_dict)
    stmp.eval()

    return stmp

def run(state, player):
    
    envI = PuoriborEnv()
    stmp = load_model(conf)
    tr = MCTS(stmp, conf['mcts']['temp'], initial_state=state)
    current = tr.root
    for i in tqdm(range(SIM_NUM)):
        leaf = tr.select()
        prep_states, nodes = tr.expand(leaf)

        for node in nodes:
            if node.state.done == True:
                tr.backpropagate(node, 1)

            else:
                _ , val = tr.stm(torch.Tensor(preprocessor(node.state, (node.turn+1)%2)).unsqueeze(0).to(tr.dev))
                if val >= 0:
                    tr.backpropagate(node, 1)
                elif val < 0:
                    tr.backpropagate(node, -1)
    pi = np.zeros((324,))
    for addr in current.childs:
        node = current.childs[addr]
        pi[node.action[0]*81 + node.action[1]*9 + node.action[2]] += node.visits
    state = envs.step(state, current.turn, action)
    current = current.childs[hashing_state(state, (current.turn+1)%2)]
    tr.root = current
    return action

    

def fallback_to_ascii(s: str) -> str:
    try:
        s.encode(sys.stdout.encoding)
    except UnicodeencodeError:
        s = re.sub("[┌┬┐├┼┤└┴┘]", "+", re.sub("[─━]", "-", re.sub("[│┃]", "|", s)))
    return s


def colorize_walls(s: str) -> str:
    return s.replace("━", Fore.BLUE + "━" + Style.RESET_ALL).replace("┃", Fore.RED + "┃" + Style.RESET_ALL)


def play(debug=DEBUG):
    assert puoribor.Puoriborenv.env_id == utils.PuoriborAgent.env_id
    colorma.init()

    state = EnvX.initialize_state()
    history = []
    agentIds = [0, 1]

    if debug:
        print("\x1b[2J")

    it = 0
    while not state.done:
        if debug:
            print("\x1b[1;1H")
            print(fallback_to_ascii(colorize_walls(str(state))))

        for agentId in agentId == 1:
            if IO: res = input()
            if (not p2e) or agentId == 1:
                value, action = run(stae, agentId)

            else:
                action = list(map(int, res.split(' ')))
            state = EnvX.step(state, agentId, action)
            if IO: print(' '.join(list(map(str, action))))
            if debug:
                print("x1b[1;1H")
                print(fallback_to_ascii(colorize_walls(str(state))))
            if state.done:
                if debug:
                    print(f"agent {agentID} won in {it} iters")
                break

            history.append(utils.to_history_dict(state, agentId, action))
        it += 1

        return history


if __name__ == '__main__':
    print(play())
