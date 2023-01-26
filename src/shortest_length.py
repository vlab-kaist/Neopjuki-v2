from collections import deque
from fights.envs import puoribor

import numpy as np

ENV = puoribor.PuoriborEnv()

def get_all_movements(state: puoribor.PuoriborState, player: int):
    actions = []
    for coordinate_x in range(puoribor.PuoriborEnv.board_size):
        for coordinate_y in range(puoribor.PuoriborEnv.board_size):
            action = [0, coordinate_x, coordinate_y]
            try:
                ENV.step(state, player, action)
            except:
                ...
            else:
                actions.append(action)

    return actions


def bfs(state: puoribor.PuoriborState, player: int) -> int:
    def checkDone(board) -> bool:
        if player == 0:
            return board[0, :, -1].sum()
        return board[1, :, 0].sum()
    sx, sy = np.argwhere(state.board[player] == 1)[0]
    Q = deque([(sx, sy)])
    visited = {(sx, sy) : 0}
    while Q:
        cx, cy = Q.popleft()
        state.board[player] = np.zeros((9, 9), dtype=np.int8)
        state.board[player][cx][cy] = 1
        if checkDone(state.board):
            return visited[(cx, cy)]
        for nx in range(9):
            for ny in range(9):
                try:
                    newState = ENV.step(state, player, [0, nx, ny])
                except:
                    continue
                if checkDone(newState.board):
                    return visited[(cx, cy)] + 1
                if (nx, ny) not in visited:
                    visited[(nx, ny)] = visited[(cx, cy)] + 1
                    Q.append((nx, ny))


def shortest_movement(state: puoribor.PuoriborState, player: int):
    moves = get_all_movements(state, player)

    optimal = None
    cost = 1_000

    for move in moves:
        next_state = ENV.step(state, player, move)
        move_cost = bfs(next_state, player)

        if move_cost < cost:
            optimal = move
            cost = move_cost

    return optimal
