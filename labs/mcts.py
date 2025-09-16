# imports
from __future__ import annotations
#from pty import CHILD
import re
import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass, field


# %% Setup
env: Env


# %%
def minimax(state: State, maxim: bool) -> int:
    if state.ended:
        return state.point
    else:
        temp: int = -10 if maxim else 10
        for action in np.where(state.legal)[0]:  # for all legal actions
            value = minimax(env.step(state, action), not maxim)
            temp = max(temp, value) if maxim else min(temp, value)
        return temp


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int) -> int:
    raise NotImplementedError  # you do this

@dataclass
class Node:
    def __init__(self, state,parent=None,action=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.untried_actions = []
        self.n = 0
        self.q = 0.0
        self.action = action

    



# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    root = Node(state)
    root.untried_actions = list(np.where(state.legal)[0])


    for _ in range(cfg.compute):
        node = tree_policy(root, cfg)
        delta = default_policy(node.state)
        backup(node, delta)

    #after run best child
    best = best_child(root, cfg.c)
    return best.action
    
    
    #raise NotImplementedError  # you do this


def tree_policy(node: Node, cfg) -> Node:
    while not node.state.ended:
        if node.untried_actions:
            return expand(node)
        else:
            node = best_child(node, cfg.c)
    return node

    #raise NotImplementedError  # you do this


def expand(v: Node) -> Node:

    action = v.untried_actions.pop()
    next_state = env.step(v.state, action)
    
    child = Node(next_state, parent=v, action=action)
    child.untried_actions = list(np.where(next_state.legal)[0])
    v.children[action] = child

    return child

    #raise NotImplementedError  # you do this


def best_child(root: Node, c) -> Node:

    weight = [(child.q / child.n) + c * np.sqrt((2 * np.log(root.n)) / child.n) for child in root.children.values()]
    return list(root.children.values())[np.argmax(weight)]

    #raise NotImplementedError  # you do this


def default_policy(state: State) -> int:

    while not state.ended:
        actions = np.where(state.legal)[0]
        if len(actions) == 0:
            break
        action = np.random.choice(actions).item()
        state = env.step(state, action)
    return state.point

    #raise NotImplementedError  # you do this


def backup(node, delta) -> None:

    while node is not None:
        node.n += 1
        node.q += delta
        node = node.parent

    #raise NotImplementedError  # you do this


# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    state = env.init()

    while not state.ended:
        actions = np.where(state.legal)[0]  # the actions to choose from

        match getattr(cfg, state.player):
            case "random":
                a = np.random.choice(actions).item()

            case "human":
                print(state, end="\n\n")
                a = int(input(f"Place your piece ({'x' if state.minim else 'o'}): "))

            case "minimax":
                values = [minimax(env.step(state, a), not state.maxim) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "alpha_beta":
                values = [alpha_beta(env.step(state, a), not state.maxim, -1, 1) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "monte_carlo":
                a = monte_carlo(state, cfg)
                #raise NotImplementedError

            case _:
                raise ValueError(f"Unknown player {state.player}")

        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")
