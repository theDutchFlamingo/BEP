import random
import numpy as np


def init(g):
    N = len(g.nodes)
    x = np.zeros(N)
    y = np.zeros(N)

    for i, v in enumerate(g.nodes):
        x[i] = random.random()*2-1
        y[i] = random.random()*2-1

    return {g.nodes[i]:(x[i],y[i]) for i in range(N)}


def next_step(dict, edges):
    for v in dict:
        force = (0,0)
        
        for w in dict:
            if v != w:
                force = 

    return dict


def draw(g):
    dict = init(g)

    for i in range(5):
        dict = next_step(dict, g.edges)
