import random
import networkx as nx
import numpy as np


def random_graph(m, q):
    g = nx.Graph()

    for i in range(m):
        for j in range(i + 1, m):
            if random.random() < q:
                g.add_edge(i, j)

    return g


def get_noiseless_eigenvalues(g: nx.Graph):
    m = nx.to_numpy_matrix(g)
    return [v for v in np.linalg.eigh(m)[1] if sum(v) == 0]
