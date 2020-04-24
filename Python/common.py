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


def has_noiseless_subsytem(g: nx.Graph):
    n = g.number_of_nodes()
    m = nx.to_numpy_matrix(g) if g.number_of_edges() > 0 else np.matrix(np.zeros([n, n]))
    return has_noiseless_eigenvector(m)


def has_noiseless_eigenvector(m: np.matrix):
    for v in np.linalg.eigh(m)[1]:
        if sum(np.asarray(v)[0]) == 0:
            return True


def count_noiseless_subsystems(g: nx.Graph):
    n = g.number_of_nodes()
    m = nx.to_numpy_matrix(g) if g.number_of_edges() > 0 else np.matrix(np.zeros([n, n]))
    return count_noiseless_eigenvalues(m)


def count_noiseless_eigenvalues(m: np.matrix):
    count = 0
    for v in np.linalg.eigh(m)[1]:
        if sum(np.asarray(v)[0]) == 0:
            count += 1
            break

    return count
