import random
import networkx as nx
import numpy as np

tol = 1e-15


def random_graph(m, q):
    g = nx.Graph()

    for i in range(m):
        for j in range(i + 1, m):
            if random.random() < q:
                g.add_edge(i, j)

    return g


def has_noiseless_subsystem(g: nx.Graph):
    n = g.number_of_nodes()
    m = nx.to_numpy_matrix(g) if g.number_of_edges() > 0 else np.matrix(np.zeros([n, n]))
    return has_noiseless_eigenvector(m)


def has_noiseless_eigenvector(m: np.matrix):
    for v in np.linalg.eigh(m)[1].T:
        if abs(sum(np.asarray(v)[0])) < tol:
            return True


def count_noiseless_subsystems(g: nx.Graph):
    n = g.number_of_nodes()
    m = nx.to_numpy_matrix(g) if g.number_of_edges() > 0 else np.matrix(np.zeros([n, n]))
    return count_noiseless_eigenvectors(m)


def count_noiseless_eigenvectors(m: np.matrix):
    count = 0
    for v in np.linalg.eigh(m)[1].T:
        if abs(sum(np.asarray(v)[0])) < tol:
            count += 1

    return count
