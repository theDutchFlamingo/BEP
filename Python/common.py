import random
import networkx as nx
import numpy as np

tol = 1e-15


def get_size(ns: np.array):
    return len([x for x in ns if x != 0])


def get_component_size(g: nx.Graph, ns: np.array):
    m = nx.laplacian_matrix(g).toarray()

    if all(m.dot(ns) == 0):
        return get_size(ns)
    else:
        prev = 0
        new = get_size(ns)

        while prev < new:
            prev = new
            ns = m.dot(ns)
            new = get_size(ns)

        return new


def random_graph(m, q):
    g = nx.Graph()

    for i in range(m):
        for j in range(i + 1, m):
            if random.random() < q:
                g.add_edge(i, j)

    return g


def get_noiseless_subsystems(g: nx.Graph):
    n = g.number_of_nodes()
    m = nx.to_numpy_matrix(g) if g.number_of_edges() > 0 else np.matrix(np.zeros([n, n]))
    return get_noiseless_eigenvectors(m)


def get_noiseless_eigenvectors(m: np.matrix):
    vecs = []
    for v in np.linalg.eigh(m)[1].T:
        if abs(sum(np.asarray(v)[0])) < tol:
            vecs.append(v)

    return vecs


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
