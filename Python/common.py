import random
import networkx as nx
import numpy as np
from scipy import linalg, compress, transpose

tol = 1e-15
tol_h = tol * 1e5


def scream():
    print("AAAASDFFDSF $T%YH^J&UBSVD h%$G what is going on? aweahfkljdhfaf")


def get_size(ns: np.array):
    return len([x for x in ns if abs(x) > tol])


def get_component_size_laplacian(m: np.array, ns: np.array):
    if all(m.dot(ns) < tol):
        return get_size(ns)
    else:
        prev = 0
        new = get_size(ns)

        while prev < new:
            prev = new
            ns = m.dot(ns)
            new = get_size(ns)

        return new


def get_component_size(g: nx.Graph, ns: np.array):
    m = nx.laplacian_matrix(g).toarray()

    return get_component_size_laplacian(m, ns)


def get_connected_count(g: nx.Graph, ns: np.array):
    m = nx.laplacian_matrix(g).toarray()

    return get_connected_count_laplacian(m, ns)


def get_connected_count_laplacian(m: np.array, ns: np.array):
    ar = [x for x in range(0, len(ns)) if ns[x] != 0]
    m = m[np.ix_(ar, ar)]

    return null_space(m).shape[1]


def get_connected_components(g: nx.Graph, ns: np.array = None):
    m = nx.laplacian_matrix(g).toarray()

    if ns is None:
        ar = range(g.number_of_nodes())
        ns = np.ones(len(ar))
    else:
        ar = [x for x in range(len(ns)) if ns[x] != 0]
    m = m[np.ix_(ar, ar)]

    visited = [False for x in ns if x != 0]

    def dfs(node):
        disc = []
        for v in range(len(visited)):
            if m[v, node] < 0 and not visited[v]:
                visited[v] = True
                disc += [v] + dfs(v)
        return disc

    ret = []
    u = 0

    for i in ns:
        if i == 0:
            continue

        if not visited[u]:
            visited[u] = True
            ret.append([u] + dfs(u))

        u += 1

    return tuple(ret)


def null_space(a, eps=tol):
    u, s, vh = linalg.svd(a)
    null_mask = (s <= eps)
    return transpose(compress(null_mask, vh, axis=0))


def random_graph(m, q):
    g = nx.Graph()

    for i in range(m):
        for j in range(i + 1, m):
            if random.random() < q:
                g.add_edge(i, j)

    return g


def get_noiseless_subsystems(g: nx.Graph):
    n = g.number_of_nodes()
    m = nx.to_numpy_array(g) if g.number_of_edges() > 0 else np.zeros([n, n])
    return get_noiseless_eigenvectors(m)


def get_noiseless_eigenvectors(m: np.array):
    vecs = []
    for v in np.linalg.eigh(m)[1].T:
        if abs(sum(v)) < tol:
            vecs.append(v)

    return vecs


def has_noiseless_subsystem(g: nx.Graph):
    n = g.number_of_nodes()
    m = nx.to_numpy_array(g) if g.number_of_edges() > 0 else np.zeros([n, n])
    return has_noiseless_eigenvector(m)


def has_noiseless_eigenvector(m: np.array):
    for v in np.linalg.eigh(m)[1].T:
        if abs(sum(v)) < tol:
            return True


def count_noiseless_subsystems(g: nx.Graph):
    n = g.number_of_nodes()
    m = nx.to_numpy_array(g) if g.number_of_edges() > 0 else np.zeros([n, n])
    return count_noiseless_eigenvectors(m)


def count_noiseless_eigenvectors(m: np.array):
    count = 0
    for v in np.linalg.eigh(m)[1].T:
        if abs(sum(v)) < tol:
            count += 1

    return count
