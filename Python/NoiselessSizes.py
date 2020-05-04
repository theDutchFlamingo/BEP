from common import *
from plotter import *
import matplotlib.pyplot as plt

p1 = .8
N = 15
It = 1000

res = np.zeros([N, N])

for i in range(N):
    for j in range(0, i):
        res[i, j] = np.nan


def old_version():
    for i in range(It):
        g = nx.erdos_renyi_graph(N, p1)
        for v in get_noiseless_subsystems(g):
            comp = get_component_size(g, v)
            ns = get_size(g, v)

            print(f"comp: {comp}, ns: {ns}")

            res[ns - 1, comp - 1] += 1 / It / count_noiseless_subsystems(g)
    return res


def new_version():
    for i in range(It):
        g = nx.erdos_renyi_graph(N, p1)
        cc = get_connected_components(g)
        for c in cc:
            m = nx.to_numpy_array(g)
            m = m[np.ix_(c, c)]
            nsc = count_noiseless_eigenvectors(m)

            for v in get_noiseless_eigenvectors(m):
                lap = np.diag(np.sum(m, axis=0)) - m
                print(get_connected_count_laplacian(lap, v))
                comp = get_component_size_laplacian(lap, v)
                ns = get_size(v)

                res[ns - 1, comp - 1] += 1 / It / len(cc) / nsc
    return res


data = new_version()

fig, ax = plt.subplots()
im, cbar = heatmap(res, ax=ax, cmap="YlGnBu")
ax.invert_yaxis()

plt.show()
