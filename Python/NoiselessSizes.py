from common import *
from plotter import *
from tqdm import tqdm
import matplotlib.pyplot as plt

p1 = .8
N = 15
It = 10000

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


def new_version(dat):
    nzit = 0
    total = 0

    for i in tqdm(range(It)):
        g = nx.erdos_renyi_graph(N, p1)
        cc = get_connected_components(g)

        # The amount of connected components with at least one noiseless eigenvector
        nzcc = 0
        temp = np.zeros([N, N], dtype=float)
        s_tot = 0.0

        for c in cc:
            m = nx.to_numpy_array(g)
            m = m[np.ix_(c, c)]

            evs = get_noiseless_eigenvectors(m)
            nsc = len(evs)

            if nsc != 0:
                nzcc += 1

            for v in evs:
                comp = len(c)
                ns = get_size(v)

                temp[ns - 1, comp - 1] += 1 / nsc
                s_tot += 1 / nsc

        if nzcc != 0:
            nzit += 1
            dat += temp / nzcc
            total += s_tot / nzcc

            if abs(np.sum(np.nan_to_num(dat)) - total) > tol * It / np.sqrt(2):
                print("Exceeded the It/sqrt(2) bound on the error")

    return dat / nzit


data = new_version(res)

fig, ax = plt.subplots()
im, cbar = heatmap(data, end=N, ax=ax, cmap="YlGnBu")
ax.invert_yaxis()

plt.show()
