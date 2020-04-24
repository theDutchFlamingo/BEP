import os
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from common import *

N = np.arange(3, 30, 3, int)
P = np.arange(0, 1, 0.1, float)

It = 1000

ratios = []


def pr(v):
    print("Hi")
    print(v)
    print(type(v))
    print(sum(v))
    print(type(sum(v)))
    print(sum(v) == 0)
    return sum(v) == 0


for i, n in enumerate(N):
    row = []

    for j, p in enumerate(P):
        G = [nx.erdos_renyi_graph(n, p) for k in range(It)]
        M = [nx.to_numpy_matrix(g) if g.number_of_edges() > 0 else np.matrix(np.zeros([n, n])) for g in G]

        V = [np.linalg.eigh(m) for m in M]

        # print(V[0][1])

        count = 0
        for g in G:
            if has_noiseless_subsytem(g):
                count += 1

        z = count/n

        print("Hi")
        print("Iteration: " + str(i*len(P)+j + 1) + " of " + str(len(N) * len(P)))

        row.append(z if z != 0 else 0.5e-3)

    ratios.append(row)

plt, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(ratios, cmap='hot', interpolation='nearest', extent=[N[0], N[-1], P[0], P[-1]],
               aspect='auto', norm=cl.LogNorm())
plt.colorbar(im)
plt.show()

nx.readwrite.gexf.write_gexf(G[0], os.path.dirname(__file__) + "\\Graphs\\0.gexf")
