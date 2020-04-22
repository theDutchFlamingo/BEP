import os
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from common import *

N = np.arange(5, 30, 1, int)
P = np.arange(0, 1, 0.05, float)


It = 1000

ratios = []

for n in N:
    row = []

    for p in P:
        G = [random_graph(n, p) for k in range(It)]
        M = [nx.to_numpy_matrix(g) for g in G]

        V = [np.linalg.eigh(m) for m in M]

        z = len([v for v in V if 0 not in v[0]])/len(V)

        row.append(z if z != 0 else 0.5e-3)

    ratios.append(row)

plt, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(ratios, cmap='hot', interpolation='nearest', extent=[N[0], N[-1], P[0], P[-1]],
               aspect='auto', norm=cl.LogNorm())
plt.colorbar(im)
plt.show()

nx.readwrite.gexf.write_gexf(G[0], os.path.dirname(__file__) + "\\Graphs\\0.gexf")
