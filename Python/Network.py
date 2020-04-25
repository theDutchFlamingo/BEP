import os
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from common import *

N = np.arange(3, 30, 3, int)
P = np.arange(0, 1, 0.1, float)

It = 1000

ratios = []
ratios2 = []

for i, n in enumerate(N):
    row = []
    row2 = []

    for j, p in enumerate(P):
        G = [nx.erdos_renyi_graph(n, p) for k in range(It)]
        # M = [nx.to_numpy_matrix(g) if g.number_of_edges() > 0 else np.matrix(np.zeros([n, n])) for g in G]

        # V = [np.linalg.eigh(m) for m in M]

        # print(V[0][1])

        count = 0
        av = 0
        for g in G:
            # print(nx.to_numpy_matrix(g))
            if has_noiseless_subsytem(g):
                count += 1
            av += count_noiseless_subsystems(g)

        av /= len(G) * It
        z = count / It
        z2 = av

        print(f"Noiseless count: {count}")
        print("Iteration: " + str(i * len(P) + j + 1) + " of " + str(len(N) * len(P)))

        if z != 0:
            row.append(z if z != 0 else 0.5e-0)
        if av != 0:
            row2.append(av if av != 0 else 0.5e-0)

    ratios.append(row)
    ratios2.append(row2)

plt, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(ratios, cmap='hot', interpolation='nearest', extent=[N[1], N[-1], P[0], P[-1]],
               aspect='auto', norm=cl.LogNorm())
plt.colorbar(im)
plt.show()

nx.readwrite.gexf.write_gexf(G[0], os.path.dirname(__file__) + "\\Graphs\\0.gexf")
