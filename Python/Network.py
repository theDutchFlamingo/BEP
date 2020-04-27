import os
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from common import *

N = np.arange(3, 30, 3, int)
P = np.arange(0, 1, 0.1, float)

It = 1000

ratios = []
ratios2 = []

for j, p in enumerate(P):
    row = []
    row2 = []

    for i, n in enumerate(N):
        G = [nx.erdos_renyi_graph(n, p) for k in range(It)]

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
        print(f"Noiseless average: {av}")
        print("Iteration: " + str(j * len(P) + i + 1) + " of " + str(len(N) * len(P)))

        row.append(z if z != 0 else 0.5e-1)
        row2.append(z2 if av != 0 else 0.5e-6)

    ratios.append(row)
    ratios2.append(row2)

# ratios = np.array(ratios, float)
# ratios2 = np.array(ratios2, float)

fig, ax = plt.subplots(2, 1, figsize=(6, 12))
im2 = ax[0].imshow(ratios2, cmap='hot', interpolation='nearest', extent=[P[1], P[-1], N[0], N[-1]],
                   aspect='auto', norm=cl.LogNorm())
ax[0].set_title("Averages")
fig.colorbar(im2, ax=ax[0])

im = ax[1].imshow(ratios, cmap='hot', interpolation='nearest', extent=[P[1], P[-1], N[0], N[-1]],
                  aspect='auto', norm=cl.LogNorm())
ax[1].set_title("Probability of 1")
fig.colorbar(im, ax=ax[1])
plt.show()

nx.readwrite.gexf.write_gexf(G[0], os.path.dirname(__file__) + "\\Graphs\\0.gexf")
