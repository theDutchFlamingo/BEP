import os
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from common import *

N = np.linspace(30, 3, 28, dtype=int)
P = np.linspace(0, 1, 25, dtype=float)

It = 1000

ratios = []
ratios2 = []

for i, n in enumerate(N):
    row = []
    row2 = []

    for j, p in enumerate(P):
        G = [nx.erdos_renyi_graph(n, p) for k in range(It)]

        count = 0
        av = 0
        for g in G:
            if has_noiseless_subsystem(g):
                count += 1
            c = count_noiseless_subsystems(g)
            av += c

        z = count / It

        av /= len(G) * It
        z2 = av

        print("Iteration: " + str(i * len(P) + j + 1) + " of " + str(len(N) * len(P)))

        row.append(z if z != 0 else 0.5e-2)
        row2.append(z2 if av != 0 else 0.5e-6)

    ratios.append(row)
    ratios2.append(row2)

extent = [P[0], P[-1], N[-1], N[0]]
im_args = {"cmap": "hot", "interpolation": "nearest", "extent": extent, "aspect": "auto"}

fig, ax = plt.subplots(2, 1, figsize=(6, 12))
im2 = ax[0].imshow(ratios2, **im_args, norm=cl.LogNorm())
ax[0].set_title("Averages")
# ax[0].set_xlim(P[0], P[-1])
# ax[0].set_ylim(N[-1], N[0])
fig.colorbar(im2, ax=ax[0])

im = ax[1].imshow(ratios, **im_args, norm=cl.LogNorm())
ax[1].set_title("Probability of 1")
fig.colorbar(im, ax=ax[1])

plt.show()

nx.readwrite.gexf.write_gexf(G[0], os.path.dirname(__file__) + "\\Graphs\\0.gexf")
