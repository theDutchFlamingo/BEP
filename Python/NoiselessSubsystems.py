import os
from matplotlib import pyplot as plt
from matplotlib import colors as cl
from tqdm import tqdm
from common import *

N = np.linspace(30, 3, 27, dtype=int)
P = np.linspace(0, 1, 25, dtype=float)
N_dense = np.linspace(30, 3, 270, dtype=float)

P_start = 0

It = 100

ratios = []
ratios2 = []
min_ = 1
min2 = 1

# Calculate the probability of 1 NC and the fraction of NCs
for i, n in enumerate(tqdm(N)):
    row = []
    row2 = []

    for j, p in enumerate(P):
        G = [nx.erdos_renyi_graph(n, p) for k in range(It)]

        count = 0
        av = 0
        for g in G:
            c = count_noiseless_subsystems(g)

            if c > 0:
                count += 1
            av += c

        z = count / It

        av /= len(G) * It
        z2 = av

        if z != 0 and z < min_:
            min_ = z

        if z2 != 0 and z2 < min2:
            min2 = z2

        row.append(z)
        row2.append(z2)

    ratios.append(row)
    ratios2.append(row2)

# Set all zero values to the minimum nonzero value
for i, r in enumerate(ratios):
    for j in range(len(r)):
        if r[j] == 0:
            r[j] = min_

for i, r in enumerate(ratios2):
    for j in range(len(r)):
        if r[j] == 0:
            r[j] = min2

# Turn them into numpy arrays
ratios = np.array(ratios)
ratios2 = np.array(ratios2)

# Add the critical probability as a function of N
P_C = np.log(N_dense)/N_dense

extent = [P[P_start], P[-1], N[-1], N[0]]
im_args = {"cmap": "hot", "interpolation": "nearest", "extent": extent, "aspect": "auto"}

fig, ax = plt.subplots(2, 1, figsize=(6, 12))
im2 = ax[0].imshow(ratios2[:, P_start:], **im_args, norm=cl.LogNorm())
ax[0].set_title("Averages")
fig.colorbar(im2, ax=ax[0])

im = ax[1].imshow(ratios[:, P_start:], **im_args, norm=cl.LogNorm())
ax[1].set_title("Probability of 1")
ax[1].plot(P_C, N_dense, color="green", linestyle="--")
fig.colorbar(im, ax=ax[1])

plt.show()

nx.readwrite.gexf.write_gexf(G[0], os.path.dirname(__file__) + "\\Graphs\\0.gexf")
