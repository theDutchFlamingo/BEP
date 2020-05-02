from common import *
import matplotlib.pyplot as plt

p1 = .05
N = 15
nx.laplacian_matrix(random_graph(N, p1)).toarray()

data = np.zeros([N, N])
data[1, -1] = .4
data[3, -1] = .25
data[5, -1] = .15
data[7, -1] = .08

for i in range(N):
    for j in range(0, i):
        data[i, j] = np.nan

fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.YlGnBu, edgecolor='white', linewidths=5)

# put the major ticks at the middle of each cell
# ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
# ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

ax.set_aspect('equal')
fig.colorbar(heatmap)

plt.show()