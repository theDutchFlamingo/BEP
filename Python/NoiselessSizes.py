from common import *
import matplotlib.pyplot as plt

p1 = .05
N = 15
nx.laplacian_matrix(random_graph(N, p1)).toarray()

column_labels = list('ABCD')
row_labels = list('WXYZ')
data = np.random.rand(N, N)

for i in range(N):
    for j in range(0, i):
        data[i, j] = np.nan

fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.YlGnBu, edgecolor='white', linewidths=5)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

ax.set_aspect('equal')

plt.show()