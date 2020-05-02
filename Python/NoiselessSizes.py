from common import *
from plotter import *
import matplotlib.pyplot as plt

p1 = .8
N = 15
It = 1000
nx.laplacian_matrix(random_graph(N, p1)).toarray()

data = np.zeros([N, N])

for i in range(N):
    for j in range(0, i):
        data[i, j] = np.nan

for i in range(It):
    g = nx.erdos_renyi_graph(N, p1)
    for v in get_noiseless_subsystems(g):
        comp = get_component_size(g, np.array(v)[0])
        ns = get_size(np.array(v)[0])
        
        print(f"comp: {comp}, ns: {ns}")

        data[ns - 1, comp - 1] += 1 / It / count_noiseless_subsystems(g)

fig, ax = plt.subplots()
im, cbar = heatmap(data, ax=ax, cmap="YlGnBu")
ax.invert_yaxis()

plt.show()
