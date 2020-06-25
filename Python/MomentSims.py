import numpy as np
from matplotlib import pyplot as plt

N = 3
T_F = 15
it = 1500
dt = T_F/it
T = np.arange(0, T_F, dt)

P_0 = np.array([1, 0.5, -1])
Q_0 = np.array([-1, 0, 1])

Omega = np.array([1.2, 1, 1.8])
gamma = 0.07 * Omega[1]
Lambda = Omega[1] * np.array([
    [0, 0.4, 0],
    [0.4, 0, 0.4],
    [0, 0.4, 0]
])

F = np.linalg.eigh(Lambda)[1]

Gamma = np.sum(F, 0)**2 * gamma

P = np.zeros([N, it])
Q = np.zeros([N, it])
X = np.concatenate((P, Q))

Q[:, 0] = Q_0
P[:, 0] = P_0
X[:, 0] = np.concatenate((Q_0, P_0))

A_S = np.array([[[-Gamma[i] / 2, -Omega[i] ** 2], [1, -Gamma[i] / 2]] for i in range(N)])


def eq_matrix():
    a = np.diag(-np.tile(Gamma, 2)/2)
    np.fill_diagonal(a[:, N:], -Omega**2)
    np.fill_diagonal(a[N:], np.ones(N))
    return a


A = eq_matrix()


def f(x):
    return A.dot(x)


def euler_forward(x):
    return np.linalg.inv(np.identity(2*N) - dt * A).dot(x)


def euler_backward(x):
    return x + dt*f(x)


def rk4(x):
    k1 = f(x)
    k2 = f(x + dt*k1/2)
    k3 = f(x + dt*k2/2)
    k4 = f(x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)


for i in range(1, it):
    X[:, i] = rk4(X[:, i-1])

for n in range(N, 2*N):
    plt.plot(T, X[n, :])

plt.legend(["Node 1", "Node 2", "Node 3"])
plt.show()
