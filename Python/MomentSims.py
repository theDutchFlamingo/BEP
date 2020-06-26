import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt

N = 3
T_F = 75
it = 15000
dt = T_F/it
T = np.arange(0, T_F, dt)

omega = np.array([1.2, 1, 1.8])
gamma = 0.07 * omega[1]
Lambda = 0.4 * omega[1] * np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# The rotation matrix diagonalizing the adjacency matrix
Omega, F = np.linalg.eigh(Lambda)
Gamma = np.sum(F, 0)**2 * gamma


def eq_matrix():
    a = np.diag(-np.tile(Gamma, 2)/2)
    np.fill_diagonal(a[:, N:], -Omega ** 2)
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


def first_order():
    p_0 = np.array([0.5, 1, -0.5])
    q_0 = np.array([-1, 0, 1])

    Q = P = np.zeros([N, it])

    Q[:, 0] = F.T.dot(q_0)
    P[:, 0] = p_0

    X = np.concatenate((P, Q))

    for i in range(1, it):
        X[:, i] = rk4(X[:, i-1])

    P, Q = np.reshape(X, [2, N, it])

    q = F.dot(Q)

    for n in range(N):
        plt.plot(T, q[n])

    plt.legend(["Node 1", "Node 2", "Node 3"])
    plt.ylabel("$\\left<q_i\\right>$")
    plt.xlabel("t")
    plt.show()

    for n in range(N):
        plt.plot(T, Q[n])

    plt.legend(["Mode 1", "Mode 2", "Mode 3"])
    plt.ylabel("$\\left<q_i\\right>$")
    plt.xlabel("t")
    plt.show()


def second_order():
    pq_0 = pp_0 = qq_0 = np.array([1]*(N*(N-1)//2))
    PQ = PP = QQ = np.zeros([N*(N-1)//2, it])
    PQ[0]


first_order()
