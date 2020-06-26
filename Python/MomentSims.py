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

D = np.zeros(N)


def tri(n):
    return n*(n+1)//2


def d(why_do_u_care_about_shadowing, and_case_of_the_parameter):
    return 1 if why_do_u_care_about_shadowing == and_case_of_the_parameter else 0


def num(urgh, blegh):
    return tri(N - 1) - tri(N - 1 - urgh) + blegh


A = np.diag(-np.tile(Gamma, 2)/2)
np.fill_diagonal(A[:, N:], -Omega ** 2)
np.fill_diagonal(A[N:], np.ones(N))

# The number of unique auto/cross-correlations for one of QiQj, PiPj, PiQj
M = tri(N)
B = np.zeros([3*M, 3*M])

for i in range(N):
    for j in range(i, N):
        ind = num(i, j)
        rev = num(j, i)
        
        # The diagonal terms
        B[ind, ind] = B[ind + M, ind + M] =\
            B[ind + 2*M, ind + 2*M] = - (Gamma[i] + Gamma[j])/2
        
        # The non-diagonal terms
        B[ind, ind + 2*M] = B[ind, rev(j, i) + 2*M] = 1/2
        
        B[ind + M, ind + 2*M] = -Omega[i]**2/2
        B[ind + M, rev + 2*M] = -Omega[j]**2/2
        
        B[ind + 2*M, ind] = -2*Omega[j]
        B[ind + 2*M, ind + M] = 2
        
    
# The constant vector to be added (dY/dt = B*Y + L)
L = np.zeros(3*M)
for i in range(M):
    L[num(i, i)] = L[num(i, i) + M] = -D[i]/2/Omega[i]**2


def a(x):
    return A.dot(x)


def b(y):
    return B.dot(y) + L


def euler_forward(x):
    return np.linalg.inv(np.identity(2*N) - dt * A).dot(x)


def euler_backward(f, x):
    return x + dt * f(x)


def rk4(f, x):
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
        X[:, i] = rk4(a, X[:, i - 1])

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
    pq_0 = pp_0 = qq_0 = np.array([1]*M)
    PQ = PP = QQ = np.zeros([M, it])
    QQ[:, 0] = 1
    
    Y = np.concatenate((QQ, PP, PQ))
    
    for i in range(1, it):
        Y[:, i] = rk4(b, Y)


first_order()
