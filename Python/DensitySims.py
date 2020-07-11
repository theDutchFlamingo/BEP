from moments_plotter import *
from numpy import sqrt
from tqdm import tqdm
from storage import *
from common import scream
import json

s = 2  # The amount of states that we consider
K = s**N  # The amount of states
K2 = K**2

rho_0 = np.zeros((K, K))/K
rho_0[0, 0] = 1


# Momentum operator for node i
def p(i):
    A = np.zeros((K, K))

    for n in range(K):
        # Obtain n_i from n
        n_i = (n % s**(i + 1))//(s**i)

        if n_i != s - 1:
            # The lowering operator is applied on the state above
            A[n + s**i, n] = sqrt(n + s**i)

        if n_i != 0:
            # The raising  operator is applied on the state below
            A[n - s**i, n] = -sqrt(n + s**i)

    return -1j * sqrt(Omega[i]/2) * A


# Position operator for node i
def q(i):
    A = np.zeros((K, K))

    for n in range(K):
        # Obtain n_i from n
        n_i = (n % s**(i + 1))//(s**i)

        if n_i != 0:
            # First the raising operator is applied on the state below
            A[n - s**i, n] = sqrt(n + s**i)

        if n_i != s - 1:
            # Next the lowering operator is applied on the state above
            A[n + s**i, n] = sqrt(n + s**i)
    
    return 1/(sqrt(2*Omega[i])) * A


def h():
    A = np.zeros((K, K))
    
    for n in range(K):
        I = np.arange(N)
        n_I = (n % s**(I + 1))//(s**I)
        A[n, n] = Omega.dot(n_I + 1/2)
        
    return A


P = np.array([p(i) for i in range(N)])
Q = np.array([q(i) for i in range(N)])
H = h()

print("P:", P)
print("Q:", Q)
print("H:", H)


def com(a, b):
    if 2 == len(b.shape) and len(a.shape) == 2:
        return a.dot(b) - b.dot(a)
    elif len(b.shape) == 2:
        return np.array([a[i].dot(b) - b.dot(a[i])
                         for i in range(a.shape[0])])
    elif len(a.shape) == 2:
        return np.array([a.dot(b[i]) - b[i].dot(a)
                         for i in range(b.shape[0])])
    else:
        return np.array([a[i].dot(b[i]) - b[i].dot(a[i])
                         for i in range(a.shape[0])])


def anti(a, b):
    if 2 == len(b.shape) and len(a.shape) == 2:
        return a.dot(b) + b.dot(a)
    elif len(b.shape) == 2:
        return np.array([a[i].dot(b) + b.dot(a[i])
                         for i in range(a.shape[0])])
    elif len(a.shape) == 2:
        return np.array([a.dot(b[i]) + b[i].dot(a)
                         for i in range(b.shape[0])])
    else:
        return np.array([a[i].dot(b[i]) + b[i].dot(a[i])
                         for i in range(a.shape[0])])


# The Liouvillian superoperator
def liouville(rho):
    Gam = Gamma[:, np.newaxis, np.newaxis]
    Ome = Omega[:, np.newaxis, np.newaxis]
    Ddd =     D[:, np.newaxis, np.newaxis]
    
    return np.sum(- 1j  * com(H, rho)
                  - 1/4 * 1j * Gam * (com(Q, anti(P, rho)) - com(P, anti(Q, rho)))
                  - 1/4 * Ddd * (com(Q, com(Q, rho)) - com(P, com(P, rho))/Ome**2),
                  axis=0)


Rho = np.zeros((it, K, K), dtype=complex)
Rho[0] = rho_0

recalculate = True

if recalculate:
    for t in (range(1, it)):
        print(Rho[t-1])
        Rho[t] = rk4(liouville, Rho[t - 1])
        if np.nan in Rho:
            scream()
        
    write("density", Rho)
else:
    Rho = read("density")


ev_Q = np.array([[np.trace(Q[i].dot(Rho[t])) for t in range(it)] for i in range(N)])
ev_P = np.array([[np.trace(P[i].dot(Rho[t])) for t in range(it)] for i in range(N)])

print(ev_Q)

first_plot_separate(ev_Q, "Q", True, ev_P)
