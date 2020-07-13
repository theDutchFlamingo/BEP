from moments_plotter import *
from numpy import sqrt, array, trace, zeros, kron
from tqdm import tqdm
from common import scream
import warnings
warnings.filterwarnings("ignore")

s = 5  # The amount of states that we consider per node
K = s ** N  # The amount of states

# The basic 2-level density matrix of one node with nonzero Q-eigenvalue
rho_Q10 = array([
    [2, 1],
    [1, 2]
])

rho_P10 = array([
    [2, -1j],
    [+1j, 2]
])

# The s-level density matrix of one node with some nonzero Q-eigenvalue
rho_Q10s = zeros((s, s))
# rho_Q10s = 1
np.fill_diagonal(rho_Q10s, 2)
np.fill_diagonal(rho_Q10s[1:], 1)
np.fill_diagonal(rho_Q10s[:, 1:], 1)

rho_P10s = zeros((s, s))
# rho_P10s = 1
np.fill_diagonal(rho_P10s[1:], 1)
np.fill_diagonal(rho_P10s[:, 1:], -1)
rho_P10s = rho_P10s * 1j
np.fill_diagonal(rho_P10s, -2)

# print(rho_P10)
# print(rho_P10s)

# The s-level density matrix of several nodes with nonzero Q-eigenvalue for node 0
rho_QN0 = kron(np.identity(s ** (N - 1)), rho_Q10s) / K
# Nonzero for node N-1
rho_QNN_1 = kron(kron(rho_Q10s, rho_Q10s), rho_Q10s) / K
rho_PNN_1 = kron(rho_P10s, np.identity(s ** (N - 1))) / K

rho_0 = rho_QNN_1 - rho_PNN_1



def assert_density(dens):
    # Assert Hermitian
    assert np.all(dens == dens.conj().T)
    # Assert positive
    assert all(np.linalg.eigvalsh(dens) > 0)


# Momentum operator for node i
def p(i):
    A = zeros((K, K))

    for n in range(K):
        # Obtain n_i from n
        n_i = (n % s ** (i + 1)) // (s ** i)

        if n_i != s - 1:
            # The lowering operator is applied on the state above
            A[n, n + s ** i] = sqrt(n + s ** i)

        if n_i != 0:
            # The raising  operator is applied on the state below
            A[n, n - s ** i] = -sqrt(n)

    return -1j * sqrt(Omega[i] / 2) * A


# Position operator for node i
def q(i):
    A = zeros((K, K))

    for n in range(K):
        # Obtain n_i from n
        n_i = (n % s ** (i + 1)) // (s ** i)

        if n_i != 0:
            # First the raising operator is applied on the state below
            A[n, n - s ** i] = sqrt(n)

        if n_i != s - 1:
            # Next the lowering operator is applied on the state above
            A[n, n + s ** i] = sqrt(n + s ** i)

    return 1 / (sqrt(2 * Omega[i])) * A


def h():
    A = zeros((K, K))

    for n in range(K):
        I = np.arange(N)
        # Digit representation of n in base s with a maximum of N digits
        n_I = (n % s ** (I + 1)) // (s ** I)
        A[n, n] = Omega.dot(n_I + 1 / 2)

    return A


def com(a, b):
    if 2 == len(b.shape) and len(a.shape) == 2:
        return a.dot(b) - b.dot(a)
    elif len(b.shape) == 2:
        return array([a[i].dot(b) - b.dot(a[i])
                      for i in range(a.shape[0])])
    elif len(a.shape) == 2:
        return array([a.dot(b[i]) - b[i].dot(a)
                      for i in range(b.shape[0])])
    else:
        return array([a[i].dot(b[i]) - b[i].dot(a[i])
                      for i in range(a.shape[0])])


def anti(a, b):
    if 2 == len(b.shape) and len(a.shape) == 2:
        return a.dot(b) + b.dot(a)
    elif len(b.shape) == 2:
        return array([a[i].dot(b) + b.dot(a[i])
                         for i in range(a.shape[0])])
    elif len(a.shape) == 2:
        return array([a.dot(b[i]) + b[i].dot(a)
                         for i in range(b.shape[0])])
    else:
        return array([a[i].dot(b[i]) + b[i].dot(a[i])
                         for i in range(a.shape[0])])


def expval(op, dens):
    return trace(op.dot(dens))


P = array([p(i) for i in range(N)])
Q = array([q(i) for i in range(N)])
H = h()

# print("P:", P)
# print("Q:", Q)
# print("H:", H)

print(np.linalg.eigvalsh(-1j*com(H, rho_0)))
# assert_density(-1j*com(H, rho_0))
# print(com(P[2], anti(Q[2], rho_0))[0])


# The Liouvillian superoperator
def liouville(rho):
    Gam = Gamma[:, np.newaxis, np.newaxis]
    Ome = Omega[:, np.newaxis, np.newaxis]
    Ddd = D[:, np.newaxis, np.newaxis]

    return np.sum(- 1j * com(H, rho)
                  - 1 / 4 * 1j * Gam * (com(Q, anti(P, rho)) - com(P, anti(Q, rho)))
                  - 1 / 4 * Ddd * (com(Q, com(Q, rho)) - com(P, com(P, rho)) / Ome ** 2),
                  axis=0)


def liouville_explicit(rho):
    Sum = 0

    # Sum over all the nodes
    for i in range(N):
        Sum += 1j * Gamma[i] * (com(Q[i], anti(P[i], rho)) - com(P[i], anti(Q[i], rho)))
        Sum += D[i] * (com(Q[i], com(Q[i], rho)) - com(P[i], com(P[i], rho)) / Omega[i] ** 2)
    
    # if random() < 0.01:
    #     print(Sum)

    return -1j * com(H, rho) - 1 / 4 * Sum


Rho = np.zeros((it, K, K), dtype=complex)
Rho[0] = rho_0

recalculate = True

ev_Q = zeros((N, it))
ev_P = zeros((N, it))
ev_QQ = zeros((M, it))
ev_PP = zeros((M, it))

if recalculate:
    for t in tqdm(range(1, it)):
        Rho[t] = rk4(liouville_explicit, Rho[t - 1])
        
        print(t)
        # assert np.all(Rho[t] == Rho[t].conj().T)
        # assert_density(Rho[t])

        ev_Q[:, t] = array([expval(Q[i], Rho[t]) for i in range(N)])
        ev_P[:, t] = array([expval(P[i], Rho[t]) for i in range(N)])
        ev_QQ[:, t] = array([expval(Q[i].dot(Q[j]), Rho[t]) for i in range(N) for j in range(i, N)])
        ev_PP[:, t] = array([expval(P[i].dot(P[j]), Rho[t]) for i in range(N) for j in range(i, N)])
        
        print(ev_Q[2, t])
        
        if np.nan == ev_Q[2, t]:
            scream()

    np.save("rho.npy", Rho)
else:
    Rho = np.load("rho.npy")
    ev_Q = array([[trace(Q[i].dot(Rho[t])) for t in range(it)] for i in range(N)])
    ev_P = array([[trace(P[i].dot(Rho[t])) for t in range(it)] for i in range(N)])
    ev_QQ = array([[trace(Q[i].dot(Q[j]).dot(Rho[t])) for t in range(it)] for i in range(N) for j in range(i, N)])
    ev_PP = array([[trace(P[i].dot(Q[j]).dot(Rho[t])) for t in range(it)] for i in range(N) for j in range(i, N)])


# print("Q:", ev_Q)
# print("PP:", ev_PP)

# Plots
plt.yscale("log")
first_plot_separate(ev_Q, "Q", True, ev_P)
plt.yscale("symlog")
plt.ylim([-1e300, 1e300])
first_plot_separate(ev_P, "P", True)
# second_plot_separate(ev_QQ, "Q", colorized=True)
