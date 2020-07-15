from moments_plotter import *
from numpy import sqrt, array, trace, zeros, kron
from tqdm import tqdm
from common import scream
import qutip
import warnings
warnings.filterwarnings("ignore")

s = 2  # The amount of states that we consider per node
K = s ** N  # The amount of states


# The s-level density matrix of one node with nonzero Q-eigenvalue
def rho_q_basic():
    rho_Q10s = zeros((s, s))
    np.fill_diagonal(rho_Q10s, 2)
    
    if s > 1:
        np.fill_diagonal(rho_Q10s[1:], 1)
        np.fill_diagonal(rho_Q10s[:, 1:], 1)
    return rho_Q10s


# The s-level density matrix of one nodes with nonzero P-eigenvalue
def rho_p_basic():
    rho_P10s = zeros((s, s), dtype=complex)
    np.fill_diagonal(rho_P10s, 2)
    
    if s > 1:
        np.fill_diagonal(rho_P10s[1:], 1j)
        np.fill_diagonal(rho_P10s[:, 1:], -1j)
    return rho_P10s


# The s-level density matrix of several nodes with nonzero eigenvalue for some operator
def rho_init(base):
    ret = sum([kron(kron(np.identity(s**n), base), np.identity(s**(N - n - 1)))
                      for n in range(N)])
    print(ret)
    return ret / trace(ret)


def assert_density(dens):
    # Assert Hermitian
    assert np.all(dens == dens.conj().T)
    # Assert unit trace
    assert abs(trace(dens) - 1) < tol
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
        return array([a[n].dot(b) - b.dot(a[n])
                      for n in range(a.shape[0])])
    elif len(a.shape) == 2:
        return array([a.dot(b[n]) - b[n].dot(a)
                      for n in range(b.shape[0])])
    else:
        return array([a[n].dot(b[n]) - b[n].dot(a[n])
                      for n in range(a.shape[0])])


def anti(a, b):
    if 2 == len(b.shape) and len(a.shape) == 2:
        return a.dot(b) + b.dot(a)
    elif len(b.shape) == 2:
        return array([a[n].dot(b) + b.dot(a[n])
                         for n in range(a.shape[0])])
    elif len(a.shape) == 2:
        return array([a.dot(b[n]) + b[n].dot(a)
                         for n in range(b.shape[0])])
    else:
        return array([a[n].dot(b[n]) + b[n].dot(a[n])
                         for n in range(a.shape[0])])


def expval(op, dens):
    return trace(op.dot(dens))


rho_0 = rho_init(rho_q_basic() + rho_p_basic())
assert_density(rho_0)
P = array([p(n) for n in range(N)])
Q = array([q(n) for n in range(N)])
H = h()

print(expval(Q[0], rho_0))
# print("P:", P)
# print("Q:", Q)
# print("H:", H)

print(np.linalg.eigvalsh(rho_0 - dt*1j*com(H, rho_0)))
# print(trace(-1j*com(H, rho_0)))  # Should be 0


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
    for n in range(N):
        Sum += 1j * Gamma[n] * (com(Q[n], anti(P[n], rho)) - com(P[n], anti(Q[n], rho)))
        Sum += D[n] * (com(Q[n], com(Q[n], rho)) - com(P[n], com(P[n], rho)) / Omega[n] ** 2)

    return -1j * com(H, rho) - 1 / 4 * Sum


print(np.linalg.eigvalsh(liouville(rho_0)))

Rho = np.zeros((it, K, K), dtype=complex)
Rho[0] = rho_0

recalculate = True

ev_Q = zeros((N, it))
ev_P = zeros((N, it))
ev_QQ = zeros((M, it))
ev_PP = zeros((M, it))

screamed = False

if recalculate:
    for t in tqdm(range(it)):
        if t != 0:
            Rho[t] = rk4(liouville_explicit, Rho[t - 1])
        
        # assert np.all(Rho[t] == Rho[t].conj().T)
        # assert_density(Rho[t])
        # print(trace(Rho[t]))

        ev_Q[:, t] = array([expval(Q[n], Rho[t]) for n in range(N)])
        ev_P[:, t] = array([expval(P[n], Rho[t]) for n in range(N)])
        ev_QQ[:, t] = array([expval(Q[n].dot(Q[j]), Rho[t]) for n in range(N) for j in range(n, N)])
        ev_PP[:, t] = array([expval(P[n].dot(P[j]), Rho[t]) for n in range(N) for j in range(n, N)])
        
        if np.isnan(ev_Q[0, t]) and not screamed:
            scream()
            screamed = True

    np.save("rho.npy", Rho)
else:
    Rho = np.load("rho.npy")
    ev_Q = array([[expval(Q[n], Rho[t]) for t in range(it)] for n in range(N)])
    ev_P = array([[expval(P[n], Rho[t]) for t in range(it)] for n in range(N)])
    ev_QQ = array([[expval(Q[n].dot(Q[j]), Rho[t]) for t in range(it)] for n in range(N) for j in range(n, N)])
    ev_PP = array([[expval(P[n].dot(P[j]), Rho[t]) for t in range(it)] for n in range(N) for j in range(n, N)])


# print("Q:", ev_Q)
# print("PP:", ev_PP)

# Plots
# plt.yscale("log")
print(ev_Q[0])
first_plot_separate(ev_Q, "Q", True, ev_P)
# plt.yscale("symlog")
# plt.ylim([-1e300, 1e300])
# first_plot_separate(ev_P, "P", True)
# second_plot_separate(ev_QQ, "Q", colorized=True)
