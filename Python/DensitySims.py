from moments_plotter import *
from numpy import array, trace
from tqdm import tqdm
from common import scream
from operators import *
import warnings
warnings.filterwarnings("ignore")


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


def is_hermitian(mat):
    return np.all(abs(mat - mat.conj().T) < tol)


def is_unit_trace(mat):
    return abs(trace(mat) - 1) < tol * 100000  # Very high tolerance


def is_pos_def(mat):
    return all(np.linalg.eigvalsh(mat) > 0)


def assert_density(dens):
    # Assert Hermitian
    assert is_hermitian(dens)
    # Assert unit trace
    assert is_unit_trace(dens)
    # Assert positive
    assert is_pos_def(dens)


def com(a, b):
    return a.dot(b) - b.dot(a)


def anti(a, b):
    return a.dot(b) + b.dot(a)


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
print("H:", H)

print(np.linalg.eigvalsh(rho_0 - dt*1j*com(H, rho_0)))
# print(trace(-1j*com(H, rho_0)))  # Should be 0


# The Liouvillian superoperator
def liouville(rho):
    Sum = 0

    # Sum over all the nodes
    for n in range(N):
        Sum += 1j * Gamma[n] * (com(Q[n], anti(P[n], rho)) - com(P[n], anti(Q[n], rho)))
        Sum += D[n] * (com(Q[n], com(Q[n], rho)) - com(P[n], com(P[n], rho)) / Omega[n] ** 2)

    return -1j * com(H, rho) - 1 / 4 * Sum


print(np.linalg.eigvalsh(liouville(rho_0)))

Rho = rho_0

recalculate = True
screamed = False

if recalculate:
    ev_Q = zeros((N, it))
    ev_P = zeros((N, it))
    ev_QQ = zeros((M, it))
    ev_PP = zeros((M, it))
    
    for t in tqdm(range(it)):
        ev_Q[:, t] = array([expval(Q[n], Rho) for n in range(N)])
        ev_P[:, t] = array([expval(P[n], Rho) for n in range(N)])
        ev_QQ[:, t] = array([expval(Q[n].dot(Q[j]), Rho) for n in range(N) for j in range(n, N)])
        ev_PP[:, t] = array([expval(P[n].dot(P[j]), Rho) for n in range(N) for j in range(n, N)])
        
        Rho = rk4(liouville, Rho)
        
        # assert_density(Rho)
        # assert is_hermitian(Rho)
        assert is_unit_trace(Rho)
        # print(trace(Rho))
        
        if np.isnan(ev_Q[0, t]) and not screamed:
            scream()
            screamed = True

    np.save("ev_Q.npy", ev_Q)
    np.save("ev_P.npy", ev_P)
    np.save("ev_QQ.npy", ev_QQ)
    np.save("ev_PP.npy", ev_PP)
else:
    ev_Q = np.load("ev_Q.npy")
    ev_P = np.load("ev_P.npy")
    ev_QQ = np.load("ev_QQ.npy")
    ev_PP = np.load("ev_PP.npy")


print(ev_Q[0])
print(ev_Q[1])
print(ev_Q[2])
# print("Q:", ev_Q)
# print("PP:", ev_PP)

# Plots
# plt.yscale("log")
first_plot_separate(ev_Q, "Q", True, ev_P)
# plt.yscale("symlog")
# plt.ylim([-1e300, 1e300])
# first_plot_separate(ev_P, "P", True)
# second_plot_separate(ev_QQ, "Q", colorized=True)
