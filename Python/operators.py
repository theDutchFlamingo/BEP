from numpy import sqrt, zeros, kron, trace, arange, diag, array
from common import tol, tol_h
from moments_common import *

s = 3  # The amount of states that we consider per node
K = s ** N  # The amount of states


## Creation, annihilation, momentum and position operators
def cr():
    return diag(sqrt(arange(1, s)), -1)


def an():
    return diag(sqrt(arange(1, s)), 1)


def p(n: int):
    base = -1j * sqrt(.5 * Omega[n]) * (an() - cr())
    return kron(kron(np.identity(s**(N - n - 1)), base), np.identity(s**n))


def q(n: int):
    base = 1/sqrt(2 * Omega[n]) * (an() + cr())
    return kron(kron(np.identity(s**(N - n - 1)), base), np.identity(s**n))


def h():
    A = zeros((K, K))

    for n in range(K):
        I = np.arange(N)
        # Digit representation of n in base s with a maximum of N digits
        n_I = (n % s ** (I + 1)) // (s ** I)
        A[n, n] = Omega.dot(n_I + 1 / 2)

    return A


## Initial conditions
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
def rho_init(base = rho_q_basic()):
    ret = sum([kron(kron(np.identity(s**(N - n - 1)), base), np.identity(s**n))
               for n in range(N)])
    return ret / trace(ret)


def rho_alt():
    ret = np.zeros((K, K), dtype=float)

    for n in range(N):
        d = 2*sqrt(s-1)/sqrt(2 * Omega[n])
        Q = 1/sqrt(2 * Omega[n]) * (an() + cr())
        np.fill_diagonal(Q, d)
        
        ret += kron(kron(np.identity(s**(N - n - 1)), Q), np.identity(s**n))
    
    return ret / trace(ret)


Q = array([q(n) for n in range(N)])
P = array([p(n) for n in range(N)])
QQ = array([q(n).dot(q(m)) for n in range(N) for m in range(n, N)])
PP = array([p(n).dot(p(m)) for n in range(N) for m in range(n, N)])
H = h()

if print_strings:
    print("H:", H)
    print("P:", P)
    print("Q:", Q)


## Some assertions
def is_hermitian(mat):
    return np.all(abs(mat - mat.conj().T) < tol)


def is_unit_trace(mat):
    return abs(trace(mat) - 1) < tol_h  # Very high tolerance


def is_pos_def(mat):
    return all(np.linalg.eigvalsh(mat) > 0)


def is_density(dens):
    return (is_hermitian(dens) and
            is_unit_trace(dens) and
            is_pos_def(dens))


## Some operators on operators
def com(a, b):
    return a.dot(b) - b.dot(a)


def anti(a, b):
    return a.dot(b) + b.dot(a)


def expval(op, dens):
    return trace(op.dot(dens))


# The Hamiltonian superoperator, for testing purposes
def hamiltonian(rho):
    return -1j * com(H, rho)


def dissipator(rho):
    Sum = 0

    # Sum over all the nodes
    for n in range(N):
        Sum += 1j * Gamma[n] * (com(Q[n], anti(P[n], rho)) - com(P[n], anti(Q[n], rho)))
        Sum += D[n] * (com(Q[n], com(Q[n], rho)) - com(P[n], com(P[n], rho)) / Omega[n] ** 2)

    return - 1 / 4 * Sum


# The Liouvillian superoperator
def liouvillian(rho):
    return hamiltonian(rho) + dissipator(rho)
