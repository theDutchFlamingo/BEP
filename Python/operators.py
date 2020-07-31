from numpy import sqrt, zeros, kron, trace, arange, diag, array
from moments_common import *

s = 4  # The amount of states that we consider per node
K = s ** N  # The amount of states


## Creation, annihilation, momentum and position operators
def cr():
    return diag(sqrt(arange(1, s)), -1)


def an():
    return diag(sqrt(arange(1, s)), 1)


# The p operator for 1 node
def p1(n: int):
    return -1j * sqrt(.5 * Omega[n]) * (an() - cr())


# The full p operator
def p(n: int):
    return kron(kron(np.identity(s**(N - n - 1)), p1(n)), np.identity(s**n))


# The q operator for 1 node
def q1(n: int):
    return 1/sqrt(2 * Omega[n]) * (an() + cr())


# The full q operator
def q(n: int):
    return kron(kron(np.identity(s**(N - n - 1)), q1(n)), np.identity(s**n))


def h():
    A = zeros((K, K))

    for n in range(K):
        I = np.arange(N)
        # Digit representation of n in base s with a maximum of N digits
        n_I = (n % s ** (I + 1)) // (s ** I)
        A[n, n] = Omega.dot(n_I + 1 / 2)

    return A


Q = array([q(n) for n in range(N)])
P = array([p(n) for n in range(N)])
QQ = array([q(n).dot(q(m)) for n in range(N) for m in range(n, N)])
PP = array([p(n).dot(p(m)) for n in range(N) for m in range(n, N)])
H = h()

if print_strings:
    print(colored(f"H: {H}", "blue"))
    print(colored(f"P: {P}", "blue"))
    print(colored(f"Q: {Q}", "blue"))


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


def assert_density(dens):
    assert is_hermitian(dens)
    assert is_unit_trace(dens)
    assert is_pos_def(dens)


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


def dissipation(rho):
    Sum = 0

    # Sum over all the nodes
    for n in range(N):
        Sum += 1j * Gamma[n] * (com(Q[n], anti(P[n], rho)) - com(P[n], anti(Q[n], rho)))
        Sum += D[n] * (com(Q[n], com(Q[n], rho)) - com(P[n], com(P[n], rho)) / Omega[n] ** 2)

    return - 1 / 4 * Sum


# The Liouvillian superoperator
def liouvillian(rho):
    return hamiltonian(rho) + dissipation(rho)
