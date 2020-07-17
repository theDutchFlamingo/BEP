from numpy import sqrt, zeros, kron
from moments_common import *

s = 2  # The amount of states that we consider per node
K = s ** N  # The amount of states


# def p_new(i):
#     p_basic = np.diag()


# Momentum operator for node i
def p(i):
    A = zeros((K, K))

    for n in range(K):
        # Obtain n_i from n
        n_i = (n % s ** (i + 1)) // (s ** i)

        if n_i != s - 1:
            # The lowering operator is applied on the state above
            A[n, n + s ** i] = sqrt(n_i + i)

        if n_i != 0:
            # The raising  operator is applied on the state below
            A[n, n - s ** i] = -sqrt(n_i)

    return -1j * sqrt(Omega[i] / 2) * A


# Position operator for node i
def q(i):
    A = zeros((K, K))

    for n in range(K):
        # Obtain n_i from n
        n_i = (n % s ** (i + 1)) // (s ** i)

        if n_i != 0:
            # First the raising operator is applied on the state below
            A[n, n - s ** i] = sqrt(n_i)

        if n_i != s - 1:
            # Next the lowering operator is applied on the state above
            A[n, n + s ** i] = sqrt(n_i + i)

    return 1 / (sqrt(2 * Omega[i])) * A


def h():
    A = zeros((K, K))

    for n in range(K):
        I = np.arange(N)
        # Digit representation of n in base s with a maximum of N digits
        n_I = (n % s ** (I + 1)) // (s ** I)
        A[n, n] = Omega.dot(n_I + 1 / 2)
        print(n_I, " = ", A[n, n])

    return A
