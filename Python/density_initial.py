from operators import *


def kronecker_power(base, exp):
    if exp == 1:
        return base
    elif exp > 1:
        return kron(kronecker_power(base, exp - 1), base)
    else:
        raise ValueError


def kronecker(arr):
    ret = 1
    for mat in arr:
        ret = kron(ret, mat)
    return ret


## Initial conditions
# The s-level density matrix of one node with correct Q_n-eigenvalue
def rho_q(n: int):
    ret = zeros((s, s))
    np.fill_diagonal(ret, 2)
    if s > 1:
        np.fill_diagonal(ret[1:], 1)
        np.fill_diagonal(ret[:, 1:], 1)

    ret *= q_0[n] / expval(q1(n), ret)
    np.fill_diagonal(ret, np.max(abs(ret)) * 2)
    return ret


# The s-level density matrix of one nodes with correct P_n-eigenvalue
def rho_p(n: int):
    ret = zeros((s, s), dtype=complex)
    if s > 1:
        np.fill_diagonal(ret[1:], 1j)
        np.fill_diagonal(ret[:, 1:], -1j)

    ret *= p_0[n] / expval(p1(n), ret)
    np.fill_diagonal(ret, np.max(abs(ret)) * 2)
    return ret


# The s-level density matrix of several nodes with correct eigenvalues for all first-order operators
def rho_init():
    # print(expval(Q[0], kronecker([rho_q(n) for n in range(N)])))
    for n in range(N):
        print(is_pos_def(rho_p(n)))
        print(is_pos_def(rho_q(n)))
    
    ret = sum([kron(kron(np.identity(s**(N - n - 1)), rho_q(n) + rho_p(n)), np.identity(s**n))
               for n in range(N)])
    
    # np.fill_diagonal(ret, 1/K)
    
    return ret / trace(ret)


def rho_alt():
    ret = np.zeros((K, K), dtype=float)

    for n in range(N):
        d = 2*sqrt(s-1)/sqrt(2 * Omega[n])
        Q = 1/sqrt(2 * Omega[n]) * (an() + cr())
        np.fill_diagonal(Q, d)

        ret += kron(kron(np.identity(s**(N - n - 1)), Q), np.identity(s**n))

    return ret / trace(ret)