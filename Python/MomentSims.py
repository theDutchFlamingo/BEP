from moments_plotter import *

A = np.diag(-np.tile(Gamma, 2)/2)
np.fill_diagonal(A[:, N:], -Omega ** 2)
np.fill_diagonal(A[N:], 1)

# The number of unique auto/cross-correlations for one of QiQj, PiPj, PiQj
B = np.zeros([2*M + N**2, 2*M + N**2])

for i in range(N):
    for j in range(N):
        # Some handy indices:
        red = num(i, j) if i <= j else num(j, i)  # Reduced index for the Q_i Q_j and P_i P_j operators
        ind = N*i + j + 2*M  # Full (unreduced) index for the P_i Q_j operator
        
        # The diagonal terms for QQ, PP and QP
        B[red, red] = B[red + M, red + M] = B[ind, ind] = -(Gamma[i] + Gamma[j])/2
        
        # The non-diagonal terms:
        B[red, ind] += 1 / (1 if i == j else 2)  # PQ influencing QQ
        B[red + M, ind] -= Omega[j] ** 2 / (1 if i == j else 2)  # PQ influencing PP
        B[ind, red] = -2 * Omega[i] ** 2  # QQ influencing PQ
        B[ind, red + M] = 2  # PP influencing PQ

    
# The constant vector to be added (dY/dt = B*Y + L)
L = np.zeros(2*M + N**2)
for i in range(N):
    L[num(i, i)] = + D[i]/2/Omega[i]**2  # I derived a - sign, but let's try +
    L[num(i, i) + M] = D[i]/2


def a(x):
    """
    Homogeneous equation with matrix A
    :param x: the vector
    :return: A * x
    """
    return A.dot(x)


def b(y):
    """
    Inhomogeneous equation with matrix B
    :param y: the vector
    :return: B * y + L
    """
    return B.dot(y) + L


def first_order():
    """
    Calculate and plot the first order moments
    :return the data which was plotted
    """
    p_0 = np.array([0.5, 1, -0.5])
    q_0 = np.array([-1, 0, 1])

    Q = P = np.zeros([N, it])

    Q[:, 0] = F.T.dot(q_0)
    P[:, 0] = p_0

    X = np.concatenate((P, Q))

    for t in range(1, it):
        X[:, t] = rk4(a, X[:, t - 1])

    P, Q = np.reshape(X, [2, N, it])

    q = F.dot(Q)
    
    return q, Q, P


def double_transform(mat: np.array):
    """
    For coordinate systems such as q1q1, q1q2, q2q2: transform each of these coordinates,
    based on a transformation in the normal domain (q1, q2)
    :param mat: the input matrix for the simple case
    :return: the matrix which can be used to transform the complicated vectors
    """
    size = tri(mat.shape[1])
    ret = np.zeros([size, size])
    
    for n in range(N):
        for m in range(n, N):
            for o in range(N):
                for p in range(N):
                    ret[num(n, m), num(o, p) if o <= p else num(p, o)] += mat[o, n]*mat[p, m]
                        
    return ret


def square_transform(mat: np.array):
    return np.kron(np.identity(N), mat)


def second_order():
    """
    Calculate and plot the second order moments
    :return the data which was plotted
    """
    qq_0 = np.array([0]*M)
    pp_0 = np.array([1, 1, -1, 1, -1, 1])
    pq_0 = np.array([0]*N**2)
    
    PP = np.zeros([M, it])
    QQ = np.zeros([M, it])
    PQ = np.zeros([N**2, it])
    QQ[:, 0] = double_transform(F.T).dot(qq_0)
    QQ[0, 0] = 0.25
    PP[:, 0] = pp_0
    PQ[:, 0] = square_transform(F.T).dot(pq_0)
    
    Y = np.concatenate((QQ, PP, PQ))
    
    for t in range(1, it):
        Y[:, t] = rk4(b, Y[:, t-1])
    
    QQ = Y[:M]
    PP = Y[M:2*M]
    PQ = Y[2*M:]
    
    qq = double_transform(F).dot(QQ)
    pp = PP
    pq = square_transform(F).dot(PQ)
    
    return qq, QQ, pp


def get_osc_data(func):
    pass


print("F:", F)
print("D:", D)
print("L:", L)
print("C:", C)
print("Gamma:", Gamma)
print("Omega:", Omega)

nodes, modes, momenta = first_order()
first_plot_separate(nodes, "q")
first_plot_separate(modes, "Q", True, momenta)

nodes, modes, momenta = second_order()
# second_plot_separate(nodes, "q")  # Not used in report
second_plot_separate(modes, "Q", True)
second_plot_diagonal(modes, "Q", True, momenta)
second_plot_cross(modes, "Q")
second_plot_u(modes, momenta, "Q", "P", True, True)
# second_plot_v(modes, momenta, "Q", "P", True, True)  # Not used in report
