import numpy as np
from common import tol
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True

N = 3
T_F = 75
it = 15000
it_med = it * 45//T_F
it_short = it * 15//T_F
dt = T_F/it
T = np.arange(0, T_F, dt)

# How many times smaller than max(Gamma) a Gamma_n
# should be to classify as noiseless
quasi_ratio = 20

omega = np.array([1.2, 1, 1.8])
# omega = np.ones(N)
temp = 10 * omega[1]
gamma = 0.07 * omega[1]

LambdaM3 = 0.4 * omega[1] * np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
LambdaM2 = 0.4 * omega[1] * np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])
Lambda = LambdaM2  # The adjacency matrix which we are currently investigating

k_m = np.max(np.sum(Lambda, 0))  # Constant to assure positivity
HO = np.diag(omega**2 + 2 * k_m)

# The normal mode frequencies and the rotation matrix diagonalizing the adjacency matrix
Omega, F = np.linalg.eigh(HO - Lambda)
Gamma = np.sum(F, 0)**2 * gamma
D = Gamma * Omega / np.tanh(Omega/2/temp)
C = Gamma * D / (4*Omega**2 + Gamma**2)


def tri(n):
    return n*(n+1)//2


def num(lesser, greater):
    """
    :param lesser: 
    :param greater: should be at least :param lesser
    :return: 
    """
    return tri(N - 1) - tri(N - 1 - lesser) + greater


A = np.diag(-np.tile(Gamma, 2)/2)
np.fill_diagonal(A[:, N:], -Omega ** 2)
np.fill_diagonal(A[N:], np.ones(N))

# The number of unique auto/cross-correlations for one of QiQj, PiPj, PiQj
M = tri(N)
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


def rk4(f, x):
    """
    An implementation of the classical Runge-Kutta method
    :param f: the function such that d/dt x = f(x)
    :param x: a data point
    :return: the next data point, based on RK4
    """
    k1 = f(x)
    k2 = f(x + dt*k1/2)
    k3 = f(x + dt*k2/2)
    k4 = f(x + dt*k3)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)


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


def color_pick(n):
    color = "red"

    if abs(Gamma[n]) < tol:
        color = "mediumspringgreen"
    elif abs(Gamma[n]) < max(Gamma) / quasi_ratio and not any(Gamma < tol):
        color = "orange"

    return color


def calc_u(pos, mom, n):
    U_0 = Omega[n]**2 * pos[num(n, n), 0] + mom[num(n, n), 0]
    return U_0 * np.exp(-Gamma[n] * T)


def calc_u_2(pos, mom, n):
    U_0 = Omega[n]**2 * pos[num(n, n), 0] + mom[num(n, n), 0]
    return (U_0-D[n]/Gamma[n]) * np.exp(-Gamma[n] * T) + D[n]/Gamma[n]  # Last term if the + is correct


def calc_v(pos, mom, n, r=None):
    m = num(n, n)
    U_0 = Omega[n]**2 * pos[m, 0] + mom[m, 0]
    B_n = C[n] - U_0 + 2*Omega[n]**2 * pos[m, 0]
    dQ_0 = (r[n, 0] if r is not None else 0) - Gamma[n] * pos[m, 0] - D[n]/2/Omega[n]**2
    A_n = (2*Omega[n]**2 * dQ_0 + (U_0 + 2*B_n) * Gamma[n])/2/Omega[n]
    return -C[n] + np.exp(-Gamma[n] * T) * (A_n * np.sin(2*Omega[n]*T) +
                                            B_n * np.cos(2*Omega[n]*T))


def calc_v_2(pos, mom, n, r=None):
    m = num(n, n)
    B_n = Omega[n]**2 * pos[m, 0] - mom[m, 0]
    A_n = Omega[n] * (r[(N+1)*n, 0] if r is not None else 0)
    return np.exp(-Gamma[n]*T) * (A_n * np.sin(2*Omega[n]*T) +
                                  B_n * np.cos(2*Omega[n]*T))


def first_plot_combined(funcs, name):
    for n in range(N):
        plt.plot(T, funcs[n])

    plt.legend([f"Node {n + 1}" for n in range(N)])
    plt.ylabel(f"$\\left<{name}_i\\right>$")
    plt.xlabel("$t$")
    plt.show()
    
    
def first_plot_separate(funcs, name, colorized=False, mom=None):
    for n in range(N):
        plt.subplot(N, 1, n + 1)
        
        color = color_pick(n) if colorized else ""
        
        plt.plot(T, funcs[n], color)
        
        if mom is not None:
            d = funcs[n, 0]
            c = (2 * mom[n, 0] + Gamma[n] * d)/(2*Omega[n])
            plt.plot(T, np.exp(-Gamma[n]*T/2) * (c * np.sin(Omega[n]*T) +
                                                 d * np.cos(Omega[n]*T)), "k:")
            plt.legend(["Solution", "Expected"])
        
        plt.ylabel(f"$\\left<{name}_{n + 1}\\right>$")
        
    plt.xlabel("$t$")
    plt.show()


def second_plot_combined(funcs, name):
    for m in range(M):
        plt.plot(T[:it_med], funcs[m, :it_med])

    plt.legend([f"Node {(n, m)}" for n in range(N) for m in range(n, N)])
    plt.ylabel(f"$\\left<{name}_i {name}_j\\right>$")
    plt.xlabel("$t$")
    plt.show()


def second_plot_separate(funcs, name, colorized=False):
    for n in range(N):
        for m in range(n, N):
            plt.subplot(N, N, 1 + N*n + m)
            
            color = ""
            
            if n == m:
                color = color_pick(n) if colorized else ""

                plt.xlabel("$t$")
                plt.ylabel(f"$\\left<{name}_{n + 1}^2\\right>$")
            else:
                plt.ylabel(f"$\\left<{name}_{n + 1} {name}_{m + 1}\\right>$")

            plt.plot(T[:it_short], funcs[num(n, m), :it_short], color)
    
    plt.show()


def second_plot_diagonal(pos, name, colorized=False, mom=None, r=None):
    for n in range(N):
        plt.subplot(N, 1, n + 1)

        color = color_pick(n) if colorized else ""
        
        plt.plot(T[:it_med], pos[num(n, n), :it_med], color)
        
        if mom is not None:
            plt.plot(T[:it_med], ((calc_u_2(pos, mom, n) + calc_v_2(pos, mom, n, r))/2/Omega[n]**2)[:it_med], "k:")
            plt.legend(["Solution", "Expected"])
        
        plt.ylabel(f"$\\left<{name}_{n + 1}^2\\right>$")

    plt.xlabel("$t$")
    plt.show()


def second_plot_cross(funcs, name):
    index = 1
    
    for n in range(N):
        for m in range(n + 1, N):
            plt.subplot(M - N, 1, index)
            index += 1
            plt.plot(T[:it_med], funcs[num(n, m), :it_med])
            plt.ylabel(f"$\\left<{name}_{n + 1}{name}_{m + 1}\\right>$")
    
    plt.xlabel("$t$")
    plt.show()


def second_plot_u(qq, pp, name1, name2, colorized=False, expect=False):
    for n in range(N):
        plt.subplot(N, 1, n + 1)

        color = color_pick(n) if colorized else ""
        
        plt.plot(T[:it_med], (Omega[n]**2 * qq[num(n, n)] + pp[num(n, n)])[:it_med], color)
        plt.ylabel(f"$\\left<\\Omega_{n + 1}^2 {name1}_{n + 1}^2 + {name2}_{n + 1}^2\\right>$")
        
        if expect:
            plt.plot(T[:it_med], calc_u_2(qq, pp, n)[:it_med], "k:")
            plt.legend(["Solution", "Expected"])
            
    plt.xlabel("$t$")
    plt.show()


def second_plot_v(qq, pp, name1, name2, colorized=False, expect=False):
    for n in range(N):
        plt.subplot(N, 1, n + 1)

        color = color_pick(n) if colorized else ""

        plt.plot(T[:it_med], (Omega[n]**2 * qq[num(n, n)] - pp[num(n, n)])[:it_med], color)
        plt.plot(T[:it_med], -C[n] * np.ones(it)[:it_med],
                 linestyle="dashed", color="lightgray")
        plt.ylabel(f"$\\left<\\Omega_{n + 1}^2 {name1}_{n + 1}^2 - {name2}_{n + 1}^2\\right>$")

        if expect:
            plt.plot(T[:it_med], calc_v(qq, pp, n)[:it_med], "k:")
            plt.legend(["Solution", "Offset (expected)", "Expected"])

    plt.xlabel("$t$")
    plt.show()


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
first_plot_separate(modes, "Q", True)

nodes, modes, momenta = second_order()
# second_plot_separate(nodes, "q")  # Not used in report
second_plot_separate(modes, "Q", True)
second_plot_diagonal(modes, "Q", True, momenta)
second_plot_cross(modes, "Q")
second_plot_u(modes, momenta, "Q", "P", True, True)
# second_plot_v(modes, momenta, "Q", "P", True, True)  # Not used in report
