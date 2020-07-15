import numpy as np

N = 3
T_F = 45
it = 150000
it_med = it * 45//T_F
it_short = it * 15//T_F
dt = T_F/it
T = np.arange(0, T_F, dt)

# How many times smaller than max(Gamma) a Gamma_n
# should be to classify as noiseless
quasi_ratio = 20

omega_1 = 1

LambdaM0 = np.array([[0]])
LambdaM1 = 0.4 * omega_1 * np.array([
    [0, 1],
    [1, 0]
])
LambdaM3 = 0.4 * omega_1 * np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
LambdaM2 = 0.4 * omega_1 * np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])
# The adjacency matrix which we are currently investigating
Lambda = LambdaM0 if N == 1 else LambdaM1 if N == 2 else LambdaM2

# omega = np.array([1.2, 1, 1.8]) if N == 3 else np.ones(N)
omega = np.ones(N)
temp = 10 * omega_1
gamma = 0.07 * omega_1

k_m = np.max(np.sum(Lambda, 0))  # Constant to assure positivity
HO = np.diag(omega**2 + 2 * k_m)

# The normal mode frequencies and the rotation matrix diagonalizing the adjacency matrix
Omega, F = np.linalg.eigh(HO - Lambda)
Gamma = np.sum(F, 0)**2 * gamma
D = Gamma * Omega / np.tanh(Omega/2/temp)
C = Gamma * D / (4*Omega**2 + Gamma**2)

print("D:", D)
print("F:", F)
print("C:", C)
print("Gamma:", Gamma)
print("Omega:", Omega)


def tri(n):
    return n*(n+1)//2


def num(lesser, greater):
    """
    :param lesser: 
    :param greater: should be at least :param lesser
    :return: 
    """
    return tri(N - 1) - tri(N - 1 - lesser) + greater


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
