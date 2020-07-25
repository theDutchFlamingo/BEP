from moments_plotter import *
from numpy import array
from tqdm import tqdm
from common import scream
from operators import *
import warnings
warnings.filterwarnings("ignore")


rho_0 = rho_init(rho_q_basic())
assert_density(rho_0)
P = array([p(n) for n in range(N)])
Q = array([q(n) for n in range(N)])
H = h()

# print("P:", P)
# print("Q:", Q)
# print("H:", H)


# The Hamiltonian superoperator, for testing purposes
def hamiltonian(rho):
    return -1j * com(H, rho)


# The Liouvillian superoperator
def liouvillian(rho):
    Sum = 0

    # Sum over all the nodes
    for n in range(N):
        Sum += 1j * Gamma[n] * (com(Q[n], anti(P[n], rho)) - com(P[n], anti(Q[n], rho)))
        Sum += D[n] * (com(Q[n], com(Q[n], rho)) - com(P[n], com(P[n], rho)) / Omega[n] ** 2)

    return -1j * com(H, rho) - 1 / 4 * Sum


# print(np.linalg.eigvalsh(liouville(rho_0)))

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
        
        Rho = rk4(liouvillian, Rho)
        
        # assert_density(Rho)
        assert is_hermitian(Rho)
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
