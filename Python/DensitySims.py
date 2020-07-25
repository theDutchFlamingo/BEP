from moments_plotter import *
from numpy import array
from tqdm import tqdm
from operators import *
import warnings
warnings.filterwarnings("ignore")


Q = array([q(n) for n in range(N)])
P = array([p(n) for n in range(N)])
QQ = array([q(n).dot(q(m)) for n in range(N) for m in range(n, N)])
PP = array([p(n).dot(p(m)) for n in range(N) for m in range(n, N)])
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


# Rho = rho_init()
Rho = rho_alt()
assert is_density(Rho)
# Test if one EF iteration preserves the properties of a density matrix
assert is_density(Rho + dt*liouvillian(Rho))

recalculate = True

mom_info = {
    "Q": (N, Q),
    "P": (N, P),
    "QQ": (M, QQ),
    "PP": (M, PP),
}

moms = {key : None for key in mom_info.keys()}

if recalculate:
    for key in moms:
        moms[key] = zeros((mom_info[key][0], it))
    
    for t in tqdm(range(it)):
        for key in moms:
            Max, op = mom_info[key]
            moms[key][:, t] = array([expval(op[n], Rho) for n in range(Max)])
        
        Rho = rk4(liouvillian, Rho)
        
        # assert is_density(Rho)
        assert is_hermitian(Rho)
        assert is_unit_trace(Rho)
    
    for key in moms:
        np.save(f"ev_{key}.npy", moms[key])
else:
    for key in moms:
        moms[key] = np.load(f"ev_{key}.npy")


# Plots
if "Q" in moms:
    for i in range(N):
        print(moms["Q"][i])
    first_plot_separate(moms["Q"], "Q", True, moms["P"] if "P" in moms else None)
# if "P" in moms:
#     first_plot_separate(moms["P"], "P", True)
# if "QQ" in moms:
#     second_plot_separate(moms["QQ"], "Q", colorized=True)
