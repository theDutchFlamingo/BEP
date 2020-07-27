from moments_plotter import *
from tqdm import tqdm
from operators import *
import warnings
warnings.filterwarnings("ignore")


# Rho = rho_init()
Rho = rho_alt()
assert is_density(Rho)
# Test if one EF iteration preserves the properties of a density matrix
assert is_density(Rho + dt*liouvillian(Rho))

# Defines which moments we will calculate, also defines
# the number of moments and the corresponding operator:
# "<name>": (<number of moments>, <operator>)
mom_info = {
    "Q": (N, Q),
    "P": (N, P),
    "QQ": (M, QQ),
    "PP": (M, PP),
}

# Where we will store the moments
moms = {key : zeros((mom_info[key][0], it)) for key in mom_info.keys()}

recalculate = True
require_density = False

print(expval(Q[0], hamiltonian(Rho)))
print(expval(Q[0], dissipator(Rho)))
print(expval(Q[0], liouvillian(Rho)))

if recalculate:
    for t in tqdm(range(it)):
        # Calculate the moments that we want
        for key in moms:
            Max, op = mom_info[key]
            moms[key][:, t] = array([expval(op[n], Rho) for n in range(Max)])
        
        # Perform one rk4 iteration
        Rho = rk4(liouvillian, Rho)
        
        # Optional assertions to check that Rho is still a density matrix
        if require_density:
            assert is_density(Rho)
        # assert is_hermitian(Rho)
        # assert is_unit_trace(Rho)
    
    for key in moms:
        np.save(f"ev_{key}.npy", moms[key])
else:
    for key in moms:
        moms[key] = np.load(f"ev_{key}.npy")


# Plots
if "Q" in moms:
    m_Q = moms["Q"]
    print("Q(0):", m_Q[0, 0])
    print("dQ1(0)/dt:", (m_Q[0, 1] - m_Q[0, 0])/dt)
    
    if print_strings:
        for i in range(N):
            print(m_Q[i])
    # first_plot_separate(m_Q, "Q", True, moms["P"] if "P" in moms else None)
# if "P" in moms:
#     first_plot_separate(moms["P"], "P", True)
# if "QQ" in moms:
#     second_plot_separate(moms["QQ"], "Q", colorized=True)
