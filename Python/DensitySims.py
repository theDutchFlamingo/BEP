from moments_plotter import *
from tqdm import tqdm
from operators import *
import warnings
warnings.filterwarnings("ignore")


Rho = rho_init()
# Rho = rho_alt()
assert is_density(Rho)
# Test if one EF iteration preserves the properties of a density matrix
# If not, this means that the time step should be taken smaller
assert_density(Rho + dt*liouvillian(Rho))

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
require_density = True
loss_its = [-1, -1, -1]  # The iteration numbers at which each property is lost

if recalculate:
    for t in tqdm(range(it)):
        # Calculate the moments that we want
        for key in moms:
            Max, op = mom_info[key]
            moms[key][:, t] = array([expval(op[n], Rho) for n in range(Max)])
        
        # Perform one rk4 iteration
        Rho = rk4(liouvillian, Rho)
        
        # Optional assertions to check when Rho is no longer a density matrix
        if require_density:
            if not is_hermitian(Rho) and loss_its[0] == -1:
                loss_its[0] = t
            if not is_pos_def(Rho) and loss_its[1] == -1:
                loss_its[1] = t
            if not is_unit_trace(Rho) and loss_its[2] == -1:
                loss_its[2] = t
    
    for key in moms:
        np.save(f"ev_{key}.npy", moms[key])
else:
    for key in moms:
        moms[key] = np.load(f"ev_{key}.npy")


if require_density:
    if loss_its[0] != -1:
        print("Hermicity lost at:", loss_its[0])
    if loss_its[1] != -1:
        print("Positivity lost at:", loss_its[1])
    if loss_its[2] != -1:
        print("Unit trace lost at:", loss_its[2])


# Plots
if "Q" in moms:
    m_Q = moms["Q"]
    
    if print_strings:
        [print(m_Q[n]) for n in range(N)]
    
    first_plot_separate(m_Q, "Q", True, moms["P"] if "P" in moms else None)
    # first_plot_rel_error(m_Q, "Q", moms["P"] if "P" in moms else None)
# 
# if "P" in moms:
#     first_plot_separate(moms["P"], "P", True)
# 
# if "QQ" in moms:
#     second_plot_separate(moms["QQ"], "Q", colorized=True)
