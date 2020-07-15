from moments_common import *
from common import tol
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

M = tri(N)


def color_pick(n):
    color = "red"

    if abs(Gamma[n]) < tol:
        color = "mediumspringgreen"
    elif abs(Gamma[n]) < max(Gamma) / quasi_ratio and not any(Gamma < tol):
        color = "orange"

    return color


def calc_u(pos, mom, n, contains_all=True):
    m = num(n, n) if contains_all else n
    U_0 = Omega[n]**2 * pos[m, 0] + mom[m, 0]
    return U_0 * np.exp(-Gamma[n] * T)


# If the + is correct
def calc_u_2(pos, mom, n, contains_all=True):
    m = num(n, n) if contains_all else n
    U_0 = Omega[n]**2 * pos[m, 0] + mom[m, 0]
    return (U_0-D[n]/Gamma[n]) * np.exp(-Gamma[n] * T) + D[n]/Gamma[n]


def calc_v(pos, mom, n, contains_all=True, r=None):
    m = num(n, n) if contains_all else n
    U_0 = Omega[n]**2 * pos[m, 0] + mom[m, 0]
    B_n = C[n] - U_0 + 2*Omega[n]**2 * pos[m, 0]
    dQ_0 = (r[n, 0] if r is not None else 0) - Gamma[n] * pos[m, 0] - D[n]/2/Omega[n]**2
    A_n = (2*Omega[n]**2 * dQ_0 + (U_0 + 2*B_n) * Gamma[n])/2/Omega[n]
    return -C[n] + np.exp(-Gamma[n] * T) * (A_n * np.sin(2*Omega[n]*T) +
                                            B_n * np.cos(2*Omega[n]*T))


# If the + is correct
def calc_v_2(pos, mom, n, contains_all=True, r=None):
    m = num(n, n) if contains_all else n
    B_n = Omega[n]**2 * pos[m, 0] - mom[m, 0]
    A_n = Omega[n] * (r[(N+1)*n, 0] if r is not None else 0)
    return np.exp(-Gamma[n]*T) * (A_n * np.sin(2*Omega[n]*T) +
                                  B_n * np.cos(2*Omega[n]*T))


# If the + sign is correct
# calc_u = calc_u_2
# calc_v = calc_v_2


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


def second_plot_diagonal(pos, name, colorized=False,
                         contains_all=True, mom=None, r=None):
    for n in range(N):
        m = num(n, n) if contains_all else n
        
        plt.subplot(N, 1, n + 1)

        color = color_pick(n) if colorized else ""

        plt.plot(T[:it_med], pos[m, :it_med], color)

        if mom is not None:
            plt.plot(T[:it_med], ((calc_u(pos, mom, n, contains_all) +
                                   calc_v(pos, mom, n, contains_all, r))/2/Omega[n]**2)[:it_med], "k:")
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
            plt.plot(T[:it_med], calc_u(qq, pp, n)[:it_med], "k:")
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
