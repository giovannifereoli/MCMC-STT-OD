import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
from itertools import product


def generate_stt_functions(mu, order):
    """
    Symbolically generate f, A and B_k up to arbitrary 'order'.
    """
    # 1) state symbols
    x_syms = sp.symbols("x y z vx vy vz")
    x, y, z, vx, vy, vz = x_syms
    mu_sym = sp.Float(mu)
    r = sp.sqrt(x**2 + y**2 + z**2)

    # 2) two‑body dynamics
    f_sym = sp.Matrix(
        [vx, vy, vz, -mu_sym * x / r**3, -mu_sym * y / r**3, -mu_sym * z / r**3]
    )

    # 3) STM generator
    X = sp.Matrix(x_syms)
    B_syms = {1: f_sym.jacobian(X)}

    # 4) build B_syms[2..order]
    for k in range(2, order + 1):
        shape = (6,) * (k + 1)
        Bk = sp.MutableDenseNDimArray.zeros(*shape)

        for idx in product(range(6), repeat=k + 1):  # (i, j1, ..., jk)
            i, *js = idx
            deriv = sp.diff(f_sym[i], *[x_syms[j] for j in js])
            Bk[idx] = deriv

        B_syms[k] = Bk

    # 5) lambdify: convert each B_syms[k] → nested lists
    f_func = sp.lambdify(x_syms, f_sym, "numpy")
    A_func = sp.lambdify(x_syms, B_syms[1], "numpy")
    B_funcs = {
        k: sp.lambdify(x_syms, B_syms[k].tolist(), "numpy") for k in range(2, order + 1)
    }

    return f_func, A_func, B_funcs


def stt_ode(t, Y, mu, order, f_func, A_func, B_funcs):
    x = Y[:6]
    # 1) dynamics + STM
    dx = np.array(f_func(*x), float).reshape(6)
    offset = 6
    Phi = Y[offset : offset + 36].reshape(6, 6)
    offset += 36
    A = np.array(A_func(*x), float)
    dPhi = A @ Phi

    # 2) unpack every T_k once
    Ts = {1: Phi}
    for k in range(2, order + 1):
        size = 6 ** (k + 1)
        Tk = Y[offset : offset + size].reshape((6,) + (6,) * k)
        Ts[k] = Tk
        offset += size

    derivs = [*dx, *dPhi.flatten()]

    # 3) build each dT_k with all partitions
    for k in range(2, order + 1):
        # start with A·T_k
        dTk = np.tensordot(A, Ts[k], axes=(1, 0))

        # full‑order term: B_k(Φ,...,Φ)
        Bk = np.array(B_funcs[k](*x), float).reshape((6,) + (6,) * k)
        term = Bk
        for _ in range(k):
            term = np.tensordot(term, Phi, axes=(1, 0))
        dTk += term

        # mixed lower‑order terms
        # for every 2 ≤ m < k, we have comb(k,m) ways to choose
        # which slots get a T_{k-m+1} instead of Φ
        for m in range(2, k):
            coef = math.comb(k, m)
            Bm = np.array(B_funcs[m](*x), float).reshape((6,) + (6,) * m)

            # first contract one slot with T_{k-m+1}
            mixed = np.tensordot(Bm, Ts[k - m + 1], axes=(1, 0))
            # then contract the remaining m−1 slots with Φ
            for _ in range(m - 1):
                mixed = np.tensordot(mixed, Phi, axes=(1, 0))

            dTk += coef * mixed

        derivs += list(dTk.flatten())

    return np.array(derivs, float)


def propagate(x0, mu, order, t_eval, **options):
    f_func, A_func, B_funcs = generate_stt_functions(mu, order)

    # initial augmented state
    Y0 = list(x0)
    Y0 += list(np.eye(6).flatten())
    for k in range(2, order + 1):
        Y0 += [0.0] * (6 ** (k + 1))
    Y0 = np.array(Y0, float)

    sol = solve_ivp(
        fun=lambda t, Y: stt_ode(t, Y, mu, order, f_func, A_func, B_funcs),
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        y0=Y0,
        **options,
    )

    n_steps = sol.y.shape[1]
    stts = {}
    offset = 6

    # k = 1 is the STM (6×6)
    phi_flat = sol.y[offset : offset + 36, :]  # shape (36, n_steps)
    phi_all = phi_flat.reshape(6, 6, n_steps)  # shape (6,6,n_steps)
    stts[1] = np.transpose(phi_all, (2, 0, 1))  # shape (n_steps,6,6)
    offset += 36

    # higher orders k=2..order
    for k in range(2, order + 1):
        block_size = 6 ** (k + 1)
        Tk_flat = sol.y[offset : offset + block_size, :]  # (6^(k+1), n_steps)
        # reshape into (6,6,...,6,n_steps)
        shape = (6,) + (6,) * k + (n_steps,)
        Tk_all = Tk_flat.reshape(shape)
        # move time axis to front → shape (n_steps,6,6,...,6)
        stts[k] = np.moveaxis(Tk_all, -1, 0)
        offset += block_size

    # return both the raw solution and the unpacked STTs
    return sol, stts


def propagate_deviation(sol, stts, delta_x0, order):
    """
    Given:
      sol       : ODE solution from propagate(…)
      stts      : dict of STTs from propagate(…)  stts[k].shape = (n_steps,6,6,...,6)
      delta_x0  : initial deviation, shape (6,)
      order     : max STT order to include
    Returns:
      delta      : ndarray (n_steps,6) of propagated deviations
      x_nom+dev  : ndarray (n_steps,6) of perturbed trajectory
    """
    n_steps = sol.y.shape[1]
    delta = np.zeros((n_steps, 6))
    x_nom = sol.y[:6, :].T  # shape (n_steps,6)

    # for each time step
    for t in range(n_steps):
        d = np.zeros(6)

        # k = 1 term: Φ • δx0
        Phi = stts[1][t]  # (6,6)
        d += Phi.dot(delta_x0)

        # higher orders
        for k in range(2, order + 1):
            Tk = stts[k][t]  # shape (6,6,...,6) with k+1 dims
            # contract Tk with δx0 repeated k times
            term = Tk
            for _ in range(k):
                term = np.tensordot(term, delta_x0, axes=(1, 0))
            d += term / math.factorial(k)

        delta[t] = d

    return delta, x_nom + delta


# === Compute orbital energy for nominal and perturbed trajectories ===
def specific_energy(x, mu):
    r_vec = x[:, :3]
    v_vec = x[:, 3:]
    r_norm = np.linalg.norm(r_vec, axis=1)
    v_norm = np.linalg.norm(v_vec, axis=1)
    return 0.5 * v_norm**2 - mu / r_norm


if __name__ == "__main__":
    μ = 398600.4418  # km^3/s^2
    x0 = np.array([7000, 0, 0, 0, 7.5, 1.0])
    order = 2
    t_eval = np.linspace(0, 2 * 3600, 1000)

    sol, stts = propagate(x0, μ, order, t_eval, rtol=1e-12, atol=1e-14, method="RK45")

    # Plot radius
    r = np.linalg.norm(sol.y[:3, :].T, axis=1)
    plt.plot(sol.t, r)
    plt.xlabel("Time (s)")
    plt.ylabel("Radius (km)")
    plt.title(f"Two-body orbit, STT order={order}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # and you have some small deviation, e.g.:
    delta_x0 = np.array([0.1, 0, 0, 0, 0, 0])  # 100 m along x

    # propagate that deviation
    delta, x_pert = propagate_deviation(sol, stts, delta_x0, order=order)
    sol_pert, stts_pert = propagate(
        x0 + delta_x0, μ, order, t_eval, rtol=1e-12, atol=1e-14, method="RK45"
    )

    # plot propagated vs STT-propagated orbit
    t = sol_pert.t
    plt.plot(t, sol_pert.y[0, :] - x_pert[:, 0], "--", label="Propagated - STT")
    plt.xlabel("Time (s)")
    plt.ylabel("x (km)")
    plt.title(f"Propagated vs STT-Propagated orbit ({order}th-order STT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    """
    # plot energy difference
    energy_perturbed = specific_energy(x_pert, μ)
    energy_diff = (energy_perturbed - energy_perturbed[0]) / energy_perturbed[0]

    # Plot energy difference
    plt.plot(t, energy_diff)  # Convert to m^2/s^2 if desired
    plt.xlabel("Time (s)")
    plt.ylabel("Δ Energy / Energy (-)")
    plt.title("Energy deviation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Monte Carlo with propagate_deviation
    N = 1000
    sigma_dev = 0.01  # km (10 m)

    # draw N random initial deviations
    delta0_samples = np.random.randn(N, 6) * sigma_dev

    # storage for final deviations
    final_delta = np.zeros((N, 6))

    for i in range(N):
        # propagate this deviation
        delta, x_pert = propagate_deviation(sol, stts, delta0_samples[i], order=order)
        # keep the deviation at the last time step
        final_delta[i] = delta[-1]
        print("", i, "of", N, "done")

    # nominal final position
    x_nom_final = sol.y[:3, -1]  # shape (3,)

    # sample points: nominal + Monte Carlo deviations (position only)
    pts = x_nom_final + final_delta[:, :3]  # shape (N,3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, alpha=0.4, label="samples")
    # highlight nominal
    ax.scatter(
        x_nom_final[0],
        x_nom_final[1],
        x_nom_final[2],
        color="r",
        marker="*",
        s=100,
        label="nominal",
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title(f"Final position cloud (N={N})")
    ax.legend()
    plt.tight_layout()
    plt.show()
    """
