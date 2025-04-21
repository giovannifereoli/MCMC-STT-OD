import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import math
from itertools import product
from scipy.stats import norm
from MCMC import MCMCModel

# NOTE for Jay:
# Increasing order of STT increase precision! OMG!!!


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


def propagate(x0, mu, order, t_eval, show_progress=True, **options):
    f_func, A_func, B_funcs = generate_stt_functions(mu, order)

    # initial augmented state
    Y0 = list(x0)
    Y0 += list(np.eye(6).flatten())
    for k in range(2, order + 1):
        Y0 += [0.0] * (6 ** (k + 1))
    Y0 = np.array(Y0, float)

    t_start = t_eval[0]
    t_end = t_eval[-1]

    last_print = -1

    def wrapped_rhs(t, y):
        nonlocal last_print
        if show_progress:
            progress = int(100 * (t - t_start) / (t_end - t_start))
            if progress > last_print:
                bar = "█" * (progress // 2) + "-" * (50 - progress // 2)
                print(f"\rProgress |{bar}| {progress:.1f}% - t = {t:.2f}", end="")
                last_print = progress
        return stt_ode(t, y, mu, order, f_func, A_func, B_funcs)

    sol = solve_ivp(
        fun=wrapped_rhs,
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


if __name__ == "__main__":
    # Constants
    mu = 398600.4418  # Earth's gravitational parameter [km^3/s^2]
    x0_ref = np.array([757.7, 5222.607, 4851.5, 2.21321, 4.67834, -5.3713])
    # x0_ref = np.array([7000, 0, 0, 0, 7.5, 1.0])
    order = 3
    t_obs = np.linspace(0, 2 * 3600, 5)

    # Simulate true deviation from nominal initial state
    true_dev = np.array([10, -5, 1, 0.1, -0.5, 0.8])  # km / km/s
    sol_ref, stts_ref = propagate(x0_ref, mu, order, t_obs, rtol=1e-12, atol=1e-14)
    _, x_true = propagate_deviation(sol_ref, stts_ref, true_dev, order=order)

    # Generate observations (range + range-rate) with noise
    sigma_range = 1e-3  # 1 mm
    sigma_rangerate = 1e-6  # 1 mm/s
    range_obs = np.linalg.norm(x_true[:, :3], axis=1) + np.random.normal(
        0, sigma_range, size=len(t_obs)
    )
    rangerate_obs = np.linalg.norm(x_true[:, 3:], axis=1) + np.random.normal(
        0, sigma_rangerate, size=len(t_obs)
    )
    y_obs = np.hstack([range_obs, rangerate_obs])

    # Residual function for MCMC
    def residuals(delta_x0):
        _, x_est = propagate_deviation(sol_ref, stts_ref, delta_x0, order=order)
        range_model = np.linalg.norm(x_est[:, :3], axis=1)
        rangerate_model = np.linalg.norm(x_est[:, 3:], axis=1)
        y_model = np.hstack([range_model, rangerate_model])
        weights = np.hstack(
            [
                np.full(len(t_obs), sigma_range),
                np.full(len(t_obs), sigma_rangerate),
            ]
        )
        return (y_obs - y_model) / weights

    # Priors and initial guess
    priors = [norm(loc=0, scale=5) for _ in range(6)]
    initial_guess = np.zeros(6)

    # Run MCMC
    # NOTE: We recommend using hundreds of walkers, collecting at least 10× the
    # autocorrelation time in samples, and allocating a burn‑in period of 10–25% of the chain.
    model = MCMCModel(
        residuals_func=residuals,
        initial_params=initial_guess,
        param_priors=priors,
        observed_data=y_obs,
    )
    model.run(n_samples=3000, n_walkers=128, burn_in=500)
    model.plot_convergence()
    model.plot_postfit_residuals()
    model.plot_log_likelihood()
    model.summary()
