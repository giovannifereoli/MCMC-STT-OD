"""
Self-contained benchmark script (NO classes) in the "MCMC-chain view":

- Two-body dynamics (6D state) with SymPy-generated STT tensors (order 1..3)
- RA/DEC angular measurements from a fixed observer
- Run YOUR MCMCModel on an STT surrogate likelihood (order chosen by you)
- Then benchmark STT orders {1,2,3} against FULL nonlinear likelihood on samples from that chain

CRITICAL NOTES (re: your pickling errors):
- If your MCMCModel/emcee uses process-based multiprocessing (mp.Pool) you WILL likely hit
  pickling errors because SymPy lambdify functions are not picklable under spawn.
- Easiest fixes (choose ONE):
    (A) Run with 1 core / no multiprocessing in MCMCModel (preferred simplest)
    (B) Modify MCMCModel to use ThreadPool (multiprocessing.dummy.Pool) instead of mp.Pool

This script keeps residuals as plain defs in main, exactly as you requested.

Assumptions about your API:
- STTPropagatorND(order, f_func, A_func, B_funcs, n).propagate(x0, t_eval, rtol, atol, method) -> (sol, stts)
- prop.propagate_deviation(sol_ref, stts_ref, delta0) -> (err?, x_est_hist)
- MCMCModel(residuals_func, initial_params, param_priors, observed_data)
  .setup_whitening_from_priors()
  .run(...)
- You can extract chain samples with one of:
    model.get_flat_samples(), or model.samples, or model.chain

"""

import time
import numpy as np
import sympy as sp
from itertools import product

from STTPropagationND import STTPropagatorND
from MCMC import MCMCModel

try:
    from tabulate import tabulate

    _HAS_TABULATE = True
except Exception:
    _HAS_TABULATE = False


# =========================
# Utils
# =========================
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rms(x):
    x = np.asarray(x).ravel()
    return float(np.sqrt(np.mean(x * x)))


def p95_abs(x):
    x = np.asarray(x).ravel()
    return float(np.percentile(np.abs(x), 95))


def loglike_from_norm_residuals(r_norm):
    r = np.asarray(r_norm).ravel()
    return float(-0.5 * np.sum(r * r))


def format_table(rows, headers):
    if _HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="github", floatfmt=".6g")
    colw = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)
    ]
    line = " | ".join(str(h).ljust(colw[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * colw[i] for i in range(len(headers)))
    out = [line, sep]
    for r in rows:
        out.append(" | ".join(str(r[i]).ljust(colw[i]) for i in range(len(headers))))
    return "\n".join(out)


def get_flat_chain_samples(model):
    if hasattr(model, "get_flat_samples") and callable(model.get_flat_samples):
        s = np.asarray(model.get_flat_samples())
        if s.ndim == 2:
            return s

    if hasattr(model, "samples"):
        s = np.asarray(model.samples)
        if s.ndim == 2:
            return s
        if s.ndim == 3:
            return s.reshape(-1, s.shape[-1])

    if hasattr(model, "chain"):
        c = np.asarray(model.chain)
        if c.ndim == 3:
            return c.reshape(-1, c.shape[-1])

    raise RuntimeError(
        "Could not extract chain samples from model. "
        "Expose model.samples (Ns,ndim), model.chain (nwalk,nstep,ndim), "
        "or implement model.get_flat_samples()."
    )


# =========================
# Two-body dynamics: symbolic STT tensors for any order (6D)
# =========================
def generate_stt_functions_twobody(order: int, mu: float):
    """
    6D dynamics:
      rdot = v
      vdot = -mu r / ||r||^3

    Returns f_func, A_func, B_funcs with signature f(*X, t).
    """
    t = sp.Symbol("t", real=True)
    x, y, z, vx, vy, vz = sp.symbols("x y z vx vy vz", real=True)
    X = sp.Matrix([x, y, z, vx, vy, vz])

    r = sp.sqrt(x**2 + y**2 + z**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3

    f = sp.Matrix([vx, vy, vz, ax, ay, az])
    A = f.jacobian(X)

    B_syms = {1: A}
    for k in range(2, order + 1):
        shape = (6,) * (k + 1)
        Bk = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(range(6), repeat=k + 1):
            i, *js = idx
            deriv = sp.diff(f[i], *[X[j] for j in js])
            Bk[idx] = deriv
        B_syms[k] = Bk

    args = (x, y, z, vx, vy, vz, t)
    f_func = sp.lambdify(args, f, "numpy")
    A_func = sp.lambdify(args, A, "numpy")
    B_funcs = {
        k: sp.lambdify(args, B_syms[k].tolist(), "numpy") for k in range(2, order + 1)
    }
    return f_func, A_func, B_funcs


# =========================
# Measurement model: RA/DEC
# =========================
def radec_from_los(los):
    x = los[:, 0]
    y = los[:, 1]
    z = los[:, 2]
    rxy2 = x * x + y * y
    rxy = np.sqrt(np.maximum(rxy2, 1e-30))
    ra = np.arctan2(y, x)  # (-pi, pi]
    dec = np.arctan2(z, rxy)  # (-pi/2, pi/2)
    return ra, dec


def y_model_from_state(x_hist, sc_pos):
    los = x_hist[:, :3] - sc_pos[None, :]
    ra, dec = radec_from_los(los)
    y = np.empty(2 * len(ra))
    y[0::2] = ra
    y[1::2] = dec
    return y


def make_noisy_measurements(x_true, sc_pos, sigma_ra, sigma_dec, seed=0):
    rng = np.random.default_rng(seed)
    y_true = y_model_from_state(x_true, sc_pos)
    y = y_true.copy()
    y[0::2] += rng.normal(0.0, sigma_ra, size=len(y[0::2]))
    y[1::2] += rng.normal(0.0, sigma_dec, size=len(y[1::2]))
    return y


# =========================
# Benchmark using chain support
# =========================
def benchmark_orders_on_chain(
    orders,
    mu,
    x0_ref1,
    t_grid,
    sc_pos,
    y_obs,
    sigma_ra,
    sigma_dec,
    chain_samples,  # (Ns,6) deltas about ref1
    n_eval=200,
    rtol=1e-10,
    atol=1e-12,
    method="LSODA",
):
    chain_samples = np.asarray(chain_samples, float)
    Ns_all = chain_samples.shape[0]
    if n_eval < Ns_all:
        idx = np.random.default_rng(0).choice(Ns_all, size=n_eval, replace=False)
        thetas = chain_samples[idx]
    else:
        thetas = chain_samples
    Ns = thetas.shape[0]

    # NONLINEAR propagator should be independent of STT order (order=1)
    f1, A1, B1 = generate_stt_functions_twobody(order=1, mu=mu)
    prop_nl = STTPropagatorND(order=1, f_func=f1, A_func=A1, B_funcs=B1, n=6)

    # weights (global for residuals_from_x)
    w = np.empty_like(y_obs, dtype=float)
    w[0::2] = sigma_ra
    w[1::2] = sigma_dec

    def residuals_from_x(x_hist):
        y_model = y_model_from_state(x_hist, sc_pos)
        res = np.empty_like(y_obs)
        res[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
        res[1::2] = y_obs[1::2] - y_model[1::2]
        return res / w  # normalized

    def residuals_nl(delta0):
        sol, _ = prop_nl.propagate(
            x0_ref1 + delta0, t_grid, rtol=rtol, atol=atol, method=method
        )
        x_est = sol.y[:6, :].T
        return residuals_from_x(x_est)

    rows = []
    for order in orders:
        # build stt funcs
        t0 = time.perf_counter()
        f_func, A_func, B_funcs = generate_stt_functions_twobody(order=order, mu=mu)
        t_build = time.perf_counter() - t0

        prop_stt = STTPropagatorND(
            order=order, f_func=f_func, A_func=A_func, B_funcs=B_funcs, n=6
        )

        # reference linearization for STT
        t0 = time.perf_counter()
        sol_ref, stts_ref = prop_stt.propagate(
            x0_ref1, t_grid, rtol=rtol, atol=atol, method=method
        )
        t_setup = time.perf_counter() - t0

        def residuals_stt(delta0):
            _, x_est = prop_stt.propagate_deviation(sol_ref, stts_ref, delta0)
            return residuals_from_x(x_est)

        dt_stt = 0.0
        dt_nl = 0.0
        dlogL = []
        dy_sig = []

        for th in thetas:
            # STT logL
            t0 = time.perf_counter()
            rS = residuals_stt(th)
            llS = loglike_from_norm_residuals(rS)
            dt_stt += time.perf_counter() - t0

            # NL logL
            t0 = time.perf_counter()
            rN = residuals_nl(th)
            llN = loglike_from_norm_residuals(rN)
            dt_nl += time.perf_counter() - t0

            dlogL.append(llS - llN)
            dy_sig.append(rS - rN)

        dlogL = np.asarray(dlogL)
        dy_sig = np.asarray(dy_sig)

        stt_ms = 1e3 * dt_stt / Ns
        nl_ms = 1e3 * dt_nl / Ns
        speedup = (dt_nl / dt_stt) if dt_stt > 0 else np.inf

        rows.append(
            [
                int(order),
                t_build,
                t_setup,
                stt_ms,
                nl_ms,
                speedup,
                rms(dlogL),
                p95_abs(dlogL),
                rms(dy_sig),
                p95_abs(dy_sig),
            ]
        )

    headers = [
        "order",
        "sym_build[s]",
        "stt_setup[s]",
        "STT logL [ms/sample]",
        "NL logL [ms/sample]",
        "speedup",
        "ΔlogL RMS",
        "p95(|ΔlogL|)",
        "RMS(Δy/σ)",
        "p95(|Δy/σ|)",
    ]
    print(
        "\n=== MCMC-chain view: STT surrogate vs FULL nonlinear (Two-body, RA/DEC) ==="
    )
    print(f"Evaluated chain samples: {Ns}")
    print(format_table(rows, headers))
    return rows


# =========================
# Publication-quality LaTeX plot (RMS only, vs FULL NL)
# =========================


def plot_chain_view_benchmark_latex(
    rows,
    out_png="stt_chain_view_rms.png",
    out_pdf="stt_chain_view_rms.pdf",
):
    """
    Fancy publication plot (LaTeX labels, RMS only).

    Panels:
      (a) Speedup vs FULL NL
      (b) RMS(Δ log L) vs FULL NL
      (c) RMS(Δ y / σ) vs FULL NL
    """
    import numpy as np
    import matplotlib.pyplot as plt

    rows = np.asarray(rows, float)

    order = rows[:, 0].astype(int)
    stt_ms = rows[:, 3]
    nl_ms = rows[:, 4]
    speedup = rows[:, 5]
    dlogL_rms = rows[:, 6]
    dy_rms = rows[:, 8]

    # -------------------------
    # Global style (journal-like)
    # -------------------------
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Consistent color palette (colorblind-safe)
    c_speed = "#1f77b4"  # blue
    c_logl = "#d62728"  # red
    c_meas = "#2ca02c"  # green
    c_nl = "#7f7f7f"  # gray

    fig = plt.figure(figsize=(11.5, 4.2))

    # =========================
    # (a) Speedup
    # =========================
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(order, speedup, "-o", color=c_speed, lw=2.5, ms=6)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"STT order")
    ax1.set_ylabel(r"Speedup $t_{\mathrm{NL}} / t_{\mathrm{STT}}$")
    ax1.set_title(r"(a) Computational gain")
    ax1.grid(True, which="both")

    # =========================
    # (b) Log-likelihood error
    # =========================
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(order, dlogL_rms, "-o", color=c_logl, lw=2.5, ms=6)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"STT order")
    ax2.set_ylabel(r"$\mathrm{RMS}(\Delta \log \mathcal{L})$")
    ax2.set_title(r"(b) Likelihood fidelity")
    ax2.grid(True, which="both")

    # =========================
    # (c) Measurement distortion
    # =========================
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(order, dy_rms, "-o", color=c_meas, lw=2.5, ms=6)
    ax3.set_yscale("log")
    ax3.set_xlabel(r"STT order")
    ax3.set_ylabel(r"$\mathrm{RMS}(\Delta y / \sigma)$")
    ax3.set_title(r"(c) Measurement-space distortion")
    ax3.grid(True, which="both")

    # =========================
    # Figure annotation
    # =========================
    nl_ref = np.median(nl_ms)
    fig.text(
        0.5,
        -0.02,
        rf"All errors referenced to FULL nonlinear likelihood "
        rf"(median cost $t_{{\mathrm{{NL}}}}\approx {nl_ref:.2f}\,\mathrm{{ms/sample}}$).",
        ha="center",
        va="top",
        fontsize=11,
    )

    fig.tight_layout()
    # fig.savefig(out_png, bbox_inches="tight")
    # fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # --------------------------
    # Scenario
    # --------------------------
    mu = 4.89044967462e-09  # km^3/s^2 (kept from your Bennu script)
    sc_pos = np.array([0.0, 0.0, 5.0], dtype=float)  # km

    T_hours = 6.0
    N_obs = 60
    t_grid = np.linspace(0.0, T_hours * 3600.0, N_obs)

    # noise (rad)
    sigma_ra = 1.0e-6
    sigma_dec = 1.0e-6

    # truth initial state
    R0 = 0.290
    x0_true = np.array([R0, 0.0, 0.0, 0.0, 1.8e-4, 1.0e-4], dtype=float)

    # propagate truth with order=1 (state-only)
    f1, A1, B1 = generate_stt_functions_twobody(order=1, mu=mu)
    prop_truth = STTPropagatorND(order=1, f_func=f1, A_func=A1, B_funcs=B1, n=6)
    sol_true, _ = prop_truth.propagate(
        x0_true, t_grid, rtol=1e-12, atol=1e-14, method="LSODA"
    )
    x_true = sol_true.y[:6, :].T

    # measurements
    y_obs = make_noisy_measurements(x_true, sc_pos, sigma_ra, sigma_dec, seed=10)

    # --------------------------
    # reference x0_ref1 (nonzero deviation)
    # --------------------------
    rng = np.random.default_rng(42)
    ref_sig_r = 0 * np.array([0.02, 0.02, 0.02])  # km
    ref_sig_v = 0 * np.array([1e-5, 1e-5, 1e-5])  # km/s
    x0_ref1 = x0_true - np.hstack(
        [rng.normal(scale=ref_sig_r), rng.normal(scale=ref_sig_v)]
    )

    print("\nTruth x0:", x0_true)
    print("Ref1  x0:", x0_ref1)
    print("Truth-ref1:", x0_true - x0_ref1)

    # --------------------------
    # Priors for your MCMCModel
    # --------------------------
    from scipy.stats import norm

    prior_sigma = np.hstack(
        [
            np.full(3, 1e-2),  # km
            np.full(3, 1e-5),  # km/s
        ]
    )
    priors = [norm(loc=0.0, scale=s) for s in prior_sigma]

    # --------------------------
    # Choose STT order used to generate the chain (surrogate posterior)
    # --------------------------
    stt_order_for_mcmc = 2

    f, A, B = generate_stt_functions_twobody(order=stt_order_for_mcmc, mu=mu)
    prop_mcmc = STTPropagatorND(
        order=stt_order_for_mcmc, f_func=f, A_func=A, B_funcs=B, n=6
    )

    rtol = 1e-10
    atol = 1e-12

    # reference propagation for STT surrogate
    sol_ref, stts_ref = prop_mcmc.propagate(
        x0_ref1, t_grid, rtol=rtol, atol=atol, method="LSODA"
    )

    # --------------------------
    # residuals (EXACTLY in-main defs like you requested)
    # --------------------------
    w = np.empty_like(y_obs, dtype=float)
    w[0::2] = sigma_ra
    w[1::2] = sigma_dec

    def residuals_from_x(x_hist):
        y_model = y_model_from_state(x_hist, sc_pos)
        res = np.empty_like(y_obs)
        # RA residual must wrap
        res[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
        res[1::2] = y_obs[1::2] - y_model[1::2]
        return res / w  # normalized

    def residuals_stt(delta0):
        _, x_est = prop_mcmc.propagate_deviation(sol_ref, stts_ref, delta0)
        return residuals_from_x(x_est)

    def residuals_nl(delta0):
        # WARNING: If prop_mcmc(order>1) augments the ODE with STT states,
        # this is NOT a fair "nonlinear" benchmark. We do NOT use this for timing below.
        sol, _ = prop_mcmc.propagate(
            x0_ref1 + delta0, t_grid, rtol=rtol, atol=atol, method="LSODA"
        )
        x_est = sol.y[:6, :].T
        return residuals_from_x(x_est)

    # --------------------------
    # Run YOUR MCMC on the surrogate likelihood
    # --------------------------
    print("\nRunning MCMC on STT surrogate residuals...")
    model = MCMCModel(
        residuals_func=residuals_stt,
        initial_params=np.zeros(6),
        param_priors=priors,
        observed_data=y_obs,
    )
    model.setup_whitening_from_priors()

    # IMPORTANT: avoid process-based multiprocessing if you see pickling errors.
    # Prefer single-core OR update MCMCModel to use ThreadPool.
    model.run(
        n_samples=3000,
        n_walkers=64,
        burn_in=300,
        thin=10,
        spherical_spread=1e-2,
        method_optimize="Powell",
        use_demoves=False,
        # If your MCMCModel supports it, set to 1 to avoid pickling:
        # n_cores=1,
        # use_multiprocessing=False,
    )

    chain = get_flat_chain_samples(model)
    print("Flat chain shape:", chain.shape)

    # --------------------------
    # Benchmark STT orders on the chain support
    # (FAIR: nonlinear uses order=1 propagator internally)
    # --------------------------
    orders_to_test = [1, 2, 3, 4]
    rows = benchmark_orders_on_chain(
        orders=orders_to_test,
        mu=mu,
        x0_ref1=x0_ref1,
        t_grid=t_grid,
        sc_pos=sc_pos,
        y_obs=y_obs,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        chain_samples=chain,
        n_eval=200,  # NL eval is the expensive part
        rtol=rtol,
        atol=atol,
        method="LSODA",
    )

    plot_chain_view_benchmark_latex(
        rows,
        out_png="stt_chain_view_rms.png",
        out_pdf="stt_chain_view_rms.pdf",
    )
