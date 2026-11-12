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

Outputs (results/stt_benchmark_<timestamp>/): chain.npz, benchmark.npz and one journal PDF
per figure (fig_stt_benchmark, fig_mcmc_traces, fig_mcmc_autocorrelation, fig_corner,
fig_corner_prior).
LOAD_CHAIN / LOAD_BENCHMARK in the main redo the figures from a saved run.
"""

import io
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp

from STTPropagationND import STTPropagatorND
from MCMC import MCMCModel

# Journal figures (Okabe-Ito colors, LaTeX text, one vector PDF per figure) live in MCMCPlots.py;
# MCMCPlots.configure() in the main applies the journal style.
import MCMCPlots

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved

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
    """
    Time and compare the STT surrogate log-likelihood of each order against the full
    nonlinear one on n_eval chain samples. Prints the summary table and returns a dict:
      orders, t_build, t_setup [s], stt_ms, nl_ms [ms/sample], speedup (n_orders,);
      dlogL = lnL_STT - lnL_NL (n_orders, Ns); dy = r_STT - r_NL, the normalized-residual
      (= predicted-measurement / sigma) difference (n_orders, Ns, n_meas); sample_idx (Ns,).
    """
    chain_samples = np.asarray(chain_samples, float)
    Ns_all = chain_samples.shape[0]
    if n_eval < Ns_all:
        idx = np.random.default_rng(0).choice(Ns_all, size=n_eval, replace=False)
    else:
        idx = np.arange(Ns_all)
    thetas = chain_samples[idx]
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
        # state only (no STM, no progress bar): the cost of a plain nonlinear likelihood
        sol = prop_nl.propagate_state_only(
            x0_ref1 + delta0,
            t_grid,
            show_progress=False,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        x_est = sol.y[:6, :].T
        return residuals_from_x(x_est)

    rows = []
    dlogL_all, dy_all = [], []
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
        dlogL_all.append(dlogL)
        dy_all.append(dy_sig)

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

    rows = np.asarray(rows, float)
    bench = dict(
        orders=rows[:, 0].astype(int),
        t_build=rows[:, 1],
        t_setup=rows[:, 2],
        stt_ms=rows[:, 3],
        nl_ms=rows[:, 4],
        speedup=rows[:, 5],
        dlogL=np.asarray(dlogL_all),
        dy=np.asarray(dy_all),
        sample_idx=np.asarray(idx),
    )
    print_benchmark_table(bench)
    return bench


def print_benchmark_table(bench):
    """Summary table of a benchmark dict (benchmark_orders_on_chain or benchmark.npz)."""
    dlogL = np.asarray(bench["dlogL"])
    dy = np.asarray(bench["dy"])
    rows = [
        [
            int(o),
            tb,
            ts,
            sm,
            nm,
            spd,
            rms(dl),
            p95_abs(dl),
            rms(d),
            p95_abs(d),
        ]
        for o, tb, ts, sm, nm, spd, dl, d in zip(
            bench["orders"],
            bench["t_build"],
            bench["t_setup"],
            bench["stt_ms"],
            bench["nl_ms"],
            bench["speedup"],
            dlogL,
            dy,
        )
    ]
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
    print(f"Evaluated chain samples: {dlogL.shape[1]}")
    print(format_table(rows, headers))


def save_benchmark(bench, path, **extra):
    """Write the benchmark dict (+ extra scalars/arrays) as .npz, OneDrive-safe."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **bench, **extra)
    path = MCMCPlots.write_output(path, buf.getvalue(), tag="Benchmark")
    print(f"[Benchmark] saved {path}")
    return path


def load_benchmark(path):
    with np.load(path) as f:
        bench = {k: f[k] for k in f.files}
    print(f"[Benchmark] loaded {path}")
    return bench


# =========================
# Figures: MCMCPlots.plot_stt_benchmark (journal style, see MCMCPlots.py)
# =========================


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

    T_hours = 1.0
    N_obs = 60
    t_grid = np.linspace(0.0, T_hours * 3600.0, N_obs)

    # Measurement noise
    sigma_pix = 0.1  # pixel noise (1-sigma)
    fov_deg = 44.0  # full FOV [deg]  (e.g. NavCam ~44 deg)
    n_pix = 2592  # pixels across FOV (e.g. NavCam)
    fov_rad = np.deg2rad(fov_deg)
    sigma_angle = sigma_pix * (fov_rad / n_pix)  # radians
    sigma_ra = sigma_angle
    sigma_dec = sigma_angle

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
    ref_sig_r = np.array([1e-3, 1e-3, 1e-3])  # km
    ref_sig_v = np.array([1e-6, 1e-6, 1e-6])  # km/s
    x0_ref1 = x0_true - np.hstack(
        [rng.normal(scale=ref_sig_r), rng.normal(scale=ref_sig_v)]
    )

    # --------------------------
    # MCMC settings, saved runs, figures
    # --------------------------
    n_walkers = 64
    n_samples = 7000  # >= 100 tau (tau ~ 55-65 steps)
    burn_in = 300
    thin = 10
    spherical_spread = 1e-2
    # Saved runs: every run writes chain.npz and benchmark.npz into its results folder. Set
    # LOAD_CHAIN to such a chain to skip the sampling, and LOAD_BENCHMARK to such a benchmark
    # to skip the STT-vs-nonlinear timing (both None: full run).
    LOAD_CHAIN = None  # e.g. "results/stt_benchmark_<stamp>/chain.npz"
    LOAD_BENCHMARK = None  # e.g. "results/stt_benchmark_<stamp>/benchmark.npz"

    # Parameter symbols and display units (km -> m, km/s -> mm/s; rescaled by powers of ten
    # from the posterior sigmas after the run); one PDF per figure in a timestamped folder.
    # In the figures delta = x0 - x0_true (error w.r.t. the truth, see the figure block).
    PARAM_SYM = [
        r"\delta x_0",
        r"\delta y_0",
        r"\delta z_0",
        r"\delta v_{x,0}",
        r"\delta v_{y,0}",
        r"\delta v_{z,0}",
    ]
    PARAM_BASE_SCALE = np.array([1e3] * 3 + [1e6] * 3)
    PARAM_BASE_UNIT = ["m"] * 3 + ["mm/s"] * 3
    MCMCPlots.configure(
        PARAM_SYM,
        PARAM_BASE_UNIT,
        PARAM_BASE_SCALE,
        body="Bennu",
        target="Particle",
        obs_word="Image",
        resid_labels=("RA", "Dec"),
        fallback_dir=Path.home() / "stt_benchmark_figures_fallback",
    )
    fig_dir = MCMCPlots.set_figure_dir(
        Path(__file__).resolve().parent
        / "results"
        / f"stt_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"[Figures] -> {fig_dir}")

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

    if LOAD_CHAIN:
        model.load_chain(LOAD_CHAIN)  # figures from a saved run (no sampling)
        burn_in, thin = model.burn_in, model.thin
    else:
        # IMPORTANT: avoid process-based multiprocessing if you see pickling errors.
        # Prefer single-core OR update MCMCModel to use ThreadPool.
        model.run(
            n_samples=n_samples,
            n_walkers=n_walkers,
            burn_in=burn_in,
            thin=thin,
            spherical_spread=spherical_spread,
            method_optimize="Powell",
            use_demoves=False,
        )
        model.save_chain(fig_dir / "chain.npz")

    chain = get_flat_chain_samples(model)
    print("Flat chain shape:", chain.shape)

    # --------------------------
    # Benchmark STT orders on the chain support
    # (FAIR: nonlinear uses order=1 propagator internally)
    # --------------------------
    orders_to_test = [1, 2, 3, 4]
    if LOAD_BENCHMARK:
        bench = load_benchmark(LOAD_BENCHMARK)
        print_benchmark_table(bench)
    else:
        bench = benchmark_orders_on_chain(
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
        save_benchmark(
            bench, fig_dir / "benchmark.npz", stt_order_for_mcmc=stt_order_for_mcmc
        )

    # --------------------------
    # Journal figures: benchmark, then the chain it was evaluated on
    # (samples are deviations about ref1; no batch solution in this script)
    # --------------------------
    MCMCPlots.plot_stt_benchmark(bench, mcmc_order=stt_order_for_mcmc)

    # The chain holds offsets from ref1, which sits ~1 m (hundreds of posterior sigmas) from
    # the truth: plotted as is, an axis reads e.g. -1043 [10^-3 m]. The figures show the
    # error w.r.t. the truth instead, delta = x0 - x0_true (truth at 0); the transform also
    # shifts the prior draws of the prior overlay.
    true_delta = x0_true - x0_ref1

    def to_error(theta):
        return np.asarray(theta, float) - true_delta

    zero = np.zeros_like(true_delta)
    disp_scale, disp_unit = MCMCPlots.auto_param_scales(
        PARAM_BASE_SCALE, PARAM_BASE_UNIT, np.std(model.samples, axis=0)
    )
    MCMCPlots.configure(PARAM_SYM, disp_unit, disp_scale)

    # The prior is 30-2000x wider than the posterior, so the prior corner gets its own units,
    # sized to the prior sigma (m, mm/s); with the posterior units above its ticks would read
    # e.g. +-20000 [10^-3 m]. The posterior is then a narrow spike: the data dominate the prior.
    prior_scale, prior_unit = MCMCPlots.auto_param_scales(
        PARAM_BASE_SCALE, PARAM_BASE_UNIT, prior_sigma
    )

    def corner_with_prior():
        MCMCPlots.configure(PARAM_SYM, prior_unit, prior_scale)
        try:
            model.plot_journal_corner(true_theta=zero, transform=to_error, prior=True)
        finally:
            MCMCPlots.configure(PARAM_SYM, disp_unit, disp_scale)

    for label, draw in (
        (
            "traces",
            lambda: model.plot_journal_traces(
                burn_in, true_theta=zero, transform=to_error
            ),
        ),
        ("autocorrelation", lambda: model.plot_journal_autocorrelation(burn_in, thin)),
        (
            "corner",
            lambda: model.plot_journal_corner(true_theta=zero, transform=to_error),
        ),
        ("corner + prior", corner_with_prior),
    ):
        try:
            draw()
        except Exception as err:
            print(f"[Figure] {label} skipped: {type(err).__name__}: {err}")
