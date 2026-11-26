"""
Paper Case 2 comparison: the posterior of the "mirror" design (case2_bennu_particle.py) under four
likelihoods, all on the same data, priors and batch solution ref1:

  exact      : the full nonlinear dynamics (degree-2 field, every walker integrated), sampled here
               as the reference;
  STT-2      : Case 2a, second-order STTs about ref1;
  STM bank   : Case 2b, the bank of first-order STMs along the batch's least-determined position
               direction;
  single STM : one first-order STM about ref1 (the "stm" run of case2_bennu_particle.py).

Usage
-----
    python case2_comparison.py                  # latest case 2 runs; samples the exact reference
    python case2_comparison.py --exact results/case2_comparison_<stamp>/exact_reference.npz
                                                # reuse an exact reference: figure + table only,
                                                # written to the folder of that file
    python case2_comparison.py --exact-only     # sample the exact reference only (data from the
                                                # newest case 2 batch.npz), no figure
    python case2_comparison.py --runs DIR_2A DIR_2B DIR_STM     # explicit run folders

Outputs (results/case2_comparison_<stamp>/)
-------
- exact_reference.npz: the exact-dynamics chains (post-burn-in, thinned; per seed and walker);
- fig_case2_comparison.pdf: (a) marginal of the launch height h0 above Bennu's equator,
  (b) h0 against the out-of-plane launch speed vh0, (c) h0 against GM; 68 / 95 % regions of the
  four posteriors overlaid, with the truth and the batch;
- summary.txt: the table printed at the end (quantiles, lobe shares, truth credible levels).

Exact reference
---------------
emcee (stretch move, 128 walkers per seed, N_SEEDS seeds in parallel processes, 100000 steps,
25000 burn-in, thin 100), in the prior-whitened coordinates of the case runs and started, as
they are, 0.1 prior sigmas around ref1. Each half-ensemble proposal integrates its 64 walkers in
one stacked DOP853 solve (rtol 1e-10, atol 1e-12) of the dynamics of
generate_stt_functions_bennu_deg2 (inertial frame, degree-2 gravity in the body frame rotated by
R_ib(t) = Rz(omega t) Rx(pi/2 - delta) Rz(alpha + pi/2)). A walker whose trajectory comes
within R_IMPACT = 0.2 km of the center has hit Bennu: it gets zero posterior density, and the
solve goes on without it from the impact time (in the first steps many walkers of the start ball
hit the body). A failed stacked solve is redone one walker at a time. The constants, data, priors
and ref1 come from batch.npz of the case 2 runs, and the exact positions at ref1 are checked
against the saved ref1 trajectory.

Every posterior drops its stranded walkers (median log-posterior more than STRANDED_DLOGP below
the ensemble median) and the samples of the other walkers that lie as far below, as
case2_bennu_particle.py does.
"""

import argparse
import io
import multiprocessing
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter

import MCMCPlots

RESULTS = Path(__file__).resolve().parent / "results"
RUN_GLOBS = {
    "stt2": "case2a_bennu_particle_stt2_*",
    "bank": "case2b_bennu_particle_stmbank_*",
    "stm": "case2_bennu_particle_stm_*",
}
STRANDED_DLOGP = 100.0  # as case2_bennu_particle.py

# Exact reference sampler
N_SEEDS = 4
N_WALKERS = 128
N_STEPS = 100000
BURN_IN = 25000
THIN = 100
SPREAD = 0.1  # walker start: ref1 + SPREAD N(0, I) in prior-whitened coordinates
R_IMPACT = 0.2  # km; a trajectory that comes this close to the center has hit Bennu (minimum
# radius of the shape model 0.224 km) and gets zero posterior density

# Figure: colours of the four likelihoods (Okabe-Ito; truth and batch keep the house colours)
ORDER = ["stm", "bank", "stt2", "exact"]  # drawing order (exact on top)
COLOR = {"exact": "k", "stt2": MCMCPlots.C_POST, "bank": "#009E73", "stm": MCMCPlots.C_MC}
LS = {"exact": "-", "stt2": "-", "bank": "-", "stm": "--"}


# ============================================================
# Exact dynamics and likelihood
# ============================================================


def _rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class Degree2Dynamics:
    """
    Vectorized point-mass + degree-2 dynamics of case2_bennu_particle.py.

    The body-frame potential U = GM / r + GM R^2 Q / r^5, with
    Q = C20 (3 z^2 - r^2) / 2 + 3 C21 x z + 3 S21 y z + 3 C22 (x^2 - y^2) + 6 S22 x y
    (unnormalized coefficients, the Cartesian form of the P2m terms of
    generate_stt_functions_bennu_deg2), body frame r_b = R_ib(t) r_i.

    Parameters
    ----------
    R_ref, alpha, delta, omega : float
        Reference radius [km], pole right ascension and declination [rad], spin rate [rad/s]
        (batch.npz dyn_const).
    """

    def __init__(self, R_ref, alpha, delta, omega):
        self.R2 = float(R_ref) ** 2
        self.omega = float(omega)
        self.M0 = _rx(np.pi / 2 - delta) @ _rz(alpha + np.pi / 2)  # R_ib(0)
        self.pole = self.M0.T @ np.array([0.0, 0.0, 1.0])  # body +z (spin axis) in J2000

    def accel(self, r_i, P, t):
        """
        Inertial acceleration.

        Parameters
        ----------
        r_i : ndarray, shape (N, 3)
            Inertial positions [km].
        P : ndarray, shape (N, 6)
            [GM, C20, C21, S21, C22, S22] per trajectory.
        t : float
            Time since the epoch [s].

        Returns
        -------
        ndarray, shape (N, 3)
        """
        Rm = _rz(self.omega * t) @ self.M0
        rb = r_i @ Rm.T
        x, y, z = rb.T
        mu, C20, C21, S21, C22, S22 = P.T
        r2 = x * x + y * y + z * z
        r = np.sqrt(r2)
        r3 = r2 * r
        r5 = r3 * r2
        r7 = r5 * r2
        Q = (
            C20 * (3 * z * z - r2) / 2
            + 3 * C21 * x * z
            + 3 * S21 * y * z
            + 3 * C22 * (x * x - y * y)
            + 6 * S22 * x * y
        )
        dQ = np.stack(
            [
                -C20 * x + 3 * C21 * z + 6 * C22 * x + 6 * S22 * y,
                -C20 * y + 3 * S21 * z - 6 * C22 * y + 6 * S22 * x,
                2 * C20 * z + 3 * C21 * x + 3 * S21 * y,
            ],
            axis=1,
        )
        ab = (-mu / r3)[:, None] * rb + (mu * self.R2)[:, None] * (
            dQ / r5[:, None] - (5 * Q / r7)[:, None] * rb
        )
        return ab @ Rm  # R_ib^T a_b, row by row

    def _solve(self, X, tau):
        """
        One stacked DOP853 solve of N trajectories (rtol 1e-10, atol 1e-12) with an impact
        event: when a trajectory reaches R_IMPACT, it is dropped (NaN) and the others continue
        from the event time. Returns positions (N, T, 3), or None if the integrator fails.
        """
        N = X.shape[0]
        P = X[:, 6:12]
        out = np.full((N, tau.size, 3), np.nan)
        y0 = X[:, :6].copy()
        # A trajectory that starts inside R_IMPACT never crosses it: drop it now
        active = np.flatnonzero(np.linalg.norm(y0[:, :3], axis=1) > R_IMPACT)
        t0, k0 = tau[0], 0
        while active.size and k0 < tau.size:
            Pa = P[active]
            na = active.size

            def rhs(t, y):
                Y = y.reshape(na, 6)
                d = np.empty_like(Y)
                d[:, :3] = Y[:, 3:]
                d[:, 3:] = self.accel(Y[:, :3], Pa, t)
                return d.ravel()

            def impact(t, y):
                return np.min(np.linalg.norm(y.reshape(na, 6)[:, :3], axis=1)) - R_IMPACT

            impact.terminal = True
            impact.direction = -1
            sol = solve_ivp(
                rhs, (t0, tau[-1]), y0[active].ravel(), t_eval=tau[k0:], rtol=1e-10,
                atol=1e-12, method="DOP853", events=impact,
            )
            if sol.status == -1:
                return None
            n = len(sol.t)  # solve_ivp returns lists when no t_eval point was reached
            if n:
                out[active, k0 : k0 + n] = sol.y.reshape(na, 6, n)[:, :3].transpose(0, 2, 1)
            k0 += n
            if sol.status == 0:
                break
            # Impact: drop the trajectory (or trajectories) at R_IMPACT, continue the rest
            ye = sol.y_events[0][0].reshape(na, 6)
            r = np.linalg.norm(ye[:, :3], axis=1)
            hit = r <= R_IMPACT * (1.0 + 1e-9)
            hit[np.argmin(r)] = True
            out[active[hit]] = np.nan
            y0[active] = ye
            active = active[~hit]
            t0 = sol.t_events[0][0]
        dead = np.setdiff1d(np.arange(N), active)
        out[dead] = np.nan
        return out

    def positions(self, X, tau):
        """
        Positions of N trajectories at the epochs tau.

        Parameters
        ----------
        X : ndarray, shape (N, 12)
            Initial states [x y z vx vy vz GM C20 C21 S21 C22 S22].
        tau : ndarray, shape (T,)
            Epochs [s] since the start (tau[0] = 0).

        Returns
        -------
        ndarray, shape (N, T, 3)
            NaN for a trajectory that hits Bennu (comes within R_IMPACT of the center) or whose
            integration fails (one at a time, after the stacked solve failed).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = self._solve(X, tau)
        if out is not None:
            return out
        out = np.full((X.shape[0], tau.size, 3), np.nan)
        for i in range(X.shape[0]):
            p = self._solve(X[i : i + 1], tau)
            if p is not None:
                out[i] = p[0]
        return out


class ExactLogPosterior:
    """
    Vectorized log-posterior of the case 2 runs with exact dynamics, in prior-whitened
    coordinates z (theta = mean + sigma z, theta the deviation about ref1). Picklable.

    log p(z) = -0.5 |r(theta)|^2 - 0.5 |z|^2, r the normalized, visibility-weighted RA/Dec
    residuals (as case2_bennu_particle.py; constants dropped).

    Parameters
    ----------
    b : mapping
        batch.npz of a case 2 run.
    """

    def __init__(self, b):
        self.x1 = np.asarray(b["x0_ref1"], dtype=float)
        self.mean = -np.asarray(b["delta_shift"], dtype=float)
        self.sig = np.asarray(b["prior_sigma"], dtype=float)
        self.tau = np.asarray(b["tau"], dtype=float)
        self.sc = np.asarray(b["sc_pos"], dtype=float)
        self.y = np.asarray(b["y_obs"], dtype=float)
        sra, sdec = np.asarray(b["sigma"], dtype=float)
        w = np.asarray(b["obs_weights"], dtype=float)
        self.w_ra = w[0::2] / sra
        self.w_dec = w[1::2] / sdec
        self.dyn = Degree2Dynamics(*np.asarray(b["dyn_const"], dtype=float))

    def theta(self, Z):
        return self.mean + np.atleast_2d(Z) * self.sig

    def residuals(self, theta):
        """Normalized residuals, shape (N, 2T), for deviations theta (N, 12) about ref1."""
        P = self.dyn.positions(self.x1 + np.atleast_2d(theta), self.tau)
        los = P - self.sc[None]
        ra = np.arctan2(los[..., 1], los[..., 0])
        dec = np.arctan2(los[..., 2], np.hypot(los[..., 0], los[..., 1]))
        dra = (self.y[None, 0::2] - ra + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([dra * self.w_ra, (self.y[None, 1::2] - dec) * self.w_dec], axis=1)

    def __call__(self, Z):
        Z = np.atleast_2d(Z)
        r = self.residuals(self.theta(Z))
        lp = -0.5 * (np.sum(r * r, axis=1) + np.sum(Z * Z, axis=1))
        return np.where(np.isfinite(lp), lp, -np.inf)  # impact / failed integration


def _run_seed(args):
    """One exact-reference emcee run (worker of sample_exact)."""
    import emcee

    logp, seed, n_steps, burn_in, thin = args
    rng = np.random.default_rng(seed)
    z1 = -logp.mean / logp.sig  # ref1 (theta = 0) in whitened coordinates
    p0 = z1 + SPREAD * rng.normal(size=(N_WALKERS, z1.size))
    np.random.seed(seed)  # emcee's moves draw from the global state
    smp = emcee.EnsembleSampler(N_WALKERS, z1.size, logp, vectorize=True)
    t0 = time.time()
    smp.run_mcmc(p0, n_steps, progress=False)
    Z = smp.get_chain(discard=burn_in, thin=thin)  # (n, nw, 12)
    lp = smp.get_log_prob(discard=burn_in, thin=thin)  # (n, nw)
    # Autocorrelation times without the stranded walkers: a walker that starts inside the
    # body (zero density) and never gets out stays constant, and makes the estimate NaN
    keep_w, _ = _drop_stranded(smp.get_log_prob(discard=burn_in))
    tau = emcee.autocorr.integrated_time(smp.get_chain(discard=burn_in)[:, keep_w], quiet=True)
    print(
        f"[Exact] seed {seed}: {n_steps} steps in {(time.time() - t0) / 60:.1f} min, "
        f"acceptance {np.mean(smp.acceptance_fraction):.3f}, tau_max {tau.max():.0f} "
        f"({(~keep_w).sum()} stranded walkers left out)",
        flush=True,
    )
    return Z, lp, np.asarray(smp.acceptance_fraction), tau


def sample_exact(b, n_seeds, n_steps, out_path):
    """
    Sample the exact-dynamics posterior (n_seeds parallel emcee runs) and save it.

    Parameters
    ----------
    b : mapping
        batch.npz of a case 2 run (data, priors, ref1, dynamics constants).
    n_seeds, n_steps : int
    out_path : Path
        exact_reference.npz.

    Returns
    -------
    dict
        The saved arrays.
    """
    logp = ExactLogPosterior(b)
    # Check: exact positions at ref1 against the ref1 trajectory of the run
    P1 = logp.dyn.positions(logp.x1[None], logp.tau)[0]
    dr = 1e3 * np.abs(P1 - np.asarray(b["r_ref1"])).max()
    z1 = -logp.mean / logp.sig
    print(
        f"[Exact] positions at ref1 vs the run's ref1 trajectory: max |dr| = {dr:.2e} m; "
        f"chi2 at ref1 = {-2 * logp(z1[None])[0] - z1 @ z1:.2f}"
    )
    if dr > 1e-2:
        raise RuntimeError(f"exact dynamics do not reproduce ref1 (max |dr| {dr:.3g} m)")
    burn_in = min(BURN_IN, n_steps // 4)
    print(
        f"[Exact] {n_seeds} seeds x {N_WALKERS} walkers x {n_steps} steps (burn-in {burn_in}, "
        f"thin {THIN}), in parallel..."
    )
    jobs = [(logp, s, n_steps, burn_in, THIN) for s in range(1, n_seeds + 1)]
    with multiprocessing.get_context("fork").Pool(n_seeds) as pool:
        res = pool.map(_run_seed, jobs)
    out = dict(
        Z=np.array([r[0] for r in res]),  # (n_seeds, n, nw, 12), whitened
        lp=np.array([r[1] for r in res]),  # (n_seeds, n, nw)
        acceptance=np.array([r[2] for r in res]),
        tau=np.array([r[3] for r in res]),
        mean=logp.mean,
        sig=logp.sig,
        x0_ref1=logp.x1,
        n_steps=n_steps,
        burn_in=burn_in,
        thin=THIN,
    )
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    MCMCPlots.write_output(out_path, buf.getvalue(), tag="Exact")
    print(f"[Exact] saved {out_path}")
    return out


# ============================================================
# Posterior samples of the four likelihoods
# ============================================================


def _drop_stranded(lp_chain):
    """
    Masks of the kept walkers and samples of a log-posterior chain.

    Parameters
    ----------
    lp_chain : ndarray, shape (n_steps, n_walkers)

    Returns
    -------
    keep_w : ndarray of bool, shape (n_walkers,)
        False for the stranded walkers (median more than STRANDED_DLOGP below the ensemble).
    keep : ndarray of bool, shape (n_steps, n_walkers)
        keep_w, and False for the samples of the other walkers that lie as far below (a walker
        that reached the posterior only after the burn-in).
    """
    lp_w = np.median(lp_chain, axis=0)
    lp_cut = np.median(lp_w) - STRANDED_DLOGP
    keep_w = lp_w >= lp_cut
    return keep_w, keep_w[None, :] & (lp_chain >= lp_cut)


def load_run(run_dir):
    """
    Physical initial states of a case 2 run, without stranded walkers.

    Returns
    -------
    X : ndarray, shape (n, 12)
        x0_ref1 + samples.
    info : dict
        Walkers dropped, acceptance, n_steps / tau_max.
    """
    ch = np.load(run_dir / "chain.npz")
    b = np.load(run_dir / "batch.npz")
    keep_w, keep = _drop_stranded(ch["log_prob_chain"])
    nw = keep_w.size
    keep = keep.reshape(-1)  # flat index = step * nw + walker
    info = dict(
        dropped=int(nw - keep_w.sum()),
        acc=float(ch["acceptance_fraction"][keep_w].mean()),
        n_tau=float(ch["n_steps"]) / float(np.max(ch["tau"])),
    )
    return b["x0_ref1"] + ch["samples"][keep], info


def load_exact(ex):
    """Physical initial states of the exact reference (all seeds, stranded walkers dropped)."""
    import emcee

    X, dropped, acc, tau = [], 0, [], np.asarray(ex["tau"], dtype=float)
    for k, (Z, lp) in enumerate(zip(ex["Z"], ex["lp"])):
        keep_w, keep = _drop_stranded(lp)
        dropped += int(keep_w.size - keep_w.sum())
        acc.append(ex["acceptance"][k][keep_w])
        X.append(ex["x0_ref1"] + ex["mean"] + Z[keep] * ex["sig"])
        if not np.all(np.isfinite(tau[k])):
            # Saved with the stranded walkers in (NaN): redo on the thinned chain without them
            tau[k] = emcee.autocorr.integrated_time(Z[:, keep_w], quiet=True) * int(ex["thin"])
    info = dict(
        dropped=dropped,
        acc=float(np.mean(np.concatenate(acc))),
        n_tau=float(ex["n_steps"]) / float(np.max(tau)),
    )
    return np.vstack(X), info


def _latest(pattern):
    runs = sorted(RESULTS.glob(pattern))
    if not runs:
        sys.exit(f"no results/{pattern} folder")
    return runs[-1]


# ============================================================
# Figure and table
# ============================================================


def _density(x, y, ext, n=90, smooth=1.6):
    H, xe, ye = np.histogram2d(x, y, bins=n, range=ext)
    H = gaussian_filter(H.T, smooth)
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H / H.sum()


def _credible_level(H, xc, yc, pt):
    """Mass of the 2D density above its value at the point pt (0 = mode, 1 = outside)."""
    i = np.clip(np.searchsorted(xc, pt[0]), 0, xc.size - 1)
    j = np.clip(np.searchsorted(yc, pt[1]), 0, yc.size - 1)
    return float(H[H > H[j, i]].sum())


def _extent(arrs, lo=0.2, hi=99.8, pad=0.12):
    a = np.min([np.percentile(x, lo) for x in arrs])
    b = np.max([np.percentile(x, hi) for x in arrs])
    return [a - pad * (b - a), b + pad * (b - a)]


def compare(post, pole, x0_true, x0_ref1, labels, out_dir):
    """
    Figure fig_case2_comparison.pdf and the summary table.

    Parameters
    ----------
    post : dict
        name -> (X (n, 12) physical initial states, info dict).
    pole : ndarray, shape (3,)
        Spin axis in J2000 (the launch height is measured along it).
    x0_true, x0_ref1 : ndarray, shape (12,)
    labels : dict
        name -> legend label.
    out_dir : Path

    Returns
    -------
    str
        The table.
    """
    def proj(X):
        X = np.atleast_2d(X)
        return (
            1e3 * X[:, :3] @ pole,  # h0 [m]
            1e6 * X[:, 3:6] @ pole,  # vh0 [mm/s]
            100.0 * (X[:, 6] / x0_true[6] - 1.0),  # GM offset [%]
        )

    names = [m for m in ORDER if m in post]
    Q = {m: proj(post[m][0]) for m in names}
    tru = [q[0] for q in proj(x0_true)]
    bat = [q[0] for q in proj(x0_ref1)]

    # Split between the lobes: minimum of the exact h0 marginal between the truth and the batch
    hb = np.linspace(*_extent([Q[m][0] for m in names]), 241)
    hc = 0.5 * (hb[1:] + hb[:-1])
    dens = gaussian_filter(np.histogram(Q["exact"][0], bins=hb, density=True)[0], 2.0)
    between = (hc > min(tru[0], bat[0])) & (hc < max(tru[0], bat[0]))
    split = float(hc[between][np.argmin(dens[between])]) if between.any() else 0.5 * (
        tru[0] + bat[0]
    )

    MCMCPlots.apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(MCMCPlots.TXT_W, 2.3), constrained_layout=True)
    ext_h = _extent([Q[m][0] for m in names])
    hist_bins = np.linspace(*ext_h, 81)
    for m in names:
        ax[0].hist(
            Q[m][0], bins=hist_bins, density=True, histtype="step", color=COLOR[m], ls=LS[m],
            lw=1.3 if m == "exact" else 1.0, zorder=3 if m == "exact" else 2,
        )
    ax[0].axvline(tru[0], color=MCMCPlots.C_TRUTH, lw=0.7, ls=":", zorder=1)
    ax[0].axvline(bat[0], color=MCMCPlots.C_BATCH, lw=0.7, ls=":", zorder=1)
    ax[0].set_xlim(ext_h)
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.12)
    ax[0].set_xlabel(r"$h_0$ [m]")
    ax[0].set_ylabel("Posterior density [1/m]")
    ax[0].set_title("(a) Launch height", loc="left")

    levels = (0.95, 0.68)
    cl = {}
    for a, j, ylab, title in (
        (ax[1], 1, r"$v_{h,0}$ [mm/s]", "(b) Height, out-of-plane speed"),
        (ax[2], 2, r"$GM$ offset [$\%$]", r"(c) Height, $GM$"),
    ):
        ext = [ext_h, _extent([Q[m][j] for m in names])]
        for m in names:
            xc, yc, H = _density(Q[m][0], Q[m][j], ext)
            lev = MCMCPlots._hpd_levels(H, levels)  # ascending density: 95 %, 68 %
            a.contour(
                xc, yc, H, levels=lev, colors=COLOR[m], linestyles=LS[m],
                linewidths=[0.7, 1.3] if m == "exact" else [0.6, 1.0],
                zorder=3 if m == "exact" else 2,
            )
            cl[(m, j)] = _credible_level(H, xc, yc, (tru[0], tru[j]))
        a.plot(tru[0], tru[j], ls="none", ms=8, marker="*", mfc=MCMCPlots.C_TRUTH, mec="white",
               mew=0.4, zorder=5)
        a.plot(bat[0], bat[j], ls="none", ms=4.5, marker="s", mfc=MCMCPlots.C_BATCH, mec="white",
               mew=0.4, zorder=5)
        a.set_xlim(ext[0])
        a.set_ylim(ext[1])
        a.set_xlabel(r"$h_0$ [m]")
        a.set_ylabel(ylab)
        a.set_title(title, loc="left")

    handles = [
        Line2D([], [], color=COLOR[m], ls=LS[m], lw=1.3 if m == "exact" else 1.0, label=labels[m])
        for m in reversed(names)
    ] + [
        Line2D([], [], ls="none", ms=7, marker="*", mfc=MCMCPlots.C_TRUTH, mec="white", mew=0.4,
               label="Truth"),
        Line2D([], [], ls="none", ms=4.5, marker="s", mfc=MCMCPlots.C_BATCH, mec="white",
               mew=0.4, label=f"Batch ({MCMCPlots.BATCH_LABEL})"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    MCMCPlots.set_figure_dir(out_dir)
    MCMCPlots.save_figure(fig, "fig_case2_comparison")

    # Table
    h_ex = Q["exact"][0]
    tv_bins = np.linspace(*ext_h, 61)
    p_ex = np.histogram(h_ex, bins=tv_bins)[0] / h_ex.size
    lines = [
        f"Case 2 comparison (h0 = launch height above the equator, vh0 = out-of-plane launch "
        f"speed; truth h0 = {tru[0]:+.1f} m, vh0 = {tru[1]:+.1f} mm/s; batch h0 = {bat[0]:+.1f} m, "
        f"vh0 = {bat[1]:+.1f} mm/s; lobe split at h0 = {split:.1f} m, the minimum of the exact "
        f"h0 marginal between them)",
        "",
        f"{'likelihood':<24s}{'n':>8s}{'drop':>6s}{'acc':>7s}{'n/tau':>7s}"
        f"{'h0 2.5/16/50/84/97.5 % [m]':>36s}{'P(truth lobe)':>15s}{'TV(h0)':>8s}"
        f"{'truth lvl (h0,vh0)':>20s}{'(h0,GM)':>9s}",
    ]
    for m in reversed(names):
        h = Q[m][0]
        info = post[m][1]
        p_m = np.histogram(h, bins=tv_bins)[0] / h.size
        tv = 0.5 * np.abs(p_m - p_ex).sum()
        pct = " ".join(f"{v:+6.1f}" for v in np.percentile(h, [2.5, 16, 50, 84, 97.5]))
        share = np.mean(h < split) if tru[0] < split else np.mean(h > split)
        lines.append(
            f"{labels[m]:<24s}{h.size:>8d}{info['dropped']:>6d}{info['acc']:>7.3f}"
            f"{info['n_tau']:>7.1f}{pct:>36s}{share:>15.3f}"
            f"{tv:>8.3f}{cl[(m, 1)]:>20.3f}{cl[(m, 2)]:>9.3f}"
        )
    lines += [
        "",
        "drop: stranded walkers left out; acc: mean acceptance; n/tau: steps per autocorrelation "
        "time (max over parameters); P(truth lobe): posterior share on the truth's side of the "
        "split; TV(h0): total variation distance of the h0 marginal from the exact one (60 bins); "
        "truth lvl: posterior mass denser than the truth in that plane (0 = mode, 1 = outside).",
    ]
    table = "\n".join(lines)
    MCMCPlots.write_output(out_dir / "summary.txt", (table + "\n").encode(), tag="Summary")
    return table


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs=3, metavar=("DIR_2A", "DIR_2B", "DIR_STM"))
    ap.add_argument(
        "--exact", metavar="NPZ", help="reuse this exact_reference.npz (outputs go next to it)"
    )
    ap.add_argument("--exact-only", action="store_true", help="sample the exact reference only")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    args = ap.parse_args()

    # A reused exact reference keeps its folder: the figure and the table are written next to it
    if args.exact:
        out_dir = Path(args.exact).resolve().parent
    else:
        out_dir = RESULTS / f"case2_comparison_{datetime.now().strftime('%y%m%d%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.exact_only:
        newest = max(
            (p for g in RUN_GLOBS.values() for p in RESULTS.glob(g) if (p / "batch.npz").exists()),
            key=lambda p: p.name.rsplit("_", 1)[-1],
        )
        print(f"[Exact] data, priors and ref1 from {newest.name}")
        sample_exact(np.load(newest / "batch.npz"), args.seeds, args.steps,
                     out_dir / "exact_reference.npz")
        return

    runs = (
        dict(zip(("stt2", "bank", "stm"), map(Path, args.runs)))
        if args.runs
        else {k: _latest(g) for k, g in RUN_GLOBS.items()}
    )
    B = {k: np.load(d / "batch.npz") for k, d in runs.items()}
    for k, d in runs.items():
        print(f"[Compare] {k:5s}: {d.name}")
    # The three runs must share the data and the batch solution
    for k in ("bank", "stm"):
        for key in ("y_obs", "x0_ref1", "x0_true", "prior_sigma", "delta_shift"):
            if not np.allclose(B[k][key], B["stt2"][key], rtol=1e-12, atol=0.0):
                sys.exit(f"{runs[k].name} and {runs['stt2'].name} differ in {key}")
    b = B["stt2"]

    if args.exact:
        print(f"[Exact] reusing {args.exact}")
        ex = dict(np.load(args.exact))
    else:
        ex = sample_exact(b, args.seeds, args.steps, out_dir / "exact_reference.npz")
    if not np.allclose(ex["x0_ref1"], b["x0_ref1"], rtol=1e-12, atol=0.0):
        sys.exit("the exact reference was sampled about a different ref1")

    n_bank = int(B["bank"]["bank_s"].size) if "bank_s" in B["bank"].files else 0
    labels = {
        "exact": "Exact dynamics",
        "stt2": "STT-2 (Case 2a)",
        "bank": f"Bank of {n_bank} STMs (Case 2b)",
        "stm": "Single STM",
    }
    post = {k: load_run(d) for k, d in runs.items()}
    post["exact"] = load_exact(ex)
    pole = Degree2Dynamics(*b["dyn_const"]).pole
    table = compare(post, pole, b["x0_true"], b["x0_ref1"], labels, out_dir)
    print("\n" + table)


if __name__ == "__main__":
    main()
