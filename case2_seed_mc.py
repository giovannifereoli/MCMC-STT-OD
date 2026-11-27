"""
Case 2 seed Monte Carlo: over random (noise, reference) seed pairs of the "mirror" design
(case2_bennu_particle.py), how often the cost along the launch height h0 has two minima, and
how often the Case 2 batch stops in the wrong one.

The paper's seeds (noise 395, reference 239) were picked so that the batch stops in the shallow
minimum 2.50 above the best fit (see case2_bennu_particle.py, Design). This script puts a rate
on that outcome: fresh seed pairs (noise 100000 + i, reference 200000 + i, disjoint from the
ranges searched in the design), everything else as Case 2.

Usage
-----
    python case2_seed_mc.py                  # N_REAL realizations, geometry from the newest Case 2
                                             # batch.npz; results/case2_seed_mc_<stamp>/
    python case2_seed_mc.py --n 200 --procs 4
    python case2_seed_mc.py --load results/case2_seed_mc_<stamp>/mc.npy
                                             # summary + figure of a finished run (written next
                                             # to that file)
    python case2_seed_mc.py --validate       # the fast batch against the saved Case 2 batch

One realization
---------------
1. Data: the truth trajectory observed from the saved spacecraft positions at the saved epochs
   and visibility, RA/Dec noise from the noise seed (as generate_opnav_measurements_from_sc).
2. Reference: truth - dev, dev ~ N(0, (20 % |r|, 20 % |v|, 1 % GM, 1 % |Cnm|)^2) from the
   reference seed (as case2_bennu_particle.py); prior N(reference, prior_sigma^2) as Case 2.
3. Batch: BatchLSQ.gn_with_stm unchanged with the Case 2 settings (SRIF, full rank, step
   halving, max_iter 30, tol 1e-8). Only its propagator is replaced: the exact degree-2
   dynamics of case2_comparison.py (Degree2Dynamics) with a central-difference STM (step 1e-4
   prior sigmas) from one stacked DOP853 solve (rtol 1e-11, atol 1e-14) of the nominal and
   the 24 perturbed trajectories; the angles need only the position rows. This reproduces the
   saved Case 2 batch exactly (--validate) in 0.1 s instead of ~17 min.
4. Continuation: the same Gauss-Newton restarted from the batch result for up to 300 more
   iterations (Gauss-Newton keeps no state but the iterate), to see where a batch that stopped
   at 30 iterations was heading.
5. Profile: the cost minimized over the 11 coordinates orthogonal to the pole at fixed h0
   (Gauss-Newton with step halving), by continuation outward from the truth over h0 in
   [-60, 80] m, step 1 m, each side stopped 60 above the lowest value. Each local minimum of the
   profile is refined by an unconstrained batch (300 iterations) and gets its barrier: the
   smallest rise on the profile to a lower point.

Outcome of the batch:
- at the best fit: converged (the continuation lowers the cost by at most STALL) at the lowest
  minimum;
- at another minimum (the wrong one), farther from or closer to the truth in h0 than the best
  fit (the paper's batch is farther);
- stopped short: the continuation lowers the cost by more than STALL (the batch stopped at 30
  iterations while still crawling along the valley with halved steps);
- failed: the propagation of the reference failed (its trajectory falls back to Bennu and
  reaches the center: no impact event in the point-mass + degree-2 field).
Two minima are the same when they differ by less than TOL_H in h0 and TOL_C in cost.

Outputs (results/case2_seed_mc_<stamp>/)
-------
- mc.npy: one dict per realization (seeds, reference and batch h0, costs, the profile, its
  refined minima and barriers);
- summary.txt: the rates (95 % binomial intervals) printed at the end;
- fig_case2_seed_mc.pdf: (a) batch h0 relative to the best fit against the reference's offset
  along the pole, by outcome, with the paper's realization (a stopped-short batch beyond the
  converged ones is drawn on the axis edge, and their count is noted); (b) cost gap between the two lowest
  minima of the realizations whose profile has two minima, by outcome.

The cost is J = -2 ln p (data and prior residuals squared, whitened); h0 is the launch height
along the pole relative to the truth.
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
from scipy.stats import norm

import MCMCPlots
from BatchLSQ import BatchLSQ
from case2_comparison import Degree2Dynamics
from case2_bennu_particle import radec_and_partials_from_los, wrap_to_pi

RESULTS = Path(__file__).resolve().parent / "results"
BATCH_GLOB = "case2b_bennu_particle_stmbank_*"  # newest plain run (any Case 2 run has the same geometry)
SHOW_FIGS = False

N_REAL = 2000
NOISE0, REF0 = 100000, 200000  # seed pair i = (NOISE0 + i, REF0 + i)
PAPER_SEEDS = (395, 239)  # (noise, reference) of case2_bennu_particle.py
PROCS = 8

KW = dict(rtol=1e-11, atol=1e-14, method="DOP853")  # as the case runs
FD_REL = 1e-4  # central-difference step of the STM, in prior sigmas
MAX_ITER = 30  # Case 2 batch
MAX_ITER_CONT = 300  # continuation and refinement
GRID = np.arange(-60.0, 80.0 + 1e-9, 1.0)  # profile h0 grid [m]
CAP = 60.0  # profile stops this far above its lowest value

BAR = 0.5  # a second minimum counts when its barrier is at least this (cost units)
TOL_H, TOL_C = 3.0, 0.05  # same minimum: |dh0| < 3 m and |dJ| < 0.05
STALL = 0.01  # the batch stopped short if the continuation lowers the cost by more

# Outcome colours (Okabe-Ito, adjacent pairs CVD-checked; markers differ as well)
OUTCOMES = [
    ("best", "At the best fit", MCMCPlots.C_POST, "o"),
    ("far", "Other minimum, farther from the truth", MCMCPlots.C_BATCH, "^"),
    ("near", "Other minimum, closer to the truth", "#E69F00", "v"),
    ("stalled", f"Stopped short at {MAX_ITER} iterations", "#CC79A7", "s"),
]


# ============================================================
# Fast exact batch
# ============================================================


class _Sol:
    def __init__(self, y, status):
        self.y, self.status = y, status


class FDPropagator:
    """
    Exact degree-2 propagation with a central-difference STM, in the interface of
    STTPropagatorND: propagate(x0, t_eval) -> (sol, {1: STM}).

    Only the positions and the position rows of the STM are filled: BatchLSQ with angle
    measurements uses no more.

    Parameters
    ----------
    dyn : Degree2Dynamics
    step : array_like, shape (12,)
        Central-difference step of each parameter (internal units).
    """

    def __init__(self, dyn, step):
        self.dyn, self.step = dyn, np.asarray(step, float)
        self.n_solves = 0

    def positions(self, X, tau, rtol=1e-11, atol=1e-14, method="DOP853"):
        """
        Positions of several trajectories in one stacked solve.

        Parameters
        ----------
        X : ndarray, shape (N, 12)
            Initial states and parameters [km, km/s, km^3/s^2, -].
        tau : ndarray, shape (T,)
            Output times [s], tau[0] the epoch of X.

        Returns
        -------
        ndarray, shape (N, T, 3), or None
            None if the solve fails or gives non-finite values.
        """
        N = X.shape[0]
        P = X[:, 6:12]
        dyn = self.dyn

        def rhs(t, y):
            Y = y.reshape(N, 6)
            d = np.empty_like(Y)
            d[:, :3] = Y[:, 3:]
            d[:, 3:] = dyn.accel(Y[:, :3], P, t)
            return d.ravel()

        self.n_solves += 1
        with np.errstate(all="ignore"):
            sol = solve_ivp(rhs, (tau[0], tau[-1]), X[:, :6].ravel(), t_eval=tau, rtol=rtol,
                            atol=atol, method=method)
        if sol.status != 0 or sol.y.shape[1] != tau.size or not np.all(np.isfinite(sol.y)):
            return None
        return sol.y.reshape(N, 6, tau.size)[:, :3].transpose(0, 2, 1)

    def propagate(self, x0, t_eval, rtol=1e-11, atol=1e-14, method="DOP853"):
        """
        Nominal trajectory and STM at t_eval.

        Returns
        -------
        sol : object
            sol.y (12, T) with the positions in rows 0-2; sol.status 0, or -1 on failure.
        stts : dict or None
            {1: Phi (T, 12, 12)} with the position rows filled; None on failure.
        """
        n, T = x0.size, t_eval.size
        H = np.diag(self.step)
        X = np.vstack([x0, x0 + H, x0 - H])
        Pos = self.positions(X, t_eval, rtol, atol, method)
        if Pos is None:
            return _Sol(np.zeros((n, 0)), -1), None
        y = np.zeros((n, T))
        y[:3] = Pos[0].T
        Phi = np.zeros((T, n, n))
        Phi[:, :3, :] = ((Pos[1 : n + 1] - Pos[n + 1 :])
                         / (2 * self.step)[:, None, None]).transpose(1, 2, 0)
        return _Sol(y, 0), {1: Phi}


class Setup:
    """
    The Case 2 geometry from a batch.npz: truth, epochs, spacecraft, noise, weights, priors and
    dynamics constants.

    Parameters
    ----------
    b : mapping
        np.load of a Case 2 batch.npz.
    """

    def __init__(self, b):
        self.x0_true = np.asarray(b["x0_true"], float)
        self.tau = np.asarray(b["tau"], float)
        self.sc = np.asarray(b["sc_pos"], float)
        self.sig = np.asarray(b["sigma"], float)
        self.w = np.asarray(b["obs_weights"], float)
        self.psig = np.asarray(b["prior_sigma"], float)
        self.dyn = Degree2Dynamics(*np.asarray(b["dyn_const"], float))
        self.pole = self.dyn.pole
        self.prop = FDPropagator(self.dyn, FD_REL * self.psig)
        self.r_true = self.prop.positions(self.x0_true[None], self.tau)[0]
        sc_state = np.zeros((self.tau.size, 6))
        sc_state[:, :3] = self.sc
        self.sc_state = sc_state

    def data(self, noise_seed):
        """RA/Dec of the truth with noise from noise_seed (as case2_bennu_particle.py)."""
        los = self.r_true - self.sc
        ra, dec, _, _ = radec_and_partials_from_los(los)
        rng = np.random.default_rng(noise_seed)
        ra = ra + rng.normal(0.0, self.sig[0], size=ra.shape)
        dec = dec + rng.normal(0.0, self.sig[1], size=dec.shape)
        y = np.empty(2 * ra.size)
        y[0::2], y[1::2] = ra, dec
        return y

    def reference(self, ref_seed):
        """Reference (prior mean, batch start) from ref_seed (as case2_bennu_particle.py)."""
        x = self.x0_true
        rng = np.random.default_rng(ref_seed)
        dev = np.hstack([
            rng.normal(scale=np.abs(x[0:3] * 0.2), size=3),
            rng.normal(scale=np.abs(x[3:6] * 0.2), size=3),
            rng.normal(scale=np.abs(x[6:7] * 0.01), size=1),
            rng.normal(scale=np.abs(x[7:12] * 0.01), size=5),
        ])
        return x - dev

    def h(self, x):
        """Launch height along the pole relative to the truth [m]."""
        return 1e3 * (x[:3] - self.x0_true[:3]) @ self.pole


class Problem:
    """
    One realization: data y and reference x_ref.

    Parameters
    ----------
    S : Setup
    y : ndarray, shape (2T,)
        Interleaved RA/Dec [rad].
    x_ref : ndarray, shape (12,)
        Reference: prior mean and batch start.
    """

    def __init__(self, S, y, x_ref):
        self.S, self.y, self.x_ref = S, y, x_ref
        self.b = BatchLSQ(propagator=S.prop, tau=S.tau, sc_state=S.sc_state, y_obs=y,
                          sigma=tuple(S.sig), obs_weights=S.w,
                          measurement=radec_and_partials_from_los, wrap=wrap_to_pi)
        self.idx = np.arange(12)

    def batch(self, x_init=None, max_iter=MAX_ITER):
        """
        The Case 2 batch (Stage 1) from x_init (default: the reference), prior N(x_ref,
        psig^2). Gauss-Newton keeps no state but the iterate, so batch(batch(x)[0], m)
        continues the same iteration.

        Returns
        -------
        x : ndarray, shape (12,)
        P : ndarray, shape (12, 12)
            Batch covariance.

        Raises
        ------
        RuntimeError
            If a propagation fails.
        """
        x_init = self.x_ref if x_init is None else x_init
        pri = [norm(loc=(self.x_ref - x_init)[i], scale=self.S.psig[i]) for i in range(12)]
        x1, _, P1 = self.b.gn_with_stm(x0_ref=x_init, priors=pri, max_iter=max_iter, tol=1e-8,
                                       verbose=False, solver="srif", rank_cutoff=False,
                                       halving=True, **KW)
        return x1, P1

    def lin(self, x):
        """Whitened residuals (data, then prior) and their Jacobian at x."""
        r, J = self.b._linearize(x, self.idx, KW)
        R = np.hstack([r, (self.x_ref - x) / self.S.psig])
        A = np.vstack([J, np.diag(1.0 / self.S.psig)])
        return R, A

    def cost(self, x):
        """J(x) = -2 ln p (inf if the propagation fails)."""
        try:
            R, _ = self.lin(x)
        except RuntimeError:
            return np.inf
        return R @ R

    def gn_fixed_h(self, x, Q, iters=20, tol=1e-6):
        """Gauss-Newton with step halving over the span of Q (h0 fixed)."""
        R, A = self.lin(x)
        c = R @ R
        for _ in range(iters):
            z = np.linalg.lstsq(A @ Q, R, rcond=None)[0]
            d = Q @ z
            for _h in range(12):
                try:
                    Rn, An = self.lin(x + d)
                    cn = Rn @ Rn
                except RuntimeError:
                    cn = np.inf
                if cn <= c:
                    break
                d = 0.5 * d
            if cn > c:
                break
            done = c - cn < tol
            x, R, A, c = x + d, Rn, An, cn
            if done:
                break
        return x, c

    def profile(self, x_start, grid, cap=CAP):
        """
        Profile of J along h0 by continuation from x_start outward over grid [m].

        Returns
        -------
        C : ndarray, shape (grid.size,)
            Profile cost (nan where not reached).
        X : ndarray, shape (grid.size, 12)
            Profile points.
        """
        S = self.S
        e = np.zeros(12)
        e[:3] = S.pole
        Q = np.linalg.svd(np.eye(12) - np.outer(e, e))[0][:, :11]
        k0 = int(np.argmin(np.abs(grid - S.h(x_start))))
        C = np.full(grid.size, np.nan)
        X = np.full((grid.size, 12), np.nan)
        for rng_k in (range(k0, grid.size), range(k0 - 1, -1, -1)):
            x = x_start.copy() if rng_k.start == k0 else X[k0].copy()
            for k in rng_k:
                x = x.copy()
                x[:3] += 1e-3 * (grid[k] - S.h(x)) * S.pole
                x, c = self.gn_fixed_h(x, Q)
                C[k], X[k] = c, x
                if c > np.nanmin(C) + cap:
                    break
        return C, X


def local_minima(grid, C, min_bar=0.0):
    """
    Interior local minima of a profile (finite part).

    Returns
    -------
    list of (int, float)
        (grid index, barrier): the barrier is the smallest rise needed to reach a lower point
        (inf for the lowest minimum).
    """
    ok = np.isfinite(C)
    ks = [k for k in range(1, grid.size - 1)
          if ok[k - 1] and ok[k] and ok[k + 1] and C[k] < C[k - 1] and C[k] <= C[k + 1]]
    out = []
    for k in ks:
        bars = []
        for step in (1, -1):
            j, top = k, C[k]
            while 0 <= j + step < grid.size and ok[j + step]:
                j += step
                top = max(top, C[j])
                if C[j] < C[k]:
                    bars.append(top - C[k])
                    break
        bar = min(bars) if bars else np.inf
        if bar >= min_bar:
            out.append((k, bar))
    return out


def one(args):
    """
    One realization.

    Parameters
    ----------
    args : (Setup, int, int)
        Geometry, noise seed, reference seed.

    Returns
    -------
    dict
        noise, ref, h_ref; batch: h_b, cost_b, x_b, sig_hb (batch sigma of h0 [m]), ok_b
        (False: the propagation failed, err holds the message); continuation: h_c, cost_c;
        profile: grid, prof, cost_true; mins (n, 5): refined h0, refined cost, grid h0,
        profile cost, barrier; wall [s]; n_solves.
    """
    S, s_noise, s_ref = args
    t0 = time.time()
    y = S.data(s_noise)
    x_ref = S.reference(s_ref)
    pb = Problem(S, y, x_ref)
    out = dict(noise=s_noise, ref=s_ref, h_ref=S.h(x_ref))
    try:
        xb, Pb = pb.batch()
        out.update(h_b=S.h(xb), cost_b=pb.cost(xb), x_b=xb,
                   sig_hb=1e3 * np.sqrt(S.pole @ Pb[:3, :3] @ S.pole), ok_b=True)
        xc, _ = pb.batch(xb, max_iter=MAX_ITER_CONT)
        out.update(h_c=S.h(xc), cost_c=pb.cost(xc))
    except Exception as ex:
        out.update(h_b=np.nan, cost_b=np.inf, x_b=np.full(12, np.nan), sig_hb=np.nan, ok_b=False,
                   err=str(ex), h_c=np.nan, cost_c=np.inf)
    C, X = pb.profile(S.x0_true.copy(), GRID)
    out.update(grid=GRID, prof=C, cost_true=pb.cost(S.x0_true))
    mins = []
    for k, bar in local_minima(GRID, C):
        try:
            xm, _ = pb.batch(X[k], max_iter=MAX_ITER_CONT)
            mins.append((S.h(xm), pb.cost(xm), GRID[k], C[k], bar))
        except Exception:
            mins.append((np.nan, np.inf, GRID[k], C[k], bar))
    out["mins"] = np.array(mins) if mins else np.zeros((0, 5))
    out["wall"] = time.time() - t0
    out["n_solves"] = S.prop.n_solves
    return out


# ============================================================
# Classification, summary, figure
# ============================================================


def _same(h1, c1, h2, c2):
    return abs(h1 - h2) < TOL_H and abs(c1 - c2) < TOL_C


def classify(r):
    """
    Distinct minima and batch outcome of one realization (dict of one()).

    Returns
    -------
    dict
        n_min / n_min_bar: distinct minima (any barrier / barrier >= BAR); dP, sep: cost gap and
        h0 separation of the second lowest minimum (barrier >= BAR; nan with one); tl_global:
        the truth's lobe (downhill from h0 = 0 on the profile) holds the lowest minimum; cat:
        best (converged at the lowest minimum), far / near (converged at another minimum,
        farther from / closer to the truth in h0 than the lowest one), stalled, failed;
        bar_b: barrier of the batch's minimum; head_best: a stopped batch continues to the
        lowest minimum; dP_b, dh_b, err_h: batch cost above, h0 minus, |h0| from the lowest
        minimum.
    """
    m = []
    for h, c, _, _, bar in sorted(r["mins"].tolist(), key=lambda v: v[1]):
        if not np.isfinite(c):
            continue
        for o in m:
            if _same(o[0], o[1], h, c):
                o[2] = max(o[2], bar)
                break
        else:
            m.append([h, c, bar])
    if (r["ok_b"] and np.isfinite(r["cost_c"])
            and not any(_same(o[0], o[1], r["h_c"], r["cost_c"]) for o in m)):
        m.append([r["h_c"], r["cost_c"], np.nan])  # a minimum outside the profile window
    m.sort(key=lambda o: o[1])
    hg, cg = m[0][0], m[0][1]
    m_bar = [o for o in m if not (o[2] < BAR)]
    g, C = r["grid"], r["prof"]
    k = int(np.argmin(np.abs(g)))
    while True:
        nb = [j for j in (k - 1, k + 1) if 0 <= j < g.size and np.isfinite(C[j]) and C[j] < C[k]]
        if not nb:
            break
        k = min(nb, key=lambda j: C[j])
    tl = min(m, key=lambda o: abs(o[0] - g[k]))
    q = dict(noise=r["noise"], ref=r["ref"], h_ref=r["h_ref"], ok=r["ok_b"], h_b=r["h_b"],
             sig=r["sig_hb"], h_g=hg, c_g=cg, n_min=len(m), n_min_bar=len(m_bar),
             dP=(m_bar[1][1] - cg) if len(m_bar) > 1 else np.nan,
             sep=abs(m_bar[1][0] - hg) if len(m_bar) > 1 else np.nan,
             tl_global=_same(tl[0], tl[1], hg, cg), wall=r["wall"],
             dP_b=r["cost_b"] - cg if r["ok_b"] else np.nan)
    if not r["ok_b"]:
        q.update(cat="failed")
        return q
    lm = next(o for o in m if _same(o[0], o[1], r["h_c"], r["cost_c"]))  # where the batch heads
    q.update(err_h=abs(r["h_b"] - hg), dh_b=r["h_b"] - hg, head_best=lm is m[0], bar_b=lm[2])
    if r["cost_b"] - r["cost_c"] > STALL:
        q["cat"] = "stalled"
    elif lm is m[0]:
        q["cat"] = "best"
    else:
        q["cat"] = "far" if abs(r["h_b"]) > abs(hg) else "near"
    return q


def summary(rows, paper, src):
    """Text summary of the classified realizations (rows) and the paper's realization."""
    N = len(rows)
    f = lambda key: np.array([q.get(key, np.nan) for q in rows], dtype=float)
    cat = np.array([q["cat"] for q in rows])

    def pct(x):
        x = np.asarray(x, bool)
        p = x.mean()
        return (f"{100 * p:5.1f} % +- {100 * 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / x.size):4.1f}"
                f"  ({int(x.sum())}/{x.size})")

    def q3(x):
        return "/".join(f"{v:.2f}" for v in np.nanpercentile(x, [10, 50, 90])) if np.any(np.isfinite(x)) else "-"

    two = f("n_min_bar") >= 2
    ok = cat != "failed"
    dP, dPb, sep, href, err, sig = f("dP"), f("dP_b"), f("sep"), f("h_ref"), f("err_h"), f("sig")
    far, near, st = cat == "far", cat == "near", cat == "stalled"
    other = far | near
    shallow = other & (f("bar_b") < BAR)
    like = far & (dPb >= 2) & (dPb <= 3) & (err >= 20)
    W = 62
    row = lambda lab, x: f"  {lab:{W}s}{pct(x)}"
    L = [
        f"Case 2 seed Monte Carlo: {N} realizations, noise seeds {int(f('noise').min())}-"
        f"{int(f('noise').max())}, reference seeds {int(f('ref').min())}-{int(f('ref').max())}; "
        f"geometry, priors and batch settings of {src}",
        f"(h0 = launch height along the pole relative to the truth; J = -2 ln p; best fit = lowest "
        f"minimum; a second minimum counts when its barrier is >= {BAR}; +- = 95 % binomial "
        f"interval; median wall time {np.median(f('wall')):.1f} s per realization on one core)",
        "",
        "Cost profile along h0",
        row("two or more minima (any barrier)", f("n_min") >= 2),
        row(f"two or more minima (barrier >= {BAR})", two),
        f"    cost gap of the second minimum 10/50/90 %: {q3(dP[two])}",
        f"    separation in h0 [m] 10/50/90 %: {q3(sep[two])}",
        row("    second minimum 2.0-3.0 above the best fit, >= 20 m away",
            two & (dP >= 2) & (dP <= 3) & (sep >= 20)),
        row("the truth's lobe holds the best fit", f("tl_global")),
        row("    given two minima", f("tl_global")[two]),
        "",
        f"Batch (SRIF, full rank, step halving, max_iter {MAX_ITER}, as Case 2)",
        row("converged at the best fit", cat == "best"),
        row("converged at another minimum (the wrong one)", other),
        row("    farther from the truth than the best fit (as the paper)", far),
        row("    closer to the truth than the best fit", near),
        row(f"    of these, minimum with a barrier < {BAR} (a shoulder)", shallow),
        row(f"stopped short at {MAX_ITER} iterations (still descending)", st),
        row("    continuing reaches the best fit", st & (f("head_best") == 1)),
        row("failed (the reference trajectory falls back to Bennu)", cat == "failed"),
        "",
        row("wrong minimum, of the batches that converged", other[ok & ~st]),
        row(f"wrong minimum, given two minima (barrier >= {BAR})", other[two]),
        row("    farther from the truth, given two minima", far[two]),
        row("like the paper (farther, 2.0-3.0 above, >= 20 m away)", like),
        f"    cost above the best fit at a wrong minimum 10/50/90 %: {q3(dPb[other])}",
        f"    stopped short: cost above the best fit 10/50/90 %: {q3(dPb[st])}",
        row("batch h0 more than 3 batch sigmas from the best fit", ok & (err > 3 * sig)),
        row("batch h0 more than 20 m from the best fit", ok & (err > 20)),
        f"  batch sigma of h0: median {np.nanmedian(sig):.2f} m",
        "",
        f"By the reference's offset along the pole (std {href.std():.1f} m): wrong minimum (farther) / "
        f"stopped short",
    ]
    edges = [-np.inf, -50, -25, 0, 25, 50, np.inf]
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (href >= a) & (href < b)
        if sel.any():
            L.append(f"  [{a:+5.0f}, {b:+5.0f}) m  n = {sel.sum():4d}: {100 * other[sel].mean():5.1f} % "
                     f"({100 * far[sel].mean():4.1f} %) / {100 * st[sel].mean():4.1f} %")
    p = paper
    L += [
        "",
        f"Paper realization (noise {PAPER_SEEDS[0]}, reference {PAPER_SEEDS[1]}): reference "
        f"offset {p['h_ref']:+.1f} m ({abs(p['h_ref']) / href.std():.1f} std; "
        f"{np.sum(np.abs(href) >= abs(p['h_ref']))} of {N} Monte Carlo references are as far); "
        f"batch h0 {p['h_b']:+.2f} m, best fit {p['h_g']:+.2f} m, cost gap {p['dP_b']:.2f}; "
        f"outcome: {p['cat']}",
        "",
        f"Outcome: converged = continuing the Gauss-Newton {MAX_ITER_CONT} more iterations lowers "
        f"the cost by at most {STALL}; two minima are the same within {TOL_H} m and {TOL_C} in "
        f"cost; minima = refined local minima of the profile (h0 in [{GRID[0]:.0f}, "
        f"{GRID[-1]:.0f}] m) and the continued batch; farther / closer: |batch h0| larger / "
        f"smaller than |best-fit h0|.",
    ]
    return "\n".join(L)


def figure(rows, paper, out_dir):
    """fig_case2_seed_mc.pdf (see the module docstring)."""
    f = lambda key: np.array([q.get(key, np.nan) for q in rows], dtype=float)
    cat = np.array([q["cat"] for q in rows])
    href, dh, dP = f("h_ref"), f("dh_b"), f("dP")
    two = f("n_min_bar") >= 2
    n_fail = int(np.sum(cat == "failed"))
    sty = {c: (col, mk) for c, _, col, mk in OUTCOMES}

    # y range from the converged batches; a stopped-short batch beyond it sits on the edge
    conv = np.isin(cat, ["best", "far", "near"])
    span = max(np.nanmax(np.abs(dh[conv])), abs(paper["dh_b"])) * 1.25
    off = np.isfinite(dh) & (np.abs(dh) > span)
    dh_plot = np.where(off, np.sign(dh) * 0.93 * span, dh)

    MCMCPlots.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(MCMCPlots.TXT_W, 2.5), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    a = ax[0]
    for c, _, col, mk in OUTCOMES:
        s = cat == c
        big = c != "best"
        a.plot(href[s], dh_plot[s], ls="none", marker=mk, ms=3.2 if big else 2.2, mfc=col,
               mec="white" if big else "none", mew=0.3, alpha=1.0 if big else 0.45,
               zorder=3 if big else 2, rasterized=not big)
    if off.any():
        lo = np.nanmin(np.abs(dh[off]))
        a.text(0.02, 0.03, rf"{int(off.sum())} off scale, $>${lo/1e3:.1f} km" if lo >= 1e3
               else rf"{int(off.sum())} off scale, $>${lo:.0f} m",
               transform=a.transAxes, ha="left", va="bottom", fontsize=6.5, color="0.35")
    col, mk = sty.get(paper["cat"], (MCMCPlots.C_MC, "s"))
    a.plot(paper["h_ref"], paper["dh_b"], ls="none", marker=mk, ms=6.5, mfc=col, mec="k",
           mew=0.8, zorder=5)
    a.annotate("Case 2", (paper["h_ref"], paper["dh_b"]), xytext=(-8, 6),
               textcoords="offset points", ha="right", va="bottom", fontsize=7)
    a.axhline(0.0, color="0.5", lw=0.5, zorder=1)
    a.set_xlabel(r"Reference offset along the pole [m]")
    a.set_ylabel(r"Batch $h_0$ $-$ best fit [m]")
    a.set_title("(a) Where the batch stops", loc="left")
    a.set_ylim(-span, span)

    b = ax[1]
    top = 10.0
    bins = np.arange(0.0, top + 0.5, 0.5)
    stack, cols = [], []
    for c, _, col_, _ in OUTCOMES + [("failed", "", MCMCPlots.C_MC, "")]:
        stack.append(np.minimum(dP[two & (cat == c)], top - 1e-9))  # the last bin holds >= top
        cols.append(col_)
    b.hist(stack, bins=bins, stacked=True, color=cols, edgecolor="white", linewidth=0.4)
    b.axvline(paper["dP_b"], color="k", lw=0.8, ls=":")
    b.text(paper["dP_b"], b.get_ylim()[1] * 0.97, " Case 2", ha="left", va="top", fontsize=7)
    b.set_xlim(bins[0], bins[-1])
    b.set_xticks(np.arange(0, top + 1, 2))
    b.set_xticklabels([f"{t:.0f}" for t in np.arange(0, top, 2)] + [rf"$\geq${top:.0f}"])
    b.set_xlabel(r"Cost gap of the second minimum $\Delta J$")
    b.set_ylabel("Realizations")
    b.set_title(f"(b) Two minima ({int(two.sum())} of {len(rows)})", loc="left")

    handles = [Line2D([], [], ls="none", marker=mk, ms=4, mfc=col_, mec="white", mew=0.3,
                      label=f"{lab} ({int(np.sum(cat == c))})") for c, lab, col_, mk in OUTCOMES]
    if n_fail:
        handles.append(Line2D([], [], ls="none", marker="s", ms=4, mfc=MCMCPlots.C_MC,
                              mec="white", mew=0.3, label=f"Failed ({n_fail})"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    MCMCPlots.set_figure_dir(out_dir)
    MCMCPlots.save_figure(fig, "fig_case2_seed_mc")


def _stamp(p):
    return next((f for f in p.name.split("_") if len(f) == 12 and f.isdigit()), "")


def _newest_batch():
    runs = sorted((p for p in RESULTS.glob(BATCH_GLOB)
                   if p.name.split("_")[-1].isdigit() and (p / "batch.npz").exists()), key=_stamp)
    if not runs:
        sys.exit(f"no results/{BATCH_GLOB}/batch.npz")
    return runs[-1] / "batch.npz"


def _save_npy(path, obj):
    buf = io.BytesIO()
    np.save(buf, np.array(obj, dtype=object), allow_pickle=True)
    MCMCPlots.write_output(path, buf.getvalue(), tag="Seed MC")


def validate(path):
    """The fast batch against the batch saved by a Case 2 run (data, reference, solution)."""
    b = np.load(path, allow_pickle=True)
    S = Setup(b)
    y = S.data(PAPER_SEEDS[0])
    print("data vs batch.npz y_obs: max |dy| / sigma =", np.max(np.abs(y - b["y_obs"])) / S.sig[0])
    xr = S.reference(PAPER_SEEDS[1])
    print("reference vs batch.npz x0_ref: max rel =",
          np.max(np.abs(xr - b["x0_ref"]) / np.abs(b["x0_ref"])))
    pb = Problem(S, np.asarray(b["y_obs"], float), np.asarray(b["x0_ref"], float))
    t = time.time()
    xb, Pb = pb.batch()
    dt = time.time() - t
    x1 = np.asarray(b["x0_ref1"], float)
    print(f"batch: {dt:.1f} s, {S.prop.n_solves} stacked solves; h0 {S.h(xb):+.3f} m (saved "
          f"{S.h(x1):+.3f}), cost {pb.cost(xb):.4f} (saved: {pb.cost(x1):.4f})")
    print("  |x - x0_ref1| / prior sigma:", np.round(np.abs(xb - x1) / S.psig, 5))
    print(f"  batch sigma(h0) {1e3 * np.sqrt(S.pole @ Pb[:3, :3] @ S.pole):.3f} m (saved cov1: "
          f"{1e3 * np.sqrt(S.pole @ b['cov1'][:3, :3] @ S.pole):.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", help="Case 2 batch.npz (default: the newest Case 2b run)")
    ap.add_argument("--n", type=int, default=N_REAL, help="realizations")
    ap.add_argument("--procs", type=int, default=PROCS)
    ap.add_argument("--load", metavar="NPY", help="summary + figure of a finished run")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    MCMCPlots.SHOW_FIGS = SHOW_FIGS
    path = Path(args.batch) if args.batch else _newest_batch()
    if args.validate:
        validate(path)
        return
    S = Setup(np.load(path, allow_pickle=True))
    t0 = time.time()
    if args.load:
        out_dir = Path(args.load).resolve().parent
        res = list(np.load(args.load, allow_pickle=True))
    else:
        out_dir = RESULTS / f"case2_seed_mc_{datetime.now().strftime('%y%m%d%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Seed MC] {args.n} realizations on {args.procs} processes; geometry from "
              f"{path.parent.name}; results in {out_dir}", flush=True)
        jobs = [(S, NOISE0 + i, REF0 + i) for i in range(args.n)]
        res = []
        with multiprocessing.get_context("fork").Pool(args.procs) as pool:
            for i, r in enumerate(pool.imap_unordered(one, jobs)):
                res.append(r)
                if (i + 1) % 50 == 0 or i + 1 == args.n:
                    print(f"[Seed MC] {i + 1}/{args.n}  {time.time() - t0:.0f} s", flush=True)
                    _save_npy(out_dir / "mc.npy", res)
        res.sort(key=lambda r: r["noise"])
        _save_npy(out_dir / "mc.npy", res)
        print(f"[Timing] {args.n} realizations in {time.time() - t0:.0f} s "
              f"({(time.time() - t0) / 60:.1f} min) on {args.procs} processes")
    paper = classify(one((S, *PAPER_SEEDS)))
    rows = [classify(r) for r in res]
    text = summary(rows, paper, path.parent.name)
    print("\n" + text)
    MCMCPlots.write_output(out_dir / "summary.txt", (text + "\n").encode(), tag="Seed MC")
    figure(rows, paper, out_dir)


if __name__ == "__main__":
    main()
