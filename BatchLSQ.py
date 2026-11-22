"""
Stage-1 batch MAP estimate shared by the scenario scripts: iterated batch square-root
information filter (SRIF), or the same batch processed with the U-D covariance factorization.

Measurements are stacked pairs y = [a0, b0, a1, b1, ...] (RA/Dec, or range/range rate), one pair
per epoch of `tau`, with sigmas (sigma_a, sigma_b) and per-row visibility weights. The model is
a function of the relative state x[:, :rel_dim] - sc_state[:, :rel_dim] that returns
(a, b, da/drel, db/drel).

gn_with_stm(...) iterates on the reference x0 (Gauss-Newton with step halving: no damping term,
no scaling). Each iteration propagates x0 with its STMs, maps the whitened measurement rows to
the epoch (H_k Phi(t_k, t0)) and solves the linearized MAP problem for the step d,

    min ||r - J d||^2 + ||R0 (d - d0)||^2,   R0 = diag(1 / prior sigma),  d0 = prior mean - delta,

(delta = sum of the previous steps: the prior stays anchored to the initial reference) with one
of two square-root solvers (Bierman 1977; Tapley, Schutz & Born 2004, ch. 5):

- solver="srif" (default): batch SRIF. The a priori information array [R0 | R0 d0] is updated
  epoch by epoch with Householder triangularizations of [R z; J_k r_k]; then R d = z by back
  substitution and P = R^-1 R^-T.
- solver="ud": the same batch, one scalar measurement at a time, with Bierman's U-D measurement
  update of P = U D U^T from P0 = diag(prior sigma^2). The measurements are mapped to the epoch,
  so there is no time update.

Both work on the raw parameters. Neither forms the normal matrix, and by default neither
truncates small singular values, so directions whose prior sigmas are many decades apart (0.1 km for r0, 1e-17 for C21)
are all kept without column scaling. The covariance is the one of the last linearization.

rank_cutoff=True (SRIF only) gives the same solution as the batch before the SRIF
(numpy.linalg.lstsq of the stacked system, LAPACK gelsd): R d = z is solved with the SVD of R
(the same singular values as the stacked system), dropping those below eps * max(m, n) times the
largest (m = measurement + prior rows), as the default rcond of lstsq does. In raw units this
drops the least-informed direction (case 1a: singular values from 3e2 to 3e16), so the solution
is the minimum-norm one, not the full MAP. The covariance is R^-1 R^-T in both cases.

The step is x0 += d when that lowers the exact MAP cost, else d/2, d/4, ... (Hartley 1961): the
direction is always the SRIF one. halving=False takes the full step every time (the iterated
batch before the SRIF). Optionally (make_chart) the step is solved for in curvilinear chart
coordinates theta, x0 += to_deviation(theta), so that it follows a curved valley (the spin-pole
rotation of case 2, extra/scenario_ISL_SH2.SymmetryChart).
"""

import numpy as np
from scipy.linalg import solve_triangular


def _prior_moments(priors, n_upd):
    """Prior mean / sigma of the solve-for deltas (flat prior when `priors` is None)."""
    if priors is None:
        return np.zeros(n_upd), np.full(n_upd, np.inf)
    prior_mean = np.array([p.mean() for p in priors], dtype=float)
    prior_sig = np.array([p.std() for p in priors], dtype=float)
    return prior_mean, prior_sig


def _embed_cov(cov_upd, update_idx, n):
    """Full n x n covariance, infinite variance for the parameters not solved for."""
    cov_full = np.full((n, n), np.inf)
    cov_full[np.ix_(update_idx, update_idx)] = cov_upd
    return cov_full


def srif_solve(R0, z0, J, r, row_blocks=None, rcond=None):
    """
    Batch SRIF: least-squares solution of [R0; J] d ~ [z0; r].

    The information array [R z] starts at [R0 z0] and takes the rows of each block with a
    Householder triangularization (LAPACK geqrf through numpy.linalg.qr), keeping the n x (n+1)
    upper block. R d = z is solved by back substitution, or, with rcond, with the SVD of R
    keeping the singular values above rcond times the largest (minimum-norm solution).

    Parameters
    ----------
    R0 : ndarray, shape (n, n)
        A priori square-root information matrix (zeros: no a priori information).
    z0 : ndarray, shape (n,)
        A priori information vector, R0 d0.
    J, r : ndarray, shapes (m, n), (m,)
        Whitened measurement rows (mapped to the epoch) and residuals.
    row_blocks : list of index arrays, or None
        Rows processed together (one block per epoch); None: all rows in one block.
    rcond : float or None
        Relative singular-value cutoff of the solve (None: full rank).

    Returns
    -------
    d : ndarray, shape (n,)
    P : ndarray, shape (n, n)
        R^-1 R^-T.
    """
    n = R0.shape[0]
    Rz = np.column_stack([R0, z0])
    for rows in [np.arange(len(r))] if row_blocks is None else row_blocks:
        Rz = np.linalg.qr(np.vstack([Rz, np.column_stack([J[rows], r[rows]])]), mode="r")[:n]
    R, z = Rz[:, :n], Rz[:, n]
    if np.any(np.diag(R) == 0.0):
        raise np.linalg.LinAlgError("SRIF: singular information matrix (unobservable parameter with a flat prior)")
    R_inv = solve_triangular(R, np.eye(n))
    if rcond is None:
        return R_inv @ z, R_inv @ R_inv.T
    U, sv, Vt = np.linalg.svd(R)
    keep = sv > rcond * sv[0]
    return Vt[keep].T @ ((U[:, keep].T @ z) / sv[keep]), R_inv @ R_inv.T


def udu_factor(P):
    """U D U^T factorization of a symmetric positive definite P (U unit upper triangular)."""
    P = np.array(P, dtype=float)
    n = P.shape[0]
    U, D = np.eye(n), np.zeros(n)
    for j in range(n - 1, -1, -1):
        D[j] = P[j, j]
        U[:j, j] = P[:j, j] / D[j]
        P[:j, :j] -= D[j] * np.outer(U[:j, j], U[:j, j])
    return U, D


def ud_solve(P0, d0, J, r):
    """
    Bierman U-D measurement updates: least-squares solution of the same problem as srif_solve,
    processing the rows of J one at a time (unit-variance scalar measurements).

    Parameters
    ----------
    P0 : ndarray, shape (n,) or (n, n)
        A priori covariance (its diagonal when 1-D: then U = I, D = P0); must be finite.
    d0 : ndarray, shape (n,)
        A priori estimate.
    J, r : ndarray, shapes (m, n), (m,)
        Whitened measurement rows (mapped to the epoch) and residuals.

    Returns
    -------
    d : ndarray, shape (n,)
    P : ndarray, shape (n, n)
        U D U^T.
    """
    n = d0.size
    P0 = np.asarray(P0, dtype=float)
    U, D = (np.eye(n), P0.copy()) if P0.ndim == 1 else udu_factor(P0)
    d = np.array(d0, dtype=float)
    for h, y in zip(J, r):
        if not np.any(h):
            continue
        f = U.T @ h
        v = D * f
        b = np.zeros(n)  # unnormalized gain
        alpha = 1.0  # innovation variance, built up column by column
        for j in range(n):
            alpha_prev, alpha = alpha, alpha + v[j] * f[j]
            D[j] *= alpha_prev / alpha
            u_j = U[:j, j].copy()
            U[:j, j] = u_j - (f[j] / alpha_prev) * b[:j]
            b[:j] += v[j] * u_j
            b[j] = v[j]
        d = d + b * ((y - h @ d) / alpha)
    return d, (U * D) @ U.T


class BatchLSQ:
    """
    Batch MAP estimate of the initial state (and parameters) from one pass of measurements.

    propagator   STTPropagatorND-like: propagate(x0, t_eval, ...) -> (sol, stts) with stts[1]
                 the STMs
    tau          (N,) epochs [s since start]
    sc_state     (N, >=rel_dim) observer state at tau, same frame as the propagated state
    y_obs        (2N,) stacked measurement pairs
    sigma        (sigma of the even rows, sigma of the odd rows)
    obs_weights  (2N,) visibility weights
    measurement  rel (N, rel_dim) -> (a, b, da_drel, db_drel)
    rel_dim      3 = line of sight (angles), 6 = relative position and velocity (Doppler)
    wrap         applied to the even-row residuals (e.g. wrap_to_pi for RA), or None
    """

    def __init__(
        self,
        propagator,
        tau,
        sc_state,
        y_obs,
        sigma,
        obs_weights,
        measurement,
        rel_dim=3,
        wrap=None,
    ):
        self.propagator = propagator
        self.tau = tau
        self.sc_state = sc_state
        self.y_obs = y_obs
        self.sigma = tuple(sigma)
        self.obs_weights = obs_weights
        self.measurement = measurement
        self.rel_dim = rel_dim
        self.wrap = wrap

        self.w = np.empty_like(y_obs, dtype=float)
        self.w[0::2] = self.sigma[0]
        self.w[1::2] = self.sigma[1]

        # SRIF row blocks: the two rows of each epoch with a nonzero weight
        w2 = np.asarray(obs_weights, dtype=float).reshape(-1, 2)
        self.epoch_rows = [np.array([2 * k, 2 * k + 1]) for k in np.flatnonzero(np.any(w2 != 0.0, axis=1))]

    # ------------------------------------------------------------------ model pieces

    def _model(self, x):
        """Measurement model at the propagated states x (N, >=rel_dim)."""
        rel = x[:, : self.rel_dim] - self.sc_state[:, : self.rel_dim]
        return self.measurement(rel)

    def _residuals(self, y0, y1):
        """Normalized, visibility-weighted residuals (2N,)."""
        res = np.empty_like(self.y_obs)
        d0 = self.y_obs[0::2] - y0
        res[0::2] = self.wrap(d0) if self.wrap is not None else d0
        res[1::2] = self.y_obs[1::2] - y1
        return res / self.w * self.obs_weights

    def _jacobian(self, dy0, dy1, Phi, update_idx):
        """Normalized Jacobian wrt the solve-for deltas at t0 (2N x n_upd), STM chain rule."""
        n = Phi.shape[1]
        J = np.zeros((2 * len(self.tau), len(update_idx)), dtype=float)
        for k in range(len(self.tau)):
            Hy = np.zeros((2, n), dtype=float)  # dy_k / dx_k
            Hy[0, : dy0.shape[1]] = dy0[k, :]
            Hy[1, : dy1.shape[1]] = dy1[k, :]
            Hx0 = Hy @ Phi[k]
            J[2 * k : 2 * k + 2, :] = Hx0[:, update_idx]
        J[0::2, :] /= self.sigma[0]
        J[1::2, :] /= self.sigma[1]
        return J * self.obs_weights[:, None]

    def _linearize(self, x0, update_idx, prop_kw):
        """Propagate x0 with its STMs: whitened residuals r (2N,) and Jacobian J (2N, n_upd)."""
        n = x0.size
        N = len(self.tau)
        sol, stts = self.propagator.propagate(x0=x0, t_eval=self.tau, **prop_kw)
        if sol.status != 0 or sol.y.shape[1] != N:
            raise RuntimeError(f"propagation of the reference failed (status {sol.status}): the iterate diverged")
        x = sol.y[:n, :].T
        Phi = np.asarray(stts[1], dtype=float).reshape(N, n, n)  # Phi(t_k, t0)
        y0, y1, dy0, dy1 = self._model(x)
        return self._residuals(y0, y1), self._jacobian(dy0, dy1, Phi, update_idx)

    # ------------------------------------------------------------------ Gauss-Newton

    def gn_with_stm(
        self,
        x0_ref,
        priors=None,
        update_idx=None,
        max_iter=10,
        tol=1e-12,
        rtol=1e-12,
        atol=1e-12,
        method="LSODA",
        verbose=True,
        solver="srif",
        make_chart=None,
        max_halvings=20,
        rank_cutoff=False,
        halving=True,
    ):
        """
        Iterated batch (Gauss-Newton). Each iteration:
        1) whitened residuals r and Jacobian J at t0 of the current x0_ref (STM chain rule)
        2) linearized MAP step d: srif_solve (solver="srif") or ud_solve (solver="ud")
        3) stop when the step norm < tol (the step is applied)
        4) x0_ref += embed(d), the step halved until the MAP cost (exact: the trial reference is
           propagated, and its r and J are those of the next iteration) decreases; at most
           max_halvings halvings, then stop

        priors      list of scipy.stats.norm on the deltas about the initial x0_ref (length n_upd),
                    or None (flat; SRIF only)
        update_idx  indices of the solve-for deltas (default: all)
        make_chart  None, or make_chart(x) -> chart about x (e.g. SymmetryChart of
                    extra/scenario_ISL_SH2.py: jacobian_at_ref(), to_deviation(theta)). The step
                    is then solved for in the chart coordinates theta (J -> J dx/dtheta, prior
                    linearized the same way) and applied as x0_ref += to_deviation(theta);
                    the covariance is mapped back to x. Needs all parameters solved for.
        rank_cutoff True: solve R d = z with the default singular-value cutoff of numpy.linalg.lstsq
                    (eps * max(m, n), m = 2N measurement rows + finite prior rows), as the batch
                    before the SRIF did (SRIF only)
        halving     False: always take the full step (no MAP-cost check), as the batch before the
                    SRIF did

        Returns x0_ref1, delta_hat1 (sum of the steps), cov1 (n x n, inf outside update_idx).
        """
        if solver not in ("srif", "ud"):
            raise ValueError(f"solver = {solver!r}: 'srif' or 'ud'")
        x0_ref = np.asarray(x0_ref, dtype=float).copy()
        n = x0_ref.size
        if update_idx is None:
            update_idx = np.arange(n)
        update_idx = np.asarray(update_idx, dtype=int)
        if make_chart is not None and not np.array_equal(update_idx, np.arange(n)):
            raise ValueError("make_chart needs update_idx = all parameters")
        prior_mean, prior_sig = _prior_moments(priors, len(update_idx))
        finite = np.isfinite(prior_sig)
        if solver == "ud" and not np.all(finite):
            raise ValueError("solver = 'ud' needs a finite prior sigma on every solve-for parameter")
        if rank_cutoff and solver != "srif":
            raise ValueError("rank_cutoff needs solver = 'srif'")
        # default rcond of numpy.linalg.lstsq on the stacked [J; prior rows] system
        rcond = np.finfo(float).eps * max(2 * len(self.tau) + int(finite.sum()), len(update_idx)) if rank_cutoff else None
        R0 = np.diag(np.where(finite, 1.0 / prior_sig, 0.0))
        prop_kw = dict(rtol=rtol, atol=atol, method=method)
        tag = solver.upper()

        def embed(d_upd):
            d_full = np.zeros(n)
            d_full[update_idx] = d_upd
            return d_full

        def map_cost(r, delta):
            d0 = prior_mean - delta[update_idx]
            return r @ r + np.sum((d0[finite] / prior_sig[finite]) ** 2)

        delta_total = np.zeros(n)
        cov_upd = np.diag(prior_sig**2)
        r, J = self._linearize(x0_ref, update_idx, prop_kw)
        cost = map_cost(r, delta_total)
        for it in range(1, max_iter + 1):
            d0 = np.where(finite, prior_mean - delta_total[update_idx], 0.0)
            if make_chart is None:
                if solver == "srif":
                    d_upd, cov_upd = srif_solve(R0, R0 @ d0, J, r, self.epoch_rows, rcond=rcond)
                else:
                    d_upd, cov_upd = ud_solve(prior_sig**2, d0, J, r)
                step = lambda a, d_upd=d_upd: a * d_upd
            else:
                # chart about x0_ref: x = x0_ref + to_deviation(theta), dx/dtheta = Jc at theta = 0
                chart = make_chart(x0_ref)
                Jc = chart.jacobian_at_ref()
                Jc_inv = np.linalg.inv(Jc)
                if solver == "srif":
                    th, cov_th = srif_solve(R0 @ Jc, R0 @ d0, J @ Jc, r, self.epoch_rows, rcond=rcond)
                else:
                    th, cov_th = ud_solve(Jc_inv @ np.diag(prior_sig**2) @ Jc_inv.T, Jc_inv @ d0, J @ Jc, r)
                d_upd = chart.to_deviation(th)
                cov_upd = Jc @ cov_th @ Jc.T
                step = lambda a, chart=chart, th=th: chart.to_deviation(a * th)

            if verbose:
                print(
                    f"\n[{tag} it {it:02d}] rms(norm res)={np.sqrt(np.mean(r**2)):.3e}  "
                    f"MAP cost={cost:.6e}  step_norm={np.linalg.norm(d_upd):.3e}"
                )
            if np.linalg.norm(d_upd) < tol:
                x0_ref = x0_ref + embed(d_upd)
                delta_total = delta_total + embed(d_upd)
                break

            # step halving: the full step, or the first of its halves that lowers the MAP cost
            # (halving=False: the full step)
            for h in range(max_halvings + 1 if halving else 1):
                d_try = embed(step(0.5**h))
                try:
                    r_try, J_try = self._linearize(x0_ref + d_try, update_idx, prop_kw)
                    cost_try = map_cost(r_try, delta_total + d_try)
                except RuntimeError:
                    if not halving:
                        raise
                    cost_try = np.inf
                if cost_try < cost or not halving:
                    break
            else:
                if verbose:
                    print(f"[{tag}] no MAP-cost decrease along the step (halved {max_halvings} times): stop")
                break
            if verbose and h > 0:
                print(f"[{tag} it {it:02d}] step halved {h} time(s): MAP cost={cost_try:.6e}")
            x0_ref, delta_total = x0_ref + d_try, delta_total + d_try
            r, J, cost = r_try, J_try, cost_try

        # Post-fit residuals are plotted by the scenario scripts (MCMCPlots.plot_mcmc_residuals),
        # batch and MCMC MAP together, so no batch-only figure is produced here.
        return x0_ref, delta_total, _embed_cov(cov_upd, update_idx, n)
