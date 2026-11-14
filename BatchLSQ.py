"""
Stage-1 batch least squares (MAP) shared by the scenario scripts.

Measurements are stacked pairs y = [a0, b0, a1, b1, ...] (RA/Dec, or range/range rate), one pair
per epoch of `tau`, with sigmas (sigma_a, sigma_b) and per-row visibility weights. The model is
a function of the relative state x[:, :rel_dim] - sc_state[:, :rel_dim] that returns
(a, b, da/drel, db/drel).

Solvers (all return x0_ref1, delta_hat1, cov1 with cov1 = inf outside update_idx):

- gn_with_stm(...)                         Gauss-Newton with first-order STMs, unscaled lstsq
                                           solve (the original Bennu batch).
- gn_with_stm(..., lm=True)                same, with the Levenberg-Marquardt safeguard on the
                                           exact MAP cost (the original Ryugu batch).
- gn_with_stm(..., scaled=True)            column-scaled GN (solve in prior-sigma units) with step
                                           halving on the exact MAP cost. The unscaled lstsq drops
                                           directions whose singular values fall below its cutoff
                                           (prior sigmas span ~17 decades in the SPH2 cases), so
                                           the plain GN can stop short along a weakly observable
                                           valley; the scaled solve keeps them.
- gn_with_stm(..., lm=True, scaled=True)   LM safeguard with the scaled solve.
- full_nonlinear_lsq(...)                  SciPy least_squares on the exact model.
"""

import numpy as np
from scipy.optimize import least_squares


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


def _column_scale(A, prior_sig):
    """Solve-for units: prior sigma where the prior is finite, else 1 / column norm of A."""
    col = np.sqrt(np.sum(A**2, axis=0))
    col = np.where(col > 0.0, col, 1.0)
    return np.where(np.isfinite(prior_sig), prior_sig, 1.0 / col)


def _inv_normal(M):
    """(A^T A)^-1, pseudoinverse if singular."""
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print("[WARNING] Covariance matrix is singular, using pseudoinverse")
        return np.linalg.pinv(M)


class BatchLSQ:
    """
    Batch MAP estimate of the initial state (and parameters) from one pass of measurements.

    propagator   STTPropagatorND-like: propagate(x0, t_eval, ...) -> (sol, stts) with stts[1]
                 the STMs; propagate_state_only(x0, t_eval, ...) -> sol
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

    @staticmethod
    def _prior_residuals(delta_total, update_idx, prior_mean, prior_sig):
        """Prior rows of the residual, anchored to the initial reference (finite priors only)."""
        finite = np.isfinite(prior_sig)
        return -(delta_total[update_idx][finite] - prior_mean[finite]) / prior_sig[finite]

    def _linearize(self, x0, delta_total, update_idx, prior_mean, prior_sig, prop_kw, check=False):
        """
        Propagate x0 with its STMs and build the stacked MAP system A d ~ b (measurement rows,
        then prior rows). Returns (r, A, b), r = measurement rows of b. With check=True a failed
        propagation returns None.
        """
        n = x0.size
        N = len(self.tau)
        sol, stts = self.propagator.propagate(x0=x0, t_eval=self.tau, **prop_kw)
        if check and (sol.status != 0 or sol.y.shape[1] != N):
            return None
        x = sol.y[:n, :].T
        Phi = np.asarray(stts[1], dtype=float).reshape(N, n, n)  # Phi(t_k, t0)

        y0, y1, dy0, dy1 = self._model(x)
        r = self._residuals(y0, y1)
        J = self._jacobian(dy0, dy1, Phi, update_idx)

        rows, rhs = [J], [r]
        finite = np.isfinite(prior_sig)
        if np.any(finite):
            Wp = np.diag(1.0 / prior_sig[finite])
            Jp = np.zeros((Wp.shape[0], len(update_idx)))
            Jp[:, finite] = Wp
            rows.append(Jp)
            rhs.append(self._prior_residuals(delta_total, update_idx, prior_mean, prior_sig))
        return r, np.vstack(rows), np.hstack(rhs)

    def _exact_map_cost(self, x0, delta_total, update_idx, prior_mean, prior_sig, prop_kw):
        """Nonlinear MAP cost ||r||^2 + ||prior||^2 (state-only propagation; inf if it fails)."""
        sol = self.propagator.propagate_state_only(
            x0=x0, t_eval=self.tau, show_progress=False, **prop_kw
        )
        if sol.status != 0 or sol.y.shape[1] != len(self.tau):
            return np.inf
        x = sol.y[: x0.size, :].T
        y0, y1, _, _ = self._model(x)
        c = np.sum(self._residuals(y0, y1) ** 2)
        if np.any(np.isfinite(prior_sig)):
            c += np.sum(self._prior_residuals(delta_total, update_idx, prior_mean, prior_sig) ** 2)
        return float(c) if np.isfinite(c) else np.inf

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
        scaled=False,
        lm=False,
        ftol=0.0,
        max_lm_tries=12,
        max_halvings=12,
        pred_tol=1e-6,
        cost_tol=1e-5,
    ):
        """
        Gauss-Newton outer loop. Each iteration:
        1) propagate at current x0_ref (state + STM)
        2) build residual vector r and Jacobian J using STM chain rule
        3) solve linearized MAP: min ||r - J d||^2 + ||(d - m)/s||^2
        4) update x0_ref += embed(d)

        priors      list of scipy.stats.norm on the deltas about x0_ref (length n_upd), or None
        update_idx  indices of the solve-for deltas (default: all)
        tol         stop when the unscaled step norm is below tol (plain GN and LM)
        scaled      column-scaled solve (prior-sigma units) and covariance; without lm, the step
                    is halved (up to max_halvings times) until the exact MAP cost decreases, and
                    the iteration stops when the predicted decrease is < pred_tol or a full step
                    gains < cost_tol
        lm          Levenberg-Marquardt safeguard (see _gn_lm); ftol = relative MAP-cost decrease
                    of an accepted step below which it stops (0 = off)
        """
        x0_ref = np.asarray(x0_ref, dtype=float).copy()
        n = x0_ref.size
        if update_idx is None:
            update_idx = np.arange(n)
        update_idx = np.asarray(update_idx, dtype=int)
        prior_mean, prior_sig = _prior_moments(priors, len(update_idx))
        prop_kw = dict(rtol=rtol, atol=atol, method=method)
        args = (x0_ref, update_idx, prior_mean, prior_sig, prop_kw, max_iter, verbose)

        if lm:
            x0_ref1, delta_total, cov_upd, r_post = self._gn_lm(
                *args, tol=tol, ftol=ftol, max_lm_tries=max_lm_tries, scaled=scaled
            )
        elif scaled:
            x0_ref1, delta_total, cov_upd, r_post = self._gn_halving(
                *args, max_halvings=max_halvings, pred_tol=pred_tol, cost_tol=cost_tol
            )
        else:
            x0_ref1, delta_total, cov_upd, r_post = self._gn_plain(*args, tol=tol)

        cov_full = _embed_cov(cov_upd, update_idx, n)
        # Post-fit residuals are plotted by the scenario scripts (MCMCPlots.plot_mcmc_residuals),
        # batch and MCMC MAP together, so no batch-only figure is produced here.
        return x0_ref1, delta_total, cov_full

    def _gn_plain(self, x0_ref, update_idx, prior_mean, prior_sig, prop_kw, max_iter, verbose, tol):
        """Plain GN: unscaled lstsq solve, full steps, stop when the step norm < tol."""
        n = x0_ref.size
        delta_total = np.zeros(n)
        for it in range(1, max_iter + 1):
            r, A, b = self._linearize(x0_ref, delta_total, update_idx, prior_mean, prior_sig, prop_kw)
            d_upd, *_ = np.linalg.lstsq(A, b, rcond=None)

            d_full = np.zeros(n)
            d_full[update_idx] = d_upd
            x0_ref = x0_ref + d_full
            delta_total = delta_total + d_full

            step_norm = np.linalg.norm(d_upd)
            rms = np.sqrt(np.mean(r**2))
            if verbose:
                print(f"\n[GN it {it:02d}] rms(norm res)={rms:.3e}  step_norm={step_norm:.3e}")
            if step_norm < tol:
                break

        # covariance (A^T A)^-1 of the last linearization
        cov_upd = np.diag(prior_sig**2) if max_iter == 0 else _inv_normal(A.T @ A)
        return x0_ref, delta_total, cov_upd, None

    def _gn_lm(
        self,
        x0_ref,
        update_idx,
        prior_mean,
        prior_sig,
        prop_kw,
        max_iter,
        verbose,
        tol,
        ftol,
        max_lm_tries,
        scaled,
    ):
        """
        GN with a Levenberg-Marquardt safeguard. Each step is checked against the exact MAP cost;
        if the cost does not decrease, the step is damped by appending the rows
        sqrt(lam) * diag(1/s) d = 0, i.e. (A^T A + lam P0^-1) d = A^T b with P0 the prior
        covariance (Marquardt column norm where the prior is flat), and lam is raised x10 until
        it does. lam is relaxed /10 after each accepted step and dropped to 0 (pure GN) once
        negligible. Stops on step norm < tol, or relative MAP-cost decrease < ftol.

        scaled: solve in the units S = 1/d_metric, where the damping rows become sqrt(lam) I:
        (As^T As + lam I) u = As^T b, d = S u. Same damped problem, full-rank solve.
        """
        n = x0_ref.size
        n_upd = len(update_idx)
        delta_total = np.zeros(n)
        finite = np.isfinite(prior_sig)

        lam = 0.0
        cost_cur = None
        for it in range(1, max_iter + 1):
            r, A, b = self._linearize(x0_ref, delta_total, update_idx, prior_mean, prior_sig, prop_kw)
            if cost_cur is None:
                cost_cur = float(np.sum(b**2))

            # damping metric: inverse prior sigma (Marquardt column norm where the prior is flat)
            d_metric = np.where(finite, 1.0 / prior_sig, np.sqrt(np.sum(A**2, axis=0)))
            lam_ref = float(np.max(np.sum(A**2, axis=0) / d_metric**2))
            if scaled:
                S = _column_scale(A, prior_sig)
                As = A * S
                AtA_s, Atb_s = As.T @ As, As.T @ b

            # pure GN first; damp until the nonlinear MAP cost decreases
            accepted = converged = False
            for _ in range(max_lm_tries):
                if scaled:
                    M = AtA_s + lam * np.eye(n_upd) if lam > 0.0 else AtA_s
                    d_upd = S * np.linalg.solve(M, Atb_s)
                else:
                    if lam > 0.0:
                        A_aug = np.vstack([A, np.sqrt(lam) * np.diag(d_metric)])
                        b_aug = np.hstack([b, np.zeros(n_upd)])
                    else:
                        A_aug, b_aug = A, b
                    d_upd, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)

                d_full = np.zeros(n)
                d_full[update_idx] = d_upd

                if lam == 0.0 and np.linalg.norm(d_upd) < tol:
                    # converged: noise-level GN step, accepted without a cost check
                    accepted = converged = True
                    cost_try = cost_cur
                    break

                cost_try = self._exact_map_cost(
                    x0_ref + d_full, delta_total + d_full, update_idx, prior_mean, prior_sig, prop_kw
                )
                if cost_try < cost_cur:
                    accepted = True
                    break
                lam = 1e-3 * lam_ref if lam == 0.0 else 10.0 * lam

            if not accepted:
                print("\n[GN] Levenberg-Marquardt could not decrease the MAP cost: stopping.")
                break

            x0_ref = x0_ref + d_full
            delta_total = delta_total + d_full

            step_norm = np.linalg.norm(d_upd)
            rms = np.sqrt(np.mean(r**2))
            rel_dec = (cost_cur - cost_try) / cost_cur
            if verbose:
                print(
                    f"\n[GN it {it:02d}] rms(norm res)={rms:.3e}  step_norm={step_norm:.3e}  "
                    f"MAP cost: {cost_cur:.4e} -> {cost_try:.4e}  (LM lam={lam:.1e})"
                )
            cost_cur = cost_try

            # relax the damping after a successful step
            lam = lam / 10.0 if lam > 1e-9 * lam_ref else 0.0

            if converged or step_norm < tol:
                break
            if rel_dec < ftol:
                if verbose:
                    print(
                        f"[GN] MAP cost decreased by {100 * rel_dec:.2f}% < "
                        f"{100 * ftol:.0f}%: stopping."
                    )
                break

        # covariance (A^T A)^-1 of the last linearization
        if max_iter == 0:
            cov_upd = np.diag(prior_sig**2)
        elif scaled:
            cov_upd = S[:, None] * _inv_normal(AtA_s) * S[None, :]
        else:
            cov_upd = _inv_normal(A.T @ A)
        return x0_ref, delta_total, cov_upd, None

    def _gn_halving(
        self,
        x0_ref,
        update_idx,
        prior_mean,
        prior_sig,
        prop_kw,
        max_iter,
        verbose,
        max_halvings,
        pred_tol,
        cost_tol,
    ):
        """
        Column-scaled GN with step halving. Solve (As^T As) u = As^T b with As = A diag(S), S the
        prior sigmas (1/column norm where the prior is flat), d = S u. Try the full step and halve
        it until the exact MAP cost ||b||^2 decreases; each trial is a propagation with STMs, so
        the accepted one is the next linearization. Stop when the predicted decrease of the full
        step is < pred_tol, or a full step gains < cost_tol.
        """
        n = x0_ref.size
        delta_total = np.zeros(n)
        prop_kw_quiet = dict(prop_kw, show_progress=False)

        r, A, b = self._linearize(x0_ref, delta_total, update_idx, prior_mean, prior_sig, prop_kw)
        cost = float(b @ b)
        if verbose:
            print(f"\n[GN scaled] start: MAP cost {cost:.4e}")

        for it in range(1, max_iter + 1):
            S = _column_scale(A, prior_sig)
            As = A * S
            u = np.linalg.solve(As.T @ As, As.T @ b)
            pred = float(b @ b - np.sum((b - As @ u) ** 2))
            if pred < pred_tol:
                if verbose:
                    print(f"[GN scaled] predicted MAP-cost decrease {pred:.2e} < {pred_tol:.0e}: converged.")
                break
            d_upd = S * u

            alpha = 1.0
            for _ in range(max_halvings):
                d_full = np.zeros(n)
                d_full[update_idx] = alpha * d_upd
                lin_try = self._linearize(
                    x0_ref + d_full,
                    delta_total + d_full,
                    update_idx,
                    prior_mean,
                    prior_sig,
                    prop_kw_quiet,
                    check=True,
                )
                cost_try = np.inf if lin_try is None else float(lin_try[2] @ lin_try[2])
                if cost_try < cost:
                    break
                alpha *= 0.5
            else:
                print("\n[GN scaled] no MAP-cost decrease along the step: stopping.")
                break

            x0_ref = x0_ref + d_full
            delta_total = delta_total + d_full
            r, A, b = lin_try
            gain, cost = cost - cost_try, cost_try
            if verbose:
                rms = np.sqrt(np.mean(r**2))
                print(
                    f"\n[GN it {it:02d}] rms(norm res)={rms:.3e}  "
                    f"step_norm={np.linalg.norm(alpha * d_upd):.3e}  "
                    f"MAP cost -> {cost:.4e}  (alpha={alpha:g})"
                )
            if alpha == 1.0 and gain < cost_tol:
                if verbose:
                    print(f"[GN scaled] full step gained {gain:.1e} < {cost_tol:.0e}: converged.")
                break

        # covariance S (As^T As)^-1 S at the final linearization (= the returned x0_ref)
        if max_iter == 0:
            cov_upd = np.diag(prior_sig**2)
        else:
            S = _column_scale(A, prior_sig)
            As = A * S
            cov_upd = S[:, None] * _inv_normal(As.T @ As) * S[None, :]
        return x0_ref, delta_total, cov_upd, r

    # ------------------------------------------------------------------ exact nonlinear

    def full_nonlinear_lsq(
        self,
        x0_ref,
        priors=None,
        update_idx=None,
        rtol=1e-10,
        atol=1e-12,
        method="LSODA",
        max_nfev=2000,
        verbose=2,
    ):
        """
        Full nonlinear batch (MAP) using SciPy least_squares (re-propagates every evaluation).
        Returns x0_ref1, delta_hat1, cov1, result.
        """
        x0_ref = np.asarray(x0_ref, dtype=float)
        n = x0_ref.size
        if update_idx is None:
            update_idx = np.arange(n)
        update_idx = np.asarray(update_idx, dtype=int)
        n_upd = len(update_idx)
        prior_mean, prior_sig = _prior_moments(priors, n_upd)
        finite = np.isfinite(prior_sig)

        def residual_vector(delta_upd):
            x0 = x0_ref.copy()
            x0[update_idx] = x0_ref[update_idx] + delta_upd
            sol = self.propagator.propagate_state_only(
                x0=x0, t_eval=self.tau, show_progress=False, rtol=rtol, atol=atol, method=method
            )
            y0, y1, _, _ = self._model(sol.y[:n, :].T)
            r_meas = self._residuals(y0, y1)
            if np.any(finite):
                r_pri = (delta_upd[finite] - prior_mean[finite]) / prior_sig[finite]
                return np.hstack([r_meas, r_pri])
            return r_meas

        result = least_squares(
            fun=residual_vector,
            x0=np.zeros(n_upd),
            method="trf",
            jac="2-point",
            max_nfev=max_nfev,
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
            verbose=verbose,
        )

        delta_hat_full = np.zeros(n)
        delta_hat_full[update_idx] = result.x
        J = result.jac
        try:
            cov_upd = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov_upd = np.linalg.pinv(J.T @ J)
        return x0_ref + delta_hat_full, delta_hat_full, _embed_cov(cov_upd, update_idx, n), result

    # ------------------------------------------------------------------ generic helper

    @staticmethod
    def stt_least_squares(residuals_func, x0, priors=None, max_nfev=40000):
        """
        least_squares on a surrogate (e.g. STT) residual function that already returns
        normalized measurement residuals; priors: list of scipy.stats.norm on the deltas.
        Returns result, cov = (J^T J)^-1.
        """
        if priors is None:
            prior_mean = np.zeros_like(x0)
            prior_sigma = np.full_like(x0, np.inf)
        else:
            prior_mean = np.array([p.mean() for p in priors], dtype=float)
            prior_sigma = np.array([p.std() for p in priors], dtype=float)
            if np.any(prior_sigma <= 0):
                raise ValueError("Prior std must be > 0.")

        def raw_residuals(delta):
            r_meas = residuals_func(delta)
            r_pri = (delta - prior_mean) / prior_sigma
            return np.hstack([r_meas, r_pri])

        result = least_squares(
            fun=raw_residuals,
            x0=x0,
            method="trf",
            jac="2-point",
            max_nfev=max_nfev,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            verbose=2,
        )
        J = result.jac
        cov = np.linalg.inv(J.T @ J)
        return result, cov
