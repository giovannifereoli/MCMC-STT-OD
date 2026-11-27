"""
MCMC orbit determination: MCMCModel (emcee sampler, priors, whitening, diagnostics,
chain save/load) and ChainReplay (a saved chain in place of the emcee sampler).

Figures live in MCMCPlots.py (the plot_* methods below forward to it); the posterior
Gaussian-mixture split and its STM propagation live in GMM.py.
"""

import io
import multiprocessing
import os
import time
from pathlib import Path

import emcee
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import (
    basinhopping,
    differential_evolution,
    dual_annealing,
    least_squares,
    minimize,
)

# Only the commented-out run_multinest/run_dynesty block at the end of MCMCModel uses these
import dynesty  # noqa: F401
from dynesty import utils as dyfunc  # noqa: F401

import MCMCPlots

# Model whose log-probability the fork-Pool workers of MCMCModel.run evaluate: set just
# before the pool forks, cleared after the run.
_FORK_MODEL = None


class _ForkedLogProb:
    """
    Picklable log-probability for the fork Pool of MCMCModel.run.

    Pool.map pickles the function it maps for every task. A bound method of the model
    pickled the whole model, self.sampler included, and emcee's EnsembleSampler.__getstate__
    sets pool = None on the live sampler: from the second step on, every walker was
    evaluated in the parent process. This wrapper pickles to a method name only; the
    workers, forked after _FORK_MODEL is set, call that method on their copy of the model.
    """

    def __init__(self, name):
        """name: MCMCModel method to call (log_prob or log_prob_whitened)."""
        self.name = name

    def __call__(self, theta):
        """Log-probability of theta on the forked model."""
        return getattr(_FORK_MODEL, self.name)(theta)


class ChainReplay:
    """
    Stand-in for the emcee sampler, rebuilt from a chain saved by MCMCModel.save_chain.

    It has get_chain / get_log_prob / get_autocorr_time / acceptance_fraction / nwalkers /
    iteration as emcee has them, so the diagnostics and journal figures run unchanged.

    The run is stored every `stride` steps (stored index i = step (i + 1) * stride - 1, as
    emcee's get_chain(thin=stride) returns it). discard / thin are in steps of the original
    run: thin is rounded to a multiple of the stride, so pass multiples of it for exact
    results (plot_mcmc_traces and plot_mcmc_autocorrelation do).

    Parameters
    ----------
    full_chain : array_like, shape (n_stored, n_walkers, ndim)
        The whole run, burn-in included, every `stride` steps, in sampled coordinates.
    full_log_prob : array_like, shape (n_stored, n_walkers)
        Log-posterior of the stored states.
    stride : int
        Steps of the original run between two stored states.
    n_steps : int
        Length of the original run (steps per walker).
    tau : array_like, shape (ndim,), optional
        Integrated autocorrelation time [steps], estimated on the full-resolution run.
    acceptance_fraction : array_like, shape (n_walkers,), optional
        Acceptance fraction per walker (NaN when not given).

    Attributes
    ----------
    full_chain, full_log_prob : ndarray
        As given (float64).
    stride : int
    iteration : int
        n_steps, as emcee's sampler.iteration.
    nwalkers, ndim : int
        From the shape of full_chain.
    tau : ndarray or None
    acceptance_fraction : ndarray, shape (n_walkers,)
    """

    def __init__(self, full_chain, full_log_prob, stride, n_steps, tau=None,
                 acceptance_fraction=None):
        """Store the saved run; see the class docstring for the parameters."""
        self.full_chain = np.asarray(full_chain, dtype=float)  # (n_stored, W, d)
        self.full_log_prob = np.asarray(full_log_prob, dtype=float)  # (n_stored, W)
        self.stride = int(stride)
        self.iteration = int(n_steps)
        self.nwalkers = self.full_chain.shape[1]
        self.ndim = self.full_chain.shape[2]
        self.tau = None if tau is None else np.asarray(tau, dtype=float)
        self.acceptance_fraction = (
            np.full(self.nwalkers, np.nan)
            if acceptance_fraction is None
            else np.asarray(acceptance_fraction, dtype=float)
        )

    def _select(self, arr, discard=0, thin=1, flat=False):
        """
        Stored rows as emcee's get_chain(discard, thin, flat) would return them.

        Parameters
        ----------
        arr : ndarray, shape (n_stored, n_walkers, ...)
            full_chain or full_log_prob.
        discard : int
            Steps of the original run dropped from the start.
        thin : int
            Keep every thin-th step of the original run, rounded to k = max(1, round(thin /
            stride)) stored rows.
        flat : bool
            Merge the step and walker axes.

        Returns
        -------
        ndarray
            arr[i0::k], with i0 the first stored row at or after step
            discard + k * stride - 1 (emcee's first kept step for that thinning);
            (n * n_walkers, ...) if flat.
        """
        k = max(1, int(round(thin / self.stride)))
        i0 = max(0, int(np.ceil((discard + k * self.stride) / self.stride)) - 1)
        out = arr[i0::k]
        return out.reshape((-1,) + out.shape[2:]) if flat else out

    def get_chain(self, discard=0, thin=1, flat=False):
        """
        The stored chain, as emcee's EnsembleSampler.get_chain.

        Parameters
        ----------
        discard, thin : int
            In steps of the original run (see _select).
        flat : bool
            Merge the step and walker axes.

        Returns
        -------
        ndarray
            (n, n_walkers, ndim), or (n * n_walkers, ndim) if flat.
        """
        return self._select(self.full_chain, discard, thin, flat)

    def get_log_prob(self, discard=0, thin=1, flat=False):
        """
        The stored log-posterior, as emcee's EnsembleSampler.get_log_prob.

        Parameters
        ----------
        discard, thin : int
            In steps of the original run (see _select).
        flat : bool
            Merge the step and walker axes.

        Returns
        -------
        ndarray
            (n, n_walkers), or (n * n_walkers,) if flat.
        """
        return self._select(self.full_log_prob, discard, thin, flat)

    def get_autocorr_time(self, tol=50, **kwargs):
        """
        The tau estimated on the full-resolution run at save time.

        Parameters
        ----------
        tol : float
            Ignored (accepted for compatibility with emcee).
        **kwargs
            Ignored.

        Returns
        -------
        ndarray, shape (ndim,)
            A copy of the stored tau [steps].

        Raises
        ------
        emcee.autocorr.AutocorrError
            If the chain was saved without tau.
        """
        if self.tau is None:
            raise emcee.autocorr.AutocorrError(np.nan, "no tau stored with the chain")
        return self.tau.copy()


class ValleyMap:
    """
    Valley-straightening coordinates: a triangular (Knothe-Rosenblatt type) transport map of
    the whitened coordinates z that unbends a curved posterior valley (MCMCModel
    .setup_valley_map, build_valley_map).

    The sampled coordinates v = (h, w) map to

        z(v) = h e + Q (y*(h) + C(h) w),

    with e the unit valley direction, Q (d, d - 1) an orthonormal basis of its complement, y*(h)
    the profile path (the best complement coordinates at fixed h) and C(h) the lower Cholesky
    factor of the conditional covariance there. The map is one-to-one and lower triangular in
    the basis [e, Q], so log |dz/dv| = sum log |diag C(h)|; the sampled density is
    p(z(v)) |dz/dv|, the same posterior. Beyond the profile grid, y* and C are held at the end
    values (a shift: still exact).

    y*(h) is a cubic spline, C(h) is interpolated entry-wise with a PCHIP (no overshoot, so
    the diagonal stays positive). Picklable.

    Parameters
    ----------
    e : array_like, shape (d,)
    Q : array_like, shape (d, d - 1)
    H : array_like, shape (n,)
        Profile grid of h (increasing).
    Y : array_like, shape (n, d - 1)
        y*(H).
    C : array_like, shape (n, d - 1, d - 1)
        Lower Cholesky factors at H.
    cost : array_like, shape (n,), optional
        Profile cost |r|^2 at H (diagnostics only).
    """

    def __init__(self, e, Q, H, Y, C, cost=None):
        """Store the profile and build its interpolants; see the class docstring."""
        self.e = np.asarray(e, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.cost = None if cost is None else np.asarray(cost, dtype=float)
        self.ndim = self.e.size
        self._il = np.tril_indices(self.ndim - 1)
        self._spY = CubicSpline(self.H, self.Y, axis=0)
        self._spC = PchipInterpolator(self.H, self.C[:, self._il[0], self._il[1]], axis=0)

    def _parts(self, h):
        """y*(h) (n, d - 1) and C(h) (n, d - 1, d - 1), held constant beyond the grid."""
        hc = np.clip(h, self.H[0], self.H[-1])
        M = np.zeros((hc.size, self.ndim - 1, self.ndim - 1))
        M[:, self._il[0], self._il[1]] = self._spC(hc)
        return self._spY(hc), M

    def to_white(self, V):
        """
        Whitened coordinates of sampled points.

        Parameters
        ----------
        V : array_like, shape (n, d) or (d,)

        Returns
        -------
        Z : ndarray, shape (n, d)
        logdet : ndarray, shape (n,)
            log |dz/dv|.
        """
        V = np.atleast_2d(np.asarray(V, dtype=float))
        h, w = V[:, 0], V[:, 1:]
        ys, M = self._parts(h)
        y = ys + np.einsum("nij,nj->ni", M, w)
        logdet = np.sum(np.log(np.abs(np.diagonal(M, axis1=1, axis2=2))), axis=1)
        return h[:, None] * self.e[None, :] + y @ self.Q.T, logdet

    def from_white(self, Z):
        """
        Sampled coordinates of whitened points (inverse of to_white).

        Parameters
        ----------
        Z : array_like, shape (n, d) or (d,)

        Returns
        -------
        ndarray, shape (n, d)
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        h = Z @ self.e
        ys, M = self._parts(h)
        w = np.linalg.solve(M, (Z @ self.Q - ys)[..., None])[..., 0]
        return np.column_stack([h, w])

    def to_white_chain(self, chain, chunk=50000):
        """
        to_white of a chain (n_steps, n_walkers, d), in chunks of rows (memory).

        Returns
        -------
        Z : ndarray, shape (n_steps, n_walkers, d)
        logdet : ndarray, shape (n_steps, n_walkers)
        """
        chain = np.asarray(chain, dtype=float)
        flat = chain.reshape(-1, chain.shape[-1])
        Z = np.empty_like(flat)
        ld = np.empty(flat.shape[0])
        for i in range(0, flat.shape[0], chunk):
            Z[i : i + chunk], ld[i : i + chunk] = self.to_white(flat[i : i + chunk])
        return Z.reshape(chain.shape), ld.reshape(chain.shape[:-1])

    def arrays(self):
        """The defining arrays, for save_chain (keys valley_<name>)."""
        out = dict(e=self.e, Q=self.Q, H=self.H, Y=self.Y, C=self.C)
        if self.cost is not None:
            out["cost"] = self.cost
        return out


def valley_basis(e):
    """
    Orthonormal complement Q (d, d - 1) of the unit vector e (QR of [e, I]).
    """
    e = np.asarray(e, dtype=float)
    Qf = np.linalg.qr(np.column_stack([e, np.eye(e.size)]))[0]
    Q = Qf[:, 1:]
    return Q - np.outer(e, e @ Q)  # exact orthogonality to e (round-off)


def build_valley_map(residuals, e, z_start, dh=0.1, dlogp_stop=25.0, h_max=50.0,
                     residuals_batch=None, fd_step=1e-4, verbose=True):
    """
    Profile the posterior along the direction e of whitened space and build its ValleyMap.

    At each h of a grid of step dh (through the start, both ways), the other d - 1 coordinates
    are fitted by Levenberg-Marquardt (y*(h), warm-started from the neighbour), and C(h) is the
    Cholesky factor of the Gauss-Newton conditional covariance inv(J^T J) there. Each side stops
    once the profile log-posterior is more than dlogp_stop below the best point found so far
    (no posterior mass beyond), at |h - h_start| > h_max, or when a fit fails. The quality of
    the map only sets the sampling efficiency: any profile gives an exact sampler.

    Parameters
    ----------
    residuals : callable
        z (d,) -> r (m,), with -log p(z) = 0.5 |r|^2 + const (data and prior residuals).
    e : array_like, shape (d,)
        Valley direction in whitened space (normalized here).
    z_start : array_like, shape (d,)
        Start of the profile (e.g. the batch or MAP point in whitened coordinates).
    dh : float
        Grid step of h [whitened units].
    dlogp_stop : float
        Stop a side this far below the best profile log-posterior.
    h_max : float
        Maximum |h - h_start|.
    residuals_batch : callable, optional
        Z (n, d) -> R (n, m), vectorized residuals; when given, the Jacobian is a central
        difference of step fd_step evaluated in one call (else least_squares differentiates
        residuals itself).
    fd_step : float
        Step of that central difference [whitened units].
    verbose : bool
        Print the profile.

    Returns
    -------
    ValleyMap
    """
    e = np.asarray(e, dtype=float)
    e = e / np.linalg.norm(e)
    Q = valley_basis(e)
    d1 = Q.shape[1]
    z_start = np.asarray(z_start, dtype=float)

    def fit(h, y0):
        base = h * e
        if residuals_batch is None:
            fun, jac = (lambda y: np.asarray(residuals(base + Q @ y), dtype=float)), "2-point"
        else:
            def fun(y):
                return np.asarray(residuals_batch((base + Q @ y)[None]), dtype=float)[0]

            def jac(y):
                Yp = y[None, :] + fd_step * np.vstack([np.eye(d1), -np.eye(d1)])
                R = np.asarray(residuals_batch(base[None, :] + Yp @ Q.T), dtype=float)
                return ((R[:d1] - R[d1:]) / (2.0 * fd_step)).T
        try:
            sol = least_squares(fun, y0, jac=jac, method="lm", xtol=1e-12, ftol=1e-12,
                                gtol=1e-12)
            J = sol.jac if residuals_batch is None else jac(sol.x)
            cov = np.linalg.inv(J.T @ J)
            Cf = np.linalg.cholesky(0.5 * (cov + cov.T))
        except (ValueError, np.linalg.LinAlgError):
            return None
        cost = 2.0 * sol.cost
        if not (np.isfinite(cost) and np.all(np.isfinite(Cf))):
            return None
        return sol.x, Cf, cost

    h0 = dh * np.round((e @ z_start) / dh)
    first = fit(h0, Q.T @ z_start)
    if first is None:
        raise RuntimeError("build_valley_map: the profile fit failed at the start point")
    pts = {0: first}
    c_best = first[2]
    for side in (1, -1):
        y, k = first[0], 0
        while k * dh < h_max:
            k += 1
            res = fit(h0 + side * k * dh, y)
            if res is None:
                if verbose:
                    print(f"[Valley] fit failed at h = {h0 + side * k * dh:+.2f}: side stopped")
                break
            pts[side * k] = res
            y = res[0]
            c_best = min(c_best, res[2])
            if 0.5 * (res[2] - c_best) > dlogp_stop:
                break
    ks = sorted(pts)
    H = h0 + dh * np.array(ks, dtype=float)
    Y = np.array([pts[k][0] for k in ks])
    C = np.array([pts[k][1] for k in ks])
    cost = np.array([pts[k][2] for k in ks])
    if verbose:
        cmin = cost.min()
        print(
            f"[Valley] profile along e: {H.size} points, h = {H[0]:+.2f} ... {H[-1]:+.2f} "
            f"(step {dh:g}); min cost {cmin:.3f} at h = {H[np.argmin(cost)]:+.2f}"
        )
        loc = [i for i in range(1, H.size - 1) if cost[i] < cost[i - 1] and cost[i] <= cost[i + 1]]
        print(
            "[Valley] local minima of the profile: "
            + ", ".join(f"h {H[i]:+.2f} (dlogp {-0.5 * (cost[i] - cmin):.3f})" for i in loc)
        )
        step = max(1, int(round(1.0 / dh)))
        print("[Valley] h, -dlogp: " + ", ".join(
            f"{H[i]:+.1f} {0.5 * (cost[i] - cmin):.2f}" for i in range(0, H.size, step)))
    return ValleyMap(e, Q, H, Y, C, cost)


class MCMCModel:
    """
    Bayesian parameter estimation with emcee.

    Priors, likelihood from normalized residuals, optional whitening, MAP start,
    sampling, diagnostics and chain save/load.

    The log-posterior is

        log p(theta) = sum_i log prior_i(theta_i) - 0.5 * |r(theta)|^2,

    with r = residuals_func(theta) the residuals already normalized by the measurement
    noise. theta is in model coordinates (physical, unwhitened: the residuals_func input).
    After setup_whitening / setup_whitening_from_priors the walkers move in the whitened
    coordinates z = L^{-1} (theta - mu) instead; the map is linear, so no Jacobian term is
    needed. run() maps the samples back to model coordinates. setup_valley_map further
    straightens a curved valley of the whitened posterior (ValleyMap, a triangular transport
    map with its Jacobian): the walkers then move in its coordinates, and run() maps the
    chain back to z.

    Parameters
    ----------
    residuals_func : callable
        theta (ndim,) -> normalized residuals (N,). The fork-pool workers inherit the model
        (only a method name is pickled per task), but keep it a top-level function or a
        class with __call__, not a closure, so that the model stays picklable.
    initial_params : array_like, shape (ndim,)
        Start of the MAP search and of the walkers (model coordinates).
    param_priors : sequence of scipy.stats frozen distributions
        One prior per parameter, assumed independent. logpdf is always used; mean / std /
        ppf by whitening, prior_residuals, the walker start and the optimizer bounds.
    observed_data : object
        Stored as self.observed_data; not used by the class.

    Attributes
    ----------
    ndim : int
        Number of parameters.
    sampler : emcee.EnsembleSampler or ChainReplay or None
        Set by run() or load_chain().
    samples : ndarray, shape (n, ndim) or None
        Flat post-burn-in, thinned posterior samples in model coordinates.
    log_probs : ndarray, shape (n,) or None
        Log-posterior of samples.
    chain : ndarray, shape (n_steps, n_walkers, ndim) or None
        Post-burn-in, thinned chain in sampled (whitened, if whitening is on) coordinates.
    log_prob_chain : ndarray, shape (n_steps, n_walkers) or None
        Log-posterior of chain.
    tau : ndarray, shape (ndim,) or None
        Integrated autocorrelation time per parameter [steps].
    burn_in, thin : int or None
        The values run() used.
    is_whitened : bool
        Only exists after whitening is set up: read it with getattr(model, "is_whitened",
        False).
    whiten_mean, whiten_L, whiten_Linv : ndarray
        Whitening map theta = whiten_L @ z + whiten_mean, and inv(whiten_L).
    valley : ValleyMap or None
        Valley-straightening coordinates of the walkers (setup_valley_map); None: off.
    """
    def __init__(self, residuals_func, initial_params, param_priors, observed_data):
        """Store the problem; see the class docstring for the parameters."""
        self.residuals_func = residuals_func
        self.initial_params = np.array(initial_params)
        self.param_priors = param_priors
        self.observed_data = observed_data
        self.ndim = len(initial_params)
        self.sampler = None
        self.samples = None
        self.log_probs = None
        self.chain = None
        self.log_prob_chain = None
        self.tau = None
        self.burn_in = None
        self.thin = None
        self.valley = None

    def log_prior(self, theta):
        """
        Sum of the independent prior log-densities.

        Parameters
        ----------
        theta : array_like, shape (ndim,)
            Parameters in model coordinates.

        Returns
        -------
        float
            sum_i param_priors[i].logpdf(theta[i]), or -inf as soon as one term is not
            finite.
        """
        # NOTE: This assumes that the priors are independent,
        # also they are scipy.stats objects!
        lp = 0.0
        for i, prior in enumerate(self.param_priors):
            lp_i = prior.logpdf(theta[i])
            if not np.isfinite(lp_i):
                return -np.inf
            lp += lp_i
        return lp

    def log_likelihood(self, theta):
        """
        Gaussian log-likelihood -0.5 * |r|^2 of the normalized residuals (constant dropped).

        Parameters
        ----------
        theta : array_like, shape (ndim,)
            Parameters in model coordinates.

        Returns
        -------
        float
        """
        # NOTE: covariance already in the residuals_func
        residuals_normalized = self.residuals_func(theta)
        return -0.5 * np.sum(residuals_normalized**2)

    def log_posterior(self, theta):
        """
        Unnormalized log-posterior log_prior + log_likelihood.

        The likelihood is not evaluated when the prior is zero.

        Parameters
        ----------
        theta : array_like, shape (ndim,)
            Parameters in model coordinates.

        Returns
        -------
        float
            -inf outside the prior support.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def log_prob(self, theta):
        """
        Log-probability passed to emcee when whitening is off (= log_posterior).

        Parameters
        ----------
        theta : array_like, shape (ndim,)
            Parameters in model coordinates.

        Returns
        -------
        float
        """
        return self.log_posterior(theta)

    def prior_residuals(self, theta):
        """
        Priors as pseudo-residuals (theta_i - mean_i) / std_i, for a least-squares MAP.

        Exact for Gaussian priors, an approximation for the others.

        Parameters
        ----------
        theta : array_like, shape (ndim,)
            Parameters in model coordinates.

        Returns
        -------
        ndarray
            One entry per prior with a finite mean and a finite std > 0. Priors without them
            (or whose mean / std raise) are skipped silently, so the result can be shorter
            than ndim.
        """
        r = []
        for i, p in enumerate(self.param_priors):
            # Try Normal-like: use mean and std if available
            try:
                mu = p.mean()
                sig = p.std()
                if np.isfinite(mu) and np.isfinite(sig) and sig > 0:
                    r.append((theta[i] - mu) / sig)
            except Exception:
                pass
        return np.array(r, dtype=float)

    def optimize_initial_guess(
        self,
        method="Powell",
        disp=True,
        n_iter=1000,
        use_bounds_from_priors=True,
    ):
        """
        Maximize the log-posterior from initial_params (run() starts the walkers there).

        With whitening on, the search runs in whitened coordinates (start and bounds mapped
        by whiten_Linv) and the optimum is mapped back.

        Parameters
        ----------
        method : str
            'dual_annealing' (maxiter=n_iter, no local search, seed 42), 'basinhopping'
            (Powell local minimizer, niter=n_iter, no bounds) or 'differential_evolution'
            (popsize 100, tol 1e-12, polished, seed 42), case-insensitive; 'LSQ' (exact
            case): scipy least_squares (trf, 2-point Jacobian) on the measurement residuals
            stacked with prior_residuals, without bounds; any other name goes to
            scipy.optimize.minimize(method=method, bounds=...).
        disp : bool
            Optimizer verbosity (least_squares: verbose=2).
        n_iter : int
            Iterations of the global optimizers.
        use_bounds_from_priors : bool
            Bounds [ppf(1e-6), ppf(1 - 1e-6)] of each prior. dual_annealing and
            differential_evolution need them.

        Returns
        -------
        ndarray, shape (ndim,)
            The optimum in model coordinates (also when the optimizer reports failure, which
            is printed as a warning).

        Notes
        -----
        'LSQ' works in whitened coordinates when whitening is on, in model coordinates
        otherwise. With whitening on, the bounds are the smallest box in whitened
        coordinates that contains the image z = whiten_Linv (x - whiten_mean) of the prior
        box. For a diagonal whiten_L (as from setup_whitening_from_priors) that image is the
        box itself; for a full whiten_L (setup_whitening(cov)) the box also admits some
        points outside the prior box, where the log-prior decides.
        """
        print("")
        print(f"\n[Optimization] Starting optimization using method: {method}")

        # Set log-posterior and whitening flags
        use_whitened = getattr(self, "is_whitened", False)
        log_prob_func = self.log_prob_whitened if use_whitened else self.log_prob

        # Prepare starting point
        x0 = self.initial_params.copy()
        if use_whitened:
            x0 = self.whiten_Linv @ (x0 - self.whiten_mean)

        def objective(theta):
            """Negative log-posterior (whitened if whitening is on)."""
            return -log_prob_func(theta)

        # Build bounds from priors if requested
        bounds = None
        if use_bounds_from_priors:
            bounds = [(p.ppf(1e-6), p.ppf(1 - 1e-6)) for p in self.param_priors]
            if use_whitened:
                # Box around the image of the prior box under z = Linv (x - m):
                # z_i ranges over sum_j min/max(Linv_ij (lo_j - m_j), Linv_ij (hi_j - m_j)).
                bounds_array = np.array(bounds, dtype=float).T  # shape (2, ndim)
                Linv = np.asarray(self.whiten_Linv, dtype=float)
                with np.errstate(invalid="ignore"):
                    A = Linv * (bounds_array[0] - self.whiten_mean)[None, :]
                    B = Linv * (bounds_array[1] - self.whiten_mean)[None, :]
                A = np.where(Linv == 0.0, 0.0, A)  # avoid 0 * inf = nan
                B = np.where(Linv == 0.0, 0.0, B)
                bounds = list(
                    zip(np.minimum(A, B).sum(axis=1), np.maximum(A, B).sum(axis=1))
                )

        # Choose optimizer
        if method.lower() == "dual_annealing":
            print("[Optimization] Using global optimizer: dual_annealing")
            result = dual_annealing(
                objective,
                bounds=bounds,
                x0=x0,
                maxiter=n_iter,
                no_local_search=True,
                seed=42,
            )
        elif method.lower() == "basinhopping":
            print("[Optimization] Using global+local optimizer: basinhopping")
            result = basinhopping(
                objective,
                x0,
                minimizer_kwargs={"method": "Powell", "options": {"disp": disp}},
                niter=n_iter,
                disp=disp,
            )
        elif method.lower() == "differential_evolution":
            print("[Optimization] Using global optimizer: differential_evolution")
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=n_iter,
                polish=True,
                seed=42,
                disp=disp,
                popsize=100,  # larger population helps refine exploration
                tol=1e-12,  # tighter tolerance for stopping
            )
        elif method == "LSQ":
            print("[Optimization] Using least squares: least_squares(method='trf')")

            def residuals_map(z):
                """
                Measurement plus prior residuals at theta = whiten_L @ z + whiten_mean
                (theta = z without whitening).
                """
                theta = self.whiten_L @ z + self.whiten_mean if use_whitened else z
                r_meas = self.residuals_func(theta)
                r_pri = self.prior_residuals(theta)
                return np.concatenate([r_meas, r_pri])

            result = least_squares(
                residuals_map,
                x0,
                method="trf",
                jac="2-point",
                verbose=2 if disp else 0,
            )
        else:
            print(f"[Optimization] Using local optimizer: {method}")
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                options={"disp": disp},
            )

        # Check result and return in original space
        if not result.success:
            print(f"[Optimization] Warning: {result.message}")
        else:
            print(f"[Optimization] Success: {result.message}")

        x_opt = result.x
        # Transform back to original space if whitened, ALWAYS
        if use_whitened:
            x_opt = self.whiten_L @ x_opt + self.whiten_mean

        print(f"[Optimization] Optimal θ: {x_opt}")
        return x_opt

    def run(
        self,
        n_samples=5000,
        n_walkers=50,
        burn_in=None,
        thin=None,
        burn_in_frac=2.0,
        thin_frac=0.5,
        spherical_spread=1e-4,
        method_optimize="Powell",
        use_demoves=False,
        stretch_a=2.0,
        use_optimize=True,
        thin_by=1,
        n_tau_target=None,
        max_samples=None,
        check_every=None,
        stranded_dlogp=None,
    ):
        """
        Run the emcee ensemble sampler, then set the burn-in and thinning from tau.

        Steps: MAP start (optimize_initial_guess, unless use_optimize=False), walkers in a
        small ball around it, sampling on a fork Pool, tau estimate (retried with tol=0),
        burn-in / thinning, and the post-burn-in thinned samples mapped back to model
        coordinates.

        Parameters
        ----------
        n_samples : int
            Steps per walker (the minimum if n_tau_target is set). A warning is printed when
            n_samples < 50 * max(tau).
        n_walkers : int
            Number of walkers.
        burn_in : int, optional
            Steps discarded. Default: int(burn_in_frac * max(tau)), or 500 if tau could not
            be estimated.
        thin : int, optional
            Thinning interval [steps]. Default: max(1, int(thin_frac * min(tau))), or 1 if
            tau could not be estimated.
        burn_in_frac, thin_frac : float
            Multiples of max(tau) / min(tau) used for the defaults above.
        spherical_spread : float
            Walker start spread: z0 + spread * N(0, I) in whitened coordinates, else
            N(theta0, spread^2 * diag(prior std^2)). Drawn from the global np.random state.
        method_optimize : str
            Optimizer of the MAP start (see optimize_initial_guess).
        use_demoves : bool
            Mixture of DEMove (0.7) and DESnookerMove (0.3) instead of the stretch move.
        stretch_a : float
            Scale parameter of emcee's StretchMove.
        use_optimize : bool
            If False, the walkers start around initial_params.
        thin_by : int
            Store every thin_by-th step only (emcee's thin_by; memory of long runs). The
            sampler is then a ChainReplay over the stored steps (stride thin_by, the emcee
            sampler in its attribute `emcee`), so n_samples, tau, burn_in and thin stay in
            steps of the run; thin is rounded to a multiple of thin_by. Default 1: emcee
            sampler, every step stored.
        n_tau_target : float, optional
            Adaptive length: after n_samples steps (the minimum), sample on in blocks of
            check_every steps until n_steps >= n_tau_target * max(tau) (tau with tol=0 on the
            whole run) or max_samples. Default None: exactly n_samples steps.
        max_samples : int, optional
            Cap of the adaptive run (default 10 * n_samples).
        check_every : int, optional
            Block of the adaptive run (default n_samples // 5), rounded to thin_by.
        stranded_dlogp : float, optional
            Leave out of every tau (adaptive check, burn-in / thinning, self.tau) the walkers
            whose median log-posterior over the second half of the run is more than this
            below the ensemble median (walkers stranded far from the posterior have a flat
            or NaN autocorrelation). Default None: all walkers.

        With a valley map (setup_valley_map) the walkers move in its coordinates v (start:
        the same draws z0 + spread * N(0, I), mapped to v) with log p(z(v)) + log |dz/dv|; the
        chain is mapped back to z (log-posterior of z) as soon as it is stored, so tau,
        burn-in, thinning, the sampler (a ChainReplay) and everything downstream refer to the
        whitened coordinates as without the map.

        Returns
        -------
        None
            Sets sampler, tau, burn_in, thin, chain and log_prob_chain (sampled
            coordinates), samples (model coordinates) and log_probs; n_steps (steps run)
            and tau_walkers (walkers of tau).
        """
        thin_by = max(1, int(thin_by))
        t_start = time.perf_counter()

        # Use optimization for better initial guess
        if use_optimize:
            optimized_guess = self.optimize_initial_guess(method=method_optimize)
        else:
            print("[Run] Skipping optimization. Using initial parameters.")
            optimized_guess = self.initial_params.copy()

        valley = getattr(self, "valley", None)
        if valley is not None and not getattr(self, "is_whitened", False):
            raise RuntimeError("the valley map needs whitening (setup_whitening)")
        if getattr(self, "is_whitened", False):
            # Whiten the optimized guess
            x0_white = self.whiten_Linv @ (optimized_guess - self.whiten_mean)

            # Sample walkers in whitened space (identity covariance, scaled)
            pos = x0_white + spherical_spread * np.random.randn(n_walkers, self.ndim)
            if valley is not None:  # the same start, in the valley coordinates
                pos = valley.from_white(pos)
        else:
            # Build realistic covariance from priors
            stds = np.array([p.std() for p in self.param_priors])
            init_cov = np.diag(stds**2)

            # Sample directly in parameter space using inflated prior covariance
            pos = np.random.multivariate_normal(
                mean=optimized_guess, cov=spherical_spread**2 * init_cov, size=n_walkers
            )

        # Determine if we need to use whitened log_prob
        log_prob_name = (
            "log_prob_whitened" if getattr(self, "is_whitened", False) else "log_prob"
        )
        if valley is not None:
            log_prob_name = "log_prob_valley"

        # Define MCMC moves
        if use_demoves:
            moves = [
                (emcee.moves.DEMove(), 0.7),
                (emcee.moves.DESnookerMove(), 0.3),
            ]
        else:
            moves = emcee.moves.StretchMove(a=stretch_a)

        # Run MCMC using emcee
        print("")
        print(
            f"[Run] Starting MCMC sampling with {n_walkers} walkers for {n_samples} steps..."
        )
        global _FORK_MODEL
        _FORK_MODEL = self  # inherited by the workers forked below (see _ForkedLogProb)
        self.map_start_s = time.perf_counter() - t_start  # MAP start (0 without use_optimize)
        t_sample = time.perf_counter()
        try:
            # Pool size: all CPUs, or the environment variable MCMC_PROCS (to share the
            # machine between runs without oversubscribing it; the chain does not depend on it)
            n_procs_env = int(os.environ.get("MCMC_PROCS", "0")) or None
            with multiprocessing.get_context("fork").Pool(n_procs_env) as pool:
                self.n_procs = pool._processes
                print(
                    "[Run] Using multiprocessing with fork context. Number of processes:",
                    pool._processes,
                )
                self.sampler = emcee.EnsembleSampler(
                    nwalkers=n_walkers,
                    ndim=self.ndim,
                    log_prob_fn=_ForkedLogProb(log_prob_name),
                    pool=pool,
                    moves=moves,
                )
                if thin_by == 1 and n_tau_target is None:
                    self.sampler.run_mcmc(pos, n_samples, progress=True)
                else:
                    n_samples = self._run_blocks(
                        pos, n_samples, thin_by, n_tau_target, max_samples, check_every,
                        stranded_dlogp,
                    )
        finally:
            _FORK_MODEL = None
        self.n_steps = n_samples
        # Wall time of the sampling (pool start to the last step); one log-posterior evaluation
        # per walker and step, plus the start
        self.run_wall_s = time.perf_counter() - t_sample
        n_evals = n_walkers * (n_samples + 1)
        print(
            f"[Timing] MCMC sampling {self.run_wall_s:.0f} s ({self.run_wall_s / 3600:.2f} h): "
            f"{n_samples} steps x {n_walkers} walkers = {n_evals:.3e} log-posterior evaluations on "
            f"{self.n_procs} processes; {1e3 * self.run_wall_s / n_samples:.2f} ms per step, "
            f"{1e3 * self.run_wall_s * self.n_procs / n_evals:.3f} ms per evaluation and process "
            f"(MAP start {self.map_start_s:.1f} s)"
        )
        replay = thin_by > 1 or valley is not None
        if replay:  # stored steps -> ChainReplay in steps of the run
            smp = self.sampler
            full, full_lp = smp.get_chain(), smp.get_log_prob()
            if valley is not None:  # back to the whitened coordinates z and log p(z)
                full, logdet = valley.to_white_chain(full)
                full_lp = full_lp - logdet
            self.sampler = ChainReplay(
                full, full_lp, stride=thin_by,
                n_steps=thin_by * smp.iteration,
                acceptance_fraction=smp.acceptance_fraction,
            )
            self.sampler.emcee = smp
        self.tau_walkers = self._kept_walkers(self.sampler.get_log_prob(), stranded_dlogp)
        if self.tau_walkers is not None and self.tau_walkers.size < n_walkers:
            print(
                f"[Run] stranded walkers left out of tau: "
                f"{np.setdiff1d(np.arange(n_walkers), self.tau_walkers).tolist()}"
            )

        # Try to estimate autocorrelation time
        print("[Run] Estimating autocorrelation time...")
        if replay or self.tau_walkers is not None:
            tau = self._tau_steps(tol=0)
            if not np.all(np.isfinite(tau)):
                tau = None
                print(
                    "[Run] Warning: Autocorrelation time could not be reliably estimated."
                )
            else:
                print("[Run] Autocorrelation estimated with tol=0"
                      + ("" if self.tau_walkers is None
                         else f" on {self.tau_walkers.size}/{n_walkers} walkers") + ".")
            if replay:
                self.sampler.tau = tau
        else:
            try:
                tau = self.sampler.get_autocorr_time()
            except emcee.autocorr.AutocorrError:
                try:
                    tau = self.sampler.get_autocorr_time(tol=0)
                    print("[Run] Autocorrelation recovered with tol=0.")
                except emcee.autocorr.AutocorrError:
                    tau = None
                    print(
                        "[Run] Warning: Autocorrelation time could not be reliably estimated."
                    )
        if tau is not None and not np.all(np.isfinite(tau)):  # e.g. walkers that never moved
            tau = None
            print("[Run] Warning: Autocorrelation time is not finite; using the fallbacks.")

        if tau is not None:
            max_tau = np.max(tau)
            min_tau = np.min(tau)
            n_tau = n_samples / max_tau

            # Warn if total steps are insufficient for autocorrelation convergence (emcee:
            # tau is reliable for runs longer than 50 tau)
            if n_samples < 50 * max_tau:
                print(
                    f"[Run] Warning: n_samples = {n_samples} corresponds to "
                    f"{n_tau:.1f} autocorrelation times. "
                    f"Recommended: at least 50 × max(tau) = {50 * max_tau:.1f} steps "
                    f"for reliable sampling."
                )
            else:
                print(
                    f"[Run] n_samples = {n_samples} corresponds to {n_tau:.1f} autocorrelation times. "
                    f"Sampling should be reliable."
                )

            # Determine burn-in and thinning if not manually specified
            if burn_in is None:
                burn_in = int(burn_in_frac * max_tau)
                print(
                    f"[Run] Auto-selected burn-in: {burn_in} steps ({burn_in_frac} x max(tau))"
                )

            if thin is None:
                thin = int(thin_frac * min_tau)
                thin = max(thin, 1)
                if thin_by > 1:  # a multiple of the stored stride
                    thin = thin_by * max(1, int(round(thin / thin_by)))
                print(
                    f"[Run] Auto-selected thinning: every {thin} steps ({thin_frac} x min(tau))"
                )
        else:
            # Fallbacks if tau unavailable
            if burn_in is None:
                burn_in = 500
                print("[Run] Fallback burn-in: 500")
            if thin is None:
                thin = 1
                print("[Run] Fallback thinning: 1")

        # Discard burn-in samples and flatten the chain
        self.tau, self.burn_in, self.thin = tau, burn_in, thin
        self.chain = self.sampler.get_chain(
            discard=burn_in, thin=thin, flat=False
        )  # (nsteps, nwalkers, ndim)
        self.log_prob_chain = self.sampler.get_log_prob(
            discard=burn_in, thin=thin, flat=False
        )  # (nsteps, nwalkers)
        self.samples = self.chain.reshape(-1, self.ndim)  # flat view for corner
        self.log_probs = self.log_prob_chain.reshape(-1)

        # If whitening was used, transform samples back to original space
        if getattr(self, "is_whitened", False):
            self.samples = (
                self.whiten_L @ self.samples.T + self.whiten_mean[:, None]
            ).T  # shape (N, ndim)

    @staticmethod
    def _kept_walkers(log_prob, stranded_dlogp):
        """
        Walkers kept for tau: None (all) if stranded_dlogp is None, else the walkers whose
        median log-posterior over the second half of the run is at most stranded_dlogp below
        the ensemble median.

        Parameters
        ----------
        log_prob : array_like, shape (n_steps, n_walkers)
            Log-posterior of the whole run (or of its stored steps).
        stranded_dlogp : float or None

        Returns
        -------
        ndarray of int or None
        """
        if stranded_dlogp is None:
            return None
        lp = np.asarray(log_prob, dtype=float)
        lp_w = np.median(lp[lp.shape[0] // 2 :], axis=0)
        return np.where(lp_w >= np.median(lp_w) - float(stranded_dlogp))[0]

    def _tau_steps(self, tol=0):
        """
        Integrated autocorrelation time [steps of the run] of the walkers in tau_walkers
        (all if None), from every stored step (emcee.autocorr.integrated_time, times the
        stored stride).

        Parameters
        ----------
        tol : float
            Passed to integrated_time (0: no length check).

        Returns
        -------
        ndarray, shape (ndim,)
        """
        smp = self.sampler
        stride = int(getattr(smp, "stride", 1))
        ch = smp.full_chain if isinstance(smp, ChainReplay) else smp.get_chain()
        w = getattr(self, "tau_walkers", None)
        if w is not None:
            ch = ch[:, w]
        with np.errstate(invalid="ignore", divide="ignore"):
            return stride * emcee.autocorr.integrated_time(ch, tol=tol, quiet=True)

    def _run_blocks(self, pos, n_min, thin_by, n_tau_target, n_max, check_every,
                    stranded_dlogp):
        """
        Sample n_min steps storing every thin_by-th, then (if n_tau_target) blocks of
        check_every steps until n_steps >= n_tau_target * max(tau) or n_max (see run).

        The emcee random state carries over from block to block (run_mcmc continues the
        previous state), so the blocks are one run. With a valley map, tau is that of the
        chain mapped back to the whitened coordinates.

        Returns
        -------
        int
            Steps run (a multiple of thin_by).
        """
        smp = self.sampler
        n_min = thin_by * int(np.ceil(n_min / thin_by))
        n_max = 10 * n_min if n_max is None else thin_by * int(np.ceil(n_max / thin_by))
        block = n_min // 5 if check_every is None else int(check_every)
        block = thin_by * max(1, int(round(block / thin_by)))
        n_done, state, n_run = 0, pos, n_min
        while True:
            print(f"[Run] steps {n_done} -> {n_done + n_run} (storing every {thin_by})")
            smp.run_mcmc(state, n_run // thin_by, thin_by=thin_by, progress=True)
            n_done, state = n_done + n_run, None
            if n_tau_target is None:
                return n_done
            valley = getattr(self, "valley", None)
            if valley is None:
                keep = self._kept_walkers(smp.get_log_prob(), stranded_dlogp)
                ch = smp.get_chain() if keep is None else smp.get_chain()[:, keep]
            else:  # tau of the whitened coordinates z, stranded walkers from log p(z)
                ch, logdet = valley.to_white_chain(smp.get_chain())
                keep = self._kept_walkers(smp.get_log_prob() - logdet, stranded_dlogp)
                ch = ch if keep is None else ch[:, keep]
            with np.errstate(invalid="ignore", divide="ignore"):
                tau = thin_by * emcee.autocorr.integrated_time(ch, tol=0, quiet=True)
            tau_max = np.max(tau) if np.all(np.isfinite(tau)) else np.inf
            print(
                f"[Run] {n_done} steps: tau_max {tau_max:.0f} steps, n_steps / tau_max = "
                f"{n_done / tau_max:.1f} (target {n_tau_target:g}), acceptance "
                f"{np.mean(smp.acceptance_fraction):.3f}"
            )
            if n_done >= n_tau_target * tau_max:
                return n_done
            if n_done >= n_max:
                print(f"[Run] Warning: stopped at max_samples = {n_max} before "
                      f"{n_tau_target:g} x max(tau).")
                return n_done
            n_run = min(block, n_max - n_done)

    def summary(self):
        """
        Print tau, suggested burn-in / thinning, acceptance rate and parameter estimates.

        The suggestions are 2 * max(tau) and max(1, int(0.5 * min(tau))) (tau retried
        with tol=0). The mean acceptance fraction is flagged outside 0.2-0.5. Each
        parameter is printed as its median +(q84 - median) / -(median - q16) of the samples
        (model coordinates).

        Returns
        -------
        None
            Prints "Run MCMC first." and returns when there are no samples.
        """
        if self.samples is None:
            print("Run MCMC first.")
            return

        print("")
        print("\n=== MCMC Summary ===")
        try:
            # First try default autocorr calculation (faster); the run's tau if it left
            # stranded walkers out
            if getattr(self, "tau_walkers", None) is not None and self.tau is not None:
                tau = np.asarray(self.tau)
            else:
                tau = self.sampler.get_autocorr_time()
        except emcee.autocorr.AutocorrError:
            try:
                # Retry with tol=0 if the default fails
                tau = self.sampler.get_autocorr_time(tol=0)
                print("Autocorrelation recovered with tol=0.")
            except emcee.autocorr.AutocorrError:
                tau = None
                print("Warning: Autocorrelation time could not be reliably estimated.")

        if tau is not None:
            print(f"Autocorr time per parameter: {tau}")
            burnin = int(2 * np.max(tau))  # Conservative default
            thin = max(1, int(0.5 * np.min(tau)))  # Decorrelation (at least 1)
            print(f"Suggested burn-in: {burnin} steps ({2} x max(tau))")
            print(f"Suggested thinning: every {thin} steps ({0.5} x min(tau))")

        # Calculate the acceptance fraction
        acceptance_fraction = self.sampler.acceptance_fraction
        mean_acceptance = np.mean(acceptance_fraction)
        print("Mean acceptance rate:", mean_acceptance)

        # Warn if acceptance rate is outside optimal range
        if mean_acceptance < 0.2 or mean_acceptance > 0.5:
            print(
                "Acceptance rate outside optimal range (0.2-0.5). Consider tuning initialization step size and stretch_a."
            )

        print("Parameter estimates:")
        for i in range(self.ndim):
            mcmc = np.percentile(self.samples[:, i], [16, 50, 84])
            q_lower, q_upper = np.diff(mcmc)
            median = mcmc[1]
            print(f"θ_{i}: {median:+.10e}  (+{q_upper:.1e} / -{q_lower:.1e})")

    def print_regression_diagnostics(self):
        """
        Print goodness-of-fit statistics of the MAP sample.

        N residuals, p = ndim parameters, dof = N - p, chi2 = |r|^2,
        reduced chi2 = chi2 / dof, log L = -chi2 / 2, AIC = 2 p - 2 log L,
        BIC = p ln N - 2 log L, the residual RMS and the MAP parameters.

        Returns
        -------
        None
            Prints "Run MCMC first." and returns when there are no samples.
        """
        if self.samples is None:
            print("Run MCMC first.")
            return

        theta_best = self.get_map_estimate()
        residuals = self.residuals_func(theta_best)

        chi2 = np.sum(residuals**2)
        n_data = len(residuals)
        n_params = self.ndim
        dof = n_data - n_params
        chi2_red = chi2 / dof if dof > 0 else float("nan")
        logL = -0.5 * chi2
        AIC = 2 * n_params - 2 * logL
        BIC = n_params * np.log(n_data) - 2 * logL
        RMS = np.sqrt(np.mean(residuals**2))

        print("")
        print("\n=== Regression Diagnostics ===")
        print(f"Number of data points: {n_data}")
        print(f"Number of parameters: {n_params}")
        print(f"Degrees of freedom: {dof}")
        print(f"Chi-squared: {chi2:.3f}")
        print(f"Reduced Chi-squared: {chi2_red:.3f}")
        print(f"Log-likelihood: {logL:.3f}")
        print(f"AIC: {AIC:.3f}")
        print(f"BIC: {BIC:.3f}")
        print(f"RMS of residuals: {RMS:.3f}")
        print(f"MAP parameters: {theta_best}")

    def setup_whitening_from_priors(self):
        """
        Turn on whitening with the diagonal prior covariance diag(std_i^2).

        Raises
        ------
        ValueError
            If a prior has no std(), or its std is not finite or not positive.
        """
        stds = []
        for i, prior in enumerate(self.param_priors):
            try:
                std = prior.std()
            except Exception:
                raise ValueError(
                    f"Prior {i} does not support .std(). Cannot use for whitening."
                )
            if not np.isfinite(std) or std <= 0:
                raise ValueError(
                    f"Prior {i} returned non-finite or non-positive std: {std}"
                )
            stds.append(std)

        cov = np.diag(np.square(stds))  # Build diagonal covariance
        self.setup_whitening(cov=cov)

    def setup_whitening(self, cov=None):
        """
        Turn on whitening: theta = L z + mu, with mu the prior means and L L^T = cov.

        The walkers then sample z, in which the posterior is roughly isotropic; the map is
        linear, so the log-posterior needs no Jacobian term.

        Parameters
        ----------
        cov : array_like, shape (ndim, ndim)
            Symmetric positive-definite whitening covariance (e.g. the batch covariance).
            Must be given: None fails in the Cholesky factorization.

        Returns
        -------
        None
            Sets whiten_mean, whiten_L, whiten_Linv and is_whitened = True.
        """
        self.whiten_mean = np.array([p.mean() for p in self.param_priors], dtype=float)
        self.whiten_L = np.linalg.cholesky(cov)
        self.whiten_Linv = np.linalg.inv(self.whiten_L)
        self.is_whitened = True

    def log_posterior_whitened(self, theta_white):
        """
        Log-posterior at theta = whiten_L @ theta_white + whiten_mean.

        Parameters
        ----------
        theta_white : array_like, shape (ndim,)

        Returns
        -------
        float

        Raises
        ------
        RuntimeError
            If is_whitened is False (AttributeError if whitening was never set up).
        """
        if not self.is_whitened:
            raise RuntimeError("Whitening not set up")

        theta = self.whiten_L @ theta_white + self.whiten_mean
        return self.log_posterior(theta)

    def log_prob_whitened(self, theta_white):
        """
        Log-probability passed to emcee when whitening is on (= log_posterior_whitened).

        Parameters
        ----------
        theta_white : array_like, shape (ndim,)

        Returns
        -------
        float
        """
        return self.log_posterior_whitened(theta_white)

    def whitened_residuals(self, z):
        """
        Data and prior residuals at theta = whiten_L @ z + whiten_mean (residuals_func, then
        prior_residuals): -log p = 0.5 |r|^2 + const for Gaussian priors. The default residuals
        of setup_valley_map.
        """
        theta = self.whiten_L @ np.asarray(z, dtype=float) + self.whiten_mean
        return np.concatenate(
            [np.atleast_1d(np.asarray(self.residuals_func(theta), dtype=float)),
             self.prior_residuals(theta)]
        )

    def setup_valley_map(self, direction, start=None, residuals=None, **kwargs):
        """
        Turn on valley-straightening sampling coordinates (ValleyMap) on top of the whitening.

        For a posterior whose high-density region is a long curved valley (e.g. a poorly
        observed launch height, along which the other parameters follow a bent path), the
        affine-invariant moves only see the chord. The map z = h e + Q (y*(h) + C(h) w) follows
        the profile path y*(h) and its conditional scale C(h) along h = e . z; the log-
        posterior gets the term log |dz/dv| = sum log diag C(h), so the target posterior is
        unchanged (only the coordinates of the walkers are). See build_valley_map.

        Parameters
        ----------
        direction : array_like, shape (ndim,)
            Valley direction g in model coordinates (e.g. the batch's least-determined position
            direction); e = whiten_L^T g / |whiten_L^T g|, so h = g . (theta - whiten_mean) /
            sqrt(g^T cov g), the deviation along g in sigmas of the whitening covariance.
        start : array_like, shape (ndim,), optional
            Start of the profile in model coordinates (default initial_params).
        residuals : callable, optional
            z (ndim,) -> residual vector with -log p(z) = 0.5 |r|^2 + const. Default
            whitened_residuals (exact for Gaussian priors; otherwise only the efficiency of
            the map suffers).
        **kwargs
            dh, dlogp_stop, h_max, residuals_batch, fd_step, verbose of build_valley_map.

        Returns
        -------
        ValleyMap
            Also stored as self.valley; the build wall time in self.valley_build_s [s].
        """
        if not getattr(self, "is_whitened", False):
            raise RuntimeError("setup_valley_map needs whitening (setup_whitening first)")
        t0 = time.perf_counter()
        g = np.asarray(direction, dtype=float)
        e = self.whiten_L.T @ g
        z0 = self.whiten_Linv @ (
            (self.initial_params if start is None else np.asarray(start, dtype=float))
            - self.whiten_mean
        )
        self.valley = build_valley_map(
            self.whitened_residuals if residuals is None else residuals, e, z0, **kwargs
        )
        self.valley_build_s = time.perf_counter() - t0
        print(f"[Timing] valley map {self.valley_build_s:.1f} s ({self.valley.H.size} profile "
              f"points)")
        return self.valley

    def log_prob_valley(self, v):
        """
        Log-probability passed to emcee with a valley map: log p(z(v)) + log |dz/dv|.

        Parameters
        ----------
        v : array_like, shape (ndim,)
            Valley coordinates (h, w).

        Returns
        -------
        float
        """
        z, logdet = self.valley.to_white(v)
        lp = self.log_posterior_whitened(z[0]) + logdet[0]
        return lp if np.isfinite(lp) else -np.inf

    def save_chain(self, path=None, max_full_steps=4000, **extra):
        """
        Save the run as a compressed .npz, enough to redo every diagnostic and figure.

        load_chain restores it without sampling again. Call it right after run(), before any
        in-place edit of self.samples, so that a loaded file reproduces the state run()
        leaves. The bytes are written with retries (OneDrive), else to
        MCMCPlots.FALLBACK_DIR.

        Parameters
        ----------
        path : str or Path, optional
            Output file. Default: chain.npz in the figure folder (MCMCPlots.set_figure_dir).
        max_full_steps : int or None
            Maximum number of stored steps of the full run: full_stride = max(1, n_steps //
            max_full_steps), capped at min(tau) // 10. None (or 0): every step.
        **extra : array_like
            Further arrays (truth, batch mean / covariance, ...), stored as extra_<key>.

        Returns
        -------
        Path or None
            The file written (in FALLBACK_DIR if the target folder could not be written);
            None if neither could be written (MCMCPlots.write_output prints an error).

        Raises
        ------
        ValueError
            If there are no samples yet.

        Notes
        -----
        Stored keys:

        - samples, log_probs: the flat posterior as run() leaves it (model
          coordinates).
        - chain, log_prob_chain: post-burn-in thinned chain in sampled (whitened, if
          whitening is on) coordinates, (n_steps, n_walkers, ndim) / (n_steps, n_walkers).
        - full_chain (float32), full_log_prob: the whole run, burn-in included, every
          full_stride steps (for the trace and autocorrelation figures).
        - n_steps, full_stride, acceptance_fraction, tau, burn_in, thin, whiten_mean / _L.
        - run_wall_s, map_start_s, n_procs: wall time of the sampling and of the MAP start [s]
          and the pool size, when the run set them.
        - valley_e / _Q / _H / _Y / _C / _cost and valley_build_s: the valley map of the walkers
          (setup_valley_map), when on; the chains are stored in the whitened coordinates
          anyway.
        - extra_<key>: the **extra arrays.

        A re-saved ChainReplay keeps its stored grid. A sampler without get_chain (e.g.
        nested sampling) stores no full_chain.
        """
        if self.samples is None:
            raise ValueError("No samples to save: run the sampler first.")
        path = Path(MCMCPlots.figure_dir() / "chain.npz" if path is None else path)
        data = {"samples": np.asarray(self.samples), "log_probs": np.asarray(self.log_probs)}
        for key in ("chain", "log_prob_chain", "tau", "burn_in", "thin", "run_wall_s", "map_start_s",
                    "n_procs"):
            val = getattr(self, key, None)
            if val is not None:
                data[key] = np.asarray(val)
        if getattr(self, "is_whitened", False):
            data["whiten_mean"] = np.asarray(self.whiten_mean)
            data["whiten_L"] = np.asarray(self.whiten_L)
        if getattr(self, "valley", None) is not None:
            for key, val in self.valley.arrays().items():
                data[f"valley_{key}"] = np.asarray(val)
            if getattr(self, "valley_build_s", None) is not None:
                data["valley_build_s"] = np.asarray(self.valley_build_s)
        smp = self.sampler
        if isinstance(smp, ChainReplay) and not hasattr(smp, "emcee"):
            # re-saving a loaded run: keep its stored grid
            stride, full, full_lp = smp.stride, smp.full_chain, smp.full_log_prob
        elif smp is not None and hasattr(smp, "get_chain"):  # emcee (or a thin_by run)
            n_steps = int(smp.iteration)
            stride = 1 if not max_full_steps else max(1, n_steps // int(max_full_steps))
            if self.tau is not None and np.all(np.isfinite(self.tau)):
                stride = max(1, min(stride, int(np.min(self.tau) // 10)))
            rs = int(getattr(smp, "stride", 1))  # thin_by run: a multiple of its stride
            stride = rs * max(1, stride // rs)
            full = smp.get_chain(thin=stride).astype(np.float32)
            full_lp = smp.get_log_prob(thin=stride)
        else:  # e.g. nested sampling: no chain
            smp = None
        if smp is not None:
            data.update(
                full_chain=full,
                full_log_prob=np.asarray(full_lp),
                full_stride=np.asarray(stride),
                n_steps=np.asarray(smp.iteration),
                acceptance_fraction=np.asarray(smp.acceptance_fraction),
            )
        for key, val in extra.items():
            data[f"extra_{key}"] = np.asarray(val)

        buf = io.BytesIO()
        np.savez_compressed(buf, **data)
        path = MCMCPlots.write_output(path, buf.getvalue(), tag="Chain")
        if path is not None:
            print(f"[Chain] saved {path} ({buf.tell() / 2**20:.1f} MB)")
        return path

    def load_chain(self, path):
        """
        Restore a run saved by save_chain (diagnostics and figures work as after run()).

        Sets samples, log_probs, chain, log_prob_chain, tau, burn_in, thin, the whitening
        (whiten_Linv recomputed) and the valley map if the run had one, and a ChainReplay in place of the emcee sampler (same
        interface; None if no full chain was stored).

        Parameters
        ----------
        path : str or Path
            A .npz written by save_chain.

        Returns
        -------
        dict
            The extra arrays, keyed without the extra_ prefix.
        """
        with np.load(path) as f:
            data = {k: f[k] for k in f.files}
        self.samples = data["samples"]
        self.log_probs = data["log_probs"]
        self.chain = data.get("chain")
        self.log_prob_chain = data.get("log_prob_chain")
        self.tau = data.get("tau")
        self.burn_in = int(data["burn_in"]) if "burn_in" in data else None
        self.thin = int(data["thin"]) if "thin" in data else None
        if "whiten_L" in data:
            self.whiten_mean = data["whiten_mean"]
            self.whiten_L = data["whiten_L"]
            self.whiten_Linv = np.linalg.inv(self.whiten_L)
            self.is_whitened = True
        if "valley_H" in data:  # the run's valley map (its chains are in z already)
            self.valley = ValleyMap(
                data["valley_e"], data["valley_Q"], data["valley_H"], data["valley_Y"],
                data["valley_C"], data.get("valley_cost"),
            )
        self.sampler = None
        if "full_chain" in data:
            self.sampler = ChainReplay(
                data["full_chain"],
                data["full_log_prob"],
                stride=int(data["full_stride"]),
                n_steps=int(data["n_steps"]),
                tau=self.tau,
                acceptance_fraction=data["acceptance_fraction"],
            )
        print(f"[Chain] loaded {path}")
        return {k[len("extra_") :]: v for k, v in data.items() if k.startswith("extra_")}

    def get_map_estimate(self):
        """
        Maximum a posteriori sample: the stored sample with the largest log-posterior.

        Returns
        -------
        ndarray, shape (ndim,)
            In model coordinates (among the post-burn-in thinned samples, not an optimum).

        Raises
        ------
        RuntimeError
            If there are no samples yet.
        """
        if self.samples is None or self.log_probs is None:
            raise RuntimeError("Run MCMC before computing MAP estimate.")

        idx_max = np.argmax(self.log_probs)
        map_params = self.samples[idx_max]
        return map_params

    def find_map(self, starts=None, log_target=None, polish=True, max_nfev=4000, verbose=True):
        """
        Posterior mode: an optimum, not a sample (unlike get_map_estimate).

        1. Least squares (trf, 3-point Jacobian, Jacobian scaling) on residuals_func stacked
           with prior_residuals, in whitened coordinates when whitening is on, from every
           start: the exact mode for Gaussian priors and this class's log_likelihood.
        2. Polish (polish=True): L-BFGS-B (3-point gradient) on log_target from the best
           point so far, then Powell from the better of the two, in coordinates whitened by
           the sample covariance (after a run; else as in 1). This finds the mode for other
           priors too (boxes, a subclass's or an assigned log_prior) and for a log_target
           other than log_posterior. Outside the support the objective is floored (start
           value - 1e6), so a box edge stops it; Powell, derivative-free, also reaches a mode
           on a box face, where the finite-difference gradient of L-BFGS-B straddles the
           edge.

        Starts and optima compete on log_target: the result is never worse than the best
        start.

        Where most of the posterior mass lies far from the mode (e.g. a volume factor along
        a weakly observed direction), the samples seldom come near the mode, so the best
        sample is a poor stand-in for it.

        Parameters
        ----------
        starts : sequence of array_like, optional
            Start points in model coordinates. Default: initial_params and, after a run, the
            best sample (get_map_estimate).
        log_target : callable, optional
            Log-density (of model coordinates) maximized by the polish and used to rank;
            default log_posterior. A plot in coordinates x(theta) shows the density
            p(theta) / |dx/dtheta|, whose mode differs from the mode in theta when x(theta)
            is nonlinear: pass log_posterior(theta) - log|dx/dtheta| and map the result
            with x(theta) to mark the mode of the plotted density.
        polish : bool
            Run stage 2.
        max_nfev : int
            Function evaluations per least-squares run.
        verbose : bool
            Print the log_target of every start and optimum.

        Returns
        -------
        ndarray, shape (ndim,)
            In model coordinates.
        """
        target = self.log_posterior if log_target is None else log_target
        if starts is None:
            starts = [self.initial_params]
            if self.samples is not None and self.log_probs is not None:
                starts.append(self.get_map_estimate())
        use_whitened = getattr(self, "is_whitened", False)

        def to_theta(z):
            return self.whiten_L @ z + self.whiten_mean if use_whitened else z

        def residuals_map(z):
            theta = to_theta(z)
            return np.concatenate([self.residuals_func(theta), self.prior_residuals(theta)])

        cands = []
        for k, x0 in enumerate(starts):
            x0 = np.asarray(x0, dtype=float)
            z0 = self.whiten_Linv @ (x0 - self.whiten_mean) if use_whitened else x0
            sol = least_squares(
                residuals_map,
                z0,
                method="trf",
                jac="3-point",
                x_scale="jac",
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=max_nfev,
            )
            cands.append((target(x0), f"start {k}", x0))
            cands.append((target(to_theta(sol.x)), f"optimum from start {k} ({sol.status})", to_theta(sol.x)))

        if polish:
            lp0, _, x0 = max(cands, key=lambda c: c[0])
            if np.isfinite(lp0):
                L, m = None, np.asarray(x0, dtype=float)
                if self.samples is not None and len(self.samples) > self.ndim:
                    C = np.cov(np.asarray(self.samples, dtype=float).T)
                    try:
                        L = np.linalg.cholesky(C + 1e-12 * np.trace(C) / self.ndim * np.eye(self.ndim))
                    except np.linalg.LinAlgError:
                        L = None
                if L is None and use_whitened:
                    L = self.whiten_L
                if L is None:
                    L = np.eye(self.ndim)
                floor = lp0 - 1e6

                def neg_target(u):
                    v = target(m + L @ u)
                    return -(v if np.isfinite(v) else floor)

                sol = minimize(neg_target, np.zeros(self.ndim), method="L-BFGS-B", jac="3-point",
                               options=dict(maxiter=2000, maxfun=50_000))
                x1 = m + L @ sol.x
                lp1 = target(x1)
                cands.append((lp1, f"polish (L-BFGS-B, {'ok' if sol.success else 'stopped'})", x1))
                # Then Powell (derivative-free) from the better point: a finite-difference
                # gradient straddling a support edge (a mode on a box face) stalls L-BFGS-B
                u0 = sol.x if lp1 > lp0 else np.zeros(self.ndim)
                sol = minimize(neg_target, u0, method="Powell",
                               options=dict(xtol=1e-8, ftol=1e-12, maxfev=20_000))
                x2 = m + L @ sol.x
                cands.append((target(x2), f"polish (Powell, {'ok' if sol.success else 'stopped'})", x2))

        best = max(range(len(cands)), key=lambda i: cands[i][0])
        theta_map = np.array(cands[best][2], dtype=float)
        if verbose:
            what = "log posterior" if log_target is None else "log target"
            print(f"\n[MAP] {what} (least squares on measurement + prior residuals, then polish):")
            for i, (lp, label, _) in enumerate(cands):
                print(f"  {label:32s} {lp:14.6f}{'  <- MAP' if i == best else ''}")
            print("[MAP] theta_map:\n", theta_map)
        return theta_map

    def gelman_rubin_diagnostic(self, split=True, threshold=1.1):
        """
        Gelman-Rubin potential scale reduction R_hat per parameter, printed with a verdict.

        The chains are the walkers of the kept samples (after burn-in and thinning), in
        model coordinates. With m chains of n samples each,

            B = n var(chain means),  W = mean(chain variances),
            V_hat = (1 - 1/n) W + B / n,  R_hat = sqrt(V_hat / W).

        Parameters
        ----------
        split : bool
            Cut every walker into its first and second half (split R_hat, 2 n_walkers
            chains); with an odd number of kept steps the middle step is dropped.
        threshold : float
            R_hat below which a parameter is reported converged.

        Returns
        -------
        ndarray, shape (ndim,)
            R_hat of each parameter.

        Raises
        ------
        RuntimeError
            If there are no samples yet, or no walker chain (e.g. nested sampling).
        ValueError
            If there are fewer than 2 chains or fewer than 2 samples per chain.

        Notes
        -----
        The walkers of an ensemble sampler are not independent chains (every move uses
        the other walkers), so R_hat is only indicative (a warning is printed).
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC before computing Gelman-Rubin diagnostic.")
        if self.chain is None:
            raise RuntimeError("Gelman-Rubin needs the walker chain of run().")

        print(
            "\n DANGER: This implementation assumes all walkers are independent chains. In EMCEE, this is not true. Use with caution. \n"
        )

        # self.samples is the step-major flattening of self.chain (n_steps, n_walkers),
        # mapped to model coordinates; regroup it per walker: (n_walkers, n_steps, ndim)
        n_steps, n_walkers = self.chain.shape[:2]
        chain = np.asarray(self.samples, dtype=float).reshape(n_steps, n_walkers, -1)
        chain = chain.transpose(1, 0, 2)

        if split:
            half = n_steps // 2
            chain = np.concatenate([chain[:, :half], chain[:, n_steps - half :]], axis=0)

        n_chains, n_samples, _ = chain.shape
        if n_chains < 2 or n_samples < 2:
            raise ValueError(
                f"Gelman-Rubin needs at least 2 chains of at least 2 samples, got "
                f"{n_chains} chains of {n_samples}" + (" after splitting." if split else ".")
            )

        chain_means = np.mean(chain, axis=1)  # (n_chains, ndim)
        chain_vars = np.var(chain, axis=1, ddof=1)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)
        W = np.mean(chain_vars, axis=0)
        V_hat = (1 - 1 / n_samples) * W + B / n_samples
        r_hat = np.sqrt(V_hat / W)

        print("\n=== Gelman-Rubin Diagnostic ===")
        for i, r in enumerate(r_hat):
            status = "OK" if r < threshold else "NOT CONVERGED"
            print(f"R_hat[θ_{i}]: {r:.4f}  is  {status}")

        if np.all(r_hat < threshold):
            print(
                "\n[Convergence] All parameters have converged (R_hat < {:.2f}).".format(
                    threshold
                )
            )
        else:
            print(
                "\n[Convergence Warning] Some parameters have not yet converged (R_hat ≥ {:.2f}).".format(
                    threshold
                )
            )

        return r_hat

    def effective_sample_size(self, min_ess_threshold=100):
        """
        Effective sample size of the kept samples per parameter, printed.

            ESS = min(n_kept * thin * n_walkers / tau, n_kept * n_walkers),

        with n_kept the kept (post-burn-in, thinned) steps, so n_kept * thin is the span
        of the kept chain [steps], and tau the integrated autocorrelation time [steps].
        Every walker contributes about one independent draw per tau steps, and thinning
        cannot make the kept samples worth more than their number.

        Parameters
        ----------
        min_ess_threshold : float
            ESS below which a parameter is reported LOW.

        Returns
        -------
        ndarray, shape (ndim,)

        Raises
        ------
        RuntimeError
            If there are no samples yet, or no walker chain (e.g. nested sampling).

        Notes
        -----
        tau is the run's (sampler.get_autocorr_time, retried with tol=0): estimated on the
        whole run in sampled coordinates, the same as in model coordinates for a diagonal
        whitening. Without it (no sampler, or a chain loaded without tau) tau is estimated
        on the kept chain with tol=0 and converted to steps (printed).
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC first.")
        if self.chain is None:
            raise RuntimeError("ESS needs the walker chain of run().")

        tau = None
        if self.sampler is not None:
            try:
                tau = self.sampler.get_autocorr_time()
            except emcee.autocorr.AutocorrError:
                try:
                    tau = self.sampler.get_autocorr_time(tol=0)
                    print("[ESS] Autocorrelation recovered with tol=0.")
                except emcee.autocorr.AutocorrError:
                    tau = None

        n_kept, n_walkers = self.chain.shape[:2]
        thin = max(1, int(self.thin or 1))
        if tau is None:
            # tau of the kept chain [kept steps] -> [steps]
            tau = thin * emcee.autocorr.integrated_time(self.chain, tol=0)
            print("[ESS] No tau from the run; estimated on the kept chain (tol=0).")

        tau = np.asarray(tau, dtype=float)
        ess = np.minimum(n_kept * thin * n_walkers / tau, n_kept * n_walkers)

        print("\n=== Effective Sample Size (ESS) ===")
        all_good = True
        for i, e in enumerate(ess):
            status = "OK" if e >= min_ess_threshold else "LOW"
            if e < min_ess_threshold:
                all_good = False
            print(f"ESS[θ_{i}]: {int(e):>5} samples is {status}")

        if all_good:
            print(
                f"\n[Convergence] All parameters have sufficient ESS (≥ {min_ess_threshold})."
            )
        else:
            print(
                f"\n[Convergence Warning] Some parameters have low ESS (< {min_ess_threshold}). Consider running longer or adjusting sampler settings."
            )

        return ess

    def get_estimate_and_covariance(self, method="map"):
        """
        Point estimate and sample covariance of the posterior.

        Parameters
        ----------
        method : {'map', 'median', 'mean'}
            Point estimate: the MAP sample (get_map_estimate), or the per-parameter median /
            mean of the samples.

        Returns
        -------
        theta_hat : ndarray, shape (ndim,)
        cov : ndarray, shape (ndim, ndim)
            np.cov of the samples. Both in model coordinates.

        Raises
        ------
        RuntimeError
            If there are no samples yet.
        ValueError
            For an unknown method.
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling this method.")

        if method == "median":
            theta_hat = np.median(self.samples, axis=0)
        elif method == "mean":
            theta_hat = np.mean(self.samples, axis=0)
        elif method == "map":
            theta_hat = self.get_map_estimate()
        else:
            raise ValueError("Method must be 'map', 'median' or 'mean'.")

        cov = np.cov(self.samples.T)
        return theta_hat, cov


    # ------------------------------------------------------------------
    # Figures: every plot lives in MCMCPlots; these methods only pass the model.
    # Journal figures (Okabe-Ito, LaTeX, one PDF each) need MCMCPlots.configure(...) and
    # MCMCPlots.set_figure_dir(...) first. The quick-look plot_* figures are the original
    # MCMCModel plots (tests/ and extra/ scripts use them).
    # ------------------------------------------------------------------
    def plot_journal_residuals(self, *args, **kwargs):
        """
        Journal figure: normalized post-fit residuals of the batch and the MCMC MAP.

        Thin wrapper around MCMCPlots.plot_mcmc_residuals(self, *args, **kwargs); see there
        for the arguments.

        Returns
        -------
        Path
            The saved PDF.
        """
        return MCMCPlots.plot_mcmc_residuals(self, *args, **kwargs)

    def plot_journal_traces(self, *args, **kwargs):
        """
        Journal figure: walker traces of every parameter and of log p.

        Thin wrapper around MCMCPlots.plot_mcmc_traces(self, *args, **kwargs); see there for
        the arguments.

        Returns
        -------
        Path
            The saved PDF.
        """
        return MCMCPlots.plot_mcmc_traces(self, *args, **kwargs)

    def plot_journal_autocorrelation(self, *args, **kwargs):
        """
        Journal figure: walker-averaged autocorrelation of every parameter.

        Thin wrapper around MCMCPlots.plot_mcmc_autocorrelation(self, *args, **kwargs); see
        there for the arguments.

        Returns
        -------
        Path
            The saved PDF.
        """
        return MCMCPlots.plot_mcmc_autocorrelation(self, *args, **kwargs)

    def plot_journal_corner(self, *args, **kwargs):
        """
        Journal figure: corner plot of the posterior against the batch and the truth
        (prior=True: with the prior overlaid, saved as fig_corner_prior).

        Thin wrapper around MCMCPlots.plot_mcmc_corner(self, *args, **kwargs); see there for
        the arguments.

        Returns
        -------
        Path
            The saved PDF.
        """
        return MCMCPlots.plot_mcmc_corner(self, *args, **kwargs)

    def plot_journal_diagnostics(self, *args, **kwargs):
        """
        The MCMC journal figures (residuals, traces, autocorrelation, corner, the corner
        with the prior unless corner_prior=False, and the corner in the RTN frame when
        corner_rtn is given).

        Thin wrapper around MCMCPlots.plot_mcmc_diagnostics(self, *args, **kwargs); see
        there for the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.plot_mcmc_diagnostics(self, *args, **kwargs)

    def plot_convergence(self, *args, **kwargs):
        """
        Quick-look figure: walker traces of the kept samples.

        Thin wrapper around MCMCPlots.legacy_convergence(self, *args, **kwargs); see there
        for the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.legacy_convergence(self, *args, **kwargs)

    def plot_log_likelihood(self, *args, **kwargs):
        """
        Quick-look figure: log-posterior trace of every walker.

        Thin wrapper around MCMCPlots.legacy_log_likelihood(self, *args, **kwargs); see
        there for the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.legacy_log_likelihood(self, *args, **kwargs)

    def plot_postfit_residuals(self, *args, **kwargs):
        """
        Quick-look figure: normalized residuals of the MAP vs observation index.

        Thin wrapper around MCMCPlots.legacy_postfit_residuals(self, *args, **kwargs); see
        there for the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.legacy_postfit_residuals(self, *args, **kwargs)

    def plot_postfit_residuals_time(self, *args, **kwargs):
        """
        Quick-look figure: paired normalized residuals of the MAP vs time.

        Thin wrapper around MCMCPlots.legacy_postfit_residuals_time(self, *args, **kwargs);
        see there for the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.legacy_postfit_residuals_time(self, *args, **kwargs)

    def plot_corner(self, *args, **kwargs):
        """
        Quick-look figure: corner plot of the samples (corner package).

        Thin wrapper around MCMCPlots.legacy_corner(self, *args, **kwargs); see there for
        the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.legacy_corner(self, *args, **kwargs)

    def plot_corner_with_batch(self, *args, **kwargs):
        """
        Quick-look figure: corner plot with batch ellipses, truth and MAP.

        Thin wrapper around MCMCPlots.legacy_corner_with_batch(self, *args, **kwargs); see
        there for the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.legacy_corner_with_batch(self, *args, **kwargs)

    def plot_marginals_overlaid(self, *args, **kwargs):
        """
        Quick-look figure: 1-D marginals of several parameters on one axis.

        Thin wrapper around MCMCPlots.legacy_marginals_overlaid(self, *args, **kwargs); see
        there for the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.legacy_marginals_overlaid(self, *args, **kwargs)

    def plot_autocorrelation(self, *args, **kwargs):
        """
        Quick-look figure: walker-averaged autocorrelation per parameter.

        Thin wrapper around MCMCPlots.legacy_autocorrelation(self, *args, **kwargs); see
        there for the arguments.

        Returns
        -------
        None
        """
        return MCMCPlots.legacy_autocorrelation(self, *args, **kwargs)

    '''
    def run_multinest(
        self,
        n_live_points=1000,
        evidence_tolerance=0.1,
        sampling_efficiency=0.3,
        outputfiles_basename="chains/multinest_",
        resume=False,
        verbose=True,
        multimodal=True,
        importance_nested_sampling=True,
        max_modes=100,
        use_MPI=False,
    ):
        """
        Run nested sampling with PyMultiNest.

        Notes
        -----
        - Assumes priors are proper scipy.stats distributions with .ppf().
        - Unlike emcee, there is no burn-in / thinning / walkers concept.
        - Stores:
            self.samples   -> equal-weight posterior samples, shape (Ns, ndim)
            self.log_probs -> recomputed log-posterior on self.samples
            self.logZ      -> global log-evidence
            self.logZerr   -> evidence uncertainty
            self.multinest_stats -> full stats dict from Analyzer
        """
        try:
            import pymultinest
        except ImportError as e:
            raise ImportError(
                "PyMultiNest is not installed. Install pymultinest and the MultiNest "
                "library first."
            ) from e

        os.makedirs(os.path.dirname(outputfiles_basename) or ".", exist_ok=True)

        # -------- prior transform: unit cube -> physical parameter space --------
        def prior_transform(cube, ndim, nparams):
            for i, p in enumerate(self.param_priors):
                u = float(cube[i])

                # keep away from exact 0/1 for bounded/infinite-tail priors
                u = min(max(u, 1e-12), 1.0 - 1e-12)

                try:
                    cube[i] = p.ppf(u)
                except Exception as e:
                    raise ValueError(
                        f"Prior {i} does not support a valid .ppf() transform required "
                        f"by PyMultiNest. Define a custom transform for that prior."
                    ) from e

                if not np.isfinite(cube[i]):
                    raise ValueError(
                        f"Prior transform produced non-finite value for parameter {i}."
                    )

        # -------- log-likelihood in physical parameter space --------
        def loglike(cube, ndim, nparams):
            theta = np.array([cube[i] for i in range(self.ndim)], dtype=float)
            ll = self.log_likelihood(theta)

            # MultiNest expects a very small finite value, not -inf/nan
            if not np.isfinite(ll):
                return -1e100
            return float(ll)

        print("")
        print(
            f"[RunMultiNest] Starting PyMultiNest with "
            f"{n_live_points} live points..."
        )

        pymultinest.run(
            LogLikelihood=loglike,
            Prior=prior_transform,
            n_dims=self.ndim,
            n_params=self.ndim,
            outputfiles_basename=outputfiles_basename,
            resume=resume,
            verbose=verbose,
            multimodal=multimodal,
            importance_nested_sampling=importance_nested_sampling,
            n_live_points=n_live_points,
            evidence_tolerance=evidence_tolerance,
            sampling_efficiency=sampling_efficiency,
            max_modes=max_modes,
            use_MPI=use_MPI,
        )

        # -------- read results back --------
        analyzer = pymultinest.Analyzer(
            n_params=self.ndim,
            outputfiles_basename=outputfiles_basename,
        )

        stats = analyzer.get_stats()
        posterior = analyzer.get_equal_weighted_posterior()

        # PyMultiNest returns samples with one extra column in this output.
        # Keep only parameter columns.
        self.samples = np.asarray(posterior[:, : self.ndim], dtype=float)

        # Recompute log-posterior so downstream methods (MAP, corner overlays, etc.)
        # continue to work with your existing class design.
        self.log_probs = np.array(
            [self.log_posterior(theta) for theta in self.samples],
            dtype=float,
        )

        self.logZ = stats["nested sampling global log-evidence"]
        self.logZerr = stats["nested sampling global log-evidence error"]
        self.multinest_stats = stats
        self.sampler = None
        self.chain = None

        print(f"[RunMultiNest] Finished.")
        print(f"[RunMultiNest] logZ    = {self.logZ:.6f}")
        print(f"[RunMultiNest] logZerr = {self.logZerr:.6f}")
        print(f"[RunMultiNest] posterior samples: {self.samples.shape[0]}")

        # Optional summary similar to your emcee printout
        print("Parameter estimates:")
        for i in range(self.ndim):
            q16, q50, q84 = np.percentile(self.samples[:, i], [16, 50, 84])
            print(
                f"θ_{i}: {q50:+.10e}  "
                f"(+{q84 - q50:.1e} / -{q50 - q16:.1e})"
            )


    def run_dynesty(
        self,
        nlive=500,
        dlogz=0.1,
        maxiter=None,
        maxcall=None,
        sample="rwalk",
        bound="multi",
        bootstrap=0,
        enlarge=None,
        walks=25,
        slices=5,
        update_interval=None,
        wt_kwargs=None,
        use_dynamic=False,
        dynamic_kwargs=None,
        resample_equal=True,
        print_progress=True,
    ):
        """
        Run nested sampling with dynesty.

        Parameters
        ----------
        nlive : int
            Number of live points.
        dlogz : float
            Stopping criterion on remaining evidence.
        maxiter, maxcall : int or None
            Optional hard limits passed to dynesty.
        sample : str
            dynesty sampling method, e.g. 'rwalk', 'rslice', 'slice', 'unif'.
        bound : str
            Bounding method, e.g. 'multi', 'single', 'balls', 'none'.
        bootstrap : int
            dynesty bootstrap setting for bounding.
        enlarge : float or None
            Optional enlargement factor for bounds.
        walks : int
            Number of walks for rwalk.
        slices : int
            Number of slices for slice/rslice.
        update_interval : int/float or None
            dynesty update interval.
        wt_kwargs : dict or None
            Passed to dynesty.utils.resample_equal if needed later.
        use_dynamic : bool
            If True, use DynamicNestedSampler. Otherwise NestedSampler.
        dynamic_kwargs : dict or None
            Extra kwargs for DynamicNestedSampler.run_nested().
        resample_equal : bool
            If True, convert weighted posterior samples into equal-weight samples.
        print_progress : bool
            Show dynesty progress bar.
        """

        print("")
        print(
            f"[Dynesty] Starting {'dynamic ' if use_dynamic else ''}nested sampling "
            f"with nlive={nlive}, sample='{sample}', bound='{bound}'"
        )

        if wt_kwargs is None:
            wt_kwargs = {}
        if dynamic_kwargs is None:
            dynamic_kwargs = {}

        # -------- Prior transform --------
        # dynesty samples u in [0,1]^ndim and maps to theta through the prior inverse CDF
        def prior_transform(u):
            theta = np.empty(self.ndim)
            for i, prior in enumerate(self.param_priors):
                val = prior.ppf(u[i])
                if not np.isfinite(val):
                    raise ValueError(
                        f"Prior {i} returned non-finite value in ppf at u={u[i]:.6e}"
                    )
                theta[i] = val
            return theta

        # -------- Log-likelihood --------
        # dynesty expects ONLY log-likelihood; priors are handled by prior_transform
        def dynesty_loglike(theta):
            return self.log_likelihood(theta)

        sampler_kwargs = dict(
            loglikelihood=dynesty_loglike,
            prior_transform=prior_transform,
            ndim=self.ndim,
            sample=sample,
            bound=bound,
            bootstrap=bootstrap,
        )

        if enlarge is not None:
            sampler_kwargs["enlarge"] = enlarge
        if update_interval is not None:
            sampler_kwargs["update_interval"] = update_interval

        # Method-specific tuning
        if sample == "rwalk":
            sampler_kwargs["walks"] = walks
        elif sample in ("slice", "rslice"):
            sampler_kwargs["slices"] = slices

        # -------- Build sampler --------
        if use_dynamic:
            self.sampler = dynesty.DynamicNestedSampler(**sampler_kwargs)
            self.sampler.run_nested(
                print_progress=print_progress,
                maxiter=maxiter,
                maxcall=maxcall,
                dlogz_init=dlogz,
                **dynamic_kwargs,
            )
        else:
            sampler_kwargs["nlive"] = nlive
            self.sampler = dynesty.NestedSampler(**sampler_kwargs)
            self.sampler.run_nested(
                dlogz=dlogz,
                print_progress=print_progress,
                maxiter=maxiter,
                maxcall=maxcall,
            )

        # -------- Store results --------
        self.results = self.sampler.results

        # Raw weighted nested-sampling outputs
        raw_samples = np.asarray(self.results.samples)  # (nsamps, ndim)
        logwt = np.asarray(self.results.logwt)  # log-weights
        logz_final = float(self.results.logz[-1])  # final log-evidence
        logzerr_final = float(self.results.logzerr[-1])  # evidence error
        logl = np.asarray(self.results.logl)  # log-likelihoods

        # Normalized posterior weights
        weights = np.exp(logwt - logz_final)

        self.logz = logz_final
        self.logzerr = logzerr_final
        self.posterior_weights = weights
        self.raw_ns_samples = raw_samples
        self.raw_ns_logl = logl

        # Equal-weight posterior resampling if requested
        if resample_equal:
            self.samples = dyfunc.resample_equal(raw_samples, weights, **wt_kwargs)
            # Posterior "log probs" only up to a constant unless priors are recomputed
            self.log_probs = np.array(
                [self.log_posterior(theta) for theta in self.samples]
            )
        else:
            self.samples = raw_samples
            self.log_probs = np.array(
                [self.log_posterior(theta) for theta in self.samples]
            )

        # dynesty does not have walkers/chains like emcee
        self.chain = None
        self.log_prob_chain = None

        # Simple posterior summaries
        mean = np.average(raw_samples, axis=0, weights=weights)
        cov = np.cov(raw_samples.T, aweights=weights)

        self.posterior_mean = mean
        self.posterior_cov = cov

        print(f"[Dynesty] Done.")
        print(f"[Dynesty] logZ  = {self.logz:.6f}")
        print(f"[Dynesty] logZerr = {self.logzerr:.6f}")
        print(f"[Dynesty] Posterior mean = {self.posterior_mean}")

        return self.results
    '''
