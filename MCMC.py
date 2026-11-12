"""
MCMC orbit determination: MCMCModel (emcee sampler, priors, whitening, diagnostics,
chain save/load) and ChainReplay (a saved chain in place of the emcee sampler).

Figures live in MCMCPlots.py (the plot_* methods below forward to it); the posterior
Gaussian-mixture split and its STM propagation live in GMM.py.
"""

import io
import multiprocessing
from pathlib import Path

import emcee
import numpy as np
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
    needed. run() maps the samples back to model coordinates.

    Parameters
    ----------
    residuals_func : callable
        theta (ndim,) -> normalized residuals (N,). The sampler pool pickles the model, so
        use a top-level function or a class with __call__, not a closure.
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
            Steps per walker. A warning is printed when n_samples < 100 * max(tau).
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

        Returns
        -------
        None
            Sets sampler, tau, burn_in, thin, chain and log_prob_chain (sampled
            coordinates), samples (model coordinates) and log_probs.
        """
        # Use optimization for better initial guess
        if use_optimize:
            optimized_guess = self.optimize_initial_guess(method=method_optimize)
        else:
            print("[Run] Skipping optimization. Using initial parameters.")
            optimized_guess = self.initial_params.copy()

        if getattr(self, "is_whitened", False):
            # Whiten the optimized guess
            x0_white = self.whiten_Linv @ (optimized_guess - self.whiten_mean)

            # Sample walkers in whitened space (identity covariance, scaled)
            pos = x0_white + spherical_spread * np.random.randn(n_walkers, self.ndim)
        else:
            # Build realistic covariance from priors
            stds = np.array([p.std() for p in self.param_priors])
            init_cov = np.diag(stds**2)

            # Sample directly in parameter space using inflated prior covariance
            pos = np.random.multivariate_normal(
                mean=optimized_guess, cov=spherical_spread**2 * init_cov, size=n_walkers
            )

        # Determine if we need to use whitened log_prob
        log_prob_func = (
            self.log_prob_whitened
            if getattr(self, "is_whitened", False)
            else self.log_prob
        )

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
        with multiprocessing.get_context("fork").Pool() as pool:
            print(
                "[Run] Using multiprocessing with fork context. Number of processes:",
                pool._processes,
            )
            self.sampler = emcee.EnsembleSampler(
                nwalkers=n_walkers,
                ndim=self.ndim,
                log_prob_fn=log_prob_func,
                pool=pool,
                moves=moves,
            )
            self.sampler.run_mcmc(pos, n_samples, progress=True)

        # Try to estimate autocorrelation time
        print("[Run] Estimating autocorrelation time...")
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

        if tau is not None:
            max_tau = np.max(tau)
            min_tau = np.min(tau)
            n_tau = n_samples / max_tau

            # Warn if total steps are insufficient for autocorrelation convergence
            if n_samples < 100 * max_tau:
                print(
                    f"[Run] Warning: n_samples = {n_samples} corresponds to "
                    f"{n_tau:.1f} autocorrelation times. "
                    f"Recommended: at least 100 × max(tau) = {100 * max_tau:.1f} steps "
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
            # First try default autocorr calculation (faster)
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
        - extra_<key>: the **extra arrays.

        A re-saved ChainReplay keeps its stored grid. A sampler without get_chain (e.g.
        nested sampling) stores no full_chain.
        """
        if self.samples is None:
            raise ValueError("No samples to save: run the sampler first.")
        path = Path(MCMCPlots.figure_dir() / "chain.npz" if path is None else path)
        data = {"samples": np.asarray(self.samples), "log_probs": np.asarray(self.log_probs)}
        for key in ("chain", "log_prob_chain", "tau", "burn_in", "thin"):
            val = getattr(self, key, None)
            if val is not None:
                data[key] = np.asarray(val)
        if getattr(self, "is_whitened", False):
            data["whiten_mean"] = np.asarray(self.whiten_mean)
            data["whiten_L"] = np.asarray(self.whiten_L)
        smp = self.sampler
        if isinstance(smp, ChainReplay):  # re-saving a loaded run: keep its stored grid
            stride, full, full_lp = smp.stride, smp.full_chain, smp.full_log_prob
        elif smp is not None and hasattr(smp, "get_chain"):  # emcee
            n_steps = int(smp.iteration)
            stride = 1 if not max_full_steps else max(1, n_steps // int(max_full_steps))
            if self.tau is not None and np.all(np.isfinite(self.tau)):
                stride = max(1, min(stride, int(np.min(self.tau) // 10)))
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

        Sets samples, log_probs, chain, log_prob_chain, tau, burn_in, thin and the whitening
        (whiten_Linv recomputed), and a ChainReplay in place of the emcee sampler (same
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
    # MCMCModel plots (Tests/ and Extra/ scripts use them).
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
