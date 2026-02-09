import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
import multiprocessing
from scipy.optimize import (
    minimize,
    basinhopping,
    dual_annealing,
    differential_evolution,
)
from matplotlib.patches import Ellipse
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from scipy.optimize import least_squares

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.7,
        "font.size": 12,
    }
)


class MCMCModel:
    def __init__(self, residuals_func, initial_params, param_priors, observed_data):
        self.residuals_func = residuals_func
        self.initial_params = np.array(initial_params)
        self.param_priors = param_priors
        self.observed_data = observed_data
        self.ndim = len(initial_params)
        self.sampler = None
        self.samples = None
        self.log_probs = None

    def log_prior(self, theta):
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
        # NOTE: covariance already in the residuals_func
        residuals_normalized = self.residuals_func(theta)
        return -0.5 * np.sum(residuals_normalized**2)

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def log_prob(self, theta):
        return self.log_posterior(theta)

    def prior_residuals(self, theta):
        """
        Turn priors into pseudo-residuals so least_squares does MAP.
        Works best for Gaussian-like priors.
        Fallback: if prior doesn't have mean/var, return empty.
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
            return -log_prob_func(theta)

        # Build bounds from priors if requested
        bounds = None
        if use_bounds_from_priors:
            bounds = [(p.ppf(1e-6), p.ppf(1 - 1e-6)) for p in self.param_priors]
            if use_whitened:
                # Transform bounds to whitened space
                bounds_array = np.array(bounds).T  # shape (2, ndim)
                lower, upper = bounds_array[0], bounds_array[1]
                bounds_white = (
                    self.whiten_Linv @ (lower - self.whiten_mean),
                    self.whiten_Linv @ (upper - self.whiten_mean),
                )
                bounds = list(zip(bounds_white[0], bounds_white[1]))

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
        elif method == "lsq":
            print(f"[Optimization] Using least squares: least_squares(method='trf')")

            def residuals_map(theta_white):
                theta = self.whiten_L @ theta_white + self.whiten_mean
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
    ):
        # Use optimization for better initial guess
        optimized_guess = self.optimize_initial_guess(method=method_optimize)

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
            moves = emcee.moves.StretchMove()

        # Run MCMC using emcee
        print("")
        print(
            f"[Run] Starting MCMC sampling with {n_walkers} walkers for {n_samples} steps..."
        )
        with multiprocessing.get_context("fork").Pool() as pool:
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
        self.chain = self.sampler.get_chain(
            discard=burn_in, thin=thin, flat=False
        )  # (nwalkers, nsteps, ndim)
        self.samples = self.chain.reshape(-1, self.ndim)  # flat view for corner
        self.log_probs = self.sampler.get_log_prob(
            discard=burn_in, thin=thin, flat=True
        )

        # If whitening was used, transform samples back to original space
        if getattr(self, "is_whitened", False):
            self.samples = (
                self.whiten_L @ self.samples.T + self.whiten_mean[:, None]
            ).T  # shape (N, ndim)

    def plot_convergence(self):
        if self.samples is None:
            print("Run MCMC first.")
            return

        # Reshape the samples to (n_walkers, n_steps, ndim)
        n_walkers = self.sampler.nwalkers
        n_steps = self.samples.shape[0] // n_walkers
        chain = self.samples.reshape(
            n_walkers, n_steps, self.ndim
        )  # (n_walkers, n_steps, ndim)

        # Plot the chain for each parameter
        fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(chain[:, :, i].T, alpha=0.5)
            ax.set_ylabel(f"$\\theta_{{{i}}}$")
            ax.grid(True)
        axes[-1].set_xlabel("Step Number")
        plt.tight_layout()
        plt.show()

    def plot_log_likelihood(self):
        if self.log_probs is None:
            print("Run MCMC first.")
            return

        # Reshape the log_probs to (n_walkers, n_steps)
        n_walkers = self.sampler.nwalkers
        n_steps = self.log_probs.shape[0] // n_walkers
        log_probs_chain = self.log_probs.reshape(n_walkers, n_steps)

        # Plot the log posterior for each walker
        plt.figure(figsize=(10, 5))
        for i in range(n_walkers):
            plt.plot(-log_probs_chain[i], alpha=0.6, linewidth=1)

        plt.xlabel("Step Number")
        plt.ylabel(r"$-\log \mathcal{P}(\theta \mid y)$")
        plt.title("Log-Posterior (Negative) per Walker After Burn-in")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()

    def plot_postfit_residuals(self):
        best_params = self.get_map_estimate()
        postfit = self.residuals_func(best_params)
        plt.figure(figsize=(8, 4))
        plt.scatter(range(len(postfit)), postfit, marker="o")
        plt.axhline(0, color="k", linestyle="--")
        plt.axhline(3, color="r", linestyle=":")
        plt.axhline(-3, color="r", linestyle=":")
        plt.xlabel("Observation Index")
        plt.ylabel("Residual Normalized")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_postfit_residuals_time(self, t_obs_used, opnav_data=False):
        # 1) Compute median (more meaningful) or MAP (best residuals) parameters and post-fit residuals
        best_params = self.get_map_estimate()
        postfit = self.residuals_func(best_params)

        # 2) Reshape: [r0, rr0, r1, rr1, ...] → (2, N)
        residuals_matrix = postfit.reshape(-1, 2).T  # shape = (2, N)

        # 3) Create subplots
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        time_hr = t_obs_used / 3600.0
        ylabels = (
            [r"RA Residual [$\sigma$]", r"DEC Residual [$\sigma$]"]
            if opnav_data
            else [r"Range Residual [$\sigma$]", r"Range-Rate Residual [$\sigma$]"]
        )

        # 4) Top residual
        ax0.plot(
            time_hr,
            residuals_matrix[0],
            "o",
            color="blue",
            markersize=4,
            label="Residual",
        )
        ax0.axhline(0, color="black", linestyle="--")
        ax0.axhline(3, color="red", linestyle=":")
        ax0.axhline(-3, color="red", linestyle=":")
        ax0.set_ylabel(ylabels[0])
        ax0.grid(True)
        ax0.legend(loc="upper right")

        # 5) Bottom residual
        ax1.plot(
            time_hr,
            residuals_matrix[1],
            "o",
            color="purple",
            markersize=4,
            label="Residual",
        )
        ax1.axhline(0, color="black", linestyle="--")
        ax1.axhline(3, color="red", linestyle=":")
        ax1.axhline(-3, color="red", linestyle=":")
        ax1.set_ylabel(ylabels[1])
        ax1.set_xlabel("Time [hours since epoch]")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def plot_corner(self, use_median_as_truth=True):
        if self.samples is None:
            print("Run MCMC first.")
            return

        # Use median as truth if specified
        truths = None
        if use_median_as_truth:
            truths = np.median(self.samples, axis=0)

        # Plotting corner plot
        fig = corner.corner(
            self.samples,
            labels=[f"$\\theta_{{{i}}}$" for i in range(self.ndim)],
            truths=truths,
            show_titles=True,
            title_fmt=".10f",
            title_kwargs={"fontsize": 12},
        )

        fig.set_size_inches(12, 12)
        plt.tight_layout()
        plt.show()

    def plot_corner_with_batch(
        self,
        batch_mean=None,
        batch_cov=None,
        use_median_as_truth=True,
        true_theta=None,
        plot_contours_labels=False,  # keep off unless you really need it
        batch_sigma_levels=(1.0,),
        idx=None,  # NEW: list/array of parameter indices to plot
        max_dims=13,
        bins=40,
        quantile_range=(0.005, 0.995),  # NEW: robust axis limits
        title_digits=3,  # NEW: avoid long titles
    ):
        if self.samples is None:
            print("Run MCMC first.")
            return

        samples = np.asarray(self.samples)

        # ----------------------------
        # Choose dimensions to plot
        # ----------------------------
        ndim = samples.shape[1]
        if idx is None:
            if ndim > max_dims:
                idx = np.arange(max_dims)  # first max_dims by default
            else:
                idx = np.arange(ndim)
        else:
            idx = np.asarray(idx, dtype=int)

        s = samples[:, idx]
        d = s.shape[1]

        # Truths & best params on the same subspace
        best_params_full = self.get_map_estimate()
        best_params = best_params_full[idx] if best_params_full is not None else None

        truths_full = np.median(samples, axis=0) if use_median_as_truth else true_theta
        truths = truths_full[idx] if truths_full is not None else None

        # Labels
        labels = [f"$\\theta_{{{k}}}$" for k in idx]

        # ----------------------------
        # Robust range so outliers don't ruin scaling
        # ----------------------------
        qlo, qhi = quantile_range
        ranges = []
        for j in range(d):
            lo, hi = np.quantile(s[:, j], [qlo, qhi])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ranges.append((lo, hi))
            else:
                # fallback
                mn, mx = np.min(s[:, j]), np.max(s[:, j])
                ranges.append((mn, mx))

        # ----------------------------
        # Figure size that scales with d
        # Rule of thumb: ~1.2–1.6 inch per dim
        # ----------------------------
        inches_per_dim = 1.35
        fig_size = max(8.0, inches_per_dim * d)

        fig = corner.corner(
            s,
            labels=labels,
            truths=truths,
            range=ranges,
            bins=bins,
            show_titles=True,
            title_fmt=f".{title_digits}g",
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 11},
            color="tab:blue",
            plot_contours=True,
            fill_contours=False,
            smooth=1.0,
            smooth1d=1.0,
            # avoid huge numbers of points making the plot slow:
            # max_n_ticks=4,  # (older corner versions)
        )

        # Corner returns a figure; axes are in row-major order
        axes = np.array(fig.axes).reshape((d, d))

        # ----------------------------
        # Overlay batch ellipses / points
        # ----------------------------
        if batch_mean is not None:
            batch_mean = np.asarray(batch_mean)
            bm = batch_mean[idx]
        else:
            bm = None

        if batch_cov is not None:
            batch_cov = np.asarray(batch_cov)
            bc = batch_cov[np.ix_(idx, idx)]
        else:
            bc = None

        if true_theta is not None:
            true_theta = np.asarray(true_theta)
            tt = true_theta[idx]
        else:
            tt = None

        for i in range(d):
            for j in range(i):
                ax = axes[i, j]

                # batch mean + ellipses
                if bm is not None:
                    ax.plot(
                        bm[j],
                        bm[i],
                        "ro",
                        ms=4.5,
                        label="Batch Mean" if (i == 1 and j == 0) else "",
                    )

                if bm is not None and bc is not None and np.all(np.isfinite(bc)):
                    cov_sub = bc[np.ix_([j, i], [j, i])]
                    mean_sub = [bm[j], bm[i]]

                    # guard against singular / negative eigenvalues
                    vals, vecs = np.linalg.eigh(cov_sub)
                    vals = np.maximum(vals, 0.0)
                    order = np.argsort(vals)[::-1]
                    vals, vecs = vals[order], vecs[:, order]

                    # if both eigenvalues are ~0, skip ellipse
                    if np.max(vals) > 0:
                        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                        for ksig in batch_sigma_levels:
                            width, height = 2 * ksig * np.sqrt(vals)
                            ell = Ellipse(
                                xy=mean_sub,
                                width=width,
                                height=height,
                                angle=angle,
                                edgecolor="red",
                                facecolor="none",
                                lw=1.4 if ksig == 1 else 1.0,
                                linestyle="-" if ksig == 1 else "--",
                                alpha=0.95,
                                label=(
                                    rf"Batch {ksig:g}$\sigma$"
                                    if (i == 1 and j == 0)
                                    else ""
                                ),
                            )
                            ax.add_patch(ell)

                # true value
                if tt is not None:
                    ax.plot(
                        tt[j],
                        tt[i],
                        "go",
                        ms=4.5,
                        label="True Value" if (i == 1 and j == 0) else "",
                    )

                # MCMC MAP
                if best_params is not None:
                    ax.plot(
                        best_params[j],
                        best_params[i],
                        "kx",
                        ms=6,
                        mew=1.6,
                        label="MCMC MAP" if (i == 1 and j == 0) else "",
                    )

                # Plot contour labels
                if plot_contours_labels:
                    x = self.samples[:, j]
                    y = self.samples[:, i]
                    data = np.vstack([x, y])
                    kde = gaussian_kde(data)
                    xi, yi = np.mgrid[
                        x.min() : x.max() : 100j, y.min() : y.max() : 100j
                    ]
                    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

                    # Choose contour levels matching corner.corner defaults
                    levels = [0.118, 0.393, 0.675, 0.864]  # ≈ 0.5σ, 1σ, 1.5σ, 2σ
                    # Estimate actual values to pass to contour
                    sorted_zi = np.sort(zi.ravel())[::-1]
                    cdf = np.cumsum(sorted_zi)
                    cdf /= cdf[-1]
                    value_levels = []
                    for lv in levels:
                        idx = np.searchsorted(cdf, lv)
                        value_levels.append(sorted_zi[idx])
                    value_levels.sort()

                    contour_set = ax.contour(
                        xi, yi, zi, levels=value_levels, colors="black", linewidths=1
                    )
                    fmt_dict = {
                        l: f"{int(100 * lv)}\\%"
                        for l, lv in zip(contour_set.levels, levels)
                    }
                    ax.clabel(contour_set, fmt=fmt_dict, inline=True, fontsize=8)

        # ----------------------------
        # Global formatting tweaks
        # ----------------------------
        for ax in fig.get_axes():
            ax.tick_params(labelsize=8)
            # reduce label padding a bit
            ax.xaxis.labelpad = 6
            ax.yaxis.labelpad = 6

        # Single legend in one axis (if it exists)
        if d >= 2:
            axes[1, 0].legend(loc="upper right", fontsize=9, frameon=True)

        fig.set_size_inches(fig_size, fig_size)
        fig.subplots_adjust(
            wspace=0.05, hspace=0.05
        )  # better than tight_layout for corner
        plt.show()

    def summary(self):
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
            thin = int(0.5 * np.min(tau))  # Decorrelation
            print(f"Suggested burn-in: {burnin} steps ({2} x max(tau))")
            print(f"Suggested thinning: every {thin} steps ({0.5} x min(tau))")

        # Calculate the acceptance fraction
        acceptance_fraction = self.sampler.acceptance_fraction
        mean_acceptance = np.mean(acceptance_fraction)
        print("Mean acceptance rate:", mean_acceptance)

        # Warn if acceptance rate is outside optimal range
        if mean_acceptance < 0.2 or mean_acceptance > 0.5:
            print(
                "Acceptance rate outside optimal range (0.2-0.5). Consider tuning initialization step size."
            )

        print("Parameter estimates:")
        for i in range(self.ndim):
            mcmc = np.percentile(self.samples[:, i], [16, 50, 84])
            q_lower, q_upper = np.diff(mcmc)
            median = mcmc[1]
            print(f"θ_{i}: {median:+.10e}  (+{q_upper:.1e} / -{q_lower:.1e})")

    def print_regression_diagnostics(self):
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
        param_uncertainties = np.std(self.samples, axis=0)

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
        # print("Parameter uncertainties (1σ):")
        # for i, std in enumerate(param_uncertainties):
        #    print(f"  θ_{i}: ±{std:.14f}")

    def setup_whitening_from_priors(self):
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
        self.whiten_mean = np.array(self.initial_params)
        self.whiten_L = np.linalg.cholesky(cov)
        self.whiten_Linv = np.linalg.inv(self.whiten_L)
        self.is_whitened = True

    def log_posterior_whitened(self, theta_white):
        if not self.is_whitened:
            raise RuntimeError("Whitening not set up")

        theta = self.whiten_L @ theta_white + self.whiten_mean
        return self.log_posterior(theta)

    def log_prob_whitened(self, theta_white):
        return self.log_posterior_whitened(theta_white)

    def save_chain(self, path="chain_data.npz"):
        if self.chain is None:
            raise ValueError("Chain is not available. Run MCMC first.")
        np.savez_compressed(
            path,
            chain=self.chain,
            log_prob_chain=self.log_prob_chain,
            samples=self.samples,
            log_probs=self.log_probs,
        )
        print(f"Chain saved to '{path}'")

    def load_chain(self, path="chain_data.npz"):
        data = np.load(path)
        self.chain = data["chain"]
        self.log_prob_chain = data["log_prob_chain"]
        self.samples = data["samples"]
        self.log_probs = data["log_probs"]
        print(f"Chain loaded from '{path}'")

    def get_map_estimate(self):
        if self.samples is None or self.log_probs is None:
            raise RuntimeError("Run MCMC before computing MAP estimate.")

        idx_max = np.argmax(self.log_probs)
        map_params = self.samples[idx_max]
        return map_params

    def gelman_rubin_diagnostic(self, split=True, threshold=1.1):
        if self.samples is None:
            raise RuntimeError("Run MCMC before computing Gelman-Rubin diagnostic.")

        print(
            "\n DANGER: This implementation assumes all walkers are independent chains. In EMCEE, this is not true. Use with caution. \n"
        )

        # Reshape: (n_walkers, n_steps, ndim)
        chain = self.chain
        n_walkers, n_steps, ndim = chain.shape

        if split:
            # Split chains to double the number of chains
            chain = chain.reshape((n_walkers * 2, n_steps // 2, ndim))
            _, n_samples, _ = chain.shape
        else:
            _, n_samples = n_walkers, n_steps

        r_hat = np.empty(ndim)
        for d in range(ndim):
            samples = chain[:, :, d]
            chain_means = np.mean(samples, axis=1)
            chain_vars = np.var(samples, axis=1, ddof=1)

            B = n_samples * np.var(chain_means, ddof=1)
            W = np.mean(chain_vars)
            V_hat = (1 - 1 / n_samples) * W + B / n_samples
            r_hat[d] = np.sqrt(V_hat / W)

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
        Estimate and print effective sample size (ESS) for each parameter.
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC first.")

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

        total_samples = self.sampler.get_chain(discard=0, flat=True).shape[0]
        ess = total_samples / tau

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

    def plot_autocorrelation(self, max_lag=100):
        if self.samples is None:
            print("Run MCMC first.")
            return

        chain = self.samples

        fig, axes = plt.subplots(self.ndim, 1, figsize=(8, 2 * self.ndim))
        if self.ndim == 1:
            axes = [axes]

        for i in range(self.ndim):
            acf_vals = acf(chain[:, i], nlags=max_lag, fft=True)
            axes[i].stem(range(max_lag + 1), acf_vals, basefmt=" ")
            axes[i].set_ylabel(r"$\mathrm{{ACF}}[\theta_{{{}}}]$".format(i))
            axes[i].grid(True)

        plt.xlabel("Lag")
        plt.tight_layout()
        plt.show()

    def get_estimate_and_covariance(self, method="map"):
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling this method.")

        if method == "median":
            theta_hat = np.median(self.samples, axis=0)
        elif method == "mean":
            theta_hat = np.mean(self.samples, axis=0)
        elif method == "map":
            theta_hat = self.get_map_estimate()
        else:
            raise ValueError("Method must be 'median' or 'mean'.")

        cov = np.cov(self.samples.T)
        return theta_hat, cov

    '''
        def run_hmc(
            self,
            n_samples=3000,
            burn_in=500,
            step_size=1e-3,
            num_integration_steps=20,
            print_every=10,
            mass_matrix=None,  # Pass a full-rank positive-definite matrix or None for identity
        ):
            """
            Run Hamiltonian Monte Carlo using NumPy with general mass matrix and momentum resampling.
            """

            # Default mass matrix = identity
            M = np.eye(self.ndim) if mass_matrix is None else np.array(mass_matrix)
            M_inv = np.linalg.inv(M)
            L = np.linalg.cholesky(M)

            def U(theta):
                lp = self.log_prior(theta)
                if not np.isfinite(lp):
                    return np.inf
                return -lp - self.log_likelihood(theta)

            def grad_U(theta):
                eps = 1e-6
                grad = np.zeros_like(theta)
                for i in range(len(theta)):
                    d = np.zeros_like(theta)
                    d[i] = eps
                    grad[i] = (U(theta + d) - U(theta - d)) / (2 * eps)
                return grad

            def leapfrog(theta, p, step_size, num_steps):
                theta_new = theta.copy()
                p_new = p - 0.5 * step_size * grad_U(theta_new)

                for _ in range(num_steps - 1):
                    theta_new += step_size * M_inv @ p_new
                    p_new -= step_size * grad_U(theta_new)

                theta_new += step_size * M_inv @ p_new
                p_new -= 0.5 * step_size * grad_U(theta_new)
                return theta_new, -p_new

            theta_current = np.array(self.initial_params)
            samples = []
            logps = []
            accepted = 0
            total_steps = n_samples + burn_in

            print("Starting HMC sampling...")
            for i in range(total_steps):
                z = np.random.randn(self.ndim)
                p_current = L @ z  # Now p ~ N(0, M)
                theta_proposed, p_proposed = leapfrog(
                    theta_current, p_current, step_size, num_integration_steps
                )

                U_current = U(theta_current)
                U_proposed = U(theta_proposed)

                K_current = 0.5 * p_current.T @ M_inv @ p_current
                K_proposed = 0.5 * p_proposed.T @ M_inv @ p_proposed
                log_accept_prob = U_current + K_current - U_proposed - K_proposed

                accepted_flag = False
                if np.log(np.random.rand()) < log_accept_prob:
                    theta_current = theta_proposed
                    accepted += 1
                    accepted_flag = True

                if i >= burn_in:
                    samples.append(theta_current.copy())
                    logps.append(-U(theta_current))

                if i % print_every == 0 or i == total_steps - 1:
                    phase = "Burn-in" if i < burn_in else "Sampling"
                    print(
                        f"[{i}/{total_steps}] {phase} | "
                        f"Accepted: {accepted}/{i+1} ({(accepted/(i+1))*100:.1f}%)"
                    )

            self.samples = np.array(samples)
            self.log_probs = np.array(logps)

            print("HMC sampling completed.")
            print(f"Final acceptance rate: {(accepted / total_steps):.2%}")
    '''

    '''
        def find_candidates_de_one_shot(
        self,
        n_keep=30,              # how many candidates to keep from the DE population
        top_k_optima=8,         # how many unique optima to return after dedup
        popsize=60,
        maxiter=250,
        seed=42,
        use_bounds_from_priors=True,
        q_bounds=1e-6,
        polish_local=True,
        method_local="Powell",
        maxiter_local=600,
        dedup_tol=1e-2,
        verbose=True,
    ):
        """
        "One-shot" approach:
        - Run differential evolution once (global search)
        - Keep best candidates from its final population
        - Optionally locally polish them
        - De-duplicate and return top optima

        Returns:
            optima_theta: (K, ndim) unique optima in ORIGINAL space
            optima_logp: (K,) log posterior values
            candidates_theta: (M, ndim) raw kept candidates (ORIGINAL space)
            candidates_logp: (M,) log posterior values
        """
        use_whitened = getattr(self, "is_whitened", False)
        log_prob_func = self.log_prob_whitened if use_whitened else self.log_prob

        def objective(theta):
            lp = log_prob_func(theta)
            if not np.isfinite(lp):
                return 1e100
            return -lp

        if use_bounds_from_priors:
            bounds = self._build_bounds_from_priors(use_whitened=use_whitened, q=q_bounds)
        else:
            raise ValueError("Provide bounds or set use_bounds_from_priors=True.")

        # ---- run DE once
        if verbose:
            print(f"[OneShot-DE] popsize={popsize}, maxiter={maxiter}, n_keep={n_keep}")

        res = differential_evolution(
            objective,
            bounds=bounds,
            seed=seed,
            popsize=popsize,
            maxiter=maxiter,
            polish=False,          # important: we want the population, we polish ourselves
            tol=1e-10,
            updating="deferred",
            workers=1,             # deterministic
            disp=False,
        )

        # ---- extract population (preferred)
        if hasattr(res, "population") and res.population is not None:
            pop = np.array(res.population)          # shape (NP, ndim)
        else:
            # Fallback for older SciPy: store population via callback
            raise RuntimeError(
                "SciPy result has no `.population`. "
                "Upgrade SciPy or implement a callback-based collector."
            )

        # Evaluate objective for the whole final population
        vals = np.array([objective(x) for x in pop])
        idx = np.argsort(vals)[: min(n_keep, len(pop))]
        cand = pop[idx]
        cand_logp = -vals[idx]

        # ---- optional local polishing of candidates
        refined = []
        if polish_local:
            if verbose:
                print(f"[OneShot-DE] Polishing {len(cand)} candidates with {method_local}...")
            for i, x0 in enumerate(cand):
                rloc = minimize(
                    objective,
                    x0,
                    method=method_local,
                    options={"maxiter": maxiter_local, "disp": False},
                )
                if np.isfinite(rloc.fun):
                    refined.append((rloc.x, -rloc.fun))
            if len(refined) == 0:
                refined = [(cand[0], cand_logp[0])]
        else:
            refined = list(zip(cand, cand_logp))

        # Sort refined by best logp
        refined.sort(key=lambda t: -t[1])
        thetas = np.array([t for (t, _) in refined])
        logps = np.array([lp for (_, lp) in refined])

        # ---- dedup in meaningful metric
        if use_whitened:
            thetas_metric = thetas.copy()
        else:
            stds = np.array([p.std() for p in self.param_priors])
            stds = np.where(stds > 0, stds, 1.0)
            thetas_metric = thetas / stds[None, :]

        uniq = []
        uniq_lp = []
        uniq_metric = []
        for t, lp, tm in zip(thetas, logps, thetas_metric):
            if len(uniq) == 0:
                uniq.append(t); uniq_lp.append(lp); uniq_metric.append(tm); continue
            d = np.min(np.linalg.norm(np.array(uniq_metric) - tm, axis=1))
            if d > dedup_tol:
                uniq.append(t); uniq_lp.append(lp); uniq_metric.append(tm)
            if len(uniq) >= top_k_optima:
                break

        optima = np.array(uniq)
        optima_lp = np.array(uniq_lp)

        # ---- map to original space if whitened
        if use_whitened:
            optima = (self.whiten_L @ optima.T + self.whiten_mean[:, None]).T
            cand  = (self.whiten_L @ cand.T + self.whiten_mean[:, None]).T

        if verbose:
            print(f"[OneShot-DE] Unique optima: {len(optima)}")
            for i, (t, lp) in enumerate(zip(optima, optima_lp), 1):
                print(f"  #{i}: logp={lp:.3f}")

        return optima, optima_lp, cand, cand_logp
    '''
