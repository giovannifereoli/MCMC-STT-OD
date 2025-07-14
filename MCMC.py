import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
import multiprocessing
from scipy.optimize import minimize, basinhopping
import warnings
from matplotlib.patches import Ellipse

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

    def optimize_initial_guess(self, method="Powell", disp=True, n_iter=100):
        def objective_logpost(theta):
            return -self.log_posterior(theta)  # minimize negative log-posterior

        x0 = self.initial_params.copy()

        print("")
        print(f"\n[Optimization] Starting optimization using method: {method}")

        if method.lower() == "basinhopping":
            # Use Powell as default local method under basinhopping
            local_method = "Powell"
            minimizer_kwargs = {"method": local_method, "options": {"disp": disp}}
            result = basinhopping(
                objective_logpost,
                x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=n_iter,
                disp=disp,
            )
        else:
            # Local optimization
            result = minimize(
                objective_logpost,
                x0,
                method=method,
                options={"disp": disp},
            )

        if result.success:
            print(f"[Optimization] Success: {result.message}")
        else:
            print(f"[Optimization] Warning: {result.message}")
        print(f"[Optimization] Optimal θ: {result.x}")

        return result.x

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

        # Run MCMC using emcee
        with multiprocessing.get_context("fork").Pool() as pool:
            self.sampler = emcee.EnsembleSampler(
                n_walkers, self.ndim, log_prob_func, pool=pool
            )
            self.sampler.run_mcmc(pos, n_samples, progress=True)

        # Try to estimate autocorrelation time
        print("")
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

            # Warn if total steps are insufficient for autocorrelation convergence
            if n_samples < 100 * max_tau:
                warnings.warn(
                    f"[Run] n_samples = {n_samples} may be too small. "
                    f"Recommended: at least 100 x max(tau) = {100 * max_tau:.1f} steps "
                    f"for reliable sampling.",
                    UserWarning,
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
        self.samples = self.sampler.get_chain(discard=burn_in, thin=thin, flat=True)
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
        best_params = np.median(self.samples, axis=0)
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
        # 1) Compute median parameters and post-fit residuals
        best_params = np.median(self.samples, axis=0)
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
        self, batch_mean=None, batch_cov=None, use_median_as_truth=True
    ):
        if self.samples is None:
            print("Run MCMC first.")
            return

        truths = np.median(self.samples, axis=0) if use_median_as_truth else None
        labels = [f"$\\theta_{{{i}}}$" for i in range(self.ndim)]

        # Plot MCMC corner
        fig = corner.corner(
            self.samples,
            labels=labels,
            truths=truths,
            show_titles=True,
            title_fmt=".6f",
            title_kwargs={"fontsize": 12},
            color="blue",
        )

        # Overlay batch mean and 1-sigma ellipses
        if batch_mean is not None and batch_cov is not None:
            axes = np.array(fig.axes).reshape((self.ndim, self.ndim))
            for i in range(self.ndim):
                for j in range(i):
                    ax = axes[i, j]
                    cov_sub = batch_cov[np.ix_([j, i], [j, i])]  # 2x2 sub-covariance
                    mean_sub = [batch_mean[j], batch_mean[i]]

                    # Red dot for mean
                    ax.plot(
                        mean_sub[0],
                        mean_sub[1],
                        "ro",
                        label="Batch Mean" if (i == 1 and j == 0) else "",
                    )

                    # Eigen-decompose 2x2 covariance for ellipse
                    vals, vecs = np.linalg.eigh(cov_sub)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                    # Width and height are 2*sqrt(eigenvalues) for 1-sigma ellipse
                    width, height = 2 * np.sqrt(vals)
                    ellipse = Ellipse(
                        xy=mean_sub,
                        width=width,
                        height=height,
                        angle=angle,
                        edgecolor="red",
                        facecolor="none",
                        lw=1.5,
                        label=r"Batch $1\sigma$ Ellipse" if (i == 1 and j == 0) else "",
                    )
                    ax.add_patch(ellipse)

            # Add legend only once
            axes[1, 0].legend(loc="upper right", fontsize=10)

        fig.set_size_inches(12, 12)
        plt.tight_layout()
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
                "Acceptance rate outside optimal range (0.2-0.5). Consider tuning initialization or step size."
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

        theta_best = np.median(self.samples, axis=0)
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
        # print("Parameter uncertainties (1σ):")
        # for i, std in enumerate(param_uncertainties):
        #    print(f"  θ_{i}: ±{std:.14f}")

    def get_unwhitened_samples(self):
        if not getattr(self, "is_whitened", False):
            return self.samples
        return (self.whiten_L @ self.samples.T + self.whiten_mean[:, None]).T

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
