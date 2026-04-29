import math
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
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from scipy.optimize import least_squares
from datetime import datetime
import os
import dynesty
from dynesty import utils as dyfunc

# Publication-ish defaults
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.7,
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
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
        elif method == "LSQ":
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
        stretch_a=2.0,
        use_optimize=True,
    ):
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

    def plot_convergence(
        self,
        idx=None,
        max_dims: int = 20,
        thin: int = 1,
        discard: int = 0,
        max_walkers_to_plot=None,
        use_x_labels: bool = True,
        save_folder: str = "results",
        save_pdf: bool = True,
        fname_prefix: str = "trace",
        ylim_quantiles=(0.005, 0.995),
        ylim_pad_frac: float = 0.12,
        show_burnin_line: bool = True,
    ):
        if getattr(self, "samples", None) is None:
            print("Run MCMC first.")
            return
        if getattr(self, "sampler", None) is None or not hasattr(
            self.sampler, "nwalkers"
        ):
            raise AttributeError("self.sampler with attribute `nwalkers` is required.")

        samples = np.asarray(self.samples)
        if samples.ndim != 2:
            raise ValueError("self.samples must be a 2D array (n_samples_total, ndim).")

        n_walkers = int(self.sampler.nwalkers)
        ndim = samples.shape[1]
        if samples.shape[0] % n_walkers != 0:
            raise ValueError(
                f"Number of rows in samples ({samples.shape[0]}) is not divisible by n_walkers ({n_walkers})."
            )

        n_steps = samples.shape[0] // n_walkers

        # reshape to (W, T, D) using emcee flattening convention
        chain = samples.reshape(n_steps, n_walkers, ndim).transpose(1, 0, 2)

        # select parameters
        if idx is None:
            idx = np.arange(min(ndim, max_dims))
        else:
            idx = np.asarray(idx, dtype=int)
            if idx.size > max_dims:
                idx = idx[:max_dims]

        # discard + thin
        discard = max(0, int(discard))
        thin = max(1, int(thin))
        chain = chain[:, discard:, :]
        chain = chain[:, ::thin, :]

        # walkers to draw
        if max_walkers_to_plot is None:
            w_plot = n_walkers
        else:
            w_plot = min(n_walkers, int(max_walkers_to_plot))
        chain_plot = chain[:w_plot, :, :]

        n_panels = int(idx.size)
        T = chain_plot.shape[1]
        x = np.arange(T)

        # grid: prefer more columns (paper-friendly)
        n_cols = int(min(5, max(2, math.ceil(math.sqrt(n_panels) + 0.5))))
        n_rows = int(math.ceil(n_panels / n_cols))

        # size: tuned for print
        fig_w = max(7.0, 2.35 * n_cols)
        fig_h = max(2.6, 1.55 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        # global style (minimal, professional)
        for ax in axes:
            ax.grid(False)
            ax.tick_params(labelsize=8, pad=2.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        # compute burn-in line location in *thinned* coordinates
        burnin_x = 0
        if show_burnin_line and discard > 0:
            burnin_x = 0  # after discard, the shown chain starts at post-burnin
            # so the "burn-in" boundary is off-plot; instead show original discard in title
            # (more honest for the displayed trace).

        for i, p in enumerate(idx):
            ax = axes[i]

            # walker traces (very light)
            for w in range(w_plot):
                ax.plot(x, chain_plot[w, :, p], alpha=0.18, linewidth=0.6)

            # median + 16–84 band across ALL walkers (post discard/thin)
            y_all = chain[:, :, p]  # (W, T)
            med = np.median(y_all, axis=0)
            q16, q84 = np.quantile(y_all, [0.16, 0.84], axis=0)
            ax.fill_between(x, q16, q84, alpha=0.12)
            ax.plot(x, med, color="black", linewidth=1.1, zorder=5)

            # label
            label = rf"$x_{{{p}}}$" if use_x_labels else rf"$\theta_{{{p}}}$"
            ax.set_ylabel(label, fontsize=9)

            # robust y-limits
            qlo, qhi = ylim_quantiles
            ylo, yhi = np.quantile(chain_plot[:, :, p], [qlo, qhi])
            if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
                pad = float(ylim_pad_frac) * (yhi - ylo)
                ax.set_ylim(ylo - pad, yhi + pad)

            ax.margins(x=0.01)

        # hide unused axes
        for j in range(n_panels, len(axes)):
            axes[j].set_visible(False)

        # x-label only bottom visible row
        for ax in axes[:n_panels]:
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("MCMC step", fontsize=9)

        # layout
        fig.subplots_adjust(
            left=0.09, right=0.99, bottom=0.10, top=0.92, wspace=0.45, hspace=0.35
        )
        fig.align_ylabels(axes[:n_panels])

        if save_pdf:
            os.makedirs(save_folder, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(save_folder, f"{fname_prefix}_{ts}.pdf")
            fig.savefig(out, format="pdf")
            print(f"Saved: {out}")

        plt.show()

    def plot_log_likelihood(
        self,
        save_folder: str = "results",
        save_pdf: bool = True,
        fname_prefix: str = "logposterior_trace",
        dpi: int = 300,
        alpha: float = 0.35,
        lw: float = 0.8,
        max_walkers_to_plot: int | None = None,
        thin: int = 1,
        discard: int = 0,
        show_summary: bool = True,
    ):
        if getattr(self, "log_probs", None) is None:
            print("Run MCMC first.")
            return
        if getattr(self, "sampler", None) is None or not hasattr(
            self.sampler, "nwalkers"
        ):
            raise AttributeError("self.sampler with attribute `nwalkers` is required.")

        log_probs = np.asarray(self.log_probs).ravel()
        n_walkers = int(self.sampler.nwalkers)

        if log_probs.size % n_walkers != 0:
            raise ValueError(
                f"log_probs length ({log_probs.size}) is not divisible by n_walkers ({n_walkers}). "
                "Cannot reshape safely."
            )

        n_steps = log_probs.size // n_walkers

        # emcee-style: flatten is step-major -> (n_steps, n_walkers) then transpose
        chain = log_probs.reshape(n_steps, n_walkers).T  # (n_walkers, n_steps)

        discard = max(0, int(discard))
        thin = max(1, int(thin))
        chain = chain[:, discard:]
        chain = chain[:, ::thin]

        finite = np.isfinite(chain)
        if not np.any(finite):
            raise ValueError("All log_probs are non-finite (NaN/Inf).")

        y = chain

        # optionally limit number of walkers drawn
        plot_walkers = (
            n_walkers
            if max_walkers_to_plot is None
            else min(n_walkers, int(max_walkers_to_plot))
        )
        y_plot = y[:plot_walkers, :]
        x = np.arange(y_plot.shape[1])
        fig, ax = plt.subplots(figsize=(10, 5))

        # walker traces
        for i in range(y_plot.shape[0]):
            ax.plot(x, y_plot[i], alpha=alpha, linewidth=lw)

        # Summary overlay: median
        if show_summary and y.shape[0] > 1:
            med = np.nanmedian(y, axis=0)
            ax.plot(x, med, linewidth=1.6, color="black", label="Median (all walkers)")

        ax.set_xlabel("MCMC step [-]")
        ax.set_ylabel(r"$\log \mathcal{P}(x \mid y)$")

        # Robust y-limits for raw log-posterior values
        ylo, yhi = np.nanquantile(y_plot[np.isfinite(y_plot)], [0.01, 0.99])
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            pad = 0.08 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)

        ax.margins(x=0.01)
        if show_summary and y.shape[0] > 1:
            ax.legend(loc="upper right", frameon=True)

        fig.tight_layout()

        if save_pdf:
            os.makedirs(save_folder, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(save_folder, f"{fname_prefix}_{ts}.pdf")
            fig.savefig(out, bbox_inches="tight", format="pdf")
            print(f"Saved: {out}")

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
        plt.ylabel(r"Residual Normalized [$\sigma$]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_postfit_residuals_time(self, t_obs_used, opnav_data=False):
        if getattr(self, "samples", None) is None:
            print("Run estimation first.")
            return

        # Compute MAP residuals
        best_params = self.get_map_estimate()
        if best_params is None:
            raise RuntimeError("MAP estimate not available.")

        postfit = np.asarray(self.residuals_func(best_params)).ravel()

        if postfit.size % 2 != 0:
            raise ValueError("Residual vector must be even-length (paired residuals).")

        # Reshape [r0, rr0, r1, rr1, ...] -> (2, N)
        residuals_matrix = postfit.reshape(-1, 2).T  # (2, N)

        time_hr = np.asarray(t_obs_used).ravel() / 3600.0
        if residuals_matrix.shape[1] != time_hr.size:
            raise ValueError("Time vector length does not match residual count.")

        # Remove zero residuals
        mask0 = residuals_matrix[0] != 0.0
        mask1 = residuals_matrix[1] != 0.0

        r0, t0 = residuals_matrix[0, mask0], time_hr[mask0]
        r1, t1 = residuals_matrix[1, mask1], time_hr[mask1]

        # Labels (explicitly whitened)
        ylabels = (
            [r"Whitened RA Residual [$\sigma$]", r"Whitened DEC Residual [$\sigma$]"]
            if opnav_data
            else [
                r"Whitened Range Residual [$\sigma$]",
                r"Whitened Range-Rate Residual [$\sigma$]",
            ]
        )

        # Figure
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        # Top panel
        ax0.scatter(t0, r0, s=50, alpha=0.85)
        ax0.axhline(3, color="red", linestyle=":", linewidth=3.0)
        ax0.axhline(-3, color="red", linestyle=":", linewidth=3.0)
        ax0.set_ylabel(ylabels[0])
        ax0.grid(True, linestyle=":")

        if r0.size > 0:
            lim = np.max(np.abs(np.quantile(r0, [0.01, 0.99])))
            lim = max(lim, 3.2)
            ax0.set_ylim(-1.1 * lim, 1.1 * lim)

        # Bottom panel
        ax1.scatter(t1, r1, s=50, alpha=0.85)
        ax1.axhline(3, color="red", linestyle=":", linewidth=3.0)
        ax1.axhline(-3, color="red", linestyle=":", linewidth=3.0)
        ax1.set_ylabel(ylabels[1])
        ax1.set_xlabel("Time since epoch [hours]")
        ax1.grid(True, linestyle=":")

        if r1.size > 0:
            lim = np.max(np.abs(np.quantile(r1, [0.01, 0.99])))
            lim = max(lim, 3.2)
            ax1.set_ylim(-1.1 * lim, 1.1 * lim)

        # Single clean legend (shared)
        legend_items = [
            Line2D(
                [0],
                [0],
                color="red",
                linestyle=":",
                linewidth=3.2,
                label=r"$\pm 3\sigma$",
            ),
        ]

        ax0.legend(
            handles=legend_items,
            loc="upper right",
            frameon=True,
        )

        fig.tight_layout()

        # Save vector PDF
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"results/postfit_residuals_{timestamp}.pdf"
        fig.savefig(fname, format="pdf", bbox_inches="tight")
        print(f"Saved: {fname}")

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
        batch_sigma_levels=(3.0,),
        idx=None,  # list/array of parameter indices to plot
        max_dims=20,
        bins=40,
        quantile_range=(0.005, 0.995),
        title_digits=3,
        legend_outside=True,
        rotation_matrix=None,
    ):
        if self.samples is None:
            print("Run MCMC first.")
            return

        samples = np.asarray(self.samples)
        if samples.ndim != 2 or samples.shape[0] < 2:
            raise ValueError("self.samples must be a 2D array with at least 2 rows.")

        # ----------------------------
        # Choose dimensions to plot
        # ----------------------------
        ndim = samples.shape[1]
        if idx is None:
            idx = np.arange(min(ndim, max_dims))
        else:
            idx = np.asarray(idx, dtype=int)

        s = samples[:, idx]
        d = s.shape[1]

        # ----------------------------
        # Truths & MAP on same subspace
        # ----------------------------
        best_params_full = self.get_map_estimate()
        best_params = (
            np.asarray(best_params_full)[idx] if best_params_full is not None else None
        )

        truths_full = np.median(samples, axis=0) if use_median_as_truth else true_theta
        truths = np.asarray(truths_full)[idx] if truths_full is not None else None

        # ----------------------------
        # Labels: x_i
        # ----------------------------
        labels = (
            [rf"$z_{{{k}}}$" for k in idx]
            if rotation_matrix is not None
            else [rf"$x_{{{k}}}$" for k in idx]
        )

        # ----------------------------
        # Batch / True on same subspace
        # ----------------------------
        bm = np.asarray(batch_mean)[idx] if batch_mean is not None else None

        bc = None
        if batch_cov is not None:
            batch_cov = np.asarray(batch_cov)
            bc = batch_cov[np.ix_(idx, idx)]
            if bc.shape != (d, d):
                raise ValueError("batch_cov shape is inconsistent with selected idx.")

        tt = np.asarray(true_theta)[idx] if true_theta is not None else None

        # ----------------------------
        # Rotate Everything if requested
        # ----------------------------
        if rotation_matrix is not None:
            R = np.asarray(rotation_matrix)
            if R.shape != (d, d):
                raise ValueError(
                    f"rotation_matrix must have shape {(d, d)}, got {R.shape}"
                )
            s = s @ R
            if best_params is not None:
                best_params = best_params @ R
            if truths is not None:
                truths = truths @ R
            if bm is not None:
                bm = bm @ R
            if batch_cov is not None:
                bc = R.T @ bc @ R
            if tt is not None:
                tt = tt @ R

        # ----------------------------
        # Robust range + ensure EVERYTHING is visible
        # ----------------------------
        qlo, qhi = quantile_range
        ranges = []
        ksig_max = float(np.max(batch_sigma_levels)) if len(batch_sigma_levels) else 0.0

        for j in range(d):
            lo, hi = np.quantile(s[:, j], [qlo, qhi])
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                lo, hi = float(np.min(s[:, j])), float(np.max(s[:, j]))

            extras = []
            if bm is not None and np.isfinite(bm[j]):
                extras.append(float(bm[j]))
            if tt is not None and np.isfinite(tt[j]):
                extras.append(float(tt[j]))
            if truths is not None and np.isfinite(truths[j]):
                extras.append(float(truths[j]))
            if best_params is not None and np.isfinite(best_params[j]):
                extras.append(float(best_params[j]))

            if bc is not None and np.isfinite(bc[j, j]) and bc[j, j] >= 0:
                sig = ksig_max * float(np.sqrt(bc[j, j]))
                if bm is not None and np.isfinite(bm[j]) and sig > 0:
                    extras.extend([float(bm[j] - sig), float(bm[j] + sig)])

            if extras:
                lo = min(lo, np.min(extras))
                hi = max(hi, np.max(extras))

            span = hi - lo
            if not np.isfinite(span) or span <= 0:
                span = 1.0
            pad = 0.03 * span
            ranges.append((lo - pad, hi + pad))

        # ----------------------------
        # Figure size scaling with d
        # ----------------------------
        inches_per_dim = 1.35
        fig_size = max(8.0, inches_per_dim * d)
        # levels_2d_sigma = [0.393, 0.865, 0.989]  # 1σ,2σ,3σ in 2D (Gaussian-equivalent)

        fig = corner.corner(
            s,
            labels=labels,
            # truths=truths,
            # truth_color="green",  # Change truth lines to green instead of blue
            range=ranges,
            bins=bins,
            show_titles=True,
            title_fmt=f".{title_digits}g",
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 11},
            color="tab:blue",
            plot_contours=True,
            fill_contours=False,
            # levels=levels_2d_sigma,
            smooth=1.0,
            smooth1d=1.0,
        )

        axes = np.array(fig.axes).reshape((d, d))

        # ----------------------------
        # Titles: position below the top edge to avoid overlap with plot above
        # ----------------------------
        for k in range(d):
            axes[k, k].title.set_y(1.15)  # reduced from 1.30 - rely on hspace instead
            axes[k, k].title.set_fontsize(9)

        # ----------------------------
        # Overlay batch ellipses / points
        # ----------------------------
        for i in range(d):
            for j in range(i):
                ax = axes[i, j]

                if bm is not None:
                    ax.plot(bm[j], bm[i], "o", ms=4.5, color="red", zorder=6)

                if bm is not None and bc is not None and np.all(np.isfinite(bc)):
                    cov_sub = bc[np.ix_([j, i], [j, i])]
                    mean_sub = [bm[j], bm[i]]

                    cov_sub = 0.5 * (cov_sub + cov_sub.T)
                    vals, vecs = np.linalg.eigh(cov_sub)
                    vals = np.maximum(vals, 0.0)

                    if np.max(vals) > 0:
                        order = np.argsort(vals)[::-1]
                        vals, vecs = vals[order], vecs[:, order]
                        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

                        for ksig in batch_sigma_levels:
                            width, height = 2.0 * float(ksig) * np.sqrt(vals)
                            ell = Ellipse(
                                xy=mean_sub,
                                width=width,
                                height=height,
                                angle=angle,
                                edgecolor="red",
                                facecolor="none",
                                lw=1.4 if float(ksig) == 3.0 else 1.0,
                                linestyle="-" if float(ksig) == 3.0 else "--",
                                alpha=0.95,
                                zorder=5,
                            )
                            ax.add_patch(ell)

                if tt is not None:
                    ax.plot(tt[j], tt[i], "o", ms=4.5, color="purple", zorder=7)

                if best_params is not None:
                    ax.plot(
                        best_params[j],
                        best_params[i],
                        "x",
                        ms=6,
                        mew=1.6,
                        color="black",
                        zorder=8,
                    )

                if plot_contours_labels:
                    x = s[:, j]
                    y = s[:, i]
                    data = np.vstack([x, y])
                    kde = gaussian_kde(data)

                    xi, yi = np.mgrid[
                        ranges[j][0] : ranges[j][1] : 100j,
                        ranges[i][0] : ranges[i][1] : 100j,
                    ]
                    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

                    levels_mass = [0.118, 0.393, 0.675, 0.864]
                    sorted_zi = np.sort(zi.ravel())[::-1]
                    cdf = np.cumsum(sorted_zi)
                    cdf /= cdf[-1]
                    value_levels = []
                    for lv in levels_mass:
                        k0 = int(np.searchsorted(cdf, lv))
                        k0 = np.clip(k0, 0, len(sorted_zi) - 1)
                        value_levels.append(sorted_zi[k0])
                    value_levels = sorted(set(value_levels))

                    cs = ax.contour(
                        xi, yi, zi, levels=value_levels, colors="black", linewidths=1
                    )
                    fmt = {
                        lvl: f"{int(100*m)}\\%"
                        for lvl, m in zip(cs.levels, levels_mass[: len(cs.levels)])
                    }
                    ax.clabel(cs, fmt=fmt, inline=True, fontsize=8)

        # ----------------------------
        # Global formatting:
        # Increase spacing to prevent overlaps
        # ----------------------------

        for ax in fig.get_axes():
            # Increase tick label size and padding significantly
            ax.tick_params(labelsize=8, pad=8.0)  # tick numbers away from axis

        # ----------------------------
        # Legend proxies
        # ----------------------------
        proxies = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="red",
                markersize=6,
                label="Batch Mean",
            ),
            Line2D(
                [0],
                [0],
                color="red",
                lw=1.4,
                linestyle="-",
                label=rf"Batch {float(batch_sigma_levels[0]):g}$\sigma$",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="purple",
                markersize=6,
                label="True Value",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="black",
                markersize=7,
                lw=0,
                label="MCMC MAP",
            ),
        ]

        if len(batch_sigma_levels) > 1:
            for ksig in batch_sigma_levels[1:]:
                proxies.insert(
                    2,
                    Line2D(
                        [0],
                        [0],
                        color="red",
                        lw=1.0,
                        linestyle="--",
                        label=rf"Batch {float(ksig):g}$\sigma$",
                    ),
                )

        filtered = []
        for p in proxies:
            lab = p.get_label()
            if "Batch" in lab:
                if bm is not None:
                    filtered.append(p)
            elif lab == "True Value":
                if tt is not None:
                    filtered.append(p)
            elif lab == "MCMC MAP":
                if best_params is not None:
                    filtered.append(p)
            else:
                filtered.append(p)

        # More breathing room with significantly increased spacing between plots
        fig.set_size_inches(fig_size, fig_size)
        fig.subplots_adjust(
            wspace=0.20, hspace=0.30, right=0.82, top=0.93, bottom=0.12, left=0.12
        )  # margins back to 0.12

        # Force a draw so text objects have final positions/sizes
        fig.canvas.draw()

        # Move ONLY the labels that exist in a corner plot:
        #   x-labels: bottom row
        #   y-labels: left column
        x_y = -0.70  # more negative -> farther down
        y_x = -0.70  # more negative -> farther left

        for j in range(d):
            ax = axes[d - 1, j]  # bottom row
            if ax.get_xlabel():
                ax.xaxis.set_label_coords(0.5, x_y)  # (x in [0,1], y in axes coords)

        for i in range(d):
            ax = axes[i, 0]  # left column
            if ax.get_ylabel():
                ax.yaxis.set_label_coords(y_x, 0.5)

        # Legend in upper-right empty triangle
        if legend_outside and len(filtered) > 0:
            leg_ax = axes[0, d - 1]
            leg_ax.axis("off")
            leg_ax.legend(
                handles=filtered,
                loc="center",
                frameon=True,
                fontsize=13,
                handlelength=2.6,
                labelspacing=0.8,
                borderpad=0.9,
            )

        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"results/corner_{ts}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.25)
        print(f"Saved: {out}")

        plt.show()

    def plot_marginals_overlaid(
        self,
        idx=None,
        max_dims=20,
        bins=80,
        quantile_range=(0.005, 0.995),
        rotation_matrix=None,
        density=True,
        log_y=False,
        alpha=0.28,
        linewidth=1.5,
        show_kde=False,
        figsize=(10, 6),
        labels=None,
        center=False,
    ):
        if self.samples is None:
            print("Run MCMC first.")
            return

        samples = np.asarray(self.samples)
        if samples.ndim != 2 or samples.shape[0] < 2:
            raise ValueError("self.samples must be a 2D array with at least 2 rows.")

        # ----------------------------
        # Choose dimensions
        # ----------------------------
        ndim = samples.shape[1]
        if idx is None:
            idx = np.arange(min(ndim, max_dims))
        else:
            idx = np.asarray(idx, dtype=int)

        s = samples[:, idx]
        d = s.shape[1]

        # ----------------------------
        # Labels
        # ----------------------------
        if labels is None:
            labels = (
                [rf"$z_{{{k}}}$" for k in idx]
                if rotation_matrix is not None
                else [rf"$x_{{{k}}}$" for k in idx]
            )

        # ----------------------------
        # Rotate if requested
        # ----------------------------
        if rotation_matrix is not None:
            R = np.asarray(rotation_matrix)
            if R.shape != (d, d):
                raise ValueError(
                    f"rotation_matrix must have shape {(d, d)}, got {R.shape}"
                )
            s = s @ R

        # ----------------------------
        # Global x-range from all marginals
        # ----------------------------
        qlo, qhi = quantile_range
        lo_all = []
        hi_all = []

        for j in range(d):
            x = s[:, j]

            if center:
                mu = np.mean(x)
                x = x - mu

            lo, hi = np.quantile(x, [qlo, qhi])
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                lo, hi = float(np.min(s[:, j])), float(np.max(s[:, j]))
            lo_all.append(lo)
            hi_all.append(hi)

        xmin = min(lo_all)
        xmax = max(hi_all)
        span = xmax - xmin
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        pad = 0.05 * span
        xmin -= pad
        xmax += pad

        # ----------------------------
        # Plot
        # ----------------------------
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.tab20(np.linspace(0, 1, d))

        for j in range(d):
            x = s[:, j]

            if center:
                mu = np.mean(x)
                x = x - mu

            ax.hist(
                x,
                bins=bins,
                range=(xmin, xmax),
                density=density,
                histtype="stepfilled",
                alpha=alpha,
                linewidth=linewidth,
                label=labels[j],
                color=colors[j],
            )

            if show_kde and np.std(x) > 0:
                try:
                    kde = gaussian_kde(x)
                    xx = np.linspace(xmin, xmax, 500)
                    yy = kde(xx)
                    ax.plot(xx, yy, linewidth=1.5)
                except Exception:
                    pass

        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Centered Value" if center else "Parameter Value", fontsize=11)
        ax.set_ylabel("Density" if density else "Count", fontsize=11)

        if log_y:
            ax.set_yscale("log")

        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, ncol=2, frameon=True)

        fig.tight_layout()

        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"results/marginals_overlaid_{ts}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.2)
        print(f"Saved: {out}")

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
                "Acceptance rate outside optimal range (0.2-0.5). Consider tuning initialization step size and stretch_a."
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
        self.whiten_mean = np.array([p.mean() for p in self.param_priors], dtype=float)
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

    def plot_autocorrelation(
        self,
        idx=None,  # parameters to include (default: all, capped)
        max_dims: int = 20,  # max number of parameters shown
        max_lag: int = 500,
        thin: int = 1,
        discard: int = 0,
        save_folder: str = "results",
        save_pdf: bool = True,
        fname_prefix: str = "acf",
    ):

        if getattr(self, "samples", None) is None:
            print("Run MCMC first.")
            return
        if getattr(self, "sampler", None) is None:
            raise AttributeError("self.sampler required.")

        samples = np.asarray(self.samples)
        n_walkers = int(self.sampler.nwalkers)
        ndim = samples.shape[1]

        if samples.shape[0] % n_walkers != 0:
            raise ValueError("Samples cannot be reshaped consistently.")

        n_steps = samples.shape[0] // n_walkers

        # Correct reshape (emcee step-major flattening)
        chain = samples.reshape(n_steps, n_walkers, ndim).transpose(1, 0, 2)

        discard = max(0, int(discard))
        thin = max(1, int(thin))
        chain = chain[:, discard:, :]
        chain = chain[:, ::thin, :]

        # Cap max_lag to available steps
        max_lag = min(int(max_lag), chain.shape[1] - 1)
        lags = np.arange(max_lag + 1)

        # Parameter selection
        if idx is None:
            idx = np.arange(min(ndim, max_dims))
        else:
            idx = np.asarray(idx, dtype=int)
            if idx.size > max_dims:
                idx = idx[:max_dims]

        fig, ax = plt.subplots(figsize=(6.8, 3.8))

        # Plot ACF curves + tau lines
        for p in idx:
            acf_vals = []
            for w in range(chain.shape[0]):
                acf_vals.append(acf(chain[w, :, p], nlags=max_lag, fft=True))
            acf_mean = np.mean(acf_vals, axis=0)
            ax.plot(lags, acf_mean, linewidth=1.6, label=rf"$x_{{{p+1}}}$")

        ax.set_xlabel("Lag [-]")
        ax.set_ylabel("Autocorrelation [-]")
        ax.set_xlim(0, max_lag)
        ax.set_ylim(-0.05, 1.05)

        # legend: keep parameter curves + ONE entry describing tau lines
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right", frameon=True)
        fig.tight_layout()

        if save_pdf:
            os.makedirs(save_folder, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(save_folder, f"{fname_prefix}_{ts}.pdf")
            fig.savefig(out, bbox_inches="tight", format="pdf")
            print(f"Saved: {out}")

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
