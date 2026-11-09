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
import io
import time
import shutil
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import logsumexp
from scipy.stats import chi2, multivariate_normal, norm
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


class ChainReplay:
    """
    Stand-in for the emcee sampler, rebuilt from a chain saved by MCMCModel.save_chain:
    get_chain / get_log_prob / get_autocorr_time / acceptance_fraction / nwalkers /
    iteration as emcee has them, so the diagnostics and journal figures run unchanged.

    The run is stored every `stride` steps (stored index i = step (i + 1) * stride - 1, as
    emcee's get_chain(thin=stride) returns it). discard / thin are in steps of the original
    run: thin is rounded to a multiple of the stride, so pass multiples of it for exact
    results (plot_mcmc_traces and plot_mcmc_autocorrelation do).
    """

    def __init__(self, full_chain, full_log_prob, stride, n_steps, tau=None,
                 acceptance_fraction=None):
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
        k = max(1, int(round(thin / self.stride)))
        i0 = max(0, int(np.ceil((discard + k * self.stride) / self.stride)) - 1)
        out = arr[i0::k]
        return out.reshape((-1,) + out.shape[2:]) if flat else out

    def get_chain(self, discard=0, thin=1, flat=False):
        return self._select(self.full_chain, discard, thin, flat)

    def get_log_prob(self, discard=0, thin=1, flat=False):
        return self._select(self.full_log_prob, discard, thin, flat)

    def get_autocorr_time(self, tol=50, **kwargs):
        """The tau estimated on the full-resolution run at save time."""
        if self.tau is None:
            raise emcee.autocorr.AutocorrError(np.nan, "no tau stored with the chain")
        return self.tau.copy()


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
        self.chain = None
        self.log_prob_chain = None
        self.tau = None
        self.burn_in = None
        self.thin = None

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

    def save_chain(self, path=None, max_full_steps=4000, **extra):
        """
        Save the run as a compressed .npz (default: chain.npz in the figure folder set by
        set_figure_dir), enough for load_chain to redo every diagnostic and journal figure
        without sampling again. Call it right after run(), before any in-place edit of
        self.samples, so that a loaded file reproduces the state run() leaves.

        Stored:
          samples, log_probs: the flat posterior as run() leaves it (physical coordinates);
          chain, log_prob_chain: post-burn-in thinned chain in SAMPLED (whitened, if
            whitening is on) coordinates, (n_steps, n_walkers, ndim) / (n_steps, n_walkers);
          full_chain (float32), full_log_prob: the whole run, burn-in included, every
            full_stride steps (for the trace and autocorrelation figures). full_stride keeps
            it to <= max_full_steps steps and <= min(tau) / 10 (max_full_steps=None: every
            step);
          n_steps, full_stride, acceptance_fraction, tau, burn_in, thin, whiten_mean/_L;
          extra: further arrays (truth, batch mean/cov, ...), stored as extra_<key>.
        The bytes are written with retries (OneDrive), else to FALLBACK_DIR. Returns the path.
        """
        if self.samples is None:
            raise ValueError("No samples to save: run the sampler first.")
        path = Path(_FIG_DIR / "chain.npz" if path is None else path)
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
        if not _write_bytes_retry(path, buf.getvalue()):
            path = FALLBACK_DIR / path.parent.name / path.name
            _write_bytes_retry(path, buf.getvalue(), tries=2)
            print("[Chain] WARNING: could not write the chain folder; saved locally instead")
        print(f"[Chain] saved {path} ({buf.tell() / 2**20:.1f} MB)")
        return path

    def load_chain(self, path):
        """
        Restore a run saved by save_chain: samples, chain, tau, burn_in, thin, whitening, and
        a ChainReplay in place of the emcee sampler (same interface), so the diagnostics and
        journal figures run as after run(). Returns the extra arrays as a dict (keys without
        the extra_ prefix).
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

        total_samples = int(self.sampler.iteration) * int(self.sampler.nwalkers)
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

    # ------------------------------------------------------------------
    # Journal figures (module-level functions in the JOURNAL FIGURES section below)
    # ------------------------------------------------------------------
    # Okabe-Ito colors, LaTeX text, printed size; one vector PDF per figure in the folder set by
    # set_figure_dir(). Call configure(...) first: it holds the parameter symbols, units and
    # display scales and applies the journal style. The plot_* methods above are unchanged
    # (older scripts in Tests/ and Extra/ still use them).

    def plot_journal_residuals(
        self,
        t_obs,
        vis_mask,
        theta_batch,
        theta_mcmc=None,
        components=None,
        name="fig_postfit_residuals",
    ):
        """
        Normalized post-fit residuals (residuals_func) of the batch estimate and of the MCMC
        MAP (default) at the visible epochs. theta_* are in sampled coordinates.
        components: measurement components shown (indices into RESID_LABELS).
        """
        if theta_mcmc is None:
            theta_mcmc = self.get_map_estimate()
        return plot_postfit_residuals(
            t_obs,
            self.residuals_func(np.asarray(theta_batch, dtype=float)),
            self.residuals_func(np.asarray(theta_mcmc, dtype=float)),
            vis_mask,
            components=components,
            name=name,
        )

    def plot_journal_traces(self, burn_in, true_theta=None, batch_theta=None, **kwargs):
        """Walker traces + ensemble mean +/-3 sigma band + log p (see plot_mcmc_traces)."""
        return plot_mcmc_traces(
            self, burn_in, true_theta=true_theta, batch_theta=batch_theta, **kwargs
        )

    def plot_journal_autocorrelation(self, burn_in, thin, **kwargs):
        """Walker-averaged ACF and integrated autocorrelation times after burn-in."""
        return plot_mcmc_autocorrelation(self, burn_in, thin, **kwargs)

    def plot_journal_corner(
        self, true_theta=None, batch_mean=None, batch_cov=None, transform=None, **kwargs
    ):
        """
        Corner plot of self.samples (posterior HDRs vs batch covariance ellipses vs truth).
        transform: maps sampled -> displayed parameters (e.g. a hierarchical parametrization);
        true_theta / batch_* are then in displayed coordinates.
        """
        samples = np.asarray(self.samples, dtype=float)
        if transform is not None:
            samples = np.asarray(transform(samples), dtype=float)
        return plot_corner(
            samples,
            true_theta=true_theta,
            batch_mean=batch_mean,
            batch_cov=batch_cov,
            **kwargs,
        )

    def plot_journal_diagnostics(
        self,
        burn_in,
        thin,
        t_obs,
        vis_mask,
        true_theta=None,
        batch_theta=None,
        batch_cov=None,
        components=None,
        acf_groups=None,
    ):
        """
        The four MCMC journal figures (residuals, traces, autocorrelation, corner) when the
        sampled parameters are the displayed ones. batch_theta defaults to zeros, i.e. deltas
        about the batch solution. A figure that fails is reported and skipped, so a long run
        is never lost to a plotting error.
        """
        batch = np.zeros(self.ndim) if batch_theta is None else np.asarray(batch_theta)
        figures = (
            (
                "post-fit residuals",
                lambda: self.plot_journal_residuals(
                    t_obs, vis_mask, batch, components=components
                ),
            ),
            (
                "traces",
                lambda: self.plot_journal_traces(
                    burn_in, true_theta=true_theta, batch_theta=batch
                ),
            ),
            (
                "autocorrelation",
                lambda: self.plot_journal_autocorrelation(
                    burn_in, thin, groups=acf_groups
                ),
            ),
            (
                "corner",
                lambda: self.plot_journal_corner(
                    true_theta=true_theta, batch_mean=batch, batch_cov=batch_cov
                ),
            ),
        )
        for label, draw in figures:
            try:
                draw()
            except Exception as err:
                print(f"[Figure] {label} skipped: {type(err).__name__}: {err}")

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


# ======================================================================
# JOURNAL FIGURES
# ======================================================================
# Style, colors and plot functions of the scenario scripts (Bennu SPH2/SPH3 particle and popper
# runs, Ryugu POLY particle run), for any number of estimated parameters, RA/Dec or
# range/range-rate measurements and any central body. Every figure is one vector PDF in the
# figure folder (see save_figure).
#
# Usage:
#
#     import MCMC
#     MCMC.configure(param_sym=..., param_unit=..., param_scale=..., body="Bennu",
#                    rotation=make_bennu_rotation_matrix)
#     MCMC.set_figure_dir("results/<run>_<timestamp>")
#     MCMC.plot_scene_body_fixed(...); model.plot_journal_diagnostics(...); ...

# Optional dependency (the heyoka conda env lacks it): scikit-learn for the GMM split.
# Without it a numpy EM is used.
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None


def _patch_type1_subset():
    """
    matplotlib 3.11 subsets the TeX Type-1 fonts it embeds in PDFs, and Type1Font.subset()
    overwrites encoding slot 0 with .notdef: the minus sign (cmsy slot 0) and \\Gamma (cmr
    slot 0) vanish from every usetex PDF. When a font really uses slot 0, embed it whole, as
    matplotlib <= 3.10 always did; every other font is still subset.
    """
    try:
        from matplotlib._type1font import Type1Font
    except ImportError:
        return
    orig = getattr(Type1Font, "subset", None)
    if orig is None or getattr(orig, "_jf_keeps_slot0", False):
        return

    def subset(self, characters, name_prefix):
        characters = frozenset(characters)
        if 0 in characters and self.prop["Encoding"].get(0, ".notdef") != ".notdef":
            return self
        return orig(self, characters, name_prefix)

    subset._jf_keeps_slot0 = True
    Type1Font.subset = subset


_patch_type1_subset()

# ============================================================
# FIGURE STYLE (journal)
# ============================================================
# Figures are drawn at their printed size (single column COL_W, full text width TXT_W) with
# LaTeX text at 7-8 pt, so they drop into a two-column journal layout without rescaling.
# Vector PDFs; only dense layers (mesh, point clouds, walker traces) are rasterized.
# The style is applied by configure(), so importing this module leaves the defaults above
# untouched for scripts that do not draw journal figures.

COL_W = 3.5  # single-column width [in]
TXT_W = 7.16  # full text width (double column) [in]
SHOW_FIGS = True  # plt.show() after each figure is saved

def _ensure_tex_on_path():
    """
    True if latex + dvipng are reachable. GUI-launched processes (VS Code, Jupyter, Spyder on
    macOS) often do not inherit the login-shell PATH, so the standard TeX locations are added
    before giving up; otherwise matplotlib would silently fall back to mathtext and DejaVu.
    """
    if shutil.which("latex") and shutil.which("dvipng"):
        return True
    import glob

    extra = ["/Library/TeX/texbin", "/opt/homebrew/bin", "/usr/local/bin", "/usr/texbin"]
    extra += sorted(glob.glob("/usr/local/texlive/*/bin/*"), reverse=True)
    path = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(path + [d for d in extra if os.path.isdir(d)])
    return bool(shutil.which("latex") and shutil.which("dvipng"))


USE_TEX = _ensure_tex_on_path()
if not USE_TEX:
    print("[figures] latex/dvipng not found: falling back to mathtext (Computer Modern)")
EN_DASH = "--" if USE_TEX else "-"  # cmr10 (mathtext fallback) has no en-dash glyph
# Every journal label renders identically in both modes: math lives inside $...$,
# percent signs are written $\%$, and only commands mathtext also knows are used.
STYLE = dict(
    {
        "text.usetex": USE_TEX,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathrsfs}",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "0.15",
        "axes.labelcolor": "0.1",
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def apply_style():
    """Set the journal rcParams. configure() calls it before any journal figure is drawn."""
    plt.rcParams.update(STYLE)

# Okabe-Ito colors (colorblind-safe; adjacent pairs checked for CVD separation).
# Each color has ONE meaning in every figure.
C_POST = "#0072B2"  # MCMC posterior / Gaussian mixture (blue)
C_BATCH = "#D55E00"  # Gauss-Newton batch / single Gaussian (vermillion)
C_SC = "#009E73"  # spacecraft, images used (bluish green)
C_OCC = "#E69F00"  # occulted epochs (orange)
C_MC = "0.62"  # nonlinear Monte Carlo (neutral grey)
C_TRUTH = "k"  # truth
C_SPAN = "0.93"  # background spans (observation arc, burn-in, +/-3 sigma)
C_XYZ = ("#56B4E9", "#CC79A7", "#E69F00")  # x / y / z components (ACF figure only)
C_PRIOR = "#CC79A7"  # hyperprior reference (corner + traces; dashed, reddish purple)

# ------------------------------------------------------------
# Per-script configuration (set by configure())
# ------------------------------------------------------------
PARAM_SYM = []  # LaTeX symbol of each estimated parameter (no $)
PARAM_UNIT = []  # display unit ("" = dimensionless, no brackets)
PARAM_SCALE = np.ones(0)  # internal units -> display units
BODY = "Bennu"  # central body (occultation labels)
TARGET = "Particle"  # tracked object (legend / axis labels)
OBS_WORD = "Image"  # one measurement epoch: "Image Used", "Image Occulted"
SITE_LABEL = "Ejection Site"  # marker of the detachment point in the scene
RESID_LABELS = ("RA", "Dec")  # measurement components (stacked pairs in the residuals)
_ROTATION = None  # R_ib(alpha, delta, omega, t, w0): inertial -> body-fixed
FALLBACK_DIR = Path.home() / "figures_fallback"  # used if the figure folder cannot be written


def configure(
    param_sym,
    param_unit,
    param_scale,
    rotation=None,
    body=None,
    target=None,
    obs_word=None,
    site_label=None,
    resid_labels=None,
    fallback_dir=None,
):
    """
    Set the parameter symbols / units / scales and the scenario names used in the labels,
    and (re)apply the journal rcParams.
    """
    global PARAM_SYM, PARAM_UNIT, PARAM_SCALE, BODY, TARGET, OBS_WORD, SITE_LABEL
    global RESID_LABELS, _ROTATION, FALLBACK_DIR
    PARAM_SYM = list(param_sym)
    PARAM_UNIT = list(param_unit)
    PARAM_SCALE = np.asarray(param_scale, dtype=float)
    if not len(PARAM_SYM) == len(PARAM_UNIT) == PARAM_SCALE.size:
        raise ValueError("param_sym, param_unit and param_scale must have the same length")
    if rotation is not None:
        _ROTATION = rotation
    if body is not None:
        BODY = body
    if target is not None:
        TARGET = target
    if obs_word is not None:
        OBS_WORD = obs_word
    if site_label is not None:
        SITE_LABEL = site_label
    if resid_labels is not None:
        RESID_LABELS = tuple(resid_labels)
    if fallback_dir is not None:
        FALLBACK_DIR = Path(fallback_dir)
    apply_style()


def _pow10_label(k, unit):
    """Unit label for a quantity displayed in units of 10^k [unit]."""
    if k == 0:
        return unit
    return rf"$10^{{{k}}}$ {unit}".strip()


def auto_param_scales(base_scale, base_unit, sigma, lo=0.05, hi=50.0):
    """
    Display scale and unit per parameter so the (batch) 1-sigma is a readable number.

    base_scale/base_unit: internal -> natural display unit (e.g. km -> m: 1e3, "m").
    sigma: 1-sigma in internal units. Parameters with lo <= sigma*base < hi keep the natural
    unit; the others are shown in units of 10^k (k = floor(log10(sigma*base))), e.g. an
    SH coefficient with sigma ~ 3e-16 is plotted in units of 10^-16.
    Returns (scale, unit).
    """
    base_scale = np.asarray(base_scale, dtype=float)
    sigma = np.abs(np.asarray(sigma, dtype=float))
    scale, unit = [], []
    for b, u, s in zip(base_scale, base_unit, sigma):
        v = s * b
        k = 0 if (not np.isfinite(v) or v == 0.0 or lo <= v < hi) else int(np.floor(np.log10(v)))
        scale.append(b * 10.0 ** (-k))
        unit.append(_pow10_label(k, u))
    return np.array(scale), unit

# ============================================================
# PLOTS
# ============================================================
# Every figure is saved as <name>.pdf in the folder set by set_figure_dir() (each script sets
# results/<run>_<timestamp>/). Colors, units and symbols come from the style block above,
# so the same quantity looks the same in every figure.

_FIG_DIR = Path("results")


def set_figure_dir(path):
    """Folder where save_figure() writes the PDFs (created if needed)."""
    global _FIG_DIR
    _FIG_DIR = Path(path)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    return _FIG_DIR


def _write_bytes_retry(fname, data, tries=6, wait=2.0):
    """Write `data` to `fname`, retrying with backoff (cloud-synced folders can time out)."""
    for k in range(tries):
        try:
            fname.parent.mkdir(parents=True, exist_ok=True)
            tmp = fname.with_name(f".{fname.name}.part")
            tmp.write_bytes(data)
            os.replace(tmp, fname)
            return True
        except OSError as err:
            print(
                f"[Figure] write of {fname.name} failed ({err}); retry {k + 1}/{tries}"
            )
            time.sleep(wait * 2**k)
    return False


def save_figure(fig, name):
    """
    Save `fig` as <figure dir>/<name>.pdf, show it if SHOW_FIGS, then close it.
    The PDF is rendered in memory first and then written with retries, so a slow
    OneDrive write cannot crash the run; if every retry fails it goes to a local folder.
    """
    fname = _FIG_DIR / f"{name}.pdf"
    _fit_span_labels(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    data = buf.getvalue()
    if not _write_bytes_retry(fname, data):
        fname = FALLBACK_DIR / _FIG_DIR.name / f"{name}.pdf"
        _write_bytes_retry(fname, data, tries=2)
        print(f"[Figure] WARNING: could not write to {_FIG_DIR}; saved locally instead")
    print(f"[Figure] saved {fname}")
    interactive = plt.get_backend().lower() not in {
        "agg",
        "pdf",
        "svg",
        "ps",
        "cairo",
        "pgf",
        "template",
    }
    if SHOW_FIGS and interactive:
        plt.show()
    plt.close(fig)
    return fname


def param_label(i, unit=True, sep=" "):
    """Axis label of estimated parameter i, e.g. '$\\delta x_0$ [m]' (sep="\\n": unit below)."""
    s = f"${PARAM_SYM[i]}$"
    return f"{s}{sep}[{PARAM_UNIT[i]}]" if (unit and PARAM_UNIT[i]) else s


def _tint(color, a):
    """Mix `color` with white: a = 0 -> white, a = 1 -> color."""
    c = np.asarray(mcolors.to_rgb(color))
    return tuple((1.0 - a) + a * c)


OCC_FILL = _tint(C_OCC, 0.2)  # opaque fill of the occulted epochs


def _occulted_spans(t, vis):
    """(start, end) of each run of occulted epochs, edges at the midpoints between epochs."""
    t = np.asarray(t, dtype=float)
    occ = ~np.asarray(vis, dtype=bool)
    if not np.any(occ):
        return []
    dt = np.diff(t)
    left = np.r_[t[0] - 0.5 * dt[0], t[1:] - 0.5 * dt]
    right = np.r_[t[:-1] + 0.5 * dt, t[-1] + 0.5 * dt[-1]]
    edges = np.diff(np.r_[0, occ.astype(int), 0])
    i0 = np.flatnonzero(edges == 1)
    i1 = np.flatnonzero(edges == -1) - 1
    return [(left[a], right[b]) for a, b in zip(i0, i1)]


def shade_occulted(ax, t, vis, text=None, text_y=0.97):
    """Shade the occulted epochs of a time axis (optionally labelled inside the first span)."""
    spans = _occulted_spans(t, vis)
    for a, b in spans:
        ax.axvspan(a, b, color=OCC_FILL, lw=0, zorder=0.4)
    if text and spans:
        a, b = spans[0]
        lo, hi = ax.get_xlim()
        label = ax.text(
            0.5 * (max(a, lo) + min(b, hi)),
            text_y,
            text,
            color="0.3",
            fontsize=7,
            ha="center",
            va="top" if text_y > 0.5 else "bottom",
            transform=ax.get_xaxis_transform(),
            zorder=5,
        )
        label._jf_span = (a, b)  # fitted into the span at save time (_fit_span_labels)
    return spans


def _fit_span_labels(fig):
    """
    Fit the span labels of shade_occulted once the layout is final: re-centre each label on
    the visible part of its span, turn it vertical (on one line) when it is wider than the
    span, and drop it when even the vertical label does not fit.
    """
    labels = [
        (ax, t) for ax in fig.axes for t in ax.texts if getattr(t, "_jf_span", None)
    ]
    if not labels:
        return
    fig.draw_without_rendering()
    for ax, t in labels:
        a, b = t._jf_span
        lo, hi = sorted(ax.get_xlim())
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            t.set_visible(False)
            continue
        t.set_x(0.5 * (a + b))
        x0, x1 = ax.transData.transform([[a, 0.0], [b, 0.0]])[:, 0]
        room = 0.9 * abs(x1 - x0)
        if t.get_window_extent().width <= room:
            continue
        t.set_text(t.get_text().replace("\n", " "))
        t.set_rotation(90)
        if t.get_window_extent().width > room:
            t.set_visible(False)


# ------------------------------------------------------------
# Scene (body-fixed frame)
# ------------------------------------------------------------


def _view_dir(elev, azim):
    """Unit vector from the scene toward the camera for ax.view_init(elev, azim)."""
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _shaded_mesh(mesh, view, light, base=(0.74, 0.71, 0.67), ambient=0.30):
    """Lambert-shaded Poly3DCollection of the front-facing triangles of `mesh`."""
    nrm = np.asarray(mesh.face_normals)
    front = nrm @ view > 0.0  # closed surface: back faces are never visible
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[front]]
    shade = ambient + (1.0 - ambient) * np.clip(nrm[front] @ light, 0.0, None)
    fc = np.clip(np.outer(shade, base), 0.0, 1.0)
    coll = Poly3DCollection(
        tri, facecolors=fc, edgecolors=fc, linewidths=0.15, antialiased=False, zorder=1
    )
    coll.set_rasterized(True)
    return coll


def _hidden_by_mesh(mesh, pts, view, eps=1e-6):
    """True where a point is hidden from the (orthographic) camera by the mesh."""
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    dirs = np.repeat(view[None, :], pts.shape[0], axis=0)
    try:
        return np.asarray(mesh.ray.intersects_any(pts + eps * view, dirs), dtype=bool)
    except ImportError:  # no rtree in this environment
        return _ray_hit_counts(mesh.vertices, mesh.faces, pts + eps * view, view) > 0


def _to_body(pos_i, tau, alpha, delta, omega, w0=0.0):
    """Rotate inertial positions (N,3) at times tau into the body-fixed frame."""
    if _ROTATION is None:
        raise RuntimeError("configure(rotation=...) was not called")
    return np.array(
        [
            np.asarray(_ROTATION(alpha, delta, omega, float(t), w0))
            @ p
            for p, t in zip(
                np.asarray(pos_i, dtype=float), np.asarray(tau, dtype=float)
            )
        ]
    )


def _ray_hit_counts(vertices, faces, origins, direction, chunk=128):
    """
    Number of mesh faces hit by the rays origins + t*direction, t > 0 (Moller-Trumbore against
    every face, one shared direction). Numpy fallback for trimesh's ray queries, which need
    rtree (or embree).
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces)
    a = V[F[:, 0]]
    e1 = V[F[:, 1]] - a
    e2 = V[F[:, 2]] - a
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    h = np.cross(d, e2)
    det = np.einsum("fi,fi->f", e1, h)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    pts = np.atleast_2d(np.asarray(origins, dtype=float))
    out = np.zeros(len(pts), dtype=int)
    for i0 in range(0, len(pts), chunk):
        s = pts[i0 : i0 + chunk, None, :] - a[None]  # (P, F, 3)
        u = np.einsum("pfi,fi->pf", s, h) * inv
        q = np.cross(s, e1[None])
        v = (q @ d) * inv
        t = np.einsum("pfi,fi->pf", q, e2) * inv
        hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
        out[i0 : i0 + chunk] = hit.sum(axis=1)
    return out


def _contains_raycast(vertices, faces, pts):
    """Inside test by ray parity, for a watertight mesh (fallback for mesh.contains)."""
    d = np.array([0.5773, 0.5779, 0.5768])  # generic direction: no edge/vertex grazing
    return _ray_hit_counts(vertices, faces, pts, d) % 2 == 1


def points_inside_mesh(mesh, pos_i, tau, alpha, delta, omega, w0=0.0):
    """True where an inertial position [km] at time tau lies inside the body-fixed mesh."""
    pb = _to_body(pos_i, tau, alpha, delta, omega, w0)
    try:
        return np.asarray(mesh.contains(pb), dtype=bool)
    except ImportError:  # no rtree in this environment
        return _contains_raycast(mesh.vertices, mesh.faces, pb)


def plot_scene_body_fixed(
    mesh,
    tau_obs,
    sc_obs,
    vis_mask,
    tau_line,
    sc_line,
    pt_line,
    alpha,
    delta,
    omega,
    w0=0.0,
    detach_bf=None,
    elev=14.0,
    azim=62.0,
    zoom=1.0,
    name="fig_scene_body_fixed",
):
    """
    Observation geometry in the body-fixed frame: shaded shape model, spacecraft and
    tracked object (truth) tracks over the observation arc, and the measurement epochs on the
    S/C track (used / occulted). Orthographic view; track points hidden behind the
    body are not drawn. Inputs are inertial positions [km] at times tau [s].
    """
    view = _view_dir(elev, azim)
    light = _view_dir(elev + 30.0, azim + 35.0)  # key light above-left of the camera

    sc_b = _to_body(sc_line[:, :3], tau_line, alpha, delta, omega, w0)
    pt_b = _to_body(pt_line[:, :3], tau_line, alpha, delta, omega, w0)
    obs_b = _to_body(sc_obs[:, :3], tau_obs, alpha, delta, omega, w0)
    vis = np.asarray(vis_mask, dtype=bool)

    def _visible_line(xyz):
        out = xyz.copy()
        out[_hidden_by_mesh(mesh, xyz, view)] = np.nan
        return out

    sc_v, pt_v = _visible_line(sc_b), _visible_line(pt_b)
    obs_hidden = _hidden_by_mesh(mesh, obs_b, view)

    fig = plt.figure(figsize=(COL_W, 2.9))
    ax = fig.add_axes((0.0, -0.03, 1.0, 0.96), projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)

    ax.add_collection3d(_shaded_mesh(mesh, view, light))
    ax.plot(*sc_v.T, color=C_SC, lw=1.1, zorder=3)
    ax.plot(*pt_v.T, color=C_TRUTH, lw=1.1, zorder=3)

    used = vis & ~obs_hidden
    occl = ~vis & ~obs_hidden
    ax.plot(
        *obs_b[used].T,
        ls="none",
        marker="o",
        ms=2.8,
        mfc=C_SC,
        mec="white",
        mew=0.35,
        zorder=4,
    )
    ax.plot(
        *obs_b[occl].T, ls="none", marker="x", ms=3.2, color=C_OCC, mew=0.9, zorder=4
    )
    if (
        detach_bf is not None
        and not _hidden_by_mesh(mesh, detach_bf[None], view)[0]
    ):
        ax.plot(
            *np.asarray(detach_bf)[:, None],
            ls="none",
            marker="*",
            ms=7.5,
            mfc=C_TRUTH,
            mec="white",
            mew=0.5,
            zorder=5,
        )

    # Equal-aspect cube around everything drawn
    xyz = np.vstack([np.asarray(mesh.vertices), sc_b, pt_b])
    mid = 0.5 * (xyz.min(0) + xyz.max(0))
    half = 0.5 * np.max(xyz.max(0) - xyz.min(0)) * 1.02
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)

    for a, lab in zip(
        (ax.xaxis, ax.yaxis, ax.zaxis),
        (
            r"$X_{\mathscr{B}}$ [km]",
            r"$Y_{\mathscr{B}}$ [km]",
            r"$Z_{\mathscr{B}}$ [km]",
        ),
    ):
        a.set_pane_color((1.0, 1.0, 1.0, 0.0))
        a.line.set_color("0.4")
        a.line.set_linewidth(0.5)
        a._axinfo["grid"].update(color="0.88", linewidth=0.4)
        a._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.25)
        a.set_major_locator(mticker.MaxNLocator(5))
        a.set_tick_params(pad=-1.5, labelsize=6.5)
        a.set_label_text(lab)
        a.labelpad = -3.0

    handles = [
        Line2D([], [], color=C_SC, lw=1.1, label="Spacecraft"),
        Line2D([], [], color=C_TRUTH, lw=1.1, label=f"{TARGET} (Truth)"),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=3.5,
            mfc=C_SC,
            mec="white",
            mew=0.35,
            label=f"{OBS_WORD} Used",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="x",
            ms=3.8,
            color=C_OCC,
            mew=0.9,
            label=f"{OBS_WORD} Occulted",
        ),
    ]
    if detach_bf is not None:
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="*",
                ms=7,
                mfc=C_TRUTH,
                mec="white",
                mew=0.5,
                label=SITE_LABEL,
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.0,
        handletextpad=0.4,
    )
    return save_figure(fig, name)


# ------------------------------------------------------------
# Observation geometry and post-fit residuals (time axis)
# ------------------------------------------------------------


def plot_observation_geometry(
    tau_obs,
    sc_obs,
    pt_obs,
    vis_mask,
    tau_line=None,
    sc_line=None,
    pt_line=None,
    name="fig_observation_geometry",
):
    """S/C-particle range over the arc; image epochs marked used / occulted (shaded)."""
    t = np.asarray(tau_obs) / 3600.0
    vis = np.asarray(vis_mask, dtype=bool)
    rho = np.linalg.norm(np.asarray(pt_obs)[:, :3] - np.asarray(sc_obs)[:, :3], axis=1)

    fig, ax = plt.subplots(figsize=(COL_W, 1.95), layout="constrained")
    ax.set_xlim(t[0], t[-1])
    shade_occulted(ax, t, vis, text=f"Occulted\nby {BODY}", text_y=0.04)
    if tau_line is not None:
        rho_l = np.linalg.norm(
            np.asarray(pt_line)[:, :3] - np.asarray(sc_line)[:, :3], axis=1
        )
        ax.plot(np.asarray(tau_line) / 3600.0, rho_l, color="0.25", lw=0.9, zorder=2)
    ax.plot(
        t[vis],
        rho[vis],
        ls="none",
        marker="o",
        ms=3.4,
        mfc=C_SC,
        mec="white",
        mew=0.4,
        zorder=3,
    )
    ax.plot(
        t[~vis],
        rho[~vis],
        ls="none",
        marker="x",
        ms=3.6,
        color=C_OCC,
        mew=1.0,
        zorder=3,
    )
    ax.set_xlabel("Time since Epoch [h]")
    # The axis label stays short so it fits the 1.95 in panel; the legend names the pair.
    ax.set_ylabel("Range [km]")
    rho_top = max(rho.max(), rho_l.max() if tau_line is not None else 0.0)
    ax.set_ylim(0.0, 1.12 * rho_top)  # headroom: markers never sit on the frame
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    handles = [
        Line2D([], [], color="0.25", lw=0.9, label=f"S/C{EN_DASH}{TARGET} Range (Truth)"),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=3.4,
            mfc=C_SC,
            mec="white",
            mew=0.4,
            label=f"{OBS_WORD} Used ({int(vis.sum())})",
        ),
        (
            Patch(facecolor=OCC_FILL, lw=0),
            Line2D([], [], ls="none", marker="x", ms=3.6, color=C_OCC, mew=1.0),
        ),
    ]
    labels = [h.get_label() for h in handles[:2]] + [
        f"{OBS_WORD} Occulted ({int((~vis).sum())})"
    ]
    ax.legend(handles, labels, loc="lower right", ncol=1)
    return save_figure(fig, name)


def plot_postfit_residuals(
    tau_obs, res_batch, res_mcmc, vis_mask, components=None, name="fig_postfit_residuals"
):
    """
    Normalized post-fit residuals at the visible epochs for the Gauss-Newton batch (open
    circles) and the MCMC MAP (dots), one panel per measurement component. Inputs are stacked
    per epoch, e.g. [ra0, dec0, ra1, ...] / sigma, with the component names in RESID_LABELS.
    `components` selects the panels (indices into RESID_LABELS; default: all), e.g. [1] to show
    range-rate only when range carries no information. `res_mcmc=None` draws the batch alone
    (Stage-1 diagnostic, before any MCMC exists).
    """
    t = np.asarray(tau_obs) / 3600.0
    vis = np.asarray(vis_mask, dtype=bool)
    n_comp = len(RESID_LABELS)
    comps = list(range(n_comp)) if components is None else list(components)
    rb = np.asarray(res_batch).reshape(-1, n_comp)[:, comps]
    has_mcmc = res_mcmc is not None
    rm = np.asarray(res_mcmc).reshape(-1, n_comp)[:, comps] if has_mcmc else None

    fig, axes = plt.subplots(
        len(comps),
        1,
        figsize=(COL_W, 0.75 + 1.075 * len(comps)),
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    axes = axes[:, 0]
    shown = np.r_[rb[vis].ravel(), rm[vis].ravel()] if has_mcmc else rb[vis].ravel()
    lim = max(3.6, 1.15 * float(np.max(np.abs(shown))))
    for k, (ax, comp) in enumerate(zip(axes, [RESID_LABELS[c] for c in comps])):
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-lim, lim)
        ax.axhspan(-3.0, 3.0, color="0.95", lw=0, zorder=0)
        ax.axhspan(-1.0, 1.0, color="0.89", lw=0, zorder=0)
        ax.axhline(0.0, color="0.45", lw=0.6, zorder=1)
        shade_occulted(ax, t, vis, text="Occulted" if k == 0 else None)
        ax.plot(
            t[vis],
            rb[vis, k],
            ls="none",
            marker="o",
            ms=4.2,
            mfc="none",
            mec=C_BATCH,
            mew=0.8,
            zorder=3,
        )
        if has_mcmc:
            ax.plot(
                t[vis], rm[vis, k], ls="none", marker="o", ms=1.9, color=C_POST, zorder=4
            )
        ax.set_yticks([-3, -1, 0, 1, 3])
        ax.yaxis.grid(False)
        ax.set_ylabel(rf"{comp} Residual [$\sigma$]")
        rms_b = np.sqrt(np.mean(rb[vis, k] ** 2))
        rms_txt = rf"RMS: {rms_b:.2f}"
        if has_mcmc:
            rms_txt = (
                rf"RMS: batch {rms_b:.2f}, MCMC {np.sqrt(np.mean(rm[vis, k] ** 2)):.2f}"
            )
        ax.text(
            0.99,
            0.04,
            rms_txt,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.2",
            zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85),
        )
    axes[-1].set_xlabel("Time since Epoch [h]")

    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=4.2,
            mfc="none",
            mec=C_BATCH,
            mew=0.8,
            label="Batch Least Squares",
        ),
        Line2D([], [], ls="none", marker="o", ms=2.4, color=C_POST, label="MCMC MAP"),
        Patch(facecolor="0.89", label=r"$\pm1\sigma$"),
        Patch(facecolor="0.95", label=r"$\pm3\sigma$"),
    ]
    if not has_mcmc:
        handles.pop(1)
    fig.legend(
        handles=handles, loc="outside upper center", ncol=len(handles), columnspacing=0.9
    )
    return save_figure(fig, name)


# ------------------------------------------------------------
# MCMC diagnostics (chain traces, autocorrelation, corner)
# ------------------------------------------------------------


def _require_configured(n_par):
    """Fail early (with the fix) when configure() does not describe n_par parameters."""
    if len(PARAM_SYM) != n_par or len(PARAM_SCALE) != n_par:
        raise RuntimeError(
            f"configure() describes {len(PARAM_SYM)} parameters but the "
            f"chain has {n_par}: call configure(param_sym, param_unit, param_scale) first"
        )


def _chain_physical(model, discard=0, thin=1, transform=None):
    """
    Full emcee chain (n_steps, n_walkers, ndim) in physical (unwhitened) coordinates.
    `transform` (optional) maps the sampled parameters to the displayed ones, applied to an
    array (..., ndim) after unwhitening (e.g. a non-centred hierarchical parametrization).
    """
    ch = np.asarray(model.sampler.get_chain(discard=discard, thin=thin), dtype=float)
    if getattr(model, "is_whitened", False):
        ch = ch @ np.asarray(model.whiten_L).T + np.asarray(model.whiten_mean)
    if transform is not None:
        ch = np.asarray(transform(ch), dtype=float)
    _require_configured(ch.shape[-1])
    return ch


def plot_mcmc_traces(
    model,
    burn_in,
    true_theta=None,
    batch_theta=None,
    n_walkers_show=24,
    max_points=1500,
    seed=0,
    transform=None,
    logp_truth_theta=None,
    name="fig_mcmc_traces",
    prior_idx=None,
    prior_label="Hyperprior Mean",
    prior_truth_label=None,
):
    """
    Walker traces of every parameter and of the log-posterior over the full run: a random
    subset of walkers (grey), the ensemble mean with its +/-3 sigma band (blue), the truth
    (black) and the batch estimate (vermillion). The burn-in is shaded. Y-limits follow the
    post-burn-in ensemble.

    The log-posterior is not Gaussian across walkers: near the mode
    log p = log p_max - chi2_d / 2, capped above with a long lower tail. A symmetric
    mean +/- 3 sigma band would sit partly above log p_max, where no walker can go, and miss
    the lower tail. Its band is therefore the equal-tailed 99.73 % interval (same mass as
    +/-3 sigma).

    transform: maps sampled -> displayed parameters (see _chain_physical); true_theta and
      batch_theta are then given in displayed coordinates.
    logp_truth_theta: the truth in SAMPLED coordinates, for log p(truth) (default:
      true_theta, i.e. when no transform is used).
    prior_idx: parameters the batch does not estimate (e.g. hierarchical ln sigma_g), whose
      batch_theta entry is their (hyper)prior mean instead: drawn dashed reddish purple with
      its own legend entry (prior_label) rather than as the batch estimate.
    prior_truth_label: if given, the true_theta of the prior_idx parameters is not a
      generative value (e.g. the realized RMS of the group deviations): drawn dotted with
      this legend entry instead of as the truth.
    Two columns of panels, three when there are more than 14 panels.
    """
    n_steps = int(model.sampler.iteration)
    rs = int(getattr(model.sampler, "stride", 1))  # > 1: chain loaded from a file
    stride = rs * max(1, round(n_steps / max_points / rs))
    ch = _chain_physical(model, thin=stride, transform=transform) * PARAM_SCALE
    lp = np.asarray(model.sampler.get_log_prob(thin=stride), dtype=float)
    steps = np.arange(ch.shape[0]) * stride
    n_par = ch.shape[2]
    post = steps >= burn_in
    rng = np.random.default_rng(seed)
    pick = rng.choice(ch.shape[1], size=min(n_walkers_show, ch.shape[1]), replace=False)

    # parameters fill the grid row by row; log p spans what is left of the bottom row
    n_cols = 2 if n_par + 1 <= 14 else 3
    n_rows = int(np.ceil((n_par + 1) / n_cols))
    fig = plt.figure(
        figsize=(TXT_W, (1.18 if n_cols == 2 else 1.1) * n_rows + 0.45),
        layout="constrained",
    )
    gs = fig.add_gridspec(n_rows, n_cols)
    axes = []
    for i in range(n_par):
        r, c = divmod(i, n_cols)
        axes.append(fig.add_subplot(gs[r, c], sharex=axes[0] if axes else None))
    r, c = divmod(n_par, n_cols)
    axes.append(fig.add_subplot(gs[r, c:], sharex=axes[0]))
    axes = np.array(axes)

    p3 = 100.0 * norm.sf(3.0)  # one tail of +/-3 sigma (0.135 %)
    band_alpha = 0.28  # shade only, drawn over the walkers: opaque enough to read on them

    def _windows(h):
        """[i0, i1) of a +/-h window around each thinned step, never crossing the end of the
        burn-in, so the post-burn-in band uses post-burn-in samples only."""
        n, nb = len(steps), int(np.argmax(post)) if post.any() else len(steps)
        idx = np.arange(n)
        seg0 = np.where(idx < nb, 0, nb)
        seg1 = np.where(idx < nb, nb, n)
        return np.maximum(idx - h, seg0), np.minimum(idx + h + 1, seg1)

    def _band(y, skewed, min_pool=4000):
        """
        Ensemble mean per step, and a +/-3 sigma band pooled over enough neighbouring
        (thinned) steps to hold ~min_pool samples, so the band edges are smooth lines rather
        than per-step noise. For a skewed quantity (log p) the band is the equal-tailed
        99.73 % interval (same mass as +/-3 sigma), from a 4x wider pool and then a 2x wider
        running mean: a 0.135 % sample quantile rests on a handful of tail samples, so it
        otherwise wobbles as they enter and leave the pool.
        """
        n, m = y.shape
        mu = np.mean(y, axis=1)
        h = int(np.ceil(0.5 * min_pool / m))
        if skewed:
            h *= 2
        i0, i1 = _windows(h)
        if not skewed:
            c1 = np.concatenate([[0.0], np.cumsum(y.sum(axis=1))])
            c2 = np.concatenate([[0.0], np.cumsum((y**2).sum(axis=1))])
            cnt = (i1 - i0) * m
            mean_w = (c1[i1] - c1[i0]) / cnt
            sig_w = np.sqrt(np.maximum((c2[i1] - c2[i0]) / cnt - mean_w**2, 0.0))
            return mu, mean_w - 3.0 * sig_w, mean_w + 3.0 * sig_w
        j0, j1 = _windows(2 * h)
        q = np.array([np.percentile(y[a:b], [p3, 100.0 - p3]) for a, b in zip(j0, j1)])
        cq = np.vstack([np.zeros((1, 2)), np.cumsum(q, axis=0)])
        q = (cq[i1] - cq[i0]) / (i1 - i0)[:, None]
        return mu, q[:, 0], q[:, 1]

    pri = set() if prior_idx is None else {int(k) for k in prior_idx}

    realized = pri if prior_truth_label else set()  # truth drawn as a realized value

    def _panel(
        ax,
        y,
        ylabel,
        ref_true=None,
        ref_batch=None,
        skewed=False,
        ref_color=C_BATCH,
        true_ls="-",
    ):
        # Walkers first and faint; the band (shade only, no edge lines) and the mean go on
        # top, so the band stays visible even where the walker traces fill it completely.
        ax.axvspan(0, burn_in, color=C_SPAN, lw=0, zorder=0)
        ax.plot(
            steps,
            y[:, pick],
            color="0.72",
            lw=0.25,
            alpha=0.45,
            zorder=1,
            rasterized=True,
        )
        mu, band_lo, band_hi = _band(y, skewed)
        ax.fill_between(
            steps, band_lo, band_hi, color=C_POST, alpha=band_alpha, lw=0, zorder=2
        )
        ax.plot(steps, mu, color=C_POST, lw=1.0, zorder=3)
        if ref_batch is not None:
            ax.axhline(ref_batch, color=ref_color, lw=0.9, ls="--", zorder=4)
        if ref_true is not None:
            ax.axhline(ref_true, color=C_TRUTH, lw=0.8, ls=true_ls, zorder=5)
        yp = y[post]
        lo, hi = np.percentile(yp, [0.5, 99.5])
        refs = [r for r in (ref_true, ref_batch) if r is not None]
        lo, hi = min([lo] + refs), max([hi] + refs)
        pad = 0.35 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    for i in range(n_par):
        _panel(
            axes[i],
            ch[:, :, i],
            param_label(i),
            None if true_theta is None else true_theta[i] * PARAM_SCALE[i],
            None if batch_theta is None else batch_theta[i] * PARAM_SCALE[i],
            ref_color=C_PRIOR if i in pri else C_BATCH,
            true_ls=":" if i in realized else "-",
        )
    logp_true = None
    theta_lp = true_theta if logp_truth_theta is None else logp_truth_theta
    if theta_lp is not None and (transform is None or logp_truth_theta is not None):
        try:
            logp_true = float(model.log_prob(np.asarray(theta_lp, dtype=float)))
        except Exception:
            logp_true = None
        if logp_true is not None and not np.isfinite(logp_true):
            logp_true = None
    _panel(
        axes[n_par],
        lp,
        r"$\log p(\theta \mid y)$",
        ref_true=logp_true,
        skewed=True,
    )
    for k, ax in enumerate(axes):  # x-axis labelled on the bottom row only
        if k >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Step [-]")
        else:
            ax.tick_params(labelbottom=False)
    for ax in axes:
        ax.set_xlim(0, n_steps)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))

    handles = [
        Line2D(
            [], [], color="0.7", lw=0.8, label=f"Walkers ({len(pick)} of {ch.shape[1]})"
        ),
        (
            Patch(facecolor=mcolors.to_rgba(C_POST, band_alpha), lw=0),
            Line2D([], [], color=C_POST, lw=0.9),
        ),
        Line2D([], [], color=C_BATCH, lw=0.9, ls="--", label="Batch Least Squares"),
        Line2D([], [], color=C_TRUTH, lw=0.8, label="Truth"),
        Patch(facecolor=C_SPAN, lw=0, label="Burn-in"),
    ]
    if realized and true_theta is not None:
        handles.insert(
            4, Line2D([], [], color=C_TRUTH, lw=0.8, ls=":", label=prior_truth_label)
        )
    if pri and batch_theta is not None:
        handles.insert(
            3, Line2D([], [], color=C_PRIOR, lw=0.9, ls="--", label=prior_label)
        )
    labels = [
        "Ensemble Mean, $\\pm 3\\sigma$" if isinstance(h, tuple) else h.get_label()
        for h in handles
    ]
    ncol = len(handles) if len(handles) <= 6 else int(np.ceil(len(handles) / 2))
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol)
    return save_figure(fig, name)


# Okabe-Ito order (minus the yellow) for parameter groups beyond the 3D state
_GROUP_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "0.2")
_GROUP_LS = ("-", "--", "-.", (0, (1, 1)), (0, (5, 1, 1, 1, 1, 1)), (0, (3, 1)), "-")


def _default_groups(n_par):
    """[r, v] and then chunks of <= 7 of the remaining parameters."""
    if n_par <= 7:
        return [list(range(n_par))]
    rest = list(range(6, n_par))
    n_chunks = int(np.ceil(len(rest) / 7))
    return [list(range(6))] + [list(c) for c in np.array_split(rest, n_chunks)]


def plot_mcmc_autocorrelation(
    model, burn_in, thin, transform=None, groups=None, name="fig_mcmc_autocorrelation"
):
    """
    Walker-averaged normalized autocorrelation of every parameter after burn-in, with the
    integrated autocorrelation time (emcee) in the legend and the thinning interval marked.
    Up to 7 parameters: one single-column panel. More: one panel per group of parameters
    (default: position + velocity, then chunks of <= 7), side by side at full text width.
    `transform`: see _chain_physical.
    """
    import emcee

    # a chain loaded from a file holds every rs-th step: lags and tau in steps of the run
    rs = int(getattr(model.sampler, "stride", 1))
    ch = _chain_physical(model, discard=burn_in, transform=transform)  # (n, W, d)
    n, n_w, n_par = ch.shape
    tau = rs * np.atleast_1d(emcee.autocorr.integrated_time(ch, tol=0))
    acf = np.array(
        [
            np.mean(
                [emcee.autocorr.function_1d(ch[:, k, i]) for k in range(n_w)], axis=0
            )
            for i in range(n_par)
        ]
    )
    n_lag = int(min(n - 1, max(6.0 * np.max(tau), 2.5 * thin) / rs))
    lag = rs * np.arange(n_lag + 1)
    max_lag = lag[-1]
    groups = _default_groups(n_par) if groups is None else [list(g) for g in groups]
    # first panel holds the whole [r, v] state (6-parameter panel, or 7 with e.g. eta)
    state_first = set(range(6)) <= set(groups[0])

    def _style(i, pos):
        """(color, linestyle) of parameter i at position `pos` in its group."""
        if state_first and i < 6:
            return C_XYZ[i % 3], "-" if i < 3 else "--"  # x/y/z colors; r solid, v dashed
        if state_first and n_par <= 7:  # Ryugu-like single panel: 7th parameter dark dash-dot
            return "0.2", "-."
        return _GROUP_COLORS[pos % 7], _GROUP_LS[pos % 7]

    def _draw(ax, idx, thin_text=True):
        ax.axhline(0.0, color="0.45", lw=0.6)
        for pos, i in enumerate(idx):
            c, ls = _style(i, pos)
            ax.plot(
                lag,
                acf[i, : n_lag + 1],
                color=c,
                ls=ls,
                lw=1.1,
                label=rf"{param_label(i, unit=False)}: $\hat\tau = {tau[i]:.0f}$",
            )
        ax.axvline(thin, color="0.3", lw=0.7, ls=":")
        if thin_text:
            ax.text(
                thin,
                0.55,
                " Thinning",
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=7,
                color="0.3",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xlim(0, max_lag)
        ax.set_ylim(-0.2, 1.02)
        ax.set_xlabel("Lag [-]")

    if len(groups) == 1:
        fig, ax = plt.subplots(figsize=(COL_W, 2.35), layout="constrained")
        _draw(ax, groups[0])
        ax.set_ylabel("Autocorrelation [-]")
        handles, labels = ax.get_legend_handles_labels()
        ncol = max(1, int(np.ceil(len(labels) / 2)))
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=ncol,
            fontsize=6.5,
            columnspacing=0.8,
            handlelength=2.2,
        )
        return save_figure(fig, name)

    n_g = len(groups)
    fig, axes = plt.subplots(
        1, n_g, figsize=(TXT_W, 2.75), sharey=True, layout="constrained"
    )
    fig.get_layout_engine().set(wspace=0.03)
    for g, (ax, idx) in enumerate(zip(axes, groups)):
        _draw(ax, idx, thin_text=(g == 0))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=6.5,
            columnspacing=0.8,
            handlelength=2.2,
            labelspacing=0.25,
            borderaxespad=0.0,
        )
        ax.text(  # panel letter in the (empty) upper-right corner: the legend sits on top
            0.97,
            0.95,
            f"({'abcdefgh'[g]})",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
    axes[0].set_ylabel("Autocorrelation [-]")
    return save_figure(fig, name)


def _label_width(s, digit_w=0.063):
    """Width of a math-mode tick label as a fraction of a corner panel's width (~63 pt).
    Measured from the usetex renderer at 6.5 pt: digit 3.96 pt, '.' 0.62 digit, minus sign
    1.57 digit."""
    return digit_w * sum(0.62 if ch == "." else 1.57 if ch == "-" else 1.0 for ch in s)


def _inner_ticks(lo, hi, margin=0.1, min_gap=None, digit_w=0.063):
    """
    2-3 round ticks (multiples of {1, 2, 2.5, 3, 4, 5, 6, 8} x 10^k) in [lo, hi], at least
    `margin` (axis fraction) from the edges and spaced so the blank space between adjacent
    labels is at least `min_gap` (axis fraction, ~one digit) in a corner panel. Prefers
    3 ticks over 2 (else 1), then the roundest multiplier, then the widest spacing. When
    three ticks crowd their labels, their outer pair (e.g. -0.1, 0.1) is tried as well.
    digit_w: width of one digit as a fraction of the panel width (default: 6.5 pt labels on
    a ~63 pt panel); min_gap defaults to 0.8 digit.
    """
    min_gap = 0.8 * digit_w if min_gap is None else min_gap
    span = hi - lo
    a, b, mid = lo + margin * span, hi - margin * span, 0.5 * (lo + hi)
    e = np.floor(np.log10(span))
    rank = {1: 0, 2: 0, 5: 0, 2.5: 1, 4: 2, 8: 2, 3: 3, 6: 3}  # lower = rounder
    steps = [(m * 10.0**k, rank[m]) for k in np.arange(e - 2, e + 1) for m in rank]
    best, best_key = None, None
    for step, rnd in steps:
        c = step * np.arange(np.ceil(a / step), np.floor(b / step) + 1)
        if c.size < 2:
            continue
        if c.size > 3:  # keep three, centred on the middle of the axis
            k = int(np.clip(np.argmin(np.abs(c - mid)) - 1, 0, c.size - 3))
            c = c[k : k + 3]
        for cand in (c, c[[0, -1]]) if c.size == 3 else (c,):
            d = _tick_decimals(cand, span)
            w = np.array([_label_width(_num(v, d), digit_w) for v in cand])
            sp = np.diff(cand)
            if sp.min() < 0.2 * span or np.min(sp / span - 0.5 * (w[:-1] + w[1:])) < min_gap:
                continue
            key = (cand.size, -rnd, sp.min())
            if best_key is None or key > best_key:
                best, best_key = cand, key
    if best is not None:
        return best
    for step, _ in sorted(steps, reverse=True):  # one tick, as round as possible, mid-axis
        c = step * np.arange(np.ceil(a / step), np.floor(b / step) + 1)
        if c.size:
            return c[[np.argmin(np.abs(c - mid))]]
    return np.array([mid])


def _hpd_levels(H, masses):
    """Density thresholds enclosing the given probability masses of a 2D histogram."""
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h)
    c /= c[-1]
    return np.array([h[min(np.searchsorted(c, m), h.size - 1)] for m in masses])


def _decimals(x, sig=2):
    """Decimals that show the positive number `x` with `sig` significant digits."""
    if not np.isfinite(x) or x <= 0:
        return 2
    return max(0, sig - 1 - int(np.floor(np.log10(x))))


def _num(x, d):
    """Fixed-point string with `d` decimals; never prints a signed zero ('-0.00')."""
    v = round(float(x), d)
    return f"{0.0 if v == 0 else v:.{d}f}"


def _tick_decimals(ticks, span):
    """Fewest decimals that represent every tick exactly (2 significant digits of span if none)."""
    t = np.asarray(ticks, dtype=float)
    tol = 1e-6 * max(np.max(np.abs(t)), span)
    d = next((k for k in range(7) if np.all(np.abs(np.round(t, k) - t) <= tol)), None)
    return _decimals(span, sig=2) if d is None else d


def _fixed_formatter(ticks, span):
    """Math-mode fixed-point tick labels sharing one number of decimals (proper minus sign)."""
    d = _tick_decimals(ticks, span)
    return mticker.FuncFormatter(lambda v, _, d=d: f"${_num(v, d)}$")


def _sig_num(x, d_min=0, sig=2, sci_below=1e-2):
    """
    Math-mode number with `sig` significant digits (and at least `d_min` decimals), written
    as a\times10^{k} when |x| < `sci_below`, so a small value never prints as 0.00.
    """
    x = float(x)
    if x == 0.0 or not np.isfinite(x):
        return _num(x, 0)
    mant, ex = f"{x:.{sig - 1}e}".split("e")
    if abs(float(f"{mant}e{ex}")) < sci_below:
        return rf"{mant}\times10^{{{int(ex)}}}"
    return _num(x, max(d_min, _decimals(abs(float(f"{mant}e{ex}")), sig)))


def _summary_title(sym, x):
    """
    'sym = median +/- half-width' of the 16-84 % interval (asymmetric if they differ > 5 %).
    Every number keeps 2 significant digits; the median also keeps the uncertainty's
    decimals, and values below 0.01 go to scientific notation.
    """
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    em, ep = q50 - q16, q84 - q50
    d = _decimals(min(em, ep), sig=2)
    med = _sig_num(q50, d_min=d)
    if abs(ep - em) <= 0.05 * 0.5 * (ep + em):
        return rf"${sym} = {med} \pm {_sig_num(0.5 * (ep + em), d_min=d)}$"
    # after a power of ten, hang the errors on an empty group so they don't read as exponents
    med = rf"{med}\,{{}}" if "times" in med else med
    return rf"${sym} = {med}^{{+{_sig_num(ep, d_min=d)}}}_{{-{_sig_num(em, d_min=d)}}}$"


def plot_corner(
    samples,
    true_theta=None,
    batch_mean=None,
    batch_cov=None,
    bins=40,
    smooth=1.1,
    name="fig_corner",
    prior_idx=None,
    prior_label="Hyperprior",
    prior_truth_label=None,
):
    """
    Corner plot of the MCMC posterior (blue: 1/2/3-sigma highest-density regions, i.e.
    39.3 / 86.5 / 98.9 % of the 2D mass) against the batch least-squares covariance
    (vermillion ellipses with the same masses) and the truth (black). Displayed in m, mm/s, %.
    Diagonal titles: posterior median and half-widths of the 16-84 % interval.

    prior_idx: parameters the batch does not estimate (e.g. hierarchical ln sigma_g), whose
    entries of batch_mean / batch_cov hold their (hyper)prior instead. Those are drawn as a
    separate reference (dashed reddish purple, own legend entry): pdf on the diagonal and
    ellipses on prior x prior panels; mixed batch x prior panels get no reference ellipse,
    since the product of a batch marginal and a hyperprior is neither.
    prior_truth_label: if given, the true_theta of the prior_idx parameters is not a
    generative value (e.g. the realized RMS of the group deviations): drawn as a hollow
    star (and a dotted line on the diagonal) with this legend entry instead of as the truth.

    The 2D densities are estimated in pair-whitened coordinates (histogram + Gaussian
    smoothing) and mapped back, so near-degenerate pairs (|rho| ~ 1) stay smooth.

    Above 9 parameters the panels are small: tick labels, titles and markers shrink, the
    axis labels carry the symbol only and the units go in a key under the legend (grouped
    by unit), and the grid is narrowed so the diagonal titles (which overhang to the right)
    stay inside the TXT_W page width.
    """
    from scipy.ndimage import gaussian_filter

    _require_configured(np.shape(samples)[1])
    s = PARAM_SCALE
    X = np.asarray(samples, dtype=float) * s
    n_par = X.shape[1]
    tr = None if true_theta is None else np.asarray(true_theta) * s
    bm = None if batch_mean is None else np.asarray(batch_mean) * s
    bc = None if batch_cov is None else np.asarray(batch_cov) * np.outer(s, s)
    pri = set() if prior_idx is None else {int(k) for k in prior_idx}
    realized = pri if prior_truth_label else set()  # truth drawn as a realized value
    star_full = dict(marker="*", mfc=C_TRUTH, mec="white", mew=0.4)
    star_hollow = dict(marker="*", mfc="white", mec=C_TRUTH, mew=0.6)

    masses = np.array([0.393, 0.865, 0.989])
    fills = [_tint(C_POST, a) for a in (0.16, 0.38, 0.68)]

    # Axis ranges: posterior, batch +/-3.5 sigma, truth
    lo, hi = np.percentile(X, [0.05, 99.95], axis=0)
    if bm is not None and bc is not None:
        sb = np.sqrt(np.diag(bc))
        lo, hi = np.minimum(lo, bm - 3.5 * sb), np.maximum(hi, bm + 3.5 * sb)
    if tr is not None:
        lo, hi = np.minimum(lo, tr), np.maximum(hi, tr)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    big = n_par > 9
    tick_fs, title_fs = (5.0, 5.5) if big else (6.5, 7.0)
    label_fs = 7.0 if big else plt.rcParams["axes.labelsize"]
    ms_star, lw_s = (4.5, 0.75) if big else (6.5, 1.0)  # truth marker, line-width factor
    wsp = 0.08
    labels = [param_label(k, unit=not big) for k in range(n_par)]
    titles = [_summary_title(PARAM_SYM[k], X[:, k]) for k in range(n_par)]

    fig, axes = plt.subplots(n_par, n_par, figsize=(TXT_W, TXT_W))
    left, right, bottom, top = 0.085, 0.995, 0.075, 0.97
    if big:
        # measure the titles with the actual renderer (fraction of the page width)
        rend = fig.canvas.get_renderer()

        def _w_frac(txt, fs):
            t = fig.text(0, 0, txt, fontsize=fs)
            w = t.get_window_extent(renderer=rend).width / fig.bbox.width
            t.remove()
            return w

        w_tit = np.array([_w_frac(t, title_fs) for t in titles])
        left, bottom = 0.06, 0.05
        # panel pitch such that title i (left edge at column i) ends inside the page
        pitch = (right - left) / (n_par - wsp / (1.0 + wsp))
        for i in range(1, n_par):
            pitch = min(pitch, (0.995 - w_tit[i] - left) / i)
        right = left + pitch * (n_par - wsp / (1.0 + wsp))
    fig.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top, wspace=wsp, hspace=wsp
    )
    panel_pt = (right - left) * TXT_W * 72.0 / (n_par + (n_par - 1) * wsp)
    digit_w = 3.96 * tick_fs / 6.5 / panel_pt
    ticks = [_inner_ticks(lo[k], hi[k], digit_w=digit_w) for k in range(n_par)]
    fmts = [_fixed_formatter(ticks[k], hi[k] - lo[k]) for k in range(n_par)]

    for i in range(n_par):
        for j in range(n_par):
            ax = axes[i, j]
            ax.grid(False)
            if j > i:
                ax.set_visible(False)
                continue
            ax.set_xlim(lo[j], hi[j])
            ax.set_xticks(ticks[j])
            ax.xaxis.set_major_formatter(fmts[j])
            ax.tick_params(labelsize=tick_fs, length=2.5 if not big else 2.0, pad=2)

            if i == j:  # ---- marginal
                ax.hist(
                    X[:, i],
                    bins=bins,
                    range=(lo[i], hi[i]),
                    density=True,
                    histtype="stepfilled",
                    color=_tint(C_POST, 0.3),
                    ec=C_POST,
                    lw=0.8 * lw_s,
                )
                if bm is not None and bc is not None:
                    xx = np.linspace(lo[i], hi[i], 300)
                    ax.plot(
                        xx,
                        norm.pdf(xx, bm[i], np.sqrt(bc[i, i])),
                        color=C_PRIOR if i in pri else C_BATCH,
                        ls="--" if i in pri else "-",
                        lw=0.9 * lw_s,
                    )
                if tr is not None:
                    ax.axvline(
                        tr[i],
                        color=C_TRUTH,
                        lw=0.8 * lw_s,
                        ls=":" if i in realized else "-",
                    )
                ax.set_ylim(bottom=0.0)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                # left-aligned: overflow goes right, into the hidden upper triangle (the last
                # one past the grid; the tight bbox at save keeps it on the page)
                ax.set_title(titles[i], loc="left", fontsize=title_fs, pad=3)
            else:  # ---- joint (x: param j, y: param i)
                ax.set_ylim(lo[i], hi[i])
                ax.set_yticks(ticks[i])
                ax.yaxis.set_major_formatter(fmts[i])
                P2 = X[:, [j, i]]
                m2 = P2.mean(0)
                Lc = np.linalg.cholesky(np.cov(P2, rowvar=False))
                U = np.linalg.solve(Lc, (P2 - m2).T).T
                r = max(4.0, float(np.percentile(np.abs(U), 99.95))) * 1.05
                H, ex, ey = np.histogram2d(
                    U[:, 0], U[:, 1], bins=bins, range=[[-r, r]] * 2
                )
                H = gaussian_filter(H, smooth)
                cu, cv = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1])
                GU, GV = np.meshgrid(cu, cv, indexing="ij")
                GX = m2[0] + Lc[0, 0] * GU
                GY = m2[1] + Lc[1, 0] * GU + Lc[1, 1] * GV
                # descending in mass -> ascending in density
                lev = np.r_[np.sort(_hpd_levels(H, masses)), H.max() * 1.0001]
                if np.all(np.diff(lev) > 0):
                    ax.contourf(GX, GY, H, levels=lev, colors=fills, zorder=1)
                    ax.contour(
                        GX,
                        GY,
                        H,
                        levels=lev[:-1],
                        colors=C_POST,
                        linewidths=0.45 * lw_s,
                        zorder=2,
                    )
                else:
                    ax.plot(P2[:, 0], P2[:, 1], ",", color=C_POST, alpha=0.2, zorder=1)
                n_pri = (i in pri) + (j in pri)  # 0: batch, 2: hyperprior, 1: neither
                if bm is not None and bc is not None and n_pri != 1:
                    C2 = bc[np.ix_([j, i], [j, i])]
                    for rad in (1.0, 2.0, 3.0):
                        e = _ellipse_2d(bm[[j, i]], C2, rad)
                        ax.plot(
                            e[0],
                            e[1],
                            color=C_PRIOR if n_pri else C_BATCH,
                            ls="--" if n_pri else "-",
                            lw=0.7 * lw_s,
                            zorder=3,
                        )
                if tr is not None:
                    hollow = i in realized or j in realized
                    ax.plot(
                        tr[j],
                        tr[i],
                        ls="none",
                        ms=ms_star,
                        zorder=4,
                        **(star_hollow if hollow else star_full),
                    )

            # outer labels only
            if i == n_par - 1:
                ax.set_xlabel(labels[j], fontsize=label_fs, labelpad=3)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=label_fs, labelpad=3)
            else:
                ax.tick_params(labelleft=False)
    fig.align_ylabels(axes[1:, 0])
    fig.align_xlabels(axes[-1, :])

    sig123 = r"($1\sigma$, $2\sigma$, $3\sigma$)"
    handles = [
        (
            Patch(facecolor=fills[2], lw=0),
            Patch(facecolor=fills[1], lw=0),
            Patch(facecolor=fills[0], lw=0),
        ),
        Line2D([], [], color=C_BATCH, lw=0.9),
        Line2D([], [], ls="none", ms=8, **star_full),
    ]
    leg_labels = [f"MCMC Posterior {sig123}", f"Batch Least Squares {sig123}", "Truth"]
    if realized and tr is not None:  # dotted line with a hollow star, as on the panels
        handles.append(Line2D([], [], color=C_TRUTH, lw=0.8, ls=":", ms=8, **star_hollow))
        leg_labels.append(prior_truth_label)
    if pri and bm is not None and bc is not None:
        handles.insert(2, Line2D([], [], color=C_PRIOR, lw=0.9, ls="--"))
        leg_labels.insert(2, f"{prior_label} {sig123}")
    # The upper-right triangle is empty: anchor the legend to the top-right (hidden) panel.
    corner_box = axes[0, -1].get_position()
    leg = fig.legend(
        handles,
        leg_labels,
        loc="upper right",
        bbox_to_anchor=(corner_box.x1, corner_box.y1),
        bbox_transform=fig.transFigure,
        fontsize=8,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        handlelength=3.0,
        handleheight=1.0,
        labelspacing=0.6,
    )
    if big:  # units key under the legend, parameters grouped by display unit
        groups = []
        for k in range(n_par):
            u = PARAM_UNIT[k] or "-"
            if groups and groups[-1][1] == u:
                groups[-1][0].append(k)
            else:
                groups.append(([k], u))
        lines = []
        for ks, u in groups:
            syms = [f"${PARAM_SYM[k]}$" for k in ks]
            rows = [", ".join(syms[a : a + 6]) for a in range(0, len(syms), 6)]
            rows = [r + "," for r in rows[:-1]] + [rows[-1] + f" [{u}]"]
            lines += rows
        bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        fig.text(
            bb.x0 + 0.012,
            bb.y0 - 0.012,
            "Axis units:\n" + "\n".join(lines),
            ha="left",
            va="top",
            fontsize=7,
            linespacing=1.35,
        )
    return save_figure(fig, name)


# ============================================================
# POSTERIOR SPLIT (GMM) + STM PROPAGATION TO A GIVEN EPOCH
# ============================================================


def fit_posterior_gmm(
    samples, n_components=8, select_bic=False, k_max=12, random_state=0
):
    """
    Split posterior samples (N, n) into a full-covariance Gaussian mixture.

    The coordinates span several orders of magnitude (km, km/s, eta), so EM runs in whitened
    coordinates z = L^{-1}(theta - m), with L L^T the sample covariance (sklearn's
    reg_covar would otherwise swamp the small coordinates). Back-mapping:
        mu_k = m + L mu_k^z,     P_k = L P_k^z L^T.
    select_bic=True picks K in 1..k_max by BIC. On a near-Gaussian posterior BIC keeps very
    few components; a fixed K splits the posterior into narrower pieces, which is what keeps
    each component's STM linearization accurate over a long prediction.
    Components are returned sorted by decreasing weight.
    """
    X = np.asarray(samples, dtype=float)
    m = X.mean(axis=0)
    # L = D chol(corr): same as chol(cov) but well conditioned for mixed units. A nearly
    # degenerate pair (|rho| -> 1 at machine precision) gets its correlation eigenvalues
    # floored so the factorization exists.
    R = np.corrcoef(X, rowvar=False)
    try:
        Lc = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(R)
        R = (vecs * np.clip(vals, 1e-10, None)) @ vecs.T
        d = np.sqrt(np.diag(R))
        Lc = np.linalg.cholesky(R / np.outer(d, d))
        print("[GMM] sample correlation is near-singular: eigenvalues floored at 1e-10")
    L = X.std(axis=0)[:, None] * Lc
    Z = np.linalg.solve(L, (X - m).T).T

    if GaussianMixture is None:
        print("[GMM] scikit-learn not installed: fitting with the built-in numpy EM")

    def _fit(k):
        """(weights, means, covariances, BIC) of a K-component fit in z coordinates."""
        if GaussianMixture is None:
            return _gmm_em(Z, k, reg_covar=1e-6, n_init=3, max_iter=500, seed=random_state)
        g = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=3,
            max_iter=500,
            random_state=random_state,
        ).fit(Z)
        return g.weights_, g.means_, g.covariances_, g.bic(Z)

    if select_bic:
        fits = [_fit(k) for k in range(1, k_max + 1)]
        bics = np.array([f[3] for f in fits])
        w_z, mu_z, P_z, _ = fits[int(np.argmin(bics))]
        print(f"[GMM] BIC for K = 1..{k_max}:", np.array2string(bics, precision=1))
    else:
        w_z, mu_z, P_z, _ = _fit(n_components)

    order = np.argsort(w_z)[::-1]
    weights = w_z[order]
    means = m + mu_z[order] @ L.T
    covs = np.einsum("ij,kjl,ml->kim", L, P_z[order], L)
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))
    return weights, means, covs


def _gmm_em(Z, k, reg_covar=1e-6, n_init=3, max_iter=500, tol=1e-3, seed=0):
    """
    Full-covariance EM with the settings of sklearn's GaussianMixture (k-means start,
    reg_covar on the diagonal, mean log-likelihood tolerance, best of n_init starts).
    Fallback when scikit-learn is missing; returns (weights, means, covs, BIC).
    """
    rng = np.random.default_rng(seed)
    N, n = Z.shape
    best = None
    for _ in range(n_init):
        # k-means++ seeding + Lloyd iterations -> hard responsibilities
        C = Z[[rng.integers(N)]]
        for _ in range(1, k):
            d2 = np.min(((Z[:, None, :] - C[None]) ** 2).sum(axis=2), axis=1)
            C = np.vstack([C, Z[rng.choice(N, p=d2 / d2.sum())]])
        for _ in range(20):
            lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(axis=2), axis=1)
            C = np.array(
                [Z[lab == j].mean(axis=0) if np.any(lab == j) else C[j] for j in range(k)]
            )
        resp = np.eye(k)[lab]
        ll_old = -np.inf
        for _ in range(max_iter):
            Nk = resp.sum(axis=0) + 10.0 * np.finfo(float).eps
            w = Nk / N
            mu = (resp.T @ Z) / Nk[:, None]
            P = np.empty((k, n, n))
            for j in range(k):
                D = Z - mu[j]
                P[j] = (resp[:, j, None] * D).T @ D / Nk[j] + reg_covar * np.eye(n)
            logp = np.column_stack(
                [np.log(w[j]) + multivariate_normal.logpdf(Z, mu[j], P[j]) for j in range(k)]
            )
            lse = logsumexp(logp, axis=1)
            resp = np.exp(logp - lse[:, None])
            ll = float(lse.mean())
            if abs(ll - ll_old) < tol:
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, w, mu, P)
    ll, w, mu, P = best
    n_par = (k - 1) + k * n + k * n * (n + 1) / 2.0
    return w, mu, P, -2.0 * ll * N + n_par * np.log(N)


def gmm_moments(weights, means, covs):
    """Mean and covariance of a Gaussian mixture."""
    m = weights @ means
    d = means - m
    P = np.einsum("k,kij->ij", weights, covs) + np.einsum("k,ki,kj->ij", weights, d, d)
    return m, P


def gmm_logpdf(x, weights, means, covs):
    """log sum_k w_k N(x; mu_k, P_k) for x of shape (M, d)."""
    x = np.atleast_2d(x)
    lp = np.array(
        [
            np.log(w)
            + np.atleast_1d(multivariate_normal(mu, P, allow_singular=True).logpdf(x))
            for w, mu, P in zip(weights, means, covs)
        ]
    )
    return logsumexp(lp, axis=0)


def _sol_ok(sol, n_t):
    """True if a propagation reached every output epoch (scipy-like or heyoka _Sol)."""
    status = getattr(sol, "status", 0 if getattr(sol, "success", True) else -1)
    return status == 0 and np.shape(sol.y)[1] == n_t and np.all(np.isfinite(sol.y))


def propagate_gmm_stm(
    propagator, x0_ref, weights, means, covs, t_eval, label="GMM", **ivp
):
    """
    Propagate every mixture component with its OWN first-order STM:
        m_k(t) = phi(t; x0_ref + mu_k),      P_k(t) = Phi_k(t) P_k Phi_k(t)^T,
    weights unchanged (no re-splitting along the way).
    Returns m (K, N_t, n) [absolute states] and P (K, N_t, n, n).
    """
    t_eval = np.asarray(t_eval, dtype=float)
    K, n = means.shape
    m_t = np.empty((K, t_eval.size, n))
    P_t = np.empty((K, t_eval.size, n, n))
    for k in range(K):
        sol, stts = propagator.propagate(
            x0_ref + means[k], t_eval, show_progress=False, **ivp
        )
        if not _sol_ok(sol, t_eval.size):
            raise RuntimeError(
                f"[{label}] component {k} propagation failed: "
                f"{getattr(sol, 'message', '')}"
            )
        Phi = stts[1]  # (N_t, n, n)
        m_t[k] = sol.y[:n].T
        P_t[k] = Phi @ covs[k] @ np.transpose(Phi, (0, 2, 1))
        print(
            f"[{label}] component {k + 1}/{K} (w = {weights[k]:.3f}) propagated",
            flush=True,
        )
    return m_t, P_t


def propagate_samples_mc(propagator, x0_ref, deltas, t_eval, **ivp):
    """Nonlinear (state-only) propagation of x0_ref + delta_i. Failed runs are left NaN."""
    t_eval = np.asarray(t_eval, dtype=float)
    M, n = deltas.shape
    out = np.full((M, t_eval.size, n), np.nan)
    step = max(1, M // 10)
    for i, d in enumerate(deltas):
        sol = propagator.propagate_state_only(
            x0_ref + d, t_eval, show_progress=False, **ivp
        )
        if _sol_ok(sol, t_eval.size):
            out[i] = sol.y[:n].T
        if (i + 1) % step == 0 or i + 1 == M:
            print(f"[MC] {i + 1}/{M} posterior samples propagated", flush=True)
    return out


def position_prediction_history(X_mc, w_g, m_g, P_g, m_1, P_1):
    """
    Per output epoch, position-only (3D) comparison against the Monte Carlo cloud:
      sig_*: sqrt(trace P_rr) [km] for MC, GMM (moment-matched) and single Gaussian
      ll_*:  mean log-density of the MC positions under the GMM / single Gaussian
    """
    n_t = X_mc.shape[1]
    out = {k: np.empty(n_t) for k in ("sig_mc", "sig_gmm", "sig_1", "ll_gmm", "ll_1")}
    for j in range(n_t):
        pos = X_mc[:, j, :3]
        _, Pg = gmm_moments(w_g, m_g[:, j, :3], P_g[:, j, :3, :3])
        out["sig_mc"][j] = np.sqrt(np.trace(np.cov(pos, rowvar=False)))
        out["sig_gmm"][j] = np.sqrt(np.trace(Pg))
        out["sig_1"][j] = np.sqrt(np.trace(P_1[0, j, :3, :3]))
        out["ll_gmm"][j] = np.mean(
            gmm_logpdf(pos, w_g, m_g[:, j, :3], P_g[:, j, :3, :3])
        )
        out["ll_1"][j] = np.mean(
            gmm_logpdf(pos, np.ones(1), m_1[:, j, :3], P_1[:, j, :3, :3])
        )
    return out


def _ellipse_2d(mean2, cov2, radius, n=181):
    """Boundary {x : (x - m)^T C^{-1} (x - m) = radius^2}, shape (2, n)."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    vals, vecs = np.linalg.eigh(cov2)
    circ = np.vstack([np.cos(t), np.sin(t)])
    return mean2[:, None] + radius * (
        vecs @ (np.sqrt(np.clip(vals, 0.0, None))[:, None] * circ)
    )


def plot_gmm_propagation(
    panels, prob=0.99, n_draw=20000, seed=0, name="fig_gmm_stm_propagation"
):
    """
    2x3 position projections (row 0: epoch, row 1: target). Each panel dict holds (km):
      mc (M,3), w (K,), mu (K,3), P (K,3,3), mu1 (3,), P1 (3,3), truth (3,), center (3,), title.
    Drawn in meters relative to `center` (the MC mean): the MC cloud, every GMM component's
    ellipse, the GMM highest-density region holding `prob` of the mass (level from mixture
    draws), the single-Gaussian ellipse with the same mass, and the truth.
    """
    r_ell = np.sqrt(chi2.ppf(prob, 2))
    pct = rf"{100 * prob:g}$\%$"
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = [r"$\delta x$ [m]", r"$\delta y$ [m]", r"$\delta z$ [m]"]
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(TXT_W, 4.9), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, hspace=0.04, wspace=0.04)
    sub = fig.subfigures(len(panels), 1)
    axes = np.array([sf.subplots(1, 3) for sf in np.atleast_1d(sub)])
    for row, pnl in enumerate(panels):
        c = np.asarray(pnl["center"], dtype=float)
        mc = (pnl["mc"] - c) * 1e3
        w = np.asarray(pnl["w"], dtype=float)
        mu = (pnl["mu"] - c) * 1e3
        P = pnl["P"] * 1e6
        mu1 = (pnl["mu1"] - c) * 1e3
        P1 = pnl["P1"] * 1e6
        tr = (pnl["truth"] - c) * 1e3

        counts = rng.multinomial(n_draw, w / w.sum())
        draws = np.vstack(
            [
                rng.multivariate_normal(mu[k], P[k], size=counts[k])
                for k in range(len(w))
            ]
        )

        for col, (i, j) in enumerate(pairs):
            ax = axes[row, col]
            ix = [i, j]
            mu2, P2 = mu[:, ix], P[:, ix][:, :, ix]
            e1 = _ellipse_2d(mu1[ix], P1[np.ix_(ix, ix)], r_ell)

            lo = np.min(
                [
                    mc[:, ix].min(0),
                    e1.min(1),
                    np.percentile(draws[:, ix], 0.1, axis=0),
                    tr[ix],
                ],
                axis=0,
            )
            hi = np.max(
                [
                    mc[:, ix].max(0),
                    e1.max(1),
                    np.percentile(draws[:, ix], 99.9, axis=0),
                    tr[ix],
                ],
                axis=0,
            )
            pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad

            ax.scatter(
                mc[:, i],
                mc[:, j],
                s=2.5,
                color=C_MC,
                alpha=0.7,
                lw=0,
                rasterized=True,
                zorder=1,
            )
            for k in range(len(w)):
                e = _ellipse_2d(mu2[k], P2[k], r_ell)
                ax.plot(e[0], e[1], color=C_POST, lw=0.45, alpha=0.55, zorder=2)

            gx, gy = np.meshgrid(
                np.linspace(lo[0], hi[0], 240), np.linspace(lo[1], hi[1], 240)
            )
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            dens = np.exp(gmm_logpdf(grid, w, mu2, P2)).reshape(gx.shape)
            level = np.quantile(
                np.exp(gmm_logpdf(draws[:, ix], w, mu2, P2)), 1.0 - prob
            )
            ax.contour(
                gx, gy, dens, levels=[level], colors=C_POST, linewidths=1.3, zorder=3
            )

            ax.plot(e1[0], e1[1], color=C_BATCH, lw=1.1, ls="--", zorder=4)
            ax.plot(
                tr[i],
                tr[j],
                ls="none",
                marker="*",
                ms=7.5,
                mfc=C_TRUTH,
                mec="white",
                mew=0.5,
                zorder=5,
            )
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.set_xlabel(names[i])
            ax.set_ylabel(names[j])
            ax.set_title(f"({'abcdef'[3 * row + col]})", loc="left")
        np.atleast_1d(sub)[row].suptitle(pnl["title"], fontsize=8)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=3, color=C_MC, label="Monte Carlo"),
        Line2D(
            [],
            [],
            color=C_POST,
            lw=0.45,
            alpha=0.7,
            label=f"GMM Components + STMs ({pct} HDR)",
        ),
        Line2D([], [], color=C_POST, lw=1.3, label=f"GMM Mixture + STMs ({pct} HDR)"),
        Line2D(
            [],
            [],
            color=C_BATCH,
            lw=1.1,
            ls="--",
            label=f"Single Gaussian + STM ({pct} HDR)",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="*",
            ms=7.5,
            mfc=C_TRUTH,
            mec="white",
            mew=0.5,
            label="Truth",
        ),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, columnspacing=1.0)
    return save_figure(fig, name)


def plot_prediction_history(
    t_hr, hist, t_obs_end_hr, name="fig_gmm_prediction_history"
):
    """Position uncertainty and MC log-density (GMM vs single Gaussian) vs time."""
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(COL_W, 3.3),
        sharex=True,
        layout="constrained",
        gridspec_kw=dict(height_ratios=(1.35, 1.0)),
    )
    for ax in (ax0, ax1):
        ax.axvspan(t_hr[0], t_obs_end_hr, color=C_SPAN, lw=0, zorder=0)
        ax.set_xlim(t_hr[0], t_hr[-1])
    ax0.text(
        0.5 * (t_hr[0] + t_obs_end_hr),
        0.96,
        "Observation arc",
        color="0.3",
        fontsize=7,
        ha="center",
        va="top",
        transform=ax0.get_xaxis_transform(),
    )

    ax0.plot(
        t_hr,
        hist["sig_mc"] * 1e3,
        color="0.78",
        lw=3.2,
        solid_capstyle="round",
        label="Monte Carlo",
        zorder=2,
    )
    ax0.plot(
        t_hr, hist["sig_gmm"] * 1e3, color=C_POST, lw=1.2, label="GMM + STMs", zorder=3
    )
    ax0.plot(
        t_hr,
        hist["sig_1"] * 1e3,
        color=C_BATCH,
        lw=1.2,
        ls="--",
        label="Single Gaussian + STM",
        zorder=4,
    )
    ax0.set_ylabel(r"$\sqrt{\mathrm{tr}(P_{rr})}$ [m]")
    ax0.set_yscale("log")
    ax0.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax0.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax0.set_title("(a) Position Uncertainty", loc="left")

    ax1.axhline(0.0, color="0.45", lw=0.6)
    ax1.plot(t_hr, hist["ll_gmm"] - hist["ll_1"], color=C_POST, lw=1.2)
    ax1.set_ylabel(r"$\Delta \overline{\log p}$ [nats]")
    ax1.set_xlabel("Time since Epoch [h]")
    ax1.set_title(r"(b) Mean Log-Density of MC Positions, GMM $-$ Single", loc="left")

    fig.legend(loc="outside upper center", ncol=3, columnspacing=0.8, handlelength=1.5)
    return save_figure(fig, name)


