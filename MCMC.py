import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee

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
        self.param_priors = param_priors  # list of scipy.stats distributions
        self.observed_data = observed_data
        self.ndim = len(initial_params)
        self.sampler = None
        self.samples = None
        self.log_probs = None

    def log_prior(self, theta):
        lp = 0.0
        for i, prior in enumerate(self.param_priors):
            lp_i = prior.logpdf(theta[i])
            if not np.isfinite(lp_i):
                return -np.inf
            lp += lp_i
        return lp

    def log_likelihood(self, theta):
        residuals = self.residuals_func(theta)
        return -0.5 * np.sum(residuals**2)  # assume identity covariance

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run(self, n_samples=5000, n_walkers=50, burn_in=500):
        def log_prob(theta):
            return self.log_posterior(theta)

        pos = self.initial_params + 1e-4 * np.random.randn(n_walkers, self.ndim)
        self.sampler = emcee.EnsembleSampler(n_walkers, self.ndim, log_prob)
        self.sampler.run_mcmc(pos, n_samples, progress=True)
        self.samples = self.sampler.get_chain(discard=burn_in, flat=True)
        self.log_probs = self.sampler.get_log_prob(discard=burn_in, flat=True)

    def plot_convergence(self):
        fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(self.sampler.get_chain()[:, :, i], alpha=0.5)
            ax.set_ylabel(f"$\\theta_{{{i}}}$")
            ax.grid(True)
        axes[-1].set_xlabel("Step Number")
        plt.tight_layout()
        plt.show()

    def plot_log_likelihood(self):
        if self.log_probs is None:
            print("Run MCMC first.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.log_probs, alpha=0.6)
        plt.xlabel("Sample Index")
        plt.ylabel("$\\log \\mathcal{P}(\\theta \mid y)$")
        plt.grid(True)
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

    def summary(self):
        if self.samples is None:
            print("Run MCMC first.")
            return

        print("Parameter estimates:")
        for i in range(self.ndim):
            mcmc = np.percentile(self.samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            print(f"$\\theta_{{{i}}}$: {mcmc[1]:.4f} (+{q[1]:.4f}/-{q[0]:.4f})")

        corner.corner(
            self.samples,
            labels=[f"$\\theta_{{{i}}}$" for i in range(self.ndim)],
            truths=np.median(self.samples, axis=0),
            show_titles=True,
            title_fmt=".4f",
            title_kwargs={"fontsize": 12},
        )
        plt.show()
