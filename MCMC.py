import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
import multiprocessing

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

    def log_prob(self, theta):
        return self.log_posterior(theta)

    def initialize_walkers_from_prior(self, n_walkers, sigma_multiplier=1.0):
        pos = np.zeros((n_walkers, self.ndim))

        for i, prior in enumerate(self.param_priors):
            try:
                pos[:, i] = prior.rvs(size=n_walkers)
            except AttributeError:
                mu = prior.mean()
                sigma = prior.std()
                pos[:, i] = mu + sigma_multiplier * sigma * np.random.randn(n_walkers)

        return pos

    def run(self, n_samples=5000, n_walkers=50, burn_in=500, init_pos=None):
        # Initialize walkers
        if init_pos is None:
            pos = self.initialize_walkers_from_prior(n_walkers)
        else:
            pos = init_pos

        # Run MCMC using emcee
        with multiprocessing.get_context("fork").Pool() as pool:
            self.sampler = emcee.EnsembleSampler(
                n_walkers, self.ndim, self.log_prob, pool=pool
            )
            self.sampler.run_mcmc(pos, n_samples, progress=True)

        # Discard burn-in samples and flatten the chain
        self.samples = self.sampler.get_chain(discard=burn_in, flat=True)
        self.log_probs = self.sampler.get_log_prob(discard=burn_in, flat=True)

    def run_hmc(
        self,
        n_samples=3000,
        burn_in=500,
        step_size=1e-3,
        num_integration_steps=20,
        print_every=500,
    ):
        """
        Run Hamiltonian Monte Carlo from scratch using NumPy, with status prints.
        """

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
                theta_new += step_size * p_new
                p_new -= step_size * grad_U(theta_new)

            theta_new += step_size * p_new
            p_new -= 0.5 * step_size * grad_U(theta_new)
            return theta_new, -p_new

        theta_current = np.array(self.initial_params)
        samples = []
        logps = []
        accepted = 0

        total_steps = n_samples + burn_in
        print("Starting HMC sampling...")
        for i in range(total_steps):
            p_current = np.random.randn(self.ndim)
            theta_proposed, p_proposed = leapfrog(
                theta_current, p_current, step_size, num_integration_steps
            )

            U_current = U(theta_current)
            U_proposed = U(theta_proposed)
            K_current = 0.5 * np.sum(p_current**2)
            K_proposed = 0.5 * np.sum(p_proposed**2)

            log_accept_prob = U_current + K_current - U_proposed - K_proposed
            accepted_flag = False

            if np.log(np.random.rand()) < log_accept_prob:
                theta_current = theta_proposed
                accepted_flag = True
                accepted += 1

            if i >= burn_in:
                samples.append(theta_current.copy())
                logps.append(-U(theta_current))

            if i % print_every == 0 or i == total_steps - 1:
                phase = "Burn-in" if i < burn_in else "Sampling"
                print(
                    f"[{i}/{total_steps}] {phase} | "
                    f"Accepted so far: {accepted}/{i+1} ({(accepted/(i+1))*100:.1f}%)"
                )

        self.samples = np.array(samples)
        self.log_probs = np.array(logps)

        print("HMC sampling completed.")
        print(f"Final acceptance rate: {(accepted / total_steps):.2%}")

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
        if self.sampler is None:
            print("Run MCMC first.")
            return

        log_probs_all = self.sampler.get_log_prob()  # shape: (n_steps, n_walkers)
        n_steps, n_walkers = log_probs_all.shape

        plt.figure(figsize=(10, 5))
        plt.yscale("log")
        for i in range(n_walkers):
            plt.plot(-log_probs_all[:, i], alpha=0.6, label=f"Walker {i}", linewidth=1)

        plt.xlabel("Step Number")
        plt.ylabel(r"$-\log \mathcal{P}(\theta \mid y)$")
        plt.title("Log-Likelihood Traces per Walker")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        # Uncomment below to see legend (can be cluttered with many walkers)
        # plt.legend(fontsize=8, ncol=4)
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

        # print("MCMC summary:")
        # tau = self.sampler.get_autocorr_time()
        # print(f"Autocorr time per parameter: {tau}")

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
