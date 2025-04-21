import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import emcee
import corner


# ------------------------------------------------------------------
# 1. Simulate True Dynamics (Generate Synthetic Data)
# ------------------------------------------------------------------
def two_body(t, state, mu):
    r = state[:3]
    v = state[3:]
    norm_r = np.linalg.norm(r)
    a = -mu * r / norm_r**3
    return np.concatenate([v, a])


# True parameters
mu_true = 398600.0  # [km^3/s^2] Earth's gravitational parameter
x0_true = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 1.0])  # [km, km/s]
t_eval = np.linspace(0, 3600, 50)  # one hour, 50 points

# Integrate dynamics to generate synthetic measurements
sol = solve_ivp(
    lambda t, y: two_body(t, y, mu_true),
    [t_eval[0], t_eval[-1]],
    x0_true,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-9,
)
X_true = sol.y.T  # shape (n_points, 6)
range_true = np.linalg.norm(X_true[:, :3], axis=1)
range_meas = range_true + np.random.normal(0, 5.0, size=range_true.shape)  # 5 km noise
print("Done generating synthetic data.")

# ------------------------------------------------------------------
# 2. Define the Log-Probability Functions for emcee
# ------------------------------------------------------------------
# Our parameter vector theta = [x0_0, ..., x0_5, mu, sigma] (8 elements)


def log_prior(theta):
    x0 = theta[:6]
    mu = theta[6]
    sigma = theta[7]
    if sigma <= 0:
        return -np.inf
    # Prior on x0: each component ~ Normal(x0_true, 50)
    lp = -0.5 * np.sum(((x0 - x0_true) / 50.0) ** 2) - 6 * np.log(
        50 * np.sqrt(2 * np.pi)
    )
    # Prior on mu: Normal(mean=400000, sigma=5000)
    lp += -0.5 * ((mu - 400000.0) / 5000.0) ** 2 - np.log(5000.0 * np.sqrt(2 * np.pi))

    # Prior on sigma: HalfNormal with sigma=10 (with proper normalization for x > 0)
    lp += np.log(2) - np.log(10 * np.sqrt(2 * np.pi)) - 0.5 * (sigma / 10.0) ** 2
    return lp


def log_likelihood(theta):
    x0 = theta[:6]
    mu = theta[6]
    sigma = theta[7]
    try:
        sol = solve_ivp(
            lambda t, y: two_body(t, y, mu),
            [t_eval[0], t_eval[-1]],
            x0,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )
        if not sol.success:
            return -np.inf
        X_model = sol.y.T  # shape (n_points, 6)
    except Exception:
        return -np.inf

    pred_range = np.linalg.norm(X_model[:, :3], axis=1)
    n = len(range_meas)
    # Gaussian likelihood: sum log(pdf) for each measurement
    ll = -0.5 * np.sum(((range_meas - pred_range) / sigma) ** 2) - n * np.log(
        sigma * np.sqrt(2 * np.pi)
    )
    return ll


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# ------------------------------------------------------------------
# 3. Run the MCMC with emcee
# ------------------------------------------------------------------
# Initialize walkers: we'll use 32 walkers and set the initial position near our "true" guess.
theta_init = np.concatenate([x0_true, [400000.0, 10.0]])
ndim = len(theta_init)
nwalkers = 32
# Add a small random scatter to the initial guess for each walker
pos0 = theta_init + 1e-2 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
print("Running MCMC using emcee...")
sampler.run_mcmc(pos0, 500, progress=True)
print("Emcee sampling complete.")

# Discard burn-in samples and flatten the chain
samples = sampler.get_chain(discard=1000, flat=True)

# ------------------------------------------------------------------
# 4. Post-Analysis
# ------------------------------------------------------------------
# Get the chain (shape: nsteps x nwalkers x ndim)
chain = sampler.get_chain()

# 4a. Convergence (Trace) Plots
labels = ["x0_0", "x0_1", "x0_2", "x0_3", "x0_4", "x0_5", "mu", "sigma"]

fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, ndim * 1.8))
for i in range(ndim):
    ax = axes[i]
    for walker in range(nwalkers):
        ax.plot(chain[:, walker, i], alpha=0.4)
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("Step")
fig.suptitle("Trace Plots for Each Parameter", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 4b. Corner Plot
# Flatten the chain by discarding burn-in samples (e.g., first 1000 steps) and combining walkers.
samples = sampler.get_chain(discard=1000, flat=True)
fig_corner = corner.corner(samples, labels=labels, show_titles=True)
plt.show()
