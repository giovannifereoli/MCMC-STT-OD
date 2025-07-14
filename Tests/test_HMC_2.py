import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import time

# Synthetic Data
np.random.seed(42)
n_data = 50
x_data = np.linspace(0, 5, n_data)
true_theta = np.array([2.0, 3.0, 0.5])  # amplitude, frequency, phase
noise_std = 0.2
y_data = true_theta[0] * np.sin(
    true_theta[1] * x_data + true_theta[2]
) + np.random.normal(0, noise_std, n_data)


# Nonlinear Model
def model(x, theta):
    return theta[0] * np.sin(theta[1] * x + theta[2])


def residuals(theta):
    return y_data - model(x_data, theta)


def U(theta):
    res = residuals(theta)
    return 0.5 * np.sum(res**2) / noise_std**2  # -log likelihood


def grad_U(theta):
    a, b, c = theta
    x = x_data
    sin_term = np.sin(b * x + c)
    cos_term = np.cos(b * x + c)
    r = residuals(theta)

    df_da = sin_term
    df_db = a * x * cos_term
    df_dc = a * cos_term

    J = np.vstack([df_da, df_db, df_dc]).T
    grad = -J.T @ r / noise_std**2
    return grad


# Leapfrog Integrator
def leapfrog(theta, p, epsilon, L):
    theta_new = theta.copy()
    p_new = p.copy()

    p_new -= 0.5 * epsilon * grad_U(theta_new)
    for _ in range(L):
        theta_new += epsilon * p_new
        if _ != L - 1:
            p_new -= epsilon * grad_U(theta_new)
    p_new -= 0.5 * epsilon * grad_U(theta_new)

    return theta_new, p_new


# HMC Sampler
def hmc_sampler(initial_theta, n_samples=2000, L=25, epsilon=0.01):
    samples = []
    theta = initial_theta.copy()

    print(f"Starting HMC with {n_samples} samples")
    t_start = time.time()

    for i in range(n_samples):
        p0 = np.random.randn(*theta.shape)
        theta_new, p_new = leapfrog(theta, p0, epsilon, L)

        def H(theta, p):
            return U(theta) + 0.5 * np.sum(p**2)

        H_current = H(theta, p0)
        H_proposed = H(theta_new, p_new)

        accept_prob = np.exp(H_current - H_proposed)
        accepted = False
        if np.random.rand() < accept_prob:
            theta = theta_new
            accepted = True
        samples.append(theta.copy())

        if i % 100 == 0 or i == n_samples - 1:
            print(
                f"Iter {i+1:4d}: U={U(theta):.3f}, accepted={accepted}, theta={theta}"
            )

    print(f"HMC finished in {time.time() - t_start:.1f} seconds")
    return np.array(samples)


# Run HMC
initial_theta = np.array([1.0, 1.0, 1.0])
samples = hmc_sampler(initial_theta, n_samples=2500, L=25, epsilon=0.01)
samples_burned = samples[500:]  # discard burn-in

# Corner Plot using ArviZ
posterior = {
    "a": samples_burned[:, 0],
    "b": samples_burned[:, 1],
    "c": samples_burned[:, 2],
}
az_data = az.from_dict(posterior=posterior)
az.plot_pair(az_data, kind="kde", marginals=True)
plt.suptitle("Posterior Corner Plot (ArviZ)", fontsize=14)
plt.tight_layout()
plt.show()

#  Plot Fit
theta_mean = np.mean(samples_burned, axis=0)
y_fit = model(x_data, theta_mean)

plt.figure(figsize=(8, 4))
plt.scatter(x_data, y_data, label="Noisy data", color="black")
plt.plot(x_data, y_fit, label="Mean HMC fit", linewidth=2)
plt.plot(x_data, model(x_data, true_theta), label="True model", linestyle="--")
plt.legend()
plt.title("Nonlinear Model Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

#  Plot Trace of Parameters
param_names = ["a (amplitude)", "b (frequency)", "c (phase)"]
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

for i in range(3):
    axes[i].plot(samples[:, i], alpha=0.7)
    axes[i].axvline(
        500, color="red", linestyle="--", label="Burn-in" if i == 0 else None
    )
    axes[i].set_ylabel(param_names[i])
    axes[i].grid(True)
axes[-1].set_xlabel("Iteration")
axes[0].legend(loc="upper right")
fig.suptitle("Trace Plot of Parameters Over HMC Iterations", fontsize=14)
plt.tight_layout()
plt.show()

"""'
# ==== Benchmark-Only MCMC ====
def benchmark_mcmc(initial_theta, n_samples=20000, proposal_std=2.0):
    theta = initial_theta.copy()
    t_start = time.time()
    for _ in range(n_samples):
        theta_prop = theta + np.random.normal(0, proposal_std, size=theta.shape)
        U_current = U(theta)
        U_prop = U(theta_prop)
        accept_prob = np.exp(U_current - U_prop)
        if np.random.rand() < accept_prob:
            theta = theta_prop
    return time.time() - t_start, theta

# ==== Run MCMC Just to Time It ====
mcmc_time, theta_mcmc = benchmark_mcmc(initial_theta)

# ==== Report CPU Time ====
print("\n==== Benchmark: HMC vs MCMC ====")
print(f"HMC  time (full sampling): {hmc_time:.4f} seconds")
print(f"MCMC time (benchmark only): {mcmc_time:.4f} seconds")
print(f"Final theta from MCMC: {theta_mcmc}, HMC: {samples_hmc[-1]}\n")
"""

# ==== Store HMC Time ====
start_hmc = time.time()
samples_hmc = hmc_sampler(initial_theta, n_samples=2500, L=25, epsilon=0.01)
hmc_time = time.time() - start_hmc

"""
import pymc as pm
import aesara.tensor as at

# ==== Run NUTS via PyMC ====
print("Starting NUTS sampling via PyMC...")
with pm.Model() as model:
    # Priors
    a = pm.Normal("a", mu=0, sigma=10)
    b = pm.Normal("b", mu=0, sigma=10)
    c = pm.Normal("c", mu=0, sigma=10)

    # Model prediction
    mu = a * at.sin(b * x_data + c)

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=noise_std, observed=y_data)

    # Sample using NUTS
    start_nuts = time.time()
    trace_nuts = pm.sample(
        draws=2500, tune=500, target_accept=0.9, chains=1, progressbar=True
    )
    nuts_time = time.time() - start_nuts

# ==== Posterior Analysis ====
az.plot_trace(trace_nuts)
plt.suptitle("Trace Plots (PyMC NUTS)", fontsize=14)
plt.tight_layout()
plt.show()

az.plot_pair(trace_nuts, var_names=["a", "b", "c"], kind="kde", marginals=True)
plt.suptitle("Posterior Corner Plot (NUTS)", fontsize=14)
plt.tight_layout()
plt.show()

# ==== Compare CPU Time ====
print("\n==== Benchmark: HMC vs MCMC vs NUTS ====")
print(f"HMC  time (full sampling): {hmc_time:.4f} seconds")
# print(f"MCMC time (benchmark only): {mcmc_time:.4f} seconds")
print(f"NUTS time (via PyMC):        {nuts_time:.4f} seconds")
"""
