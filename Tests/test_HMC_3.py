import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import arviz as az
import time

# ------------------ Ground Truth ------------------ #
np.random.seed(42)
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 50)
true_x0 = np.array([7000, 0, 0, 0, 7.5, 0])  # [km, km/s] - circular LEO-ish

mu = 398600.4418  # [km^3/s^2] - Earth gravitational parameter


# 2-body dynamics
def two_body_ode(t, state):
    r = state[:3]
    v = state[3:]
    norm_r = np.linalg.norm(r)
    a = -mu * r / norm_r**3
    return np.concatenate([v, a])


# Generate true trajectory
sol = solve_ivp(two_body_ode, t_span, true_x0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
true_states = sol.y.T

# Observations: range only, with noise
range_obs = np.linalg.norm(true_states[:, :3], axis=1)
range_obs += np.random.normal(0, 1e-1, size=range_obs.shape)  # add noise (100m std)

noise_std = 1e-1  # standard deviation of range noise


# ------------------ HMC Setup ------------------ #
def propagate_and_compute_range(x0):
    sol = solve_ivp(two_body_ode, t_span, x0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return np.linalg.norm(sol.y[:3].T, axis=1)


def residuals(x0):
    return range_obs - propagate_and_compute_range(x0)


def U(x0):
    res = residuals(x0)
    return 0.5 * np.sum(res**2) / noise_std**2


def grad_U(x0, epsilon=1e-8):
    grad = np.zeros_like(x0)
    for i in range(len(x0)):
        dx = np.zeros_like(x0)
        dx[i] = epsilon
        grad[i] = (U(x0 + dx) - U(x0 - dx)) / (2 * epsilon)
    return grad


# ------------------ HMC Core ------------------ #
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


def hmc_sampler(initial_theta, n_samples=500, L=10, epsilon=0.001):
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
            print(f"Iter {i+1:4d}: U={U(theta):.3f}, accepted={accepted}")

    print(f"HMC finished in {time.time() - t_start:.1f} seconds")
    return np.array(samples)


# ------------------ Run ------------------ #
initial_guess = true_x0 + np.random.randn(6) * 5.0  # Add some error
samples = hmc_sampler(initial_guess, n_samples=1500, L=20, epsilon=0.005)
samples_burned = samples[300:]

# ------------------ Plotting ------------------ #
param_names = ["x", "y", "z", "vx", "vy", "vz"]
theta_mean = np.mean(samples_burned, axis=0)
print("Estimated Initial State:", theta_mean)

fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
for i in range(6):
    axes[i].plot(samples[:, i])
    axes[i].axvline(300, color="red", linestyle="--")
    axes[i].set_ylabel(param_names[i])
axes[-1].set_xlabel("Iteration")
plt.suptitle("Trace of Estimated Initial Conditions", fontsize=16)
plt.tight_layout()
plt.show()

az_data = az.from_dict(
    posterior={k: samples_burned[:, i] for i, k in enumerate(param_names)}
)
az.plot_pair(az_data, kind="kde", marginals=True)
plt.suptitle("Posterior Corner Plot", fontsize=16)
plt.tight_layout()
plt.show()
