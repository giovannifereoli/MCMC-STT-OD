import numpy as np
from scipy.stats import norm
from MCMC import MCMCModel  # adjust filename if needed

# === Problem setup ===
np.random.seed(42)

N = 100  # number of time steps
n_state = 2  # dimension of the state
A = np.array([[0.9, 0.3], [-0.1, 0.95]])  # stable dynamics matrix
sigma_e = 0.1  # measurement noise std

# === True initial state and propagation ===
x0_true = np.array([1.0, -0.5])
x_true = np.zeros((N, n_state))
x_true[0] = x0_true
for k in range(1, N):
    x_true[k] = A @ x_true[k - 1]

# === Observation model: observe first component via nonlinear sensor ===
y_obs = np.sin(x_true[:, 0]) + np.random.normal(0, sigma_e, size=N)


# === Residual function: only a function of x0 ===
def residuals(x0_est):
    x_est = np.zeros((N, n_state))
    x_est[0] = x0_est
    for k in range(1, N):
        x_est[k] = A @ x_est[k - 1]
    residuals = (y_obs - np.sin(x_est[:, 0])) / sigma_e
    return residuals


# === Prior: wide Gaussian on x0 ===
priors = [norm(loc=0.0, scale=1.0) for _ in range(n_state)]

# === Initial guess for x0 ===
initial_guess = np.zeros(n_state)

# === Run MCMC ===
model = MCMCModel(
    residuals_func=residuals,
    initial_params=initial_guess,
    param_priors=priors,
    observed_data=y_obs,
)


model.run(n_samples=5000, n_walkers=40, burn_in=1000)
# model.run_hmc(n_samples=3000)
model.plot_convergence()
model.plot_postfit_residuals()
model.plot_log_likelihood()
model.summary()
