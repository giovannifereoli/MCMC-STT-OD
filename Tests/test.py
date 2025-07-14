import numpy as np
from scipy.stats import uniform
from MCMC import MCMCModel  # adjust if filename is different

# === Synthetic Data: y = 2.5 * x + 1.0 + noise ===
np.random.seed(42)
true_params = [2.5, 1.0]
x_data = np.linspace(0, 10, 50)
y_true = true_params[0] * x_data + true_params[1]
y_obs = y_true + np.random.normal(0, 0.5, size=len(x_data))


# === Define residuals function ===
def residuals(theta):
    a, b = theta
    y_model = a * x_data + b
    return y_obs - y_model


# === Priors ===
priors = [uniform(loc=0.0, scale=5.0), uniform(loc=-5.0, scale=10.0)]  # for a  # for b

# === Initial guess ===
initial_guess = [1.0, 0.0]

# === Create and run model ===
model = MCMCModel(
    residuals_func=residuals,
    initial_params=initial_guess,
    param_priors=priors,
    observed_data=y_obs,
)

model.run(n_samples=3000, n_walkers=32, burn_in=500)
model.plot_convergence()
model.plot_postfit_residuals()
model.summary()
