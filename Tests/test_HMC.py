import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation
import arviz as az


#  Simulate True Dynamics (Generate Synthetic Data)
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

# Integrate dynamics
sol = solve_ivp(
    lambda t, y: two_body(t, y, mu_true),
    [t_eval[0], t_eval[-1]],
    x0_true,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-9,
)
X_true = sol.y.T  # [N x 6]
range_true = np.linalg.norm(X_true[:, :3], axis=1)
range_meas = range_true + np.random.normal(0, 5.0, size=range_true.shape)  # 5 km noise
print("Done generating synthetic data.")


#  PyMC Model
# Define the two-body dynamics for PyMC
def two_body_rhs(x, t, mu):
    # Convert mu from a 1D array to a scalar
    mu = mu[0]
    r = x[:3]
    v = x[3:]
    norm_r = pt.sqrt(pt.sum(r**2))
    a = -mu * r / norm_r**3
    return pt.concatenate([v, a])


two_body_model = DifferentialEquation(
    func=two_body_rhs, times=t_eval, n_states=6, n_theta=1, t0=0.0
)
with pm.Model() as model:
    # Priors on initial state
    x0 = pm.Normal("x0", mu=x0_true, sigma=50.0, shape=6)

    # Prior on mu
    mu = pm.Normal("mu", mu=400000.0, sigma=5000.0)  # Close to Earth's mu

    # ODE integration with corrected theta shape
    x_sol = two_body_model(y0=x0, theta=[mu])

    # Predicted ranges
    pos = x_sol[:, :3]
    pred_range = pt.sqrt(pt.sum(pos**2, axis=1))

    # Observation noise
    sigma = pm.HalfNormal("sigma", sigma=10.0)

    # Likelihood
    y = pm.Normal("y", mu=pred_range, sigma=sigma, observed=range_meas)

    # Inference
    trace = pm.sample(
        1000, tune=1000, target_accept=0.95, return_inferencedata=True, progressbar=True
    )

#  Post-analysis
az.plot_trace(trace, var_names=["mu", "x0", "sigma"])
plt.show()

print(az.summary(trace, var_names=["mu", "x0", "sigma"], round_to=3))
