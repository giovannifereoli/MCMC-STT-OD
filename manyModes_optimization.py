import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import emcee

ndim = 10
nwalkers = 100
n_modes_guess = 5


# Step 1: Find modes via optimization
def neg_log_prob(x):  # minimize negative log-prob
    return -log_prob(x)


modes = []
for _ in range(50):  # run 50 optimizations
    x0 = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds])
    result = minimize(neg_log_prob, x0, method="Nelder-Mead")
    if result.success:
        modes.append(result.x)
modes = np.array(modes)

# Step 2: Cluster modes
kmeans = KMeans(n_clusters=n_modes_guess).fit(modes)
mode_centers = kmeans.cluster_centers_

# Step 3: Initialize walkers
walkers_per_mode = nwalkers // len(mode_centers)
initial_positions = []
for mode in mode_centers:
    ball = mode + 0.01 * np.random.randn(walkers_per_mode, ndim)  # small ball
    initial_positions.append(ball)
initial_positions = np.vstack(initial_positions)

# Step 4: Run emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(initial_positions, nsteps=5000, progress=True)
