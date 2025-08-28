import numpy as np
import matplotlib.pyplot as plt
import emcee

# Toy model for asteroid internal density structures with ellipsoidal shape
# Assume triaxial ellipsoid with semi-axes a >= b >= c, fix a=1, p=b/a, q=c/a, 0 < q <= p <= 1
# Observable: normalized moment of inertia factor k = I_c / (M R_eq^2), rotation about c-axis
# Observed value with uncertainty
k_obs = 0.33  # Example observed value (indicating central density concentration)
sigma = 0.01  # Uncertainty


# Define beta(shape) = (1/5) (a^2 + b^2) / R^2 with a=1, R=(p q)^{1/3}
def beta(p, q):
    return (1.0 / 5) * (1 + p**2) / (p * q) ** (2.0 / 3)


# Compute k for each model
def compute_k(m, theta):
    p, q = theta[:2]
    b = beta(p, q)
    if m == 0:  # uniform
        return b
    alpha = theta[2]
    if m == 1:  # shell
        if alpha == 0:
            return b
        else:
            return b * (1 - alpha**5) / (1 - alpha**3)
    elif m == 2:  # mascon (dense core, gamma=5)
        gamma = 5
        return b * (1 + (gamma - 1) * alpha**5) / (1 + (gamma - 1) * alpha**3)


# Log likelihood (Gaussian)
def log_lik(k):
    return -0.5 * ((k_obs - k) / sigma) ** 2


# Log prior (uniform in regions)
def log_prior(m, theta):
    p, q = theta[:2]
    if not (0 < q <= p <= 1):
        return -np.inf
    if m == 0:
        return 0.0
    alpha = theta[2]
    if not (0 <= alpha <= 1):
        return -np.inf
    return 0.0


# Log posterior
def log_prob(theta, m):
    lp = log_prior(m, theta)
    if not np.isfinite(lp):
        return -np.inf
    k = compute_k(m, theta)
    ll = log_lik(k)
    return lp + ll


# MCMC parameters
np.random.seed(42)
nwalkers = 50
n_steps = 5000
burnin = 1000


# Function to generate initial positions
def get_initial_positions(nwalkers, ndim):
    pos = np.random.rand(nwalkers, ndim)
    pos[:, 0] = np.random.uniform(0, 1, nwalkers)  # p
    pos[:, 1] = np.random.uniform(0, 1, nwalkers) * pos[:, 0]  # q < p
    if ndim == 3:
        pos[:, 2] = np.random.uniform(0, 1, nwalkers)  # alpha
    return pos


# Run emcee for each model and store likelihoods
models = [0, 1, 2]
model_names = ["Uniform", "Shell", "Mascon"]
evidences = []
chains = []
flat_log_probs = []
max_log_liks = []

for m in models:
    ndim = 2 if m == 0 else 3
    initial_pos = get_initial_positions(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(m,))
    sampler.run_mcmc(initial_pos, n_steps, progress=False)

    # Get flat chain and log probs after burn-in
    flat_chain = sampler.get_chain(discard=burnin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burnin, flat=True)

    # Since log_prior=0 in valid region, log_prob = log_lik
    liks = np.exp(flat_log_prob)

    # Harmonic mean estimator for evidence Z
    if np.all(liks == 0):  # Avoid division by zero
        z = 0.0
    else:
        z = len(liks) / np.sum(1.0 / liks)

    # Store maximum log likelihood
    max_log_lik = np.max(flat_log_prob)

    evidences.append(z)
    chains.append(flat_chain)
    flat_log_probs.append(flat_log_prob)
    max_log_liks.append(max_log_lik)

# Compute posterior probabilities for each structure
total_evidence = np.sum(evidences)
probs = [z / total_evidence if total_evidence > 0 else 0 for z in evidences]

# Identify model with highest maximum likelihood
max_lik_index = np.argmax(max_log_liks)
max_lik_model = model_names[max_lik_index]
max_lik_value = max_log_liks[max_lik_index]

# Print results
print("Posterior probabilities for each structure:")
for name, prob in zip(model_names, probs):
    print(f"  {name}: {prob:.4f}")
print(f"\nModel with highest maximum likelihood: {max_lik_model}")
print(f"Maximum log likelihood: {max_lik_value:.4f}")

# Visualize the posterior for the mascon model (as example)
if probs[2] > 0:
    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    chain = chains[2]
    labels = ["p (b/a)", "q (c/a)", r"$\alpha$ (core fraction)"]
    for i in range(3):
        axes[i].plot(chain[:, i], alpha=0.5)
        axes[i].set_ylabel(labels[i])
    axes[2].set_xlabel("Sample Index")
    plt.suptitle("MCMC Chains for Mascon Model Parameters")
    plt.tight_layout()
    plt.show()

# Histogram of k from posterior for all models
plt.figure(figsize=(8, 4))
all_ks = []
for m, chain in enumerate(chains):
    thetas = chain
    ks = np.array([compute_k(m, theta) for theta in thetas])
    all_ks.append(ks)
    plt.hist(ks, bins=50, density=True, alpha=0.5, label=model_names[m])
plt.axvline(k_obs, color="r", linestyle="--", label="Observed k")
plt.xlabel("k (Moment of Inertia Factor)")
plt.ylabel("Posterior Density")
plt.legend()
plt.title("Posterior Distribution of k for Each Model")
plt.show()
