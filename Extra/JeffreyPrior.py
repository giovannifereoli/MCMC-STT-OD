import numpy as np


class JeffreysPrior:
    def __init__(self, lower=1e-12, upper=1e12):
        self.lower = lower
        self.upper = upper

    def logpdf(self, x):
        if x <= self.lower or x >= self.upper:
            return -np.inf
        return -np.log(x)

    def rvs(self):
        # Sample uniformly in log-space
        u = np.random.uniform(np.log(self.lower), np.log(self.upper))
        return np.exp(u)

    def std(self):
        # Approximate std for whitening (bounded Jeffreys prior)
        from scipy.integrate import quad

        def mean_integrand(x):
            return x * (1 / x)  # Flat in log-space

        def var_integrand(x, mean_val):
            return (x - mean_val) ** 2 * (1 / x)

        mean, _ = quad(mean_integrand, self.lower, self.upper)
        mean /= np.log(self.upper / self.lower)

        var, _ = quad(var_integrand, self.lower, self.upper, args=(mean,))
        var /= np.log(self.upper / self.lower)

        return np.sqrt(var)
