"""
Gaussian mixtures of an MCMC posterior and their propagation with first-order STMs.

    from GMM import GMM, propagate_samples_mc, position_prediction_history

    gmm = GMM.fit(model.samples, n_components=8)            # posterior split (deltas)
    one = GMM.gaussian(samples.mean(0), np.cov(samples.T))   # single-Gaussian baseline
    gmm_t = gmm.propagate_stm(propagator, x0_ref, t_eval)    # one GMM per output epoch
    m, P = gmm_t[-1].marginal(slice(0, 3)).moments()         # position mean / covariance

The figures of these results (plot_gmm_propagation, plot_prediction_history) are in
MCMCPlots.py.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# Optional dependency (the heyoka conda env lacks it): scikit-learn for the EM fit.
# Without it a numpy EM with the same settings is used.
try:
    from sklearn.mixture import GaussianMixture as _SkGaussianMixture
except ImportError:
    _SkGaussianMixture = None


class GMM:
    """
    Full-covariance Gaussian mixture  p(x) = sum_k w_k N(x; mu_k, P_k).

    Components are kept in the order given (fit() returns them sorted by decreasing
    weight). The weights are not renormalized.

    Parameters
    ----------
    weights : array_like, shape (K,)
        Component weights (sum to 1).
    means : array_like, shape (K, n)
        Component means.
    covs : array_like, shape (K, n, n) or (n, n)
        Component covariances; a single (n, n) matrix is read as K = 1.

    Attributes
    ----------
    weights : ndarray, shape (K,)
    means : ndarray, shape (K, n)
    covs : ndarray, shape (K, n, n)
    n_components : int
        K (also len(gmm)).
    dim : int
        n.

    Raises
    ------
    ValueError
        If the shapes of weights, means and covs do not agree.
    """

    def __init__(self, weights, means, covs):
        """Store the components; see the class docstring for the parameters."""
        self.weights = np.atleast_1d(np.asarray(weights, dtype=float))
        self.means = np.atleast_2d(np.asarray(means, dtype=float))
        covs = np.asarray(covs, dtype=float)
        self.covs = covs[None] if covs.ndim == 2 else covs
        K, n = self.means.shape
        if self.weights.shape != (K,) or self.covs.shape != (K, n, n):
            raise ValueError(
                f"inconsistent shapes: weights {self.weights.shape}, means {self.means.shape},"
                f" covs {self.covs.shape}"
            )

    @property
    def n_components(self):
        """Number of components K."""
        return self.weights.size

    @property
    def dim(self):
        """Dimension n of the state."""
        return self.means.shape[1]

    def __len__(self):
        """Number of components K."""
        return self.n_components

    def __repr__(self):
        """Short form 'GMM(K=.., n=..)'."""
        return f"GMM(K={self.n_components}, n={self.dim})"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def gaussian(cls, mean, cov):
        """
        Single Gaussian N(mean, cov) as a one-component mixture.

        Parameters
        ----------
        mean : array_like, shape (n,)
        cov : array_like, shape (n, n)

        Returns
        -------
        GMM
            K = 1, weight 1.
        """
        return cls(np.ones(1), np.asarray(mean, dtype=float)[None], np.asarray(cov)[None])

    @classmethod
    def fit(cls, samples, n_components=8, select_bic=False, k_max=12, random_state=0):
        """
        Split posterior samples into a full-covariance Gaussian mixture (EM).

        The coordinates span several orders of magnitude (km, km/s, eta), so EM runs in
        whitened coordinates z = L^{-1}(theta - m), with m the sample mean and L L^T the
        sample covariance (sklearn's reg_covar would otherwise swamp the small coordinates).
        L is built as diag(std) chol(corr), well conditioned for mixed units; a
        near-singular correlation matrix gets its eigenvalues floored at 1e-10 (printed).
        Back-mapping:

            mu_k = m + L mu_k^z,     P_k = L P_k^z L^T.

        On a near-Gaussian posterior BIC keeps very few components; a fixed K splits the
        posterior into narrower pieces, which is what keeps each component's STM
        linearization accurate over a long prediction.

        Parameters
        ----------
        samples : array_like, shape (N, n)
            Posterior samples (e.g. model.samples, deltas about a reference state).
        n_components : int
            Number of components K (ignored if select_bic).
        select_bic : bool
            Fit K = 1..k_max and keep the lowest BIC (the BICs are printed).
        k_max : int
            Largest K tried with select_bic.
        random_state : int
            Seed of the EM initialization.

        Returns
        -------
        GMM
            Components sorted by decreasing weight, covariances symmetrized.

        Notes
        -----
        EM uses scikit-learn's GaussianMixture (full covariance, reg_covar 1e-6, n_init 3,
        max_iter 500); without scikit-learn, the numpy fallback _em with the same settings
        (printed).
        """
        X = np.asarray(samples, dtype=float)
        m = X.mean(axis=0)
        # L = D chol(corr): same as chol(cov) but well conditioned for mixed units. A nearly
        # degenerate pair (|rho| -> 1 at machine precision) gets its correlation eigenvalues
        # floored so the factorization exists.
        R = np.corrcoef(X, rowvar=False)
        try:
            Lc = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(R)
            R = (vecs * np.clip(vals, 1e-10, None)) @ vecs.T
            d = np.sqrt(np.diag(R))
            Lc = np.linalg.cholesky(R / np.outer(d, d))
            print("[GMM] sample correlation is near-singular: eigenvalues floored at 1e-10")
        L = X.std(axis=0)[:, None] * Lc
        Z = np.linalg.solve(L, (X - m).T).T

        if _SkGaussianMixture is None:
            print("[GMM] scikit-learn not installed: fitting with the built-in numpy EM")

        def _fit(k):
            """(weights, means, covariances, BIC) of a K-component fit in z coordinates."""
            if _SkGaussianMixture is None:
                return _em(Z, k, reg_covar=1e-6, n_init=3, max_iter=500, seed=random_state)
            g = _SkGaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-6,
                n_init=3,
                max_iter=500,
                random_state=random_state,
            ).fit(Z)
            return g.weights_, g.means_, g.covariances_, g.bic(Z)

        if select_bic:
            fits = [_fit(k) for k in range(1, k_max + 1)]
            bics = np.array([f[3] for f in fits])
            w_z, mu_z, P_z, _ = fits[int(np.argmin(bics))]
            print(f"[GMM] BIC for K = 1..{k_max}:", np.array2string(bics, precision=1))
        else:
            w_z, mu_z, P_z, _ = _fit(n_components)

        order = np.argsort(w_z)[::-1]
        means = m + mu_z[order] @ L.T
        covs = np.einsum("ij,kjl,ml->kim", L, P_z[order], L)
        covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))
        return cls(w_z[order], means, covs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def moments(self):
        """
        Mean and covariance of the mixture (moment matching).

        Returns
        -------
        m : ndarray, shape (n,)
            sum_k w_k mu_k.
        P : ndarray, shape (n, n)
            sum_k w_k [P_k + (mu_k - m)(mu_k - m)^T]: within- plus between-component spread.
        """
        m = self.weights @ self.means
        d = self.means - m
        P = np.einsum("k,kij->ij", self.weights, self.covs) + np.einsum(
            "k,ki,kj->ij", self.weights, d, d
        )
        return m, P

    def logpdf(self, x):
        """
        Log-density of the mixture.

        Parameters
        ----------
        x : array_like, shape (M, n) or (n,)

        Returns
        -------
        ndarray, shape (M,)
            log p(x) (logsumexp over the components; singular covariances allowed).
        """
        x = np.atleast_2d(x)
        lp = np.array(
            [
                np.log(w)
                + np.atleast_1d(multivariate_normal(mu, P, allow_singular=True).logpdf(x))
                for w, mu, P in zip(self.weights, self.means, self.covs)
            ]
        )
        return logsumexp(lp, axis=0)

    def marginal(self, idx):
        """
        Marginal mixture of a subset of the coordinates (same weights).

        Parameters
        ----------
        idx : list of int or slice
            Coordinates kept, e.g. slice(0, 3) = position.

        Returns
        -------
        GMM
        """
        idx = np.arange(self.dim)[idx]
        return GMM(self.weights, self.means[:, idx], self.covs[:, idx][:, :, idx])

    def linear_map(self, A):
        """
        Image of the mixture under x -> A x (e.g. a rotation to another frame).

        Parameters
        ----------
        A : array_like, shape (m, n)

        Returns
        -------
        GMM
            Same weights, means A mu_k, covariances A P_k A^T.
        """
        A = np.asarray(A, dtype=float)
        return GMM(
            self.weights,
            self.means @ A.T,
            np.einsum("ij,kjl,ml->kim", A, self.covs, A),
        )

    def sample(self, n, rng=None):
        """
        Random draws from the mixture: multinomial component counts, then Gaussian draws.

        Parameters
        ----------
        n : int
            Number of draws.
        rng : int or numpy.random.Generator, optional
            Seed or generator (np.random.default_rng(rng)).

        Returns
        -------
        ndarray, shape (n, dim)
            Grouped by component (component 0 first), not shuffled.
        """
        rng = np.random.default_rng(rng)
        counts = rng.multinomial(n, self.weights / self.weights.sum())
        return np.vstack(
            [
                rng.multivariate_normal(self.means[k], self.covs[k], size=counts[k])
                for k in range(self.n_components)
            ]
        )

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------
    def propagate_stm(self, propagator, x0_ref, t_eval, label="GMM", **ivp):
        """
        Propagate every component with its own first-order STM.

            m_k(t) = phi(t; x0_ref + mu_k),      P_k(t) = Phi_k(t) P_k Phi_k(t)^T,

        with phi the nonlinear flow and Phi_k the STM along the trajectory of component k.
        Weights are unchanged (no re-splitting along the way).

        Parameters
        ----------
        propagator : object
            propagator.propagate(x0, t_eval, show_progress=False, **ivp) must return
            (sol, stts): sol.y the states (rows :n used, one column per output epoch) and
            stts[1] the STM history (N_t, n, n), as STTPropagatorND and its drop-ins.
        x0_ref : array_like, shape (n,)
            Reference state; the component means are deviations about it.
        t_eval : array_like, shape (N_t,)
            Output epochs.
        label : str
            Prefix of the progress and error messages.
        **ivp
            Passed to propagator.propagate (e.g. rtol, atol).

        Returns
        -------
        list of GMM
            One mixture per output epoch (N_t), with absolute (not deviation) means.

        Raises
        ------
        RuntimeError
            If a component's propagation does not reach every output epoch with finite
            states.
        """
        t_eval = np.asarray(t_eval, dtype=float)
        K, n = self.means.shape
        m_t = np.empty((t_eval.size, K, n))
        P_t = np.empty((t_eval.size, K, n, n))
        for k in range(K):
            sol, stts = propagator.propagate(
                x0_ref + self.means[k], t_eval, show_progress=False, **ivp
            )
            if not _sol_ok(sol, t_eval.size):
                raise RuntimeError(
                    f"[{label}] component {k} propagation failed: "
                    f"{getattr(sol, 'message', '')}"
                )
            Phi = stts[1]  # (N_t, n, n)
            m_t[:, k] = sol.y[:n].T
            P_t[:, k] = Phi @ self.covs[k] @ np.transpose(Phi, (0, 2, 1))
            print(
                f"[{label}] component {k + 1}/{K} (w = {self.weights[k]:.3f}) propagated",
                flush=True,
            )
        return [GMM(self.weights, m_t[j], P_t[j]) for j in range(t_eval.size)]


# ======================================================================
# Monte Carlo reference and comparison metrics
# ======================================================================


def propagate_samples_mc(propagator, x0_ref, deltas, t_eval, **ivp):
    """
    Monte Carlo reference: nonlinear state-only propagation of x0_ref + delta_i.

    Parameters
    ----------
    propagator : object
        propagator.propagate_state_only(x0, t_eval, show_progress=False, **ivp) must
        return a scipy-like solution (sol.y, status or success).
    x0_ref : array_like, shape (n,)
        Reference state.
    deltas : ndarray, shape (M, n)
        Deviations about x0_ref (e.g. posterior samples).
    t_eval : array_like, shape (N_t,)
        Output epochs.
    **ivp
        Passed to propagate_state_only.

    Returns
    -------
    ndarray, shape (M, N_t, n)
        Absolute states. Runs that fail are left NaN. Progress is printed every M // 10.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    M, n = deltas.shape
    out = np.full((M, t_eval.size, n), np.nan)
    step = max(1, M // 10)
    for i, d in enumerate(deltas):
        sol = propagator.propagate_state_only(
            x0_ref + d, t_eval, show_progress=False, **ivp
        )
        if _sol_ok(sol, t_eval.size):
            out[i] = sol.y[:n].T
        if (i + 1) % step == 0 or i + 1 == M:
            print(f"[MC] {i + 1}/{M} posterior samples propagated", flush=True)
    return out


def position_prediction_history(X_mc, gmm_t, single_t, batch_t=None):
    """
    Per-epoch position (3D) check of the propagated GMM and Gaussian vs Monte Carlo.

    Parameters
    ----------
    X_mc : ndarray, shape (M, N_t, n)
        Monte Carlo states (propagate_samples_mc), positions in [:3] [km]. Must hold no
        NaN rows.
    gmm_t, single_t : list of GMM
        The per-epoch mixtures returned by GMM.propagate_stm for the GMM and for the single
        Gaussian.
    batch_t : list of GMM, optional
        A further propagated Gaussian (e.g. the batch least-squares solution and its
        covariance), checked the same way.

    Returns
    -------
    dict of ndarray, each shape (N_t,)
        sig_mc, sig_gmm, sig_1 : sqrt(trace P_rr) [km] of the MC sample covariance, of the
        moment-matched GMM and of the single Gaussian.
        ll_gmm, ll_1 : mean log-density of the MC positions under the GMM / the single
        Gaussian position marginal (higher = better fit of the MC cloud).
        sig_batch, ll_batch : the same for batch_t (only if given).
    """
    n_t = X_mc.shape[1]
    keys = ["sig_mc", "sig_gmm", "sig_1", "ll_gmm", "ll_1"]
    if batch_t is not None:
        keys += ["sig_batch", "ll_batch"]
    out = {k: np.empty(n_t) for k in keys}
    pos_idx = slice(0, 3)
    for j in range(n_t):
        pos = X_mc[:, j, pos_idx]
        g = gmm_t[j].marginal(pos_idx)
        g1 = single_t[j].marginal(pos_idx)
        out["sig_mc"][j] = np.sqrt(np.trace(np.cov(pos, rowvar=False)))
        out["sig_gmm"][j] = np.sqrt(np.trace(g.moments()[1]))
        out["sig_1"][j] = np.sqrt(np.trace(g1.moments()[1]))
        out["ll_gmm"][j] = np.mean(g.logpdf(pos))
        out["ll_1"][j] = np.mean(g1.logpdf(pos))
        if batch_t is not None:
            gb = batch_t[j].marginal(pos_idx)
            out["sig_batch"][j] = np.sqrt(np.trace(gb.moments()[1]))
            out["ll_batch"][j] = np.mean(gb.logpdf(pos))
    return out


# ======================================================================
# Helpers
# ======================================================================


def _sol_ok(sol, n_t):
    """True if a propagation reached every output epoch (scipy-like or heyoka _Sol)."""
    status = getattr(sol, "status", 0 if getattr(sol, "success", True) else -1)
    return status == 0 and np.shape(sol.y)[1] == n_t and np.all(np.isfinite(sol.y))


def _em(Z, k, reg_covar=1e-6, n_init=3, max_iter=500, tol=1e-3, seed=0):
    """
    Full-covariance EM with the settings of sklearn's GaussianMixture.

    Fallback of GMM.fit when scikit-learn is missing: k-means++ seeding and 20 Lloyd
    iterations for the start, reg_covar on the diagonal, stop when the mean log-likelihood
    changes by less than tol, best of n_init starts.

    Parameters
    ----------
    Z : ndarray, shape (N, n)
        Data (whitened coordinates).
    k : int
        Number of components.
    reg_covar, n_init, max_iter, tol : float, int, int, float
        As in sklearn.mixture.GaussianMixture.
    seed : int
        Seed of the k-means++ start.

    Returns
    -------
    weights : ndarray, shape (k,)
    means : ndarray, shape (k, n)
    covs : ndarray, shape (k, n, n)
    bic : float
        -2 N mean_loglik + n_par ln N.
    """
    rng = np.random.default_rng(seed)
    N, n = Z.shape
    best = None
    for _ in range(n_init):
        # k-means++ seeding + Lloyd iterations -> hard responsibilities
        C = Z[[rng.integers(N)]]
        for _ in range(1, k):
            d2 = np.min(((Z[:, None, :] - C[None]) ** 2).sum(axis=2), axis=1)
            C = np.vstack([C, Z[rng.choice(N, p=d2 / d2.sum())]])
        for _ in range(20):
            lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(axis=2), axis=1)
            C = np.array(
                [Z[lab == j].mean(axis=0) if np.any(lab == j) else C[j] for j in range(k)]
            )
        resp = np.eye(k)[lab]
        ll_old = -np.inf
        for _ in range(max_iter):
            Nk = resp.sum(axis=0) + 10.0 * np.finfo(float).eps
            w = Nk / N
            mu = (resp.T @ Z) / Nk[:, None]
            P = np.empty((k, n, n))
            for j in range(k):
                D = Z - mu[j]
                P[j] = (resp[:, j, None] * D).T @ D / Nk[j] + reg_covar * np.eye(n)
            logp = np.column_stack(
                [np.log(w[j]) + multivariate_normal.logpdf(Z, mu[j], P[j]) for j in range(k)]
            )
            lse = logsumexp(logp, axis=1)
            resp = np.exp(logp - lse[:, None])
            ll = float(lse.mean())
            if abs(ll - ll_old) < tol:
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, w, mu, P)
    ll, w, mu, P = best
    n_par = (k - 1) + k * n + k * n * (n + 1) / 2.0
    return w, mu, P, -2.0 * ll * N + n_par * np.log(N)
