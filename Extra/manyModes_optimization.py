def initialize_walkers_lhs_priorquant_full_map_unbounded(
    self,
    n_walkers=50,
    n_starts=200,
    top_k=20,
    ball_sigma=1e-2,
    seed=0,
    method="Powell",      # robust without bounds
    maxiter=500,
    u_eps=1e-10,          # keeps u away from exactly 0/1 -> avoids +/-inf for unbounded priors
    verbose=True,
):
    """
    LHS + unbounded full MAP (no clipping, no hard bounds):

    - Latin hypercube sample u in (0,1)^ndim
    - Map to parameter space via theta_i = prior_i.ppf(u_i)  (quantile transform)
    - Optimize full theta from each start (unconstrained)
    - Keep top_k best optima
    - Scatter walkers around them
    - Return pos in sampler space (whitened if self.is_whitened)
    """
    rng = np.random.default_rng(seed)

    # LHS on (0,1)^ndim
    def lhs_unit(n, d):
        u = rng.random((n, d))
        a = (np.arange(n)[:, None] + u) / n
        for j in range(d):
            rng.shuffle(a[:, j])
        return a

    U = lhs_unit(n_starts, self.ndim)
    U = np.clip(U, u_eps, 1.0 - u_eps)  # avoid ppf(0) / ppf(1)

    # Map to theta via prior ppf (no "bounds", just prior quantiles) 
    starts = np.zeros((n_starts, self.ndim), dtype=float)
    for i, p in enumerate(self.param_priors):
        starts[:, i] = p.ppf(U[:, i])

    def objective(theta):
        lp = self.log_prob(theta)
        if not np.isfinite(lp):
            return 1e300
        return -lp

    if verbose:
        print("")
        print("[Init-LHS-Unbounded] LHS in prior-quantile space + full MAP optimize each")
        print(f"[Init-LHS-Unbounded] ndim={self.ndim}, n_starts={n_starts}, top_k={top_k}")
        print(f"[Init-LHS-Unbounded] optimizer={method}, maxiter={maxiter}, u_eps={u_eps:g}")

    candidates = []
    n_fail = 0
    for x0 in starts:
        res = minimize(
            objective,
            x0,
            method=method,
            options=dict(maxiter=maxiter, disp=False),
        )
        x_opt = res.x
        lp = self.log_prob(x_opt)
        if np.isfinite(lp):
            candidates.append((lp, x_opt))
        else:
            n_fail += 1

    if len(candidates) == 0:
        raise RuntimeError(
            "Init-LHS-Unbounded failed: no finite optima found. "
            "Your posterior may be numerically unstable far from nominal."
        )

    candidates.sort(key=lambda t: t[0], reverse=True)
    seeds = np.array([x for _, x in candidates[: min(top_k, len(candidates))]], dtype=float)

    if verbose:
        print(f"[Init-LHS-Unbounded] successes={len(candidates)}, failed={n_fail}")
        print(f"[Init-LHS-Unbounded] best logpost = {candidates[0][0]:.6f}")

    # Distribute walkers across seeds
    n_seeds = seeds.shape[0]
    counts = np.full(n_seeds, n_walkers // n_seeds, dtype=int)
    counts[: (n_walkers - counts.sum())] += 1

    pos_orig = []
    for s, c in zip(seeds, counts):
        ball = s + ball_sigma * rng.standard_normal((c, self.ndim))
        pos_orig.append(ball)
    pos_orig = np.vstack(pos_orig)[:n_walkers, :]

    # Return in sampler space
    if getattr(self, "is_whitened", False):
        pos_white = (self.whiten_Linv @ (pos_orig.T - self.whiten_mean[:, None])).T
        return pos_white

    return pos_orig

# FOR MCMC CLASS
pos = self.initialize_walkers_lhs_priorquant_full_map_unbounded(
    n_walkers=n_walkers,
    n_starts=300,          # ↑ increase if you suspect multiple basins
    top_k=min(20, n_walkers),
    ball_sigma=spherical_spread,
    seed=42,
    method="Powell",       # safest without bounds
    maxiter=500,
    u_eps=1e-10,
)
