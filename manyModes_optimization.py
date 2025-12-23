    def initialize_walkers_lhs_full_map(
        self,
        n_walkers=50,
        n_starts=200,          # how many LHS start points
        top_k=20,              # keep best K optima as seeds
        ball_sigma=1e-2,       # scatter of walkers around each seed
        ppf_eps=1e-6,
        seed=0,
        method="L-BFGS-B",
        maxiter=300,
        verbose=True,
    ):
        """
        Features:
        - LHS sample *all* dims within prior bounds
        - run a local optimizer in full space from each start
        - keep top_k best solutions (by log posterior)
        - scatter walkers around those seeds

        Returns pos in sampler space:
        - unwhitened if self.is_whitened is False
        - whitened if self.is_whitened is True
        """
        rng = np.random.default_rng(seed)

        # Bounds from priors (original space)
        bounds = [(p.ppf(ppf_eps), p.ppf(1 - ppf_eps)) for p in self.param_priors]
        bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        for i, (lo, hi) in enumerate(bounds):
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                raise ValueError(
                    f"Invalid bounds from prior[{i}]: ({lo}, {hi}). "
                    "Check prior support / ppf."
                )

        # LHS on [0,1]^ndim then map to bounds
        def lhs_unit(n, d):
            u = rng.random((n, d))
            a = (np.arange(n)[:, None] + u) / n
            for j in range(d):
                rng.shuffle(a[:, j])
            return a

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        starts = lo + lhs_unit(n_starts, self.ndim) * (hi - lo)

        # Full-space optimization objective in ORIGINAL space
        def objective(theta):
            lp = self.log_prob(theta)
            if not np.isfinite(lp):
                return 1e100
            return -lp

        if verbose:
            print("")
            print("[Init-FullMAP] LHS over all dims + full MAP optimize each start")
            print(f"[Init-FullMAP] ndim={self.ndim}, n_starts={n_starts}, top_k={top_k}")
            print(f"[Init-FullMAP] optimizer={method}, maxiter={maxiter}")

        candidates = []  # (lp, theta_opt)
        n_fail = 0

        for x0 in starts:
            res = minimize(
                objective,
                x0,
                method=method,
                options=dict(maxiter=maxiter),
            )

            x_opt = res.x
            lp = self.log_prob(x_opt)
            if np.isfinite(lp):
                candidates.append((lp, x_opt))
            else:
                n_fail += 1

        if len(candidates) == 0:
            raise RuntimeError(
                "Init-FullMAP failed: no finite optima found. "
                "Check priors/bounds and log_prob stability."
            )

        candidates.sort(key=lambda t: t[0], reverse=True)
        seeds = np.array([x for _, x in candidates[: min(top_k, len(candidates))]], dtype=float)

        if verbose:
            print(f"[Init-FullMAP] successes={len(candidates)}, failed={n_fail}")
            print(f"[Init-FullMAP] best logpost = {candidates[0][0]:.6f}")

        # Distribute walkers across seeds
        n_seeds = seeds.shape[0]
        counts = np.full(n_seeds, n_walkers // n_seeds, dtype=int)
        counts[: (n_walkers - counts.sum())] += 1

        pos_orig = []
        for s, c in zip(seeds, counts):
            ball = s + ball_sigma * rng.standard_normal((c, self.ndim))
            for i, (l, h) in enumerate(bounds):
                ball[:, i] = np.clip(ball[:, i], l, h)
            pos_orig.append(ball)

        pos_orig = np.vstack(pos_orig)[:n_walkers, :]
        if pos_orig.shape != (n_walkers, self.ndim):
            raise RuntimeError("Init-FullMAP produced wrong shape for pos.")

        # Return in sampler space (whitened if needed)
        if getattr(self, "is_whitened", False):
            pos_white = (self.whiten_Linv @ (pos_orig.T - self.whiten_mean[:, None])).T
            return pos_white

        return pos_orig

# FOR MCMC CLASS
pos = self.initialize_walkers_lhs_full_map(
    n_walkers=n_walkers,
    n_starts=300,          # bump if you suspect many basins
    top_k=min(20, n_walkers),
    ball_sigma=spherical_spread,
    seed=42,
    method="L-BFGS-B",
    maxiter=300,
)
