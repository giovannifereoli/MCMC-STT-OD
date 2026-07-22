"""
Adaptive Domain Splitting (ADS) variant of STTPropagatorND.

`STTPropagatorADS` is a DROP-IN subclass of `STTPropagatorND`: same
constructor signature and same public methods (`propagate`,
`propagate_deviation`, `propagate_covariance`, `propagate_state_only`,
...), so any script written against `STTPropagatorND` keeps working.

What it adds
------------
Given an initial uncertainty box around x0 (the `domain` half-widths),
`propagate` PRECOMPUTES an ADS decomposition of that box:

  - it recursively splits the box along the direction of worst
    linearization error (an ADS-style criterion), and
  - for every resulting patch it propagates a REFERENCE trajectory and
    its full State Transition Tensors (the SAME variational system as
    STTPropagatorND: state + A(x,t)Phi + B_k terms), all anchored at t0.

So after `propagate`, the object "knows the STTs at t0" for a whole
patchwork of local references, not just one global reference.

`propagate_deviation(sol, stts, delta_x0)` then routes delta_x0 to the
patch that contains it and expands about that patch's centre with that
patch's STTs, instead of about the single global reference. For a large
deviation the patch centre is closer, so the local expansion is far more
accurate -- that is the entire point of ADS.

Relationship to daceypy ADS
---------------------------
daceypy's ADS splits ONLINE during a Differential-Algebra propagation.
Here the dynamics are the ordinary numpy `f_func(*x, t)` you already use,
and each patch gets the SAME variational STT propagation as your base
class. Only the split DECISION uses the true (nonlinear) flow: a patch
is split when the true propagation of its edges departs from the patch's
STM prediction by more than `ads_tol`. The patches, like in daceypy ADS,
partition the INITIAL domain at t0; each patch gets its own from-t0
reference + STTs.

Safety / equivalence
--------------------
If `domain is None` (or all-zero), no split is ever made: there is a
single patch equal to x0 and `STTPropagatorADS` reproduces
`STTPropagatorND` bit-for-bit. ADS only activates when you pass a domain.

Cost note
---------
Every patch runs the full variational STT propagation, so the precompute
cost is (number of patches) x (one STTPropagatorND.propagate). When the
Jacobian `A_func` is expensive (e.g. a large lambdified spherical
harmonic field) keep the patch count modest: restrict `ads_axes` to the
dynamically relevant components (position/velocity) and choose `ads_tol`
no tighter than the accuracy you actually need.

Using it in a script (e.g. J1_scenarioA_..._hierSH)
---------------------------------------------------
    prop = STTPropagatorADS(
        order=stt_order, f_func=f_func, A_func=A_func, B_funcs=B_funcs,
        n=12,
        domain=np.concatenate([sig_prior_r, sig_prior_v,
                               [sig_prior_mu], sig_prior_c]),  # 1-sigma box
        ads_tol=0.25,             # split when position error over a patch > 0.25 km
        ads_error_idx=(0, 1, 2),  # criterion measured on position (km)
        ads_axes=(0, 1, 2, 3, 4, 5),  # only split position/velocity
    )
    sol, stts = prop.propagate(x0_ref, tau_full, rtol=1e-8, atol=1e-10,
                               method="LSODA")
    delta, x_est = prop.propagate_deviation(sol, stts, delta_x0)  # routed
"""

import numpy as np

from STTPropagationND import STTPropagatorND


class STTPropagatorADS(STTPropagatorND):
    """ADS drop-in for STTPropagatorND (see module docstring)."""

    def __init__(
        self,
        order,
        f_func,
        A_func,
        B_funcs=None,
        n=None,
        domain=None,
        ads_tol=1.0,
        ads_max_splits=6,
        ads_max_patches=256,
        ads_error_idx=(0, 1, 2),
        ads_axes=None,
        impact_radius=None,
        escape_radius=None,
        pos_idx=(0, 1, 2),
        max_step=None,
        verbose=True,
    ):
        """
        Args (in addition to STTPropagatorND):
            domain: array-like (n,) of HALF-WIDTHS of the initial
                uncertainty box around x0 (same units as the state). Axes
                with 0 are treated as certain and never split. If None,
                the class behaves exactly like STTPropagatorND.
            ads_tol: split threshold. A patch is split while the max
                linearization error over it (measured on the components
                in `ads_error_idx`, in state units) exceeds this value.
            ads_max_splits: max splits along any single axis per patch.
            ads_max_patches: hard cap on the number of patches.
            ads_error_idx: state components used for the split criterion
                and its tolerance (default position 0,1,2).
            ads_axes: iterable of axis indices allowed to split
                (default: every axis with domain>0). Restrict this to the
                dynamically relevant axes to keep the patch count (and the
                cost) down.
            impact_radius: if set, patch/edge trajectories are integrated
                with a TERMINAL solve_ivp event at ||pos|| <= impact_radius.
                This is essential for a body-centred `mu/r` potential: a
                patch whose initial condition is displaced into (or onto a
                near-collision arc with) the body would otherwise drive the
                integrator step size to zero at the r->0 singularity and
                HANG. With the event, such an integration stops cleanly and
                the offending patch/edge is rejected instead of hanging.
            escape_radius: if set, a terminal event at ||pos|| >=
                escape_radius bounds runaway/hyperbolic patches the same way.
            pos_idx: state indices holding position (used by the events;
                default 0,1,2).
            max_step: optional solve_ivp max_step for the ADS integrations,
                so a fast periapsis passage cannot step over the impact
                event undetected. Only applied to the ADS precompute.
            verbose: print a short splitting log.
        """
        super().__init__(order, f_func, A_func, B_funcs, n)
        self.domain = None if domain is None else np.asarray(domain, float).reshape(-1)
        self.ads_tol = float(ads_tol)
        self.ads_max_splits = int(ads_max_splits)
        self.ads_max_patches = int(ads_max_patches)
        self.ads_error_idx = list(ads_error_idx)
        self.ads_axes = None if ads_axes is None else np.asarray(ads_axes, int)
        self.impact_radius = None if impact_radius is None else float(impact_radius)
        self.escape_radius = None if escape_radius is None else float(escape_radius)
        self.pos_idx = list(pos_idx)
        self.max_step = None if max_step is None else float(max_step)
        self.verbose = bool(verbose)

        # filled by propagate()
        self._patches = None        # list of dicts: center, half, sol, stts
        self._global_patch = None   # the centre-0 reference, kept as fallback
        self._active = None         # axes eligible to split
        self._n_run = None

    # ------------------------------------------------------------------
    # Integration guards (prevent the r->0 singularity hang)
    # ------------------------------------------------------------------
    def _make_events(self):
        """Terminal solve_ivp events for impact / escape, or None.

        The event value is measured on ||Y[pos_idx]||, so it works for both
        the state-only integration (Y is the state) and the full variational
        integration (Y[:n] is the state, Y[pos_idx] its position).
        """
        if self.impact_radius is None and self.escape_radius is None:
            return None
        idx = list(self.pos_idx)
        events = []
        if self.impact_radius is not None:
            r_imp = self.impact_radius

            def impact(t, Y, r_imp=r_imp, idx=idx):
                p = Y[idx]
                return float(np.sqrt(np.dot(p, p)) - r_imp)

            impact.terminal = True
            impact.direction = -1.0   # only when crossing inward
            events.append(impact)
        if self.escape_radius is not None:
            r_esc = self.escape_radius

            def escape(t, Y, r_esc=r_esc, idx=idx):
                p = Y[idx]
                return float(np.sqrt(np.dot(p, p)) - r_esc)

            escape.terminal = True
            escape.direction = 1.0    # only when crossing outward
            events.append(escape)
        return events

    def _ic_in_domain(self, x):
        """False if the initial position is already inside the impact radius
        or already beyond the escape radius.

        A terminal event only fires on a *crossing*, so an IC that STARTS out
        of range would never trigger it and would drive the integrator into
        the r->0 singularity. Reject those up front (no integration at all).
        """
        if self.impact_radius is None and self.escape_radius is None:
            return True
        p = np.asarray(x, float).reshape(-1)[self.pos_idx]
        r = float(np.sqrt(np.dot(p, p)))
        if self.impact_radius is not None and r <= self.impact_radius:
            return False
        if self.escape_radius is not None and r >= self.escape_radius:
            return False
        return True

    def _augment_opts(self, opts):
        """Add the terminal events / max_step to the solve_ivp options."""
        o = dict(opts)
        events = self._make_events()
        if events is not None:
            o["events"] = events
        if self.max_step is not None:
            o.setdefault("max_step", self.max_step)
        return o

    @staticmethod
    def _reached_end(sol, t_eval):
        """True if the integration covered the full t_eval window.

        A terminal event (impact/escape) sets sol.status=1 and truncates the
        returned samples; such a trajectory is not a usable patch reference.
        """
        n_want = np.asarray(t_eval, float).reshape(-1).size
        return getattr(sol, "status", 0) == 0 and sol.t.size >= n_want

    # ------------------------------------------------------------------
    # ADS construction
    # ------------------------------------------------------------------
    def _splits_along(self, axis, half):
        """How many times `axis` has been halved from the full domain."""
        d = self.domain[axis]
        if axis < 0 or half[axis] <= 0 or d <= 0:
            return 0
        return int(round(np.log2(d / half[axis])))

    def _worst_axis(self, x0, center, half, sol_c, stts_c, t_eval, opts):
        """
        ADS-style split criterion.

        For each eligible axis j, propagate the true (nonlinear) flow of
        the patch edge  x0 + center + half_j e_j  and compare its final
        state with the STM prediction  x_c(tf) + Phi(tf)(half_j e_j). The
        mismatch on `ads_error_idx` is the linearization error along j.
        (Only the split DECISION uses this; the patch STTs are the full
        variational tensors.)

        Returns (worst_axis, worst_error).
        """
        n = self._n_run
        Phi_tf = stts_c[1][-1]                       # (n, n)
        x_c_tf = sol_c.y[:n, -1]                      # patch reference at tf
        eidx = self.ads_error_idx
        worst_axis, worst = -1, 0.0
        for j in self._active:
            hj = half[j]
            if hj <= 0:
                continue
            e_j = np.zeros(n)
            e_j[j] = hj
            if not self._ic_in_domain(x0 + center + e_j):
                # Edge IC already inside body / past escape: maximally
                # nonlinear, split toward it. No integration attempted.
                if np.inf > worst:
                    worst, worst_axis = np.inf, j
                continue
            sol_p = self.propagate_state_only(
                x0 + center + e_j, t_eval, show_progress=False, **opts
            )
            if not self._reached_end(sol_p, t_eval):
                # Edge left the physical domain (impact/escape): the linear
                # model is meaningless there. Flag it as maximally nonlinear
                # so the box is split toward this axis (bringing the edge in).
                err = np.inf
            else:
                x_true_tf = sol_p.y[:, -1]
                x_lin_tf = x_c_tf + Phi_tf @ e_j
                err = np.linalg.norm((x_true_tf - x_lin_tf)[eidx])
            if err > worst:
                worst, worst_axis = err, j
        return worst_axis, worst

    def _build_patches(self, x0, t_eval, opts):
        n = self._n_run
        if self.ads_axes is not None:
            self._active = self.ads_axes
        else:
            self._active = np.where(self.domain > 0)[0]

        # Impact/escape terminal events + max_step so no single integration
        # can hang on the r->0 singularity (see _make_events).
        opts = self._augment_opts(opts)

        patches = []
        stack = [(np.zeros(n), self.domain.copy())]   # (center, half)
        while stack:
            center, half = stack.pop()

            # Reject up front if the patch-centre IC is already inside the body
            # / past escape (a terminal event would never fire -> hang).
            if not self._ic_in_domain(x0 + center):
                if self.verbose:
                    print(
                        "[ADS] reject patch (centre IC out of physical range); "
                        f"center offset on active axes = {center[self._active]}"
                    )
                continue

            # full variational STT propagation for this patch reference
            sol_c, stts_c = super().propagate(
                x0 + center, t_eval, show_progress=False, **opts
            )

            # Reject any patch whose reference left the physical domain
            # (impact/escape truncated the integration): it cannot serve as a
            # basis over the full window. Dropping it is safe -- deviations
            # routed there fall back to the nearest surviving patch centre.
            if not self._reached_end(sol_c, t_eval):
                if self.verbose:
                    print(
                        f"[ADS] reject patch (reference left physical domain "
                        f"at t={sol_c.t[-1]:.1f}); center offset on active "
                        f"axes = {center[self._active]}"
                    )
                continue

            # stop splitting if we hit the patch cap
            if len(patches) + len(stack) + 1 >= self.ads_max_patches:
                patches.append(dict(center=center, half=half, sol=sol_c, stts=stts_c))
                continue

            axis, err = self._worst_axis(x0, center, half, sol_c, stts_c, t_eval, opts)
            can_split = (
                axis >= 0
                and err > self.ads_tol
                and self._splits_along(axis, half) < self.ads_max_splits
            )
            if can_split:
                h = half.copy()
                h[axis] *= 0.5
                c1 = center.copy()
                c1[axis] -= h[axis]
                c2 = center.copy()
                c2[axis] += h[axis]
                stack.append((c1, h))
                stack.append((c2, h))
                if self.verbose:
                    print(
                        f"[ADS] split axis {axis} (err {err:.3e} > tol "
                        f"{self.ads_tol:.3e}); final={len(patches)}, "
                        f"queue={len(stack)}"
                    )
            else:
                patches.append(dict(center=center, half=half, sol=sol_c, stts=stts_c))

        self._patches = patches
        if self.verbose:
            print(f"[ADS] precomputed {len(patches)} patch reference(s) + STT(s)")

    def _find_patch(self, dx):
        """Route dx to the patch whose CENTRE is nearest (normalised by the
        domain half-widths).

        The global centre-0 reference is always one of the candidates (added
        in `propagate`), so a deviation near the nominal routes to the EXACT
        global expansion rather than an off-centre leaf. This is the crucial
        property that keeps ADS from ever being worse than the single global
        expansion near x0: at dx=0 the nearest centre is 0, dloc=0, and
        propagate_deviation reproduces the nominal trajectory exactly. Far
        from the nominal, a split leaf centre is closer and wins, giving the
        ADS benefit. (The previous first-box-containment rule had no patch
        centred at 0 once the root was split, so it forced dx~0 onto a distant
        leaf and reintroduced the very nonlinearity ADS is meant to remove.)
        """
        act = self._active
        if act is None or not act.size:
            return self._patches[0]

        # 1) Proper ADS tiling routing: the leaf whose box CONTAINS dx (the
        #    tiling is a partition, so there is one). If dx falls in a gap
        #    (e.g. next to a rejected patch) take the nearest leaf centre,
        #    scored in units of each leaf's own half-widths.
        leaf, leaf_score = None, np.inf
        for pt in self._patches:
            c, h = pt["center"], pt["half"]
            hh = np.where(h[act] > 0, h[act], 1e-30)
            if np.all(np.abs(dx[act] - c[act]) <= h[act] + 1e-9 * (1 + h[act])):
                leaf = pt
                break
            score = float(np.max(np.abs(dx[act] - c[act]) / hh))
            if score < leaf_score:
                leaf_score, leaf = score, pt

        # 2) Fall back to the global centre-0 reference when dx is ABSOLUTELY
        #    closer to the nominal than to that leaf's centre (distances
        #    normalised per-axis by the domain half-widths so position,
        #    velocity, ... are comparable). Taylor error grows with distance
        #    from the expansion centre, so the closer centre is the better
        #    expansion. At dx=0 the global centre wins (dist 0) -> the nominal
        #    is reproduced EXACTLY; far out, the leaf centre wins -> ADS.
        if self._global_patch is not None and leaf is not None:
            scale = np.where(self.domain[act] > 0, self.domain[act], 1.0)
            dist_leaf = float(np.max(np.abs(dx[act] - leaf["center"][act]) / scale))
            dist_glob = float(np.max(np.abs(dx[act]) / scale))
            if dist_glob <= dist_leaf:
                return self._global_patch
        return leaf if leaf is not None else self._global_patch

    # ------------------------------------------------------------------
    # Public API (same signatures as STTPropagatorND)
    # ------------------------------------------------------------------
    @property
    def n_patches(self):
        return 0 if self._patches is None else len(self._patches)

    def propagate(self, x0, t_eval, show_progress=True, compute_stt=True, **options):
        """
        Same contract as STTPropagatorND.propagate: returns (sol, stts)
        for the GLOBAL reference x0. As a side effect, when `domain` is
        set and compute_stt is True, it also precomputes the ADS patch
        decomposition (references + STTs) used by propagate_deviation.
        """
        sol, stts = super().propagate(
            x0, t_eval, show_progress=show_progress, compute_stt=compute_stt, **options
        )

        self._n_run = self._infer_n(x0)
        self._global_patch = None
        if self.domain is not None and compute_stt:
            if self.domain.size != self._n_run:
                raise ValueError(
                    f"domain length {self.domain.size} != state dimension {self._n_run}"
                )
            self._build_patches(np.asarray(x0, float).reshape(-1), t_eval, dict(options))
            if not self._patches:
                # Every candidate patch left the physical domain. Fall back to
                # the single global reference (the base-class expansion), which
                # is exactly the un-split STTPropagatorND behaviour.
                if self.verbose:
                    print("[ADS] all patches rejected; using single global "
                          "reference (== STTPropagatorND).")
                self._active = np.array([], dtype=int)
                self._patches = [
                    dict(
                        center=np.zeros(self._n_run),
                        half=np.zeros(self._n_run),
                        sol=sol,
                        stts=stts,
                    )
                ]
            else:
                # Keep the global centre-0 reference as a fallback (NOT part of
                # the leaf tiling). _find_patch routes a deviation near the
                # nominal to it, so ADS never does worse than the single global
                # expansion near x0 -- it only adds local references that help
                # FAR from the nominal.
                self._global_patch = dict(
                    center=np.zeros(self._n_run),
                    half=self.domain.copy(),
                    sol=sol,
                    stts=stts,
                )
        else:
            # single global patch -> identical to STTPropagatorND
            self._active = np.array([], dtype=int)
            self._patches = [
                dict(
                    center=np.zeros(self._n_run),
                    half=np.zeros(self._n_run),
                    sol=sol,
                    stts=stts,
                )
            ]

        return sol, stts

    def propagate_deviation(self, sol, stts, delta_x0):
        """
        Same contract as the base method, but the expansion is taken
        about the ADS patch containing delta_x0 (about x0 if unsplit),
        using that patch's full STTs.

        Returns (delta, x_est) where x_est is the ADS estimate of the
        perturbed trajectory and delta = x_est - x_nom(global).
        """
        delta_x0 = np.asarray(delta_x0, float).reshape(-1)
        if not self._patches:
            return super().propagate_deviation(sol, stts, delta_x0)

        patch = self._find_patch(delta_x0)
        dloc = delta_x0 - patch["center"]
        # expand about the patch reference using its STTs (order-aware)
        _, x_est = super().propagate_deviation(patch["sol"], patch["stts"], dloc)

        # report delta relative to the GLOBAL nominal for API compatibility
        n = stts[1].shape[1]
        x_nom_global = sol.y[:n, :].T
        delta = x_est - x_nom_global
        return delta, x_est

    def patch_of(self, delta_x0):
        """Return (center, half) of the patch a deviation is routed to."""
        p = self._find_patch(np.asarray(delta_x0, float).reshape(-1))
        return p["center"], p["half"]


# ======================================================================
# Self-test / demo: 3D two-body
#   (1) unsplit STTPropagatorADS reproduces STTPropagatorND exactly
#   (2) with a large uncertainty box, ADS patches beat the single expansion
# ======================================================================
if __name__ == "__main__":
    from scipy.integrate import solve_ivp

    MU = 1.0

    def f_func(x, y, z, vx, vy, vz, t):
        r3 = (x * x + y * y + z * z) ** 1.5
        return [vx, vy, vz, -MU * x / r3, -MU * y / r3, -MU * z / r3]

    def A_func(x, y, z, vx, vy, vz, t):
        r = np.sqrt(x * x + y * y + z * z)
        rvec = np.array([x, y, z])
        G = 3.0 * np.outer(rvec, rvec) / r ** 5 - np.eye(3) / r ** 3   # d(acc)/d(pos)
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = G
        return A

    # second-order tensor B2[i,j,k] = d^2 f_i / dx_j dx_k, for order-2 STTs
    def B2_func(x, y, z, vx, vy, vz, t):
        r2 = x * x + y * y + z * z
        r = np.sqrt(r2)
        rv = np.array([x, y, z])
        B = np.zeros((6, 6, 6))
        for a in range(3):          # acceleration component (rows 3..5)
            for i in range(3):      # d/dpos_i
                for j in range(3):  # d/dpos_j
                    di = 1.0 if a == i else 0.0
                    dj = 1.0 if a == j else 0.0
                    dij = 1.0 if i == j else 0.0
                    val = (
                        -3.0 * (di * rv[j] + dj * rv[i] + dij * rv[a]) / r ** 5
                        + 15.0 * rv[a] * rv[i] * rv[j] / r ** 7
                    )
                    B[3 + a, i, j] = val
        return B

    def truth_state(ic, t_eval):
        def rhs(t, s):
            r3 = (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) ** 1.5
            return [s[3], s[4], s[5], -s[0] / r3, -s[1] / r3, -s[2] / r3]
        return solve_ivp(rhs, (t_eval[0], t_eval[-1]), ic, method="DOP853",
                         rtol=1e-12, atol=1e-12, t_eval=t_eval).y.T

    ecc = 0.3
    vp = np.sqrt((1 + ecc) / (1 - ecc))
    x0 = np.array([1 - ecc, 0.0, 0.0, 0.0, vp, 0.0])
    n_rev = 2
    t_eval = np.linspace(0.0, n_rev * 2 * np.pi, 120)
    opts = dict(rtol=1e-11, atol=1e-12, method="LSODA")
    ORDER = 2
    Bf = {2: B2_func}

    print("=" * 66)
    print("(1) unsplit ADS == base STTPropagatorND (full STTs)")
    print("=" * 66)
    base = STTPropagatorND(order=ORDER, f_func=f_func, A_func=A_func, B_funcs=Bf, n=6)
    ads_off = STTPropagatorADS(order=ORDER, f_func=f_func, A_func=A_func, B_funcs=Bf,
                               n=6, domain=None, verbose=False)
    sol_b, stts_b = base.propagate(x0, t_eval, show_progress=False, **opts)
    sol_a, stts_a = ads_off.propagate(x0, t_eval, show_progress=False, **opts)
    dx = np.array([1e-3, 0, 0, 0, 0, 0])
    db, _ = base.propagate_deviation(sol_b, stts_b, dx)
    da, _ = ads_off.propagate_deviation(sol_a, stts_a, dx)
    print(f"  max |delta_base - delta_ads| = {np.abs(db - da).max():.3e} "
          "(should be ~0)")

    print()
    print("=" * 66)
    print("(2) large uncertainty box: ADS patches beat the single expansion")
    print("=" * 66)
    # position uncertainty half-width 0.03 (3% of a) -> nonlinear over 2 revs
    domain = np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.0])
    ads = STTPropagatorADS(order=ORDER, f_func=f_func, A_func=A_func, B_funcs=Bf,
                           n=6, domain=domain, ads_tol=2e-2, ads_max_splits=6,
                           ads_error_idx=(0, 1, 2), ads_axes=(0, 1, 2),
                           verbose=True)
    sol_g, stts_g = ads.propagate(x0, t_eval, show_progress=False, **opts)
    print(f"[ADS] number of patches: {ads.n_patches}")

    rng = np.random.default_rng(0)
    samples = rng.uniform(-1, 1, size=(200, 3)) * domain[:3]
    err_single, err_ads = [], []
    for s in samples:
        dx0 = np.array([s[0], s[1], s[2], 0, 0, 0])
        xt = truth_state(x0 + dx0, t_eval)[-1, :3]
        _, xe_single = base.propagate_deviation(sol_b, stts_b, dx0)   # global STTs
        _, xe_ads = ads.propagate_deviation(sol_g, stts_g, dx0)       # ADS-routed
        err_single.append(np.linalg.norm(xe_single[-1, :3] - xt))
        err_ads.append(np.linalg.norm(xe_ads[-1, :3] - xt))
    err_single, err_ads = np.array(err_single), np.array(err_ads)
    print(f"  final position error over {len(samples)} samples "
          f"(both use order-{ORDER} STTs):")
    print(f"    single global expansion : max {err_single.max():.3e}, "
          f"mean {err_single.mean():.3e}")
    print(f"    ADS patch expansions    : max {err_ads.max():.3e}, "
          f"mean {err_ads.mean():.3e}")
    print(f"    improvement factor (mean): {err_single.mean() / err_ads.mean():.1f}x")
