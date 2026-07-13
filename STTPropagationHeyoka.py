"""
STTPropagationHeyoka
====================

A drop-in replacement for :class:`STTPropagationND.STTPropagatorND` that uses
**heyoka.py** (https://bluescarni.github.io/heyoka.py) instead of hand-derived
SymPy tensors + ``scipy.integrate.solve_ivp``.

Why heyoka
----------
heyoka integrates ODEs with a high-order, adaptive **Taylor** method and can
build the **variational equations** of a system automatically to arbitrary
order.  That gives us, for free and to machine precision:

* order 1  -> the State Transition Matrix  Phi(t, t0) = d x(t) / d x0
* order k  -> the higher-order State Transition Tensors
* the full **Taylor map** of the flow w.r.t. the initial conditions, evaluated
  natively with ``eval_taylor_map`` (this is exactly ``propagate_deviation``).

So there is no need to derive/lambdify Jacobians (``A_func``) and Hessians+
(``B_funcs``) by hand: you hand heyoka the symbolic right-hand side once.

Interface parity with STTPropagatorND
-------------------------------------
The public surface matches the SciPy version so existing scripts change only the
*construction* of the propagator:

    propagate(x0, t_eval, **opts)            -> (sol, stts)
    propagate_state_only(x0, t_eval, **opts) -> sol
    propagate_deviation(sol, stts, delta_x0) -> (delta, x_est)
    propagate_covariance(sol, stts, P0)      -> P_t

* ``sol.y``   has shape ``(n, N)`` (state history), like solve_ivp's ``sol.y``.
* ``stts[1]`` has shape ``(N, n, n)`` (the STM history), like before.
* ``propagate_deviation`` uses heyoka's Taylor map, so it is accurate to the
  integrator's variational ``order`` (for ``order == 1`` it reduces exactly to
  the first-order expansion ``x_nom + Phi @ delta``).

Legacy ``solve_ivp`` keywords (``rtol``, ``atol``, ``method``) are accepted and
ignored -- heyoka's accuracy is governed by the ``tol`` given at construction.
"""

from __future__ import annotations

import numpy as np
import heyoka as hy


class _Sol:
    """Lightweight stand-in for a solve_ivp solution (only what callers use)."""

    def __init__(self, t, y, aug, grid):
        self.t = np.asarray(t, dtype=float)  # (N,)
        self.y = np.asarray(y, dtype=float)  # (n, N)  state history
        self._aug = np.asarray(aug, dtype=float)  # (N, n_aug) full variational state
        self._grid = np.asarray(grid, dtype=float)
        self.success = True


class HeyokaSTTPropagator:
    """State Transition Tensor propagator backed by heyoka's variational ODEs.

    Parameters
    ----------
    order : int
        Maximum variational order (1 = STM only, >=2 adds higher STTs to the
        Taylor map used by ``propagate_deviation``).
    sys : list[tuple]
        heyoka ODE system: a list of ``(state_variable, rhs_expression)`` pairs,
        e.g. built by :func:`build_bennu_deg2_heyoka`.
    state_vars : list
        The heyoka state variables, in the same order as ``sys`` / ``x0``.
    n : int, optional
        System dimension (defaults to ``len(state_vars)``).
    tol : float
        heyoka adaptive tolerance (default 1e-16 ~ machine precision).
    """

    def __init__(self, order, sys, state_vars, n=None, tol=1e-16):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = int(order)
        self.sys = list(sys)
        self.state_vars = list(state_vars)
        self.n = int(n) if n is not None else len(self.state_vars)
        self.tol = float(tol)

        # Build the variational system (STM + higher STTs) ONCE, JIT-compile ONCE.
        # compact_mode=True is essential for large right-hand sides (gravity
        # fields, variational equations): it compiles shared subexpressions once
        # and turns minutes of LLVM codegen into seconds.
        self._vsys = hy.var_ode_sys(self.sys, hy.var_args.vars, order=self.order)
        self._ta = hy.taylor_adaptive(
            self._vsys, [0.0] * self.n, tol=self.tol, compact_mode=True
        )
        self._n_aug = len(self._ta.state)

        # The variational initial conditions (identity at order 1, zeros above)
        # do NOT depend on x0, so capture them once and reuse.
        self._ic_var = np.array(self._ta.state[self.n :], dtype=float)

        # Precompute the columns of the augmented state that hold the STM, mapping
        # each to its (row i, col j) via the heyoka multi-index [sv, e0, e1, ...].
        sl1 = self._ta.get_vslice(1)
        self._stm_cols = []
        for col in range(sl1.start, sl1.stop):
            mi = self._ta.get_mindex(col)  # [state_var, exp_wrt_varg0, ...]
            i = mi[0]
            j = mi[1:].index(1)  # the single arg differentiated once
            self._stm_cols.append((col, i, j))

        # A cheaper, non-variational integrator for state-only propagation.
        self._ta_state = hy.taylor_adaptive(
            self.sys, [0.0] * self.n, tol=self.tol, compact_mode=True
        )

    # ------------------------------------------------------------------ helpers
    def _set_ic(self, integrator, x0, t0, variational):
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != self.n:
            raise ValueError(f"x0 length {x0.size} does not match n={self.n}")
        if variational:
            y0 = np.empty(self._n_aug, dtype=float)
            y0[: self.n] = x0
            y0[self.n :] = self._ic_var
            integrator.state[:] = y0
        else:
            integrator.state[:] = x0
        integrator.time = float(t0)

    @staticmethod
    def _grid_array(out):
        # heyoka propagate_grid returns a tuple whose last element is the
        # (n_grid, state_len) array of states at the requested times.
        return np.asarray(out[-1], dtype=float)

    # ------------------------------------------------------------------- public
    def propagate(self, x0, t_eval, show_progress=False, compute_stt=True, **opts):
        """Integrate state + variationals; return ``(sol, stts)``.

        ``stts[1]`` is the STM history, shape ``(N, n, n)``.  The full variational
        state at every epoch is stashed on ``sol`` for ``propagate_deviation``.
        """
        if not compute_stt:
            return self.propagate_state_only(x0, t_eval, **opts), None

        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        if t_eval.size < 2:
            raise ValueError("t_eval must have at least 2 points")

        self._set_ic(self._ta, x0, t_eval[0], variational=True)
        out = self._ta.propagate_grid(t_eval)
        aug = self._grid_array(out)  # (N, n_aug)

        x_hist = aug[:, : self.n]  # (N, n)
        N = t_eval.size
        Phi = np.zeros((N, self.n, self.n), dtype=float)
        for col, i, j in self._stm_cols:
            Phi[:, i, j] = aug[:, col]
        stts = {1: Phi}

        sol = _Sol(t=t_eval, y=x_hist.T, aug=aug, grid=t_eval)
        return sol, stts

    def propagate_state_only(self, x0, t_eval, show_progress=False, **opts):
        """Integrate only the state (no variationals). Returns a ``_Sol``."""
        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        if t_eval.size < 2:
            raise ValueError("t_eval must have at least 2 points")
        self._set_ic(self._ta_state, x0, t_eval[0], variational=False)
        out = self._ta_state.propagate_grid(t_eval)
        y = self._grid_array(out)[:, : self.n]  # (N, n)
        return _Sol(t=t_eval, y=y.T, aug=y, grid=t_eval)

    def propagate_deviation(self, sol, stts, delta_x0):
        """Map an initial deviation through the flow's Taylor map.

        For each epoch, restore the saved variational state and evaluate heyoka's
        native Taylor map at ``delta_x0``.  This is the truncated Taylor expansion
        of the flow to the integrator's ``order`` (for ``order == 1`` it equals
        ``x_nom + Phi @ delta_x0``).

        Returns ``(delta, x_est)`` with shapes ``(N, n)``.
        """
        aug = sol._aug
        grid = sol._grid
        # heyoka's eval_taylor_map requires a C-contiguous input array; MCMC
        # estimators can hand us strided views (e.g. a covariance-derived mean),
        # so force contiguity here.
        delta = np.ascontiguousarray(delta_x0, dtype=float).reshape(-1)
        if delta.size != self.n:
            raise ValueError(f"delta_x0 length {delta.size} does not match n={self.n}")

        N = aug.shape[0]
        x_est = np.empty((N, self.n), dtype=float)
        for k in range(N):
            self._ta.state[:] = aug[k]
            self._ta.time = float(grid[k])
            mapped = np.asarray(self._ta.eval_taylor_map(delta), dtype=float)
            x_est[k] = mapped[: self.n]

        x_nom = aug[:, : self.n]
        return x_est - x_nom, x_est

    def propagate_covariance(self, sol, stts, P0):
        """First-order covariance propagation ``P(t) = Phi P0 Phi^T``.

        (Matches the linear term of STTPropagatorND.propagate_covariance; higher-
        order moment terms are available from the Taylor map if needed.)
        """
        Phi_hist = stts[1]
        N, n, _ = Phi_hist.shape
        P0 = np.asarray(P0, dtype=float).reshape(n, n)
        P_t = np.empty((N, n, n), dtype=float)
        for k in range(N):
            Phi = Phi_hist[k]
            P = Phi @ P0 @ Phi.T
            P_t[k] = 0.5 * (P + P.T)
        return P_t


# =============================================================================
# Dynamics builder: degree-2 Bennu gravity (inertial, body-fixed harmonics)
# =============================================================================
def build_bennu_deg2_heyoka(R_ref_km, alpha_rad, delta_rad, omega_rad_s, w0_rad=0.0):
    r"""Build the heyoka ODE system for the 12-D augmented Bennu deg-2 problem.

    State (same layout as the SymPy version):
        X = [x y z vx vy vz mu C20 C21 S21 C22 S22]

    Dynamics are inertial, Bennu-centered.  The gravity field is defined in the
    body-fixed frame and rotated to inertial via ``R_ib(t)`` with the pole
    (alpha, delta) and spin rate ``omega``.  Because the potential depends on the
    inertial position only through ``r_b = R_ib(t) r_i``, the inertial
    acceleration is simply ``grad_{r_i} U`` -- heyoka differentiates it for us.

    Returns
    -------
    sys : list[(var, rhs)]
    state_vars : list of the 12 heyoka variables in order.
    """
    x, y, z, vx, vy, vz, mu, C20, C21, S21, C22, S22 = hy.make_vars(
        "x", "y", "z", "vx", "vy", "vz", "mu", "C20", "C21", "S21", "C22", "S22"
    )
    t = hy.time
    W = w0_rad + omega_rad_s * t

    # Numeric constant rotations (Rx(pi/2 - delta) and Rz(alpha + pi/2)).
    def Rz_num(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def Rx_num(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    M = Rx_num(np.pi / 2 - delta_rad) @ Rz_num(alpha_rad + np.pi / 2)  # constant 3x3

    # Rz(W) with symbolic W (time-dependent) times the constant part M.
    cW, sW = hy.cos(W), hy.sin(W)
    RzW = [[cW, -sW, 0.0], [sW, cW, 0.0], [0.0, 0.0, 1.0]]  # rows of heyoka exprs
    # R_ib = Rz(W) @ M   (3x3 heyoka expression matrix)
    R_ib = [
        [sum(RzW[i][k] * M[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]

    # Body-fixed position r_b = R_ib @ r_i
    ri = [x, y, z]
    xb = sum(R_ib[0][k] * ri[k] for k in range(3))
    yb = sum(R_ib[1][k] * ri[k] for k in range(3))
    zb = sum(R_ib[2][k] * ri[k] for k in range(3))

    # Degree-2 potential, body-fixed.  Written in Cartesian form: this is
    # algebraically identical to the classic P_2m(sin phi)*cos/sin(m*lambda)
    # expansion, but the sqrt(xb^2+yb^2) longitude denominators cancel exactly,
    # leaving a rational field (no pole singularity, far cheaper to compile).
    #   sin(phi) = zb/r,   and after cancellation:
    #   F2 = C20*(3 zb^2/r^2 - 1)/2
    #      + 3 zb (C21 xb + S21 yb)/r^2
    #      + 3 (C22 (xb^2 - yb^2) + 2 S22 xb yb)/r^2
    r2 = xb * xb + yb * yb + zb * zb
    r = hy.sqrt(r2)

    F2 = (
        C20 * 0.5 * (3.0 * zb * zb / r2 - 1.0)
        + 3.0 * zb * (C21 * xb + S21 * yb) / r2
        + 3.0 * (C22 * (xb * xb - yb * yb) + 2.0 * S22 * xb * yb) / r2
    )

    R2 = R_ref_km**2
    U = mu / r * (1.0 + (R2 / r2) * F2)

    # Inertial acceleration = grad_{r_i} U (chain rule through R_ib is automatic).
    ax, ay, az = hy.diff(U, x), hy.diff(U, y), hy.diff(U, z)

    zero = hy.expression(0.0)
    sys = [
        (x, vx),
        (y, vy),
        (z, vz),
        (vx, ax),
        (vy, ay),
        (vz, az),
        (mu, zero),
        (C20, zero),
        (C21, zero),
        (S21, zero),
        (C22, zero),
        (S22, zero),
    ]
    state_vars = [x, y, z, vx, vy, vz, mu, C20, C21, S21, C22, S22]
    return sys, state_vars


if __name__ == "__main__":
    # Smoke test: build the Bennu deg-2 propagator and run a short arc.
    R_bennu = 0.290
    alpha = np.deg2rad(85.65)
    delta = np.deg2rad(-60.17)
    omega = 2 * np.pi / (4.296057 * 3600.0)

    sys, svars = build_bennu_deg2_heyoka(R_bennu, alpha, delta, omega)
    prop = HeyokaSTTPropagator(order=1, sys=sys, state_vars=svars, n=12)

    x0 = np.array(
        [
            0.05,
            0.18,
            0.22,
            1e-4,
            -0.5e-4,
            0.8e-4,
            4.89e-9,
            6.09e-2,
            0.0,
            0.0,
            1.98e-3,
            -7.06e-4,
        ]
    )
    tau = np.linspace(0.0, 3600.0, 20)
    sol, stts = prop.propagate(x0, tau)
    print("state hist shape:", sol.y.shape, " STM hist shape:", stts[1].shape)
    print("Phi(0) is identity:", np.allclose(stts[1][0], np.eye(12)))
    d, xe = prop.propagate_deviation(sol, stts, 1e-3 * np.ones(12))
    print("deviation @last epoch:", np.round(d[-1, :3], 8))
