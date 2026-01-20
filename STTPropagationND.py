import math
import numpy as np
from scipy.integrate import solve_ivp


class STTPropagatorND:
    """
    State Transition Tensor (STT) Propagator for N-Dimensional Nonlinear Dynamical Systems

    A high-order Taylor expansion propagator that computes state transition tensors (STTs)
    for nonlinear systems of the form:

        dx/dt = f(x, t)

    where x ∈ ℝⁿ. The propagator computes:
      - Φ (order 1): State Transition Matrix (STM)
      - T_k (order k≥2): Higher-order State Transition Tensors

    The STTs enable high-order sensitivity analysis, deviation propagation, and
    covariance propagation without explicit Monte Carlo sampling.

    Attributes:
        order (int): Maximum order of STT to compute (≥1).
        f_func (callable): State derivative function f(x_1, ..., x_n, t) → ℝⁿ.
        A_func (callable): Jacobian function A(x_1, ..., x_n, t) → ℝⁿˣⁿ.
        B_funcs (dict): Maps order k → callable for k-th order derivative tensor
                        B_k(x_1, ..., x_n, t) → ℝⁿˣⁿˣ...ˣⁿ (n^(k+1) elements).
        n (int or None): System dimension. If None, inferred from x0 at propagation time.

    Methods:
        propagate(x0, t_eval, show_progress=True, **options):
            Integrate the STT equations from initial condition x0 over time points t_eval.

            Args:
                x0 (array-like): Initial state (n,).
                t_eval (array-like): Time points for solution output.
                show_progress (bool): Display integration progress bar.
                **options: Keyword arguments passed to scipy.integrate.solve_ivp.

                tuple: (sol, stts) where
                    sol: ODE solution object from solve_ivp.
                    stts: dict mapping order k → array of shape (n_steps, n, n, ..., n).

        propagate_deviation(sol, stts, delta_x0):
            Compute high-order deviation from nominal trajectory using STT expansion.

            Args:
                sol: ODE solution from propagate().
                stts: Unpacked STT tensors from propagate().
                delta_x0 (array-like): Initial deviation (n,).

                tuple: (delta, x_est) where
                    delta: Deviation trajectory (n_steps, n).
                    x_est: Estimated trajectory (n_steps, n).

        propagate_covariance(sol, stts, P0):
            Propagate covariance using moment-propagation with STT expansion.

            Args:
                sol: ODE solution from propagate().
                stts: Unpacked STT tensors from propagate().
                P0 (array-like): Initial covariance matrix (n, n).

                array: Covariance trajectory (n_steps, n, n).

    Notes:
        - B_funcs must contain entries for all orders k ∈ [2, order].
        - STT computation uses Einstein summation (np.einsum) and tensor contractions.
        - Supports orders up to 4 in deviation/covariance methods; higher orders
          require extending the contraction patterns.
    """

    def __init__(self, order: int, f_func, A_func, B_funcs=None, n: int | None = None):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = int(order)
        self.f_func = f_func
        self.A_func = A_func
        self.B_funcs = {} if B_funcs is None else dict(B_funcs)
        self.n = n  # if None, inferred from x0 at propagate-time

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _as_float_array(x):
        return np.asarray(x, dtype=float)

    def _infer_n(self, x0):
        if self.n is not None:
            return int(self.n)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        return int(x0.size)

    def _pack_initial_Y0(self, x0, n):
        # State
        Y0 = list(np.asarray(x0, dtype=float).reshape(-1))
        if len(Y0) != n:
            raise ValueError(f"x0 length {len(Y0)} does not match n={n}")

        # STM (k=1): n x n
        Y0 += list(np.eye(n, dtype=float).reshape(-1))

        # Higher-order STTs: for k>=2, size = n^(k+1)
        for k in range(2, self.order + 1):
            Y0 += [0.0] * (n ** (k + 1))

        return np.asarray(Y0, dtype=float)

    def _unpack_stts(self, sol, n):
        n_steps = sol.y.shape[1]
        stts = {}
        offset = n

        # k=1: STM
        phi_flat = sol.y[offset : offset + n * n, :]  # (n*n, n_steps)
        phi_all = phi_flat.reshape(n, n, n_steps)  # (n,n,n_steps)
        stts[1] = np.transpose(phi_all, (2, 0, 1))  # (n_steps,n,n)
        offset += n * n

        # k>=2
        for k in range(2, self.order + 1):
            block_size = n ** (k + 1)
            Tk_flat = sol.y[offset : offset + block_size, :]  # (n^(k+1), n_steps)
            shape = (n,) + (n,) * k + (n_steps,)
            Tk_all = Tk_flat.reshape(shape)
            stts[k] = np.moveaxis(Tk_all, -1, 0)  # (n_steps, n, n, ..., n)
            offset += block_size

        return stts

    # -----------------------------
    # Core RHS
    # -----------------------------
    def rhs(self, t, Y, n):
        """
        Y packs: [x (n), Phi (n*n), T2 (n^3), ..., Torder (n^(order+1))]
        """
        # 1) state
        x = Y[:n]
        dx = self._as_float_array(self.f_func(*x, t)).reshape(n)

        # 2) STM
        offset = n
        Phi = Y[offset : offset + n * n].reshape(n, n)
        offset += n * n

        A = self._as_float_array(self.A_func(*x, t)).reshape(n, n)
        dPhi = A @ Phi

        # 3) unpack T_k
        Ts = {1: Phi}
        for k in range(2, self.order + 1):
            size = n ** (k + 1)
            Tk = Y[offset : offset + size].reshape((n,) + (n,) * k)
            Ts[k] = Tk
            offset += size

        derivs = [*dx, *dPhi.reshape(-1)]

        # 4) dT_k recursion (matches your existing structure)
        for k in range(2, self.order + 1):
            if k not in self.B_funcs:
                raise ValueError(
                    f"B_funcs missing entry for k={k} (needed for order={self.order})"
                )

            # A · T_k  (contract A's columns with T_k's first axis)
            dTk = np.tensordot(
                A, Ts[k], axes=(1, 0)
            )  # -> shape (n, n, ..., n) k+1 axes

            # full-order term: B_k(Φ,...,Φ)
            Bk = self._as_float_array(self.B_funcs[k](*x, t)).reshape((n,) + (n,) * k)
            term = Bk
            for _ in range(k):
                term = np.tensordot(term, Phi, axes=(1, 0))
            dTk += term

            # mixed terms: choose m slots at once as in your original (one insertion of T_{k-m+1})
            for m in range(2, k):
                if m not in self.B_funcs:
                    raise ValueError(
                        f"B_funcs missing entry for m={m} (needed for mixed terms up to order={self.order})"
                    )

                coef = math.comb(k, m)
                Bm = self._as_float_array(self.B_funcs[m](*x, t)).reshape(
                    (n,) + (n,) * m
                )

                # First contract one slot with T_{k-m+1}
                mixed = np.tensordot(Bm, Ts[k - m + 1], axes=(1, 0))
                # Contract remaining m-1 slots with Φ
                for _ in range(m - 1):
                    mixed = np.tensordot(mixed, Phi, axes=(1, 0))

                dTk += coef * mixed

            derivs += list(dTk.reshape(-1))

        return np.asarray(derivs, dtype=float)

    # -----------------------------
    # Public API
    # -----------------------------
    def propagate(self, x0, t_eval, show_progress=True, compute_stt=True, **options):
        if not compute_stt:
            sol = self.propagate_state_only(
                x0, t_eval, show_progress=show_progress, **options
            )
            return sol, None

        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        if t_eval.size < 2:
            raise ValueError("t_eval must have at least 2 points")

        n = self._infer_n(x0)
        Y0 = self._pack_initial_Y0(x0, n)

        t_start = float(t_eval[0])
        t_end = float(t_eval[-1])
        last_print = -1

        def wrapped_rhs(t, y):
            nonlocal last_print
            if show_progress:
                denom = (t_end - t_start) if (t_end - t_start) != 0 else 1.0
                progress = int(100 * (t - t_start) / denom)
                if progress > last_print:
                    bar = "█" * (progress // 2) + "-" * (50 - progress // 2)
                    print(f"\rProgress |{bar}| {progress:.1f}% - t = {t:.2f}", end="")
                    last_print = progress
            return self.rhs(t, y, n)

        sol = solve_ivp(
            fun=wrapped_rhs,
            t_span=(t_start, t_end),
            t_eval=t_eval,
            y0=Y0,
            **options,
        )
        if show_progress:
            print("")

        return sol, self._unpack_stts(sol, n)

    def propagate_state_only(self, x0, t_eval, show_progress=True, **options):
        """
        Integrate only the state dx/dt = f(x,t), no STM/STTs.
        Returns:
          sol: solve_ivp solution with sol.y shape (n, n_steps)
        """
        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        if t_eval.size < 2:
            raise ValueError("t_eval must have at least 2 points")

        n = self._infer_n(x0)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != n:
            raise ValueError(f"x0 length {x0.size} does not match n={n}")

        t_start = float(t_eval[0])
        t_end = float(t_eval[-1])
        last_print = -1

        def rhs_state(t, x):
            nonlocal last_print
            if show_progress:
                denom = (t_end - t_start) if (t_end - t_start) != 0 else 1.0
                progress = int(100 * (t - t_start) / denom)
                if progress > last_print:
                    bar = "█" * (progress // 2) + "-" * (50 - progress // 2)
                    print(f"\rProgress |{bar}| {progress:.1f}% - t = {t:.2f}", end="")
                    last_print = progress
            return self._as_float_array(self.f_func(*x, t)).reshape(n)

        sol = solve_ivp(
            fun=rhs_state,
            t_span=(t_start, t_end),
            t_eval=t_eval,
            y0=x0,
            **options,
        )
        if show_progress:
            print("")
        return sol

    def propagate_deviation(self, sol, stts, delta_x0):
        """
        sol: output of propagate()
        stts: unpacked tensors from propagate()
        delta_x0: (n,)
        Returns:
          delta(t): (n_steps,n)
          x_est(t): (n_steps,n)
        """
        x_nom = sol.y[: stts[1].shape[1], :].T  # (n_steps,n)

        delta_x0 = np.asarray(delta_x0, dtype=float).reshape(-1)
        n = delta_x0.size
        n_steps = x_nom.shape[0]

        # First-order
        delta = np.einsum("tij,j->ti", stts[1], delta_x0)

        # Higher orders
        if self.order >= 2:
            delta += 0.5 * np.einsum("tijk,j,k->ti", stts[2], delta_x0, delta_x0)

        if self.order >= 3:
            delta += (1.0 / 6.0) * np.einsum(
                "tijkl,j,k,l->ti", stts[3], delta_x0, delta_x0, delta_x0
            )

        if self.order >= 4:
            delta += (1.0 / 24.0) * np.einsum(
                "tijklm,j,k,l,m->ti", stts[4], delta_x0, delta_x0, delta_x0, delta_x0
            )

        # If you need >4, extend with factorial + einsum pattern.

        return delta, x_nom + delta

    def propagate_covariance(self, sol, stts, P0):
        """
        Moment-propagation approximation consistent with your original class.
        P0: (n,n)
        Returns P_t: (n_steps,n,n)
        """
        n_steps = stts[1].shape[0]
        n = stts[1].shape[1]
        P0 = np.asarray(P0, dtype=float).reshape(n, n)

        P_t = np.zeros((n_steps, n, n), dtype=float)

        for ti in range(n_steps):
            Phi = stts[1][ti]
            P = Phi @ P0 @ Phi.T

            if self.order >= 2:
                T2 = stts[2][ti]
                term2 = 0.5 * np.einsum("ijk,jl,km->im", T2, P0, P0)
                P += term2 + term2.T

            if self.order >= 3:
                T3 = stts[3][ti]
                term3 = (1.0 / 6.0) * np.einsum("ijkl,jm,kn,lo->io", T3, P0, P0, P0)
                P += term3 + term3.T

            if self.order >= 4:
                T4 = stts[4][ti]
                term4 = (1.0 / 24.0) * np.einsum(
                    "ijklm,jn,ko,lp,mq->iq", T4, P0, P0, P0, P0
                )
                P += term4 + term4.T

            P_t[ti] = 0.5 * (P + P.T)

        return P_t
