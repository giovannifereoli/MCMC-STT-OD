import numpy as np
import math
from scipy.integrate import solve_ivp


class STTPropagator:
    def __init__(self, order, f_func, A_func, B_funcs):
        self.order = order
        self.f_func = f_func
        self.A_func = A_func
        self.B_funcs = B_funcs

    def rhs(self, t, Y):
        # 1) dynamics + STM
        x = Y[:6]  # unpack state
        dx = np.array(self.f_func(*x), float).reshape(6)
        offset = 6
        Phi = Y[offset : offset + 36].reshape(6, 6)
        offset += 36
        A = np.array(self.A_func(*x), float)
        dPhi = A @ Phi

        # 2) unpack every T_k once
        Ts = {1: Phi}
        for k in range(2, self.order + 1):
            size = 6 ** (k + 1)
            Tk = Y[offset : offset + size].reshape((6,) + (6,) * k)
            Ts[k] = Tk
            offset += size

        derivs = [*dx, *dPhi.flatten()]

        # 3) build each dT_k with all partitions
        for k in range(2, self.order + 1):
            # start with A·T_k
            dTk = np.tensordot(A, Ts[k], axes=(1, 0))

            # full‑order term: B_k(Φ,...,Φ)
            Bk = np.array(self.B_funcs[k](*x), float).reshape((6,) + (6,) * k)
            term = Bk
            for _ in range(k):
                term = np.tensordot(term, Phi, axes=(1, 0))
            dTk += term

            # mixed lower‑order terms
            # for every 2 ≤ m < k, we have comb(k,m) ways to choose
            # which slots get a T_{k-m+1} instead of Φ
            for m in range(2, k):
                coef = math.comb(k, m)
                Bm = np.array(self.B_funcs[m](*x), float).reshape((6,) + (6,) * m)

                # first contract one slot with T_{k-m+1}
                mixed = np.tensordot(Bm, Ts[k - m + 1], axes=(1, 0))
                # then contract the remaining m−1 slots with Φ
                for _ in range(m - 1):
                    mixed = np.tensordot(mixed, Phi, axes=(1, 0))

                dTk += coef * mixed

            derivs += list(dTk.flatten())

        return np.array(derivs, float)

    def propagate(self, x0, t_eval, show_progress=True, **options):
        # 2) Initial
        Y0 = list(x0)
        Y0 += list(np.eye(6).flatten())
        for k in range(2, self.order + 1):
            Y0 += [0.0] * (6 ** (k + 1))
        Y0 = np.array(Y0, float)

        t_start = t_eval[0]
        t_end = t_eval[-1]

        last_print = -1

        def wrapped_rhs(t, y):
            nonlocal last_print
            if show_progress:
                progress = int(100 * (t - t_start) / (t_end - t_start))
                if progress > last_print:
                    bar = "█" * (progress // 2) + "-" * (50 - progress // 2)
                    print(f"\rProgress |{bar}| {progress:.1f}% - t = {t:.2f}", end="")
                    last_print = progress
            return self.rhs(t, y)

        # 3) Integrate
        sol = solve_ivp(
            fun=wrapped_rhs,
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
            y0=Y0,
            **options,
        )

        return sol, self._unpack_stts(sol)

    def _unpack_stts(self, sol):
        # 4) Decode the solution
        n_steps = sol.y.shape[1]
        stts = {}
        offset = 6

        # k = 1 is the STM (6×6)
        phi_flat = sol.y[offset : offset + 36, :]  # shape (36, n_steps)
        phi_all = phi_flat.reshape(6, 6, n_steps)  # shape (6,6,n_steps)
        stts[1] = np.transpose(phi_all, (2, 0, 1))  # shape (n_steps,6,6)
        offset += 36

        # higher orders k=2..order
        for k in range(2, self.order + 1):
            block_size = 6 ** (k + 1)
            Tk_flat = sol.y[offset : offset + block_size, :]  # (6^(k+1), n_steps)
            # reshape into (6,6,...,6,n_steps)
            shape = (6,) + (6,) * k + (n_steps,)
            Tk_all = Tk_flat.reshape(shape)
            # move time axis to front → shape (n_steps,6,6,...,6)
            stts[k] = np.moveaxis(Tk_all, -1, 0)
            offset += block_size

        return stts

    def propagate_deviation(self, sol, stts, delta_x0):
        # 1) unpack the solution
        x_nom = sol.y[:6, :].T  # (n_steps,6)

        # 2) Start with first order: Phi @ δx0
        delta = np.einsum("tij,j->ti", stts[1], delta_x0)

        # 3) Add higher orders
        if self.order >= 2:
            delta += 0.5 * np.einsum("tijk,j,k->ti", stts[2], delta_x0, delta_x0)

        if self.order >= 3:
            delta += (1 / 6) * np.einsum(
                "tijkl,j,k,l->ti", stts[3], delta_x0, delta_x0, delta_x0
            )

        if self.order >= 4:
            delta += (1 / 24) * np.einsum(
                "tijklm,j,k,l,m->ti", stts[4], delta_x0, delta_x0, delta_x0, delta_x0
            )

        return delta, x_nom + delta

    def propagate_covariance(self, sol, stts, P0):
        n_steps = sol.y.shape[1]
        P_t = np.zeros((n_steps, 6, 6))

        # First order term: Φ P0 Φᵀ
        for t in range(n_steps):
            Phi = stts[1][t]
            P = Phi @ P0 @ Phi.T

            # Second order: sum_i,j T_ijk P0_jl P0_kl
            if self.order >= 2:
                T2 = stts[2][t]
                term2 = 0.5 * np.einsum("ijk,jl,km->im", T2, P0, P0)
                P += term2 + term2.T  # ensure symmetry

            if self.order >= 3:
                T3 = stts[3][t]
                term3 = (1 / 6) * np.einsum("ijkl,jm,kn,lo->io", T3, P0, P0, P0)
                P += term3 + term3.T

            if self.order >= 4:
                T4 = stts[4][t]
                term4 = (1 / 24) * np.einsum(
                    "ijklm,jn,ko,lp,mq->iq", T4, P0, P0, P0, P0
                )
                P += term4 + term4.T

            P_t[t] = 0.5 * (P + P.T)  # final symmetrization

        return P_t
