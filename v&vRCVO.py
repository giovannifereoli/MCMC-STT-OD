#!/usr/bin/env python3
"""
RCVO linearized residual model in dx0 using MATLAB SRIF prep file (v7.3 MAT).

Model:
    dx_k = Phi0k @ dx0
    e_k(dx0) = r_k - H_k @ dx_k = r_k - (H_k @ Phi0k) @ dx0
Whitening (optional, done block-by-block):
    e_w = L^{-1} e,   where R = L L^T

This script:
- loads MATLAB v7.3 MAT (HDF5) robustly with deref
- applies MATLAB->NumPy transpose fix for 2D arrays
- builds stacked (r_w, A_w) where A_w stacks (Hk Phi0k) whitened by Rk^{-1/2}
- exposes a picklable residual function for emcee multiprocessing
- runs your MCMCModel (if importable)
- plots prefit/postfit residuals (whitened units)

Deps:
  pip install numpy scipy h5py matplotlib emcee
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import norm


# =========================
# Data containers
# =========================


@dataclass
class MeasBlock:
    time: float
    r: np.ndarray  # (Nk,)
    Hk: np.ndarray  # (Nk, n_state)
    R: np.ndarray  # (Nk, Nk)


@dataclass
class TrajNode:
    time: float
    Phi0k: np.ndarray  # (n_state, n_state)


# =========================
# MATLAB v7.3 (HDF5) helpers
# =========================


def _is_hdf5_ref_dtype(dtype) -> bool:
    try:
        return h5py.check_dtype(ref=dtype) is not None
    except Exception:
        return False


def _deref(f: h5py.File, obj):
    if isinstance(obj, h5py.Group):
        return obj

    if isinstance(obj, h5py.Dataset):
        obj = obj[()]

    if isinstance(obj, h5py.Reference):
        target = f[obj]
        if isinstance(target, h5py.Dataset):
            return np.array(target[()])
        return target

    if isinstance(obj, np.ndarray):
        if not _is_hdf5_ref_dtype(obj.dtype):
            return np.array(obj)

        arr = np.asarray(obj).squeeze()
        if arr.size == 1:
            ref = arr.reshape(()).item()
            target = f[ref]
            if isinstance(target, h5py.Dataset):
                return np.array(target[()])
            return target

        out = []
        for ref in arr.ravel():
            target = f[ref]
            if isinstance(target, h5py.Dataset):
                out.append(np.array(target[()]))
            else:
                out.append(target)
        return out

    return obj


def _matlab_to_numpy_numeric(a: np.ndarray) -> np.ndarray:
    """
    MATLAB->NumPy fix for numeric arrays read via h5py.

    In practice for v7.3 MAT:
      - 2D matrices commonly need transpose (A.T).
      - 1D vectors: keep as is (we later reshape).
      - Higher-d arrays: we do NOT guess here (not needed for this file).
    """
    a = np.array(a)
    if a.ndim == 2:
        return a.T
    return a


def _read_scalar(f: h5py.File, ref) -> float:
    a = _deref(f, ref)
    if isinstance(a, h5py.Group):
        raise TypeError("Expected scalar dataset/ref, got group.")
    a = np.array(a).squeeze()
    return float(a)


def _read_array(f: h5py.File, ref, dtype=float, matlab_fix: bool = True) -> np.ndarray:
    a = _deref(f, ref)
    if isinstance(a, h5py.Group):
        raise TypeError("Expected array dataset/ref, got group.")
    a = np.array(a, dtype=dtype)
    if matlab_fix:
        a = _matlab_to_numpy_numeric(a)
    return a


def _read_struct_array_refs(f: h5py.File, group_name: str) -> Tuple[h5py.Group, int]:
    """
    For a MATLAB struct array saved in v7.3, typical layout:
      group_name is a Group
      each field is a Dataset of object refs with shape (1, M)
    Returns (group, M).
    """
    if group_name not in f:
        raise KeyError(f"Missing group '{group_name}' in MAT.")
    g = f[group_name]
    if not isinstance(g, h5py.Group):
        raise TypeError(f"'{group_name}' is not a group.")

    fields = list(g.keys())
    if not fields:
        raise ValueError(f"Group '{group_name}' has no fields.")

    ds0 = g[fields[0]]
    if not isinstance(ds0, h5py.Dataset) or ds0.ndim != 2 or ds0.shape[0] != 1:
        raise TypeError(
            f"Unexpected struct-array field storage for '{group_name}'. "
            f"Field '{fields[0]}' has shape {getattr(ds0,'shape',None)}."
        )
    M = ds0.shape[1]
    return g, M


# =========================
# Loaders
# =========================


def load_sorted_measurements_v73(mat_path: str) -> List[MeasBlock]:
    blocks: List[MeasBlock] = []

    with h5py.File(mat_path, "r") as f:
        sm, M = _read_struct_array_refs(f, "sorted_measurements")

        time_refs = sm["time"][0, :]
        res_refs = sm["residual"][0, :]
        cov_refs = sm["covariance"][0, :]
        par_refs = sm["partials"][0, :]

        for k in range(M):
            t = _read_scalar(f, time_refs[k])

            r = _read_array(f, res_refs[k], dtype=float, matlab_fix=True).reshape(-1)
            Nk = r.size

            R = _read_array(f, cov_refs[k], dtype=float, matlab_fix=True)
            R = np.atleast_2d(R)
            if R.shape != (Nk, Nk):
                raise ValueError(
                    f"Block {k}: R shape {R.shape} incompatible with Nk={Nk}."
                )

            par_g = _deref(f, par_refs[k])
            if not isinstance(par_g, h5py.Group):
                raise TypeError(f"partials[{k}] did not resolve to a group.")
            if "wrt_X" not in par_g:
                raise KeyError(f"partials[{k}] missing 'wrt_X'.")

            Hk = _deref(f, par_g["wrt_X"])
            if isinstance(Hk, list):
                if not Hk:
                    raise ValueError(f"partials[{k}].wrt_X is empty.")
                Hk = Hk[0]

            Hk = np.array(Hk, dtype=float)
            Hk = _matlab_to_numpy_numeric(Hk)  # transpose fix
            Hk = np.atleast_2d(Hk)

            # extra safety: if still transposed, fix
            if Hk.shape[0] != Nk and Hk.shape[1] == Nk:
                Hk = Hk.T

            if Hk.shape[0] != Nk:
                raise ValueError(
                    f"Block {k}: Hk shape {Hk.shape} incompatible with Nk={Nk}."
                )

            blocks.append(MeasBlock(time=t, r=r, Hk=Hk, R=R))

    return blocks


def load_trajectory_ref_v73(mat_path: str) -> List[TrajNode]:
    nodes: List[TrajNode] = []

    with h5py.File(mat_path, "r") as f:
        tr, M = _read_struct_array_refs(f, "trajectory_ref")

        time_refs = tr["time"][0, :]
        stm_refs = tr["STM"][0, :]

        for k in range(M):
            t = _read_scalar(f, time_refs[k])
            Phi = _read_array(f, stm_refs[k], dtype=float, matlab_fix=True)
            Phi = np.atleast_2d(Phi)

            if Phi.shape[0] != Phi.shape[1]:
                raise ValueError(f"trajectory_ref[{k}].STM not square: {Phi.shape}")

            nodes.append(TrajNode(time=t, Phi0k=Phi))

    return nodes


def load_P0_v73(mat_path: str) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        if "P0" not in f:
            raise KeyError("Missing P0 in MAT.")
        P0 = np.array(f["P0"][()], dtype=float)
        P0 = _matlab_to_numpy_numeric(P0)  # transpose fix
        return P0


# =========================
# Whitening
# =========================


def whiten_block(
    r: np.ndarray, A: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    R = np.atleast_2d(R)

    diag = np.diag(R)
    if np.allclose(R, np.diag(diag), rtol=0.0, atol=0.0):
        s = np.sqrt(np.maximum(diag, 1e-30))
        return r / s, A / s[:, None]

    try:
        L = np.linalg.cholesky(R)
        return np.linalg.solve(L, r), np.linalg.solve(L, A)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(R)
        w = np.maximum(w, 1e-30)
        Winvhalf = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        return Winvhalf @ r, Winvhalf @ A


# =========================
# Linearized dx0 model -> stacked (r, A)
# =========================


class LinearizedRCVODx0:
    def __init__(
        self,
        blocks: List[MeasBlock],
        traj: List[TrajNode],
        whiten: bool = True,
        time_tol: float = 1e-5,
    ):
        if not blocks:
            raise ValueError("No measurement blocks loaded.")
        if not traj:
            raise ValueError("No trajectory_ref nodes loaded.")

        self.blocks = blocks
        self.traj = traj
        self.whiten = whiten
        self.time_tol = float(time_tol)

        self._t_traj = np.array([n.time for n in traj], dtype=float)
        self._Phi = [n.Phi0k for n in traj]
        self.n_state = self._Phi[0].shape[0]

        rs = []
        As = []
        ts = []

        for b in blocks:
            Phi0k = self._get_Phi_at_time(b.time)
            A = b.Hk @ Phi0k  # Nk x n_state

            if whiten:
                rw, Aw = whiten_block(b.r, A, b.R)
            else:
                rw, Aw = b.r, A

            rs.append(rw.reshape(-1))
            As.append(Aw)
            ts.append(np.full(rw.size, b.time, dtype=float))

        self.r = np.concatenate(rs, axis=0)
        self.A = np.vstack(As)
        self.t_scalar = np.concatenate(ts, axis=0)

    def _get_Phi_at_time(self, t: float) -> np.ndarray:
        j = int(np.argmin(np.abs(self._t_traj - t)))
        dt = abs(self._t_traj[j] - t)
        if dt > self.time_tol:
            raise ValueError(
                f"Measurement time {t:.9f} not on trajectory_ref grid; nearest dt={dt:.3e} s. "
                "Increase time_tol or re-build trajectory_ref on measurement epochs."
            )
        Phi0k = self._Phi[j]
        if Phi0k.shape != (self.n_state, self.n_state):
            raise ValueError(f"Phi0k wrong shape {Phi0k.shape}")
        return Phi0k


# =========================
# PICKLABLE residual callable (for multiprocessing emcee)
# =========================


class ResidualDx0:
    """
    Picklable callable: e(dx0) = r - A[:,idx] @ dx0

    IMPORTANT: emcee multiprocessing requires the log_prob function to be picklable.
    Closures / nested funcs capturing non-picklable objects often fail on macOS spawn.
    This class holds only numpy arrays -> picklable.
    """

    def __init__(
        self, r: np.ndarray, A: np.ndarray, idx_solve: Optional[np.ndarray] = None
    ):
        self.r = np.asarray(r, dtype=float).reshape(-1)
        self.A = np.asarray(A, dtype=float)
        if idx_solve is None:
            self.idx = None
        else:
            self.idx = np.asarray(idx_solve, dtype=int).reshape(-1)

    def __call__(self, dx0: np.ndarray) -> np.ndarray:
        dx0 = np.asarray(dx0, dtype=float).reshape(-1)
        if self.idx is None:
            return self.r - self.A @ dx0
        return self.r - self.A[:, self.idx] @ dx0


# =========================
# Plotting
# =========================


def plot_prefit_postfit(
    t_scalar_s: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    save_prefix: Optional[str] = None,
    num_bins: int = 60,
    fontsize: int = 16,
    markersize: float = 3.0,
):
    t_hr = np.asarray(t_scalar_s, dtype=float) / 3600.0

    def _make_fig(data: np.ndarray, label: str):
        data = np.asarray(data).reshape(-1)
        finite = np.isfinite(data)
        y = data[finite]
        tt = t_hr[finite]

        if y.size == 0:
            ylo, yhi = -1.0, 1.0
        else:
            yabs = np.max(np.abs(y))
            ylo, yhi = -yabs, yabs

        fig = plt.figure(figsize=(12.5, 4.8))
        gs = fig.add_gridspec(1, 2, wspace=0.15)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(tt, y, ".", markersize=markersize)
        ax1.axhline(0.0, linewidth=1.6)
        ax1.grid(True, linestyle=":")
        ax1.set_xlabel("Time since epoch [hr]")
        ax1.set_ylabel(label)
        ax1.set_ylim([ylo, yhi])
        ax1.tick_params(labelsize=fontsize)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(y, bins=num_bins, density=True, orientation="horizontal", alpha=0.7)
        ax2.axhline(0.0, linewidth=1.6)
        ax2.grid(True, linestyle=":")
        ax2.set_xlabel("PDF [-]")
        ax2.set_ylim([ylo, yhi])
        ax2.tick_params(labelsize=fontsize)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        fig.suptitle(label, fontsize=fontsize + 2)
        fig.tight_layout()

        print(f"\n==== {label} stats (whitened) ====")
        print(f"N     = {y.size}")
        print(f"Mean  = {np.mean(y):.6e}")
        print(f"Sigma = {np.std(y):.6e}")
        print(f"RMS   = {np.sqrt(np.mean(y**2)):.6e}")

        return fig

    fig1 = _make_fig(pre, "Prefit residuals (whitened)")
    if save_prefix:
        fig1.savefig(f"{save_prefix}_prefit.png", dpi=200)

    fig2 = _make_fig(post, "Postfit residuals (whitened)")
    if save_prefix:
        fig2.savefig(f"{save_prefix}_postfit.png", dpi=200)

    plt.show()


# =========================
# Main
# =========================


def main():
    # ---- USER SETTINGS ----
    MAT_FILE = "Extra/srif_inputs_from_RZ.mat"
    SAVE_PREFIX = "results/rcvo_linear_dx0"
    TIME_TOL = 1e-5

    # MCMC settings
    N_SAMPLES = 2000
    N_WALKERS = 128
    BURN_IN = 1
    THIN = 1
    SPHERICAL_SPREAD = 1e-1
    METHOD_OPTIMIZE = "lsq"

    if not os.path.isfile(MAT_FILE):
        raise FileNotFoundError(f"MAT file not found: {MAT_FILE}")

    os.makedirs(os.path.dirname(SAVE_PREFIX), exist_ok=True)

    print(f"[LOAD] {MAT_FILE}")
    blocks = load_sorted_measurements_v73(MAT_FILE)
    traj = load_trajectory_ref_v73(MAT_FILE)
    P0 = load_P0_v73(MAT_FILE)

    print(f"[OK] blocks: {len(blocks)}")
    print(f"[OK] traj nodes: {len(traj)}")

    # STM sanity: Phi(t0) ~ I
    Phi0 = traj[0].Phi0k
    Ierr = np.linalg.norm(Phi0 - np.eye(Phi0.shape[0]))
    print(f"[CHECK] ||Phi(t0)-I|| = {Ierr:.3e}")

    model_lin = LinearizedRCVODx0(blocks, traj, whiten=True, time_tol=TIME_TOL)

    n_state = P0.shape[0]
    if model_lin.n_state != n_state:
        raise ValueError(
            f"State size mismatch: P0 is {n_state}, Phi is {model_lin.n_state}"
        )

    idx_solve = np.arange(n_state, dtype=int)

    # Build PICKLABLE residual callable (IMPORTANT for multiprocessing)
    residual_callable = ResidualDx0(model_lin.r, model_lin.A, idx_solve=idx_solve)

    # Prefit at dx0=0
    pre = residual_callable(np.zeros(idx_solve.size))
    dof = pre.size - idx_solve.size
    chi2 = float(pre @ pre)
    print(f"[SANITY] chi2={chi2:.6e} dof={dof} chi2_red={chi2/max(dof,1):.6e}")

    # Priors from P0 diagonal
    sig0 = np.sqrt(np.maximum(np.diag(P0), 0.0))
    priors = [norm(loc=0.0, scale=(s if s > 0 else 1.0)) for s in sig0[idx_solve]]

    # Run your MCMC
    try:
        from MCMC import MCMCModel
    except Exception as e:
        raise ImportError(
            "Could not import your MCMCModel. Ensure MCMC.py is on PYTHONPATH and provides MCMCModel.\n"
            f"Import error: {e}"
        ) from e

    mcmc = MCMCModel(
        residuals_func=residual_callable,  # <- picklable callable
        initial_params=np.zeros(idx_solve.size),
        param_priors=priors,
        observed_data=np.zeros(1),
    )
    mcmc.setup_whitening_from_priors()

    mcmc.run(
        n_samples=N_SAMPLES,
        n_walkers=N_WALKERS,
        burn_in=BURN_IN,
        thin=THIN,
        spherical_spread=SPHERICAL_SPREAD,
        method_optimize=METHOD_OPTIMIZE,
    )

    dx0_hat, P_mcmc = mcmc.get_estimate_and_covariance()
    post = residual_callable(dx0_hat)

    chi2_post = float(post @ post)
    print(f"[MCMC] chi2_red @ dx0_hat = {chi2_post/max(dof,1):.6e}")
    print(f"[MCMC] ||dx0_hat|| = {np.linalg.norm(dx0_hat):.6e}")

    plot_prefit_postfit(
        t_scalar_s=model_lin.t_scalar,
        pre=pre,
        post=post,
        save_prefix=SAVE_PREFIX,
        num_bins=60,
        fontsize=16,
        markersize=3.0,
    )

    # Optional diagnostics if available
    for fn in [
        "plot_convergence",
        "plot_log_likelihood",
        "plot_autocorrelation",
        "summary",
        "gelman_rubin_diagnostic",
        "plot_corner",
    ]:
        try:
            getattr(mcmc, fn)()
        except Exception as e:
            print(f"[INFO] {fn} not available / failed:", e)


if __name__ == "__main__":
    main()
