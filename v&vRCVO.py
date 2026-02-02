#!/usr/bin/env python3
"""
RCVO linearized residual model in dx0 using MATLAB SRIF prep file (v7.3 MAT).

Core model (dx0 sampling + STM mapping):
    dx_k = Phi0k @ dx0
    e_k(dx0) = r_k - H_k @ dx_k = r_k - (H_k @ Phi0k) @ dx0

Key points:
- MATLAB v7.3 MAT is HDF5. Reading with h5py requires careful deref of object refs.
- VERY IMPORTANT: MATLAB arrays are column-major. When you read 2D arrays with h5py,
  they commonly appear transposed relative to what you expect in Python.
  This script applies a consistent "MATLAB->NumPy" fix:
      for any 2D numeric matrix A: use A.T

Input MAT must contain (as saved by your MATLAB script):
  - sorted_measurements: struct array with fields {time, residual, partials, covariance}
      where partials.wrt_X is H_k wrt x_k at measurement epoch k
  - trajectory_ref: struct array with fields {time, STM, ...} where STM is Phi0k
  - P0: a priori covariance on dx0 (n_state x n_state)

Dependencies:
  pip install numpy scipy h5py matplotlib

Optional:
  your MCMC class must be importable: from MCMC import MCMCModel
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
    time: float  # seconds since t0 (MATLAB: t2 - t0)
    r: np.ndarray  # (Nk,)
    Hk: np.ndarray  # (Nk, n_state) Jacobian wrt x_k at epoch k
    R: np.ndarray  # (Nk, Nk) covariance


@dataclass
class TrajNode:
    time: float
    Phi0k: np.ndarray  # (n_state, n_state)


# =========================
# MATLAB v7.3 (HDF5) helpers
# =========================


def _is_hdf5_ref_dtype(dtype) -> bool:
    """True if dtype is an HDF5 object reference dtype."""
    try:
        return h5py.check_dtype(ref=dtype) is not None
    except Exception:
        return False


def _deref(f: h5py.File, obj):
    """
    Robustly interpret MATLAB v7.3 fields:
      - if obj is an HDF5 Dataset: read it (may yield numeric array OR refs)
      - if obj is an HDF5 Group: return group
      - if obj is a scalar HDF5 reference: dereference it
      - if obj is an ndarray of references:
          * if size==1: dereference scalar
          * else: return list of dereferenced objects (cell-array style)
      - if obj is numeric ndarray: return it directly
    """
    if isinstance(obj, h5py.Group):
        return obj

    if isinstance(obj, h5py.Dataset):
        obj = obj[()]  # numpy array or scalar (maybe refs)

    if isinstance(obj, h5py.Reference):
        target = f[obj]
        if isinstance(target, h5py.Dataset):
            return np.array(target[()])
        return target

    if isinstance(obj, np.ndarray):
        # numeric directly
        if not _is_hdf5_ref_dtype(obj.dtype):
            return np.array(obj)

        # array of refs
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

            R = _read_array(f, cov_refs[k], dtype=float, matlab_fix=True)
            R = np.atleast_2d(R)

            par_g = _deref(f, par_refs[k])
            if not isinstance(par_g, h5py.Group):
                raise TypeError(f"partials[{k}] did not resolve to a group.")

            if "wrt_X" not in par_g:
                raise KeyError(f"partials[{k}] missing 'wrt_X'.")

            wrt_obj = par_g["wrt_X"]
            Hk = _deref(f, wrt_obj)
            if isinstance(Hk, list):
                if len(Hk) == 0:
                    raise ValueError(f"partials[{k}].wrt_X is an empty ref list.")
                Hk = Hk[0]

            Hk = np.array(Hk, dtype=float)
            Hk = _matlab_to_numpy_numeric(Hk)  # <-- critical transpose fix
            Hk = np.atleast_2d(Hk)

            Nk = r.size

            # If it still came in as (n_state x Nk), fix by transpose (extra safety)
            if Hk.shape[0] != Nk and Hk.shape[1] == Nk:
                Hk = Hk.T

            if Hk.shape[0] != Nk:
                raise ValueError(
                    f"Block {k}: Hk shape {Hk.shape} incompatible with len(r) Nk={Nk}. "
                    "This usually means MATLAB->NumPy orientation is still wrong."
                )

            if R.shape != (Nk, Nk):
                raise ValueError(
                    f"Block {k}: R shape {R.shape} incompatible with Nk={Nk}."
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

            # extra safety: must be square
            if Phi.shape[0] != Phi.shape[1]:
                raise ValueError(
                    f"trajectory_ref[{k}].STM not square: shape={Phi.shape}"
                )

            nodes.append(TrajNode(time=t, Phi0k=Phi))

    return nodes


def load_P0_v73(mat_path: str) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        if "P0" not in f:
            raise KeyError("Missing P0 in MAT.")
        P0 = np.array(f["P0"][()], dtype=float)
        P0 = _matlab_to_numpy_numeric(P0)  # P0 is 2D -> transpose fix
        return P0


# =========================
# Whitening + mapping dx0 -> residuals
# =========================


def whiten_block(
    r: np.ndarray, A: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (r_w, A_w) such that:
        r_w - A_w dx = L^{-1}(r - A dx),  with R = L L^T
    Fast path if diagonal.
    """
    R = np.atleast_2d(R)
    diag = np.diag(R)

    # diagonal fast-path
    if np.allclose(R, np.diag(diag), rtol=0.0, atol=0.0):
        s = np.sqrt(np.maximum(diag, 1e-30))
        return r / s, A / s[:, None]

    # SPD path
    try:
        L = np.linalg.cholesky(R)
        return np.linalg.solve(L, r), np.linalg.solve(L, A)
    except np.linalg.LinAlgError:
        # fallback eigen-whitening
        w, V = np.linalg.eigh(R)
        w = np.maximum(w, 1e-30)
        Winvhalf = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        return Winvhalf @ r, Winvhalf @ A


class LinearizedRCVODx0:
    """
    Residual model for sampling in dx0:
        e(dx0) = r - (Hk Phi0k) dx0
    Optionally whiten block-by-block with Rk^{-1/2}.
    """

    def __init__(
        self,
        blocks: List[MeasBlock],
        traj: List[TrajNode],
        whiten: bool = True,
        time_tol: float = 1e-5,  # seconds
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

        # Precompute stacked r and A = Hk Phi0k
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

        self.r = np.concatenate(rs, axis=0)  # (Ntot,)
        self.A = np.vstack(As)  # (Ntot, n_state)
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
            raise ValueError(f"Phi0k has wrong shape {Phi0k.shape}")
        return Phi0k

    def residuals(
        self, dx0: np.ndarray, idx_solve: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Return e(dx0) in whitened units (if whiten=True):
            e = r - A dx0
        where A = stacked (Hk Phi0k).
        """
        dx0 = np.asarray(dx0, dtype=float).reshape(-1)

        if idx_solve is None:
            if dx0.size != self.n_state:
                raise ValueError(f"dx0 size {dx0.size} != n_state {self.n_state}")
            return self.r - self.A @ dx0

        idx_solve = np.asarray(idx_solve, dtype=int).reshape(-1)
        if dx0.size != idx_solve.size:
            raise ValueError(f"dx0 size {dx0.size} != len(idx_solve) {idx_solve.size}")

        return self.r - self.A[:, idx_solve] @ dx0


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
    """
    Two figures:
      1) Prefit: time series + histogram
      2) Postfit: time series + histogram
    Residuals assumed already normalized/whitened (dimensionless).
    """
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

    # MCMC settings (edit freely)
    N_SAMPLES = 2000
    N_WALKERS = 128
    BURN_IN = 300
    THIN = 20
    SPHERICAL_SPREAD = 1e-1
    METHOD_OPTIMIZE = "Powell"

    if not os.path.isfile(MAT_FILE):
        raise FileNotFoundError(f"MAT file not found: {MAT_FILE}")

    os.makedirs(os.path.dirname(SAVE_PREFIX), exist_ok=True)

    print(f"[LOAD] {MAT_FILE}")
    blocks = load_sorted_measurements_v73(MAT_FILE)
    traj = load_trajectory_ref_v73(MAT_FILE)
    P0 = load_P0_v73(MAT_FILE)

    print(f"[OK] blocks: {len(blocks)}")
    print(f"[OK] traj nodes: {len(traj)}")

    # Quick STM sanity at start: Phi(t0) should be ~I
    Phi0 = traj[0].Phi0k
    Ierr = np.linalg.norm(Phi0 - np.eye(Phi0.shape[0]))
    print(f"[CHECK] ||Phi(t0)-I|| = {Ierr:.3e}")

    # Build dx0 model with STM mapping + whitening
    model_lin = LinearizedRCVODx0(blocks, traj, whiten=True, time_tol=TIME_TOL)

    n_state = P0.shape[0]
    if model_lin.n_state != n_state:
        raise ValueError(
            f"State size mismatch: P0 is {n_state}, Phi is {model_lin.n_state}"
        )

    idx_solve = np.arange(n_state, dtype=int)

    # Prefit at dx0=0
    pre = model_lin.residuals(np.zeros(idx_solve.size), idx_solve=idx_solve)
    dof = pre.size - idx_solve.size
    chi2 = float(pre @ pre)
    print(f"[SANITY] chi2={chi2:.6e} dof={dof} chi2_red={chi2/max(dof,1):.6e}")

    # Priors from P0 diagonal (independent normals; consistent with your whitening workflow)
    sig0 = np.sqrt(np.maximum(np.diag(P0), 0.0))
    priors = [norm(loc=0.0, scale=(s if s > 0 else 1.0)) for s in sig0[idx_solve]]

    def residuals_func(dx0_solve: np.ndarray) -> np.ndarray:
        return model_lin.residuals(dx0_solve, idx_solve=idx_solve)

    # ---- Run your MCMC ----
    try:
        from MCMC import MCMCModel
    except Exception as e:
        raise ImportError(
            "Could not import your MCMCModel. Ensure MCMC.py is on PYTHONPATH and provides MCMCModel.\n"
            f"Import error: {e}"
        ) from e

    mcmc = MCMCModel(
        residuals_func=residuals_func,
        initial_params=np.zeros(idx_solve.size),
        param_priors=priors,
        observed_data=np.zeros(1),  # placeholder
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
    post = residuals_func(dx0_hat)

    chi2_post = float(post @ post)
    print(f"[MCMC] chi2_red @ dx0_hat = {chi2_post/max(dof,1):.6e}")
    print(f"[MCMC] ||dx0_hat|| = {np.linalg.norm(dx0_hat):.6e}")

    # ---- Plots ----
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
    ]:
        try:
            getattr(mcmc, fn)()
        except Exception as e:
            print(f"[INFO] {fn} not available / failed:", e)


if __name__ == "__main__":
    main()
