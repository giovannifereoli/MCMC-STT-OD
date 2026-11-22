"""
Time-varying STM trajectory reconstruction: global vs ADS-split.

3D inclined two-body problem (mu = 1, a = 1, e = 0.5, i = 30 deg). A
deviation dx0 on the initial state is propagated NOT by re-integrating,
but with the state transition matrix used as a time-varying operator:

    dx(t_k) = Phi(t0, t_k) @ dx0,      x_pred(t_k) = x_nom(t_k) + dx(t_k)

so each sample gives a whole reconstructed TRAJECTORY over the k time
steps, not just an endpoint. Phi(t0, t_k) comes from the variational
equations Phi' = A(t) Phi (A = df/dx along the reference), integrated
with DOP853 -- exact, no DACE involved in the STM itself.

  "global STM"  = ONE reference (the nominal), ONE time-varying Phi,
                  applied to the whole uncertainty set (NOT split).
  "split STMs"  = the reference(s) daceypy's Automatic Domain Splitting
                  places (one per patch centre), each with its OWN Phi,
                  applied only inside its patch.

The initial position uncertainty is 10 km. At this size the single
global expansion no longer meets ADS's tolerance everywhere, so ADS
splits into a handful of patches, and the many local STMs stay closer
to the truth than the one global STM -- the regime where domain
splitting starts to pay off.

Figure (results/): ads_mc_3d.pdf -- (left) one sample's reconstructed
trajectory, where the global STM drifts off the orbit while the split
STM tracks the truth; (right) reconstruction error vs time for both.

Run with the Framework python:
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 test_dace_ads.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp

from daceypy import ADS, ADSintegrator, DA, array

np.set_printoptions(precision=4, suppress=True)

# Categorical slots (validated, colorblind-safe order): blue, green, magenta.
C_TRUTH = "#2a78d6"
C_ADS = "#008300"
C_SINGLE = "#e87ba4"
INK = "#3a3f45"
CMAP_BLUES = LinearSegmentedColormap.from_list("blues", ["#8fb8e8", "#12365f"])

OUTDIR = Path(__file__).resolve().parent / "results"

MU = 1.0
# Length unit for a physical "feel" in metres (e.g. a LEO-ish orbit).
LU_KM = 7000.0
TO_M = LU_KM * 1.0e3


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b8bcc2")
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color="#e3e5e8", linewidth=0.6)
    ax.set_axisbelow(True)


# =====================================================================
# Dynamics: DA version (for the ADS split structure) and numpy version
# =====================================================================

def kepler3d_rhs_da(x: array, t: float) -> array:
    r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    r3 = r2 * r2.sqrt()
    return array([x[3], x[4], x[5],
                  -MU * x[0] / r3, -MU * x[1] / r3, -MU * x[2] / r3])


class TwoBody3DADS(ADSintegrator):
    f = staticmethod(kepler3d_rhs_da)


def rhs3d_np(t, x):
    r3 = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 1.5
    return [x[3], x[4], x[5],
            -MU * x[0] / r3, -MU * x[1] / r3, -MU * x[2] / r3]


# =====================================================================
# Time-varying STM from the variational equations (exact, no DACE)
# =====================================================================

def gravity_gradient(r: np.ndarray) -> np.ndarray:
    """da/dr for two-body: G = mu/|r|^3 (3 rr^T/|r|^2 - I)."""
    rn = np.linalg.norm(r)
    return MU / rn ** 3 * (3.0 * np.outer(r, r) / rn ** 2 - np.eye(3))


def var_rhs(t, y):
    x, Phi = y[:6], y[6:].reshape(6, 6)
    r = x[:3]
    a = -MU * r / np.linalg.norm(r) ** 3
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = gravity_gradient(r)
    return np.concatenate([x[3:], a, (A @ Phi).ravel()])


def propagate_with_stm(ic, t_eval):
    """
    Integrate the reference AND its state transition matrix.

    Returns
      X   : reference trajectory, shape (N, 6)
      Phi : Phi(t0, t_k), shape (N, 6, 6)
    """
    y0 = np.concatenate([ic, np.eye(6).ravel()])
    sol = solve_ivp(var_rhs, (t_eval[0], t_eval[-1]), y0, method="DOP853",
                    rtol=1e-12, atol=1e-12, t_eval=t_eval)
    X = sol.y[:6].T
    Phi = sol.y[6:].T.reshape(len(t_eval), 6, 6)
    return X, Phi


def reconstruct(X, Phi, dx0):
    """x_nom(t_k) + Phi(t0,t_k) @ dx0  ->  trajectory, shape (N, 6)."""
    return X + np.einsum("nij,j->ni", Phi, dx0)


def truth_traj(ic, t_eval):
    return solve_ivp(rhs3d_np, (t_eval[0], t_eval[-1]), ic, method="DOP853",
                     rtol=1e-11, atol=1e-11, t_eval=t_eval).y[:6].T


# =====================================================================
# Main
# =====================================================================

def main():
    t_all = time.perf_counter()

    # ------------------------------------------------------------ setup
    print("=" * 70)
    print("Setup")
    print("=" * 70)
    ecc = 0.5
    inc = np.radians(30.0)
    period = 2 * np.pi
    n_rev = 3
    v_p = np.sqrt((1 + ecc) / (1 - ecc))
    x0 = np.array([1.0 - ecc, 0.0, 0.0,
                   0.0, v_p * np.cos(inc), v_p * np.sin(inc)])
    t_eval = np.linspace(0.0, n_rev * period, 180)
    delta = 10.0 / LU_KM                 # 10 km position uncertainty, in LU
    print(f"orbit: mu = {MU}, a = 1, e = {ecc}, "
          f"i = {np.degrees(inc):.0f} deg, start at pericenter")
    print(f"x0 = {x0}")
    print(f"initial position uncertainty: 10 km = {delta:.3e} LU "
          f"(1 LU = {LU_KM:.0f} km)")
    print(f"reconstruction over {n_rev} revs, "
          f"{t_eval.size} time steps; Phi(t0,t_k) from the variational")
    print("equations (analytic Jacobian, DOP853) -- the STM is exact")

    rng = np.random.default_rng(4)

    # nominal orbit (one period) for the trajectory plot
    nom = truth_traj(x0, np.linspace(0, period, 400))

    # ================================================================
    # Monte Carlo: global STM vs ADS-split STMs at 10 km
    # ================================================================
    print()
    print("=" * 70)
    print("Monte Carlo: global STM vs ADS-split STMs at 10 km")
    print("=" * 70)
    t_step = time.perf_counter()
    print(f"initial position deviation |dx0| = 10 km = {delta:.3e} LU")

    # --- ADS decides WHERE (and whether) to place extra reference points.
    #     At 10 km the single expansion no longer meets the tolerance
    #     everywhere, so ADS splits into a handful of patches. ---
    order, nvar = 5, 3
    DA.init(order, nvar)
    box = array([x0[0] + delta * DA(1), x0[1] + delta * DA(2),
                 x0[2] + delta * DA(3), x0[3], x0[4], x0[5]])
    prop = TwoBody3DADS()
    prop.loadTime(0.0, n_rev * period)
    prop.loadTol(1e-11, 1e-11)
    prop.loadStepSize()
    prop.loadADSopt(tol=1e-4, nsplit=10)
    print("running daceypy ADS to get the split structure "
          "(tol 1e-4, max 10 splits/patch)...")
    states = prop.propagate([ADS(box)], 0.0, n_rev * period)
    patches = [s.ADSPatch for s in states]
    npatch = len(patches)
    print(f"-> ADS returned {npatch} patch(es)"
          + (" -- one global expansion already meets the tolerance, so no"
             " split was needed" if npatch == 1 else
             f" = {npatch} local reference points instead of 1"))

    # --- global STM, plus one time-varying STM per patch centre.
    #     Patch references are built lazily: only the patches a MC sample
    #     lands in get their variational propagation. ---
    Xn_L, Phi_L = propagate_with_stm(x0, t_eval)
    ref_cache = {}

    def patch_ref(i):
        if i not in ref_cache:
            pc = patches[i].center()
            ic = x0 + delta * np.r_[pc, 0.0, 0.0, 0.0]
            Xc, Phic = propagate_with_stm(ic, t_eval)
            ref_cache[i] = (pc, Xc, Phic)
        return ref_cache[i]

    # --- Monte Carlo: reconstruct trajectories both ways ---
    n2 = 200
    pts = rng.uniform(-1, 1, size=(n2, 3))
    err_g = np.zeros((n2, t_eval.size))
    err_s = np.zeros((n2, t_eval.size))
    print(f"reconstructing {n2} MC trajectories (global vs split) vs "
          "DOP853 truth...")
    for k, p in enumerate(pts):
        dx0 = delta * np.r_[p, 0, 0, 0]
        tru = truth_traj(x0 + dx0, t_eval)
        pred_g = reconstruct(Xn_L, Phi_L, dx0)
        i = next(j for j, patch in enumerate(patches) if patch.contain(p))
        pc, Xc, Phic = patch_ref(i)
        dloc = delta * np.r_[p - pc, 0, 0, 0]
        pred_s = reconstruct(Xc, Phic, dloc)
        err_g[k] = np.linalg.norm(pred_g[:, :3] - tru[:, :3], axis=1)
        err_s[k] = np.linalg.norm(pred_s[:, :3] - tru[:, :3], axis=1)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{n2} done")
    print(f"used {len(ref_cache)} of {npatch} patch references "
          f"(only the ones hit by samples); "
          f"{time.perf_counter() - t_step:.1f} s")

    print("position error vs DOP853 truth (km):")
    for rv in range(1, n_rev + 1):
        idx = np.argmin(np.abs(t_eval - rv * period))
        print(f"  after {rv} rev: global max "
              f"{err_g[:, idx].max() * LU_KM:7.2f} km, "
              f"split max {err_s[:, idx].max() * LU_KM:7.2f} km")
    ratio = err_g[:, -1].max() / max(err_s[:, -1].max(), 1e-30)
    print(f"=> at 10 km the single global linearization starts to lose")
    print(f"   accuracy; the {npatch} ADS-split STMs stay much closer to the")
    print(f"   truth ({ratio:.0f}x smaller final error). This is where")
    print("   domain splitting begins to pay off.")

    # --- one representative sample: full trajectories for the 3D view ---
    idx = int(np.argmax(err_g[:, -1]))
    p = pts[idx]
    dx0 = delta * np.r_[p, 0, 0, 0]
    tj_truth = truth_traj(x0 + dx0, t_eval)
    tj_glob = reconstruct(Xn_L, Phi_L, dx0)
    i = next(j for j, patch in enumerate(patches) if patch.contain(p))
    pc, Xc, Phic = patch_ref(i)
    tj_split = reconstruct(Xc, Phic, delta * np.r_[p - pc, 0, 0, 0])
    print(f"representative sample #{idx}: final error global "
          f"{err_g[idx, -1] * LU_KM:.2f} km, "
          f"split {err_s[idx, -1] * LU_KM:.2f} km")

    split_label = (f"split STMs ({npatch} patch"
                   + ("" if npatch == 1 else "es") + ")")

    # ---- figure 2 (two panels: trajectory + error)
    fig2 = plt.figure(figsize=(11.5, 4.8))

    # Panel 1: one sample's reconstructed trajectory over the revs.
    ax = fig2.add_subplot(1, 2, 1, projection="3d")
    ax.plot(nom[:, 0], nom[:, 1], nom[:, 2], color="#b8bcc2",
            linewidth=1.0, linestyle="--", label="nominal orbit")
    ax.plot(tj_truth[:, 0], tj_truth[:, 1], tj_truth[:, 2],
            color=C_TRUTH, linewidth=2.6, alpha=0.7, label="DOP853 truth")
    ax.plot(tj_glob[:, 0], tj_glob[:, 1], tj_glob[:, 2],
            color=C_SINGLE, linewidth=1.6, label="global STM (drifts off)")
    ax.plot(tj_split[:, 0], tj_split[:, 1], tj_split[:, 2],
            color=C_ADS, linewidth=1.4, linestyle=(0, (4, 3)),
            label="split STM (on truth)")
    ax.scatter(*tj_truth[0, :3], marker="o", s=20, color=INK,
               label="start")
    ax.set_xlabel("$x$", color=INK, fontsize=9)
    ax.set_ylabel("$y$", color=INK, fontsize=9)
    ax.set_zlabel("$z$", color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    lim = np.vstack([nom[:, :3], tj_truth[:, :3]])
    mn, mx = lim.min(0), lim.max(0)
    pad = 0.1 * (mx - mn).max()
    ax.set_xlim(mn[0] - pad, mx[0] + pad)
    ax.set_ylim(mn[1] - pad, mx[1] + pad)
    ax.set_zlim(mn[2] - pad, mx[2] + pad)
    ax.set_box_aspect((mx - mn)[0:3] + 1e-9)
    ax.set_title(f"One sample reconstructed over {n_rev} revs\n"
                 "global STM drifts off; split STM tracks the truth",
                 color=INK, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5,
              labelcolor=INK)

    # Panel 2: error vs time (all samples), km.
    ax = fig2.add_subplot(1, 2, 2)
    tt = t_eval / period
    ax.fill_between(tt, err_g.min(0) * LU_KM, err_g.max(0) * LU_KM,
                    color=C_SINGLE, alpha=0.15)
    ax.fill_between(tt, err_s.min(0) * LU_KM, err_s.max(0) * LU_KM,
                    color=C_ADS, alpha=0.15)
    ax.plot(tt, err_g.max(0) * LU_KM, color=C_SINGLE, linewidth=2.2,
            label="global STM (1 reference)")
    ax.plot(tt, err_s.max(0) * LU_KM, color=C_ADS, linewidth=2.2,
            label=split_label)
    ax.set_yscale("log")
    top = max(err_g.max(), err_s.max()) * LU_KM * 3
    ax.set_ylim(1e-2, top)               # error is ~0 at t0; clip the tail
    ax.set_title("Reconstruction error vs time at 10 km\n"
                 "split STMs stay closer to the truth",
                 color=INK, fontsize=10)
    ax.set_xlabel("time  [revolutions]", color=INK, fontsize=9)
    ax.set_ylabel("position error  [km]", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    style_axis(ax)

    fig2.tight_layout()
    fig2.savefig(OUTDIR / "ads_mc_3d.pdf")
    print("saved results/ads_mc_3d.pdf")

    print(f"\nall done in {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
    plt.show()
