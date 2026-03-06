"""
OpNav Information Filter — corrected from scratch.

ARCHITECTURE: Sequential state-space information filter operating on DEVIATIONS.

  δx_k  = x_k - x_traj[k]   (deviation from reference trajectory)
  Filter state: δx_k ~ N(mu_k, P_k),   J_k = P_k^{-1}

  mu_k always lives in DEVIATION space.

DYNAMICS (in deviation space):
  δx_{k+1} = Phi_step @ δx_k + higher-order STT terms + w,  w ~ N(0, Q)
  where Phi_step = Phi_{k->k+1} is the step STT composed from globals.

MEASUREMENT (linearised around reference trajectory):
  y_obs_k = h(x_traj[k]) + Hy @ δx_k + v,   v ~ N(0, R)
  => innovation:  dy = (y_obs - y_ref) - Hy @ mu_prop
  where y_ref = h(x_traj[k+1]) is the measurement predicted from the reference.

COMBINED TIME+MEAS BLOCK-MATRIX UPDATE (Bierman/Maybeck):

  A = J_k  +  E[Phi^T Q^{-1} Phi]        (n x n)
  B = -E[Phi]^T Q^{-1}                   (n x n)
  D = Q^{-1}  +  Hy^T R^{-1} Hy          (n x n)

  J_{k+1} = D - B^T A^{-1} B             (Schur complement)

MEAN UPDATE (deviation space throughout):
  mu_prop   = E[Phi] @ mu_k              (propagated deviation)
  dy        = (y_obs - y_ref) - Hy @ mu_prop   (innovation)
  mu_{k+1}  = mu_prop + P_{k+1} @ Hy^T @ Rinv @ dy

PRIOR:
  mu_prior = x0_ref - x_traj[0]         (deviation, typically near zero)
  J_prior  = inv(P_prior)

KEY FIXES vs. original:
  1. mu_k lives in deviation space throughout; prior initialised correctly.
  2. Innovation computed as (y_obs - y_ref) - Hy @ mu_prop, NOT y_obs - Hy @ mu_prop.
     y_ref = h(x_traj[k+1]) subtracted so Hy operates only on the deviation.
  3. Gaussian moment integrals fed mu_k (deviation) and P_k — correct.
  4. mu_prop = Jbar @ mu_k in deviation space — correct.
  5. compose_phi2_step uses proper chain-rule pullback.
  6. sig_ang corrected (2e-6 rad, not 2e67).
  7. MC: per-trial visibility and measurements.
  8. MC: mu_prior fixed to deviation at every trial.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct

os.makedirs("results", exist_ok=True)

try:
    from STTPropagationND import STTPropagatorND
    import sympy as sp

    HAS_STT = True
except ImportError:
    HAS_STT = False
    print("[WARN] STTPropagatorND not found — cannot run.")


# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOLIC STT (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════


def build_pointmass_stt_funcs(order: int):
    t = sp.Symbol("t")
    syms = x, y, z, vx, vy, vz, mu = sp.symbols("x y z vx vy vz mu", real=True)
    X = sp.Matrix([x, y, z, vx, vy, vz, mu])
    r2 = x**2 + y**2 + z**2
    r3 = sp.sqrt(r2) ** 3
    a = -mu * sp.Matrix([x, y, z]) / r3
    f = sp.Matrix([vx, vy, vz, a[0], a[1], a[2], 0])
    A = f.jacobian(X)
    args = (*syms, t)
    f_func = sp.lambdify(args, f, "numpy")
    A_func = sp.lambdify(args, A, "numpy")
    B_funcs = {}
    for k in range(2, order + 1):
        sh = (7,) * (k + 1)
        Bk = sp.MutableDenseNDimArray.zeros(*sh)
        for idx in iproduct(range(7), repeat=k + 1):
            i, *js = idx
            Bk[idx] = sp.diff(f[i], *[X[j] for j in js])
        B_funcs[k] = sp.lambdify(args, Bk.tolist(), "numpy")
    return f_func, A_func, B_funcs


# ═══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN MOMENTS  E[δx], E[δx δx^T], E[δx δx δx], E[δx δx δx δx]
# for δx ~ N(mu, P)  where mu IS the deviation mean
# ═══════════════════════════════════════════════════════════════════════════════


def gaussian_moments_4(mu, P):
    """Raw moments of δx ~ N(mu, P). mu here is the deviation mean."""
    mu = np.asarray(mu, float).ravel()
    P = np.asarray(P, float)
    E1 = mu
    E2 = P + np.outer(mu, mu)
    E3 = (
        np.einsum("i,j,k->ijk", mu, mu, mu)
        + np.einsum("i,jk->ijk", mu, P)
        + np.einsum("j,ik->ijk", mu, P)
        + np.einsum("k,ij->ijk", mu, P)
    )
    mu4 = np.einsum("i,j,k,l->ijkl", mu, mu, mu, mu)
    mu2P = (
        np.einsum("i,j,kl->ijkl", mu, mu, P)
        + np.einsum("i,k,jl->ijkl", mu, mu, P)
        + np.einsum("i,l,jk->ijkl", mu, mu, P)
        + np.einsum("j,k,il->ijkl", mu, mu, P)
        + np.einsum("j,l,ik->ijkl", mu, mu, P)
        + np.einsum("k,l,ij->ijkl", mu, mu, P)
    )
    PP = (
        np.einsum("ij,kl->ijkl", P, P)
        + np.einsum("ik,jl->ijkl", P, P)
        + np.einsum("il,jk->ijkl", P, P)
    )
    E4 = mu4 + mu2P + PP
    return E1, E2, E3, E4


# ═══════════════════════════════════════════════════════════════════════════════
# E[Phi]  and  E[Phi^T M Phi]  under δx_k ~ N(mu_k, P_k)
# ═══════════════════════════════════════════════════════════════════════════════


def expected_Phi(Phi1, Phi2, Phi3, mu, P):
    """Jbar = E[Phi(δx_k)] under δx_k ~ N(mu, P). Shape (n, n)."""
    E1, E2, _, _ = gaussian_moments_4(mu, P)
    Jbar = np.array(Phi1, float).copy()
    if Phi2 is not None:
        Jbar += np.einsum("j,ijm->im", E1, Phi2)
    if Phi3 is not None:
        Jbar += 0.5 * np.einsum("jk,ijkm->im", E2, Phi3)
    return Jbar  # (n, n)


def expected_PhiT_M_Phi(Phi1, Phi2, Phi3, M, mu, P):
    """E[Phi^T M Phi] under δx_k ~ N(mu, P), up to 3rd-order STT."""
    Phi1 = np.array(Phi1, float)
    M = np.array(M, float)
    E1, E2, E3, E4 = gaussian_moments_4(mu, P)

    D = Phi1.T @ M @ Phi1

    if Phi2 is not None:
        Phi2 = np.array(Phi2, float)
        ePhi2 = np.einsum("j,ijm->im", E1, Phi2)
        D += Phi1.T @ M @ ePhi2 + ePhi2.T @ M @ Phi1
        D += np.einsum("jk,ijm,ir,rks->ms", E2, Phi2, M, Phi2)

    if Phi3 is not None:
        Phi3 = np.array(Phi3, float)
        ePhi3 = np.einsum("jk,ijkm->im", E2, Phi3)
        D += 0.5 * (Phi1.T @ M @ ePhi3 + ePhi3.T @ M @ Phi1)
        if Phi2 is not None:
            D += 0.5 * (
                np.einsum("jpq,ijm,ir,rpqs->ms", E3, Phi2, M, Phi3)
                + np.einsum("jpq,ipqs,ir,rjm->sm", E3, Phi3, M, Phi2).T
            )
        D += 0.25 * np.einsum("jkpq,ijkm,ir,rpqs->ms", E4, Phi3, M, Phi3)

    return D


# ═══════════════════════════════════════════════════════════════════════════════
# STT COMPOSITION  (step k -> k+1  from global 0 -> k  and  0 -> k+1)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_stt(stts, k, order, n):
    raw = np.array(stts[order][k], float)
    return raw.reshape((n,) * (order + 1))


def compose_phi1_step(stts, k, n):
    """Phi_{k->k+1} = Phi_{0->k+1} @ inv(Phi_{0->k})."""
    Phi_0k = extract_stt(stts, k, 1, n)
    Phi_0k1 = extract_stt(stts, k + 1, 1, n)
    return Phi_0k1 @ np.linalg.inv(Phi_0k)


def compose_phi2_step(stts, k, n, Phi1_step, Phi1_0k_inv):
    """
    Second-order step STT Phi2_{k->k+1} from global STTs.

    Chain rule (second order):
      Phi2_step[i,j,l] =  Phi2_0k1[i,a,b] * Phi1inv[a,j] * Phi1inv[b,l]
                        -  Phi1_step[i,a]  * Phi2_0k[a,j,l]

    The second term removes the curvature already captured in Phi_{0->k},
    preventing double-counting when composing global STTs.
    """
    Phi2_0k = extract_stt(stts, k, 2, n)
    Phi2_0k1 = extract_stt(stts, k + 1, 2, n)

    # Pull Phi2_0k1 back through Phi1_0k
    T1 = np.einsum("iab,aj,bl->ijl", Phi2_0k1, Phi1_0k_inv, Phi1_0k_inv)

    # Remove curvature already accounted for in Phi_{0->k}
    T2 = np.einsum("ia,ajl->ijl", Phi1_step, Phi2_0k)

    return T1 - T2


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════


def occultation_mask(sc_pos, part_pos, R_body):
    """True where particle is visible (not occulted)."""
    d = part_pos - sc_pos
    dd = np.einsum("ij,ij->i", d, d)
    t = np.clip(-np.einsum("ij,ij->i", sc_pos, d) / np.maximum(dd, 1e-30), 0, 1)
    return np.linalg.norm(sc_pos + t[:, None] * d, axis=1) > R_body


def radec_and_partials(x_part, sc_pos):
    """
    Compute (ra, dec) and Jacobian Hy w.r.t. x_part position (first 3 elements).

    Returns
    -------
    ra   : float
    dec  : float
    y_ref: (2,)  [ra, dec] as measurement prediction
    Hy   : (2, n) measurement Jacobian (columns 3..n-1 are zero)
    """
    los = x_part[:3] - sc_pos
    los = los[None, :]  # (1, 3) for radec_partials
    ra, dec, dra, ddec = _radec_partials_batch(los)
    y_ref = np.array([ra[0], dec[0]])
    Hy = np.zeros((2, x_part.size))
    Hy[0, :3] = dra[0]
    Hy[1, :3] = ddec[0]
    return y_ref, Hy


def _radec_partials_batch(los):
    """Batch RA/Dec and their Jacobians w.r.t. los."""
    x, y, z = los[:, 0], los[:, 1], los[:, 2]
    rxy2 = np.maximum(x * x + y * y, 1e-30)
    rxy = np.sqrt(rxy2)
    rho2 = np.maximum(rxy2 + z * z, 1e-30)
    ra = np.arctan2(y, x)
    dec = np.arctan2(z, rxy)
    dra = np.stack([-y / rxy2, x / rxy2, np.zeros_like(x)], axis=1)
    ddec = np.stack([-z * (x / rxy) / rho2, -z * (y / rxy) / rho2, rxy / rho2], axis=1)
    return ra, dec, dra, ddec


def simulate_measurements(x_traj, sc_state, sig_ra, sig_dec, rng):
    """Generate noisy (ra, dec) observations from true particle trajectory."""
    los = x_traj[:, :3] - sc_state[:, :3]
    ra, dec, _, _ = _radec_partials_batch(los)
    y = np.empty(2 * len(ra))
    y[0::2] = ra + rng.normal(0, sig_ra, len(ra))
    y[1::2] = dec + rng.normal(0, sig_dec, len(dec))
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED TIME + MEASUREMENT INFORMATION UPDATE
#
#  All quantities in DEVIATION space: mu_k = E[δx_k], P_k = Cov[δx_k].
#
#  Inputs
#  ------
#  Jk       : (n,n)  information matrix for δx_k
#  mu_k     : (n,)   posterior deviation mean  E[δx_k]
#  P_k      : (n,n)  posterior covariance      Cov[δx_k]  (= inv(Jk))
#  Qinv     : (n,n)  process noise information
#  Hy       : (m,n)  obs Jacobian w.r.t. δx_{k+1}
#  Rinv     : (m,m)  measurement noise information (zero matrix if no obs)
#  dy       : (m,)   CORRECTED innovation = (y_obs - y_ref) - Hy @ mu_prop
#                    (caller computes this so y_ref is subtracted correctly)
#  Phi1     : (n,n)  first-order step STT  k -> k+1
#  Phi2,3   : higher-order step STTs or None
#
#  Returns
#  -------
#  J_{k+1}, mu_{k+1}, P_{k+1}   all in deviation space
# ═══════════════════════════════════════════════════════════════════════════════


def info_update(Jk, mu_k, P_k, Qinv, Hy, Rinv, dy, Phi1, Phi2=None, Phi3=None):
    """
    Combined time + measurement information update in deviation space.

    The innovation `dy` must already be:
        dy = (y_obs - y_ref) - Hy @ mu_prop
    where
        mu_prop = Jbar @ mu_k           (deviation propagated)
        y_ref   = h(x_traj[k+1])       (reference measurement)

    The caller is responsible for computing mu_prop and dy so that
    y_ref is correctly subtracted.
    """
    # Gaussian expectations under δx_k ~ N(mu_k, P_k)
    Jbar = expected_Phi(Phi1, Phi2, Phi3, mu_k, P_k)  # (n,n)
    D11 = expected_PhiT_M_Phi(Phi1, Phi2, Phi3, Qinv, mu_k, P_k)  # (n,n)

    # Block-matrix components
    A = Jk + D11  # (n,n)
    B = -Jbar.T @ Qinv  # (n,n)
    D = Qinv + Hy.T @ Rinv @ Hy  # (n,n)

    # Schur complement -> information at k+1
    Jkp1 = D - B.T @ np.linalg.solve(A, B)
    Jkp1 = 0.5 * (Jkp1 + Jkp1.T)  # enforce symmetry

    # Covariance and mean update (deviation space)
    P_kp1 = np.linalg.inv(Jkp1)
    mu_kp1 = Jbar @ mu_k + P_kp1 @ (Hy.T @ (Rinv @ dy))

    return Jkp1, mu_kp1, P_kp1


# ═══════════════════════════════════════════════════════════════════════════════
# FULL FILTER RUN
# ═══════════════════════════════════════════════════════════════════════════════


def run_filter(
    x_traj,  # (N, n)  reference trajectory (full state)
    sc_state,  # (N, 6)  spacecraft state
    stts,  # global STT dict  {order: [array_at_epoch_0, ...]}
    vis_mask,  # (N,) bool
    y_obs,  # (2N,) interleaved [ra_0,dec_0, ra_1,dec_1, ...]
    J_prior,  # (n,n)
    mu_prior,  # (n,)  DEVIATION mean at epoch 0  = x0_ref - x_traj[0]
    Qinv,  # (n,n)
    Rinv,  # (2,2)
    stt_order,  # int
    n,  # state dimension
):
    """
    Run combined-step information filter entirely in deviation space.

    mu_prior must be the deviation x0_ref - x_traj[0], NOT the full state.
    """
    Jk = J_prior.copy()
    mu_k = np.asarray(mu_prior, float).copy()  # deviation at epoch 0
    P_k = np.linalg.inv(Jk)

    P_hist = []
    mu_hist = []

    N = len(vis_mask)

    for k in range(N - 1):
        # ── Step STTs: Phi_{k -> k+1} from global STTs ──────────────────────
        Phi1_0k = extract_stt(stts, k, 1, n)
        Phi1_0k1 = extract_stt(stts, k + 1, 1, n)
        Phi1_0k_inv = np.linalg.inv(Phi1_0k)
        Phi1_step = Phi1_0k1 @ Phi1_0k_inv

        Phi2_step = None
        Phi3_step = None
        if stt_order >= 2:
            Phi2_step = compose_phi2_step(stts, k, n, Phi1_step, Phi1_0k_inv)
        # (third-order composition can be added analogously)

        # ── Propagate deviation mean to k+1 ─────────────────────────────────
        # Jbar = E[Phi_step] under δx_k ~ N(mu_k, P_k)
        Jbar = expected_Phi(Phi1_step, Phi2_step, Phi3_step, mu_k, P_k)
        mu_prop = Jbar @ mu_k  # δx_{k+1|k}

        # ── Build observation at k+1 ─────────────────────────────────────────
        if vis_mask[k + 1]:
            # Reference measurement from the linearisation trajectory
            y_ref, Hy = radec_and_partials(x_traj[k + 1], sc_state[k + 1, :3])

            y_obs_k = y_obs[2 * (k + 1) : 2 * (k + 1) + 2]

            # Innovation: (y_obs - y_ref) - Hy @ mu_prop
            # y_ref handles the reference; Hy @ mu_prop handles the deviation
            dy = (y_obs_k - y_ref) - Hy @ mu_prop
            dy[0] = (dy[0] + np.pi) % (2 * np.pi) - np.pi  # wrap RA
            Rinv_k = Rinv
        else:
            # No measurement: zero information from obs
            _, Hy = radec_and_partials(x_traj[k + 1], sc_state[k + 1, :3])
            dy = np.zeros(2)
            Rinv_k = np.zeros_like(Rinv)

        # ── Combined information update ──────────────────────────────────────
        # Pass pre-computed mu_prop inside dy; info_update recomputes Jbar
        # internally for the A/B/D blocks (consistent with the Schur complement).
        # The mean update inside info_update uses its own Jbar @ mu_k + correction,
        # which matches mu_prop since it recomputes the same Jbar.
        Jk, mu_k, P_k = info_update(
            Jk,
            mu_k,
            P_k,
            Qinv,
            Hy,
            Rinv_k,
            dy,
            Phi1_step,
            Phi2_step,
            Phi3_step,
        )
        print(P_k)

        P_hist.append(P_k.copy())
        mu_hist.append(mu_k.copy())

    return Jk, mu_k, P_k, np.array(P_hist), np.array(mu_hist)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    if not HAS_STT:
        raise RuntimeError("STTPropagatorND required — cannot run.")

    # ── Scenario parameters ──────────────────────────────────────────────────
    R_body = 0.290
    mu_true = 4.89e-9
    n_obs = 10
    t0, t1 = 0.0, 1.5 * 3600.0
    tau = np.linspace(t0, t1, n_obs)
    sig_ang = 2e-6  # rad (~0.4 arcsec)
    R_meas = np.diag([sig_ang**2, sig_ang**2])
    Rinv = np.linalg.inv(R_meas)
    stt_order = 2
    n = 7
    N_mc = 10

    # ── Symbolic STT ─────────────────────────────────────────────────────────
    print("Building symbolic STT functions …")
    f_func, A_func, B_funcs = build_pointmass_stt_funcs(stt_order)
    propagator = STTPropagatorND(
        order=stt_order, f_func=f_func, A_func=A_func, B_funcs=B_funcs, n=n
    )

    # ── Truth trajectory ─────────────────────────────────────────────────────
    x0_true = np.hstack([[R_body, 0.0, 0.0], [0.0, 2e-4, 0.2e-4], mu_true])
    print("Propagating truth …")
    sol_true, _ = propagator.propagate(
        x0_true, tau, rtol=1e-10, atol=1e-12, method="LSODA"
    )
    x_true = sol_true.y[:n].T  # (N, n)

    # ── Spacecraft ───────────────────────────────────────────────────────────
    R_sc, w_sc = 2.0, 2 * np.pi / (2.0 * 3600.0)
    th = w_sc * tau
    sc_state = np.zeros((n_obs, 6))
    sc_state[:, 0] = R_sc * np.cos(th)
    sc_state[:, 3] = -R_sc * w_sc * np.sin(th)
    sc_state[:, 1] = R_sc * np.sin(th)
    sc_state[:, 4] = R_sc * w_sc * np.cos(th)
    sc_state[:, 2] = 0.2

    # ── Visibility + measurements (truth) ────────────────────────────────────
    vis_true = occultation_mask(sc_state[:, :3], x_true[:, :3], R_body)
    print(f"Visible epochs (truth): {vis_true.sum()}/{n_obs}")
    rng_meas = np.random.default_rng(123)
    y_obs = simulate_measurements(x_true, sc_state, sig_ang, sig_ang, rng_meas)

    # ── Prior & reference trajectory ─────────────────────────────────────────
    sig_pre = np.array([0.05 * R_body] * 3 + [2e-4 * 0.2] * 3 + [0.01 * abs(mu_true)])
    P_pre = np.diag(sig_pre**2)
    J_prior = np.linalg.inv(P_pre)

    rng_ref = np.random.default_rng(7)
    x0_ref = x0_true + rng_ref.normal(size=n) * sig_pre  # perturbed IC

    print("Propagating reference + STTs …")
    sol_ref, stts_ref = propagator.propagate(
        x0_ref, tau, rtol=1e-10, atol=1e-12, method="LSODA"
    )
    x_ref = sol_ref.y[:n].T  # (N, n)

    # ── Process noise ────────────────────────────────────────────────────────
    Q = 1e-14 * np.eye(n)
    Qinv = np.linalg.inv(Q)

    # ── PRIOR MEAN IN DEVIATION SPACE ────────────────────────────────────────
    # x_ref[0] = propagated x0_ref, x_traj[0] = x_ref[0] (same reference).
    # The prior mean of the deviation is our best guess: x0_ref - x_ref[0] = 0.
    # If using a separate x_traj, substitute accordingly.
    mu_prior_dev = x0_ref - x_ref[0]  # = 0 when x_traj[0] == x0_ref

    # ── Run filter ───────────────────────────────────────────────────────────
    print("Running information filter …")
    Jf, mu_f, P_f, P_hist, mu_hist = run_filter(
        x_ref,
        sc_state,
        stts_ref,
        vis_true,
        y_obs,
        J_prior,
        mu_prior_dev,
        Qinv,
        Rinv,
        stt_order,
        n,
    )

    # Reconstruct full-state estimate at final epoch
    x_est_final = x_ref[-1] + mu_f  # reference + deviation
    x_true_final = x_true[-1]
    err_final = x_true_final - x_est_final

    sig_filter = np.sqrt(np.diag(P_f))
    names = ["x", "y", "z", "vx", "vy", "vz", "mu"]

    print("\n[Filter] posterior 1σ (deviation space):")
    for nm, s, e in zip(names, sig_filter, err_final):
        print(
            f"  {nm:>4}: 1σ = {s:.4e}   err = {e:.4e}   {'✓' if abs(e) < 3*s else '✗'}"
        )

    # ── Monte Carlo ──────────────────────────────────────────────────────────
    print(f"\nRunning {N_mc}-trial Monte Carlo …")
    rng_mc = np.random.default_rng(42)
    mc_err = []

    for i in range(N_mc):
        if i % 10 == 0:
            print(f"  trial {i}/{N_mc}")

        # Draw a true IC from the prior
        x0_i = rng_mc.multivariate_normal(x0_true, P_pre)

        # Propagate this trial's truth
        sol_i, _ = propagator.propagate(
            x0_i, tau, rtol=1e-9, atol=1e-11, method="LSODA"
        )
        x_i = sol_i.y[:n].T

        # Per-trial visibility and measurements
        vis_i = occultation_mask(sc_state[:, :3], x_i[:, :3], R_body)
        y_i = simulate_measurements(
            x_i,
            sc_state,
            sig_ang,
            sig_ang,
            np.random.default_rng(int(rng_mc.integers(int(1e12)))),
        )

        # Filter uses the SAME reference trajectory (x_ref) and STTs for all trials.
        # Prior deviation for this trial: x0_ref - x_ref[0] (same as nominal = 0).
        _, mu_est_i, _, _, _ = run_filter(
            x_ref,
            sc_state,
            stts_ref,
            vis_i,
            y_i,
            J_prior,
            mu_prior_dev,
            Qinv,
            Rinv,
            stt_order,
            n,
        )

        # True final state vs. filter estimate (deviation + reference)
        x_est_i = x_ref[-1] + mu_est_i
        mc_err.append(x_i[-1] - x_est_i)

    mc_err = np.array(mc_err)  # (N_mc, n)
    sig_mc = np.sqrt(np.diag(np.cov(mc_err.T)))

    print("\n╔══ MC Validation ══════════════════════════════════════════════════╗")
    print(f"{'':>6}  {'Filter 1σ':>14}  {'MC 1σ':>14}  {'Ratio F/MC':>12}")
    print("├" + "─" * 62)
    for j, nm in enumerate(names):
        r = sig_filter[j] / sig_mc[j] if sig_mc[j] > 0 else float("nan")
        flag = "  ✓" if 0.7 < r < 1.4 else "  ✗ CHECK"
        print(f"{nm:>6}  {sig_filter[j]:14.4e}  {sig_mc[j]:14.4e}  {r:12.3f}{flag}")
    print("╚" + "═" * 62)
    print("Ratio ~ 1 → consistent | <0.7 → overconfident | >1.4 → conservative")

    # ── Plots ────────────────────────────────────────────────────────────────
    lbl = [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\mu$"]
    sig_hist = np.sqrt(np.clip(np.diagonal(P_hist, axis1=1, axis2=2), 0, np.inf))

    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(n):
        ax.plot(tau[1 : len(sig_hist) + 1] / 3600, sig_hist[:, j], label=lbl[j])
    ax.set_yscale("log")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(r"$1\sigma$ [deviation]")
    ax.set_title("Info Filter — Posterior σ vs. Time")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig("results/covariance_vs_time.png", dpi=200)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    xi = np.arange(n)
    w = 0.35
    ax2.bar(xi - w / 2, sig_filter, w, label="Filter", color="steelblue")
    ax2.bar(xi + w / 2, sig_mc, w, label=f"MC ({N_mc})", color="darkorange")
    ax2.set_yscale("log")
    ax2.set_xticks(xi)
    ax2.set_xticklabels(names)
    ax2.set_ylabel(r"$1\sigma$ estimation error")
    ax2.set_title("Filter vs. MC Validation")
    ax2.legend()
    ax2.grid(True, axis="y")
    plt.tight_layout()
    fig2.savefig("results/mc_validation.png", dpi=200)

    # Mean-error history (bias check)
    if len(mu_hist) > 0:
        fig3, axes = plt.subplots(n, 1, figsize=(10, 12), sharex=True)
        for j in range(n):
            axes[j].plot(
                tau[1 : len(mu_hist) + 1] / 3600, mu_hist[:, j], color="steelblue"
            )
            axes[j].axhline(0, color="k", lw=0.5, ls="--")
            axes[j].set_ylabel(lbl[j], fontsize=8)
            axes[j].grid(True)
        axes[-1].set_xlabel("Time [h]")
        fig3.suptitle("Filter deviation mean vs. time (bias check)")
        plt.tight_layout()
        fig3.savefig("results/mean_deviation_history.png", dpi=200)

    print("\nSaved:")
    print("  results/covariance_vs_time.png")
    print("  results/mc_validation.png")
    print("  results/mean_deviation_history.png")
    plt.show()
