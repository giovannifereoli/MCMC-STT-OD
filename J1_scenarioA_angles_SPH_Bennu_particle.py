"""
Scenario (from scratch, but uses your existing STTPropagator + MCMCModel):

- Spacecraft (OSIRIS-REx) trajectory is TRUTH from SPICE SPK (Bennu-centered, J2000).
- Particle detaches from Bennu surface (point in body-fixed), then propagates in J2000 (Bennu-centered).
- Bennu rotates with constant spin about its pole (truth alpha/delta). No tau state.
- Particle dynamics uses degree-2 gravity potential in body-fixed:
    U = mu/r * (1 + (R_ref/r)^2 * sum_{m=0..2} P2m(sinφ)*(C2m cos mλ + S2m sin mλ))
  Acceleration is computed as a = -∇U in body-fixed, then rotated to inertial.

- You estimate:
    theta = [δr0(3), δv0(3), δmu(1), δC20, δC21, δS21, δC22, δS22]  -> 12 params
  about a reference x0_ref (also 12D state).

- Measurements: RA/DEC from spacecraft to particle, with occultation mask (sphere proxy).

Pipeline:
  1) Propagate truth particle with truth params.
  2) Query SPICE for spacecraft truth at the same epochs.
  3) Generate noisy RA/DEC on visible epochs only.
  4) Build reference (perturb truth) and propagate ref + STTs with STTPropagator.
  5) Stage-1 full nonlinear MAP (optional, slow): re-propagate each evaluation.
  6) Stage-2 STT-based MAP: use propagate_deviation with your STTPropagator.
  7) MCMC around STT-based residual function.

IMPORTANT REQUIREMENT:
- Your STTPropagator must support time-dependent dynamics: f(x,t), A(x,t), Bk(x,t).
  If your STTPropagator currently calls f_func(*x) without t, you must patch it to pass t:
      f = f_func(*x, t)
      A = A_func(*x, t)
      Bk = B_funcs[k](*x, t)
  Everything else in this script assumes that.

You MUST provide:
- A meta-kernel that furnsh() loads:
    * OSIRIS-REx SPK (spacecraft)
    * Bennu SPK (if needed by your setup) / leapseconds
- Names consistent with kernels:
    SC_NAME, CENTER, FRAME (J2000 typically).

Units:
- km, km/s, seconds (ET)
"""

import sympy as sp
import numpy as np
from itertools import product

import spiceypy as spice
import trimesh
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm

from STTPropagation import STTPropagator
from MCMC import MCMCModel


# ============================================================
# Geometry / utility
# ============================================================


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def radec_from_los(los_vec):
    """
    los_vec: (N,3) from observer to target in inertial
    returns ra in [0,2pi), dec in [-pi/2, pi/2]
    """
    u = los_vec / np.linalg.norm(los_vec, axis=1, keepdims=True)
    ra = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
    dec = np.arcsin(u[:, 2])
    return ra, dec


def occultation_mask(sc_pos, part_pos, R_body):
    """
    Visibility test: does the segment SC->particle intersect sphere of radius R_body at origin?
    sc_pos:   (N,3)
    part_pos: (N,3)
    returns: mask True if visible
    """
    r1 = sc_pos
    r2 = part_pos
    d = r2 - r1
    dd = np.sum(d * d, axis=1)

    # closest approach to origin along segment
    # minimize ||r1 + t d||, t in [0,1]
    t = -np.sum(r1 * d, axis=1) / dd
    t = np.clip(t, 0.0, 1.0)
    closest = r1 + t[:, None] * d

    return np.linalg.norm(closest, axis=1) > R_body


def make_bennu_rotation_matrix(alpha, delta, omega, t, w0=0.0):
    """
    inertial->body-fixed rotation for a "IAU-like" constant spin model:
      pole is (alpha, delta) in J2000, spin about body +Z by W = w0 + omega*t

    Returns R_ib (3,3) mapping r_i -> r_b.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cd, sd = np.cos(delta), np.sin(delta)
    W = w0 + omega * t
    cW, sW = np.cos(W), np.sin(W)

    # Using common IAU convention-like build:
    # R_ib = Rz(W) * Rx(pi/2 - delta) * Rz(alpha + pi/2)
    # (sign conventions differ between references; this is consistent internally if used everywhere.)
    def Rz(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def Rx(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    return Rz(W) @ Rx(np.pi / 2 - delta) @ Rz(alpha + np.pi / 2)


# ============================================================
# Degree-2 potential in body-fixed + symbolic derivatives up to any STT order
# (time dependence enters only through R_ib(t) which we embed symbolically via cos/sin(omega*t + w0))
# ============================================================


def generate_stt_functions_bennu_deg2(
    order, R_ref_km, alpha_rad, delta_rad, omega_rad_s, w0_rad=0.0
):
    """
    Build symbolic f, A, B_k for augmented 12D state:
      X = [x y z vx vy vz mu C20 C21 S21 C22 S22]
    where x,y,z,vx,vy,vz are in inertial Bennu-centered J2000,
    and gravity params are constants.
    Bennu rotates via known alpha/delta and constant spin omega.

    f_func, A_func, B_funcs MUST be called as f_func(*X, t) (t in seconds).
    """

    # ---- symbols ----
    t = sp.Symbol("t", real=True)

    x, y, z, vx, vy, vz = sp.symbols("x y z vx vy vz", real=True)
    mu, C20, C21, S21, C22, S22 = sp.symbols("mu C20 C21 S21 C22 S22", real=True)

    X = sp.Matrix([x, y, z, vx, vy, vz, mu, C20, C21, S21, C22, S22])
    r_i = sp.Matrix([x, y, z])

    # ---- rotation inertial->body: R_ib(t) ----
    alpha = sp.Float(alpha_rad)
    delta = sp.Float(delta_rad)
    omega = sp.Float(omega_rad_s)
    w0 = sp.Float(w0_rad)
    W = w0 + omega * t

    def Rz(th):
        c, s = sp.cos(th), sp.sin(th)
        return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def Rx(th):
        c, s = sp.cos(th), sp.sin(th)
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])

    R_ib = Rz(W) * Rx(sp.pi / 2 - delta) * Rz(alpha + sp.pi / 2)
    R_bi = R_ib.T

    r_b = R_ib * r_i
    xb, yb, zb = r_b[0], r_b[1], r_b[2]

    # ---- body-fixed degree-2 potential ----
    r2 = xb**2 + yb**2 + zb**2
    r = sp.sqrt(r2)

    lam = sp.atan2(yb, xb)
    phi = sp.atan2(zb, sp.sqrt(xb**2 + yb**2))
    sphi = sp.sin(phi)
    cphi = sp.cos(phi)

    # unnormalized P2m(sin(phi))
    P20 = sp.Rational(1, 2) * (3 * sphi**2 - 1)
    P21 = 3 * sphi * cphi
    P22 = 3 * cphi**2

    cos1, sin1 = sp.cos(lam), sp.sin(lam)
    cos2, sin2 = sp.cos(2 * lam), sp.sin(2 * lam)

    F2 = C20 * P20 + P21 * (C21 * cos1 + S21 * sin1) + P22 * (C22 * cos2 + S22 * sin2)

    R2 = sp.Float(R_ref_km**2)
    U = mu / r * (1 + (R2 / r2) * F2)

    # body-fixed acceleration
    a_b = -sp.Matrix([sp.diff(U, xb), sp.diff(U, yb), sp.diff(U, zb)])

    # inertial acceleration
    a_i = R_bi * a_b

    # ---- augmented dynamics (12D) ----
    f = sp.Matrix([vx, vy, vz, a_i[0], a_i[1], a_i[2], 0, 0, 0, 0, 0, 0])

    # ---- A and B tensors ----
    A = f.jacobian(X)
    B_syms = {1: A}

    for k in range(2, order + 1):
        shape = (12,) * (k + 1)
        Bk = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(range(12), repeat=k + 1):
            i, *js = idx
            deriv = sp.diff(f[i], *[X[j] for j in js])
            Bk[idx] = deriv
        B_syms[k] = Bk

    # ---- lambdify ----
    args = (x, y, z, vx, vy, vz, mu, C20, C21, S21, C22, S22, t)
    f_func = sp.lambdify(args, f, "numpy")
    A_func = sp.lambdify(args, B_syms[1], "numpy")
    B_funcs = {
        k: sp.lambdify(args, B_syms[k].tolist(), "numpy") for k in range(2, order + 1)
    }

    return f_func, A_func, B_funcs


# ============================================================
# Measurement generation (RA/DEC from SPICE SC to particle)
# ============================================================


def generate_opnav_measurements_from_sc(x_part, sc_state, sigma_ra, sigma_dec, rng):
    """
    x_part: (N,12) or (N,6) particle in inertial
    sc_state: (N,6) spacecraft inertial
    """
    los = x_part[:, :3] - sc_state[:, :3]
    ra, dec = radec_from_los(los)

    ra_meas = np.mod(ra + rng.normal(0.0, sigma_ra, size=ra.shape), 2 * np.pi)
    dec_meas = dec + rng.normal(0.0, sigma_dec, size=dec.shape)

    y = np.empty(2 * len(ra))
    y[0::2] = ra_meas
    y[1::2] = dec_meas
    return y


# ============================================================
# Batch MAP helper (same style as you already used)
# ============================================================


def compute_STT_batch_solution(residuals_func, x0, priors=None, max_nfev=40000):
    """
    residuals_func already returns normalized measurement residuals (sigma-weighted).
    priors: list of scipy.stats.norm on the parameter deltas (len n)
    """
    if priors is None:
        prior_mean = np.zeros_like(x0)
        prior_sigma = np.full_like(x0, np.inf)
    else:
        prior_mean = np.array([p.mean() for p in priors], dtype=float)
        prior_sigma = np.array([p.std() for p in priors], dtype=float)
        if np.any(prior_sigma <= 0):
            raise ValueError("Prior std must be > 0.")

    def raw_residuals(delta):
        r_meas = residuals_func(delta)
        r_pri = (delta - prior_mean) / prior_sigma
        return np.hstack([r_meas, r_pri])

    result = least_squares(
        fun=raw_residuals,
        x0=x0,
        method="trf",
        jac="2-point",
        max_nfev=max_nfev,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        verbose=2,
    )
    J = result.jac
    cov = np.linalg.inv(J.T @ J)
    return result, cov


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    # --------------------------
    # USER SETTINGS (edit these)
    # --------------------------
    META_KERNEL = "kernels/bennu_meta.tm"  # <-- EDIT
    SC_NAME = "ORX"  # <-- EDIT: name in SPK (e.g., "OSIRIS-REX" or "ORX")
    CENTER = "BENNU"  # <-- EDIT: center name in SPICE
    FRAME_I = "J2000"

    # Observation window (must be covered by SPK)
    utc0 = "2019-03-01T00:00:00"  # <-- EDIT
    utc1 = "2019-03-01T02:00:00"  # <-- EDIT
    n_obs = 120

    # Bennu physical
    R_bennu = 0.290  # km
    R_ref = R_bennu

    # Bennu spin + pole truth (use your preferred truth values)
    alpha_true = np.deg2rad(85.65)  # rad (EDIT)
    delta_true = np.deg2rad(-60.17)  # rad (EDIT)
    spin_period = 4.296057 * 3600.0  # s (EDIT)
    omega_true = 2 * np.pi / spin_period  # rad/s

    # Dynamics / STT order
    stt_order = 2  # can be 2,3,4,... (careful: symbolic cost grows fast)

    # Truth gravity params
    mu_true = 4.892e-9
    C20_true = -2.0e-5
    C21_true = 3.0e-7
    S21_true = -2.0e-7
    C22_true = 1.0e-7
    S22_true = 2.0e-7
    params_true = np.array(
        [mu_true, C20_true, C21_true, S21_true, C22_true, S22_true], dtype=float
    )

    # Measurement noise
    sigma_ra = np.deg2rad(0.005)
    sigma_dec = np.deg2rad(0.005)

    # Reference perturbation scales (for x0_ref)
    rng_ref = np.random.default_rng(42)
    ref_sigma_r = 0.05  # km
    ref_sigma_v = 2e-5  # km/s
    ref_sigma_mu = 5e-10
    ref_sigma_c = 5e-6

    # Prior sigmas (on deltas about reference)
    sig_r = 0.20
    sig_v = 5e-4
    sig_mu = 2e-9
    sig_c20 = 2e-5
    sig_c21s21 = 5e-6
    sig_c22s22 = 5e-6

    # MCMC settings
    n_walkers = 128
    n_samples = 20000
    burn_in = 3000
    thin = 10
    spherical_spread = 1e-2

    # --------------------------
    # Load SPICE & spacecraft truth
    # --------------------------
    spice.furnsh(META_KERNEL)

    et0 = spice.utc2et(utc0)
    et1 = spice.utc2et(utc1)
    ets_full = np.linspace(et0, et1, n_obs)

    sc_state_full = np.zeros((n_obs, 6))
    for i, et in enumerate(ets_full):
        st, _ = spice.spkezr(SC_NAME, float(et), FRAME_I, "NONE", CENTER)
        sc_state_full[i, :] = np.array(st, dtype=float)

    # --------------------------
    # Particle detach point on Bennu surface (use mesh point like your old script)
    # --------------------------
    mesh_path = "ObjFiles/BennuRadar.obj"  # <-- EDIT if needed
    bennu_mesh = trimesh.load(mesh_path, force="mesh")
    vertices = bennu_mesh.vertices

    lat_desired = np.deg2rad(45.0)
    lon_desired = np.deg2rad(80.0)
    pos_target = np.array(
        [
            R_bennu * np.cos(lat_desired) * np.cos(lon_desired),
            R_bennu * np.cos(lat_desired) * np.sin(lon_desired),
            R_bennu * np.sin(lat_desired),
        ]
    )
    dists = np.linalg.norm(vertices - pos_target, axis=1)
    closest_idx = np.argmin(dists)
    pos_detach_bf = vertices[closest_idx]  # interpret as body-fixed coords at t0

    # surface normal (mesh)
    normal_bf = bennu_mesh.vertex_normals[closest_idx]
    normal_bf = normal_bf / np.linalg.norm(normal_bf)

    # Convert detach position to inertial at start time using your rotation model:
    R_ib0 = make_bennu_rotation_matrix(
        alpha_true, delta_true, omega_true, t=0.0, w0=0.0
    )
    R_bi0 = R_ib0.T
    r0_true = R_bi0 @ pos_detach_bf

    # Outward initial velocity (random hemisphere w.r.t. r0_true)
    rng = np.random.default_rng(7)
    vmag = 2e-4
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    if np.dot(u, r0_true) < 0:
        u = -u
    v0_true = vmag * u

    # Full truth initial state (12)
    x0_true = np.hstack([r0_true, v0_true, params_true])

    # --------------------------
    # Build STT functions + STTPropagator (YOUR propagator)
    # --------------------------
    f_func, A_func, B_funcs = generate_stt_functions_bennu_deg2(
        order=stt_order,
        R_ref_km=R_ref,
        alpha_rad=alpha_true,
        delta_rad=delta_true,
        omega_rad_s=omega_true,
        w0_rad=0.0,
    )

    propagator = STTPropagator(
        order=stt_order, f_func=f_func, A_func=A_func, B_funcs=B_funcs
    )

    # --------------------------
    # Propagate particle truth (numeric via your propagator)
    # --------------------------
    print("\nPropagating particle truth...")
    sol_true, stts_true = propagator.propagate(
        x0_true, ets_full, rtol=1e-11, atol=1e-13
    )
    x_true_full = sol_true.y[:12, :].T  # (N,12)

    # --------------------------
    # Observability mask (occultation by Bennu sphere proxy)
    # --------------------------
    vis_mask = occultation_mask(sc_state_full[:, 0:3], x_true_full[:, 0:3], R_bennu)

    ets = ets_full[vis_mask]
    sc_state = sc_state_full[vis_mask, :]
    x_true = x_true_full[vis_mask, :]

    print(f"\nVisibility: {np.sum(vis_mask)}/{len(vis_mask)} epochs kept.")

    # --------------------------
    # Generate noisy RA/DEC measurements
    # --------------------------
    rng_meas = np.random.default_rng(123)
    y_obs = generate_opnav_measurements_from_sc(
        x_part=x_true,
        sc_state=sc_state,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        rng=rng_meas,
    )

    # --------------------------
    # Reference initial condition (perturb truth) and propagate reference + STTs
    # --------------------------
    print("\nPropagating reference trajectory (12D) + STTs...")

    ref_dev = np.hstack(
        [
            rng_ref.normal(scale=ref_sigma_r, size=3),
            rng_ref.normal(scale=ref_sigma_v, size=3),
            rng_ref.normal(scale=ref_sigma_mu, size=1),
            rng_ref.normal(scale=ref_sigma_c, size=5),
        ]
    )
    x0_ref = x0_true - ref_dev

    # propagate reference about the visible arc
    sol_ref, stts_ref = propagator.propagate(
        x0=x0_ref, t_eval=ets, rtol=1e-12, atol=1e-14
    )
    x_ref = sol_ref.y[:12, :].T

    # --------------------------
    # Residual function (STT-based): delta(t) produced by propagate_deviation
    # --------------------------
    def residuals_normalized(delta0):
        # Uses your STTPropagator deviation propagation
        # (must support 12D)
        _, x_est = propagator.propagate_deviation(sol_ref, stts_ref, delta0)

        los = x_est[:, :3] - sc_state[:, :3]
        ra_model, dec_model = radec_from_los(los)

        y_model = np.empty_like(y_obs)
        y_model[0::2] = ra_model
        y_model[1::2] = dec_model

        res = np.empty_like(y_obs)
        res[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
        res[1::2] = y_obs[1::2] - y_model[1::2]

        w = np.empty_like(y_obs)
        w[0::2] = sigma_ra
        w[1::2] = sigma_dec
        return res / w

    # --------------------------
    # Priors on 12D delta0
    # --------------------------
    prior_sigma = np.array(
        [
            sig_r,
            sig_r,
            sig_r,
            sig_v,
            sig_v,
            sig_v,
            sig_mu,
            sig_c20,
            sig_c21s21,
            sig_c21s21,
            sig_c22s22,
            sig_c22s22,
        ],
        dtype=float,
    )

    priors = [norm(loc=0.0, scale=s) for s in prior_sigma]

    # --------------------------
    # Stage-2 STM/STT-based MAP (fast)
    # --------------------------
    print("\n[Batch] STT-based MAP...")
    batch_res, batch_cov = compute_STT_batch_solution(
        residuals_func=residuals_normalized,
        x0=np.zeros(12),
        priors=priors,
        max_nfev=20000,
    )
    delta_map = batch_res.x

    chi2 = np.sum(residuals_normalized(delta_map) ** 2)
    dof = len(y_obs) - len(delta_map)
    print(f"\n[Batch] chi2_red = {chi2/dof:.3f}  (chi2={chi2:.2f}, dof={dof})")
    print("[Batch] delta_map:\n", delta_map)

    # --------------------------
    # MCMC (your existing MCMCModel)
    # --------------------------
    print("\n[MCMC] Running...")
    model = MCMCModel(
        residuals_func=residuals_normalized,
        initial_params=np.zeros(12),
        param_priors=priors,
        observed_data=y_obs,
    )
    model.setup_whitening_from_priors()
    model.run(
        n_samples=n_samples,
        n_walkers=n_walkers,
        burn_in=burn_in,
        thin=thin,
        spherical_spread=spherical_spread,
        method_optimize="Powell",
        use_demoves=True,
    )

    theta_hat, P_mcmc = model.get_estimate_and_covariance()

    # Truth delta about reference
    true_delta = x0_true - x0_ref

    # --------------------------
    # Plot: Bennu sphere + SC truth + particle truth + particle MAP
    # --------------------------
    _, x_map = propagator.propagate_deviation(sol_ref, stts_ref, delta_map)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Bennu sphere
    uu = np.linspace(0, 2 * np.pi, 40)
    vv = np.linspace(-np.pi / 2, np.pi / 2, 20)
    xs = R_bennu * np.outer(np.cos(uu), np.cos(vv))
    ys = R_bennu * np.outer(np.sin(uu), np.cos(vv))
    zs = R_bennu * np.outer(np.ones_like(uu), np.sin(vv))
    ax.plot_surface(xs, ys, zs, alpha=0.12, linewidth=0)

    ax.plot(
        sc_state_full[:, 0],
        sc_state_full[:, 1],
        sc_state_full[:, 2],
        label="SC (SPICE truth)",
    )
    ax.plot(
        x_true_full[:, 0], x_true_full[:, 1], x_true_full[:, 2], label="Particle truth"
    )
    ax.plot(
        x_map[:, 0],
        x_map[:, 1],
        x_map[:, 2],
        label="Particle MAP (STT-based, visible arc)",
    )

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("OSIRIS-REx + Particle Trajectories (Bennu-centered J2000)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Plot visibility mask
    # --------------------------
    t_hr = (ets_full - ets_full[0]) / 3600.0
    plt.figure(figsize=(10, 2.6))
    plt.plot(t_hr, vis_mask.astype(int), "o-")
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ["occulted", "visible"])
    plt.xlabel("Time since start [hr]")
    plt.title("Occultation / Visibility Mask (sphere proxy)")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Corner plot using your model samples (if your MCMCModel exposes `samples`)
    # --------------------------
    try:
        labels = [
            r"$\delta x_0$",
            r"$\delta y_0$",
            r"$\delta z_0$",
            r"$\delta v_{x0}$",
            r"$\delta v_{y0}$",
            r"$\delta v_{z0}$",
            r"$\delta \mu$",
            r"$\delta C_{20}$",
            r"$\delta C_{21}$",
            r"$\delta S_{21}$",
            r"$\delta C_{22}$",
            r"$\delta S_{22}$",
        ]
        model.plot_corner_with_batch(
            batch_mean=delta_map,
            batch_cov=batch_cov,
            use_median_as_truth=False,
            true_theta=true_delta,
        )
    except Exception as e:
        print(
            "\n[Corner] Skipped (your MCMCModel may not have plot_corner_with_batch here).",
            e,
        )

    # Clean up SPICE
    spice.kclear()
