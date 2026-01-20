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

# TODO: missing stage 1
# TODO: check all the values
# TODO: check all the math and workflow

# TODO: corner plots is horrible now

import os
import sys
from pathlib import Path

import sympy as sp
import numpy as np
from itertools import product

import spiceypy as spice
import trimesh
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm

from STTPropagationND import STTPropagatorND
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
    Dynamics are inertial, Bennu-centered. Gravity is defined in body-fixed and rotated.
    f_func, A_func, B_funcs are called as f_func(*X, t).
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

    # ---- IMPORTANT FIX: define body-fixed coordinates as independent symbols ----
    xb_s, yb_s, zb_s = sp.symbols("xb yb zb", real=True)
    r_b_s = sp.Matrix([xb_s, yb_s, zb_s])

    # geometry in body-fixed (independent variables)
    r2 = xb_s**2 + yb_s**2 + zb_s**2
    r = sp.sqrt(r2)

    lam = sp.atan2(yb_s, xb_s)
    phi = sp.atan2(zb_s, sp.sqrt(xb_s**2 + yb_s**2))
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
    U_b = mu / r * (1 + (R2 / r2) * F2)

    # body-fixed acceleration as gradient wrt (xb_s, yb_s, zb_s) SYMBOLS (now SymPy is happy)
    a_b_s = -sp.Matrix([sp.diff(U_b, xb_s), sp.diff(U_b, yb_s), sp.diff(U_b, zb_s)])

    # now substitute actual r_b(t) = R_ib(t) * r_i into that acceleration
    r_b_expr = R_ib * r_i
    subs_rb = {xb_s: r_b_expr[0], yb_s: r_b_expr[1], zb_s: r_b_expr[2]}
    a_b = sp.Matrix([a_b_s[i].subs(subs_rb) for i in range(3)])

    # inertial acceleration
    a_i = R_bi * a_b

    # ---- augmented dynamics (12D) ----
    f = sp.Matrix([vx, vy, vz, a_i[0], a_i[1], a_i[2], 0, 0, 0, 0, 0, 0])

    # ---- A and higher-order tensors ----
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

    # NOTE: for B_k, lambdify on .tolist() returns nested Python lists; that’s fine if your
    # STTPropagator handles it. If not, wrap with np.array(...) inside your propagator.
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
# SPICE kernel loading
# ============================================================


def list_files(d: Path):
    if not d.exists():
        return []
    return sorted([str(p) for p in d.iterdir() if p.is_file()])


def safe_furnsh(kpath: str):
    """Load a kernel, with a helpful error if it fails."""
    try:
        spice.furnsh(kpath)
    except Exception as e:
        raise RuntimeError(f"Failed to load kernel:\n  {kpath}\nError:\n  {e}") from e


def load_kernels(kernel_root: Path):
    # OREx Trajectories
    orex_traj_kernels_path = kernel_root / "orex" / "orex_trajectories"
    traj_kernel_files = list_files(orex_traj_kernels_path)

    # OREx Instrument Kernels
    instrument_kernels = [
        str(kernel_root / "orex" / "instrument_kernels" / "orx_navcam_v02.ti"),
        str(kernel_root / "orex" / "instrument_kernels" / "orx_ocams_v07.ti"),
    ]

    # OREx Frame Kernels
    frame_kernels = [
        str(kernel_root / "orex" / "frame_kernels" / "orx_v14.tf"),
    ]

    # OREx Attitude Kernels
    attitude_kernels_path = kernel_root / "orex" / "attitude_kernels"
    attitude_kernels = list_files(attitude_kernels_path)

    # OREx Clock Kernels
    clock_kernels = [
        str(kernel_root / "orex" / "clock_kernels" / "orx_sclkscet_00093.tsc"),
    ]

    # Other SPICE Kernels
    other_kernels = [
        str(kernel_root / "pck00010.tpc"),
        str(kernel_root / "naif0012.tls"),
        str(kernel_root / "de424.bsp"),
        str(kernel_root / "gm_de440.tpc"),
        str(kernel_root / "bennu_v17.tpc"),
        str(kernel_root / "orex" / "bennu_refdrmc_v1.bsp"),
        str(
            kernel_root
            / "orex"
            / "bennu_shape_models"
            / "bennu_g_12600mm_alt_obj_0000n00000_v021a.bds"
        ),
        str(kernel_root / "orex" / "orx_struct_v04.bsp"),
        str(kernel_root / "trajectories" / "de432s.bsp"),
    ]

    kernels = (
        traj_kernel_files
        + instrument_kernels
        + frame_kernels
        + attitude_kernels
        + clock_kernels
        + other_kernels
    )

    # Load all kernels (and fail fast if something is missing)
    for k in kernels:
        if not os.path.isfile(k):
            raise FileNotFoundError(f"Kernel not found: {k}")
        safe_furnsh(k)

    return kernels


# ============================================================
# BETTER TRAJECTORY PLOT USING THE SHAPE MODEL (trimesh)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def set_axes_equal_3d(ax):
    """Make 3D axes have equal scale so the mesh doesn't look like a pancake."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def add_trimesh_to_ax(
    ax, mesh, color=(0.7, 0.7, 0.7, 1.0), alpha=0.35, edge_alpha=0.15, max_faces=30000
):
    """
    Render a trimesh.Trimesh on a matplotlib 3D axis.
    Downsamples faces if mesh is huge (matplotlib dies otherwise).
    """
    # Downsample faces if necessary
    faces = mesh.faces
    verts = mesh.vertices

    if faces.shape[0] > max_faces:
        idx = np.random.default_rng(0).choice(
            faces.shape[0], size=max_faces, replace=False
        )
        faces = faces[idx]

    poly3d = verts[faces]  # (F,3,3)
    coll = Poly3DCollection(poly3d, linewidths=0.2)
    coll.set_facecolor(color)
    coll.set_alpha(alpha)
    coll.set_edgecolor((0.0, 0.0, 0.0, edge_alpha))
    ax.add_collection3d(coll)


def normalize_mesh_scale(mesh, target_radius_km, mode="rms"):
    """
    Optional: scale the mesh so its "radius" matches target_radius_km.
    mode:
      - "max": use max vertex norm
      - "rms": use RMS vertex norm (usually smoother)
    """
    verts = mesh.vertices
    r = np.linalg.norm(verts, axis=1)
    if mode == "max":
        scale = target_radius_km / np.max(r)
    else:
        scale = target_radius_km / np.sqrt(np.mean(r**2))
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    return mesh, scale


def plot_bennu_scene(
    bennu_mesh,
    sc_state_full,
    x_true_full,
    x_map,
    ets_full=None,
    vis_mask=None,
    title="OSIRIS-REx + Particle around Bennu (mesh)",
    mesh_target_radius_km=None,
    mesh_scale_mode="rms",
    downsample=3,
):
    """
    bennu_mesh: trimesh.Trimesh in Bennu body-fixed coordinates, centered at origin.
               For visualization, we treat it as "static" in Bennu-centered inertial.
               (If you want time-varying attitude of the mesh, we can animate, but this is already a huge improvement.)
    sc_state_full: (N,6) inertial
    x_true_full:   (N,12 or 6) inertial
    x_map:         (N_visible,12 or 6) inertial (MAP on visible arc)
    vis_mask:      (N,) bool; if provided, show visible vs occulted points
    """

    # Optional mesh scaling
    if mesh_target_radius_km is not None:
        bennu_mesh_plot, scale = normalize_mesh_scale(
            bennu_mesh, mesh_target_radius_km, mode=mesh_scale_mode
        )
        print(
            f"[Plot] Mesh scaled by factor {scale:.6g} to match radius ~{mesh_target_radius_km} km ({mesh_scale_mode})."
        )
    else:
        bennu_mesh_plot = bennu_mesh

    # Downsample trajectories for rendering
    sl = slice(None, None, max(1, int(downsample)))
    sc = sc_state_full[sl, :3]
    pt = x_true_full[sl, :3]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Mesh
    add_trimesh_to_ax(ax, bennu_mesh_plot, alpha=0.35, edge_alpha=0.10, max_faces=25000)

    # Spacecraft (full arc)
    ax.plot(sc[:, 0], sc[:, 1], sc[:, 2], linewidth=2.0, label="SC (SPICE truth)")

    # Particle truth
    ax.plot(pt[:, 0], pt[:, 1], pt[:, 2], linewidth=2.0, label="Particle truth")

    # Particle MAP (only visible arc typically)
    ax.plot(
        x_map[:, 0],
        x_map[:, 1],
        x_map[:, 2],
        linewidth=2.5,
        label="Particle MAP (visible arc)",
    )

    # Optional: show visible vs occulted points on spacecraft track (makes geometry readable)
    if vis_mask is not None and ets_full is not None:
        vis_mask_ds = vis_mask[sl]
        sc_vis = sc[vis_mask_ds]
        sc_occ = sc[~vis_mask_ds]
        if sc_vis.shape[0] > 0:
            ax.scatter(
                sc_vis[:, 0],
                sc_vis[:, 1],
                sc_vis[:, 2],
                s=18,
                marker="o",
                label="SC epochs used (visible)",
            )
        if sc_occ.shape[0] > 0:
            ax.scatter(
                sc_occ[:, 0],
                sc_occ[:, 1],
                sc_occ[:, 2],
                s=14,
                marker="x",
                label="SC epochs dropped (occulted)",
            )

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title(title)

    # Better view
    ax.view_init(elev=25, azim=35)

    # Auto limits around trajectories + mesh
    all_xyz = np.vstack([sc_state_full[:, :3], x_true_full[:, :3], x_map[:, :3]])
    pad = 0.2 * np.max(np.linalg.norm(all_xyz, axis=1))
    ax.set_xlim(np.min(all_xyz[:, 0]) - pad, np.max(all_xyz[:, 0]) + pad)
    ax.set_ylim(np.min(all_xyz[:, 1]) - pad, np.max(all_xyz[:, 1]) + pad)
    ax.set_zlim(np.min(all_xyz[:, 2]) - pad, np.max(all_xyz[:, 2]) + pad)
    set_axes_equal_3d(ax)

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN SCRIPT (corrected: uses tau = ET-ET0 for dynamics)
# ============================================================

if __name__ == "__main__":

    # --------------------------
    # USER SETTINGS (edit these)
    # --------------------------
    KERNEL_ROOT = Path("./kernels")
    SC_NAME = "OSIRIS-REX"
    CENTER = "BENNU"
    FRAME_I = "J2000"
    ABCORR = "NONE"

    # Observation window (must be covered by SPK)
    utc0 = "2019-03-01T00:00:00"
    utc1 = "2019-03-01T02:00:00"
    n_obs = 120

    # Bennu physical
    R_bennu = 0.290  # km
    R_ref = R_bennu

    # Bennu spin + pole truth
    alpha_true = np.deg2rad(85.65)  # rad
    delta_true = np.deg2rad(-60.17)  # rad
    spin_period = 4.296057 * 3600.0  # s
    omega_true = 2 * np.pi / spin_period  # rad/s

    # Dynamics / STT order
    stt_order = 1

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
    prior_looseness = 1e1

    # MCMC settings
    n_walkers = 128
    n_samples = 2000
    burn_in = 300
    thin = 10
    spherical_spread = 1e-2

    # --------------------------
    # Load SPICE & spacecraft truth (SPICE remains in ET)
    # --------------------------
    _ = load_kernels(KERNEL_ROOT)

    et0 = spice.utc2et(utc0)
    et1 = spice.utc2et(utc1)
    ets_full = np.linspace(et0, et1, n_obs)  # ET (for SPICE)
    tau_full = ets_full - ets_full[0]  # seconds since start (for dynamics)

    sc_state_full = np.zeros((n_obs, 6))
    for i, et in enumerate(ets_full):
        st, _ = spice.spkezr(SC_NAME, float(et), FRAME_I, ABCORR, CENTER)
        sc_state_full[i, :] = np.array(st, dtype=float)

    # --------------------------
    # Particle detach point on Bennu surface
    # --------------------------
    mesh_path = "ObjFiles/BennuRadar.obj"  # <-- EDIT if needed
    bennu_mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(bennu_mesh.vertices)

    # ---- mesh unit sanity check ----
    rverts = np.linalg.norm(vertices, axis=1)
    print(
        "[Mesh] vertex radius stats (raw): min/mean/max =",
        rverts.min(),
        rverts.mean(),
        rverts.max(),
    )
    # If mesh looks like meters (~300), scale to km:
    if rverts.max() > 10.0:  # crude but effective for Bennu
        print("[Mesh] Detected likely meters-scale OBJ; scaling mesh by 1e-3 to km.")
        bennu_mesh = bennu_mesh.copy()
        bennu_mesh.apply_scale(1e-3)
        vertices = np.asarray(bennu_mesh.vertices)

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
    pos_detach_bf = vertices[closest_idx]  # body-fixed at tau=0

    # surface normal (mesh)
    normal_bf = bennu_mesh.vertex_normals[closest_idx]
    normal_bf = normal_bf / np.linalg.norm(normal_bf)

    # Convert detach position to inertial at start time using rotation at tau=0
    R_ib0 = make_bennu_rotation_matrix(
        alpha_true, delta_true, omega_true, t=0.0, w0=0.0
    )
    r0_true = R_ib0.T @ pos_detach_bf  # inertial

    # Outward initial velocity (random hemisphere w.r.t. r0_true)
    rng = np.random.default_rng(7)
    vmag = 2e-4  # km/s
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    if np.dot(u, r0_true) < 0:
        u = -u
    v0_true = vmag * u

    # Full truth initial state (12)
    x0_true = np.hstack([r0_true, v0_true, params_true])

    # --------------------------
    # Build STT functions + propagator (must accept t argument)
    # --------------------------
    f_func, A_func, B_funcs = generate_stt_functions_bennu_deg2(
        order=stt_order,
        R_ref_km=R_ref,
        alpha_rad=alpha_true,
        delta_rad=delta_true,
        omega_rad_s=omega_true,
        w0_rad=0.0,
    )

    propagator = STTPropagatorND(
        order=stt_order, f_func=f_func, A_func=A_func, B_funcs=B_funcs, n=12
    )

    # --------------------------
    # Propagate particle truth (DYNAMICS on tau_full)
    # --------------------------
    print("\nPropagating particle truth...")
    sol_true, stts_true = propagator.propagate(
        x0_true, tau_full, rtol=1e-8, atol=1e-10, method="LSODA"
    )
    x_true_full = sol_true.y[:12, :].T  # (N,12)

    # --------------------------
    # Observability mask (occultation by Bennu sphere proxy)
    # --------------------------
    vis_mask = occultation_mask(sc_state_full[:, 0:3], x_true_full[:, 0:3], R_bennu)

    ets = ets_full[vis_mask]  # ETs for labeling / plots
    tau = tau_full[vis_mask]  # taus for propagation / residuals
    sc_state = sc_state_full[vis_mask, :]
    x_true = x_true_full[vis_mask, :]

    print(f"\nVisibility: {np.sum(vis_mask)}/{len(vis_mask)} epochs kept.")

    # --------------------------
    # Generate noisy RA/DEC measurements (uses SC state and particle truth at same indices)
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
    # Reference initial condition (perturb truth)
    # --------------------------
    print("\nBuilding reference initial state (12D)...")

    ref_dev = np.hstack(
        [
            rng_ref.normal(scale=ref_sigma_r, size=3),
            rng_ref.normal(scale=ref_sigma_v, size=3),
            rng_ref.normal(scale=ref_sigma_mu, size=1),
            rng_ref.normal(scale=ref_sigma_c, size=5),
        ]
    )
    x0_ref = x0_true - ref_dev

    # --------------------------
    # Priors on 12D delta0 (about whichever reference you're optimizing around)
    # --------------------------
    prior_sigma = prior_looseness * np.array(
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
    # Stage 1: full nonlinear batch (NO STTs) on visible arc (tau grid)
    # --------------------------
    print("\n[Stage 1] Full nonlinear batch (NO STTs) to convergence...")

    def solve_batch_nonlinear_full(x0_ref, delta0_init, priors=None, max_nfev=20000):
        if priors is None:
            prior_mean = np.zeros_like(delta0_init)
            prior_sigma_loc = np.full_like(delta0_init, np.inf)
        else:
            prior_mean = np.array([p.mean() for p in priors], dtype=float)
            prior_sigma_loc = np.array([p.std() for p in priors], dtype=float)

        def fun(delta):
            # propagate full nonlinear, about x0_ref + delta, on tau grid
            sol, _ = propagator.propagate(
                x0=x0_ref + delta,
                t_eval=tau,
                rtol=1e-8,
                atol=1e-10,
                method="LSODA",
            )
            x_est = sol.y[:12, :].T

            los = x_est[:, :3] - sc_state[:, :3]
            ra_model, dec_model = radec_from_los(los)

            y_model = np.empty_like(y_obs)
            y_model[0::2] = ra_model
            y_model[1::2] = dec_model

            r = np.empty_like(y_obs)
            r[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
            r[1::2] = y_obs[1::2] - y_model[1::2]

            w = np.empty_like(y_obs)
            w[0::2] = sigma_ra
            w[1::2] = sigma_dec

            r_meas = r / w
            r_prior = (delta - prior_mean) / prior_sigma_loc
            return np.hstack([r_meas, r_prior])

        result = least_squares(
            fun=fun,
            x0=delta0_init,
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

    batch1, cov1 = solve_batch_nonlinear_full(
        x0_ref=x0_ref,
        delta0_init=np.zeros(12),
        priors=priors,
        max_nfev=20000,
    )
    delta_hat1 = batch1.x
    x0_ref1 = x0_ref + delta_hat1
    print("[Stage 1] delta_hat1 =", delta_hat1)

    # --------------------------
    # Stage 2: relinearize STTs about ref1 (propagate ref1 + STTs on tau grid)
    # --------------------------
    print("\n[Stage 2] Propagating ref1 and computing STTs about ref1...")
    sol_ref, stts_ref = propagator.propagate(
        x0=x0_ref1, t_eval=tau, rtol=1e-8, atol=1e-10, method="LSODA"
    )

    # --------------------------
    # Residual function (STT-based)
    # --------------------------
    def residuals_normalized(delta0):
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
    # Stage 2: STT-based MAP
    # --------------------------
    print("\n[Stage 2] STT-based MAP...")
    batch_res, batch_cov = compute_STT_batch_solution(
        residuals_func=residuals_normalized,
        x0=np.zeros(12),
        priors=priors,
        max_nfev=20000,
    )
    delta_map = batch_res.x

    chi2 = np.sum(residuals_normalized(delta_map) ** 2)
    dof = len(y_obs) - len(delta_map)
    print(f"\n[Stage 2] chi2_red = {chi2/dof:.3f}  (chi2={chi2:.2f}, dof={dof})")
    print("[Stage 2] delta_map:\n", delta_map)

    # --------------------------
    # MCMC
    # --------------------------
    print("\n[MCMC] Running...")
    model = MCMCModel(
        residuals_func=residuals_normalized,
        initial_params=np.zeros(12),  # delta about ref1
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

    # Truth delta about ref1
    true_delta = x0_true - x0_ref1

    # --------------------------
    # Diagnostics
    # --------------------------
    model.plot_convergence()
    model.plot_postfit_residuals_time(
        t_obs_used=tau, opnav_data=True
    )  # tau is the dynamics time
    model.plot_log_likelihood()
    model.summary()
    model.print_regression_diagnostics()
    model.gelman_rubin_diagnostic()
    model.plot_autocorrelation()

    # --------------------------
    # Plot scene
    # --------------------------
    _, x_map = propagator.propagate_deviation(sol_ref, stts_ref, delta_map)

    plot_bennu_scene(
        bennu_mesh=bennu_mesh,
        sc_state_full=sc_state_full,
        x_true_full=x_true_full,
        x_map=x_map,
        ets_full=ets_full,
        vis_mask=vis_mask,
        mesh_target_radius_km=R_bennu,
        mesh_scale_mode="rms",
        downsample=2,
    )

    # --------------------------
    # Plot visibility mask (still labeled in ET-time since start, so tau is fine here too)
    # --------------------------
    t_hr = tau_full / 3600.0
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
    # Corner plot
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
        print("\n[Corner] Skipped.", e)

    spice.kclear()
