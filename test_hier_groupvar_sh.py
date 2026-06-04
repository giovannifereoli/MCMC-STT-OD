"""
Non-Gaussian SPH posterior scenario: HIERARCHICAL PER-GROUP SH-VARIANCE PRIOR
+ fused radio + OpNav

This is the "prior-side" companion to the three likelihood-side scenarios
(test_normal_gmm = Gaussian mixture, test_normal_studen_t = Student-t,
test_uniform_gmm = uniform + mixture).  Here the non-Gaussianity lives in the
PRIOR ON THE SPHERICAL-HARMONIC COEFFICIENTS, not in the likelihood, and the
likelihood is a plain Gaussian over a *fused* measurement set:

  * OpNav angles (RA/Dec line-of-sight), and
  * radio Doppler (range / range-rate),

both generated from the same SC->GravityPopper geometry and stacked into one
normalised residual vector.

HIERARCHICAL PER-GROUP SH-DEVIATION-VARIANCE PRIOR (the "fancy" part)
--------------------------------------------------------------------
The sampler estimates DEVIATIONS about the reference ref1 (the Stage-1 GN
solution) for everything: the state (r, v, mu) AND the 12 SH coefficients.  The
SH coefficient deviations delta_C_lm are split into GROUPS by (degree, zonal vs
non-zonal); zonal = order m=0 (the C_l0 coefficients), non-zonal = m>0 (the
sectoral/tesseral C_lm, S_lm).  Each group g has its own deviation scale sigma_g,
and each coefficient's prior is Gaussian, centred on the a-priori reference x0_ref
(NOT on the data-derived GN solution ref1), with per-group width sigma_g:

    C_lm ~ N(x0_ref_lm, sigma_g^2).

In the chain's "deviation about ref1" coordinates this reads
    delta_C_lm ~ N(-delta_shift_lm, sigma_g^2),   delta_shift = ref1 - x0_ref,
exactly like the state priors (loc=-delta_shift) — so the GN and MCMC priors are
consistent and neither double-counts the data via ref1.

The per-group scale sigma_g is ITSELF a sampled hyperparameter (log-parameterised,
s_g = ln sigma_g, with an independent weakly-informative Gaussian hyperprior),
so marginalising it makes each coefficient deviation's prior heavy-tailed -> the
SH posterior is genuinely non-Gaussian by construction, and we obtain a posterior
on the per-group SH-deviation scale sigma_g.  Each hyperprior centre is set to the
GN PRIOR SH-deviation scale (the RMS of the batch prior sigmas per group) — NOT
to cov1, the GN posterior, which would double-count the measurements (cov1 already
absorbed them, and the MCMC likelihood uses them again).

WHY PER-GROUP SIGMAS (NOT A KAULA K/l^alpha LAW) AND WHY ZONAL / NON-ZONAL
-------------------------------------------------------------------------
An earlier version used Kaula's rule sigma_l = K / l^alpha and sampled the two
hyperparameters (kappa = ln K, alpha).  With only two degrees (l = 2, 3),
(kappa, alpha) is a near-degenerate 2-point line fit, and the weakly-informative
hyperprior makes the implied per-degree (ln sigma_2, ln sigma_3) ~0.996
CORRELATED — a badly-conditioned direction that wrecked the ensemble sampler (3%
acceptance, huge autocorrelation).  We instead sample the per-group log-sigmas
DIRECTLY, with INDEPENDENT hyperpriors, which removes that correlation.  We
further split each degree into ZONAL (order m=0, the axisymmetric C_l0) and
NON-ZONAL (m>0, the sectoral/tesseral terms): zonal terms are systematically
larger than the tesserals (cf. OSIRIS-REx RSX Table 4, which tabulates SEPARATE
zonal vs RMS spectra), so giving them their own scale is physically motivated and
further decouples the hyperparameters.  -> 4 groups.

WHY NON-CENTERED (the funnel)
-----------------------------
Each coefficient DEVIATION is written delta_C_lm = w_lm * sigma_g with
w_lm ~ N(0,1), and the group scale sigma_g enters the LIKELIHOOD, not the SH
prior.  The natural CENTERED form — sampling delta_C_lm directly with prior
N(0, sigma_g^2) while ALSO sampling sigma_g — is a Neal funnel: the width of
delta_C_lm collapses as sigma_g -> 0, so no single emcee step size works
everywhere (small steps stall in the wide mouth, large steps are rejected in the
narrow neck) -> 3% acceptance and a bias toward the shrunk-to-zero neck.
Non-centering gives w a FIXED unit width independent of sigma_g, so the geometry
is benign; it is an exact change of variables (the Jacobian sigma_g cancels the
Gaussian 1/sigma_g), so the target posterior is identical — only the sampler's
coordinates change.

Sampling state vector (23D, NON-CENTERED):
    delta_state(7) = [dr(3), dv(3), dmu]              (deviations about ref1)
    w_SH(12)        = standardized SH deviations, delta_C_lm = w_lm * sigma_g, w ~ N(0,1)
    s_g(4)          = ln sigma_g, one per (degree, zonal/non-zonal) GROUP:
                      {2_zonal, 2_nonzonal, 3_zonal, 3_nonzonal}

State (r, v) and mu keep their original Gaussian priors.  The 12 SH coefficient
deviations are governed by the hierarchical per-group-variance prior through their
group scale sigma_g, which has its own independent, weakly-informative Gaussian
hyperprior on s_g; the four per-group sigma_g posteriors are reported afterwards.

Units: km, km/s, seconds.
"""

import math
import os
from pathlib import Path
from datetime import datetime


import sympy as sp
import numpy as np
from itertools import product

import spiceypy as spice
import trimesh
import matplotlib.pyplot as plt
from scipy.stats import norm

from STTPropagationND import STTPropagatorND
from MCMC import MCMCModel

# Publication-ish defaults
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.7,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}"

# ============================================================
# Geometry / utility
# ============================================================


def occultation_mask(sc_pos, part_pos, R_body):
    """
    Visibility test: does the segment SC->GravityPopper intersect sphere of radius R_body at origin?
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
# Degree-3 potential in body-fixed + symbolic derivatives up to any STT order
# (time dependence enters only through R_ib(t) which we embed symbolically via cos/sin(omega*t + w0))
# ============================================================


def generate_stt_functions_bennu_deg3(
    order, R_ref_km, alpha_rad, delta_rad, omega_rad_s, w0_rad=0.0
):
    t = sp.Symbol("t", real=True)
    x, y, z, vx, vy, vz = sp.symbols("x y z vx vy vz", real=True)
    mu, C20, C21, S21, C22, S22, C30, C31, S31, C32, S32, C33, S33 = sp.symbols(
        "mu C20 C21 S21 C22 S22 C30 C31 S31 C32 S32 C33 S33", real=True
    )

    n_state = 19
    X = sp.Matrix(
        [
            x,
            y,
            z,
            vx,
            vy,
            vz,
            mu,
            C20,
            C21,
            S21,
            C22,
            S22,
            C30,
            C31,
            S31,
            C32,
            S32,
            C33,
            S33,
        ]
    )
    r_i = sp.Matrix([x, y, z])

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

    xb_s, yb_s, zb_s = sp.symbols("xb yb zb", real=True)

    r2 = xb_s**2 + yb_s**2 + zb_s**2
    r = sp.sqrt(r2)
    lam = sp.atan2(yb_s, xb_s)
    phi = sp.atan2(zb_s, sp.sqrt(xb_s**2 + yb_s**2))
    sphi = sp.sin(phi)
    cphi = sp.cos(phi)

    # --- degree-2 Legendre functions (unnormalized) ---
    P20 = sp.Rational(1, 2) * (3 * sphi**2 - 1)
    P21 = 3 * sphi * cphi
    P22 = 3 * cphi**2

    # --- degree-3 Legendre functions (unnormalized) ---
    P30 = sp.Rational(1, 2) * sphi * (5 * sphi**2 - 3)
    P31 = sp.Rational(3, 2) * cphi * (5 * sphi**2 - 1)
    P32 = 15 * sphi * cphi**2
    P33 = 15 * cphi**3

    cos1, sin1 = sp.cos(lam), sp.sin(lam)
    cos2, sin2 = sp.cos(2 * lam), sp.sin(2 * lam)
    cos3, sin3 = sp.cos(3 * lam), sp.sin(3 * lam)

    R2 = sp.Float(R_ref_km**2)
    R3 = sp.Float(R_ref_km**3)

    F2 = C20 * P20 + P21 * (C21 * cos1 + S21 * sin1) + P22 * (C22 * cos2 + S22 * sin2)
    F3 = (
        C30 * P30
        + P31 * (C31 * cos1 + S31 * sin1)
        + P32 * (C32 * cos2 + S32 * sin2)
        + P33 * (C33 * cos3 + S33 * sin3)
    )

    U_b = mu / r * (1 + (R2 / r2) * F2 + (R3 / r**3) * F3)

    a_b_s = sp.Matrix([sp.diff(U_b, xb_s), sp.diff(U_b, yb_s), sp.diff(U_b, zb_s)])

    r_b_expr = R_ib * r_i
    subs_rb = {xb_s: r_b_expr[0], yb_s: r_b_expr[1], zb_s: r_b_expr[2]}
    a_b = sp.Matrix([a_b_s[i].subs(subs_rb) for i in range(3)])
    a_i = R_bi * a_b

    # 19D dynamics — gravity params are constants (zero time derivative)
    f = sp.Matrix(
        [vx, vy, vz, a_i[0], a_i[1], a_i[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    A = f.jacobian(X)
    B_syms = {1: A}

    for k in range(2, order + 1):
        shape = (n_state,) * (k + 1)
        Bk = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(range(n_state), repeat=k + 1):
            i, *js = idx
            deriv = sp.diff(f[i], *[X[j] for j in js])
            Bk[idx] = deriv
        B_syms[k] = Bk

    args = (
        x,
        y,
        z,
        vx,
        vy,
        vz,
        mu,
        C20,
        C21,
        S21,
        C22,
        S22,
        C30,
        C31,
        S31,
        C32,
        S32,
        C33,
        S33,
        t,
    )
    f_func = sp.lambdify(args, f, "numpy")
    A_func = sp.lambdify(args, B_syms[1], "numpy")
    B_funcs = {
        k: sp.lambdify(args, B_syms[k].tolist(), "numpy") for k in range(2, order + 1)
    }

    return f_func, A_func, B_funcs


# ============================================================
# Measurement generation (Range / Range-rate from SPICE SC to GravityPopper)
# ============================================================


def generate_radio_measurements_from_sc(
    x_part, sc_state, sigma_range, sigma_range_rate, rng, add_outliers=False
):
    """
    x_part:   (N,19) or (N,6) GravityPopper inertial state
    sc_state: (N,6) spacecraft inertial state

    returns:
      y = [range_0, range_rate_0, range_1, range_rate_1, ...]
    """
    rel_state = x_part[:, :6] - sc_state[:, :6]
    rho, rhodot, _, _ = range_rate_and_partials_from_rel_state(rel_state)

    rho_meas = rho + rng.normal(0.0, sigma_range, size=rho.shape)
    rhodot_meas = rhodot + rng.normal(0.0, sigma_range_rate, size=rhodot.shape)

    if add_outliers:
        p_out = 0.02
        out_scale_rho = 20.0 * sigma_range
        out_scale_rhodot = 20.0 * sigma_range_rate
        mask = rng.random(size=rho.shape) < p_out
        rho_meas[mask] += rng.normal(0.0, out_scale_rho, size=np.sum(mask))
        rhodot_meas[mask] += rng.normal(0.0, out_scale_rhodot, size=np.sum(mask))

    y = np.empty(2 * len(rho))
    y[0::2] = rho_meas
    y[1::2] = rhodot_meas
    return y


def range_rate_and_partials_from_rel_state(rel_state):
    """
    rel_state: (N,6) = [dx, dy, dz, dvx, dvy, dvz]
               relative state from SC to GravityPopper

    returns:
      rho:           (N,)   range
      rhodot:        (N,)   range rate
      d_rho_d_r:     (N,3)  partial of range wrt relative position
      d_rhodot_d_x:  (N,6)  partial of range rate wrt relative state
                            [d/dx, d/dy, d/dz, d/dvx, d/dvy, d/dvz]
    """
    r = rel_state[:, :3]
    v = rel_state[:, 3:6]

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]

    rho2 = x * x + y * y + z * z
    rho = np.sqrt(np.maximum(rho2, 1e-30))

    rv = x * vx + y * vy + z * vz
    rhodot = rv / rho

    # ---- range partial wrt position ----
    d_rho_dx = x / rho
    d_rho_dy = y / rho
    d_rho_dz = z / rho
    d_rho_d_r = np.stack([d_rho_dx, d_rho_dy, d_rho_dz], axis=1)

    # ---- range-rate partial wrt state ----
    # rhodot = (r·v)/rho
    # d(rhodot)/dr = v/rho - (r·v) r / rho^3
    # d(rhodot)/dv = r/rho

    rho3 = np.maximum(rho**3, 1e-30)

    d_rhodot_dx = vx / rho - rv * x / rho3
    d_rhodot_dy = vy / rho - rv * y / rho3
    d_rhodot_dz = vz / rho - rv * z / rho3

    d_rhodot_dvx = x / rho
    d_rhodot_dvy = y / rho
    d_rhodot_dvz = z / rho

    d_rhodot_d_x = np.stack(
        [
            d_rhodot_dx,
            d_rhodot_dy,
            d_rhodot_dz,
            d_rhodot_dvx,
            d_rhodot_dvy,
            d_rhodot_dvz,
        ],
        axis=1,
    )

    return rho, rhodot, d_rho_d_r, d_rhodot_d_x


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
        str(kernel_root / "particles_pub_03Mar2020.bsp"),
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


def plot_bennu_scene_body_fixed(
    bennu_mesh,
    sc_state_full,
    x_true_full,
    x_map,
    tau_full,
    tau_map,
    alpha,
    delta,
    omega,
    vis_mask=None,
    title="OSIRIS-REx + GravityPopper around Bennu (BODY-FIXED)",
    mesh_target_radius_km=None,
    mesh_scale_mode="rms",
    downsample=3,
):
    """
    BODY-FIXED visualization (publication-ready).

    Notes
    -----
    - Assumes `bennu_mesh` is already in Bennu body-fixed frame and centered at origin.
    - `make_bennu_rotation_matrix(alpha, delta, omega, t)` is assumed to return a DCM
      that maps BODY->INERTIAL (R_bi) or INERTIAL->BODY (R_ib). Many implementations
      return BODY->INERTIAL; in that common case, the correct inertial->body transform
      is the transpose.
    """

    # ------------------------------------------------------------
    # Downsample
    # ------------------------------------------------------------
    sl = slice(None, None, max(1, int(downsample)))

    sc_i = np.asarray(sc_state_full)[sl, :3]
    pt_i = np.asarray(x_true_full)[sl, :3]
    tau_i = np.asarray(tau_full)[sl]

    x_map = np.asarray(x_map)
    tau_map = np.asarray(tau_map)
    mesh = bennu_mesh

    # ------------------------------------------------------------
    # Rotate INERTIAL -> BODY-FIXED
    # ------------------------------------------------------------
    sc_b = np.zeros_like(sc_i)
    pt_b = np.zeros_like(pt_i)

    for k, tk in enumerate(tau_i):
        R_ib = np.asarray(make_bennu_rotation_matrix(alpha, delta, omega, float(tk)))
        sc_b[k] = R_ib @ sc_i[k]
        pt_b[k] = R_ib @ pt_i[k]

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    add_trimesh_to_ax(ax, mesh, alpha=0.35, edge_alpha=0.08, max_faces=25000)

    # Trajectories: force colors explicitly
    ax.plot(
        sc_b[:, 0],
        sc_b[:, 1],
        sc_b[:, 2],
        linewidth=1.8,
        alpha=0.95,
        color="tab:blue",
        label="OSIRIS-REx (-64)",
    )
    ax.plot(
        pt_b[:, 0],
        pt_b[:, 1],
        pt_b[:, 2],
        linewidth=1.6,
        alpha=0.95,
        color="tab:red",
        label="GravityPopper Truth",
    )

    # ------------------------------------------------------------
    # Visible / occulted points (both black)
    # ------------------------------------------------------------
    if vis_mask is not None:
        vis_mask = np.asarray(vis_mask).astype(bool)
        vis_mask_ds = (
            vis_mask[sl]
            if vis_mask.shape[0] >= tau_full[sl].shape[0]
            else vis_mask[: sc_b.shape[0]]
        )

        sc_vis = sc_b[vis_mask_ds]
        sc_occ = sc_b[~vis_mask_ds]

        if sc_vis.size:
            ax.scatter(
                sc_vis[:, 0],
                sc_vis[:, 1],
                sc_vis[:, 2],
                s=16,
                marker="o",
                color="black",
                alpha=0.9,
                label="Measurement Used",
            )
        if sc_occ.size:
            ax.scatter(
                sc_occ[:, 0],
                sc_occ[:, 1],
                sc_occ[:, 2],
                s=18,
                marker="x",
                color="black",
                alpha=0.9,
                label="Measurement Unavailable",
            )

    # ------------------------------------------------------------
    # Axes / view / limits
    # ------------------------------------------------------------
    ax.set_xlabel(r"$X_\mathscr{B}$ [km]", labelpad=10)
    ax.set_ylabel(r"$Y_\mathscr{B}$ [km]", labelpad=10)
    ax.set_zlabel(r"$Z_\mathscr{B}$ [km]", labelpad=10)
    fig.canvas.draw()

    # No title (caption-driven for papers)
    # ax.set_title(title)

    ax.view_init(elev=10, azim=-120)

    mesh_v = np.asarray(mesh.vertices)
    all_xyz = np.vstack([mesh_v, sc_b, pt_b])

    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    span = maxs - mins
    span_max = float(np.max(span)) if np.all(np.isfinite(span)) else 1.0
    pad = 0.08 * span_max

    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(mins[2] - pad, maxs[2] + pad)

    set_axes_equal_3d(ax)

    # Legend: bigger and closer (inside, not outside)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.94)

    # Save figure with timestamp (won't overwrite previous)
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.canvas.draw()
    fig.savefig(
        f"results/bennu_scene_body_fixed_{timestamp}.pdf",
        format="pdf",
        # bbox_inches="tight",
        pad_inches=0.20,
    )
    print(f"Saved: results/bennu_scene_body_fixed_{timestamp}.pdf")

    # plt.show()


# ============================================================
# RA/Dec measurement model (angles-only, from SC camera to GravityPopper)
# ============================================================


def wrap_to_pi(x):
    """Wrap angle (radians) to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def radec_and_partials_from_los(los):
    """
    los: (N,3) vector from SC to GravityPopper in J2000.
    Returns:
      ra, dec: (N,)  RA = arctan2(y,x), Dec = arctan2(z, rho_eq)
      d_ra_d_r, d_dec_d_r: (N,3)  partials wrt particle position
    """
    x = los[:, 0]
    y = los[:, 1]
    z = los[:, 2]
    rxy2 = np.maximum(x * x + y * y, 1e-30)
    rxy = np.sqrt(rxy2)
    rho2 = np.maximum(rxy2 + z * z, 1e-30)

    ra = np.arctan2(y, x)
    dec = np.arctan2(z, rxy)

    d_ra_dx = -y / rxy2
    d_ra_dy = x / rxy2
    d_ra_dz = np.zeros_like(z)

    d_dec_drxy = -z / rho2
    d_dec_dx = d_dec_drxy * (x / rxy)
    d_dec_dy = d_dec_drxy * (y / rxy)
    d_dec_dz = rxy / rho2

    d_ra_d_r = np.stack([d_ra_dx, d_ra_dy, d_ra_dz], axis=1)
    d_dec_d_r = np.stack([d_dec_dx, d_dec_dy, d_dec_dz], axis=1)
    return ra, dec, d_ra_d_r, d_dec_d_r


def generate_opnav_measurements_from_sc(x_part, sc_state, sigma_ra, sigma_dec, rng):
    """
    x_part:   (N,19) or (N,6) GravityPopper inertial state.
    sc_state: (N,6)  spacecraft inertial state.
    Returns y = [ra_0, dec_0, ra_1, dec_1, ...] (2N,).
    """
    los = x_part[:, :3] - sc_state[:, :3]
    ra, dec, _, _ = radec_and_partials_from_los(los)
    ra_meas = ra + rng.normal(0.0, sigma_ra, size=ra.shape)
    dec_meas = dec + rng.normal(0.0, sigma_dec, size=dec.shape)
    y = np.empty(2 * len(ra))
    y[0::2] = ra_meas
    y[1::2] = dec_meas
    return y


def solve_stage1_gn_fused(
    propagator,
    x0_ref,
    tau,
    sc_state,
    y_ang,  # (2N,) stacked [ra0, dec0, ra1, dec1, ...]
    y_radio,  # (2N,) stacked [rho0, rhodot0, rho1, rhodot1, ...]
    sigma_ra,
    sigma_dec,
    sigma_range,
    sigma_range_rate,
    vis_mask,  # (N,) bool per-epoch visibility (applied to all 4 measurements)
    priors=None,
    update_idx=None,
    max_iter=10,
    tol=1e-12,
    rtol=1e-8,
    atol=1e-10,
    method="LSODA",
    verbose=True,
):
    """
    FUSED Gauss-Newton MAP using BOTH radio (range/range-rate) and OpNav (RA/Dec)
    with the STM chain rule.  Per epoch the 4 measurements [RA, Dec, range,
    range-rate] are stacked; angle partials depend on position only, the
    range-rate partial also on velocity.  Linearised MAP per iteration:
    min ||r - J d||^2 + ||(d - m)/s||^2 (Gaussian-prior regularisation), with the
    measurement Jacobian chained to the initial state via Phi(t,0).
    """
    if update_idx is None:
        update_idx = np.arange(19)
    update_idx = np.asarray(update_idx, dtype=int)
    n_upd = len(update_idx)

    if priors is None:
        prior_mean = np.zeros(n_upd)
        prior_sig = np.full(n_upd, np.inf)
    else:
        prior_mean = np.array([p.mean() for p in priors], dtype=float)
        prior_sig = np.array([p.std() for p in priors], dtype=float)

    delta_total = np.zeros(19)
    N = len(tau)
    vis = np.asarray(vis_mask).astype(float)  # (N,) 1.0 visible / 0.0 occulted

    rows = [None]  # populated each iteration; kept for covariance after loop
    for it in range(1, max_iter + 1):
        sol_ref, stts_ref = propagator.propagate(
            x0=x0_ref, t_eval=tau, rtol=rtol, atol=atol, method=method
        )
        x_ref = sol_ref.y[:19, :].T  # (N,19)

        Phi_list = np.array(
            [np.array(stts_ref[1][k], dtype=float).reshape(19, 19) for k in range(N)]
        )

        # measurement models + partials
        los = x_ref[:, :3] - sc_state[:, :3]
        ra_m, dec_m, d_ra_d_r, d_dec_d_r = radec_and_partials_from_los(los)
        rel = x_ref[:, :6] - sc_state[:, :6]
        rho_m, rhodot_m, d_rho_d_r, d_rhodot_d_x = (
            range_rate_and_partials_from_rel_state(rel)
        )

        r = np.zeros(4 * N)
        J = np.zeros((4 * N, n_upd))
        for k in range(N):
            wk = vis[k]

            # normalized residuals (RA wrapped)
            r_ra = wrap_to_pi(y_ang[2 * k] - ra_m[k]) / sigma_ra
            r_dec = (y_ang[2 * k + 1] - dec_m[k]) / sigma_dec
            r_rho = (y_radio[2 * k] - rho_m[k]) / sigma_range
            r_rhod = (y_radio[2 * k + 1] - rhodot_m[k]) / sigma_range_rate
            r[4 * k : 4 * k + 4] = wk * np.array([r_ra, r_dec, r_rho, r_rhod])

            # measurement partials wrt the 19D state at epoch k (sigma-normalized)
            Hy = np.zeros((4, 19))
            Hy[0, 0:3] = d_ra_d_r[k, :] / sigma_ra
            Hy[1, 0:3] = d_dec_d_r[k, :] / sigma_dec
            Hy[2, 0:3] = d_rho_d_r[k, :] / sigma_range
            Hy[3, 0:6] = d_rhodot_d_x[k, :] / sigma_range_rate

            Hx0 = Hy @ Phi_list[k]  # chain to initial state
            J[4 * k : 4 * k + 4, :] = wk * Hx0[:, update_idx]

        rows, rhs = [J], [r]
        finite = np.isfinite(prior_sig)
        if np.any(finite):
            Wp = np.diag(1.0 / prior_sig[finite])
            Jp = np.zeros((Wp.shape[0], n_upd))
            Jp[:, finite] = Wp
            rp = (
                -(delta_total[update_idx][finite] - prior_mean[finite])
                / prior_sig[finite]
            )
            rows.append(Jp)
            rhs.append(rp)

        A = np.vstack(rows)
        b = np.hstack(rhs)
        d_upd, *_ = np.linalg.lstsq(A, b, rcond=None)

        d_full = np.zeros(19)
        d_full[update_idx] = d_upd
        x0_ref = x0_ref + d_full
        delta_total = delta_total + d_full

        step_norm = np.linalg.norm(d_upd)
        rms = np.sqrt(np.mean(r**2))
        if verbose:
            print(f"[GN-fused it {it:02d}] rms={rms:.3e}  step={step_norm:.3e}")
        if step_norm < tol:
            break

    A_full = np.vstack(rows)
    try:
        cov_upd = np.linalg.inv(A_full.T @ A_full)
    except np.linalg.LinAlgError:
        cov_upd = np.linalg.pinv(A_full.T @ A_full)

    cov_full = np.full((19, 19), np.inf)
    for i, idx_i in enumerate(update_idx):
        for j, idx_j in enumerate(update_idx):
            cov_full[idx_i, idx_j] = cov_upd[i, j]

    return x0_ref, delta_total, cov_full


# ============================================================
# Hierarchical per-group SH-variance log-prior
# (this scenario's "fancy" ingredient — lives in the PRIOR, not the likelihood)
# ============================================================


class _HierGroupVarLogPrior:
    """Picklable NON-CENTERED hierarchical per-group SH-deviation-variance log-prior.

    Sampling state: theta = [delta_state(7), w(12), s_g(...)], where the 12
    spherical-harmonic coordinates are STANDARDIZED DEVIATIONS, w_lm ~ N(0,1).  The
    physical SH coefficient DEVIATION about ref1 is reconstructed deterministically
    (in the likelihood / residual transform, NOT here) as

        delta_C_lm = w_lm * sigma_g ,     sigma_g = exp(s_g)

    where g is the coefficient's (degree, zonal/non-zonal) GROUP.  Because the
    group scale sigma_g no longer multiplies the SAMPLED coordinate inside the
    prior, the group scales and the SH coordinates are a-priori INDEPENDENT and
    the prior factorizes completely.  This is an exact reparameterization of the
    centered hierarchical model — same target posterior — but it removes the Neal
    funnel that made emcee collapse to small-sigma / deviations-shrunk-to-zero
    (the source of the observed overconfidence).

    Prior factors:
      * state (r, v) and mu (theta indices 0..6): original Gaussian priors,
        passed in as `state_priors` (frozen scipy distributions, picklable);
      * w (theta indices in `sh_index`): iid standard normal N(0,1);
      * hyperparameters (theta indices in `hyper_index`): INDEPENDENT Gaussian
        hyperpriors with the given loc/scale.  Here they are the per-group
        log-deviation-scale s_g = ln sigma_g (one per (degree, zonal/non-zonal) group).
    """

    def __init__(
        self,
        state_priors,
        sh_index,
        hyper_index,
        hyper_loc,
        hyper_scale,
    ):
        self.state_priors = list(state_priors)  # 7 priors: r(3), v(3), mu(1)
        # theta indices of the 12 STANDARDIZED SH coordinates (z_lm), i.e. 7..18
        self.sh_index = [int(j) for j in sh_index]
        # theta indices + independent Gaussian hyperprior loc/scale (per degree)
        self.hyper_index = [int(j) for j in hyper_index]
        self.hyper_loc = [float(x) for x in hyper_loc]
        self.hyper_scale = [float(x) for x in hyper_scale]
        self._half_ln2pi = 0.5 * math.log(2.0 * math.pi)

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)

        # --- state (r, v) + mu : original Gaussian priors, indices 0..6 ---
        lp = 0.0
        for i in range(7):
            lpi = self.state_priors[i].logpdf(theta[i])
            if not np.isfinite(lpi):
                return -np.inf
            lp += lpi

        # --- hyperparameters: independent Gaussians on the per-degree log-RMS
        # s_l = ln sigma_l (weakly informative, decorrelated across degrees) ---
        for idx, loc, scale in zip(self.hyper_index, self.hyper_loc, self.hyper_scale):
            h = theta[idx]
            lp += -0.5 * ((h - loc) / scale) ** 2 - math.log(scale) - self._half_ln2pi

        # --- NON-CENTERED SH block: the 12 coordinates are STANDARDIZED z_lm with
        # an iid N(0,1) prior.  The per-group scale sigma_g = exp(s_g) is applied in
        # the LIKELIHOOD (residual transform), not here, so the group scales no
        # longer couple into the SH prior -> the funnel disappears.  This is an exact
        # reparameterization (same posterior) but a geometry emcee can traverse. ---
        for j in self.sh_index:
            z = theta[j]
            lp += -self._half_ln2pi - 0.5 * z * z

        if not np.isfinite(lp):
            return -np.inf
        return float(lp)


def plot_gn_postfit_residuals(r_fused, n_obs, tau, title_suffix="Stage-1 Gauss-Newton"):
    """Plot the FUSED Gauss-Newton postfit residuals (normalised) vs time.

    r_fused is the output of residuals_normalized(np.zeros(19)) evaluated at the
    converged GN reference (x0_ref1): a length-(4*n_obs) vector laid out as
    [RA,Dec]*n_obs  followed by  [range,range-rate]*n_obs, each divided by its
    measurement sigma and zeroed at occulted epochs.  Each panel shows the
    normalised residuals for visible epochs with +/-3 sigma reference lines.
    """
    r_fused = np.asarray(r_fused, dtype=float).ravel()
    n_ang = 2 * n_obs
    r_ang, r_rad = r_fused[:n_ang], r_fused[n_ang:]

    t_hr = np.asarray(tau, dtype=float).ravel() / 3600.0

    channels = [
        (r_ang[0::2], r"Whitened RA Residual [$\sigma$]"),
        (r_ang[1::2], r"Whitened Dec Residual [$\sigma$]"),
        (r_rad[0::2], r"Whitened Range Residual [$\sigma$]"),
        (r_rad[1::2], r"Whitened Range-Rate Residual [$\sigma$]"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (res, ylabel) in zip(axes.ravel(), channels):
        res = np.asarray(res, dtype=float)
        mask = res != 0.0  # drop occulted (weighted-to-zero) epochs
        ax.scatter(t_hr[mask], res[mask], s=45, alpha=0.85, color="tab:blue")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.axhline(3.0, color="red", linestyle=":", linewidth=2.5)
        ax.axhline(-3.0, color="red", linestyle=":", linewidth=2.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":")
        if np.any(mask):
            lim = max(np.max(np.abs(res[mask])) * 1.15, 3.4)
            ax.set_ylim(-lim, lim)
            rms = np.sqrt(np.mean(res[mask] ** 2))
            ax.text(
                0.02,
                0.95,
                rf"RMS $= {rms:.2f}\,\sigma$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
            )

    axes[1, 0].set_xlabel("Time since epoch [hours]")
    axes[1, 1].set_xlabel("Time since epoch [hours]")
    fig.suptitle(f"Postfit Residuals --- {title_suffix}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/gn_postfit_residuals_{timestamp}.pdf"
    fig.savefig(fname, format="pdf", bbox_inches="tight")
    print(f"Saved: {fname}")
    # plt.show()


def make_hier_group_var_log_prior(
    state_priors,
    sh_index,
    hyper_index,
    hyper_loc,
    hyper_scale,
):
    """Return a picklable NON-CENTERED hierarchical per-group-variance log-prior.

    `sh_index` are the theta indices (7..18) of the standardized SH coordinates;
    `hyper_index`/`hyper_loc`/`hyper_scale` are the theta indices and independent
    Gaussian hyperprior loc/scale of the per-group log-RMS s_g = ln sigma_g.
    Patch it onto a plain MCMCModel instance:
        model.log_prior = make_hier_group_var_log_prior(...)
    """
    return _HierGroupVarLogPrior(
        state_priors,
        sh_index,
        hyper_index,
        hyper_loc,
        hyper_scale,
    )


# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    # --------------------------
    # USER SETTINGS — same proximity-ops scenario as the other test_* cases.
    # This scenario's distinguishing features:
    #   1) Hierarchical per-group SH-variance PRIOR (per-group ln sigma_g sampled)
    #   2) Plain Gaussian likelihood over a FUSED radio (range/range-rate) + OpNav
    #      (RA/Dec) measurement set
    # --------------------------
    KERNEL_ROOT = Path("./kernels")
    SC_NAME = "OSIRIS-REX"
    CENTER = "BENNU"
    FRAME_I = "J2000"
    ABCORR = "NONE"

    # REALISTIC ARC: 1-hour, 5-min cadence (standard proximity-ops imaging cadence)
    utc0 = "2019-03-01T00:00:00"
    arc_hours = 1.0  # total arc length [hours]
    cadence_min = 1.0  # measurement cadence [minutes]
    # n_obs and ets_full derived after SPICE load

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

    # ============================================================
    # Bennu Gravity Field Truth Values (degree 2×2 + degree 3×3)
    # ============================================================
    # CONVENTION: unnormalized Stokes coefficients.
    # Reference radius R_ref = R_bennu = 0.290 km.

    mu_true = 4.89044967462e-09  # km^3/s^2

    # Degree 2
    C20_true = 0.060908668621940644
    C21_true = -2.8120664615284112e-14
    S21_true = 3.874234999952248e-15
    C22_true = 0.001978445533807606
    S22_true = -0.0007064992913094132

    # Degree 3
    C30_true = -0.004572082563573552
    C31_true = 0.0008801840896940344
    S31_true = -0.0005870017273132463
    C32_true = -0.0003193368868974497
    S32_true = -0.000183688614279846
    C33_true = 0.0001632924069308578
    S33_true = -4.32290988621995e-05

    params_true = np.array(
        [
            mu_true,
            C20_true,
            C21_true,
            S21_true,
            C22_true,
            S22_true,
            C30_true,
            C31_true,
            S31_true,
            C32_true,
            S32_true,
            C33_true,
            S33_true,
        ],
        dtype=float,
    )

    # REALISTIC OPTICAL NOISE: OSIRIS-REx NavCam ~13.5 µrad/pixel.
    pixel_scale_rad = 13.5e-6  # rad/pixel (NavCam angular resolution)
    noise_pixels = 0.25  # noise level in pixels
    sigma_angle = noise_pixels * pixel_scale_rad  # rad
    sigma_ra = sigma_angle
    sigma_dec = sigma_angle

    # REALISTIC RADIO (DSN-class Doppler) NOISE — fused with the optical angles.
    # range:      1 m  (1e-3 km)        — sequential-ranging class
    # range-rate: 0.1 mm/s (1e-7 km/s)  — X-band two-way Doppler over ~10 s count
    sigma_range = 1.0e-3  # km
    sigma_range_rate = 1.0e-7  # km/s

    rng_ref = np.random.default_rng(42)

    # Reference is perturbed off truth by these fractions of |truth| (matching
    # J1_scenarioA2): the Stage-1 GN then has a real error to correct, and the SH
    # deviations (truth - ref1) are sizable enough that the per-group sigma_g
    # hyperparameters become identifiable rather than collapsing to the prior.
    ref_pct_r = 0.2  # 20% of each position component
    ref_pct_v = 0.2  # 20% of each velocity component
    ref_pct_mu = 0.01  # 1% of mu
    ref_pct_c = 0.01  # 1% of each C/S coefficient

    # HIERARCHICAL PER-GROUP SH-DEVIATION-VARIANCE PRIOR
    # ------------------------------------------------------------------------
    # The sampler estimates DEVIATIONS about ref1 for the state (r, v, mu) AND for
    # the 12 SH coefficients.  The SH deviations are split into GROUPS by (degree,
    # zonal vs non-zonal) and given a zero-mean Gaussian prior
    #     delta_C_lm ~ N(0, sigma_g^2)
    # whose per-group scale sigma_g is ITSELF a sampled hyperparameter (log-
    # parameterised, s_g = ln sigma_g).  So the SH prior is centered on the
    # REFERENCE (ref1, the Stage-1 GN solution) and sigma_g is the learned
    # SH-deviation scale.  Zonal = order m=0 (C_l0); non-zonal = m>0 (tesseral).
    #
    # SH absolute layout x0[7:19] = C20,C21,S21,C22,S22 | C30,C31,S31,C32,S32,C33,S33
    # (theta indices 7..18).  Group -> SH theta indices:
    group_index_map = {
        "2_zonal": [7],  # C20
        "2_nonzonal": [8, 9, 10, 11],  # C21,S21,C22,S22
        "3_zonal": [12],  # C30
        "3_nonzonal": [13, 14, 15, 16, 17, 18],  # C31,S31,C32,S32,C33,S33
    }
    group_labels = list(group_index_map.keys())
    n_hyper = len(group_labels)  # 4
    n_phys = 19  # state(7) + SH(12)
    ndim_full = n_phys + n_hyper  # 23
    # theta index of each group's log-sigma hyperparameter (appended after phys 19)
    group_to_hyper = {g: n_phys + k for k, g in enumerate(group_labels)}
    # (The sigma_g hyperprior centres are set below from sh_prior_sigma — the GN
    #  PRIOR scale — to avoid double-counting the data; see that block.)

    # GAUSSIAN PRECONDITIONER PRIORS (Stage-1 GN + whitening seed)
    # ------------------------------
    # NOTE: r, v, mu keep these Gaussian priors in the MCMC.  The 12 SH entries are
    # used ONLY as a Stage-1-GN / whitening preconditioner — the actual SH prior in
    # the MCMC is the hierarchical per-group deviation-variance prior (patched onto
    # log_prior below).  Position / velocity: Gaussian from pre-detachment imaging.
    sig_prior_r = np.full(3, 0.250)  # 250 m  position uncertainty [km]
    sig_prior_v = np.full(3, 3.0e-4)  # 0.3 mm/s velocity uncertainty [km/s]

    # mu: Gaussian centred on zero delta, sigma = 1% of truth GM.  This single
    # value feeds BOTH the Stage-1 batch (via `priors`) AND the MCMC (via
    # priors_ref1[6]), so "batch 1% on GM" and "MCMC keeps 1% on GM" are one knob.
    mu_prior_sigma = np.abs(mu_true) * 0.1

    # SH preconditioner sigmas (Stage-1 GN seed) = 1% of each TRUE coefficient.
    # This is ALSO the a-priori scale that seeds the MCMC sigma_g hyperprior centres
    # below (the prior, not cov1).  The np.maximum floors the coefficient at 1e-6 so
    # the two MACHINE-ZERO terms (C21, S21 ~ 1e-14 by Bennu symmetry) don't get a
    # ~1e-16 sigma -> ~1e15 GN weight, which would blow up AtA's condition number
    # (~1e25) and return garbage in cov1 (the GN covariance shown in the corner
    # overlay).  Every real coefficient is >> 1e-6, so the floor only touches C21/S21.
    sh_prior_sigma = 0.1 * np.maximum(np.abs(params_true[1:13]), 1e-6)

    # MCMC settings
    n_walkers = 128
    n_samples = 10000
    burn_in = 1000
    thin = 100
    spherical_spread = 1e-4

    # SMOKE TEST: set env SPH_SMOKE=1 for a fast end-to-end shakeout run.
    if os.environ.get("SPH_SMOKE", "0") == "1":
        n_walkers = 48
        n_samples = 400
        burn_in = 100
        thin = 1
        print("[Smoke] SPH_SMOKE=1 -> tiny MCMC settings for a shakeout run.")

    # --------------------------
    # Load SPICE & spacecraft truth (SPICE remains in ET)
    # --------------------------
    _ = load_kernels(KERNEL_ROOT)

    et0 = spice.utc2et(utc0)
    et1 = et0 + arc_hours * 3600.0
    n_obs = round(arc_hours * 60.0 / cadence_min) + 1  # 13 for 1-hr / 5-min
    ets_full = np.linspace(et0, et1, n_obs)  # ET (for SPICE)
    tau_full = ets_full - ets_full[0]  # seconds since start (for dynamics)
    print(
        f"[Setup] arc={arc_hours:.1f} hr, cadence={cadence_min:.0f} min, n_obs={n_obs}"
    )

    sc_state_full = np.zeros((n_obs, 6))
    for i, et in enumerate(ets_full):
        st, _ = spice.spkezr(SC_NAME, float(et), FRAME_I, ABCORR, CENTER)
        sc_state_full[i, :] = np.array(st, dtype=float)

    # --------------------------
    # GravityPopper detach point on Bennu surface
    # --------------------------
    mesh_path = "ObjFiles/BennuRadar.obj"
    bennu_mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(bennu_mesh.vertices)

    # mesh unit sanity check
    rverts = np.linalg.norm(vertices, axis=1)
    print(
        "[Mesh] vertex radius stats (raw): min/mean/max =",
        rverts.min(),
        rverts.mean(),
        rverts.max(),
    )
    vertices = np.asarray(bennu_mesh.vertices)

    # NEAR-EQUATORIAL DETACHMENT at body-fixed longitude 45 deg.
    lat_desired = np.deg2rad(3.0)
    lon_desired = np.deg2rad(45.0)
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

    # tangent launch direction (physically plausible: lofting off surface)
    rhat = r0_true / np.linalg.norm(r0_true)

    # pick random vector not parallel to rhat
    u = rng.normal(size=3)
    u -= np.dot(u, rhat) * rhat
    u /= np.linalg.norm(u)

    # SMALL radial fraction: with only 45 min of data the sign of the radial kick
    # is poorly constrained, adding mild ambiguity that couples into the SH estimates.
    rad_frac = 0.03  # 3% radial (vs 10% in original) -> weaker radial signature
    sign = 1.0
    u = np.sqrt(1 - rad_frac**2) * u + sign * rad_frac * rhat

    v0_true = vmag * u

    # Full truth initial state (19)
    x0_true = np.hstack([r0_true, v0_true, params_true])

    # --------------------------
    # Build STT functions + propagator (must accept t argument)
    # --------------------------
    f_func, A_func, B_funcs = generate_stt_functions_bennu_deg3(
        order=stt_order,
        R_ref_km=R_ref,
        alpha_rad=alpha_true,
        delta_rad=delta_true,
        omega_rad_s=omega_true,
        w0_rad=0.0,
    )

    propagator = STTPropagatorND(
        order=stt_order, f_func=f_func, A_func=A_func, B_funcs=B_funcs, n=19
    )

    # --------------------------
    # Propagate GravityPopper truth (DYNAMICS on tau_full)
    # --------------------------
    print("\nPropagating GravityPopper truth...")
    sol_true, stts_true = propagator.propagate(
        x0_true, tau_full, rtol=1e-8, atol=1e-10, method="LSODA"
    )
    x_true_full = sol_true.y[:19, :].T  # (N,19)

    # --------------------------
    # Observability mask (occultation by Bennu sphere proxy)
    # --------------------------
    vis_mask_full = occultation_mask(
        sc_state_full[:, 0:3], x_true_full[:, 0:3], R_bennu
    )

    # Use FULL time grid for propagation (uniform spacing required for time-dependent dynamics)
    ets = ets_full
    tau = tau_full
    sc_state = sc_state_full
    x_true = x_true_full
    vis_mask = vis_mask_full

    print(f"\nVisibility: {np.sum(vis_mask)}/{len(vis_mask)} epochs visible.")
    print(f"Using all {len(tau)} epochs for propagation (uniform time grid).")

    # --------------------------
    # Generate noisy OPTICAL (RA/Dec) measurements
    # --------------------------
    rng_meas = np.random.default_rng(123)
    y_obs_full = generate_opnav_measurements_from_sc(
        x_part=x_true,
        sc_state=sc_state,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        rng=rng_meas,
    )

    # Zero weight for occulted observations
    obs_weights = np.ones_like(y_obs_full)
    obs_weights[0::2][~vis_mask] = 0.0
    obs_weights[1::2][~vis_mask] = 0.0

    y_obs = y_obs_full

    # --------------------------
    # Generate noisy RADIO (range / range-rate) measurements — FUSED with optical
    # --------------------------
    y_radio_full = generate_radio_measurements_from_sc(
        x_part=x_true,
        sc_state=sc_state,
        sigma_range=sigma_range,
        sigma_range_rate=sigma_range_rate,
        rng=rng_meas,
        add_outliers=False,  # plain-Gaussian likelihood here; keep radio clean
    )
    # Radio visibility weights (same occultation geometry as the optical block)
    radio_weights = np.ones_like(y_radio_full)
    radio_weights[0::2][~vis_mask] = 0.0
    radio_weights[1::2][~vis_mask] = 0.0
    y_radio = y_radio_full
    print(
        f"[Measurements] Fused set: {n_obs} OpNav (RA/Dec) + {n_obs} radio "
        f"(range/range-rate) epochs; {int(np.sum(vis_mask))} visible."
    )

    # --------------------------
    # Reference initial condition (perturb truth)
    # --------------------------
    print("\nBuilding reference initial state (19D)...")

    sig_ref_r = np.abs(x0_true[0:3] * ref_pct_r)
    sig_ref_v = np.abs(x0_true[3:6] * ref_pct_v)
    sig_ref_mu = np.abs(x0_true[6:7] * ref_pct_mu)
    sig_ref_c = np.abs(x0_true[7:19] * ref_pct_c)

    ref_dev = np.hstack(
        [
            rng_ref.normal(scale=sig_ref_r, size=3),
            rng_ref.normal(scale=sig_ref_v, size=3),
            rng_ref.normal(scale=sig_ref_mu, size=1),
            rng_ref.normal(scale=sig_ref_c, size=12),
        ]
    )
    x0_ref = x0_true - ref_dev
    print("\n[Reference] deviation from truth:", ref_dev)

    # --------------------------
    # Priors on 19D delta0 — ALL GAUSSIAN
    # --------------------------
    priors_r = [norm(loc=0.0, scale=s) for s in sig_prior_r]
    priors_v = [norm(loc=0.0, scale=s) for s in sig_prior_v]
    priors_mu = [norm(loc=0.0, scale=mu_prior_sigma)]
    priors_sh = [norm(loc=0.0, scale=s) for s in sh_prior_sigma]

    priors = priors_r + priors_v + priors_mu + priors_sh

    # Equivalent sigma vector (for GN solver and diagnostics)
    prior_sigma = np.array([p.std() for p in priors], dtype=float)

    print("\n[Prior] r  sigmas [km]:", sig_prior_r)
    print("[Prior] v  sigmas [km/s]:", sig_prior_v)
    print("[Prior] mu sigma (Gaussian) [km³/s²]:", mu_prior_sigma)
    print("[Prior] SH sigmas (Gaussian):", sh_prior_sigma)

    # --------------------------
    # Stage 1: FUSED radio + OpNav Gauss-Newton batch (STM chain rule)
    # --------------------------
    print("\n[Stage 1] Fused radio+OpNav Gauss-Newton batch (STM) to convergence...")

    x0_ref1, delta_hat1, cov1 = solve_stage1_gn_fused(
        propagator=propagator,
        x0_ref=x0_ref,
        tau=tau,
        sc_state=sc_state,
        y_ang=y_obs,
        y_radio=y_radio,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        sigma_range=sigma_range,
        sigma_range_rate=sigma_range_rate,
        vis_mask=vis_mask,
        priors=priors,
        max_iter=15,
        tol=1e-6,
        rtol=1e-8,
        atol=1e-10,
        verbose=True,
    )

    print("\n[Stage 1] Covariance diagonal (stdev):")
    print(np.sqrt(np.diag(cov1)))
    print("[Stage 1] delta_hat1:\n", delta_hat1)

    # --------------------------
    # Stage 2: relinearize STTs about ref1
    # --------------------------
    print("\n[Stage 2] Propagating ref1 and computing STTs about ref1...")
    sol_ref, stts_ref = propagator.propagate(
        x0=x0_ref1, t_eval=tau, rtol=1e-8, atol=1e-10, method="LSODA"
    )

    # --------------------------
    # Residual function (STT-based) with visibility weighting
    # --------------------------
    def residuals_normalized(theta):
        # 21D-safe: only the first 19 entries (state + SH deltas) drive the
        # measurements; the per-group log-RMS hyperparameters do not.
        delta0 = np.asarray(theta, dtype=float)[:19]
        _, x_est = propagator.propagate_deviation(sol_ref, stts_ref, delta0)

        # ---- OpNav block: RA/Dec line-of-sight angles ----
        los = x_est[:, :3] - sc_state[:, :3]
        ra_m, dec_m, _, _ = radec_and_partials_from_los(los)
        y_ang = np.empty_like(y_obs)
        y_ang[0::2] = ra_m
        y_ang[1::2] = dec_m
        res_ang = np.empty_like(y_obs)
        res_ang[0::2] = wrap_to_pi(y_obs[0::2] - y_ang[0::2])
        res_ang[1::2] = y_obs[1::2] - y_ang[1::2]
        w_ang = np.empty_like(y_obs)
        w_ang[0::2] = sigma_ra
        w_ang[1::2] = sigma_dec
        res_ang = (res_ang / w_ang) * obs_weights

        # ---- Radio block: range / range-rate Doppler ----
        rel = x_est[:, :6] - sc_state[:, :6]
        rho_m, rhodot_m, _, _ = range_rate_and_partials_from_rel_state(rel)
        y_rad = np.empty_like(y_radio)
        y_rad[0::2] = rho_m
        y_rad[1::2] = rhodot_m
        res_rad = y_radio - y_rad
        w_rad = np.empty_like(y_radio)
        w_rad[0::2] = sigma_range
        w_rad[1::2] = sigma_range_rate
        res_rad = (res_rad / w_rad) * radio_weights

        # ---- fused, normalised residual vector (OpNav first, then radio) ----
        return np.concatenate([res_ang, res_rad])

    # --------------------------
    # Chi2 at ref1
    # --------------------------
    chi2_at_ref = np.sum(residuals_normalized(np.zeros(19)) ** 2)
    n_visible = np.sum(vis_mask)
    # 4 measurements per visible epoch (RA, Dec, range, range-rate); 19 physical
    # parameters fit to the data (the 4 per-group hyperparameters are prior-regularised).
    dof = 4 * n_visible - 19
    print(
        f"\n[Stage 2] At ref1 (delta=0): chi2_red = {chi2_at_ref/dof:.3f}  "
        f"(chi2={chi2_at_ref:.2f}, dof={dof}, n_vis={n_visible})"
    )

    # --------------------------
    # GN-step postfit residuals (delta=0 == converged Stage-1 GN solution)
    # --------------------------
    r_gn = residuals_normalized(np.zeros(19))
    n_ang_gn = y_obs.size
    print(
        f"[GN postfit] normalised RMS  OpNav={np.sqrt(np.mean(r_gn[:n_ang_gn]**2)):.3f}  "
        f"radio={np.sqrt(np.mean(r_gn[n_ang_gn:]**2)):.3f}  (target ~1.0)"
    )
    plot_gn_postfit_residuals(r_gn, n_obs, tau, title_suffix="Stage-1 Gauss-Newton")

    # Fast path: GN_POSTFIT_ONLY=1 stops here, before the (expensive) MCMC.
    if os.environ.get("GN_POSTFIT_ONLY", "0") == "1":
        print("[GN postfit] GN_POSTFIT_ONLY=1 -> done (skipping MCMC).")
        spice.kclear()
        raise SystemExit(0)

    # Preconditioner priors for whitening / LSQ init.  State (r,v,mu) keep their
    # Gaussian priors centred on the Stage-1 MAP; the SH preconditioner priors
    # just give whitening a sensible scale — the *actual* SH prior is the
    # hierarchical per-group-variance prior patched onto model.log_prior below.
    delta_shift = x0_ref1 - x0_ref
    priors_ref1 = [norm(loc=-delta_shift[i], scale=priors[i].std()) for i in range(19)]

    # --------------------------------------------------------------------------
    # PER-GROUP sigma_g HYPERPRIOR + 23D NON-CENTERED sampling priors + init.
    # We sample the SH coefficients in standardized form w = (C - x0_ref)/sigma_g
    # (w ~ N(0,1)); delta_C = -delta_shift_SH + w * sigma_g is reconstructed below
    # (centred on the a-priori x0_ref, like the state priors).  The per-group
    # sigma_g hyperprior is centred at the GN *PRIOR* SH-deviation scale: the RMS
    # over the group of the batch prior sigmas (sh_prior_sigma).
    #
    # IMPORTANT — no double counting: we deliberately do NOT use cov1 here.  cov1 is
    # the Stage-1 GN *posterior* covariance (it already absorbed the measurements),
    # and the MCMC likelihood uses those SAME measurements again — seeding the
    # sigma_g prior from cov1 would feed the data in twice and overstate confidence.
    # Taking the centre from the a-priori scale keeps the SH hyperprior consistent
    # with the (relinearised) state priors, which likewise use their original
    # a-priori sigmas, so each datum informs the posterior exactly once.
    # --------------------------------------------------------------------------
    logsig_loc = {}
    for g, idxs in group_index_map.items():
        _var_g = float(
            np.mean([sh_prior_sigma[j - 7] ** 2 for j in idxs])
        )  # a-priori var
        logsig_loc[g] = float(0.5 * math.log(_var_g))  # ln(RMS of a-priori sigmas)
    # Hyperprior width on s_g = ln sigma_g (NOT on sigma_g directly).  Because it is
    # a std in NATURAL-LOG units, it is a MULTIPLICATIVE bracket on sigma_g itself:
    #   +/-1 sigma -> factor exp(+/-0.75) ~= x2.1   (centre/2.1 ... centre*2.1)
    #   +/-2 sigma -> factor exp(+/-1.50) ~= x4.5   (centre/4.5 ... centre*4.5)
    # So 0.75 says the SH-deviation scale is a-priori known only to ~a factor of 2
    # (1 sigma) / ~4-5 (2 sigma): it does NOT pin sigma_g to a number, it brackets
    # its order of magnitude and lets the DATA pick within that bracket (weakly
    # informative).  We can afford this broad a scale prior only because the model
    # is NON-CENTERED — in a centered form a broad sigma_g prior deepens the funnel.
    hyper_logsig_scale = 0.75
    logsig_scale = {g: hyper_logsig_scale for g in group_labels}
    print(
        "[Prior] per-group SH-deviation log-sigma hyperpriors (centre = GN a-priori scale):\n   "
        + "\n   ".join(
            f"ln sigma[{g}] ~ N({logsig_loc[g]:.2f}, {logsig_scale[g]:.2f}^2)  "
            f"(sigma0={math.exp(logsig_loc[g]):.3e})"
            for g in group_labels
        )
    )

    # 23D NON-CENTERED sampling priors: state(7) physical preconditioner priors,
    # w_SH(12) ~ N(0,1) (standardized SH deviations, whitening identity), and the
    # 4 per-group log-sigma INDEPENDENT Gaussian hyperpriors.
    hyper_priors = [
        norm(loc=logsig_loc[g], scale=logsig_scale[g]) for g in group_labels
    ]
    w_priors = [norm(loc=0.0, scale=1.0) for _ in range(12)]
    param_priors_full = list(priors_ref1[0:7]) + w_priors + hyper_priors

    # Init: start the chain at ref1 (all PHYSICAL deviations about ref1 = 0), like
    # the state block (delta_state = 0).  Since the SH prior is now centred on the
    # a-priori (delta_C = -delta_shift_SH + w*sigma_g), delta_C = 0 means
    # w = delta_shift_SH / sigma_g0.
    w_init = np.zeros(12)
    for g, idxs in group_index_map.items():
        _sig0 = math.exp(logsig_loc[g])
        for _j in idxs:
            w_init[_j - 7] = delta_shift[_j] / _sig0
    initial_params_full = np.hstack(
        [np.zeros(7), w_init, [logsig_loc[g] for g in group_labels]]
    )

    # --------------------------
    # Non-centered <-> physical transform.
    # Sampling theta = [delta_state(7), w_SH(12), s_g(4)]; the propagator /
    # residuals_normalized expect the 19D PHYSICAL deviation about ref1.  The SH
    # DEVIATION is reconstructed as  delta_C_lm = -delta_shift_SH + w_lm * sigma_g,
    # sigma_g = exp(s_g) from that coefficient's group.  The -delta_shift_SH offset
    # (delta_shift = ref1 - x0_ref) centres the SH PRIOR on the a-priori x0_ref
    # (prior mean of delta about ref1 = x0_ref - ref1), EXACTLY like the state priors
    # (loc=-delta_shift).  So GN and MCMC use the same a-priori, and we do not anchor
    # the SH prior on the data-derived ref1 (no double-counting).
    # --------------------------
    def physical_delta_from_sampling(theta_s):
        theta_s = np.asarray(theta_s, dtype=float)
        delta19 = theta_s[:19].copy()
        for g, idxs in group_index_map.items():
            _sigma_g = math.exp(theta_s[group_to_hyper[g]])  # sigma_g = exp(s_g)
            for _j in idxs:
                delta19[_j] = -delta_shift[_j] + theta_s[_j] * _sigma_g
        return delta19

    def residuals_sampling(theta_s):
        # Likelihood residuals evaluated in the non-centered sampling space.
        return residuals_normalized(physical_delta_from_sampling(theta_s))

    # --------------------------
    # MCMC with HIERARCHICAL PER-GROUP SH-VARIANCE PRIOR + plain Gaussian likelihood
    # The fused radio+OpNav residuals enter a vanilla Gaussian likelihood;
    # the non-Gaussianity comes entirely from the patched hierarchical prior.
    # --------------------------
    print(
        "\n[MCMC] Running with hierarchical per-group SH-variance PRIOR (per-group "
        "ln sigma_g sampled) and a plain Gaussian likelihood over the fused residuals..."
    )
    model = MCMCModel(
        residuals_func=residuals_sampling,
        initial_params=initial_params_full,
        param_priors=param_priors_full,
        observed_data=y_obs,
    )
    # Patch the log_prior in-place (picklable __call__ class — same pattern the
    # other scenarios use to patch the likelihood).  Likelihood stays the default
    # plain-Gaussian -0.5*sum(r^2) over the fused residual vector.
    model.log_prior = make_hier_group_var_log_prior(
        state_priors=priors_ref1[0:7],
        sh_index=list(range(7, 19)),
        hyper_index=[group_to_hyper[g] for g in group_labels],
        hyper_loc=[logsig_loc[g] for g in group_labels],
        hyper_scale=[logsig_scale[g] for g in group_labels],
    )
    model.setup_whitening_from_priors()
    model.run(
        n_samples=n_samples,
        n_walkers=n_walkers,
        burn_in=burn_in,
        thin=thin,
        spherical_spread=spherical_spread,
        method_optimize="LSQ",
    )

    # --------------------------
    # Back-transform the chain to PHYSICAL coordinates for ALL reporting.
    # Each sample's standardized SH deviation w -> physical deviation about ref1
    # via that sample's own per-group scale: delta_SH = -delta_shift_SH + w*exp(s_g)
    # (same a-priori-centred map as the transform).  The 4 ln-sigma columns (19..22)
    # are left untouched.  After this, model.samples, theta_hat, the covariance, the
    # corner plot, and residuals_normalized all share the 23D layout [delta(19), s_g(4)].
    # --------------------------
    for g, idxs in group_index_map.items():
        _sigma_g = np.exp(model.samples[:, group_to_hyper[g]])  # (n_samples,)
        for _j in idxs:
            model.samples[:, _j] = -delta_shift[_j] + model.samples[:, _j] * _sigma_g
    # Post-run diagnostics evaluate residuals on PHYSICAL samples now.
    model.residuals_func = residuals_normalized

    theta_hat, P_mcmc = model.get_estimate_and_covariance()

    chi2_mcmc = np.sum(residuals_normalized(theta_hat) ** 2)
    print(
        f"\n[MCMC] At theta_hat: chi2_red = {chi2_mcmc/dof:.3f}  (chi2={chi2_mcmc:.2f}, dof={dof})"
    )
    print("[MCMC] theta_hat:\n", theta_hat)
    print("\n[MCMC] Covariance diagonal (stdev):")
    print(np.sqrt(np.diag(P_mcmc)))

    # --------------------------
    # Per-group SH-deviation scale posterior (sigma_g) vs the realized truth scale
    # --------------------------
    samples = model.samples
    sig_g_s = {g: np.exp(samples[:, group_to_hyper[g]]) for g in group_labels}

    # "True" per-group SH-deviation scale: sigma_g is the prior scale of
    # (C - x0_ref), so its realized value is the RMS over the group of the actual
    # a-priori offset (truth - x0_ref) = ref_dev (the reference perturbation).
    true_delta = x0_true - x0_ref1  # truth as a deviation about ref1 (for the corner)
    sig_g_true = {
        g: float(np.sqrt(np.mean(np.array([ref_dev[j] for j in idxs]) ** 2)))
        for g, idxs in group_index_map.items()
    }

    def _pct(a):
        return np.percentile(a, [16, 50, 84])

    print(
        "\n[Prior] Per-group SH-deviation scale posterior "
        "(sigma_g = RMS|delta_C| within the group):"
    )
    for g in group_labels:
        lo, med, hi = _pct(sig_g_s[g])
        print(
            f"   sigma[{g}]: {med:.4e}  (16/84: {lo:.4e} / {hi:.4e})   "
            f"truth(dev)≈{sig_g_true[g]:.4e}"
        )

    # 23D truth overlay: physical delta(19) + per-group log-RMS of the true devs(4).
    true_theta = np.hstack(
        [true_delta, [math.log(max(sig_g_true[g], 1e-300)) for g in group_labels]]
    )
    print("\n[Truth] true_delta about ref1:\n", true_delta)

    # --------------------------
    # Diagnostics  (guarded: some built-in plots assume a single data type)
    # --------------------------
    model.plot_convergence()
    try:
        model.plot_postfit_residuals_time(t_obs_used=tau, opnav_data=True)
    except Exception as e:
        print("[Diag] postfit-time plot skipped (fused residual layout):", e)

    # Fused postfit RMS, split by data type, at the MCMC estimate.
    r_fused = residuals_normalized(theta_hat)
    n_ang = y_obs.size
    r_ang, r_rad = r_fused[:n_ang], r_fused[n_ang:]
    print(
        f"\n[Postfit] normalised RMS  OpNav={np.sqrt(np.mean(r_ang**2)):.3f}  "
        f"radio={np.sqrt(np.mean(r_rad**2)):.3f}  (target ~1.0)"
    )

    model.summary()
    model.print_regression_diagnostics()
    model.plot_autocorrelation()
    model.plot_log_likelihood()

    # --------------------------
    # Plot scene (using MCMC mean estimate)
    # --------------------------
    _, x_map = propagator.propagate_deviation(sol_ref, stts_ref, theta_hat[:19])

    plot_bennu_scene_body_fixed(
        bennu_mesh=bennu_mesh,
        sc_state_full=sc_state_full,
        x_true_full=x_true_full,
        x_map=x_map,
        tau_full=tau_full,
        tau_map=tau,
        alpha=alpha_true,
        delta=delta_true,
        omega=omega_true,
        vis_mask=vis_mask,
        mesh_target_radius_km=R_bennu,
        mesh_scale_mode="rms",
        downsample=2,
    )

    # --------------------------
    # Plot visibility mask
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
    # plt.show()

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
            r"$\delta C_{30}$",
            r"$\delta C_{31}$",
            r"$\delta S_{31}$",
            r"$\delta C_{32}$",
            r"$\delta S_{32}$",
            r"$\delta C_{33}$",
            r"$\delta S_{33}$",
            r"$\ln \sigma_{2,\mathrm{zon}}$",
            r"$\ln \sigma_{2,\mathrm{nz}}$",
            r"$\ln \sigma_{3,\mathrm{zon}}$",
            r"$\ln \sigma_{3,\mathrm{nz}}$",
        ]
        # Pad the 19D Stage-1 batch (cov1) to the full 23D.  The 19 PHYSICAL params
        # (pos, vel, mu, 12 SH) get the real linear-batch overlay (mean=0, cov1).
        # The 4 hyperparameters have NO batch analog — GN does not estimate sigma_g —
        # so for them the red ellipse is the PRIOR N(logsig_loc, logsig_scale^2), and
        # the purple point is ln(RMS of the realized truth deviations) for reference.
        batch_mean_full = np.hstack(
            [np.zeros(19), [logsig_loc[g] for g in group_labels]]
        )
        batch_cov_full = np.zeros((ndim_full, ndim_full))
        batch_cov_full[:19, :19] = cov1
        for g in group_labels:
            j = group_to_hyper[g]
            batch_cov_full[j, j] = logsig_scale[g] ** 2

        # (1) PHYSICAL 19D corner: pos, vel, mu, 12 SH — MCMC (blue) vs Stage-1 batch
        # GN (red mean+cov1 ellipses) vs truth (purple), all as deviations about ref1.
        # This is the apples-to-apples comparison (both estimate the same 19 deltas).
        model.plot_corner_with_batch(
            batch_mean=batch_mean_full,
            batch_cov=batch_cov_full,
            use_median_as_truth=False,
            true_theta=true_theta,
            idx=list(range(19)),
        )

        # (2) HYPERPARAMETER corner: the 4 ln sigma_g, shown SEPARATELY so the
        # prior-vs-realized scale mismatch does not distort the physical corner's
        # axes.  Red ellipse = sigma_g PRIOR; purple = ln(realized deviation RMS).
        model.plot_corner_with_batch(
            batch_mean=batch_mean_full,
            batch_cov=batch_cov_full,
            use_median_as_truth=False,
            true_theta=true_theta,
            idx=list(range(19, ndim_full)),
        )
    except Exception as e:
        print("\n[Corner] Skipped.", e)

    spice.kclear()
