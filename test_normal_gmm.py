"""
test_Claude3.py  —  Non-Gaussian SPH posterior scenario: Gaussian priors + Gaussian-mixture likelihood

Same physical setup as test_Claude.py and test_Claude2.py; two changes only:
  1. Gaussian priors on ALL parameters (µ and SH coefficients included), as in test_Claude2.py.
  2. Gaussian-mixture contamination likelihood (ε, k) instead of Student-t or plain Gaussian.

The combination isolates the effect of the likelihood choice: Gaussian priors are what a
Kalman filter implicitly assumes, while the mixture likelihood provides explicit outlier
modelling (each residual is either "clean" N(0,1) or "outlier" N(0,k²) with mixing weight ε).
Comparing with test_Claude.py (Uniform priors + mixture) and test_Claude2.py (Gaussian priors
+ Student-t) reveals how much the prior shape vs. the likelihood shape drives the posterior.

Units: km, km/s, seconds.
"""

import math
import os
import sys
from pathlib import Path
from datetime import datetime


import sympy as sp
import numpy as np
from itertools import product

import spiceypy as spice
import trimesh
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.stats import uniform as scipy_uniform

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
    r1 = sc_pos
    r2 = part_pos
    d = r2 - r1
    dd = np.sum(d * d, axis=1)
    t = -np.sum(r1 * d, axis=1) / dd
    t = np.clip(t, 0.0, 1.0)
    closest = r1 + t[:, None] * d
    return np.linalg.norm(closest, axis=1) > R_body


def make_bennu_rotation_matrix(alpha, delta, omega, t, w0=0.0):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cd, sd = np.cos(delta), np.sin(delta)
    W = w0 + omega * t
    cW, sW = np.cos(W), np.sin(W)

    def Rz(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def Rx(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    return Rz(W) @ Rx(np.pi / 2 - delta) @ Rz(alpha + np.pi / 2)


def occultation_mask_shape(
    sc_pos_i,
    part_pos_i,
    bennu_mesh_bf,
    alpha,
    delta,
    omega,
    w0=0.0,
    t0=0.0,
    tau=None,
    eps=1e-9,
    use_embree=True,
):
    sc_pos_i = np.asarray(sc_pos_i, dtype=float)
    part_pos_i = np.asarray(part_pos_i, dtype=float)
    N = sc_pos_i.shape[0]

    if tau is None:
        tau = np.zeros(N, dtype=float)
    else:
        tau = np.asarray(tau, dtype=float)
        if tau.shape[0] != N:
            raise ValueError("tau must have same length as sc_pos_i/part_pos_i")

    if use_embree:
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector
            intersector = RayMeshIntersector(bennu_mesh_bf)
        except Exception:
            intersector = bennu_mesh_bf.ray
    else:
        intersector = bennu_mesh_bf.ray

    sc_b = np.zeros_like(sc_pos_i)
    pt_b = np.zeros_like(part_pos_i)
    for k in range(N):
        R_ib = make_bennu_rotation_matrix(alpha, delta, omega, t=t0 + tau[k], w0=w0)
        sc_b[k] = R_ib @ sc_pos_i[k]
        pt_b[k] = R_ib @ part_pos_i[k]

    d = pt_b - sc_b
    rng = np.linalg.norm(d, axis=1)
    good = rng > 0.0
    dirs = np.zeros_like(d)
    dirs[good] = d[good] / rng[good, None]

    dist_hit = intersector.intersects_first(ray_origins=sc_b, ray_directions=dirs)

    visible = np.ones(N, dtype=bool)
    no_hit = np.isnan(dist_hit) | (dist_hit < 0.0)
    visible[~no_hit] = dist_hit[~no_hit] >= (rng[~no_hit] - eps)
    visible[~good] = True

    return visible


# ============================================================
# Degree-3 potential in body-fixed + symbolic derivatives up to any STT order
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
        [x, y, z, vx, vy, vz, mu, C20, C21, S21, C22, S22, C30, C31, S31, C32, S32, C33, S33]
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

    P20 = sp.Rational(1, 2) * (3 * sphi**2 - 1)
    P21 = 3 * sphi * cphi
    P22 = 3 * cphi**2
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

    args = (x, y, z, vx, vy, vz, mu, C20, C21, S21, C22, S22, C30, C31, S31, C32, S32, C33, S33, t)
    f_func = sp.lambdify(args, f, "numpy")
    A_func = sp.lambdify(args, B_syms[1], "numpy")
    B_funcs = {
        k: sp.lambdify(args, B_syms[k].tolist(), "numpy") for k in range(2, order + 1)
    }

    return f_func, A_func, B_funcs


# ============================================================
# Measurement generation
# ============================================================


def generate_radio_measurements_from_sc(
    x_part, sc_state, sigma_range, sigma_range_rate, rng, add_outliers=False
):
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

    d_rho_dx = x / rho
    d_rho_dy = y / rho
    d_rho_dz = z / rho
    d_rho_d_r = np.stack([d_rho_dx, d_rho_dy, d_rho_dz], axis=1)

    rho3 = np.maximum(rho**3, 1e-30)

    d_rhodot_dx = vx / rho - rv * x / rho3
    d_rhodot_dy = vy / rho - rv * y / rho3
    d_rhodot_dz = vz / rho - rv * z / rho3

    d_rhodot_dvx = x / rho
    d_rhodot_dvy = y / rho
    d_rhodot_dvz = z / rho

    d_rhodot_d_x = np.stack(
        [d_rhodot_dx, d_rhodot_dy, d_rhodot_dz, d_rhodot_dvx, d_rhodot_dvy, d_rhodot_dvz],
        axis=1,
    )

    return rho, rhodot, d_rho_d_r, d_rhodot_d_x


# ============================================================
# Batch MAP helper
# ============================================================


def compute_STT_batch_solution(residuals_func, x0, priors=None, max_nfev=40000):
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
    try:
        spice.furnsh(kpath)
    except Exception as e:
        raise RuntimeError(f"Failed to load kernel:\n  {kpath}\nError:\n  {e}") from e


def load_kernels(kernel_root: Path):
    orex_traj_kernels_path = kernel_root / "orex" / "orex_trajectories"
    traj_kernel_files = list_files(orex_traj_kernels_path)

    instrument_kernels = [
        str(kernel_root / "orex" / "instrument_kernels" / "orx_navcam_v02.ti"),
        str(kernel_root / "orex" / "instrument_kernels" / "orx_ocams_v07.ti"),
    ]
    frame_kernels = [
        str(kernel_root / "orex" / "frame_kernels" / "orx_v14.tf"),
    ]
    attitude_kernels_path = kernel_root / "orex" / "attitude_kernels"
    attitude_kernels = list_files(attitude_kernels_path)
    clock_kernels = [
        str(kernel_root / "orex" / "clock_kernels" / "orx_sclkscet_00093.tsc"),
    ]
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

    for k in kernels:
        if not os.path.isfile(k):
            raise FileNotFoundError(f"Kernel not found: {k}")
        safe_furnsh(k)

    return kernels


# ============================================================
# 3-D scene plot
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def set_axes_equal_3d(ax):
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
    faces = mesh.faces
    verts = mesh.vertices

    if faces.shape[0] > max_faces:
        idx = np.random.default_rng(0).choice(faces.shape[0], size=max_faces, replace=False)
        faces = faces[idx]

    poly3d = verts[faces]
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
    sl = slice(None, None, max(1, int(downsample)))

    sc_i = np.asarray(sc_state_full)[sl, :3]
    pt_i = np.asarray(x_true_full)[sl, :3]
    tau_i = np.asarray(tau_full)[sl]

    x_map = np.asarray(x_map)
    tau_map = np.asarray(tau_map)
    mesh = bennu_mesh

    sc_b = np.zeros_like(sc_i)
    pt_b = np.zeros_like(pt_i)

    for k, tk in enumerate(tau_i):
        R_ib = np.asarray(make_bennu_rotation_matrix(alpha, delta, omega, float(tk)))
        sc_b[k] = R_ib @ sc_i[k]
        pt_b[k] = R_ib @ pt_i[k]

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    add_trimesh_to_ax(ax, mesh, alpha=0.35, edge_alpha=0.08, max_faces=25000)

    ax.plot(sc_b[:, 0], sc_b[:, 1], sc_b[:, 2], linewidth=1.8, alpha=0.95,
            color="tab:blue", label="OSIRIS-REx (-64)")
    ax.plot(pt_b[:, 0], pt_b[:, 1], pt_b[:, 2], linewidth=1.6, alpha=0.95,
            color="tab:red", label="GravityPopper Truth")

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
            ax.scatter(sc_vis[:, 0], sc_vis[:, 1], sc_vis[:, 2], s=16, marker="o",
                       color="black", alpha=0.9, label="Measurement Used")
        if sc_occ.size:
            ax.scatter(sc_occ[:, 0], sc_occ[:, 1], sc_occ[:, 2], s=18, marker="x",
                       color="black", alpha=0.9, label="Measurement Unavailable")

    ax.set_xlabel(r"$X_\mathscr{B}$ [km]", labelpad=10)
    ax.set_ylabel(r"$Y_\mathscr{B}$ [km]", labelpad=10)
    ax.set_zlabel(r"$Z_\mathscr{B}$ [km]", labelpad=10)
    fig.canvas.draw()
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

    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, borderaxespad=0.2)
    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.94)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.canvas.draw()
    fig.savefig(f"results/bennu_scene_body_fixed_{timestamp}.pdf", format="pdf", pad_inches=0.20)
    print(f"Saved: results/bennu_scene_body_fixed_{timestamp}.pdf")
    plt.show()


# ============================================================
# Gauss-Newton batch solver with STMs (RA/Dec)
# ============================================================


def solve_stage1_gn_with_stm(
    propagator,
    x0_ref,
    tau,
    sc_state,
    y_obs,
    sigma_range,
    sigma_range_rate,
    obs_weights,
    priors=None,
    update_idx=None,
    max_iter=10,
    tol=1e-12,
    rtol=1e-12,
    atol=1e-12,
    method="LSODA",
    verbose=True,
):
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
    w = np.empty_like(y_obs, dtype=float)
    w[0::2] = sigma_range
    w[1::2] = sigma_range_rate

    for it in range(1, max_iter + 1):
        sol_ref, stts_ref = propagator.propagate(
            x0=x0_ref, t_eval=tau, rtol=rtol, atol=atol, method=method
        )
        x_ref = sol_ref.y[:19, :].T

        Phi_list = []
        for k in range(len(tau)):
            Phi_k = np.array(stts_ref[1][k], dtype=float).reshape(19, 19)
            Phi_list.append(Phi_k)
        Phi_list = np.array(Phi_list)

        rel_state = x_ref[:, :6] - sc_state[:, :6]
        range_model, range_rate_model, d_range_dx, d_range_rate_dx = (
            range_rate_and_partials_from_rel_state(rel_state)
        )

        y_model = np.empty_like(y_obs)
        y_model[0::2] = range_model
        y_model[1::2] = range_rate_model

        res = np.empty_like(y_obs)
        res[0::2] = y_obs[0::2] - y_model[0::2]
        res[1::2] = y_obs[1::2] - y_model[1::2]
        r = res / w * obs_weights

        J = np.zeros((2 * len(tau), n_upd), dtype=float)
        for k in range(len(tau)):
            Hy = np.zeros((2, 19), dtype=float)
            Hy[0, 0:3] = d_range_dx[k, :]
            Hy[1, 0:6] = d_range_rate_dx[k, :]
            Hx0 = Hy @ Phi_list[k]
            J[2 * k : 2 * k + 2, :] = Hx0[:, update_idx]

        J[0::2, :] /= sigma_range
        J[1::2, :] /= sigma_range_rate
        J = J * obs_weights[:, None]

        rows = [J]
        rhs = [r]
        finite = np.isfinite(prior_sig)
        if np.any(finite):
            Wp = np.diag(1.0 / prior_sig[finite])
            Jp = np.zeros((Wp.shape[0], n_upd))
            Jp[:, finite] = Wp
            rp = -(delta_total[update_idx][finite] - prior_mean[finite]) / prior_sig[finite]
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
            print(f"\n[GN it {it:02d}] rms(norm res)={rms:.3e}  step_norm={step_norm:.3e}")
        if step_norm < tol:
            break

    try:
        cov_upd = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        cov_upd = np.linalg.pinv(A.T @ A)

    cov_full = np.full((19, 19), np.inf)
    for i, idx_i in enumerate(update_idx):
        for j, idx_j in enumerate(update_idx):
            cov_full[idx_i, idx_j] = cov_upd[i, j]

    return x0_ref, delta_total, cov_full


def solve_stage1_full_nonlinear_lsq(
    propagator,
    x0_ref,
    tau,
    sc_state,
    y_obs,
    obs_weights,
    sigma_ra,
    sigma_dec,
    priors=None,
    update_idx=None,
    rtol=1e-10,
    atol=1e-12,
    method="LSODA",
    max_nfev=2000,
    verbose=2,
):
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

    finite = np.isfinite(prior_sig)
    w = np.empty_like(y_obs, dtype=float)
    w[0::2] = sigma_ra
    w[1::2] = sigma_dec

    def residual_vector(delta_upd):
        x0 = x0_ref.copy()
        x0[update_idx] = x0_ref[update_idx] + delta_upd
        sol = propagator.propagate_state_only(x0=x0, t_eval=tau, rtol=rtol, atol=atol, method=method)
        x = sol.y[:19, :].T
        rel_state = x[:, :6] - sc_state[:, :6]
        range_model, range_rate_model, _, _ = range_rate_and_partials_from_rel_state(rel_state)
        y_model = np.empty_like(y_obs)
        y_model[0::2] = range_model
        y_model[1::2] = range_rate_model
        res = np.empty_like(y_obs)
        res[0::2] = y_obs[0::2] - y_model[0::2]
        res[1::2] = y_obs[1::2] - y_model[1::2]
        r_meas = (res / w) * obs_weights
        if np.any(finite):
            r_pri = (delta_upd[finite] - prior_mean[finite]) / prior_sig[finite]
            return np.hstack([r_meas, r_pri])
        return r_meas

    result = least_squares(
        fun=residual_vector, x0=np.zeros(n_upd), method="trf", jac="2-point",
        max_nfev=max_nfev, ftol=1e-14, xtol=1e-14, gtol=1e-14, verbose=verbose,
    )
    delta_hat_full = np.zeros(19)
    delta_hat_full[update_idx] = result.x
    x0_ref1 = x0_ref + delta_hat_full

    J = result.jac
    try:
        cov_upd = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov_upd = np.linalg.pinv(J.T @ J)

    cov_full = np.full((19, 19), np.inf)
    for i, ii in enumerate(update_idx):
        for j, jj in enumerate(update_idx):
            cov_full[ii, jj] = cov_upd[i, j]

    return x0_ref1, delta_hat_full, cov_full, result


# ============================================================
# RA/Dec measurement model
# ============================================================


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def radec_and_partials_from_los(los):
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
    los = x_part[:, :3] - sc_state[:, :3]
    ra, dec, _, _ = radec_and_partials_from_los(los)
    ra_meas = ra + rng.normal(0.0, sigma_ra, size=ra.shape)
    dec_meas = dec + rng.normal(0.0, sigma_dec, size=dec.shape)
    y = np.empty(2 * len(ra))
    y[0::2] = ra_meas
    y[1::2] = dec_meas
    return y


def solve_stage1_gn_angles(
    propagator,
    x0_ref,
    tau,
    sc_state,
    y_obs,
    sigma_ra,
    sigma_dec,
    obs_weights,
    priors=None,
    update_idx=None,
    max_iter=10,
    tol=1e-12,
    rtol=1e-8,
    atol=1e-10,
    method="LSODA",
    verbose=True,
):
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
    w = np.empty_like(y_obs, dtype=float)
    w[0::2] = sigma_ra
    w[1::2] = sigma_dec

    for it in range(1, max_iter + 1):
        sol_ref, stts_ref = propagator.propagate(
            x0=x0_ref, t_eval=tau, rtol=rtol, atol=atol, method=method
        )
        x_ref = sol_ref.y[:19, :].T

        Phi_list = np.array(
            [np.array(stts_ref[1][k], dtype=float).reshape(19, 19) for k in range(len(tau))]
        )

        los = x_ref[:, :3] - sc_state[:, :3]
        ra_m, dec_m, d_ra_d_r, d_dec_d_r = radec_and_partials_from_los(los)

        y_model = np.empty_like(y_obs)
        y_model[0::2] = ra_m
        y_model[1::2] = dec_m

        res = np.empty_like(y_obs)
        res[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
        res[1::2] = y_obs[1::2] - y_model[1::2]
        r = (res / w) * obs_weights

        J = np.zeros((2 * len(tau), n_upd), dtype=float)
        for k in range(len(tau)):
            Hy = np.zeros((2, 19), dtype=float)
            Hy[0, 0:3] = d_ra_d_r[k, :]
            Hy[1, 0:3] = d_dec_d_r[k, :]
            Hx0 = Hy @ Phi_list[k]
            J[2 * k : 2 * k + 2, :] = Hx0[:, update_idx]

        J[0::2, :] /= sigma_ra
        J[1::2, :] /= sigma_dec
        J = J * obs_weights[:, None]

        rows, rhs = [J], [r]
        finite = np.isfinite(prior_sig)
        if np.any(finite):
            Wp = np.diag(1.0 / prior_sig[finite])
            Jp = np.zeros((Wp.shape[0], n_upd))
            Jp[:, finite] = Wp
            rp = -(delta_total[update_idx][finite] - prior_mean[finite]) / prior_sig[finite]
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
            print(f"[GN it {it:02d}] rms={rms:.3e}  step={step_norm:.3e}")
        if step_norm < tol:
            break

    try:
        cov_upd = np.linalg.inv(np.vstack(rows).T @ np.vstack(rows))
    except np.linalg.LinAlgError:
        cov_upd = np.linalg.pinv(np.vstack(rows).T @ np.vstack(rows))

    cov_full = np.full((19, 19), np.inf)
    for i, idx_i in enumerate(update_idx):
        for j, idx_j in enumerate(update_idx):
            cov_full[idx_i, idx_j] = cov_upd[i, j]

    return x0_ref, delta_total, cov_full


# ============================================================
# Gaussian-mixture likelihood helpers
# ============================================================


class _MixtureLogLikelihood:
    """Picklable callable for the Gaussian-mixture contamination likelihood."""

    def __init__(self, residuals_func, epsilon, k_outlier):
        self.residuals_func = residuals_func
        self.log_1meps = math.log(1.0 - epsilon)
        self.log_eps = math.log(epsilon)
        self.log_k = math.log(k_outlier)
        self.k_outlier = k_outlier
        self.half_log_2pi = 0.5 * math.log(2.0 * math.pi)

    def __call__(self, theta):
        r = self.residuals_func(theta)
        log_clean = self.log_1meps - 0.5 * r**2 - self.half_log_2pi
        log_outlier = self.log_eps - 0.5 * (r / self.k_outlier) ** 2 - self.log_k - self.half_log_2pi
        return float(np.sum(np.logaddexp(log_clean, log_outlier)))


def make_mixture_log_likelihood(residuals_func, epsilon, k_outlier):
    """Return a picklable Gaussian-mixture log-likelihood callable.

    Patch it onto a plain MCMCModel instance:
        model.log_likelihood = make_mixture_log_likelihood(...)
    """
    return _MixtureLogLikelihood(residuals_func, epsilon, k_outlier)


def mixture_outlier_probs(residuals_func, epsilon, k_outlier, theta):
    """Posterior P(outlier | rᵢ, θ) for each normalised residual."""
    r = residuals_func(theta)
    log_clean = math.log(1.0 - epsilon) - 0.5 * r**2
    log_outlier = math.log(epsilon) - 0.5 * (r / k_outlier) ** 2 - math.log(k_outlier)
    return np.exp(log_outlier - np.logaddexp(log_clean, log_outlier))


# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    # --------------------------
    # USER SETTINGS
    # Changes vs. test_Claude.py  : Gaussian priors (Uniform → Gaussian for µ and SH)
    # Changes vs. test_Claude2.py : Gaussian-mixture likelihood (Student-t → mixture)
    # --------------------------
    KERNEL_ROOT = Path("./kernels")
    SC_NAME = "OSIRIS-REX"
    CENTER = "BENNU"
    FRAME_I = "J2000"
    ABCORR = "NONE"

    utc0 = "2019-03-01T00:00:00"
    arc_hours = 1.0
    cadence_min = 5.0

    R_bennu = 0.290  # km
    R_ref = R_bennu

    alpha_true = np.deg2rad(85.65)
    delta_true = np.deg2rad(-60.17)
    spin_period = 4.296057 * 3600.0
    omega_true = 2 * np.pi / spin_period

    stt_order = 1

    mu_true = 4.89044967462e-09
    C20_true = 0.060908668621940644
    C21_true = -2.8120664615284112e-14
    S21_true = 3.874234999952248e-15
    C22_true = 0.001978445533807606
    S22_true = -0.0007064992913094132
    C30_true = -0.004572082563573552
    C31_true = 0.0008801840896940344
    S31_true = -0.0005870017273132463
    C32_true = -0.0003193368868974497
    S32_true = -0.000183688614279846
    C33_true = 0.0001632924069308578
    S33_true = -4.32290988621995e-05

    params_true = np.array(
        [mu_true, C20_true, C21_true, S21_true, C22_true, S22_true,
         C30_true, C31_true, S31_true, C32_true, S32_true, C33_true, S33_true],
        dtype=float,
    )

    pixel_scale_rad = 13.5e-6
    noise_pixels = 1.0
    sigma_angle = noise_pixels * pixel_scale_rad
    sigma_ra = sigma_angle
    sigma_dec = sigma_angle

    outlier_frac = 0.12
    outlier_scale = 25.0

    rng_ref = np.random.default_rng(42)
    ref_pct_r = ref_pct_v = ref_pct_mu = ref_pct_c = 0.0

    # GAUSSIAN PRIORS (all parameters)
    sig_prior_r = np.full(3, 0.250)    # 250 m position uncertainty [km]
    sig_prior_v = np.full(3, 3.0e-4)   # 0.3 mm/s velocity uncertainty [km/s]

    # mu: Gaussian, sigma = 30% of truth value
    mu_prior_sigma = np.abs(mu_true) * 0.3

    # SH coefficients: Gaussian, sigma = scale × |truth|  (3× deg-2, 5× deg-3)
    sh_scale = np.where(np.arange(12) < 5, 3.0, 5.0)
    sh_prior_sigma = sh_scale * np.abs(params_true[1:])
    sh_floor = np.abs(C22_true) * 0.5
    sh_prior_sigma = np.maximum(sh_prior_sigma, sh_floor)

    # MCMC settings
    n_walkers = 128
    n_samples = 80_000
    burn_in = 10_000
    thin = 100
    spherical_spread = 0.02

    # --------------------------
    # Load SPICE & spacecraft truth
    # --------------------------
    _ = load_kernels(KERNEL_ROOT)

    et0 = spice.utc2et(utc0)
    et1 = et0 + arc_hours * 3600.0
    n_obs = round(arc_hours * 60.0 / cadence_min) + 1
    ets_full = np.linspace(et0, et1, n_obs)
    tau_full = ets_full - ets_full[0]
    print(f"[Setup] arc={arc_hours:.1f} hr, cadence={cadence_min:.0f} min, n_obs={n_obs}")

    sc_state_full = np.zeros((n_obs, 6))
    for i, et in enumerate(ets_full):
        st, _ = spice.spkezr(SC_NAME, float(et), FRAME_I, ABCORR, CENTER)
        sc_state_full[i, :] = np.array(st, dtype=float)

    # --------------------------
    # GravityPopper detach point
    # --------------------------
    mesh_path = "ObjFiles/BennuRadar.obj"
    bennu_mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(bennu_mesh.vertices)

    rverts = np.linalg.norm(vertices, axis=1)
    print("[Mesh] vertex radius stats (raw): min/mean/max =",
          rverts.min(), rverts.mean(), rverts.max())
    vertices = np.asarray(bennu_mesh.vertices)

    lat_desired = np.deg2rad(3.0)
    lon_desired = np.deg2rad(45.0)
    pos_target = np.array([
        R_bennu * np.cos(lat_desired) * np.cos(lon_desired),
        R_bennu * np.cos(lat_desired) * np.sin(lon_desired),
        R_bennu * np.sin(lat_desired),
    ])
    dists = np.linalg.norm(vertices - pos_target, axis=1)
    closest_idx = np.argmin(dists)
    pos_detach_bf = vertices[closest_idx]

    normal_bf = bennu_mesh.vertex_normals[closest_idx]
    normal_bf = normal_bf / np.linalg.norm(normal_bf)

    R_ib0 = make_bennu_rotation_matrix(alpha_true, delta_true, omega_true, t=0.0, w0=0.0)
    r0_true = R_ib0.T @ pos_detach_bf

    rng = np.random.default_rng(7)
    vmag = 2e-4

    rhat = r0_true / np.linalg.norm(r0_true)
    u = rng.normal(size=3)
    u -= np.dot(u, rhat) * rhat
    u /= np.linalg.norm(u)

    rad_frac = 0.03
    sign = 1.0
    u = np.sqrt(1 - rad_frac**2) * u + sign * rad_frac * rhat
    v0_true = vmag * u

    x0_true = np.hstack([r0_true, v0_true, params_true])

    # --------------------------
    # STT propagator
    # --------------------------
    f_func, A_func, B_funcs = generate_stt_functions_bennu_deg3(
        order=stt_order, R_ref_km=R_ref, alpha_rad=alpha_true,
        delta_rad=delta_true, omega_rad_s=omega_true, w0_rad=0.0,
    )
    propagator = STTPropagatorND(
        order=stt_order, f_func=f_func, A_func=A_func, B_funcs=B_funcs, n=19
    )

    # --------------------------
    # Propagate truth
    # --------------------------
    print("\nPropagating GravityPopper truth...")
    sol_true, stts_true = propagator.propagate(
        x0_true, tau_full, rtol=1e-8, atol=1e-10, method="LSODA"
    )
    x_true_full = sol_true.y[:19, :].T

    # --------------------------
    # Visibility mask
    # --------------------------
    vis_mask_full = occultation_mask(sc_state_full[:, 0:3], x_true_full[:, 0:3], R_bennu)
    ets = ets_full
    tau = tau_full
    sc_state = sc_state_full
    x_true = x_true_full
    vis_mask = vis_mask_full

    print(f"\nVisibility: {np.sum(vis_mask)}/{len(vis_mask)} epochs visible.")
    print(f"Using all {len(tau)} epochs for propagation (uniform time grid).")

    # --------------------------
    # Generate noisy RA/Dec measurements + outliers
    # --------------------------
    rng_meas = np.random.default_rng(123)
    y_obs_full = generate_opnav_measurements_from_sc(
        x_part=x_true, sc_state=sc_state,
        sigma_ra=sigma_ra, sigma_dec=sigma_dec, rng=rng_meas,
    )

    epoch_is_outlier = rng_meas.random(n_obs) < outlier_frac
    n_outliers = int(np.sum(epoch_is_outlier))
    if n_outliers > 0:
        y_obs_full[0::2][epoch_is_outlier] += (
            outlier_scale * sigma_ra * rng_meas.choice([-1, 1], n_outliers)
        )
        y_obs_full[1::2][epoch_is_outlier] += (
            outlier_scale * sigma_dec * rng_meas.choice([-1, 1], n_outliers)
        )
    print(
        f"[Measurements] Injected {n_outliers}/{n_obs} outlier epochs "
        f"(amplitude ≈ {outlier_scale:.0f}σ, target rate {100*outlier_frac:.0f}%)"
    )

    obs_weights = np.ones_like(y_obs_full)
    obs_weights[0::2][~vis_mask] = 0.0
    obs_weights[1::2][~vis_mask] = 0.0

    y_obs = y_obs_full

    # --------------------------
    # Reference initial condition
    # --------------------------
    print("\nBuilding reference initial state (19D)...")

    sig_ref_r = np.abs(x0_true[0:3] * ref_pct_r)
    sig_ref_v = np.abs(x0_true[3:6] * ref_pct_v)
    sig_ref_mu = np.abs(x0_true[6:7] * ref_pct_mu)
    sig_ref_c = np.abs(x0_true[7:19] * ref_pct_c)

    ref_dev = np.hstack([
        rng_ref.normal(scale=sig_ref_r, size=3),
        rng_ref.normal(scale=sig_ref_v, size=3),
        rng_ref.normal(scale=sig_ref_mu, size=1),
        rng_ref.normal(scale=sig_ref_c, size=12),
    ])
    x0_ref = x0_true - 0 * ref_dev
    print("\n[Reference] deviation from truth:", ref_dev)

    # --------------------------
    # Priors on 19D delta0 — ALL GAUSSIAN
    # --------------------------
    priors_r  = [norm(loc=0.0, scale=s) for s in sig_prior_r]
    priors_v  = [norm(loc=0.0, scale=s) for s in sig_prior_v]
    priors_mu = [norm(loc=0.0, scale=mu_prior_sigma)]
    priors_sh = [norm(loc=0.0, scale=s) for s in sh_prior_sigma]

    priors = priors_r + priors_v + priors_mu + priors_sh

    prior_sigma = np.array([p.std() for p in priors], dtype=float)

    print("\n[Prior] r  sigmas [km]:", sig_prior_r)
    print("[Prior] v  sigmas [km/s]:", sig_prior_v)
    print("[Prior] mu sigma (Gaussian) [km³/s²]:", mu_prior_sigma)
    print("[Prior] SH sigmas (Gaussian):", sh_prior_sigma)

    # --------------------------
    # Stage 1: Gauss-Newton MAP
    # --------------------------
    print("\n[Stage 1] Gauss-Newton batch to convergence...")

    x0_ref1, delta_hat1, cov1 = solve_stage1_gn_angles(
        propagator=propagator,
        x0_ref=x0_ref,
        tau=tau,
        sc_state=sc_state,
        y_obs=y_obs,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        obs_weights=obs_weights,
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
    # Residual function (STT-based)
    # --------------------------
    def residuals_normalized(delta0):
        _, x_est = propagator.propagate_deviation(sol_ref, stts_ref, delta0)

        los = x_est[:, :3] - sc_state[:, :3]
        ra_m, dec_m, _, _ = radec_and_partials_from_los(los)

        y_model = np.empty_like(y_obs)
        y_model[0::2] = ra_m
        y_model[1::2] = dec_m

        res = np.empty_like(y_obs)
        res[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
        res[1::2] = y_obs[1::2] - y_model[1::2]

        w = np.empty_like(y_obs)
        w[0::2] = sigma_ra
        w[1::2] = sigma_dec

        return (res / w) * obs_weights

    # --------------------------
    # Chi2 at ref1
    # --------------------------
    chi2_at_ref = np.sum(residuals_normalized(np.zeros(19)) ** 2)
    n_visible = np.sum(vis_mask)
    dof = 2 * n_visible - 19
    print(
        f"\n[Stage 2] At ref1 (delta=0): chi2_red = {chi2_at_ref/dof:.3f}  "
        f"(chi2={chi2_at_ref:.2f}, dof={dof}, n_vis={n_visible})"
    )

    # Priors for MCMC — all Gaussian, shifted to Stage-1 MAP
    delta_shift = x0_ref1 - x0_ref
    priors_ref1 = [
        norm(loc=-delta_shift[i], scale=priors[i].std()) for i in range(19)
    ]

    # --------------------------
    # MCMC with Gaussian-mixture likelihood + Gaussian priors
    # --------------------------
    print(
        f"\n[MCMC] Running with Gaussian-mixture likelihood "
        f"(ε={outlier_frac:.2f}, k={outlier_scale:.0f}σ) and Gaussian priors..."
    )
    model = MCMCModel(
        residuals_func=residuals_normalized,
        initial_params=np.zeros(19),
        param_priors=priors_ref1,
        observed_data=y_obs,
    )
    model.log_likelihood = make_mixture_log_likelihood(
        residuals_normalized, outlier_frac, outlier_scale
    )
    model.setup_whitening_from_priors()
    model.run(
        n_samples=n_samples,
        n_walkers=n_walkers,
        burn_in=burn_in,
        thin=thin,
        spherical_spread=spherical_spread,
        method_optimize="LSQ",
        use_demoves=True,
        stretch_a=1.2,
    )

    theta_hat, P_mcmc = model.get_estimate_and_covariance()

    chi2_mcmc = np.sum(residuals_normalized(theta_hat) ** 2)
    print(
        f"\n[MCMC] At theta_hat: chi2_red = {chi2_mcmc/dof:.3f}  (chi2={chi2_mcmc:.2f}, dof={dof})"
    )
    print("[MCMC] theta_hat:\n", theta_hat)
    print("\n[MCMC] Covariance diagonal (stdev):")
    print(np.sqrt(np.diag(P_mcmc)))

    # Per-epoch outlier probabilities
    p_out = mixture_outlier_probs(
        residuals_normalized, outlier_frac, outlier_scale, theta_hat
    )
    p_out_ra = p_out[0::2]
    t_hr = tau / 3600.0
    print("\n[Mixture] P(outlier | data) per epoch:")
    for k, (t, po, flag) in enumerate(zip(t_hr, p_out_ra, epoch_is_outlier)):
        marker = "  <-- INJECTED" if flag else ""
        print(f"  t={t:.2f} hr: P_out={po:.3f}{marker}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(
        t_hr,
        p_out_ra,
        width=cadence_min / 60 * 0.7,
        color=["tab:red" if f else "tab:blue" for f in epoch_is_outlier],
        alpha=0.8,
        label="P(outlier | data)",
    )
    ax.axhline(
        outlier_frac, color="k", ls="--", lw=1.2,
        label=f"Prior $\\varepsilon$={outlier_frac:.2f}",
    )
    ax.set_xlabel("Time since epoch [hr]")
    ax.set_ylabel(r"$P(\mathrm{outlier}\,|\,\mathbf{y})$")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(f"results/outlier_probs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    plt.show()

    true_delta = x0_true - x0_ref1
    print("\n[Truth] true_delta about ref1:\n", true_delta)

    # --------------------------
    # Diagnostics
    # --------------------------
    model.plot_convergence()
    model.plot_postfit_residuals_time(t_obs_used=tau, opnav_data=True)
    model.summary()
    model.print_regression_diagnostics()
    model.plot_autocorrelation()
    model.plot_log_likelihood()

    # --------------------------
    # Plot scene
    # --------------------------
    _, x_map = propagator.propagate_deviation(sol_ref, stts_ref, theta_hat)

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
    # Visibility mask plot
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
            r"$\delta x_0$", r"$\delta y_0$", r"$\delta z_0$",
            r"$\delta v_{x0}$", r"$\delta v_{y0}$", r"$\delta v_{z0}$",
            r"$\delta \mu$",
            r"$\delta C_{20}$", r"$\delta C_{21}$", r"$\delta S_{21}$",
            r"$\delta C_{22}$", r"$\delta S_{22}$",
            r"$\delta C_{30}$", r"$\delta C_{31}$", r"$\delta S_{31}$",
            r"$\delta C_{32}$", r"$\delta S_{32}$",
            r"$\delta C_{33}$", r"$\delta S_{33}$",
        ]
        model.plot_corner_with_batch(
            batch_mean=np.zeros(19),
            batch_cov=cov1,
            use_median_as_truth=False,
            true_theta=true_delta,
        )
    except Exception as e:
        print("\n[Corner] Skipped.", e)

    spice.kclear()
