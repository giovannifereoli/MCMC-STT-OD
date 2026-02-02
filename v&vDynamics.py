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

# TODO: add SRP and attitude?
# TODO: read chelseay and make realistic
# TODO: check all the math
# TODO: improve all plots in general

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
    Dynamics are inertial, Bennu-centered. Gravity defined in body-fixed then rotated to inertial.

    f_func, A_func, B_funcs are called as f_func(*X, t).
    """

    import sympy as sp
    import numpy as np
    from itertools import product

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

    # define body-fixed coordinates as independent symbols
    xb_s, yb_s, zb_s = sp.symbols("xb yb zb", real=True)

    # geometry in body-fixed (NO atan2)
    r2 = xb_s**2 + yb_s**2 + zb_s**2
    r = sp.sqrt(r2)
    rho2 = xb_s**2 + yb_s**2
    rho = sp.sqrt(rho2)

    # sin(phi)=z/r, cos(phi)=rho/r
    sphi = zb_s / r
    cphi = rho / r

    # unnormalized P2m(sin(phi)) (your same convention)
    P20 = sp.Rational(1, 2) * (3 * sphi**2 - 1)
    P21 = 3 * sphi * cphi
    P22 = 3 * cphi**2

    # cosλ, sinλ, cos2λ, sin2λ without λ = atan2(y,x)
    # NOTE: still has rho in denom; that's fine and avoids piecewise atan2 derivatives.
    cos1 = xb_s / rho
    sin1 = yb_s / rho
    cos2 = (xb_s**2 - yb_s**2) / rho2
    sin2 = (2 * xb_s * yb_s) / rho2

    F2 = C20 * P20 + P21 * (C21 * cos1 + S21 * sin1) + P22 * (C22 * cos2 + S22 * sin2)

    R2 = sp.Float(R_ref_km**2)
    U_b = mu / r * (1 + (R2 / r2) * F2)

    # Correct sign: a = ∇U
    a_b_s = sp.Matrix([sp.diff(U_b, xb_s), sp.diff(U_b, yb_s), sp.diff(U_b, zb_s)])

    # substitute r_b(t) = R_ib(t) * r_i into the body-fixed acceleration
    r_b_expr = R_ib * r_i
    subs_rb = {xb_s: r_b_expr[0], yb_s: r_b_expr[1], zb_s: r_b_expr[2]}
    a_b = sp.Matrix([a_b_s[i].subs(subs_rb) for i in range(3)])

    # inertial acceleration
    a_i = R_bi * a_b

    # augmented dynamics (12D)
    f = sp.Matrix([vx, vy, vz, a_i[0], a_i[1], a_i[2], 0, 0, 0, 0, 0, 0])

    # Jacobian and higher-order tensors
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

    # lambdify
    args = (x, y, z, vx, vy, vz, mu, C20, C21, S21, C22, S22, t)
    f_func = sp.lambdify(args, f, "numpy")
    A_func = sp.lambdify(args, B_syms[1], "numpy")
    B_funcs = {
        k: sp.lambdify(args, B_syms[k].tolist(), "numpy") for k in range(2, order + 1)
    }

    return f_func, A_func, B_funcs


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
# MAIN SCRIPT  (propagate OREX state + validate vs SPICE)
# ============================================================
if __name__ == "__main__":

    # --------------------------
    # USER SETTINGS (edit these)
    # --------------------------
    KERNEL_ROOT = Path("./kernels")
    SC_NAME = "-64000401"
    CENTER = "BENNU"
    FRAME_I = "J2000"
    ABCORR = "NONE"

    utc0 = "2019-01-29T12:00:00"
    utc1 = "2019-02-02T12:00:00"
    n_obs = 200  # denser helps validation plots

    # Bennu physical
    R_bennu = 0.290  # km
    R_ref = R_bennu

    # Bennu spin + pole truth (only needed because your deg2 model rotates gravity)
    alpha_true = np.deg2rad(85.65)  # rad
    delta_true = np.deg2rad(-60.17)  # rad
    spin_period = 4.296057 * 3600.0  # s
    omega_true = 2.0 * np.pi / spin_period  # rad/s

    # Dynamics / STT order
    stt_order = 1

    # Truth gravity params (same as before)
    mu_true = 4.89044967462e-09
    C20_true = 6.09086686e-02
    C21_true = -2.81206646e-14
    S21_true = 3.87423500e-15
    C22_true = 1.97844553e-03
    S22_true = -7.06499291e-04
    params_true = np.array(
        [mu_true, C20_true, C21_true, S21_true, C22_true, S22_true], dtype=float
    )

    # --------------------------
    # Load SPICE & pull spacecraft truth (SPICE remains in ET)
    # --------------------------
    _ = load_kernels(KERNEL_ROOT)

    et0 = spice.utc2et(utc0)
    et1 = spice.utc2et(utc1)
    ets_full = np.linspace(et0, et1, n_obs)  # ET for SPICE
    tau_full = ets_full - ets_full[0]  # seconds since start for dynamics

    sc_state_full = np.zeros((n_obs, 6))
    for i, et in enumerate(ets_full):
        st, _ = spice.spkezr(SC_NAME, float(et), FRAME_I, ABCORR, CENTER)
        sc_state_full[i, :] = np.array(st, dtype=float)

    # --------------------------
    # Initial condition for propagation = SPICE truth at start
    # IMPORTANT: your 12D "state" is [r,v, mu,C20,C21,S21,C22,S22] with params constant.
    # --------------------------
    x0_orex_12 = np.hstack([sc_state_full[0, :], params_true])

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
    # Propagate OREX using your model (deg-2 rotating gravity, Bennu-centered J2000)
    # --------------------------
    print("\nPropagating OREX with your dynamics...")
    sol_prop, stts_prop = propagator.propagate(
        x0=x0_orex_12, t_eval=tau_full, rtol=1e-6, atol=1e-8, method="LSODA"
    )
    x_prop_full = sol_prop.y[:12, :].T  # (N,12)
    sc_prop = x_prop_full[:, 0:6]  # (N,6)

    # --------------------------
    # Validate vs SPICE: state error time history
    # --------------------------
    err = sc_prop - sc_state_full  # (N,6)
    pos_err = np.linalg.norm(err[:, 0:3], axis=1)  # km
    vel_err = np.linalg.norm(err[:, 3:6], axis=1)  # km/s

    # component-wise too (often more diagnostic)
    t_hr = tau_full / 3600.0

    print("\nValidation summary:")
    print(f"  |dr| RMS   = {np.sqrt(np.mean(pos_err**2)):.6e} km")
    print(f"  |dr| MAX   = {np.max(pos_err):.6e} km")
    print(f"  |dv| RMS   = {np.sqrt(np.mean(vel_err**2)):.6e} km/s")
    print(f"  |dv| MAX   = {np.max(vel_err):.6e} km/s")

    # --------------------------
    # Plots: trajectory + errors
    # --------------------------
    # 3D inertial trajectory compare
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        sc_state_full[:, 0],
        sc_state_full[:, 1],
        sc_state_full[:, 2],
        lw=2.5,
        label="SPICE truth",
    )
    ax.plot(
        sc_prop[:, 0],
        sc_prop[:, 1],
        sc_prop[:, 2],
        lw=2.0,
        label="Propagated (deg2 model)",
    )
    ax.scatter([0], [0], [0], s=35, label="Bennu center")
    ax.set_title("OSIRIS-REx around Bennu (Bennu-centered, J2000)")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend()
    plt.tight_layout()

    # Norm errors
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax0.plot(t_hr, pos_err)
    ax0.set_ylabel(r"$||\Delta r||$ [km]")
    ax0.grid(True, linestyle=":")
    ax1.plot(t_hr, vel_err)
    ax1.set_ylabel(r"$||\Delta v||$ [km/s]")
    ax1.set_xlabel("Time since start [hr]")
    ax1.grid(True, linestyle=":")
    plt.tight_layout()

    # Component errors (km and km/s)
    fig, axs = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    labs_r = ["dx", "dy", "dz"]
    labs_v = ["dvx", "dvy", "dvz"]
    for k in range(3):
        axs[k, 0].plot(t_hr, err[:, k])
        axs[k, 0].set_ylabel(f"{labs_r[k]} [km]")
        axs[k, 0].grid(True, linestyle=":")
        axs[k, 1].plot(t_hr, err[:, 3 + k])
        axs[k, 1].set_ylabel(f"{labs_v[k]} [km/s]")
        axs[k, 1].grid(True, linestyle=":")
    axs[2, 0].set_xlabel("Time since start [hr]")
    axs[2, 1].set_xlabel("Time since start [hr]")
    axs[0, 0].set_title("Position component error")
    axs[0, 1].set_title("Velocity component error")
    plt.tight_layout()

    plt.show()
    spice.kclear()
