"""
Paper Case 2 (Bennu particle, "mirror" design): OSIRIS-REx optical angles (RA/Dec, 22 images over
1.5 h) of a particle launched from Bennu's equator while the spacecraft looks along the spin axis.
The full-rank SRIF batch converges to a shallow local minimum 26 m from the truth; the MCMC
likelihood decides whether the posterior shows the second lobe that holds the truth.

Usage
-----
    python case2_bennu_particle.py 2a     # Case 2a: second-order STT likelihood
    python case2_bennu_particle.py 2b     # Case 2b: bank of first-order STMs (STMs only)
    python case2_bennu_particle.py stm    # single STM about the batch (for case2_comparison.py)
    # redo only the post-processing (clean RTN corner + summary) of a finished run:
    python case2_bennu_particle.py 2a --post results/case2a_bennu_particle_stt2_<stamp>
    # walkers in valley-straightening coordinates (same posterior; folder suffix _valley):
    python case2_bennu_particle.py 2a --valley

Cases
-----
2a  : the Stage 2 STTs about the batch solution ref1 are second order (DOP853; the symbolic
      B_2 tensors take ~17 min). The posterior is a curved, two-lobed
      valley in launch height and out-of-plane speed (h0, vh0): one lobe at the batch, one at
      the truth. Results in results/case2a_bennu_particle_stt2_<stamp>/.
2b  : STMs only. One STM about ref1 gives positions affine in the deviation, hence a single
      Gaussian lobe (the "stm" run). The bank (StmBankResiduals) holds BANK_SMAX / BANK_DS
      anchors on each side of ref1 along the batch's least-determined position direction
      (21 STMs over +-5 batch sigmas), and blends the two neighbouring STM predictions
      linearly. It follows the curved valley and shows both lobes.
      Results in results/case2b_bennu_particle_stmbank_<stamp>/.
stm : the single STM about ref1 (first-order Stage 2), not a paper case on its own: its chain
      is the "single STM" curve of case2_comparison.py.
      Results in results/case2_bennu_particle_stm_<stamp>/.

Valley coordinates (--valley)
-----------------------------
The two lobes lie along a bent valley in launch height; the affine-invariant stretch move only
sees its chord, so the autocorrelation time grows with the run and the lobe shares stay biased
toward the start (the batch lobe). With --valley the walkers move in the coordinates of a
triangular transport map of the whitened coordinates z (MCMCModel.setup_valley_map, ValleyMap):
z = h e + Q (y*(h) + C(h) w), h the deviation along the batch's least-determined position
direction in batch sigmas, y*(h) the profile path (the best other coordinates at fixed h,
Levenberg-Marquardt on the likelihood and prior residuals, step 0.1, until the profile is 25
below its best) and C(h) the Cholesky factor of the Gauss-Newton conditional covariance there.
The log-posterior gets log |dz/dv| = sum log diag C(h): the posterior is the same, only the
walkers' coordinates change; the chain is mapped back to z before tau, burn-in and thinning.

Design (redone 2026-11-26 in the SPICE frame IAU_BENNU)
--------------------------------------------------------
- Epoch 2019-04-29T18:35:00 UTC (+90 min, 22 images); the spacecraft is 4.1-4.3 km from Bennu,
  within 2.5 deg of the south pole axis, looking along the spin axis, so the camera barely sees
  the height h0 of the launch point along the pole.
- Launch site: nearest mesh vertex to lat 3 deg / lon -60.76 deg (body-fixed, IAU_BENNU).
- Launch velocity 160 mm/s, azimuth 188 deg (from local east toward north), elevation 12 deg,
  built in the local frame of the site: v0 = vmag R_ib0^T (cos EL (cos AZ e + sin AZ n) + sin EL u).
- Reference seed 239 (20 % / 20 % / 1 % / 1 % offsets; the reference starts ~80 m from the truth
  along the pole) and noise seed 395. They were picked, among the seeds whose linearized cost
  profile along h0 has two minima about 2.5 apart, so that the batch stops in the far one.
- Batch: full rank (rank_cutoff=False), Gauss-Newton with step halving (the BatchLSQ default,
  as in Case 1). With C21 = S21 = 0 the degree-2 field is symmetric about the equator; the cost
  along h0 is a valley with the best fit at h0 = -6.2 m (cost 34.30) and a shallow minimum at
  h0 = +25.7 m, where the batch lands (cost 36.80, 2.50 above; the linearized profile puts it
  2.45 above).
- Earlier designs: lon 110 deg, AZ -8 deg, 2019-05-20T14:35, seeds 42 / 123 (in the frame with
  the pole at RA -85.65 deg, before the fix of make_bennu_rotation_matrix), then lon -129.65 deg,
  seeds 3 / 631 (pole fixed, but a rounded pole and W(et0) = 0 instead of the IAU model). In
  IAU_BENNU, W(et0) = 291.11 deg; lon -60.76 = -129.65 - 291.11 deg puts the site where the
  second design had it in inertial space at et0, which keeps the flat cost valley.
- MCMC: the settings shared by all the paper cases (see "MCMC settings" in the main).
Everything else (noise 0.1 px, priors 0.25 km / 0.3 m/s / 1 %) is the setting of the earlier
particle arc.

Outputs
-------
- the usual figures (scene, geometry, traces without the stranded walkers, ACF, post-fits,
  corners);
- chain.npz (MCMCModel.save_chain) and batch.npz (x0_ref1, cov1, true_delta, x0_true, the data
  y_obs / obs_weights / sc_pos / sigma, the ref1 trajectory, the dynamics constants; for 2b the
  bank anchors), read by case2_comparison.py;
- fig_corner_rtn_<tag>.pdf (tag stt2 / stmbank / stm): RTN corner redrawn without stranded
  walkers (a walker whose post-burn-in median log-posterior is > STRANDED_DLOGP below the
  ensemble median), plus a printed summary (acceptance, tau, truth Mahalanobis distances under
  the batch and the sampled covariance).

The MCMC walker start and the emcee moves draw from the global NumPy state, seeded with
SEED_WALKERS before the run, so a rerun on the same machine reproduces the chain.

Original scenario docstring follows.
-------------------------------------------------------------------------------------------
Scenario (from scratch, but uses your existing STTPropagator + MCMCModel):

- Spacecraft (OSIRIS-REx) trajectory is TRUTH from SPICE SPK (Bennu-centered, J2000).
- Particle detaches from Bennu surface (point in body-fixed), then propagates in J2000 (Bennu-centered).
- Bennu rotates with constant spin about its pole: the IAU model of bennu_v17.tpc (pole, rate and
  prime meridian at utc0, get_CS_unnormalized.bennu_spin_spice), so the body-fixed frame is SPICE
  IAU_BENNU. No tau state.
- Particle dynamics uses degree-2 gravity potential in body-fixed:
    U = mu/r * (1 + (R_ref/r)^2 * sum_{m=0..2} P2m(sinφ)*(C2m cos mλ + S2m sin mλ))
  Acceleration is computed as a = ∇U in body-fixed, then rotated to inertial.

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

# TODO: change plot from x to z!
# TODO: Add SRP and attitude? maybe create measurement gaps
# TODO: Read chelseay and make realistic
# TODO: Check all the math
# TODO: Implement ADS? To improve performance and correctness

# NOTE: For second order, you generally need tighter tolerances, use `RK45`,
# and call `propagate_state` or `propagate_state_stm_only` instead of `propagate`.


import os
import shutil
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace


import sympy as sp
import numpy as np
from itertools import product

import spiceypy as spice
import trimesh
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

from STTPropagationND import STTPropagatorND
from MCMC import MCMCModel
from BatchLSQ import BatchLSQ
from get_CS_unnormalized import bennu_10_10, bennu_spin_spice

# Journal figures (Okabe-Ito colors, LaTeX text, one vector PDF per figure) live in MCMCPlots.py;
# MCMCPlots.configure() in the main applies the journal style.
import MCMCPlots

MCMCPlots.SHOW_FIGS = False  # long unattended runs: save the PDFs only

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
    W = w0 + omega * t

    # IAU build from frame (passive) rotations, R3(th) = [[c, s, 0], [-s, c, 0], [0, 0, 1]]:
    # R_ib = R3(W) R1(pi/2 - delta) R3(alpha + pi/2). Body +Z is then the pole
    # (cos d cos a, cos d sin a, sin d) and the body spins at +omega about it. (Active rotations
    # here put the pole at RA = -alpha and reversed the spin: fixed 2026-11-25.)
    def Rz(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def Rx(th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])

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
    """
    Shape-based visibility test using ray-mesh intersection.

    Inputs
    ------
    sc_pos_i    : (N,3) spacecraft position in inertial (Bennu-centered, J2000)
    part_pos_i  : (N,3) particle position in inertial (Bennu-centered, J2000)
    bennu_mesh_bf : trimesh.Trimesh, Bennu shape in BODY-FIXED frame, centered at origin
    alpha,delta,omega,w0 : Bennu pole/spin model used by make_bennu_rotation_matrix
    tau         : (N,) seconds since start (same "tau" you use for dynamics); if None assumes t=0 for all
    eps         : small margin to avoid classifying limb/touch as occulted
    use_embree  : try to use pyembree for speed if available

    Returns
    -------
    visible : (N,) bool
        True if particle is visible (NOT occulted by the mesh).
    """
    sc_pos_i = np.asarray(sc_pos_i, dtype=float)
    part_pos_i = np.asarray(part_pos_i, dtype=float)
    N = sc_pos_i.shape[0]

    if tau is None:
        tau = np.zeros(N, dtype=float)
    else:
        tau = np.asarray(tau, dtype=float)
        if tau.shape[0] != N:
            raise ValueError("tau must have same length as sc_pos_i/part_pos_i")

    # Build a ray intersector
    if use_embree:
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector

            intersector = RayMeshIntersector(bennu_mesh_bf)
        except Exception:
            intersector = bennu_mesh_bf.ray
    else:
        intersector = bennu_mesh_bf.ray

    # Rotate inertial -> body-fixed at each epoch
    sc_b = np.zeros_like(sc_pos_i)
    pt_b = np.zeros_like(part_pos_i)
    for k in range(N):
        R_ib = make_bennu_rotation_matrix(alpha, delta, omega, t=t0 + tau[k], w0=w0)
        sc_b[k] = R_ib @ sc_pos_i[k]
        pt_b[k] = R_ib @ part_pos_i[k]

    # Ray origins and directions in BODY-FIXED
    d = pt_b - sc_b
    rng = np.linalg.norm(d, axis=1)
    # Handle degenerate SC==particle
    good = rng > 0.0
    dirs = np.zeros_like(d)
    dirs[good] = d[good] / rng[good, None]

    # Query first hit distance along each ray
    # trimesh expects (M,3) origins and directions
    # intersects_first returns distance; np.nan if no hit
    dist_hit = intersector.intersects_first(ray_origins=sc_b, ray_directions=dirs)

    # Visible if:
    # - no hit at all   OR
    # - first hit is beyond the particle range (with margin eps)
    # Occulted if hit occurs before particle.
    # Note: dist_hit is in same units as mesh coords (km if you scaled mesh).
    visible = np.ones(N, dtype=bool)
    # If intersects_first returns -1 sometimes depending on backend, treat as "no hit"
    no_hit = np.isnan(dist_hit) | (dist_hit < 0.0)
    visible[~no_hit] = dist_hit[~no_hit] >= (rng[~no_hit] - eps)

    # Degenerate rays: if range==0, treat as visible (or set False; your choice)
    visible[~good] = True

    return visible


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

    # frame (passive) rotations, as make_bennu_rotation_matrix
    def Rz(th):
        c, s = sp.cos(th), sp.sin(th)
        return sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    def Rx(th):
        c, s = sp.cos(th), sp.sin(th)
        return sp.Matrix([[1, 0, 0], [0, c, s], [0, -s, c]])

    R_ib = Rz(W) * Rx(sp.pi / 2 - delta) * Rz(alpha + sp.pi / 2)
    R_bi = R_ib.T

    # define body-fixed coordinates as independent symbols
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
    a_b_s = sp.Matrix([sp.diff(U_b, xb_s), sp.diff(U_b, yb_s), sp.diff(U_b, zb_s)])

    # now substitute actual r_b(t) = R_ib(t) * r_i into that acceleration
    r_b_expr = R_ib * r_i
    subs_rb = {xb_s: r_b_expr[0], yb_s: r_b_expr[1], zb_s: r_b_expr[2]}
    a_b = sp.Matrix([a_b_s[i].subs(subs_rb) for i in range(3)])

    # inertial acceleration
    a_i = R_bi * a_b

    # augmented dynamics (12D)
    f = sp.Matrix([vx, vy, vz, a_i[0], a_i[1], a_i[2], 0, 0, 0, 0, 0, 0])

    # A and higher-order tensors
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

    # NOTE: for B_k, lambdify on .tolist() returns nested Python lists; that’s fine if your
    # STTPropagator handles it. If not, wrap with np.array(...) inside your propagator.
    B_funcs = {
        k: sp.lambdify(args, B_syms[k].tolist(), "numpy") for k in range(2, order + 1)
    }

    return f_func, A_func, B_funcs


# ============================================================
# Measurement generation (RA/DEC from SPICE SC to particle)
# ============================================================


def generate_opnav_measurements_from_sc(
    x_part, sc_state, sigma_ra, sigma_dec, rng, add_outliers=False
):
    """
    x_part: (N,12) or (N,6) particle in inertial
    sc_state: (N,6) spacecraft inertial
    """
    los = x_part[:, :3] - sc_state[:, :3]
    ra, dec, _, _ = radec_and_partials_from_los(los)

    ra_meas = ra + rng.normal(0.0, sigma_ra, size=ra.shape)
    dec_meas = dec + rng.normal(0.0, sigma_dec, size=dec.shape)

    if add_outliers:
        p_out = 0.02
        mask = rng.random(size=ra.shape) < p_out
        ra_meas[mask] += rng.normal(0, 20 * sigma_ra, size=np.sum(mask))
        dec_meas[mask] += rng.normal(0, 20 * sigma_dec, size=np.sum(mask))

    y = np.empty(2 * len(ra))
    y[0::2] = ra_meas
    y[1::2] = dec_meas
    return y


def radec_and_partials_from_los(los):
    """
    los: (N,3) from SC to particle
    returns:
      ra, dec: (N,)
      d_ra_d_r, d_dec_d_r: (N,3) wrt particle position (same as wrt los)
    """
    x = los[:, 0]
    y = los[:, 1]
    z = los[:, 2]

    rxy2 = x * x + y * y
    rxy = np.sqrt(np.maximum(rxy2, 1e-30))
    rho2 = rxy2 + z * z
    rho = np.sqrt(np.maximum(rho2, 1e-30))

    # RA: use raw atan2, NO modulo
    ra = np.arctan2(y, x)  # (-pi, pi]
    denom_ra = np.maximum(rxy2, 1e-30)
    d_ra_dx = -y / denom_ra
    d_ra_dy = x / denom_ra
    d_ra_dz = 0.0 * z

    # -DEC: use atan2(z, rxy)
    dec = np.arctan2(z, rxy)

    # ddec/dx, ddec/dy, ddec/dz
    # dec = atan2(z, rxy)
    # ddec/dz = rxy / (rxy^2 + z^2) = rxy / rho^2
    # ddec/drxy = -z / (rxy^2 + z^2) = -z / rho^2
    # drxy/dx = x/rxy, drxy/dy = y/rxy
    denom_dec = np.maximum(rho2, 1e-30)

    d_dec_dz = rxy / denom_dec
    d_dec_drxy = -z / denom_dec

    d_dec_dx = d_dec_drxy * (x / rxy)
    d_dec_dy = d_dec_drxy * (y / rxy)

    d_ra_d_r = np.stack([d_ra_dx, d_ra_dy, d_ra_dz], axis=1)
    d_dec_d_r = np.stack([d_dec_dx, d_dec_dy, d_dec_dz], axis=1)

    return ra, dec, d_ra_d_r, d_dec_d_r


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
# Figure symbols / units (shared by the main run and the post-processing)
# ============================================================

# Display units: km -> m, km/s -> mm/s, km^3/s^2 -> m^3/s^2 (rescaled by powers of ten from
# the batch sigmas after Stage 1)
PARAM_SYM = [
    r"\delta x_0",
    r"\delta y_0",
    r"\delta z_0",
    r"\delta v_{x,0}",
    r"\delta v_{y,0}",
    r"\delta v_{z,0}",
    r"\delta GM",
    r"\delta C_{20}",
    r"\delta C_{21}",
    r"\delta S_{21}",
    r"\delta C_{22}",
    r"\delta S_{22}",
]
PARAM_BASE_SCALE = np.array([1e3] * 3 + [1e6] * 3 + [1e9] + [1.0] * 5)
PARAM_BASE_UNIT = ["m"] * 3 + ["mm/s"] * 3 + [r"m$^3$/s$^2$"] + [""] * 5

# A walker is "stranded" if its post-burn-in median log-posterior is more than this below the
# ensemble median (a walker left far out in GM, median log-posterior ~ -1e3 to -1e5 against
# ~ +120 for the ensemble, drawing outlier islands in the corner plot); its samples are left
# out of fig_corner_rtn_<tag> and of the summary. So are the samples of other walkers that lie
# this far below (a walker that reached the posterior only after the burn-in).
STRANDED_DLOGP = 100.0

# Mark the posterior mode (MCMCModel.find_map, an optimum, not the best sample theta_hat) and
# the batch estimate ref1 (delta = 0) on every corner plot, fig_corner_rtn_<tag> included.
# The mode is saved in batch.npz (theta_map), so --post redraws it; folders from runs before
# it was saved get no mode marker.
SHOW_MAP = True
SHOW_BATCH_MEAN = True


# ============================================================
# Post-processing: stranded walkers, clean RTN corner, summary
# ============================================================


def _mahalanobis(d, P):
    return float(np.sqrt(d @ np.linalg.solve(P, d)))


def postprocess_run(run_dir, tag):
    """
    From chain.npz and batch.npz in run_dir: drop stranded walkers, redraw the RTN corner as
    fig_corner_rtn_<tag>.pdf, and print the summary quoted in the paper.
    """
    run_dir = Path(run_dir)
    ch = np.load(run_dir / "chain.npz")
    b = np.load(run_dir / "batch.npz")
    samples = ch["samples"]
    lpc = ch["log_prob_chain"]  # (n_steps, n_walkers), post-burn-in thinned
    nw = lpc.shape[1]

    # Stranded walkers. Flat sample index = step * n_walkers + walker (emcee get_chain(flat))
    # Samples of the other walkers below the same threshold (a walker that reached the posterior
    # only after the burn-in) are dropped too
    lp_w = np.median(lpc, axis=0)
    lp_cut = np.median(lp_w) - STRANDED_DLOGP
    bad = np.where(lp_w < lp_cut)[0]
    keep = ~np.isin(np.arange(samples.shape[0]) % nw, bad)
    late = keep & (ch["log_probs"] < lp_cut)
    keep &= ~late
    print(f"\n[Post] {run_dir.name}")
    print(
        f"[Post] stranded walkers: {bad.tolist()} (median log-post "
        f"{np.round(lp_w[bad], 1).tolist()} vs ensemble {np.median(lp_w):.1f}); "
        f"late samples of other walkers: {late.sum()}; kept {keep.sum()}/{keep.size} samples"
    )
    # Acceptance and autocorrelation times: as stored by the run (all walkers) and recomputed
    # on the stored full chain without the stranded walkers (the numbers quoted in the paper)
    import emcee

    good = np.setdiff1d(np.arange(nw), bad)
    acc = ch["acceptance_fraction"]
    tau_all = ch["tau"]
    tau_good = (
        emcee.autocorr.integrated_time(ch["full_chain"][:, good, :], quiet=True)
        * int(ch["full_stride"])
    )
    print(
        f"[Post] n_steps {int(ch['n_steps'])}, burn-in {int(ch['burn_in'])}, thin "
        f"{int(ch['thin'])}\n"
        f"[Post] all walkers : acceptance {acc.mean():.3f}, tau {tau_all.min():.0f}-"
        f"{tau_all.max():.0f} steps\n"
        f"[Post] kept walkers: acceptance {acc[good].mean():.3f}, tau {tau_good.min():.0f}-"
        f"{tau_good.max():.0f} steps (n_steps / tau_max = "
        f"{int(ch['n_steps']) / tau_good.max():.0f})"
    )

    # Truth Mahalanobis distances over position, velocity, GM (7) and all 12 parameters
    cov1, t = b["cov1"], b["true_delta"]
    for k in (7, 12):
        line = f"[Post] Mahalanobis {k:2d}D: truth|batch = {_mahalanobis(t[:k], cov1[:k, :k]):6.2f}"
        for lab, X in (("all", samples), ("kept", samples[keep])):
            mu, S = X.mean(0), np.cov(X.T)
            line += f"   truth|MCMC({lab}) = {_mahalanobis(t[:k] - mu[:k], S[:k, :k]):6.2f}"
        print(line)

    # RTN corner without stranded walkers (the layout of MCMCPlots' fig_corner_rtn)
    MCMCPlots.configure(PARAM_SYM, PARAM_BASE_UNIT, PARAM_BASE_SCALE, body="Bennu", target="Particle")
    disp_scale, disp_unit = MCMCPlots.auto_param_scales(
        PARAM_BASE_SCALE, PARAM_BASE_UNIT, np.sqrt(np.diag(cov1))
    )
    MCMCPlots.configure(PARAM_SYM, disp_unit, disp_scale)
    MCMCPlots.set_figure_dir(run_dir)
    A = MCMCPlots.rtn_corner_map(b["x0_ref1"], 12)
    rtn_scale, rtn_unit = MCMCPlots.auto_param_scales(
        PARAM_BASE_SCALE, PARAM_BASE_UNIT, np.sqrt(np.diag(A @ cov1 @ A.T))
    )
    theta_map = b["theta_map"] if SHOW_MAP and "theta_map" in b.files else None
    kw = MCMCPlots._mapped_corner_kw(A, t, np.zeros(12), cov1, theta_map)
    if SHOW_BATCH_MEAN:
        kw["show_batch_mean"] = True
    kw["param_sym"] = MCMCPlots.RTN_SYM + PARAM_SYM[6:]
    kw.update(param_unit=rtn_unit, param_scale=rtn_scale)
    out = MCMCPlots.plot_mcmc_corner(
        SimpleNamespace(samples=samples[keep], param_priors=None),
        name=f"fig_corner_rtn_{tag}",
        **kw,
    )
    print(f"[Post] RTN corner -> {out}")


# ============================================================
# Case 2b: bank of STMs along the batch's least-determined position direction
# ============================================================


class StmBankResiduals:
    """
    Normalized RA/Dec residuals from a bank of first-order STMs (Case 2b). Picklable.

    A single STM about ref1 makes the positions affine in the deviation delta0, so the
    posterior is one Gaussian lobe. The bank puts anchors delta_k = s_k v on the line along
    which the batch's least-determined position direction g (unit vector on the position
    slots) moves while every other parameter follows its batch correlation with it:
    v = cov1 g / sqrt(g' cov1 g), s_k in batch sigmas. Each anchor has its own trajectory
    x_k(t) and STM Phi_k(t). A sample delta0 has line coordinate l = g' delta0 / sqrt(g' cov1 g);
    with s_i <= l < s_i+1 and w = (l - s_i) / (s_i+1 - s_i), the positions are
        r(t) = (1 - w) [r_i + Phi_i (delta0 - delta_i)] + w [r_i+1 + Phi_i+1 (delta0 - delta_i+1)],
    and beyond the end anchors the end STM extrapolates linearly. RA/Dec are exact functions
    of r(t), as in the STT likelihoods.
    """

    def __init__(self, s, g, sl, delta_k, r_k, phi_k, y_obs, sc_pos, sigma, obs_weights):
        self.s = np.asarray(s, dtype=float)  # (K,) anchor coordinates [batch sigmas]
        self.g = np.asarray(g, dtype=float)  # (12,) direction (position slots only)
        self.sl = float(sl)  # batch sigma along g [km]
        self.delta_k = np.asarray(delta_k, dtype=float)  # (K, 12) anchor deviations about ref1
        self.r_k = np.asarray(r_k, dtype=float)  # (K, T, 3) anchor positions
        self.phi_k = np.asarray(phi_k, dtype=float)  # (K, T, 3, 12) position rows of the STMs
        self.y_obs = np.asarray(y_obs, dtype=float)
        self.sc_pos = np.asarray(sc_pos, dtype=float)
        self.w = np.empty_like(self.y_obs)
        self.w[0::2], self.w[1::2] = sigma
        self.obs_weights = np.asarray(obs_weights, dtype=float)

    def positions(self, delta0):
        delta0 = np.asarray(delta0, dtype=float).reshape(-1)[:12]
        l = float(self.g @ delta0) / self.sl
        i = int(np.clip(np.searchsorted(self.s, l) - 1, 0, self.s.size - 2))
        w = float(np.clip((l - self.s[i]) / (self.s[i + 1] - self.s[i]), 0.0, 1.0))
        ra = self.r_k[i] + np.einsum("tij,j->ti", self.phi_k[i], delta0 - self.delta_k[i])
        rb = self.r_k[i + 1] + np.einsum(
            "tij,j->ti", self.phi_k[i + 1], delta0 - self.delta_k[i + 1]
        )
        return (1.0 - w) * ra + w * rb

    def __call__(self, delta0):
        ra_model, dec_model, _, _ = radec_and_partials_from_los(
            self.positions(delta0) - self.sc_pos
        )
        res = np.empty_like(self.y_obs)
        res[0::2] = wrap_to_pi(self.y_obs[0::2] - ra_model)
        res[1::2] = self.y_obs[1::2] - dec_model
        return (res / self.w) * self.obs_weights


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description=__doc__.split("Cases")[0])
    ap.add_argument("case", nargs="?", default="2a", choices=["2a", "2b", "stm"])
    ap.add_argument(
        "--post",
        metavar="RUN_DIR",
        help="only redo the post-processing of a finished run folder",
    )
    ap.add_argument(
        "--demoves",
        action="store_true",
        help="sample with emcee's DE + DE-snooker moves (MCMCModel.run use_demoves) instead of "
        "the stretch move; adds _de to the results folder (a sampler test, not a paper setting)",
    )
    ap.add_argument(
        "--valley",
        action="store_true",
        help="walkers in valley-straightening coordinates (MCMCModel.setup_valley_map along the "
        "batch's least-determined position direction; same posterior, stretch move); adds "
        "_valley to the results folder",
    )
    args = ap.parse_args()
    T0_SCRIPT = time.perf_counter()  # [Timing] summary at the end (run.log)
    CASE = args.case
    # results folder and fig_corner_rtn_<TAG>
    TAG = {"2a": "stt2", "2b": "stmbank", "stm": "stm"}[CASE]
    RUN_NAME = {
        "2a": "case2a_bennu_particle_stt2",
        "2b": "case2b_bennu_particle_stmbank",
        "stm": "case2_bennu_particle_stm",
    }[CASE]
    if args.post:
        postprocess_run(args.post, TAG)
        sys.exit(0)

    # --------------------------
    # USER SETTINGS (edit these)
    # --------------------------
    KERNEL_ROOT = Path("./data/kernels")
    SC_NAME = "OSIRIS-REX"
    CENTER = "BENNU"
    FRAME_I = "J2000"
    ABCORR = "NONE"

    # Observation window (must be covered by SPK): the spacecraft is 4.1-4.3 km from Bennu,
    # within 2.5 deg of the south pole axis, looking along the spin axis
    utc0 = "2019-04-29T18:35:00"
    utc1 = "2019-04-29T20:05:00"
    n_obs = 22

    # Bennu physical
    R_bennu = 0.290  # km
    R_ref = R_bennu

    # Bennu spin + pole truth: the IAU model of the PCK (bennu_v17.tpc) about et0, set after the
    # kernels are loaded (alpha_true, delta_true, omega_true, w0_true)

    # Dynamics / STT order. Case 2a: order 2 for the Stage 2 STTs (MCMC likelihood) only;
    # truth, data and batch stay on a first-order propagator (propagator1 below). Cases 2b and
    # stm are first order throughout.
    stt_order = 2 if CASE == "2a" else 1

    # Case 2b bank of STMs: anchors at s = -BANK_SMAX, ..., +BANK_SMAX batch sigmas, step
    # BANK_DS, along the batch's least-determined position direction (21 STMs)
    BANK_SMAX = 5.0
    BANK_DS = 0.5

    # Stage-1 batch solver: "srif" (square-root information filter, Householder) or "ud"
    # (Bierman U-D, one scalar measurement at a time), both full rank here (rank_cutoff=False):
    # Gauss-Newton with step halving from the reference converges to the shallow local
    # minimum at h0 = +25.7 m (cost 36.80; the best fit is at h0 = -6.2 m, cost 34.30).
    # "ud" adds "_ud" to the results folder.
    BATCH_SOLVER = "srif"

    # Truth gravity params: unnormalized degree-2 field of get_CS_unnormalized.bennu_10_10
    # (OSIRIS-REx radio science, R_ref = 0.290 km, C20 = -J2 N20 < 0: oblate), as in case 2
    mu_true, C_unn, S_unn = bennu_10_10()
    C20_true, C21_true, S21_true = C_unn[2, 0], C_unn[2, 1], S_unn[2, 1]
    C22_true, S22_true = C_unn[2, 2], S_unn[2, 2]
    params_true = np.array(
        [mu_true, C20_true, C21_true, S21_true, C22_true, S22_true], dtype=float
    )

    # Measurement noise
    sigma_pix = 0.1  # pixel noise (1-sigma)
    fov_deg = 44.0  # full FOV [deg]  (e.g. NavCam ~44 deg)
    n_pix = 2592  # pixels across FOV (e.g. NavCam)
    fov_rad = np.deg2rad(fov_deg)
    sigma_angle = sigma_pix * (fov_rad / n_pix)  # radians
    sigma_ra = sigma_angle
    sigma_dec = sigma_angle

    # reference is perturbed by these fractions of |truth|
    rng_ref = np.random.default_rng(239)
    ref_pct_r = 0.2  # 20% of each position component
    ref_pct_v = 0.2  # 20% of each velocity component
    ref_pct_mu = 0.01  # 1% of mu
    ref_pct_c = 0.01  # 1% of each C/S coefficient

    # priors (on deltas about the reference) are these fractions of |truth|
    # NOTE: ach component of the particle's position is assigned a 250 m a priori
    # uncertainty, which in three dimensions roughly equates to Bennu's volume.
    # Each component of the particle's velocity is assigned a 30 cm/s a priori
    # uncertainty, which is the same order of magnitude as the escape speeds on
    # Bennu's surface.
    # prior_pct_r = ref_pct_r  # 1% of each position component
    # prior_pct_v = ref_pct_v  # 1% of each velocity component
    # prior_pct_mu = ref_pct_mu  # 1% of mu
    # prior_pct_c = ref_pct_c  # 1% of each C/S coefficient
    sig_prior_r = np.full(3, 0.250)  # km
    sig_prior_v = np.full(3, 3.0e-4)  # km/s
    sig_prior_mu = np.abs(mu_true) * 1e-2  # 1% prior on mu
    sig_prior_c = np.abs(params_true[1:]) * 1e-2  # 1% on C/S terms
    prior_looseness = 1  # 1.1 * 1e2

    # Integration of every propagation (truth, batch, Stage 2 STTs / STM bank), as in all the
    # paper cases
    ivp = dict(method="DOP853", rtol=1e-11, atol=1e-14)

    # MCMC settings (the same in all the paper cases): walkers start from N(MAP, batch
    # covariance) (whitened by cov1, spread 1); at least n_samples steps, then on in blocks of
    # CHECK_EVERY steps until n_steps >= N_TAU x max(tau) (at most MAX_SAMPLES; tau without
    # the stranded walkers); every THIN_BY-th step stored; burn-in 2 x max(tau), thinning
    # 0.5 x min(tau)
    n_walkers = 128
    n_samples = 100000  # minimum
    N_TAU = 50
    CHECK_EVERY = 50000
    MAX_SAMPLES = 1000000
    THIN_BY = 10
    burn_in = None  # None: 2 x max(tau)
    thin = None  # None: 0.5 x min(tau)
    spherical_spread = 1.0  # walker start, whitened by the batch covariance
    SEED_WALKERS = 7
    # Saved chain: every run writes chain.npz into its results folder. Set LOAD_CHAIN to
    # such a file to skip the sampling and redo the diagnostics and figures from it.
    LOAD_CHAIN = None  # e.g. "results/case2a_bennu_particle_stt2_<stamp>/chain.npz"
    # Also save fig_corner_rtn: the corner plot with the position and velocity deviations in
    # the RTN frame (radial, along-track, orbit normal) of the batch solution ref1 about the
    # central body (MCMCPlots.rtn_corner_map), from every post-burn-in sample. Off: the RTN
    # corner of this script is fig_corner_rtn_<tag>, drawn by postprocess_run without the
    # stranded walkers.
    CORNER_RTN = False

    # Figures: parameter symbols and display units are PARAM_SYM / PARAM_BASE_* (module
    # level), one PDF per figure in a timestamped folder
    MCMCPlots.configure(
        PARAM_SYM,
        PARAM_BASE_UNIT,
        PARAM_BASE_SCALE,
        rotation=make_bennu_rotation_matrix,
        body="Bennu",
        target="Particle",
        obs_word="Image",
        site_label="Ejection Site",
        resid_labels=("RA", "Dec"),
        fallback_dir=Path.home() / "bennu_figures_fallback",
    )
    # Folder name: <RUN_NAME>_YYMMDDHHMMSS
    fig_dir = MCMCPlots.set_figure_dir(
        Path(__file__).resolve().parent
        / "results"
        / f"{RUN_NAME}_{datetime.now().strftime('%y%m%d%H%M%S')}{'_ud' if BATCH_SOLVER == 'ud' else ''}{'_de' if args.demoves else ''}{'_valley' if args.valley else ''}"
    )
    print(
        f"[Case] {CASE}: n_obs = {n_obs}, MCMC likelihood = "
        f"{dict(stt2='second-order STT', stmbank='bank of STMs', stm='single STM')[TAG]}"
    )
    print(f"[Figures] -> {fig_dir}")

    # --------------------------
    # Load SPICE & spacecraft truth (SPICE remains in ET)
    # --------------------------
    _ = load_kernels(KERNEL_ROOT)

    et0 = spice.utc2et(utc0)
    # Bennu pole, spin rate and prime meridian at et0 (R_ib(tau) = spice.pxform("J2000",
    # "IAU_BENNU", et0 + tau))
    alpha_true, delta_true, omega_true, w0_true = bennu_spin_spice(et0)
    print(
        f"[SPICE] Bennu pole RA {np.rad2deg(alpha_true):.5f} deg, Dec {np.rad2deg(delta_true):.5f} deg, "
        f"period {2 * np.pi / omega_true / 3600:.6f} h, W(et0) {np.rad2deg(w0_true):.4f} deg"
    )
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
    mesh_path = "data/shape_models/BennuRadar.obj"
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

    lat_desired = np.deg2rad(3.0)
    lon_desired = np.deg2rad(-60.76)
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
        alpha_true, delta_true, omega_true, t=0.0, w0=w0_true
    )
    r0_true = R_ib0.T @ pos_detach_bf  # inertial

    # Launch velocity: 160 mm/s, azimuth AZ from local east toward north, elevation EL above
    # the local horizontal, in the local frame of the site (up = radial), then inertial
    vmag = 1.6e-4  # km/s
    az_launch = np.deg2rad(188.0)
    el_launch = np.deg2rad(12.0)
    up_bf = pos_detach_bf / np.linalg.norm(pos_detach_bf)
    east_bf = np.cross([0.0, 0.0, 1.0], up_bf)
    east_bf /= np.linalg.norm(east_bf)
    north_bf = np.cross(up_bf, east_bf)
    u = R_ib0.T @ (
        np.cos(el_launch) * (np.cos(az_launch) * east_bf + np.sin(az_launch) * north_bf)
        + np.sin(el_launch) * up_bf
    )
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
        w0_rad=w0_true,
    )

    propagator = STTPropagatorND(
        order=stt_order, f_func=f_func, A_func=A_func, B_funcs=B_funcs, n=12
    )
    # First-order propagator for the truth, the batch and the Case 2b bank: the data and the
    # batch are identical in all three cases
    propagator1 = STTPropagatorND(
        order=1, f_func=f_func, A_func=A_func, B_funcs=B_funcs, n=12
    )

    # --------------------------
    # Propagate particle truth (DYNAMICS on tau_full)
    # --------------------------
    print("\nPropagating particle truth...")
    sol_true, stts_true = propagator1.propagate(x0_true, tau_full, **ivp)
    x_true_full = sol_true.y[:12, :].T  # (N,12)

    # --------------------------
    # Observability mask (occultation by Bennu sphere proxy)
    # --------------------------
    # CRITICAL FIX: Computing visibility mask but NOT filtering the time grid.
    # Filtering creates irregular time spacing which causes numerical issues
    # in time-dependent dynamics. Instead, we propagate on the FULL uniform
    # time grid and only exclude occulted observations from the cost function.

    vis_mask_full = occultation_mask(
        sc_state_full[:, 0:3], x_true_full[:, 0:3], R_bennu
    )

    # Use FULL time grid for propagation (uniform spacing required for time-dependent dynamics)
    ets = ets_full
    tau = tau_full
    sc_state = sc_state_full
    x_true = x_true_full
    vis_mask = vis_mask_full  # Keep mask for plotting/diagnostics

    print(f"\nVisibility: {np.sum(vis_mask)}/{len(vis_mask)} epochs visible.")
    print(f"Using all {len(tau)} epochs for propagation (uniform time grid).")

    # --------------------------
    # Observation scene + geometry (truth on a dense grid over the arc)
    # --------------------------
    tau_dense = np.linspace(tau_full[0], tau_full[-1], 400)
    sol_dense = propagator.propagate_state_only(
        x0_true, tau_dense, show_progress=False, **ivp
    )
    pt_dense = sol_dense.y[:3, :].T
    r_dense = np.linalg.norm(pt_dense, axis=1)
    # Epoch 0 is the launch point, on the surface itself, where the inside test is a coin
    # flip: check the arc after it.
    n_in_dense = int(
        np.sum(
            MCMCPlots.points_inside_mesh(
                bennu_mesh, pt_dense[1:], tau_dense[1:], alpha_true, delta_true, omega_true, w0_true
            )
        )
    )
    print(
        f"[Scene] truth |r| over the arc: {r_dense.min() * 1e3:.1f}-"
        f"{r_dense.max() * 1e3:.1f} m; inside the shape model at "
        f"{n_in_dense}/{tau_dense.size - 1} epochs after launch"
    )
    if n_in_dense:
        print("[WARNING] The truth trajectory crosses the shape model during the arc.")
    sc_dense = np.array(
        [
            spice.spkezr(SC_NAME, float(et0 + t), FRAME_I, ABCORR, CENTER)[0]
            for t in tau_dense
        ],
        dtype=float,
    )
    MCMCPlots.plot_scene_body_fixed(
        bennu_mesh,
        tau_obs=tau_full,
        sc_obs=sc_state_full,
        vis_mask=vis_mask,
        tau_line=tau_dense,
        sc_line=sc_dense,
        pt_line=pt_dense,
        alpha=alpha_true,
        delta=delta_true,
        omega=omega_true,
        w0=w0_true,
        detach_bf=pos_detach_bf,
    )
    MCMCPlots.plot_observation_geometry(
        tau_full,
        sc_state_full,
        x_true_full,
        vis_mask,
        tau_line=tau_dense,
        sc_line=sc_dense,
        pt_line=pt_dense,
    )

    # --------------------------
    # Generate noisy RA/DEC measurements
    # --------------------------
    # Generate measurements for ALL epochs (including occulted ones)
    # We'll handle visibility in the residual weighting
    rng_meas = np.random.default_rng(395)
    y_obs_full = generate_opnav_measurements_from_sc(
        x_part=x_true,
        sc_state=sc_state,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        rng=rng_meas,
    )

    # Create a weight vector: zero weight for occulted observations
    obs_weights = np.ones_like(y_obs_full)
    obs_weights[0::2][~vis_mask] = 0.0  # Zero weight for occulted RA
    obs_weights[1::2][~vis_mask] = 0.0  # Zero weight for occulted DEC

    y_obs = y_obs_full  # Use all measurements, rely on weights

    # --------------------------
    # Reference initial condition (perturb truth)
    # --------------------------
    print("\nBuilding reference initial state (12D)...")

    sig_ref_r = np.abs(x0_true[0:3] * ref_pct_r)
    sig_ref_v = np.abs(x0_true[3:6] * ref_pct_v)
    sig_ref_mu = np.abs(x0_true[6:7] * ref_pct_mu)
    sig_ref_c = np.abs(x0_true[7:12] * ref_pct_c)

    ref_dev = np.hstack(
        [
            rng_ref.normal(scale=sig_ref_r, size=3),
            rng_ref.normal(scale=sig_ref_v, size=3),
            rng_ref.normal(scale=sig_ref_mu, size=1),
            rng_ref.normal(scale=sig_ref_c, size=5),
        ]
    )
    x0_ref = x0_true - ref_dev
    print("\n[Reference] deviation from truth:", ref_dev)

    # --------------------------
    # Priors on 12D delta0
    # --------------------------
    # r_scale = np.linalg.norm(x0_true[0:3])
    # v_scale = np.linalg.norm(x0_true[3:6])
    # sig_prior_r = prior_pct_r * r_scale * np.ones(3)
    # sig_prior_v = prior_pct_v * v_scale * np.ones(3)
    # sig_prior_mu = np.abs(x0_true[6:7] * prior_pct_mu)
    # sig_prior_c = np.abs(x0_true[7:12] * prior_pct_c)

    prior_sigma = prior_looseness * np.hstack(
        [sig_prior_r, sig_prior_v, sig_prior_mu, sig_prior_c]
    )

    priors = [norm(loc=0.0, scale=s) for s in prior_sigma]

    print("\n[Prior] sigmas:", prior_sigma)

    # --------------------------
    # Stage 1: batch SRIF with visibility weighting
    # --------------------------
    print("\n[Stage 1] Batch SRIF (first-order STMs, Gauss-Newton iterations) to convergence...")

    batch = BatchLSQ(
        propagator=propagator1,
        tau=tau,  # seconds since start
        sc_state=sc_state,
        y_obs=y_obs,
        sigma=(sigma_ra, sigma_dec),
        obs_weights=obs_weights,  # visibility weights
        measurement=radec_and_partials_from_los,
        wrap=wrap_to_pi,
    )
    t_batch = time.perf_counter()
    x0_ref1, delta_hat1, cov1 = batch.gn_with_stm(
        x0_ref=x0_ref,
        priors=priors,  # match update_idx length
        max_iter=30,
        tol=1e-8,
        **ivp,
        verbose=True,
        solver=BATCH_SOLVER,
        rank_cutoff=False,  # full rank, see BATCH_SOLVER
        halving=True,  # step halving (the default), as in Case 1
    )
    dt_batch = time.perf_counter() - t_batch
    print(f"[Timing] batch (Stage 1) wall time {dt_batch:.1f} s")

    print("\n[Stage 1] Covariance diagonal (stdev):")
    print(np.sqrt(np.diag(cov1)))
    print("[Stage 1] delta_hat1:\n", delta_hat1)

    # Display units: rescale each parameter by a power of ten so its batch sigma reads well
    disp_scale, disp_unit = MCMCPlots.auto_param_scales(
        PARAM_BASE_SCALE, PARAM_BASE_UNIT, np.sqrt(np.diag(cov1))
    )
    MCMCPlots.configure(PARAM_SYM, disp_unit, disp_scale)

    # --------------------------
    # Stage 2: relinearize about ref1 (2a: order-2 STTs; stm: one STM; 2b: a bank of STMs)
    # --------------------------
    if CASE == "2b":
        # Least-determined position direction of the batch (largest eigenvalue of the
        # position block of cov1) and the line delta = s v on which it moves, every other
        # parameter following its batch correlation (v = cov1 g / sqrt(g' cov1 g))
        _, evec_r = np.linalg.eigh(cov1[:3, :3])
        g_bank = np.zeros(12)
        g_bank[:3] = evec_r[:, -1]
        sl_bank = float(np.sqrt(g_bank @ cov1 @ g_bank))
        v_bank = cov1 @ g_bank / sl_bank
        s_bank = np.arange(-BANK_SMAX, BANK_SMAX + 1e-9, BANK_DS)
        delta_bank = s_bank[:, None] * v_bank[None, :]
        print(
            f"\n[Stage 2] Bank of {s_bank.size} STMs along the batch's least-determined "
            f"position direction {np.round(evec_r[:, -1], 3)} (sigma {sl_bank * 1e3:.1f} m), "
            f"s = {s_bank[0]:+.1f} ... {s_bank[-1]:+.1f} batch sigmas, step {BANK_DS}"
        )
        t_stage2 = time.time()
        r_bank, phi_bank = [], []
        for d_k in delta_bank:
            sol_k, stts_k = propagator1.propagate(
                x0=x0_ref1 + d_k, t_eval=tau, show_progress=False, **ivp
            )
            r_bank.append(sol_k.y[:3, :].T)
            phi_bank.append(stts_k[1][:, :3, :])
        r_bank, phi_bank = np.array(r_bank), np.array(phi_bank)
        dt_stage2 = time.time() - t_stage2
        print(f"[Stage 2] bank propagation wall time {dt_stage2:.1f} s")
        r_ref1 = r_bank[np.argmin(np.abs(s_bank))]  # the s = 0 anchor is ref1
        residuals_normalized = StmBankResiduals(
            s_bank,
            g_bank,
            sl_bank,
            delta_bank,
            r_bank,
            phi_bank,
            y_obs,
            sc_state[:, :3],
            (sigma_ra, sigma_dec),
            obs_weights,
        )
        bank_kw = dict(
            bank_s=s_bank, bank_g=g_bank, bank_sl=sl_bank, bank_v=v_bank, bank_r=r_bank
        )
    else:
        print(f"\n[Stage 2] Propagating ref1 and computing order-{stt_order} STTs about ref1...")
        t_stage2 = time.time()
        sol_ref, stts_ref = propagator.propagate(x0=x0_ref1, t_eval=tau, **ivp)
        dt_stage2 = time.time() - t_stage2
        print(
            f"[Stage 2] STT propagation wall time {dt_stage2:.1f} s, "
            f"nfev = {sol_ref.nfev}"
        )
        r_ref1 = sol_ref.y[:3, :].T
        bank_kw = {}

        # --------------------------
        # Residual function (STT-based) with visibility weighting
        # --------------------------
        def residuals_normalized(delta0):
            _, x_est = propagator.propagate_deviation(sol_ref, stts_ref, delta0)

            los = x_est[:, :3] - sc_state[:, :3]
            ra_model, dec_model, _, _ = radec_and_partials_from_los(los)

            y_model = np.empty_like(y_obs)
            y_model[0::2] = ra_model
            y_model[1::2] = dec_model

            res = np.empty_like(y_obs)
            res[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
            res[1::2] = y_obs[1::2] - y_model[1::2]

            w = np.empty_like(y_obs)
            w[0::2] = sigma_ra
            w[1::2] = sigma_dec

            # Apply visibility weighting
            return (res / w) * obs_weights

    # --------------------------
    # Chi2 at ref1
    # --------------------------
    chi2_at_ref = np.sum(residuals_normalized(np.zeros(12)) ** 2)
    n_visible = np.sum(vis_mask)
    dof = 2 * n_visible - 12  # Only count visible observations
    print(
        f"\n[Stage 2] At ref1 (delta=0): chi2_red = {chi2_at_ref/dof:.3f}  "
        f"(chi2={chi2_at_ref:.2f}, dof={dof}, n_vis={n_visible})"
    )

    # Priors for MCMC
    delta_shift = x0_ref1 - x0_ref
    priors_ref1 = [norm(loc=-ds, scale=s) for ds, s in zip(delta_shift, prior_sigma)]

    # --------------------------
    # MCMC
    # --------------------------
    print("\n[MCMC] Running (starting from zeros about ref1)...")
    model = MCMCModel(
        residuals_func=residuals_normalized,
        initial_params=np.zeros(12),
        param_priors=priors_ref1,
        observed_data=y_obs,
    )
    model.setup_whitening(cov=0.5 * (cov1 + cov1.T))  # walker start N(MAP, cov1)
    # Batch solution, truth, data, ref1 trajectory and dynamics constants (case2_comparison.py
    # can sample the exact reference from this file while the MCMC runs); written again with
    # the MCMC estimate after the run
    batch_kw = dict(
        x0_ref1=x0_ref1,
        cov1=cov1,
        t_batch_s=dt_batch,
        t_stage2_s=dt_stage2,
        true_delta=x0_true - x0_ref1,
        x0_true=x0_true,
        x0_ref=x0_ref,
        vis_mask=vis_mask,
        tau=tau,
        prior_sigma=prior_sigma,
        delta_shift=delta_shift,
        case=CASE,
        y_obs=y_obs,
        obs_weights=obs_weights,
        sc_pos=sc_state[:, :3],
        sigma=np.array([sigma_ra, sigma_dec]),
        r_ref1=r_ref1,
        dyn_const=np.array([R_ref, alpha_true, delta_true, omega_true, w0_true]),
        **bank_kw,
    )
    if not LOAD_CHAIN:
        np.savez(fig_dir / "batch.npz", **batch_kw)
    if args.valley and not LOAD_CHAIN:
        # Valley-straightening coordinates (--valley): profile of the posterior along the
        # batch's least-determined position direction g (largest eigenvalue of the position
        # block of cov1, the launch height), h = g . (theta - prior mean) / sqrt(g' cov1 g);
        # the walkers follow the profile path and its conditional scale (same posterior)
        _, evec_v = np.linalg.eigh(cov1[:3, :3])
        g_valley = np.zeros(12)
        g_valley[:3] = evec_v[:, -1]
        print(f"\n[Valley] direction g = {np.round(evec_v[:, -1], 3)} (position block of cov1)")
        model.setup_valley_map(g_valley)
    if LOAD_CHAIN:
        model.load_chain(LOAD_CHAIN)  # figures from a saved run (no sampling)
        burn_in, thin = model.burn_in, model.thin
        # postprocess_run reads chain.npz from fig_dir
        shutil.copyfile(LOAD_CHAIN, fig_dir / "chain.npz")
    else:
        np.random.seed(SEED_WALKERS)  # walker start (MCMCModel.run draws from np.random)
        model.run(
            n_samples=n_samples,
            n_walkers=n_walkers,
            burn_in=burn_in,
            thin=thin,
            spherical_spread=spherical_spread,
            method_optimize="Powell",
            thin_by=THIN_BY,
            n_tau_target=N_TAU,
            max_samples=MAX_SAMPLES,
            check_every=CHECK_EVERY,
            stranded_dlogp=STRANDED_DLOGP,
            use_demoves=args.demoves,
        )
        burn_in, thin = model.burn_in, model.thin
        model.save_chain(fig_dir / "chain.npz")

    theta_hat, P_mcmc = model.get_estimate_and_covariance()

    chi2_mcmc = np.sum(residuals_normalized(theta_hat) ** 2)
    print(
        f"\n[MCMC] At theta_hat: chi2_red = {chi2_mcmc/dof:.3f}  (chi2={chi2_mcmc:.2f}, dof={dof})"
    )
    print("[MCMC] theta_hat:\n", theta_hat)
    print("\n[MCMC] Covariance diagonal (stdev):")
    print(np.sqrt(np.diag(P_mcmc)))

    true_delta = x0_true - x0_ref1
    print("\n[Truth] true_delta about ref1:\n", true_delta)
    # Posterior mode (for the corners and the post-processing)
    map_kw = dict(theta_map=model.find_map()) if SHOW_MAP else {}
    # Batch solution and truth, for the post-processing (and comparisons across runs); first
    # written before the MCMC (above), for case2_comparison.py --exact-only
    np.savez(fig_dir / "batch.npz", **batch_kw, theta_hat=theta_hat, P_mcmc=P_mcmc, **map_kw)

    # --------------------------
    # Diagnostics + journal figures (residuals, traces, autocorrelation, corner)
    # --------------------------
    # Samples are deviations about the batch solution ref1, so the batch estimate is 0 with
    # covariance cov1.
    model.summary()
    model.print_regression_diagnostics()
    # RTN corner: display units sized to the batch sigmas in R, T, N (as disp_unit is to
    # the x, y, z ones)
    A_rtn = MCMCPlots.rtn_corner_map(x0_ref1, model.ndim)
    rtn_scale, rtn_unit = MCMCPlots.auto_param_scales(
        PARAM_BASE_SCALE, PARAM_BASE_UNIT, np.sqrt(np.diag(A_rtn @ cov1 @ A_rtn.T))
    )
    # Trace figure without the stranded walkers (STRANDED_DLOGP): far from the posterior,
    # they widen the +/-3 sigma band and the y-limits until the other walkers are a flat line
    lp_w = np.median(model.log_prob_chain, axis=0)
    trace_walkers = np.where(lp_w >= np.median(lp_w) - STRANDED_DLOGP)[0]
    model.plot_journal_diagnostics(
        burn_in,
        thin,
        tau,
        vis_mask,
        true_theta=true_delta,
        batch_cov=cov1,
        corner_rtn=x0_ref1 if CORNER_RTN else None,
        corner_rtn_kw=dict(param_unit=rtn_unit, param_scale=rtn_scale),
        show_map=SHOW_MAP,
        map_theta=map_kw.get("theta_map"),
        show_batch_mean=SHOW_BATCH_MEAN,
        trace_walkers=trace_walkers,
    )

    spice.kclear()

    # Paper figure fig_corner_rtn_<tag>.pdf (stranded walkers excluded) + summary
    postprocess_run(fig_dir, TAG)

    # Wall times of the stages (the MCMC one is also in chain.npz, run_wall_s); the rest is the
    # setup (symbolic derivatives, truth, data), the posterior mode and the figures
    dt_total = time.perf_counter() - T0_SCRIPT
    dt_mcmc = getattr(model, "run_wall_s", None)
    dt_valley = getattr(model, "valley_build_s", None) if args.valley else None
    print(
        f"\n[Timing] total {dt_total:.0f} s ({dt_total / 3600:.2f} h): batch {dt_batch:.0f} s, "
        f"Stage 2 {dt_stage2:.0f} s, "
        + (f"valley map {dt_valley:.0f} s, " if dt_valley is not None else "")
        + "MCMC sampling "
        + (f"{dt_mcmc:.0f} s" if dt_mcmc is not None else "- (LOAD_CHAIN)")
        + f", rest {dt_total - dt_batch - dt_stage2 - (dt_valley or 0.0) - (dt_mcmc or 0.0):.0f} s"
    )
