"""
========================================================================
WHY HIERARCHICAL BAYESIAN ESTIMATION HERE?
========================================================================
In ordinary Bayesian estimation the posterior is

    p(theta | y) ∝ p(y | theta) · p(theta),

i.e. likelihood × prior, where the prior p(theta) is FIXED by the analyst:
its hyperparameters (here the SH-deviation scales σ_g) are chosen by hand
BEFORE looking at the data.  When the data are informative the likelihood is
sharply peaked and dominates, so this hand-picked prior scale barely affects
the answer -- the result is "let the data speak."

OpNav of a detached particle is the OPPOSITE regime.  The measurements are
angles only (RA/DEC), the arc is short (~1.5 h), and there are occultation
gaps; the line-of-sight geometry pins the particle's POSITION/VELOCITY well
but carries almost no information about the gravity-field coefficients, whose
signal on the trajectory over such a short arc is tiny.  Along those weakly
observable directions the likelihood is nearly flat, so the posterior simply
RETURNS THE PRIOR: p(theta|y) ≈ p(theta).  Whatever σ_g the analyst typed in
(say "1% of the truth") then BECOMES the answer -- a subjective modeling
choice masquerading as an estimation result, and a direct source of bias if
that guess is wrong.  A point estimate would look confident for no reason.

Hierarchical Bayes fixes this by refusing to commit to a single prior scale.
Instead of fixing σ_g, we place a (hyper)prior ON σ_g and let the data inform
it too -- a "prior on the prior":

    p(theta, σ_g | y) ∝ p(y | theta) · p(theta | σ_g) · p(σ_g).

The sensitive choice (how wide is the SH prior?) is moved up one level into a
weakly-informative hyperprior, and the model LEARNS how much spread is actually
consistent with the data rather than asserting it.  Three consequences matter
here: (i) the arbitrary "1%" is no longer hard-coded into the result; (ii) the
uncertainty about σ_g is propagated DOWN into the coefficient posterior, giving
honestly wider error bars exactly where the data are weak (adaptive shrinkage /
partial pooling); and (iii) coefficients in the same group borrow strength from
one another.  This is precisely the weak / partially-identified regime where a
fixed point-prior would dictate the answer and where hierarchical modeling pays
off.  The specific grouping, the choice over Kaula's rule, and the non-centered
parameterization that makes this samplable are justified further below.

========================================================================
Scenario (from scratch, but uses your existing STTPropagator + MCMCModel):

- Spacecraft (OSIRIS-REx) trajectory is TRUTH from SPICE SPK (Bennu-centered, J2000).
- Particle detaches from Bennu surface (point in body-fixed), then propagates in J2000 (Bennu-centered).
- Bennu rotates with constant spin about its pole (truth alpha/delta). No tau state.
- Particle dynamics uses degree-2 gravity potential in body-fixed:
    U = mu/r * (1 + (R_ref/r)^2 * sum_{m=0..2} P2m(sinφ)*(C2m cos mλ + S2m sin mλ))
  Acceleration is computed as a = ∇U in body-fixed, then rotated to inertial.

- You estimate (HIERARCHICAL variant of the base scenario):
    theta_phys = [δr0(3), δv0(3), δmu(1), δC20, δC21, δS21, δC22, δS22]  -> 12 params
  about a reference x0_ref (12D state), PLUS one hierarchical hyperparameter PER SH
  GROUP -- the per-group SH-deviation scale, sampled in log space.  The degree-2
  coefficients are grouped by spherical-harmonic ORDER m (see justification below):
    s_zonal    = ln σ_zonal     (governs C20            , m=0),
    s_tesseral = ln σ_tesseral  (governs C21, S21        , m=1),
    s_sectoral = ln σ_sectoral  (governs C22, S22        , m=2),
  for a 15D sampling vector theta = [theta_phys(12), s_zonal, s_tesseral, s_sectoral].
  The 5 SH coefficients are sampled NON-CENTERED: C = x0_ref + w·σ_g with w ~ N(0,1),
  so each group's prior width σ_g is LEARNED from the data (weakly-informative
  log-Gaussian hyperprior centred at the a-priori 1% scale).  This removes the Neal
  funnel a centered (coefficient, σ_g) parameterization would create, and the SH
  prior stays centred on x0_ref -- consistent with the Stage-1 batch and the state
  priors (no double-counting of the data).

  WHY GROUP BY ORDER m (and not lump all non-zonal terms together):
    The pooled scale σ_g of a group is set, a priori, by the RMS magnitude of its
    members.  For Bennu in its principal-axis body-fixed frame the order-1 (tesseral)
    terms C21,S21 ~ 1e-14 vanish by construction (the pole is aligned with the
    maximum-inertia axis), whereas the order-0 (zonal) oblateness C20 ~ 6e-2 and the
    order-2 (sectoral) equatorial-ellipticity terms C22,S22 ~ 1e-3 are O(1)-relevant.
    These three families differ by up to ~12 orders of magnitude.  A single
    "non-zonal" group {C21,S21,C22,S22} would let the sectoral terms (1e-3) set a
    shared scale σ_g ~ 1e-5 that the tesseral terms (1e-14) then inherit, inflating
    the C21/S21 posterior by ~10 orders of magnitude relative to the batch (which
    keeps each coefficient's own 1% prior).  Grouping by order keeps each σ_g
    representative of its members, so partial pooling shrinks like with like and the
    near-zero tesseral terms stay near zero, matching the linearized covariance.

  WHY NOT KAULA'S RULE (or a fixed power-law degree-variance prior):
    Kaula-type rules σ_n^2 ~ (K/n^2)/(2n+1) are SPECTRAL priors that regularize the
    DECAY of power ACROSS degrees n; they say nothing about the relative size of
    coefficients of the same degree but different order m.  Here (i) we estimate a
    SINGLE degree (n=2), so there is no across-degree spectrum to regularize -- the
    one quantity Kaula constrains is absent; (ii) Kaula's constant is calibrated for
    large, near-hydrostatic planetary bodies, whereas Bennu's low-degree field is
    dominated by its irregular SHAPE and rubble-pile structure and does not follow a
    smooth power law; and (iii) Kaula is a FIXED deterministic scale, not learned,
    so it cannot adapt σ_g to the (here, very weak) information the short OpNav arc
    actually carries about gravity.  The per-order hierarchical scale is the natural
    small-body, single-degree analogue: a weakly-informative, data-adaptive width
    that distinguishes the physically distinct zonal/tesseral/sectoral families
    instead of imposing an Earth-calibrated spectral slope that does not apply.

  WHY NON-CENTERED (the funnel):
    In the CENTERED form one samples (C_j, σ_g) jointly with C_j ~ N(x0_ref, σ_g^2).
    The joint geometry is a funnel (Neal 2003): as σ_g -> 0 the admissible C_j region
    pinches to a point, so the posterior curvature varies by orders of magnitude with
    σ_g and any single step size is simultaneously too large in the neck and too small
    in the mouth -- ensemble/HMC samplers stall and the σ_g tail is never explored.
    The NON-CENTERED form C_j = x0_ref + w_j·σ_g with w_j ~ N(0,1) factorizes the
    prior into a fixed unit Gaussian on w and a smooth log-Gaussian on s_g = ln σ_g,
    so the prior geometry is INDEPENDENT of σ_g (no pinch) and the sampler mixes at a
    single scale (Papaspiliopoulos et al. 2007; Betancourt & Girolami 2015).  Sampling
    s_g = ln σ_g (rather than σ_g) additionally enforces positivity and makes the
    weakly-informative hyperprior symmetric in scale (a factor ~exp(0.75) ~ 2 spread).

- Measurements: RA/DEC from spacecraft to particle, with occultation mask (sphere proxy).

Pipeline:
  1) Propagate truth particle with truth params.
  2) Query SPICE for spacecraft truth at the same epochs.
  3) Generate noisy RA/DEC on visible epochs only.
  4) Build reference (perturb truth) and propagate ref + STTs with STTPropagator.
  5) Stage-1 full nonlinear MAP (optional, slow): re-propagate each evaluation.
  6) Stage-2 STT-based MAP: use propagate_deviation with your STTPropagator.
  7) MCMC around the STT-based residual function, with a HIERARCHICAL per-group
     SH-variance prior: the per-order (zonal/tesseral/sectoral) σ_g are sampled
     alongside the state.

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

import os
import sys
import math
from pathlib import Path
from datetime import datetime


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

# Journal figures (Okabe-Ito colors, LaTeX text, one vector PDF per figure) live in MCMCPlots.py;
# MCMCPlots.configure() in the main applies the journal style.
import MCMCPlots

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved

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
# Hierarchical per-group SH-variance log-prior (NON-CENTERED, picklable)
# ============================================================
class _HierGroupVarLogPrior:
    """Picklable non-centered hierarchical per-group SH-variance log-prior.

    Sampling theta = [state(7), w_SH(n_sh), s_g(n_grp)].  The SH coordinates are
    STANDARDIZED, w ~ N(0,1); the physical coefficient is C = x0_ref + w*sigma_g
    (reconstructed in the likelihood, not here), so the group scales sigma_g and the
    SH coordinates are a-priori independent (no funnel).  Prior factors:
      * state (indices 0..6): the given Gaussian `state_priors`;
      * w (indices in `sh_index`): iid N(0,1);
      * s_g = ln sigma_g (indices in `hyper_index`): independent Gaussian hyperpriors.
    """

    def __init__(self, state_priors, sh_index, hyper_index, hyper_loc, hyper_scale):
        self.state_priors = list(state_priors)
        self.sh_index = [int(j) for j in sh_index]
        self.hyper_index = [int(j) for j in hyper_index]
        self.hyper_loc = [float(x) for x in hyper_loc]
        self.hyper_scale = [float(x) for x in hyper_scale]
        self._half_ln2pi = 0.5 * math.log(2.0 * math.pi)

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        lp = 0.0
        for i in range(len(self.state_priors)):
            lpi = self.state_priors[i].logpdf(theta[i])
            if not np.isfinite(lpi):
                return -np.inf
            lp += lpi
        for idx, loc, scale in zip(self.hyper_index, self.hyper_loc, self.hyper_scale):
            h = theta[idx]
            lp += -0.5 * ((h - loc) / scale) ** 2 - math.log(scale) - self._half_ln2pi
        for j in self.sh_index:
            z = theta[j]
            lp += -self._half_ln2pi - 0.5 * z * z
        if not np.isfinite(lp):
            return -np.inf
        return float(lp)


def make_hier_group_var_log_prior(
    state_priors, sh_index, hyper_index, hyper_loc, hyper_scale
):
    return _HierGroupVarLogPrior(
        state_priors, sh_index, hyper_index, hyper_loc, hyper_scale
    )


# ============================================================
# MAIN SCRIPT
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
    utc1 = "2019-03-01T01:30:00"
    n_obs = 22

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

    # Stage-1 batch solve. False: plain Gauss-Newton with an unscaled lstsq solve (the results
    # so far). True: column-scaled GN (solve in prior-sigma units) with step halving on the exact
    # MAP cost; the prior sigmas span ~17 decades, so the unscaled lstsq drops the weakest
    # direction and the plain GN stops short of the MAP along it. Runs go to a "_scaledGN" folder.
    SCALED_BATCH = False

    # Truth gravity params
    mu_true = 4.89044967462e-09
    C20_true = 6.09086686e-02
    C21_true = -2.81206646e-14
    S21_true = 3.87423500e-15
    C22_true = 1.97844553e-03
    S22_true = -7.06499291e-04
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
    rng_ref = np.random.default_rng(42)
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

    # MCMC settings
    # NOTE: always do a run with burn_in and thin not activated
    n_walkers = 128  # 10 *
    n_samples = 50000  # 5 *
    burn_in = 5000
    thin = 100
    spherical_spread = 1e-1
    # Saved chain: every run writes chain.npz into its results folder. Set LOAD_CHAIN to
    # such a file to skip the sampling and redo the diagnostics and figures from it.
    LOAD_CHAIN = None  # e.g. "results/bennu_particle_hiersh_<stamp>/chain.npz"

    # Figures: parameter symbols and display units (km -> m, km/s -> mm/s,
    # km^3/s^2 -> m^3/s^2; rescaled by powers of ten from the batch sigmas after Stage 1),
    # one PDF per figure in a timestamped folder (MCMCPlots.SHOW_FIGS at the top of the file)
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
        r"\log_{10}\sigma_{\mathrm{zon}}",
        r"\log_{10}\sigma_{\mathrm{tes}}",
        r"\log_{10}\sigma_{\mathrm{sec}}",
    ]
    # the chain holds s_g = ln sigma_g; the figures show log10 sigma_g = s_g / ln 10 (decades)
    LOG10_SCALE = 1.0 / np.log(10.0)
    PARAM_BASE_SCALE = np.array([1e3] * 3 + [1e6] * 3 + [1e9] + [1.0] * 5 + [LOG10_SCALE] * 3)
    PARAM_BASE_UNIT = ["m"] * 3 + ["mm/s"] * 3 + [r"m$^3$/s$^2$"] + [""] * 8
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
    fig_dir = MCMCPlots.set_figure_dir(
        Path(__file__).resolve().parent
        / "results"
        / f"bennu_particle_hiersh{'_scaledGN' if SCALED_BATCH else ''}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"[Figures] -> {fig_dir}")

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

    # tangent launch direction (physically plausible: lofting off surface)
    rhat = r0_true / np.linalg.norm(r0_true)

    # pick random vector not parallel to rhat
    u = rng.normal(size=3)
    u -= np.dot(u, rhat) * rhat
    u /= np.linalg.norm(u)

    # add small radial component (+/-) to create two near-degenerate families
    rad_frac = 0.10  # 10% radial, rest tangential
    sign = 1.0  # <-- TRICK: if you set this wrong in the reference, batch may lock on wrong lobe
    u = np.sqrt(1 - rad_frac**2) * u + sign * rad_frac * rhat

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
        x0_true, tau_dense, show_progress=False, rtol=1e-8, atol=1e-10, method="LSODA"
    )
    pt_dense = sol_dense.y[:3, :].T
    r_dense = np.linalg.norm(pt_dense, axis=1)
    # Epoch 0 is the launch point, on the surface itself, where the inside test is a coin
    # flip: check the arc after it.
    n_in_dense = int(
        np.sum(
            MCMCPlots.points_inside_mesh(
                bennu_mesh, pt_dense[1:], tau_dense[1:], alpha_true, delta_true, omega_true
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
    rng_meas = np.random.default_rng(123)
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
    prior_sigma = prior_looseness * np.hstack(
        [sig_prior_r, sig_prior_v, sig_prior_mu, sig_prior_c]
    )

    priors = [norm(loc=0.0, scale=s) for s in prior_sigma]

    print("\n[Prior] sigmas:", prior_sigma)

    # --------------------------
    # Stage 1: full nonlinear batch with visibility weighting
    # --------------------------
    print("\n[Stage 1] Gauss-Newton batch (first-order STMs) to convergence...")

    batch = BatchLSQ(
        propagator=propagator,
        tau=tau,  # seconds since start
        sc_state=sc_state,
        y_obs=y_obs,
        sigma=(sigma_ra, sigma_dec),
        obs_weights=obs_weights,  # visibility weights
        measurement=radec_and_partials_from_los,
        wrap=wrap_to_pi,
    )
    x0_ref1, delta_hat1, cov1 = batch.gn_with_stm(
        x0_ref=x0_ref,
        priors=priors,  # match update_idx length
        max_iter=30,
        tol=1e-8,
        rtol=1e-8,
        atol=1e-10,
        verbose=True,
        scaled=SCALED_BATCH,
    )

    print("\n[Stage 1] Covariance diagonal (stdev):")
    print(np.sqrt(np.diag(cov1)))
    print("[Stage 1] delta_hat1:\n", delta_hat1)

    # Display units: rescale each physical parameter by a power of ten so its batch sigma
    # reads well (the three log10 sigma_g are dimensionless)
    disp_scale, disp_unit = MCMCPlots.auto_param_scales(
        PARAM_BASE_SCALE[:12], PARAM_BASE_UNIT[:12], np.sqrt(np.diag(cov1))
    )
    MCMCPlots.configure(
        PARAM_SYM,
        list(disp_unit) + [""] * 3,
        np.hstack([disp_scale, np.full(3, LOG10_SCALE)]),
    )

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
    def residuals_normalized(delta0):
        delta0 = np.asarray(delta0)[:12]  # robust to 15D theta (drops the 3 ln sigma_g)
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

    # Priors for MCMC (state r,v,mu keep their Gaussian priors, re-centred about ref1)
    delta_shift = x0_ref1 - x0_ref
    priors_ref1 = [norm(loc=-ds, scale=s) for ds, s in zip(delta_shift, prior_sigma)]

    # --------------------------------------------------------------------------
    # HIERARCHICAL SH PRIOR: sample a per-group SH-deviation scale sigma_g, grouped by
    # spherical-harmonic order m -- ZONAL (C20), TESSERAL (C21,S21), SECTORAL (C22,S22).
    # Non-centered: C = x0_ref + w*sigma_g, w ~ N(0,1); sigma_g = exp(s_g) is sampled
    # with a weakly-informative log-Gaussian hyperprior centred at the a-priori SH scale.
    # Sampling state: theta = [delta_state(7), w_SH(5), s_g(one per group)].
    # --------------------------------------------------------------------------
    # Group SH coefficients by spherical-harmonic ORDER m, NOT into one "non-zonal"
    # bucket.  Pooling C21,S21 (m=1, truth ~1e-14, i.e. ~0 for a pole-aligned body)
    # with C22,S22 (m=2, truth ~1e-3) forces the tiny tesseral terms to inherit the
    # sectoral scale (~1e-5), ballooning their posterior far beyond the batch (which
    # keeps each coefficient's own 1% prior).  Separate groups -> each group's sigma_g
    # reflects its own member magnitudes, so C21/S21 stay ~1e-16 like the batch.
    group_index_map = {
        "zonal": [7],  # C20      (m=0)
        "tesseral": [8, 9],  # C21, S21 (m=1)  ~0 for Bennu
        "sectoral": [10, 11],  # C22, S22 (m=2)
    }
    group_labels = list(group_index_map.keys())
    n_phys = 12
    ndim_full = n_phys + len(group_labels)  # 12 + n_groups
    group_to_hyper = {g: n_phys + k for k, g in enumerate(group_labels)}

    # sigma_g hyperprior centre = RMS over the group of the a-priori SH sigmas
    # (prior_sigma[7:12], the SAME 1% scale the Stage-1 batch used) -> GN/MCMC
    # consistent, and NOT taken from cov1, so the data is not double-counted.
    logsig_loc = {
        g: float(0.5 * math.log(np.mean([prior_sigma[j] ** 2 for j in idxs])))
        for g, idxs in group_index_map.items()
    }
    hyper_logsig_scale = 0.75  # weakly informative on ln sigma_g (~factor exp(0.75))
    logsig_scale = {g: hyper_logsig_scale for g in group_labels}
    print("\n[Prior] per-group SH log-sigma hyperpriors (centre = a-priori scale):")
    for g in group_labels:
        print(
            f"   ln sigma[{g}] ~ N({logsig_loc[g]:.2f}, {logsig_scale[g]:.2f}^2)  "
            f"(sigma0={math.exp(logsig_loc[g]):.3e})"
        )

    hyper_priors = [
        norm(loc=logsig_loc[g], scale=logsig_scale[g]) for g in group_labels
    ]
    w_priors = [norm(loc=0.0, scale=1.0) for _ in range(5)]
    param_priors_full = list(priors_ref1[0:7]) + w_priors + hyper_priors

    # Init at ref1 (physical delta = 0): SH delta_C = -delta_shift_SH + w*sigma_g0,
    # so delta_C = 0 means w = delta_shift_SH / sigma_g0.
    w_init = np.zeros(5)
    for g, idxs in group_index_map.items():
        _sig0 = math.exp(logsig_loc[g])
        for j in idxs:
            w_init[j - 7] = delta_shift[j] / _sig0
    initial_params_full = np.hstack(
        [np.zeros(7), w_init, [logsig_loc[g] for g in group_labels]]
    )

    # Non-centered sampling -> physical 12D delta about ref1.  The SH coefficient
    # prior is centred on the a-priori x0_ref (delta_C = -delta_shift_SH + w*sigma_g),
    # exactly like the state priors (loc=-delta_shift) -> GN and MCMC are consistent.
    def physical_delta_from_sampling(theta_s):
        theta_s = np.asarray(theta_s, dtype=float)
        delta12 = theta_s[:12].copy()
        for g, idxs in group_index_map.items():
            _sig = math.exp(theta_s[group_to_hyper[g]])
            for j in idxs:
                delta12[j] = -delta_shift[j] + theta_s[j] * _sig
        return delta12

    def residuals_sampling(theta_s):
        return residuals_normalized(physical_delta_from_sampling(theta_s))

    # --------------------------
    # MCMC
    # --------------------------
    print("\n[MCMC] Running (hierarchical per-group SH sigma; starting at ref1)...")
    model = MCMCModel(
        residuals_func=residuals_sampling,
        initial_params=initial_params_full,
        param_priors=param_priors_full,
        observed_data=y_obs,
    )
    model.log_prior = make_hier_group_var_log_prior(
        state_priors=priors_ref1[0:7],
        sh_index=list(range(7, 12)),
        hyper_index=[group_to_hyper[g] for g in group_labels],
        hyper_loc=[logsig_loc[g] for g in group_labels],
        hyper_scale=[logsig_scale[g] for g in group_labels],
    )
    model.setup_whitening_from_priors()
    if LOAD_CHAIN:
        model.load_chain(LOAD_CHAIN)  # figures from a saved run (no sampling)
        burn_in, thin = model.burn_in, model.thin
    else:
        model.run(
            n_samples=n_samples,
            n_walkers=n_walkers,
            burn_in=burn_in,
            thin=thin,
            spherical_spread=spherical_spread,
            method_optimize="Powell",
        )
        model.save_chain(fig_dir / "chain.npz")  # before samples are edited in place

    theta_hat_s, _ = model.get_estimate_and_covariance()  # sampling-space MAP
    chi2_mcmc = np.sum(residuals_sampling(theta_hat_s) ** 2)
    print(
        f"\n[MCMC] At theta_hat: chi2_red = {chi2_mcmc/dof:.3f}  (chi2={chi2_mcmc:.2f}, dof={dof})"
    )

    # Back-transform the chain SH columns (w -> physical delta_C about ref1) so all
    # reporting/plots are in physical deviation space; sigma_g columns left as-is.
    for g, idxs in group_index_map.items():
        _sig = np.exp(model.samples[:, group_to_hyper[g]])
        for j in idxs:
            model.samples[:, j] = -delta_shift[j] + model.samples[:, j] * _sig
    model.residuals_func = residuals_normalized  # diagnostics now use physical delta

    theta_hat, P_mcmc = model.get_estimate_and_covariance()
    print("[MCMC] theta_hat:\n", theta_hat)
    print("\n[MCMC] Covariance diagonal (stdev):")
    print(np.sqrt(np.diag(P_mcmc)))

    # Per-group sigma_g posterior vs the realized truth deviation-from-a-priori scale
    # (sigma_g is the scale of C - x0_ref, whose realized value is ref_dev).
    sig_g_s = {g: np.exp(model.samples[:, group_to_hyper[g]]) for g in group_labels}
    sig_g_true = {
        g: float(np.sqrt(np.mean([ref_dev[j] ** 2 for j in idxs])))
        for g, idxs in group_index_map.items()
    }
    print("\n[Prior] Per-group SH-deviation scale posterior (sigma_g):")
    for g in group_labels:
        lo, med, hi = np.percentile(sig_g_s[g], [16, 50, 84])
        print(
            f"   sigma[{g}]: {med:.4e}  (16/84: {lo:.4e} / {hi:.4e})   "
            f"truth(dev)≈{sig_g_true[g]:.4e}"
        )

    true_delta = x0_true - x0_ref1
    true_theta = np.hstack(
        [true_delta, [math.log(max(sig_g_true[g], 1e-300)) for g in group_labels]]
    )
    print("\n[Truth] true_delta about ref1:\n", true_delta)

    # --------------------------
    # Diagnostics + journal figures (residuals, traces, autocorrelation, corner)
    # --------------------------
    model.summary()
    model.print_regression_diagnostics()

    # The chain is sampled in [state(7), w_SH(5), ln sigma_g(3)]; the figures show the
    # physical deviations about ref1 plus the three ln sigma_g (model.samples already is),
    # the latter displayed as log10 sigma_g through PARAM_SCALE.
    def chain_to_display(theta_s):
        th = np.array(theta_s, dtype=float, copy=True)
        for g, idxs in group_index_map.items():
            sig = np.exp(th[..., group_to_hyper[g]])
            for j in idxs:
                th[..., j] = -delta_shift[j] + th[..., j] * sig
        return th

    # truth in SAMPLED coordinates (log p(truth) in the trace plot)
    theta_truth_sampling = np.array(true_theta, dtype=float, copy=True)
    for g, idxs in group_index_map.items():
        sig_t = math.exp(true_theta[group_to_hyper[g]])
        for j in idxs:
            theta_truth_sampling[j] = (true_delta[j] + delta_shift[j]) / sig_t

    # Batch analog in displayed coordinates: (0, cov1) for the 12 physical deltas. The
    # sigma_g have no batch estimate (GN does not estimate them): their ellipse/line is
    # the hyperprior.
    batch_mean_full = np.hstack([np.zeros(12), [logsig_loc[g] for g in group_labels]])
    batch_cov_full = np.zeros((ndim_full, ndim_full))
    batch_cov_full[:12, :12] = cov1
    for g in group_labels:
        batch_cov_full[group_to_hyper[g], group_to_hyper[g]] = logsig_scale[g] ** 2

    def _draw_traces():
        model.residuals_func = residuals_sampling  # log p(truth) in sampled coordinates
        try:
            model.plot_journal_traces(
                burn_in,
                true_theta=true_theta,
                batch_theta=batch_mean_full,
                transform=chain_to_display,
                logp_truth_theta=theta_truth_sampling,
                prior_idx=[group_to_hyper[g] for g in group_labels],  # hyperprior mean
                prior_truth_label="Realized Deviation Scale",
            )
        finally:
            model.residuals_func = residuals_normalized

    corner_kw = dict(
        true_theta=true_theta,
        batch_mean=batch_mean_full,
        batch_cov=batch_cov_full,
        # ln sigma_g entries of batch_* are the hyperprior, not a batch estimate:
        # drawn in their own style with a "Hyperprior" legend entry. Their "truth"
        # is the realized RMS of the group deviations (no generative sigma_g):
        # hollow star / dotted line with its own legend entry
        prior_idx=[group_to_hyper[g] for g in group_labels],
        prior_truth_label="Realized Deviation Scale",
    )
    journal_figs = (
        (
            "post-fit residuals",
            lambda: model.plot_journal_residuals(tau, vis_mask, np.zeros(12)),
        ),
        ("traces", _draw_traces),
        (
            "autocorrelation",
            lambda: model.plot_journal_autocorrelation(
                burn_in,
                thin,
                transform=chain_to_display,
                groups=[list(range(6)), list(range(6, 12)), [12, 13, 14]],
            ),
        ),
        ("corner", lambda: model.plot_journal_corner(**corner_kw)),
        (
            # prior drawn in the sampled coordinates [state, w, ln sigma_g] (the
            # param_priors) and mapped like the chain; the hyperprior is part of it
            "corner + prior",
            lambda: model.plot_journal_corner(
                prior=True, prior_transform=chain_to_display, **corner_kw
            ),
        ),
    )
    for label, draw in journal_figs:
        try:
            draw()
        except Exception as err:
            print(f"[Figure] {label} skipped: {type(err).__name__}: {err}")

    spice.kclear()
