"""
Scenario (Ryugu twin of J1_scenarioA_angles_SPH2_Bennu_particle.py, uses your existing
STTPropagatorND + MCMCModel):

- Spacecraft: a NOTIONAL 2030s Japanese comet-sample-return mission concept ("CSR-30",
  not a real mission) that performs a close-proximity checkout phase at (162173) Ryugu.
  Its trajectory is TRUTH and is treated as KNOWN. There is no SPK for a notional
  spacecraft, so it is generated here: a ~2 km terminator orbit propagated with the
  same force model (Ryugu polyhedron + Sun point mass + SRP cannonball), Ryugu-centered J2000.
- Ryugu heliocentric ephemeris: JPL Horizons SPK (kernels/ryugu/), used for the Sun
  position relative to Ryugu (third-body + SRP) and for the Sun-RTN frame (S/C orbit plane).
- Particle detaches from Ryugu's surface (vertex of the SfM 49k shape model, lifted a few
  meters along the local normal; ejection velocity on top of the local rotational velocity),
  then propagates in J2000 (Ryugu-centered).
- Ryugu rotates with constant spin about its pole (IAU-style: pole RA/DEC, W = w0 + omega*t).

- Particle dynamics (inertial, Ryugu-centered):
    a = R_bi(t) g(R_ib(t) r)                             polyhedral gravity, constant density rho
                                                          (ESA polyhedral-gravity-model; GM = G rho V
                                                          is fixed by the shape + density, not estimated)
      + mu_sun [ (r_s - r)/|r_s - r|^3 - r_s/|r_s|^3 ]   Sun third body
      + eta * k_srp * AU^2 (r - r_s)/|r - r_s|^3         SRP cannonball (no shadow)
  The dynamics matrix A = df/dX is analytic: the gravity block uses the polyhedral
  gravity-gradient tensor returned by the same GravityEvaluable call (no finite differences).

- You estimate (7 params):
    theta = [δr0(3), δv0(3), δeta]
  about a reference x0_ref (also 7D state).
  eta (SRP scale) absorbs mismodeled SRP (unknown Cr, A/m, grain size).

- Measurements: RA/DEC from spacecraft to particle, with occultation from the shape model
  (ray casting on the mesh).

- Propagation: STMs ONLY (order 1). No higher-order STTs.

Pipeline (same as the Bennu script):
  1) Propagate truth particle with truth params.
  2) Spacecraft truth at the same epochs.
  3) Generate noisy RA/DEC on visible epochs only (occulted epochs get zero weight).
  4) Build reference (perturb truth).
  5) Stage-1 Gauss-Newton batch with STMs.
  6) Stage-2: relinearize STMs about the Stage-1 solution.
  7) MCMC around the STM-based residual function.
  8) Split the MCMC posterior at epoch into a Gaussian mixture (GMM), propagate every
     component to a user-given epoch with its own STM (P_k(t) = Phi_k P_k Phi_k^T), and
     check the predicted mixture against a nonlinear Monte Carlo of posterior samples.

Units:
- km, km/s, seconds (tau = seconds since start; SPICE uses ET)
"""

import os
import copy
from pathlib import Path
from datetime import datetime

import numpy as np

import spiceypy as spice
import trimesh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as mticker
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.stats import chi2, multivariate_normal, norm
from sklearn.mixture import GaussianMixture

from polyhedral_gravity import (
    GravityEvaluable,
    MetricUnit,
    Polyhedron,
    PolyhedronIntegrity,
)

from STTPropagationND import STTPropagatorND
from MCMC import MCMCModel

# ============================================================
# FIGURE STYLE (journal)
# ============================================================
# Figures are drawn at their printed size (single column COL_W, full text width TXT_W) with
# LaTeX text at 7-8 pt, so they drop into a two-column journal layout without rescaling.
# Vector PDFs; only dense layers (mesh, point clouds, walker traces) are rasterized.
# MCMC.py sets its own rcParams on import, so this block must come after that import.

COL_W = 3.5  # single-column width [in]
TXT_W = 7.16  # full text width (double column) [in]
SHOW_FIGS = True  # plt.show() after each figure is saved

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}\usepackage{mathrsfs}"
        ),
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "0.15",
        "axes.labelcolor": "0.1",
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Okabe-Ito colors (colorblind-safe; adjacent pairs checked for CVD separation).
# Each color has ONE meaning in every figure.
C_POST = "#0072B2"  # MCMC posterior / Gaussian mixture (blue)
C_BATCH = "#D55E00"  # Gauss-Newton batch / single Gaussian (vermillion)
C_SC = "#009E73"  # spacecraft, images used (bluish green)
C_OCC = "#E69F00"  # occulted epochs (orange)
C_MC = "0.62"  # nonlinear Monte Carlo (neutral grey)
C_TRUTH = "k"  # truth
C_SPAN = "0.93"  # background spans (observation arc, burn-in, +/-3 sigma)
C_XYZ = ("#56B4E9", "#CC79A7", "#E69F00")  # x / y / z components (ACF figure only)

# Estimated parameters: symbols, display units and scale from internal units (km, km/s, -)
PARAM_SYM = [
    r"\delta x_0",
    r"\delta y_0",
    r"\delta z_0",
    r"\delta \dot{x}_0",
    r"\delta \dot{y}_0",
    r"\delta \dot{z}_0",
    r"\delta \eta",
]
PARAM_UNIT = ["m", "m", "m", r"mm\,s$^{-1}$", r"mm\,s$^{-1}$", r"mm\,s$^{-1}$", r"\%"]
PARAM_SCALE = np.array([1e3, 1e3, 1e3, 1e6, 1e6, 1e6, 1e2])

# ============================================================
# Constants
# ============================================================

G_KM = 6.67430e-20  # gravitational constant [km^3 / (kg s^2)]
AU_KM = 1.495978707e8  # [km]
P_SUN_1AU = 4.56e-6  # solar radiation pressure at 1 AU [N/m^2]

N_STATE = 7  # [x y z vx vy vz eta]
STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz", "eta"]

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
    t = -np.sum(r1 * d, axis=1) / dd
    t = np.clip(t, 0.0, 1.0)
    closest = r1 + t[:, None] * d

    return np.linalg.norm(closest, axis=1) > R_body


def _Rz_frame(th):
    """Frame (passive) rotation about z."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _Rx_frame(th):
    """Frame (passive) rotation about x."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def make_ryugu_rotation_matrix(alpha, delta, omega, t, w0=0.0):
    """
    inertial->body-fixed rotation for an IAU constant-spin model:
      pole is (alpha, delta) in J2000, prime meridian angle W = w0 + omega*t

        R_ib = R3(W) R1(pi/2 - delta) R3(alpha + pi/2)     (frame rotations)

    With this convention the body +Z axis is the pole (cos d cos a, cos d sin a, sin d)
    and the body spins with angular velocity +omega about it.
    Returns R_ib (3,3) mapping r_i -> r_b.
    """
    W = w0 + omega * t
    return _Rz_frame(W) @ _Rx_frame(np.pi / 2 - delta) @ _Rz_frame(alpha + np.pi / 2)


def occultation_mask_shape(
    sc_pos_i,
    part_pos_i,
    body_mesh_bf,
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
    sc_pos_i     : (N,3) spacecraft position in inertial (body-centered, J2000)
    part_pos_i   : (N,3) particle position in inertial (body-centered, J2000)
    body_mesh_bf : trimesh.Trimesh, shape in BODY-FIXED frame, centered at origin
    alpha,delta,omega,w0 : pole/spin model used by make_ryugu_rotation_matrix
    tau          : (N,) seconds since start; if None assumes t=0 for all
    eps          : small margin to avoid classifying limb/touch as occulted
    use_embree   : try to use pyembree for speed if available

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

            intersector = RayMeshIntersector(body_mesh_bf)
        except Exception:
            intersector = body_mesh_bf.ray
    else:
        intersector = body_mesh_bf.ray

    # Rotate inertial -> body-fixed at each epoch
    sc_b = np.zeros_like(sc_pos_i)
    pt_b = np.zeros_like(part_pos_i)
    for k in range(N):
        R_ib = make_ryugu_rotation_matrix(alpha, delta, omega, t=t0 + tau[k], w0=w0)
        sc_b[k] = R_ib @ sc_pos_i[k]
        pt_b[k] = R_ib @ part_pos_i[k]

    # Ray origins and directions in BODY-FIXED
    d = pt_b - sc_b
    rng = np.linalg.norm(d, axis=1)
    good = rng > 0.0
    dirs = np.zeros_like(d)
    dirs[good] = d[good] / rng[good, None]

    # First hit distance along each ray (nan / -1 if no hit, depending on backend)
    dist_hit = np.asarray(
        intersector.intersects_first(ray_origins=sc_b, ray_directions=dirs)
    )
    if dist_hit.dtype.kind in "iu":
        # Some backends return the hit triangle index instead of a distance.
        locs, idx_ray, _ = intersector.intersects_location(
            ray_origins=sc_b, ray_directions=dirs, multiple_hits=True
        )
        dist_hit = np.full(N, np.nan)
        for loc, ir in zip(locs, idx_ray):
            dh = np.linalg.norm(loc - sc_b[ir])
            if np.isnan(dist_hit[ir]) or dh < dist_hit[ir]:
                dist_hit[ir] = dh

    visible = np.ones(N, dtype=bool)
    no_hit = np.isnan(dist_hit) | (dist_hit < 0.0)
    visible[~no_hit] = dist_hit[~no_hit] >= (rng[~no_hit] - eps)
    visible[~good] = True

    return visible


def points_inside_mesh(mesh_bf, pos_i, tau, alpha, delta, omega, w0=0.0):
    """True where an inertial position (at time tau) lies inside the body-fixed mesh."""
    pos_i = np.asarray(pos_i, dtype=float)
    r_b = np.array(
        [
            make_ryugu_rotation_matrix(alpha, delta, omega, float(t), w0) @ p
            for p, t in zip(pos_i, np.asarray(tau, dtype=float))
        ]
    )
    return np.asarray(mesh_bf.contains(r_b), dtype=bool)


# ============================================================
# Dynamics: polyhedral gravity (ESA polyhedral-gravity-model) + Sun + SRP
# ============================================================


class RyuguPolyhedralDynamics:
    """
    f(X, t) and A(X, t) = df/dX for the augmented 7D state
        X = [x y z vx vy vz eta]
    in Ryugu-centered J2000. Called as f_func(*X, t), A_func(*X, t) (STTPropagatorND API).

    Gravity: constant-density polyhedron (mesh in km, rho in kg/km^3), so the evaluable
    returns g(r_b) [km/s^2] and the gravity-gradient tensor T = d g / d r_b [1/s^2], and
    GM = G rho V follows from the shape + density (self.gm). Then
        a_grav      = R_bi g(R_ib r)
        d a / d r   = R_bi T R_ib
    f and A at the same (r, t) share ONE polyhedral evaluation (one-entry cache).
    Beyond r_far (default 50 km ~ 110 Ryugu radii) the field is replaced by the point mass
    (relative difference < 1e-5); this only matters for wild trial iterates of the
    estimator and avoids the polyhedral library's far-field precision warnings.

    Picklable: the GravityEvaluable is dropped on pickling and rebuilt lazily.
    """

    def __init__(
        self,
        vertices_km,
        faces,
        alpha,
        delta,
        omega,
        w0,
        density,
        sun_pos_spline,
        mu_sun,
        srp_k_1au,
        parallel=True,
        r_far=50.0,
    ):
        self.vertices_km = np.ascontiguousarray(vertices_km, dtype=float)
        self.density = float(density)  # bulk density [kg/m^3]
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.omega = float(omega)
        self.w0 = float(w0)
        self.sun_pos_spline = sun_pos_spline  # tau -> Sun position wrt Ryugu [km], J2000
        self.mu_sun = float(mu_sun)
        self.srp_k_1au = float(srp_k_1au)  # Cr * (A/m) * P_1AU [km/s^2] at 1 AU
        self.parallel = bool(parallel)
        self.r_far = float(r_far)

        self._M0 = _Rx_frame(np.pi / 2 - self.delta) @ _Rz_frame(self.alpha + np.pi / 2)

        self._evaluable = None
        self._key = None
        self._geo = None
        self._ensure_evaluable()

    # ---- polyhedral model ----
    def _ensure_evaluable(self):
        if self._evaluable is not None:
            return
        tri = self.vertices_km[self.faces]
        volume = np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
        if volume <= 0:
            raise ValueError("Mesh volume <= 0: faces must be oriented outward.")
        rho_km = self.density * 1e9  # [kg/m^3] -> [kg/km^3]
        poly = Polyhedron(
            (self.vertices_km.tolist(), self.faces.tolist()),
            rho_km,
            integrity_check=PolyhedronIntegrity.VERIFY,
            metric_unit=MetricUnit.KILOMETER,
        )
        self._evaluable = GravityEvaluable(poly)
        self.volume_km3 = volume
        self.gm = G_KM * rho_km * volume  # [km^3/s^2]

        # Self-check: far field must be -GM r/|r|^3
        p_far = np.array([200.0, 0.0, 0.0])
        _, g_far, _ = self._evaluable(p_far, parallel=self.parallel)
        g_far = np.asarray(g_far, dtype=float)
        g_pm = -self.gm * p_far / np.linalg.norm(p_far) ** 3
        if np.linalg.norm(g_far - g_pm) > 1e-3 * np.linalg.norm(g_pm):
            raise RuntimeError(
                f"Polyhedral normalization check failed: {g_far} vs point mass {g_pm}"
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_evaluable"] = None
        state["_key"] = None
        state["_geo"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def with_srp_k(self, srp_k_1au):
        """Copy sharing the polyhedron, with a different SRP coefficient (e.g. the S/C)."""
        other = copy.copy(self)
        other.srp_k_1au = float(srp_k_1au)
        other._key = None
        other._geo = None
        return other

    # ---- frames ----
    def R_ib(self, t):
        return _Rz_frame(self.w0 + self.omega * t) @ self._M0

    # ---- everything that depends on (r, t) only ----
    def _geometry(self, r, t):
        key = (float(t), float(r[0]), float(r[1]), float(r[2]))
        if key == self._key:
            return self._geo
        self._ensure_evaluable()

        I3 = np.eye(3)
        rn = np.linalg.norm(r)
        if rn > self.r_far:
            # far field: point mass
            g_i = -self.gm * r / rn**3
            T_i = self.gm * (3.0 * np.outer(r, r) / rn**5 - I3 / rn**3)
        else:
            R = self.R_ib(t)
            r_b = R @ r
            _, g_b, T_b = self._evaluable(r_b, parallel=self.parallel)
            g_b = np.asarray(g_b, dtype=float)
            Txx, Tyy, Tzz, Txy, Txz, Tyz = T_b
            T_b = np.array(
                [[Txx, Txy, Txz], [Txy, Tyy, Tyz], [Txz, Tyz, Tzz]], dtype=float
            )
            g_i = R.T @ g_b
            T_i = R.T @ T_b @ R

        # Sun third body
        r_s = np.asarray(self.sun_pos_spline(t), dtype=float)
        d = r_s - r
        dn = np.linalg.norm(d)
        rsn = np.linalg.norm(r_s)
        a_3b = self.mu_sun * (d / dn**3 - r_s / rsn**3)
        G_3b = self.mu_sun * (-I3 / dn**3 + 3.0 * np.outer(d, d) / dn**5)

        # SRP cannonball (per unit eta), u = Sun -> particle
        u = -d
        s = self.srp_k_1au * AU_KM**2
        a_srp1 = s * u / dn**3
        G_srp1 = s * (I3 / dn**3 - 3.0 * np.outer(u, u) / dn**5)

        self._key = key
        self._geo = (g_i, T_i, a_3b, G_3b, a_srp1, G_srp1)
        return self._geo

    # ---- STTPropagatorND interface ----
    def f_func(self, *args):
        X = np.asarray(args[:N_STATE], dtype=float)
        t = float(args[N_STATE])
        g_i, _, a_3b, _, a_srp1, _ = self._geometry(X[0:3], t)
        acc = g_i + a_3b + X[6] * a_srp1
        out = np.zeros(N_STATE)
        out[0:3] = X[3:6]
        out[3:6] = acc
        return out

    def A_func(self, *args):
        X = np.asarray(args[:N_STATE], dtype=float)
        t = float(args[N_STATE])
        g_i, T_i, _, G_3b, a_srp1, G_srp1 = self._geometry(X[0:3], t)
        A = np.zeros((N_STATE, N_STATE))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = T_i + G_3b + X[6] * G_srp1
        A[3:6, 6] = a_srp1
        return A


def check_A_against_fd(dyn, x, t, rel_step=1e-6):
    """Central finite differences of f vs analytic A. Returns max relative column error."""
    x = np.asarray(x, dtype=float)
    A = dyn.A_func(*x, t)
    A_fd = np.zeros_like(A)
    for j in range(N_STATE):
        h = rel_step * max(abs(x[j]), 1e-3 if j < 3 else (1e-7 if j < 6 else 1e-2))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        A_fd[:, j] = (dyn.f_func(*xp, t) - dyn.f_func(*xm, t)) / (2 * h)
    err = np.linalg.norm(A - A_fd, axis=0) / np.maximum(np.linalg.norm(A_fd, axis=0), 1e-300)
    return err


# ============================================================
# Measurement generation (RA/DEC from S/C to particle)
# ============================================================


def generate_opnav_measurements_from_sc(
    x_part, sc_state, sigma_ra, sigma_dec, rng, add_outliers=False
):
    """
    x_part: (N,n) particle in inertial
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

    # RA: use raw atan2, NO modulo
    ra = np.arctan2(y, x)  # (-pi, pi]
    denom_ra = np.maximum(rxy2, 1e-30)
    d_ra_dx = -y / denom_ra
    d_ra_dy = x / denom_ra
    d_ra_dz = 0.0 * z

    # DEC = atan2(z, rxy)
    dec = np.arctan2(z, rxy)
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
    """
    Generic kernels + Ryugu SPK.

    The Ryugu SPK was generated with JPL Horizons (162173 Ryugu, 2030-2040), e.g.:
      https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='DES=2162173;'
          &EPHEM_TYPE=SPK&START_TIME='2030-01-01'&STOP_TIME='2040-01-01'
      (base64-decode the "spk" field into kernels/ryugu/ryugu_2162173_2030_2040.bsp)
    """
    kernels = [
        str(kernel_root / "naif0012.tls"),
        str(kernel_root / "de424.bsp"),
        str(kernel_root / "gm_de440.tpc"),
        str(kernel_root / "ryugu" / "ryugu_2162173_2030_2040.bsp"),
    ]

    for k in kernels:
        if not os.path.isfile(k):
            raise FileNotFoundError(f"Kernel not found: {k}")
        safe_furnsh(k)

    return kernels


def build_sun_spline(et0, tau_end, body_id, frame="J2000", abcorr="NONE", dt=60.0, pad=1800.0):
    """Cubic spline tau -> Sun position wrt the body [km] (so the dynamics need no SPICE)."""
    tau_grid = np.arange(-pad, tau_end + pad + dt, dt)
    r_sun = np.array(
        [spice.spkpos("SUN", float(et0 + tg), frame, abcorr, body_id)[0] for tg in tau_grid]
    )
    return CubicSpline(tau_grid, r_sun, axis=0)


def sun_rtn_frame(et0, body_id, frame="J2000", abcorr="NONE"):
    """
    Sun-RTN frame of the body at et0 (columns of C_rtn2i):
      R = Sun -> body (anti-Sun), N = heliocentric orbit normal, T = N x R.
    """
    st, _ = spice.spkezr(body_id, float(et0), frame, abcorr, "SUN")
    r_h, v_h = np.asarray(st[:3]), np.asarray(st[3:])
    R = r_h / np.linalg.norm(r_h)
    N = np.cross(r_h, v_h)
    N /= np.linalg.norm(N)
    T = np.cross(N, R)
    return np.column_stack([R, T, N]), np.linalg.norm(r_h)


# ============================================================
# PLOTS
# ============================================================
# Every figure is saved as <name>.pdf in the folder set by set_figure_dir() (main sets
# results/ryugu_poly_<timestamp>/). Colors, units and symbols come from the style block at
# the top of the file, so the same quantity looks the same in every figure.

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_FIG_DIR = Path("results")


def set_figure_dir(path):
    """Folder where save_figure() writes the PDFs (created if needed)."""
    global _FIG_DIR
    _FIG_DIR = Path(path)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    return _FIG_DIR


def save_figure(fig, name):
    """Save `fig` as <figure dir>/<name>.pdf, show it if SHOW_FIGS, then close it."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    fname = _FIG_DIR / f"{name}.pdf"
    fig.savefig(fname)
    print(f"[Figure] saved {fname}")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)
    return fname


def param_label(i, unit=True):
    """Axis label of estimated parameter i, e.g. '$\\delta x_0$ [m]'."""
    s = f"${PARAM_SYM[i]}$"
    return f"{s} [{PARAM_UNIT[i]}]" if unit else s


def _tint(color, a):
    """Mix `color` with white: a = 0 -> white, a = 1 -> color."""
    c = np.asarray(mcolors.to_rgb(color))
    return tuple((1.0 - a) + a * c)


OCC_FILL = _tint(C_OCC, 0.2)  # opaque fill of the occulted epochs


def _occulted_spans(t, vis):
    """(start, end) of each run of occulted epochs, edges at the midpoints between epochs."""
    t = np.asarray(t, dtype=float)
    occ = ~np.asarray(vis, dtype=bool)
    if not np.any(occ):
        return []
    dt = np.diff(t)
    left = np.r_[t[0] - 0.5 * dt[0], t[1:] - 0.5 * dt]
    right = np.r_[t[:-1] + 0.5 * dt, t[-1] + 0.5 * dt[-1]]
    edges = np.diff(np.r_[0, occ.astype(int), 0])
    i0 = np.flatnonzero(edges == 1)
    i1 = np.flatnonzero(edges == -1) - 1
    return [(left[a], right[b]) for a, b in zip(i0, i1)]


def shade_occulted(ax, t, vis, text=None, text_y=0.97):
    """Shade the occulted epochs of a time axis (optionally labelled inside the first span)."""
    spans = _occulted_spans(t, vis)
    for a, b in spans:
        ax.axvspan(a, b, color=OCC_FILL, lw=0, zorder=0.4)
    if text and spans:
        a, b = spans[0]
        lo, hi = ax.get_xlim()
        ax.text(
            0.5 * (max(a, lo) + min(b, hi)), text_y, text, color="0.3", fontsize=7,
            ha="center", va="top" if text_y > 0.5 else "bottom",
            transform=ax.get_xaxis_transform(), zorder=5,
        )
    return spans


# ------------------------------------------------------------
# Scene (body-fixed frame)
# ------------------------------------------------------------


def _view_dir(elev, azim):
    """Unit vector from the scene toward the camera for ax.view_init(elev, azim)."""
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _shaded_mesh(mesh, view, light, base=(0.74, 0.71, 0.67), ambient=0.30):
    """Lambert-shaded Poly3DCollection of the front-facing triangles of `mesh`."""
    nrm = np.asarray(mesh.face_normals)
    front = nrm @ view > 0.0  # closed surface: back faces are never visible
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[front]]
    shade = ambient + (1.0 - ambient) * np.clip(nrm[front] @ light, 0.0, None)
    fc = np.clip(np.outer(shade, base), 0.0, 1.0)
    coll = Poly3DCollection(
        tri, facecolors=fc, edgecolors=fc, linewidths=0.15, antialiased=False, zorder=1
    )
    coll.set_rasterized(True)
    return coll


def _hidden_by_mesh(mesh, pts, view, eps=1e-6):
    """True where a point is hidden from the (orthographic) camera by the mesh."""
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    dirs = np.repeat(view[None, :], pts.shape[0], axis=0)
    return np.asarray(mesh.ray.intersects_any(pts + eps * view, dirs), dtype=bool)


def _to_body(pos_i, tau, alpha, delta, omega, w0=0.0):
    """Rotate inertial positions (N,3) at times tau into the body-fixed frame."""
    return np.array(
        [
            np.asarray(make_ryugu_rotation_matrix(alpha, delta, omega, float(t), w0)) @ p
            for p, t in zip(np.asarray(pos_i, dtype=float), np.asarray(tau, dtype=float))
        ]
    )


def plot_scene_body_fixed(
    ryugu_mesh,
    tau_obs,
    sc_obs,
    vis_mask,
    tau_line,
    sc_line,
    pt_line,
    alpha,
    delta,
    omega,
    w0=0.0,
    detach_bf=None,
    elev=14.0,
    azim=62.0,
    zoom=1.0,
    name="fig_scene_body_fixed",
):
    """
    Observation geometry in the Ryugu body-fixed frame: shaded shape model, spacecraft and
    particle (truth) tracks over the observation arc, and the image epochs on the S/C track
    (used / occulted by the shape model). Orthographic view; track points hidden behind the
    body are not drawn. Inputs are inertial positions [km] at times tau [s].
    """
    view = _view_dir(elev, azim)
    light = _view_dir(elev + 30.0, azim + 35.0)  # key light above-left of the camera

    sc_b = _to_body(sc_line[:, :3], tau_line, alpha, delta, omega, w0)
    pt_b = _to_body(pt_line[:, :3], tau_line, alpha, delta, omega, w0)
    obs_b = _to_body(sc_obs[:, :3], tau_obs, alpha, delta, omega, w0)
    vis = np.asarray(vis_mask, dtype=bool)

    def _visible_line(xyz):
        out = xyz.copy()
        out[_hidden_by_mesh(ryugu_mesh, xyz, view)] = np.nan
        return out

    sc_v, pt_v = _visible_line(sc_b), _visible_line(pt_b)
    obs_hidden = _hidden_by_mesh(ryugu_mesh, obs_b, view)

    fig = plt.figure(figsize=(COL_W, 2.9))
    ax = fig.add_axes((0.0, -0.03, 1.0, 0.96), projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)

    ax.add_collection3d(_shaded_mesh(ryugu_mesh, view, light))
    ax.plot(*sc_v.T, color=C_SC, lw=1.1, zorder=3)
    ax.plot(*pt_v.T, color=C_TRUTH, lw=1.1, zorder=3)

    used = vis & ~obs_hidden
    occl = ~vis & ~obs_hidden
    ax.plot(
        *obs_b[used].T, ls="none", marker="o", ms=2.8, mfc=C_SC, mec="white", mew=0.35,
        zorder=4,
    )
    ax.plot(*obs_b[occl].T, ls="none", marker="x", ms=3.2, color=C_OCC, mew=0.9, zorder=4)
    if detach_bf is not None and not _hidden_by_mesh(ryugu_mesh, detach_bf[None], view)[0]:
        ax.plot(
            *np.asarray(detach_bf)[:, None], ls="none", marker="*", ms=7.5, mfc=C_TRUTH,
            mec="white", mew=0.5, zorder=5,
        )

    # Equal-aspect cube around everything drawn
    xyz = np.vstack([np.asarray(ryugu_mesh.vertices), sc_b, pt_b])
    mid = 0.5 * (xyz.min(0) + xyz.max(0))
    half = 0.5 * np.max(xyz.max(0) - xyz.min(0)) * 1.02
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)

    for a, lab in zip(
        (ax.xaxis, ax.yaxis, ax.zaxis),
        (r"$X_{\mathscr{B}}$ [km]", r"$Y_{\mathscr{B}}$ [km]", r"$Z_{\mathscr{B}}$ [km]"),
    ):
        a.set_pane_color((1.0, 1.0, 1.0, 0.0))
        a.line.set_color("0.4")
        a.line.set_linewidth(0.5)
        a._axinfo["grid"].update(color="0.88", linewidth=0.4)
        a._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.25)
        a.set_major_locator(mticker.MaxNLocator(5))
        a.set_tick_params(pad=-1.5, labelsize=6.5)
        a.set_label_text(lab)
        a.labelpad = -3.0

    handles = [
        Line2D([], [], color=C_SC, lw=1.1, label="Spacecraft"),
        Line2D([], [], color=C_TRUTH, lw=1.1, label="Particle (truth)"),
        Line2D([], [], ls="none", marker="o", ms=3.5, mfc=C_SC, mec="white", mew=0.35,
               label="Image used"),
        Line2D([], [], ls="none", marker="x", ms=3.8, color=C_OCC, mew=0.9,
               label="Image occulted"),
    ]
    if detach_bf is not None:
        handles.append(
            Line2D([], [], ls="none", marker="*", ms=7, mfc=C_TRUTH, mec="white", mew=0.5,
                   label="Ejection site")
        )
    fig.legend(
        handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.0, handletextpad=0.4,
    )
    return save_figure(fig, name)


# ------------------------------------------------------------
# Observation geometry and post-fit residuals (time axis)
# ------------------------------------------------------------


def plot_observation_geometry(
    tau_obs, sc_obs, pt_obs, vis_mask, tau_line=None, sc_line=None, pt_line=None,
    name="fig_observation_geometry",
):
    """S/C-particle range over the arc; image epochs marked used / occulted (shaded)."""
    t = np.asarray(tau_obs) / 3600.0
    vis = np.asarray(vis_mask, dtype=bool)
    rho = np.linalg.norm(np.asarray(pt_obs)[:, :3] - np.asarray(sc_obs)[:, :3], axis=1)

    fig, ax = plt.subplots(figsize=(COL_W, 1.95), layout="constrained")
    ax.set_xlim(t[0], t[-1])
    shade_occulted(ax, t, vis, text="occulted\nby Ryugu", text_y=0.04)
    if tau_line is not None:
        rho_l = np.linalg.norm(np.asarray(pt_line)[:, :3] - np.asarray(sc_line)[:, :3], axis=1)
        ax.plot(np.asarray(tau_line) / 3600.0, rho_l, color="0.25", lw=0.9, zorder=2)
    ax.plot(
        t[vis], rho[vis], ls="none", marker="o", ms=3.4, mfc=C_SC, mec="white", mew=0.4,
        zorder=3,
    )
    ax.plot(t[~vis], rho[~vis], ls="none", marker="x", ms=3.6, color=C_OCC, mew=1.0, zorder=3)
    ax.set_xlabel("Time since epoch [h]")
    ax.set_ylabel(r"S/C--particle range [km]")
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    handles = [
        Line2D([], [], color="0.25", lw=0.9, label="Range (truth)"),
        Line2D([], [], ls="none", marker="o", ms=3.4, mfc=C_SC, mec="white", mew=0.4,
               label=f"Image used ({int(vis.sum())})"),
        (
            Patch(facecolor=OCC_FILL, lw=0),
            Line2D([], [], ls="none", marker="x", ms=3.6, color=C_OCC, mew=1.0),
        ),
    ]
    labels = [h.get_label() for h in handles[:2]] + [f"Image occulted ({int((~vis).sum())})"]
    ax.legend(handles, labels, loc="lower right", ncol=1)
    return save_figure(fig, name)


def plot_postfit_residuals(
    tau_obs, res_batch, res_mcmc, vis_mask, name="fig_postfit_residuals"
):
    """
    Normalized RA / Dec post-fit residuals at the visible epochs for the Gauss-Newton batch
    (open circles) and the MCMC MAP (dots). Inputs are stacked [ra0, dec0, ra1, ...] / sigma.
    """
    t = np.asarray(tau_obs) / 3600.0
    vis = np.asarray(vis_mask, dtype=bool)
    rb = np.asarray(res_batch).reshape(-1, 2)
    rm = np.asarray(res_mcmc).reshape(-1, 2)

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True, layout="constrained")
    lim = max(3.6, 1.15 * float(np.max(np.abs(np.r_[rb[vis].ravel(), rm[vis].ravel()]))))
    for k, (ax, comp) in enumerate(zip(axes, ("RA", "Dec"))):
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-lim, lim)
        ax.axhspan(-3.0, 3.0, color="0.95", lw=0, zorder=0)
        ax.axhspan(-1.0, 1.0, color="0.89", lw=0, zorder=0)
        ax.axhline(0.0, color="0.45", lw=0.6, zorder=1)
        shade_occulted(ax, t, vis, text="occulted" if k == 0 else None)
        ax.plot(
            t[vis], rb[vis, k], ls="none", marker="o", ms=4.2, mfc="none", mec=C_BATCH,
            mew=0.8, zorder=3,
        )
        ax.plot(t[vis], rm[vis, k], ls="none", marker="o", ms=1.9, color=C_POST, zorder=4)
        ax.set_yticks([-3, -1, 0, 1, 3])
        ax.yaxis.grid(False)
        ax.set_ylabel(rf"{comp} residual [$\sigma$]")
        rms_b = np.sqrt(np.mean(rb[vis, k] ** 2))
        rms_m = np.sqrt(np.mean(rm[vis, k] ** 2))
        ax.text(
            0.99, 0.04, rf"RMS: batch {rms_b:.2f}, MCMC {rms_m:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="0.2", zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85),
        )
    axes[1].set_xlabel("Time since epoch [h]")

    handles = [
        Line2D([], [], ls="none", marker="o", ms=4.2, mfc="none", mec=C_BATCH, mew=0.8,
               label="Gauss--Newton batch"),
        Line2D([], [], ls="none", marker="o", ms=2.4, color=C_POST, label="MCMC MAP"),
        Patch(facecolor="0.89", label=r"$\pm1\sigma$"),
        Patch(facecolor="0.95", label=r"$\pm3\sigma$"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, columnspacing=0.9)
    return save_figure(fig, name)


# ------------------------------------------------------------
# MCMC diagnostics (chain traces, autocorrelation, corner)
# ------------------------------------------------------------


def _chain_physical(model, discard=0, thin=1):
    """Full emcee chain (n_steps, n_walkers, ndim) in physical (unwhitened) coordinates."""
    ch = np.asarray(model.sampler.get_chain(discard=discard, thin=thin), dtype=float)
    if getattr(model, "is_whitened", False):
        ch = ch @ np.asarray(model.whiten_L).T + np.asarray(model.whiten_mean)
    return ch


def plot_mcmc_traces(
    model, burn_in, true_theta=None, batch_theta=None, n_walkers_show=24, max_points=1500,
    seed=0, name="fig_mcmc_traces",
):
    """
    Walker traces of every parameter and of the log-posterior over the full run: a random
    subset of walkers (grey), the ensemble median with its 16-84% band (blue), the truth
    (black) and the batch estimate (vermillion). The burn-in is shaded. Y-limits follow the
    post-burn-in ensemble, so the initial transient enters from outside the panel.
    """
    n_steps = model.sampler.get_chain().shape[0]
    stride = max(1, n_steps // max_points)
    ch = _chain_physical(model, thin=stride) * PARAM_SCALE  # display units
    lp = np.asarray(model.sampler.get_log_prob(thin=stride), dtype=float)
    steps = np.arange(ch.shape[0]) * stride
    n_par = ch.shape[2]
    post = steps >= burn_in
    rng = np.random.default_rng(seed)
    pick = rng.choice(ch.shape[1], size=min(n_walkers_show, ch.shape[1]), replace=False)

    n_rows = int(np.ceil((n_par + 1) / 2))
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(TXT_W, 1.18 * n_rows + 0.45), sharex=True, layout="constrained"
    )
    axes = axes.ravel()

    def _panel(ax, y, ylabel, ref_true=None, ref_batch=None):
        ax.axvspan(0, burn_in, color=C_SPAN, lw=0, zorder=0)
        ax.plot(steps, y[:, pick], color="0.7", lw=0.25, alpha=0.5, zorder=1, rasterized=True)
        q16, q50, q84 = np.percentile(y, [16, 50, 84], axis=1)
        ax.fill_between(steps, q16, q84, color=C_POST, alpha=0.25, lw=0, zorder=2)
        ax.plot(steps, q50, color=C_POST, lw=0.9, zorder=3)
        if ref_batch is not None:
            ax.axhline(ref_batch, color=C_BATCH, lw=0.9, ls="--", zorder=4)
        if ref_true is not None:
            ax.axhline(ref_true, color=C_TRUTH, lw=0.8, zorder=5)
        yp = y[post]
        lo, hi = np.percentile(yp, [0.5, 99.5])
        refs = [r for r in (ref_true, ref_batch) if r is not None]
        lo, hi = min([lo] + refs), max([hi] + refs)
        pad = 0.35 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    for i in range(n_par):
        _panel(
            axes[i], ch[:, :, i], param_label(i),
            None if true_theta is None else true_theta[i] * PARAM_SCALE[i],
            None if batch_theta is None else batch_theta[i] * PARAM_SCALE[i],
        )
    _panel(axes[n_par], lp, r"$\log p(\bm{\theta}\,|\,\bm{y})$")
    for ax in axes[n_par + 1:]:
        ax.set_visible(False)

    for ax in axes[n_par - 1: n_par + 1]:
        ax.set_xlabel("Step")
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes:
        ax.set_xlim(0, n_steps)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))

    handles = [
        Line2D([], [], color="0.7", lw=0.8, label=f"Walkers ({len(pick)} of {ch.shape[1]})"),
        (
            Patch(facecolor=C_POST, alpha=0.25, lw=0),
            Line2D([], [], color=C_POST, lw=0.9),
        ),
        Line2D([], [], color=C_BATCH, lw=0.9, ls="--", label="Gauss--Newton batch"),
        Line2D([], [], color=C_TRUTH, lw=0.8, label="Truth"),
        Patch(facecolor=C_SPAN, lw=0, label="Burn-in"),
    ]
    labels = [handles[0].get_label(), "Ensemble median, 16--84\\%", handles[2].get_label(),
              handles[3].get_label(), handles[4].get_label()]
    fig.legend(handles, labels, loc="outside upper center", ncol=5)
    return save_figure(fig, name)


def plot_mcmc_autocorrelation(model, burn_in, thin, name="fig_mcmc_autocorrelation"):
    """
    Walker-averaged normalized autocorrelation of every parameter after burn-in, with the
    integrated autocorrelation time (emcee) in the legend and the thinning interval marked.
    """
    import emcee

    ch = _chain_physical(model, discard=burn_in)  # (n, W, d)
    n, n_w, n_par = ch.shape
    tau = np.atleast_1d(emcee.autocorr.integrated_time(ch, tol=0))
    acf = np.array(
        [
            np.mean([emcee.autocorr.function_1d(ch[:, k, i]) for k in range(n_w)], axis=0)
            for i in range(n_par)
        ]
    )
    max_lag = int(min(n - 1, max(6.0 * np.max(tau), 2.5 * thin)))
    lag = np.arange(max_lag + 1)

    styles = [(C_XYZ[i % 3], "-" if i < 3 else "--") for i in range(6)] + [("0.2", "-.")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.35), layout="constrained")
    ax.axhline(0.0, color="0.45", lw=0.6)
    for i in range(n_par):
        c, ls = styles[i] if i < len(styles) else ("0.5", ":")
        ax.plot(
            lag, acf[i, : max_lag + 1], color=c, ls=ls, lw=1.1,
            label=rf"{param_label(i, unit=False)}: $\hat\tau = {tau[i]:.0f}$",
        )
    ax.axvline(thin, color="0.3", lw=0.7, ls=":")
    ax.text(
        thin, 0.55, " thinning", rotation=90, ha="right", va="bottom", fontsize=7,
        color="0.3", transform=ax.get_xaxis_transform(),
    )
    ax.set_xlim(0, max_lag)
    ax.set_ylim(-0.2, 1.02)
    ax.set_xlabel("Lag [steps]")
    ax.set_ylabel("Autocorrelation")
    ax.legend(loc="upper right", ncol=2, fontsize=6.5, columnspacing=0.8, handlelength=2.2)
    return save_figure(fig, name)


def _inner_ticks(lo, hi, margin=0.12):
    """2-3 'nice' ticks in [lo, hi], each at least `margin` (axis fraction) from the edges."""
    d = margin * (hi - lo)
    t = np.array([0.5 * (lo + hi)])
    for n in (3, 4, 5, 6):
        c = mticker.MaxNLocator(n).tick_values(lo, hi)
        c = c[(c >= lo + d) & (c <= hi - d)]
        if c.size >= 2:
            return c[:3] if c.size <= 3 else c[::2][:3]
        if c.size:
            t = c
    return t


def _hpd_levels(H, masses):
    """Density thresholds enclosing the given probability masses of a 2D histogram."""
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h)
    c /= c[-1]
    return np.array([h[min(np.searchsorted(c, m), h.size - 1)] for m in masses])


def plot_corner(
    samples, true_theta=None, batch_mean=None, batch_cov=None, bins=40, smooth=1.1,
    name="fig_corner",
):
    """
    Corner plot of the MCMC posterior (blue: 1/2/3-sigma highest-density regions, i.e.
    39.3 / 86.5 / 98.9% of the 2D mass) against the Gauss-Newton batch covariance
    (vermillion ellipses with the same masses) and the truth (black). Displayed in m, mm/s, %.

    The 2D densities are estimated in pair-whitened coordinates (histogram + Gaussian
    smoothing) and mapped back, so near-degenerate pairs (|rho| ~ 1) stay smooth.
    """
    from scipy.ndimage import gaussian_filter

    s = PARAM_SCALE
    X = np.asarray(samples, dtype=float) * s
    n_par = X.shape[1]
    tr = None if true_theta is None else np.asarray(true_theta) * s
    bm = None if batch_mean is None else np.asarray(batch_mean) * s
    bc = None if batch_cov is None else np.asarray(batch_cov) * np.outer(s, s)

    masses = np.array([0.393, 0.865, 0.989])
    fills = [_tint(C_POST, a) for a in (0.16, 0.38, 0.68)]

    # Axis ranges: posterior, batch +/-3.5 sigma, truth
    lo, hi = np.percentile(X, [0.05, 99.95], axis=0)
    if bm is not None and bc is not None:
        sb = np.sqrt(np.diag(bc))
        lo, hi = np.minimum(lo, bm - 3.5 * sb), np.maximum(hi, bm + 3.5 * sb)
    if tr is not None:
        lo, hi = np.minimum(lo, tr), np.maximum(hi, tr)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(n_par, n_par, figsize=(TXT_W, TXT_W))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.075, top=0.965, wspace=0.07,
                        hspace=0.07)

    for i in range(n_par):
        for j in range(n_par):
            ax = axes[i, j]
            ax.grid(False)
            if j > i:
                ax.set_visible(False)
                continue
            ax.set_xlim(lo[j], hi[j])
            ax.set_xticks(_inner_ticks(lo[j], hi[j]))

            if i == j:  # ---- marginal
                ax.hist(
                    X[:, i], bins=bins, range=(lo[i], hi[i]), density=True,
                    histtype="stepfilled", color=_tint(C_POST, 0.3), ec=C_POST, lw=0.8,
                )
                if bm is not None and bc is not None:
                    xx = np.linspace(lo[i], hi[i], 300)
                    ax.plot(xx, norm.pdf(xx, bm[i], np.sqrt(bc[i, i])), color=C_BATCH, lw=0.9)
                if tr is not None:
                    ax.axvline(tr[i], color=C_TRUTH, lw=0.8)
                ax.set_ylim(bottom=0.0)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                q16, q50, q84 = np.percentile(X[:, i], [16, 50, 84])
                em, ep = q50 - q16, q84 - q50
                dec = int(np.clip(1 - np.floor(np.log10(min(em, ep))), 0, 6))
                ax.set_title(
                    rf"${PARAM_SYM[i]} = {q50:.{dec}f}^{{+{ep:.{dec}f}}}_{{-{em:.{dec}f}}}$",
                    fontsize=6.5, pad=3,
                )
            else:  # ---- joint (x: param j, y: param i)
                ax.set_ylim(lo[i], hi[i])
                ax.set_yticks(_inner_ticks(lo[i], hi[i]))
                P2 = X[:, [j, i]]
                m2 = P2.mean(0)
                Lc = np.linalg.cholesky(np.cov(P2, rowvar=False))
                U = np.linalg.solve(Lc, (P2 - m2).T).T
                r = max(4.0, float(np.percentile(np.abs(U), 99.95))) * 1.05
                H, ex, ey = np.histogram2d(U[:, 0], U[:, 1], bins=bins, range=[[-r, r]] * 2)
                H = gaussian_filter(H, smooth)
                cu, cv = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1])
                GU, GV = np.meshgrid(cu, cv, indexing="ij")
                GX = m2[0] + Lc[0, 0] * GU
                GY = m2[1] + Lc[1, 0] * GU + Lc[1, 1] * GV
                lev = _hpd_levels(H, masses)  # descending in mass -> ascending in density
                lev = np.r_[np.sort(lev), H.max() * 1.0001]
                if np.all(np.diff(lev) > 0):
                    ax.contourf(GX, GY, H, levels=lev, colors=fills, zorder=1)
                    ax.contour(GX, GY, H, levels=lev[:-1], colors=C_POST, linewidths=0.45,
                               zorder=2)
                else:
                    ax.plot(P2[:, 0], P2[:, 1], ",", color=C_POST, alpha=0.2, zorder=1)
                if bm is not None and bc is not None:
                    C2 = bc[np.ix_([j, i], [j, i])]
                    for rad in (1.0, 2.0, 3.0):
                        e = _ellipse_2d(bm[[j, i]], C2, rad)
                        ax.plot(e[0], e[1], color=C_BATCH, lw=0.7, zorder=3)
                if tr is not None:
                    ax.plot(
                        tr[j], tr[i], ls="none", marker="*", ms=5.5, mfc=C_TRUTH, mec="white",
                        mew=0.4, zorder=4,
                    )

            # outer labels only
            if i == n_par - 1:
                ax.set_xlabel(param_label(j))
                ax.tick_params(axis="x", labelrotation=45, labelsize=6.5)
                for lab in ax.get_xticklabels():
                    lab.set_ha("right")
                    lab.set_rotation_mode("anchor")
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(param_label(i))
                ax.tick_params(axis="y", labelsize=6.5)
            elif i != j:
                ax.set_yticklabels([])
    fig.align_ylabels(axes[:, 0])
    fig.align_xlabels(axes[-1, :])

    handles = [
        (Patch(facecolor=fills[2], lw=0), Patch(facecolor=fills[1], lw=0),
         Patch(facecolor=fills[0], lw=0)),
        Line2D([], [], color=C_BATCH, lw=0.9),
        Line2D([], [], ls="none", marker="*", ms=7, mfc=C_TRUTH, mec="white", mew=0.4),
    ]
    fig.legend(
        handles,
        ["MCMC posterior", "Gauss--Newton batch (linear)", "Truth"],
        loc="upper right", bbox_to_anchor=(0.97, 0.93), fontsize=8,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, handlelength=2.6,
    )
    fig.text(
        0.97, 0.93 - 0.085, r"Contours: $1\sigma$, $2\sigma$, $3\sigma$ regions"
        "\n" r"(39.3, 86.5, 98.9\% of the 2D mass)",
        ha="right", va="top", fontsize=7, color="0.3",
    )
    return save_figure(fig, name)


# ============================================================
# Gauss-Newton batch solver with STMs
# ============================================================


def solve_stage1_gn_with_stm(
    propagator,
    x0_ref,
    tau,  # (N,) seconds since start (NOT ET)
    sc_state,  # (N,6) inertial, same epochs as tau
    y_obs,  # (2N,) stacked [ra0,dec0, ra1,dec1,...]
    sigma_ra,
    sigma_dec,
    obs_weights,  # visibility weights
    priors=None,  # list of scipy.stats.norm length n_update, OR None
    update_idx=None,  # indices of the delta you're solving for (default: all)
    max_iter=10,
    tol=1e-8,
    rtol=1e-12,
    atol=1e-12,
    method="LSODA",
    verbose=True,
    max_lm_tries=12,
):
    """
    Gauss-Newton outer loop. Each iteration:
    1) propagate at current x0_ref (state + STM)
    2) build residual vector r and Jacobian J using STM chain rule
    3) solve linearized MAP: min ||r - J d||^2 + ||(d - m)/s||^2
    4) update x0_ref += embed(d)

    The linear solves are done on columns scaled by the prior sigmas (the state spans
    ~20 orders of magnitude: km, km/s, km^3/s^2, -, km/s^2). Convergence is declared
    when the scaled step (in prior-sigma units) is below `tol`.

    Levenberg-Marquardt safeguard: the pure GN step is tried first; if it does not
    decrease the true (nonlinear) MAP cost, it is damped (lam * I in prior-sigma units)
    until it does. A low-altitude bound particle around Ryugu is strongly nonlinear over
    a few hours, and the undamped GN step diverges from a 20%-perturbed reference.
    """
    x0_ref = np.asarray(x0_ref, dtype=float).copy()
    n = x0_ref.size
    if update_idx is None:
        update_idx = np.arange(n)
    update_idx = np.asarray(update_idx, dtype=int)
    n_upd = len(update_idx)

    if priors is None:
        prior_mean = np.zeros(n_upd)
        prior_sig = np.full(n_upd, np.inf)
    else:
        prior_mean = np.array([p.mean() for p in priors], dtype=float)
        prior_sig = np.array([p.std() for p in priors], dtype=float)

    col_scale = np.where(np.isfinite(prior_sig), prior_sig, 1.0)

    delta_total = np.zeros(n)

    w = np.empty_like(y_obs, dtype=float)
    w[0::2] = sigma_ra
    w[1::2] = sigma_dec

    finite = np.isfinite(prior_sig)

    def prior_rows(delta_tot):
        return -(delta_tot[update_idx][finite] - prior_mean[finite]) / prior_sig[finite]

    def map_cost(x0_try, delta_tot_try):
        """True nonlinear MAP cost (state-only propagation)."""
        sol = propagator.propagate_state_only(
            x0=x0_try, t_eval=tau, show_progress=False, rtol=rtol, atol=atol, method=method
        )
        if sol.status != 0 or sol.y.shape[1] != len(tau):
            return np.inf
        x = sol.y[:n, :].T
        ra_m, dec_m, _, _ = radec_and_partials_from_los(x[:, :3] - sc_state[:, :3])
        res_t = np.empty_like(y_obs)
        res_t[0::2] = wrap_to_pi(y_obs[0::2] - ra_m)
        res_t[1::2] = y_obs[1::2] - dec_m
        c = np.sum((res_t / w * obs_weights) ** 2)
        if np.any(finite):
            c += np.sum(prior_rows(delta_tot_try) ** 2)
        return float(c) if np.isfinite(c) else np.inf

    A_s = None
    lam = 0.0
    cost_prev = None
    for it in range(1, max_iter + 1):
        sol_ref, stts_ref = propagator.propagate(
            x0=x0_ref, t_eval=tau, rtol=rtol, atol=atol, method=method
        )
        x_ref = sol_ref.y[:n, :].T  # (N,n)
        Phi_list = np.asarray(stts_ref[1], dtype=float).reshape(len(tau), n, n)

        los = x_ref[:, :3] - sc_state[:, :3]
        ra_model, dec_model, d_ra_dr, d_dec_dr = radec_and_partials_from_los(los)

        res = np.empty_like(y_obs)
        res[0::2] = wrap_to_pi(y_obs[0::2] - ra_model)
        res[1::2] = y_obs[1::2] - dec_model

        r = res / w * obs_weights

        # Jacobian wrt update variables (2N x n_upd); dy/dx_k only depends on position
        J = np.zeros((2 * len(tau), n_upd), dtype=float)
        for k in range(len(tau)):
            Hy = np.zeros((2, n), dtype=float)
            Hy[0, 0:3] = d_ra_dr[k, :]
            Hy[1, 0:3] = d_dec_dr[k, :]
            Hx0 = Hy @ Phi_list[k]
            J[2 * k : 2 * k + 2, :] = Hx0[:, update_idx]

        J[0::2, :] /= sigma_ra
        J[1::2, :] /= sigma_dec
        J = J * obs_weights[:, None]

        # linearized MAP solve: (J; W_prior) d = (r; r_prior)
        rows = [J]
        rhs = [r]

        if np.any(finite):
            Wp = np.diag(1.0 / prior_sig[finite])
            Jp = np.zeros((Wp.shape[0], n_upd))
            Jp[:, finite] = Wp

            # anchor to initial reference using the accumulated delta
            rp = prior_rows(delta_total)
            rows.append(Jp)
            rhs.append(rp)

        A = np.vstack(rows)
        b = np.hstack(rhs)

        A_s = A * col_scale[None, :]
        cost_cur = float(np.sum(b**2)) if cost_prev is None else cost_prev
        H_diag_max = float(np.max(np.sum(A_s**2, axis=0)))

        accepted = False
        converged = False
        for _ in range(max_lm_tries):
            if lam > 0.0:
                A_aug = np.vstack([A_s, np.sqrt(lam) * np.eye(n_upd)])
                b_aug = np.hstack([b, np.zeros(n_upd)])
            else:
                A_aug, b_aug = A_s, b
            d_s, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            d_upd = col_scale * d_s
            d_full = np.zeros(n)
            d_full[update_idx] = d_upd

            if lam == 0.0 and np.linalg.norm(d_s) < tol:
                # converged: tiny GN step, accept without a (noise-level) cost check
                accepted = converged = True
                cost_try = None
                break

            cost_try = map_cost(x0_ref + d_full, delta_total + d_full)
            if cost_try < cost_cur:
                accepted = True
                break
            lam = 1e-3 * H_diag_max if lam == 0.0 else 10.0 * lam

        if not accepted:
            print("\n[GN] Levenberg-Marquardt could not decrease the MAP cost: stopping.")
            break

        x0_ref = x0_ref + d_full
        delta_total = delta_total + d_full
        cost_prev = cost_try

        step_norm = np.linalg.norm(d_upd)
        step_scaled = np.linalg.norm(d_s)
        rms = np.sqrt(np.sum(r**2) / max(np.count_nonzero(r), 1))

        if verbose:
            cost_str = f"{cost_try:.4e}" if cost_try is not None else "n/a"
            print(
                f"\n[GN it {it:02d}] rms(norm res, visible)={rms:.3e}  "
                f"step_norm={step_norm:.3e}  step(prior-sigma units)={step_scaled:.3e}  "
                f"MAP cost: {cost_cur:.4e} -> {cost_str}  (LM lam={lam:.1e})"
            )

        # relax the damping after a successful step
        lam = lam / 10.0 if lam > 1e-9 * H_diag_max else 0.0

        if converged or step_scaled < tol:
            break

    # Covariance of the update variables at convergence (scaled for conditioning)
    if max_iter == 0 or A_s is None:
        cov_upd = np.diag(prior_sig**2)
    else:
        AtA_s = A_s.T @ A_s
        try:
            cov_s = np.linalg.inv(AtA_s)
        except np.linalg.LinAlgError:
            print("[WARNING] Covariance matrix is singular, using pseudoinverse")
            cov_s = np.linalg.pinv(AtA_s)
        cov_upd = col_scale[:, None] * cov_s * col_scale[None, :]

    cov_full = np.full((n, n), np.inf)
    for i, idx_i in enumerate(update_idx):
        for j, idx_j in enumerate(update_idx):
            cov_full[idx_i, idx_j] = cov_upd[i, j]

    # (post-fit residuals of this solution are plotted in main, next to the MCMC MAP)
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
    """
    Full nonlinear batch (MAP) using SciPy least_squares (re-propagates every evaluation;
    slow with the polyhedron). Includes observation weights to handle occultation.
    """
    x0_ref = np.asarray(x0_ref, dtype=float)
    n = x0_ref.size
    if update_idx is None:
        update_idx = np.arange(n)
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

        sol = propagator.propagate_state_only(
            x0=x0, t_eval=tau, show_progress=False, rtol=rtol, atol=atol, method=method
        )
        x = sol.y[:n, :].T

        los = x[:, :3] - sc_state[:, :3]
        ra_model, dec_model, _, _ = radec_and_partials_from_los(los)

        res = np.empty_like(y_obs)
        res[0::2] = wrap_to_pi(y_obs[0::2] - ra_model)
        res[1::2] = y_obs[1::2] - dec_model

        r_meas = (res / w) * obs_weights

        if np.any(finite):
            r_pri = (delta_upd[finite] - prior_mean[finite]) / prior_sig[finite]
            return np.hstack([r_meas, r_pri])

        return r_meas

    result = least_squares(
        fun=residual_vector,
        x0=np.zeros(n_upd),
        method="trf",
        jac="2-point",
        x_scale=np.where(finite, prior_sig, 1.0),
        max_nfev=max_nfev,
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        verbose=verbose,
    )

    delta_hat_full = np.zeros(n)
    delta_hat_full[update_idx] = result.x
    x0_ref1 = x0_ref + delta_hat_full

    J = result.jac
    try:
        cov_upd = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov_upd = np.linalg.pinv(J.T @ J)

    cov_full = np.full((n, n), np.inf)
    for i, ii in enumerate(update_idx):
        for j, jj in enumerate(update_idx):
            cov_full[ii, jj] = cov_upd[i, j]

    return x0_ref1, delta_hat_full, cov_full, result


# ============================================================
# POSTERIOR SPLIT (GMM) + STM PROPAGATION TO A GIVEN EPOCH
# ============================================================

def fit_posterior_gmm(samples, n_components=8, select_bic=False, k_max=12, random_state=0):
    """
    Split posterior samples (N, n) into a full-covariance Gaussian mixture.

    The coordinates span several orders of magnitude (km, km/s, eta), so EM runs in whitened
    coordinates z = L^{-1}(theta - m), with L L^T the sample covariance (sklearn's
    reg_covar would otherwise swamp the small coordinates). Back-mapping:
        mu_k = m + L mu_k^z,     P_k = L P_k^z L^T.
    select_bic=True picks K in 1..k_max by BIC. On a near-Gaussian posterior BIC keeps very
    few components; a fixed K splits the posterior into narrower pieces, which is what keeps
    each component's STM linearization accurate over a long prediction.
    Components are returned sorted by decreasing weight.
    """
    X = np.asarray(samples, dtype=float)
    m = X.mean(axis=0)
    # L = D chol(corr): same as chol(cov) but well conditioned for mixed units
    L = X.std(axis=0)[:, None] * np.linalg.cholesky(np.corrcoef(X, rowvar=False))
    Z = np.linalg.solve(L, (X - m).T).T

    def _fit(k):
        return GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=3,
            max_iter=500,
            random_state=random_state,
        ).fit(Z)

    if select_bic:
        fits = [_fit(k) for k in range(1, k_max + 1)]
        bics = np.array([g.bic(Z) for g in fits])
        gmm = fits[int(np.argmin(bics))]
        print(f"[GMM] BIC for K = 1..{k_max}:", np.array2string(bics, precision=1))
    else:
        gmm = _fit(n_components)

    order = np.argsort(gmm.weights_)[::-1]
    weights = gmm.weights_[order]
    means = m + gmm.means_[order] @ L.T
    covs = np.einsum("ij,kjl,ml->kim", L, gmm.covariances_[order], L)
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))
    return weights, means, covs


def gmm_moments(weights, means, covs):
    """Mean and covariance of a Gaussian mixture."""
    m = weights @ means
    d = means - m
    P = np.einsum("k,kij->ij", weights, covs) + np.einsum("k,ki,kj->ij", weights, d, d)
    return m, P


def gmm_logpdf(x, weights, means, covs):
    """log sum_k w_k N(x; mu_k, P_k) for x of shape (M, d)."""
    x = np.atleast_2d(x)
    lp = np.array(
        [
            np.log(w)
            + np.atleast_1d(multivariate_normal(mu, P, allow_singular=True).logpdf(x))
            for w, mu, P in zip(weights, means, covs)
        ]
    )
    return logsumexp(lp, axis=0)


def propagate_gmm_stm(propagator, x0_ref, weights, means, covs, t_eval, label="GMM", **ivp):
    """
    Propagate every mixture component with its OWN first-order STM:
        m_k(t) = phi(t; x0_ref + mu_k),      P_k(t) = Phi_k(t) P_k Phi_k(t)^T,
    weights unchanged (no re-splitting along the way).
    Returns m (K, N_t, n) [absolute states] and P (K, N_t, n, n).
    """
    t_eval = np.asarray(t_eval, dtype=float)
    K, n = means.shape
    m_t = np.empty((K, t_eval.size, n))
    P_t = np.empty((K, t_eval.size, n, n))
    for k in range(K):
        sol, stts = propagator.propagate(
            x0_ref + means[k], t_eval, show_progress=False, **ivp
        )
        if sol.status != 0 or sol.y.shape[1] != t_eval.size:
            raise RuntimeError(f"[{label}] component {k} propagation failed: {sol.message}")
        Phi = stts[1]  # (N_t, n, n)
        m_t[k] = sol.y[:n].T
        P_t[k] = Phi @ covs[k] @ np.transpose(Phi, (0, 2, 1))
        print(f"[{label}] component {k + 1}/{K} (w = {weights[k]:.3f}) propagated", flush=True)
    return m_t, P_t


def propagate_samples_mc(propagator, x0_ref, deltas, t_eval, **ivp):
    """Nonlinear (state-only) propagation of x0_ref + delta_i. Failed runs are left NaN."""
    t_eval = np.asarray(t_eval, dtype=float)
    M, n = deltas.shape
    out = np.full((M, t_eval.size, n), np.nan)
    step = max(1, M // 10)
    for i, d in enumerate(deltas):
        sol = propagator.propagate_state_only(
            x0_ref + d, t_eval, show_progress=False, **ivp
        )
        if sol.status == 0 and sol.y.shape[1] == t_eval.size:
            out[i] = sol.y[:n].T
        if (i + 1) % step == 0 or i + 1 == M:
            print(f"[MC] {i + 1}/{M} posterior samples propagated", flush=True)
    return out


def position_prediction_history(X_mc, w_g, m_g, P_g, m_1, P_1):
    """
    Per output epoch, position-only (3D) comparison against the Monte Carlo cloud:
      sig_*: sqrt(trace P_rr) [km] for MC, GMM (moment-matched) and single Gaussian
      ll_*:  mean log-density of the MC positions under the GMM / single Gaussian
    """
    n_t = X_mc.shape[1]
    out = {k: np.empty(n_t) for k in ("sig_mc", "sig_gmm", "sig_1", "ll_gmm", "ll_1")}
    for j in range(n_t):
        pos = X_mc[:, j, :3]
        _, Pg = gmm_moments(w_g, m_g[:, j, :3], P_g[:, j, :3, :3])
        out["sig_mc"][j] = np.sqrt(np.trace(np.cov(pos, rowvar=False)))
        out["sig_gmm"][j] = np.sqrt(np.trace(Pg))
        out["sig_1"][j] = np.sqrt(np.trace(P_1[0, j, :3, :3]))
        out["ll_gmm"][j] = np.mean(gmm_logpdf(pos, w_g, m_g[:, j, :3], P_g[:, j, :3, :3]))
        out["ll_1"][j] = np.mean(gmm_logpdf(pos, np.ones(1), m_1[:, j, :3], P_1[:, j, :3, :3]))
    return out


def _ellipse_2d(mean2, cov2, radius, n=181):
    """Boundary {x : (x - m)^T C^{-1} (x - m) = radius^2}, shape (2, n)."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    vals, vecs = np.linalg.eigh(cov2)
    circ = np.vstack([np.cos(t), np.sin(t)])
    return mean2[:, None] + radius * (vecs @ (np.sqrt(np.clip(vals, 0.0, None))[:, None] * circ))


def plot_gmm_propagation(
    panels, prob=0.99, n_draw=20000, seed=0, name="fig_gmm_stm_propagation"
):
    """
    2x3 position projections (row 0: epoch, row 1: target). Each panel dict holds (km):
      mc (M,3), w (K,), mu (K,3), P (K,3,3), mu1 (3,), P1 (3,3), truth (3,), center (3,), title.
    Drawn in meters relative to `center` (the MC mean): the MC cloud, every GMM component's
    ellipse, the GMM highest-density region holding `prob` of the mass (level from mixture
    draws), the single-Gaussian ellipse with the same mass, and the truth.
    """
    r_ell = np.sqrt(chi2.ppf(prob, 2))
    pct = f"{100 * prob:g}\\%"
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = [r"$\Delta x$ [m]", r"$\Delta y$ [m]", r"$\Delta z$ [m]"]
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(TXT_W, 4.9), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, hspace=0.04, wspace=0.04)
    sub = fig.subfigures(len(panels), 1)
    axes = np.array([sf.subplots(1, 3) for sf in np.atleast_1d(sub)])
    for row, pnl in enumerate(panels):
        c = np.asarray(pnl["center"], dtype=float)
        mc = (pnl["mc"] - c) * 1e3
        w = np.asarray(pnl["w"], dtype=float)
        mu = (pnl["mu"] - c) * 1e3
        P = pnl["P"] * 1e6
        mu1 = (pnl["mu1"] - c) * 1e3
        P1 = pnl["P1"] * 1e6
        tr = (pnl["truth"] - c) * 1e3

        counts = rng.multinomial(n_draw, w / w.sum())
        draws = np.vstack(
            [rng.multivariate_normal(mu[k], P[k], size=counts[k]) for k in range(len(w))]
        )

        for col, (i, j) in enumerate(pairs):
            ax = axes[row, col]
            ix = [i, j]
            mu2, P2 = mu[:, ix], P[:, ix][:, :, ix]
            e1 = _ellipse_2d(mu1[ix], P1[np.ix_(ix, ix)], r_ell)

            lo = np.min(
                [mc[:, ix].min(0), e1.min(1), np.percentile(draws[:, ix], 0.1, axis=0), tr[ix]],
                axis=0,
            )
            hi = np.max(
                [mc[:, ix].max(0), e1.max(1), np.percentile(draws[:, ix], 99.9, axis=0), tr[ix]],
                axis=0,
            )
            pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad

            ax.scatter(
                mc[:, i], mc[:, j], s=2.5, color=C_MC, alpha=0.7, lw=0, rasterized=True,
                zorder=1,
            )
            for k in range(len(w)):
                e = _ellipse_2d(mu2[k], P2[k], r_ell)
                ax.plot(e[0], e[1], color=C_POST, lw=0.45, alpha=0.55, zorder=2)

            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 240), np.linspace(lo[1], hi[1], 240))
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            dens = np.exp(gmm_logpdf(grid, w, mu2, P2)).reshape(gx.shape)
            level = np.quantile(np.exp(gmm_logpdf(draws[:, ix], w, mu2, P2)), 1.0 - prob)
            ax.contour(gx, gy, dens, levels=[level], colors=C_POST, linewidths=1.3, zorder=3)

            ax.plot(e1[0], e1[1], color=C_BATCH, lw=1.1, ls="--", zorder=4)
            ax.plot(
                tr[i], tr[j], ls="none", marker="*", ms=7.5, mfc=C_TRUTH, mec="white",
                mew=0.5, zorder=5,
            )
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.set_xlabel(names[i])
            ax.set_ylabel(names[j])
            ax.set_title(f"({'abcdef'[3 * row + col]})", loc="left")
        np.atleast_1d(sub)[row].suptitle(pnl["title"], fontsize=8)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=3, color=C_MC, label="Monte Carlo (nonlinear)"),
        Line2D([], [], color=C_POST, lw=0.45, alpha=0.7, label=f"GMM components ({pct})"),
        Line2D([], [], color=C_POST, lw=1.3, label=f"GMM + STMs ({pct} HDR)"),
        Line2D([], [], color=C_BATCH, lw=1.1, ls="--", label=f"Single Gaussian + STM ({pct})"),
        Line2D([], [], ls="none", marker="*", ms=7.5, mfc=C_TRUTH, mec="white", mew=0.5,
               label="Truth"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, columnspacing=1.0)
    return save_figure(fig, name)


def plot_prediction_history(t_hr, hist, t_obs_end_hr, name="fig_gmm_prediction_history"):
    """Position uncertainty and MC log-density (GMM vs single Gaussian) vs time."""
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(COL_W, 3.3), sharex=True, layout="constrained",
        gridspec_kw=dict(height_ratios=(1.35, 1.0)),
    )
    for ax in (ax0, ax1):
        ax.axvspan(t_hr[0], t_obs_end_hr, color=C_SPAN, lw=0, zorder=0)
        ax.set_xlim(t_hr[0], t_hr[-1])
    ax0.text(
        0.5 * (t_hr[0] + t_obs_end_hr), 0.96, "observation arc", color="0.3", fontsize=7,
        ha="center", va="top", transform=ax0.get_xaxis_transform(),
    )

    ax0.plot(t_hr, hist["sig_mc"] * 1e3, color="0.78", lw=3.2, solid_capstyle="round",
             label="Monte Carlo", zorder=2)
    ax0.plot(t_hr, hist["sig_gmm"] * 1e3, color=C_POST, lw=1.2, label="GMM + STMs", zorder=3)
    ax0.plot(t_hr, hist["sig_1"] * 1e3, color=C_BATCH, lw=1.2, ls="--",
             label="Single Gaussian + STM", zorder=4)
    ax0.set_ylabel(r"$\sqrt{\mathrm{tr}\,\bm{P}_{rr}}$ [m]")
    ax0.set_yscale("log")
    ax0.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax0.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax0.set_title("(a) Position uncertainty", loc="left")

    ax1.axhline(0.0, color="0.45", lw=0.6)
    ax1.plot(t_hr, hist["ll_gmm"] - hist["ll_1"], color=C_POST, lw=1.2)
    ax1.set_ylabel(r"$\Delta\,\overline{\log p}$ [nats]")
    ax1.set_xlabel("Time since epoch [h]")
    ax1.set_title(
        r"(b) Mean log-density of MC positions, GMM $-$ single", loc="left"
    )

    fig.legend(loc="outside upper center", ncol=3, columnspacing=0.8, handlelength=1.5)
    return save_figure(fig, name)


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    # --------------------------
    # USER SETTINGS (edit these)
    # --------------------------
    KERNEL_ROOT = Path("./kernels")
    SC_NAME = "CSR-30 (notional)"  # notional 2030s Japanese comet sample-return concept
    CENTER = "20162173"  # Ryugu (NAIF ID of the Horizons SPK)
    FRAME_I = "J2000"
    ABCORR = "NONE"

    # Observation window (must be covered by the Ryugu SPK: 2030-2040)
    utc0 = "2034-03-01T00:00:00"
    utc1 = "2034-03-01T03:00:00"
    n_obs = 37  # one image every 5 min

    # Ryugu physical (Watanabe et al. 2019, Science 364, 268)
    shape_path = "ObjFiles/Ryugu/SHAPE_SFM_49k_v20180804.obj"  # km, body-fixed
    R_ryugu = 0.448  # mean radius [km] (detach target + sphere fallback only)

    # Ryugu spin + pole truth (IAU-style positive pole; retrograde rotator)
    alpha_true = np.deg2rad(96.4)  # rad
    delta_true = np.deg2rad(-66.4)  # rad
    spin_period = 7.63262 * 3600.0  # s
    omega_true = 2 * np.pi / spin_period  # rad/s
    w0_true = 0.0

    # Dynamics: STMs only
    stt_order = 1

    # Ryugu bulk density (Watanabe et al. 2019: 1190 +/- 20 kg/m^3). With the shape model it
    # fixes the polyhedral gravity (GM = G rho V ~ 30 m^3/s^2), so GM is NOT in the state.
    ryugu_density = 1190.0  # kg/m^3

    # Truth dynamic parameters
    eta_true = 1.0  # SRP scale (truth); the reference / prior is centered elsewhere
    params_true = np.array([eta_true])

    # Particle SRP (cannonball): ~2 cm diameter grain, Ryugu-like bulk density
    part_radius_m = 0.01
    part_density = 1500.0  # kg/m^3
    part_Cr = 1.2
    part_A_over_m = 3.0 / (4.0 * part_density * part_radius_m)  # m^2/kg
    k_srp_part = part_Cr * part_A_over_m * P_SUN_1AU / 1000.0  # km/s^2 at 1 AU

    # Notional spacecraft (CSR-30): ~2 km terminator orbit, treated as known
    sc_Cr = 1.3
    sc_A_over_m = 0.02  # m^2/kg (~600 kg, ~12 m^2 incl. solar arrays)
    k_srp_sc = sc_Cr * sc_A_over_m * P_SUN_1AU / 1000.0  # km/s^2 at 1 AU
    a_sc = 2.0  # orbit radius [km]
    phi_sc0 = np.deg2rad(110.0)  # initial phase in the terminator plane

    # Measurement noise (notional wide-angle navcam)
    sigma_pix = 0.1  # pixel noise (1-sigma)
    fov_deg = 30.0  # full FOV [deg]
    n_pix = 2048  # pixels across FOV
    fov_rad = np.deg2rad(fov_deg)
    sigma_angle = sigma_pix * (fov_rad / n_pix)  # radians
    sigma_ra = sigma_angle
    sigma_dec = sigma_angle

    # reference is perturbed by these fractions of |truth| (or absolute values)
    rng_ref = np.random.default_rng(42)
    ref_pct_r = 0.2  # 20% of each position component
    ref_pct_v = 0.2  # 20% of each velocity component
    ref_sig_eta = 0.05  # absolute, on eta

    # priors (on deltas about the reference)
    sig_prior_r = np.full(3, 0.250)  # km
    sig_prior_v = np.full(3, 3.0e-4)  # km/s (~ surface escape speed)
    sig_prior_eta = 0.10  # 10% on SRP scale
    prior_looseness = 1

    # MCMC settings
    # NOTE: always do a run with burn_in and thin not activated
    n_walkers = 128
    n_samples = 20000
    burn_in = 2000
    thin = 100
    spherical_spread = 1e-1

    # Posterior split (GMM) + STM propagation to a given epoch
    # NOTE: the truth particle re-impacts Ryugu at ~3.99 h after utc0; keep the target before
    # that (the polyhedral field is not valid inside the body).
    utc_target = "2034-03-01T03:45:00"
    gmm_n_components = 8  # fixed K (narrow components keep each STM accurate)
    gmm_select_bic = False  # True: pick K in 1..12 by BIC instead
    n_mc_prop = 300  # posterior samples propagated nonlinearly (Monte Carlo check)
    n_t_prop = 46  # output epochs on [epoch, target]
    gmm_prob = 0.99  # probability mass of the plotted regions

    # Figures: one PDF per figure in a timestamped folder (SHOW_FIGS at the top of the file)
    fig_dir = set_figure_dir(
        Path(__file__).resolve().parent / "results"
        / f"ryugu_poly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"[Figures] -> {fig_dir}")

    # --------------------------
    # Load SPICE (ET) + time grid
    # --------------------------
    _ = load_kernels(KERNEL_ROOT)

    et0 = spice.utc2et(utc0)
    et1 = spice.utc2et(utc1)
    ets_full = np.linspace(et0, et1, n_obs)  # ET (for SPICE)
    tau_full = ets_full - ets_full[0]  # seconds since start (for dynamics)
    tau_target = spice.utc2et(utc_target) - et0  # seconds since start
    if tau_target <= 0.0:
        raise ValueError("utc_target must be after utc0.")

    mu_sun = float(spice.bodvrd("SUN", "GM", 1)[1][0])
    sun_spline = build_sun_spline(
        et0, max(tau_full[-1], tau_target), CENTER, FRAME_I, ABCORR
    )
    C_rtn2i, d_sun = sun_rtn_frame(et0, CENTER, FRAME_I, ABCORR)
    print(f"[Ryugu] heliocentric distance at epoch: {d_sun / AU_KM:.4f} AU")
    print(
        f"[SRP] particle A/m = {part_A_over_m:.3e} m^2/kg -> |a_srp| ~ "
        f"{k_srp_part * (AU_KM / d_sun) ** 2:.3e} km/s^2"
    )

    # --------------------------
    # Shape model + polyhedral dynamics
    # --------------------------
    ryugu_mesh = trimesh.load(shape_path, force="mesh", process=False)
    vertices = np.asarray(ryugu_mesh.vertices)
    rverts = np.linalg.norm(vertices, axis=1)
    print(
        "[Mesh] faces =",
        len(ryugu_mesh.faces),
        "| vertex radius min/mean/max =",
        rverts.min(),
        rverts.mean(),
        rverts.max(),
        "| volume =",
        ryugu_mesh.volume,
        "km^3",
    )

    print("\nBuilding polyhedral gravity evaluable (ESA polyhedral-gravity-model)...")
    dyn_part = RyuguPolyhedralDynamics(
        vertices_km=vertices,
        faces=np.asarray(ryugu_mesh.faces),
        alpha=alpha_true,
        delta=delta_true,
        omega=omega_true,
        w0=w0_true,
        density=ryugu_density,
        sun_pos_spline=sun_spline,
        mu_sun=mu_sun,
        srp_k_1au=k_srp_part,
        parallel=True,
    )
    dyn_sc = dyn_part.with_srp_k(k_srp_sc)
    print(
        f"[Poly] density = {ryugu_density:.0f} kg/m^3, volume = {dyn_part.volume_km3:.4f} km^3 "
        f"-> GM = {dyn_part.gm * 1e9:.2f} m^3/s^2"
    )

    propagator = STTPropagatorND(
        order=stt_order,
        f_func=dyn_part.f_func,
        A_func=dyn_part.A_func,
        B_funcs=None,
        n=N_STATE,
    )
    propagator_sc = STTPropagatorND(
        order=1, f_func=dyn_sc.f_func, A_func=dyn_sc.A_func, B_funcs=None, n=N_STATE
    )

    # --------------------------
    # Spacecraft truth (notional, KNOWN): terminator orbit
    # --------------------------
    T_hat = C_rtn2i[:, 1]
    N_hat = C_rtn2i[:, 2]
    r_sc0 = a_sc * (np.cos(phi_sc0) * T_hat + np.sin(phi_sc0) * N_hat)
    v_sc0 = np.sqrt(dyn_part.gm / a_sc) * (-np.sin(phi_sc0) * T_hat + np.cos(phi_sc0) * N_hat)
    x0_sc = np.hstack([r_sc0, v_sc0, 1.0])

    print("\nPropagating notional spacecraft (truth, treated as known)...")
    sol_sc = propagator_sc.propagate_state_only(
        x0_sc, tau_full, rtol=1e-10, atol=1e-13, method="LSODA"
    )
    sc_state_full = sol_sc.y[:6, :].T  # (N,6)
    r_sc_norm = np.linalg.norm(sc_state_full[:, :3], axis=1)
    print(f"[S/C] range to Ryugu center: {r_sc_norm.min():.3f} - {r_sc_norm.max():.3f} km")

    # --------------------------
    # Particle detach point on Ryugu surface
    # --------------------------
    lat_desired = np.deg2rad(30.0)
    lon_desired = np.deg2rad(60.0)
    h_lift = 0.005  # km above the vertex along its normal (polyhedral tensor is singular on edges/vertices)
    pos_target = np.array(
        [
            R_ryugu * np.cos(lat_desired) * np.cos(lon_desired),
            R_ryugu * np.cos(lat_desired) * np.sin(lon_desired),
            R_ryugu * np.sin(lat_desired),
        ]
    )
    dists = np.linalg.norm(vertices - pos_target, axis=1)
    closest_idx = np.argmin(dists)
    normal_bf = np.asarray(ryugu_mesh.vertex_normals[closest_idx], dtype=float)
    normal_bf = normal_bf / np.linalg.norm(normal_bf)
    pos_detach_bf = vertices[closest_idx] + h_lift * normal_bf  # body-fixed at tau=0

    R_ib0 = make_ryugu_rotation_matrix(
        alpha_true, delta_true, omega_true, t=0.0, w0=w0_true
    )
    r0_true = R_ib0.T @ pos_detach_bf  # inertial
    nhat = R_ib0.T @ normal_bf  # local surface normal (inertial)

    # Rotational velocity of the surface point (inertial)
    pole_i = R_ib0.T @ np.array([0.0, 0.0, 1.0])
    v_rot = omega_true * np.cross(pole_i, r0_true)

    # Ejection velocity: mostly tangential + small lofting component along the normal
    rng = np.random.default_rng(7)
    vmag = 2.5e-4  # km/s, relative to the rotating surface (escape ~3.7e-4 km/s)
    u = rng.normal(size=3)
    u -= np.dot(u, nhat) * nhat
    u /= np.linalg.norm(u)
    rad_frac = 0.30
    u = np.sqrt(1 - rad_frac**2) * u + rad_frac * nhat

    v0_true = v_rot + vmag * u

    # Full truth initial state (7)
    x0_true = np.hstack([r0_true, v0_true, params_true])

    # --------------------------
    # Check analytic A (polyhedral tensor) against finite differences of f
    # --------------------------
    x_chk = x0_true.copy()
    x_chk[0:3] = R_ib0.T @ (vertices[closest_idx] + 0.05 * normal_bf)
    err_A = check_A_against_fd(dyn_part, x_chk, t=0.0)
    print("\n[Check] |A - A_fd| / |A_fd| per column:", np.array2string(err_A, precision=2))

    # --------------------------
    # Propagate particle truth (DYNAMICS on tau_full)
    # --------------------------
    print("\nPropagating particle truth...")
    sol_true, stts_true = propagator.propagate(
        x0_true, tau_full, rtol=1e-10, atol=1e-13, method="LSODA"
    )
    x_true_full = sol_true.y[:N_STATE, :].T  # (N,7)

    # Surface clearance check on a dense grid (impact => polyhedral field is not valid)
    tau_dense = np.linspace(tau_full[0], tau_full[-1], 400)
    sol_dense = propagator.propagate_state_only(
        x0_true, tau_dense, show_progress=False, rtol=1e-10, atol=1e-13, method="LSODA"
    )
    inside = points_inside_mesh(
        ryugu_mesh, sol_dense.y[:3, :].T, tau_dense, alpha_true, delta_true, omega_true
    )
    r_dense = np.linalg.norm(sol_dense.y[:3, :], axis=0)
    print(
        f"[Particle] radius along arc: {r_dense.min():.3f} - {r_dense.max():.3f} km | "
        f"inside-mesh samples: {np.sum(inside)}"
    )
    if np.any(inside):
        print("[WARNING] Particle truth trajectory impacts Ryugu: change detach/launch.")

    # --------------------------
    # Observability mask (occultation by the shape model)
    # --------------------------
    # Propagate on the FULL uniform time grid and only exclude occulted observations
    # from the cost function (zero weights).
    vis_mask_full = occultation_mask_shape(
        sc_state_full[:, 0:3],
        x_true_full[:, 0:3],
        ryugu_mesh,
        alpha_true,
        delta_true,
        omega_true,
        w0=w0_true,
        tau=tau_full,
    )

    ets = ets_full
    tau = tau_full
    sc_state = sc_state_full
    x_true = x_true_full
    vis_mask = vis_mask_full

    print(f"\nVisibility: {np.sum(vis_mask)}/{len(vis_mask)} epochs visible.")
    print(f"Using all {len(tau)} epochs for propagation (uniform time grid).")

    # --------------------------
    # Figures: observation geometry (truth only)
    # --------------------------
    sc_dense = propagator_sc.propagate_state_only(
        x0_sc, tau_dense, show_progress=False, rtol=1e-10, atol=1e-13, method="LSODA"
    ).y[:6].T
    pt_dense = sol_dense.y[:3].T

    plot_scene_body_fixed(
        ryugu_mesh,
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
    plot_observation_geometry(
        tau_full, sc_state_full, x_true_full, vis_mask,
        tau_line=tau_dense, sc_line=sc_dense, pt_line=pt_dense,
    )

    # --------------------------
    # Generate noisy RA/DEC measurements
    # --------------------------
    rng_meas = np.random.default_rng(123)
    y_obs_full = generate_opnav_measurements_from_sc(
        x_part=x_true,
        sc_state=sc_state,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        rng=rng_meas,
    )

    # zero weight for occulted observations
    obs_weights = np.ones_like(y_obs_full)
    obs_weights[0::2][~vis_mask] = 0.0
    obs_weights[1::2][~vis_mask] = 0.0

    y_obs = y_obs_full

    # --------------------------
    # Reference initial condition (perturb truth)
    # --------------------------
    print(f"\nBuilding reference initial state ({N_STATE}D)...")

    sig_ref_r = np.abs(x0_true[0:3] * ref_pct_r)
    sig_ref_v = np.abs(x0_true[3:6] * ref_pct_v)

    ref_dev = np.hstack(
        [
            rng_ref.normal(scale=sig_ref_r, size=3),
            rng_ref.normal(scale=sig_ref_v, size=3),
            rng_ref.normal(scale=ref_sig_eta, size=1),
        ]
    )
    x0_ref = x0_true - ref_dev
    print("\n[Reference] deviation from truth:", ref_dev)

    # --------------------------
    # Priors on 7D delta0
    # --------------------------
    prior_sigma = prior_looseness * np.hstack(
        [sig_prior_r, sig_prior_v, sig_prior_eta]
    )
    priors = [norm(loc=0.0, scale=s) for s in prior_sigma]
    print("\n[Prior] sigmas:", prior_sigma)

    # --------------------------
    # Stage 1: Gauss-Newton batch with STMs and visibility weighting
    # --------------------------
    print("\n[Stage 1] Gauss-Newton batch (first-order STMs) to convergence...")

    x0_ref1, delta_hat1, cov1 = solve_stage1_gn_with_stm(
        propagator=propagator,
        x0_ref=x0_ref,
        tau=tau,
        sc_state=sc_state,
        y_obs=y_obs,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        obs_weights=obs_weights,
        priors=priors,
        max_iter=30,
        tol=1e-6,
        rtol=1e-10,
        atol=1e-13,
        verbose=True,
    )
    """
    x0_ref1, delta_hat1, cov1, res_nl = solve_stage1_full_nonlinear_lsq(
        propagator=propagator,
        x0_ref=x0_ref,
        tau=tau,
        sc_state=sc_state,
        y_obs=y_obs,
        obs_weights=obs_weights,
        sigma_ra=sigma_ra,
        sigma_dec=sigma_dec,
        priors=priors,
        rtol=1e-10,
        atol=1e-13,
        max_nfev=4000,
        verbose=2,
    )"""

    print("\n[Stage 1] Covariance diagonal (stdev):")
    print(np.sqrt(np.diag(cov1)))
    print("[Stage 1] delta_hat1:\n", delta_hat1)
    err1 = (x0_true - x0_ref1) / np.sqrt(np.diag(cov1))
    print("[Stage 1] (truth - estimate) / sigma:\n", np.array2string(err1, precision=2))

    # --------------------------
    # Stage 2: relinearize STMs about ref1
    # --------------------------
    print("\n[Stage 2] Propagating ref1 and computing STMs about ref1...")
    sol_ref, stts_ref = propagator.propagate(
        x0=x0_ref1, t_eval=tau, rtol=1e-10, atol=1e-13, method="LSODA"
    )

    # --------------------------
    # Residual function (STM-based) with visibility weighting
    # (module-level function in __main__ -> picklable by reference for the fork pool)
    # --------------------------
    def residuals_normalized(delta0):
        _, x_est = propagator.propagate_deviation(sol_ref, stts_ref, delta0)

        los = x_est[:, :3] - sc_state[:, :3]
        ra_model, dec_model, _, _ = radec_and_partials_from_los(los)

        res = np.empty_like(y_obs)
        res[0::2] = wrap_to_pi(y_obs[0::2] - ra_model)
        res[1::2] = y_obs[1::2] - dec_model

        w = np.empty_like(y_obs)
        w[0::2] = sigma_ra
        w[1::2] = sigma_dec

        return (res / w) * obs_weights

    # --------------------------
    # Chi2 at ref1
    # --------------------------
    chi2_at_ref = np.sum(residuals_normalized(np.zeros(N_STATE)) ** 2)
    n_visible = np.sum(vis_mask)
    dof = 2 * n_visible - N_STATE
    print(
        f"\n[Stage 2] At ref1 (delta=0): chi2_red = {chi2_at_ref/dof:.3f}  "
        f"(chi2={chi2_at_ref:.2f}, dof={dof}, n_vis={n_visible})"
    )

    # Priors for MCMC (same priors as Stage 1, re-centered on ref1)
    delta_shift = x0_ref1 - x0_ref
    priors_ref1 = [norm(loc=-ds, scale=s) for ds, s in zip(delta_shift, prior_sigma)]

    # --------------------------
    # MCMC
    # --------------------------
    print("\n[MCMC] Running (starting from zeros about ref1)...")
    model = MCMCModel(
        residuals_func=residuals_normalized,
        initial_params=np.zeros(N_STATE),
        param_priors=priors_ref1,
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
    )

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

    # --------------------------
    # Diagnostics + figures
    # --------------------------
    model.summary()
    model.print_regression_diagnostics()

    # Post-fit residuals: GN batch (delta = 0 about ref1) vs MCMC MAP
    plot_postfit_residuals(
        tau,
        residuals_normalized(np.zeros(N_STATE)),
        residuals_normalized(theta_hat),
        vis_mask,
    )
    plot_mcmc_traces(
        model, burn_in, true_theta=true_delta, batch_theta=np.zeros(N_STATE)
    )
    plot_mcmc_autocorrelation(model, burn_in, thin)

    # Corner plot (deltas about ref1: dx0, dy0, dz0, dvx0, dvy0, dvz0, deta)
    try:
        plot_corner(
            model.samples,
            true_theta=true_delta,
            batch_mean=np.zeros(N_STATE),
            batch_cov=cov1,
        )
    except Exception as e:
        print("\n[Corner] Skipped.", e)

    # --------------------------
    # Posterior at epoch -> GMM split -> STM propagation to utc_target
    # --------------------------
    # Samples are deviations about ref1 at tau = 0. Every mixture component k is propagated
    # about its own mean with its own STM (P_k(t) = Phi_k P_k Phi_k^T); the single-Gaussian
    # baseline is the same with K = 1 (sample mean + covariance). A nonlinear Monte Carlo of
    # posterior samples is the reference both are checked against.
    post = np.asarray(model.samples, dtype=float)
    print(
        f"\n[GMM] Splitting {post.shape[0]} posterior samples at epoch ({utc0}) and "
        f"propagating to {utc_target} (+{tau_target / 3600.0:.2f} h)"
    )
    w_g, mu_g, P_g = fit_posterior_gmm(
        post, n_components=gmm_n_components, select_bic=gmm_select_bic
    )
    print("[GMM] weights:", np.array2string(w_g, precision=3))

    w_1 = np.ones(1)
    mu_1 = post.mean(axis=0, keepdims=True)
    P_1 = np.cov(post, rowvar=False)[None]

    t_prop = np.linspace(0.0, tau_target, n_t_prop)
    ivp_prop = dict(rtol=1e-10, atol=1e-13, method="LSODA")

    m_g_t, P_g_t = propagate_gmm_stm(propagator, x0_ref1, w_g, mu_g, P_g, t_prop, **ivp_prop)
    m_1_t, P_1_t = propagate_gmm_stm(
        propagator, x0_ref1, w_1, mu_1, P_1, t_prop, label="Single Gaussian", **ivp_prop
    )

    rng_mc = np.random.default_rng(2024)
    idx_mc = rng_mc.choice(post.shape[0], size=min(n_mc_prop, post.shape[0]), replace=False)
    X_mc = propagate_samples_mc(propagator, x0_ref1, post[idx_mc], t_prop, **ivp_prop)
    ok_mc = np.all(np.isfinite(X_mc[:, :, 0]), axis=1)
    if not np.all(ok_mc):
        print(f"[MC] {np.sum(~ok_mc)} sample propagations failed and are dropped.")
    X_mc = X_mc[ok_mc]

    sol_tp = propagator.propagate_state_only(x0_true, t_prop, show_progress=False, **ivp_prop)
    x_true_prop = sol_tp.y[:N_STATE, :].T  # (n_t_prop, n)

    # Polyhedral field is only valid outside the body: flag any impact along the prediction
    n_in_truth = int(
        np.sum(
            points_inside_mesh(
                ryugu_mesh, x_true_prop[:, :3], t_prop, alpha_true, delta_true, omega_true
            )
        )
    )
    n_in_mc = sum(
        bool(
            np.any(
                points_inside_mesh(
                    ryugu_mesh, X_mc[i, :, :3], t_prop, alpha_true, delta_true, omega_true
                )
            )
        )
        for i in range(X_mc.shape[0])
    )
    print(
        f"[GMM] inside-mesh epochs along the prediction: truth {n_in_truth}, "
        f"MC samples that impact {n_in_mc}/{X_mc.shape[0]}"
    )
    if n_in_truth or n_in_mc:
        print("[WARNING] Prediction crosses the surface: move utc_target earlier.")

    # Metrics at the target epoch (position, meters)
    pos_mc = X_mc[:, -1, :3]
    m_mc = pos_mc.mean(axis=0)
    mg_T, Pg_T = gmm_moments(w_g, m_g_t[:, -1, :3], P_g_t[:, -1, :3, :3])
    m1_T, P1_T = m_1_t[0, -1, :3], P_1_t[0, -1, :3, :3]
    sig_mc_T = np.sqrt(np.diag(np.cov(pos_mc, rowvar=False)))
    print(f"\n[GMM] Position at target ({utc_target}), per-axis 1-sigma [m]:")
    print("   MC             :", np.array2string(sig_mc_T * 1e3, precision=2))
    print("   GMM + STMs     :", np.array2string(np.sqrt(np.diag(Pg_T)) * 1e3, precision=2))
    print("   Single Gaussian:", np.array2string(np.sqrt(np.diag(P1_T)) * 1e3, precision=2))
    print("[GMM] Mean offset from the MC mean [m]:")
    print("   GMM + STMs     :", np.array2string((mg_T - m_mc) * 1e3, precision=2))
    print("   Single Gaussian:", np.array2string((m1_T - m_mc) * 1e3, precision=2))

    ll_g = np.mean(gmm_logpdf(pos_mc, w_g, m_g_t[:, -1, :3], P_g_t[:, -1, :3, :3]))
    ll_1 = np.mean(gmm_logpdf(pos_mc, w_1, m_1_t[:, -1, :3], P_1_t[:, -1, :3, :3]))
    print(
        f"[GMM] Mean log-density of the MC positions: GMM {ll_g:.3f} | single {ll_1:.3f} "
        f"(GMM - single = {ll_g - ll_1:+.3f} nats)"
    )

    r_tr = x_true_prop[-1, :3]
    d_tr = r_tr - m1_T
    maha_1 = float(np.sqrt(d_tr @ np.linalg.solve(P1_T, d_tr)))
    lp_tr_g = gmm_logpdf(r_tr, w_g, m_g_t[:, -1, :3], P_g_t[:, -1, :3, :3])[0]
    lp_tr_1 = gmm_logpdf(r_tr, w_1, m_1_t[:, -1, :3], P_1_t[:, -1, :3, :3])[0]
    print(
        f"[GMM] Truth at target: log p GMM {lp_tr_g:.3f} | single {lp_tr_1:.3f} | "
        f"Mahalanobis (single) {maha_1:.2f}"
    )

    # Figures
    def _panel(j, title):
        Pg_j = P_g_t[:, j, :3, :3]
        return dict(
            mc=X_mc[:, j, :3],
            w=w_g,
            mu=m_g_t[:, j, :3],
            P=Pg_j,
            mu1=m_1_t[0, j, :3],
            P1=P_1_t[0, j, :3, :3],
            truth=x_true_prop[j, :3],
            center=X_mc[:, j, :3].mean(axis=0),
            title=title,
        )

    plot_gmm_propagation(
        [
            _panel(0, f"Posterior at epoch ({utc0.replace('T', ' ')} UTC)"),
            _panel(
                -1,
                f"Predicted at {utc_target.replace('T', ' ')} UTC "
                f"(+{tau_target / 3600.0:.2f} h)",
            ),
        ],
        prob=gmm_prob,
    )

    hist = position_prediction_history(X_mc, w_g, m_g_t, P_g_t, m_1_t, P_1_t)
    plot_prediction_history(t_prop / 3600.0, hist, tau_full[-1] / 3600.0)

    spice.kclear()
