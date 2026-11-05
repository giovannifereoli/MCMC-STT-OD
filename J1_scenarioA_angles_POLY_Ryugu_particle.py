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
from matplotlib.lines import Line2D
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
        "grid.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage{mathrsfs}"

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


def plot_ryugu_scene_body_fixed(
    ryugu_mesh,
    sc_state_full,
    x_true_full,
    x_map,
    tau_full,
    tau_map,
    alpha,
    delta,
    omega,
    vis_mask=None,
    sc_label="Notional S/C (CSR-30)",
    downsample=1,
):
    """
    BODY-FIXED visualization (publication-ready).
    Assumes `ryugu_mesh` is in Ryugu body-fixed frame, centered at origin;
    make_ryugu_rotation_matrix returns R_ib (inertial -> body).
    """
    sl = slice(None, None, max(1, int(downsample)))

    sc_i = np.asarray(sc_state_full)[sl, :3]
    pt_i = np.asarray(x_true_full)[sl, :3]
    tau_i = np.asarray(tau_full)[sl]
    mesh = ryugu_mesh

    # Rotate INERTIAL -> BODY-FIXED
    sc_b = np.zeros_like(sc_i)
    pt_b = np.zeros_like(pt_i)
    for k, tk in enumerate(tau_i):
        R_ib = np.asarray(make_ryugu_rotation_matrix(alpha, delta, omega, float(tk)))
        sc_b[k] = R_ib @ sc_i[k]
        pt_b[k] = R_ib @ pt_i[k]

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    add_trimesh_to_ax(ax, mesh, alpha=0.35, edge_alpha=0.08, max_faces=25000)

    ax.plot(
        sc_b[:, 0],
        sc_b[:, 1],
        sc_b[:, 2],
        linewidth=1.8,
        alpha=0.95,
        color="tab:blue",
        label=sc_label,
    )
    ax.plot(
        pt_b[:, 0],
        pt_b[:, 1],
        pt_b[:, 2],
        linewidth=1.6,
        alpha=0.95,
        color="tab:red",
        label="Particle Truth",
    )

    # Visible / occulted points (both black)
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
                label="Measurement Included",
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

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.94)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.canvas.draw()
    fig.savefig(
        f"results/ryugu_scene_body_fixed_{timestamp}.pdf",
        format="pdf",
        pad_inches=0.20,
    )
    print(f"Saved: results/ryugu_scene_body_fixed_{timestamp}.pdf")

    plt.show()


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

    # ----------------------------
    # Plot postfit (Stage-1 GN)
    # ----------------------------
    if verbose:
        sol_pf, _ = propagator.propagate(
            x0=x0_ref, t_eval=tau, rtol=rtol, atol=atol, method=method
        )
        x_pf = sol_pf.y[:n, :].T

        los_pf = x_pf[:, :3] - sc_state[:, :3]
        ra_pf, dec_pf, _, _ = radec_and_partials_from_los(los_pf)

        res_pf = np.empty_like(y_obs)
        res_pf[0::2] = wrap_to_pi(y_obs[0::2] - ra_pf)
        res_pf[1::2] = y_obs[1::2] - dec_pf

        postfit = (res_pf / w) * obs_weights

        residuals_matrix = postfit.reshape(-1, 2).T
        time_hr = np.asarray(tau).ravel() / 3600.0

        mask0 = residuals_matrix[0] != 0.0
        mask1 = residuals_matrix[1] != 0.0

        r0, t0 = residuals_matrix[0, mask0], time_hr[mask0]
        r1, t1 = residuals_matrix[1, mask1], time_hr[mask1]

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        ax0.scatter(t0, r0, s=50, alpha=0.85)
        ax0.axhline(3, color="red", linestyle=":", linewidth=3.0)
        ax0.axhline(-3, color="red", linestyle=":", linewidth=3.0)
        ax0.set_ylabel(r"Whitened RA Residual [$\sigma$]")
        ax0.grid(True, linestyle=":")
        if r0.size > 0:
            lim = np.max(np.abs(np.quantile(r0, [0.01, 0.99])))
            lim = max(lim, 3.2)
            ax0.set_ylim(-1.1 * lim, 1.1 * lim)

        ax1.scatter(t1, r1, s=50, alpha=0.85)
        ax1.axhline(3, color="red", linestyle=":", linewidth=3.0)
        ax1.axhline(-3, color="red", linestyle=":", linewidth=3.0)
        ax1.set_ylabel(r"Whitened DEC Residual [$\sigma$]")
        ax1.set_xlabel("Time since epoch [hours]")
        ax1.grid(True, linestyle=":")
        if r1.size > 0:
            lim = np.max(np.abs(np.quantile(r1, [0.01, 0.99])))
            lim = max(lim, 3.2)
            ax1.set_ylim(-1.1 * lim, 1.1 * lim)

        legend_items = [
            Line2D(
                [0],
                [0],
                color="red",
                linestyle=":",
                linewidth=3.2,
                label=r"$\pm 3\sigma$",
            ),
        ]
        ax0.legend(handles=legend_items, loc="upper right", frameon=True)

        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"results/postfit_residuals_batch_ryugu_{timestamp}.pdf"
        fig.savefig(fname, format="pdf", bbox_inches="tight")
        fig.tight_layout()
        plt.show()

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

GMM_COLOR = "#2a78d6"  # Gaussian mixture (categorical slot 1)
GAUSS_COLOR = "#eb6834"  # single Gaussian (categorical slot 2)
MC_COLOR = "#8c8c8c"  # Monte Carlo samples (neutral)


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


def plot_gmm_propagation(panels, prob=0.99, n_draw=20000, seed=0):
    """
    2x3 position projections (row 0: epoch, row 1: target). Each panel dict holds (km):
      mc (M,3), w (K,), mu (K,3), P (K,3,3), mu1 (3,), P1 (3,3), truth (3,), center (3,), title.
    Drawn in meters relative to `center`: the MC cloud, every GMM component's ellipse, the
    GMM highest-density region holding `prob` of the mass (level from mixture draws), the
    single-Gaussian ellipse with the same mass, and the truth.
    """
    r_ell = np.sqrt(chi2.ppf(prob, 2))
    pct = f"{100 * prob:g}\\%"
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = [r"$\delta x$ [m]", r"$\delta y$ [m]", r"$\delta z$ [m]"]
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.6))
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
                mc[:, i], mc[:, j], s=5, color=MC_COLOR, alpha=0.45, lw=0,
                rasterized=True, zorder=1,
            )
            for k in range(len(w)):
                e = _ellipse_2d(mu2[k], P2[k], r_ell)
                ax.plot(e[0], e[1], color=GMM_COLOR, lw=0.8, alpha=0.5, zorder=2)

            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 220))
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            dens = np.exp(gmm_logpdf(grid, w, mu2, P2)).reshape(gx.shape)
            level = np.quantile(np.exp(gmm_logpdf(draws[:, ix], w, mu2, P2)), 1.0 - prob)
            ax.contour(gx, gy, dens, levels=[level], colors=GMM_COLOR, linewidths=2.0, zorder=3)

            ax.plot(e1[0], e1[1], color=GAUSS_COLOR, lw=2.0, ls="--", zorder=4)
            ax.plot(
                tr[i], tr[j], ls="none", marker="*", ms=13, color="k",
                mec="white", mew=0.8, zorder=5,
            )
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_xlabel(names[i])
            ax.set_ylabel(names[j])
        axes[row, 0].set_title(pnl["title"], loc="left")

    handles = [
        Line2D([], [], ls="none", marker="o", ms=5, color=MC_COLOR, label="Monte Carlo (nonlinear)"),
        Line2D([], [], color=GMM_COLOR, lw=0.8, alpha=0.6, label=f"GMM components ({pct})"),
        Line2D([], [], color=GMM_COLOR, lw=2.0, label=f"GMM + STMs ({pct} region)"),
        Line2D([], [], color=GAUSS_COLOR, lw=2.0, ls="--", label=f"Single Gaussian + STM ({pct})"),
        Line2D([], [], ls="none", marker="*", ms=12, color="k", label="Truth"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/gmm_stm_propagation_ryugu_{timestamp}.pdf"
    fig.savefig(fname, format="pdf", bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


def plot_prediction_history(t_hr, hist, t_obs_end_hr):
    """Position uncertainty and MC log-density (GMM vs single Gaussian) vs time."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)
    for ax in (ax0, ax1):
        ax.axvspan(t_hr[0], t_obs_end_hr, color="0.93", lw=0, zorder=0)
    ax0.text(
        0.5 * (t_hr[0] + t_obs_end_hr), 0.97, "observation arc", color="0.35",
        ha="center", va="top", transform=ax0.get_xaxis_transform(),
    )

    ax0.plot(t_hr, hist["sig_mc"] * 1e3, color="0.25", lw=2.0, label="Monte Carlo (nonlinear)")
    ax0.plot(t_hr, hist["sig_gmm"] * 1e3, color=GMM_COLOR, lw=2.0, label="GMM + STMs")
    ax0.plot(
        t_hr, hist["sig_1"] * 1e3, color=GAUSS_COLOR, lw=2.0, ls="--",
        label="Single Gaussian + STM",
    )
    ax0.set_ylabel(r"$\sqrt{\mathrm{tr}\,P_{rr}}$ [m]")
    ax0.set_yscale("log")
    ax0.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax0.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax1.axhline(0.0, color="0.5", lw=0.8)
    ax1.plot(t_hr, hist["ll_gmm"] - hist["ll_1"], color=GMM_COLOR, lw=2.0)
    ax1.set_ylabel(r"$\overline{\log p}_{\mathrm{GMM}} - \overline{\log p}_{\mathcal{N}}$ [nats]")
    ax1.set_xlabel("Time since epoch [hr]")
    ax1.set_title(r"Mean log-density of the MC positions ($>0$: mixture fits better)", loc="left")

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/gmm_prediction_history_ryugu_{timestamp}.pdf"
    fig.savefig(fname, format="pdf", bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()


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
    # Diagnostics
    # --------------------------
    model.plot_convergence()
    model.plot_postfit_residuals_time(t_obs_used=tau, opnav_data=True)
    model.summary()
    model.print_regression_diagnostics()
    model.plot_autocorrelation()
    model.plot_log_likelihood()

    # --------------------------
    # Plot scene (using MCMC estimate)
    # --------------------------
    _, x_map = propagator.propagate_deviation(sol_ref, stts_ref, theta_hat)

    plot_ryugu_scene_body_fixed(
        ryugu_mesh=ryugu_mesh,
        sc_state_full=sc_state_full,
        x_true_full=x_true_full,
        x_map=x_map,
        tau_full=tau_full,
        tau_map=tau,
        alpha=alpha_true,
        delta=delta_true,
        omega=omega_true,
        vis_mask=vis_mask,
        downsample=1,
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
    plt.title("Occultation / Visibility Mask (shape model)")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Corner plot
    # (x_0..x_6 = dx0, dy0, dz0, dvx0, dvy0, dvz0, deta)
    # --------------------------
    try:
        model.plot_corner_with_batch(
            batch_mean=np.zeros(N_STATE),
            batch_cov=cov1,
            use_median_as_truth=False,
            true_theta=true_delta,
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
