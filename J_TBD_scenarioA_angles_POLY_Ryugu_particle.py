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
from scipy.interpolate import CubicSpline
from scipy.stats import norm

from polyhedral_gravity import (
    GravityEvaluable,
    MetricUnit,
    Polyhedron,
    PolyhedronIntegrity,
)

from STTPropagationND import STTPropagatorND
from MCMC import MCMCModel
from BatchLSQ import BatchLSQ

# Journal figures (Okabe-Ito colors, LaTeX text, one vector PDF per figure) live in MCMCPlots.py;
# MCMCPlots.configure() in the main applies the journal style.
import MCMCPlots
from GMM import GMM, position_prediction_history, propagate_samples_mc

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved


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
        self.sun_pos_spline = (
            sun_pos_spline  # tau -> Sun position wrt Ryugu [km], J2000
        )
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
        volume = (
            np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])))
            / 6.0
        )
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
    err = np.linalg.norm(A - A_fd, axis=0) / np.maximum(
        np.linalg.norm(A_fd, axis=0), 1e-300
    )
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


def build_sun_spline(
    et0, tau_end, body_id, frame="J2000", abcorr="NONE", dt=60.0, pad=1800.0
):
    """Cubic spline tau -> Sun position wrt the body [km] (so the dynamics need no SPICE)."""
    tau_grid = np.arange(-pad, tau_end + pad + dt, dt)
    r_sun = np.array(
        [
            spice.spkpos("SUN", float(et0 + tg), frame, abcorr, body_id)[0]
            for tg in tau_grid
        ]
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

    # Stage-1 batch solve. False: LM-safeguarded GN with an unscaled lstsq solve (the results so
    # far). True: same LM, solved in prior-sigma units (same damped problem, full-rank solve, so
    # no direction is dropped by the lstsq cutoff). Runs go to a "_scaledGN" folder.
    SCALED_BATCH = False

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
    # Saved chain: every run writes chain.npz into its results folder. Set LOAD_CHAIN to
    # such a file to skip the sampling and redo the diagnostics and figures from it.
    LOAD_CHAIN = None  # e.g. "results/ryugu_poly_<stamp>/chain.npz"

    # Posterior split (GMM) + STM propagation to a given epoch
    # NOTE: the truth particle re-impacts Ryugu at ~3.99 h after utc0; keep the target before
    # that (the polyhedral field is not valid inside the body).
    utc_target = "2034-03-01T03:45:00"
    gmm_n_components = 8  # fixed K (narrow components keep each STM accurate)
    gmm_select_bic = False  # True: pick K in 1..12 by BIC instead
    n_mc_prop = 300  # posterior samples propagated nonlinearly (Monte Carlo check)
    n_t_prop = 46  # output epochs on [epoch, target]
    gmm_prob = 0.99  # probability mass of the plotted regions

    # Figures: parameter symbols and display units (km -> m, km/s -> mm/s, eta -> %), one PDF
    # per figure in a timestamped folder (MCMCPlots.SHOW_FIGS at the top of the file)
    MCMCPlots.configure(
        [
            r"\delta x_0",
            r"\delta y_0",
            r"\delta z_0",
            r"\delta v_{x,0}",
            r"\delta v_{y,0}",
            r"\delta v_{z,0}",
            r"\delta \eta",
        ],
        ["m", "m", "m", "mm/s", "mm/s", "mm/s", r"$\%$"],
        np.array([1e3, 1e3, 1e3, 1e6, 1e6, 1e6, 1e2]),
        rotation=make_ryugu_rotation_matrix,
        body="Ryugu",
        target="Particle",
        obs_word="Image",
        site_label="Ejection Site",
        resid_labels=("RA", "Dec"),
        fallback_dir=Path.home() / "ryugu_figures_fallback",
    )
    fig_dir = MCMCPlots.set_figure_dir(
        Path(__file__).resolve().parent
        / "results"
        / f"ryugu_poly{'_scaledGN' if SCALED_BATCH else ''}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    v_sc0 = np.sqrt(dyn_part.gm / a_sc) * (
        -np.sin(phi_sc0) * T_hat + np.cos(phi_sc0) * N_hat
    )
    x0_sc = np.hstack([r_sc0, v_sc0, 1.0])

    print("\nPropagating notional spacecraft (truth, treated as known)...")
    sol_sc = propagator_sc.propagate_state_only(
        x0_sc, tau_full, rtol=1e-8, atol=1e-10, method="LSODA"
    )
    sc_state_full = sol_sc.y[:6, :].T  # (N,6)
    r_sc_norm = np.linalg.norm(sc_state_full[:, :3], axis=1)
    print(
        f"[S/C] range to Ryugu center: {r_sc_norm.min():.3f} - {r_sc_norm.max():.3f} km"
    )

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
    print(
        "\n[Check] |A - A_fd| / |A_fd| per column:", np.array2string(err_A, precision=2)
    )

    # --------------------------
    # Propagate particle truth (DYNAMICS on tau_full)
    # --------------------------
    print("\nPropagating particle truth...")
    sol_true, stts_true = propagator.propagate(
        x0_true, tau_full, rtol=1e-8, atol=1e-10, method="LSODA"
    )
    x_true_full = sol_true.y[:N_STATE, :].T  # (N,7)

    # Surface clearance check on a dense grid (impact => polyhedral field is not valid)
    tau_dense = np.linspace(tau_full[0], tau_full[-1], 400)
    sol_dense = propagator.propagate_state_only(
        x0_true, tau_dense, show_progress=False, rtol=1e-8, atol=1e-10, method="LSODA"
    )
    inside = MCMCPlots.points_inside_mesh(
        ryugu_mesh, sol_dense.y[:3, :].T, tau_dense, alpha_true, delta_true, omega_true
    )
    r_dense = np.linalg.norm(sol_dense.y[:3, :], axis=0)
    print(
        f"[Particle] radius along arc: {r_dense.min():.3f} - {r_dense.max():.3f} km | "
        f"inside-mesh samples: {np.sum(inside)}"
    )
    if np.any(inside):
        print(
            "[WARNING] Particle truth trajectory impacts Ryugu: change detach/launch."
        )

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
    sc_dense = (
        propagator_sc.propagate_state_only(
            x0_sc,
            tau_dense,
            show_progress=False,
            rtol=1e-8,
            atol=1e-10,
            method="LSODA",
        )
        .y[:6]
        .T
    )
    pt_dense = sol_dense.y[:3].T

    MCMCPlots.plot_scene_body_fixed(
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
    prior_sigma = prior_looseness * np.hstack([sig_prior_r, sig_prior_v, sig_prior_eta])
    priors = [norm(loc=0.0, scale=s) for s in prior_sigma]
    print("\n[Prior] sigmas:", prior_sigma)

    # --------------------------
    # Stage 1: Gauss-Newton batch with STMs and visibility weighting
    # --------------------------
    print("\n[Stage 1] Gauss-Newton batch (first-order STMs) to convergence...")

    batch = BatchLSQ(
        propagator=propagator,
        tau=tau,
        sc_state=sc_state,
        y_obs=y_obs,
        sigma=(sigma_ra, sigma_dec),
        obs_weights=obs_weights,
        measurement=radec_and_partials_from_los,
        wrap=wrap_to_pi,
        postfit_plot=False,  # no batch post-fit residual figure for Ryugu
    )
    x0_ref1, delta_hat1, cov1 = batch.gn_with_stm(
        x0_ref=x0_ref,
        priors=priors,
        max_iter=30,
        tol=1e-8,
        ftol=1e-2,  # stop once an LM step gains < 1% in MAP cost: converged, not exact
        rtol=1e-8,
        atol=1e-10,
        verbose=True,
        lm=True,  # Levenberg-Marquardt safeguard: the plain GN step diverges from x0_ref
        scaled=SCALED_BATCH,
    )
    """
    x0_ref1, delta_hat1, cov1, res_nl = batch.full_nonlinear_lsq(
        x0_ref=x0_ref,
        priors=priors,
        rtol=1e-8,
        atol=1e-10,
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
        x0=x0_ref1, t_eval=tau, rtol=1e-8, atol=1e-10, method="LSODA"
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

    # --------------------------
    # Diagnostics + figures
    # --------------------------
    model.summary()
    model.print_regression_diagnostics()

    # Samples are deviations about the batch solution ref1, so the batch estimate is 0 with
    # covariance cov1 (corner: deltas dx0, dy0, dz0, dvx0, dvy0, dvz0, deta).
    model.plot_journal_diagnostics(
        burn_in,
        thin,
        tau,
        vis_mask,
        true_theta=true_delta,
        batch_cov=cov1,
    )

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
    gmm = GMM.fit(post, n_components=gmm_n_components, select_bic=gmm_select_bic)
    print("[GMM] weights:", np.array2string(gmm.weights, precision=3))
    single = GMM.gaussian(post.mean(axis=0), np.cov(post, rowvar=False))

    t_prop = np.linspace(0.0, tau_target, n_t_prop)
    ivp_prop = dict(rtol=1e-8, atol=1e-10, method="LSODA")

    # One GMM per output epoch (absolute states)
    gmm_t = gmm.propagate_stm(propagator, x0_ref1, t_prop, **ivp_prop)
    single_t = single.propagate_stm(
        propagator, x0_ref1, t_prop, label="Single Gaussian", **ivp_prop
    )

    rng_mc = np.random.default_rng(2024)
    idx_mc = rng_mc.choice(
        post.shape[0], size=min(n_mc_prop, post.shape[0]), replace=False
    )
    X_mc = propagate_samples_mc(propagator, x0_ref1, post[idx_mc], t_prop, **ivp_prop)
    ok_mc = np.all(np.isfinite(X_mc[:, :, 0]), axis=1)
    if not np.all(ok_mc):
        print(f"[MC] {np.sum(~ok_mc)} sample propagations failed and are dropped.")
    X_mc = X_mc[ok_mc]

    sol_tp = propagator.propagate_state_only(
        x0_true, t_prop, show_progress=False, **ivp_prop
    )
    x_true_prop = sol_tp.y[:N_STATE, :].T  # (n_t_prop, n)

    # Polyhedral field is only valid outside the body: flag any impact along the prediction
    n_in_truth = int(
        np.sum(
            MCMCPlots.points_inside_mesh(
                ryugu_mesh,
                x_true_prop[:, :3],
                t_prop,
                alpha_true,
                delta_true,
                omega_true,
            )
        )
    )
    n_in_mc = sum(
        bool(
            np.any(
                MCMCPlots.points_inside_mesh(
                    ryugu_mesh,
                    X_mc[i, :, :3],
                    t_prop,
                    alpha_true,
                    delta_true,
                    omega_true,
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
    gmm_T = gmm_t[-1].marginal(slice(0, 3))
    single_T = single_t[-1].marginal(slice(0, 3))
    mg_T, Pg_T = gmm_T.moments()
    m1_T, P1_T = single_T.moments()
    sig_mc_T = np.sqrt(np.diag(np.cov(pos_mc, rowvar=False)))
    print(f"\n[GMM] Position at target ({utc_target}), per-axis 1-sigma [m]:")
    print("   MC             :", np.array2string(sig_mc_T * 1e3, precision=2))
    print(
        "   GMM + STMs     :",
        np.array2string(np.sqrt(np.diag(Pg_T)) * 1e3, precision=2),
    )
    print(
        "   Single Gaussian:",
        np.array2string(np.sqrt(np.diag(P1_T)) * 1e3, precision=2),
    )
    print("[GMM] Mean offset from the MC mean [m]:")
    print("   GMM + STMs     :", np.array2string((mg_T - m_mc) * 1e3, precision=2))
    print("   Single Gaussian:", np.array2string((m1_T - m_mc) * 1e3, precision=2))

    ll_g = np.mean(gmm_T.logpdf(pos_mc))
    ll_1 = np.mean(single_T.logpdf(pos_mc))
    print(
        f"[GMM] Mean log-density of the MC positions: GMM {ll_g:.3f} | single {ll_1:.3f} "
        f"(GMM - single = {ll_g - ll_1:+.3f} nats)"
    )

    r_tr = x_true_prop[-1, :3]
    d_tr = r_tr - m1_T
    maha_1 = float(np.sqrt(d_tr @ np.linalg.solve(P1_T, d_tr)))
    lp_tr_g = gmm_T.logpdf(r_tr)[0]
    lp_tr_1 = single_T.logpdf(r_tr)[0]
    print(
        f"[GMM] Truth at target: log p GMM {lp_tr_g:.3f} | single {lp_tr_1:.3f} | "
        f"Mahalanobis (single) {maha_1:.2f}"
    )

    # Figures
    def _panel(j, title):
        return dict(
            mc=X_mc[:, j, :3],
            gmm=gmm_t[j].marginal(slice(0, 3)),
            single=single_t[j].marginal(slice(0, 3)),
            truth=x_true_prop[j, :3],
            center=X_mc[:, j, :3].mean(axis=0),
            title=title,
        )

    MCMCPlots.plot_gmm_propagation(
        [
            _panel(0, f"Posterior at Epoch ({utc0.replace('T', ' ')} UTC)"),
            _panel(
                -1,
                f"Predicted at {utc_target.replace('T', ' ')} UTC "
                f"(+{tau_target / 3600.0:.2f} h)",
            ),
        ],
        prob=gmm_prob,
    )

    hist = position_prediction_history(X_mc, gmm_t, single_t)
    MCMCPlots.plot_prediction_history(t_prop / 3600.0, hist, tau_full[-1] / 3600.0)

    spice.kclear()
