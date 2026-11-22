"""
Scenario ISL-SH2: Hera-like mothercraft + CubeSat inter-satellite link (ISL) at Bennu, built
so that the degree-2 gravity posterior is both informed by the data and non-Gaussian.

Geometry
- Bennu: unnormalized degree-2 field of get_CS_unnormalized.bennu_10_10 (R_ref = 0.290 km),
  pole (85.65, -60.17) deg, period 4.296057 h, IAU constant spin (frame rotations, body +Z =
  pole, spin +omega, as in scenario_NGSR.py).
- Mothercraft (MC): MC_DIST km above the north spin pole, fixed in inertial space (station-kept,
  known ephemeris; it plays the role of the SPICE spacecraft of the J1 scripts). MC_TILT_DEG
  moves it off the spin axis.
- CubeSat: polar orbit of radius ORBIT_RADIUS, ARC_HOURS arc, two-way ISL range and range-rate
  every DT_OBS_MIN minutes, occulted by the sphere of radius R_ref.
- Estimated: x0 = [r0, v0, GM, C20, C21, S21, C22, S22] of the CubeSat (12 parameters).

Why this posterior is not Gaussian
With the MC on the spin axis, the ISL data are exactly invariant under the one-parameter group
g_delta: rotate r0 and v0 by delta about the pole, (C21, S21) by delta and (C22, S22) by
2 delta. The rotated CubeSat orbit in the rotated field is the original orbit rotated by delta
(the spin is about the same axis), and range / range-rate to a point on the axis do not change.
So the data fix J22 = hypot(C22, S22) and the relative phase phi0 - lambda22 (phi0: body
longitude of r0 at t0, lambda22 = atan2(S22, C22) / 2), but not lambda22 itself: the posterior
is a curved ridge along the symmetry orbit, a crescent in (C22, S22) and in the position, whose
length is set by the prior (~10-15 deg in delta), while its width is set by the data.
Off the axis (MC_TILT_DEG > 0) or with an angle measurement (a camera on the MC) the symmetry is
broken, delta becomes observable and the posterior turns back into a narrow Gaussian.

Sampling the ridge
- A plain STT about the batch solution x0_ref1 fails along the arc: 10 deg of delta is ~70 m of
  position, far outside the STT region. The symmetry-reduced STT is exact in delta: rotate the
  sample onto the slice through x0_ref1 (position azimuth about the pole = that of x0_ref1),
  evaluate the STT there, rotate the predicted trajectory back by delta (SymmetryChart).
- emcee (stretch move, also whitened) does not cross a ridge this curved: the walkers collapse
  to a short piece of it (sigma_delta 0.2-0.7 deg instead of 9-14 deg in tests). The MCMC
  therefore samples chart coordinates theta (MCMCModelChart): theta is the Cartesian deviation
  about x0_ref1 whose position component along e_phi (azimuth direction about the pole) is the
  arc length s = rho_ref * delta, so x = g_delta(x0_ref1 + ds) with ds = theta minus that
  component. In theta the ridge is straight; the prior is evaluated at dx = x - x0_ref1, with
  the Jacobian log(1 + ds.e_rho / rho_ref). USE_SYMMETRY_CHART = False samples dx directly with
  the plain STT, for comparison.
- STT_ORDER = 1 (default): the Stage-2 surrogate uses the STMs only, and Stage 1 ends with an
  exact trust-region least-squares MAP in chart coordinates (exact_map_chart); 2 adds the
  second-order STT, and Stage 1 ends with the STT re-centering (results folder suffix _stt2).
- SH_PRIOR = "uniform" replaces the Gaussian priors of C20, C21, S21, C22, S22 by boxes
  x0_ref +- UNIFORM_HALF_WIDTH prior sigmas (GM, r0, v0 stay Gaussian; results folder suffix
  _uniform). Stage 1 only needs a reference point: it uses Gaussians with the mean and sd of
  the boxes; the MCMC samples the box prior itself.
- An exact check (nonlinear propagation of N_EXACT_CHECK posterior samples) reports the
  importance-sampling ESS of the STT surrogate against the exact likelihood.

Pipeline
  1) Truth: CubeSat orbit, MC position, noisy range / range-rate on the visible epochs.
  2) A priori x0_ref = truth - prior draw.
  3) Stage 1: BatchLSQ Gauss-Newton (STMs, Levenberg-Marquardt, scaled) on growing arcs, then
     to the exact MAP x0_ref1 (exact_map_chart with STMs, STT re-centering with STTs);
     cov1 = batch covariance there.
  4) Stage 2: STMs (STT_ORDER = 1) or STTs about x0_ref1, symmetry demo (exact / plain STT /
     chart).
  5) MCMC in chart coordinates; chain saved to chain.npz (LOAD_CHAIN replots a saved run).
  6) Checks (arc angle, Gelman-Rubin, exact importance check), summary table and figures:
     residuals, traces, autocorrelation, corners (deviations, RTN, absolute C/S, derived
     C20 / J22 / lambda22 / phi0, each with and without the prior).
  7) Posterior split (GMM) + per-component STM propagation to gmm_target_hours (past the
     arc), checked against an exact Monte Carlo of posterior samples (gmm_prediction, the GMM
     block of scenario_NGSR.py): fig_gmm_stm_propagation (+ _rtn), fig_gmm_prediction_history.

Units: km, km/s, s.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sympy as sp
import trimesh
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import kurtosis, norm, skew, uniform

# The script lives in extra/: the project modules are in the parent folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MCMCPlots
from BatchLSQ import BatchLSQ
from get_CS_unnormalized import bennu_10_10
from GMM import GMM, position_prediction_history
from MCMC import MCMCModel
from STTPropagationND import STTPropagatorND

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved


# ============================================================
# Geometry / measurements
# ============================================================


def wrap_to_pi(a):
    """Angle(s) wrapped to (-pi, pi]."""
    return np.pi - np.mod(np.pi - np.asarray(a, dtype=float), 2.0 * np.pi)


def _Rz_frame(th):
    """Frame (passive) rotation about z."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _Rx_frame(th):
    """Frame (passive) rotation about x."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _Rz_act(th):
    """Active rotation about z."""
    return _Rz_frame(th).T


def _Rx_act(th):
    """Active rotation about x."""
    return _Rx_frame(th).T


def make_bennu_rotation_matrix(alpha, delta, omega, t, w0=0.0):
    """
    Inertial -> body-fixed rotation of an IAU constant-spin model (frame rotations):

        R_ib = R3(W) R1(pi/2 - delta) R3(alpha + pi/2),   W = w0 + omega t

    Body +Z is the pole (cos d cos a, cos d sin a, sin d) and the body spins at +omega.
    """
    return _Rz_frame(w0 + omega * t) @ _Rx_frame(np.pi / 2 - delta) @ _Rz_frame(alpha + np.pi / 2)


def occultation_mask(p1, p2, R_body):
    """
    True where the segment p1 -> p2 (each (N, 3)) misses the sphere of radius R_body at the
    origin, i.e. where the link is visible.
    """
    d = p2 - p1
    t = np.clip(-np.sum(p1 * d, axis=-1) / np.sum(d * d, axis=-1), 0.0, 1.0)
    return np.linalg.norm(p1 + t[..., None] * d, axis=-1) > R_body


def range_rangerate_and_partials(rel):
    """
    Range and range-rate of the relative state rel = CubeSat - MC, shape (N, 6).

    Returns
    -------
    rho, rhodot : ndarray, shape (N,)
    d_rho : ndarray, shape (N, 3)
        Partials of the range with respect to the relative position.
    d_rhodot : ndarray, shape (N, 6)
        Partials of the range-rate with respect to the relative position and velocity.
    """
    d = rel[:, :3]
    v = rel[:, 3:6]
    rho = np.linalg.norm(d, axis=1)
    u = d / rho[:, None]
    rhodot = np.sum(u * v, axis=1)
    d_rhodot = np.hstack([(v - rhodot[:, None] * u) / rho[:, None], u])
    return rho, rhodot, u, d_rhodot


def isl_residuals(x6, mc_state, y_obs, sigma, obs_weights):
    """
    Normalized, visibility-weighted ISL residuals [rho_0, rhodot_0, rho_1, ...].

    Parameters
    ----------
    x6 : ndarray, shape (..., N, >=6)
        Inertial CubeSat states at the N epochs (any leading sample axes).
    mc_state : ndarray, shape (N, 6)
    y_obs : ndarray, shape (2N,)
    sigma : (float, float)
        Range and range-rate noise.
    obs_weights : ndarray, shape (2N,)

    Returns
    -------
    ndarray, shape (..., 2N)
    """
    rel = np.asarray(x6)[..., :6] - mc_state
    d, v = rel[..., :3], rel[..., 3:6]
    rho = np.linalg.norm(d, axis=-1)
    rhodot = np.sum(d * v, axis=-1) / rho
    res = np.empty(rel.shape[:-2] + (y_obs.size,))
    res[..., 0::2] = (y_obs[0::2] - rho) / sigma[0]
    res[..., 1::2] = (y_obs[1::2] - rhodot) / sigma[1]
    return res * obs_weights


# ============================================================
# Degree-2 gravity: state X = [r v GM C20 C21 S21 C22 S22], inertial, rotating body
# ============================================================

# 12-state indices of the 9 field variables (r, GM, C20 ... S22)
_W_IDX = np.array([0, 1, 2, 6, 7, 8, 9, 10, 11])

# Q = r_b^T M r_b with M = sum_i c_i _MB[i], c = (C20, C21, S21, C22, S22):
# Q = C20 (z^2 - (x^2 + y^2)/2) + 3 C21 xz + 3 S21 yz + 3 C22 (x^2 - y^2) + 6 S22 xy
_MB = np.zeros((5, 3, 3))
_MB[0] = np.diag([-0.5, -0.5, 1.0])
_MB[1][0, 2] = _MB[1][2, 0] = 1.5
_MB[2][1, 2] = _MB[2][2, 1] = 1.5
_MB[3] = np.diag([3.0, -3.0, 0.0])
_MB[4][0, 1] = _MB[4][1, 0] = 3.0


def _degree2_lambdas(R_ref, max_order):
    """
    Body-frame acceleration of U = GM (1/r + R_ref^2 Q / r^5) and its derivatives with respect
    to w = (x, y, z, GM, C20, C21, S21, C22, S22), up to max_order, as numpy functions of w.

    Entry k returns the flattened array D_k[w_k, ..., w_1, i] (derive_by_array prepends the new
    variable axis; the last axis is the acceleration component).
    """
    x, y, z, gm, c20, c21, s21, c22, s22 = sp.symbols(
        "x y z gm c20 c21 s21 c22 s22", real=True
    )
    w = [x, y, z, gm, c20, c21, s21, c22, s22]
    r2 = x**2 + y**2 + z**2
    Q = (
        c20 * (z**2 - (x**2 + y**2) / 2)
        + 3 * c21 * x * z
        + 3 * s21 * y * z
        + 3 * c22 * (x**2 - y**2)
        + 6 * s22 * x * y
    )
    U = gm * (r2 ** sp.Rational(-1, 2) + R_ref**2 * Q * r2 ** sp.Rational(-5, 2))
    a = [sp.diff(U, v) for v in (x, y, z)]
    funcs = [sp.lambdify(w, a, modules="numpy", cse=True)]
    D = sp.Array(a)
    for _ in range(max_order):
        D = sp.derive_by_array(D, w)
        flat = list(D.reshape(int(np.prod(D.shape))))
        funcs.append(sp.lambdify(w, flat, modules="numpy", cse=True))
    return funcs


class Degree2Gravity:
    """
    Inertial 12-D dynamics X = [r v GM C20 C21 S21 C22 S22] about a uniformly rotating body
    with a degree-2 field, for STTPropagatorND (f_func, A_func and B(k) take (*x, t)).

    The body-frame derivatives are sympy-exact; they are rotated to the inertial frame
    numerically (block diag(R_ib, I) on every variable axis). One evaluation per (t, x) is
    cached, so f, A and B_k of the same right-hand side call share it.

    Parameters
    ----------
    R_ref : float
        Reference radius [km].
    alpha, delta, omega, w0 : float
        Pole [rad], spin rate [rad/s] and prime-meridian angle at t = 0 [rad].
    max_order : int
        Highest derivative order evaluated (1: f and A; 2: also B_2; ...).
    """

    def __init__(self, R_ref, alpha, delta, omega, w0=0.0, max_order=1):
        self.R_ref = float(R_ref)
        self.alpha, self.delta, self.omega, self.w0 = alpha, delta, omega, w0
        self.max_order = int(max_order)
        self.M0 = _Rx_frame(np.pi / 2 - delta) @ _Rz_frame(alpha + np.pi / 2)
        self.pole = self.M0[2].copy()  # inertial spin axis (body +Z)
        self._funcs = _degree2_lambdas(self.R_ref, self.max_order)
        self._key, self._val, self._order = None, None, -1

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_funcs"], d["_key"], d["_val"], d["_order"] = None, None, None, -1
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._funcs = _degree2_lambdas(self.R_ref, self.max_order)

    def R_ib(self, t):
        """Inertial -> body-fixed rotation at t."""
        return _Rz_frame(self.w0 + self.omega * t) @ self.M0

    def _eval(self, x, t, order):
        key = (float(t), x.tobytes())
        if key == self._key and self._order >= order:
            return self._val
        Rm = self.R_ib(t)
        wb = (*(Rm @ x[0:3]), x[6], *x[7:12])
        L = np.eye(9)
        L[0:3, 0:3] = Rm
        out = {}
        f = np.zeros(12)
        f[0:3] = x[3:6]
        f[3:6] = Rm.T @ np.asarray(self._funcs[0](*wb), dtype=float).ravel()
        out[0] = f
        for k in range(1, order + 1):
            Db = np.asarray(self._funcs[k](*wb), dtype=float).reshape((9,) * k + (3,))
            Di = np.tensordot(Rm.T, np.moveaxis(Db, -1, 0), axes=(1, 0))
            for _ in range(k):
                Di = np.tensordot(Di, L, axes=(1, 0))  # body -> inertial variable, cyclic
            T = np.zeros((12,) * (k + 1))
            T[np.ix_(np.arange(3, 6), *([_W_IDX] * k))] = Di
            if k == 1:
                T[0:3, 3:6] = np.eye(3)
            out[k] = T
        self._key, self._val, self._order = key, out, order
        return out

    def f_func(self, *args):
        return self._eval(np.asarray(args[:12], dtype=float), args[12], 0)[0]

    def A_func(self, *args):
        return self._eval(np.asarray(args[:12], dtype=float), args[12], self.max_order)[1]

    def B(self, k):
        """Picklable B_k callable for STTPropagatorND(B_funcs={k: gravity.B(k)})."""
        if not 2 <= k <= self.max_order:
            raise ValueError(f"B_{k} needs 2 <= k <= max_order = {self.max_order}")
        return _DerivativeTensor(self, k)

    def acceleration_body(self, rb, gm, cs):
        """Body-frame acceleration, vectorized: rb (n, 3), gm (n,), cs (n, 5) -> (n, 3)."""
        M = np.einsum("ni,ijk->njk", cs, _MB)
        r = np.linalg.norm(rb, axis=1)[:, None]
        Mr = np.einsum("njk,nk->nj", M, rb)
        Q = np.sum(rb * Mr, axis=1)[:, None]
        a = -rb / r**3 + self.R_ref**2 * (2.0 * Mr / r**5 - 5.0 * Q * rb / r**7)
        return gm[:, None] * a

    def propagate_many(self, X0s, t_eval, method="DOP853", rtol=1e-11, atol=1e-14):
        """
        Exact propagation of many 12-D initial states at once (one stacked solve_ivp).

        Returns
        -------
        ndarray, shape (n, len(t_eval), 6)
            Inertial r, v at t_eval.
        """
        X0s = np.atleast_2d(np.asarray(X0s, dtype=float))
        n = X0s.shape[0]
        gm, cs = X0s[:, 6], X0s[:, 7:12]

        def rhs(t, Y):
            Y = Y.reshape(n, 6)
            Rm = self.R_ib(t)
            out = np.empty_like(Y)
            out[:, 0:3] = Y[:, 3:6]
            out[:, 3:6] = self.acceleration_body(Y[:, 0:3] @ Rm.T, gm, cs) @ Rm
            return out.ravel()

        sol = solve_ivp(
            rhs,
            (t_eval[0], t_eval[-1]),
            X0s[:, 0:6].ravel(),
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if sol.status != 0:
            raise RuntimeError(f"propagate_many failed: {sol.message}")
        return sol.y.reshape(n, 6, -1).transpose(0, 2, 1)


class _DerivativeTensor:
    """B_k(*x, t) of a Degree2Gravity (a class, not a closure, so that it pickles)."""

    def __init__(self, gravity, k):
        self.gravity, self.k = gravity, k

    def __call__(self, *args):
        g = self.gravity
        return g._eval(np.asarray(args[:12], dtype=float), args[12], g.max_order)[self.k]


# ============================================================
# Symmetry chart (coordinates sampled by the MCMC)
# ============================================================


class SymmetryChart:
    """
    Coordinates theta adapted to the rotation symmetry g_delta about the spin axis.

    g_delta rotates r and v by delta about the inertial pole, (C21, S21) by delta and
    (C22, S22) by 2 delta. With e_rho the unit vector from the axis to r_ref, e_phi =
    pole x e_rho and rho_ref the distance of r_ref from the axis:

        s  = theta_pos . e_phi                 (arc length along the symmetry orbit)
        delta = s / rho_ref
        ds = theta with its e_phi position component removed   (deviation on the slice)
        x  = g_delta(x_ref + ds),   dx = x - x_ref

    The slice is the half-plane through the axis and r_ref; the map is one-to-one for
    |delta| < pi and ds.e_rho > -rho_ref, with log|det dx/dtheta| = log(1 + ds.e_rho / rho_ref).
    All methods work on arrays with any leading axes (..., 12).

    Parameters
    ----------
    x_ref : array_like, shape (12,)
        Reference state (the batch solution x0_ref1).
    pole : array_like, shape (3,)
        Inertial spin axis.
    """

    def __init__(self, x_ref, pole):
        self.x_ref = np.asarray(x_ref, dtype=float).copy()
        self.pole = np.asarray(pole, dtype=float) / np.linalg.norm(pole)
        r = self.x_ref[:3]
        r_perp = r - (r @ self.pole) * self.pole
        self.rho_ref = float(np.linalg.norm(r_perp))
        self.e_rho = r_perp / self.rho_ref
        self.e_phi = np.cross(self.pole, self.e_rho)

    def _rot3(self, v, c, s):
        """Rodrigues rotation of vectors v (..., 3) about the pole, cos / sin arrays (...)."""
        p = self.pole
        return v * c[..., None] + np.cross(p, v) * s[..., None] + p * ((v @ p) * (1.0 - c))[..., None]

    def rotate(self, X, ang):
        """g_ang(X): r, v (and C21..S22 if X has 12 columns) rotated by ang (broadcast)."""
        X = np.asarray(X, dtype=float)
        ang = np.broadcast_to(np.asarray(ang, dtype=float), X.shape[:-1])
        c, s = np.cos(ang), np.sin(ang)
        out = X.copy()
        out[..., 0:3] = self._rot3(X[..., 0:3], c, s)
        out[..., 3:6] = self._rot3(X[..., 3:6], c, s)
        if X.shape[-1] >= 12:
            c2, s2 = np.cos(2.0 * ang), np.sin(2.0 * ang)
            out[..., 8] = c * X[..., 8] - s * X[..., 9]
            out[..., 9] = s * X[..., 8] + c * X[..., 9]
            out[..., 10] = c2 * X[..., 10] - s2 * X[..., 11]
            out[..., 11] = s2 * X[..., 10] + c2 * X[..., 11]
        return out

    def angle(self, theta):
        """delta = theta_pos . e_phi / rho_ref."""
        return (np.asarray(theta, dtype=float)[..., :3] @ self.e_phi) / self.rho_ref

    def slice_deviation(self, theta):
        """ds: theta without its e_phi position component."""
        ds = np.array(theta, dtype=float)
        ds[..., :3] -= (ds[..., :3] @ self.e_phi)[..., None] * self.e_phi
        return ds

    def to_deviation(self, theta):
        """dx = g_delta(x_ref + ds) - x_ref."""
        return self.rotate(self.x_ref + self.slice_deviation(theta), self.angle(theta)) - self.x_ref

    def from_deviation(self, dx):
        """Inverse of to_deviation (delta = azimuth of r about the pole, from r_ref)."""
        x = self.x_ref + np.asarray(dx, dtype=float)
        ang = np.arctan2(x[..., :3] @ self.e_phi, x[..., :3] @ self.e_rho)
        theta = self.rotate(x, -ang) - self.x_ref
        theta[..., :3] += (ang * self.rho_ref)[..., None] * self.e_phi
        return theta

    def azimuth(self, dx):
        """Arc angle delta of absolute deviations dx (the angle part of from_deviation)."""
        x = self.x_ref + np.asarray(dx, dtype=float)
        return np.arctan2(x[..., :3] @ self.e_phi, x[..., :3] @ self.e_rho)

    def log_jacobian(self, theta):
        """log|det d dx / d theta| = log(1 + ds.e_rho / rho_ref) (nan beyond the axis)."""
        a = self.slice_deviation(theta)[..., :3] @ self.e_rho
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log1p(a / self.rho_ref)

    def generator(self, X):
        """d g_delta(X) / d delta at delta = 0."""
        X = np.asarray(X, dtype=float)
        t = np.zeros_like(X)
        t[..., 0:3] = np.cross(self.pole, X[..., 0:3])
        t[..., 3:6] = np.cross(self.pole, X[..., 3:6])
        t[..., 8], t[..., 9] = -X[..., 9], X[..., 8]
        t[..., 10], t[..., 11] = -2.0 * X[..., 11], 2.0 * X[..., 10]
        return t

    def jacobian_at_ref(self):
        """d dx / d theta at theta = 0: I + (generator without position) e_phi^T / rho_ref."""
        t = self.generator(self.x_ref)
        t[:3] = 0.0
        e = np.zeros(self.x_ref.size)
        e[:3] = self.e_phi
        return np.eye(self.x_ref.size) + np.outer(t, e) / self.rho_ref


class IdentityChart:
    """theta = dx (plain sampling, plain STT): the SymmetryChart interface with delta = 0."""

    def __init__(self, x_ref):
        self.x_ref = np.asarray(x_ref, dtype=float).copy()

    def rotate(self, X, ang):
        return np.array(X, dtype=float)

    def angle(self, theta):
        return np.zeros(np.shape(theta)[:-1])

    def slice_deviation(self, theta):
        return np.array(theta, dtype=float)

    def to_deviation(self, theta):
        return np.array(theta, dtype=float)

    def from_deviation(self, dx):
        return np.array(dx, dtype=float)

    def log_jacobian(self, theta):
        return np.zeros(np.shape(theta)[:-1])

    def jacobian_at_ref(self):
        return np.eye(self.x_ref.size)


class MCMCModelChart(MCMCModel):
    """
    MCMCModel whose sampled parameters are chart coordinates theta.

    param_priors are independent priors on the deviation dx = chart.to_deviation(theta)
    (about x0_ref1, as in the other scripts): Gaussians (scipy norm), or boxes (scipy uniform,
    SH_PRIOR = "uniform"); the log-prior in theta adds the chart Jacobian. The log-densities are
    evaluated with numpy (same value as the sum of the logpdf).
    theta is restricted to |delta| < pi: x(theta) is 2 pi periodic in delta, and without the
    cut the MAP search can land on a copy of the posterior one period away.
    """

    def __init__(self, residuals_func, initial_params, param_priors, observed_data, chart):
        super().__init__(residuals_func, initial_params, param_priors, observed_data)
        self.chart = chart
        self._prior_mean = np.array([p.mean() for p in param_priors], dtype=float)
        self._prior_std = np.array([p.std() for p in param_priors], dtype=float)
        self._unif = np.array([p.dist.name == "uniform" for p in param_priors], dtype=bool)
        if self._unif.any():  # boxes [lo, hi] on these, Gaussians on the others
            support = np.array([p.support() for p in param_priors], dtype=float)
            self._lo, self._hi = support[self._unif, 0], support[self._unif, 1]
            g = ~self._unif
            self._prior_norm = -np.sum(np.log(self._prior_std[g] * np.sqrt(2.0 * np.pi))) - np.sum(
                np.log(self._hi - self._lo)
            )
        else:
            self._prior_norm = -np.sum(np.log(self._prior_std * np.sqrt(2.0 * np.pi)))

    def log_prior(self, theta):
        theta = np.asarray(theta, dtype=float)
        if abs(float(self.chart.angle(theta))) >= np.pi:  # one period of g_delta (one-to-one)
            return -np.inf
        lj = float(self.chart.log_jacobian(theta))
        if not np.isfinite(lj):
            return -np.inf
        if self._unif.any():
            dx = self.chart.to_deviation(theta)
            du = dx[self._unif]
            if np.any(du < self._lo) or np.any(du > self._hi):
                return -np.inf
            g = ~self._unif
            z = (dx[g] - self._prior_mean[g]) / self._prior_std[g]
        else:
            z = (self.chart.to_deviation(theta) - self._prior_mean) / self._prior_std
        return self._prior_norm - 0.5 * float(z @ z) + lj

    def optimize_initial_guess(self, *args, **kwargs):
        """
        MCMCModel.optimize_initial_guess without the prior bounds by default, and with a finite
        floor LP_FLOOR on the log-probability while it runs (emcee still sees -inf).

        Outside the support (|delta| >= pi, outside a uniform box) the log-probability is -inf:
        Powell's Brent line search turns it into NaN parabolic steps, and its bounded line
        search (used with bounds) never compares with the current point, so on a box prior,
        whose whitened image spans thousands of posterior sigmas, it returns a point on the
        -inf plateau. Unbounded Brent on the floored objective brackets from the current point.
        """
        kwargs.setdefault("use_bounds_from_priors", False)
        self._lp_floor = self.LP_FLOOR
        try:
            return super().optimize_initial_guess(*args, **kwargs)
        finally:
            self._lp_floor = None

    LP_FLOOR = -1e30
    _lp_floor = None

    def log_prob_whitened(self, theta_white):
        lp = super().log_prob_whitened(theta_white)
        if self._lp_floor is not None and not lp > self._lp_floor:  # also NaN
            return self._lp_floor
        return lp


# ============================================================
# Derived quantities and small helpers
# ============================================================


def derived_quantities(x, M0, ref=None):
    """
    (C20, J22, lambda22, phi0) of absolute states x (..., 12).

    J22 = hypot(C22, S22), lambda22 = atan2(S22, C22) / 2 (longitude of the long axis),
    phi0 = body longitude of r0 at t = 0 (M0 = R_ib(0)). With ref (a 4-vector), the angles are
    unwrapped next to ref (lambda22 has period pi).
    """
    x = np.asarray(x, dtype=float)
    rb = x[..., :3] @ M0.T
    out = np.stack(
        [
            x[..., 7],
            np.hypot(x[..., 10], x[..., 11]),
            0.5 * np.arctan2(x[..., 11], x[..., 10]),
            np.arctan2(rb[..., 1], rb[..., 0]),
        ],
        axis=-1,
    )
    if ref is not None:
        out[..., 2] = ref[2] + 0.5 * wrap_to_pi(2.0 * (out[..., 2] - ref[2]))
        out[..., 3] = ref[3] + wrap_to_pi(out[..., 3] - ref[3])
    return out


def symmetry_step(x, x_prior, prior_sigma, chart):
    """
    Angle delta minimizing the prior cost |(g_delta(x) - x_prior) / prior_sigma|^2 along the
    symmetry orbit of x (grid over (-pi, pi], then a bounded 1-D refinement).

    With the mothercraft on the spin axis the data cost is the same at every g_delta(x), so
    this is an exact MAP step along the valley that Gauss-Newton can only crawl along.

    Returns
    -------
    x_new : ndarray, shape (12,)
    delta : float
    """
    grid = np.linspace(-np.pi, np.pi, 3601)
    cost_grid = np.sum(((chart.rotate(np.tile(x, (grid.size, 1)), grid) - x_prior) / prior_sigma) ** 2, axis=1)
    a0, h = grid[np.argmin(cost_grid)], grid[1] - grid[0]
    res = minimize_scalar(
        lambda a: np.sum(((chart.rotate(x, a) - x_prior) / prior_sigma) ** 2),
        bounds=(a0 - h, a0 + h),
        method="bounded",
        options=dict(xatol=1e-12),
    )
    return chart.rotate(x, res.x), float(res.x)


def exact_map_chart(x_start, res_many, make_chart, cov, h=1e-3, max_nfev=300, max_outer=5, verbose=True):
    """
    Exact MAP by trust-region least squares (scipy trf) in chart coordinates.

    Used with STMs only (STT_ORDER = 1), where the STT re-centering is Gauss-Newton with step
    halving and crawls along the curved MAP valley (a fraction of a MAP-cost unit per
    iteration); trf adapts its trust region instead and follows the valley. Each chart is
    centered on the current state and scaled by cov: theta = L z, L L^T = J^-1 cov J^-T (J the
    chart Jacobian at its reference), so z is of order one. The Jacobian (first-order, STM-like
    information of the exact flow) is a central difference in z: the 24 perturbed states go
    through one res_many call. The chart is re-centered at each solution until the step
    |z| < 1e-6.

    Parameters
    ----------
    x_start : ndarray, shape (12,)
    res_many : callable
        res_many(X) -> ndarray (n, m): measurement and prior residuals of the absolute states
        X (n, 12), exact propagation (one stacked solve).
    make_chart : callable
        make_chart(x) -> SymmetryChart or IdentityChart about x.
    cov : ndarray, shape (12, 12)
        Covariance of the deviation (e.g. the batch covariance), for the scaling.
    h : float
        Difference step in z.
    max_nfev : int
        Function evaluations per least_squares call.
    max_outer : int
        Maximum number of charts.
    verbose : bool

    Returns
    -------
    x : ndarray, shape (12,)
    cost : float
        Sum of squared residuals at x (the MAP cost of the other Stage-1 prints).
    """
    x = np.asarray(x_start, dtype=float).copy()
    cost = float(np.sum(res_many(x) ** 2))
    eye = np.eye(x.size)
    for outer in range(1, max_outer + 1):
        chart = make_chart(x)
        J = np.linalg.inv(chart.jacobian_at_ref())
        P = J @ cov @ J.T
        L = np.linalg.cholesky(0.5 * (P + P.T))

        def states(Z, x=x, chart=chart, L=L):
            return x + np.array([chart.to_deviation(L @ z) for z in np.atleast_2d(Z)])

        def fun(z, states=states):
            return res_many(states(z))[0]

        def jac(z, states=states):
            R = res_many(states(np.vstack([z + h * eye, z - h * eye])))
            return (R[: x.size] - R[x.size :]).T / (2.0 * h)

        sol = least_squares(
            fun, np.zeros(x.size), jac=jac, method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=max_nfev
        )
        x, cost = states(sol.x)[0], 2.0 * float(sol.cost)
        step = float(np.linalg.norm(sol.x))
        if verbose:
            print(
                f"[Stage 1] exact MAP, chart {outer}: cost {cost:.6f} ({sol.nfev} evaluations, "
                f"|z| = {step:.2e}, delta {np.rad2deg(chart.angle(L @ sol.x)):+.3f} deg)"
            )
        if step < 1e-6:
            break
    return x, cost


def numeric_jacobian(fun, x0, h):
    """Central-difference Jacobian of fun at x0 with steps h (one per input)."""
    cols = []
    for i, hi in enumerate(h):
        e = np.zeros_like(x0)
        e[i] = hi
        cols.append((fun(x0 + e) - fun(x0 - e)) / (2.0 * hi))
    return np.column_stack(cols)


def moments_row(name, s, prior_sd, batch_sd):
    """One line of the summary table: sigmas, ratio to the prior, skewness, excess kurtosis."""
    sd = np.std(s)
    return (
        f"  {name:>10s}  prior {prior_sd:10.3e}  batch {batch_sd:10.3e}  MCMC {sd:10.3e}"
        f"  MCMC/prior {sd / prior_sd:7.3f}  skew {skew(s):+6.2f}  ex.kurt {kurtosis(s):+6.2f}"
    )


class _Tee:
    """Write to several streams (screen + run.log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

    def __getattr__(self, name):  # isatty, encoding, ... of the screen stream
        return getattr(self.streams[0], name)


# ============================================================
# Posterior split (GMM) + STM propagation (as scenario_NGSR.py)
# ============================================================


def gmm_prediction(
    samples_dx,
    x0_ref1,
    cov1,
    x0_true,
    propagator,
    propagate_many,
    t_prop,
    t_obs_end,
    ivp,
    titles,
    n_components=8,
    select_bic=False,
    n_mc=1000,
    prob=0.99,
    batch=True,
    rtn=True,
    show_mc=False,
    show_components=False,
    show_epoch=False,
    r_ref=None,
    chart=None,
    seed=2024,
    figure=None,
):
    """
    Split the posterior at the epoch into a Gaussian mixture, propagate every component with
    its own STM and check the predicted mixture against a nonlinear Monte Carlo of posterior
    samples (the GMM block of scenario_NGSR.py, for the 12-D degree-2 state):

        m_k(t) = phi(t; x0_ref1 + mu_k),      P_k(t) = Phi_k(t) P_k Phi_k(t)^T.

    By default the mixture is fit to the Cartesian deviations dx, which the STMs map. The
    crescent along g_delta then takes many components: a few straight ones put part of its sag
    into the stiff mean-motion direction, and the prediction comes out too wide (with K = 8 about
    4x the Monte Carlo spread at 48 h, close with K = 32). With chart, the mixture is fit to the
    chart coordinates theta instead, where the crescent is straight, and each component is
    mapped to dx by the chart Jacobian at its mean. The prediction then has the Monte Carlo size
    even with K = 8, because the STMs carry the symmetry generator exactly, but the thin mapped
    components do not follow the sag, so the mixture log-density stays poor. With batch, the
    Stage-1 Gaussian N(0, cov1) (STM about ref1) is propagated the same way. The Monte Carlo
    propagates n_mc posterior samples exactly in one stacked solve (propagate_many).

    Parameters
    ----------
    samples_dx : ndarray, shape (M, 12)
        Posterior deviations about x0_ref1 at t = 0.
    x0_ref1, x0_true : ndarray, shape (12,)
        Batch solution (reference of the deviations) and truth.
    cov1 : ndarray, shape (12, 12)
        Batch covariance at x0_ref1.
    propagator : STTPropagatorND
        Order-1 propagator (states + STMs), for GMM.propagate_stm.
    propagate_many : callable
        propagate_many(X0s (n, 12), t_eval, **ivp) -> (n, N_t, 6): exact r, v
        (Degree2Gravity.propagate_many).
    t_prop : ndarray, shape (N_t,)
        Output epochs [s]; t_prop[0] = 0, the epoch of the samples.
    t_obs_end : float
        End of the observation arc [s] (shaded in fig_gmm_prediction_history).
    ivp : dict
        solve_ivp options (method, rtol, atol) of all propagations.
    titles : (str, str)
        Row titles of the epoch and of the target panels.
    n_components, select_bic : int, bool
        GMM.fit settings (fixed K, or K in 1..12 by BIC).
    n_mc : int
        Posterior samples propagated exactly.
    prob : float
        Probability mass of the plotted regions.
    batch : bool
        Also draw the propagated batch Gaussian N(0, cov1) and print its metrics.
    rtn : bool
        Also save fig_gmm_stm_propagation_rtn (RTN frame of the propagated batch solution at
        each row's epoch).
    show_mc, show_components : bool
        Draw the MC cloud / every component's ellipse in the propagation figures.
    show_epoch : bool
        Also draw the posterior at the epoch (top row) in the propagation figures; default:
        only the prediction at t_prop[-1].
    r_ref : float, optional
        Reference radius of the field: warn if the truth or a MC sample comes within 5 % of it
        along the prediction.
    chart : SymmetryChart, optional
        Fit the mixture in its coordinates theta = chart.from_deviation(dx) (see above);
        default: fit dx.
    seed : int
        Seed of the MC subsample.
    figure : callable, optional
        figure(label, draw) wrapping each figure (e.g. the scripts' _figure, which reports and
        skips a plotting error); default: draw() directly.

    Returns
    -------
    dict
        gmm (GMM at t = 0), gmm_t, batch_t (lists of GMM, one per output epoch; batch_t None
        without batch and rtn), X_mc (n_mc, N_t, 6), x_true_prop (N_t, 6), hist
        (GMM.position_prediction_history).
    """
    figure = figure or (lambda label, draw: draw())
    post = np.asarray(samples_dx, dtype=float)
    x0_ref1 = np.asarray(x0_ref1, dtype=float)
    t_prop = np.asarray(t_prop, dtype=float)
    pos = slice(0, 3)
    print(
        f"\n[GMM] Splitting {post.shape[0]} posterior samples at the epoch and propagating to "
        f"+{t_prop[-1] / 3600.0:.2f} h (observation arc 0-{t_obs_end / 3600.0:.2f} h)"
    )
    if chart is None:
        gmm = GMM.fit(post, n_components=n_components, select_bic=select_bic)
    else:
        # Fit in chart coordinates; component k -> N(dx(mu_k), J_k P_k J_k^T), J_k = d dx / d theta
        theta = chart.from_deviation(post)
        g_th = GMM.fit(theta, n_components=n_components, select_bic=select_bic)
        h = 1e-4 * theta.std(axis=0)
        jac = [numeric_jacobian(chart.to_deviation, mu, h) for mu in g_th.means]
        covs = np.array([J @ P @ J.T for J, P in zip(jac, g_th.covs)])
        gmm = GMM(g_th.weights, chart.to_deviation(g_th.means), 0.5 * (covs + np.transpose(covs, (0, 2, 1))))
        print("[GMM] fit in chart coordinates, components mapped to dx by the chart Jacobian")
    print("[GMM] weights:", np.array2string(gmm.weights, precision=3))

    # One GMM per output epoch (absolute states)
    gmm_t = gmm.propagate_stm(propagator, x0_ref1, t_prop, **ivp)
    # Batch LSQ Gaussian: mean ref1 (delta 0) on its nonlinear trajectory, cov1 mapped by the
    # STM about ref1. Its mean also defines the RTN frame of fig_gmm_stm_propagation_rtn.
    batch_t = (
        GMM.gaussian(np.zeros(x0_ref1.size), cov1).propagate_stm(
            propagator, x0_ref1, t_prop, label="Batch LSQ", **ivp
        )
        if batch or rtn
        else None
    )

    rng = np.random.default_rng(seed)
    idx = rng.choice(post.shape[0], size=min(n_mc, post.shape[0]), replace=False)
    print(f"[MC] Propagating {idx.size} posterior samples exactly (one stacked solve)...")
    X_mc = propagate_many(x0_ref1 + post[idx], t_prop, **ivp)
    x_true_prop = propagate_many(np.asarray(x0_true, dtype=float)[None], t_prop, **ivp)[0]

    # The degree-2 field is only valid outside the reference sphere
    if r_ref is not None:
        r_tr = float(np.linalg.norm(x_true_prop[:, :3], axis=-1).min())
        r_mc = float(np.linalg.norm(X_mc[:, :, :3], axis=-1).min())
        print(
            f"[GMM] closest approach along the prediction: truth {r_tr:.3f} km, MC samples "
            f"{r_mc:.3f} km (R_ref {r_ref} km)"
        )
        if min(r_tr, r_mc) < 1.05 * r_ref:
            print("[WARNING] The prediction comes within 5 % of R_ref (field not valid inside)")

    # Metrics at the target epoch (position, meters)
    pos_mc = X_mc[:, -1, pos]
    m_mc = pos_mc.mean(axis=0)
    gmm_T = gmm_t[-1].marginal(pos)
    mg_T, Pg_T = gmm_T.moments()
    print(f"\n[GMM] Position at +{t_prop[-1] / 3600.0:.2f} h, per-axis 1-sigma [m]:")
    print("   MC             :", np.array2string(np.std(pos_mc, axis=0, ddof=1) * 1e3, precision=2))
    print("   GMM + STMs     :", np.array2string(np.sqrt(np.diag(Pg_T)) * 1e3, precision=2))
    print("[GMM] Mean offset from the MC mean [m]:")
    print("   GMM + STMs     :", np.array2string((mg_T - m_mc) * 1e3, precision=2))
    ll_g = np.mean(gmm_T.logpdf(pos_mc))
    print(f"[GMM] Mean log-density of the MC positions: GMM {ll_g:.3f}")
    r_tr_T = x_true_prop[-1, pos]
    print(f"[GMM] Truth at target: log p GMM {gmm_T.logpdf(r_tr_T)[0]:.3f}")
    if batch:
        batch_T = batch_t[-1].marginal(pos)
        mb_T, Pb_T = batch_T.moments()
        d_trb = r_tr_T - mb_T
        print("[GMM] Batch LSQ + STM at target (same metrics):")
        print("   1-sigma [m]          :", np.array2string(np.sqrt(np.diag(Pb_T)) * 1e3, precision=2))
        print("   mean - MC mean [m]   :", np.array2string((mb_T - m_mc) * 1e3, precision=2))
        ll_b = np.mean(batch_T.logpdf(pos_mc))
        print(
            f"   mean log-density of the MC positions {ll_b:.3f} (GMM - batch = "
            f"{ll_g - ll_b:+.3f} nats) | truth log p {batch_T.logpdf(r_tr_T)[0]:.3f} | truth "
            f"Mahalanobis {float(np.sqrt(d_trb @ np.linalg.solve(Pb_T, d_trb))):.2f}"
        )

    # Figures
    def _panel(j, title, C=None):
        """Position panel at output epoch j; C (3, 3) rotates it (e.g. to RTN)."""
        p = dict(
            mc=X_mc[:, j, pos],
            gmm=gmm_t[j].marginal(pos),
            truth=x_true_prop[j, pos],
            center=X_mc[:, j, pos].mean(axis=0),
            title=title,
        )
        if batch:
            p["batch"] = batch_t[j].marginal(pos)
        if C is not None:
            p.update(
                mc=p["mc"] @ C.T,
                truth=C @ p["truth"],
                center=C @ p["center"],
                **{k: p[k].linear_map(C) for k in ("gmm", "batch") if k in p},
            )
        return p

    # Rows: the prediction at the target epoch, after the posterior at the epoch with show_epoch
    title_0, title_T = titles
    rows = [(0, title_0), (-1, title_T)] if show_epoch else [(-1, title_T)]
    rtn_title = {
        0: ", RTN Frame of the Batch Solution",
        -1: ", RTN Frame of the Propagated Batch Solution",
    }
    figure(
        "GMM propagation",
        lambda: MCMCPlots.plot_gmm_propagation(
            [_panel(j, title) for j, title in rows],
            prob=prob,
            show_mc=show_mc,
            show_components=show_components,
        ),
    )
    if rtn:
        figure(
            "GMM propagation RTN",
            lambda: MCMCPlots.plot_gmm_propagation(
                [
                    _panel(j, title + rtn_title[j], C=MCMCPlots.rtn_matrix(batch_t[j].means[0]))
                    for j, title in rows
                ],
                prob=prob,
                name="fig_gmm_stm_propagation_rtn",
                axis_names=[r"$\delta r_R$ [m]", r"$\delta r_T$ [m]", r"$\delta r_N$ [m]"],
                show_mc=show_mc,
                show_components=show_components,
            ),
        )
    hist = position_prediction_history(X_mc, gmm_t, batch_t=batch_t if batch else None)
    figure(
        "GMM prediction history",
        lambda: MCMCPlots.plot_prediction_history(t_prop / 3600.0, hist, t_obs_end / 3600.0),
    )
    return dict(
        gmm=gmm,
        gmm_t=gmm_t,
        batch_t=batch_t,
        X_mc=X_mc,
        x_true_prop=x_true_prop,
        hist=hist,
    )


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent

    # --------------------------
    # Settings
    # --------------------------
    STT_ORDER = 1  # surrogate order: 1 = STMs only, 2 = second-order STTs (on the slice; folder + _stt2)
    USE_SYMMETRY_CHART = True  # False: sample dx with the plain STT (fails along the arc)
    SH_PRIOR = "gaussian"  # "uniform": boxes on C20, C21, S21, C22, S22 (folder + _uniform)
    UNIFORM_HALF_WIDTH = 3.0  # box half-width in prior sigmas, centered on the a priori x0_ref
    # Saved chain: every run writes chain.npz into its results folder. Set LOAD_CHAIN to such
    # a file to skip the sampling and redo the diagnostics and figures from it.
    LOAD_CHAIN = None  # e.g. "extra/results/scenario_ISL_SH2_<stamp>/chain.npz"
    CORNER_RTN = True
    # Mark the posterior mode (MCMCModel.find_map, an optimum, not the best sample) and the
    # batch estimate ref1 on every corner plot.
    SHOW_MAP = True
    SHOW_BATCH_MEAN = True

    # Bennu (unnormalized degree-2 coefficients of the OSIRIS-REx field)
    R_ref = 0.290  # km
    GM_true, C_unn, S_unn = bennu_10_10()
    cs_true = np.array([C_unn[2, 0], C_unn[2, 1], S_unn[2, 1], C_unn[2, 2], S_unn[2, 2]])
    alpha_true = np.deg2rad(85.65)
    delta_true = np.deg2rad(-60.17)
    omega_true = 2.0 * np.pi / (4.296057 * 3600.0)
    SHAPE_PATH = HERE.parent / "data" / "kernels" / "orex" / "bennu_radar_v20.obj"  # orbit scene figure only

    # Mothercraft: MC_DIST km from the center, on the spin axis (MC_TILT_DEG = 0) or tilted
    # from it towards body longitude MC_TILT_AZ_DEG (at t = 0); fixed in inertial space.
    MC_DIST = 10.0  # km
    MC_TILT_DEG = 0.0
    MC_TILT_AZ_DEG = 0.0

    # CubeSat: circular (point-mass speed) orbit in the body-equatorial frame at t = 0
    ORBIT_RADIUS = 0.4  # km
    ORBIT_INC_DEG = 90.0
    ORBIT_RAAN_DEG = 0.0
    ORBIT_U0_DEG = 0.0  # argument of latitude at t = 0

    # Tracking: two-way ISL range / range-rate
    ARC_HOURS = 24.0
    DT_OBS_MIN = 10.0
    sigma_range = 1e-3  # km (1 m)
    sigma_range_rate = 1e-7  # km/s (0.1 mm/s)

    # Prior (1 sigma) on x0 = [r0, v0, GM, C20, C21, S21, C22, S22]
    sig_prior_r = 0.1  # km
    sig_prior_v = 2e-5  # km/s (2 cm/s)
    sig_prior_gm = 0.05 * GM_true  # 5 %
    sig_prior_c20 = 0.5 * abs(cs_true[0])  # 50 %
    sig_prior_c2m = 3e-3  # C21, S21, C22, S22 (J22 = 2.1e-3: not known a priori)

    # Integration (truth, batch, STTs)
    ivp = dict(method="DOP853", rtol=1e-11, atol=1e-14)
    STAGE1_ARCS_H = [2.0, 4.0, 8.0, 16.0, ARC_HOURS]  # batch continuation (see Stage 1)
    STAGE1_ROUNDS = 3  # max symmetry step + Gauss-Newton rounds per arc
    STAGE1_ITERS = 10  # max Gauss-Newton iterations per round
    STAGE1_STT_ITERS = 10  # max STT re-centering iterations (end of Stage 1)
    STAGE1_STT_FTOL = 1e-7  # stop when the exact MAP cost decreases by less (relative)

    # MCMC
    n_walkers = 96
    # tau: ~500-700 steps for C20, C22, S22, lambda22; ~2300-3300 for GM, J22, phi0 - lambda22
    # (curved mean-motion valley): 60000 steps = 75-110 tau / 18-25 tau. For final figures
    # use ~150000 (R_hat of the velocity / GM / C22 chart coordinates was 1.12-1.18 at 60000).
    n_samples = 60000
    burn_in = None  # None: 2 x max(tau)
    thin = None  # None: 0.5 x min(tau)
    spherical_spread = 0.5  # walker start, whitened by the batch covariance in theta
    SEED_WALKERS = 7
    N_EXACT_CHECK = 1000  # posterior samples propagated exactly (importance check)

    # Posterior split (GMM) + STM propagation past the arc (as scenario_NGSR.py)
    GMM_PROPAGATION = True
    gmm_target_hours = 48.0  # [h since the epoch]; the arc ends at ARC_HOURS
    # K = 8 (as NGSR) puts part of the crescent's sag into the mean-motion direction: ~4x the
    # Monte Carlo spread at 48 h. K = 32 is within 1.5x (128 vs 84 m at 48 h), ~2 min.
    gmm_n_components = 32
    gmm_select_bic = False  # True: K in 1..12 by BIC
    GMM_FIT_CHART = False  # True: fit in chart coordinates (right spread, poor log-density)
    n_mc_prop = 1000  # posterior samples propagated exactly (one stacked solve)
    n_t_prop = 97  # output epochs (every 30 min to 48 h)
    gmm_prob = 0.99
    GMM_BATCH = True  # also propagate the batch Gaussian N(0, cov1)
    GMM_RTN = True  # also save the RTN version of fig_gmm_stm_propagation
    GMM_SHOW_MC = False
    GMM_SHOW_COMPONENTS = False
    GMM_SHOW_EPOCH = False  # True: also the posterior at the epoch (top row) in fig_gmm_stm_propagation(_rtn)

    # Figures: parameter symbols and display units (km -> m, km/s -> mm/s, km^3/s^2 ->
    # m^3/s^2; rescaled by powers of ten from the batch sigmas after Stage 1)
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
    MCMCPlots.configure(
        PARAM_SYM,
        PARAM_BASE_UNIT,
        PARAM_BASE_SCALE,
        rotation=make_bennu_rotation_matrix,
        body="Bennu",
        target="CubeSat",
        obs_word="Epoch",
        site_label="Deployment Site",
        resid_labels=("Range", "Range-Rate"),
        fallback_dir=Path.home() / "bennu_figures_fallback",
    )
    # Folder name: <script name>_YYMMDDHHMMSS (+ _stt<order> with STT_ORDER > 1, + _plain without
    # the symmetry chart, + _uniform with the box priors)
    fig_dir = MCMCPlots.set_figure_dir(
        HERE
        / "results"
        / (
            Path(__file__).stem
            + "_"
            + datetime.now().strftime("%y%m%d%H%M%S")
            + (f"_stt{STT_ORDER}" if STT_ORDER > 1 else "")
            + ("" if USE_SYMMETRY_CHART else "_plain")
            + ("_uniform" if SH_PRIOR == "uniform" else "")
        )
    )
    # Screen output also goes to run.log in the results folder
    _run_log = open(Path(fig_dir) / "run.log", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _run_log)
    sys.stderr = _Tee(sys.__stderr__, _run_log)
    print(f"[Figures] -> {fig_dir}")

    def _figure(label, draw):
        """Draw one figure; a plotting error is reported and skipped."""
        try:
            draw()
        except Exception as err:
            print(f"[Figure] {label} skipped: {type(err).__name__}: {err}")

    # --------------------------
    # Dynamics and propagators
    # --------------------------
    print("\n[Setup] Building the degree-2 field derivatives (sympy)...")
    gravity1 = Degree2Gravity(R_ref, alpha_true, delta_true, omega_true, max_order=1)
    gravity2 = (
        gravity1
        if STT_ORDER == 1
        else Degree2Gravity(R_ref, alpha_true, delta_true, omega_true, max_order=STT_ORDER)
    )
    prop1 = STTPropagatorND(order=1, f_func=gravity1.f_func, A_func=gravity1.A_func, n=12)
    prop2 = STTPropagatorND(
        order=STT_ORDER,
        f_func=gravity2.f_func,
        A_func=gravity2.A_func,
        B_funcs={k: gravity2.B(k) for k in range(2, STT_ORDER + 1)},
        n=12,
    )
    M0 = gravity1.M0  # R_ib(0)
    pole_i = gravity1.pole

    # --------------------------
    # Truth: CubeSat initial state, MC position, measurements
    # --------------------------
    u0 = np.deg2rad(ORBIT_U0_DEG)
    Q_orb = _Rz_act(np.deg2rad(ORBIT_RAAN_DEG)) @ _Rx_act(np.deg2rad(ORBIT_INC_DEG))
    v_circ = np.sqrt(GM_true / ORBIT_RADIUS)
    r0_true = M0.T @ Q_orb @ (ORBIT_RADIUS * np.array([np.cos(u0), np.sin(u0), 0.0]))
    v0_true = M0.T @ Q_orb @ (v_circ * np.array([-np.sin(u0), np.cos(u0), 0.0]))
    x0_true = np.concatenate([r0_true, v0_true, [GM_true], cs_true])

    tilt, tilt_az = np.deg2rad(MC_TILT_DEG), np.deg2rad(MC_TILT_AZ_DEG)
    mc_dir_b = np.array([np.sin(tilt) * np.cos(tilt_az), np.sin(tilt) * np.sin(tilt_az), np.cos(tilt)])
    r_mc = MC_DIST * (M0.T @ mc_dir_b)

    tau = np.arange(0.0, ARC_HOURS * 3600.0 + 1e-6, DT_OBS_MIN * 60.0)
    N = tau.size
    mc_state = np.tile(np.concatenate([r_mc, np.zeros(3)]), (N, 1))

    print("\n[Truth] Propagating the CubeSat...")
    sol_true = prop1.propagate_state_only(x0_true, tau, show_progress=False, **ivp)
    x_true = sol_true.y[:12].T
    tau_dense = np.linspace(0.0, tau[-1], 2001)
    x_dense = prop1.propagate_state_only(x0_true, tau_dense, show_progress=False, **ivp).y[:12].T
    r_dense = np.linalg.norm(x_dense[:, :3], axis=1)
    vis_mask = occultation_mask(mc_state[:, :3], x_true[:, :3], R_ref)

    print(f"[Truth] GM = {GM_true:.6e} km^3/s^2, (C20, C21, S21, C22, S22) = {cs_true}")
    J22_true = np.hypot(cs_true[3], cs_true[4])
    print(
        f"[Truth] J22 = {J22_true:.4e}, lambda22 = "
        f"{np.rad2deg(0.5 * np.arctan2(cs_true[4], cs_true[3])):.2f} deg"
    )
    print(
        f"[Truth] CubeSat radius {r_dense.min():.3f}-{r_dense.max():.3f} km (R_ref {R_ref} km), "
        f"speed {v_circ * 1e5:.2f} cm/s, period {2 * np.pi * np.sqrt(ORBIT_RADIUS**3 / GM_true) / 3600:.2f} h"
    )
    print(
        f"[Truth] MC at {MC_DIST} km, {MC_TILT_DEG} deg from the spin axis; "
        f"{vis_mask.sum()}/{N} epochs visible"
    )
    if r_dense.min() < 1.05 * R_ref:
        print("[WARNING] The CubeSat orbit comes within 5 % of R_ref (field not valid inside)")

    rng_meas = np.random.default_rng(123)
    rho_t, rhodot_t, _, _ = range_rangerate_and_partials(x_true[:, :6] - mc_state)
    y_obs = np.empty(2 * N)
    y_obs[0::2] = rho_t + sigma_range * rng_meas.standard_normal(N)
    y_obs[1::2] = rhodot_t + sigma_range_rate * rng_meas.standard_normal(N)
    obs_weights = np.ones(2 * N)
    obs_weights[0::2][~vis_mask] = 0.0
    obs_weights[1::2][~vis_mask] = 0.0
    sigma = (sigma_range, sigma_range_rate)
    n_visible = int(vis_mask.sum())
    dof = 2 * n_visible - 12

    _figure(
        "observation geometry",
        lambda: MCMCPlots.plot_observation_geometry(
            tau,
            mc_state,
            x_true,
            vis_mask,
            tau_line=tau_dense,
            sc_line=np.tile(mc_state[0], (tau_dense.size, 1)),
            pt_line=x_dense,
        ),
    )
    _figure(
        "orbit scene",
        lambda: MCMCPlots.plot_scene_orbit(
            trimesh.load(SHAPE_PATH, force="mesh", process=False),
            tau,
            x_true,
            vis_mask,
            pt_line=x_dense,
            sc_line=np.tile(mc_state[0], (tau_dense.size, 1)),
            alpha=alpha_true,
            delta=delta_true,
            omega=omega_true,
            observer="Mothercraft",
        ),
    )

    # --------------------------
    # A priori reference and priors
    # --------------------------
    prior_sigma = np.array(
        [sig_prior_r] * 3
        + [sig_prior_v] * 3
        + [sig_prior_gm, sig_prior_c20]
        + [sig_prior_c2m] * 4
    )
    rng_ref = np.random.default_rng(42)
    x0_ref = x0_true - prior_sigma * rng_ref.standard_normal(12)
    print("\n[Prior] sigmas:", prior_sigma)
    print("[Reference] truth - a priori, in prior sigmas:", np.round((x0_true - x0_ref) / prior_sigma, 2))
    # Box priors on the degree-2 coefficients (SH_PRIOR = "uniform"): x0_ref +- UNIFORM_HALF_WIDTH
    # prior sigmas. From here on prior_sigma is the sd of each prior (box: half-width / sqrt 3);
    # Stage 1 uses Gaussians with these means and sds, Stage 2 the boxes (priors_ref1).
    is_unif = np.zeros(12, dtype=bool)
    if SH_PRIOR == "uniform":
        is_unif[7:12] = True
        unif_lo = x0_ref - UNIFORM_HALF_WIDTH * prior_sigma
        unif_hi = x0_ref + UNIFORM_HALF_WIDTH * prior_sigma
        prior_sigma = np.where(is_unif, UNIFORM_HALF_WIDTH / np.sqrt(3.0) * prior_sigma, prior_sigma)
        print(
            f"[Prior] uniform on C20, C21, S21, C22, S22: a priori +- {UNIFORM_HALF_WIDTH:g} sigma; "
            f"sd {prior_sigma[7:12]}"
        )
    elif SH_PRIOR != "gaussian":
        raise ValueError(f"SH_PRIOR = {SH_PRIOR!r}: 'gaussian' or 'uniform'")

    # --------------------------
    # Stage 1: Gauss-Newton batch (STMs) on growing arcs
    # --------------------------
    # From the a priori state (~100 m and ~2 cm/s off on a 0.4 km, 11 cm/s orbit) Gauss-Newton
    # on the full arc stalls in a local minimum: each arc of STAGE1_ARCS_H starts from the
    # solution of the previous one, with the priors kept centered on x0_ref. With the MC on the
    # spin axis the MAP sits in a curved valley (the symmetry orbit) along which the data cost
    # is flat and Gauss-Newton only crawls; each round first moves to the prior-optimal point of
    # the valley (symmetry_step, exact), then iterates Gauss-Newton. A second, stiff and curved
    # valley (mean motion: vy, vz, GM, C22 trade off along the 24 h arc) is not straightened by
    # the symmetry step: LM crawls along it (~1 MAP-cost unit per iteration, hundreds of
    # iterations), so Stage 1 ends with the STT re-centering below instead.
    print("\n[Stage 1] Gauss-Newton batch (first-order STMs, LM, scaled) on growing arcs...")
    sym_prior = SymmetryChart(x0_ref, pole_i)  # rotations about the spin axis
    use_symmetry_step = MC_TILT_DEG == 0.0
    x_start = x0_ref.copy()
    for T_arc in STAGE1_ARCS_H:
        k = int(np.searchsorted(tau, T_arc * 3600.0 + 1e-6))
        batch = BatchLSQ(
            propagator=prop1,
            tau=tau[:k],
            sc_state=mc_state[:k],
            y_obs=y_obs[: 2 * k],
            sigma=sigma,
            obs_weights=obs_weights[: 2 * k],
            measurement=range_rangerate_and_partials,
            rel_dim=6,
        )
        for rnd in range(STAGE1_ROUNDS):
            x_prev = x_start.copy()
            msg = f"\n[Stage 1] arc 0-{tau[k - 1] / 3600:.1f} h ({int(vis_mask[:k].sum())} visible), round {rnd + 1}"
            if use_symmetry_step:
                x_start, a_sym = symmetry_step(x_start, x0_ref, prior_sigma, sym_prior)
                msg += f": symmetry step {np.rad2deg(a_sym):+.4f} deg"
            print(msg)
            x_start, delta_hat1, cov1 = batch.gn_with_stm(
                x0_ref=x_start,
                priors=[norm(loc=m, scale=s) for m, s in zip(x0_ref - x_start, prior_sigma)],
                max_iter=STAGE1_ITERS,
                tol=1e-10,
                rtol=ivp["rtol"],
                atol=ivp["atol"],
                method=ivp["method"],
                verbose=True,
                scaled=True,
                lm=True,
                ftol=1e-7,
            )
            if np.max(np.abs(x_start - x_prev) / prior_sigma) < 1e-6:
                print("[Stage 1] converged (round change < 1e-6 prior sigma)")
                break

    # STT re-centering: the MAP of the order-STT_ORDER surrogate about the current state (in
    # chart coordinates, so exact along the symmetry orbit), accepted when it lowers the exact
    # MAP cost (step halving in theta otherwise); the STTs are then recomputed about the new
    # state. With STT_ORDER = 2 the second-order terms carry the curvature of the mean-motion
    # valley, so this reaches the MAP in a few iterations where LM crawls. A fixed point is an
    # exact stationary point (the expansion is exact to first order at its center). With STMs
    # only (STT_ORDER = 1) it would be Gauss-Newton in chart coordinates, which crawls as well:
    # exact_map_chart (trust-region least squares on the exact residuals) is used instead.
    def exact_map_cost(X):
        """Exact MAP cost (chi2 + prior chi2, as BatchLSQ) of absolute states X (12,)."""
        traj = gravity1.propagate_many(np.atleast_2d(X), tau, **ivp)
        chi2 = np.sum(isl_residuals(traj, mc_state, y_obs, sigma, obs_weights) ** 2)
        return float(chi2 + np.sum(((X - x0_ref) / prior_sigma) ** 2))

    def exact_residuals_many(X):
        """Exact measurement and prior residuals (n, 2N + 12) of absolute states X (n, 12)."""
        X = np.atleast_2d(X)
        traj = gravity1.propagate_many(X, tau, **ivp)
        r = isl_residuals(traj, mc_state, y_obs, sigma, obs_weights)
        return np.concatenate([r, (X - x0_ref) / prior_sigma], axis=-1)

    if STT_ORDER == 1:
        # STMs only: the re-centering below would be Gauss-Newton in chart coordinates, which
        # crawls along the curved MAP valley; exact trust-region least squares instead.
        print("\n[Stage 1] Exact MAP (trust-region least squares in chart coordinates, full arc)...")
        x_c, cost_c = exact_map_chart(
            x_start,
            exact_residuals_many,
            lambda x: SymmetryChart(x, pole_i) if USE_SYMMETRY_CHART else IdentityChart(x),
            cov1,
        )
    else:
        print(f"\n[Stage 1] STT re-centering (order {STT_ORDER}, full arc)...")
        x_c, cost_c = x_start.copy(), exact_map_cost(x_start)
        for it in range(1, STAGE1_STT_ITERS + 1):
            sol_c, stts_c = prop2.propagate(x0=x_c, t_eval=tau, **ivp)
            stts_c6 = {k: np.asarray(v)[:, :6] for k, v in stts_c.items()}
            chart_c = SymmetryChart(x_c, pole_i) if USE_SYMMETRY_CHART else IdentityChart(x_c)
            J_c = np.linalg.inv(chart_c.jacobian_at_ref())
            P_c = J_c @ cov1 @ J_c.T
            L_c = np.linalg.cholesky(0.5 * (P_c + P_c.T))  # theta = L_c z: z of order one

            def surrogate_map_residuals(z):
                th = L_c @ z
                dx6 = prop2.propagate_deviation(sol_c, stts_c6, chart_c.slice_deviation(th))[1]
                r = isl_residuals(chart_c.rotate(dx6, chart_c.angle(th)), mc_state, y_obs, sigma, obs_weights)
                return np.concatenate([r, (chart_c.to_deviation(th) + x_c - x0_ref) / prior_sigma])

            res_c = least_squares(surrogate_map_residuals, np.zeros(12), method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
            a = 1.0
            while a > 1e-3:
                x_try = x_c + chart_c.to_deviation(L_c @ (a * res_c.x))
                cost_try = exact_map_cost(x_try)
                if cost_try < cost_c:
                    break
                a *= 0.5
            if a <= 1e-3:
                print(f"[Stage 1] re-centering {it}: no exact decrease from {cost_c:.6f}: converged")
                break
            print(
                f"[Stage 1] re-centering {it}: surrogate MAP cost {2.0 * res_c.cost:.6f}, exact "
                f"{cost_c:.6f} -> {cost_try:.6f} (step x {a:g}, |z| = {a * np.linalg.norm(res_c.x):.2e}, "
                f"delta {np.rad2deg(chart_c.angle(L_c @ (a * res_c.x))):+.3f} deg)"
            )
            x_c, cost_dec, cost_c = x_try, cost_c - cost_try, cost_try
            if cost_dec < STAGE1_STT_FTOL * cost_c:
                break

    # Batch covariance at the MAP (one LM iteration on the full arc: a noise-level step)
    print("\n[Stage 1] Covariance at the MAP (BatchLSQ, full arc)")
    x0_ref1, delta_hat1, cov1 = batch.gn_with_stm(
        x0_ref=x_c,
        priors=[norm(loc=m, scale=s) for m, s in zip(x0_ref - x_c, prior_sigma)],
        max_iter=1,
        tol=1e-10,
        rtol=ivp["rtol"],
        atol=ivp["atol"],
        method=ivp["method"],
        verbose=True,
        scaled=True,
        lm=True,
    )
    print(f"[Stage 1] exact MAP cost at ref1: {exact_map_cost(x0_ref1):.6f}")
    sd1 = np.sqrt(np.diag(cov1))
    print("\n[Stage 1] batch sigma:", sd1)
    print("[Stage 1] batch / prior sigma:", np.round(sd1 / prior_sigma, 4))
    print("[Stage 1] (truth - batch) / batch sigma:", np.round((x0_true - x0_ref1) / sd1, 2))

    # Display units: a power of ten per parameter so that its batch sigma reads well
    disp_scale, disp_unit = MCMCPlots.auto_param_scales(PARAM_BASE_SCALE, PARAM_BASE_UNIT, sd1)
    MCMCPlots.configure(PARAM_SYM, disp_unit, disp_scale)

    # --------------------------
    # Stage 2: STTs about ref1, symmetry chart
    # --------------------------
    print(f"\n[Stage 2] Propagating ref1 with {'STMs' if STT_ORDER == 1 else f'STTs of order {STT_ORDER}'}...")
    sol_ref, stts_ref = prop2.propagate(x0=x0_ref1, t_eval=tau, **ivp)
    stts_ref6 = {k: np.asarray(v)[:, :6] for k, v in stts_ref.items()}  # r, v rows only

    sym = SymmetryChart(x0_ref1, pole_i)  # always used for the diagnostics
    chart = sym if USE_SYMMETRY_CHART else IdentityChart(x0_ref1)
    print(
        f"[Chart] {'symmetry chart' if USE_SYMMETRY_CHART else 'identity (plain STT)'}; "
        f"distance of r0 from the spin axis rho_ref = {sym.rho_ref:.4f} km"
    )

    def stt_trajectory(dx):
        """r, v (N, 6) of the deviation dx about ref1 (STT)."""
        return prop2.propagate_deviation(sol_ref, stts_ref6, dx)[1]

    def residuals_normalized(theta):
        """ISL residuals of chart coordinates theta: STT on the slice, rotated back."""
        x6 = chart.rotate(stt_trajectory(chart.slice_deviation(theta)), chart.angle(theta))
        return isl_residuals(x6, mc_state, y_obs, sigma, obs_weights)

    def chi2_plain_stt(dx):
        return float(np.sum(isl_residuals(stt_trajectory(dx), mc_state, y_obs, sigma, obs_weights) ** 2))

    def chi2_sym_stt(dx):
        theta = sym.from_deviation(dx)
        x6 = sym.rotate(stt_trajectory(sym.slice_deviation(theta)), sym.angle(theta))
        return float(np.sum(isl_residuals(x6, mc_state, y_obs, sigma, obs_weights) ** 2))

    chi2_ref = float(np.sum(residuals_normalized(np.zeros(12)) ** 2))
    print(
        f"[Stage 2] At ref1: chi2_red = {chi2_ref / dof:.3f} (chi2 = {chi2_ref:.2f}, dof = {dof}, "
        f"n_vis = {n_visible})"
    )

    # Symmetry demo: data chi2 along g_delta(ref1), exact vs plain STT vs symmetry-reduced STT
    print("\n[Symmetry] data chi2 - chi2(ref1) at g_delta(ref1):")
    print("   delta [deg]       exact    plain STT   symmetry STT")
    angles = np.deg2rad([-20.0, -10.0, -5.0, 5.0, 10.0, 20.0])
    X_rot = sym.rotate(np.tile(x0_ref1, (angles.size, 1)), angles)
    traj_rot = gravity1.propagate_many(np.vstack([x0_ref1, X_rot]), tau, **ivp)
    chi2_ex = np.sum(isl_residuals(traj_rot, mc_state, y_obs, sigma, obs_weights) ** 2, axis=-1)
    for a, X, c_ex in zip(angles, X_rot, chi2_ex[1:]):
        dx = X - x0_ref1
        print(
            f"   {np.rad2deg(a):+8.1f}   {c_ex - chi2_ex[0]:+11.3e}  {chi2_plain_stt(dx) - chi2_ref:+11.3e}"
            f"  {chi2_sym_stt(dx) - chi2_ref:+11.3e}"
        )

    # Priors on dx about ref1 (centered on the a priori x0_ref)
    priors_ref1 = [norm(loc=-ds, scale=s) for ds, s in zip(x0_ref1 - x0_ref, prior_sigma)]
    for i in np.flatnonzero(is_unif):  # boxes (SH_PRIOR = "uniform")
        priors_ref1[i] = uniform(loc=unif_lo[i] - x0_ref1[i], scale=unif_hi[i] - unif_lo[i])
    true_delta = x0_true - x0_ref1

    # --------------------------
    # MCMC in chart coordinates
    # --------------------------
    print("\n[MCMC] Running in chart coordinates (starting about ref1)...")
    model = MCMCModelChart(
        residuals_func=residuals_normalized,
        initial_params=np.zeros(12),
        param_priors=priors_ref1,
        observed_data=y_obs,
        chart=chart,
    )
    # Whitening: batch covariance mapped to theta (exact Jacobian of the chart at theta = 0)
    J_chart = chart.jacobian_at_ref()
    J_inv = np.linalg.inv(J_chart)
    P_theta = J_inv @ cov1 @ J_inv.T
    model.setup_whitening(cov=0.5 * (P_theta + P_theta.T))
    if LOAD_CHAIN:
        model.load_chain(LOAD_CHAIN)  # figures from a saved run (no sampling)
        burn_in, thin = model.burn_in, model.thin
    else:
        np.random.seed(SEED_WALKERS)  # walker start (MCMCModel.run draws from np.random)
        model.run(
            n_samples=n_samples,
            n_walkers=n_walkers,
            burn_in=burn_in,
            thin=thin,
            spherical_spread=spherical_spread,
            method_optimize="Powell",
        )
        burn_in, thin = model.burn_in, model.thin
        model.save_chain(
            fig_dir / "chain.npz",
            x0_ref1=x0_ref1,
            cov1=cov1,
            x0_true=x0_true,
            x0_ref=x0_ref,
            use_symmetry_chart=np.array(USE_SYMMETRY_CHART),
            stt_order=np.array(STT_ORDER),
            sh_prior=np.array(SH_PRIOR),
            uniform_half_width=np.array(UNIFORM_HALF_WIDTH),
        )

    theta_hat, _ = model.get_estimate_and_covariance()
    chi2_mcmc = float(np.sum(residuals_normalized(theta_hat) ** 2))
    print(f"\n[MCMC] At the MAP sample: chi2_red = {chi2_mcmc / dof:.3f} (chi2 = {chi2_mcmc:.2f}, dof = {dof})")
    model.summary()
    model.print_regression_diagnostics()
    try:
        model.gelman_rubin_diagnostic()
    except Exception as err:
        print(f"[Gelman-Rubin] skipped: {type(err).__name__}: {err}")

    samples_dx = chart.to_deviation(np.asarray(model.samples, dtype=float))
    x_post = x0_ref1 + samples_dx

    # Posterior mode of the plotted density, the one of dx (the log-prior in theta holds the
    # chart Jacobian, taken out here), mapped to dx; on the SH and derived corners its image.
    dx_map = None
    if SHOW_MAP:
        theta_map = model.find_map(
            log_target=lambda th: model.log_posterior(th) - float(chart.log_jacobian(th))
        )
        dx_map = chart.to_deviation(theta_map)

    def _map(f):
        return None if dx_map is None else f(dx_map)

    # --------------------------
    # Checks
    # --------------------------
    # Arc angle about the spin axis: MCMC vs batch (linearized) spread
    g_delta = np.zeros(12)
    g_delta[:3] = sym.e_phi / sym.rho_ref
    sd_delta_batch = np.sqrt(g_delta @ cov1 @ g_delta)
    sd_delta_mcmc = np.std(sym.azimuth(samples_dx))
    print(
        f"\n[Check] arc angle delta: MCMC sigma {np.rad2deg(sd_delta_mcmc):.2f} deg, "
        f"batch (linearized) {np.rad2deg(sd_delta_batch):.2f} deg"
    )

    # Exact check: nonlinear propagation of posterior samples, data chi2 exact vs surrogate
    rng_chk = np.random.default_rng(0)
    idx = rng_chk.choice(samples_dx.shape[0], size=min(N_EXACT_CHECK, samples_dx.shape[0]), replace=False)
    print(f"[Check] Propagating {idx.size} posterior samples exactly...")
    traj = gravity1.propagate_many(x0_ref1 + samples_dx[idx], tau, **ivp)
    c_exact = np.sum(isl_residuals(traj, mc_state, y_obs, sigma, obs_weights) ** 2, axis=-1)
    c_used = np.array([np.sum(residuals_normalized(model.samples[i]) ** 2) for i in idx])
    c_plain = np.array([chi2_plain_stt(samples_dx[i]) for i in idx])
    for lab, c_s in (("sampled surrogate", c_used), ("plain STT", c_plain)):
        dc = c_exact - c_s
        lw = -0.5 * (dc - dc.min())
        w = np.exp(lw - lw.max())
        ess = w.sum() ** 2 / np.sum(w**2)
        print(
            f"[Check] exact - {lab}: ESS {ess:.0f}/{idx.size}, median |dchi2| "
            f"{np.median(np.abs(dc)):.3g}, 90% |dchi2| {np.quantile(np.abs(dc), 0.9):.3g}"
        )

    # Summary: sigmas, skewness and excess kurtosis of every parameter and derived quantity
    print("\n[Summary] dx about ref1 (sigma in internal units; prior / batch / MCMC):")
    names = ["dx0", "dy0", "dz0", "dvx0", "dvy0", "dvz0", "dGM", "dC20", "dC21", "dS21", "dC22", "dS22"]
    for i, nm in enumerate(names):
        print(moments_row(nm, samples_dx[:, i], prior_sigma[i], sd1[i]))

    ref_der = derived_quantities(x0_ref1, M0)
    der_post = derived_quantities(x_post, M0, ref=ref_der)
    rng_pr = np.random.default_rng(1)
    x_prior = x0_ref + prior_sigma * rng_pr.standard_normal((200_000, 12))
    if is_unif.any():
        x_prior[:, is_unif] = rng_pr.uniform(unif_lo[is_unif], unif_hi[is_unif], (200_000, int(is_unif.sum())))
    der_prior = derived_quantities(x_prior, M0, ref=ref_der)
    J_der = numeric_jacobian(lambda x: derived_quantities(x, M0, ref=ref_der), x0_ref1, 1e-4 * sd1)
    cov_der = J_der @ cov1 @ J_der.T
    der_true = derived_quantities(x0_true, M0, ref=ref_der)
    print("[Summary] derived quantities (angles in rad):")
    for i, nm in enumerate(["C20", "J22", "lambda22", "phi0"]):
        print(moments_row(nm, der_post[:, i], np.std(der_prior[:, i]), np.sqrt(cov_der[i, i])))
    phase = der_post[:, 3] - der_post[:, 2]
    print(
        f"  phi0 - lambda22: MCMC sigma {np.rad2deg(np.std(phase)):.3f} deg "
        f"(lambda22 alone {np.rad2deg(np.std(der_post[:, 2])):.2f} deg)"
    )
    print(
        "[Summary] truth (C20, J22, lambda22 [deg], phi0 [deg]): "
        f"{der_true[0]:.5e}, {der_true[1]:.5e}, {np.rad2deg(der_true[2]):.2f}, {np.rad2deg(der_true[3]):.2f}"
    )
    print(
        "[Summary] MCMC median                            : "
        f"{np.median(der_post[:, 0]):.5e}, {np.median(der_post[:, 1]):.5e}, "
        f"{np.rad2deg(np.median(der_post[:, 2])):.2f}, {np.rad2deg(np.median(der_post[:, 3])):.2f}"
    )

    # --------------------------
    # Figures
    # --------------------------
    to_dx = chart.to_deviation
    _figure(
        "post-fit residuals",
        lambda: MCMCPlots.plot_mcmc_residuals(model, tau, vis_mask, np.zeros(12), components=[0, 1]),
    )
    _figure(
        "traces",
        lambda: MCMCPlots.plot_mcmc_traces(
            model,
            burn_in,
            true_theta=true_delta,
            batch_theta=np.zeros(12),
            transform=to_dx,
            logp_truth_theta=chart.from_deviation(true_delta),
        ),
    )
    _figure(
        "autocorrelation",
        lambda: MCMCPlots.plot_mcmc_autocorrelation(model, burn_in, thin, transform=to_dx),
    )
    corner_kw = dict(
        true_theta=true_delta,
        batch_mean=np.zeros(12),
        batch_cov=cov1,
        map_theta=_map(lambda d: d),
        show_batch_mean=SHOW_BATCH_MEAN,
    )
    _figure("corner", lambda: MCMCPlots.plot_corner(samples_dx, name="fig_corner", **corner_kw))
    _figure(
        "corner + prior",
        lambda: MCMCPlots.plot_corner(samples_dx, name="fig_corner_prior", prior=priors_ref1, **corner_kw),
    )
    if CORNER_RTN:
        A_rtn = MCMCPlots.rtn_corner_map(x0_ref1, 12)
        rtn_scale, rtn_unit = MCMCPlots.auto_param_scales(
            PARAM_BASE_SCALE, PARAM_BASE_UNIT, np.sqrt(np.diag(A_rtn @ cov1 @ A_rtn.T))
        )
        _figure(
            "corner RTN",
            lambda: MCMCPlots.plot_corner(
                samples_dx @ A_rtn.T,
                true_theta=A_rtn @ true_delta,
                batch_mean=np.zeros(12),
                batch_cov=A_rtn @ cov1 @ A_rtn.T,
                map_theta=_map(lambda d: A_rtn @ d),
                show_batch_mean=SHOW_BATCH_MEAN,
                name="fig_corner_rtn",
                param_sym=MCMCPlots.RTN_SYM + PARAM_SYM[6:],
                param_unit=rtn_unit,
                param_scale=rtn_scale,
            ),
        )

    # Absolute degree-2 coefficients (5 parameters), with and without the prior
    sh_sym = [r"C_{20}", r"C_{21}", r"S_{21}", r"C_{22}", r"S_{22}"]
    sh_scale, sh_unit = MCMCPlots.auto_param_scales(np.ones(5), [""] * 5, sd1[7:12])
    MCMCPlots.configure(sh_sym, sh_unit, sh_scale)
    sh_kw = dict(
        true_theta=x0_true[7:12],
        batch_mean=x0_ref1[7:12],
        batch_cov=cov1[7:12, 7:12],
        map_theta=_map(lambda d: (x0_ref1 + d)[7:12]),
        show_batch_mean=SHOW_BATCH_MEAN,
    )
    _figure("corner SH", lambda: MCMCPlots.plot_corner(x_post[:, 7:12], name="fig_corner_sh2", **sh_kw))
    _figure(
        "corner SH + prior",
        lambda: MCMCPlots.plot_corner(
            x_post[:, 7:12],
            name="fig_corner_sh2_prior",
            prior=[
                uniform(loc=unif_lo[i], scale=unif_hi[i] - unif_lo[i])
                if is_unif[i]
                else norm(loc=x0_ref[i], scale=prior_sigma[i])
                for i in range(7, 12)
            ],
            **sh_kw,
        ),
    )

    # Derived: C20, J22, lambda22, phi0 (angles in degrees)
    der_sym = [r"C_{20}", r"J_{22}", r"\lambda_{22}", r"\varphi_0"]
    der_scale, der_unit = MCMCPlots.auto_param_scales(
        np.array([1.0, 1.0, np.rad2deg(1.0), np.rad2deg(1.0)]),
        ["", "", "deg", "deg"],
        np.sqrt(np.diag(cov_der)),
    )
    MCMCPlots.configure(der_sym, der_unit, der_scale)
    der_kw = dict(
        true_theta=der_true,
        batch_mean=ref_der,
        batch_cov=cov_der,
        map_theta=_map(lambda d: derived_quantities(x0_ref1 + d, M0, ref=ref_der)),
        show_batch_mean=SHOW_BATCH_MEAN,
    )
    _figure("corner derived", lambda: MCMCPlots.plot_corner(der_post, name="fig_corner_derived", **der_kw))
    _figure(
        "corner derived + prior",
        lambda: MCMCPlots.plot_corner(der_post, name="fig_corner_derived_prior", prior=der_prior, **der_kw),
    )
    MCMCPlots.configure(PARAM_SYM, disp_unit, disp_scale)  # back to the 12 parameters

    # --------------------------
    # Posterior split (GMM) + STM propagation, checked against an exact Monte Carlo
    # --------------------------
    if GMM_PROPAGATION:
        gmm_prediction(
            samples_dx,
            x0_ref1,
            cov1,
            x0_true,
            prop1,
            gravity1.propagate_many,
            np.linspace(0.0, gmm_target_hours * 3600.0, n_t_prop),
            tau[-1],
            ivp,
            ("Posterior at Epoch", f"Predicted at Epoch + {gmm_target_hours:g} h"),
            n_components=gmm_n_components,
            select_bic=gmm_select_bic,
            n_mc=n_mc_prop,
            prob=gmm_prob,
            batch=GMM_BATCH,
            rtn=GMM_RTN,
            show_mc=GMM_SHOW_MC,
            show_components=GMM_SHOW_COMPONENTS,
            show_epoch=GMM_SHOW_EPOCH,
            r_ref=R_ref,
            chart=sym if GMM_FIT_CHART else None,
            figure=_figure,
        )

    print(f"\n[Done] Results in {fig_dir}")
