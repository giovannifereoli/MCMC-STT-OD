"""
Paper Case 2 (Bennu popper): OSIRIS-REx (SPICE trajectory, as in Case 1) tracks a radio popper
on a low polar orbit about Bennu with Doppler only, built so that the degree-2 gravity
posterior is both informed by the data and non-Gaussian (a crescent in C22-S22).

Paper cases (CASE in the settings)
- "2a": Gaussian priors on C20, C21, S21, C22, S22 + posterior split (GMM) and STM propagation
  past the arc (step 7). Results in results/case2a_bennu_popper_<stamp>/.
- "2b": uniform (box) priors on C20, C21, S21, C22, S22 (SH_PRIOR = "uniform"), no GMM
  propagation. Results in results/case2b_bennu_popper_uniform_<stamp>/.

Geometry
- Bennu: unnormalized degree-2 field of get_CS_unnormalized.bennu_10_10 (R_ref = 0.290 km),
  pole (85.65, -60.17) deg, period 4.296057 h, IAU constant spin (frame rotations, body +Z =
  pole, spin +omega, as in extra_ryugu_probe.py and extra/scenario_ISL_SH2.py).
- Observer: OSIRIS-REx from SPICE (OSIRIS-REX w.r.t. BENNU, J2000, no aberration correction)
  during Detailed Survey, north polar station (2019-03-16T18:00 + 24 h: 5.9-6.7 km from Bennu,
  2.4-3.3 deg from the spin axis, azimuth about the axis drifting by 25 deg).
- Popper: a radio probe without propulsion, released by a spring from a mothercraft hovering at
  the release point. Planned orbit: circular polar orbit of radius ORBIT_RADIUS whose plane
  contains the mean azimuth of OREx about the spin axis (ORBIT_RAAN_DEG = None). DELIVERY = True
  (default) adds the release error, DELIVERY_SIGMA_R / DELIVERY_SIGMA_V per axis (20 m, 1 cm/s),
  drawn with DELIVERY_SEED; a draw that comes within DELIVERY_R_MIN of the center or beyond
  DELIVERY_R_MAX within DELIVERY_CHECK_HOURS is a failed deployment and is redrawn (the scenario
  is conditioned on a probe that survives the arc and the prediction). The delivered orbit is
  inclined, eccentric and off the OREx meridian by a few degrees. DELIVERY = False keeps the exact
  planned orbit (results folder suffix _circular). ARC_HOURS arc, one-way Doppler (range-rate) every DT_OBS_MIN minutes, occulted
  by the sphere of radius R_ref. USE_RANGE adds range.
- Estimated: x0 = [r0, v0, GM, C20, C21, S21, C22, S22] of the popper (12 parameters).

Why this posterior is not Gaussian
The dynamics are invariant under the one-parameter group g_delta (scenario_ISL_SH2.py): rotate
r0, v0 by delta about the pole, (C21, S21) by delta and (C22, S22) by 2 delta; the new orbit is
the old one rotated by delta. An observer on the spin axis cannot see delta, so the data fix
J22 = hypot(C22, S22) and the phase of the popper relative to the long axis but not lambda22:
the posterior is a crescent along the symmetry orbit (scenario_ISL_SH2.py). OREx is 2-3 deg off
the axis, which breaks the symmetry at first order through the transverse terms of the
range-rate. With the popper orbit in the meridian plane of OREx these terms vanish (the motion
along g_delta is normal to the plane that contains OREx, the popper position and its velocity)
and only the 25 deg azimuth drift of OREx and the 2-3 deg tilt are left. The pure g_delta still
costs data chi2 (the [Symmetry] table), but g_delta combined with small changes of the other
parameters is nearly free: the data pin J22 to a few % and lambda22 only to several degrees
(2 lambda22 to ~10-20 deg), and the posterior bends along the symmetry orbit. With the orbit
plane 90 deg away from OREx, lambda22 is known to a few degrees and the posterior is close to a
Gaussian ellipse.

Sampling (as scenario_ISL_SH2.py)
- The STT is evaluated on the slice through x0_ref1 (symmetry chart) and the trajectory rotated
  back by delta: exact in delta for the dynamics for any observer (OREx enters only through
  the measurement of the rotated trajectory). USE_SYMMETRY_CHART = False samples dx directly.
- MCMC in chart coordinates theta (MCMCModelChart); the exact check (nonlinear propagation of
  N_EXACT_CHECK posterior samples) reports the importance-sampling ESS of the surrogate.
- STT_ORDER = 1 (default): the Stage-2 surrogate uses the STMs only; 2 adds the second-order
  STT (results folder suffix _stt2). Stage 1 does not depend on it.
- SH_PRIOR = "uniform" (CASE = "2b") replaces the Gaussian priors of C20, C21, S21, C22, S22 by
  boxes x0_ref +- UNIFORM_HALF_WIDTH prior sigmas (GM, r0, v0 stay Gaussian; results folder
  suffix _uniform). Stage 1 only needs a reference point: it uses Gaussians with the mean and sd
  of the boxes; the MCMC samples the box prior itself.

Pipeline
  1) SPICE: OREx states at the epochs; truth popper orbit, noisy Doppler on the visible epochs.
  2) A priori x0_ref = truth - prior draw.
  3) Stage 1: batch SRIF (BatchLSQ, STMs; BATCH_SOLVER = "ud": Bierman U-D) on growing arcs,
     Gauss-Newton iterations with step halving on the exact MAP cost, the step solved for in
     symmetry-chart coordinates: x0_ref1 = exact MAP of the full arc, cov1 = SRIF covariance.
  4) Stage 2: STMs (STT_ORDER = 1) or STTs about x0_ref1; data chi2 along g_delta (exact /
     plain STT / chart STT).
  5) MCMC in chart coordinates; chain saved to chain.npz (LOAD_CHAIN replots a saved run).
  6) Checks (arc angle, Gelman-Rubin, exact importance check), summary table, curvature of the
     degree-2 pairs, figures: geometry, residuals, traces, autocorrelation, corners
     (deviations, RTN, absolute C/S, derived C20 / J22 / lambda22 / phi0, each with and without
     the prior).
  7) Case 2a only (GMM_PROPAGATION): posterior split (GMM) + per-component STM propagation to
     gmm_target_hours (past the arc), checked against an exact Monte Carlo of posterior samples
     (gmm_prediction, the GMM block of extra_ryugu_probe.py): fig_gmm_stm_propagation (+ _rtn),
     fig_gmm_prediction_history.

The degree-2 dynamics, the symmetry chart and the helpers (gmm_prediction included) are
imported from extra/scenario_ISL_SH2.py.

Units: km, km/s, s.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import spiceypy as spice
import trimesh
from scipy.integrate import solve_ivp
from scipy.stats import norm, uniform

import MCMCPlots
from BatchLSQ import BatchLSQ
from get_CS_unnormalized import bennu_10_10
from extra.scenario_ISL_SH2 import (
    Degree2Gravity,
    IdentityChart,
    MCMCModelChart,
    SymmetryChart,
    _Rx_act,
    _Rz_act,
    _Tee,
    derived_quantities,
    gmm_prediction,
    isl_residuals,
    make_bennu_rotation_matrix,
    moments_row,
    numeric_jacobian,
    occultation_mask,
    range_rangerate_and_partials,
)
from STTPropagationND import STTPropagatorND

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved


# ============================================================
# SPICE
# ============================================================


def load_orex_kernels(kernel_root):
    """Furnish the OREx trajectory kernels, leap seconds, PCK and the Bennu ephemeris."""
    kernel_root = Path(kernel_root)
    kernels = sorted(str(p) for p in (kernel_root / "orex" / "orex_trajectories").iterdir() if p.is_file())
    kernels += [
        str(kernel_root / "naif0012.tls"),
        str(kernel_root / "pck00010.tpc"),
        str(kernel_root / "de424.bsp"),
        str(kernel_root / "bennu_v17.tpc"),
        str(kernel_root / "orex" / "bennu_refdrmc_v1.bsp"),
    ]
    for k in kernels:
        if not Path(k).is_file():
            raise FileNotFoundError(f"Kernel not found: {k}")
        spice.furnsh(k)
    return kernels


def spice_states(target, et0, tau, frame, abcorr, center):
    """States (N, 6) of target w.r.t. center at et0 + tau [km, km/s]."""
    return np.array([spice.spkezr(target, float(et0 + t), frame, abcorr, center)[0] for t in tau], dtype=float)


# ============================================================
# Helpers
# ============================================================


def curvature_metric(a, b):
    """
    Bend of b along a: least squares b = c0 + c1 a + c2 a^2 on standardized samples, returned
    as |c2| * 4 / sd(residual) (the sagitta of the fitted parabola over a = +-2 sd in units of
    the scatter about it). Gaussian samples give ~0 (sampling noise ~0.1); a value above ~1.5
    is a visibly curved pair.
    """
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    A = np.column_stack([np.ones_like(a), a, a * a])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return abs(c[2]) * 4.0 / np.std(b - A @ c)


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent

    # --------------------------
    # Settings
    # --------------------------
    # Paper case: "2a" = Gaussian SH priors + GMM propagation, "2b" = uniform SH priors (no GMM)
    CASE = "2a"
    STT_ORDER = 1  # surrogate order: 1 = STMs only, 2 = second-order STTs (on the slice; folder + _stt2)
    USE_SYMMETRY_CHART = True  # False: sample dx with the plain STT (fails along the crescent)
    # Stage-1 batch solver: "srif" (square-root information filter, Householder) or "ud"
    # (Bierman U-D, one scalar measurement at a time); same MAP and covariance. "ud" adds
    # "_ud" to the results folder.
    BATCH_SOLVER = "srif"
    # "uniform": boxes on C20, C21, S21, C22, S22 (folder + _uniform); set by CASE
    SH_PRIOR = "uniform" if CASE == "2b" else "gaussian"
    UNIFORM_HALF_WIDTH = 3.0  # box half-width in prior sigmas, centered on the a priori x0_ref
    # Saved chain: every run writes chain.npz into its results folder. Set LOAD_CHAIN to such
    # a file to skip the sampling and redo the diagnostics and figures from it (set CASE to the
    # case of that run).
    LOAD_CHAIN = None  # e.g. "results/case2a_bennu_popper_<stamp>/chain.npz"
    CORNER_RTN = True
    # Mark the posterior mode (MCMCModel.find_map, an optimum, not the best sample) and the
    # batch estimate ref1 on every corner plot.
    SHOW_MAP = True
    SHOW_BATCH_MEAN = True

    # SPICE (as Case 1)
    KERNEL_ROOT = HERE / "data" / "kernels"
    SC_NAME = "OSIRIS-REX"
    CENTER = "BENNU"
    FRAME_I = "J2000"
    ABCORR = "NONE"
    utc0 = "2019-03-16T18:00:00"  # Detailed Survey, north polar station

    # Bennu (unnormalized degree-2 coefficients of the OSIRIS-REx field)
    R_ref = 0.290  # km
    GM_true, C_unn, S_unn = bennu_10_10()
    cs_true = np.array([C_unn[2, 0], C_unn[2, 1], S_unn[2, 1], C_unn[2, 2], S_unn[2, 2]])
    alpha_true = np.deg2rad(85.65)
    delta_true = np.deg2rad(-60.17)
    omega_true = 2.0 * np.pi / (4.296057 * 3600.0)
    SHAPE_PATH = HERE / "data" / "kernels" / "orex" / "bennu_radar_v20.obj"  # orbit scene figure only

    # Popper: planned release state = circular (point-mass speed) orbit in the body-equatorial
    # frame at t = 0
    ORBIT_RADIUS = 0.4  # km
    ORBIT_INC_DEG = 90.0
    ORBIT_RAAN_DEG = None  # None: orbit plane through the mean azimuth of OREx about the axis
    ORBIT_U0_DEG = 0.0  # argument of latitude at t = 0
    # Release error of the probe (no propulsion: the delivered orbit is what the spring gives).
    # False: the probe starts exactly on the planned orbit (folder + _circular).
    DELIVERY = True
    DELIVERY_SIGMA_R = 0.02  # km per axis (20 m: mothercraft navigation at release)
    DELIVERY_SIGMA_V = 1e-5  # km/s per axis (1 cm/s: spring separation speed and direction)
    DELIVERY_SEED = 1
    DELIVERY_CHECK_HOURS = 48.0  # survival window (arc + GMM prediction)
    DELIVERY_R_MIN = 1.05 * R_ref  # km: closer = contact with the ridge / field not valid
    DELIVERY_R_MAX = 2.0  # km: farther = escaping

    # Tracking: one-way Doppler (range-rate) from OREx
    ARC_HOURS = 24.0
    DT_OBS_MIN = 10.0
    USE_RANGE = False  # True: range as well
    sigma_range = 1e-3  # km (1 m; weight 0 unless USE_RANGE)
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
    STAGE1_ITERS = 40  # max SRIF iterations per arc
    STAGE1_TOL = 1e-6  # stop when the step is shorter (norm in km, km/s, ...; 1 mm)

    # MCMC
    n_walkers = 96
    n_samples = 180000  # ~80-110 x max(tau) (tau ~1600-2100 steps)
    burn_in = None  # None: 2 x max(tau)
    thin = None  # None: 0.5 x min(tau)
    spherical_spread = 0.5  # walker start, whitened by the batch covariance in theta
    SEED_WALKERS = 7
    N_EXACT_CHECK = 1000  # posterior samples propagated exactly (importance check)

    # Posterior split (GMM) + STM propagation past the arc (as extra_ryugu_probe.py); Case 2a
    GMM_PROPAGATION = CASE == "2a"
    gmm_target_hours = 48.0  # [h since utc0]; the arc ends at ARC_HOURS
    # K = 8 (as the Ryugu probe) puts part of the crescent's sag into the mean-motion direction:
    # ~2.5x the Monte Carlo spread at 48 h. K = 32 matches it (45 vs 48 m at 48 h), ~40 s.
    gmm_n_components = 32
    gmm_select_bic = False  # True: K in 1..12 by BIC
    GMM_FIT_CHART = False  # True: fit in chart coordinates (see gmm_prediction)
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
        target="Popper",
        obs_word="Epoch",
        site_label="Deployment Site",
        resid_labels=("Range", "Range-Rate"),
        fallback_dir=Path.home() / "bennu_figures_fallback",
    )
    # Folder name: case<CASE>_bennu_popper(_uniform)_YYMMDDHHMMSS, e.g.
    # results/case2a_bennu_popper_261122133644 (+ _circular without the release error,
    # + _stt<order> with STT_ORDER > 1, + _plain without the symmetry chart, + _ud with the
    # U-D batch)
    fig_dir = MCMCPlots.set_figure_dir(
        HERE
        / "results"
        / (
            f"case{CASE}_bennu_popper"
            + ("_uniform" if SH_PRIOR == "uniform" else "")
            + "_"
            + datetime.now().strftime("%y%m%d%H%M%S")
            + ("" if DELIVERY else "_circular")
            + (f"_stt{STT_ORDER}" if STT_ORDER > 1 else "")
            + ("" if USE_SYMMETRY_CHART else "_plain")
            + ("_ud" if BATCH_SOLVER == "ud" else "")
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
    # SPICE: OREx states
    # --------------------------
    print("\n[SPICE] Loading kernels and OREx states...")
    load_orex_kernels(KERNEL_ROOT)
    et0 = spice.str2et(utc0)
    tau = np.arange(0.0, ARC_HOURS * 3600.0 + 1e-6, DT_OBS_MIN * 60.0)
    N = tau.size
    tau_dense = np.linspace(0.0, tau[-1], 2001)
    orex_state = spice_states(SC_NAME, et0, tau, FRAME_I, ABCORR, CENTER)
    orex_dense = spice_states(SC_NAME, et0, tau_dense, FRAME_I, ABCORR, CENTER)

    # OREx about the spin axis: distance, angle from the axis, azimuth in the body-equatorial
    # frame at t = 0 (inertial)
    ro_eq = orex_state[:, :3] @ M0.T
    dist_o = np.linalg.norm(ro_eq, axis=1)
    beta_o = np.rad2deg(np.arccos(ro_eq[:, 2] / dist_o))
    az_o = np.arctan2(ro_eq[:, 1], ro_eq[:, 0])
    az_mean = np.arctan2(np.mean(np.sin(az_o)), np.mean(np.cos(az_o)))
    print(
        f"[SPICE] OREx {utc0} + {ARC_HOURS:g} h: {dist_o.min():.2f}-{dist_o.max():.2f} km from Bennu, "
        f"{beta_o.min():.2f}-{beta_o.max():.2f} deg from the spin axis, azimuth "
        f"{np.rad2deg(az_o.min()):.1f} to {np.rad2deg(az_o.max()):.1f} deg (mean {np.rad2deg(az_mean):.1f})"
    )

    # --------------------------
    # Truth: popper initial state, measurements
    # --------------------------
    raan = az_mean if ORBIT_RAAN_DEG is None else np.deg2rad(ORBIT_RAAN_DEG)
    u0 = np.deg2rad(ORBIT_U0_DEG)
    Q_orb = _Rz_act(raan) @ _Rx_act(np.deg2rad(ORBIT_INC_DEG))
    v_circ = np.sqrt(GM_true / ORBIT_RADIUS)
    r0_true = M0.T @ Q_orb @ (ORBIT_RADIUS * np.array([np.cos(u0), np.sin(u0), 0.0]))
    v0_true = M0.T @ Q_orb @ (v_circ * np.array([-np.sin(u0), np.cos(u0), 0.0]))
    x0_true = np.concatenate([r0_true, v0_true, [GM_true], cs_true])

    def body_elements(x):
        """Osculating inclination, node [deg] (body-equatorial frame at t = 0), e, a [km] of x."""
        rb, vb = M0 @ x[:3], M0 @ x[3:6]
        h = np.cross(rb, vb)
        e_vec = np.cross(vb, h) / GM_true - rb / np.linalg.norm(rb)
        a = 1.0 / (2.0 / np.linalg.norm(rb) - vb @ vb / GM_true)
        inc = np.rad2deg(np.arccos(h[2] / np.linalg.norm(h)))
        return inc, np.rad2deg(np.arctan2(h[0], -h[1])), np.linalg.norm(e_vec), a

    if DELIVERY:
        # Spring release without propulsion: draw the release error, redraw failed deployments

        def _contact(t, y):
            return np.linalg.norm(y[:3]) - DELIVERY_R_MIN

        def _escape(t, y):
            return np.linalg.norm(y[:3]) - DELIVERY_R_MAX

        _contact.terminal = _escape.terminal = True
        rng_del = np.random.default_rng(DELIVERY_SEED)
        x0_plan = x0_true.copy()
        for n_draw in range(1, 101):
            dr0 = DELIVERY_SIGMA_R * rng_del.standard_normal(3)
            dv0 = DELIVERY_SIGMA_V * rng_del.standard_normal(3)
            x_try = x0_plan.copy()
            x_try[0:3] += dr0
            x_try[3:6] += dv0
            chk = solve_ivp(
                lambda t, y: gravity1.f_func(*y, t),
                (0.0, DELIVERY_CHECK_HOURS * 3600.0),
                x_try,
                events=(_contact, _escape),
                **ivp,
            )
            if chk.t_events[0].size == 0 and chk.t_events[1].size == 0:
                break
            fate = "contact" if chk.t_events[0].size else "escape"
            print(
                f"[Delivery] draw {n_draw}: |dr| {np.linalg.norm(dr0) * 1e3:.1f} m, |dv| "
                f"{np.linalg.norm(dv0) * 1e5:.2f} cm/s -> {fate} at {chk.t[-1] / 3600:.1f} h, redrawn"
            )
        else:
            raise RuntimeError("[Delivery] no surviving release in 100 draws")
        x0_true = x_try
        r0_true, v0_true = x0_true[0:3], x0_true[3:6]
        i_p, n_p, _, _ = body_elements(x0_plan)
        i_d, n_d, e_d, a_d = body_elements(x0_true)
        print(
            f"[Delivery] seed {DELIVERY_SEED}, draw {n_draw} survives {DELIVERY_CHECK_HOURS:g} h: release "
            f"error dr = {np.round(dr0 * 1e3, 1)} m, dv = {np.round(dv0 * 1e5, 2)} cm/s "
            f"(sigma {DELIVERY_SIGMA_R * 1e3:g} m, {DELIVERY_SIGMA_V * 1e5:g} cm/s per axis)"
        )
        print(
            f"[Delivery] planned: inclination {i_p:.2f} deg, node {n_p:.2f} deg, e 0 | delivered: "
            f"inclination {i_d:.2f} deg, node {n_d:.2f} deg ({(n_d - n_p + 180) % 360 - 180:+.2f} deg "
            f"from the OREx meridian), e {e_d:.3f}, a {a_d:.3f} km (osculating, t = 0)"
        )

    print("\n[Truth] Propagating the popper...")
    sol_true = prop1.propagate_state_only(x0_true, tau, show_progress=False, **ivp)
    x_true = sol_true.y[:12].T
    x_dense = prop1.propagate_state_only(x0_true, tau_dense, show_progress=False, **ivp).y[:12].T
    r_dense = np.linalg.norm(x_dense[:, :3], axis=1)
    vis_mask = occultation_mask(orex_state[:, :3], x_true[:, :3], R_ref)

    print(f"[Truth] GM = {GM_true:.6e} km^3/s^2, (C20, C21, S21, C22, S22) = {cs_true}")
    J22_true = np.hypot(cs_true[3], cs_true[4])
    print(
        f"[Truth] J22 = {J22_true:.4e}, lambda22 = "
        f"{np.rad2deg(0.5 * np.arctan2(cs_true[4], cs_true[3])):.2f} deg"
    )
    if DELIVERY:
        print(
            f"[Truth] popper (delivered): radius {r_dense.min():.3f}-{r_dense.max():.3f} km (R_ref "
            f"{R_ref} km), speed {np.linalg.norm(v0_true) * 1e5:.2f} cm/s (planned {v_circ * 1e5:.2f}), "
            f"period {2 * np.pi * np.sqrt(a_d**3 / GM_true) / 3600:.2f} h (osculating)"
        )
    else:
        print(
            f"[Truth] popper: inclination {ORBIT_INC_DEG:g} deg, node {np.rad2deg(raan):.1f} deg "
            f"({'OREx mean azimuth' if ORBIT_RAAN_DEG is None else 'set'}), radius "
            f"{r_dense.min():.3f}-{r_dense.max():.3f} km (R_ref {R_ref} km), speed {v_circ * 1e5:.2f} cm/s, "
            f"period {2 * np.pi * np.sqrt(ORBIT_RADIUS**3 / GM_true) / 3600:.2f} h"
        )
    print(f"[Truth] {vis_mask.sum()}/{N} epochs visible from OREx")
    if r_dense.min() < 1.05 * R_ref:
        print("[WARNING] The popper orbit comes within 5 % of R_ref (field not valid inside)")

    rng_meas = np.random.default_rng(123)
    rho_t, rhodot_t, _, _ = range_rangerate_and_partials(x_true[:, :6] - orex_state)
    y_obs = np.empty(2 * N)
    y_obs[0::2] = rho_t + sigma_range * rng_meas.standard_normal(N)
    y_obs[1::2] = rhodot_t + sigma_range_rate * rng_meas.standard_normal(N)
    obs_weights = np.ones(2 * N)
    obs_weights[0::2][~vis_mask] = 0.0
    obs_weights[1::2][~vis_mask] = 0.0
    if not USE_RANGE:
        obs_weights[0::2] = 0.0
    sigma = (sigma_range, sigma_range_rate)
    n_visible = int(vis_mask.sum())
    n_meas = int(np.count_nonzero(obs_weights))
    dof = n_meas - 12

    _figure(
        "observation geometry",
        lambda: MCMCPlots.plot_observation_geometry(
            tau, orex_state, x_true, vis_mask, tau_line=tau_dense, sc_line=orex_dense, pt_line=x_dense
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
            sc_line=orex_dense,
            alpha=alpha_true,
            delta=delta_true,
            omega=omega_true,
            observer="OSIRIS-REx",
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

    def exact_map_cost_many(X, k=N):
        """Exact MAP cost (chi2 + prior chi2, as BatchLSQ) of absolute states X (n, 12), first k epochs."""
        X = np.atleast_2d(X)
        traj = gravity1.propagate_many(X, tau[:k], **ivp)
        chi2 = np.sum(
            isl_residuals(traj, orex_state[:k], y_obs[: 2 * k], sigma, obs_weights[: 2 * k]) ** 2, axis=-1
        )
        return chi2 + np.sum(((X - x0_ref) / prior_sigma) ** 2, axis=-1)

    def exact_map_cost(X):
        return float(exact_map_cost_many(X)[0])

    # --------------------------
    # Stage 1: batch SRIF (STMs) on growing arcs
    # --------------------------
    # From the a priori state (~100 m and ~2 cm/s off on a 0.4 km, 9 cm/s orbit) Gauss-Newton
    # on the full arc does not converge: each arc of STAGE1_ARCS_H starts from the solution of
    # the previous one, with the priors kept centered on x0_ref. The MAP sits in a curved valley
    # close to g_delta (OREx near the spin axis), ~10 deg of rotation away from the 8 h solution.
    # Each iteration solves the SRIF for the step in symmetry-chart coordinates theta (delta
    # along g_delta + the slice), x += to_deviation(theta), so that the step follows the valley
    # (in Cartesian coordinates Gauss-Newton crawls along it). The step is halved until it
    # lowers the exact MAP cost (BatchLSQ.gn_with_stm). The chart is used here whatever
    # USE_SYMMETRY_CHART (only a parametrization of the step: same MAP and covariance).
    print(f"\n[Stage 1] Batch {BATCH_SOLVER.upper()} (first-order STMs, Gauss-Newton iterations) on growing arcs...")
    x0_ref1 = x0_ref.copy()
    for T_arc in STAGE1_ARCS_H:
        k = int(np.searchsorted(tau, T_arc * 3600.0 + 1e-6))
        print(f"\n[Stage 1] arc 0-{tau[k - 1] / 3600:.1f} h ({int(vis_mask[:k].sum())} visible)")
        batch = BatchLSQ(
            propagator=prop1,
            tau=tau[:k],
            sc_state=orex_state[:k],
            y_obs=y_obs[: 2 * k],
            sigma=sigma,
            obs_weights=obs_weights[: 2 * k],
            measurement=range_rangerate_and_partials,
            rel_dim=6,
        )
        x0_ref1, delta_hat1, cov1 = batch.gn_with_stm(
            x0_ref=x0_ref1,
            priors=[norm(loc=m, scale=s) for m, s in zip(x0_ref - x0_ref1, prior_sigma)],
            max_iter=STAGE1_ITERS,
            tol=STAGE1_TOL,
            rtol=ivp["rtol"],
            atol=ivp["atol"],
            method=ivp["method"],
            verbose=True,
            solver=BATCH_SOLVER,
            make_chart=lambda x: SymmetryChart(x, pole_i),
        )
        print(f"[Stage 1] exact MAP cost on the arc: {exact_map_cost_many(x0_ref1, k)[0]:.6f}")

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
        """Doppler residuals of chart coordinates theta: STT on the slice, rotated back."""
        x6 = chart.rotate(stt_trajectory(chart.slice_deviation(theta)), chart.angle(theta))
        return isl_residuals(x6, orex_state, y_obs, sigma, obs_weights)

    def chi2_plain_stt(dx):
        return float(np.sum(isl_residuals(stt_trajectory(dx), orex_state, y_obs, sigma, obs_weights) ** 2))

    def chi2_sym_stt(dx):
        theta = sym.from_deviation(dx)
        x6 = sym.rotate(stt_trajectory(sym.slice_deviation(theta)), sym.angle(theta))
        return float(np.sum(isl_residuals(x6, orex_state, y_obs, sigma, obs_weights) ** 2))

    chi2_ref = float(np.sum(residuals_normalized(np.zeros(12)) ** 2))
    print(
        f"[Stage 2] At ref1: chi2_red = {chi2_ref / dof:.3f} (chi2 = {chi2_ref:.2f}, dof = {dof}, "
        f"n_vis = {n_visible})"
    )

    # Near-symmetry: data chi2 along the pure g_delta(ref1) (OREx breaks it; the posterior
    # follows g_delta combined with small changes of the other parameters), exact vs plain STT
    # vs symmetry-reduced STT (exact in delta for the dynamics)
    print("\n[Symmetry] data chi2 - chi2(ref1) at g_delta(ref1):")
    print("   delta [deg]       exact    plain STT   symmetry STT")
    angles = np.deg2rad([-20.0, -10.0, -5.0, 5.0, 10.0, 20.0])
    X_rot = sym.rotate(np.tile(x0_ref1, (angles.size, 1)), angles)
    traj_rot = gravity1.propagate_many(np.vstack([x0_ref1, X_rot]), tau, **ivp)
    chi2_ex = np.sum(isl_residuals(traj_rot, orex_state, y_obs, sigma, obs_weights) ** 2, axis=-1)
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
    c_exact = np.sum(isl_residuals(traj, orex_state, y_obs, sigma, obs_weights) ** 2, axis=-1)
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

    # Curvature of the degree-2 pairs (absolute coefficients): b bent along a
    sh_names = ["C20", "C21", "S21", "C22", "S22"]
    bends = sorted(
        (
            (curvature_metric(x_post[:, 7 + i], x_post[:, 7 + j]), sh_names[i], sh_names[j])
            for i in range(5)
            for j in range(5)
            if i != j
        ),
        reverse=True,
    )
    print("[Summary] curvature of the C/S pairs (sagitta / scatter over +-2 sd; Gaussian ~0, curved > 1.5):")
    print("  " + ", ".join(f"{b} along {a}: {m:.2f}" for m, a, b in bends[:6]))

    # --------------------------
    # Figures
    # --------------------------
    to_dx = chart.to_deviation
    _figure(
        "post-fit residuals",
        lambda: MCMCPlots.plot_mcmc_residuals(
            model, tau, vis_mask, np.zeros(12), components=[0, 1] if USE_RANGE else [1]
        ),
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
        t_gmm = gmm_target_hours * 3600.0
        utc_gmm = spice.et2utc(et0 + t_gmm, "ISOC", 0).replace("T", " ")
        gmm_prediction(
            samples_dx,
            x0_ref1,
            cov1,
            x0_true,
            prop1,
            gravity1.propagate_many,
            np.linspace(0.0, t_gmm, n_t_prop),
            tau[-1],
            ivp,
            (
                f"Posterior at Epoch ({utc0.replace('T', ' ')} UTC)",
                f"Predicted at {utc_gmm} UTC (+{gmm_target_hours:.2f} h)",
            ),
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
