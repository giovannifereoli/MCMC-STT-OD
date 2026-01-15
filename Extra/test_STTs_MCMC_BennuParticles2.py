import sympy as sp
import numpy as np
from itertools import product
from scipy.stats import norm
from MCMC import MCMCModel
from astropy.time import Time
import matplotlib.pyplot as plt
from STTPropagation import STTPropagator
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares

# NOTE: Hypothesis (HP)
# Spacecraft hovering in Bennu’s body-fixed frame. OpNav measurements are generated from a fixed observer.

# TODO: Scenario Setup & Physical Model
# - Verify correctness and physical units of: gravity model, J2 perturbation, solar radiation pressure (SRP)
# - Verify correctness of __main__ setup
# - Increase realism / complexity of the scenario: visibility constraints, realistic observation geometry, OSIRIS-REx SPKs
# - Consolidate and clearly separate the three scenarios:
#     1) Nominal case (well-behaved, working baseline)
#     2) Banana-shaped posterior case
#     3) IOD / multimodal / batch OD failing case
# - Improve plotting routines (labels, spacing, reference handling, etc.)

# NOTE:
# This MCMC framework is meant to help the OD analyst explore the posterior distribution of the state estimate, especially
# when different OD solutions obtained under different assumptions are mutually inconsistent.

# Typical usage:
# - Take the converged OD solution with the highest confidence
# - Sample the likelihood locally around that reference solution

# Important:
# - MCMC is a *sampler*, not an optimizer. It can behave like an optimizer in: IOD problems AND ill-posed or strongly multimodal likelihoods.
# - However, MCMC does NOT naturally funnel toward low-likelihood regions and tends to wander through high-likelihood space.
#   Results in poorly constrained or strongly multimodal cases. should be interpreted with caution. Chains can look
#   converged while still hiding valid solutions (“bee vs. house” analogy).
#
# In this framework:
# - Starting from STTs close to a valid OD solution is sufficient and efficient for local posterior sampling
# - Keeping DEMoves enabled generally improves exploration

# For IOD problems it is recommended to:
# - Initialize walkers inside admissible regions and leave walkers in the virtual-asteroid space
# - Alternatively, brute-force a hypercube if the state vector includes more than position and velocity
# - Genetic algorithms over admissible regions can be tried, but they are not equivalent in rigor to this approach
# - Walkers stuck in persistently low-likelihood regions should be removed, as they degrade chain statistics
# - Use uniform grids or Adaptive Domain Splitting (ADS) following Armellin et al. to generate STTs per sub-domain


def generate_stt_functions(
    mu, order, R_eq=0.290, J2=1.962e-5, Cr=1.2, A_m=0.1, P0=4.56e-6
):
    """
    Generate f, A, and B_k symbolically for STT propagation including:
    - Point-mass gravity
    - J2 perturbation (central body assumed aligned with z-axis)
    - Solar radiation pressure (SRP) in inertial frame

    Parameters:
    - mu: gravitational parameter [km^3/s^2]
    - order: max STT order
    - R_eq: reference radius of the body [km]
    - J2: J2 coefficient
    - Cr: radiation pressure coefficient
    - A_m: area-to-mass ratio [m^2/kg]
    - P0: solar radiation pressure at 1 AU [N/m^2] in km units
    """
    x_syms = sp.symbols("x y z vx vy vz")
    x, y, z, vx, vy, vz = x_syms
    r_vec = sp.Matrix([x, y, z])
    v_vec = sp.Matrix([vx, vy, vz])
    r = sp.sqrt(x**2 + y**2 + z**2)
    r2 = x**2 + y**2 + z**2
    r5 = r2 ** (5 / 2)
    z2 = z**2

    # Gravity: point-mass + J2
    a_pm = -mu * r_vec / r**3
    a_j2 = (
        (3 / 2)
        * J2
        * mu
        * R_eq**2
        / r**5
        * sp.Matrix(
            [x * (5 * z2 / r2 - 1), y * (5 * z2 / r2 - 1), z * (5 * z2 / r2 - 3)]
        )
    )
    a_grav = a_pm + a_j2

    # SRP: from Sun at fixed direction (here assume along +x), scaled in km/s²
    # 1 N/kg = 1e-3 km/s² (since 1 m/s² = 1e-3 km/s²)
    P_srp = P0 * 1e-3  # convert to km/s²
    sun_dir = sp.Matrix([1, 0, 0])
    a_srp = Cr * A_m * P_srp * sun_dir

    # Total acceleration
    a_total = a_grav + a_srp

    # Full symbolic dynamics
    f_sym = sp.Matrix([vx, vy, vz, *a_total])

    # First-order STM
    X = sp.Matrix(x_syms)
    B_syms = {1: f_sym.jacobian(X)}

    # Higher-order tensors
    for k in range(2, order + 1):
        shape = (6,) * (k + 1)
        Bk = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(range(6), repeat=k + 1):
            i, *js = idx
            deriv = sp.diff(f_sym[i], *[x_syms[j] for j in js])
            Bk[idx] = deriv
        B_syms[k] = Bk

    # Lambdify everything
    f_func = sp.lambdify(x_syms, f_sym, "numpy")
    A_func = sp.lambdify(x_syms, B_syms[1], "numpy")
    B_funcs = {
        k: sp.lambdify(x_syms, B_syms[k].tolist(), "numpy") for k in range(2, order + 1)
    }

    return f_func, A_func, B_funcs


# Generate angular measurements (RA/DEC) from particle to spacecraft
def generate_opnav_measurements(x_true, sc_pos, sigma_ra, sigma_dec):
    los_vec = x_true[:, :3] - sc_pos  # shape (N, 3)
    los_unit = los_vec / np.linalg.norm(los_vec, axis=1, keepdims=True)

    ra = np.arctan2(los_unit[:, 1], los_unit[:, 0])
    dec = np.arcsin(los_unit[:, 2])
    ra = np.mod(ra, 2 * np.pi)  # wrap to [0, 2pi]

    # Add noise
    ra += np.random.normal(0, sigma_ra, size=ra.shape)
    ra = np.mod(ra, 2 * np.pi)  # wrap to [0, 2pi]
    dec += np.random.normal(0, sigma_dec, size=dec.shape)

    # Interleave: [ra0, dec0, ra1, dec1, ...]
    y_obs = np.empty(2 * len(ra))
    y_obs[0::2] = ra
    y_obs[1::2] = dec

    return y_obs


def compute_STT_batch_solution(residuals_func, x0, sigma):
    # Define raw (normalized) residual function for LS
    def raw_residuals(delta_x0):
        res = residuals_func(delta_x0)
        return res  # normalized residuals

    # Run nonlinear least-squares (trust-region or LM)
    result = least_squares(
        fun=raw_residuals, x0=x0, method="trf", jac="2-point", verbose=2
    )

    # Estimate covariance from inverse JTJ
    J = result.jac  # shape (m, n)
    JTJ = J.T @ J
    cov = np.linalg.inv(JTJ)

    return result, cov


def plot_normalized_residuals_vs_time(
    residuals_func, delta_x0_list, labels, colors, t_obs_used
):
    assert len(delta_x0_list) == len(labels) == len(colors), "Mismatched input lengths."

    time_hr = (t_obs_used - t_obs_used[0]) / 3600.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for delta_x0, label, color in zip(delta_x0_list, labels, colors):
        res = residuals_func(delta_x0)
        ra_res = res[0::2]
        dec_res = res[1::2]

        ax1.plot(time_hr, ra_res, "o", markersize=4, color=color, label=f"RA {label}")
        ax2.plot(time_hr, dec_res, "o", markersize=4, color=color, label=f"DEC {label}")

    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linestyle="--")
        ax.axhline(3, color="red", linestyle=":")
        ax.axhline(-3, color="red", linestyle=":")
        ax.grid(True)
        ax.legend(loc="upper right")

    ax1.set_ylabel("RA Residual [$\\sigma$]")
    ax2.set_ylabel("DEC Residual [$\\sigma$]")
    ax2.set_xlabel("Time [hours since epoch]")
    plt.tight_layout()
    plt.show()


def plot_estimation_error_and_covariance(
    stts,
    sol,
    x_truth,
    delta_batch,
    P_batch,
    delta_mcmc,
    P_mcmc,
    propagator,
    t_obs,
    labels=["Batch", "MCMC"],
    colors=["blue", "orange"],
):
    # Propagate deviations
    err_batch, traj_batch = propagator.propagate_deviation(
        stts=stts, sol=sol, delta_x0=delta_batch
    )
    err_mcmc, traj_mcmc = propagator.propagate_deviation(
        stts=stts, sol=sol, delta_x0=delta_mcmc
    )

    # Propagate covariances
    P_batch_t = propagator.propagate_covariance(sol, stts, P_batch)
    P_mcmc_t = propagator.propagate_covariance(sol, stts, P_mcmc)

    # Compute true error
    err_batch = traj_batch - x_truth
    err_mcmc = traj_mcmc - x_truth

    # Time in minutes
    time_min = (t_obs - t_obs[0]) / 60.0

    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    components = ["x", "y", "z"]

    for i in range(3):  # Position components
        for err, P_t, label, color in zip(
            [err_batch, err_mcmc],
            [P_batch_t, P_mcmc_t],
            labels,
            colors,
        ):
            # Absolute error for semilogy
            axs[i].semilogy(
                time_min,
                np.abs(err[:, i]),
                label=f"{label} Error",
                color=color,
                linewidth=2.0,
            )
            axs[i].scatter(
                time_min,
                np.abs(err[:, i]),
                color=color,
                s=9,
                alpha=0.3,
            )
            axs[i].fill_between(
                time_min,
                3 * np.sqrt(P_t[:, i, i]),
                color=color,
                alpha=0.2,
                label=rf"{label} $\pm3\sigma$" if i == 0 else None,
            )

        axs[i].set_ylabel(rf"$|\Delta {components[i]}|$ [km]")
        axs[i].grid(True, which="both")

    for i in range(3):  # Velocity components
        for err, P_t, label, color in zip(
            [err_batch, err_mcmc],
            [P_batch_t, P_mcmc_t],
            labels,
            colors,
        ):
            axs[i + 3].semilogy(
                time_min,
                np.abs(err[:, i + 3]),
                label=f"{label} Error",
                color=color,
                linewidth=2.0,
            )
            axs[i + 3].scatter(
                time_min,
                np.abs(err[:, i + 3]),
                color=color,
                s=9,
                alpha=0.3,
            )
            axs[i + 3].fill_between(
                time_min,
                3 * np.sqrt(P_t[:, i + 3, i + 3]),
                color=color,
                alpha=0.2,
                label=rf"{label} $\pm3\sigma$" if i == 0 else None,
            )

        axs[i + 3].set_ylabel(rf"$|\Delta \dot{{{components[i]}}}|$ [km/s]")
        axs[i + 3].grid(True, which="both")

    axs[-1].set_xlabel("Time [min]")
    axs[0].legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()


# ============================================================
# NOTE: Concrete recipe to generate a "banana-shaped" posterior
# ============================================================
# Goal: induce a curved likelihood ridge (angles-only, short-arc, weak parallax),
#       so MCMC corner plots show a banana correlation between state components.
#
# Recipe (minimal edits):
#
# 1) Short arc + fewer observations (range/along-track weakly observed)
#    t_obs = JD0_seconds + np.linspace(0, 900, num=10)     # 15 min, 10 obs
#    # or even stronger:
#    # t_obs = JD0_seconds + np.linspace(0, 600, num=8)    # 10 min, 8 obs
#
# 2) Reduce parallax by moving the observer farther (still fixed)
#    sc_pos = np.array([0.0, 0.0, 10.0])                  # was 2.0 km
#
# 3) Make motion mostly tangential to surface (LOS changes slowly)
#    rng = np.random.default_rng(24)
#    rand_vec = rng.normal(size=3); rand_vec /= np.linalg.norm(rand_vec)
#    tang = rand_vec - np.dot(rand_vec, normal) * normal   # remove radial component
#    tang /= np.linalg.norm(tang)
#    v_detach = v_mag * tang
#
# 4) Increase measurement noise (widens ridge so curvature is visible)
#    sigma_ra  = np.deg2rad(0.05)                         # 10x baseline
#    sigma_dec = np.deg2rad(0.05)
#
# 5) Loosen priors so likelihood geometry dominates (avoid Gaussianizing)
#    prior_sigma_pos *= 5.0
#    prior_sigma_vel *= 5.0
#
# Expected outcome:
# - Short arc + weak parallax + tangential motion -> strong nonlinear coupling
# - MCMC corner plot shows curved (banana) correlations in position/velocity.


if __name__ == "__main__":
    # Constants for Bennu
    R_bennu = 0.290  # [km] approximate mean radius
    mu = 4.892e-9  # [km^3/s^2] Bennu GM
    order = 2

    # Load Bennu mesh
    mesh_path = "ObjFiles/BennuRadar.obj"  # Update with correct path if needed
    bennu_mesh = trimesh.load(mesh_path, force="mesh")
    vertices = bennu_mesh.vertices  # shape (N, 3)
    faces = bennu_mesh.faces  # shape (M, 3)

    # Convert (lat, lon) to target position
    lat_desired = np.deg2rad(45.0)  # latitude in radians
    lon_desired = np.deg2rad(80.0)  # longitude in radians

    x_des = R_bennu * np.cos(lat_desired) * np.cos(lon_desired)
    y_des = R_bennu * np.cos(lat_desired) * np.sin(lon_desired)
    z_des = R_bennu * np.sin(lat_desired)
    pos_target = np.array([x_des, y_des, z_des])

    # Find closest vertex on mesh
    dists = np.linalg.norm(vertices - pos_target, axis=1)
    closest_idx = np.argmin(dists)
    pos_detach = vertices[closest_idx]

    # Get surface normal at that point
    normal = bennu_mesh.vertex_normals[closest_idx]
    normal = normal / np.linalg.norm(normal)

    # Initial velocity parameters
    v_mag = 2 * 1e-4  # [km/s]

    # Generate random unit vector
    np.random.seed(24)  # For reproducibility
    rand_vec = np.random.randn(3)
    rand_vec /= np.linalg.norm(rand_vec)

    # Ensure it lies in the positive hemisphere relative to the normal
    if np.dot(rand_vec, normal) < 0:
        rand_vec = -rand_vec  # Flip to ensure outward direction

    # Apply magnitude
    v_detach = v_mag * rand_vec

    # Construct state vector
    x0_true = np.hstack((pos_detach, v_detach))

    # Time setup
    JD0 = Time("2025-04-24T00:00:00", scale="utc").jd
    JD0_seconds = (JD0 - Time("2000-01-01T12:00:00", scale="utc").jd) * 86400.0
    t_obs = JD0_seconds + np.linspace(0, 6 * 3600, num=6)  # CASE 3

    # Generate symbolic dynamics functions externally
    f_func, A_func, B_funcs = generate_stt_functions(mu, order)

    # Instantiate STT propagator with provided symbolic functions
    propagator = STTPropagator(
        order=order, f_func=f_func, A_func=A_func, B_funcs=B_funcs
    )

    # Propagate truth
    print("\nPropagating true trajectory:")
    sol_true, stts_true = propagator.propagate(x0_true, t_obs, rtol=1e-10, atol=1e-12)
    x_true = sol_true.y[:6, :].T  # shape (n_steps, 6)

    # Measurement model (OpNav angles)
    sigma_ra = np.deg2rad(0.005)  # 20 arcsec
    sigma_dec = np.deg2rad(0.005)

    sc_pos = np.array(
        [0.0, 0.0, 5.0]
    )  # Fixed observer in Bennu-centered inertial frame

    y_obs = generate_opnav_measurements(x_true, sc_pos, sigma_ra, sigma_dec)
    t_obs_used = t_obs  # all times used in this simplified case

    # ============================================================
    # Initial reference trajectory and Priors
    # ============================================================
    print("\nPropagating reference trajectory:")

    # Fractional deviation (tunable)
    pos_dev_frac = 0.01  # 1% position
    vel_dev_frac = 0.01  # 1% velocity

    rng = np.random.default_rng(42)

    # Randomized reference deviation (same scale as fractions)
    ref_dev_pos = pos_dev_frac * x0_true[:3] * rng.normal(size=3)
    ref_dev_vel = vel_dev_frac * x0_true[3:] * rng.normal(size=3)
    ref_dev = np.hstack([ref_dev_pos, ref_dev_vel])

    print(f"Reference deviation: {ref_dev}")

    # truth - ref = ref_dev
    idx0 = np.searchsorted(t_obs, t_obs_used[0])
    x0_ref0 = sol_true.y[:6, idx0] - ref_dev

    # Propagate reference
    sol_ref0, stts_ref0 = propagator.propagate(
        x0=x0_ref0,
        t_eval=t_obs_used,
        rtol=1e-12,
        atol=1e-14,
    )

    # ============================================================
    # Priors (matched to fractional deviation scale)
    # ============================================================
    # Prior std proportional to the same fractional deviation
    increase_factor = 1.0  # Loosen priors to allow more exploration
    prior_sigma_pos = increase_factor * pos_dev_frac * np.abs(x0_ref0[:3])
    prior_sigma_vel = increase_factor * vel_dev_frac * np.abs(x0_ref0[3:])

    # Avoid zero-variance if a component is near zero
    prior_sigma_pos = np.maximum(prior_sigma_pos, 1e-6)  # km
    prior_sigma_vel = np.maximum(prior_sigma_vel, 1e-9)  # km/s

    priors = [norm(loc=0.0, scale=s) for s in prior_sigma_pos] + [
        norm(loc=0.0, scale=s) for s in prior_sigma_vel
    ]

    initial_guess = np.zeros(6)

    # ============================================================
    # Helpers (keep your local-function style)
    # ============================================================
    def wrap_to_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    # Stage 1 residual: FULL nonlinear (NO STTs)
    def residuals_full_normalized(delta_x0, x0_ref):
        sol, _ = propagator.propagate(
            x0=x0_ref + delta_x0,
            t_eval=t_obs_used,
            rtol=1e-12,
            atol=1e-14,
        )
        x_est = sol.y[:6, :].T

        los_vec = x_est[:, :3] - sc_pos
        los_unit = los_vec / np.linalg.norm(los_vec, axis=1, keepdims=True)

        ra_model = np.mod(np.arctan2(los_unit[:, 1], los_unit[:, 0]), 2 * np.pi)
        dec_model = np.arcsin(los_unit[:, 2])

        y_model = np.empty_like(y_obs)
        y_model[0::2] = ra_model
        y_model[1::2] = dec_model

        residuals = np.empty_like(y_obs)
        residuals[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
        residuals[1::2] = y_obs[1::2] - y_model[1::2]

        weights = np.empty_like(y_obs)
        weights[0::2] = sigma_ra
        weights[1::2] = sigma_dec

        return residuals / weights

    def solve_batch_nonlinear_full(
        x0_ref,
        x0_delta0,
        priors=None,  # list of scipy.stats.norm, length 6
        max_nfev=40000,
    ):
        # Build prior mean/sigma from your norm() objects
        if priors is None:
            prior_mean = np.zeros_like(x0_delta0)
            prior_sigma = np.full_like(x0_delta0, np.inf)  # no prior
        else:
            prior_mean = np.array([p.mean() for p in priors], dtype=float)
            prior_sigma = np.array([p.std() for p in priors], dtype=float)

            if np.any(prior_sigma <= 0):
                raise ValueError("Prior std must be > 0 for all parameters.")

        def fun(d):
            # measurement residuals (already normalized by sigma_ra/sigma_dec)
            r_meas = residuals_full_normalized(d, x0_ref)

            # prior residuals (MAP): (d - mu)/sigma
            r_prior = (d - prior_mean) / prior_sigma

            return np.hstack([r_meas, r_prior])

        result = least_squares(
            fun=fun,
            x0=x0_delta0,
            method="trf",
            jac="2-point",
            max_nfev=max_nfev,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            verbose=2,
        )

        # Posterior covariance (Gauss-Newton approx around MAP)
        J = result.jac
        cov = np.linalg.inv(J.T @ J)

        return result, cov

    # ============================================================
    # Stage 1: FULL nonlinear batch to convergence (NO STTs)
    # ============================================================
    print("\n[Stage 1] Full nonlinear batch (NO STTs) to convergence...")
    batch1, cov1 = solve_batch_nonlinear_full(
        x0_ref0, initial_guess, priors=priors, max_nfev=40000
    )
    delta_hat1 = batch1.x

    chi2_1 = np.sum(residuals_full_normalized(delta_hat1, x0_ref0) ** 2)
    dof_1 = len(y_obs) - len(delta_hat1)
    chi2_red_1 = chi2_1 / dof_1

    print("\n[Stage 1] delta_hat1:", delta_hat1)
    print(f"[Stage 1] chi2_red = {chi2_red_1:.3f}  (chi2={chi2_1:.1f}, dof={dof_1})")

    # ============================================================
    # Stage 2: relinearize STTs about ref1 = ref0 + delta_hat1
    # ============================================================
    x0_ref1 = x0_ref0 + delta_hat1

    print("\n[Stage 2] Propagating ref1 and computing STTs about ref1...")
    sol_ref, stts_ref = propagator.propagate(
        x0=x0_ref1,
        t_eval=t_obs_used,
        rtol=1e-12,
        atol=1e-14,
    )

    # ============================================================
    # Residual function for MCMC — OpNav angular case (STT-based)
    # ============================================================
    def residuals_normalized(delta_x0):
        # 1. Propagate the perturbed trajectory
        _, x_est = propagator.propagate_deviation(sol_ref, stts_ref, delta_x0)

        # 2. Line-of-sight vector: target - observer
        los_vec = x_est[:, :3] - sc_pos

        # 3. Normalize to get unit vectors
        los_unit = los_vec / np.linalg.norm(los_vec, axis=1, keepdims=True)

        # 4. Convert to RA and DEC
        ra_model = np.arctan2(los_unit[:, 1], los_unit[:, 0])
        dec_model = np.arcsin(los_unit[:, 2])

        # 5. Wrap RA to [0, 2pi] to match y_obs
        ra_model = np.mod(ra_model, 2 * np.pi)

        # 6. Stack and flatten to match y_obs structure
        y_model = np.empty_like(y_obs)
        y_model[0::2] = ra_model
        y_model[1::2] = dec_model

        # 7. Residuals
        residuals = np.empty_like(y_obs)
        residuals[0::2] = wrap_to_pi(y_obs[0::2] - y_model[0::2])
        residuals[1::2] = y_obs[1::2] - y_model[1::2]

        # 8. Normalize
        weights = np.empty_like(y_obs)
        weights[0::2] = sigma_ra
        weights[1::2] = sigma_dec
        return residuals / weights

    # Correct interleaved weights (matches y_obs layout)
    weights = np.empty_like(y_obs)
    weights[0::2] = sigma_ra
    weights[1::2] = sigma_dec

    # ============================================================
    # Stage 2 batch refine (STT-based)
    # ============================================================
    print("\n[Stage 2] Running STT batch least-squares estimation...")
    batch_result, batch_cov = compute_STT_batch_solution(
        residuals_func=residuals_normalized, x0=np.zeros(6), sigma=weights
    )
    batch_estimate = batch_result.x

    chi2_2 = np.sum(residuals_normalized(batch_estimate) ** 2)
    dof_2 = len(y_obs) - len(batch_estimate)
    chi2_red_2 = chi2_2 / dof_2

    print("\n[Stage 2] delta_hat2:", batch_estimate)
    print(f"[Stage 2] chi2_red = {chi2_red_2:.3f}  (chi2={chi2_2:.1f}, dof={dof_2})")

    # ============================================================
    # MCMC
    # ============================================================
    model = MCMCModel(
        residuals_func=residuals_normalized,
        initial_params=initial_guess,
        param_priors=priors,
        observed_data=y_obs,
    )
    model.setup_whitening_from_priors()
    model.run(
        n_samples=5000,
        n_walkers=128,
        burn_in=2000,
        thin=10,
        spherical_spread=1e-3,
        method_optimize="Powell",
        use_demoves=False,
    )

    # Truth delta w.r.t ref1 (since MCMC is about ref1)
    true_theta_about_ref1 = x0_true - x0_ref1  # == ref_dev - delta_hat1

    model.plot_convergence()
    model.plot_postfit_residuals_time(t_obs_used=t_obs_used, opnav_data=True)
    model.plot_log_likelihood()
    model.plot_corner_with_batch(
        batch_mean=batch_estimate,
        batch_cov=batch_cov,
        use_median_as_truth=False,
        true_theta=true_theta_about_ref1,
    )
    model.summary()
    model.print_regression_diagnostics()
    model.gelman_rubin_diagnostic()
    model.plot_autocorrelation()

    theta_hat, cov = model.get_estimate_and_covariance()

    plot_estimation_error_and_covariance(
        stts=stts_ref,
        sol=sol_ref,
        x_truth=x_true,
        delta_batch=batch_estimate,
        P_batch=batch_cov,
        delta_mcmc=theta_hat,
        P_mcmc=cov,
        propagator=propagator,
        t_obs=t_obs_used,
    )
