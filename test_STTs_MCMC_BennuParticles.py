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

# TODOs for this script:
# TODO: Check correctness of gravity, J2, and SRP (formulas, numbers, and units)
# TODO: Validate nominal propagation: does the trajectory make physical sense?
# TODO: Implement visibility constraints if needed (line-of-sight to Sun or observer)
# TODO: Fix plots, especially post-fits residuals
# TODO: Add SPH
# TODO: How to make the scenario more realistic/challenging?


def generate_stt_functions(
    mu, order, R_eq=0.246, J2=8.64e-5, Cr=1.0, A_m=1.0, P0=4.56e-6
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


if __name__ == "__main__":
    # Constants for Bennu
    R_bennu = 0.246  # [km] approximate mean radius
    mu = 4.892e-9  # [km^3/s^2] Bennu GM
    order = 3

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
    v_mag = 2 * 1e-4  # [km/s], small detachment velocity

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
    # t_obs = JD0_seconds + np.linspace(
    #    0, 0.05 * 3600, int((0.05 * 3600) / 20)
    # )  # 20-sec cadence
    t_obs = JD0_seconds + np.linspace(0, 0.05 * 3600, num=100)

    # Generate symbolic dynamics functions externally
    f_func, A_func, B_funcs = generate_stt_functions(mu, order)

    # Instantiate STT propagator with provided symbolic functions
    propagator = STTPropagator(
        order=order, f_func=f_func, A_func=A_func, B_funcs=B_funcs
    )

    # Propagate truth
    print("")
    print(f"\nPropagating true trajectory:")
    sol_true, stts_true = propagator.propagate(x0_true, t_obs, rtol=1e-10, atol=1e-12)
    x_true = sol_true.y[:6, :].T  # shape (n_steps, 6)

    # Measurement model (OpNav angles)
    sigma_ra = np.deg2rad(0.005)  # ~18 arcsec
    sigma_dec = np.deg2rad(0.005)

    sc_pos = np.array(
        [0.0, 0.0, 2.0]
    )  # Fixed observer in Bennu-centered inertial frame

    y_obs = generate_opnav_measurements(x_true, sc_pos, sigma_ra, sigma_dec)
    t_obs_used = t_obs  # all times used in this simplified case

    # Propagate reference trajectory
    print("")
    print("\nPropagating reference trajectory:")

    # Define relative deviation fractions (e.g., 1% for position, 0.5% for velocity)
    pos_dev_frac = 0.01  # 1% of position
    vel_dev_frac = 0.005  # 0.5% of velocity

    # Compute component-wise deviation
    ref_dev = np.hstack([pos_dev_frac * x0_true[:3], vel_dev_frac * x0_true[3:]])
    print(f"Reference deviation: {ref_dev}")
    idx0 = np.searchsorted(t_obs, t_obs_used[0])
    x0_ref = sol_true.y[:6, idx0] - ref_dev

    sol_ref, stts_ref = propagator.propagate(
        x0=x0_ref,
        t_eval=t_obs_used,
        rtol=1e-12,
        atol=1e-14,
    )

    # Plot: 3D trajectory and observer LOS vectors
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Facecolor (e.g., light brown or gray)
    face_color = "#b88b4a"

    # Plot asteroid mesh
    mesh = Poly3DCollection(
        vertices[faces],
        alpha=0.3,
        edgecolor="k",
        linewidths=0.2,
        facecolor=face_color,
    )
    ax.add_collection3d(mesh)

    # Particle Trajectory
    ax.plot(
        x_true[:, 0],
        x_true[:, 1],
        x_true[:, 2],
        label="Particle Trajectory",
        color="blue",
    )

    # Detachment Point
    ax.scatter(*pos_detach, color="green", label="Detachment Point", s=30)

    # Spacecraft Position
    ax.scatter(*sc_pos, color="red", label="Spacecraft", s=50)

    # Line-of-Sight Vectors
    N_skip = 30
    for i in range(0, len(x_true), N_skip):
        ax.plot(
            [sc_pos[0], x_true[i, 0]],
            [sc_pos[1], x_true[i, 1]],
            [sc_pos[2], x_true[i, 2]],
            color="gray",
            alpha=0.3,
        )

    # Plot Labels and Appearance
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    ax.set_aspect("equal")
    plt.show()

    # Plot RA and DEC over time
    ra_vals = y_obs[0::2]  # Extract RA measurements
    dec_vals = y_obs[1::2]  # Extract DEC measurements

    # Convert observation times to minutes since start
    time_minutes = (t_obs_used - t_obs_used[0]) / 60.0

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # RA scatter
    ax2[0].scatter(time_minutes, np.rad2deg(ra_vals), color="purple", s=10, label="RA")
    ax2[0].set_ylabel("RA [deg]")
    ax2[0].grid(True)
    ax2[0].legend()

    # DEC scatter
    ax2[1].scatter(
        time_minutes, np.rad2deg(dec_vals), color="darkorange", s=10, label="DEC"
    )
    ax2[1].set_xlabel("Time [min]")
    ax2[1].set_ylabel("DEC [deg]")
    ax2[1].grid(True)
    ax2[1].legend()

    fig2.suptitle("Simulated OpNav Angular Measurements (Scatter)")
    plt.tight_layout()
    plt.show()

    # Residual function for MCMC — OpNav angular case
    def residuals_normalized(delta_x0):
        # 1. Propagate the perturbed trajectory
        _, x_est = propagator.propagate_deviation(sol_ref, stts_ref, delta_x0)

        # 2. Line-of-sight vector: target - observer
        los_vec = x_est[:, :3] - sc_pos  # shape (N, 3)

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

        # 7. Residuals (assuming y_obs in radians and wrapped consistently)
        residuals = y_obs - y_model

        # 8. Normalize by measurement uncertainties
        weights = np.empty_like(y_obs)
        weights[0::2] = sigma_ra
        weights[1::2] = sigma_dec

        return residuals / weights

    # Assign weights to the residuals
    weights = np.hstack(
        [
            np.full(len(t_obs_used), sigma_ra),
            np.full(len(t_obs_used), sigma_dec),
        ]
    )

    # Evaluate residuals at delta_x0 = 0 (prefit)
    delta_prefit = np.zeros(6)

    # Run Batch Estimation
    print("")
    print("\nRunning STT batch least-squares estimation...")
    batch_result, batch_cov = compute_STT_batch_solution(
        residuals_func=residuals_normalized, x0=delta_prefit, sigma=weights
    )
    batch_estimate = batch_result.x

    # Call the unified plot function
    plot_normalized_residuals_vs_time(
        residuals_func=residuals_normalized,
        delta_x0_list=[delta_prefit, batch_estimate],
        labels=["Prefit", "Batch"],
        colors=["blue", "green"],
        t_obs_used=t_obs_used,
    )
    plt.show()

    # Print batch solution summary
    print("\n=== Batch Least-Squares Estimate ===")
    param_names = ["Δx", "Δy", "Δz", "Δvx", "Δvy", "Δvz"]
    batch_std = np.sqrt(np.diag(batch_cov))
    for i, name in enumerate(param_names):
        estimate = batch_estimate[i]
        sigma = batch_std[i]
        print(f"{name:>5}: {estimate:+.6e} ± {sigma:.2e} [σ]")

    # Priors
    initial_guess = np.zeros(6)
    pos_lower, pos_upper = -1e-1, 1e-1  # Position in km
    vel_lower, vel_upper = -1e-3, 1e-3  # Velocity in km/s
    priors = [norm(loc=0.0, scale=pos_upper) for _ in range(3)] + [  # position in km
        norm(loc=0.0, scale=vel_upper) for _ in range(3)  # velocity in km/s
    ]

    # Run MCMC
    model = MCMCModel(
        residuals_func=residuals_normalized,
        initial_params=initial_guess,
        param_priors=priors,
        observed_data=y_obs,
    )
    model.setup_whitening_from_priors()
    model.run(n_samples=1000, n_walkers=128, burn_in=500, thin=1, spherical_spread=1e-1)
    model.plot_convergence()
    model.plot_postfit_residuals_time(t_obs_used=t_obs_used, opnav_data=True)
    model.plot_log_likelihood()
    model.plot_corner_with_batch(batch_mean=batch_estimate, batch_cov=batch_cov)
    model.summary()
    model.print_regression_diagnostics()
    model.gelman_rubin_diagnostic()
    model.effective_sample_size()
    model.plot_autocorrelation()
