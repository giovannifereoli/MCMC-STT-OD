import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import math
from itertools import product
from scipy.stats import norm, uniform
from MCMC import MCMCModel
from scipy.constants import pi
from scipy.spatial.transform import Rotation as R
from astropy.time import Time
import matplotlib.pyplot as plt


def generate_stt_functions(mu, order, beta=0.0):
    """
    Symbolically generate f, A and B_k up to arbitrary 'order',
    including a drag term modeled as a = -beta * |v| * v.
    """
    # 1) State symbols
    x_syms = sp.symbols("x y z vx vy vz")
    x, y, z, vx, vy, vz = x_syms
    mu_sym = sp.Float(mu)
    beta_sym = sp.Float(beta)

    # 2) Define position and velocity magnitude
    r = sp.sqrt(x**2 + y**2 + z**2)
    v = sp.sqrt(vx**2 + vy**2 + vz**2)

    # 3) Two-body + drag dynamics
    a_grav = -mu_sym * sp.Matrix([x, y, z]) / r**3
    a_drag = -beta_sym * v * sp.Matrix([vx, vy, vz])
    a_total = a_grav + a_drag

    # 4) Full f vector
    f_sym = sp.Matrix([vx, vy, vz, *a_total])

    # 3) STM generator
    X = sp.Matrix(x_syms)
    B_syms = {1: f_sym.jacobian(X)}

    # 4) build B_syms[2..order]
    for k in range(2, order + 1):
        shape = (6,) * (k + 1)
        Bk = sp.MutableDenseNDimArray.zeros(*shape)

        for idx in product(range(6), repeat=k + 1):  # (i, j1, ..., jk)
            i, *js = idx
            deriv = sp.diff(f_sym[i], *[x_syms[j] for j in js])
            Bk[idx] = deriv

        B_syms[k] = Bk

    # 5) lambdify: convert each B_syms[k] → nested lists
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


if __name__ == "__main__":
    # Initialize parameters
    mu = 4.892e-9  # [km^3/s^2] — Bennu's GM (from OSIRIS-REx SPICE kernels)
    order = 3  # STT order

    # Initial state of particle (detaching from Bennu surface)
    x0_true = np.array(
        [
            0.3,
            0.0,
            0.0,  # position [km] (near surface)
            0.0,
            0.02,
            0.01,  # velocity [km/s]
        ]
    )

    # Epoch definition
    JD0 = Time("2025-04-24T00:00:00", scale="utc").jd
    JD0_seconds = (JD0 - Time("2000-01-01T12:00:00", scale="utc").jd) * 86400.0

    # Observation times (e.g., every 20 seconds for 6 hours)
    t_obs = JD0_seconds + np.linspace(0, 6 * 3600, int((6 * 3600) / 20))

    # Propagate true trajectory
    print(f"\nPropagating true trajectory:")
    sol_true, stts_true = propagate(x0_true, mu, order, t_obs, rtol=1e-10, atol=1e-12)
    x_true = sol_true.y[:6, :].T  # shape (n_steps, 6)

    # Measurement noise
    JD0 = Time("2025-04-24T00:00:00", scale="utc").jd
    sigma_ra = np.deg2rad(0.005)  # ~18 arcsec
    sigma_dec = np.deg2rad(0.005)  # same

    # Define fixed spacecraft position in ECI (e.g., 2 km above Bennu surface)
    sc_pos = np.array([0.0, 0.0, -2.0])  # [km], Bennu-centered inertial frame

    # Generate synthetic angular observations
    y_obs = generate_opnav_measurements(x_true, sc_pos, sigma_ra, sigma_dec)

    # Use all times (no visibility check here)
    t_obs_used = t_obs

    # Propagate reference trajectory
    print("\nPropagating reference trajectory:")

    # Define small deviation from truth (in km and km/s), and add it to the truth at t_obs_used[0]
    ref_dev = np.array([2, -3, 1, 0.1e-3, -0.5e-3, 0.8e-3])
    idx0 = np.searchsorted(t_obs, t_obs_used[0])
    x0_ref = sol_true.y[:6, idx0] - ref_dev

    # Propagate reference trajectory at used observation times
    sol_ref, stts_ref = propagate(
        x0=x0_ref,
        mu=mu,
        order=order,
        t_eval=t_obs_used,
        rtol=1e-12,
        atol=1e-14,
    )

    # Residual function for MCMC — OpNav angular case
    def residuals_normalized(delta_x0):
        # 1. Propagate the perturbed trajectory
        _, x_est = propagate_deviation(sol_ref, stts_ref, delta_x0, order=order)

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
    model.run(n_samples=1000, n_walkers=128, burn_in=500, thin=1)
    model.plot_convergence()
    model.plot_postfit_residuals()
    model.plot_postfit_residuals_time(t_obs_used=t_obs_used)
    model.plot_log_likelihood()
    model.plot_corner()
    model.summary()
    model.print_regression_diagnostics()


# EXTRA CODE:
#
# 1) Plot the elevation angles of the stations
# for name in stations:
#    elevations = np.arcsin(
#        np.sum((x_true[:, :3] - station_eci[name]) * station_eci[name], axis=1)
#        / (
#            np.linalg.norm(x_true[:, :3] - station_eci[name], axis=1)
#            * np.linalg.norm(station_eci[name], axis=1)
#        )
#    )
#    plt.plot(t_obs / 3600, np.degrees(elevations), label=name)
# plt.axhline(10, color="k", linestyle="--", label="10 deg limit")
# plt.xlabel("Time (hours)")
# plt.ylabel("Elevation (deg)")
# plt.legend()
# plt.grid()
# plt.show()
#
# 2) ...
