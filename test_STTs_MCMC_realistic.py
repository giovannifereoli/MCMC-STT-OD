import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import math
from itertools import product
from scipy.stats import norm
from MCMC import MCMCModel
from scipy.constants import pi
from scipy.spatial.transform import Rotation as R
from astropy.time import Time
import matplotlib.pyplot as plt

# TODO: fix sphere initialization. For 0 init params just do 1e-4 * randn. Detach from pror. Just use init_params
# TODO: put data paper 'Multiple-Shooting for IOD', emulate it, find more cases
# TODO: look autocorr time, fix steps (10*tau) and burn-in (3-5*tau)
# TODO: look acceptance fraction to asses initialization (i.e., should be 0.2-0.5)
# TODO: try PTSampler
# TODO: check correctness log prior
# TODO: make HMC work


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


def stt_ode(t, Y, mu, order, f_func, A_func, B_funcs):
    x = Y[:6]
    # 1) dynamics + STM
    dx = np.array(f_func(*x), float).reshape(6)
    offset = 6
    Phi = Y[offset : offset + 36].reshape(6, 6)
    offset += 36
    A = np.array(A_func(*x), float)
    dPhi = A @ Phi

    # 2) unpack every T_k once
    Ts = {1: Phi}
    for k in range(2, order + 1):
        size = 6 ** (k + 1)
        Tk = Y[offset : offset + size].reshape((6,) + (6,) * k)
        Ts[k] = Tk
        offset += size

    derivs = [*dx, *dPhi.flatten()]

    # 3) build each dT_k with all partitions
    for k in range(2, order + 1):
        # start with A·T_k
        dTk = np.tensordot(A, Ts[k], axes=(1, 0))

        # full‑order term: B_k(Φ,...,Φ)
        Bk = np.array(B_funcs[k](*x), float).reshape((6,) + (6,) * k)
        term = Bk
        for _ in range(k):
            term = np.tensordot(term, Phi, axes=(1, 0))
        dTk += term

        # mixed lower‑order terms
        # for every 2 ≤ m < k, we have comb(k,m) ways to choose
        # which slots get a T_{k-m+1} instead of Φ
        for m in range(2, k):
            coef = math.comb(k, m)
            Bm = np.array(B_funcs[m](*x), float).reshape((6,) + (6,) * m)

            # first contract one slot with T_{k-m+1}
            mixed = np.tensordot(Bm, Ts[k - m + 1], axes=(1, 0))
            # then contract the remaining m−1 slots with Φ
            for _ in range(m - 1):
                mixed = np.tensordot(mixed, Phi, axes=(1, 0))

            dTk += coef * mixed

        derivs += list(dTk.flatten())

    return np.array(derivs, float)


def propagate(x0, mu, order, t_eval, show_progress=True, **options):
    f_func, A_func, B_funcs = generate_stt_functions(mu, order)

    # initial augmented state
    Y0 = list(x0)
    Y0 += list(np.eye(6).flatten())
    for k in range(2, order + 1):
        Y0 += [0.0] * (6 ** (k + 1))
    Y0 = np.array(Y0, float)

    t_start = t_eval[0]
    t_end = t_eval[-1]

    last_print = -1

    def wrapped_rhs(t, y):
        nonlocal last_print
        if show_progress:
            progress = int(100 * (t - t_start) / (t_end - t_start))
            if progress > last_print:
                bar = "█" * (progress // 2) + "-" * (50 - progress // 2)
                print(f"\rProgress |{bar}| {progress:.1f}% - t = {t:.2f}", end="")
                last_print = progress
        return stt_ode(t, y, mu, order, f_func, A_func, B_funcs)

    sol = solve_ivp(
        fun=wrapped_rhs,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        y0=Y0,
        **options,
    )

    n_steps = sol.y.shape[1]
    stts = {}
    offset = 6

    # k = 1 is the STM (6×6)
    phi_flat = sol.y[offset : offset + 36, :]  # shape (36, n_steps)
    phi_all = phi_flat.reshape(6, 6, n_steps)  # shape (6,6,n_steps)
    stts[1] = np.transpose(phi_all, (2, 0, 1))  # shape (n_steps,6,6)
    offset += 36

    # higher orders k=2..order
    for k in range(2, order + 1):
        block_size = 6 ** (k + 1)
        Tk_flat = sol.y[offset : offset + block_size, :]  # (6^(k+1), n_steps)
        # reshape into (6,6,...,6,n_steps)
        shape = (6,) + (6,) * k + (n_steps,)
        Tk_all = Tk_flat.reshape(shape)
        # move time axis to front → shape (n_steps,6,6,...,6)
        stts[k] = np.moveaxis(Tk_all, -1, 0)
        offset += block_size

    # return both the raw solution and the unpacked STTs
    return sol, stts


def propagate_deviation(sol, stts, delta_x0, order):
    """
    Given:
      sol       : ODE solution from propagate(…)
      stts      : dict of STTs from propagate(…)  stts[k].shape = (n_steps,6,6,...,6)
      delta_x0  : initial deviation, shape (6,)
      order     : max STT order to include
    Returns:
      delta      : ndarray (n_steps,6) of propagated deviations
      x_nom+dev  : ndarray (n_steps,6) of perturbed trajectory
    """
    x_nom = sol.y[:6, :].T  # (n_steps,6)

    # Start with first order: Φ @ δx0
    delta = np.einsum("tij,j->ti", stts[1], delta_x0)

    if order >= 2:
        delta += 0.5 * np.einsum("tijk,j,k->ti", stts[2], delta_x0, delta_x0)

    if order >= 3:
        delta += (1 / 6) * np.einsum(
            "tijkl,j,k,l->ti", stts[3], delta_x0, delta_x0, delta_x0
        )

    if order >= 4:
        delta += (1 / 24) * np.einsum(
            "tijklm,j,k,l,m->ti", stts[4], delta_x0, delta_x0, delta_x0, delta_x0
        )

    return delta, x_nom + delta


def geodetic_to_ecef(lat, lon, alt):
    # WGS-84 ellipsoid constants
    a = 6378.137  # Equatorial radius [km]
    e2 = 6.69437999014e-3  # Square of eccentricity

    lat = np.radians(lat)
    lon = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_eci(ecef_pos, t_obs):
    omega_earth = 7.2921150e-5  # rad/s

    station_eci_t = []
    station_vel_eci_t = []

    for t in t_obs:
        theta = omega_earth * t  # rotation angle from JD0 (in radians)
        Rz = R.from_euler("z", theta).as_matrix()

        r_eci = Rz @ ecef_pos
        v_eci = np.cross([0, 0, omega_earth], r_eci)

        station_eci_t.append(r_eci)
        station_vel_eci_t.append(v_eci)

    return np.array(station_eci_t), np.array(station_vel_eci_t)


def is_visible(station_pos_eci, sc_pos_eci):
    rho = sc_pos_eci - station_pos_eci
    rho_norm = np.linalg.norm(rho, axis=1)
    station_norm = np.linalg.norm(station_pos_eci, axis=1)
    dot_product = np.sum(rho * station_pos_eci, axis=1)
    elevation = np.arcsin(dot_product / (rho_norm * station_norm))  # in radians
    return elevation > np.deg2rad(10)  # visibility if elevation > 10 deg


if __name__ == "__main__":
    from astropy.time import Time
    import numpy as np
    from scipy.stats import norm

    # Constants
    mu = 398600.4418  # Earth's gravitational parameter [km^3/s^2]
    x0_true = np.array([757.7, 5222.607, 4851.5, 2.21321, 4.67834, -5.3713])
    order = 3
    t_obs = np.linspace(0, 24 * 3600, int((24 * 3600) / 20))

    # Simulate true deviation
    sol_true, stts_true = propagate(x0_true, mu, order, t_obs, rtol=1e-10, atol=1e-12)
    x_true = sol_true.y[:6, :].T  # shape (n_steps, 6)

    # Measurement noise
    JD0 = Time("2025-04-24T00:00:00", scale="utc").jd
    sigma_range = 1e-3  # 1 meter
    sigma_rangerate = 1e-6  # 1 mm/s

    # DSN stations ECEF
    stations = {
        "Goldstone": geodetic_to_ecef(35.2472, -116.7933, 1.0),  # Goldstone
        "Canberra": geodetic_to_ecef(-35.3981, 148.9819, 1.0),  # Canberra
        "Madrid": geodetic_to_ecef(40.4314, -4.2486, 1.0),  # Madrid
    }

    # Convert all stations to ECI at each time
    station_eci = {}
    station_vel_eci = {}
    for name, pos in stations.items():
        r_eci, v_eci = ecef_to_eci(pos, t_obs)
        station_eci[name] = r_eci
        station_vel_eci[name] = v_eci

    # Build measurements
    t_obs_used = []
    y_obs = []
    station_eci_used = []
    station_vel_eci_used = []

    for i in range(len(t_obs)):
        visible_stations = []
        for name in stations:
            if is_visible(station_eci[name], x_true[:, :3])[i]:
                visible_stations.append(name)

        if visible_stations:
            # Choose first visible station (could improve by picking best elevation)
            station = visible_stations[0]

            sc_pos = x_true[i, :3]
            sc_vel = x_true[i, 3:]
            st_pos = station_eci[station][i]
            st_vel = station_vel_eci[station][i]

            los_vec = sc_pos - st_pos
            los_vel = sc_vel - st_vel
            range_meas = np.linalg.norm(los_vec) + np.random.normal(0, sigma_range)
            rangerate_meas = (
                np.dot(los_vec, los_vel) / np.linalg.norm(los_vec)
            ) + np.random.normal(0, sigma_rangerate)

            y_obs.append([range_meas, rangerate_meas])
            t_obs_used.append(t_obs[i])
            station_eci_used.append(st_pos)
            station_vel_eci_used.append(st_vel)

    y_obs = np.array(
        y_obs
    ).flatten()  # stack as [range1, rangerate1, range2, rangerate2, ...]
    t_obs_used = np.array(t_obs_used)
    station_eci_used = np.array(station_eci_used)
    station_vel_eci_used = np.array(station_vel_eci_used)
    print(f"\n")
    print(f"\nTotal measurements collected: {len(t_obs_used)}")

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

    # Recompute sol_ref and stts_ref for only the used times
    ref_dev = np.array(
        [2 * 1e-3, -3 * 1e-3, 1 * 1e-3, 0.1 * 1e-6, -0.5 * 1e-6, 0.8 * 1e-6]
    )  # km / km/s
    print(f"\n")
    sol_ref, stts_ref = propagate(
        x0_true - ref_dev, mu, order, t_obs_used, rtol=1e-12, atol=1e-14
    )

    # Residual function for MCMC
    def residuals(delta_x0):
        _, x_est = propagate_deviation(sol_ref, stts_ref, delta_x0, order=order)
        los_vec_model = x_est[:, :3] - station_eci_used
        range_model = np.linalg.norm(los_vec_model, axis=1)
        los_vel_model = x_est[:, 3:] - station_vel_eci_used
        rangerate_model = np.sum(los_vec_model * los_vel_model, axis=1) / range_model
        y_model = np.hstack([range_model, rangerate_model])
        weights = np.hstack(
            [
                np.full(len(t_obs_used), sigma_range),
                np.full(len(t_obs_used), sigma_rangerate),
            ]
        )
        return (y_obs - y_model) / weights

    # Priors
    initial_guess = np.zeros(6)
    priors = [
        (
            norm(loc=initial_guess[i], scale=1e10)
            if i < 3
            else norm(loc=initial_guess[i], scale=1e10)
        )
        for i in range(6)
    ]

    # Run MCMC
    model = MCMCModel(
        residuals_func=residuals,
        initial_params=initial_guess,
        param_priors=priors,
        observed_data=y_obs,
    )
    model.run(n_samples=5000, n_walkers=128, burn_in=1000)
    model.plot_convergence()
    model.plot_postfit_residuals()
    model.plot_log_likelihood()
    model.summary()
