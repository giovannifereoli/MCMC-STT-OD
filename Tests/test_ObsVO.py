"""
Two-Body Scale Ambiguity Demo with GM Sensitivity
=================================================

This script shows the exact scale symmetry

    r  -> lambda r
    v  -> lambda v
    GM -> lambda^3 GM

for the two-body problem when time is kept fixed.

The script also propagates the sensitivity with respect to GM:

    d r(t) / d GM
    d v(t) / d GM

This is useful for orbit determination because it shows how the GM column of
the design matrix can become strongly coupled with a radial/scale perturbation
of the trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Dynamics + extended variational equations
# ---------------------------------------------------------------------


def two_body_with_extended_stm(t, y):
    """
    Propagate two-body dynamics and the 7x7 STM.

    Extended state:
        z = [r(3), v(3), GM]

    Propagated vector:
        y = [r(3), v(3), GM, Phi(49)]

    Phi maps perturbations in [r0, v0, GM] to perturbations in [r(t), v(t), GM].
    """
    r = y[0:3]
    v = y[3:6]
    gm = y[6]
    phi = y[7:].reshape(7, 7)

    radius = np.linalg.norm(r)
    identity_3 = np.eye(3)

    acceleration = -gm * r / radius**3

    # da/dr: gravity-gradient tensor
    da_dr = -gm / radius**3 * (identity_3 - 3.0 * np.outer(r, r) / radius**2)

    # da/dGM
    da_dgm = -r / radius**3

    # Extended A matrix for [r, v, GM]
    a_matrix = np.zeros((7, 7))
    a_matrix[0:3, 3:6] = identity_3
    a_matrix[3:6, 0:3] = da_dr
    a_matrix[3:6, 6] = da_dgm

    # GM is constant: d(GM)/dt = 0
    # Therefore the last row is zero.

    phi_dot = a_matrix @ phi

    dydt = np.zeros_like(y)
    dydt[0:3] = v
    dydt[3:6] = acceleration
    dydt[6] = 0.0
    dydt[7:] = phi_dot.ravel()

    return dydt


def propagate(r0, v0, gm, t_span, n_points=600):
    """Propagate the state and the 7x7 STM."""
    phi0 = np.eye(7).ravel()
    y0 = np.concatenate((r0, v0, np.array([gm]), phi0))

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    return solve_ivp(
        two_body_with_extended_stm,
        t_span,
        y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def unit_bearing_to_origin(r_traj, camera_offset):
    """Unit line-of-sight vectors from the camera to the asteroid center."""
    camera_position = r_traj + camera_offset[:, None]
    los = -camera_position
    return los / np.linalg.norm(los, axis=0)


def angular_separation(u, v):
    """Angular separation between two unit-vector time histories."""
    dot_product = np.sum(u * v, axis=0)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))


def normalized_trajectory(r_traj):
    """Trajectory normalized by instantaneous radius."""
    return r_traj / np.linalg.norm(r_traj, axis=0)


def extract_stm(sol):
    """Return 7x7 STM history with shape (N, 7, 7)."""
    return np.moveaxis(sol.y[7:, :], 1, 0).reshape(-1, 7, 7)


def block_norms(phi):
    """Return Frobenius norms of the 3x3 STM blocks for [r, v]."""
    return {
        "rr": np.linalg.norm(phi[:, 0:3, 0:3], axis=(1, 2)),
        "rv": np.linalg.norm(phi[:, 0:3, 3:6], axis=(1, 2)),
        "vr": np.linalg.norm(phi[:, 3:6, 0:3], axis=(1, 2)),
        "vv": np.linalg.norm(phi[:, 3:6, 3:6], axis=(1, 2)),
    }


def relative_vector_error(a, b):
    """Relative norm error between two vector histories."""
    numerator = np.linalg.norm(a - b, axis=0)
    denominator = np.maximum(np.linalg.norm(a, axis=0), 1e-15)
    return numerator / denominator


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def setup_plot_style():
    plt.rcParams.update(
        {
            "figure.figsize": (11, 8),
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "grid.alpha": 0.35,
            "lines.linewidth": 2.0,
        }
    )


def finish_axis(ax):
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_geometry_figure(t, r_a, r_b, bearing_error_deg, radius_ratio, scale):
    r_a_hat = normalized_trajectory(r_a)
    r_b_hat = normalized_trajectory(r_b)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(r_a[0], r_a[1], label="Case A")
    ax.plot(r_b[0], r_b[1], "--", label=rf"Case B, $\lambda={scale:g}$")
    ax.scatter(0.0, 0.0, s=60, c="k", label="Asteroid")
    ax.set_title("Metric trajectories are different")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    finish_axis(ax)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(r_a_hat[0], r_a_hat[1], label="Case A normalized")
    ax.plot(r_b_hat[0], r_b_hat[1], "--", label="Case B normalized")
    ax.set_title("Angular shape is identical")
    ax.set_xlabel(r"$x / \|r\|$")
    ax.set_ylabel(r"$y / \|r\|$")
    ax.axis("equal")
    finish_axis(ax)
    ax.legend()

    ax = axes[1, 0]
    ax.semilogy(t, np.maximum(bearing_error_deg, 1e-16))
    ax.set_title("Bearing difference is numerical zero")
    ax.set_xlabel("time")
    ax.set_ylabel("angular difference [deg]")
    finish_axis(ax)

    ax = axes[1, 1]
    ax.plot(t, radius_ratio)
    ax.axhline(scale, linestyle="--", color="k", label=rf"$\lambda={scale:g}$")
    ax.set_title("Radius ratio remains constant")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\|r_B\| / \|r_A\|$")
    ax.set_ylim(scale * 0.995, scale * 1.005)
    finish_axis(ax)
    ax.legend()

    fig.suptitle(
        "VO-only scale ambiguity in two-body dynamics", fontsize=15, fontweight="bold"
    )
    return fig


def make_dynamics_figure(t, curvature_a, curvature_b, acceleration_a, acceleration_b):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(t, curvature_a, label=r"Case A: $GM/\|r\|^3$")
    ax.plot(t, curvature_b, "--", label=r"Case B: $GM/\|r\|^3$")
    ax.set_title(r"Invariant dynamical curvature $GM/r^3$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$GM/\|r\|^3$")
    finish_axis(ax)
    ax.legend()

    ax = axes[1]
    ax.plot(t, acceleration_a, label=r"Case A: $GM/\|r\|^2$")
    ax.plot(t, acceleration_b, "--", label=r"Case B: $GM/\|r\|^2$")
    ax.set_title("Metric acceleration is not invariant")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\|a\|$")
    finish_axis(ax)
    ax.legend()

    return fig


def make_stm_block_figure(t, phi_a, phi_b):
    norms_a = block_norms(phi_a)
    norms_b = block_norms(phi_b)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    entries = [
        ("rr", r"$\Phi_{rr}$", axes[0, 0]),
        ("rv", r"$\Phi_{rv}$", axes[0, 1]),
        ("vr", r"$\Phi_{vr}$", axes[1, 0]),
        ("vv", r"$\Phi_{vv}$", axes[1, 1]),
    ]

    for key, label, ax in entries:
        ax.plot(t, norms_a[key], label=rf"Case A: $\|{label}\|_F$")
        ax.plot(t, norms_b[key], "--", label=rf"Case B: $\|{label}\|_F$")
        ax.set_title(f"STM block {label}")
        ax.set_xlabel("time")
        ax.set_ylabel("Frobenius norm")
        finish_axis(ax)
        ax.legend()

    fig.suptitle("State STM blocks", fontsize=15, fontweight="bold")
    return fig


def make_gm_sensitivity_figure(t, phi_a, phi_b, gm_a, gm_b, scale):
    """
    Plot propagated GM sensitivities.

    In the 7x7 STM, the GM sensitivity column is:

        Phi[:, 6] = d [r, v, GM](t) / d GM_0

    Important scaling:
        GM_B = lambda^3 GM_A

    A unit perturbation in GM_B is not equivalent to a unit perturbation in GM_A.
    To compare the same fractional GM perturbation, multiply the B sensitivity by:

        dGM_B/dGM_A = lambda^3

    So:
        d r_B / d GM_A = lambda^3 d r_B / d GM_B

    The expected scale relation is:
        d r_B / d GM_A = lambda d r_A / d GM_A
        d v_B / d GM_A = lambda d v_A / d GM_A

    Therefore:
        lambda^3 Phi_B[:, GM_B] should match lambda Phi_A[:, GM_A].
    """
    dr_dgm_a = phi_a[:, 0:3, 6].T
    dv_dgm_a = phi_a[:, 3:6, 6].T

    dr_dgm_b = phi_b[:, 0:3, 6].T
    dv_dgm_b = phi_b[:, 3:6, 6].T

    # Convert B sensitivity to derivative with respect to the base GM_A.
    dr_dgm_b_equiv = scale**3 * dr_dgm_b
    dv_dgm_b_equiv = scale**3 * dv_dgm_b

    # Expected scaled A sensitivities.
    dr_dgm_a_scaled = scale * dr_dgm_a
    dv_dgm_a_scaled = scale * dv_dgm_a

    dr_error = relative_vector_error(dr_dgm_a_scaled, dr_dgm_b_equiv)
    dv_error = relative_vector_error(dv_dgm_a_scaled, dv_dgm_b_equiv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        t,
        np.linalg.norm(dr_dgm_a, axis=0),
        label=r"Case A: $\|\partial r/\partial GM_A\|$",
    )
    ax.plot(
        t,
        np.linalg.norm(dr_dgm_b, axis=0),
        "--",
        label=r"Case B: $\|\partial r/\partial GM_B\|$",
    )
    ax.set_title("Raw position sensitivity to GM")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\|\partial r/\partial GM\|$")
    finish_axis(ax)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(
        t,
        np.linalg.norm(dv_dgm_a, axis=0),
        label=r"Case A: $\|\partial v/\partial GM_A\|$",
    )
    ax.plot(
        t,
        np.linalg.norm(dv_dgm_b, axis=0),
        "--",
        label=r"Case B: $\|\partial v/\partial GM_B\|$",
    )
    ax.set_title("Raw velocity sensitivity to GM")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\|\partial v/\partial GM\|$")
    finish_axis(ax)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        t,
        np.linalg.norm(dr_dgm_a_scaled, axis=0),
        label=r"$\lambda \|\partial r_A/\partial GM_A\|$",
    )
    ax.plot(
        t,
        np.linalg.norm(dr_dgm_b_equiv, axis=0),
        "--",
        label=r"$\lambda^3 \|\partial r_B/\partial GM_B\|$",
    )
    ax.set_title("Position GM sensitivity after scale normalization")
    ax.set_xlabel("time")
    ax.set_ylabel("normalized sensitivity")
    finish_axis(ax)
    ax.legend()

    ax = axes[1, 1]
    ax.semilogy(t, np.maximum(dr_error, 1e-16), label=r"$r$ sensitivity error")
    ax.semilogy(t, np.maximum(dv_error, 1e-16), "--", label=r"$v$ sensitivity error")
    ax.set_title("Relative error after scale normalization")
    ax.set_xlabel("time")
    ax.set_ylabel("relative error")
    finish_axis(ax)
    ax.legend()

    fig.suptitle(
        r"Propagated GM sensitivity: $\partial r/\partial GM$, $\partial v/\partial GM$",
        fontsize=15,
        fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------


def main():
    setup_plot_style()

    gm_a = 1.0
    r0_a = np.array([-5.0, 1.0, 0.0])
    v0_a = np.array([0.5, 0.0, 0.0])

    scale = 2.0
    gm_b = scale**3 * gm_a
    r0_b = scale * r0_a
    v0_b = scale * v0_a

    t_span = (0.0, 30.0)

    sol_a = propagate(r0_a, v0_a, gm_a, t_span)
    sol_b = propagate(r0_b, v0_b, gm_b, t_span)

    if not sol_a.success or not sol_b.success:
        raise RuntimeError("Numerical propagation failed.")

    t = sol_a.t

    r_a = sol_a.y[0:3, :]
    r_b = sol_b.y[0:3, :]

    phi_a = extract_stm(sol_a)
    phi_b = extract_stm(sol_b)

    camera_offset_a = np.array([0.1, 0.05, 0.0])
    camera_offset_b = scale * camera_offset_a

    bearing_a = unit_bearing_to_origin(r_a, camera_offset_a)
    bearing_b = unit_bearing_to_origin(r_b, camera_offset_b)
    bearing_error_deg = np.rad2deg(angular_separation(bearing_a, bearing_b))

    radius_a = np.linalg.norm(r_a, axis=0)
    radius_b = np.linalg.norm(r_b, axis=0)

    radius_ratio = radius_b / radius_a

    curvature_a = gm_a / radius_a**3
    curvature_b = gm_b / radius_b**3

    acceleration_a = gm_a / radius_a**2
    acceleration_b = gm_b / radius_b**2

    make_geometry_figure(t, r_a, r_b, bearing_error_deg, radius_ratio, scale)
    make_dynamics_figure(t, curvature_a, curvature_b, acceleration_a, acceleration_b)
    make_stm_block_figure(t, phi_a, phi_b)
    make_gm_sensitivity_figure(t, phi_a, phi_b, gm_a, gm_b, scale)

    plt.show()


if __name__ == "__main__":
    main()
