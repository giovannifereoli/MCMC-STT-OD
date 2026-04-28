"""
Bearing-Only Batch Fit: Same Postfits, Different Scale
======================================================

This script performs an actual batch least-squares fit using bearing-only
measurements.

Important distinction:
    - The STM is used only to build the linearized design matrix H.
    - Postfit residuals are recomputed with a fresh nonlinear propagation
      of the updated state.

That is the correct OD workflow:

    1. propagate reference trajectory + STM
    2. build H from STM
    3. solve normal equations for delta p
    4. update p
    5. recompute postfit residuals with nonlinear propagation

The point:
    Starting from different metric scales, the filter reaches solutions with
    essentially identical nonlinear postfit residuals, but different absolute
    scale and different GM.

This demonstrates the bearing-only / VO-only scale ambiguity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Dynamics + STM
# ---------------------------------------------------------------------


def two_body_with_stm(t, y):
    """
    Propagate state and 7x7 STM for z = [r, v, GM].

    y = [r(3), v(3), GM, Phi(49)]
    """
    r = y[0:3]
    v = y[3:6]
    gm = y[6]
    phi = y[7:].reshape(7, 7)

    radius = np.linalg.norm(r)
    i3 = np.eye(3)

    acc = -gm * r / radius**3

    da_dr = -gm / radius**3 * (i3 - 3.0 * np.outer(r, r) / radius**2)
    da_dgm = -r / radius**3

    a_mat = np.zeros((7, 7))
    a_mat[0:3, 3:6] = i3
    a_mat[3:6, 0:3] = da_dr
    a_mat[3:6, 6] = da_dgm

    ydot = np.zeros_like(y)
    ydot[0:3] = v
    ydot[3:6] = acc
    ydot[6] = 0.0
    ydot[7:] = (a_mat @ phi).ravel()

    return ydot


def propagate_with_stm(p, t_eval):
    """Propagate [r0, v0, GM] and its 7x7 STM."""
    phi0 = np.eye(7).ravel()
    y0 = np.concatenate((p[0:3], p[3:6], [p[6]], phi0))

    sol = solve_ivp(
        two_body_with_stm,
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )

    if not sol.success:
        raise RuntimeError("Propagation with STM failed.")

    r = sol.y[0:3, :]
    v = sol.y[3:6, :]
    phi = np.moveaxis(sol.y[7:, :], 1, 0).reshape(-1, 7, 7)

    return r, v, phi


def two_body(t, y, gm):
    """State-only two-body dynamics."""
    r = y[0:3]
    v = y[3:6]
    radius = np.linalg.norm(r)
    acc = -gm * r / radius**3

    dydt = np.zeros(6)
    dydt[0:3] = v
    dydt[3:6] = acc
    return dydt


def propagate_state_only(p, t_eval):
    """
    Fresh nonlinear propagation of [r0, v0, GM].

    This is what is used for postfit residuals.
    """
    y0 = np.concatenate((p[0:3], p[3:6]))
    gm = p[6]

    sol = solve_ivp(
        two_body,
        (t_eval[0], t_eval[-1]),
        y0,
        args=(gm,),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )

    if not sol.success:
        raise RuntimeError("State-only propagation failed.")

    return sol.y[0:3, :], sol.y[3:6, :]


# ---------------------------------------------------------------------
# Bearing measurement model
# ---------------------------------------------------------------------


def bearing(r):
    """Bearing from spacecraft to asteroid center."""
    return -r / np.linalg.norm(r, axis=0)


def bearing_single(r):
    return -r / np.linalg.norm(r)


def bearing_jacobian_wrt_position(r):
    """
    b = -r / ||r||

    db/dr = -(I - uu^T) / ||r||
    """
    radius = np.linalg.norm(r)
    u = r / radius
    return -(np.eye(3) - np.outer(u, u)) / radius


def nonlinear_weighted_residual(p, t_eval, y_obs, sigma):
    """
    Nonlinear postfit residual.

    No STM is used here.
    """
    r, _ = propagate_state_only(p, t_eval)
    y_calc = bearing(r)
    return ((y_obs - y_calc) / sigma).ravel(order="F")


def build_linearized_system(p, t_eval, y_obs, sigma):
    """
    Build weighted residual and design matrix using STM.

    residual = y_obs - y_calc
    H = d y_calc / d p

    The linearized model is:
        residual_new_predicted = residual - H dp

    Equivalently:
        residual ~= H dp
    """
    r, _, phi = propagate_with_stm(p, t_eval)

    residual_blocks = []
    h_blocks = []

    for k in range(len(t_eval)):
        y_calc_k = bearing_single(r[:, k])
        residual_k = y_obs[:, k] - y_calc_k

        db_dr = bearing_jacobian_wrt_position(r[:, k])
        dr_dp = phi[k, 0:3, :]

        h_k = db_dr @ dr_dp

        residual_blocks.append(residual_k / sigma)
        h_blocks.append(h_k / sigma)

    residual = np.concatenate(residual_blocks)
    h = np.vstack(h_blocks)

    return residual, h


def batch_fit(p0, t_eval, y_obs, sigma, max_iter=12, tol=1e-11, damping=1e-8):
    """
    Batch Gauss-Newton fit.

    STM usage:
        At each iteration, STM builds H and the linearized residual.

    Postfit usage:
        After the update, residuals are recomputed by nonlinear propagation.
    """
    p = p0.copy().astype(float)
    history = []

    for iteration in range(max_iter):
        residual_prefit, h = build_linearized_system(p, t_eval, y_obs, sigma)

        normal = h.T @ h
        rhs = h.T @ residual_prefit

        dp = np.linalg.solve(normal + damping * np.eye(7), rhs)

        # Linearized predicted residual after update
        residual_predicted_postfit = residual_prefit - h @ dp

        p_new = p + dp

        if p_new[6] <= 0:
            p_new[6] = p[6]

        # True nonlinear postfit residual after update
        residual_nonlinear_postfit = nonlinear_weighted_residual(
            p_new, t_eval, y_obs, sigma
        )

        history.append(
            {
                "iteration": iteration,
                "p_before": p.copy(),
                "dp": dp.copy(),
                "p_after": p_new.copy(),
                "prefit_rms": np.sqrt(np.mean(residual_prefit**2)),
                "predicted_postfit_rms": np.sqrt(
                    np.mean(residual_predicted_postfit**2)
                ),
                "nonlinear_postfit_rms": np.sqrt(
                    np.mean(residual_nonlinear_postfit**2)
                ),
                "step_norm": np.linalg.norm(dp),
            }
        )

        p = p_new

        if np.linalg.norm(dp) < tol * max(np.linalg.norm(p), 1.0):
            break

    residual_final = nonlinear_weighted_residual(p, t_eval, y_obs, sigma)
    _, h_final = build_linearized_system(p, t_eval, y_obs, sigma)

    return p, residual_final, h_final, history


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def scaled_parameters(p, lam):
    out = p.copy()
    out[0:3] *= lam
    out[3:6] *= lam
    out[6] *= lam**3
    return out


def best_scale_against_truth(p_fit, p_true):
    lam_r = np.linalg.norm(p_fit[0:3]) / np.linalg.norm(p_true[0:3])
    lam_v = np.linalg.norm(p_fit[3:6]) / np.linalg.norm(p_true[3:6])
    lam_gm = (p_fit[6] / p_true[6]) ** (1.0 / 3.0)
    return lam_r, lam_v, lam_gm


def scale_direction(p):
    q = np.zeros(7)
    q[0:3] = p[0:3]
    q[3:6] = p[3:6]
    q[6] = 3.0 * p[6]
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def setup_plot_style():
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
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


def make_fit_summary_plot(results, p_true):
    labels = [rf"$\lambda_0={res['lambda0']}$" for res in results]
    x = np.arange(len(results))

    final_rms = np.array([res["rms"] for res in results])
    final_cost = np.array([res["cost"] for res in results])
    gm_values = np.array([res["p_fit"][6] for res in results])
    lam_r = np.array([res["lambda_r"] for res in results])
    lam_v = np.array([res["lambda_v"] for res in results])
    lam_gm = np.array([res["lambda_gm"] for res in results])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x, final_rms)
    ax.set_title("Same nonlinear postfit RMS")
    ax.set_ylabel("weighted RMS residual")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    finish_axis(ax)

    ax = axes[0, 1]
    ax.bar(x, gm_values)
    ax.axhline(p_true[6], color="k", linestyle="--", label="true GM")
    ax.set_title("Different fitted GM")
    ax.set_ylabel("fitted GM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    finish_axis(ax)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, lam_r, "o-", label=r"from $\|r_0\|$")
    ax.plot(x, lam_v, "s--", label=r"from $\|v_0\|$")
    ax.plot(x, lam_gm, "d-.", label=r"from $GM^{1/3}$")
    ax.axhline(1.0, color="k", linestyle=":", label="truth scale")
    ax.set_title("Solutions stay at different scales")
    ax.set_ylabel("scale relative to truth")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    finish_axis(ax)
    ax.legend()

    ax = axes[1, 1]
    ax.semilogy(x, np.maximum(final_cost, 1e-16), "o-")
    ax.set_title("Same nonlinear least-squares cost")
    ax.set_ylabel(r"$J=\frac{1}{2}e^T e$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    finish_axis(ax)

    fig.suptitle(
        "Bearing-only batch fit: same nonlinear postfits, different scale/GM",
        fontsize=15,
        fontweight="bold",
    )
    return fig


def make_iteration_plot(results):
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

    for res in results:
        hist = res["history"]
        it = [h["iteration"] for h in hist]
        pre = [h["prefit_rms"] for h in hist]
        pred = [h["predicted_postfit_rms"] for h in hist]
        nonlin = [h["nonlinear_postfit_rms"] for h in hist]

        ax.semilogy(it, pre, "o-", label=rf"prefit, $\lambda_0={res['lambda0']}$")
        ax.semilogy(
            it, nonlin, "s--", label=rf"nonlinear postfit, $\lambda_0={res['lambda0']}$"
        )

    ax.set_title("STM builds update, nonlinear propagation evaluates postfits")
    ax.set_xlabel("iteration")
    ax.set_ylabel("weighted RMS residual")
    finish_axis(ax)
    ax.legend(ncol=2)
    return fig


def make_postfit_residual_plot(t_eval, results):
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

    for res in results:
        residual = res["residual"]
        residual_3n = residual.reshape(len(t_eval), 3).T
        norm_residual = np.linalg.norm(residual_3n, axis=0)
        ax.plot(t_eval, norm_residual, label=rf"$\lambda_0={res['lambda0']}$")

    ax.set_title("Nonlinear postfit residual norm time history")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\|y_{obs}-y_{calc}\|/\sigma$")
    finish_axis(ax)
    ax.legend()
    return fig


def make_orbit_comparison_plot(t_eval, p_true, results):
    r_true, _ = propagate_state_only(p_true, t_eval)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    ax.plot(r_true[0], r_true[1], "k", linewidth=3, label="truth")
    for res in results:
        r_fit, _ = propagate_state_only(res["p_fit"], t_eval)
        ax.plot(r_fit[0], r_fit[1], "--", label=rf"fit $\lambda_0={res['lambda0']}$")
    ax.scatter(0.0, 0.0, c="k", s=60)
    ax.set_title("Metric fitted orbits are different")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    finish_axis(ax)
    ax.legend()

    ax = axes[1]
    rhat_true = r_true / np.linalg.norm(r_true, axis=0)
    ax.plot(rhat_true[0], rhat_true[1], "k", linewidth=3, label="truth normalized")
    for res in results:
        r_fit, _ = propagate_state_only(res["p_fit"], t_eval)
        rhat_fit = r_fit / np.linalg.norm(r_fit, axis=0)
        ax.plot(
            rhat_fit[0], rhat_fit[1], "--", label=rf"fit $\lambda_0={res['lambda0']}$"
        )
    ax.set_title("Normalized fitted orbits overlap")
    ax.set_xlabel(r"$x/\|r\|$")
    ax.set_ylabel(r"$y/\|r\|$")
    ax.axis("equal")
    finish_axis(ax)
    ax.legend()

    return fig


def make_cost_along_scale_family_plot(t_eval, y_obs, sigma, p_fit):
    lambdas = np.linspace(0.4, 2.2, 80)
    costs = []

    for lam in lambdas:
        p_lam = scaled_parameters(p_fit, lam)
        residual = nonlinear_weighted_residual(p_lam, t_eval, y_obs, sigma)
        costs.append(0.5 * residual @ residual)

    costs = np.array(costs)

    _, h = build_linearized_system(p_fit, t_eval, y_obs, sigma)
    q = scale_direction(p_fit)
    hq_norm = np.linalg.norm(h @ q)

    s = np.linalg.svd(h, compute_uv=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(lambdas, costs)
    ax.set_title("Nonlinear cost is flat along scale family")
    ax.set_xlabel(r"scale applied to fitted solution")
    ax.set_ylabel(r"$J=\frac{1}{2}e^T e$")
    finish_axis(ax)

    ax = axes[1]
    ax.semilogy(np.arange(1, len(s) + 1), s, marker="o")
    ax.set_title(
        rf"STM design matrix is rank-deficient: $\|Hq_{{scale}}\|={hq_norm:.2e}$"
    )
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    finish_axis(ax)

    return fig


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    setup_plot_style()
    rng = np.random.default_rng(7)

    p_true = np.array(
        [
            -5.0,
            1.0,
            0.0,
            0.5,
            0.0,
            0.0,
            1.0,
        ]
    )

    t_eval = np.linspace(0.0, 30.0, 80)

    r_true, _ = propagate_state_only(p_true, t_eval)
    y_clean = bearing(r_true)

    sigma = 1.0e-6
    y_obs = y_clean + sigma * rng.normal(size=y_clean.shape)
    y_obs /= np.linalg.norm(y_obs, axis=0)

    base_guess = p_true.copy()
    base_guess[0:3] += np.array([0.03, -0.02, 0.015])
    base_guess[3:6] += np.array([0.004, -0.002, 0.001])
    base_guess[6] *= 1.01

    initial_scales = [0.9, 1.0, 1.8]
    results = []

    print("\nTruth:")
    print(p_true)

    for lam0 in initial_scales:
        p0 = scaled_parameters(base_guess, lam0)

        p_fit, residual, h, history = batch_fit(
            p0,
            t_eval,
            y_obs,
            sigma,
            max_iter=12,
            damping=1e-8,
        )

        cost = 0.5 * residual @ residual
        rms = np.sqrt(np.mean(residual**2))
        lam_r, lam_v, lam_gm = best_scale_against_truth(p_fit, p_true)

        results.append(
            {
                "lambda0": lam0,
                "p0": p0,
                "p_fit": p_fit,
                "residual": residual,
                "h": h,
                "history": history,
                "cost": cost,
                "rms": rms,
                "lambda_r": lam_r,
                "lambda_v": lam_v,
                "lambda_gm": lam_gm,
            }
        )

        print("\nInitial scale:", lam0)
        print("Fitted p:")
        print(p_fit)
        print("Nonlinear postfit weighted RMS:", rms)
        print("Nonlinear cost:", cost)
        print("Scale from r, v, GM^(1/3):", lam_r, lam_v, lam_gm)
        print("Final iteration diagnostic:")
        print(history[-1])

    make_fit_summary_plot(results, p_true)
    make_iteration_plot(results)
    make_postfit_residual_plot(t_eval, results)
    make_orbit_comparison_plot(t_eval, p_true, results)
    make_cost_along_scale_family_plot(t_eval, y_obs, sigma, results[1]["p_fit"])

    plt.show()


if __name__ == "__main__":
    main()
