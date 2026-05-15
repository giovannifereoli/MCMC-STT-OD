"""
OD Solution Closeness Measure
==============================
Implements the statistical distance measure between two Orbit Determination (OD)
solutions as described in the AIAA 2004-4982 appendix.

Given two OD solutions S1 = (c1, P1) and S2 = (c2, P2), computes the smallest
t such that the t-sigma ellipsoids around each solution first become tangent.
This tangency value d(S1, S2) is the closeness measure.
"""

import numpy as np
from numpy.polynomial import polynomial as P
import warnings


def _build_polynomial_coefficients(
    dc: np.ndarray, P1: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    """
    Build coefficients of the polynomial in alpha derived from equation (3):

        0 = dc^T (P2 + alpha*P1)^C (alpha^2 * P1 - P2) (P2 + alpha*P1)^C dc

    where M^C = det(M) * M^{-1} is the matrix of cofactors.

    For numerical stability we work with the equivalent form using actual inverses:

        0 = dc^T (P2 + alpha*P1)^{-1} (alpha^2 * P1 - P2) (P2 + alpha*P1)^{-1} dc

    We evaluate this as a scalar function of alpha and find its roots numerically.

    Parameters
    ----------
    dc  : (n,) difference vector c2 - c1
    P1  : (n, n) covariance of solution 1
    P2  : (n, n) covariance of solution 2

    Returns
    -------
    f   : callable, the scalar function f(alpha) whose roots we seek
    """

    def f(alpha: float) -> float:
        M = P2 + alpha * P1
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.nan
        inner = alpha**2 * P1 - P2
        val = dc @ Minv @ inner @ Minv @ dc
        return val

    return f


def _tangency_value(
    alpha: float, dc: np.ndarray, P1: np.ndarray, P2: np.ndarray
) -> float:
    """
    Compute the squared closeness value for a given alpha:

        t^2 = dc^T (P2 + alpha*P1)^{-1} P2 (P2 + alpha*P1)^{-1} dc

    This is the value of (x - c2)^T Lambda2 (x - c2) at the tangency point,
    which equals t^2 by construction (equation 1 in the paper).

    Parameters
    ----------
    alpha : float, positive root of the polynomial equation
    dc    : (n,) difference vector c2 - c1
    P1    : (n, n) covariance of solution 1
    P2    : (n, n) covariance of solution 2

    Returns
    -------
    t_squared : float
    """
    M = P2 + alpha * P1
    Minv = np.linalg.inv(M)
    return float(dc @ Minv @ P2 @ Minv @ dc)


def od_closeness(
    c1: np.ndarray,
    P1: np.ndarray,
    c2: np.ndarray,
    P2: np.ndarray,
    alpha_search_range: tuple = (1e-6, 1e6),
    n_grid: int = 10000,
    tol: float = 1e-10,
) -> dict:
    """
    Compute the closeness measure d(S1, S2) between two OD solutions.

    The closeness measure is defined as the smallest t > 0 such that the
    t-sigma uncertainty ellipsoids around c1 (with covariance P1) and c2
    (with covariance P2) are tangent.

    Algorithm
    ---------
    1. Form the scalar function f(alpha) = dc^T (P2+alpha*P1)^{-1}
       (alpha^2 P1 - P2) (P2+alpha*P1)^{-1} dc.
    2. Find positive real roots of f(alpha) = 0 via sign-change detection
       on a grid followed by bisection refinement.
    3. For each root alpha_i, evaluate the tangency sigma-value t_i.
    4. Return d = min_i(t_i).

    Parameters
    ----------
    c1 : array_like, shape (n,)
        Mean state vector of solution 1.
    P1 : array_like, shape (n, n)
        Covariance matrix of solution 1 (symmetric positive definite).
    c2 : array_like, shape (n,)
        Mean state vector of solution 2.
    P2 : array_like, shape (n, n)
        Covariance matrix of solution 2 (symmetric positive definite).
    alpha_search_range : (float, float)
        (min, max) range for searching alpha roots. Default (1e-6, 1e6).
    n_grid : int
        Number of grid points for initial root bracket search. Default 10000.
    tol : float
        Bisection convergence tolerance for alpha roots. Default 1e-10.

    Returns
    -------
    result : dict with keys:
        'd'       : float, the closeness measure (number of sigma)
        't_sq'    : float, d^2
        'alpha'   : float, the alpha value achieving the minimum
        'x_tang'  : ndarray shape (n,), the tangency point in state space
        'all_roots'   : list of all positive alpha roots found
        'all_t_sq'    : list of t^2 values for each root
    """
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)

    assert c1.shape == c2.shape, "c1 and c2 must have the same shape"
    assert (
        P1.shape == P2.shape == (len(c1), len(c1))
    ), "Covariance shapes must match state dimension"

    dc = c2 - c1

    # Degenerate case: solutions are identical
    if np.allclose(dc, 0):
        return {
            "d": 0.0,
            "t_sq": 0.0,
            "alpha": None,
            "x_tang": c1.copy(),
            "all_roots": [],
            "all_t_sq": [],
        }

    f = _build_polynomial_coefficients(dc, P1, P2)

    # --- Grid search for sign changes (bracket root locations) ---
    alpha_min, alpha_max = alpha_search_range
    alphas = np.geomspace(alpha_min, alpha_max, n_grid)
    fvals = np.array([f(a) for a in alphas])

    roots = []
    for i in range(len(fvals) - 1):
        if np.isnan(fvals[i]) or np.isnan(fvals[i + 1]):
            continue
        if fvals[i] * fvals[i + 1] < 0:
            # Bisection to refine the root
            lo, hi = alphas[i], alphas[i + 1]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fmid = f(mid)
                if abs(fmid) < tol or (hi - lo) < tol * mid:
                    break
                if f(lo) * fmid < 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))

    if not roots:
        warnings.warn(
            "No positive real roots found for alpha. The ellipsoids may never become tangent "
            "in the searched range, or the solutions may be nested. "
            "Try expanding alpha_search_range."
        )
        return {
            "d": np.nan,
            "t_sq": np.nan,
            "alpha": None,
            "x_tang": None,
            "all_roots": [],
            "all_t_sq": [],
        }

    # --- Evaluate t^2 at each root and pick the minimum ---
    t_sq_values = [_tangency_value(a, dc, P1, P2) for a in roots]
    best_idx = int(np.argmin(t_sq_values))
    best_alpha = roots[best_idx]
    best_t_sq = t_sq_values[best_idx]

    # --- Compute tangency point x* ---
    Lambda1 = np.linalg.inv(P1)
    Lambda2 = np.linalg.inv(P2)
    M = Lambda1 + best_alpha * Lambda2
    x_tang = np.linalg.solve(M, Lambda1 @ c1 + best_alpha * Lambda2 @ c2)

    return {
        "d": float(np.sqrt(max(best_t_sq, 0.0))),
        "t_sq": float(best_t_sq),
        "alpha": float(best_alpha),
        "x_tang": x_tang,
        "all_roots": roots,
        "all_t_sq": t_sq_values,
    }


# ---------------------------------------------------------------------------
# Convenience: project to B-plane and compute 2-D closeness
# ---------------------------------------------------------------------------


def od_closeness_2d(
    c1_2d: np.ndarray,
    P1_2d: np.ndarray,
    c2_2d: np.ndarray,
    P2_2d: np.ndarray,
    **kwargs,
) -> dict:
    """
    Thin wrapper around od_closeness for 2-D B-plane cross sections.
    Inputs are 2-vectors and 2x2 covariance matrices.
    """
    return od_closeness(c1_2d, P1_2d, c2_2d, P2_2d, **kwargs)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    rng = np.random.default_rng(42)

    def make_ellipse_patch(mean, cov, n_sigma, **kw):
        """Return a matplotlib Ellipse patch for a 2-D Gaussian."""
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * n_sigma * np.sqrt(vals)
        return Ellipse(xy=mean, width=w, height=h, angle=angle, **kw)

    # --- Example 1: simple 2-D case ---
    print("=" * 60)
    print("Example 1: 2-D solutions")
    print("=" * 60)

    c1 = np.array([0.0, 0.0])
    P1 = np.array([[4.0, 1.0], [1.0, 1.0]])

    c2 = np.array([3.0, 1.5])
    P2 = np.array([[1.0, -0.3], [-0.3, 2.0]])

    result = od_closeness(c1, P1, c2, P2)
    print(f"  Closeness d(S1,S2)  = {result['d']:.6f} sigma")
    print(f"  Best alpha          = {result['alpha']:.6f}")
    print(f"  Tangency point x*   = {result['x_tang']}")
    print(f"  All alpha roots     = {[f'{a:.4f}' for a in result['all_roots']]}")
    print(f"  All t^2 values      = {[f'{t:.4f}' for t in result['all_t_sq']]}")

    # Verify: tangency point should lie on the d-sigma ellipsoid of both solutions
    x = result["x_tang"]
    d = result["d"]
    L1 = np.linalg.inv(P1)
    L2 = np.linalg.inv(P2)
    maha1 = np.sqrt((x - c1) @ L1 @ (x - c1))
    maha2 = np.sqrt((x - c2) @ L2 @ (x - c2))
    print(
        f"\n  Verification (should both equal d={d:.4f}, tangency point should lie on the d-sigma ellipsoid of both solutions):"
    )
    print(f"    Mahalanobis dist from S1: {maha1:.6f}")
    print(f"    Mahalanobis dist from S2: {maha2:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    d_val = result["d"]

    for n_sig in [1.0, d_val]:
        e1 = make_ellipse_patch(
            c1,
            P1,
            n_sig,
            fill=False,
            edgecolor="royalblue",
            linestyle="--" if n_sig == 1.0 else "-",
            linewidth=1.5 if n_sig == 1.0 else 2.5,
            label=f"S1 {n_sig:.2f}-σ" if n_sig != 1.0 else "S1 1-σ",
        )
        e2 = make_ellipse_patch(
            c2,
            P2,
            n_sig,
            fill=False,
            edgecolor="tomato",
            linestyle="--" if n_sig == 1.0 else "-",
            linewidth=1.5 if n_sig == 1.0 else 2.5,
            label=f"S2 {n_sig:.2f}-σ" if n_sig != 1.0 else "S2 1-σ",
        )
        ax.add_patch(e1)
        ax.add_patch(e2)

    ax.plot(*c1, "o", color="royalblue", ms=8, label="c1")
    ax.plot(*c2, "o", color="tomato", ms=8, label="c2")
    if result["x_tang"] is not None:
        ax.plot(
            *result["x_tang"],
            "*",
            color="gold",
            ms=14,
            markeredgecolor="k",
            zorder=5,
            label=f"Tangency point (d={d_val:.3f}σ)",
        )

    ax.set_xlim(-6, 7)
    ax.set_ylim(-4, 6)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.set_title(f"OD Closeness Measure  d(S1,S2) = {d_val:.4f} σ")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Example 2: 6-D state space ---
    print("\n" + "=" * 60)
    print("Example 2: 6-D state space (position + velocity)")
    print("=" * 60)

    n = 6
    A1 = rng.standard_normal((n, n))
    P1_6d = A1 @ A1.T / n + np.eye(n) * 0.5
    A2 = rng.standard_normal((n, n))
    P2_6d = A2 @ A2.T / n + np.eye(n) * 0.5

    c1_6d = np.zeros(n)
    c2_6d = rng.standard_normal(n) * 1.5

    result_6d = od_closeness(c1_6d, P1_6d, c2_6d, P2_6d)
    print(f"  Closeness d(S1,S2)  = {result_6d['d']:.6f} sigma")
    print(f"  Best alpha          = {result_6d['alpha']:.6f}")

    x = result_6d["x_tang"]
    L1 = np.linalg.inv(P1_6d)
    L2 = np.linalg.inv(P2_6d)
    maha1 = np.sqrt((x - c1_6d) @ L1 @ (x - c1_6d))
    maha2 = np.sqrt((x - c2_6d) @ L2 @ (x - c2_6d))
    print(
        f"\n  Verification (should both equal d={result_6d['d']:.4f}, tangency point should lie on the d-sigma ellipsoid of both solutions):"
    )
    print(f"    Mahalanobis dist from S1: {maha1:.6f}")
    print(f"    Mahalanobis dist from S2: {maha2:.6f}")


# ---------------------------------------------------------------------------
# Multi-solution consistency matrix
# ---------------------------------------------------------------------------


def compute_consistency_matrix(solutions):
    """
    Compute the N×N pairwise closeness matrix for a list of OD solutions.

    Parameters
    ----------
    solutions : list of (c, P) tuples
        Each entry is (mean_vector, covariance_matrix).

    Returns
    -------
    D : (N, N) ndarray, symmetric, zeros on diagonal.
        D[i, j] = d(Si, Sj) in units of sigma.
    """
    N = len(solutions)
    D = np.zeros((N, N))
    total = N * (N - 1) // 2
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            c1, P1 = solutions[i]
            c2, P2 = solutions[j]
            res = od_closeness(c1, P1, c2, P2)
            D[i, j] = D[j, i] = res["d"]
            count += 1
            print(f"  [{count}/{total}] d(S{i+1}, S{j+1}) = {res['d']:.4f} σ")
    return D


def plot_consistency_matrix(
    solutions,
    labels=None,
    scheme="merb",
    title="OD Solution Consistency Matrix",
    show_values=True,
    figsize=None,
    save_path=None,
):
    """
    Compute and plot the N×N pairwise solution consistency matrix.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import BoundaryNorm, ListedColormap

    N = len(solutions)
    if labels is None:
        labels = [f"S{i+1}" for i in range(N)]

    print(f"Computing {N*(N-1)//2} pairwise closeness measures…")
    D = compute_consistency_matrix(solutions)
    print("Done.\n")

    # ---- colormap setup ------------------------------------------------
    MERB_BOUNDS = [0.0, 0.2, 0.5, 0.8, 1.2, 9999]
    MERB_COLORS = ["#1a9e74", "#6cc9a9", "#3a7abf", "#d95f30", "#8b1a1a"]
    MERB_LABELS = ["< 0.2", "0.2–0.5", "0.5–0.8", "0.8–1.2", "≥ 1.2"]

    MERA_BOUNDS = [0.0, 0.8, 9999]
    MERA_COLORS = ["#3a7abf", "#d95f30"]
    MERA_LABELS = ["≤ 0.8  (consistent)", "> 0.8  (inconsistent)"]

    SIGMA_BOUNDS = [0.0, 1.0, 2.0, 3.0, 9999]
    SIGMA_COLORS = ["#1a9e74", "#6cc9a9", "#d95f30", "#8b1a1a"]
    SIGMA_LABELS = ["≤ 1σ", "1σ – 2σ", "2σ – 3σ", "> 3σ"]

    if scheme == "mera":
        bounds, colors, leg_labels = MERA_BOUNDS, MERA_COLORS, MERA_LABELS
    elif scheme == "sigma":
        bounds, colors, leg_labels = SIGMA_BOUNDS, SIGMA_COLORS, SIGMA_LABELS
    elif scheme == "continuous":
        bounds, colors, leg_labels = None, None, None
    else:
        bounds, colors, leg_labels = MERB_BOUNDS, MERB_COLORS, MERB_LABELS

    # ---- figure --------------------------------------------------------
    cell_in = 0.62
    margin = 1.8
    if figsize is None:
        sz = N * cell_in + margin
        figsize = (sz + 1.0, sz)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    # Mask lower triangle since D is symmetric
    mask = np.tril(np.ones_like(D, dtype=bool), k=-1)
    D_plot = np.ma.masked_where(mask, D)

    if bounds is not None:
        cmap = ListedColormap(colors)
        cmap.set_bad(color="#f8f8f8")
        norm = BoundaryNorm(bounds, cmap.N)
        ax.imshow(D_plot, cmap=cmap, norm=norm, aspect="equal")
    else:
        vmax = max(float(np.nanmax(D)), 1.5)
        cmap = plt.cm.RdYlGn_r.copy()
        cmap.set_bad(color="#f8f8f8")
        im = ax.imshow(D_plot, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")

    # ---- diagonal (grey) -----------------------------------------------
    for i in range(N):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="#cccccc", zorder=2))
        ax.text(
            i, i, "—", ha="center", va="center", fontsize=8, color="#888888", zorder=3
        )

    # ---- cell text -----------------------------------------------------
    if show_values:
        for i in range(N):
            for j in range(i + 1, N):  # upper triangle only
                val = D[i, j]

                if bounds is not None:
                    idx = int(np.searchsorted(bounds[1:], val))
                    idx = min(idx, len(colors) - 1)

                    r, g, b = (
                        int(colors[idx].lstrip("#")[k : k + 2], 16) / 255
                        for k in (0, 2, 4)
                    )

                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    tc = "white" if lum < 0.55 else "#1a1a1a"
                else:
                    tc = "white"

                fs = max(6, min(9, 90 // N))
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=tc,
                    fontweight="bold",
                    zorder=4,
                )

    # ---- axes ----------------------------------------------------------
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(length=0)

    for sp in ax.spines.values():
        sp.set_visible(False)

    for k in range(N + 1):
        ax.axhline(k - 0.5, color="white", linewidth=0.8, zorder=1)
        ax.axvline(k - 0.5, color="white", linewidth=0.8, zorder=1)

    # ---- legend --------------------------------------------------------
    if bounds is not None:
        patches = [
            mpatches.Patch(facecolor=c, edgecolor="#999", linewidth=0.5, label=l)
            for c, l in zip(colors, leg_labels)
        ]
        ax.legend(
            handles=patches,
            title="Closeness d (σ)",
            title_fontsize=8,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            framealpha=0.9,
            edgecolor="#cccccc",
        )
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Closeness d (σ)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    # ---- title & footer ------------------------------------------------
    ax.set_title(title, fontsize=11, fontweight="500", pad=12)
    ax.set_xlabel("Solution", fontsize=9, labelpad=6)
    ax.set_ylabel("Solution", fontsize=9, labelpad=6)

    upper = D[np.triu_indices(N, k=1)]
    footer = (
        f"N = {N} solutions   |   "
        f"min d = {np.nanmin(upper):.3f} σ   "
        f"mean d = {np.nanmean(upper):.3f} σ   "
        f"max d = {np.nanmax(upper):.3f} σ"
    )
    fig.text(
        0.5, 0.01, footer, ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return D


# ---------------------------------------------------------------------------
# Example: many solutions → consistency matrix
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ----- build a set of solutions -----
    # Replace this block with your real (c, P) pairs.
    rng_m = np.random.default_rng(99)

    def _make_sol(center, spread, n=6):
        c = center + rng_m.standard_normal(n) * spread
        A = rng_m.standard_normal((n, n)) * spread
        P = A @ A.T / n + np.eye(n) * spread**2 * 0.4
        return c, P

    fA = np.zeros(6)
    fB = np.array([1.0, 0.8, 0.4, 0.2, 0.1, 0.05])

    solutions = [_make_sol(fA, 0.2) for _ in range(4)] + [  # family A — tight cluster
        _make_sol(fB, 0.3) for _ in range(4)
    ]  # family B — offset cluster
    labels = [f"A{i+1}" for i in range(4)] + [f"B{i+1}" for i in range(4)]

    # ----- plot all three schemes -----
    for scheme in ("merb", "mera", "continuous", "sigma"):
        D = plot_consistency_matrix(
            solutions,
            labels=labels,
            scheme=scheme,
            title=f"OD Solution Consistency  ({scheme.upper()} style)",
        )

    print("\nFull consistency matrix D (σ):")
    print(np.round(D, 3))
