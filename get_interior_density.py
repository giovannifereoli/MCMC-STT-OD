# ======================================================================
# FROM SCRATCH: mascons strictly INSIDE Bennu + SH mapping + emcee + plots
#   - Mascons are sampled uniformly in VOLUME inside the closed OBJ mesh
#   - Layered densities are assigned by radial bins (or you can change rule)
#   - Corner plot at the end + density plot inside shape (surface mesh)
# ======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict, Optional
import math
import numpy as np
import matplotlib.pyplot as plt

# Optional (fast contains test + fast plotting helpers). If unavailable, we fall back.
_TRIMESH_AVAILABLE = False
try:
    import trimesh  # pip install trimesh

    _TRIMESH_AVAILABLE = True
except Exception:
    _TRIMESH_AVAILABLE = False


# ============================================================
# OBJ loading / triangulation (minimal)
# ============================================================


def load_obj(filepath: str) -> Tuple[np.ndarray, List[List[int]]]:
    verts = []
    faces = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = []
                for p in parts:
                    v_str = p.split("/")[0]
                    vi = int(v_str)
                    if vi < 0:
                        vi = len(verts) + vi + 1
                    idx.append(vi - 1)
                if len(idx) >= 3:
                    faces.append(idx)
    return np.asarray(verts, dtype=float), faces


def triangulate_faces(faces: List[List[int]]) -> np.ndarray:
    tris = []
    for face in faces:
        a = face[0]
        for i in range(1, len(face) - 1):
            tris.append([a, face[i], face[i + 1]])
    return np.asarray(tris, dtype=int)


def signed_volume_of_mesh(V: np.ndarray, T: np.ndarray) -> float:
    v0 = V[T[:, 0]]
    v1 = V[T[:, 1]]
    v2 = V[T[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def enforce_outward_orientation(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    if signed_volume_of_mesh(V, T) < 0.0:
        return T[:, [0, 2, 1]]
    return T


# ============================================================
# Mascon model
# ============================================================


@dataclass
class Mascon:
    pos: np.ndarray  # (3,)
    volume: float  # [m^3]
    layer: int  # layer id
    mass: float = 0.0  # [kg] (set from density * volume)


# ============================================================
# Point-in-mesh (robust, fast if trimesh is installed)
# ============================================================


def build_trimesh(V: np.ndarray, T: np.ndarray):
    if not _TRIMESH_AVAILABLE:
        return None
    return trimesh.Trimesh(vertices=V, faces=T, process=False)


# ---- fallback ray casting (dependency-free, slower) ----


def _tri_ray_intersect_moller(
    tri: np.ndarray, ray_o: np.ndarray, ray_d: np.ndarray, eps=1e-12
) -> bool:
    v0, v1, v2 = tri
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(ray_d, e2)
    det = float(np.dot(e1, pvec))
    if abs(det) < eps:
        return False
    inv_det = 1.0 / det
    tvec = ray_o - v0
    u = float(np.dot(tvec, pvec)) * inv_det
    if u < 0.0 or u > 1.0:
        return False
    qvec = np.cross(tvec, e1)
    v = float(np.dot(ray_d, qvec)) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return False
    t = float(np.dot(e2, qvec)) * inv_det
    return t > eps


def contains_points_ray_cast(V: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Parity test (odd intersections) along a fixed non-axis ray direction.
    Works for closed watertight meshes. Slower than trimesh.contains.
    """
    ray_d = np.array([1.0, 0.137, 0.291], dtype=float)
    ray_d /= np.linalg.norm(ray_d)

    tris = V[T]  # (Nt,3,3)
    inside = np.zeros((P.shape[0],), dtype=bool)
    for i, p in enumerate(P):
        cnt = 0
        for tri in tris:
            if _tri_ray_intersect_moller(tri, p, ray_d):
                cnt += 1
        inside[i] = (cnt % 2) == 1
    return inside


def mesh_contains_points(
    mesh_tm, V: np.ndarray, T: np.ndarray, P: np.ndarray
) -> np.ndarray:
    if mesh_tm is not None:
        # trimesh expects shape (N,3)
        return mesh_tm.contains(P)
    # fallback
    return contains_points_ray_cast(V, T, P)


# ============================================================
# SAMPLE MASCONS STRICTLY INSIDE MESH (uniform in volume)
# ============================================================


def sample_points_in_aabb(
    rng: np.random.Generator, bmin: np.ndarray, bmax: np.ndarray, n: int
) -> np.ndarray:
    return rng.uniform(bmin, bmax, size=(n, 3))


def sample_mascons_inside_mesh_fast(
    V: np.ndarray,
    T: np.ndarray,
    n_total: int,
    n_layers: int,
    rho_init: np.ndarray,
    layer_rule: str = "radial_bins",
    seed: int = 0,
    chunk: int = 5000,
    max_tries: int = 50,
):
    """
    Fast uniform-in-volume sampling inside a watertight triangular mesh using trimesh.
    Robust to cases where trimesh returns fewer points than requested by sampling repeatedly.
    """
    rng = np.random.default_rng(seed)

    mesh = trimesh.Trimesh(vertices=V, faces=T, process=False)

    # Lightweight cleanup that sometimes fixes "almost watertight" meshes
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    # Try to fill holes (won't always fix)
    try:
        mesh.fill_holes()
    except Exception:
        pass

    if not mesh.is_watertight:
        raise ValueError(
            "Mesh is not watertight; trimesh.sample.volume_mesh may return too few points.\n"
            "Fix the OBJ (closed surface) or use the voxel fallback sampler."
        )

    # --- sample until we have n_total points ---
    np.random.seed(seed)  # trimesh uses numpy RNG internally
    pts_list = []
    got = 0
    tries = 0

    while got < n_total and tries < max_tries:
        tries += 1
        n_req = min(chunk, n_total - got)
        P = trimesh.sample.volume_mesh(mesh, n_req)

        # P can (rarely) come back shorter than requested; keep it anyway
        if P is None:
            P = np.zeros((0, 3), dtype=float)
        if P.ndim != 2 or P.shape[1] != 3:
            raise RuntimeError(
                f"Unexpected volume_mesh output shape: {getattr(P,'shape',None)}"
            )

        if P.shape[0] > 0:
            pts_list.append(P)
            got += P.shape[0]

        print(f"volume_mesh sampling: {got}/{n_total}", end="\r")

    print()

    if got < n_total:
        raise RuntimeError(
            f"volume_mesh could only generate {got} points after {tries} tries (wanted {n_total}).\n"
            "This usually indicates a mesh issue even if is_watertight=True, or numerical problems.\n"
            "Try voxel fallback."
        )

    P = np.vstack(pts_list)[:n_total]  # EXACT length

    # --- per-mascon volume ---
    Vmesh = float(mesh.volume)
    if not np.isfinite(Vmesh) or Vmesh <= 0.0:
        raise ValueError(f"Mesh volume invalid: {Vmesh}")
    vol_i = Vmesh / float(n_total)

    # --- assign layers ---
    rho_init = np.array(rho_init, dtype=float).reshape(-1)

    if layer_rule == "none":
        n_layers = 1
        layers = np.zeros((n_total,), dtype=int)
        if rho_init.size == 0:
            rho_init = np.array([1500.0])
        else:
            rho_init = rho_init[:1]

    elif layer_rule == "radial_bins":
        if n_layers < 1:
            n_layers = 1

        if rho_init.size != n_layers:
            if rho_init.size < n_layers:
                rho_init = np.pad(rho_init, (0, n_layers - rho_init.size), mode="edge")
            else:
                rho_init = rho_init[:n_layers]

        r = np.linalg.norm(P, axis=1)
        rmax = float(np.max(np.linalg.norm(V, axis=1)))  # safe upper bound
        edges = np.linspace(0.0, rmax, n_layers + 1)
        layers = np.clip(np.digitize(r, edges) - 1, 0, n_layers - 1).astype(int)

    else:
        raise ValueError("layer_rule must be 'none' or 'radial_bins'")

    # --- build mascons ---
    mascons = []
    for i in range(
        P.shape[0]
    ):  # NOTE: use len(P) not n_total (but len(P)==n_total anyway)
        lay = int(layers[i])
        mass = float(rho_init[lay]) * vol_i
        mascons.append(Mascon(pos=P[i], volume=vol_i, layer=lay, mass=mass))

    return mascons


# ============================================================
# Fully-normalized Legendre + Mascons -> Cbar/Sbar
# ============================================================


def fully_normalized_legendre(nmax: int, x: float) -> np.ndarray:
    P = np.zeros((nmax + 1, nmax + 1), dtype=float)
    P[0, 0] = 1.0
    if nmax == 0:
        return P

    sx = math.sqrt(max(0.0, 1.0 - x * x))
    for m in range(1, nmax + 1):
        P[m, m] = math.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sx * P[m - 1, m - 1]

    for m in range(0, nmax):
        P[m + 1, m] = math.sqrt(2.0 * m + 3.0) * x * P[m, m]

    for m in range(0, nmax + 1):
        for n in range(m + 2, nmax + 1):
            a = math.sqrt(((2.0 * n + 1.0) * (2.0 * n - 1.0)) / ((n - m) * (n + m)))
            b = math.sqrt(
                ((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0))
                / ((2.0 * n - 3.0) * (n - m) * (n + m))
            )
            P[n, m] = a * x * P[n - 1, m] - b * P[n - 2, m]
    return P


def mascons_to_cs(
    mascons: Iterable[Mascon],
    n_degree: int,
    ref_radius: float,
    exclude_c00: bool = True,
):
    mascons = list(mascons)
    Mtot = float(sum(mc.mass for mc in mascons))
    if Mtot <= 0.0:
        raise ValueError("Total mass must be > 0")

    C = np.zeros((n_degree + 1, n_degree + 1), dtype=float)
    S = np.zeros((n_degree + 1, n_degree + 1), dtype=float)

    for mc in mascons:
        x, y, z = float(mc.pos[0]), float(mc.pos[1]), float(mc.pos[2])
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0.0:
            continue
        lam = math.atan2(y, x)
        sinphi = z / r
        Pbar = fully_normalized_legendre(n_degree, sinphi)

        rho = r / ref_radius
        rpow = 1.0
        w = float(mc.mass) / Mtot

        for n in range(0, n_degree + 1):
            if n > 0:
                rpow *= rho
            for m in range(0, n + 1):
                if exclude_c00 and (n == 0 and m == 0):
                    continue
                basis = rpow * Pbar[n, m]
                C[n, m] += w * basis * math.cos(m * lam)
                S[n, m] += w * basis * math.sin(m * lam)

    if exclude_c00:
        C[0, 0] = 0.0
        S[0, 0] = 0.0
    else:
        C[0, 0] = 1.0
        S[0, 0] = 0.0

    return C, S


def flatten_cs(
    C: np.ndarray, S: np.ndarray, n_degree: int, exclude_c00: bool = True
) -> np.ndarray:
    c_list, s_list = [], []
    for n in range(0, n_degree + 1):
        for m in range(0, n + 1):
            if exclude_c00 and (n == 0 and m == 0):
                continue
            c_list.append(C[n, m])
            s_list.append(S[n, m])
    return np.array(c_list + s_list, dtype=float)


def build_mascon_design_matrix(
    mascons: List[Mascon], n_degree: int, ref_radius: float, exclude_c00: bool = True
) -> np.ndarray:
    mascons = list(mascons)
    Nmc = len(mascons)

    n_terms = 0
    for n in range(0, n_degree + 1):
        for m in range(0, n + 1):
            if exclude_c00 and (n == 0 and m == 0):
                continue
            n_terms += 1

    A = np.zeros((2 * n_terms, Nmc), dtype=float)

    for i, mc in enumerate(mascons):
        x, y, z = float(mc.pos[0]), float(mc.pos[1]), float(mc.pos[2])
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0.0:
            continue
        lam = math.atan2(y, x)
        sinphi = z / r
        Pbar = fully_normalized_legendre(n_degree, sinphi)

        rho = r / ref_radius
        rpow = 1.0

        row_c = 0
        row_s = n_terms
        for n in range(0, n_degree + 1):
            if n > 0:
                rpow *= rho
            for m in range(0, n + 1):
                if exclude_c00 and (n == 0 and m == 0):
                    continue
                basis = rpow * Pbar[n, m]
                A[row_c, i] = basis * math.cos(m * lam)
                A[row_s, i] = basis * math.sin(m * lam)
                row_c += 1
                row_s += 1

    return A


# ============================================================
# Bennu "truth" CS (exact values YOU already have elsewhere)
# Here: keep your function, or swap for your own tables.
# ============================================================


def bennu_truth_cs_from_literature(ndeg: int):
    # Keep this minimal: only up to degree 5 filled as before, higher = 0
    Rref = 290.0
    GM = 4.890450

    C = np.zeros((ndeg + 1, ndeg + 1), dtype=float)
    S = np.zeros((ndeg + 1, ndeg + 1), dtype=float)
    C[0, 0] = 1.0

    J = {2: 1.926e-2, 3: -1.22e-3, 4: -6.50e-3, 5: 6.7e-5}
    for n, Jn in J.items():
        if n <= ndeg:
            C[n, 0] = -Jn

    if ndeg >= 2:
        C[2, 2] = 3.06e-3
        S[2, 2] = -1.09e-3

    if ndeg >= 3:
        C[3, 1] = 8.15e-4
        S[3, 1] = -5.43e-4
        C[3, 2] = -9.35e-4
        S[3, 2] = -5.38e-4
        C[3, 3] = 1.17e-3
        S[3, 3] = -3.1e-4

    if ndeg >= 4:
        C[4, 1] = -8.82e-4
        S[4, 1] = -5.8e-4
        C[4, 2] = -8.71e-4
        S[4, 2] = -8.4e-5
        C[4, 3] = -7.6e-5
        S[4, 3] = -3.9e-4
        C[4, 4] = 7.7e-4
        S[4, 4] = 2.25e-3

    if ndeg >= 5:
        C[5, 1] = -3.5e-4
        S[5, 1] = 1.6e-4
        C[5, 2] = -3.7e-5
        S[5, 2] = -2.7e-4
        C[5, 3] = -2.2e-6
        S[5, 3] = -9.5e-6
        C[5, 4] = 3.2e-4
        S[5, 4] = 5.0e-5
        C[5, 5] = -2.3e-5
        S[5, 5] = 3.0e-4

    return C, S, Rref, GM


# ============================================================
# emcee fit of layer densities (fast, uses design matrix)
# ============================================================


def fit_layer_densities_with_emcee(
    mascons: List[Mascon],
    C_obs: np.ndarray,
    S_obs: np.ndarray,
    n_degree: int,
    ref_radius: float,
    sigma_cs: float | np.ndarray,
    rho0: np.ndarray,
    rho_bounds: List[Tuple[float, float]],
    n_walkers: int = 48,
    n_steps: int = 3000,
    burn: int = 1000,
    thin: int = 5,
    exclude_c00: bool = True,
    seed: int = 0,
):
    try:
        import emcee
    except Exception as e:
        raise ImportError("Need emcee: pip install emcee") from e

    rng = np.random.default_rng(seed)

    A = build_mascon_design_matrix(
        mascons, n_degree, ref_radius, exclude_c00=exclude_c00
    )
    y_obs = flatten_cs(C_obs, S_obs, n_degree, exclude_c00=exclude_c00)

    if np.isscalar(sigma_cs):
        sigma = np.full_like(y_obs, float(sigma_cs))
    else:
        sigma = np.array(sigma_cs, dtype=float).reshape(-1)
        if sigma.shape[0] != y_obs.shape[0]:
            raise ValueError("sigma_cs has wrong length.")
    inv_var = 1.0 / (sigma * sigma)

    K = int(len(rho0))
    if len(rho_bounds) != K:
        raise ValueError("rho_bounds must match rho0 length")

    volumes = np.array([mc.volume for mc in mascons], dtype=float)
    layers = np.array([mc.layer for mc in mascons], dtype=int)

    def log_prior(rho: np.ndarray) -> float:
        for k in range(K):
            lo, hi = rho_bounds[k]
            if rho[k] < lo or rho[k] > hi:
                return -np.inf
        return 0.0

    def y_model(rho: np.ndarray) -> np.ndarray:
        w = rho[layers] * volumes
        Mtot = float(np.sum(w))
        if Mtot <= 0.0:
            return np.full_like(y_obs, np.nan)
        return (A @ w) / Mtot

    def log_like(rho: np.ndarray) -> float:
        ym = y_model(rho)
        if not np.all(np.isfinite(ym)):
            return -np.inf
        r = y_obs - ym
        return -0.5 * float(np.sum(r * r * inv_var))

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_like(theta)

    p0 = rho0 + 1e-6 * rng.standard_normal((n_walkers, K))

    sampler = emcee.EnsembleSampler(n_walkers, K, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=True)

    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
    mean = np.mean(flat, axis=0)
    cov = np.cov(flat.T)
    return sampler, flat, (mean, cov)


# ============================================================
# PLOTS: corner + density-in-shape
# ============================================================


def plot_corner(
    flat_samples: np.ndarray,
    labels: Optional[List[str]] = None,
    truths: Optional[np.ndarray] = None,
):
    try:
        import corner

        fig = corner.corner(
            flat_samples,
            labels=labels,
            truths=truths,
            show_titles=True,
            title_fmt=".3g",
            quantiles=[0.16, 0.5, 0.84],
            bins=40,
        )
        plt.show()
        return fig
    except Exception:
        # fallback
        X = flat_samples
        K = X.shape[1]
        if labels is None:
            labels = [f"p{k}" for k in range(K)]
        fig, axes = plt.subplots(K, K, figsize=(2.6 * K, 2.6 * K))
        for i in range(K):
            for j in range(K):
                ax = axes[i, j]
                if i == j:
                    ax.hist(X[:, j], bins=40, density=True, alpha=0.85)
                    if truths is not None:
                        ax.axvline(truths[j], linewidth=2)
                elif i > j:
                    ax.plot(X[:, j], X[:, i], ".", ms=1.2, alpha=0.15)
                    if truths is not None:
                        ax.plot(truths[j], truths[i], "x", ms=8, mew=2)
                else:
                    ax.axis("off")
                if i == K - 1 and j < K:
                    ax.set_xlabel(labels[j])
                if j == 0 and i < K:
                    ax.set_ylabel(labels[i])
        plt.tight_layout()
        plt.show()
        return fig


def plot_density_in_shape(
    V: np.ndarray,
    T: np.ndarray,
    mascons: List[Mascon],
    rho_layers: np.ndarray,
    title: str,
):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    P = np.array([mc.pos for mc in mascons], dtype=float)
    dens = np.array([float(rho_layers[mc.layer]) for mc in mascons], dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # mesh surface (subsample for speed)
    step = max(1, len(T) // 8000)
    tris = V[T[::step]]
    surf = Poly3DCollection(tris, alpha=0.08, linewidths=0.0)
    ax.add_collection3d(surf)

    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=dens, s=500, alpha=0.95)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
    cb.set_label(r"Density $\rho$ [kg/m$^3$]")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    mins = np.min(V, axis=0)
    maxs = np.max(V, axis=0)
    ctr = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    ax.set_xlim(ctr[0] - 0.5 * span, ctr[0] + 0.5 * span)
    ax.set_ylim(ctr[1] - 0.5 * span, ctr[1] + 0.5 * span)
    ax.set_zlim(ctr[2] - 0.5 * span, ctr[2] + 0.5 * span)

    plt.tight_layout()
    plt.show()
    return fig


def pca_body_axes(V: np.ndarray):
    """
    PCA axes from vertices.
    Returns centroid, axes (3x3 with columns = principal axes), and half-lengths along axes.
    """
    ctr = np.mean(V, axis=0)
    X = V - ctr
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)  # ascending
    # sort descending variance -> axis 0 is "long" axis
    idx = np.argsort(evals)[::-1]
    axes = evecs[:, idx]

    # extents along axes
    proj = X @ axes  # (N,3)
    mins = np.min(proj, axis=0)
    maxs = np.max(proj, axis=0)
    half = 0.5 * (maxs - mins)
    return ctr, axes, half


def pull_point_inside(
    p: np.ndarray,
    ctr: np.ndarray,
    mesh_tm,
    V: np.ndarray,
    T: np.ndarray,
    max_iter: int = 60,
) -> np.ndarray:
    """
    If p is outside, shrink it toward ctr until inside (bisection on scale).
    Assumes ctr is inside (reasonable for Bennu-like meshes; if not, pick an interior point).
    """
    p = np.array(p, dtype=float)
    ctr = np.array(ctr, dtype=float)

    # quick accept
    if bool(mesh_contains_points(mesh_tm, V, T, p.reshape(1, 3))[0]):
        return p

    # bisection on alpha in [0,1]: ctr + alpha*(p-ctr)
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pmid = ctr + mid * (p - ctr)
        inside = bool(mesh_contains_points(mesh_tm, V, T, pmid.reshape(1, 3))[0])
        if inside:
            lo = mid  # can go further out
        else:
            hi = mid  # too far, pull in
    return ctr + lo * (p - ctr)


def build_strategic_big_mascons(
    V: np.ndarray,
    T: np.ndarray,
    K: int = 7,
    seed: int = 0,
    fractions: Tuple[float, float, float] = (0.65, 0.45, 0.45),
):
    """
    Build K 'big mascons' in strategic positions based on PCA body axes:
      - center
      - ± long-axis lobes
      - ± intermediate-axis
      - ± short-axis (poles)
    If K is smaller, it truncates the list. If K is bigger, it adds a few random interior points.

    Returns:
      mascons (list[Mascon]) with layer = unique per mascon (so you fit one rho per mascon)
      rho_init (K,)
    """
    rng = np.random.default_rng(seed)
    mesh_tm = build_trimesh(V, T)

    ctr, axes, half = pca_body_axes(V)

    # build candidate positions in body principal-axis frame
    f1, f2, f3 = fractions
    a1 = half[0] * f1
    a2 = half[1] * f2
    a3 = half[2] * f3

    # helper: ctr + axes @ [dx,dy,dz]
    def p_body(dx, dy, dz):
        return ctr + axes @ np.array([dx, dy, dz], dtype=float)

    candidates = [
        p_body(0, 0, 0),  # core
        p_body(+a1, 0, 0),  # +long lobe
        p_body(-a1, 0, 0),  # -long lobe
        p_body(0, +a2, 0),  # +mid
        p_body(0, -a2, 0),  # -mid
        p_body(0, 0, +a3),  # +pole
        p_body(0, 0, -a3),  # -pole
    ]

    # ensure inside (shrink toward centroid if needed)
    cand_in = []
    for p in candidates:
        cand_in.append(pull_point_inside(p, ctr, mesh_tm, V, T))

    # truncate to K
    P = cand_in[:K]

    # if user asks for more than 7, add random interior points (still “big” but extra)
    # note: this is a lightweight fill-in; you can replace with a smarter “neck” locator if you want.
    if K > len(P):
        # build an AABB sampler + inside test
        bmin = np.min(V, axis=0)
        bmax = np.max(V, axis=0)
        need = K - len(P)
        added = 0
        tries = 0
        while added < need and tries < 200:
            tries += 1
            Q = rng.uniform(bmin, bmax, size=(max(1000, 50 * need), 3))
            inside = mesh_contains_points(mesh_tm, V, T, Q)
            Q_in = Q[inside]
            if Q_in.shape[0] > 0:
                take = min(Q_in.shape[0], need - added)
                for j in range(take):
                    P.append(Q_in[j])
                added += take

        if added < need:
            raise RuntimeError(f"Could not find enough interior points for K={K}")

    P = np.asarray(P, dtype=float)

    # use equal mascon volume = mesh volume / K (units consistent with your V units!)
    if _TRIMESH_AVAILABLE:
        mesh = trimesh.Trimesh(vertices=V, faces=T, process=False)
        Vmesh = float(mesh.volume)
    else:
        # fallback: volume from signed volume of triangles (needs outward orientation)
        Vmesh = abs(signed_volume_of_mesh(V, T))
    vol_i = Vmesh / float(K)

    # make each mascon its own "layer" index so emcee estimates one rho per mascon
    mascons = []
    for i in range(K):
        mascons.append(Mascon(pos=P[i], volume=vol_i, layer=i, mass=0.0))

    # init densities (kg/m^3) per mascon — start uniform
    rho_init = np.full((K,), 1400.0, dtype=float)

    return mascons, rho_init


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    obj_path = "ObjFiles/BennuRadar.obj"

    # --- Load Bennu shape mesh
    V, faces = load_obj(obj_path)
    T = triangulate_faces(faces)
    T = enforce_outward_orientation(V, T)

    # --- Observed Bennu field (replace with your own truth if you have more degrees)
    ndeg = 10
    C_obs, S_obs, Rref, GM_obs = bennu_truth_cs_from_literature(ndeg)
    print("Bennu truth loaded: Rref =", Rref, "GM =", GM_obs)

    # --- 1) Generate mascons strictly INSIDE the mesh
    # Choose many points so it looks dense and the mapping is smoother.
    n_total = 12000

    # Layers: choose number of density parameters you want to estimate.
    n_layers = 10

    # initial densities (kg/m^3)
    rho_init = np.array([1400.0] * n_layers, dtype=float)

    # IMPORTANT: this samples inside the polyhedron via mesh containment test.
    """'
    mascons = sample_mascons_inside_mesh_fast(
        V,
        T,
        n_total=n_total,
        n_layers=n_layers,
        rho_init=rho_init,
        layer_rule="radial_bins",
        seed=1,
    )"""

    # --- 1) Strategic BIG mascons instead of layered dense sampling
    K = 7  # try 5–9 first; 7 = core + ±long + ±mid + ±pole
    mascons, rho_init = build_strategic_big_mascons(
        V, T, K=K, seed=1, fractions=(0.65, 0.45, 0.45)
    )

    # each mascon is its own parameter
    n_layers = K  # keep variable name so the rest of the script is unchanged

    # quick sanity: check max radius of mascons vs mesh
    r_mc = np.linalg.norm(np.array([mc.pos for mc in mascons]), axis=1)
    r_v = np.linalg.norm(V, axis=1)
    print(
        "max ||r|| mascons =",
        float(np.max(r_mc)),
        " max ||r|| vertices =",
        float(np.max(r_v)),
    )

    # --- 2) Fit layer densities with emcee
    rho0 = rho_init.copy()
    rho_bounds = [(500.0, 2000.0)] * n_layers
    sigma_cs = 1e-6

    sampler, flat_samples, (rho_mean, rho_cov) = fit_layer_densities_with_emcee(
        mascons=mascons,
        C_obs=C_obs,
        S_obs=S_obs,
        n_degree=ndeg,
        ref_radius=Rref,
        sigma_cs=sigma_cs,
        rho0=rho0,
        rho_bounds=rho_bounds,
        n_walkers=48,
        n_steps=3000,
        burn=1000,
        thin=5,
        exclude_c00=True,
        seed=2,
    )

    print("Posterior mean densities [kg/m^3] =", rho_mean)
    print("Posterior std  densities [kg/m^3] =", np.sqrt(np.diag(rho_cov)))

    # --- 3) Corner plot
    labels = [f"rho_{k} [kg/m^3]" for k in range(n_layers)]
    plot_corner(flat_samples, labels=labels, truths=None)

    # --- 4) Density plot inside Bennu
    plot_density_in_shape(
        V,
        T,
        mascons,
        rho_mean,
        title="Posterior mean density (mascons strictly inside Bennu)",
    )
