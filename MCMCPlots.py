"""
Figures of the MCMC orbit-determination runs: journal figures and quick-look diagnostics.

Journal figures (Okabe-Ito colors, LaTeX text, printed size, one vector PDF each):

    import MCMCPlots
    MCMCPlots.SHOW_FIGS = False                   # plt.show() after each saved figure
    MCMCPlots.configure(param_sym=..., param_unit=..., param_scale=..., body="Bennu",
                        rotation=make_bennu_rotation_matrix)
    MCMCPlots.set_figure_dir("results/<run>_<timestamp>")
    MCMCPlots.plot_scene_body_fixed(...)
    model.plot_journal_diagnostics(...)           # = MCMCPlots.plot_mcmc_diagnostics(model, ...)

Sections:
    FIGURE STYLE           style, colors, configure(), auto_param_scales()
    FIGURE OUTPUT          set_figure_dir(), save_figure(), write_output()
    SCENE / MEASUREMENTS   scene, observation geometry, post-fit residuals
    MCMC DIAGNOSTICS       traces, autocorrelation, corner (+ prior, + RTN frame:
                           rtn_corner_map), all-in-one plot_mcmc_diagnostics
    GMM PROPAGATION        figures of GMM.GMM.propagate_stm results
    STT BENCHMARK          STT surrogate vs full nonlinear likelihood (J1_benchmark_STTs.py)
    QUICK-LOOK (legacy)    the original MCMCModel.plot_* figures (Tests/, Extra/ scripts)

The quick-look rcParams below are applied on import (as MCMC.py always did); configure()
replaces them with the journal style.
"""

import io
import math
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import corner
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import chi2, gaussian_kde, norm
from statsmodels.tsa.stattools import acf

from GMM import GMM

# Quick-look (legacy MCMCModel.plot_*) defaults, applied on import
LEGACY_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.7,
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(LEGACY_STYLE)


def _patch_type1_subset():
    r"""
    Keep encoding slot 0 of the usetex Type-1 fonts embedded in PDFs (matplotlib 3.11).

    matplotlib 3.11 subsets the TeX Type-1 fonts it embeds, and Type1Font.subset()
    overwrites encoding slot 0 with .notdef: the minus sign (cmsy slot 0) and \Gamma (cmr
    slot 0) vanish from every usetex PDF. When a font really uses slot 0 it is embedded
    whole, as matplotlib <= 3.10 always did; every other font is still subset. Idempotent,
    and a no-op on matplotlib versions without Type1Font.subset.
    """
    try:
        from matplotlib._type1font import Type1Font
    except ImportError:
        return
    orig = getattr(Type1Font, "subset", None)
    if orig is None or getattr(orig, "_jf_keeps_slot0", False):
        return

    def subset(self, characters, name_prefix):
        """Embed the whole font when slot 0 is really used, else subset as usual."""
        characters = frozenset(characters)
        if 0 in characters and self.prop["Encoding"].get(0, ".notdef") != ".notdef":
            return self
        return orig(self, characters, name_prefix)

    subset._jf_keeps_slot0 = True
    Type1Font.subset = subset


_patch_type1_subset()

# ============================================================
# FIGURE STYLE (journal)
# ============================================================
# Figures are drawn at their printed size (single column COL_W, full text width TXT_W) with
# LaTeX text at 7-8 pt, so they drop into a two-column journal layout without rescaling.
# Vector PDFs; only dense layers (mesh, point clouds, walker traces) are rasterized.
# The style is applied by configure(), so importing this module leaves the quick-look
# defaults (LEGACY_STYLE) untouched for scripts that do not draw journal figures.

COL_W = 3.5  # single-column width [in]
TXT_W = 7.16  # full text width (double column) [in]
SHOW_FIGS = True  # plt.show() after each figure is saved

def _ensure_tex_on_path():
    """
    True if latex + dvipng are reachable. GUI-launched processes (VS Code, Jupyter, Spyder on
    macOS) often do not inherit the login-shell PATH, so the standard TeX locations are added
    before giving up; otherwise matplotlib would silently fall back to mathtext and DejaVu.
    """
    if shutil.which("latex") and shutil.which("dvipng"):
        return True
    import glob

    extra = ["/Library/TeX/texbin", "/opt/homebrew/bin", "/usr/local/bin", "/usr/texbin"]
    extra += sorted(glob.glob("/usr/local/texlive/*/bin/*"), reverse=True)
    path = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(path + [d for d in extra if os.path.isdir(d)])
    return bool(shutil.which("latex") and shutil.which("dvipng"))


USE_TEX = _ensure_tex_on_path()
if not USE_TEX:
    print("[figures] latex/dvipng not found: falling back to mathtext (Computer Modern)")
EN_DASH = "--" if USE_TEX else "-"  # cmr10 (mathtext fallback) has no en-dash glyph
# Every journal label renders identically in both modes: math lives inside $...$,
# percent signs are written $\%$, and only commands mathtext also knows are used.
STYLE = dict(
    {
        "text.usetex": USE_TEX,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathrsfs}",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "0.15",
        "axes.labelcolor": "0.1",
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def apply_style():
    """Set the journal rcParams. configure() calls it before any journal figure is drawn."""
    plt.rcParams.update(STYLE)

# Okabe-Ito colors (colorblind-safe; adjacent pairs checked for CVD separation).
# Each color has ONE meaning in every figure.
C_POST = "#0072B2"  # MCMC posterior / Gaussian mixture (blue)
C_BATCH = "#D55E00"  # Gauss-Newton batch / single Gaussian (vermillion)
C_SC = "#009E73"  # spacecraft, images used (bluish green)
C_OCC = "#E69F00"  # occulted epochs (orange)
C_MC = "0.62"  # nonlinear Monte Carlo (neutral grey)
C_TRUTH = "k"  # truth
C_SPAN = "0.93"  # background spans (observation arc, burn-in, +/-3 sigma)
C_XYZ = ("#56B4E9", "#CC79A7", "#E69F00")  # x / y / z components (ACF figure only)
C_PRIOR = "#CC79A7"  # hyperprior reference (corner + traces; dashed, reddish purple)
C_LSQ = "#009E73"  # batch LSQ Gaussian in the GMM propagation figures (dash-dot, bluish green)

# ------------------------------------------------------------
# Per-script configuration (set by configure())
# ------------------------------------------------------------
PARAM_SYM = []  # LaTeX symbol of each estimated parameter (no $)
PARAM_UNIT = []  # display unit ("" = dimensionless, no brackets)
PARAM_SCALE = np.ones(0)  # internal units -> display units
BODY = "Bennu"  # central body (occultation labels)
TARGET = "Particle"  # tracked object (legend / axis labels)
OBS_WORD = "Image"  # one measurement epoch: "Image Used", "Image Occulted"
SITE_LABEL = "Ejection Site"  # marker of the detachment point in the scene
RESID_LABELS = ("RA", "Dec")  # measurement components (stacked pairs in the residuals)
_ROTATION = None  # R_ib(alpha, delta, omega, t, w0): inertial -> body-fixed
FALLBACK_DIR = Path.home() / "figures_fallback"  # used if the figure folder cannot be written


def configure(
    param_sym,
    param_unit,
    param_scale,
    rotation=None,
    body=None,
    target=None,
    obs_word=None,
    site_label=None,
    resid_labels=None,
    fallback_dir=None,
):
    r"""
    Describe the estimated parameters and the scenario, and apply the journal style.

    Must be called before any journal figure: the labels, units and display scales of every
    figure come from here. The scenario arguments left at None keep their current value.

    Parameters
    ----------
    param_sym : sequence of str
        LaTeX symbol of each estimated parameter, without the $ (e.g. r"\delta x_0").
    param_unit : sequence of str
        Display unit of each parameter; "" for a dimensionless one (no brackets).
    param_scale : array_like, shape (ndim,)
        Factor from internal to displayed units (e.g. 1e3 for km -> m).
    rotation : callable, optional
        R_ib(alpha, delta, omega, t, w0): inertial -> body-fixed rotation matrix (3, 3),
        needed by plot_scene_body_fixed and points_inside_mesh.
    body : str, optional
        Name of the central body in the occultation labels (default "Bennu").
    target : str, optional
        Name of the tracked object in the legends (default "Particle").
    obs_word : str, optional
        Name of one measurement in the legends (default "Image").
    site_label : str, optional
        Legend entry of the detachment / ejection site (default "Ejection Site").
    resid_labels : sequence of str, optional
        Names of the stacked measurement components of one epoch (default ("RA", "Dec")).
    fallback_dir : str or Path, optional
        Local folder used by write_output when the figure folder cannot be written
        (default ~/figures_fallback).

    Raises
    ------
    ValueError
        If param_sym, param_unit and param_scale differ in length.
    """
    global PARAM_SYM, PARAM_UNIT, PARAM_SCALE, BODY, TARGET, OBS_WORD, SITE_LABEL
    global RESID_LABELS, _ROTATION, FALLBACK_DIR
    PARAM_SYM = list(param_sym)
    PARAM_UNIT = list(param_unit)
    PARAM_SCALE = np.asarray(param_scale, dtype=float)
    if not len(PARAM_SYM) == len(PARAM_UNIT) == PARAM_SCALE.size:
        raise ValueError("param_sym, param_unit and param_scale must have the same length")
    if rotation is not None:
        _ROTATION = rotation
    if body is not None:
        BODY = body
    if target is not None:
        TARGET = target
    if obs_word is not None:
        OBS_WORD = obs_word
    if site_label is not None:
        SITE_LABEL = site_label
    if resid_labels is not None:
        RESID_LABELS = tuple(resid_labels)
    if fallback_dir is not None:
        FALLBACK_DIR = Path(fallback_dir)
    apply_style()


def _pow10_label(k, unit):
    """Unit label for a quantity displayed in units of 10^k [unit]."""
    if k == 0:
        return unit
    return rf"$10^{{{k}}}$ {unit}".strip()


def auto_param_scales(base_scale, base_unit, sigma, lo=0.05, hi=50.0):
    """
    Display scale and unit per parameter so that the 1-sigma is a readable number.

    A parameter whose 1-sigma in the natural display unit, v = sigma * base_scale, lies in
    [lo, hi) keeps that unit (as do v = 0 and non-finite v). The others are shown in units
    of 10^k [unit], k = floor(log10 v); e.g. an SH coefficient with sigma ~ 3e-16 is
    plotted in units of 10^-16.

    Parameters
    ----------
    base_scale : array_like, shape (ndim,)
        Internal -> natural display unit (e.g. 1e3 for km -> m).
    base_unit : sequence of str
        Natural display unit of each parameter (e.g. "m").
    sigma : array_like, shape (ndim,)
        1-sigma in internal units (e.g. from the batch covariance; the sign is ignored).
    lo, hi : float
        Range of v kept in the natural unit.

    Returns
    -------
    scale : ndarray, shape (ndim,)
        base_scale * 10^-k, for configure(param_scale=...).
    unit : list of str
        base_unit, or "$10^{k}$ base_unit" when k != 0, for configure(param_unit=...).
    """
    base_scale = np.asarray(base_scale, dtype=float)
    sigma = np.abs(np.asarray(sigma, dtype=float))
    scale, unit = [], []
    for b, u, s in zip(base_scale, base_unit, sigma):
        v = s * b
        k = 0 if (not np.isfinite(v) or v == 0.0 or lo <= v < hi) else int(np.floor(np.log10(v)))
        scale.append(b * 10.0 ** (-k))
        unit.append(_pow10_label(k, u))
    return np.array(scale), unit

# ============================================================
# FIGURE OUTPUT
# ============================================================
# Every figure is saved as <name>.pdf in the folder set by set_figure_dir() (each script sets
# results/<run>_<timestamp>/; MCMCModel.save_chain writes chain.npz there too). Colors, units
# and symbols come from the style block above, so the same quantity looks the same in every
# figure.

_FIG_DIR = Path("results")


def set_figure_dir(path):
    """
    Set the folder where save_figure() writes the PDFs (created if needed).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Path
        The folder.
    """
    global _FIG_DIR
    _FIG_DIR = Path(path)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    return _FIG_DIR


def figure_dir():
    """Current figure folder (set_figure_dir)."""
    return _FIG_DIR


def _write_bytes_retry(fname, data, tries=6, wait=2.0):
    """Write `data` to `fname`, retrying with backoff (cloud-synced folders can time out)."""
    for k in range(tries):
        try:
            fname.parent.mkdir(parents=True, exist_ok=True)
            tmp = fname.with_name(f".{fname.name}.part")
            tmp.write_bytes(data)
            os.replace(tmp, fname)
            return True
        except OSError as err:
            print(
                f"[Figure] write of {fname.name} failed ({err}); retry {k + 1}/{tries}"
            )
            time.sleep(wait * 2**k)
    return False


def write_output(fname, data, tag="Figure"):
    """
    Write bytes to a file, robust to a slow cloud-synced (OneDrive) folder.

    Writes atomically (fname.part, then rename) with up to 6 tries and exponential backoff.
    If every try fails, the bytes go to FALLBACK_DIR/<folder name>/<file name> instead (2
    tries) and a warning is printed. If that fails too, an error is printed and None is
    returned; nothing is raised, so a long run is not lost to a failed write.

    Parameters
    ----------
    fname : str or Path
        Target file.
    data : bytes
        File contents.
    tag : str
        Prefix of the printed messages.

    Returns
    -------
    Path or None
        fname, or the fallback copy; None if neither could be written (every failed try
        is printed).
    """
    fname = Path(fname)
    if _write_bytes_retry(fname, data):
        return fname
    local = FALLBACK_DIR / fname.parent.name / fname.name
    if _write_bytes_retry(local, data, tries=2):
        print(f"[{tag}] WARNING: could not write to {fname.parent}; saved {local} instead")
        return local
    print(
        f"[{tag}] ERROR: could not write {fname.name} to {fname.parent} "
        f"nor to {local.parent}; NOT saved"
    )
    return None


def save_figure(fig, name):
    """
    Save `fig` as <figure dir>/<name>.pdf, show it if SHOW_FIGS, then close it.

    The span labels of shade_occulted are fitted first. The PDF is rendered in memory and
    written by write_output, so a slow OneDrive write cannot crash a long run. plt.show() is
    skipped on non-interactive backends (Agg, PDF, ...).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File name without the .pdf extension.

    Returns
    -------
    Path or None
        The PDF written (see write_output); None if it could not be written.
    """
    _fit_span_labels(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    fname = write_output(_FIG_DIR / f"{name}.pdf", buf.getvalue())
    if fname is not None:
        print(f"[Figure] saved {fname}")
    interactive = plt.get_backend().lower() not in {
        "agg",
        "pdf",
        "svg",
        "ps",
        "cairo",
        "pgf",
        "template",
    }
    if SHOW_FIGS and interactive:
        plt.show()
    plt.close(fig)
    return fname


def param_label(i, unit=True, sep=" "):
    r"""
    Axis label of estimated parameter i, e.g. '$\delta x_0$ [m]'.

    Parameters
    ----------
    i : int
        Parameter index (into PARAM_SYM / PARAM_UNIT).
    unit : bool
        Append the unit in brackets (never for a dimensionless parameter).
    sep : str
        Between symbol and unit; "\n" puts the unit on a second line.

    Returns
    -------
    str
    """
    s = f"${PARAM_SYM[i]}$"
    return f"{s}{sep}[{PARAM_UNIT[i]}]" if (unit and PARAM_UNIT[i]) else s


def _wrap_label(sym, unit, max_w, width, unit_line=False):
    r"""
    Axis label '$sym$ [unit]' as lines no wider than max_w, as measured by width(text).

    Breaks, where needed, between symbol and unit (always if unit_line, with [-] for a
    dimensionless parameter), after a leading \log_{..} of the symbol, and at the spaces of
    the unit outside math mode. A piece that cannot be broken keeps a line of its own.
    """
    u = f"[{unit or '-'}]" if (unit or unit_line) else ""
    one = f"${sym}$ {u}".rstrip()
    if not unit_line and width(one) <= max_w:
        return one
    m = re.match(r"\s*(\\log_\{[^}]*\}|\\log_\w|\\log|\\ln)\s*(\S.*)$", sym)
    groups = [(list(m.groups()) if m else [sym], "${}$")]
    if u:
        parts, cur, in_math = [], "", False
        for c in u:
            in_math ^= c == "$"
            if c == " " and not in_math:
                parts.append(cur)
                cur = ""
            else:
                cur += c
        groups.append((parts + [cur], "{}"))
    lines = []
    for parts, fmt in groups:
        cur = []
        for part in parts:
            if cur and width(fmt.format(" ".join(cur + [part]))) > max_w:
                lines.append(fmt.format(" ".join(cur)))
                cur = []
            cur.append(part)
        lines.append(fmt.format(" ".join(cur)))
    return "\n".join(lines)


def _tint(color, a):
    """Mix `color` with white: a = 0 -> white, a = 1 -> color."""
    c = np.asarray(mcolors.to_rgb(color))
    return tuple((1.0 - a) + a * c)


OCC_FILL = _tint(C_OCC, 0.2)  # opaque fill of the occulted epochs


def _occulted_spans(t, vis):
    """(start, end) of each run of occulted epochs, edges at the midpoints between epochs."""
    t = np.asarray(t, dtype=float)
    occ = ~np.asarray(vis, dtype=bool)
    if not np.any(occ):
        return []
    dt = np.diff(t)
    left = np.r_[t[0] - 0.5 * dt[0], t[1:] - 0.5 * dt]
    right = np.r_[t[:-1] + 0.5 * dt, t[-1] + 0.5 * dt[-1]]
    edges = np.diff(np.r_[0, occ.astype(int), 0])
    i0 = np.flatnonzero(edges == 1)
    i1 = np.flatnonzero(edges == -1) - 1
    return [(left[a], right[b]) for a, b in zip(i0, i1)]


def shade_occulted(ax, t, vis, text=None, text_y=0.97):
    """
    Shade the occulted epochs of a time axis, optionally with a label in the first span.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    t : array_like, shape (N,)
        Epochs, in the units of the x axis.
    vis : array_like of bool, shape (N,)
        True where the measurement is usable; each run of False epochs is shaded, edges at
        the midpoints to the neighbouring epochs.
    text : str, optional
        Label centred in the first span; save_figure shrinks it to fit the span.
    text_y : float
        Height of the label in axes fraction (top-aligned above 0.5).

    Returns
    -------
    list of (float, float)
        The shaded (start, end) spans.
    """
    spans = _occulted_spans(t, vis)
    for a, b in spans:
        ax.axvspan(a, b, color=OCC_FILL, lw=0, zorder=0.4)
    if text and spans:
        a, b = spans[0]
        lo, hi = ax.get_xlim()
        label = ax.text(
            0.5 * (max(a, lo) + min(b, hi)),
            text_y,
            text,
            color="0.3",
            fontsize=7,
            ha="center",
            va="top" if text_y > 0.5 else "bottom",
            transform=ax.get_xaxis_transform(),
            zorder=5,
        )
        label._jf_span = (a, b)  # fitted into the span at save time (_fit_span_labels)
    return spans


def _fit_span_labels(fig):
    """
    Fit the span labels of shade_occulted once the layout is final: re-centre each label on
    the visible part of its span, turn it vertical (on one line) when it is wider than the
    span, and drop it when even the vertical label does not fit.
    """
    labels = [
        (ax, t) for ax in fig.axes for t in ax.texts if getattr(t, "_jf_span", None)
    ]
    if not labels:
        return
    fig.draw_without_rendering()
    for ax, t in labels:
        a, b = t._jf_span
        lo, hi = sorted(ax.get_xlim())
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            t.set_visible(False)
            continue
        t.set_x(0.5 * (a + b))
        x0, x1 = ax.transData.transform([[a, 0.0], [b, 0.0]])[:, 0]
        room = 0.9 * abs(x1 - x0)
        if t.get_window_extent().width <= room:
            continue
        t.set_text(t.get_text().replace("\n", " "))
        t.set_rotation(90)
        if t.get_window_extent().width > room:
            t.set_visible(False)


# ------------------------------------------------------------
# Scene (body-fixed frame)
# ------------------------------------------------------------


def _view_dir(elev, azim):
    """Unit vector from the scene toward the camera for ax.view_init(elev, azim)."""
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _shaded_mesh(mesh, view, light, base=(0.74, 0.71, 0.67), ambient=0.30):
    """Lambert-shaded Poly3DCollection of the front-facing triangles of `mesh`."""
    nrm = np.asarray(mesh.face_normals)
    front = nrm @ view > 0.0  # closed surface: back faces are never visible
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[front]]
    shade = ambient + (1.0 - ambient) * np.clip(nrm[front] @ light, 0.0, None)
    fc = np.clip(np.outer(shade, base), 0.0, 1.0)
    coll = Poly3DCollection(
        tri, facecolors=fc, edgecolors=fc, linewidths=0.15, antialiased=False, zorder=1
    )
    coll.set_rasterized(True)
    return coll


def _hidden_by_mesh(mesh, pts, view, eps=1e-6):
    """True where a point is hidden from the (orthographic) camera by the mesh."""
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    dirs = np.repeat(view[None, :], pts.shape[0], axis=0)
    try:
        return np.asarray(mesh.ray.intersects_any(pts + eps * view, dirs), dtype=bool)
    except ImportError:  # no rtree in this environment
        return _ray_hit_counts(mesh.vertices, mesh.faces, pts + eps * view, view) > 0


def _to_body(pos_i, tau, alpha, delta, omega, w0=0.0):
    """Rotate inertial positions (N,3) at times tau into the body-fixed frame."""
    if _ROTATION is None:
        raise RuntimeError("configure(rotation=...) was not called")
    return np.array(
        [
            np.asarray(_ROTATION(alpha, delta, omega, float(t), w0))
            @ p
            for p, t in zip(
                np.asarray(pos_i, dtype=float), np.asarray(tau, dtype=float)
            )
        ]
    )


def _ray_hit_counts(vertices, faces, origins, direction, chunk=128):
    """
    Number of mesh faces hit by the rays origins + t*direction, t > 0 (Moller-Trumbore against
    every face, one shared direction). Numpy fallback for trimesh's ray queries, which need
    rtree (or embree).
    """
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces)
    a = V[F[:, 0]]
    e1 = V[F[:, 1]] - a
    e2 = V[F[:, 2]] - a
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    h = np.cross(d, e2)
    det = np.einsum("fi,fi->f", e1, h)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    pts = np.atleast_2d(np.asarray(origins, dtype=float))
    out = np.zeros(len(pts), dtype=int)
    for i0 in range(0, len(pts), chunk):
        s = pts[i0 : i0 + chunk, None, :] - a[None]  # (P, F, 3)
        u = np.einsum("pfi,fi->pf", s, h) * inv
        q = np.cross(s, e1[None])
        v = (q @ d) * inv
        t = np.einsum("pfi,fi->pf", q, e2) * inv
        hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
        out[i0 : i0 + chunk] = hit.sum(axis=1)
    return out


def _contains_raycast(vertices, faces, pts):
    """Inside test by ray parity, for a watertight mesh (fallback for mesh.contains)."""
    d = np.array([0.5773, 0.5779, 0.5768])  # generic direction: no edge/vertex grazing
    return _ray_hit_counts(vertices, faces, pts, d) % 2 == 1


def points_inside_mesh(mesh, pos_i, tau, alpha, delta, omega, w0=0.0):
    """
    True where an inertial position at time tau lies inside the body-fixed mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Watertight shape model in the body-fixed frame [km].
    pos_i : array_like, shape (N, 3)
        Inertial positions [km].
    tau : array_like, shape (N,)
        Epochs [s].
    alpha, delta, omega, w0 : float
        Arguments of the rotation set by configure(rotation=...): pole right ascension and
        declination [rad], spin rate [rad/s] and rotation angle at t = 0 [rad].

    Returns
    -------
    ndarray of bool, shape (N,)

    Raises
    ------
    RuntimeError
        If configure(rotation=...) was not called.

    Notes
    -----
    Uses mesh.contains, or a ray-parity fallback (_contains_raycast) when rtree is missing.
    """
    pb = _to_body(pos_i, tau, alpha, delta, omega, w0)
    try:
        return np.asarray(mesh.contains(pb), dtype=bool)
    except ImportError:  # no rtree in this environment
        return _contains_raycast(mesh.vertices, mesh.faces, pb)


def plot_scene_body_fixed(
    mesh,
    tau_obs,
    sc_obs,
    vis_mask,
    tau_line,
    sc_line,
    pt_line,
    alpha,
    delta,
    omega,
    w0=0.0,
    detach_bf=None,
    elev=14.0,
    azim=62.0,
    zoom=1.0,
    name="fig_scene_body_fixed",
):
    """
    Observation geometry in the body-fixed frame (single column).

    Shaded shape model, spacecraft track (green) and tracked-object truth track (black)
    over the arc, and the measurement epochs on the spacecraft track: used (dots) and
    occulted (crosses). Orthographic view; track points and markers hidden behind the body
    are not drawn. The axes are an equal-aspect cube around the mesh and the tracks.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Shape model in the body-fixed frame [km].
    tau_obs : array_like, shape (N,)
        Measurement epochs [s].
    sc_obs : array_like, shape (N, >=3)
        Inertial spacecraft states at tau_obs [km] (first 3 columns used).
    vis_mask : array_like of bool, shape (N,)
        True for used measurements, False for occulted ones.
    tau_line : array_like, shape (M,)
        Dense epochs of the track lines [s].
    sc_line, pt_line : array_like, shape (M, >=3)
        Inertial spacecraft and tracked-object (truth) states at tau_line [km].
    alpha, delta, omega, w0 : float
        Rotation arguments (see points_inside_mesh).
    detach_bf : array_like, shape (3,), optional
        Detachment / ejection site in the body-fixed frame [km]: a star (legend entry
        SITE_LABEL) if visible.
    elev, azim : float
        View angles [deg].
    zoom : float
        Box-aspect zoom; keep <= 1 or the axis labels get clipped.
    name : str
        File name of the PDF.

    Returns
    -------
    Path
        The saved PDF.
    """
    view = _view_dir(elev, azim)
    light = _view_dir(elev + 30.0, azim + 35.0)  # key light above-left of the camera

    sc_b = _to_body(sc_line[:, :3], tau_line, alpha, delta, omega, w0)
    pt_b = _to_body(pt_line[:, :3], tau_line, alpha, delta, omega, w0)
    obs_b = _to_body(sc_obs[:, :3], tau_obs, alpha, delta, omega, w0)
    vis = np.asarray(vis_mask, dtype=bool)

    def _visible_line(xyz):
        """Copy of xyz with the points hidden by the mesh set to NaN."""
        out = xyz.copy()
        out[_hidden_by_mesh(mesh, xyz, view)] = np.nan
        return out

    sc_v, pt_v = _visible_line(sc_b), _visible_line(pt_b)
    obs_hidden = _hidden_by_mesh(mesh, obs_b, view)

    fig = plt.figure(figsize=(COL_W, 2.9))
    ax = fig.add_axes((0.0, -0.03, 1.0, 0.96), projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)

    ax.add_collection3d(_shaded_mesh(mesh, view, light))
    ax.plot(*sc_v.T, color=C_SC, lw=1.1, zorder=3)
    ax.plot(*pt_v.T, color=C_TRUTH, lw=1.1, zorder=3)

    used = vis & ~obs_hidden
    occl = ~vis & ~obs_hidden
    ax.plot(
        *obs_b[used].T,
        ls="none",
        marker="o",
        ms=2.8,
        mfc=C_SC,
        mec="white",
        mew=0.35,
        zorder=4,
    )
    ax.plot(
        *obs_b[occl].T, ls="none", marker="x", ms=3.2, color=C_OCC, mew=0.9, zorder=4
    )
    if (
        detach_bf is not None
        and not _hidden_by_mesh(mesh, detach_bf[None], view)[0]
    ):
        ax.plot(
            *np.asarray(detach_bf)[:, None],
            ls="none",
            marker="*",
            ms=7.5,
            mfc=C_TRUTH,
            mec="white",
            mew=0.5,
            zorder=5,
        )

    # Equal-aspect cube around everything drawn
    xyz = np.vstack([np.asarray(mesh.vertices), sc_b, pt_b])
    mid = 0.5 * (xyz.min(0) + xyz.max(0))
    half = 0.5 * np.max(xyz.max(0) - xyz.min(0)) * 1.02
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)

    for a, lab in zip(
        (ax.xaxis, ax.yaxis, ax.zaxis),
        (
            r"$X_{\mathscr{B}}$ [km]",
            r"$Y_{\mathscr{B}}$ [km]",
            r"$Z_{\mathscr{B}}$ [km]",
        ),
    ):
        a.set_pane_color((1.0, 1.0, 1.0, 0.0))
        a.line.set_color("0.4")
        a.line.set_linewidth(0.5)
        a._axinfo["grid"].update(color="0.88", linewidth=0.4)
        a._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.25)
        a.set_major_locator(mticker.MaxNLocator(5))
        a.set_tick_params(pad=-1.5, labelsize=6.5)
        a.set_label_text(lab)
        a.labelpad = -3.0

    handles = [
        Line2D([], [], color=C_SC, lw=1.1, label="Spacecraft"),
        Line2D([], [], color=C_TRUTH, lw=1.1, label=f"{TARGET} (Truth)"),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=3.5,
            mfc=C_SC,
            mec="white",
            mew=0.35,
            label=f"{OBS_WORD} Used",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="x",
            ms=3.8,
            color=C_OCC,
            mew=0.9,
            label=f"{OBS_WORD} Occulted",
        ),
    ]
    if detach_bf is not None:
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="*",
                ms=7,
                mfc=C_TRUTH,
                mec="white",
                mew=0.5,
                label=SITE_LABEL,
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.0,
        handletextpad=0.4,
    )
    return save_figure(fig, name)


# ------------------------------------------------------------
# Observation geometry and post-fit residuals (time axis)
# ------------------------------------------------------------


def plot_observation_geometry(
    tau_obs,
    sc_obs,
    pt_obs,
    vis_mask,
    tau_line=None,
    sc_line=None,
    pt_line=None,
    name="fig_observation_geometry",
):
    """
    Spacecraft-target range over the arc, with the measurement epochs (single column).

    The range |pt - sc| is drawn as a grey truth line and at the measurement epochs as
    used (dots) and occulted (crosses) markers, with their counts in the legend; the
    occulted epochs are shaded and labelled "Occulted by BODY".

    Parameters
    ----------
    tau_obs : array_like, shape (N,)
        Measurement epochs [s] (plotted in hours).
    sc_obs, pt_obs : array_like, shape (N, >=3)
        Inertial spacecraft and target states at tau_obs [km] (first 3 columns used).
    vis_mask : array_like of bool, shape (N,)
        True for used measurements.
    tau_line, sc_line, pt_line : array_like, optional
        Dense epochs [s] and states [km] of the range line; no line if omitted.
    name : str
        File name of the PDF.

    Returns
    -------
    Path
        The saved PDF.
    """
    t = np.asarray(tau_obs) / 3600.0
    vis = np.asarray(vis_mask, dtype=bool)
    rho = np.linalg.norm(np.asarray(pt_obs)[:, :3] - np.asarray(sc_obs)[:, :3], axis=1)

    fig, ax = plt.subplots(figsize=(COL_W, 1.95), layout="constrained")
    ax.set_xlim(t[0], t[-1])
    shade_occulted(ax, t, vis, text=f"Occulted\nby {BODY}", text_y=0.04)
    if tau_line is not None:
        rho_l = np.linalg.norm(
            np.asarray(pt_line)[:, :3] - np.asarray(sc_line)[:, :3], axis=1
        )
        ax.plot(np.asarray(tau_line) / 3600.0, rho_l, color="0.25", lw=0.9, zorder=2)
    ax.plot(
        t[vis],
        rho[vis],
        ls="none",
        marker="o",
        ms=3.4,
        mfc=C_SC,
        mec="white",
        mew=0.4,
        zorder=3,
    )
    ax.plot(
        t[~vis],
        rho[~vis],
        ls="none",
        marker="x",
        ms=3.6,
        color=C_OCC,
        mew=1.0,
        zorder=3,
    )
    ax.set_xlabel("Time since Epoch [h]")
    # The axis label stays short so it fits the 1.95 in panel; the legend names the pair.
    ax.set_ylabel("Range [km]")
    rho_top = max(rho.max(), rho_l.max() if tau_line is not None else 0.0)
    ax.set_ylim(0.0, 1.12 * rho_top)  # headroom: markers never sit on the frame
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    handles = [
        Line2D([], [], color="0.25", lw=0.9, label=f"S/C{EN_DASH}{TARGET} Range (Truth)"),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=3.4,
            mfc=C_SC,
            mec="white",
            mew=0.4,
            label=f"{OBS_WORD} Used ({int(vis.sum())})",
        ),
        (
            Patch(facecolor=OCC_FILL, lw=0),
            Line2D([], [], ls="none", marker="x", ms=3.6, color=C_OCC, mew=1.0),
        ),
    ]
    labels = [h.get_label() for h in handles[:2]] + [
        f"{OBS_WORD} Occulted ({int((~vis).sum())})"
    ]
    ax.legend(handles, labels, loc="lower right", ncol=1)
    return save_figure(fig, name)


def plot_postfit_residuals(
    tau_obs, res_batch, res_mcmc, vis_mask, components=None, name="fig_postfit_residuals"
):
    """
    Normalized post-fit residuals at the used epochs, one panel per measurement component.

    Batch (Gauss-Newton) residuals are open vermillion circles, MCMC MAP residuals blue
    dots, with grey +/-1 sigma and +/-3 sigma bands and the RMS of each set in the panel.

    Parameters
    ----------
    tau_obs : array_like, shape (N,)
        All measurement epochs [s] (plotted in hours).
    res_batch : array_like, shape (N * n_comp,)
        Batch residuals divided by sigma, stacked per epoch (e.g. [ra0, dec0, ra1, ...]),
        with the component names in RESID_LABELS; one entry per epoch of tau_obs.
    res_mcmc : array_like, shape (N * n_comp,) or None
        Same for the MCMC MAP. None draws the batch alone (Stage-1 check, before any MCMC).
    vis_mask : array_like of bool, shape (N,)
        Epochs plotted (used measurements).
    components : list of int, optional
        Panels shown (indices into RESID_LABELS; default all), e.g. [1] to show range-rate
        only when range carries no information.
    name : str
        File name of the PDF.

    Returns
    -------
    Path
        The saved PDF.
    """
    t = np.asarray(tau_obs) / 3600.0
    vis = np.asarray(vis_mask, dtype=bool)
    n_comp = len(RESID_LABELS)
    comps = list(range(n_comp)) if components is None else list(components)
    rb = np.asarray(res_batch).reshape(-1, n_comp)[:, comps]
    has_mcmc = res_mcmc is not None
    rm = np.asarray(res_mcmc).reshape(-1, n_comp)[:, comps] if has_mcmc else None

    fig, axes = plt.subplots(
        len(comps),
        1,
        figsize=(COL_W, 0.75 + 1.075 * len(comps)),
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    axes = axes[:, 0]
    shown = np.r_[rb[vis].ravel(), rm[vis].ravel()] if has_mcmc else rb[vis].ravel()
    lim = max(3.6, 1.15 * float(np.max(np.abs(shown))))
    for k, (ax, comp) in enumerate(zip(axes, [RESID_LABELS[c] for c in comps])):
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-lim, lim)
        ax.axhspan(-3.0, 3.0, color="0.95", lw=0, zorder=0)
        ax.axhspan(-1.0, 1.0, color="0.89", lw=0, zorder=0)
        ax.axhline(0.0, color="0.45", lw=0.6, zorder=1)
        shade_occulted(ax, t, vis, text="Occulted" if k == 0 else None)
        ax.plot(
            t[vis],
            rb[vis, k],
            ls="none",
            marker="o",
            ms=4.2,
            mfc="none",
            mec=C_BATCH,
            mew=0.8,
            zorder=3,
        )
        if has_mcmc:
            ax.plot(
                t[vis], rm[vis, k], ls="none", marker="o", ms=1.9, color=C_POST, zorder=4
            )
        ax.set_yticks([-3, -1, 0, 1, 3])
        ax.yaxis.grid(False)
        ax.set_ylabel(rf"{comp} Residual [$\sigma$]")
        rms_b = np.sqrt(np.mean(rb[vis, k] ** 2))
        rms_txt = rf"RMS: {rms_b:.2f}"
        if has_mcmc:
            rms_txt = (
                rf"RMS: batch {rms_b:.2f}, MCMC {np.sqrt(np.mean(rm[vis, k] ** 2)):.2f}"
            )
        ax.text(
            0.99,
            0.04,
            rms_txt,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.2",
            zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85),
        )
    axes[-1].set_xlabel("Time since Epoch [h]")

    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=4.2,
            mfc="none",
            mec=C_BATCH,
            mew=0.8,
            label="Batch Least Squares",
        ),
        Line2D([], [], ls="none", marker="o", ms=2.4, color=C_POST, label="MCMC MAP"),
        Patch(facecolor="0.89", label=r"$\pm1\sigma$"),
        Patch(facecolor="0.95", label=r"$\pm3\sigma$"),
    ]
    if not has_mcmc:
        handles.pop(1)
    fig.legend(
        handles=handles, loc="outside upper center", ncol=len(handles), columnspacing=0.9
    )
    return save_figure(fig, name)


# ============================================================
# MCMC DIAGNOSTICS (chain traces, autocorrelation, corner)
# ============================================================
# The plot_mcmc_* functions take the MCMCModel; MCMCModel.plot_journal_* forward to them.


def _require_configured(n_par):
    """Fail early (with the fix) when configure() does not describe n_par parameters."""
    if len(PARAM_SYM) != n_par or len(PARAM_SCALE) != n_par:
        raise RuntimeError(
            f"configure() describes {len(PARAM_SYM)} parameters but the "
            f"chain has {n_par}: call configure(param_sym, param_unit, param_scale) first"
        )


def _chain_physical(model, discard=0, thin=1, transform=None):
    """
    emcee chain in model coordinates (unwhitened), optionally transformed for display.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run (or with a ChainReplay sampler).
    discard, thin : int
        As in emcee's get_chain.
    transform : callable, optional
        Maps an (n, ndim) array of model coordinates to displayed coordinates (e.g. a
        hierarchical reparametrization); its output width must match configure().

    Returns
    -------
    ndarray, shape (n_steps, n_walkers, n_par)

    Raises
    ------
    RuntimeError
        If configure() does not describe n_par parameters (_require_configured).
    """
    ch = np.asarray(model.sampler.get_chain(discard=discard, thin=thin), dtype=float)
    if getattr(model, "is_whitened", False):
        ch = ch @ np.asarray(model.whiten_L).T + np.asarray(model.whiten_mean)
    if transform is not None:
        ch = np.asarray(transform(ch), dtype=float)
    _require_configured(ch.shape[-1])
    return ch


def plot_mcmc_traces(
    model,
    burn_in,
    true_theta=None,
    batch_theta=None,
    n_walkers_show=24,
    max_points=1500,
    seed=0,
    transform=None,
    logp_truth_theta=None,
    name="fig_mcmc_traces",
    prior_idx=None,
    prior_label="Hyperprior Mean",
    prior_truth_label=None,
):
    """
    Walker traces of every parameter and of the log-posterior over the whole run.

    Each panel shows a random subset of walkers (grey), the ensemble mean with its +/-3
    sigma band (blue), the truth (black) and the batch estimate (dashed vermillion); the
    burn-in is shaded and the y-limits follow the post-burn-in ensemble. The chain is
    thinned to about max_points steps for drawing. Two columns of panels (three above 14
    panels), log p spanning the rest of the bottom row.

    The log-posterior is not Gaussian across walkers: near the mode
    log p = log p_max - chi2_d / 2, capped above with a long lower tail. A symmetric
    mean +/- 3 sigma band would sit partly above log p_max, where no walker can go, and miss
    the lower tail, so its band is the equal-tailed 99.73 % interval (same mass as +/-3
    sigma).

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run (or with a ChainReplay sampler).
    burn_in : int
        Burn-in length [steps] (shaded; excluded from the y-limits and the bands).
    true_theta, batch_theta : array_like, shape (n_par,), optional
        Truth and batch estimate in displayed coordinates (internal units); no line and no
        legend entry when None.
    n_walkers_show : int
        Number of walkers drawn.
    max_points : int
        Approximate number of steps drawn per walker.
    seed : int
        Seed of the walker subset.
    transform : callable, optional
        Model -> displayed coordinates (see _chain_physical).
    logp_truth_theta : array_like, shape (ndim,), optional
        Truth in model coordinates (the input of model.log_prob), for the log p(truth) line.
        Defaults to true_theta when no transform is used; no line if log p is not finite.
    name : str
        File name of the PDF.
    prior_idx : sequence of int, optional
        Parameters the batch does not estimate (e.g. hierarchical ln sigma_g), whose
        batch_theta entry holds their (hyper)prior mean instead: drawn dashed reddish purple
        with its own legend entry.
    prior_label : str
        Legend entry of that prior mean.
    prior_truth_label : str, optional
        If given, true_theta of the prior_idx parameters is not a generative value (e.g. the
        realized RMS of the group deviations): drawn dotted with this legend entry instead
        of as the truth.

    Returns
    -------
    Path
        The saved PDF.
    """
    n_steps = int(model.sampler.iteration)
    rs = int(getattr(model.sampler, "stride", 1))  # > 1: chain loaded from a file
    stride = rs * max(1, round(n_steps / max_points / rs))
    ch = _chain_physical(model, thin=stride, transform=transform) * PARAM_SCALE
    lp = np.asarray(model.sampler.get_log_prob(thin=stride), dtype=float)
    steps = np.arange(ch.shape[0]) * stride
    n_par = ch.shape[2]
    post = steps >= burn_in
    rng = np.random.default_rng(seed)
    pick = rng.choice(ch.shape[1], size=min(n_walkers_show, ch.shape[1]), replace=False)

    # parameters fill the grid row by row; log p spans what is left of the bottom row
    n_cols = 2 if n_par + 1 <= 14 else 3
    n_rows = int(np.ceil((n_par + 1) / n_cols))
    fig = plt.figure(
        figsize=(TXT_W, (1.18 if n_cols == 2 else 1.1) * n_rows + 0.45),
        layout="constrained",
    )
    gs = fig.add_gridspec(n_rows, n_cols)
    axes = []
    for i in range(n_par):
        r, c = divmod(i, n_cols)
        axes.append(fig.add_subplot(gs[r, c], sharex=axes[0] if axes else None))
    r, c = divmod(n_par, n_cols)
    axes.append(fig.add_subplot(gs[r, c:], sharex=axes[0]))
    axes = np.array(axes)

    p3 = 100.0 * norm.sf(3.0)  # one tail of +/-3 sigma (0.135 %)
    band_alpha = 0.28  # shade only, drawn over the walkers: opaque enough to read on them

    def _windows(h):
        """[i0, i1) of a +/-h window around each thinned step, never crossing the end of the
        burn-in, so the post-burn-in band uses post-burn-in samples only."""
        n, nb = len(steps), int(np.argmax(post)) if post.any() else len(steps)
        idx = np.arange(n)
        seg0 = np.where(idx < nb, 0, nb)
        seg1 = np.where(idx < nb, nb, n)
        return np.maximum(idx - h, seg0), np.minimum(idx + h + 1, seg1)

    def _band(y, skewed, min_pool=4000):
        """
        Ensemble mean per step, and a +/-3 sigma band pooled over enough neighbouring
        (thinned) steps to hold ~min_pool samples, so the band edges are smooth lines rather
        than per-step noise. For a skewed quantity (log p) the band is the equal-tailed
        99.73 % interval (same mass as +/-3 sigma), from a 4x wider pool and then a 2x wider
        running mean: a 0.135 % sample quantile rests on a handful of tail samples, so it
        otherwise wobbles as they enter and leave the pool.
        """
        n, m = y.shape
        mu = np.mean(y, axis=1)
        h = int(np.ceil(0.5 * min_pool / m))
        if skewed:
            h *= 2
        i0, i1 = _windows(h)
        if not skewed:
            c1 = np.concatenate([[0.0], np.cumsum(y.sum(axis=1))])
            c2 = np.concatenate([[0.0], np.cumsum((y**2).sum(axis=1))])
            cnt = (i1 - i0) * m
            mean_w = (c1[i1] - c1[i0]) / cnt
            sig_w = np.sqrt(np.maximum((c2[i1] - c2[i0]) / cnt - mean_w**2, 0.0))
            return mu, mean_w - 3.0 * sig_w, mean_w + 3.0 * sig_w
        j0, j1 = _windows(2 * h)
        q = np.array([np.percentile(y[a:b], [p3, 100.0 - p3]) for a, b in zip(j0, j1)])
        cq = np.vstack([np.zeros((1, 2)), np.cumsum(q, axis=0)])
        q = (cq[i1] - cq[i0]) / (i1 - i0)[:, None]
        return mu, q[:, 0], q[:, 1]

    pri = set() if prior_idx is None else {int(k) for k in prior_idx}

    realized = pri if prior_truth_label else set()  # truth drawn as a realized value

    def _panel(
        ax,
        y,
        ylabel,
        ref_true=None,
        ref_batch=None,
        skewed=False,
        ref_color=C_BATCH,
        true_ls="-",
    ):
        """
        One trace panel: burn-in span, faint walkers, band and mean on top, reference lines.
        """
        # Walkers first and faint; the band (shade only, no edge lines) and the mean go on
        # top, so the band stays visible even where the walker traces fill it completely.
        ax.axvspan(0, burn_in, color=C_SPAN, lw=0, zorder=0)
        ax.plot(
            steps,
            y[:, pick],
            color="0.72",
            lw=0.25,
            alpha=0.45,
            zorder=1,
            rasterized=True,
        )
        mu, band_lo, band_hi = _band(y, skewed)
        ax.fill_between(
            steps, band_lo, band_hi, color=C_POST, alpha=band_alpha, lw=0, zorder=2
        )
        ax.plot(steps, mu, color=C_POST, lw=1.0, zorder=3)
        if ref_batch is not None:
            ax.axhline(ref_batch, color=ref_color, lw=0.9, ls="--", zorder=4)
        if ref_true is not None:
            ax.axhline(ref_true, color=C_TRUTH, lw=0.8, ls=true_ls, zorder=5)
        yp = y[post]
        lo, hi = np.percentile(yp, [0.5, 99.5])
        refs = [r for r in (ref_true, ref_batch) if r is not None]
        lo, hi = min([lo] + refs), max([hi] + refs)
        pad = 0.35 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    for i in range(n_par):
        _panel(
            axes[i],
            ch[:, :, i],
            param_label(i),
            None if true_theta is None else true_theta[i] * PARAM_SCALE[i],
            None if batch_theta is None else batch_theta[i] * PARAM_SCALE[i],
            ref_color=C_PRIOR if i in pri else C_BATCH,
            true_ls=":" if i in realized else "-",
        )
    logp_true = None
    theta_lp = true_theta if logp_truth_theta is None else logp_truth_theta
    if theta_lp is not None and (transform is None or logp_truth_theta is not None):
        try:
            logp_true = float(model.log_prob(np.asarray(theta_lp, dtype=float)))
        except Exception:
            logp_true = None
        if logp_true is not None and not np.isfinite(logp_true):
            logp_true = None
    _panel(
        axes[n_par],
        lp,
        r"$\log p(\theta \mid y)$",
        ref_true=logp_true,
        skewed=True,
    )
    for k, ax in enumerate(axes):  # x-axis labelled on the bottom row only
        if k >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Step [-]")
        else:
            ax.tick_params(labelbottom=False)
    for ax in axes:
        ax.set_xlim(0, n_steps)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))

    handles = [
        Line2D(
            [], [], color="0.7", lw=0.8, label=f"Walkers ({len(pick)} of {ch.shape[1]})"
        ),
        (
            Patch(facecolor=mcolors.to_rgba(C_POST, band_alpha), lw=0),
            Line2D([], [], color=C_POST, lw=0.9),
        ),
        Line2D([], [], color=C_BATCH, lw=0.9, ls="--", label="Batch Least Squares"),
        Line2D([], [], color=C_TRUTH, lw=0.8, label="Truth"),
        Patch(facecolor=C_SPAN, lw=0, label="Burn-in"),
    ]
    # no batch solution (e.g. the STT benchmark) or no truth: no legend entry for it
    drop = {"Batch Least Squares"} if batch_theta is None else set()
    drop |= {"Truth"} if true_theta is None else set()
    handles = [h for h in handles if isinstance(h, tuple) or h.get_label() not in drop]
    if realized and true_theta is not None:
        handles.insert(
            4, Line2D([], [], color=C_TRUTH, lw=0.8, ls=":", label=prior_truth_label)
        )
    if pri and batch_theta is not None:
        handles.insert(
            3, Line2D([], [], color=C_PRIOR, lw=0.9, ls="--", label=prior_label)
        )
    labels = [
        "Ensemble Mean, $\\pm 3\\sigma$" if isinstance(h, tuple) else h.get_label()
        for h in handles
    ]
    ncol = len(handles) if len(handles) <= 6 else int(np.ceil(len(handles) / 2))
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol)
    return save_figure(fig, name)


# Okabe-Ito order (minus the yellow) for parameter groups beyond the 3D state
_GROUP_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "0.2")
_GROUP_LS = ("-", "--", "-.", (0, (1, 1)), (0, (5, 1, 1, 1, 1, 1)), (0, (3, 1)), "-")


def _default_groups(n_par):
    """[r, v] and then chunks of <= 7 of the remaining parameters."""
    if n_par <= 7:
        return [list(range(n_par))]
    rest = list(range(6, n_par))
    n_chunks = int(np.ceil(len(rest) / 7))
    return [list(range(6))] + [list(c) for c in np.array_split(rest, n_chunks)]


def plot_mcmc_autocorrelation(
    model, burn_in, thin, transform=None, groups=None, name="fig_mcmc_autocorrelation"
):
    """
    Walker-averaged normalized autocorrelation of every parameter after burn-in.

    Each curve is labelled with its integrated autocorrelation time tau (emcee,
    integrated_time with tol=0); a dotted line marks the thinning interval. Lags run to
    max(6 max tau, 2.5 thin), in steps of the original run (also for a ChainReplay with
    stride > 1). One group of parameters: a single-column panel. More: one panel per group,
    side by side at full text width, lettered (a), (b), ...

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run (or with a ChainReplay sampler).
    burn_in : int
        Steps discarded before the estimate.
    thin : int
        Thinning interval used for model.samples [steps] (marked).
    transform : callable, optional
        Model -> displayed coordinates (see _chain_physical).
    groups : list of list of int, optional
        Parameters of each panel (default: _default_groups, i.e. one panel up to 7
        parameters, else [r, v] and then chunks of <= 7).
    name : str
        File name of the PDF.

    Returns
    -------
    Path
        The saved PDF.
    """
    import emcee

    # a chain loaded from a file holds every rs-th step: lags and tau in steps of the run
    rs = int(getattr(model.sampler, "stride", 1))
    ch = _chain_physical(model, discard=burn_in, transform=transform)  # (n, W, d)
    n, n_w, n_par = ch.shape
    tau = rs * np.atleast_1d(emcee.autocorr.integrated_time(ch, tol=0))
    acf = np.array(
        [
            np.mean(
                [emcee.autocorr.function_1d(ch[:, k, i]) for k in range(n_w)], axis=0
            )
            for i in range(n_par)
        ]
    )
    n_lag = int(min(n - 1, max(6.0 * np.max(tau), 2.5 * thin) / rs))
    lag = rs * np.arange(n_lag + 1)
    max_lag = lag[-1]
    groups = _default_groups(n_par) if groups is None else [list(g) for g in groups]
    # first panel holds the whole [r, v] state (6-parameter panel, or 7 with e.g. eta)
    state_first = set(range(6)) <= set(groups[0])

    def _style(i, pos):
        """(color, linestyle) of parameter i at position `pos` in its group."""
        if state_first and i < 6:
            return C_XYZ[i % 3], "-" if i < 3 else "--"  # x/y/z colors; r solid, v dashed
        if state_first and n_par <= 7:  # Ryugu-like single panel: 7th parameter dark dash-dot
            return "0.2", "-."
        return _GROUP_COLORS[pos % 7], _GROUP_LS[pos % 7]

    def _draw(ax, idx, thin_text=True):
        """
        ACF curves of the parameters idx, with tau labels, zero line and thinning marker.
        """
        ax.axhline(0.0, color="0.45", lw=0.6)
        for pos, i in enumerate(idx):
            c, ls = _style(i, pos)
            ax.plot(
                lag,
                acf[i, : n_lag + 1],
                color=c,
                ls=ls,
                lw=1.1,
                label=rf"{param_label(i, unit=False)}: $\hat\tau = {tau[i]:.0f}$",
            )
        ax.axvline(thin, color="0.3", lw=0.7, ls=":")
        if thin_text:
            # Label left of the line, or right of it when the line hugs the y-axis (no room
            # in between). Vertically: above the curves, or below them when the curves still
            # pass the upper half over the label's width (e.g. thin = 10 << tau = 50).
            k_thin = min(int(round(thin / rs)), n_lag)
            right = thin < 0.08 * max_lag
            w = max(1, int(round(0.05 * max_lag / rs)))  # label width ~5 % of the axis
            win = (
                acf[idx, k_thin : min(k_thin + w, n_lag) + 1]
                if right
                else acf[idx, max(k_thin - w, 0) : k_thin + 1]
            )
            y_curve = (np.max(win) + 0.2) / 1.22  # axes fraction, ylim below
            ax.annotate(
                " Thinning",
                xy=(thin, 0.55 if y_curve < 0.5 else 0.19),
                xycoords=ax.get_xaxis_transform(),
                xytext=(1.5 if right else -1.5, 0.0),
                textcoords="offset points",
                rotation=90,
                ha="left" if right else "right",
                va="bottom",
                fontsize=7,
                color="0.3",
            )
        ax.set_xlim(0, max_lag)
        ax.set_ylim(-0.2, 1.02)
        ax.set_xlabel("Lag [-]")

    if len(groups) == 1:
        fig, ax = plt.subplots(figsize=(COL_W, 2.35), layout="constrained")
        _draw(ax, groups[0])
        ax.set_ylabel("Autocorrelation [-]")
        handles, labels = ax.get_legend_handles_labels()
        ncol = max(1, int(np.ceil(len(labels) / 2)))
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=ncol,
            fontsize=6.5,
            columnspacing=0.8,
            handlelength=2.2,
        )
        return save_figure(fig, name)

    n_g = len(groups)
    fig, axes = plt.subplots(
        1, n_g, figsize=(TXT_W, 2.75), sharey=True, layout="constrained"
    )
    fig.get_layout_engine().set(wspace=0.03)
    for g, (ax, idx) in enumerate(zip(axes, groups)):
        _draw(ax, idx, thin_text=(g == 0))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=6.5,
            columnspacing=0.8,
            handlelength=2.2,
            labelspacing=0.25,
            borderaxespad=0.0,
        )
        ax.text(  # panel letter in the (empty) upper-right corner: the legend sits on top
            0.97,
            0.95,
            f"({'abcdefgh'[g]})",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )
    axes[0].set_ylabel("Autocorrelation [-]")
    return save_figure(fig, name)


def _label_width(s, digit_w=0.063):
    """Width of a math-mode tick label as a fraction of a corner panel's width (~63 pt).
    Measured from the usetex renderer at 6.5 pt: digit 3.96 pt, '.' 0.62 digit, minus sign
    1.57 digit."""
    return digit_w * sum(0.62 if ch == "." else 1.57 if ch == "-" else 1.0 for ch in s)


def _label_overhang(ticks, lo, hi, digit_w):
    """How far (axis fraction) the first / last tick label reaches past the left / right
    axis end (negative: room left inside the axis)."""
    span = hi - lo
    d = _tick_decimals(ticks, span)
    w0, w1 = (_label_width(_num(v, d), digit_w) for v in (ticks[0], ticks[-1]))
    return 0.5 * w0 - (ticks[0] - lo) / span, 0.5 * w1 - (hi - ticks[-1]) / span


_TICK_RANK = {1: 0, 2: 0, 5: 0, 2.5: 1, 4: 2, 8: 2, 3: 3, 6: 3}  # tick multipliers, lower = rounder


def _tick_options(lo, hi, margin=0.1, min_gap=None, digit_w=0.063):
    """
    Candidate tick sets: 2-3 round ticks (multiples of {1, 2, 2.5, 3, 4, 5, 6, 8} x 10^k) in
    [lo, hi], at least `margin` (axis fraction) from the edges and spaced so the blank space
    between adjacent labels is at least `min_gap` (axis fraction, default 0.8 digit) in a
    corner panel. Every three-tick set also offers its outer pair (e.g. -0.1, 0.1), for when
    three labels crowd, and its two adjacent pairs (e.g. 0, 500), for when an end label would
    meet the next panel's (_row_ticks). digit_w: width of one digit as a fraction of the
    panel width.
    Returns [(n ticks, roundness rank, min spacing / span, min label gap, ticks)].
    """
    min_gap = 0.8 * digit_w if min_gap is None else min_gap
    span = hi - lo
    a, b, mid = lo + margin * span, hi - margin * span, 0.5 * (lo + hi)
    e = np.floor(np.log10(span))
    out = []
    for k in np.arange(e - 2, e + 1):
        for m, rnd in _TICK_RANK.items():
            step = m * 10.0**k
            c = step * np.arange(np.ceil(a / step), np.floor(b / step) + 1)
            if c.size < 2:
                continue
            if c.size > 3:  # keep three, centred on the middle of the axis
                i = int(np.clip(np.argmin(np.abs(c - mid)) - 1, 0, c.size - 3))
                c = c[i : i + 3]
            for cand in (c, c[[0, -1]], c[:2], c[1:]) if c.size == 3 else (c,):
                d = _tick_decimals(cand, span)
                w = np.array([_label_width(_num(v, d), digit_w) for v in cand])
                sp = np.diff(cand)
                g = np.min(sp / span - 0.5 * (w[:-1] + w[1:]))
                if sp.min() >= 0.2 * span and g >= min_gap:
                    out.append((cand.size, rnd, sp.min() / span, g, cand))
    return out


def _single_tick(lo, hi, margin=0.1):
    """One tick, as round as possible, nearest the middle of [lo, hi]."""
    span = hi - lo
    a, b, mid = lo + margin * span, hi - margin * span, 0.5 * (lo + hi)
    e = np.floor(np.log10(span))
    steps = sorted(m * 10.0**k for k in np.arange(e - 2, e + 1) for m in _TICK_RANK)
    for step in reversed(steps):
        c = step * np.arange(np.ceil(a / step), np.floor(b / step) + 1)
        if c.size:
            return c[[np.argmin(np.abs(c - mid))]]
    return np.array([mid])


def _inner_ticks(lo, hi, margin=0.1, min_gap=None, digit_w=0.063):
    """
    2-3 round ticks in [lo, hi] (see _tick_options). Prefers 3 ticks over 2 (else 1), then
    the roundest multiplier, then the widest spacing. digit_w: width of one digit as a
    fraction of the panel width (default: 6.5 pt labels on a ~63 pt panel).
    """
    opts = _tick_options(lo, hi, margin, min_gap, digit_w)
    if opts:
        return max(opts, key=lambda o: (o[0], -o[1], o[2]))[4]
    return _single_tick(lo, hi, margin)


def _row_ticks(lo, hi, digit_w, wsp, gap, min_gap, widen=1.2, edge_w=None):
    """
    Ticks for corner axes whose labels sit side by side along the bottom row, so the label
    at the end of one axis never runs into the facing label of the next panel (e.g. "20"
    against "-200", or -500 against -500 when a prior widens the axes).
    Per axis: a tick set (_tick_options, labels at least `min_gap` apart) and a widening of
    the axis about its centre (up to `widen`, in 2.5% steps), such that the facing labels
    of consecutive panels keep `gap` between them (axis fractions; panels `wsp` apart).
    Dynamic programming over the row minimises, in order: single-tick axes, total widening,
    labels closer than 0.8 digit, then the _inner_ticks preferences.
    edge_w: digit width for the facing-label check (default digit_w), e.g. the measured one
    while the tick sets keep the calibrated digit_w of _inner_ticks.
    Returns ticks, lo, hi (the widened ranges).
    """
    edge_w = digit_w if edge_w is None else edge_w
    n_step = int(round((widen - 1.0) / 0.025))
    loose = 0.8 * digit_w
    rows, backs, tot, R_prev = [], [], None, None
    for k in range(len(lo)):
        mid, half = 0.5 * (lo[k] + hi[k]), 0.5 * (hi[k] - lo[k])
        opts = [(1e13, _single_tick(lo[k], hi[k]), lo[k], hi[k])]
        for m in range(n_step + 1):
            a, b = mid - (1.0 + 0.025 * m) * half, mid + (1.0 + 0.025 * m) * half
            for size, rnd, sp, g, tk in _tick_options(a, b, min_gap=min_gap, digit_w=digit_w):
                cost = 1e9 * m + 1e7 * (g < loose) - 1e4 * size + 1e2 * rnd - sp
                opts.append((cost, tk, a, b))
        cost = np.array([o[0] for o in opts])
        LR = np.array([_label_overhang(o[1], o[2], o[3], edge_w) for o in opts])
        if tot is None:
            tot, back = cost, None
        else:
            c = np.where(R_prev[:, None] + LR[None, :, 0] <= wsp - gap, tot[:, None], np.inf)
            back = np.argmin(c, axis=0)
            tot = cost + c[back, np.arange(len(opts))]
        rows.append(opts)
        backs.append(back)
        R_prev = LR[:, 1]
    o = int(np.argmin(tot))
    sel = []
    for k in range(len(lo) - 1, -1, -1):
        sel.append(rows[k][o])
        if backs[k] is not None:
            o = int(backs[k][o])
    sel.reverse()
    return [t[1] for t in sel], np.array([t[2] for t in sel]), np.array([t[3] for t in sel])


def _hpd_levels(H, masses):
    """Density thresholds enclosing the given probability masses of a 2D histogram."""
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h)
    c /= c[-1]
    return np.array([h[min(np.searchsorted(c, m), h.size - 1)] for m in masses])


def _whitened_density(P2, bins, smooth, masses):
    """
    Smoothed 2D density of the samples P2 (N, 2), for highest-density contours.

    Histogram in pair-whitened coordinates (+ Gaussian smoothing of `smooth` bins), mapped
    back to the coordinates of P2, so near-degenerate pairs (|rho| ~ 1) stay smooth.

    Returns
    -------
    tuple or None
        (GX, GY, H, lev): grid (bins, bins) in P2 coordinates, density on it, and the
        density thresholds of `masses` in ascending order (descending in mass). None if the
        pair covariance is singular.
    """
    from scipy.ndimage import gaussian_filter

    m2 = P2.mean(0)
    try:
        Lc = np.linalg.cholesky(np.cov(P2, rowvar=False))
    except np.linalg.LinAlgError:
        return None
    U = np.linalg.solve(Lc, (P2 - m2).T).T
    r = max(4.0, float(np.percentile(np.abs(U), 99.95))) * 1.05
    H, ex, ey = np.histogram2d(U[:, 0], U[:, 1], bins=bins, range=[[-r, r]] * 2)
    H = gaussian_filter(H, smooth)
    cu, cv = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1])
    GU, GV = np.meshgrid(cu, cv, indexing="ij")
    GX = m2[0] + Lc[0, 0] * GU
    GY = m2[1] + Lc[1, 0] * GU + Lc[1, 1] * GV
    return GX, GY, H, np.sort(_hpd_levels(H, masses))


def _decimals(x, sig=2):
    """Decimals that show the positive number `x` with `sig` significant digits."""
    if not np.isfinite(x) or x <= 0:
        return 2
    return max(0, sig - 1 - int(np.floor(np.log10(x))))


def _num(x, d):
    """Fixed-point string with `d` decimals; never prints a signed zero ('-0.00')."""
    v = round(float(x), d)
    return f"{0.0 if v == 0 else v:.{d}f}"


def _tick_decimals(ticks, span):
    """Fewest decimals that represent every tick exactly (2 significant digits of span if none)."""
    t = np.asarray(ticks, dtype=float)
    tol = 1e-6 * max(np.max(np.abs(t)), span)
    d = next((k for k in range(7) if np.all(np.abs(np.round(t, k) - t) <= tol)), None)
    return _decimals(span, sig=2) if d is None else d


def _fixed_formatter(ticks, span):
    """Math-mode fixed-point tick labels sharing one number of decimals (proper minus sign)."""
    d = _tick_decimals(ticks, span)
    return mticker.FuncFormatter(lambda v, _, d=d: f"${_num(v, d)}$")


def _sig_num(x, d_min=0, sig=2, sci_below=1e-2):
    r"""
    Math-mode number with `sig` significant digits and at least `d_min` decimals.

    Written as a\times10^{k} when |x| < sci_below, so a small value never prints as 0.00.
    0 and non-finite x are printed plainly.
    """
    x = float(x)
    if x == 0.0 or not np.isfinite(x):
        return _num(x, 0)
    mant, ex = f"{x:.{sig - 1}e}".split("e")
    if abs(float(f"{mant}e{ex}")) < sci_below:
        return rf"{mant}\times10^{{{int(ex)}}}"
    return _num(x, max(d_min, _decimals(abs(float(f"{mant}e{ex}")), sig)))


def _summary_title(sym, x):
    """
    'sym = median +/- half-width' of the 16-84 % interval (asymmetric if they differ > 5 %).
    Every number keeps 2 significant digits; the median also keeps the uncertainty's
    decimals, and values below 0.01 go to scientific notation.
    """
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    em, ep = q50 - q16, q84 - q50
    d = _decimals(min(em, ep), sig=2)
    med = _sig_num(q50, d_min=d)
    if abs(ep - em) <= 0.05 * 0.5 * (ep + em):
        return rf"${sym} = {med} \pm {_sig_num(0.5 * (ep + em), d_min=d)}$"
    # after a power of ten, hang the errors on an empty group so they don't read as exponents
    med = rf"{med}\,{{}}" if "times" in med else med
    return rf"${sym} = {med}^{{+{_sig_num(ep, d_min=d)}}}_{{-{_sig_num(em, d_min=d)}}}$"


def plot_corner(
    samples,
    true_theta=None,
    batch_mean=None,
    batch_cov=None,
    bins=40,
    smooth=1.1,
    name="fig_corner",
    prior_idx=None,
    prior_label="Hyperprior",
    prior_truth_label=None,
    prior=None,
    n_prior=200_000,
    param_sym=None,
    param_unit=None,
    param_scale=None,
):
    """
    Corner plot of the posterior against the batch covariance and the truth (square, TXT_W).

    Posterior: blue 1/2/3-sigma highest-density regions (39.3 / 86.5 / 98.9 % of the 2D
    mass) and histograms. Batch least squares: vermillion ellipses with the same masses
    and Gaussian pdfs on the diagonal. Truth: black star (2D) and line (diagonal).
    Diagonal titles: posterior median and half-widths of the 16-84 % interval
    (_summary_title). Axis ranges cover the 0.05-99.95 % percentiles, the batch +/-3.5
    sigma and the truth.

    Optional prior overlay (`prior`): dashed reddish-purple highest-density contours with
    the same masses and its marginals on the diagonal. The axis ranges then also cover the
    0.1-99.9 % prior quantiles, and every diagonal curve is scaled to unit peak, since the
    prior is often orders of magnitude wider than the posterior (the posterior histogram
    keeps the bins of the figure without prior). The hyperprior drawn through prior_idx is
    part of the prior and is then not drawn a second time.

    The 2D densities are estimated in pair-whitened coordinates (histogram + Gaussian
    smoothing) and mapped back, so near-degenerate pairs (|rho| ~ 1) stay smooth. Axis
    labels hold only the symbols and stay in line; one wider than a panel wraps after its
    log_10 instead of running into its neighbour. The units are listed in an 'Axis units'
    key under the legend ([-] when dimensionless).

    Above 9 parameters the panels are small: tick labels, titles and markers shrink, and
    the grid is narrowed so the diagonal titles (which overhang to the right) stay inside
    the page width.

    Parameters
    ----------
    samples : array_like, shape (N, n_par)
        Posterior samples in displayed coordinates, internal units (scaled by PARAM_SCALE
        for display).
    true_theta : array_like, shape (n_par,), optional
        Truth (no star and no legend entry when None).
    batch_mean, batch_cov : array_like, shapes (n_par,) and (n_par, n_par), optional
        Batch estimate and covariance (no batch reference and no legend entry when either is
        None).
    bins : int
        Histogram bins (1D and 2D).
    smooth : float
        Gaussian smoothing of the 2D histograms [bins].
    name : str
        File name of the PDF.
    prior_idx : sequence of int, optional
        Parameters the batch does not estimate (e.g. hierarchical ln sigma_g), whose entries
        of batch_mean / batch_cov hold their (hyper)prior instead. Drawn as a separate
        reference (dashed reddish purple, own legend entry): pdf on the diagonal and
        ellipses on prior x prior panels; mixed batch x prior panels get no reference
        ellipse, since the product of a batch marginal and a hyperprior is neither.
    prior_label : str
        Legend entry of that prior.
    prior_truth_label : str, optional
        If given, true_theta of the prior_idx parameters is not a generative value (e.g. the
        realized RMS of the group deviations): drawn as a hollow star (dotted line on the
        diagonal) with this legend entry instead of as the truth.
    prior : sequence of scipy.stats frozen distributions, or array_like, optional
        Prior overlay, in the coordinates of `samples` (internal units). Either one
        independent distribution per parameter (drawn exactly: pdfs on the diagonal,
        contours of their product on the 2D panels), or prior samples of shape
        (M, n_par), e.g. drawn in the sampled coordinates and mapped through a
        reparametrization (drawn like the posterior: smoothed histograms).
    n_prior : int
        Draws (fixed seed) used for the highest-density levels of a distribution prior.
    param_sym : list of str, optional
        Symbols of the displayed coordinates, when they differ from the configured ones
        (PARAM_SYM, the default), e.g. after a rotation of the position.
    param_unit, param_scale : optional
        Display units and scales replacing PARAM_UNIT and PARAM_SCALE (the defaults), e.g.
        auto_param_scales of the sigmas in rotated coordinates.

    Returns
    -------
    Path
        The saved PDF.
    """
    from scipy.ndimage import gaussian_filter1d

    _require_configured(np.shape(samples)[1])
    n_dim = np.shape(samples)[1]
    sym = PARAM_SYM if param_sym is None else list(param_sym)
    if len(sym) != n_dim:
        raise ValueError(f"param_sym: {len(sym)} symbols for {n_dim} parameters")
    s = PARAM_SCALE if param_scale is None else np.asarray(param_scale, dtype=float)
    units = PARAM_UNIT if param_unit is None else list(param_unit)
    if len(s) != n_dim or len(units) != n_dim:
        raise ValueError(f"param_unit/param_scale: need {n_dim} entries")
    X = np.asarray(samples, dtype=float) * s
    n_par = X.shape[1]
    tr = None if true_theta is None else np.asarray(true_theta) * s
    bm = None if batch_mean is None else np.asarray(batch_mean) * s
    bc = None if batch_cov is None else np.asarray(batch_cov) * np.outer(s, s)
    pri = set() if prior_idx is None else {int(k) for k in prior_idx}
    realized = pri if prior_truth_label else set()  # truth drawn as a realized value
    star_full = dict(marker="*", mfc=C_TRUTH, mec="white", mew=0.4)
    star_hollow = dict(marker="*", mfc="white", mec=C_TRUTH, mew=0.6)

    masses = np.array([0.393, 0.865, 0.989])
    fills = [_tint(C_POST, a) for a in (0.16, 0.38, 0.68)]

    # Prior overlay: independent distributions (exact) or samples (smoothed histograms)
    dists = Ps = None
    if prior is not None:
        if all(hasattr(p, "pdf") for p in prior):
            dists = list(prior)
            if len(dists) != n_par:
                raise ValueError(f"prior: {len(dists)} distributions for {n_par} parameters")
            rng = np.random.default_rng(0)
            draws = np.column_stack(
                [np.ravel(p.rvs(size=int(n_prior), random_state=rng)) for p in dists]
            )
            pdf_draws = np.column_stack([p.pdf(draws[:, k]) for k, p in enumerate(dists)])
            p_lo = np.array([p.ppf(1e-3) for p in dists]) * s
            p_hi = np.array([p.ppf(1.0 - 1e-3) for p in dists]) * s
        else:
            Ps = np.asarray(prior, dtype=float) * s
            if Ps.ndim != 2 or Ps.shape[1] != n_par:
                raise ValueError(f"prior samples: shape {Ps.shape}, expected (M, {n_par})")
            Ps = Ps[np.all(np.isfinite(Ps), axis=1)]
            p_lo, p_hi = np.percentile(Ps, [0.1, 99.9], axis=0)
    has_prior = dists is not None or Ps is not None
    hyper_ref = not has_prior  # draw the prior_idx hyperprior from the batch slot

    # Axis ranges: posterior, batch +/-3.5 sigma, truth (+ prior quantiles)
    lo, hi = np.percentile(X, [0.05, 99.95], axis=0)
    if bm is not None and bc is not None:
        sb = np.sqrt(np.diag(bc))
        lo, hi = np.minimum(lo, bm - 3.5 * sb), np.maximum(hi, bm + 3.5 * sb)
    if tr is not None:
        lo, hi = np.minimum(lo, tr), np.maximum(hi, tr)
    pad = 0.06 * (hi - lo)
    h_lo, h_hi = lo - pad, hi + pad  # posterior histogram range (as without the prior)
    if has_prior:
        lo, hi = np.minimum(lo, p_lo), np.maximum(hi, p_hi)
        pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    big = n_par > 9
    tick_fs, title_fs = (5.0, 5.5) if big else (6.5, 7.0)
    label_fs = 7.0 if big else plt.rcParams["axes.labelsize"]
    ms_star, lw_s = (4.5, 0.75) if big else (6.5, 1.0)  # truth marker, line-width factor
    wsp = 0.08
    titles = [_summary_title(sym[k], X[:, k]) for k in range(n_par)]

    fig, axes = plt.subplots(n_par, n_par, figsize=(TXT_W, TXT_W))
    left, right, bottom, top = 0.085, 0.995, 0.075, 0.97
    # measure text with the actual renderer (fraction of the page width)
    rend = fig.canvas.get_renderer()

    def _w_frac(txt, fs):
        """Rendered width of txt at font size fs, as a fraction of the figure width."""
        t = fig.text(0, 0, txt, fontsize=fs)
        w = t.get_window_extent(renderer=rend).width / fig.bbox.width
        t.remove()
        return w

    if big:
        w_tit = np.array([_w_frac(t, title_fs) for t in titles])
        left, bottom = 0.06, 0.05
        # panel pitch such that title i (left edge at column i) ends inside the page
        pitch = (right - left) / (n_par - wsp / (1.0 + wsp))
        for i in range(1, n_par):
            pitch = min(pitch, (0.995 - w_tit[i] - left) / i)
        right = left + pitch * (n_par - wsp / (1.0 + wsp))
    fig.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top, wspace=wsp, hspace=wsp
    )
    # Axis labels hold only the symbols (the units go in the key under the legend), in line
    # per row/column: a symbol wider than the panel pitch (less 2 pt) wraps after its
    # leading log_10, so neighbours never meet.
    n_eff = n_par - wsp / (1.0 + wsp)
    w_max = min(right - left, top - bottom) / n_eff * TXT_W * 72.0 - 2.0
    labels = [
        _wrap_label(
            sym[k], None, w_max, lambda txt: _w_frac(txt, label_fs) * TXT_W * 72.0
        )
        for k in range(n_par)
    ]
    panel_frac = (right - left) / (n_par + (n_par - 1) * wsp)
    # Ticks are chosen along the bottom row (_row_ticks), so the label at the end of one
    # axis never runs into the facing label of the next panel (e.g. "20" against "-200").
    # Facing labels are checked with the measured digit width: TeX's small optical sizes
    # are wider than the 6.5 pt calibration scaled down (5 pt: 3.39 pt, not 3.05 pt).
    digit_m = _w_frac("$" + "0" * 10 + "$", tick_fs) / 10.0 / panel_frac
    if has_prior:
        # The prior widens the axes to near-symmetric ranges with long labels at both ends
        # (-500, 500 next to -500, 500 reads as one row): facing labels keep 0.5 digit, which
        # leaves e.g. 0, 600 there; labels within a panel may come to 0.3 digit, else the
        # log10 sigma axes drop to -4.0, -2.4.
        ticks, lo, hi = _row_ticks(
            lo, hi, digit_m, wsp, gap=0.5 * digit_m, min_gap=0.3 * digit_m
        )
    else:
        # Tick sets as _inner_ticks (calibrated digit width, labels 0.8 digit apart), so
        # an axis changes only where its end label would meet the next panel's.
        digit_w = 3.96 * tick_fs / 6.5 / (panel_frac * TXT_W * 72.0)
        ticks, lo, hi = _row_ticks(
            lo, hi, digit_w, wsp, gap=0.5 * digit_m, min_gap=0.8 * digit_w, edge_w=digit_m
        )
    fmts = [_fixed_formatter(ticks[k], hi[k] - lo[k]) for k in range(n_par)]

    for i in range(n_par):
        for j in range(n_par):
            ax = axes[i, j]
            ax.grid(False)
            if j > i:
                ax.set_visible(False)
                continue
            ax.set_xlim(lo[j], hi[j])
            ax.set_xticks(ticks[j])
            ax.xaxis.set_major_formatter(fmts[j])
            ax.tick_params(labelsize=tick_fs, length=2.5 if not big else 2.0, pad=2)

            if i == j:  # ---- marginal
                # with a prior: every curve scaled to unit peak (weights = 1 / max count)
                peak = np.histogram(X[:, i], bins=bins, range=(h_lo[i], h_hi[i]))[0].max()
                ax.hist(
                    X[:, i],
                    bins=bins,
                    range=(h_lo[i], h_hi[i]),
                    density=not has_prior,
                    weights=np.full(X.shape[0], 1.0 / peak) if has_prior else None,
                    histtype="stepfilled",
                    color=_tint(C_POST, 0.3),
                    ec=C_POST,
                    lw=0.8 * lw_s,
                )
                if bm is not None and bc is not None and (hyper_ref or i not in pri):
                    sbi = np.sqrt(bc[i, i])
                    xx = np.unique(  # dense around the batch: it may be a narrow spike
                        np.r_[
                            np.linspace(lo[i], hi[i], 300),
                            np.linspace(bm[i] - 5 * sbi, bm[i] + 5 * sbi, 200),
                        ]
                    )
                    xx = xx[(xx >= lo[i]) & (xx <= hi[i])]
                    ax.plot(
                        xx,
                        norm.pdf(xx, bm[i], sbi) * (np.sqrt(2 * np.pi) * sbi if has_prior else 1),
                        color=C_PRIOR if i in pri else C_BATCH,
                        ls="--" if i in pri else "-",
                        lw=0.9 * lw_s,
                    )
                if dists is not None:
                    xx = np.linspace(lo[i], hi[i], 600)
                    yy = dists[i].pdf(xx / s[i])
                    ax.plot(xx, yy / yy.max(), color=C_PRIOR, ls="--", lw=0.9 * lw_s)
                elif Ps is not None:
                    hp, ep = np.histogram(Ps[:, i], bins=3 * bins, range=(lo[i], hi[i]))
                    hp = gaussian_filter1d(hp.astype(float), 1.5)
                    ax.plot(
                        0.5 * (ep[1:] + ep[:-1]),
                        hp / hp.max(),
                        color=C_PRIOR,
                        ls="--",
                        lw=0.9 * lw_s,
                    )
                if tr is not None:
                    ax.axvline(
                        tr[i],
                        color=C_TRUTH,
                        lw=0.8 * lw_s,
                        ls=":" if i in realized else "-",
                    )
                ax.set_ylim(bottom=0.0)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                # left-aligned: overflow goes right, into the hidden upper triangle (the last
                # one past the grid; the tight bbox at save keeps it on the page)
                ax.set_title(titles[i], loc="left", fontsize=title_fs, pad=3)
            else:  # ---- joint (x: param j, y: param i)
                ax.set_ylim(lo[i], hi[i])
                ax.set_yticks(ticks[i])
                ax.yaxis.set_major_formatter(fmts[i])
                P2 = X[:, [j, i]]
                dens = _whitened_density(P2, bins, smooth, masses)
                if dens is not None:
                    GX, GY, H, lev = dens
                    # descending in mass -> ascending in density
                    lev = np.r_[lev, H.max() * 1.0001]
                if dens is not None and np.all(np.diff(lev) > 0):
                    ax.contourf(GX, GY, H, levels=lev, colors=fills, zorder=1)
                    ax.contour(
                        GX,
                        GY,
                        H,
                        levels=lev[:-1],
                        colors=C_POST,
                        linewidths=0.45 * lw_s,
                        zorder=2,
                    )
                else:
                    ax.plot(P2[:, 0], P2[:, 1], ",", color=C_POST, alpha=0.2, zorder=1)
                pdens = None
                if dists is not None:  # exact: product of the two marginal pdfs
                    gx = np.linspace(lo[j], hi[j], 241)
                    gy = np.linspace(lo[i], hi[i], 241)
                    GXp, GYp = np.meshgrid(gx, gy)
                    Hp = dists[i].pdf(GYp / s[i]) * dists[j].pdf(GXp / s[j])
                    # level of mass m: (1 - m) quantile of the density at prior draws
                    levp = np.quantile(pdf_draws[:, i] * pdf_draws[:, j], 1.0 - masses)
                    pdens = GXp, GYp, Hp, np.sort(levp)
                elif Ps is not None:
                    # a little more smoothing: dashed lines show the sampling noise
                    pdens = _whitened_density(Ps[:, [j, i]], bins, 1.5 * smooth, masses)
                if pdens is not None and np.all(np.diff(pdens[3]) > 0):
                    ax.contour(
                        *pdens,
                        colors=C_PRIOR,
                        linestyles="--",
                        linewidths=0.7 * lw_s,
                        zorder=2.5,
                    )
                n_pri = (i in pri) + (j in pri)  # 0: batch, 2: hyperprior, 1: neither
                if bm is not None and bc is not None and n_pri != 1 and (hyper_ref or not n_pri):
                    C2 = bc[np.ix_([j, i], [j, i])]
                    for rad in (1.0, 2.0, 3.0):
                        e = _ellipse_2d(bm[[j, i]], C2, rad)
                        ax.plot(
                            e[0],
                            e[1],
                            color=C_PRIOR if n_pri else C_BATCH,
                            ls="--" if n_pri else "-",
                            lw=0.7 * lw_s,
                            zorder=3,
                        )
                if tr is not None:
                    hollow = i in realized or j in realized
                    ax.plot(
                        tr[j],
                        tr[i],
                        ls="none",
                        ms=ms_star,
                        zorder=4,
                        **(star_hollow if hollow else star_full),
                    )

            # outer labels only
            if i == n_par - 1:
                ax.set_xlabel(labels[j], fontsize=label_fs, labelpad=3)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=label_fs, labelpad=3)
            else:
                ax.tick_params(labelleft=False)
    fig.align_ylabels(axes[1:, 0])
    fig.align_xlabels(axes[-1, :])

    sig123 = r"($1\sigma$, $2\sigma$, $3\sigma$)"
    handles = [
        (
            Patch(facecolor=fills[2], lw=0),
            Patch(facecolor=fills[1], lw=0),
            Patch(facecolor=fills[0], lw=0),
        ),
        Line2D([], [], color=C_BATCH, lw=0.9),
        Line2D([], [], ls="none", ms=8, **star_full),
    ]
    leg_labels = [f"MCMC Posterior {sig123}", f"Batch Least Squares {sig123}", "Truth"]
    if tr is None:  # no truth: no truth entry
        del handles[2], leg_labels[2]
    if bm is None or bc is None:  # no batch solution (e.g. the STT benchmark): no batch entry
        del handles[1], leg_labels[1]
    if realized and tr is not None:  # dotted line with a hollow star, as on the panels
        handles.append(Line2D([], [], color=C_TRUTH, lw=0.8, ls=":", ms=8, **star_hollow))
        leg_labels.append(prior_truth_label)
    if pri and bm is not None and bc is not None and hyper_ref:
        handles.insert(2, Line2D([], [], color=C_PRIOR, lw=0.9, ls="--"))
        leg_labels.insert(2, f"{prior_label} {sig123}")
    if has_prior:  # after the batch entry (or the posterior one, without a batch)
        k = 2 if bm is not None and bc is not None else 1
        handles.insert(k, Line2D([], [], color=C_PRIOR, lw=0.9, ls="--"))
        leg_labels.insert(k, f"Prior {sig123}")
    # The upper-right triangle is empty: anchor the legend to the top-right (hidden) panel.
    corner_box = axes[0, -1].get_position()
    leg = fig.legend(
        handles,
        leg_labels,
        loc="upper right",
        bbox_to_anchor=(corner_box.x1, corner_box.y1),
        bbox_transform=fig.transFigure,
        fontsize=8,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        handlelength=3.0,
        handleheight=1.0,
        labelspacing=0.6,
    )
    # Units key under the legend: consecutive parameters sharing a display unit on one line
    # (at most 6 symbols per row), [-] when dimensionless.
    groups = []
    for k in range(n_par):
        u = units[k] or "-"
        if groups and groups[-1][1] == u:
            groups[-1][0].append(k)
        else:
            groups.append(([k], u))
    lines = []
    for ks, u in groups:
        syms = [f"${sym[k]}$" for k in ks]
        rows = [", ".join(syms[a : a + 6]) for a in range(0, len(syms), 6)]
        lines += [r + "," for r in rows[:-1]] + [rows[-1] + f" [{u}]"]
    bb = leg.get_window_extent(rend).transformed(fig.transFigure.inverted())
    fig.text(
        bb.x0 + 0.012,
        bb.y0 - 0.012,
        "Axis units:\n" + "\n".join(lines),
        ha="left",
        va="top",
        fontsize=label_fs,
        linespacing=1.35,
    )
    return save_figure(fig, name)



def plot_mcmc_residuals(
    model,
    t_obs,
    vis_mask,
    theta_batch,
    theta_mcmc=None,
    components=None,
    name="fig_postfit_residuals",
):
    """
    Normalized post-fit residuals of the batch estimate and of the MCMC MAP.

    Runs model.residuals_func on both estimates and calls plot_postfit_residuals.

    Parameters
    ----------
    model : MCMC.MCMCModel
    t_obs : array_like, shape (N,)
        All measurement epochs [s].
    vis_mask : array_like of bool, shape (N,)
        Epochs plotted (used measurements).
    theta_batch : array_like, shape (ndim,)
        Batch estimate in model coordinates (the input of model.residuals_func).
    theta_mcmc : array_like, shape (ndim,), optional
        Defaults to the MAP sample (model.get_map_estimate()).
    components : list of int, optional
        Panels shown (indices into RESID_LABELS; default all).
    name : str
        File name of the PDF.

    Returns
    -------
    Path
        The saved PDF.
    """
    if theta_mcmc is None:
        theta_mcmc = model.get_map_estimate()
    return plot_postfit_residuals(
        t_obs,
        model.residuals_func(np.asarray(theta_batch, dtype=float)),
        model.residuals_func(np.asarray(theta_mcmc, dtype=float)),
        vis_mask,
        components=components,
        name=name,
    )


def plot_mcmc_corner(
    model,
    true_theta=None,
    batch_mean=None,
    batch_cov=None,
    transform=None,
    prior=False,
    prior_transform=None,
    n_prior=200_000,
    **kwargs,
):
    """
    Corner plot of model.samples (posterior HDRs vs batch covariance vs truth).

    With prior=True the prior (model.param_priors, independent) is overlaid and the file
    defaults to "fig_corner_prior", so a run can save both versions.

    Parameters
    ----------
    model : MCMC.MCMCModel
    true_theta, batch_mean, batch_cov : array_like, optional
        In displayed coordinates (see plot_corner).
    transform : callable, optional
        Model -> displayed coordinates, applied to model.samples (e.g. a hierarchical
        reparametrization).
    prior : bool
        Overlay the prior (see plot_corner).
    prior_transform : callable, optional
        Sampled -> displayed coordinates, applied to prior draws; defaults to `transform`.
        Needed when model.samples were mapped in place after the run (e.g. a non-centred
        hierarchical chain), since model.param_priors live in the sampled coordinates.
        Without one, the priors are drawn exactly from their pdfs; with one, from
        n_prior mapped draws.
    n_prior : int
        Prior draws (fixed seed).
    **kwargs
        Passed to plot_corner (bins, smooth, name, prior_idx, prior_label,
        prior_truth_label, param_sym, param_unit, param_scale).

    Returns
    -------
    Path
        The saved PDF.
    """
    samples = np.asarray(model.samples, dtype=float)
    if transform is not None:
        samples = np.asarray(transform(samples), dtype=float)
    prior_arg = None
    if prior:
        kwargs.setdefault("name", "fig_corner_prior")
        f = transform if prior_transform is None else prior_transform
        if f is None:
            prior_arg = list(model.param_priors)
        else:
            rng = np.random.default_rng(0)
            draws = np.column_stack(
                [np.ravel(p.rvs(size=int(n_prior), random_state=rng)) for p in model.param_priors]
            )
            prior_arg = np.asarray(f(draws), dtype=float)
    return plot_corner(
        samples,
        true_theta=true_theta,
        batch_mean=batch_mean,
        batch_cov=batch_cov,
        prior=prior_arg,
        n_prior=n_prior,
        **kwargs,
    )


# Symbols of the RTN corner (fig_corner_rtn): the six state slots, epoch subscript 0 as in
# the scripts' PARAM_SYM (\delta x_0 ... \delta v_{z,0}).
RTN_SYM = [
    r"\delta r_{R,0}",
    r"\delta r_{T,0}",
    r"\delta r_{N,0}",
    r"\delta v_{R,0}",
    r"\delta v_{T,0}",
    r"\delta v_{N,0}",
]


def rtn_matrix(x_ref):
    """
    Rotation from the inertial frame to the RTN frame of the orbit through x_ref.

    R = r / |r| (radial), N = r x v / |r x v| (orbit normal), T = N x R (along-track,
    in-plane), with r, v the position and velocity of x_ref relative to the frame origin
    (the central body).

    Parameters
    ----------
    x_ref : array_like, shape (>=6,)
        Reference state [r, v, ...] (any further entries are ignored).

    Returns
    -------
    ndarray, shape (3, 3)
        Rows R, T, N in inertial components, so that d_rtn = C @ d_inertial.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    r, v = x_ref[:3], x_ref[3:6]
    e_r = r / np.linalg.norm(r)
    h = np.cross(r, v)
    e_n = h / np.linalg.norm(h)
    e_t = np.cross(e_n, e_r)
    return np.vstack([e_r, e_t, e_n])


def rtn_corner_map(x_ref, ndim):
    """
    Corner map (see plot_mcmc_diagnostics) to the RTN frame of the orbit through x_ref.

    The position deviation (slots 0-2) and the velocity deviation (slots 3-5) are both
    resolved along R, T, N of rtn_matrix(x_ref) (the inertial velocity deviation in RTN
    components, no omega x r term); the other parameters are unchanged.

    Parameters
    ----------
    x_ref : array_like, shape (>=6,)
        Reference state [r, v, ...], e.g. the batch solution about which the samples are
        deviations.
    ndim : int
        Number of parameters (>= 6, the first six being the position and velocity).

    Returns
    -------
    ndarray, shape (ndim, ndim)
    """
    C = rtn_matrix(x_ref)
    A = np.eye(int(ndim))
    A[0:3, 0:3] = C
    A[3:6, 3:6] = C
    return A


def _mapped_corner_kw(A, true_theta, batch_mean, batch_cov):
    """plot_mcmc_corner kwargs for displayed coordinates A @ theta."""
    A = np.asarray(A, dtype=float)
    return dict(
        true_theta=None if true_theta is None else A @ np.asarray(true_theta),
        batch_mean=A @ np.asarray(batch_mean),
        batch_cov=None if batch_cov is None else A @ np.asarray(batch_cov) @ A.T,
        transform=lambda S: np.asarray(S) @ A.T,
    )


def plot_mcmc_diagnostics(
    model,
    burn_in,
    thin,
    t_obs,
    vis_mask,
    true_theta=None,
    batch_theta=None,
    batch_cov=None,
    components=None,
    acf_groups=None,
    corner_prior=True,
    corner_map=None,
    corner_sym=None,
    corner_rtn=None,
    corner_rtn_kw=None,
):
    """
    The MCMC journal figures: residuals, traces, autocorrelation, corner and (optionally)
    the corner with the prior overlaid (fig_corner_prior) and the corner in the RTN frame
    (fig_corner_rtn).

    For runs whose sampled parameters are the displayed ones (no transform). A figure that
    fails is reported and skipped, so a long run is never lost to a plotting error.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run.
    burn_in, thin : int
        Burn-in and thinning of the run [steps].
    t_obs : array_like, shape (N,)
        All measurement epochs [s].
    vis_mask : array_like of bool, shape (N,)
        Used measurements.
    true_theta : array_like, shape (ndim,), optional
        Truth.
    batch_theta : array_like, shape (ndim,), optional
        Batch estimate; defaults to zeros, i.e. deltas about the batch solution.
    batch_cov : array_like, shape (ndim, ndim), optional
        Batch covariance (corner plot).
    components : list of int, optional
        Residual panels shown (indices into RESID_LABELS).
    acf_groups : list of list of int, optional
        Panels of the autocorrelation figure.
    corner_prior : bool
        Also save the corner plot with the prior (model.param_priors) overlaid.
    corner_map : array_like, shape (ndim, ndim), optional
        Linear map A from the model to the displayed coordinates of the two corner plots
        (e.g. a rotation of the position into the observer's line-of-sight frame), applied
        to the samples, the prior draws, the truth and the batch estimate, and as A C A^T to
        the batch covariance. Each displayed coordinate keeps the unit and display scale of
        the parameter in its slot. The traces and the autocorrelation stay in the model
        coordinates.
    corner_sym : list of str, optional
        Symbols of the displayed corner coordinates (default PARAM_SYM).
    corner_rtn : array_like, shape (>=6,), optional
        Reference state [r, v, ...] relative to the central body (e.g. the batch solution
        x0_ref1). If given, a third corner plot, fig_corner_rtn, is saved with the position
        and velocity deviations (slots 0-5) resolved in the RTN frame of the orbit through
        it (rtn_corner_map; independent of corner_map, no prior overlay). Off by default.
    corner_rtn_kw : dict, optional
        Further plot_corner kwargs of the RTN corner: param_sym (default RTN_SYM +
        PARAM_SYM[6:]), and e.g. param_unit/param_scale when the display units were sized
        to the x, y, z sigmas (auto_param_scales).

    Returns
    -------
    None
    """
    batch = np.zeros(model.ndim) if batch_theta is None else np.asarray(batch_theta)
    corner_kw = dict(true_theta=true_theta, batch_mean=batch, batch_cov=batch_cov)
    if corner_map is not None:
        corner_kw = _mapped_corner_kw(corner_map, true_theta, batch, batch_cov)
    if corner_sym is not None:
        corner_kw["param_sym"] = corner_sym
    figures = (
        (
            "post-fit residuals",
            lambda: plot_mcmc_residuals(
                model, t_obs, vis_mask, batch, components=components
            ),
        ),
        (
            "traces",
            lambda: plot_mcmc_traces(
                model, burn_in, true_theta=true_theta, batch_theta=batch
            ),
        ),
        (
            "autocorrelation",
            lambda: plot_mcmc_autocorrelation(model, burn_in, thin, groups=acf_groups),
        ),
        ("corner", lambda: plot_mcmc_corner(model, **corner_kw)),
    )
    if corner_prior:
        figures += (
            ("corner + prior", lambda: plot_mcmc_corner(model, prior=True, **corner_kw)),
        )
    if corner_rtn is not None:
        rtn_kw = _mapped_corner_kw(
            rtn_corner_map(corner_rtn, model.ndim), true_theta, batch, batch_cov
        )
        rtn_kw["param_sym"] = RTN_SYM + list(PARAM_SYM[6:])
        rtn_kw.update(corner_rtn_kw or {})
        figures += (
            ("corner RTN", lambda: plot_mcmc_corner(model, name="fig_corner_rtn", **rtn_kw)),
        )
    for label, draw in figures:
        try:
            draw()
        except Exception as err:
            print(f"[Figure] {label} skipped: {type(err).__name__}: {err}")


# ============================================================
# GMM PROPAGATION (figures of GMM.GMM.propagate_stm results)
# ============================================================


def _ellipse_2d(mean2, cov2, radius, n=181):
    """Boundary {x : (x - m)^T C^{-1} (x - m) = radius^2}, shape (2, n)."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    vals, vecs = np.linalg.eigh(cov2)
    circ = np.vstack([np.cos(t), np.sin(t)])
    return mean2[:, None] + radius * (
        vecs @ (np.sqrt(np.clip(vals, 0.0, None))[:, None] * circ)
    )


def plot_gmm_propagation(
    panels,
    prob=0.99,
    n_draw=20000,
    seed=0,
    name="fig_gmm_stm_propagation",
    axis_names=None,
    show_mc=True,
    show_components=True,
):
    """
    Position projections (xy, xz, yz): propagated GMM, single Gaussian and Monte Carlo.

    One row per panel dict (e.g. row 0 at an epoch, row 1 at the target epoch), drawn in m
    relative to its center: the MC cloud (grey), every GMM component's ellipse, the GMM
    highest-density region holding `prob` of the mass (level from mixture draws), the
    single-Gaussian ellipse with the same mass (dashed vermillion), optionally the batch LSQ
    Gaussian's ellipse (dash-dot bluish green) and the truth (star). Full text width.

    Parameters
    ----------
    panels : list of dict
        At most 2 rows (panels lettered a-f). Keys, in km: "mc" (M, 3) Monte Carlo
        positions; "gmm" and "single" (GMM.GMM of the position: mixture and single
        Gaussian); "truth" (3,); "center" (3,), e.g. the MC mean; "title" (str). Optional
        "batch" (GMM.GMM, K = 1): the propagated batch least-squares Gaussian.
    prob : float
        Probability mass of the ellipses and of the GMM region.
    n_draw : int
        Mixture draws used to find the density level of the GMM region.
    seed : int
        Seed of those draws.
    name : str
        File name of the PDF.
    axis_names : list of str, optional
        Labels of the three position axes (default delta x, delta y, delta z in m), e.g. for
        panels rotated to the RTN frame.
    show_mc : bool
        Draw the MC cloud (False: it is also left out of the axis limits and the legend).
    show_components : bool
        Draw every GMM component's ellipse (False: only the mixture region).

    Returns
    -------
    Path
        The saved PDF.
    """
    r_ell = np.sqrt(chi2.ppf(prob, 2))
    pct = rf"{100 * prob:g}$\%$"
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = (
        [r"$\delta x$ [m]", r"$\delta y$ [m]", r"$\delta z$ [m]"]
        if axis_names is None
        else list(axis_names)
    )
    has_batch = any("batch" in pnl for pnl in panels)
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(TXT_W, 4.9), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, hspace=0.04, wspace=0.04)
    sub = fig.subfigures(len(panels), 1)
    axes = np.array([sf.subplots(1, 3) for sf in np.atleast_1d(sub)])
    for row, pnl in enumerate(panels):
        c = np.asarray(pnl["center"], dtype=float)
        mc = (pnl["mc"] - c) * 1e3
        w = pnl["gmm"].weights
        mu = (pnl["gmm"].means - c) * 1e3
        P = pnl["gmm"].covs * 1e6
        mu1 = (pnl["single"].means[0] - c) * 1e3
        P1 = pnl["single"].covs[0] * 1e6
        tr = (pnl["truth"] - c) * 1e3
        if "batch" in pnl:
            mub = (pnl["batch"].means[0] - c) * 1e3
            Pb = pnl["batch"].covs[0] * 1e6
        draws = GMM(w, mu, P).sample(n_draw, rng)

        for col, (i, j) in enumerate(pairs):
            ax = axes[row, col]
            ix = [i, j]
            g2 = GMM(w, mu[:, ix], P[:, ix][:, :, ix])
            e1 = _ellipse_2d(mu1[ix], P1[np.ix_(ix, ix)], r_ell)
            ext_lo = [
                e1.min(1),
                np.percentile(draws[:, ix], 0.1, axis=0),
                tr[ix],
            ]
            ext_hi = [
                e1.max(1),
                np.percentile(draws[:, ix], 99.9, axis=0),
                tr[ix],
            ]
            if show_mc:
                ext_lo.insert(0, mc[:, ix].min(0))
                ext_hi.insert(0, mc[:, ix].max(0))
            if "batch" in pnl:
                eb = _ellipse_2d(mub[ix], Pb[np.ix_(ix, ix)], r_ell)
                ext_lo.append(eb.min(1))
                ext_hi.append(eb.max(1))

            lo = np.min(ext_lo, axis=0)
            hi = np.max(ext_hi, axis=0)
            pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad

            if show_mc:
                ax.scatter(
                    mc[:, i],
                    mc[:, j],
                    s=2.5,
                    color=C_MC,
                    alpha=0.7,
                    lw=0,
                    rasterized=True,
                    zorder=1,
                )
            if show_components:
                for k in range(len(w)):
                    e = _ellipse_2d(g2.means[k], g2.covs[k], r_ell)
                    ax.plot(e[0], e[1], color=C_POST, lw=0.45, alpha=0.55, zorder=2)

            gx, gy = np.meshgrid(
                np.linspace(lo[0], hi[0], 240), np.linspace(lo[1], hi[1], 240)
            )
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            dens = np.exp(g2.logpdf(grid)).reshape(gx.shape)
            level = np.quantile(np.exp(g2.logpdf(draws[:, ix])), 1.0 - prob)
            ax.contour(
                gx, gy, dens, levels=[level], colors=C_POST, linewidths=1.3, zorder=3
            )

            ax.plot(e1[0], e1[1], color=C_BATCH, lw=1.1, ls="--", zorder=4)
            if "batch" in pnl:
                ax.plot(eb[0], eb[1], color=C_LSQ, lw=1.1, ls="-.", zorder=4)
            ax.plot(
                tr[i],
                tr[j],
                ls="none",
                marker="*",
                ms=7.5,
                mfc=C_TRUTH,
                mec="white",
                mew=0.5,
                zorder=5,
            )
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.set_xlabel(names[i])
            ax.set_ylabel(names[j])
            ax.set_title(f"({'abcdef'[3 * row + col]})", loc="left")
        np.atleast_1d(sub)[row].suptitle(pnl["title"], fontsize=8)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=3, color=C_MC, label="Monte Carlo"),
        Line2D(
            [],
            [],
            color=C_POST,
            lw=0.45,
            alpha=0.7,
            label=f"GMM Components + STMs ({pct} HDR)",
        ),
        Line2D([], [], color=C_POST, lw=1.3, label=f"GMM Mixture + STMs ({pct} HDR)"),
        Line2D(
            [],
            [],
            color=C_BATCH,
            lw=1.1,
            ls="--",
            label=f"Single Gaussian + STM ({pct} HDR)",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="*",
            ms=7.5,
            mfc=C_TRUTH,
            mec="white",
            mew=0.5,
            label="Truth",
        ),
    ]
    if has_batch:
        handles.insert(
            4,
            Line2D(
                [],
                [],
                color=C_LSQ,
                lw=1.1,
                ls="-.",
                label=f"Batch LSQ + STM ({pct} HDR)",
            ),
        )
    if not show_components:
        handles.pop(1)
    if not show_mc:
        handles.pop(0)
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=3 if len(handles) >= 6 else len(handles),
        columnspacing=1.0,
    )
    return save_figure(fig, name)


def plot_prediction_history(
    t_hr, hist, t_obs_end_hr, name="fig_gmm_prediction_history"
):
    """
    Position uncertainty and Monte Carlo log-density vs time: GMM vs single Gaussian.

    (a) sqrt(trace P_rr) [m] on a log scale: Monte Carlo (thick grey), GMM + STMs (blue),
    single Gaussian (dashed vermillion). (b) Difference of the mean log-density of the MC
    positions, GMM minus single Gaussian [nats] (> 0: the GMM fits the MC cloud better). The
    observation arc is shaded. Single column. If hist holds the batch LSQ Gaussian
    (sig_batch, ll_batch), it is added to both panels (dash-dot bluish green; in (b) batch
    minus single Gaussian).

    Parameters
    ----------
    t_hr : array_like, shape (N_t,)
        Output epochs [h].
    hist : dict
        From GMM.position_prediction_history (km; plotted in m).
    t_obs_end_hr : float
        End of the observation arc [h] (shaded from t_hr[0]).
    name : str
        File name of the PDF.

    Returns
    -------
    Path
        The saved PDF.
    """
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(COL_W, 3.3),
        sharex=True,
        layout="constrained",
        gridspec_kw=dict(height_ratios=(1.35, 1.0)),
    )
    for ax in (ax0, ax1):
        ax.axvspan(t_hr[0], t_obs_end_hr, color=C_SPAN, lw=0, zorder=0)
        ax.set_xlim(t_hr[0], t_hr[-1])
    if (t_obs_end_hr - t_hr[0]) >= 0.3 * (t_hr[-1] - t_hr[0]):
        ax0.text(
            0.5 * (t_hr[0] + t_obs_end_hr),
            0.96,
            "Observation arc",
            color="0.3",
            fontsize=7,
            ha="center",
            va="top",
            transform=ax0.get_xaxis_transform(),
        )
    else:
        # arc too short to hold the label: put it to the right, arrow back to the arc
        ax0.annotate(
            "Observation arc",
            xy=(t_obs_end_hr, 0.9),
            xycoords=ax0.get_xaxis_transform(),
            xytext=(12, 0),
            textcoords="offset points",
            color="0.3",
            fontsize=7,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="0.3", lw=0.6, shrinkA=1, shrinkB=0),
        )

    ax0.plot(
        t_hr,
        hist["sig_mc"] * 1e3,
        color="0.78",
        lw=3.2,
        solid_capstyle="round",
        label="Monte Carlo",
        zorder=2,
    )
    ax0.plot(
        t_hr, hist["sig_gmm"] * 1e3, color=C_POST, lw=1.2, label="GMM + STMs", zorder=3
    )
    ax0.plot(
        t_hr,
        hist["sig_1"] * 1e3,
        color=C_BATCH,
        lw=1.2,
        ls="--",
        label="Single Gaussian + STM",
        zorder=4,
    )
    has_batch = "sig_batch" in hist
    if has_batch:
        ax0.plot(
            t_hr,
            hist["sig_batch"] * 1e3,
            color=C_LSQ,
            lw=1.2,
            ls="-.",
            label="Batch LSQ + STM",
            zorder=4,
        )
    ax0.set_ylabel(r"$\sqrt{\mathrm{tr}(P_{rr})}$ [m]")
    ax0.set_yscale("log")
    ax0.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax0.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax0.set_title("(a) Position Uncertainty", loc="left")

    ax1.axhline(0.0, color="0.45", lw=0.6)
    ax1.plot(t_hr, hist["ll_gmm"] - hist["ll_1"], color=C_POST, lw=1.2)
    if has_batch:
        ax1.plot(t_hr, hist["ll_batch"] - hist["ll_1"], color=C_LSQ, lw=1.2, ls="-.")
    ax1.set_ylabel(r"$\Delta \overline{\log p}$ [nats]")
    ax1.set_xlabel("Time since Epoch [h]")
    ax1.set_title(
        r"(b) Mean Log-Density of MC Positions $-$ Single"
        if has_batch
        else r"(b) Mean Log-Density of MC Positions, GMM $-$ Single",
        loc="left",
    )

    fig.legend(
        loc="outside upper center",
        ncol=2 if has_batch else 3,
        columnspacing=0.8,
        handlelength=1.5,
    )
    return save_figure(fig, name)




# ============================================================
# STT BENCHMARK (J1_benchmark_STTs.py)
# ============================================================


def plot_stt_benchmark(bench, mcmc_order=None, name="fig_stt_benchmark"):
    """
    STT surrogate likelihood vs the full nonlinear one, per STT order, on posterior samples
    of an MCMC chain (full text width, three panels):
      (a) wall-clock time per log-likelihood evaluation, STT (blue) vs full nonlinear (grey),
          with the speedup t_NL / t_STT printed above each STT point;
      (b) |ln L_STT - ln L_NL| over the chain samples;
      (c) RMS over the measurements of the predicted-measurement error
          (y_STT - y_NL) / sigma, over the chain samples.
    (b) and (c): one box per order (25-75 %, median line, whiskers 5-95 %) over the
    individual samples (dots). `mcmc_order` shades the order whose surrogate generated the
    chain.

    bench: dict from J1_benchmark_STTs.benchmark_orders_on_chain (or its benchmark.npz):
      orders (n_o,), stt_ms (n_o,), nl_ms (n_o,), dlogL (n_o, n_s), dy (n_o, n_s, n_meas).
    """
    orders = np.asarray(bench["orders"], dtype=int)
    stt_ms = np.asarray(bench["stt_ms"], dtype=float)
    nl_ms = float(np.mean(bench["nl_ms"]))  # same computation for every order
    tiny = np.finfo(float).tiny
    dlogl = np.maximum(np.abs(np.asarray(bench["dlogL"], dtype=float)), tiny)
    dy = np.asarray(bench["dy"], dtype=float)
    dy_rms = np.maximum(np.sqrt(np.mean(dy * dy, axis=2)), tiny)  # per sample
    n_s = dlogl.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(TXT_W, 2.2), sharex=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    for ax in axes:
        if mcmc_order is not None and mcmc_order in orders:
            ax.axvspan(mcmc_order - 0.5, mcmc_order + 0.5, color=C_SPAN, lw=0, zorder=0)
        ax.set_yscale("log")
        ax.grid(False, axis="x")
        ax.grid(False, axis="y", which="minor")
        ax.set_xticks(orders)
        ax.set_xlim(orders.min() - 0.6, orders.max() + 0.6)
        ax.set_xlabel("STT Order [-]")
        ax.tick_params(axis="x", which="minor", bottom=False)

    # (a) cost per evaluation
    ax = axes[0]
    ax.axhline(nl_ms, color="0.78", lw=3.2, solid_capstyle="butt", zorder=2)
    ax.plot(
        orders, stt_ms, "-o", color=C_POST, lw=1.2, ms=4.0, mec="white", mew=0.5, zorder=3
    )
    for k, t in zip(orders, stt_ms):
        sp = nl_ms / t
        ax.annotate(
            rf"${sp:.0f}\times$" if sp >= 10 else rf"${sp:.1f}\times$",
            (k, t),
            xytext=(0, 4.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=C_POST,
        )
    lo = min(stt_ms.min(), nl_ms)
    hi = max(stt_ms.max(), nl_ms)
    ax.set_ylim(lo / 2.5, hi * 3.0)
    ax.set_ylabel("Time per Evaluation [ms]")
    ax.set_title("(a) Log-Likelihood Cost", loc="left")

    # (b), (c): error distributions over the chain samples
    rng = np.random.default_rng(0)
    box = dict(
        widths=0.46,
        whis=(5.0, 95.0),
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="none", edgecolor=C_POST, lw=0.8),
        medianprops=dict(color=C_POST, lw=1.3),
        whiskerprops=dict(color=C_POST, lw=0.8),
        capprops=dict(color=C_POST, lw=0.8),
        manage_ticks=False,
        zorder=3,
    )
    dot_c = _tint(C_POST, 0.45)
    for ax, data, ylabel, title in (
        (axes[1], dlogl, r"$|\Delta \ln L|$ [-]", "(b) Log-Likelihood Error"),
        (axes[2], dy_rms, r"$\mathrm{RMS}(\Delta y) / \sigma$ [-]",
         "(c) Predicted-Measurement Error"),
    ):
        for k, v in zip(orders, data):
            ax.plot(
                k + rng.uniform(-0.17, 0.17, v.size),
                v,
                ls="none",
                marker="o",
                ms=1.3,
                mew=0,
                color=dot_c,
                alpha=0.55,
                rasterized=True,
                zorder=2,
            )
        ax.boxplot(list(data), positions=orders, **box)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        lo, hi = np.log10(data.min()), np.log10(data.max())
        if hi - lo > 8:  # label every other decade
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda v, _: rf"$10^{{{round(np.log10(v))}}}$"
                    if round(np.log10(v)) % 2 == 0
                    else ""
                )
            )
        pad = 0.06 * (hi - lo)
        ax.set_ylim(10 ** (lo - pad), 10 ** (hi + pad))

    handles = [
        Line2D([], [], color=C_POST, lw=1.2, marker="o", ms=4.0, mec="white", mew=0.5),
        Line2D([], [], color="0.78", lw=3.2),
        (
            Line2D([], [], ls="none", marker="o", ms=2.0, mew=0, color=dot_c),
            Patch(facecolor="none", edgecolor=C_POST, lw=0.8),
        ),
    ]
    labels = [
        "STT Surrogate",
        "Full Nonlinear",
        f"{n_s} Chain Samples (Box: 25{EN_DASH}75$\\%$, Whiskers: 5{EN_DASH}95$\\%$)",
    ]
    if mcmc_order is not None and mcmc_order in orders:
        handles.append(Patch(facecolor=C_SPAN, lw=0))
        labels.append("Order Used for Sampling")
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(handles),
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        columnspacing=1.4,
    )
    return save_figure(fig, name)


# ============================================================
# QUICK-LOOK (legacy) MCMC figures
# ============================================================
# The original MCMCModel.plot_* figures (Tests/ and Extra/ scripts call them through the
# MCMCModel methods). They use LEGACY_STYLE and are not part of the journal figure set.


def legacy_convergence(
    model,
    idx=None,
    max_dims: int = 20,
    thin: int = 1,
    discard: int = 0,
    max_walkers_to_plot=None,
    use_x_labels: bool = True,
    save_folder: str = "results",
    save_pdf: bool = True,
    fname_prefix: str = "trace",
    ylim_quantiles=(0.005, 0.995),
    ylim_pad_frac: float = 0.12,
    show_burnin_line: bool = True,
):
    r"""
    Walker traces of the kept samples, with the median and 16-84 % band over all walkers.

    Quick-look figure (LEGACY_STYLE, shown with plt.show()), called by
    MCMCModel.plot_convergence.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run; model.samples is reshaped to (walkers, steps, ndim).
    idx : sequence of int, optional
        Parameters shown (default: the first max_dims).
    max_dims : int
        Cap on the number of parameters.
    thin, discard : int
        Extra thinning / discard applied to the kept samples [kept steps].
    max_walkers_to_plot : int, optional
        Walkers drawn (default all).
    use_x_labels : bool
        Panel labels $x_p$ (else $\theta_p$), p the 0-based parameter index.
    save_folder : str
        Folder of the PDF.
    save_pdf : bool
        Save <save_folder>/<fname_prefix>_<timestamp>.pdf.
    fname_prefix : str
    ylim_quantiles : (float, float)
        Quantiles of the drawn values that set the y-limits.
    ylim_pad_frac : float
        Padding of the y-limits (fraction of their span).
    show_burnin_line : bool
        Ignored, kept for compatibility: model.samples already starts after the burn-in,
        so there is no boundary to draw.

    Returns
    -------
    None
        Prints "Run MCMC first." and returns if there are no samples.

    Raises
    ------
    AttributeError
        If model.sampler has no nwalkers.
    ValueError
        If model.samples is not 2D or not a whole number of steps.
    """
    if getattr(model, "samples", None) is None:
        print("Run MCMC first.")
        return
    if getattr(model, "sampler", None) is None or not hasattr(
        model.sampler, "nwalkers"
    ):
        raise AttributeError("model.sampler with attribute `nwalkers` is required.")

    samples = np.asarray(model.samples)
    if samples.ndim != 2:
        raise ValueError("model.samples must be a 2D array (n_samples_total, ndim).")

    n_walkers = int(model.sampler.nwalkers)
    ndim = samples.shape[1]
    if samples.shape[0] % n_walkers != 0:
        raise ValueError(
            f"Number of rows in samples ({samples.shape[0]}) is not divisible by n_walkers ({n_walkers})."
        )

    n_steps = samples.shape[0] // n_walkers

    # reshape to (W, T, D) using emcee flattening convention
    chain = samples.reshape(n_steps, n_walkers, ndim).transpose(1, 0, 2)

    # select parameters
    if idx is None:
        idx = np.arange(min(ndim, max_dims))
    else:
        idx = np.asarray(idx, dtype=int)
        if idx.size > max_dims:
            idx = idx[:max_dims]

    # discard + thin
    discard = max(0, int(discard))
    thin = max(1, int(thin))
    chain = chain[:, discard:, :]
    chain = chain[:, ::thin, :]

    # walkers to draw
    if max_walkers_to_plot is None:
        w_plot = n_walkers
    else:
        w_plot = min(n_walkers, int(max_walkers_to_plot))
    chain_plot = chain[:w_plot, :, :]

    n_panels = int(idx.size)
    T = chain_plot.shape[1]
    x = np.arange(T)

    # grid: prefer more columns (paper-friendly)
    n_cols = int(min(5, max(2, math.ceil(math.sqrt(n_panels) + 0.5))))
    n_rows = int(math.ceil(n_panels / n_cols))

    # size: tuned for print
    fig_w = max(7.0, 2.35 * n_cols)
    fig_h = max(2.6, 1.55 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    # global style (minimal, professional)
    for ax in axes:
        ax.grid(False)
        ax.tick_params(labelsize=8, pad=2.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    for i, p in enumerate(idx):
        ax = axes[i]

        # walker traces (very light)
        for w in range(w_plot):
            ax.plot(x, chain_plot[w, :, p], alpha=0.18, linewidth=0.6)

        # median + 16–84 band across ALL walkers (post discard/thin)
        y_all = chain[:, :, p]  # (W, T)
        med = np.median(y_all, axis=0)
        q16, q84 = np.quantile(y_all, [0.16, 0.84], axis=0)
        ax.fill_between(x, q16, q84, alpha=0.12)
        ax.plot(x, med, color="black", linewidth=1.1, zorder=5)

        # label
        label = rf"$x_{{{p}}}$" if use_x_labels else rf"$\theta_{{{p}}}$"
        ax.set_ylabel(label, fontsize=9)

        # robust y-limits
        qlo, qhi = ylim_quantiles
        ylo, yhi = np.quantile(chain_plot[:, :, p], [qlo, qhi])
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            pad = float(ylim_pad_frac) * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)

        ax.margins(x=0.01)

    # hide unused axes
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    # x-label only bottom visible row
    for ax in axes[:n_panels]:
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("MCMC step", fontsize=9)

    # layout
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.10, top=0.92, wspace=0.45, hspace=0.35
    )
    fig.align_ylabels(axes[:n_panels])

    if save_pdf:
        os.makedirs(save_folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(save_folder, f"{fname_prefix}_{ts}.pdf")
        fig.savefig(out, format="pdf")
        print(f"Saved: {out}")

    plt.show()

def legacy_log_likelihood(
    model,
    save_folder: str = "results",
    save_pdf: bool = True,
    fname_prefix: str = "logposterior_trace",
    dpi: int = 300,
    alpha: float = 0.35,
    lw: float = 0.8,
    max_walkers_to_plot: int | None = None,
    thin: int = 1,
    discard: int = 0,
    show_summary: bool = True,
):
    """
    Log-posterior trace of every walker (model.log_probs), with the walker median.

    Quick-look figure (LEGACY_STYLE, shown with plt.show()), called by
    MCMCModel.plot_log_likelihood.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run.
    save_folder : str
        Folder of the PDF.
    save_pdf : bool
        Save <save_folder>/<fname_prefix>_<timestamp>.pdf.
    fname_prefix : str
    dpi : int
        savefig dpi (resolution of any rasterized content of the PDF).
    alpha, lw : float
        Opacity and width of the walker lines.
    max_walkers_to_plot : int, optional
        Walkers drawn (default all).
    thin, discard : int
        Extra thinning / discard applied to the kept samples [kept steps].
    show_summary : bool
        Draw the median over all walkers (if there is more than one).

    Returns
    -------
    None
        Prints "Run MCMC first." and returns if there are no samples.

    Raises
    ------
    AttributeError
        If model.sampler has no nwalkers.
    ValueError
        If log_probs is not a whole number of steps, or holds no finite value.
    """
    if getattr(model, "log_probs", None) is None:
        print("Run MCMC first.")
        return
    if getattr(model, "sampler", None) is None or not hasattr(
        model.sampler, "nwalkers"
    ):
        raise AttributeError("model.sampler with attribute `nwalkers` is required.")

    log_probs = np.asarray(model.log_probs).ravel()
    n_walkers = int(model.sampler.nwalkers)

    if log_probs.size % n_walkers != 0:
        raise ValueError(
            f"log_probs length ({log_probs.size}) is not divisible by n_walkers ({n_walkers}). "
            "Cannot reshape safely."
        )

    n_steps = log_probs.size // n_walkers

    # emcee-style: flatten is step-major -> (n_steps, n_walkers) then transpose
    chain = log_probs.reshape(n_steps, n_walkers).T  # (n_walkers, n_steps)

    discard = max(0, int(discard))
    thin = max(1, int(thin))
    chain = chain[:, discard:]
    chain = chain[:, ::thin]

    finite = np.isfinite(chain)
    if not np.any(finite):
        raise ValueError("All log_probs are non-finite (NaN/Inf).")

    y = chain

    # optionally limit number of walkers drawn
    plot_walkers = (
        n_walkers
        if max_walkers_to_plot is None
        else min(n_walkers, int(max_walkers_to_plot))
    )
    y_plot = y[:plot_walkers, :]
    x = np.arange(y_plot.shape[1])
    fig, ax = plt.subplots(figsize=(10, 5))

    # walker traces
    for i in range(y_plot.shape[0]):
        ax.plot(x, y_plot[i], alpha=alpha, linewidth=lw)

    # Summary overlay: median
    if show_summary and y.shape[0] > 1:
        med = np.nanmedian(y, axis=0)
        ax.plot(x, med, linewidth=1.6, color="black", label="Median (all walkers)")

    ax.set_xlabel("MCMC step [-]")
    ax.set_ylabel(r"$\log \mathcal{P}(x \mid y)$")

    # Robust y-limits for raw log-posterior values
    ylo, yhi = np.nanquantile(y_plot[np.isfinite(y_plot)], [0.01, 0.99])
    if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
        pad = 0.08 * (yhi - ylo)
        ax.set_ylim(ylo - pad, yhi + pad)

    ax.margins(x=0.01)
    if show_summary and y.shape[0] > 1:
        ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()

    if save_pdf:
        os.makedirs(save_folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(save_folder, f"{fname_prefix}_{ts}.pdf")
        fig.savefig(out, bbox_inches="tight", format="pdf", dpi=dpi)
        print(f"Saved: {out}")

    plt.show()

def legacy_postfit_residuals(model):
    """
    Normalized MAP residuals (model.residuals_func) vs measurement index, with +/-3 lines.

    Quick-look figure (LEGACY_STYLE, shown with plt.show()), called by
    MCMCModel.plot_postfit_residuals. Not saved.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run.

    Returns
    -------
    None
    """
    best_params = model.get_map_estimate()
    postfit = model.residuals_func(best_params)
    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(postfit)), postfit, marker="o")
    plt.axhline(0, color="k", linestyle="--")
    plt.axhline(3, color="r", linestyle=":")
    plt.axhline(-3, color="r", linestyle=":")
    plt.xlabel("Observation Index")
    plt.ylabel(r"Residual Normalized [$\sigma$]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def legacy_postfit_residuals_time(model, t_obs_used, opnav_data=False):
    """
    Normalized MAP residuals vs time, two panels (e.g. range / range-rate), +/-3 lines.

    Quick-look figure (LEGACY_STYLE, shown with plt.show()), called by
    MCMCModel.plot_postfit_residuals_time.
    Saved as results/postfit_residuals_<timestamp>.pdf.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run.
    t_obs_used : array_like, shape (N,)
        Epochs of the residual pairs.
    opnav_data : bool
        Label the panels "Whitened RA / DEC" instead of range / range-rate.

    Returns
    -------
    None
        Prints "Run estimation first." and returns if there are no samples.

    Raises
    ------
    RuntimeError
        If there is no MAP estimate.
    ValueError
        If the residual vector has odd length or does not match t_obs_used.

    Notes
    -----
    The residuals are read as pairs [r0, rr0, r1, rr1, ...]; exact zeros (missing
    measurements) are not drawn.
    """
    if getattr(model, "samples", None) is None:
        print("Run estimation first.")
        return

    # Compute MAP residuals
    best_params = model.get_map_estimate()
    if best_params is None:
        raise RuntimeError("MAP estimate not available.")

    postfit = np.asarray(model.residuals_func(best_params)).ravel()

    if postfit.size % 2 != 0:
        raise ValueError("Residual vector must be even-length (paired residuals).")

    # Reshape [r0, rr0, r1, rr1, ...] -> (2, N)
    residuals_matrix = postfit.reshape(-1, 2).T  # (2, N)

    time_hr = np.asarray(t_obs_used).ravel() / 3600.0
    if residuals_matrix.shape[1] != time_hr.size:
        raise ValueError("Time vector length does not match residual count.")

    # Remove zero residuals
    mask0 = residuals_matrix[0] != 0.0
    mask1 = residuals_matrix[1] != 0.0

    r0, t0 = residuals_matrix[0, mask0], time_hr[mask0]
    r1, t1 = residuals_matrix[1, mask1], time_hr[mask1]

    # Labels (explicitly whitened)
    ylabels = (
        [r"Whitened RA Residual [$\sigma$]", r"Whitened DEC Residual [$\sigma$]"]
        if opnav_data
        else [
            r"Whitened Range Residual [$\sigma$]",
            r"Whitened Range-Rate Residual [$\sigma$]",
        ]
    )

    # Figure
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Top panel
    ax0.scatter(t0, r0, s=50, alpha=0.85)
    ax0.axhline(3, color="red", linestyle=":", linewidth=3.0)
    ax0.axhline(-3, color="red", linestyle=":", linewidth=3.0)
    ax0.set_ylabel(ylabels[0])
    ax0.grid(True, linestyle=":")

    if r0.size > 0:
        lim = np.max(np.abs(np.quantile(r0, [0.01, 0.99])))
        lim = max(lim, 3.2)
        ax0.set_ylim(-1.1 * lim, 1.1 * lim)

    # Bottom panel
    ax1.scatter(t1, r1, s=50, alpha=0.85)
    ax1.axhline(3, color="red", linestyle=":", linewidth=3.0)
    ax1.axhline(-3, color="red", linestyle=":", linewidth=3.0)
    ax1.set_ylabel(ylabels[1])
    ax1.set_xlabel("Time since epoch [hours]")
    ax1.grid(True, linestyle=":")

    if r1.size > 0:
        lim = np.max(np.abs(np.quantile(r1, [0.01, 0.99])))
        lim = max(lim, 3.2)
        ax1.set_ylim(-1.1 * lim, 1.1 * lim)

    # Single clean legend (shared)
    legend_items = [
        Line2D(
            [0],
            [0],
            color="red",
            linestyle=":",
            linewidth=3.2,
            label=r"$\pm 3\sigma$",
        ),
    ]

    ax0.legend(
        handles=legend_items,
        loc="upper right",
        frameon=True,
    )

    fig.tight_layout()

    # Save vector PDF
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/postfit_residuals_{timestamp}.pdf"
    fig.savefig(fname, format="pdf", bbox_inches="tight")
    print(f"Saved: {fname}")

    plt.show()

def legacy_corner(model, use_median_as_truth=True):
    r"""
    Plain corner.corner plot of model.samples, labels $\theta_i$.

    Quick-look figure (LEGACY_STYLE, shown with plt.show()), called by
    MCMCModel.plot_corner. Not saved.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run.
    use_median_as_truth : bool
        Mark the posterior median as "truths".

    Returns
    -------
    None
        Prints "Run MCMC first." and returns if there are no samples.
    """
    if model.samples is None:
        print("Run MCMC first.")
        return

    # Use median as truth if specified
    truths = None
    if use_median_as_truth:
        truths = np.median(model.samples, axis=0)

    # Plotting corner plot
    fig = corner.corner(
        model.samples,
        labels=[f"$\\theta_{{{i}}}$" for i in range(model.ndim)],
        truths=truths,
        show_titles=True,
        title_fmt=".10f",
        title_kwargs={"fontsize": 12},
    )

    fig.set_size_inches(12, 12)
    plt.tight_layout()
    plt.show()

def legacy_corner_with_batch(
    model,
    batch_mean=None,
    batch_cov=None,
    use_median_as_truth=True,
    true_theta=None,
    plot_contours_labels=False,  # keep off unless you really need it
    batch_sigma_levels=(3.0,),
    idx=None,  # list/array of parameter indices to plot
    max_dims=20,
    bins=40,
    quantile_range=(0.005, 0.995),
    title_digits=3,
    legend_outside=True,
    rotation_matrix=None,
):
    """
    corner.corner plot of model.samples with the batch solution overlaid.

    Quick-look figure (LEGACY_STYLE, shown with plt.show()), called by
    MCMCModel.plot_corner_with_batch.
    Saved as results/corner_<timestamp>.pdf. On the 2D panels: batch mean (red dot) and
    covariance ellipses, true_theta (purple dot) and MAP (black cross). Reference lines on
    every panel: the posterior median (green) or true_theta (purple), see
    use_median_as_truth.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run.
    batch_mean : array_like, shape (ndim,), optional
    batch_cov : array_like, shape (ndim, ndim), optional
    use_median_as_truth : bool
        Reference values, drawn as corner's truth lines and included in the axis ranges:
        the posterior median (green, legend 'MCMC Median'), else true_theta (purple; none
        when true_theta is None).
    true_theta : array_like, shape (ndim,), optional
        Truth (purple dot).
    plot_contours_labels : bool
        Add labelled KDE contours to the 2D panels (slow).
    batch_sigma_levels : sequence of float
        Sigma levels of the batch ellipses; the largest also widens the axis ranges.
    idx : sequence of int, optional
        Parameters shown (default: the first max_dims).
    max_dims : int
    bins : int
    quantile_range : (float, float)
        Sample quantiles that set the axis ranges (widened to include the references).
    title_digits : int
        Significant digits of the panel titles.
    legend_outside : bool
        Legend in the empty upper-right panel.
    rotation_matrix : array_like, shape (d, d), optional
        Plot z = x R instead (samples, means, MAP; covariance R^T C R), labels $z_k$; d =
        number of parameters shown.

    Returns
    -------
    None
        Prints "Run MCMC first." and returns if there are no samples.

    Raises
    ------
    ValueError
        If there are fewer than 2 samples, or batch_cov / rotation_matrix have the wrong
        shape.
    """
    if model.samples is None:
        print("Run MCMC first.")
        return

    samples = np.asarray(model.samples)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("model.samples must be a 2D array with at least 2 rows.")

    # ----------------------------
    # Choose dimensions to plot
    # ----------------------------
    ndim = samples.shape[1]
    if idx is None:
        idx = np.arange(min(ndim, max_dims))
    else:
        idx = np.asarray(idx, dtype=int)

    s = samples[:, idx]
    d = s.shape[1]

    # ----------------------------
    # Truths & MAP on same subspace
    # ----------------------------
    best_params_full = model.get_map_estimate()
    best_params = (
        np.asarray(best_params_full)[idx] if best_params_full is not None else None
    )

    truths_full = np.median(samples, axis=0) if use_median_as_truth else true_theta
    truths = np.asarray(truths_full)[idx] if truths_full is not None else None

    # ----------------------------
    # Labels: x_i
    # ----------------------------
    labels = (
        [rf"$z_{{{k}}}$" for k in idx]
        if rotation_matrix is not None
        else [rf"$x_{{{k}}}$" for k in idx]
    )

    # ----------------------------
    # Batch / True on same subspace
    # ----------------------------
    bm = np.asarray(batch_mean)[idx] if batch_mean is not None else None

    bc = None
    if batch_cov is not None:
        batch_cov = np.asarray(batch_cov)
        bc = batch_cov[np.ix_(idx, idx)]
        if bc.shape != (d, d):
            raise ValueError("batch_cov shape is inconsistent with selected idx.")

    tt = np.asarray(true_theta)[idx] if true_theta is not None else None

    # ----------------------------
    # Rotate Everything if requested
    # ----------------------------
    if rotation_matrix is not None:
        R = np.asarray(rotation_matrix)
        if R.shape != (d, d):
            raise ValueError(
                f"rotation_matrix must have shape {(d, d)}, got {R.shape}"
            )
        s = s @ R
        if best_params is not None:
            best_params = best_params @ R
        if truths is not None:
            truths = truths @ R
        if bm is not None:
            bm = bm @ R
        if batch_cov is not None:
            bc = R.T @ bc @ R
        if tt is not None:
            tt = tt @ R

    # ----------------------------
    # Robust range + ensure EVERYTHING is visible
    # ----------------------------
    qlo, qhi = quantile_range
    ranges = []
    ksig_max = float(np.max(batch_sigma_levels)) if len(batch_sigma_levels) else 0.0

    for j in range(d):
        lo, hi = np.quantile(s[:, j], [qlo, qhi])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            lo, hi = float(np.min(s[:, j])), float(np.max(s[:, j]))

        extras = []
        if bm is not None and np.isfinite(bm[j]):
            extras.append(float(bm[j]))
        if tt is not None and np.isfinite(tt[j]):
            extras.append(float(tt[j]))
        if truths is not None and np.isfinite(truths[j]):
            extras.append(float(truths[j]))
        if best_params is not None and np.isfinite(best_params[j]):
            extras.append(float(best_params[j]))

        if bc is not None and np.isfinite(bc[j, j]) and bc[j, j] >= 0:
            sig = ksig_max * float(np.sqrt(bc[j, j]))
            if bm is not None and np.isfinite(bm[j]) and sig > 0:
                extras.extend([float(bm[j] - sig), float(bm[j] + sig)])

        if extras:
            lo = min(lo, np.min(extras))
            hi = max(hi, np.max(extras))

        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        pad = 0.03 * span
        ranges.append((lo - pad, hi + pad))

    # ----------------------------
    # Figure size scaling with d
    # ----------------------------
    inches_per_dim = 1.35
    fig_size = max(8.0, inches_per_dim * d)
    # levels_2d_sigma = [0.393, 0.865, 0.989]  # 1σ,2σ,3σ in 2D (Gaussian-equivalent)

    fig = corner.corner(
        s,
        labels=labels,
        truths=truths,
        truth_color="green" if use_median_as_truth else "purple",
        range=ranges,
        bins=bins,
        show_titles=True,
        title_fmt=f".{title_digits}g",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        color="tab:blue",
        plot_contours=True,
        fill_contours=False,
        # levels=levels_2d_sigma,
        smooth=1.0,
        smooth1d=1.0,
    )

    axes = np.array(fig.axes).reshape((d, d))

    # ----------------------------
    # Titles: position below the top edge to avoid overlap with plot above
    # ----------------------------
    for k in range(d):
        axes[k, k].title.set_y(1.15)  # reduced from 1.30 - rely on hspace instead
        axes[k, k].title.set_fontsize(9)

    # ----------------------------
    # Overlay batch ellipses / points
    # ----------------------------
    for i in range(d):
        for j in range(i):
            ax = axes[i, j]

            if bm is not None:
                ax.plot(bm[j], bm[i], "o", ms=4.5, color="red", zorder=6)

            if bm is not None and bc is not None and np.all(np.isfinite(bc)):
                cov_sub = bc[np.ix_([j, i], [j, i])]
                mean_sub = [bm[j], bm[i]]

                cov_sub = 0.5 * (cov_sub + cov_sub.T)
                vals, vecs = np.linalg.eigh(cov_sub)
                vals = np.maximum(vals, 0.0)

                if np.max(vals) > 0:
                    order = np.argsort(vals)[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

                    for ksig in batch_sigma_levels:
                        width, height = 2.0 * float(ksig) * np.sqrt(vals)
                        ell = Ellipse(
                            xy=mean_sub,
                            width=width,
                            height=height,
                            angle=angle,
                            edgecolor="red",
                            facecolor="none",
                            lw=1.4 if float(ksig) == 3.0 else 1.0,
                            linestyle="-" if float(ksig) == 3.0 else "--",
                            alpha=0.95,
                            zorder=5,
                        )
                        ax.add_patch(ell)

            if tt is not None:
                ax.plot(tt[j], tt[i], "o", ms=4.5, color="purple", zorder=7)

            if best_params is not None:
                ax.plot(
                    best_params[j],
                    best_params[i],
                    "x",
                    ms=6,
                    mew=1.6,
                    color="black",
                    zorder=8,
                )

            if plot_contours_labels:
                x = s[:, j]
                y = s[:, i]
                data = np.vstack([x, y])
                kde = gaussian_kde(data)

                xi, yi = np.mgrid[
                    ranges[j][0] : ranges[j][1] : 100j,
                    ranges[i][0] : ranges[i][1] : 100j,
                ]
                zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

                levels_mass = [0.118, 0.393, 0.675, 0.864]
                sorted_zi = np.sort(zi.ravel())[::-1]
                cdf = np.cumsum(sorted_zi)
                cdf /= cdf[-1]
                value_levels = []
                for lv in levels_mass:
                    k0 = int(np.searchsorted(cdf, lv))
                    k0 = np.clip(k0, 0, len(sorted_zi) - 1)
                    value_levels.append(sorted_zi[k0])
                value_levels = sorted(set(value_levels))

                cs = ax.contour(
                    xi, yi, zi, levels=value_levels, colors="black", linewidths=1
                )
                fmt = {
                    lvl: f"{int(100*m)}\\%"
                    for lvl, m in zip(cs.levels, levels_mass[: len(cs.levels)])
                }
                ax.clabel(cs, fmt=fmt, inline=True, fontsize=8)

    # ----------------------------
    # Global formatting:
    # Increase spacing to prevent overlaps
    # ----------------------------

    for ax in fig.get_axes():
        # Increase tick label size and padding significantly
        ax.tick_params(labelsize=8, pad=8.0)  # tick numbers away from axis

    # ----------------------------
    # Legend proxies
    # ----------------------------
    proxies = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="red",
            markersize=6,
            label="Batch Mean",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            lw=1.4,
            linestyle="-",
            label=rf"Batch {float(batch_sigma_levels[0]):g}$\sigma$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="purple",
            markersize=6,
            label="True Value",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            markersize=7,
            lw=0,
            label="MCMC MAP",
        ),
        Line2D([0], [0], color="green", lw=1.2, label="MCMC Median"),
    ]

    if len(batch_sigma_levels) > 1:
        for ksig in batch_sigma_levels[1:]:
            proxies.insert(
                2,
                Line2D(
                    [0],
                    [0],
                    color="red",
                    lw=1.0,
                    linestyle="--",
                    label=rf"Batch {float(ksig):g}$\sigma$",
                ),
            )

    filtered = []
    for p in proxies:
        lab = p.get_label()
        if "Batch" in lab:
            if bm is not None:
                filtered.append(p)
        elif lab == "True Value":
            if tt is not None:
                filtered.append(p)
        elif lab == "MCMC MAP":
            if best_params is not None:
                filtered.append(p)
        elif lab == "MCMC Median":
            if use_median_as_truth and truths is not None:
                filtered.append(p)
        else:
            filtered.append(p)

    # More breathing room with significantly increased spacing between plots
    fig.set_size_inches(fig_size, fig_size)
    fig.subplots_adjust(
        wspace=0.20, hspace=0.30, right=0.82, top=0.93, bottom=0.12, left=0.12
    )  # margins back to 0.12

    # Force a draw so text objects have final positions/sizes
    fig.canvas.draw()

    # Move ONLY the labels that exist in a corner plot:
    #   x-labels: bottom row
    #   y-labels: left column
    x_y = -0.70  # more negative -> farther down
    y_x = -0.70  # more negative -> farther left

    for j in range(d):
        ax = axes[d - 1, j]  # bottom row
        if ax.get_xlabel():
            ax.xaxis.set_label_coords(0.5, x_y)  # (x in [0,1], y in axes coords)

    for i in range(d):
        ax = axes[i, 0]  # left column
        if ax.get_ylabel():
            ax.yaxis.set_label_coords(y_x, 0.5)

    # Legend in upper-right empty triangle
    if legend_outside and len(filtered) > 0:
        leg_ax = axes[0, d - 1]
        leg_ax.axis("off")
        leg_ax.legend(
            handles=filtered,
            loc="center",
            frameon=True,
            fontsize=13,
            handlelength=2.6,
            labelspacing=0.8,
            borderpad=0.9,
        )

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/corner_{ts}.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.25)
    print(f"Saved: {out}")

    plt.show()

def legacy_marginals_overlaid(
    model,
    idx=None,
    max_dims=20,
    bins=80,
    quantile_range=(0.005, 0.995),
    rotation_matrix=None,
    density=True,
    log_y=False,
    alpha=0.28,
    linewidth=1.5,
    show_kde=False,
    figsize=(10, 6),
    labels=None,
    center=False,
):
    """
    1D marginal histograms of several parameters overlaid on one axis.

    Quick-look figure (LEGACY_STYLE, shown with plt.show()), called by
    MCMCModel.plot_marginals_overlaid.
    Saved as results/marginals_overlaid_<timestamp>.pdf.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run.
    idx : sequence of int, optional
        Parameters shown (default: the first max_dims).
    max_dims : int
    bins : int
    quantile_range : (float, float)
        Quantiles that set the common x-range (5 % padding).
    rotation_matrix : array_like, shape (d, d), optional
        Histogram z = x R instead, labels $z_k$.
    density : bool
        Normalized histograms (else counts).
    log_y : bool
        Logarithmic y axis.
    alpha, linewidth : float
        Fill opacity and outline width.
    show_kde : bool
        Add a Gaussian KDE line per parameter.
    figsize : (float, float)
    labels : list of str, optional
        Legend entries (default $x_k$ / $z_k$).
    center : bool
        Subtract each parameter's mean.

    Returns
    -------
    None
        Prints "Run MCMC first." and returns if there are no samples.

    Raises
    ------
    ValueError
        If there are fewer than 2 samples, or rotation_matrix has the wrong shape.
    """
    if model.samples is None:
        print("Run MCMC first.")
        return

    samples = np.asarray(model.samples)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("model.samples must be a 2D array with at least 2 rows.")

    # ----------------------------
    # Choose dimensions
    # ----------------------------
    ndim = samples.shape[1]
    if idx is None:
        idx = np.arange(min(ndim, max_dims))
    else:
        idx = np.asarray(idx, dtype=int)

    s = samples[:, idx]
    d = s.shape[1]

    # ----------------------------
    # Labels
    # ----------------------------
    if labels is None:
        labels = (
            [rf"$z_{{{k}}}$" for k in idx]
            if rotation_matrix is not None
            else [rf"$x_{{{k}}}$" for k in idx]
        )

    # ----------------------------
    # Rotate if requested
    # ----------------------------
    if rotation_matrix is not None:
        R = np.asarray(rotation_matrix)
        if R.shape != (d, d):
            raise ValueError(
                f"rotation_matrix must have shape {(d, d)}, got {R.shape}"
            )
        s = s @ R

    # ----------------------------
    # Global x-range from all marginals
    # ----------------------------
    qlo, qhi = quantile_range
    lo_all = []
    hi_all = []

    for j in range(d):
        x = s[:, j]

        if center:
            mu = np.mean(x)
            x = x - mu

        lo, hi = np.quantile(x, [qlo, qhi])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            lo, hi = float(np.min(s[:, j])), float(np.max(s[:, j]))
        lo_all.append(lo)
        hi_all.append(hi)

    xmin = min(lo_all)
    xmax = max(hi_all)
    span = xmax - xmin
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    pad = 0.05 * span
    xmin -= pad
    xmax += pad

    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, d))

    for j in range(d):
        x = s[:, j]

        if center:
            mu = np.mean(x)
            x = x - mu

        ax.hist(
            x,
            bins=bins,
            range=(xmin, xmax),
            density=density,
            histtype="stepfilled",
            alpha=alpha,
            linewidth=linewidth,
            label=labels[j],
            color=colors[j],
        )

        if show_kde and np.std(x) > 0:
            try:
                kde = gaussian_kde(x)
                xx = np.linspace(xmin, xmax, 500)
                yy = kde(xx)
                ax.plot(xx, yy, linewidth=1.5)
            except Exception:
                pass

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Centered Value" if center else "Parameter Value", fontsize=11)
    ax.set_ylabel("Density" if density else "Count", fontsize=11)

    if log_y:
        ax.set_yscale("log")

    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, ncol=2, frameon=True)

    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/marginals_overlaid_{ts}.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.2)
    print(f"Saved: {out}")

    plt.show()

def legacy_autocorrelation(
    model,
    idx=None,  # parameters to include (default: all, capped)
    max_dims: int = 20,  # max number of parameters shown
    max_lag: int = 500,
    thin: int = 1,
    discard: int = 0,
    save_folder: str = "results",
    save_pdf: bool = True,
    fname_prefix: str = "acf",
):
    """
    Walker-averaged autocorrelation (statsmodels acf) of each parameter vs lag.

    Quick-look figure (LEGACY_STYLE, shown with plt.show()), called by
    MCMCModel.plot_autocorrelation. A dashed line in each curve's colour marks its
    integrated autocorrelation time tau_int, estimated on the displayed chain (after
    discard/thin) the way emcee.autocorr.integrated_time does (walker-mean ACF, Sokal
    window c = 5), in displayed lags; a tau beyond max_lag (or NaN) is not drawn.

    Parameters
    ----------
    model : MCMC.MCMCModel
        After run; model.samples is reshaped to (walkers, steps, ndim).
    idx : sequence of int, optional
        Parameters shown (default: the first max_dims); labels $x_{p+1}$.
    max_dims : int
    max_lag : int
        Largest lag [kept steps] (capped by the chain length).
    thin, discard : int
        Extra thinning / discard applied to the kept samples [kept steps].
    save_folder : str
        Folder of the PDF.
    save_pdf : bool
        Save <save_folder>/<fname_prefix>_<timestamp>.pdf.
    fname_prefix : str

    Returns
    -------
    None
        Prints "Run MCMC first." and returns if there are no samples.

    Raises
    ------
    AttributeError
        If model.sampler is None.
    ValueError
        If model.samples is not a whole number of steps.
    """

    if getattr(model, "samples", None) is None:
        print("Run MCMC first.")
        return
    if getattr(model, "sampler", None) is None:
        raise AttributeError("model.sampler required.")

    samples = np.asarray(model.samples)
    n_walkers = int(model.sampler.nwalkers)
    ndim = samples.shape[1]

    if samples.shape[0] % n_walkers != 0:
        raise ValueError("Samples cannot be reshaped consistently.")

    n_steps = samples.shape[0] // n_walkers

    # Correct reshape (emcee step-major flattening)
    chain = samples.reshape(n_steps, n_walkers, ndim).transpose(1, 0, 2)

    discard = max(0, int(discard))
    thin = max(1, int(thin))
    chain = chain[:, discard:, :]
    chain = chain[:, ::thin, :]

    # Cap max_lag to available steps
    max_lag = min(int(max_lag), chain.shape[1] - 1)
    lags = np.arange(max_lag + 1)

    # Parameter selection
    if idx is None:
        idx = np.arange(min(ndim, max_dims))
    else:
        idx = np.asarray(idx, dtype=int)
        if idx.size > max_dims:
            idx = idx[:max_dims]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    # Plot ACF curves + tau lines. The full-length walker-mean ACF gives tau_int with
    # emcee's estimator: tau(M) = 2 sum_{k<=M} rho_k - 1, Sokal window M >= 5 tau(M).
    n_disp = chain.shape[1]
    any_tau = False
    for p in idx:
        acf_full = np.mean(
            [acf(chain[w, :, p], nlags=n_disp - 1, fft=True) for w in range(chain.shape[0])],
            axis=0,
        )
        (line,) = ax.plot(
            lags, acf_full[: max_lag + 1], linewidth=1.6, label=rf"$x_{{{p+1}}}$"
        )
        taus = 2.0 * np.cumsum(acf_full) - 1.0
        in_window = np.arange(taus.size) < 5.0 * taus
        tau_p = taus[np.argmin(in_window)] if np.any(in_window) else taus[-1]
        if np.isfinite(tau_p) and tau_p <= max_lag:
            ax.axvline(tau_p, color=line.get_color(), ls="--", lw=1.0, alpha=0.8)
            any_tau = True

    ax.set_xlabel("Lag [-]")
    ax.set_ylabel("Autocorrelation [-]")
    ax.set_xlim(0, max_lag)
    ax.set_ylim(-0.05, 1.05)

    # legend: keep parameter curves + ONE entry describing tau lines
    handles, labels = ax.get_legend_handles_labels()
    if any_tau:
        handles.append(Line2D([0], [0], color="0.3", ls="--", lw=1.0))
        labels.append(r"$\tau_{\mathrm{int}}$")
    ax.legend(handles, labels, loc="upper right", frameon=True)
    fig.tight_layout()

    if save_pdf:
        os.makedirs(save_folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(save_folder, f"{fname_prefix}_{ts}.pdf")
        fig.savefig(out, bbox_inches="tight", format="pdf")
        print(f"Saved: {out}")

    plt.show()
