import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

# ============================================================
# Publication-ready style
# ============================================================
plt.rcParams.update(
    {
        "figure.figsize": (7.4, 4.8),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "lines.linewidth": 2.2,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "text.usetex": True,
    }
)

# ============================================================
# Colorblind-safe palette (Okabe–Ito)
# ============================================================
COLORS = {
    "true": "#000000",  # black
    "low": "#0072B2",  # blue
    "mid": "#E69F00",  # orange
    "high": "#D55E00",  # vermillion
}


# ============================================================
# Runge function
# ============================================================
def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def interpolant(n, x_eval):
    x_nodes = np.linspace(-1.0, 1.0, n + 1)
    y_nodes = f(x_nodes)
    interp = BarycentricInterpolator(x_nodes, y_nodes)
    return interp(x_eval)


# ============================================================
# Domain (ZOOMED OUT)
# ============================================================
x = np.linspace(-1.5, 1.5, 6000)
y_true = f(x)

# Orders (stronger contrast)
orders = [3, 7, 10]

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots()

# True function
ax.plot(x, y_true, color=COLORS["true"], lw=2.6, label="True function")

# Interpolants
ax.plot(
    x,
    interpolant(orders[0], x),
    color=COLORS["low"],
    ls="--",
    label="Low order (3rd degree)",
)

ax.plot(
    x,
    interpolant(orders[1], x),
    color=COLORS["mid"],
    ls=":",
    label="Medium order (7th degree)",
)

ax.plot(
    x,
    interpolant(orders[2], x),
    color=COLORS["high"],
    ls="-",
    label="High order (10th degree)",
)

# Reference (center)
ax.axvline(0.0, color="0.7", lw=1.0, ls=(0, (4, 3)))

# Labels
ax.set_xlabel(r"State Deviation")
ax.set_ylabel(r"Function Value")

# Limits (zoomed out to expose oscillations)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.2, 1.2)

# Grid
ax.grid(True, alpha=0.2)

# Legend
leg = ax.legend(loc="lower center", frameon=True)
leg.get_frame().set_edgecolor("0.8")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

# Save
plt.savefig("results/runge_zoomed_colorblind.pdf")

plt.show()
