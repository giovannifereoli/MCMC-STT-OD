import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Publication-ready style
# ============================================================
plt.rcParams.update(
    {
        "figure.figsize": (8.5, 4.5),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "text.usetex": True,
    }
)

# ============================================================
# Colorblind-safe palette (Okabe–Ito)
# ============================================================
COLORS = {
    "true": "#000000",
    "n1": "#0072B2",
    "n3": "#009E73",
    "n7": "#D55E00",
    "n8": "#CC79A7",
    "ref": "#666666",
    "lim": "#999999",
}


# ============================================================
# Function and Taylor series
# ============================================================
def f(x):
    return 1.0 / x


def taylor_partial_sum(x, N):
    z = x - 1.0
    y = np.zeros_like(x)
    for n in range(N + 1):
        y += (-1.0) ** n * z**n
    return y


# Domain
x = np.linspace(0.05, 3.2, 4000)
y_true = f(x)

# Orders
orders = [1, 3, 7, 8]

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots()

# True function
(line_true,) = ax.plot(x, y_true, color=COLORS["true"], lw=2.6, label=r"$f(x)=1/x$")

# Taylor approximations
(line_n1,) = ax.plot(
    x,
    taylor_partial_sum(x, 1),
    color=COLORS["n1"],
    ls="--",
    label="Order 1 about $x_0=1$",
)

(line_n3,) = ax.plot(
    x,
    taylor_partial_sum(x, 3),
    color=COLORS["n3"],
    ls="-.",
    label="Order 3 about $x_0=1$",
)

(line_n7,) = ax.plot(
    x,
    taylor_partial_sum(x, 7),
    color=COLORS["n7"],
    ls="-",
    label="Order 7 about $x_0=1$",
)

(line_n8,) = ax.plot(
    x,
    taylor_partial_sum(x, 8),
    color=COLORS["n8"],
    ls=":",
    label="Order 8 about $x_0=1$",
)

# ============================================================
# Vertical lines
# ============================================================
ax.axvline(1.0, color=COLORS["ref"], lw=1.2, ls=(0, (4, 3)))

# Custom legend handles for vertical lines
ref_handle = Line2D(
    [0], [0], color=COLORS["ref"], lw=1.5, ls=(0, (4, 3)), label=r"Reference $x_0=1$"
)


# ============================================================
# Labels and limits
# ============================================================
ax.set_xlabel(r"State Deviation")
ax.set_ylabel(r"Function Value")

ax.set_xlim(0.05, 3.2)
ax.set_ylim(-10, 10)

ax.grid(True, alpha=0.18)

# ============================================================
# Legend (upper right, ordered)
# ============================================================
handles = [line_true, line_n1, line_n3, line_n7, line_n8, ref_handle]

leg = ax.legend(handles=handles, loc="lower left", frameon=True)
leg.get_frame().set_edgecolor("0.8")
leg.get_frame().set_linewidth(0.8)

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# Save
plt.savefig("results/taylor_radius_with_legend.pdf")

plt.show()
