import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from tqdm import tqdm

# Define observation epoch
t = Time("2025-08-20")

# Get Earth's barycentric position and velocity in heliocentric frame
earth = get_body_barycentric_posvel("earth", t)
P_earth = np.array([earth[0].x.value, earth[0].y.value, earth[0].z.value])
dot_P_earth = np.array([earth[1].x.value, earth[1].y.value, earth[1].z.value])

# Example attributable for a spacecraft near an asteroid
alpha = 180 * np.pi / 180
delta = 0 * np.pi / 180
dot_alpha = 0.05 * np.pi / 180
dot_delta = 0.01 * np.pi / 180

# Compute unit vector (hat_r) from Earth to spacecraft
hat_r = np.array(
    [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)]
)

# Compute time derivative of unit vector (dot_hat_r)
dot_hat_r = np.array(
    [
        -np.cos(delta) * np.sin(alpha) * dot_alpha
        - np.sin(delta) * np.cos(alpha) * dot_delta,
        np.cos(delta) * np.cos(alpha) * dot_alpha
        - np.sin(delta) * np.sin(alpha) * dot_delta,
        np.cos(delta) * dot_delta,
    ]
)

# Compute angular proper motion magnitude (eta)
eta = np.sqrt(dot_alpha**2 * np.cos(delta) ** 2 + dot_delta**2)

# Define astronomical constants
k = 0.01720209895
mu_earth = 1 / 328900.5614

# Compute coefficients for energy constraint equations
c0 = np.dot(P_earth, P_earth)
c5 = 2 * np.dot(P_earth, hat_r)
c2 = eta**2
c3 = 2 * np.dot(dot_P_earth, dot_hat_r)
c4 = np.dot(dot_P_earth, dot_P_earth)
c1 = 2 * np.dot(dot_P_earth, hat_r)


# Function to compute heliocentric dot_rho bounds for a given rho
def get_helio_dot_r_bounds(r):
    S = r**2 + c5 * r + c0
    if S <= 0:
        return None, None
    sqrt_S = np.sqrt(S)
    C_r = c2 * r**2 + c3 * r + c4
    quad_C = C_r - 2 * k**2 / sqrt_S
    quad_B = c1
    quad_A = 1.0
    disc = quad_B**2 - 4 * quad_A * quad_C
    if disc < 0:
        return None, None
    sqrt_disc = np.sqrt(disc)
    dot_r_min = (-quad_B - sqrt_disc) / (2 * quad_A)
    dot_r_max = (-quad_B + sqrt_disc) / (2 * quad_A)
    return dot_r_min, dot_r_max


# Function to get allowed intervals for dot_rho
def get_allowed_intervals(r):
    h_min, h_max = get_helio_dot_r_bounds(r)
    if h_min is None:
        return []
    G_r = 2 * k**2 * mu_earth / r - eta**2 * r**2
    allowed = []
    if G_r <= 0:
        allowed.append((h_min, h_max))
    else:
        geo = np.sqrt(G_r)
        left_max = min(h_max, -geo)
        if h_min < left_max:
            allowed.append((h_min, left_max))
        right_min = max(h_min, geo)
        if right_min < h_max:
            allowed.append((right_min, h_max))
    return allowed


# Function to compute geocentric constraint bounds (for plotting)
def get_dot_r_geo_bounds(r):
    G_r = 2 * k**2 * mu_earth / r - eta**2 * r**2
    if G_r > 0:
        dot_r_geo = np.sqrt(G_r)
        return -dot_r_geo, dot_r_geo
    return None, None


# Function to convert (rho, dot_rho) to heliocentric (x, y, z, xdot, ydot, zdot)
def rho_to_cartesian(rho, dot_rho):
    pos = P_earth + rho * hat_r
    vel = dot_P_earth + dot_rho * hat_r + rho * dot_hat_r
    return pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]


# Sample rho values for boundary
rho_min = 0.001
rho_max = 1
num_samples_plot = 200
desired_num_triangles = 50
num_samples_tri = desired_num_triangles // 4 + 1
rho_list_plot = np.linspace(rho_min, rho_max, num_samples_plot)
rho_list_tri = np.linspace(rho_min, rho_max, num_samples_tri)

# Collect boundary points for plotting
upper_min_points_plot = []
upper_max_points_plot = []
lower_min_points_plot = []
lower_max_points_plot = []
for rho in tqdm(rho_list_plot, desc="Computing plot boundaries"):
    intervals = get_allowed_intervals(rho)
    for dmin, dmax in intervals:
        if dmin < 0:
            l_max = min(dmax, 0)
            if dmin < l_max:
                lower_min_points_plot.append([rho, dmin])
                lower_max_points_plot.append([rho, l_max])
        if dmax > 0:
            u_min = max(dmin, 0)
            if u_min < dmax:
                upper_min_points_plot.append([rho, u_min])
                upper_max_points_plot.append([rho, dmax])

# Collect boundary points for triangulation
upper_min_points_tri = []
upper_max_points_tri = []
lower_min_points_tri = []
lower_max_points_tri = []
for rho in rho_list_tri:
    intervals = get_allowed_intervals(rho)
    for dmin, dmax in intervals:
        if dmin < 0:
            l_max = min(dmax, 0)
            if dmin < l_max:
                lower_min_points_tri.append([rho, dmin])
                lower_max_points_tri.append([rho, l_max])
        if dmax > 0:
            u_min = max(dmin, 0)
            if u_min < dmax:
                upper_min_points_tri.append([rho, u_min])
                upper_max_points_tri.append([rho, dmax])

# Convert to arrays for plotting
upper_rho_plot = None
upper_min_dot_plot = None
upper_max_dot_plot = None
lower_rho_plot = None
lower_min_dot_plot = None
lower_max_dot_plot = None
if upper_min_points_plot:
    upper_min_plot = np.array(upper_min_points_plot)
    upper_max_plot = np.array(upper_max_points_plot)
    upper_rho_plot = upper_min_plot[:, 0]
    upper_min_dot_plot = upper_min_plot[:, 1]
    upper_max_dot_plot = upper_max_plot[:, 1]
if lower_min_points_plot:
    lower_min_plot = np.array(lower_min_points_plot)
    lower_max_plot = np.array(lower_max_points_plot)
    lower_rho_plot = lower_min_plot[:, 0]
    lower_min_dot_plot = lower_min_plot[:, 1]
    lower_max_dot_plot = lower_max_plot[:, 1]

# Convert to arrays for triangulation
upper_min_tri = np.array(upper_min_points_tri) if upper_min_points_tri else None
upper_max_tri = np.array(upper_max_points_tri) if upper_max_points_tri else None
lower_min_tri = np.array(lower_min_points_tri) if lower_min_points_tri else None
lower_max_tri = np.array(lower_max_points_tri) if lower_max_points_tri else None

# Compute triangle centroids
centroids_list = []
cartesian_coords = []
branches = [
    ("upper", upper_min_tri, upper_max_tri),
    ("lower", lower_min_tri, lower_max_tri),
]
for branch_name, min_tri, max_tri in branches:
    if min_tri is None or len(min_tri) < 2:
        continue
    m = len(min_tri)
    for i in range(m - 1):
        p1 = min_tri[i]
        p2 = min_tri[i + 1]
        p3 = max_tri[i + 1]
        p4 = max_tri[i]
        cent1 = np.mean([p1, p2, p4], axis=0)
        centroids_list.append(cent1)
        x, y, z, xdot, ydot, zdot = rho_to_cartesian(cent1[0], cent1[1])
        cartesian_coords.append([x, y, z, xdot, ydot, zdot])
        cent2 = np.mean([p2, p3, p4], axis=0)
        centroids_list.append(cent2)
        x, y, z, xdot, ydot, zdot = rho_to_cartesian(cent2[0], cent2[1])
        cartesian_coords.append([x, y, z, xdot, ydot, zdot])
centroids = np.array(centroids_list) if centroids_list else np.array([])
cartesian_coords = np.array(cartesian_coords) if cartesian_coords else np.array([])

# Plot the admissible region
fig, ax = plt.subplots(figsize=(10, 6))

# Plot branch boundaries
if upper_rho_plot is not None:
    ax.plot(
        upper_rho_plot,
        upper_max_dot_plot,
        "b-",
        label="Upper Branch (Heliocentric E_s ≤ 0)",
        linewidth=2,
    )
if lower_rho_plot is not None:
    ax.plot(
        lower_rho_plot,
        lower_min_dot_plot,
        "r-",
        label="Lower Branch (Heliocentric E_s ≤ 0)",
        linewidth=2,
    )

# Plot geocentric constraint bounds
rho_fine = np.linspace(rho_min, rho_max, num_samples_plot)
rho_geo = []
geo_min_plot = []
geo_max_plot = []
for rho in rho_fine:
    dot_r_geo_min, dot_r_geo_max = get_dot_r_geo_bounds(rho)
    if dot_r_geo_min is not None:
        rho_geo.append(rho)
        geo_min_plot.append(dot_r_geo_min)
        geo_max_plot.append(dot_r_geo_max)
if rho_geo:
    ax.plot(
        rho_geo, geo_max_plot, "g--", label="Geocentric Constraint (E_e ≥ 0)", alpha=0.7
    )
    ax.plot(rho_geo, geo_min_plot, "g--", alpha=0.7)

# Plot admissible region (filled)
filled = False
if upper_rho_plot is not None:
    x_upper = np.concatenate((upper_rho_plot, upper_rho_plot[::-1]))
    y_upper = np.concatenate((upper_min_dot_plot, upper_max_dot_plot[::-1]))
    ax.fill(x_upper, y_upper, "gray", alpha=0.2, label="Admissible Region")
    filled = True
if lower_rho_plot is not None:
    x_lower = np.concatenate((lower_rho_plot, lower_rho_plot[::-1]))
    y_lower = np.concatenate((lower_min_dot_plot, lower_max_dot_plot[::-1]))
    ax.fill(
        x_lower,
        y_lower,
        "gray",
        alpha=0.2,
        label="Admissible Region" if not filled else None,
    )

# Plot triangulation lines
for branch_name, min_tri, max_tri in branches:
    if min_tri is None or len(min_tri) < 2:
        continue
    rho_tri = min_tri[:, 0]
    min_dot_tri = min_tri[:, 1]
    max_dot_tri = max_tri[:, 1]
    m = len(rho_tri)
    for i in range(m - 1):
        rho_i = rho_tri[i]
        rho_ip1 = rho_tri[i + 1]
        ax.plot([rho_i, rho_i], [min_dot_tri[i], max_dot_tri[i]], "k-", linewidth=1)
        ax.plot(
            [rho_ip1, rho_ip1],
            [min_dot_tri[i + 1], max_dot_tri[i + 1]],
            "k-",
            linewidth=1,
        )
        ax.plot(
            [rho_ip1, rho_i], [min_dot_tri[i + 1], max_dot_tri[i]], "k-", linewidth=1
        )
        ax.plot(
            [rho_i, rho_ip1], [min_dot_tri[i], min_dot_tri[i + 1]], "k-", linewidth=1
        )
        ax.plot(
            [rho_i, rho_ip1], [max_dot_tri[i], max_dot_tri[i + 1]], "k-", linewidth=1
        )

# Plot centroids
if len(centroids) > 0:
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="orange",
        marker="o",
        s=50,
        label="Triangle Centroids (Virtual Asteroids)",
    )

# Customize plot
ax.set_xlabel("Range (AU)", fontsize=12)
ax.set_ylabel("Range-Rate (AU/day)", fontsize=12)
ax.set_title(
    "Admissible Region for Spacecraft near Asteroid",
    fontsize=14,
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.5)

# Set limits
all_dot_r = []
if upper_min_dot_plot is not None:
    all_dot_r.extend(upper_min_dot_plot)
    all_dot_r.extend(upper_max_dot_plot)
if lower_min_dot_plot is not None:
    all_dot_r.extend(lower_min_dot_plot)
    all_dot_r.extend(lower_max_dot_plot)
if all_dot_r:
    rho_margin = 0.05 * (rho_max - rho_min)
    ax.set_xlim(rho_min - rho_margin, rho_max + rho_margin)
    dot_r_min_val = min(all_dot_r)
    dot_r_max_val = max(all_dot_r)
    dot_r_margin = 0.1 * (dot_r_max_val - dot_r_min_val)
    ax.set_ylim(dot_r_min_val - dot_r_margin, dot_r_max_val + dot_r_margin)
plt.tight_layout()
plt.show()

# Print results
num_triangles = len(cartesian_coords)
print(f"Number of triangles: {num_triangles}")
print(
    "\nCartesian coordinates of triangle centroids (x, y, z, xdot, ydot, zdot) in AU, AU/day:"
)
for i, coords in enumerate(cartesian_coords):
    print(
        f"Centroid {i+1}: x={coords[0]:.6f}, y={coords[1]:.6f}, z={coords[2]:.6f}, "
        f"xdot={coords[3]:.6f}, ydot={coords[4]:.6f}, zdot={coords[5]:.6f}"
    )
