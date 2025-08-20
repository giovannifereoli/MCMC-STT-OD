# Import required libraries
import numpy as np  # For numerical computations, array handling, and vector operations
import matplotlib.pyplot as plt  # For plotting the admissible region and triangulation
from astropy.coordinates import (
    get_body_barycentric_posvel,
)  # To get Earth's position/velocity
from astropy.time import Time  # To handle observation epoch
from tqdm import tqdm  # For progress bars during boundary computation
from scipy.spatial import (
    Delaunay,
)  # For Delaunay triangulation of the admissible region

# Define observation epoch
t = Time("2025-08-20")  # Observation date (example: current date)

# Get Earth's barycentric position and velocity in heliocentric frame
earth = get_body_barycentric_posvel("earth", t)
P_earth = np.array(
    [earth[0].x.value, earth[0].y.value, earth[0].z.value]
)  # Earth's position (AU)
dot_P_earth = np.array(
    [earth[1].x.value, earth[1].y.value, earth[1].z.value]
)  # Earth's velocity (AU/day)

# Example attributable for a spacecraft near an asteroid (angles and rates from Earth observation)
alpha = 180 * np.pi / 180  # Right ascension (rad), example for equatorial plane
delta = 0 * np.pi / 180  # Declination (rad), example for simplicity
dot_alpha = 0.05 * np.pi / 180  # Rate of right ascension (rad/day)
dot_delta = 0.01 * np.pi / 180  # Rate of declination (rad/day)

# Compute unit vector (hat_r) from Earth to spacecraft (line-of-sight direction)
hat_r = np.array(
    [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)]
)

# Compute time derivative of unit vector (dot_hat_r) for angular motion
dot_hat_r = np.array(
    [
        -np.cos(delta) * np.sin(alpha) * dot_alpha
        - np.sin(delta) * np.cos(alpha) * dot_delta,
        np.cos(delta) * np.cos(alpha) * dot_alpha
        - np.sin(delta) * np.sin(alpha) * dot_delta,
        np.cos(delta) * dot_delta,
    ]
)

# Compute angular proper motion magnitude (eta) for geocentric constraint
eta = np.sqrt(dot_alpha**2 * np.cos(delta) ** 2 + dot_delta**2)

# Define astronomical constants
k = 0.01720209895  # Gaussian gravitational constant (AU^{3/2}/day)
mu_earth = 1 / 328900.5614  # Earth's gravitational parameter (AU^3/day^2)

# Compute coefficients for energy constraint equations
c0 = np.dot(P_earth, P_earth)  # Squared magnitude of Earth's position
c5 = 2 * np.dot(P_earth, hat_r)  # Projection of Earth's position onto line-of-sight
c2 = eta**2  # Squared angular proper motion
c3 = 2 * np.dot(dot_P_earth, dot_hat_r)  # Cross-term for velocity
c4 = np.dot(dot_P_earth, dot_P_earth)  # Squared magnitude of Earth's velocity
c1 = 2 * np.dot(dot_P_earth, hat_r)  # Projection of Earth's velocity onto line-of-sight


# Function to compute dot_rho bounds for a given rho
def get_dot_r_bounds(r):
    """
    Calculate min/max dot_rho for a given rho based on heliocentric/geocentric constraints.
    Args:
        r (float): Topocentric range (rho) in AU
    Returns:
        tuple: (dot_r_min, dot_r_max) in AU/day, or (None, None) if invalid
    """
    S = r**2 + c5 * r + c0  # Heliocentric distance squared
    if S <= 0:
        return None, None
    sqrt_S = np.sqrt(S)
    C_r = c2 * r**2 + c3 * r + c4  # Velocity term for heliocentric energy
    quad_C = C_r - 2 * k**2 / sqrt_S  # Heliocentric energy constraint (E <= 0)
    quad_B = c1
    quad_A = 1.0
    disc = quad_B**2 - 4 * quad_A * quad_C  # Discriminant of quadratic
    if disc < 0:
        return None, None
    sqrt_disc = np.sqrt(disc)
    dot_r_min = (-quad_B - sqrt_disc) / (2 * quad_A)  # Lower root (lower branch)
    dot_r_max = (-quad_B + sqrt_disc) / (2 * quad_A)  # Upper root (upper branch)

    # Geocentric constraint (E_earth >= 0)
    G_r = 2 * k**2 * mu_earth / r - eta**2 * r**2
    if G_r > 0:
        dot_r_geo = np.sqrt(G_r)  # Geocentric escape velocity bound
        dot_r_min = max(dot_r_min, -dot_r_geo)  # Clip to ensure escape from Earth
        dot_r_max = min(dot_r_max, dot_r_geo)
        if dot_r_min > dot_r_max:
            return None, None

    return dot_r_min, dot_r_max


# Function to compute geocentric constraint bounds (for plotting)
def get_dot_r_geo_bounds(r):
    """
    Compute geocentric constraint bounds for plotting.
    Args:
        r (float): Topocentric range (rho) in AU
    Returns:
        tuple: (dot_r_geo_min, dot_r_geo_max) in AU/day
    """
    G_r = 2 * k**2 * mu_earth / r - eta**2 * r**2
    if G_r > 0:
        dot_r_geo = np.sqrt(G_r)
        return -dot_r_geo, dot_r_geo
    return None, None


# Sample rho values for boundary (minimal for triangulation)
num_samples = 6  # Points per branch (~12 vertices total for minimal triangles)
rho_min = 0.01  # Minimum range (AU), avoids Earth surface
rho_max = 2.0  # Maximum range (AU), typical for near-Earth asteroids
rho_list = np.linspace(rho_min, rho_max, num_samples)

# Collect boundary points for upper and lower branches
upper_points = []
lower_points = []
valid_rho = []
for rho in tqdm(rho_list, desc="Computing upper boundary"):
    dot_r_min, dot_r_max = get_dot_r_bounds(rho)
    if dot_r_min is not None and dot_r_max is not None:
        upper_points.append([rho, dot_r_max])  # Upper branch (heliocentric constraint)
        lower_points.append([rho, dot_r_min])  # Lower branch (heliocentric constraint)
        valid_rho.append(rho)

# Combine points into a closed polygon (upper then lower in reverse)
points = upper_points + lower_points[::-1]
vertices = np.array(points)  # Shape (n, 2) for triangulation

# Triangulate using Delaunay (minimal n-2 triangles)
if len(vertices) >= 3:
    tri = Delaunay(vertices)
    indices = tri.simplices
else:
    print("Insufficient points for triangulation")
    indices = None

# Compute triangle centroids for virtual asteroids
centroids = []
if indices is not None:
    for tri in indices:
        tri_verts = vertices[tri]
        centroid = np.mean(tri_verts, axis=0)  # Average vertices for centroid
        centroids.append(centroid)
centroids = np.array(centroids)

# Plot the admissible region with detailed annotations
fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure for clarity
# Plot heliocentric constraint (upper and lower branches)
if upper_points:
    upper_points = np.array(upper_points)
    ax.plot(
        upper_points[:, 0],
        upper_points[:, 1],
        "b-",
        label="Upper Branch (Heliocentric E ≤ 0)",
        linewidth=2,
    )
if lower_points:
    lower_points = np.array(lower_points)
    ax.plot(
        lower_points[:, 0],
        lower_points[:, 1],
        "r-",
        label="Lower Branch (Heliocentric E ≤ 0)",
        linewidth=2,
    )

# Plot geocentric constraint bounds
rho_fine = np.linspace(rho_min, rho_max, 100)  # Finer grid for smooth curve
geo_min = []
geo_max = []
for rho in rho_fine:
    dot_r_geo_min, dot_r_geo_max = get_dot_r_geo_bounds(rho)
    geo_min.append(dot_r_geo_min)
    geo_max.append(dot_r_geo_max)
ax.plot(rho_fine, geo_max, "g--", label="Geocentric Constraint (E_⊕ ≥ 0)", alpha=0.7)
ax.plot(rho_fine, geo_min, "g--", alpha=0.7)

# Plot admissible region (filled)
if len(vertices) > 0:
    ax.fill(
        vertices[:, 0], vertices[:, 1], "gray", alpha=0.2, label="Admissible Region"
    )

# Plot triangulation
if indices is not None:
    for tri in indices:
        tri_verts = vertices[tri]
        ax.plot(tri_verts[:, 0], tri_verts[:, 1], "k-", linewidth=1)
        ax.plot(
            [tri_verts[-1, 0], tri_verts[0, 0]],
            [tri_verts[-1, 1], tri_verts[0, 1]],
            "k-",
            linewidth=1,
        )

# Plot centroids of triangles (virtual asteroids)
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
ax.set_xlabel("Range (ρ, AU)", fontsize=12)
ax.set_ylabel("Range-Rate (ρ̇, AU/day)", fontsize=12)
ax.set_title(
    "Admissible Region for Spacecraft near Asteroid (Earth Observation, 2025-08-20)",
    fontsize=14,
)
ax.legend(fontsize=10, loc="best")
ax.grid(True, linestyle="--", alpha=0.5)
# Adjust axis limits for better focus
if len(vertices) > 0:
    rho_margin = 0.05 * (rho_max - rho_min)
    ax.set_xlim(rho_min - rho_margin, rho_max + rho_margin)
    dot_r_values = vertices[:, 1]
    dot_r_margin = 0.1 * (max(dot_r_values) - min(dot_r_values))
    ax.set_ylim(min(dot_r_values) - dot_r_margin, max(dot_r_values) + dot_r_margin)
plt.tight_layout()
plt.show()

# Print results
if indices is not None:
    print(f"Number of triangles: {len(indices)} (minimal for {len(vertices)} vertices)")
