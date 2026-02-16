# MIT License
#
# Original work:
#   SHARMLib
#   Copyright (c) 2012 Yu Takahashi
#
# Adapted in 2019 by Benjamin Bercovici.
# Python adaptation and extensions:
#   Copyright (c) 2026 Giovanni Fereoli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

import math
import numpy as np


# -----------------------------
# Trinomial indexing utilities
# -----------------------------


def _trinomial_coeff_count(n: int) -> int:
    return (n + 1) * (n + 2) // 2


def _build_trinomial_tables(max_degree: int):
    """
    Precompute, for each degree n <= max_degree:
      - terms[n]: list of (i, j, k) in the exact linear storage order
      - index_map[n]: dict mapping (i, j, k) -> linear index
    """
    terms: List[List[Tuple[int, int, int]]] = []
    index_map: List[Dict[Tuple[int, int, int], int]] = []
    for n in range(max_degree + 1):
        t: List[Tuple[int, int, int]] = []
        m: Dict[Tuple[int, int, int], int] = {}
        idx = 0
        for i in range(n, -1, -1):
            for k in range(0, n - i + 1):
                j = n - i - k
                tup = (i, j, k)
                t.append(tup)
                m[tup] = idx
                idx += 1
        terms.append(t)
        index_map.append(m)
        assert idx == _trinomial_coeff_count(n)
    return terms, index_map


@dataclass
class Trinomial:
    degree: int = 0
    data: np.ndarray = None  # shape (coeff_count(degree),)

    def __post_init__(self):
        if self.data is None:
            self.data = np.zeros((_trinomial_coeff_count(self.degree),), dtype=float)

    def resize(self, degree: int):
        self.degree = degree
        self.data = np.zeros((_trinomial_coeff_count(degree),), dtype=float)


# -----------------------------
# Core arithmetic on trinomials
# -----------------------------
def tri_add(result: Trinomial, left: Trinomial, right: Trinomial):
    result.degree = left.degree
    result.data = left.data + right.data


def tri_sub(result: Trinomial, left: Trinomial, right: Trinomial):
    result.degree = left.degree
    result.data = left.data - right.data


def tri_copy(target: Trinomial, source: Trinomial):
    target.degree = source.degree
    target.data = source.data.copy()


def tri_mult_scalar(result: Trinomial, scalar: float):
    result.data *= scalar


def tri_mult(
    result: Trinomial,
    left: Trinomial,
    right: Trinomial,
    terms: List[List[Tuple[int, int, int]]],
    index_map: List[Dict[Tuple[int, int, int], int]],
):
    deg_l = left.degree
    deg_r = right.degree
    deg = deg_l + deg_r
    result.degree = deg
    result.data = np.zeros((_trinomial_coeff_count(deg),), dtype=float)

    t_left = terms[deg_l]
    t_right = terms[deg_r]
    idx_res_map = index_map[deg]

    # Match C++ loops (skip exact-zero coeffs)
    for idx_l, (i_l, j_l, k_l) in enumerate(t_left):
        c_l = left.data[idx_l]
        if c_l == 0.0:
            continue
        for idx_r, (i_r, j_r, k_r) in enumerate(t_right):
            c_r = right.data[idx_r]
            if c_r == 0.0:
                continue
            i = i_l + i_r
            j = j_l + j_r
            k = k_l + k_r
            idx_res = idx_res_map[(i, j, k)]
            result.data[idx_res] += c_l * c_r


# -----------------------------
# Table calculations
# -----------------------------
def calculate_basic_tables(
    n_degree: int,
):
    """
    Returns:
      factorials: list up to (n_degree+3)
      trinomialCoefficientCount: list length (n_degree+1)
      mixing_vec: list where mixing_vec[n] is a 1D array aligned with coeff order for degree n
    """
    factorials = [1.0] * (n_degree + 4)
    for i in range(1, n_degree + 4):
        factorials[i] = factorials[i - 1] * i

    trinomialCoefficientCount = [0] * (n_degree + 1)
    for n in range(0, n_degree + 1):
        trinomialCoefficientCount[n] = _trinomial_coeff_count(n)

    return factorials, trinomialCoefficientCount


def _build_mixing_vec(
    n_degree: int,
    factorials: List[float],
    terms: List[List[Tuple[int, int, int]]],
):
    """
    mixingFactors[i][j][k] in C++ equals (i!)(j!)(k!) / (n+3)! where n=i+j+k.
    We build per-degree vectors aligned with coeff storage order for IntegrateOneSimplex.
    """
    mixing_vec: List[np.ndarray] = [None] * (n_degree + 1)
    for n in range(0, n_degree + 1):
        denom = factorials[n + 3]
        vec = np.zeros((_trinomial_coeff_count(n),), dtype=float)
        for idx, (i, j, k) in enumerate(terms[n]):
            vec[idx] = (factorials[i] * factorials[j] * factorials[k]) / denom
        mixing_vec[n] = vec
    return mixing_vec


def calculate_fully_normalized_tables(n_degree: int):
    diagonalFactors = np.zeros((n_degree + 1,), dtype=float)
    subdiagonalFactors = np.zeros((n_degree + 1,), dtype=float)
    vertical1Factors = np.zeros((n_degree + 1, n_degree + 1), dtype=float)
    vertical2Factors = np.zeros((n_degree + 1, n_degree + 1), dtype=float)

    # diagonalFactors[0] not used
    if n_degree >= 1:
        diagonalFactors[1] = 1.0 / math.sqrt(3.0)
    for n in range(2, n_degree + 1):
        diagonalFactors[n] = (2.0 * n - 1.0) / math.sqrt(2.0 * n * (2 * n + 1))

    for n in range(0, n_degree + 1):
        subdiagonalFactors[n] = (
            (2.0 * n - 1.0) / math.sqrt(2.0 * n + 1.0) if (2.0 * n + 1.0) != 0 else 0.0
        )

    for n in range(2, n_degree + 1):
        for m in range(0, n - 1):  # m <= n-2
            denom = (2 * n + 1) * (n + m) * (n - m)
            vertical1Factors[n, m] = (2 * n - 1) * math.sqrt((2.0 * n - 1) / denom)
            vertical2Factors[n, m] = math.sqrt(
                ((2.0 * n - 3) * (n + m - 1) * (n - m - 1)) / denom
            )

    return diagonalFactors, subdiagonalFactors, vertical1Factors, vertical2Factors


def calculate_unnormalized_tables(n_degree: int):
    diagonalFactors = np.zeros((n_degree + 1,), dtype=float)
    subdiagonalFactors = np.zeros((n_degree + 1,), dtype=float)
    vertical1Factors = np.zeros((n_degree + 1, n_degree + 1), dtype=float)
    vertical2Factors = np.zeros((n_degree + 1, n_degree + 1), dtype=float)

    if n_degree >= 1:
        diagonalFactors[1] = 1.0
    for n in range(2, n_degree + 1):
        diagonalFactors[n] = 1.0 / float(2 * n)

    for n in range(0, n_degree + 1):
        subdiagonalFactors[n] = 1.0

    for n in range(2, n_degree + 1):
        for m in range(0, n - 1):  # m <= n-2
            vertical1Factors[n, m] = float(2 * n - 1) / float(n + m)
            vertical2Factors[n, m] = float(n - m - 1) / float(n + m)

    return diagonalFactors, subdiagonalFactors, vertical1Factors, vertical2Factors


# -----------------------------
# Integration
# -----------------------------
def integrate_one_simplex(tri: Trinomial, mixing_vec: List[np.ndarray]) -> float:
    n = tri.degree
    # mixing_vec[n] aligns with tri.data ordering
    return float(np.dot(mixing_vec[n], tri.data))


# -----------------------------
# AccumulateOneSimplex
# -----------------------------
def accumulate_one_simplex(
    n_degree: int,
    Cnm: np.ndarray,  # shape (n_degree+1, n_degree+1)
    Snm: np.ndarray,  # shape (n_degree+1, n_degree+1)
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    diagonalFactors: np.ndarray,
    subdiagonalFactors: np.ndarray,
    vertical1Factors: np.ndarray,
    vertical2Factors: np.ndarray,
    mixing_vec: List[np.ndarray],
    ref_radius: float,
    terms: List[List[Tuple[int, int, int]]],
    index_map: List[Dict[Tuple[int, int, int], int]],
):
    xTri = Trinomial(0)
    yTri = Trinomial(0)
    zTri = Trinomial(0)
    rSquared = Trinomial(0)
    diagonalC = Trinomial(0)
    diagonalS = Trinomial(0)

    verticalC = [Trinomial(0), Trinomial(0), Trinomial(0)]
    verticalS = [Trinomial(0), Trinomial(0), Trinomial(0)]

    prior2 = 0
    prior1 = 1
    present = 2

    overallFactor = (
        x0 * (y1 * z2 - y2 * z1) + x1 * (y2 * z0 - z2 * y0) + x2 * (y0 * z1 - y1 * z0)
    )
    if overallFactor == 0.0:
        return

    # normalize coordinates
    x0 /= ref_radius
    y0 /= ref_radius
    z0 /= ref_radius
    x1 /= ref_radius
    y1 /= ref_radius
    z1 /= ref_radius
    x2 /= ref_radius
    y2 /= ref_radius
    z2 /= ref_radius

    # initialize xTri = x0*X + x1*Y + x2*Z
    xTri.resize(1)
    xTri.data[index_map[1][(1, 0, 0)]] = x0
    xTri.data[index_map[1][(0, 1, 0)]] = x1
    xTri.data[index_map[1][(0, 0, 1)]] = x2

    # yTri
    yTri.resize(1)
    yTri.data[index_map[1][(1, 0, 0)]] = y0
    yTri.data[index_map[1][(0, 1, 0)]] = y1
    yTri.data[index_map[1][(0, 0, 1)]] = y2

    # zTri
    zTri.resize(1)
    zTri.data[index_map[1][(1, 0, 0)]] = z0
    zTri.data[index_map[1][(0, 1, 0)]] = z1
    zTri.data[index_map[1][(0, 0, 1)]] = z2

    # rSquared = x*x + y*y + z*z
    rSquared.resize(2)
    rSquared.data[index_map[2][(2, 0, 0)]] = x0 * x0 + y0 * y0 + z0 * z0
    rSquared.data[index_map[2][(0, 2, 0)]] = x1 * x1 + y1 * y1 + z1 * z1
    rSquared.data[index_map[2][(0, 0, 2)]] = x2 * x2 + y2 * y2 + z2 * z2
    rSquared.data[index_map[2][(1, 1, 0)]] = (x0 * x1 + y0 * y1 + z0 * z1) * 2.0
    rSquared.data[index_map[2][(0, 1, 1)]] = (x1 * x2 + y1 * y2 + z1 * z2) * 2.0
    rSquared.data[index_map[2][(1, 0, 1)]] = (x2 * x0 + y2 * y0 + z2 * z0) * 2.0

    temp1 = Trinomial(0)
    temp2 = Trinomial(0)

    for m in range(0, n_degree + 1):
        for n in range(m, n_degree + 1):

            if n == m:
                # diagonal
                if m == 0:
                    verticalC[prior2].resize(0)
                    verticalC[prior2].data[index_map[0][(0, 0, 0)]] = overallFactor

                    verticalS[prior2].resize(0)
                    verticalS[prior2].data[index_map[0][(0, 0, 0)]] = 0.0

                elif m == 1:
                    fac = overallFactor * diagonalFactors[1]

                    verticalC[prior2].resize(1)
                    verticalC[prior2].data[index_map[1][(1, 0, 0)]] = x0 * fac
                    verticalC[prior2].data[index_map[1][(0, 1, 0)]] = x1 * fac
                    verticalC[prior2].data[index_map[1][(0, 0, 1)]] = x2 * fac

                    verticalS[prior2].resize(1)
                    verticalS[prior2].data[index_map[1][(1, 0, 0)]] = y0 * fac
                    verticalS[prior2].data[index_map[1][(0, 1, 0)]] = y1 * fac
                    verticalS[prior2].data[index_map[1][(0, 0, 1)]] = y2 * fac

                else:
                    # general diagonal shift using prior column's diagonal
                    tri_mult(temp1, xTri, diagonalC, terms, index_map)
                    tri_mult(temp2, yTri, diagonalS, terms, index_map)
                    tri_sub(verticalC[prior2], temp1, temp2)
                    tri_mult_scalar(verticalC[prior2], diagonalFactors[m])

                    tri_mult(temp1, yTri, diagonalC, terms, index_map)
                    tri_mult(temp2, xTri, diagonalS, terms, index_map)
                    tri_add(verticalS[prior2], temp1, temp2)
                    tri_mult_scalar(verticalS[prior2], diagonalFactors[m])

                # accumulate
                Cnm[n, m] += integrate_one_simplex(verticalC[prior2], mixing_vec)
                Snm[n, m] += integrate_one_simplex(verticalS[prior2], mixing_vec)

                # remember diagonal for next column init
                tri_copy(diagonalC, verticalC[prior2])
                tri_copy(diagonalS, verticalS[prior2])

            elif n == m + 1:
                # subdiagonal
                tri_mult(verticalC[prior1], verticalC[prior2], zTri, terms, index_map)
                tri_mult(verticalS[prior1], verticalS[prior2], zTri, terms, index_map)

                tri_mult_scalar(verticalC[prior1], subdiagonalFactors[n])
                tri_mult_scalar(verticalS[prior1], subdiagonalFactors[n])

                Cnm[n, m] += integrate_one_simplex(verticalC[prior1], mixing_vec)
                Snm[n, m] += integrate_one_simplex(verticalS[prior1], mixing_vec)

            else:
                # ordinary vertical recurrence
                tri_mult(temp1, verticalC[prior1], zTri, terms, index_map)
                tri_mult_scalar(temp1, float(vertical1Factors[n, m]))
                tri_mult(temp2, verticalC[prior2], rSquared, terms, index_map)
                tri_mult_scalar(temp2, float(vertical2Factors[n, m]))
                tri_sub(verticalC[present], temp1, temp2)

                tri_mult(temp1, verticalS[prior1], zTri, terms, index_map)
                tri_mult_scalar(temp1, float(vertical1Factors[n, m]))
                tri_mult(temp2, verticalS[prior2], rSquared, terms, index_map)
                tri_mult_scalar(temp2, float(vertical2Factors[n, m]))
                tri_sub(verticalS[present], temp1, temp2)

                Cnm[n, m] += integrate_one_simplex(verticalC[present], mixing_vec)
                Snm[n, m] += integrate_one_simplex(verticalS[present], mixing_vec)

                # cycle indices
                i = prior2
                prior2 = prior1
                prior1 = present
                present = i


# -----------------------------
# ComputePolyhedralCS
# -----------------------------
def compute_polyhedral_cs(
    n_degree: int,
    ref_radius: float,
    r0: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    normalized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python equivalent of ComputePolyhedralCS for a SINGLE simplex (triangle) given by r0,r1,r2.

    Returns:
      Cnm2f, Snm2f as numpy arrays with shape (n_degree+1, n_degree+1),
      where only entries with n>=m are filled (others remain 0).
    """
    if n_degree < 0:
        raise ValueError("n_degree must be >= 0")

    # Max trinomial degree that can occur in products:
    # In the recurrence we multiply things like (vertical degree ~ n) by (rSquared degree 2),
    # so intermediates can go to n+2. Conservative max:
    max_tri_degree = n_degree + 2

    terms, index_map = _build_trinomial_tables(max_tri_degree)

    factorials, _ = calculate_basic_tables(
        max_tri_degree
    )  # need up to (max_tri_degree+3)!
    mixing_vec = _build_mixing_vec(max_tri_degree, factorials, terms)

    if normalized:
        diagonalFactors, subdiagonalFactors, vertical1Factors, vertical2Factors = (
            calculate_fully_normalized_tables(n_degree)
        )
    else:
        diagonalFactors, subdiagonalFactors, vertical1Factors, vertical2Factors = (
            calculate_unnormalized_tables(n_degree)
        )

    Cnm = np.zeros((n_degree + 1, n_degree + 1), dtype=float)
    Snm = np.zeros((n_degree + 1, n_degree + 1), dtype=float)

    x0, y0, z0 = float(r0[0]), float(r0[1]), float(r0[2])
    x1, y1, z1 = float(r1[0]), float(r1[1]), float(r1[2])
    x2, y2, z2 = float(r2[0]), float(r2[1]), float(r2[2])

    # simplexCount = 1 in your snippet
    accumulate_one_simplex(
        n_degree=n_degree,
        Cnm=Cnm,
        Snm=Snm,
        x0=x0,
        y0=y0,
        z0=z0,
        x1=x1,
        y1=y1,
        z1=z1,
        x2=x2,
        y2=y2,
        z2=z2,
        diagonalFactors=diagonalFactors,
        subdiagonalFactors=subdiagonalFactors,
        vertical1Factors=vertical1Factors,
        vertical2Factors=vertical2Factors,
        mixing_vec=mixing_vec,
        ref_radius=ref_radius,
        terms=terms,
        index_map=index_map,
    )

    # C++ writes Cnm2f(n,m)=Cnm[n][m] for n>=m
    Cnm2f = np.zeros_like(Cnm)
    Snm2f = np.zeros_like(Snm)
    for m in range(0, n_degree + 1):
        for n in range(m, n_degree + 1):
            Cnm2f[n, m] = Cnm[n, m]
            Snm2f[n, m] = Snm[n, m]

    return Cnm2f, Snm2f


# -----------------------------
# GetBnmNormalizedExterior
# -----------------------------
def get_bnm_normalized_exterior(
    n_degree: int,
    pos: np.ndarray,
    ref_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python equivalent of GetBnmNormalizedExterior.

    Returns:
      b_bar_real, b_bar_imag as numpy arrays sized (n_degree+3, n_degree+3),
      matching the C++ loops up to n_degree+2 inclusive.
    """
    # C++ uses (n_degree+2) in loops; allocate +3 so index n_degree+2 is valid.
    N = n_degree + 3
    b_bar_real = np.zeros((N, N), dtype=float)
    b_bar_imag = np.zeros((N, N), dtype=float)

    r_sat = float(np.linalg.norm(pos))
    x_sat, y_sat, z_sat = float(pos[0]), float(pos[1]), float(pos[2])

    for mm in range(0, n_degree + 3):  # 0..n_degree+2 inclusive
        m = float(mm)
        for nn in range(mm, n_degree + 3):
            n = float(nn)

            if mm == nn:
                if mm == 0:
                    b_bar_real[0, 0] = ref_radius / r_sat
                    b_bar_imag[0, 0] = 0.0
                else:
                    delta_1_n = 1.0 if nn == 1 else 0.0
                    fac = math.sqrt((1.0 + delta_1_n) * (2.0 * n + 1.0) / (2.0 * n)) * (
                        ref_radius / r_sat
                    )
                    b_bar_real[nn, nn] = fac * (
                        (x_sat / r_sat) * b_bar_real[nn - 1, nn - 1]
                        - (y_sat / r_sat) * b_bar_imag[nn - 1, nn - 1]
                    )
                    b_bar_imag[nn, nn] = fac * (
                        (y_sat / r_sat) * b_bar_real[nn - 1, nn - 1]
                        + (x_sat / r_sat) * b_bar_imag[nn - 1, nn - 1]
                    )
            else:
                if nn >= 2:
                    a = (
                        math.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
                        * (ref_radius / r_sat)
                        * (z_sat / r_sat)
                    )
                    b = (
                        math.sqrt(
                            (2.0 * n + 1.0)
                            * (((n - 1.0) * (n - 1.0) - m * m))
                            / ((2.0 * n - 3.0) * (n * n - m * m))
                        )
                        * (ref_radius / r_sat)
                        * (ref_radius / r_sat)
                    )

                    b_bar_real[nn, mm] = (
                        a * b_bar_real[nn - 1, mm] - b * b_bar_real[nn - 2, mm]
                    )
                    b_bar_imag[nn, mm] = (
                        a * b_bar_imag[nn - 1, mm] - b * b_bar_imag[nn - 2, mm]
                    )
                else:
                    a = (
                        math.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
                        * (ref_radius / r_sat)
                        * (z_sat / r_sat)
                    )
                    b_bar_real[nn, mm] = a * b_bar_real[nn - 1, mm]
                    b_bar_imag[nn, mm] = a * b_bar_imag[nn - 1, mm]

    return b_bar_real, b_bar_imag


def load_obj(filepath: str):
    """
    Minimal OBJ loader: reads 'v' and 'f' only.
    Faces may be triangles, quads, or ngons; will be triangulated later.
    Returns:
      V: (Nv,3) float
      F: list of faces, each a list of vertex indices (0-based), length >=3
    """
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
                    # formats: v, v/vt, v//vn, v/vt/vn
                    v_str = p.split("/")[0]
                    vi = int(v_str)
                    # OBJ indices are 1-based; negatives are relative to end
                    if vi < 0:
                        vi = len(verts) + vi + 1
                    idx.append(vi - 1)
                if len(idx) >= 3:
                    faces.append(idx)
    V = np.asarray(verts, dtype=float)
    return V, faces


def triangulate_faces(faces):
    """
    Fan triangulation: face [a,b,c,d,...] -> (a,b,c), (a,c,d), ...
    Returns:
      T: (Nt,3) int
    """
    tris = []
    for face in faces:
        a = face[0]
        for i in range(1, len(face) - 1):
            tris.append([a, face[i], face[i + 1]])
    return np.asarray(tris, dtype=int)


def signed_volume_of_mesh(V, T):
    """
    Signed volume using tetrahedra w.r.t origin:
      V = (1/6) sum dot(v0, cross(v1,v2)) over triangles (v0,v1,v2)
    Positive if triangle winding is consistent with outward normals
    (assuming origin is inside / near body).
    """
    v0 = V[T[:, 0]]
    v1 = V[T[:, 1]]
    v2 = V[T[:, 2]]
    return np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0


def enforce_outward_orientation(V, T):
    """
    If volume is negative, flip all triangles (swap two indices).
    """
    vol = signed_volume_of_mesh(V, T)
    if vol < 0:
        T = T[:, [0, 2, 1]]
    return T


def ref_radius_from_vertices(V, mode="max_norm"):
    """
    mode:
      - "max_norm": max ||v|| (bounding sphere wrt origin)
      - "mean_norm": mean ||v||
      - "rms_norm": sqrt(mean(||v||^2))
    """
    r = np.linalg.norm(V, axis=1)
    if mode == "max_norm":
        return float(np.max(r))
    if mode == "mean_norm":
        return float(np.mean(r))
    if mode == "rms_norm":
        return float(np.sqrt(np.mean(r * r)))
    raise ValueError(f"Unknown mode: {mode}")


def compute_polyhedral_cs_from_obj(
    obj_path: str,
    n_degree: int,
    ref_radius: float | None = None,
    normalized: bool = True,
    center: str | None = "centroid",  # None, "centroid", or "mean"
    ref_radius_mode: str = "max_norm",  # if ref_radius is None
    enforce_orientation: bool = True,
):
    """
    Loads OBJ, triangulates, optionally recenters, optionally fixes winding,
    and accumulates Cnm/Snm over all triangles.

    IMPORTANT:
      - OBJ vertices must already be in the body-fixed frame with origin at COM/CF.
      - Mesh must be closed and non-self-intersecting for physical meaning.
    """
    V, faces = load_obj(obj_path)
    T = triangulate_faces(faces)

    # Optional recentering (use only if your OBJ is not already COM-centered!)
    if center is not None:
        if center == "centroid" or center == "mean":
            c = np.mean(V, axis=0)
            V = V - c
        else:
            raise ValueError("center must be None, 'centroid', or 'mean'")

    if enforce_orientation:
        T = enforce_outward_orientation(V, T)

    if ref_radius is None:
        ref_radius = ref_radius_from_vertices(V, mode=ref_radius_mode)

    # Accumulate over triangles:
    C = np.zeros((n_degree + 1, n_degree + 1), dtype=float)
    S = np.zeros((n_degree + 1, n_degree + 1), dtype=float)

    # We call the single-triangle routine and sum.
    for t, (i0, i1, i2) in enumerate(T):
        print(f"Percent complete: {100.0 * (t+1) / len(T):.2f}%", end="\r")
        C_tri, S_tri = compute_polyhedral_cs(
            n_degree=n_degree,
            ref_radius=ref_radius,
            r0=V[i0],
            r1=V[i1],
            r2=V[i2],
            normalized=normalized,
        )
        C += C_tri
        S += S_tri

    return C, S, ref_radius


if __name__ == "__main__":
    obj_path = "ObjFiles/BennuRadar.obj"
    C, S, Rref = compute_polyhedral_cs_from_obj(
        obj_path=obj_path,
        n_degree=10,
        ref_radius=None,  # auto
        normalized=True,
        center=None,  # set None if OBJ already centered at COM
        enforce_orientation=True,
    )
    print("ref_radius =", Rref)
    print("C[0:4,0:4]=\n", C[:4, :4])
    print("S[0:4,0:4]=\n", S[:4, :4])

    # RMS Power Spectral Coefficients (per degree)
    print("\nRMS Power Spectral Coefficients (Fully Normalized):")
    print("Degree   P_n")

    Pn = np.zeros(C.shape[0])

    for n in range(C.shape[0]):
        power = 0.0
        for m in range(n + 1):
            power += C[n, m] ** 2 + S[n, m] ** 2
        Pn[n] = np.sqrt(power)
        print(f"{n:3d}   {Pn[n]:.6e}")

    degrees = np.arange(len(Pn))

    plt.figure(figsize=(7, 5))

    # start from degree 2
    plt.plot(degrees[2:], Pn[2:], marker="o", linewidth=1.5)

    plt.yscale("log")

    plt.xlabel("Spherical Harmonic Degree $n$")
    plt.ylabel(r"$P_n = \sqrt{\sum_{m=0}^{n}(\bar C_{nm}^2 + \bar S_{nm}^2)}$")
    plt.title(f"Degree RMS Power Spectrum (Fully Normalized)\n$P_0$ = {Pn[0]:.3e}")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
