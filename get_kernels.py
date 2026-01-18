"""
Plot OSIRIS-REx trajectory in the Bennu-centered J2000 inertial frame for a chosen epoch.

Assumptions:
- You already have kernels locally in the folder structure you showed.
- Uses spiceypy (pip install spiceypy) and matplotlib.

What it does:
1) Loads all kernels (your directory layout).
2) Builds a time grid over a user-chosen UTC interval.
3) Computes OREx position wrt Bennu in J2000.
4) Plots 3D trajectory + range vs time.

Tip:
- “particle ejection” imaging is often around Jan 2019; below defaults to a Jan 19–21, 2019 window.
  Change START_UTC / STOP_UTC to whatever interval you want to inspect.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import spiceypy as spice


# ----------------------------
# User inputs (edit as needed)
# ----------------------------
KERNEL_ROOT = Path("./kernels")

# Default window (adjust to the exact particle-ejecta imaging interval you care about)
START_UTC = "2019-01-19T00:00:00"
STOP_UTC = "2019-01-20T00:00:00"
DT_SEC = 60.0  # sample step [s]

# Body names as defined in your kernels
TARGET = "BENNU"  # center
OBSERVER = "OSIRIS-REX"  # spacecraft
FRAME = "J2000"
ABCORR = "NONE"


def list_files(d: Path):
    if not d.exists():
        return []
    return sorted([str(p) for p in d.iterdir() if p.is_file()])


def safe_furnsh(kpath: str):
    """Load a kernel, with a helpful error if it fails."""
    try:
        spice.furnsh(kpath)
    except Exception as e:
        raise RuntimeError(f"Failed to load kernel:\n  {kpath}\nError:\n  {e}") from e


def load_kernels(kernel_root: Path):
    # OREx Trajectories
    orex_traj_kernels_path = kernel_root / "orex" / "orex_trajectories"
    traj_kernel_files = list_files(orex_traj_kernels_path)

    # OREx Instrument Kernels
    instrument_kernels = [
        str(kernel_root / "orex" / "instrument_kernels" / "orx_navcam_v02.ti"),
        str(kernel_root / "orex" / "instrument_kernels" / "orx_ocams_v07.ti"),
    ]

    # OREx Frame Kernels
    frame_kernels = [
        str(kernel_root / "orex" / "frame_kernels" / "orx_v14.tf"),
    ]

    # OREx Attitude Kernels
    attitude_kernels_path = kernel_root / "orex" / "attitude_kernels"
    attitude_kernels = list_files(attitude_kernels_path)

    # OREx Clock Kernels
    clock_kernels = [
        str(kernel_root / "orex" / "clock_kernels" / "orx_sclkscet_00093.tsc"),
    ]

    # Other SPICE Kernels
    other_kernels = [
        str(kernel_root / "pck00010.tpc"),
        str(kernel_root / "naif0012.tls"),
        str(kernel_root / "de424.bsp"),
        str(kernel_root / "gm_de440.tpc"),
        str(kernel_root / "bennu_v17.tpc"),
        str(kernel_root / "orex" / "bennu_refdrmc_v1.bsp"),
        str(
            kernel_root
            / "orex"
            / "bennu_shape_models"
            / "bennu_g_12600mm_alt_obj_0000n00000_v021a.bds"
        ),
        str(kernel_root / "orex" / "orx_struct_v04.bsp"),
        str(kernel_root / "trajectories" / "de432s.bsp"),
    ]

    kernels = (
        traj_kernel_files
        + instrument_kernels
        + frame_kernels
        + attitude_kernels
        + clock_kernels
        + other_kernels
    )

    # Load all kernels (and fail fast if something is missing)
    for k in kernels:
        if not os.path.isfile(k):
            raise FileNotFoundError(f"Kernel not found: {k}")
        safe_furnsh(k)

    return kernels


def get_orex_pos_wrt_bennu(et_array: np.ndarray):
    """
    Returns Nx3 positions [km] of OSIRIS-REX w.r.t. BENNU in J2000.
    """
    r = np.zeros((len(et_array), 3), dtype=float)
    for i, et in enumerate(et_array):
        # spkpos returns position vector of TARGET as seen from OBSERVER.
        # We want spacecraft wrt Bennu, so we ask for OBSERVER as target, BENNU as observer.
        # Position of OSIRIS-REX relative to BENNU:
        pos, _ = spice.spkpos(OBSERVER, et, FRAME, ABCORR, TARGET)
        r[i, :] = pos
    return r


def main():
    kernels_loaded = []
    try:
        # Load kernels
        kernels_loaded = load_kernels(KERNEL_ROOT)

        # Time grid
        et0 = spice.str2et(START_UTC)
        et1 = spice.str2et(STOP_UTC)
        if et1 <= et0:
            raise ValueError("STOP_UTC must be after START_UTC")

        ets = np.arange(et0, et1 + DT_SEC, DT_SEC)
        utc = [spice.et2utc(et, "ISOC", 3) for et in ets]

        # States
        r = get_orex_pos_wrt_bennu(ets)
        rng = np.linalg.norm(r, axis=1)

        # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=1.5)
        ax.scatter([0], [0], [0], s=30, marker="o")
        ax.set_xlabel("X (km)  [Bennu-centered J2000]")
        ax.set_ylabel("Y (km)  [Bennu-centered J2000]")
        ax.set_zlabel("Z (km)  [Bennu-centered J2000]")
        ax.set_title(
            f"OSIRIS-REx wrt Bennu in {FRAME}\n{START_UTC} to {STOP_UTC}, dt={DT_SEC:.0f}s"
        )
        ax.set_box_aspect((1, 1, 1))

        # Range vs time
        plt.figure()
        t_hours = (ets - ets[0]) / 3600.0
        plt.plot(t_hours, rng, linewidth=1.5)
        plt.xlabel(f"Time since {START_UTC} (hours)")
        plt.ylabel("Range to Bennu (km)")
        plt.title("OSIRIS-REx range to Bennu")
        plt.grid(True)

        plt.show()

    finally:
        # Clean unload to avoid kernel pool contamination in interactive runs
        try:
            spice.kclear()
        except Exception:
            pass


if __name__ == "__main__":
    main()
