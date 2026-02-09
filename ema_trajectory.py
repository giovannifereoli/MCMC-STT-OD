import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spiceypy
from matplotlib.collections import LineCollection

kernel_directory = Path("kernelsJustitia")
ema_kernel_path = kernel_directory / "ema_khamis_reftraj_280323_v2.bsp"
EMA_CODE = -60000

spiceypy.furnsh(ema_kernel_path.as_posix())
spiceypy.furnsh((kernel_directory / "de440s.bsp").as_posix())
spiceypy.furnsh((kernel_directory / "mar099s.bsp").as_posix())
spiceypy.furnsh((kernel_directory / "naif0012.tls").as_posix())

# constants in km
AU = 1.495978707e8
GM_SUN = 1.32712440041279419e11


# generic definitions for EMA trajectory
PLANET_PROPERTIES = {
    "Mercury": {"a": 0.387098309, "spk_id": 1, "color": "#A9A9A9"},
    "Venus": {"a": 0.72332982, "spk_id": 2, "color": "#FFCC00"},
    "Earth": {"a": 1.0000010178, "spk_id": 399, "color": "#0000FF"},
    "Mars": {"a": 1.52367934, "spk_id": 499, "color": "#FF0000"},
    "Jupiter": {"a": 5.202603191, "spk_id": 599, "color": "#FFA500"},
    "Saturn": {"a": 9.554909595, "spk_id": 699, "color": "#FFFF00"},
    "Uranus": {"a": 19.218446061, "spk_id": 7, "color": "#00FFFF"},
    "Neptune": {"a": 30.11038687, "spk_id": 8, "color": "#00008B"},
    "Pluto": {"a": 39.544674, "spk_id": 9, "color": "#8B4513"},
}

PLANET_FLYBY_PROPERTIES = {
    2: {
        "name": "Venus",
        "flyby_date": [datetime.datetime(2028, 7, 19)],
    },
    399: {
        "name": "Earth",
        "flyby_date": [datetime.datetime(2029, 5, 22)],
    },
    499: {
        "name": "Mars",
        "flyby_date": [datetime.datetime(2031, 9, 27)],
    },
}

# asteroid encounters (flyby + arrival) in chronological order
ORDERED_ENCOUNTER_SPKIDS = (
    20010253,
    20000623,
    20013294,
    20088055,
    20023871,
    20059980,
    20000269,
)

ENCOUNTER_TARGET_PROPERTIES = {
    20010253: {
        "name": "Westerwald",
        "flyby_date": datetime.datetime(2030, 2, 18),
    },
    20000623: {
        "name": "Chimaera",
        "flyby_date": datetime.datetime(2030, 6, 14),
    },
    20013294: {
        "name": "Rockox",
        "flyby_date": datetime.datetime(2031, 1, 14),
    },
    20088055: {
        "name": "2000 VA28",
        "flyby_date": datetime.datetime(2032, 7, 24),
    },
    20023871: {
        "name": "1998 RC76",
        "flyby_date": datetime.datetime(2032, 12, 15),
    },
    20059980: {
        "name": "1999 SG6",
        "flyby_date": datetime.datetime(2033, 9, 2),
    },
    20000269: {
        "name": "Justitia",
        "flyby_date": datetime.datetime(2034, 10, 30),
    },
}


def get_coverage_windows(kernel_path: Path, spk_id: int) -> tuple:
    """
    Get the coverage windows for a given SPK ID in a kernel file.

    Parameters
    ----------
    kernel_path : pathlib.Path
        Path to the SPK kernel file.
    spk_id : int
        SPK ID to get coverage for.

    Returns
    -------
    tuple of tuples
        Tuple of tuples representing the start and end epoch of coverage windows.

    Notes
    -----
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkcov_c.html
    """

    coverage_window_cells = spiceypy.spkcov(kernel_path.as_posix(), spk_id)

    # convert Spice cell to Python type
    number_of_windows = spiceypy.wncard(coverage_window_cells)
    coverage_windows = []
    for i in range(number_of_windows):
        coverage_windows.append(spiceypy.wnfetd(coverage_window_cells, i))

    return tuple(coverage_windows)


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def compute_orbital_period(sma, gm):
    return 2 * np.pi * np.sqrt(sma**3 / gm)


def compute_heliocentric_revolution_states(
    sma: float, gm: float, spice_id: str, N_STEPS=1000
):
    ets = np.linspace(
        0,
        compute_orbital_period(sma, gm),
        N_STEPS,
    )

    states = np.array(
        [spiceypy.spkezr(spice_id, et, "ECLIPJ2000", "NONE", "Sun")[0] for et in ets]
    )

    return states


def add_planet_outlines(
    ax,
    planets: list[str],
    epochs_per_planet: dict = None,
    markersize=10,
    markers="o",
    linewidth=0.5,
    add_labels=True,
):
    for planet in planets:
        states = compute_heliocentric_revolution_states(
            PLANET_PROPERTIES[planet]["a"] * AU,
            GM_SUN,
            str(PLANET_PROPERTIES[planet]["spk_id"]),
        )
        states /= AU
        x, y = states[:, 0], states[:, 1]

        ax.plot(
            x,
            y,
            color="grey",
            linewidth=linewidth,
        )
    if epochs_per_planet is not None:
        for planet, epochs in epochs_per_planet.items():
            epochs = np.atleast_1d(epochs)
            markersize = np.atleast_1d(markersize)

            for ii, epoch in enumerate(epochs):
                epoch_state = spiceypy.spkezr(
                    str(PLANET_PROPERTIES[planet]["spk_id"]),
                    epoch,
                    "ECLIPJ2000",
                    "NONE",
                    "Sun",
                )[0]

                epoch_state /= AU
                epoch_x, epoch_y = epoch_state[0], epoch_state[1]

                ax.scatter(
                    epoch_x,
                    epoch_y,
                    s=markersize,
                    marker=markers[ii],
                    zorder=10,
                    label=planet if add_labels and ii == 0 else None,
                    color=PLANET_PROPERTIES[planet]["color"],
                )

    if add_labels:
        planet_leg = ax.legend(
            title="Gravity assists",
            ncols=3,
            loc="upper left",
            bbox_to_anchor=(-0.2, -0.12),
        )
        ax.add_artist(planet_leg)


def add_trajectory_legend(ax):
    # legend for start/end and flybys
    handles = [
        plt.Line2D(
            [],
            [],
            color="black",
            marker="s",
            linestyle="",
            label="Earth Launch",
            markersize=6,
        ),
        plt.Line2D(
            [],
            [],
            color="black",
            marker="^",
            linestyle="",
            label=f"{ENCOUNTER_TARGET_PROPERTIES[ORDERED_ENCOUNTER_SPKIDS[-1]]['name']} Arrival",
            markersize=6,
        ),
        plt.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            linestyle="",
            label="Asteroid Flybys",
            markersize=6,
        ),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.85, -0.12))


def plot_flyby_targets(ax):
    # exclude Justitia
    for spk in ORDERED_ENCOUNTER_SPKIDS[:-1]:
        flyby_date = ENCOUNTER_TARGET_PROPERTIES[spk]["flyby_date"]
        flyby_epoch = spiceypy.datetime2et(flyby_date)
        flyby_state = spiceypy.spkezr(
            str(EMA_CODE), flyby_epoch, "ECLIPJ2000", "NONE", "Sun"
        )[0]
        flyby_name = ENCOUNTER_TARGET_PROPERTIES[spk]["name"]

        ax.scatter(
            flyby_state[0] / AU,
            flyby_state[1] / AU,
            color="grey",
            marker="o",
            zorder=5,
            s=10,
        )

        # manually adjust label positions for better visibility
        # this is trial and error to make it look nice
        offset_points = (3, 3)
        if flyby_name == "Chimaera":
            offset_points = (3, -6)
        elif flyby_name == "1999 SG6":
            offset_points = (5, -2)
        elif flyby_name == "1998 RC76":
            offset_points = (6, -0.5)

        ax.annotate(
            flyby_name,
            xy=(flyby_state[0] / AU, flyby_state[1] / AU),
            xytext=offset_points,
            textcoords="offset points",
            fontsize=8,
        )


def plot_ema_trajectory(fig, ax):
    ema_kernel_coverage_windows = get_coverage_windows(ema_kernel_path, EMA_CODE)[0]

    plot_start_epoch = ema_kernel_coverage_windows[0]
    # end the plot at Justitia arrival
    plot_end_epoch = spiceypy.datetime2et(
        ENCOUNTER_TARGET_PROPERTIES[ORDERED_ENCOUNTER_SPKIDS[-1]]["flyby_date"]
    )
    plotting_epochs = np.linspace(plot_start_epoch, plot_end_epoch, 1000)

    ema_states = np.array(
        [
            spiceypy.spkezr(str(EMA_CODE), et, "ECLIPJ2000", "NONE", "Sun")[0]
            for et in plotting_epochs
        ]
    )
    lines = colored_line(
        ema_states[:, 0] / AU,
        ema_states[:, 1] / AU,
        plotting_epochs,
        ax,
        cmap="viridis",
    )
    cbar = fig.colorbar(lines)
    cbar.set_label("Epoch")

    # set colorbar ticks to years
    year_ticks = np.arange(
        spiceypy.et2datetime(plot_start_epoch).year + 1,
        spiceypy.et2datetime(plot_end_epoch).year + 1,
        1,
    )
    et_ticks = [
        spiceypy.datetime2et(datetime.datetime(year, 1, 1, 0, 0, 0))
        for year in year_ticks
    ]
    tick_labels = [str(year) for year in year_ticks]
    cbar.set_ticks(
        et_ticks,
        labels=tick_labels,
    )

    # launch and arrival markers
    ax.scatter(
        ema_states[0, 0] / AU,
        ema_states[0, 1] / AU,
        color="black",
        marker="s",
        zorder=5,
        s=20,
    )
    ax.scatter(
        ema_states[-1, 0] / AU,
        ema_states[-1, 1] / AU,
        color="black",
        marker="^",
        zorder=5,
        s=20,
    )


def main():
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.set_aspect("equal")

    # ema trajectory with color scale by epoch
    plot_ema_trajectory(fig, ax)

    # outlines of planets in light gray with markers at flyby epochs
    add_planet_outlines(
        ax,
        ["Mercury", "Venus", "Earth", "Mars"],
        epochs_per_planet={
            PLANET_FLYBY_PROPERTIES[planet_spk]["name"]: [
                spiceypy.datetime2et(dt)
                for dt in PLANET_FLYBY_PROPERTIES[planet_spk]["flyby_date"]
            ]
            for planet_spk in PLANET_FLYBY_PROPERTIES.keys()
        },
    )

    # plot flyby target markers along the trajectory
    plot_flyby_targets(ax)

    # manually add legend for start/end and asteroid flybys
    add_trajectory_legend(ax)

    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_title("EMA Trajectory")

    fig.savefig("ema_trajectory.png")


if __name__ == "__main__":
    main()
