"""Compare Nikkaluokta and Tarfala daily temperatures in a saved PNG chart.

This script loads the prepared data through kalix.data, selects the shared
temperature period, and draws one line per station. Prepare the data first
with ``python -m kalix.prepare``. Missing observations remain gaps in the lines.
"""

from pathlib import Path

import matplotlib

# Select a backend that saves image files without opening a graphical window.
# This must happen before importing pyplot, which creates the figures.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from kalix.data import load_processed_data


def main() -> None:
    """Load observations, build the temperature chart, and save it to disk."""
    # __file__ is this script's path; resolve makes it absolute, parent takes
    # its folder. The data loader searches upward from there for the project root.
    root = Path(__file__).resolve().parent
    daily = load_processed_data(root)
    # A dictionary connects legend labels to Series (individual table columns).
    series = {
        "Nikkaluokta": daily["nikkaluokta_temperature_c"],
        "Tarfala": daily["tarfala_temperature_c"],
    }

    # Ignore missing values at the ends. The latest start and earliest end
    # define the interval in which both stations have observation coverage.
    start = max(values.first_valid_index() for values in series.values())
    end = min(values.last_valid_index() for values in series.values())
    if start > end:
        raise ValueError("The temperature datasets have no shared timeframe.")

    # The prepared daily calendar preserves gaps in missing observations.
    # fig is the whole image; ax is the plotting area. Size is given in inches,
    # and constrained layout automatically makes room for labels and legends.
    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    for label, values in series.items():
        # loc slices by date label and includes both endpoints, unlike list slices.
        shared = values.loc[start:end]
        # The index supplies x coordinates; values supply y coordinates.
        # alpha controls opacity (0 is invisible, 1 is fully opaque).
        ax.plot(shared.index, shared, label=label, linewidth=0.6, alpha=0.75)

    # f-strings insert values; :%Y-%m-%d formats dates as year-month-day.
    ax.set(
        title=f"Daily mean temperature | {start:%Y-%m-%d} to {end:%Y-%m-%d}",
        xlabel="Date",
        ylabel="Temperature (°C)",
        xlim=(start, end),
    )
    # A zero-degree reference helps show freezing and thawing periods.
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.grid(alpha=0.2)
    ax.legend()

    # Path's / operator joins folders; it is not numerical division here.
    output = root / "outputs" / "figures" / "temperature_comparison.png"
    # parents=True creates intermediate folders; exist_ok=True allows reruns.
    output.parent.mkdir(parents=True, exist_ok=True)
    # dpi controls saved resolution. Close the figure afterward to release memory.
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    # Execute main when run as a script; importing this module does not draw a plot.
    main()
