"""Plot three daily SWE stores and their sum for two editable, inclusive dates.

Set START_DATE and END_DATE below, then run: python src/plot/plot_snow.py
The full cleaned record is simulated before selecting the displayed interval.
"""

from datetime import date
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from kalix.cleanup import load_cleaned_data
from kalix.data import project_root
from kalix.snow import LAYERS, analyze_cleaned_snow

# Edit these two dates (YYYY-MM-DD). Both endpoints are included in the plot.
START_DATE = "2022-08-01"
END_DATE = "2026-06-30"


def parse_date(value: str) -> pd.Timestamp:
    """Require an unambiguous calendar date in YYYY-MM-DD format."""
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
    except ValueError as error:
        raise ValueError("START_DATE and END_DATE must use YYYY-MM-DD format.") from error
    return pd.Timestamp(parsed)


def create_snow_plot(snow: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> plt.Figure:
    """Select already simulated storage and draw the three layers and their sum."""
    if start > end:
        raise ValueError("Start date must be on or before end date.")
    if start < snow.index[0] or end > snow.index[-1]:
        raise ValueError(
            f"Dates must be within {snow.index[0]:%Y-%m-%d} to {snow.index[-1]:%Y-%m-%d}."
        )
    selected = snow.loc[start:end]
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    colors = ("#2563eb", "#9333ea", "#15803d")
    for layer, color in zip(LAYERS, colors, strict=True):
        ax.plot(
            selected.index,
            selected[f"{layer}_swe_mm"],
            label=layer.title(),
            color=color,
            linewidth=1.1,
        )
    total = selected[[f"{layer}_swe_mm" for layer in LAYERS]].sum(axis=1)
    ax.plot(selected.index, total, label="Sum of three stores", color="#111827", linewidth=1.8)

    # Retain month labels while spacing them out for multi-year selections.
    months = (end.year - start.year) * 12 + end.month - start.month + 1
    locator = mdates.MonthLocator(interval=max(1, ceil(months / 18)))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    if not any(
        start <= pd.Timestamp(mdates.num2date(tick)).tz_localize(None) <= end
        for tick in locator.tick_values(start, end)
    ):
        ax.set_xticks([start])
    ax.set(
        title=f"Daily snow water equivalent | {start:%Y-%m-%d} to {end:%Y-%m-%d}",
        xlabel=(
            "Time (month / year)\nSum adds the three layer SWE depths; equal-area mean = sum / 3."
        ),
        ylabel="Snow water equivalent (mm)",
        xlim=(start, end + pd.Timedelta(days=1)),
        ylim=(0, None),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    if start == end:
        # Make all four values visible when the requested interval is one day.
        for line in ax.lines:
            line.set_marker("o")
    return fig


def main() -> None:
    """Load cleaned forcing, simulate continuously, and save a dated PNG plot."""
    try:
        start = parse_date(START_DATE)
        end = parse_date(END_DATE)
        if start > end:
            raise ValueError("START_DATE must be on or before END_DATE.")
        root = project_root(Path(__file__))
        snow = analyze_cleaned_snow(load_cleaned_data(root))
        fig = create_snow_plot(snow, start, end)
        output = (
            root
            / "outputs"
            / "figures"
            / (f"snow_water_equivalent_{start:%Y-%m-%d}_{end:%Y-%m-%d}.png")
        )
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=200)
        finally:
            plt.close(fig)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Cannot create snow plot: {error}") from None
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
