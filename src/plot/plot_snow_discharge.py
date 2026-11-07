"""Plot three snow stores, their sum, and observed discharge for editable dates.

Set START_DATE and END_DATE below, then run: python src/plot/plot_snow_discharge.py
Snow uses the left axis (mm water equivalent); discharge uses the right (m³/s).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kalix.cleanup import load_cleaned_data
from kalix.data import project_root
from kalix.snow import analyze_cleaned_snow
from plot.plot_snow import create_snow_plot, parse_date

# Edit these two dates (YYYY-MM-DD). Both endpoints are included in the plot.
START_DATE = "2022-08-01"
END_DATE = "2026-06-30"


def create_snow_discharge_plot(
    snow: pd.DataFrame,
    discharge: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> plt.Figure:
    """Add date-aligned, unmodified discharge to the existing four-line SWE chart."""
    fig = create_snow_plot(snow, start, end)
    try:
        selected_dates = snow.loc[start:end].index
        if not isinstance(discharge.index, pd.DatetimeIndex) or not discharge.index.is_unique:
            raise ValueError("Discharge requires unique daily date labels.")
        values = discharge.reindex(selected_dates).to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError("Discharge must be finite and complete for every selected day.")
        if (values < 0).any():
            raise ValueError("Discharge must be nonnegative.")

        snow_axis = fig.axes[0]
        discharge_axis = snow_axis.twinx()
        color = "#ea580c"
        discharge_axis.plot(
            selected_dates,
            values,
            color=color,
            linewidth=1.2,
            label="Discharge",
            marker="o" if start == end else None,
        )
        discharge_axis.set_ylabel("Discharge (m³/s)", color=color)
        discharge_axis.tick_params(axis="y", colors=color)
        discharge_axis.set_ylim(bottom=0)
        snow_axis.set_title(
            f"Daily snow water equivalent and discharge | {start:%Y-%m-%d} to {end:%Y-%m-%d}"
        )
        snow_lines, snow_labels = snow_axis.get_legend_handles_labels()
        discharge_lines, discharge_labels = discharge_axis.get_legend_handles_labels()
        snow_axis.legend(snow_lines + discharge_lines, snow_labels + discharge_labels)
    except Exception:
        plt.close(fig)
        raise
    return fig


def main() -> None:
    """Load cleaned data once, simulate continuously, and save the combined chart."""
    try:
        start = parse_date(START_DATE)
        end = parse_date(END_DATE)
        if start > end:
            raise ValueError("START_DATE must be on or before END_DATE.")
        root = project_root(Path(__file__))
        cleaned = load_cleaned_data(root)
        if "discharge_m3_s" not in cleaned:
            raise ValueError("Cleaned data require the discharge_m3_s column.")
        snow = analyze_cleaned_snow(cleaned)
        fig = create_snow_discharge_plot(snow, cleaned.discharge_m3_s, start, end)
        output = (
            root / "outputs" / "figures" / (f"snow_discharge_{start:%Y-%m-%d}_{end:%Y-%m-%d}.png")
        )
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=200)
        finally:
            plt.close(fig)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Cannot create snow/discharge plot: {error}") from None
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
