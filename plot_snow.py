"""Plot an exploratory two-station snow-storage proxy for the 2024/25 season."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kalix.data import load_processed_data
from kalix.snow import SnowParameters, simulate_snow

WARMUP_START = "2024-05-08"
DISPLAY_START = "2024-09-01"
DISPLAY_END = "2025-08-31"
MELT_FACTORS = (2.0, 3.0, 5.0)
HIGH_WEIGHTS = (0.25, 0.50, 0.75)
OUTPUT_NAME = "snow_water_equivalent_2024_2025"


def build_chart_data(daily: pd.DataFrame) -> pd.DataFrame:
    """Build the central run and nine scenarios, checking the warm-up first.

    Initial SWE is the storage immediately before WARMUP_START. The 0 and
    1,000 mm initial-state runs must agree at both stations before DISPLAY_START.
    This checks the selected scenarios, not observed snow-free conditions.
    """
    expected = pd.date_range(WARMUP_START, DISPLAY_END, freq="D", name=daily.index.name)
    forcing = daily.loc[WARMUP_START:DISPLAY_END]
    if not forcing.index.equals(expected):
        raise ValueError(
            f"The chart requires every daily date from {WARMUP_START} to {DISPLAY_END}."
        )
    if "discharge_m3_s" not in forcing:
        raise ValueError("The chart requires the observed discharge_m3_s column.")

    station_swe = ["nikkaluokta_swe_mm", "tarfala_swe_mm"]
    scenarios = {}
    convergence = {}
    central = None
    for factor in MELT_FACTORS:
        parameters = SnowParameters(degree_day_factor=factor)
        reference = simulate_snow(forcing, parameters=parameters)
        initially_snowy = simulate_snow(
            forcing, parameters=parameters, initial_swe_mm=(1000.0, 1000.0)
        )
        agrees = np.isclose(
            reference[station_swe].to_numpy(),
            initially_snowy[station_swe].to_numpy(),
            rtol=0.0,
            atol=1e-8,
        ).all(axis=1)
        before_display = reference.index < pd.Timestamp(DISPLAY_START)
        if not agrees[before_display][-1] or not agrees[~before_display].all():
            raise ValueError(
                f"Initial SWE scenarios have not converged before {DISPLAY_START} "
                f"for melt factor {factor:g}; use a longer warm-up."
            )
        convergence[f"degree_day_factor_{factor:g}"] = {
            "first_station_state_convergence_date": reference.index[agrees][0].strftime("%Y-%m-%d"),
            "maximum_displayed_station_difference_mm": float(
                (
                    reference.loc[DISPLAY_START:, station_swe]
                    - initially_snowy.loc[DISPLAY_START:, station_swe]
                )
                .abs()
                .to_numpy()
                .max()
            ),
        }
        for weight in HIGH_WEIGHTS:
            name = f"scenario_swe_d{factor:g}_w{weight * 100:g}_mm"
            scenarios[name] = (1.0 - weight) * reference["nikkaluokta_swe_mm"] + weight * reference[
                "tarfala_swe_mm"
            ]
        if factor == 3.0:
            central = reference

    # The central run's combined fields use simulate_snow's explicit 50/50 default.
    result = central.loc[DISPLAY_START:DISPLAY_END].copy()
    scenario_frame = pd.DataFrame(scenarios).loc[DISPLAY_START:DISPLAY_END]
    for name in scenario_frame:
        result[name] = scenario_frame[name]
    result["scenario_swe_min_mm"] = scenario_frame.min(axis=1)
    result["scenario_swe_max_mm"] = scenario_frame.max(axis=1)
    result["discharge_m3_s"] = forcing.loc[DISPLAY_START:DISPLAY_END, "discharge_m3_s"]
    result.attrs = {
        "warmup_start": WARMUP_START,
        "display_start": DISPLAY_START,
        "display_end": DISPLAY_END,
        "central_degree_day_factor_mm_per_c_day": 3.0,
        "degree_day_factors_mm_per_c_day": list(MELT_FACTORS),
        "high_station_weights": list(HIGH_WEIGHTS),
        "central_high_station_weight": 0.5,
        "snow_temperature_c": -1.0,
        "rain_temperature_c": 1.0,
        "melt_temperature_c": 0.0,
        "initial_station_swe_scenarios_mm": [0.0, 1000.0],
        "initialization_convergence": convergence,
        "interpretation": (
            "Seasonal two-station snow-storage proxy. The range across selected "
            "assumptions is not a confidence interval or guaranteed catchment bounds."
        ),
        "assumptions": [
            "Both zones use the same Nikkaluokta precipitation without correction.",
            "Station weights are illustrative, not measured catchment area fractions.",
            "Daily source labels are retained despite differing observation windows.",
            "No glacier, refreezing, liquid retention, sublimation or runoff routing model.",
            "Initialization convergence applies only to the selected starting states.",
        ],
    }
    return result


def plot_chart(chart_data: pd.DataFrame, output: Path) -> None:
    """Save three aligned panels without equating SWE or melt with river discharge."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.17, hspace=0.22)
    fig.suptitle("Estimated seasonal snow water equivalent | 2024/25", fontsize=17, y=0.965)
    fig.text(
        0.08,
        0.925,
        "Two-station snow-storage proxy  |  1 September 2024 – 31 August 2025",
        fontsize=11,
        color="#46525d",
    )

    snow_ax, input_ax, flow_ax = axes
    dates = chart_data.index
    snow_ax.fill_between(
        dates,
        chart_data["scenario_swe_min_mm"].to_numpy(),
        chart_data["scenario_swe_max_mm"].to_numpy(),
        color="#a6c5d7",
        alpha=0.5,
        label="range across selected assumptions",
    )
    snow_ax.plot(
        dates,
        chart_data["nikkaluokta_swe_mm"],
        color="#ba7d40",
        linewidth=1.1,
        label="Nikkaluokta temperature regime (~467 m)",
    )
    snow_ax.plot(
        dates,
        chart_data["tarfala_swe_mm"],
        color="#557f9e",
        linewidth=1.1,
        label="Tarfala temperature regime (~1,144 m)",
    )
    snow_ax.plot(
        dates,
        chart_data["combined_swe_mm"],
        color="#163f50",
        linewidth=2.3,
        label="Illustrative 50/50 combination (melt factor 3)",
    )
    snow_ax.set(ylabel="Remaining SWE (mm)", ylim=(0, None))
    snow_ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    input_ax.bar(
        dates,
        chart_data["combined_rain_mm"],
        width=1.0,
        color="#5c9dbf",
        alpha=0.65,
        label="Rainfall (central scenario)",
    )
    input_ax.plot(
        dates,
        chart_data["combined_melt_mm"],
        color="#cf7133",
        linewidth=1.2,
        label="Modeled melt (central scenario)",
    )
    input_ax.set(ylabel="Water input (mm/day)", ylim=(0, None))
    input_ax.legend(loc="upper left", fontsize=8.5, ncols=2, framealpha=0.95)

    flow_ax.plot(
        dates,
        chart_data["discharge_m3_s"],
        color="#74588d",
        linewidth=1.5,
        label="KAALASJÄRVI observed discharge",
    )
    flow_ax.set(ylabel="Discharge (m³/s)", xlabel="Date", ylim=(0, None))
    flow_ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    flow_ax.xaxis.set_major_locator(mdates.MonthLocator())
    flow_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_xlim(pd.Timestamp(DISPLAY_START), pd.Timestamp(DISPLAY_END))
        axis.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.08,
        0.035,
        "Shading: high-station weights 25%, 50%, 75% × melt factors 2, 3, 5 mm/°C/day. "
        "It is not a confidence interval or guaranteed catchment bounds.\n"
        "Both zones use Nikkaluokta precipitation; temperature observation windows differ "
        "from precipitation. Glacier processes are excluded.\n"
        "Warm-up starts 8 May 2024: initial SWE scenarios 0 and 1,000 mm converge before "
        "September. Discharge provides timing context only.",
        fontsize=8.5,
        color="#46525d",
        linespacing=1.6,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    chart_data = build_chart_data(load_processed_data(root))
    output = root / "outputs" / "figures" / OUTPUT_NAME
    plot_chart(chart_data, output.with_suffix(".png"))
    chart_data.to_csv(output.with_suffix(".csv"), index_label="date")
    output.with_suffix(".json").write_text(
        json.dumps(chart_data.attrs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for suffix in (".png", ".csv", ".json"):
        print(f"Saved {output.with_suffix(suffix)}")
    for name, check in chart_data.attrs["initialization_convergence"].items():
        print(f"{name}: initial states converge on {check['first_station_state_convergence_date']}")


if __name__ == "__main__":
    main()
