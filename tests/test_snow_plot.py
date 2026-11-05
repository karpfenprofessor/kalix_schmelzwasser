"""Checks that the chart uses complete inputs and the intended assumption range."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalix.data import load_daily_data
from kalix.snow import SnowParameters, simulate_snow


@pytest.fixture(scope="module")
def plot_module():
    """Load the root-level script without assuming it is an installed module."""
    path = Path(__file__).resolve().parents[1] / "plot_snow.py"
    spec = importlib.util.spec_from_file_location("plot_snow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def daily():
    return load_daily_data()


def test_real_chart_covers_one_season_with_the_central_run_and_nine_scenarios(plot_module, daily):
    chart = plot_module.build_chart_data(daily)
    assert chart.index.equals(pd.date_range("2024-09-01", "2025-08-31", name="date"))
    assert len(chart) == 365
    forcing = daily.loc["2024-05-08":"2025-08-31"]
    central = simulate_snow(forcing).loc["2024-09-01":]
    np.testing.assert_allclose(chart[central.columns], central)
    np.testing.assert_allclose(chart.discharge_m3_s, daily.loc[chart.index, "discharge_m3_s"])

    scenarios = []
    for factor in (2, 3, 5):
        for weight in (0.25, 0.5, 0.75):
            column = f"scenario_swe_d{factor}_w{weight * 100:g}_mm"
            scenarios.append(column)
            expected = simulate_snow(
                forcing, parameters=SnowParameters(degree_day_factor=factor), high_weight=weight
            ).loc["2024-09-01":, "combined_swe_mm"]
            np.testing.assert_allclose(chart[column], expected)
    assert len(scenarios) == 9
    np.testing.assert_allclose(chart.scenario_swe_min_mm, chart[scenarios].min(axis=1))
    np.testing.assert_allclose(chart.scenario_swe_max_mm, chart[scenarios].max(axis=1))
    assert (chart.scenario_swe_min_mm <= chart.combined_swe_mm).all()
    assert (chart.combined_swe_mm <= chart.scenario_swe_max_mm).all()
    for check in chart.attrs["initialization_convergence"].values():
        assert pd.Timestamp(check["first_station_state_convergence_date"]) < pd.Timestamp(
            "2024-09-01"
        )
        assert check["maximum_displayed_station_difference_mm"] <= 1e-8


@pytest.mark.parametrize("missing_date", ["2024-05-08", "2024-12-01", "2025-08-31"])
def test_chart_rejects_missing_warmup_day_season_day_or_endpoint(plot_module, daily, missing_date):
    incomplete = daily.drop(pd.Timestamp(missing_date))
    with pytest.raises(ValueError, match="requires every daily date"):
        plot_module.build_chart_data(incomplete)


def test_chart_rejects_missing_weather_instead_of_carrying_storage_through_it(plot_module, daily):
    incomplete = daily.copy()
    incomplete.loc["2024-10-01", "precipitation_mm"] = np.nan
    with pytest.raises(ValueError, match="finite and complete"):
        plot_module.build_chart_data(incomplete)


def test_chart_requires_discharge_for_the_timing_comparison(plot_module, daily):
    with pytest.raises(ValueError, match="discharge_m3_s"):
        plot_module.build_chart_data(daily.drop(columns="discharge_m3_s"))


def test_chart_rejects_a_warmup_that_has_not_removed_initial_state_differences(plot_module):
    dates = pd.date_range("2024-05-08", "2025-08-31", name="date")
    cold = pd.DataFrame(
        {
            "nikkaluokta_temperature_c": -5.0,
            "tarfala_temperature_c": -5.0,
            "precipitation_mm": 0.0,
            "discharge_m3_s": 10.0,
        },
        index=dates,
    )
    with pytest.raises(ValueError, match="Initial SWE scenarios have not converged"):
        plot_module.build_chart_data(cold)
