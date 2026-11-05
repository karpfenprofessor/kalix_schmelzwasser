"""Water-balance, input, and warm-up checks for the exploratory snow stores."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kalix.data import load_daily_data
from kalix.snow import SnowParameters, simulate_snow

ZONES = ("nikkaluokta", "tarfala", "combined")
FLUXES = ("snowfall", "rain", "melt", "swe")


def weather(low, precipitation, high=None, start="2024-01-01"):
    """Build a complete daily forcing window without quality metadata."""
    return pd.DataFrame(
        {
            "nikkaluokta_temperature_c": low,
            "tarfala_temperature_c": low if high is None else high,
            "precipitation_mm": precipitation,
        },
        index=pd.date_range(start, periods=len(low), freq="D", name="date"),
    )


def test_cold_days_accumulate_precipitation_and_keep_starting_storage():
    result = simulate_snow(
        weather([-5, -3, -8], [10, 0, 5]), initial_swe_mm=(2, 4), high_weight=0.25
    )
    np.testing.assert_allclose(result.nikkaluokta_swe_mm, [12, 12, 17])
    np.testing.assert_allclose(result.tarfala_swe_mm, [14, 14, 19])
    np.testing.assert_allclose(result.combined_swe_mm, [12.5, 12.5, 17.5])
    assert (result.filter(regex="_(rain|melt)_mm$") == 0).all().all()


def test_precipitation_phase_includes_transition_endpoints_and_mixed_snow():
    result = simulate_snow(
        weather([-2, -1, 0, 1, 2], [10] * 5),
        parameters=SnowParameters(degree_day_factor=0),
    )
    for zone in ZONES:
        np.testing.assert_allclose(result[f"{zone}_snowfall_mm"], [10, 10, 5, 0, 0])
        np.testing.assert_allclose(result[f"{zone}_rain_mm"], [0, 0, 5, 10, 10])
        np.testing.assert_allclose(result[f"{zone}_swe_mm"], [10, 20, 25, 25, 25])


def test_warm_days_deplete_snow_without_melting_more_than_available():
    result = simulate_snow(weather([-5, 2, 4], [10, 4, 0]))
    np.testing.assert_allclose(result.combined_snowfall_mm, [10, 0, 0])
    np.testing.assert_allclose(result.combined_rain_mm, [0, 4, 0])
    np.testing.assert_allclose(result.combined_melt_mm, [0, 6, 4])
    np.testing.assert_allclose(result.combined_swe_mm, [10, 4, 0])


def test_same_day_snowfall_can_melt_but_total_melt_is_capped():
    result = simulate_snow(weather([0.5], [1]))
    assert result.combined_snowfall_mm.iloc[0] == pytest.approx(0.25)
    assert result.combined_rain_mm.iloc[0] == pytest.approx(0.75)
    assert result.combined_melt_mm.iloc[0] == pytest.approx(0.25)
    assert result.combined_swe_mm.iloc[0] == 0


def test_water_is_conserved_in_each_zone_and_weighted_store():
    generator = np.random.default_rng(42)
    daily = weather(
        generator.uniform(-8, 8, 100),
        generator.uniform(0, 20, 100),
        high=generator.uniform(-8, 8, 100),
    )
    weight = 0.3
    initial = (12, 34)
    result = simulate_snow(daily, initial_swe_mm=initial, high_weight=weight)
    initial_by_zone = dict(zip(ZONES, (*initial, (1 - weight) * 12 + weight * 34)))
    for zone in ZONES:
        snowfall = result[f"{zone}_snowfall_mm"]
        rain = result[f"{zone}_rain_mm"]
        melt = result[f"{zone}_melt_mm"]
        storage = result[f"{zone}_swe_mm"]
        previous = storage.shift(1, fill_value=initial_by_zone[zone])
        np.testing.assert_allclose(snowfall + rain, daily.precipitation_mm)
        np.testing.assert_allclose(storage, previous + snowfall - melt, atol=1e-12)
        assert (storage >= 0).all()
        assert (melt >= 0).all()
        assert (melt <= previous + snowfall).all()


def test_temperature_inversion_does_not_force_the_high_station_to_be_colder():
    result = simulate_snow(weather([-8], [10], high=[2]), initial_swe_mm=(5, 5))
    assert result.nikkaluokta_swe_mm.iloc[0] == 15
    assert result.tarfala_snowfall_mm.iloc[0] == 0
    assert result.tarfala_rain_mm.iloc[0] == 10
    assert result.tarfala_melt_mm.iloc[0] == 5
    assert result.tarfala_swe_mm.iloc[0] == 0


@pytest.mark.parametrize("weight", [0, 0.25, 0.5, 0.75, 1])
def test_every_combined_flux_uses_the_selected_station_weight(weight):
    result = simulate_snow(
        weather([-5, 0, 4], [10, 6, 2], high=[-8, -3, 2]),
        initial_swe_mm=(3, 7),
        high_weight=weight,
    )
    for flux in FLUXES:
        expected = (1 - weight) * result[f"nikkaluokta_{flux}_mm"] + weight * result[
            f"tarfala_{flux}_mm"
        ]
        np.testing.assert_allclose(result[f"combined_{flux}_mm"], expected)


def test_separate_stores_are_calculated_before_combining_temperatures():
    separate = simulate_snow(weather([-4], [10], high=[2]))
    averaged = simulate_snow(weather([-1], [10]))
    assert separate.combined_swe_mm.iloc[0] == 5
    assert averaged.combined_swe_mm.iloc[0] == 10


@pytest.mark.parametrize("start", ["2024-08-30", "2024-12-30"])
def test_storage_carries_through_season_and_calendar_year_boundaries(start):
    result = simulate_snow(weather([-5] * 4, [10, 0, 0, 0], start=start))
    np.testing.assert_allclose(result.combined_swe_mm, [10] * 4)


def test_custom_thresholds_and_melt_factor_are_used():
    parameters = SnowParameters(
        degree_day_factor=4,
        snow_temperature_c=-2,
        rain_temperature_c=2,
        melt_temperature_c=1,
    )
    result = simulate_snow(weather([0, 1, 2], [8, 4, 0]), parameters=parameters)
    np.testing.assert_allclose(result.combined_snowfall_mm, [4, 1, 0])
    np.testing.assert_allclose(result.combined_melt_mm, [0, 0, 4])
    np.testing.assert_allclose(result.combined_swe_mm, [4, 5, 1])


def test_simulation_can_resume_from_the_previous_windows_final_station_states():
    daily = weather([-5, -2, 2, 4, -3, 1], [10, 4, 0, 2, 8, 0], high=[-8, -4, 0, 2, -5, -1])
    whole = simulate_snow(daily, initial_swe_mm=(3, 7), high_weight=0.25)
    first = simulate_snow(daily.iloc[:3], initial_swe_mm=(3, 7), high_weight=0.25)
    second = simulate_snow(
        daily.iloc[3:],
        initial_swe_mm=(first.nikkaluokta_swe_mm.iloc[-1], first.tarfala_swe_mm.iloc[-1]),
        high_weight=0.25,
    )
    np.testing.assert_allclose(pd.concat([first, second]), whole)


def test_input_values_index_and_metadata_are_not_modified():
    daily = weather([-3, 2], [10, 0], high=[-5, 0])
    daily.attrs = {"source": {"station": "test"}}
    original = daily.copy(deep=True)
    result = simulate_snow(daily)
    assert_frame_equal(daily, original)
    assert daily.attrs == original.attrs
    assert result.index.equals(daily.index)
    assert set(result.columns) == {f"{zone}_{flux}_mm" for zone in ZONES for flux in FLUXES}
    assert result.attrs


@pytest.mark.parametrize(
    "column", ["nikkaluokta_temperature_c", "tarfala_temperature_c", "precipitation_mm"]
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_missing_or_nonfinite_weather_is_rejected(column, invalid):
    daily = weather([-3.0, 2.0], [10.0, 0.0])
    daily.loc[daily.index[0], column] = invalid
    with pytest.raises(ValueError):
        simulate_snow(daily)


def test_negative_precipitation_is_rejected():
    with pytest.raises(ValueError):
        simulate_snow(weather([-3], [-0.1]))


def test_missing_weather_column_is_rejected():
    daily = weather([-3], [10]).drop(columns="tarfala_temperature_c")
    with pytest.raises(ValueError):
        simulate_snow(daily)


@pytest.mark.parametrize("kind", ["empty", "not_dates", "duplicate", "gap", "unsorted", "time"])
def test_invalid_daily_calendar_is_rejected(kind):
    daily = weather([-3, -2, -1], [10, 5, 0])
    if kind == "empty":
        daily = daily.iloc[:0]
    elif kind == "not_dates":
        daily = daily.reset_index(drop=True)
    elif kind == "duplicate":
        daily.index = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    elif kind == "gap":
        daily = daily.drop(daily.index[1])
    elif kind == "unsorted":
        daily = daily.iloc[::-1]
    else:
        daily.index = daily.index + pd.Timedelta(hours=6)
    with pytest.raises(ValueError):
        simulate_snow(daily)


@pytest.mark.parametrize(
    "parameters",
    [
        {"degree_day_factor": -1},
        {"degree_day_factor": np.inf},
        {"snow_temperature_c": np.nan},
        {"rain_temperature_c": np.inf},
        {"melt_temperature_c": np.nan},
        {"snow_temperature_c": 1, "rain_temperature_c": 1},
        {"snow_temperature_c": 2, "rain_temperature_c": -1},
    ],
)
def test_invalid_model_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        simulate_snow(weather([-3], [10]), parameters=SnowParameters(**parameters))


@pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_weight_is_rejected(weight):
    with pytest.raises(ValueError):
        simulate_snow(weather([-3], [10]), high_weight=weight)


@pytest.mark.parametrize("initial", [(-1, 0), (0, -1), (np.nan, 0), (0, np.inf), (1,), (1, 2, 3)])
def test_invalid_starting_storage_is_rejected(initial):
    with pytest.raises(ValueError):
        simulate_snow(weather([-3], [10]), initial_swe_mm=initial)


@pytest.fixture(scope="module")
def chart_forcing():
    daily = load_daily_data()
    forcing = daily.loc["2024-05-08":"2025-08-31"]
    assert forcing.index.equals(pd.date_range("2024-05-08", "2025-08-31", name="date"))
    return forcing


@pytest.mark.parametrize("factor", [2, 3, 5])
def test_real_data_initialization_scenarios_converge_before_chart(chart_forcing, factor):
    parameters = SnowParameters(degree_day_factor=factor)
    low_initial = simulate_snow(chart_forcing, parameters=parameters)
    high_initial = simulate_snow(chart_forcing, parameters=parameters, initial_swe_mm=(1000, 1000))
    for zone in ("nikkaluokta", "tarfala"):
        column = f"{zone}_swe_mm"
        np.testing.assert_allclose(
            low_initial.loc["2024-09-01":, column],
            high_initial.loc["2024-09-01":, column],
            rtol=0,
            atol=1e-8,
        )
        assert (high_initial.loc[:"2024-08-31", column] == 0).any()
