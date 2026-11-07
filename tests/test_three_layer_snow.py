"""Three-layer water balances, cleaned forcing policy, and initialization diagnostics."""

import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kalix.snow import (
    FLUXES,
    LAYERS,
    SnowParameters,
    analyze_cleaned_snow,
    simulate_snow,
    simulate_three_layer_snow,
)


def weather(low, high, precipitation, start="2024-01-01"):
    return pd.DataFrame(
        {
            "nikkaluokta_temperature_c": low,
            "tarfala_temperature_c": high,
            "precipitation_mm": precipitation,
        },
        index=pd.date_range(start, periods=len(low), name="date"),
    )


def cleaned_weather(low, high, precipitation, start="2024-01-01"):
    frame = weather(low, high, precipitation, start)
    for station in ("nikkaluokta", "tarfala"):
        frame[f"{station}_temperature_candidate_c"] = frame[f"{station}_temperature_c"]
        frame[f"{station}_temperature_imputed"] = False
    frame.attrs = {"source": {"policy": "preserve original observations"}}
    return frame


def test_midpoint_including_inversions_and_equal_temperatures():
    daily = weather([2, -8, -3], [-4, 2, -3], [10, 10, 10])
    before = daily.copy(deep=True)
    result = simulate_three_layer_snow(daily)
    np.testing.assert_allclose(result.middle_temperature_c, [-1, -3, -3])
    np.testing.assert_allclose(result.nikkaluokta_temperature_c, daily.nikkaluokta_temperature_c)
    np.testing.assert_allclose(result.tarfala_temperature_c, daily.tarfala_temperature_c)
    # The middle gets its own full snowfall; averaging temperatures for all
    # layers, or dividing the forcing by three, would give different storage.
    assert result.combined_swe_mm.iloc[0] == pytest.approx(20 / 3)
    assert result.middle_snowfall_mm.iloc[0] == 10
    assert_frame_equal(daily, before)
    assert result.attrs["snow_model"]["area_weights"] == dict.fromkeys(LAYERS, 1 / 3)
    assert result.attrs["snow_model"]["layer_elevations_m"]["middle"] == 805.209


def test_phase_thresholds_and_cold_accumulation_in_every_layer():
    daily = weather([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [10] * 5)
    result = simulate_three_layer_snow(
        daily, parameters=SnowParameters(degree_day_factor=0), initial_swe_mm=(2, 4, 6)
    )
    for zone, initial in zip((*LAYERS, "combined"), (2, 4, 6, 4), strict=True):
        np.testing.assert_allclose(result[f"{zone}_snowfall_mm"], [10, 10, 5, 0, 0])
        np.testing.assert_allclose(result[f"{zone}_rain_mm"], [0, 0, 5, 10, 10])
        np.testing.assert_allclose(
            result[f"{zone}_swe_mm"], np.array([10, 20, 25, 25, 25]) + initial
        )


def test_melt_caps_and_same_day_snowfall():
    result = simulate_three_layer_snow(weather([-5, 2, 4, 0.5], [-5, 2, 4, 0.5], [10, 4, 0, 1]))
    for zone in (*LAYERS, "combined"):
        np.testing.assert_allclose(result[f"{zone}_swe_mm"], [10, 4, 0, 0])
        np.testing.assert_allclose(result[f"{zone}_melt_mm"], [0, 6, 4, 0.25])


def test_water_balance_equal_weights_and_legacy_endpoint_results():
    rng = np.random.default_rng(23)
    daily = weather(rng.uniform(-10, 10, 400), rng.uniform(-10, 10, 400), rng.uniform(0, 20, 400))
    initial = (12, 23, 34)
    result = simulate_three_layer_snow(daily, initial_swe_mm=initial)
    legacy = simulate_snow(daily, initial_swe_mm=(12, 34))
    for zone, starting in zip((*LAYERS, "combined"), (*initial, 23), strict=True):
        previous = result[f"{zone}_swe_mm"].shift(1, fill_value=starting)
        snow, rain, melt, swe = (result[f"{zone}_{flux}_mm"] for flux in FLUXES)
        np.testing.assert_allclose(snow + rain, daily.precipitation_mm)
        np.testing.assert_allclose(swe, previous + snow - melt, atol=1e-12)
        assert (swe >= 0).all()
        assert (melt >= 0).all()
        assert (melt <= previous + snow).all()
    for flux in FLUXES:
        np.testing.assert_allclose(
            result[f"combined_{flux}_mm"],
            result[[f"{layer}_{flux}_mm" for layer in LAYERS]].mean(axis=1),
        )
        for zone in ("nikkaluokta", "tarfala"):
            np.testing.assert_allclose(result[f"{zone}_{flux}_mm"], legacy[f"{zone}_{flux}_mm"])


def test_custom_parameters_and_resume_across_year_boundary():
    daily = weather([-5, -3, 2, 3, 0], [-3, -1, 4, 5, 2], [10, 4, 0, 0, 6], "2024-12-30")
    parameters = SnowParameters(
        degree_day_factor=4, snow_temperature_c=-2, rain_temperature_c=2, melt_temperature_c=1
    )
    whole = simulate_three_layer_snow(daily, parameters=parameters, initial_swe_mm=(3, 5, 7))
    first = simulate_three_layer_snow(
        daily.iloc[:2], parameters=parameters, initial_swe_mm=(3, 5, 7)
    )
    resumed = simulate_three_layer_snow(
        daily.iloc[2:],
        parameters=parameters,
        initial_swe_mm=tuple(first[f"{zone}_swe_mm"].iloc[-1] for zone in LAYERS),
    )
    np.testing.assert_allclose(pd.concat([first, resumed]), whole)
    np.testing.assert_allclose(whole.middle_snowfall_mm, [10, 4, 0, 0, 1.5])
    np.testing.assert_allclose(whole.middle_melt_mm, [0, 0, 8, 11, 0])


@pytest.mark.parametrize("initial", [(-1, 0, 0), (0, np.nan, 0), (0, 0, np.inf), (1, 2), (1,) * 4])
def test_invalid_three_layer_initial_storage(initial):
    with pytest.raises(ValueError, match="3 finite nonnegative"):
        simulate_three_layer_snow(weather([-3], [-5], [1]), initial_swe_mm=initial)


@pytest.mark.parametrize(
    "column", ["nikkaluokta_temperature_c", "tarfala_temperature_c", "precipitation_mm"]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_forcing_is_rejected(column, value):
    daily = weather([-3.0], [-5.0], [1.0])
    daily.loc[daily.index[0], column] = value
    with pytest.raises(ValueError, match="finite and complete"):
        simulate_three_layer_snow(daily)


@pytest.mark.parametrize("kind", ["empty", "not_dates", "duplicate", "gap", "unsorted", "time"])
def test_invalid_calendars(kind):
    daily = weather([-3] * 3, [-5] * 3, [1] * 3)
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
        daily.index += pd.Timedelta(hours=6)
    with pytest.raises(ValueError):
        simulate_three_layer_snow(daily)


def test_negative_precipitation_missing_columns_and_invalid_parameter_type():
    daily = weather([-3], [-5], [-1])
    with pytest.raises(ValueError, match="nonnegative"):
        simulate_three_layer_snow(daily)
    with pytest.raises(ValueError, match="Missing weather columns"):
        simulate_three_layer_snow(daily.drop(columns="precipitation_mm"))
    with pytest.raises(TypeError, match="SnowParameters"):
        simulate_three_layer_snow(daily, parameters={})


def test_cleaned_analysis_selects_candidates_preserves_flags_and_only_fills_missing_june():
    cleaned = cleaned_weather([-3.0] * 32, [-5.0] * 32, [2.0] * 32, "1999-05-31")
    cleaned.loc["1999-06-01":"1999-06-30", "precipitation_mm"] = np.nan
    cleaned.loc["1999-06-15", "precipitation_mm"] = 7.0
    cleaned.loc["1999-06-01", "nikkaluokta_temperature_c"] = np.nan
    cleaned.loc["1999-06-01", "nikkaluokta_temperature_imputed"] = True
    before = cleaned.copy(deep=True)
    original_attrs = copy.deepcopy(cleaned.attrs)
    result = analyze_cleaned_snow(cleaned)
    assert_frame_equal(cleaned, before)
    assert cleaned.attrs == original_attrs
    assert result.index.equals(cleaned.index)
    assert result.precipitation_assumed_zero.sum() == 29
    assert result.loc["1999-06-15", "precipitation_used_mm"] == 7
    assert result.loc["1999-06-01":"1999-06-14", "precipitation_used_mm"].eq(0).all()
    pd.testing.assert_series_equal(result.precipitation_mm, cleaned.precipitation_mm)
    assert result.loc["1999-06-01", "nikkaluokta_temperature_c"] == -3
    for station in ("nikkaluokta", "tarfala"):
        column = f"{station}_temperature_imputed"
        pd.testing.assert_series_equal(result[column], cleaned[column])
    assert result.combined_swe_mm.iloc[-1] == 11
    assert result.attrs["forcing_provenance"]["precipitation_assumed_zero_days"] == 29
    assert result.attrs["period"]["days"] == 32


@pytest.mark.parametrize("start", ["1999-05-31", "1999-07-01", "2000-06-01"])
def test_analysis_rejects_missing_precipitation_outside_june_1999(start):
    with pytest.raises(ValueError, match="finite and complete"):
        analyze_cleaned_snow(cleaned_weather([-3], [-5], [np.nan], start))


@pytest.mark.parametrize("value", [np.inf, -1.0])
def test_analysis_does_not_replace_invalid_june_values(value):
    with pytest.raises(ValueError):
        analyze_cleaned_snow(cleaned_weather([-3], [-5], [value], "1999-06-01"))


@pytest.mark.parametrize("kind", ["missing_column", "missing_flag", "numeric_flag", "unresolved"])
def test_analysis_rejects_incomplete_cleaned_inputs(kind):
    cleaned = cleaned_weather([-3.0], [-5.0], [1.0])
    if kind == "missing_column":
        cleaned = cleaned.drop(columns="tarfala_temperature_candidate_c")
    elif kind == "missing_flag":
        cleaned["tarfala_temperature_imputed"] = pd.array([pd.NA], dtype="boolean")
    elif kind == "numeric_flag":
        cleaned["tarfala_temperature_imputed"] = 1
    else:
        cleaned["tarfala_temperature_candidate_c"] = np.nan
    with pytest.raises(ValueError):
        analyze_cleaned_snow(cleaned)


def test_initialization_requires_all_layers_to_converge():
    # The low layer melts its comparison store on day 1, middle on day 2,
    # and high layer on day 3. Only then is initialization marked converged.
    result = analyze_cleaned_snow(cleaned_weather([400, 400, 400], [0, 200, 200], [0, 0, 0]))
    np.testing.assert_allclose(result.initialization_max_difference_mm, [1000, 400, 0])
    assert result.initialization_converged.tolist() == [False, False, True]
    assert result.attrs["initialization"]["first_convergence_date"] == "2024-01-03"


def test_initialization_nonconvergence_is_explicit():
    result = analyze_cleaned_snow(cleaned_weather([-3] * 3, [-5] * 3, [1] * 3))
    np.testing.assert_allclose(result.initialization_max_difference_mm, [1000] * 3)
    assert not result.initialization_converged.any()
    assert result.attrs["initialization"]["first_convergence_date"] is None
