"""Exploratory seasonal snow storage, expressed in mm of water equivalent.

The stations represent temperature regimes, not known catchment area fractions.
All stores receive the same precipitation depth. This simple accounting model omits
snowpack liquid-water retention, refreezing, sublimation, and glacier processes.
Parameters are illustrative assumptions, not calibrated estimates.

Snow water equivalent (SWE) is the depth of water obtained by melting the snow,
not its physical snow depth. Each day adds snowfall to yesterday's SWE and then
subtracts melt. This module calculates those quantities; plotting is separate.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd

ZONES = ("nikkaluokta", "tarfala")
LAYERS = ("nikkaluokta", "middle", "tarfala")
LAYER_ELEVATIONS_M = {"nikkaluokta": 466.765, "middle": 805.209, "tarfala": 1143.653}
INITIALIZATION_TOLERANCE_MM = 1e-8
FLUXES = ("snowfall", "rain", "melt", "swe")
WEATHER_COLUMNS = (
    "nikkaluokta_temperature_c",
    "tarfala_temperature_c",
    "precipitation_mm",
)


@dataclass(frozen=True)
class SnowParameters:
    """Daily phase thresholds (degrees C) and melt factor (mm / degree C / day)."""

    # dataclass creates an initializer from these fields. frozen=True prevents
    # normal reassignment, so a run's settings are not accidentally changed.
    degree_day_factor: float = 3.0
    snow_temperature_c: float = -1.0
    rain_temperature_c: float = 1.0
    melt_temperature_c: float = 0.0

    def __post_init__(self) -> None:
        """Validate settings immediately after the dataclass is initialized."""
        # asdict turns the fields into a dictionary so each can be checked alike.
        for name, value in asdict(self).items():
            if not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.degree_day_factor < 0:
            raise ValueError("degree_day_factor must be nonnegative.")
        if self.snow_temperature_c >= self.rain_temperature_c:
            raise ValueError("snow_temperature_c must be less than rain_temperature_c.")


def simulate_snow(
    daily: pd.DataFrame,
    *,
    parameters: SnowParameters | None = None,
    initial_swe_mm: tuple[float, float] = (0.0, 0.0),
    high_weight: float = 0.5,
) -> pd.DataFrame:
    """Run two independent snow stores and combine their outputs afterward.

    ``daily`` must contain the three WEATHER_COLUMNS on a complete, increasing
    daily DatetimeIndex. Missing/nonfinite weather and negative precipitation are
    rejected: skipping a day would invalidate later snow storage.

    Initial SWE is the storage *before* the first input day, ordered Nikkaluokta,
    Tarfala. ``high_weight`` is Tarfala's illustrative representation weight in
    [0, 1]; Nikkaluokta receives the remaining weight. There are no annual resets.

    Returns ``{zone}_{flux}_mm`` columns for zones ``nikkaluokta``, ``tarfala``,
    and ``combined`` and fluxes ``snowfall``, ``rain``, ``melt``, and ``swe``.
    Snowfall/rain/melt are daily water amounts; SWE is end-of-day storage. Melt
    leaves the store immediately. Rain never enters this simplified snow store.
    Neither the input dataframe nor its metadata is modified.
    """
    # The * in the signature makes subsequent arguments keyword-only: callers
    # write high_weight=0.5, making the meaning of each setting visible.
    # None means "use default settings"; annotations describe expected types.
    parameters = SnowParameters() if parameters is None else parameters
    if not isinstance(parameters, SnowParameters):
        raise TypeError("parameters must be a SnowParameters instance.")
    if not isinstance(high_weight, Real) or not isfinite(high_weight) or not 0 <= high_weight <= 1:
        raise ValueError("high_weight must be a finite number between 0 and 1.")
    initial = _initial_storage(initial_swe_mm, 2)
    weather = _weather_inputs(daily)
    output = _simulate_stores(weather[:, :2], weather[:, 2], initial, parameters, ZONES)

    # A weighted average: high_weight belongs to Tarfala; the rest to Nikkaluokta.
    for flux in FLUXES:
        output[f"combined_{flux}_mm"] = (1 - high_weight) * output[
            f"nikkaluokta_{flux}_mm"
        ] + high_weight * output[f"tarfala_{flux}_mm"]
    result = pd.DataFrame(output, index=daily.index.copy())
    result.attrs = {
        "snow_model": {
            "parameters": asdict(parameters),
            "initial_swe_mm": initial.tolist(),
            "high_weight": float(high_weight),
            "units": "mm water equivalent; fluxes are daily amounts, SWE is end-of-day storage",
            "precipitation": "Same Nikkaluokta precipitation applied independently to both zones.",
            "alignment": "Source daily labels; observation windows have not been synchronized.",
            "interpretation": (
                "Uncalibrated seasonal snow-storage proxy, not observed catchment SWE."
            ),
        }
    }
    return result


def _initial_storage(values, count: int) -> np.ndarray:
    """Validate one starting SWE value per independent store."""
    message = f"initial_swe_mm must contain {count} finite nonnegative values."
    try:
        initial = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if initial.shape != (count,) or not np.isfinite(initial).all() or (initial < 0).any():
        raise ValueError(message)
    return initial


def _weather_inputs(daily: pd.DataFrame) -> np.ndarray:
    """Validate the daily calendar and return the two temperatures and precipitation."""
    # Validate the calendar before calculating: skipping a day would skip its
    # snowfall/melt and change every later day's stored snow.
    index = daily.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0 or index.hasnans:
        raise ValueError("Weather inputs require a nonempty daily DatetimeIndex without NaT.")
    if not index.equals(index.normalize()) or not index.equals(
        pd.date_range(index[0], periods=len(index), freq="D", name=index.name)
    ):
        raise ValueError("Weather dates must be unique, increasing, and complete daily dates.")
    # Set difference gives the required names that the input table does not have.
    missing_columns = set(WEATHER_COLUMNS).difference(daily.columns)
    if missing_columns:
        raise ValueError(f"Missing weather columns: {', '.join(sorted(missing_columns))}.")
    if not daily.columns.is_unique:
        raise ValueError("Weather column names must be unique.")
    try:
        # loc[:, names] selects all rows and the named columns in this order.
        # Convert the table to an array for the repeated numerical calculations.
        weather = daily.loc[:, list(WEATHER_COLUMNS)].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError("Weather inputs must be numeric, finite, and complete.") from error
    if not np.isfinite(weather).all():
        raise ValueError("Weather inputs must be finite and complete; missing values are invalid.")
    # Array positions start at zero: column 2 is the third weather column.
    precipitation = weather[:, 2]
    if (precipitation < 0).any():
        raise ValueError("Precipitation must be nonnegative.")
    return weather


def _simulate_stores(
    temperatures: np.ndarray,
    precipitation: np.ndarray,
    initial: np.ndarray,
    parameters: SnowParameters,
    zones: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Apply the same phase, melt, and daily water balance to every store."""
    output = {}
    # Calculate each station separately before weighting outputs. Averaging
    # temperatures first would change the threshold-dependent snow/melt results.
    for zone_index, zone in enumerate(zones):
        temperature = temperatures[:, zone_index]
        # The fraction falls linearly from 1 (all snow) to 0 (all rain).
        # clip keeps the result between 0 and 1 beyond the temperature thresholds.
        snow_fraction = np.clip(
            (parameters.rain_temperature_c - temperature)
            / (parameters.rain_temperature_c - parameters.snow_temperature_c),
            0.0,
            1.0,
        )
        snowfall = precipitation * snow_fraction
        rain = precipitation - snowfall
        # maximum(..., 0) makes potential melt zero below the melt threshold.
        # Example: factor 3 and temperature 2 degrees above it gives 6 mm/day.
        potential_melt = parameters.degree_day_factor * np.maximum(
            temperature - parameters.melt_temperature_c, 0.0
        )
        # Allocate result arrays up front. empty leaves entries uninitialized;
        # the loop below must assign every entry before we use the results.
        melt = np.empty(len(precipitation), dtype=float)
        swe = np.empty(len(precipitation), dtype=float)
        storage = initial[zone_index]
        # Storage depends on yesterday, so this part proceeds one day at a time.
        for day in range(len(precipitation)):
            available = storage + snowfall[day]
            # Actual melt cannot exceed the snow available, including new snow.
            melt[day] = min(available, potential_melt[day])
            storage = available - melt[day]
            swe[day] = storage
        # zip pairs names with arrays; strict=True detects unequal sequence lengths.
        for flux, values in zip(FLUXES, (snowfall, rain, melt, swe), strict=True):
            output[f"{zone}_{flux}_mm"] = values

    return output


def simulate_three_layer_snow(
    daily: pd.DataFrame,
    *,
    parameters: SnowParameters | None = None,
    initial_swe_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> pd.DataFrame:
    """Run three equal-area stores with a daily elevation-midpoint temperature.

    Input uses WEATHER_COLUMNS on a complete daily DatetimeIndex, with finite
    temperatures and finite, nonnegative precipitation. Middle temperature is
    the arithmetic mean of both station temperatures, including inversions.
    Each layer receives the full precipitation depth; combined fluxes/storage
    are their arithmetic mean, calculated after simulating independent stores.

    Initial SWE precedes the first day, ordered Nikkaluokta, middle, Tarfala.
    Returns three ``{layer}_temperature_c`` columns and ``{zone}_{flux}_mm``
    for each layer plus combined. SWE is end-of-day water-equivalent storage;
    snowfall/rain/melt are daily amounts. No annual reset or input mutation.
    """
    parameters = SnowParameters() if parameters is None else parameters
    if not isinstance(parameters, SnowParameters):
        raise TypeError("parameters must be a SnowParameters instance.")
    initial = _initial_storage(initial_swe_mm, 3)
    weather = _weather_inputs(daily)
    temperatures = np.column_stack(
        (weather[:, 0], weather[:, 0] / 2 + weather[:, 1] / 2, weather[:, 1])
    )
    output = {
        f"{layer}_temperature_c": temperatures[:, position] for position, layer in enumerate(LAYERS)
    }
    output.update(_simulate_stores(temperatures, weather[:, 2], initial, parameters, LAYERS))
    for flux in FLUXES:
        output[f"combined_{flux}_mm"] = np.mean(
            [output[f"{layer}_{flux}_mm"] for layer in LAYERS], axis=0
        )
    result = pd.DataFrame(output, index=daily.index.copy())
    result.attrs = {
        "snow_model": {
            "model": "three_equal_area_layers",
            "parameters": asdict(parameters),
            "initial_swe_mm": initial.tolist(),
            "layer_order": list(LAYERS),
            "layer_elevations_m": dict(LAYER_ELEVATIONS_M),
            "area_weights": dict.fromkeys(LAYERS, 1 / 3),
            "middle_temperature": "Daily arithmetic mean; linear interpolation in elevation.",
            "units": "mm water equivalent; fluxes are daily amounts, SWE is end-of-day storage",
            "precipitation": "Same full Nikkaluokta precipitation depth applied to each layer.",
            "alignment": "Source daily labels; observation windows have not been synchronized.",
            "interpretation": (
                "Uncalibrated seasonal snow-storage proxy, not observed catchment SWE."
            ),
        }
    }
    return result


def analyze_cleaned_snow(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Analyze cleaned candidate forcing without changing observations or metadata.

    Use both temperature candidate columns and preserve their imputation flags.
    Only missing precipitation in June 1999 is replaced by zero, explicitly as
    a modelling assumption. Any other missing/nonfinite forcing is rejected.
    ``precipitation_mm`` retains the original, ``precipitation_used_mm`` records
    the forcing, and ``precipitation_assumed_zero`` identifies substitutions.

    Compare zero initial SWE against 1,000 mm per layer, using identical forcing.
    Report the maximum absolute layer SWE difference each day and convergence
    at <= 1e-8 mm. This diagnostic does not assess missing-precipitation error.
    The entire provided daily calendar is simulated, without resetting storage.
    """
    required = {"precipitation_mm"}
    for station in ZONES:
        required.update((f"{station}_temperature_candidate_c", f"{station}_temperature_imputed"))
    missing = required.difference(cleaned.columns)
    if missing:
        raise ValueError(f"Missing cleaned columns: {', '.join(sorted(missing))}.")
    if not cleaned.columns.is_unique:
        raise ValueError("Cleaned column names must be unique.")
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        raise ValueError("Weather inputs require a nonempty daily DatetimeIndex without NaT.")
    forcing = cleaned[["precipitation_mm"]].copy(deep=True)
    for station in ZONES:
        flag = cleaned[f"{station}_temperature_imputed"]
        if not pd.api.types.is_bool_dtype(flag.dtype) or flag.isna().any():
            raise ValueError(f"{station}_temperature_imputed must contain complete Boolean flags.")
        forcing[f"{station}_temperature_c"] = cleaned[f"{station}_temperature_candidate_c"]
    assumed_zero = forcing.precipitation_mm.isna() & (
        (forcing.index.year == 1999) & (forcing.index.month == 6)
    )
    forcing.loc[assumed_zero, "precipitation_mm"] = 0.0
    result = simulate_three_layer_snow(forcing)
    high_initial = simulate_three_layer_snow(forcing, initial_swe_mm=(1000.0,) * 3)
    swe_columns = [f"{layer}_swe_mm" for layer in LAYERS]
    difference = (high_initial[swe_columns] - result[swe_columns]).abs().max(axis=1)
    converged = difference.le(INITIALIZATION_TOLERANCE_MM)
    result["precipitation_mm"] = cleaned.precipitation_mm
    result["precipitation_used_mm"] = forcing.precipitation_mm
    result["precipitation_assumed_zero"] = assumed_zero
    for station in ZONES:
        result[f"{station}_temperature_imputed"] = cleaned[f"{station}_temperature_imputed"]
    result["initialization_max_difference_mm"] = difference
    result["initialization_converged"] = converged
    result.attrs["forcing_provenance"] = {
        "temperature_columns": [f"{station}_temperature_candidate_c" for station in ZONES],
        "temperature_policy": (
            "Observed temperatures retained; gaps use cleaned candidate estimates."
        ),
        "precipitation_column": "precipitation_mm",
        "precipitation_gap_policy": "Only missing June 1999 precipitation is assumed zero.",
        "precipitation_assumed_zero_days": int(assumed_zero.sum()),
        "precipitation_uncertainty": (
            "Effects of assumed precipitation can persist in storage after June 1999; "
            "initialization convergence does not resolve this uncertainty."
        ),
    }
    result.attrs["initialization"] = {
        "reference_initial_swe_mm": [0.0] * 3,
        "comparison_initial_swe_mm": [1000.0] * 3,
        "tolerance_mm": INITIALIZATION_TOLERANCE_MM,
        "first_convergence_date": (
            str(result.index[converged][0].date()) if converged.any() else None
        ),
        "interpretation": (
            "Sensitivity to these two starting states only; not evidence of observed "
            "snow-free conditions or accuracy of the precipitation assumption."
        ),
    }
    result.attrs["period"] = {
        "start": str(result.index[0].date()),
        "end": str(result.index[-1].date()),
        "days": len(result),
    }
    return result
