"""Exploratory two-station seasonal snow storage, expressed in mm of water.

The stations represent temperature regimes, not known catchment area fractions.
Both receive the same measured precipitation. This simple accounting model omits
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
    try:
        # NumPy arrays support numeric checks and calculations over many values.
        initial = np.asarray(initial_swe_mm, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("initial_swe_mm must contain two finite nonnegative values.") from error
    if initial.shape != (2,) or not np.isfinite(initial).all() or (initial < 0).any():
        raise ValueError("initial_swe_mm must contain two finite nonnegative values.")

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

    output = {}
    # Calculate each station separately before weighting outputs. Averaging
    # temperatures first would change the threshold-dependent snow/melt results.
    for zone_index, zone in enumerate(ZONES):
        temperature = weather[:, zone_index]
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
        melt = np.empty(len(index), dtype=float)
        swe = np.empty(len(index), dtype=float)
        storage = initial[zone_index]
        # Storage depends on yesterday, so this part proceeds one day at a time.
        for day in range(len(index)):
            available = storage + snowfall[day]
            # Actual melt cannot exceed the snow available, including new snow.
            melt[day] = min(available, potential_melt[day])
            storage = available - melt[day]
            swe[day] = storage
        # zip pairs names with arrays; strict=True detects unequal sequence lengths.
        for flux, values in zip(FLUXES, (snowfall, rain, melt, swe), strict=True):
            output[f"{zone}_{flux}_mm"] = values

    # A weighted average: high_weight belongs to Tarfala; the rest to Nikkaluokta.
    for flux in FLUXES:
        output[f"combined_{flux}_mm"] = (1 - high_weight) * output[
            f"nikkaluokta_{flux}_mm"
        ] + high_weight * output[f"tarfala_{flux}_mm"]
    # Each dictionary entry becomes a named result column on the same daily dates.
    result = pd.DataFrame(output, index=index.copy())
    # Store assumptions alongside the table without mixing them into numeric data.
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
