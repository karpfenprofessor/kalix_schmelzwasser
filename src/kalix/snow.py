"""Exploratory two-station seasonal snow storage, expressed in mm of water.

The stations represent temperature regimes, not known catchment area fractions.
Both receive the same measured precipitation. This simple accounting model omits
snowpack liquid-water retention, refreezing, sublimation, and glacier processes.
Parameters are illustrative assumptions, not calibrated estimates.
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

    degree_day_factor: float = 3.0
    snow_temperature_c: float = -1.0
    rain_temperature_c: float = 1.0
    melt_temperature_c: float = 0.0

    def __post_init__(self) -> None:
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
    parameters = SnowParameters() if parameters is None else parameters
    if not isinstance(parameters, SnowParameters):
        raise TypeError("parameters must be a SnowParameters instance.")
    if not isinstance(high_weight, Real) or not isfinite(high_weight) or not 0 <= high_weight <= 1:
        raise ValueError("high_weight must be a finite number between 0 and 1.")
    try:
        initial = np.asarray(initial_swe_mm, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("initial_swe_mm must contain two finite nonnegative values.") from error
    if initial.shape != (2,) or not np.isfinite(initial).all() or (initial < 0).any():
        raise ValueError("initial_swe_mm must contain two finite nonnegative values.")

    index = daily.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0 or index.hasnans:
        raise ValueError("Weather inputs require a nonempty daily DatetimeIndex without NaT.")
    if not index.equals(index.normalize()) or not index.equals(
        pd.date_range(index[0], periods=len(index), freq="D", name=index.name)
    ):
        raise ValueError("Weather dates must be unique, increasing, and complete daily dates.")
    missing_columns = set(WEATHER_COLUMNS).difference(daily.columns)
    if missing_columns:
        raise ValueError(f"Missing weather columns: {', '.join(sorted(missing_columns))}.")
    try:
        weather = daily.loc[:, list(WEATHER_COLUMNS)].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError("Weather inputs must be numeric, finite, and complete.") from error
    if not np.isfinite(weather).all():
        raise ValueError("Weather inputs must be finite and complete; missing values are invalid.")
    precipitation = weather[:, 2]
    if (precipitation < 0).any():
        raise ValueError("Precipitation must be nonnegative.")

    output = {}
    for zone_index, zone in enumerate(ZONES):
        temperature = weather[:, zone_index]
        snow_fraction = np.clip(
            (parameters.rain_temperature_c - temperature)
            / (parameters.rain_temperature_c - parameters.snow_temperature_c),
            0.0,
            1.0,
        )
        snowfall = precipitation * snow_fraction
        rain = precipitation - snowfall
        potential_melt = parameters.degree_day_factor * np.maximum(
            temperature - parameters.melt_temperature_c, 0.0
        )
        melt = np.empty(len(index), dtype=float)
        swe = np.empty(len(index), dtype=float)
        storage = initial[zone_index]
        for day in range(len(index)):
            available = storage + snowfall[day]
            melt[day] = min(available, potential_melt[day])
            storage = available - melt[day]
            swe[day] = storage
        for flux, values in zip(FLUXES, (snowfall, rain, melt, swe), strict=True):
            output[f"{zone}_{flux}_mm"] = values

    for flux in FLUXES:
        output[f"combined_{flux}_mm"] = (1 - high_weight) * output[
            f"nikkaluokta_{flux}_mm"
        ] + high_weight * output[f"tarfala_{flux}_mm"]
    result = pd.DataFrame(output, index=index.copy())
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
