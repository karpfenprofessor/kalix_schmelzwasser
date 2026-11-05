# Kalix snowmelt and river flow

Explore daily SMHI weather and discharge records to develop spring river-flow
predictions. An exploratory snow water equivalent (SWE) model estimates seasonal
snow storage from the station observations. No river-flow forecast has been fitted;
feature, forecast model, and evaluation modules remain placeholders.

## Getting started

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage Python and dependencies. In PowerShell, from the project root:

```powershell
.\.tools\uv.exe sync --locked
.\.tools\uv.exe run python -m kalix.prepare
.\.tools\uv.exe run jupyter lab
```

Open `notebooks/01_data_quality.ipynb` and select the project's Python kernel.
Sync installs the package in editable mode, so notebooks also work from the
`notebooks/` directory. No environment activation is needed.

If bare `uv` is not recognized, use the project-local `.\.tools\uv.exe` above;
its folder does not need to be on `PATH` (Windows' list of command search folders).
Check the project's Python and installed uv-managed versions with:

```powershell
.\.tools\uv.exe run python --version
.\.tools\uv.exe python list --only-installed
```

`python --version` checks the Python found through `PATH`, which can differ from
the project's `.venv` environment. If `uv` is installed on `PATH`, it can replace
`.\.tools\uv.exe` in these commands.

## Temperature comparison plot

Plot both daily temperature datasets over their shared timeframe:

```powershell
.\.tools\uv.exe run python plot_temperatures.py
```

The script uses `load_processed_data()` to read `data/processed/daily.parquet`
and saves `outputs/figures/temperature_comparison.png`. The loader checks that
prepared data are current. Missing observations appear as gaps.

## Snow water equivalent chart

Generate the exploratory chart and its daily data:

```powershell
.\.tools\uv.exe run python plot_snow.py
```

The script reads current prepared data and saves
`outputs/figures/snow_water_equivalent_2024_2025.png` and
`outputs/figures/snow_water_equivalent_2024_2025.csv`, plus a JSON file with the
same basename recording parameters, dates, weights, and initialization checks.
The CSV includes the central model, all nine SWE scenarios and their range, and
river discharge for comparison. The chart covers
**1 September 2024 to 31 August 2025**, a complete season for both temperature
stations and precipitation.

Each station temperature drives a separate snow store using the same Nikkaluokta
precipitation. All precipitation becomes snow at or below -1 °C and rain at or
above +1 °C; the snow fraction changes linearly between these thresholds. Melt
starts above 0 °C and is capped by available snow, including that day's snowfall.
Precipitation is already in mm water equivalent, so no snow-density conversion is
needed. The model retains the observed temperatures, including winter inversions.

The central estimate gives each store 50% weight and uses a degree-day melt factor
of **3 mm/(°C day)**. The assumption range spans nine combinations: Tarfala weights
of **0.25, 0.50, and 0.75**, each with melt factors of **2, 3, and 5 mm/(°C day)**.
These weights are assumptions, not measured catchment elevation fractions. The
range is neither a confidence interval nor guaranteed bounds on actual SWE.

Simulation starts on **8 May 2024** for warm-up. The script checks that starting
each store at 0 versus 1,000 mm produces the same state by the chart start for
every selected melt factor. Storage then carries forward continuously without an
annual reset. This sensitivity check does not establish observed snow-free ground
or represent glacier and perennial snow storage.

This is an uncalibrated two-station snow-storage proxy. Unknown catchment elevation
weights, mountain precipitation differences, wind redistribution, and other snow
processes limit interpretation as catchment-average SWE. Temperature and
precipitation daily windows end at 00:00 and 06:00 UTC respectively; joining source
date labels approximates their alignment. River discharge can provide an indirect
timing check, but it does not directly measure SWE or uniquely determine the
weights and melt factor. Missing forcing is rejected rather than filled with zero
or skipped, because it can affect storage after the gap.

Reusable calculations are available separately from plotting:

```python
from kalix.data import load_processed_data
from kalix.snow import SnowParameters, simulate_snow

daily = load_processed_data()
snow = simulate_snow(
    daily.loc["2024-05-08":"2025-08-31"],
    parameters=SnowParameters(degree_day_factor=3),
    high_weight=0.5,
    initial_swe_mm=(0.0, 0.0),
)
```

The result retains the daily index and returns `{zone}_{flux}_mm` columns. Zones
are `nikkaluokta`, `tarfala`, and `combined`; flux names are `snowfall`, `rain`,
`melt`, and `swe`. SWE is end-of-day storage; the other quantities are daily water
amounts.

## Layout

| Path | Contents |
| --- | --- |
| `resources/weather_station/` | Original SMHI CSV exports; do not edit in place |
| `src/kalix/` | Import, preparation, snow calculations, and future forecast code |
| `notebooks/` | Data quality, exploration, and model comparison |
| `data/processed/` | Generated daily datasets, metadata, and quality report |
| `outputs/figures/`, `outputs/models/` | Generated plots and fitted models |
| `tests/` | Import, preparation, and freshness checks |
| `pyproject.toml`, `uv.lock` | Package configuration and locked dependencies |

Keep reusable calculations in `src/kalix/` and exploration in notebooks. Generated
data, outputs, and local environments are ignored by Git.

## Source data

The four exports in `resources/weather_station/` contain:

| File | Daily measurement | Unit |
| --- | --- | --- |
| `smhi-opendata_nikkaluokta_temperatur.csv` | Nikkaluokta mean temperature | °C |
| `smhi-opendata_nikkaluokta_niederschlag.csv` | Nikkaluokta precipitation | mm water equivalent |
| `smhi-opendata_kalix_durchfluss.csv` | KAALASJÄRVI discharge | m³/s |
| `smhi-opendata_tarfala_temperatur.csv` | Tarfala A mean temperature | °C |

**Tarfala A** (station **178970**, elevation **1,143.653 m**) has observations from
**22 November 1995 to 31 July 2026**. All four sources are included in preparation
and the daily loaders. Nikkaluokta uses `nikkaluokta_temperature_c`; Tarfala uses the separate
`tarfala_temperature_c` column, with its own quality flags and UTC intervals.

The reader handles metadata preambles and semicolon-delimited tables, preserving
quality flags, available UTC observation intervals, and source metadata in
dataframe `attrs`. All observations in both temperature exports carry flag `Y`;
review source definitions before filtering. Precipitation does not directly measure
snowfall, snow depth, or snow-water storage.

## Prepared data

Run `.\.tools\uv.exe run python -m kalix.prepare` from the project root after
changing a source export. Preparation preserves
the CSVs and writes these ignored files to `data/processed/`:

- `nikkaluokta_temperature.parquet`, `tarfala_temperature.parquet`: separate
  temperature series named for their stations.
- `precipitation.parquet`, `discharge.parquet`: individual precipitation and
  discharge series.
- `daily.parquet` and `daily.csv`: joined working data and an inspection export.
- `metadata.json`: source metadata, units, conventions, checksums, and build details.
- `quality_report.md`: gaps, quality counts, value checks, and monthly/annual completeness.

Each station's temperature file includes its measurement (`nikkaluokta_temperature_c`
or `tarfala_temperature_c`), quality flag, UTC intervals, and missing indicator,
all with the same station prefix. The joined daily files use these column names too.
Rebuilding upgrades the prepared schema to version 3 and removes the former
`temperature.parquet` after successfully writing its replacement.

The four-series overlap is **22 November 1995–30 June 2026**:
**11,179 days**, with **10,409 (93.11%)** complete across all four measurements.

| Series | Missing days | Gaps |
| --- | ---: | --- |
| Nikkaluokta temperature | 100 | Seven gaps of 3–36 days |
| Tarfala temperature | 640 | 44 gaps of 1–128 days |
| Nikkaluokta precipitation | 30 | June 1999 |
| KAALASJÄRVI discharge | 0 | None |

All calendar days, numeric precision, quality flags, and genuine zero precipitation
are retained. Missing values remain missing and have separate Boolean
`nikkaluokta_temperature_missing`, `tarfala_temperature_missing`, `precipitation_missing`,
and `discharge_missing` columns.
No interpolation, imputation, or quality filtering is applied.

```python
from kalix.data import load_processed_data

daily = load_processed_data()
```

The loader verifies source and output checksums and requests a rebuild if files
are missing, stale, or invalid. Prefer it over the inspection CSV to retain types,
UTC timestamps, and metadata. `load_daily_data()` reads all four original exports
directly. Regenerate prepared files instead of editing them.

## Modelling workflow

1. Inspect coverage, gaps, quality flags, and observation intervals.
2. Explore seasonality, pairing each winter with the following spring.
3. Define the forecast issue date, lead time, target, and spring season.
4. Compare a seasonal baseline and regression with a calibrated snow accumulation
   and degree-day melt model, validating on later winters and springs.

Daily joins use source date labels; observation windows differ and are not precisely
synchronized. Use only inputs available at forecast issue time, with weather
scenarios or forecasts for later periods. Fit preprocessing and model parameters on
training years only; reserve the latest years for final evaluation.

Require complete input and target windows unless a missing-data method is explicitly
validated. Do not discard whole years automatically or sum incomplete precipitation
windows as complete totals. Winter gaps can affect later snow-storage estimates;
leave June 1999 precipitation missing. Any future imputation must respect forecast
availability and training/validation boundaries.

## Checks

```powershell
.\.tools\uv.exe run pytest
.\.tools\uv.exe run ruff check .
.\.tools\uv.exe run ruff format --check .
```

Commit `uv.lock` and rerun checks when updating dependencies.
