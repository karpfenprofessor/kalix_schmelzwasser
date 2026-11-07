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

## Three-layer snow storage from cleaned data

Set `START_DATE` and `END_DATE` near the top of `src/plot/plot_snow.py`
(both inclusive, in `YYYY-MM-DD` format), then generate the PNG chart:

```powershell
.venv\Scripts\python.exe src/plot/plot_snow.py
```

The chart labels time by month and year and shows four lines: Nikkaluokta,
middle, Tarfala, and the **sum of the three stores**. The sum adds their SWE depths;
the equal-area average is this sum divided by three. Storage is simulated over the
full cleaned record before selecting the dates. The output filename includes the
selected dates, for example
`outputs/figures/snow_water_equivalent_2024-09-01_2025-08-31.png`.

For the same four snow-storage lines with observed discharge on a separate
right-hand axis, edit `START_DATE` and `END_DATE` in
`src/plot/plot_snow_discharge.py` and run:

```powershell
.venv\Scripts\python.exe src/plot/plot_snow_discharge.py
```

Snow storage uses mm water equivalent on the left axis; the orange discharge
line uses m³/s on the right. Discharge is aligned to the same daily dates,
without smoothing or normalization. Missing, nonfinite, or negative discharge
within the selected interval is rejected; observed zero values are retained.
The output is `outputs/figures/snow_discharge_<start>_<end>.png`.

Open `notebooks/check_snow.ipynb` and run all cells to calculate the full daily
record, **22 November 1995 to 30 June 2026**. Edit `VIEW_START`, `VIEW_END`,
`TABLE_START`, and `TABLE_DAYS` to inspect a selected interval without restarting
the simulation. The notebook includes temperature and SWE charts, initialization
sensitivity, a daily table, and a detailed view of the June 1999 assumption.

The model represents three equal-area layers:

| Layer | Elevation | Daily temperature | Area weight |
| --- | --- | --- | --- |
| Nikkaluokta | 466.765 m | Cleaned Nikkaluokta candidate | 1/3 |
| Middle | 805.209 m | `(Nikkaluokta + Tarfala) / 2` | 1/3 |
| Tarfala | 1,143.653 m | Cleaned Tarfala candidate | 1/3 |

The middle temperature assumes a linear profile between station elevations each
day, including inversions. Candidate temperatures preserve observations and fill
their gaps with the existing experimental estimates. Each layer receives the
**full precipitation depth**; it is not divided by three. Simulate each layer
independently before averaging their fluxes and stores.

For each layer and day, the existing snow parameters give:

```text
snow_fraction = clip((1 - temperature_c) / 2, 0, 1)
snowfall_mm = precipitation_used_mm * snow_fraction
rain_mm = precipitation_used_mm - snowfall_mm
potential_melt_mm = 3 * max(temperature_c, 0)
melt_mm = min(previous_swe_mm + snowfall_mm, potential_melt_mm)
swe_mm = previous_swe_mm + snowfall_mm - melt_mm
combined_swe_mm = (nikkaluokta_swe_mm + middle_swe_mm + tarfala_swe_mm) / 3
```

All snow falls at or below -1 °C, all rain at or above +1 °C, and melt begins above
0 °C. Storage carries forward without annual resets. SWE is **mm water equivalent**,
not physical snow depth or total water volume. Equal areas, uniform precipitation,
and the melt parameters are modelling assumptions, not calibrated catchment facts.

```python
from kalix.cleanup import load_cleaned_data
from kalix.snow import analyze_cleaned_snow

cleaned = load_cleaned_data()
snow = analyze_cleaned_snow(cleaned)
```

The analysis uses both `*_temperature_candidate_c` columns and retains their
`*_temperature_imputed` flags. It assumes **zero precipitation only where readings
are missing in June 1999**, without editing cleaned files. `precipitation_mm`
retains the original missing values; `precipitation_used_mm` records model forcing;
`precipitation_assumed_zero` flags substitutions. Unexpected missing forcing,
nonfinite values, and negative precipitation are rejected. Missing June rainfall
can affect storage after the month ends.

Each layer starts at zero before the first day. A comparison run starts each at
1,000 mm under identical weather. `initialization_max_difference_mm` reports the
largest absolute layer SWE difference each day; `initialization_converged` is true
when all three differences are at most `1e-8` mm. The first convergence date is
recorded in `snow.attrs["initialization"]`, or `None` if no convergence occurs.
This comparison does not establish actual snow-free ground or resolve uncertainty
from the assumed precipitation. Other assumptions and the simulated period are
also recorded in dataframe metadata.

For custom forcing or initial storage, use the strict calculation interface:

```python
from kalix.snow import SnowParameters, simulate_three_layer_snow

# forcing has the original two station temperature column names and precipitation_mm.
snow = simulate_three_layer_snow(
    forcing,
    parameters=SnowParameters(degree_day_factor=3),
    initial_swe_mm=(0.0, 0.0, 0.0),  # Nikkaluokta, middle, Tarfala
)
```

This function returns three `{layer}_temperature_c` columns and
`{zone}_{flux}_mm` for `nikkaluokta`, `middle`, `tarfala`, and `combined`, with
fluxes `snowfall`, `rain`, `melt`, and `swe`. It requires complete daily forcing
and makes no missing-precipitation substitutions. Both interfaces preserve their
inputs. The original two-station `simulate_snow()` API and its cleanup-validation
benchmarks remain available with their existing behaviour.

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

## Experimental temperature cleanup

Generate separate candidate datasets and a missing-data experiment report:

```powershell
python -m kalix.cleanup
```

Use the project's Python environment (for example, `.venv\Scripts\python.exe`),
or run `uv run python -m kalix.cleanup`. The command also accepts `--root`.
It validates the prepared inputs and writes five daily Parquet files to
`data/cleaned/`: `daily.parquet` and the four station/variable files using the
same names as `data/processed/`. It also writes `validation.parquet`,
`quality_report.md`, and a checksum/provenance manifest in `metadata.json`.
Generated cleaned outputs are ignored by Git.

All original columns, observations, source quality codes, missing flags, UTC
intervals, and dates remain intact. Each temperature station gains three columns:

- `<station>_temperature_candidate_c`: observed temperature where available,
  an experimental regression estimate where possible, otherwise missing.
- `<station>_temperature_imputed`: true only for an estimated missing observation.
- `<station>_temperature_imputation_method`: `observed`,
  `cross_station_regression`, or `unresolved`.

Each station has its own seasonal linear regression using the other station's
observed daily temperature, annual sine/cosine terms, and their interactions.
Models for a calendar year use only earlier years and require at least 730 paired
observations. Estimated readings never supply training pairs or donor values.
Precipitation and discharge are not filled; June 1999 precipitation stays missing.

```python
from kalix.cleanup import load_cleaned_data

cleaned = load_cleaned_data()
```

The loader checks prepared-input provenance and every cleaned output checksum.
Regenerate outputs instead of editing Parquet files. Estimates are separate
columns; existing plots and snow calculations continue using observations.

Open `notebooks/check_cleaned.ipynb` to compare both temperature curves and inspect
inserted values in a different colour. It includes six example gaps, a complete
gap inventory, and a selectable detailed chart and daily table. By default,
each zoom includes seven observed readings before and after the gap.

Validation hides fully observed blocks of 1, 3, 7, 14, 30, 62, and 128 days in
later years. It compares regression with earlier-year monthly climatology and,
for gaps up to three days, bounded interpolation. Interpolation is a benchmark
only: both endpoints must be available before issuing a forecast. The validation
table records every station/year/length/starting-quarter/method case, its status,
skip reason, temperature error, and snow sensitivity when sufficient complete
weather exists. Blocks may overlap; the scores are descriptive.

SWE sensitivity uses identical zero initialization 180 days before each block,
default snow parameters, equal station weights, and a 90-day follow-up. It
compares candidate forcing against observed forcing, not measured SWE. Missing
forcing can affect storage after the gap; do not restart SWE at zero without a
documented initialization or reconstruction.

Candidates remain experimental even when scores look good. Donor observations
must have been available at forecast issuance; archived interval timestamps do
not prove publication latency or absence of later revisions. Annually refitted
candidates fit rolling yearly evaluation, but fixed multi-year holdouts require
refitting and freezing preprocessing at their own training cutoff. Define the
forecast target and issue date before comparing forecast performance on common
evaluation dates, with coverage reported separately.

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
