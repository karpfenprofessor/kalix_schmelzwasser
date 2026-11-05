# Kalix snowmelt and river flow

Explore daily SMHI weather and discharge records to develop spring river-flow
predictions. Feature, snow, model, and evaluation modules are placeholders; no model
has been trained.

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

## Layout

| Path | Contents |
| --- | --- |
| `resources/weather_station/` | Original SMHI CSV exports; do not edit in place |
| `src/kalix/` | Import, preparation, and future modelling code |
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
