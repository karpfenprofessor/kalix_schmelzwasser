# Kalix snowmelt and river flow

Python project for exploring daily SMHI weather and discharge records, then developing
and evaluating spring river-flow predictions.

## Getting started

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the project root:

```powershell
uv sync --locked
uv run python -m kalix.prepare
uv run jupyter lab
```

Open `notebooks/01_data_quality.ipynb` first and select the project's Python kernel.
The project package is installed in editable mode by `uv sync`; notebooks work when
launched from either the root directory or the `notebooks` directory.

If using the project-local uv executable, replace `uv` in these commands with
`.\.tools\uv.exe`. No environment activation is required.

## Layout

```text
resources/weather_station/     Original SMHI exports; do not edit in place
data/processed/                Generated, cleaned daily datasets
notebooks/                    Data quality, exploration, and model comparison
src/kalix/data.py             SMHI import, daily alignment, and prepared-data loading
src/kalix/prepare.py          Reproducible common-range working copies and quality report
src/kalix/features.py         Future predictor calculations
src/kalix/snow.py             Future snow accumulation and melt model
src/kalix/models.py           Future baseline and prediction models
src/kalix/evaluation.py       Future chronological evaluation
tests/                        Import, preparation, and freshness checks
outputs/figures/              Generated plots
outputs/models/               Fitted models and associated metadata
pyproject.toml                Package, dependencies, and check configuration
uv.lock                       Exact dependency versions; commit this file
```

Keep reusable calculations in `src/kalix/`. Use notebooks for exploration and
explanation. Generated datasets, figures, model files, and local environments are
ignored by Git; their directories contain tracked placeholders.

## Source data and conventions

The three files in `resources/weather_station/` contain:

- Nikkaluokta daily mean temperature, in degrees Celsius.
- Nikkaluokta daily precipitation, in millimetres of water equivalent.
- KAALASJÄRVI daily discharge, in cubic metres per second.

The exports contain metadata before the semicolon-delimited observation table and
may have trailing explanatory columns. The reader finds the table header and keeps
observation quality flags. Metadata are available in each source dataframe's
`attrs`. Temperature and precipitation observation intervals are retained as UTC
timestamps. The discharge source's date convention is retained in its metadata.

`load_processed_data()` loads the joined Parquet working copy, checking source and
output checksums before use. `load_daily_data()` remains available for reading the
original exports directly; `read_station_csv()` still reads a single source.
Both daily loaders join by source date labels. Temperature, precipitation, and
discharge use different observation windows, so the daily table must not yet be
treated as precisely synchronized forecast inputs. Check observation availability
against the forecast issue time before building predictors.

All temperature records in the current export have quality flag `Y`. Investigate
the source definitions before choosing a quality policy. The precipitation series
does not directly measure snowfall, snow depth, or snow-water storage.

## Prepared working copies

Run `uv run python -m kalix.prepare` whenever the source exports change. From another
directory, supply the checkout root with `--root "C:/path/to/kalix_schmelzwasser"`.
Preparation preserves the original CSVs and rebuilds the following ignored outputs
under `data/processed/`:

- `temperature.parquet`, `precipitation.parquet`, and `discharge.parquet`: individual
  measurements, source quality flags, available UTC intervals, and missing indicators.
- `daily.parquet`: authoritative joined working data for Python.
- `daily.csv`: a human-readable inspection export of the same daily table.
- `metadata.json`: station/source metadata, units, date conventions, quality-code
  explanations, checksums, processing version, and build time.
- `quality_report.md`: every missing-data gap, quality-code counts, basic value
  checks, and monthly/annual completeness.

Every working table uses the same continuous daily calendar, bounded by the overlap
of the three exports. With the current files this is **1 October 1995 through
30 June 2026**, inclusive: **11,231 days**, of which **11,100 (98.83%)** have all
three measurements.

| Measurement | Missing days | Main gaps |
| --- | ---: | --- |
| Temperature | 101 | Eight gaps, lasting 1 to 36 days |
| Precipitation | 30 | All of June 1999 |
| Discharge | 0 | None |

The joined data include Boolean `temperature_missing`, `precipitation_missing`, and
`discharge_missing` columns. Missingness is separate from source quality: a value
with a quality flag is still an observation, and unchecked recent discharge remains
identifiable by its `O` flag. No dates are dropped, no measurements are imputed, no
quality categories are filtered, and genuine zero precipitation is preserved.
Numeric precision and available observation timestamps are retained.

Use `from kalix.data import load_processed_data` and call
`daily = load_processed_data()` inside the checkout. Missing, stale, or invalid
prepared files produce a rebuild instruction. Do not edit generated files manually;
regenerate them from the original exports. The CSV is for inspection; prefer the
Parquet loader to retain types, UTC timestamps, and source metadata.

For later models, exclude samples whose required input or target windows contain
unresolved gaps, rather than removing an entire year automatically. Never sum an
incomplete precipitation window as though it were complete. Missing winter weather
can affect a calculated snow-storage state beyond the gap itself. A later experiment
may test interpolation of bounded temperature gaps of at most three days in separate
estimated-value columns (only four days in these exports), but this is not enabled
in the preparation pipeline. Leave June 1999 precipitation missing: neither zero
rainfall nor an interpolated daily series is supported by these files. Any later
imputation must respect forecast-time availability and training/validation boundaries.

## Workflow

1. Inspect coverage, gaps, quality flags, and daily observation intervals.
2. Explore seasonality and associate each winter with the following spring.
3. Define the forecast issue date, lead time, target, and spring season.
4. Establish a historical seasonal baseline and a small regression.
5. Develop a calibrated snow accumulation and degree-day melt calculation.
6. Compare models using complete later winters and springs as validation blocks.

Future observed weather cannot be an input to a forecast issued before those
observations become available. Seasonal forecasts will need weather scenarios or
forecasts for the period after the issue date. Fit preprocessing and model parameters
on training years only, and reserve the latest years for a final evaluation.

The feature, snow, model, and evaluation modules are intentionally placeholders.
No model has been trained, and no snowfall estimates are presented as observations.

## Checks

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Git is initialized locally; no initial commit has been made. Review files with
`git status` before making the first commit. Dependency updates should be deliberate:
update `uv.lock` and rerun the checks together.
