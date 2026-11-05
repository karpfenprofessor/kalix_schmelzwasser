# Kalix snowmelt and river flow

Python project for exploring daily SMHI weather and discharge records, then developing
and evaluating spring river-flow predictions.

## Getting started

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the project root:

```powershell
uv sync --locked
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
src/kalix/data.py             SMHI import and daily date alignment
src/kalix/features.py         Future predictor calculations
src/kalix/snow.py             Future snow accumulation and melt model
src/kalix/models.py           Future baseline and prediction models
src/kalix/evaluation.py       Future chronological evaluation
tests/                        Data-import checks
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

`load_daily_data()` returns one row per calendar date within the overlapping date
bounds of the three sources. Absent observations remain missing; no interpolation,
quality filtering, or replacement with zero is performed. This is a join by source
date labels. Temperature, precipitation, and discharge use different observation
windows, so the daily table must not yet be treated as precisely synchronized
forecast inputs. Check observation availability against the forecast issue time
before building predictors.

All temperature records in the current export have quality flag `Y`. Investigate
the source definitions before choosing a quality policy. The precipitation series
does not directly measure snowfall, snow depth, or snow-water storage.

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
