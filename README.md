# Kalix snowmelt and river flow

Explore daily SMHI weather and discharge records to develop spring river-flow
predictions. Feature, snow, model, and evaluation modules are placeholders; no model
has been trained.

## Getting started

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the project root:

```powershell
uv sync --locked
uv run python -m kalix.prepare
uv run jupyter lab
```

Open `notebooks/01_data_quality.ipynb` and select the project's Python kernel.
`uv sync` installs the package in editable mode, so notebooks also work from the
`notebooks/` directory. For the project-local executable, replace `uv` with
`.\.tools\uv.exe`; no environment activation is needed.

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

The new **Tarfala A** export (station **178970**, elevation **1,143.653 m**) has
observations from **22 November 1995 to 31 July 2026**. It is not yet included in
the daily loaders or preparation pipeline, which still use the three original
series. Read it separately with:

```python
from kalix.data import read_station_csv

tarfala = read_station_csv(
    "resources/weather_station/smhi-opendata_tarfala_temperatur.csv", "temperature"
)
```

The reader handles metadata preambles and semicolon-delimited tables, preserving
quality flags, available UTC observation intervals, and source metadata in
dataframe `attrs`. All observations in both temperature exports carry flag `Y`;
review source definitions before filtering. Precipitation does not directly measure
snowfall, snow depth, or snow-water storage.

## Prepared data

Run `uv run python -m kalix.prepare` after changing a pipeline source. From outside
the checkout, add `--root "C:/path/to/kalix_schmelzwasser"`. Preparation preserves
the CSVs and writes these ignored files to `data/processed/`:

- `temperature.parquet`, `precipitation.parquet`, `discharge.parquet`: individual
  series with quality flags, available UTC intervals, and missing indicators.
- `daily.parquet` and `daily.csv`: joined working data and an inspection export.
- `metadata.json`: source metadata, units, conventions, checksums, and build details.
- `quality_report.md`: gaps, quality counts, value checks, and monthly/annual completeness.

The current three-series overlap is **1 October 1995–30 June 2026**:
**11,231 days**, with **11,100 (98.83%)** complete across all three measurements.
Tarfala is excluded from these figures.

| Series | Missing days | Gaps |
| --- | ---: | --- |
| Nikkaluokta temperature | 101 | Eight gaps of 1–36 days |
| Nikkaluokta precipitation | 30 | June 1999 |
| KAALASJÄRVI discharge | 0 | None |

All calendar days, numeric precision, quality flags, and genuine zero precipitation
are retained. Missing values remain missing and have separate Boolean
`temperature_missing`, `precipitation_missing`, and `discharge_missing` columns.
No interpolation, imputation, or quality filtering is applied.

```python
from kalix.data import load_processed_data

daily = load_processed_data()
```

The loader verifies source and output checksums and requests a rebuild if files
are missing, stale, or invalid. Prefer it over the inspection CSV to retain types,
UTC timestamps, and metadata. `load_daily_data()` reads the original three exports
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
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Commit `uv.lock` and rerun checks when updating dependencies.
