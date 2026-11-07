"""Export experimental temperature estimates without changing observations.

Run ``python -m kalix.cleanup`` after preparation. Candidate temperatures use
only earlier calendar years for fitting and observed same-day donor readings.
They are not approved forecast inputs; interpolation is a validation benchmark
only and requires its right endpoint to be available before forecast issuance.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from kalix.data import (
    PREPARED_FILENAMES,
    SOURCES,
    file_sha256,
    load_processed_data,
    project_root,
)
from kalix.prepare import _gap_intervals, _markdown_table
from kalix.snow import WEATHER_COLUMNS, SnowParameters, simulate_snow

CLEANED_SCHEMA_VERSION = 1
TEMPERATURES = ("nikkaluokta_temperature", "tarfala_temperature")
MIN_PAIRED_DAYS = 730
GAP_LENGTHS = (1, 3, 7, 14, 30, 62, 128)
WINDOW_LENGTHS = (1, 7, 30, 90, 180, 365)
FEATURES = ("donor_c", "annual_sin", "annual_cos", "donor_sin", "donor_cos")
CLEANED_FILENAMES = (
    *(f"{variable}.parquet" for variable in SOURCES),
    "daily.parquet",
    "validation.parquet",
    "quality_report.md",
)
POLICY = (
    "Original observations, quality flags, missing indicators, and dates are unchanged. "
    "Candidate temperature columns are experimental, not automatically approved for forecasting. "
    "Precipitation and discharge are not filled."
)
INTERPOLATION_AVAILABILITY = (
    "Requires both observed endpoints before forecast issuance; not an online gap-filling method."
)


@dataclass
class _YearModel:
    variable: str
    donor: str
    year: int
    paired_days: int
    target_days: int
    training_start: str | None
    training_end: str | None
    regression: LinearRegression | None
    climatology: pd.Series

    def metadata(self) -> dict:
        """Store enough information to reproduce predictions without pickles."""
        regression = self.regression
        return {
            "variable": self.variable,
            "donor": self.donor,
            "prediction_year": self.year,
            "training_before": f"{self.year}-01-01",
            "paired_days": self.paired_days,
            "target_days": self.target_days,
            "paired_training_start": self.training_start,
            "paired_training_end": self.training_end,
            "intercept": None if regression is None else float(regression.intercept_),
            "coefficients": None if regression is None else regression.coef_.tolist(),
            "monthly_climatology_c": {
                str(month): float(value) for month, value in self.climatology.items()
            },
        }


def _features(donor: pd.Series) -> np.ndarray:
    """Use a full annual cycle in both leap years and ordinary years."""
    index = donor.index
    phase = 2 * np.pi * (index.day_of_year.to_numpy() - 1) / np.where(index.is_leap_year, 366, 365)
    values = donor.to_numpy(dtype=float)
    sine, cosine = np.sin(phase), np.cos(phase)
    return np.column_stack((values, sine, cosine, values * sine, values * cosine))


def _validate_daily(daily: pd.DataFrame) -> None:
    index = daily.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or len(index) == 0
        or index.hasnans
        or not index.equals(index.normalize())
        or not index.equals(pd.date_range(index[0], periods=len(index), freq="D", name=index.name))
    ):
        raise ValueError(
            "Cleanup requires a nonempty, unique, increasing, complete daily calendar."
        )
    for variable, (_, _, column) in SOURCES.items():
        required = {column, f"{variable}_quality", f"{variable}_missing"}
        if not required.issubset(daily.columns):
            raise ValueError(f"Missing prepared columns for {variable}.")
        observed = daily[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"Nonfinite observed values in {column}.")


def _fit_models(daily: pd.DataFrame) -> dict[tuple[str, int], _YearModel]:
    """Freeze each year's models before any observation in that year."""
    models = {}
    for variable, donor in (TEMPERATURES, TEMPERATURES[::-1]):
        target_column, donor_column = f"{variable}_c", f"{donor}_c"
        for year in sorted(daily.index.year.unique()):
            year = int(year)
            history = daily.loc[daily.index.year < year]
            paired = history[[target_column, donor_column]].dropna()
            target = history[target_column].dropna()
            regression = None
            if len(paired) >= MIN_PAIRED_DAYS:
                regression = LinearRegression().fit(
                    _features(paired[donor_column]), paired[target_column].to_numpy()
                )
            climatology = target.groupby(target.index.month).mean()
            if len(target) < MIN_PAIRED_DAYS:
                climatology = climatology.iloc[:0]
            models[variable, year] = _YearModel(
                variable=variable,
                donor=donor,
                year=year,
                paired_days=len(paired),
                target_days=len(target),
                training_start=None if paired.empty else str(paired.index[0].date()),
                training_end=None if paired.empty else str(paired.index[-1].date()),
                regression=regression,
                climatology=climatology,
            )
    return models


def _candidate_data(daily: pd.DataFrame, models: dict) -> pd.DataFrame:
    cleaned = daily.copy(deep=True)
    for variable in TEMPERATURES:
        original = daily[f"{variable}_c"]
        candidate = original.copy()
        method = pd.Series("observed", index=daily.index, dtype="string")
        method.loc[original.isna()] = "unresolved"
        for year in sorted(daily.index.year.unique()):
            model = models[variable, int(year)]
            eligible = (
                (daily.index.year == year) & original.isna() & daily[f"{model.donor}_c"].notna()
            )
            if model.regression is not None and eligible.any():
                candidate.loc[eligible] = model.regression.predict(
                    _features(daily.loc[eligible, f"{model.donor}_c"])
                )
                method.loc[eligible] = "cross_station_regression"
        cleaned[f"{variable}_candidate_c"] = candidate.astype("float64")
        cleaned[f"{variable}_imputed"] = original.isna() & candidate.notna()
        cleaned[f"{variable}_imputation_method"] = method
    cleaned.attrs = {**daily.attrs, "cleanup_policy": POLICY}
    return cleaned


def _select_block(daily: pd.DataFrame, model: _YearModel, quarter: int, days: int):
    """Select by coverage alone, never by the resulting prediction error."""
    columns = [f"{model.variable}_c", f"{model.donor}_c"]
    complete = daily[columns].notna().all(axis=1).to_numpy()
    # Prefix counts let every candidate block be checked without slicing a frame.
    absent = np.r_[0, np.cumsum(~complete)]
    starts = np.flatnonzero((daily.index.year == model.year) & (daily.index.quarter == quarter))
    for start in starts:
        stop = start + days
        if (
            stop <= len(daily)
            and daily.index[stop - 1].year == model.year
            and absent[stop] == absent[start]
        ):
            return daily.index[start:stop]
    return None


def _temperature_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {
        "n_days": len(error),
        "squared_error_sum_c2": float(error @ error),
        "error_sum_c": float(error.sum()),
        "rmse_c": float(np.sqrt(np.mean(error**2))),
        "bias_c": float(error.mean()),
        "snow_threshold_errors": int(np.count_nonzero((actual <= -1) != (predicted <= -1))),
        "melt_threshold_errors": int(np.count_nonzero((actual > 0) != (predicted > 0))),
        "rain_threshold_errors": int(np.count_nonzero((actual >= 1) != (predicted >= 1))),
    }


def _snow_metrics(daily: pd.DataFrame, variable: str, block, predicted: np.ndarray) -> dict:
    """Measure perturbations against the same observed-forcing reference run."""
    start, end = block[0] - pd.Timedelta(days=180), block[-1] + pd.Timedelta(days=90)
    calendar = pd.date_range(start, end, freq="D", name=daily.index.name)
    reference = daily.reindex(calendar)[list(WEATHER_COLUMNS)].copy()
    if not np.isfinite(reference.to_numpy(dtype=float)).all():
        return {"snow_status": "skipped", "snow_reason": "incomplete_weather_or_context"}
    if (reference.precipitation_mm < 0).any():
        return {"snow_status": "skipped", "snow_reason": "negative_precipitation"}
    changed = reference.copy()
    changed.loc[block, f"{variable}_c"] = predicted
    # Both runs start at zero 180 days before the artificial gap. This is an
    # identical assumption, not evidence that the actual catchment was snow-free.
    observed_snow = simulate_snow(reference).loc[block[0] :]
    candidate_snow = simulate_snow(changed).loc[block[0] :]
    difference = candidate_snow.combined_swe_mm - observed_snow.combined_swe_mm
    post_gap = difference.loc[block[-1] + pd.Timedelta(days=1) :]
    observed_melt, candidate_melt = observed_snow.combined_melt_mm, candidate_snow.combined_melt_mm
    shift = None
    if observed_melt.max() > 0 and candidate_melt.max() > 0:
        # idxmax selects the earliest date when a peak is tied.
        shift = int((candidate_melt.idxmax() - observed_melt.idxmax()).days)
    return {
        "snow_status": "ok",
        "snow_reason": None,
        "swe_n_days": len(difference),
        "swe_squared_error_sum_mm2": float(difference.to_numpy() @ difference.to_numpy()),
        "swe_rmse_mm": float(np.sqrt(np.mean(difference.to_numpy() ** 2))),
        "swe_max_abs_difference_mm": float(difference.abs().max()),
        "swe_post_gap_max_abs_difference_mm": float(post_gap.abs().max()),
        "swe_end_difference_mm": float(difference.iloc[-1]),
        "peak_melt_shift_days": shift,
    }


def _validate_gaps(daily: pd.DataFrame, models: dict) -> pd.DataFrame:
    rows = []
    for (variable, year), model in models.items():
        for days in GAP_LENGTHS:
            methods = ["cross_station_regression", "monthly_climatology"]
            if days <= 3:
                methods.append("linear_interpolation")
            for quarter in range(1, 5):
                block = _select_block(daily, model, quarter, days)
                for method in methods:
                    row = {
                        "variable": variable,
                        "donor": model.donor,
                        "year": year,
                        "starting_quarter": quarter,
                        "gap_days": days,
                        "method": method,
                        "start": pd.NaT if block is None else block[0],
                        "end": pd.NaT if block is None else block[-1],
                        "training_before": pd.Timestamp(year=year, month=1, day=1),
                        "training_days": (
                            model.paired_days
                            if method == "cross_station_regression"
                            else model.target_days
                            if method == "monthly_climatology"
                            else 0
                        ),
                        "availability": (
                            INTERPOLATION_AVAILABILITY
                            if method == "linear_interpolation"
                            else "Observed donor interval must end before forecast issuance."
                            if method == "cross_station_regression"
                            else "Earlier-year observations only."
                        ),
                        "status": "skipped",
                        "reason": "no_fully_observed_block",
                        "n_days": 0,
                        "snow_status": "skipped",
                        "snow_reason": "temperature_case_skipped",
                    }
                    predicted = None
                    if block is not None:
                        row["reason"] = "insufficient_training_history"
                        if method == "cross_station_regression" and model.regression is not None:
                            predicted = model.regression.predict(
                                _features(daily.loc[block, f"{model.donor}_c"])
                            )
                        elif method == "monthly_climatology":
                            values = model.climatology.reindex(block.month).to_numpy(dtype=float)
                            if np.isfinite(values).all():
                                predicted = values
                        elif method == "linear_interpolation":
                            row["reason"] = "missing_interpolation_endpoints"
                            endpoints = daily[f"{variable}_c"].reindex(
                                [block[0] - pd.Timedelta(days=1), block[-1] + pd.Timedelta(days=1)]
                            )
                            if endpoints.notna().all():
                                predicted = np.linspace(
                                    endpoints.iloc[0], endpoints.iloc[1], days + 2
                                )[1:-1]
                    if predicted is not None:
                        actual = daily.loc[block, f"{variable}_c"].to_numpy(dtype=float)
                        row.update(status="ok", reason=None)
                        row.update(_temperature_metrics(actual, predicted))
                        row.update(_snow_metrics(daily, variable, block, predicted))
                    rows.append(row)
    validation = pd.DataFrame(rows)
    # Stable types also cover small datasets with no successful experiments.
    float_columns = (
        "squared_error_sum_c2",
        "error_sum_c",
        "rmse_c",
        "bias_c",
        "swe_squared_error_sum_mm2",
        "swe_rmse_mm",
        "swe_max_abs_difference_mm",
        "swe_post_gap_max_abs_difference_mm",
        "swe_end_difference_mm",
    )
    int_columns = (
        "snow_threshold_errors",
        "melt_threshold_errors",
        "rain_threshold_errors",
        "swe_n_days",
        "peak_melt_shift_days",
    )
    for column in float_columns:
        validation[column] = validation.get(column, pd.Series(dtype=float)).astype("float64")
    for column in int_columns:
        validation[column] = pd.array(
            validation.get(column, [None] * len(validation)), dtype="Int64"
        )
    for column in (
        "variable",
        "donor",
        "method",
        "availability",
        "status",
        "reason",
        "snow_status",
        "snow_reason",
    ):
        validation[column] = validation[column].astype("string")
    validation.attrs = {
        "interpretation": (
            "Artificial blocks can overlap; aggregate scores are descriptive, "
            "not independent samples."
        ),
        "snow_parameters": asdict(SnowParameters()),
        "snow_initial_swe_mm": [0.0, 0.0],
        "snow_high_weight": 0.5,
        "snow_warmup_days": 180,
        "snow_followup_days": 90,
    }
    return validation


def _coverage(daily: pd.DataFrame) -> list[dict]:
    original = [value for _, _, value in SOURCES.values()]
    candidate = [
        f"{variable}_candidate_c" if variable in TEMPERATURES else value
        for variable, (_, _, value) in SOURCES.items()
    ]
    rows = []
    for label, columns in (
        ("all_original", original),
        ("all_candidates", candidate),
        ("nikkaluokta_precipitation_original", [original[0], "precipitation_mm", "discharge_m3_s"]),
        (
            "nikkaluokta_precipitation_candidate",
            [candidate[0], "precipitation_mm", "discharge_m3_s"],
        ),
    ):
        complete = daily[columns].notna().all(axis=1)
        for days in WINDOW_LENGTHS:
            possible = max(0, len(daily) - days + 1)
            count = int(complete.rolling(days, min_periods=days).sum().eq(days).sum())
            rows.append(
                {"dataset": label, "window_days": days, "possible": possible, "complete": count}
            )
    return rows


def _report(daily: pd.DataFrame, validation: pd.DataFrame, metadata: dict) -> str:
    lines = [
        "# Cleaned-data experimental quality report",
        "",
        POLICY,
        "",
        f"Built (UTC): {metadata['created_at_utc']}",
        "",
        "## Temperature candidates",
        "",
        "Separate regressions use observed donor temperature, annual sine/cosine, and their "
        "interactions. Each prediction year uses only earlier calendar years, with at least "
        "730 observed pairs. No estimated donor or target enters fitting. "
        "Early gaps or unavailable donors remain unresolved.",
        "",
    ]
    lines += _markdown_table(
        ["Variable", "Original missing", "Experimental estimates", "Unresolved"],
        [
            [variable, info["original_missing"], info["estimated"], info["unresolved"]]
            for variable, info in metadata["temperatures"].items()
        ],
    )
    lines += [
        "## Actual temperature gaps",
        "",
        "Long gaps (more than three days) remain experimental even when candidate values exist. "
        "In particular, review Tarfala's winter 2010/11 and spring 2014 outages.",
        "",
    ]
    gaps = []
    for variable in TEMPERATURES:
        for gap in _gap_intervals(daily[f"{variable}_c"]):
            section = daily.loc[gap["start"] : gap["end"]]
            gaps.append(
                [
                    variable,
                    gap["start"],
                    gap["end"],
                    gap["days"],
                    int(section[f"{variable}_imputed"].sum()),
                    "long; experimental" if gap["days"] > 3 else "experimental",
                ]
            )
    lines += _markdown_table(["Variable", "Start", "End", "Days", "Estimated", "Status"], gaps)
    lines += [
        "## Artificial-gap temperature validation",
        "",
        "For every station, year, gap length, and starting quarter, use the earliest "
        "fully paired observed block ending in that year. The target block is withheld "
        "from prediction; yearly models were fitted before the year began. "
        "Monthly climatology uses earlier observed target readings (minimum 730 days). "
        "Interpolation is evaluated only for one- and three-day gaps. "
        + INTERPOLATION_AVAILABILITY,
        "",
        "Cases can overlap across lengths and quarters. Scores weight cases by their day "
        "counts, including overlap; they do not measure independent forecast accuracy. "
        "Threshold error rates compare T <= -1, T > 0, and T >= 1 degrees Celsius.",
        "",
    ]
    summary = []
    for (variable, method, length), cases in validation.groupby(
        ["variable", "method", "gap_days"], sort=True
    ):
        valid = cases.loc[cases.status.eq("ok")]
        n = int(valid.n_days.sum())
        summary.append(
            [
                variable,
                method,
                int(length),
                len(valid),
                len(cases) - len(valid),
                n,
                f"{np.sqrt(valid.squared_error_sum_c2.sum() / n):.3f}" if n else "n/a",
                f"{valid.error_sum_c.sum() / n:.3f}" if n else "n/a",
                *[
                    f"{valid[column].sum() / n:.2%}" if n else "n/a"
                    for column in (
                        "snow_threshold_errors",
                        "melt_threshold_errors",
                        "rain_threshold_errors",
                    )
                ],
            ]
        )
    lines += _markdown_table(
        [
            "Variable",
            "Method",
            "Gap days",
            "Scored",
            "Skipped",
            "Scored days",
            "RMSE C",
            "Bias C",
            "Snow threshold error",
            "Melt threshold error",
            "Rain threshold error",
        ],
        summary,
    )
    lines += ["## Skipped temperature cases", ""]
    skipped = validation.loc[validation.status.eq("skipped")]
    lines += _markdown_table(
        ["Method", "Reason", "Cases"],
        [
            [method, reason, len(group)]
            for (method, reason), group in skipped.groupby(["method", "reason"], sort=True)
        ],
    )
    lines += [
        "## SWE sensitivity",
        "",
        "Use complete original weather for 180 days before each block through 90 days "
        "after it. Both simulations start at (0, 0) mm, use the existing default snow "
        "parameters and Tarfala weight 0.5, and differ only in the withheld temperature "
        "block. This does not establish actual snow-free ground or measured SWE accuracy. "
        "Errors are combined-store SWE from gap start through follow-up. Peak shifts "
        "use the earliest maximum daily combined melt in that same interval; shifts "
        "are undefined if either run has no melt.",
        "",
    ]
    snow_rows = []
    for (variable, method, length), cases in validation.groupby(
        ["variable", "method", "gap_days"], sort=True
    ):
        snow = cases.loc[cases.snow_status.eq("ok")]
        n = int(snow.swe_n_days.sum())
        shifts = snow.peak_melt_shift_days.dropna()
        snow_rows.append(
            [
                variable,
                method,
                int(length),
                len(snow),
                len(cases) - len(snow),
                f"{np.sqrt(snow.swe_squared_error_sum_mm2.sum() / n):.3f}" if n else "n/a",
                f"{snow.swe_max_abs_difference_mm.max():.3f}" if n else "n/a",
                f"{snow.swe_post_gap_max_abs_difference_mm.max():.3f}" if n else "n/a",
                f"{shifts.abs().mean():.2f}" if len(shifts) else "n/a",
            ]
        )
    lines += _markdown_table(
        [
            "Variable",
            "Method",
            "Gap days",
            "Scored",
            "Skipped",
            "SWE RMSE mm",
            "Max SWE difference mm",
            "Max post-gap difference mm",
            "Mean absolute peak shift days",
        ],
        snow_rows,
    )
    lines += [
        "## Complete input and target windows",
        "",
        "Counts require discharge as well as the named weather inputs. They describe "
        "daily windows, not independent training samples or valid initialized SWE states.",
        "",
    ]
    lines += _markdown_table(
        ["Dataset", "Window days", "Possible", "Complete", "Complete fraction"],
        [
            [
                row["dataset"],
                row["window_days"],
                row["possible"],
                row["complete"],
                f"{row['complete'] / row['possible']:.2%}" if row["possible"] else "n/a",
            ]
            for row in metadata["coverage"]
        ],
    )
    lines += [
        "## Remaining limits and forecast use",
        "",
        f"- Precipitation still has {int(daily.precipitation_mm.isna().sum())} missing days. "
        "June 1999 remains missing; never replace it with zero or report partial "
        "totals as complete.",
        "- Missing winter forcing can affect SWE after a gap. Resume only with documented "
        "initialization or reconstruction; do not automatically reset storage to zero.",
        "- Candidates require the donor observation to have been available at forecast issue "
        "time. Source interval timestamps are retained, but do not establish publication latency "
        "or whether an archived observation was later revised.",
        "- The existing source daily windows are not synchronized; cleanup does not change them.",
        "- No candidates are approved automatically. Existing plots and snow calculations "
        "continue using observations unless an explicit candidate-column selection is made.",
        "- Define the forecast target and issue date before testing forecast benefit. Refit "
        "preprocessing within each training split; fixed multi-year holdouts must also freeze "
        "fitting at their training cutoff rather than reuse these annually refitted candidates.",
        "- Compare forecasting methods on the same evaluation dates and report coverage "
        "separately. These imputation experiments are not final forecast evaluation.",
        "",
    ]
    return "\n".join(lines)


def _input_hashes(root: Path) -> dict[str, str]:
    directory = root / "data" / "processed"
    return {name: file_sha256(directory / name) for name in (*PREPARED_FILENAMES, "metadata.json")}


def cleanup_data(root: str | Path | None = None) -> dict[str, Path]:
    """Validate prepared inputs, evaluate candidates, and publish separate outputs."""
    root = project_root(root)
    inputs = _input_hashes(root)
    daily = load_processed_data(root)
    _validate_daily(daily)
    models = _fit_models(daily)
    cleaned = _candidate_data(daily, models)
    validation = _validate_gaps(daily, models)
    metadata = {
        "schema_version": CLEANED_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": POLICY,
        "experimental": True,
        "common_range": {
            "start": str(daily.index[0].date()),
            "end": str(daily.index[-1].date()),
            "rows": len(daily),
        },
        "inputs": inputs,
        "features": list(FEATURES),
        "minimum_training_days": MIN_PAIRED_DAYS,
        "gap_lengths": list(GAP_LENGTHS),
        "models": [model.metadata() for model in models.values()],
        "temperatures": {
            variable: {
                "original_missing": int(daily[f"{variable}_c"].isna().sum()),
                "estimated": int(cleaned[f"{variable}_imputed"].sum()),
                "unresolved": int(cleaned[f"{variable}_candidate_c"].isna().sum()),
            }
            for variable in TEMPERATURES
        },
        "coverage": _coverage(cleaned),
        "validation": {
            **validation.attrs,
            "cases": len(validation),
            "scored": int(validation.status.eq("ok").sum()),
            "snow_scored": int(validation.snow_status.eq("ok").sum()),
            "interpolation_availability": INTERPOLATION_AVAILABILITY,
        },
        "outputs": {},
    }
    report = _report(cleaned, validation, metadata)
    directory = root / "data" / "cleaned"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cleanup-", dir=directory) as temporary:
        staging = Path(temporary)
        for variable in SOURCES:
            frame = cleaned[[column for column in cleaned if column.startswith(f"{variable}_")]]
            frame.to_parquet(
                staging / f"{variable}.parquet", engine="pyarrow", compression="snappy"
            )
        cleaned.to_parquet(staging / "daily.parquet", engine="pyarrow", compression="snappy")
        validation.to_parquet(
            staging / "validation.parquet", engine="pyarrow", compression="snappy", index=False
        )
        (staging / "quality_report.md").write_text(report, encoding="utf-8", newline="\n")
        metadata["outputs"] = {name: file_sha256(staging / name) for name in CLEANED_FILENAMES}
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if _input_hashes(root) != inputs:
            raise ValueError("Prepared inputs changed during cleanup. Run cleanup again.")
        load_processed_data(root)  # Also catch edits to the original source exports.
        for name in (*CLEANED_FILENAMES, "metadata.json"):
            (staging / name).replace(directory / name)
    return {name: directory / name for name in (*CLEANED_FILENAMES, "metadata.json")}


def load_cleaned_data(root: str | Path | None = None) -> pd.DataFrame:
    """Return observations plus candidates only when all provenance checks pass."""
    root = project_root(root)
    directory = root / "data" / "cleaned"
    rebuild = f'Rebuild with: uv run python -m kalix.cleanup --root "{root}"'
    try:
        load_processed_data(root)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != CLEANED_SCHEMA_VERSION:
            raise ValueError("Cleaned-data schema version has changed.")
        if metadata["inputs"] != _input_hashes(root):
            raise ValueError("Prepared inputs have changed.")
        for name in CLEANED_FILENAMES:
            if file_sha256(directory / name) != metadata["outputs"][name]:
                raise ValueError(f"Cleaned output has changed: {name}.")
        daily = pd.read_parquet(directory / "daily.parquet", engine="pyarrow")
        _validate_daily(daily)
        daily.index = pd.DatetimeIndex(daily.index, freq="D")
        return daily
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cleaned data are missing, stale, or invalid. {rebuild}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="Project root (otherwise found from the current directory)."
    )
    arguments = parser.parse_args()
    try:
        paths = cleanup_data(arguments.root)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cleanup failed: {error}\n")
    metadata = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
    for variable, info in metadata["temperatures"].items():
        print(
            f"{variable}: {info['estimated']} experimental estimates; "
            f"{info['unresolved']} unresolved."
        )
    print(f"Cleaned Parquet files and report: {paths['metadata.json'].parent}")


if __name__ == "__main__":
    main()
