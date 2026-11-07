"""Causal temperature candidates, withheld-block metrics, and export provenance."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalix import cleanup
from kalix.data import SOURCES, file_sha256, load_processed_data
from kalix.prepare import prepare_data


@pytest.fixture(scope="module")
def daily():
    index = pd.date_range("2000-01-01", "2003-12-31", freq="D", name="date")
    phase = 2 * np.pi * (index.day_of_year.to_numpy() - 1) / np.where(index.is_leap_year, 366, 365)
    donor = -1 + 12 * np.sin(phase) + 2 * np.cos(np.arange(len(index)) * 1.137)
    features = cleanup._features(pd.Series(donor, index=index))
    target = -3 + features @ np.array([0.7, 0.8, 0.5, 0.1, -0.03])
    frame = pd.DataFrame(index=index)
    for variable, (_, _, column) in SOURCES.items():
        values = {
            "nikkaluokta_temperature": donor,
            "tarfala_temperature": target,
            "precipitation": np.where(np.arange(len(index)) % 3, 0.0, 5.0),
            "discharge": 10 + np.arange(len(index)) % 30,
        }
        frame[column] = np.asarray(values[variable], dtype=float)
        frame[f"{variable}_quality"] = pd.Series("Y", index=index, dtype="string")
        if variable != "discharge":
            offset = pd.Timedelta(hours=6 if variable == "precipitation" else 0)
            frame[f"{variable}_start_utc"] = (
                index.tz_localize("UTC") + offset + pd.Timedelta(seconds=1)
            )
            frame[f"{variable}_end_utc"] = index.tz_localize("UTC") + offset + pd.Timedelta(days=1)
    # Exactly 730 paired training days remain before 2002.
    for column in ("nikkaluokta_temperature_c", "tarfala_temperature_c"):
        frame.loc[["2000-04-05", "2002-03-01"], column] = np.nan
    frame.loc["2002-02-01":"2002-02-10", "nikkaluokta_temperature_c"] = np.nan
    frame.loc["2002-05-01":"2002-05-05", "tarfala_temperature_c"] = np.nan
    frame.loc["2002-06-01":"2002-06-30", "precipitation_mm"] = np.nan
    for variable, (_, _, column) in SOURCES.items():
        frame[f"{variable}_missing"] = frame[column].isna()
    frame.attrs = {"test_source": "observations"}
    return frame


def _write_sources(root, daily):
    """Exercise the real preparation and cleanup loaders with small source exports."""
    (root / "pyproject.toml").touch()
    directory = root / "resources" / "weather_station"
    directory.mkdir(parents=True)
    for variable, (filename, source_column, column) in SOURCES.items():
        if variable == "discharge":
            exported = pd.DataFrame(
                {
                    "Datum (svensk sommartid)": daily.index.strftime("%Y-%m-%d"),
                    source_column: daily[column].to_numpy(),
                    "Kvalitet": "G",
                }
            )
        else:
            exported = pd.DataFrame(
                {
                    "Från Datum Tid (UTC)": daily[f"{variable}_start_utc"]
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .to_numpy(),
                    "Till Datum Tid (UTC)": daily[f"{variable}_end_utc"]
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .to_numpy(),
                    "Representativt dygn": daily.index.strftime("%Y-%m-%d"),
                    source_column: daily[column].to_numpy(),
                    "Kvalitet": "Y",
                }
            )
        exported.to_csv(directory / filename, sep=";", index=False, encoding="utf-8")
    prepare_data(root)


@pytest.fixture(scope="module")
def built_root(tmp_path_factory, daily):
    root = tmp_path_factory.mktemp("cleanup-project")
    _write_sources(root, daily)
    cleanup.cleanup_data(root)
    return root


@pytest.fixture
def clean_root(tmp_path, built_root):
    return Path(shutil.copytree(built_root, tmp_path / "project"))


def test_candidates_preserve_inputs_and_fill_only_eligible_missing_values(daily):
    before = daily.copy(deep=True)
    models = cleanup._fit_models(daily)
    result = cleanup._candidate_data(daily, models)
    pd.testing.assert_frame_equal(daily, before)
    pd.testing.assert_frame_equal(result[daily.columns], daily)
    assert models["tarfala_temperature", 2002].paired_days == 730
    assert models["tarfala_temperature", 2002].training_end == "2001-12-31"
    for variable in cleanup.TEMPERATURES:
        original, candidate = result[f"{variable}_c"], result[f"{variable}_candidate_c"]
        pd.testing.assert_series_equal(
            candidate[original.notna()], original.dropna(), check_names=False
        )
        assert result[f"{variable}_imputed"].equals(original.isna() & candidate.notna())
        assert str(result[f"{variable}_imputation_method"].dtype).startswith("string")
        assert pd.isna(result.loc["2000-04-05", f"{variable}_candidate_c"])
        assert pd.isna(result.loc["2002-03-01", f"{variable}_candidate_c"])
        assert result.loc["2002-03-01", f"{variable}_imputation_method"] == "unresolved"
    assert result.loc["2002-05-01":"2002-05-05", "tarfala_temperature_imputed"].all()
    # Recover the known synthetic seasonal relationship in the forward direction.
    expected = -3 + cleanup._features(
        daily.loc["2002-05-01":"2002-05-05", "nikkaluokta_temperature_c"]
    ) @ np.array([0.7, 0.8, 0.5, 0.1, -0.03])
    np.testing.assert_allclose(
        result.loc["2002-05-01":"2002-05-05", "tarfala_temperature_candidate_c"],
        expected,
        atol=1e-10,
    )


def test_same_year_and_future_observations_cannot_change_earlier_predictions(daily):
    original_models = cleanup._fit_models(daily)
    expected = cleanup._candidate_data(daily, original_models)
    changed = daily.copy()
    selection = (changed.index.year >= 2002) & changed.tarfala_temperature_c.notna()
    changed.loc[selection, "tarfala_temperature_c"] += 1000
    changed_models = cleanup._fit_models(changed)
    result = cleanup._candidate_data(changed, changed_models)
    np.testing.assert_allclose(
        result.loc["2002-05-01":"2002-05-05", "tarfala_temperature_candidate_c"],
        expected.loc["2002-05-01":"2002-05-05", "tarfala_temperature_candidate_c"],
    )
    assert (
        original_models["tarfala_temperature", 2002].metadata()
        == changed_models["tarfala_temperature", 2002].metadata()
    )
    assert (
        original_models["tarfala_temperature", 2003].metadata()
        != changed_models["tarfala_temperature", 2003].metadata()
    )


def test_estimated_columns_never_supply_training_pairs_or_donors(daily):
    models = cleanup._fit_models(daily)
    injected = daily.copy()
    for variable in cleanup.TEMPERATURES:
        injected[f"{variable}_candidate_c"] = 1000.0
    injected_models = cleanup._fit_models(injected)
    result = cleanup._candidate_data(injected, injected_models)
    for key, model in models.items():
        assert injected_models[key].metadata() == model.metadata()
    assert pd.isna(result.loc["2002-03-01", "nikkaluokta_temperature_candidate_c"])
    assert pd.isna(result.loc["2002-03-01", "tarfala_temperature_candidate_c"])
    # A single missing paired day moves this year below the 730-day threshold.
    insufficient = daily.copy()
    insufficient.loc["2001-01-01", "nikkaluokta_temperature_c"] = np.nan
    result = cleanup._candidate_data(insufficient, cleanup._fit_models(insufficient))
    assert pd.isna(result.loc["2002-05-01", "tarfala_temperature_candidate_c"])


def test_gap_selection_and_withheld_block_metrics(daily):
    models = cleanup._fit_models(daily)
    model = models["tarfala_temperature", 2002]
    assert cleanup._select_block(daily, model, 1, 14)[0] == pd.Timestamp("2002-01-01")
    assert cleanup._select_block(daily, model, 4, 128) is None
    validation = cleanup._validate_gaps(daily, models)
    case = validation.loc[
        validation.variable.eq("tarfala_temperature")
        & validation.year.eq(2002)
        & validation.gap_days.eq(14)
        & validation.starting_quarter.eq(1)
        & validation.method.eq("cross_station_regression")
    ].iloc[0]
    assert case.status == "ok"
    assert case.rmse_c < 1e-10
    assert case.training_before <= case.start  # Exclusive training cutoff.
    assert case.n_days == 14
    skipped = validation.loc[
        validation.year.eq(2000) & validation.method.eq("cross_station_regression")
    ]
    assert skipped.status.eq("skipped").all()
    assert (validation.loc[validation.method.eq("linear_interpolation"), "gap_days"] <= 3).all()
    assert (
        validation.loc[validation.method.eq("linear_interpolation"), "availability"]
        .str.contains("both observed endpoints")
        .all()
    )
    metrics = cleanup._temperature_metrics(np.array([-2.0, 0.0, 2.0]), np.array([0.0, 2.0, -2.0]))
    assert metrics["snow_threshold_errors"] == 2
    assert metrics["melt_threshold_errors"] == 2
    assert metrics["rain_threshold_errors"] == 2
    assert metrics["squared_error_sum_c2"] == 24


def test_snow_sensitivity_includes_errors_after_the_gap_and_missing_context(daily):
    complete = daily.copy()
    complete[list(cleanup.WEATHER_COLUMNS)] = [-5.0, -5.0, 10.0]
    block = complete.loc["2002-01-01":"2002-01-03"].index
    unchanged = cleanup._snow_metrics(complete, "tarfala_temperature", block, np.full(3, -5.0))
    assert unchanged["swe_max_abs_difference_mm"] == 0
    assert unchanged["peak_melt_shift_days"] is None
    changed = cleanup._snow_metrics(complete, "tarfala_temperature", block, np.full(3, 5.0))
    assert changed["swe_post_gap_max_abs_difference_mm"] > 0
    assert changed["swe_n_days"] == 93
    complete.loc["2001-12-01", "precipitation_mm"] = np.nan
    assert (
        cleanup._snow_metrics(complete, "tarfala_temperature", block, np.full(3, -5.0))[
            "snow_status"
        ]
        == "skipped"
    )
    edge = complete.loc["2000-01-01":"2000-01-03"].index
    assert (
        cleanup._snow_metrics(daily, "tarfala_temperature", edge, np.zeros(3))["snow_status"]
        == "skipped"
    )


def test_parquet_exports_preserve_observations_metadata_and_checksums(built_root):
    observed = load_processed_data(built_root)
    cleaned = cleanup.load_cleaned_data(built_root)
    pd.testing.assert_frame_equal(cleaned[observed.columns], observed)
    directory = built_root / "data" / "cleaned"
    assert {path.name for path in directory.iterdir()} == {
        *cleanup.CLEANED_FILENAMES,
        "metadata.json",
    }
    for variable in SOURCES:
        frame = pd.read_parquet(directory / f"{variable}.parquet")
        pd.testing.assert_frame_equal(frame, cleaned[frame.columns], check_freq=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["experimental"] is True
    assert metadata["inputs"] == cleanup._input_hashes(built_root)
    for name, checksum in metadata["outputs"].items():
        assert file_sha256(directory / name) == checksum
    validation = pd.read_parquet(directory / "validation.parquet")
    assert validation.status.eq("ok").any()
    assert validation.status.eq("skipped").any()
    report = (directory / "quality_report.md").read_text(encoding="utf-8")
    for text in (
        "experimental",
        "128",
        "SWE sensitivity",
        "Complete input",
        "unresolved",
        "June 1999",
        "publication latency",
    ):
        assert text.lower() in report.lower()
    for row in metadata["coverage"]:
        assert row["possible"] == max(0, len(observed) - row["window_days"] + 1)
    original = metadata["temperatures"]["tarfala_temperature"]
    assert original == {"original_missing": 7, "estimated": 5, "unresolved": 2}


@pytest.mark.parametrize(
    "change",
    [
        "metadata_missing",
        "invalid_json",
        "invalid_shape",
        "schema",
        "output",
        "source",
        "processed",
        "rebuilt_processed",
    ],
)
def test_loader_rejects_stale_missing_or_corrupt_data(clean_root, change):
    directory = clean_root / "data" / "cleaned"
    metadata_path = directory / "metadata.json"
    if change == "metadata_missing":
        metadata_path.unlink()
    elif change in ("invalid_json", "invalid_shape"):
        metadata_path.write_text("not json" if change == "invalid_json" else "[]", encoding="utf-8")
    elif change == "schema":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["schema_version"] = -1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif change == "output":
        (directory / "validation.parquet").write_bytes(b"corrupt")
    elif change == "source":
        path = clean_root / "resources" / "weather_station" / SOURCES["precipitation"][0]
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "processed":
        (clean_root / "data" / "processed" / "daily.parquet").write_bytes(b"corrupt")
    else:
        prepare_data(clean_root)
    with pytest.raises(ValueError, match="Rebuild.*kalix.cleanup"):
        cleanup.load_cleaned_data(clean_root)


def test_rebuild_is_reproducible_and_failed_export_preserves_existing_files(
    clean_root, monkeypatch
):
    directory = clean_root / "data" / "cleaned"
    processed_before = cleanup._input_hashes(clean_root)
    before = cleanup.load_cleaned_data(clean_root)
    validation_before = pd.read_parquet(directory / "validation.parquet")
    cleanup.cleanup_data(clean_root)
    pd.testing.assert_frame_equal(cleanup.load_cleaned_data(clean_root), before)
    pd.testing.assert_frame_equal(
        pd.read_parquet(directory / "validation.parquet"), validation_before
    )
    assert cleanup._input_hashes(clean_root) == processed_before
    hashes = {path.name: file_sha256(path) for path in directory.iterdir()}
    original_export = pd.DataFrame.to_parquet
    calls = 0

    def fail_second_export(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated export failure")
        return original_export(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_second_export)
    with pytest.raises(OSError, match="simulated export failure"):
        cleanup.cleanup_data(clean_root)
    assert {path.name: file_sha256(path) for path in directory.iterdir()} == hashes
    pd.testing.assert_frame_equal(cleanup.load_cleaned_data(clean_root), before)


def test_cli_and_insufficient_history_export(clean_root):
    result = subprocess.run(
        [sys.executable, "-m", "kalix.cleanup", "--root", str(clean_root)],
        cwd=clean_root.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "experimental estimates" in result.stdout
    assert len(cleanup.load_cleaned_data(clean_root)) == 1461


def test_tiny_dataset_exports_typed_skipped_cases_and_unresolved_gaps(tmp_path, daily):
    tiny = daily.loc["2000-04-01":"2000-04-10"]
    _write_sources(tmp_path, tiny)
    outputs = cleanup.cleanup_data(tmp_path)
    cleaned = cleanup.load_cleaned_data(tmp_path)
    assert cleaned.index.equals(tiny.index)
    for variable in cleanup.TEMPERATURES:
        assert pd.isna(cleaned.loc["2000-04-05", f"{variable}_candidate_c"])
        assert not cleaned[f"{variable}_imputed"].any()
    validation = pd.read_parquet(outputs["validation.parquet"])
    assert (
        validation.loc[validation.method.ne("linear_interpolation"), "status"].eq("skipped").all()
    )
    assert validation.snow_status.eq("skipped").all()
    assert validation.swe_rmse_mm.dtype == "float64"
    assert str(validation.peak_melt_shift_days.dtype) == "Int64"


def test_source_change_during_cleanup_does_not_publish_outputs(clean_root, monkeypatch):
    directory = clean_root / "data" / "cleaned"
    before = {path.name: file_sha256(path) for path in directory.iterdir()}
    validate = cleanup._validate_gaps

    def edit_source_after_validation(*args, **kwargs):
        result = validate(*args, **kwargs)
        source = clean_root / "resources" / "weather_station" / SOURCES["discharge"][0]
        source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(cleanup, "_validate_gaps", edit_source_after_validation)
    with pytest.raises(ValueError, match="Rebuild.*kalix.prepare"):
        cleanup.cleanup_data(clean_root)
    assert {path.name: file_sha256(path) for path in directory.iterdir()} == before


@pytest.mark.parametrize("calendar", ["missing_day", "duplicate", "unsorted", "empty"])
def test_cleanup_rejects_invalid_calendars(daily, calendar):
    if calendar == "missing_day":
        invalid = daily.drop(daily.index[3])
    elif calendar == "duplicate":
        invalid = pd.concat([daily, daily.iloc[:1]])
    elif calendar == "unsorted":
        invalid = daily.iloc[::-1]
    else:
        invalid = daily.iloc[:0]
    with pytest.raises(ValueError, match="calendar"):
        cleanup._validate_daily(invalid)
