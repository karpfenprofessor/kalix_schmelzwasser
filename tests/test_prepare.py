"""Checks for reproducible, common-calendar working data and its provenance."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from kalix.data import SOURCES, load_daily_data, load_processed_data
from kalix.prepare import prepare_data

OUTPUT_NAMES = {
    "temperature.parquet",
    "precipitation.parquet",
    "discharge.parquet",
    "daily.parquet",
    "daily.csv",
    "metadata.json",
    "quality_report.md",
}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_checksums(root):
    directory = root / "resources" / "weather_station"
    return {filename: checksum(directory / filename) for filename, _, _ in SOURCES.values()}


def output_checksums(root):
    directory = root / "data" / "processed"
    return {name: checksum(directory / name) for name in OUTPUT_NAMES}


@pytest.fixture
def source_root(tmp_path):
    """Unequal source coverage, absent days, a blank reading, and real zero rain."""
    (tmp_path / "pyproject.toml").touch()
    directory = tmp_path / "resources" / "weather_station"
    directory.mkdir(parents=True)
    weather_values = {
        "temperature": [
            (1, "-1.0", "Y"),
            (2, "-8.123456789012345", "Y"),
            (4, "", "Y"),
            (5, "0.0", "Y"),
            (6, "1.25", "Y"),
            (7, "2.0", "Y"),
        ],
        "precipitation": [
            (2, "0.0", "G"),
            (3, "1.125", "Y"),
            (4, "2.0", "G"),
            (6, "3.5", "G"),
        ],
    }
    for variable, (filename, column, _) in SOURCES.items():
        lines = ["Stationsnamn;Stationsnummer", f"Test {variable};1234", ""]
        if variable == "discharge":
            lines.append(f"Datum (svensk sommartid);{column};Kvalitet;;")
            for day in range(1, 9):
                quality = "O" if day == 6 else "G"
                value = "1.234567890123456" if day == 2 else str(day)
                lines.append(f"2000-01-{day:02};{value};{quality};;")
        else:
            lines.append(
                "Från Datum Tid (UTC);Till Datum Tid (UTC);Representativt dygn;"
                f"{column};Kvalitet;;Tidsutsnitt:"
            )
            hour = "06" if variable == "precipitation" else "00"
            for day, value, quality in weather_values[variable]:
                lines.append(
                    f"2000-01-{day:02} {hour}:00:01;"
                    f"2000-01-{day + 1:02} {hour}:00:00;"
                    f"2000-01-{day:02};{value};{quality};;Source note"
                )
        (directory / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_working_copies_share_calendar_and_preserve_observations(source_root):
    raw_hashes = source_checksums(source_root)
    expected = load_daily_data(source_root)
    outputs = prepare_data(source_root)
    assert set(outputs) == OUTPUT_NAMES
    assert all(path.is_file() for path in outputs.values())
    daily = load_processed_data(source_root)
    calendar = pd.date_range("2000-01-02", "2000-01-06", name="date")
    pd.testing.assert_index_equal(daily.index, calendar)
    pd.testing.assert_frame_equal(daily[expected.columns], expected, check_dtype=False)

    for variable, (_, _, value) in SOURCES.items():
        frame = pd.read_parquet(outputs[f"{variable}.parquet"])
        pd.testing.assert_index_equal(frame.index, calendar, check_freq=False)
        pd.testing.assert_series_equal(frame[value], daily[value], check_freq=False)
        assert str(frame[value].dtype) == "float64"
        assert pd.api.types.is_string_dtype(frame[f"{variable}_quality"].dtype)
        for candidate in (frame, daily):
            indicator = candidate[f"{variable}_missing"]
            assert indicator.dtype == bool
            pd.testing.assert_series_equal(indicator, candidate[value].isna(), check_names=False)
        if variable != "discharge":
            for boundary in ("start", "end"):
                name = f"{variable}_{boundary}_utc"
                pd.testing.assert_series_equal(frame[name], expected[name], check_freq=False)
                assert str(frame[name].dt.tz) == "UTC"

    assert daily.loc["2000-01-02", "precipitation_mm"] == 0.0
    assert daily.loc["2000-01-02", "temperature_c"] == float("-8.123456789012345")
    assert daily.loc["2000-01-02", "discharge_m3_s"] == float("1.234567890123456")
    assert daily.loc["2000-01-04", "temperature_quality"] == "Y"
    assert pd.isna(daily.loc["2000-01-04", "temperature_c"])
    assert pd.isna(daily.loc["2000-01-03", "temperature_quality"])
    assert daily.loc["2000-01-06", "discharge_quality"] == "O"
    assert source_checksums(source_root) == raw_hashes

    csv = pd.read_csv(outputs["daily.csv"], index_col="date", float_precision="round_trip")
    csv.index = pd.to_datetime(csv.index)
    pd.testing.assert_index_equal(csv.index, daily.index, check_names=True, check_freq=False)
    for _, _, value in SOURCES.values():
        pd.testing.assert_series_equal(
            csv[value], daily[value], check_dtype=False, check_freq=False
        )
    for variable in ("temperature", "precipitation"):
        for boundary in ("start", "end"):
            column = f"{variable}_{boundary}_utc"
            pd.testing.assert_series_equal(
                pd.to_datetime(csv[column], utc=True), daily[column], check_freq=False
            )


def test_metadata_report_and_rebuild_are_traceable(source_root):
    raw_hashes = source_checksums(source_root)
    outputs = prepare_data(source_root)
    first = load_processed_data(source_root)
    metadata = json.loads(outputs["metadata.json"].read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert pd.Timestamp(metadata["created_at_utc"]).tzinfo is not None
    assert metadata["common_range"] == {"start": "2000-01-02", "end": "2000-01-06", "rows": 5}
    assert metadata["missing_days"] == {"temperature": 2, "precipitation": 1, "discharge": 0}
    assert metadata["complete_days"] == 2
    assert set(metadata["outputs"]) == OUTPUT_NAMES - {"metadata.json"}
    for name, expected_hash in metadata["outputs"].items():
        assert checksum(outputs[name]) == expected_hash
    for variable, (filename, _, _) in SOURCES.items():
        info = metadata["sources"][variable]
        assert info["source_file"] == filename
        assert info["sha256"] == raw_hashes[filename]
        assert info["units"]
        assert info["source_start"] <= "2000-01-02"
        assert info["source_end"] >= "2000-01-06"
        assert info["metadata_rows"][1][0] == f"Test {variable}"

    report = outputs["quality_report.md"].read_text(encoding="utf-8")
    for term in ("2000-01-03", "2000-01-04", "2000-01-05", "2000-01", "2000"):
        assert term in report
    assert "temperature" in report.lower()
    assert "precipitation" in report.lower()
    assert "UTC" in report
    assert all(flag in report for flag in ("G", "Y", "O"))

    prepare_data(source_root)
    pd.testing.assert_frame_equal(load_processed_data(source_root), first)
    assert source_checksums(source_root) == raw_hashes


@pytest.mark.parametrize(
    "change",
    [
        "missing_metadata",
        "invalid_json",
        "invalid_shape",
        "schema_version",
        "missing_output",
        "missing_source",
        "source",
        "output",
    ],
)
def test_loader_explains_how_to_rebuild_missing_or_stale_data(source_root, change):
    outputs = prepare_data(source_root)
    metadata_path = outputs["metadata.json"]
    if change == "missing_metadata":
        metadata_path.unlink()
    elif change == "invalid_json":
        metadata_path.write_text("not json", encoding="utf-8")
    elif change == "invalid_shape":
        metadata_path.write_text("[]", encoding="utf-8")
    elif change == "schema_version":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["schema_version"] = -1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif change == "missing_output":
        outputs["temperature.parquet"].unlink()
    elif change == "missing_source":
        (source_root / "resources" / "weather_station" / SOURCES["temperature"][0]).unlink()
    elif change == "source":
        path = source_root / "resources" / "weather_station" / SOURCES["temperature"][0]
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif change == "output":
        outputs["daily.csv"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Rebuild.*python -m kalix.prepare"):
        load_processed_data(source_root)


@pytest.mark.parametrize("invalid", ["date", "number", "nonfinite", "duplicate", "no_overlap"])
def test_invalid_sources_do_not_replace_existing_outputs(source_root, invalid):
    prepare_data(source_root)
    before = output_checksums(source_root)
    path = source_root / "resources" / "weather_station" / SOURCES["temperature"][0]
    content = path.read_text(encoding="utf-8")
    if invalid == "date":
        content = content.replace(";2000-01-02;", ";2000-02-30;")
    elif invalid == "number":
        content = content.replace("-8.123456789012345", "not-a-number")
    elif invalid == "nonfinite":
        content = content.replace("-8.123456789012345", "inf")
    elif invalid == "no_overlap":
        content = content.replace("2000-01-", "2000-02-")
    else:
        content += next(line for line in content.splitlines() if ";2000-01-02;" in line) + "\n"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_data(source_root)
    assert output_checksums(source_root) == before


def test_failed_export_preserves_previous_working_data(source_root, monkeypatch):
    prepare_data(source_root)
    before = output_checksums(source_root)
    to_parquet = pd.DataFrame.to_parquet
    calls = 0

    def fail_second_export(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated export failure")
        return to_parquet(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_second_export)
    with pytest.raises(OSError, match="simulated export failure"):
        prepare_data(source_root)
    assert output_checksums(source_root) == before
    assert len(load_processed_data(source_root)) == 5


def test_cli_accepts_explicit_project_root(source_root):
    result = subprocess.run(
        [sys.executable, "-m", "kalix.prepare", "--root", str(source_root)],
        cwd=source_root.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(load_processed_data(source_root)) == 5


def test_current_exports_match_accepted_coverage_and_missing_counts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "pyproject.toml").touch()
    source_directory = tmp_path / "resources" / "weather_station"
    source_directory.mkdir(parents=True)
    for filename, _, _ in SOURCES.values():
        shutil.copyfile(
            root / "resources" / "weather_station" / filename, source_directory / filename
        )
    originals = source_checksums(tmp_path)
    outputs = prepare_data(tmp_path)
    daily = load_processed_data(tmp_path)
    calendar = pd.date_range("1995-10-01", "2026-06-30", name="date")
    pd.testing.assert_index_equal(daily.index, calendar)
    assert len(daily) == 11231
    measurements = [value for _, _, value in SOURCES.values()]
    assert daily[measurements].isna().sum().tolist() == [101, 30, 0]
    assert daily[measurements].notna().all(axis=1).sum() == 11100
    for variable in SOURCES:
        frame = pd.read_parquet(outputs[f"{variable}.parquet"])
        pd.testing.assert_index_equal(frame.index, calendar, check_freq=False)
    assert source_checksums(tmp_path) == originals
