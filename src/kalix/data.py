"""Read SMHI daily exports without changing values or filtering quality flags."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd

Variable = Literal["nikkaluokta_temperature", "precipitation", "discharge", "tarfala_temperature"]

SOURCES = {
    "nikkaluokta_temperature": (
        "smhi-opendata_nikkaluokta_temperatur.csv",
        "Lufttemperatur",
        "nikkaluokta_temperature_c",
    ),
    "precipitation": (
        "smhi-opendata_nikkaluokta_niederschlag.csv",
        "Nederbördsmängd",
        "precipitation_mm",
    ),
    "discharge": ("smhi-opendata_kalix_durchfluss.csv", "Vattenföring (Dygn)", "discharge_m3_s"),
    "tarfala_temperature": (
        "smhi-opendata_tarfala_temperatur.csv",
        "Lufttemperatur",
        "tarfala_temperature_c",
    ),
}

PROCESSED_SCHEMA_VERSION = 3
PREPARED_FILENAMES = (
    *(f"{variable}.parquet" for variable in SOURCES),
    "daily.parquet",
    "daily.csv",
    "quality_report.md",
)


def file_sha256(path: str | Path) -> str:
    """Hash a source or generated file without changing it."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def project_root(start: str | Path | None = None) -> Path:
    """Find this source checkout from the working directory or a supplied location."""
    location = Path.cwd() if start is None else Path(start).resolve()
    if location.is_file():
        location = location.parent
    for candidate in (location, *location.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "resources" / "weather_station"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Run inside the project checkout or pass its root directory.")


def read_station_csv(path: str | Path, variable: Variable) -> pd.DataFrame:
    """Read one export, preserving daily labels, quality flags, and UTC intervals.

    Empty measurements remain missing. Invalid dates/numbers and duplicate daily
    labels raise errors. Source preamble lines are retained as dataframe metadata.
    """
    path = Path(path)
    _, source_column, value_column = SOURCES[variable]
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source, delimiter=";"))

    for header_number, header in enumerate(rows):
        is_daily = "Representativt dygn" in header or (header and header[0].startswith("Datum ("))
        if is_daily and source_column in header and "Kvalitet" in header:
            break
    else:
        raise ValueError(f"No daily {variable} observation header found in {path}.")

    date_column = "Representativt dygn" if variable != "discharge" else header[0]
    selected = {
        "date": header.index(date_column),
        value_column: header.index(source_column),
        f"{variable}_quality": header.index("Kvalitet"),
    }
    if variable != "discharge":
        selected[f"{variable}_start_utc"] = header.index("Från Datum Tid (UTC)")
        selected[f"{variable}_end_utc"] = header.index("Till Datum Tid (UTC)")

    records = []
    for line_number, row in enumerate(rows[header_number + 1 :], start=header_number + 2):
        if not any(field.strip() for field in row):
            continue
        if len(row) <= max(selected.values()):
            raise ValueError(f"Incomplete observation at {path}:{line_number}.")
        records.append({name: row[index].strip() for name, index in selected.items()})
    if not records:
        raise ValueError(f"No daily observations found in {path}.")

    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if frame["date"].isna().any():
        raise ValueError(f"Missing observation dates found in {path}.")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    if frame[value_column].isin([float("inf"), float("-inf")]).any():
        raise ValueError(f"Non-finite measurements found in {path}.")
    for name in selected:
        if name.endswith("_utc"):
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
            if frame[name].isna().any():
                raise ValueError(f"Missing observation timestamps found in {path}.")
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate observation dates found in {path}.")
    frame = frame.set_index("date").sort_index()
    frame.attrs = {
        "source_file": path.name,
        "source_date_column": date_column,
        "metadata_rows": rows[:header_number],
        "source_notes": [
            field.strip()
            for row in rows[header_number + 1 :]
            for field in row[header.index("Kvalitet") + 1 :]
            if field.strip()
        ],
    }
    return frame


def load_daily_data(root: str | Path | None = None) -> pd.DataFrame:
    """Join source date labels across their common bounds, preserving missing days."""
    source_directory = project_root(root) / "resources" / "weather_station"
    frames = [
        read_station_csv(source_directory / filename, variable)
        for variable, (filename, _, _) in SOURCES.items()
    ]
    start = max(frame.index.min() for frame in frames)
    end = min(frame.index.max() for frame in frames)
    if start > end:
        raise ValueError("The source observation periods do not overlap.")
    calendar = pd.date_range(start, end, freq="D", name="date")
    combined = pd.concat(frames, axis=1, sort=True).reindex(calendar)
    combined.attrs = {
        "sources": [frame.attrs for frame in frames],
        "alignment": "Source daily labels; observation windows have not been synchronized.",
    }
    return combined


def load_processed_data(root: str | Path | None = None) -> pd.DataFrame:
    """Load prepared daily data after checking source and output fingerprints.

    Preparation is explicit: missing, stale, or damaged outputs never trigger an
    implicit rebuild. Source import functions remain available independently.
    """
    root = project_root(root)
    directory = root / "data" / "processed"
    rebuild = f'Rebuild with: uv run python -m kalix.prepare --root "{root}"'
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != PROCESSED_SCHEMA_VERSION:
            raise ValueError("Prepared-data schema version has changed.")
        for variable, (filename, _, _) in SOURCES.items():
            source = root / "resources" / "weather_station" / filename
            if file_sha256(source) != metadata["sources"][variable]["sha256"]:
                raise ValueError(f"Source has changed: {filename}.")
        for filename in PREPARED_FILENAMES:
            if file_sha256(directory / filename) != metadata["outputs"][filename]:
                raise ValueError(f"Prepared output has changed: {filename}.")
        daily = pd.read_parquet(directory / "daily.parquet", engine="pyarrow")
        # Parquet retains dates but not pandas' daily frequency annotation.
        daily.index = pd.DatetimeIndex(daily.index, freq="D")
        return daily
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Prepared data are missing, stale, or invalid. {rebuild}") from error
