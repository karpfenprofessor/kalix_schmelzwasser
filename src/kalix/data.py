"""Read SMHI daily exports without changing values or filtering quality flags."""

import csv
from pathlib import Path
from typing import Literal

import pandas as pd

Variable = Literal["temperature", "precipitation", "discharge"]

SOURCES = {
    "temperature": ("smhi-opendata_nikkaluokta_temperatur.csv", "Lufttemperatur", "temperature_c"),
    "precipitation": (
        "smhi-opendata_nikkaluokta_niederschlag.csv",
        "Nederbördsmängd",
        "precipitation_mm",
    ),
    "discharge": ("smhi-opendata_kalix_durchfluss.csv", "Vattenföring (Dygn)", "discharge_m3_s"),
}


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
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    if frame[value_column].isin([float("inf"), float("-inf")]).any():
        raise ValueError(f"Non-finite measurements found in {path}.")
    for name in selected:
        if name.endswith("_utc"):
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate observation dates found in {path}.")
    frame = frame.set_index("date").sort_index()
    frame.attrs = {
        "source_file": path.name,
        "source_date_column": date_column,
        "metadata_rows": rows[:header_number],
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
