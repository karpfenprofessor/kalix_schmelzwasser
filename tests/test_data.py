"""Checks for source-format quirks and missing-day preservation."""

import pandas as pd
import pytest

from kalix.data import SOURCES, load_daily_data, read_station_csv


def test_weather_reader_skips_station_metadata_and_keeps_quality(tmp_path):
    source = tmp_path / "temperature.csv"
    source.write_text(
        "Stationsnamn;Stationsnummer\nNikkaluokta A;179960\n\n"
        "1995-11-01 00:00:00;2026-10-01 06:20:12;466.765;67.8527;19.0212\n\n"
        "Från Datum Tid (UTC);Till Datum Tid (UTC);Representativt dygn;"
        "Lufttemperatur;Kvalitet;;Tidsutsnitt:\n"
        "2000-01-01 00:00:01;2000-01-02 00:00:00;2000-01-01;-8.5;Y;;note\n"
        "2000-01-02 00:00:01;2000-01-03 00:00:00;2000-01-02;;Y\n",
        encoding="utf-8",
    )
    frame = read_station_csv(source, "temperature")
    assert len(frame) == 2
    assert frame.iloc[0]["temperature_c"] == -8.5
    assert frame.iloc[0]["temperature_quality"] == "Y"
    assert pd.isna(frame.iloc[1]["temperature_c"])
    assert str(frame["temperature_end_utc"].dt.tz) == "UTC"
    assert frame.attrs["metadata_rows"][1][0] == "Nikkaluokta A"


def test_duplicate_dates_are_rejected(tmp_path):
    source = tmp_path / "flow.csv"
    source.write_text(
        "Datum (svensk sommartid);Vattenföring (Dygn);Kvalitet\n"
        "2000-01-01;10.0;G\n2000-01-01;11.0;G\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate observation dates"):
        read_station_csv(source, "discharge")


def test_join_preserves_missing_dates_and_real_zero_precipitation(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    directory = tmp_path / "resources" / "weather_station"
    directory.mkdir(parents=True)
    for variable, (filename, source_column, _) in SOURCES.items():
        if variable == "discharge":
            content = (
                f"Datum (svensk sommartid);{source_column};Kvalitet\n"
                "2000-01-01;10;G\n2000-01-02;12;G\n2000-01-03;11;G\n"
            )
        else:
            content = (
                "Från Datum Tid (UTC);Till Datum Tid (UTC);Representativt dygn;"
                f"{source_column};Kvalitet\n"
                "2000-01-01 00:00:01;2000-01-02 00:00:00;2000-01-01;0;Y\n"
                "2000-01-03 00:00:01;2000-01-04 00:00:00;2000-01-03;2;Y\n"
            )
        (directory / filename).write_text(content, encoding="utf-8")
    daily = load_daily_data(tmp_path)
    assert len(daily) == 3
    assert daily.loc["2000-01-01", "precipitation_mm"] == 0
    assert pd.isna(daily.loc["2000-01-02", "precipitation_mm"])
    assert daily.loc["2000-01-02", "discharge_m3_s"] == 12
