"""Build common-range working copies without filling or discarding observations."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from kalix.data import (
    PROCESSED_SCHEMA_VERSION,
    SOURCES,
    file_sha256,
    project_root,
    read_station_csv,
)

UNITS = {
    "nikkaluokta_temperature": "degrees Celsius",
    "precipitation": "mm",
    "discharge": "m³/s",
    "tarfala_temperature": "degrees Celsius",
}
ALIGNMENT = "Source daily labels; observation windows have not been synchronized."
POLICY = (
    "All calendar days and source quality flags are retained. Missing measurements remain missing. "
    "No interpolation, zero filling, outlier removal, or quality filtering is applied."
)


def _gap_intervals(series: pd.Series) -> list[dict]:
    missing = series.isna()
    groups = missing.ne(missing.shift()).cumsum()
    return [
        {"start": str(gap.index[0].date()), "end": str(gap.index[-1].date()), "days": len(gap)}
        for _, gap in series[missing].groupby(groups[missing])
    ]


def _markdown_table(headers: list[str], rows: list[list]) -> list[str]:
    def line(values):
        return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"

    return [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows), ""]


def _quality_report(daily: pd.DataFrame, metadata: dict) -> str:
    measurements = [value for _, _, value in SOURCES.values()]
    total = len(daily)
    common = metadata["common_range"]
    lines = [
        "# Prepared-data quality report",
        "",
        "Source: SMHI exports in resources/weather_station/. Original files are unchanged.",
        f"Built (UTC): {metadata['created_at_utc']}",
        "",
        f"Common range: **{common['start']} to {common['end']}**, inclusive; **{total:,} days**.",
        f"Fully observed days: **{metadata['complete_days']:,}** "
        f"(**{metadata['complete_days'] / total:.2%}**).",
        "",
        POLICY,
        "",
        "## Source coverage and value checks",
        "",
    ]
    rows = []
    for variable, (_, _, value) in SOURCES.items():
        source = metadata["sources"][variable]
        series = daily[value]
        minimum = "n/a" if series.dropna().empty else f"{series.min():.15g}"
        maximum = "n/a" if series.dropna().empty else f"{series.max():.15g}"
        rows.append(
            [
                variable,
                source["units"],
                source["source_start"],
                source["source_end"],
                int(series.notna().sum()),
                int(series.isna().sum()),
                minimum,
                maximum,
                int(series.eq(0).sum()),
                int(series.lt(0).sum()),
            ]
        )
    lines += _markdown_table(
        [
            "Variable",
            "Unit",
            "Source start",
            "Source end",
            "Observed",
            "Missing",
            "Min",
            "Max",
            "Zero",
            "Negative",
        ],
        rows,
    )
    lines += [
        "Negative temperatures and zero precipitation are retained. Negative precipitation or "
        "discharge, if present, require review; no measurements are silently removed.",
        "",
        "## All missing-measurement intervals",
        "",
    ]
    gaps = [
        [variable, gap["start"], gap["end"], gap["days"]]
        for variable, (_, _, value) in SOURCES.items()
        for gap in _gap_intervals(daily[value])
    ]
    lines += _markdown_table(["Variable", "Start", "End", "Days"], gaps)
    if not gaps:
        lines += ["No missing measurements in the common range.", ""]
    lines += ["## Quality codes in the common range", ""]
    quality_rows = []
    for variable in SOURCES:
        counts = daily[f"{variable}_quality"].value_counts(dropna=False)
        for code, count in counts.items():
            label = "missing" if pd.isna(code) else (code if code else "blank source flag")
            quality_rows.append([variable, label, int(count)])
    lines += _markdown_table(["Variable", "Source quality code", "Days"], quality_rows)
    lines += [
        "Quality codes describe source observations, independently of missing-measurement flags. "
        "Original definitions are retained below and in metadata.json. A blanket G-only filter "
        "would remove all temperature observations in the current exports.",
        "",
        "## Observation conventions and source notes",
        "",
        ALIGNMENT,
        "",
    ]
    for variable, (_, _, value) in SOURCES.items():
        source = metadata["sources"][variable]
        lines += [f"### {variable}", "", f"Source date column: {source['source_date_column']}.", ""]
        if variable != "discharge":
            observed = daily.loc[daily[value].notna()]
            if not observed.empty:
                first = observed.iloc[0]
                lines += [
                    f"Example daily label: {observed.index[0].date()}; UTC observation interval: "
                    f"{first[f'{variable}_start_utc'].isoformat()} to "
                    f"{first[f'{variable}_end_utc'].isoformat()}.",
                    "",
                ]
        lines.extend(f"- {note}" for note in source["source_notes"])
        lines.append("")
    observed = daily[measurements].notna()
    for title, labels in (
        ("Annual completeness", daily.index.strftime("%Y")),
        ("Monthly completeness", daily.index.strftime("%Y-%m")),
    ):
        lines += [f"## {title}", "", "Denominators include only days in the common range.", ""]
        rows = []
        for label, group in observed.groupby(labels, sort=True):
            counts = group.sum()
            rows.append(
                [label, len(group)]
                + [
                    f"{int(counts[value])}/{len(group)} ({counts[value] / len(group):.2%})"
                    for value in measurements
                ]
                + [int(group.all(axis=1).sum())]
            )
        lines += _markdown_table(
            [
                "Period",
                "Calendar days",
                *SOURCES,
                "All observed",
            ],
            rows,
        )
    lines += [
        "## Use in later calculations and training",
        "",
        "- Keep the calendar intact when creating lags or rolling windows.",
        "- Require complete input and target windows for each training sample unless an explicitly "
        "validated missing-data method is introduced. Do not automatically discard entire years.",
        "- Do not report a partial precipitation sum as a complete monthly or seasonal total.",
        "- Missing winter forcing can invalidate subsequent snow-storage states until a documented "
        "reset or reconstruction. Review gaps at both temperature stations, including "
        "Tarfala's winter 2010/11 and spring 2014 gaps.",
        "- Short bounded temperature gaps of at most three days may be evaluated later in separate "
        "estimated-value columns. Nothing is interpolated in these files.",
        "- Any later imputation must respect forecast-time availability and training boundaries; "
        "future observations must not enter predictors before they would have been available.",
        "",
    ]
    return "\n".join(lines)


def prepare_data(root: str | Path | None = None) -> dict[str, Path]:
    """Build all working copies on the sources' intersecting daily calendar.

    Source validation finishes before existing outputs are touched. Files are
    staged and the checksum manifest is published last, so interrupted builds
    cannot silently be consumed as complete by load_processed_data().
    """
    root = project_root(root)
    source_directory = root / "resources" / "weather_station"
    source_hashes = {
        variable: file_sha256(source_directory / filename)
        for variable, (filename, _, _) in SOURCES.items()
    }
    frames = {
        variable: read_station_csv(source_directory / filename, variable)
        for variable, (filename, _, _) in SOURCES.items()
    }
    start = max(frame.index.min() for frame in frames.values())
    end = min(frame.index.max() for frame in frames.values())
    if start > end:
        raise ValueError("The source observation periods do not overlap.")
    calendar = pd.date_range(start, end, freq="D", name="date")
    sources = {}
    trimmed = {}
    for variable, frame in frames.items():
        sources[variable] = {
            **frame.attrs,
            "sha256": source_hashes[variable],
            "units": UNITS[variable],
            "source_start": str(frame.index.min().date()),
            "source_end": str(frame.index.max().date()),
        }
        table = frame.reindex(calendar).copy()
        value = SOURCES[variable][2]
        table[value] = table[value].astype("float64")
        table[f"{variable}_quality"] = table[f"{variable}_quality"].astype("string")
        table[f"{variable}_missing"] = table[value].isna()
        table.attrs = sources[variable]
        trimmed[variable] = table
    daily = pd.concat(trimmed.values(), axis=1)
    daily.attrs = {"sources": list(sources.values()), "alignment": ALIGNMENT, "policy": POLICY}
    measurements = [value for _, _, value in SOURCES.values()]
    metadata = {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sources": sources,
        "common_range": {"start": str(start.date()), "end": str(end.date()), "rows": len(calendar)},
        "alignment": ALIGNMENT,
        "missing_data_policy": POLICY,
        "missing_days": {
            variable: int(daily[value].isna().sum()) for variable, (_, _, value) in SOURCES.items()
        },
        "complete_days": int(daily[measurements].notna().all(axis=1).sum()),
        "outputs": {},
    }
    report = _quality_report(daily, metadata)
    directory = root / "data" / "processed"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=directory) as temporary:
        staging = Path(temporary)
        for variable, frame in trimmed.items():
            frame.to_parquet(
                staging / f"{variable}.parquet", engine="pyarrow", compression="snappy"
            )
        daily.to_parquet(staging / "daily.parquet", engine="pyarrow", compression="snappy")
        daily.to_csv(
            staging / "daily.csv", index_label="date", encoding="utf-8", lineterminator="\n"
        )
        (staging / "quality_report.md").write_text(report, encoding="utf-8", newline="\n")
        metadata["outputs"] = {path.name: file_sha256(path) for path in sorted(staging.iterdir())}
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for variable, (filename, _, _) in SOURCES.items():
            if file_sha256(source_directory / filename) != source_hashes[variable]:
                raise ValueError(
                    f"Source changed during preparation: {filename}. Run preparation again."
                )
        for filename in (*metadata["outputs"], "metadata.json"):
            (staging / filename).replace(directory / filename)
    # Retire the former Nikkaluokta filename only after publishing the new build.
    (directory / "temperature.parquet").unlink(missing_ok=True)
    return {filename: directory / filename for filename in (*metadata["outputs"], "metadata.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="Project root (otherwise found from the current directory)."
    )
    arguments = parser.parse_args()
    try:
        paths = prepare_data(arguments.root)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    metadata = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
    common = metadata["common_range"]
    print(f"Prepared {common['rows']:,} days: {common['start']} to {common['end']} (inclusive).")
    print(
        "Missing measurements: "
        + ", ".join(f"{key}={value}" for key, value in metadata["missing_days"].items())
    )
    print(f"Working copies and quality report: {paths['metadata.json'].parent}")


if __name__ == "__main__":
    main()
