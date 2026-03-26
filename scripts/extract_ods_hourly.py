#!/usr/bin/env python3
"""Extract hourly temperature rows from ODS into a tidy CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import zipfile
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET


NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}

OFFICE_VALUE = f"{{{NS['office']}}}value"
OFFICE_DATE_VALUE = f"{{{NS['office']}}}date-value"
TABLE_REPEAT = f"{{{NS['table']}}}number-columns-repeated"
TABLE_NAME = f"{{{NS['table']}}}name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract date + 24 hourly temperature columns from ODS."
    )
    parser.add_argument("input_ods", type=Path, help="Path to .ods file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/hourly_temperature.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=["hour_start", "hour_end", "none"],
        default="hour_start",
        help=(
            "hour_start: hour=1 => 00:00, hour=24 => 23:00. "
            "hour_end: hour=1 => 01:00, hour=24 => next day 00:00."
        ),
    )
    return parser.parse_args()


def iter_expanded_cells(row: ET.Element) -> Iterable[ET.Element]:
    for cell in row.findall("table:table-cell", NS):
        repeat = int(cell.attrib.get(TABLE_REPEAT, "1"))
        for _ in range(repeat):
            yield cell


def cell_text(cell: ET.Element) -> str:
    parts = []
    for p in cell.findall("text:p", NS):
        if p.text:
            parts.append(p.text.strip())
    return " ".join(x for x in parts if x)


def parse_date(cell: ET.Element) -> Optional[dt.date]:
    raw = cell.attrib.get(OFFICE_DATE_VALUE)
    if raw:
        return dt.date.fromisoformat(raw[:10])
    text = cell_text(cell)
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_float(cell: ET.Element) -> Optional[float]:
    raw = cell.attrib.get(OFFICE_VALUE)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            return None
    text = cell_text(cell).replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def make_timestamp(date_value: dt.date, hour: int, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "hour_start":
        ts = dt.datetime.combine(date_value, dt.time()) + dt.timedelta(hours=hour - 1)
    else:
        ts = dt.datetime.combine(date_value, dt.time()) + dt.timedelta(hours=hour)
    return ts.isoformat(timespec="minutes")


def extract_rows(input_ods: Path, timestamp_mode: str) -> list[dict[str, object]]:
    with zipfile.ZipFile(input_ods) as zf:
        content_xml = zf.read("content.xml")
    root = ET.parse(io.BytesIO(content_xml)).getroot()

    rows_out: list[dict[str, object]] = []
    for table in root.findall(".//table:table", NS):
        sheet_name = table.attrib.get(TABLE_NAME, "")
        for row in table.findall("table:table-row", NS):
            expanded = list(iter_expanded_cells(row))
            if len(expanded) < 25:
                continue
            date_value = parse_date(expanded[0])
            if not date_value:
                continue
            for hour in range(1, 25):
                temp = parse_float(expanded[hour])
                if temp is None:
                    continue
                rows_out.append(
                    {
                        "sheet": sheet_name,
                        "date": date_value.isoformat(),
                        "hour": hour,
                        "temp_c": temp,
                        "timestamp": make_timestamp(date_value, hour, timestamp_mode),
                    }
                )
    return rows_out


def main() -> None:
    args = parse_args()
    rows = extract_rows(args.input_ods, args.timestamp_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sheet", "date", "hour", "temp_c", "timestamp"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

