#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_CSV = Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/registry/sources/external_sources.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a source row to the dam research source registry.")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--access-date", required=True)
    parser.add_argument("--relevance", required=True)
    parser.add_argument("--status", default="active")
    parser.add_argument("--note", default="")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = {
        "source_id": args.source_id,
        "category": args.category,
        "title": args.title,
        "organization_or_journal": args.org,
        "year": args.year,
        "url": args.url,
        "access_date": args.access_date,
        "relevance": args.relevance,
        "status": args.status,
        "note": args.note,
    }

    with args.csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

    print(args.csv_path)


if __name__ == "__main__":
    main()
