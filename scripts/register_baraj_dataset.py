#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_CSV = Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/registry/datasets/local_data_inventory.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a dataset row to the dam research local-data registry.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--granularity", required=True)
    parser.add_argument("--status", default="active")
    parser.add_argument("--note", default="")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = {
        "dataset_id": args.dataset_id,
        "kind": args.kind,
        "path": args.path,
        "coverage": args.coverage,
        "granularity": args.granularity,
        "status": args.status,
        "note": args.note,
    }

    with args.csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

    print(args.csv_path)


if __name__ == "__main__":
    main()
