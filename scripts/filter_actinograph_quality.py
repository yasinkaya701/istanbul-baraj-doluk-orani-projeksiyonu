#!/usr/bin/env python3
"""Filter daily actinograph report by quality metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter actinograph daily report by quality thresholds")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/universal_datasets/daily_solar_radiation_report.csv"),
        help="Input daily report",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/universal_datasets/daily_solar_radiation_report_filtered.csv"),
        help="Filtered output report",
    )
    parser.add_argument(
        "--out-excluded",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/universal_datasets/daily_solar_radiation_report_excluded.csv"),
        help="Excluded rows",
    )
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument("--min-strength", type=float, default=0.20)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for col in ["valid_coverage", "trace_strength"]:
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")

    keep = (df["valid_coverage"] >= args.min_coverage) & (df["trace_strength"] >= args.min_strength)
    filtered = df[keep].copy()
    excluded = df[~keep].copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out, index=False)
    excluded.to_csv(args.out_excluded, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Kept rows: {len(filtered)}")
    print(f"Excluded rows: {len(excluded)}")
    print(f"Output: {args.out}")
    print(f"Excluded: {args.out_excluded}")


if __name__ == "__main__":
    main()
