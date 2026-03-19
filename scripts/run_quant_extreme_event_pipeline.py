#!/usr/bin/env python3
"""Run the extreme-event and anomaly-day pipeline for a quant output directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect quant outputs to extreme-event extraction and anomaly-day climate tables."
    )
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="Observation table used by the quant run (timestamp,variable,value or compatible parquet/csv).",
    )
    parser.add_argument(
        "--quant-output-dir",
        type=Path,
        required=True,
        help="Quant run output directory that contains reports/top_anomalies_global_context_input.csv.",
    )
    parser.add_argument(
        "--daily-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/spreadsheet/es_ea_newdata_daily.csv"),
        help="Daily climate CSV to join on anomaly days.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Extreme-event output directory. Default: <quant-output-dir>/extreme_events",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to call downstream scripts.",
    )
    parser.add_argument("--low-quantile", type=float, default=0.01)
    parser.add_argument("--high-quantile", type=float, default=0.99)
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--jump-quantile", type=float, default=0.99)
    parser.add_argument("--min-points", type=int, default=25)
    parser.add_argument("--context-max-day-distance", type=int, default=45)
    parser.add_argument("--context-day-window", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perm-iter", type=int, default=3000)
    parser.add_argument("--boot-iter", type=int, default=3000)
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> None:
    print(f"[EXTREME_PIPELINE] {name}")
    print(f"[EXTREME_PIPELINE] CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    scripts_dir = Path(__file__).resolve().parent
    quant_output_dir = args.quant_output_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (quant_output_dir / "extreme_events")
    output_dir.mkdir(parents=True, exist_ok=True)

    context_csv = quant_output_dir / "reports" / "top_anomalies_global_context_input.csv"
    if not context_csv.exists():
        raise SystemExit(f"Quant context CSV not found: {context_csv}")
    if not args.observations.exists():
        raise SystemExit(f"Observations file not found: {args.observations}")
    if not args.daily_csv.exists():
        raise SystemExit(f"Daily climate CSV not found: {args.daily_csv}")

    extract_events_py = scripts_dir / "extract_extreme_events.py"
    enrich_events_py = scripts_dir / "enrich_extreme_events.py"
    scan_causes_py = scripts_dir / "scan_internet_causes.py"
    validate_py = scripts_dir / "scientific_validate_internet_causes.py"
    anomaly_day_py = scripts_dir / "extract_anomaly_day_climate_data.py"

    run_step(
        "Extract extreme events",
        [
            args.python_bin,
            str(extract_events_py),
            "--observations",
            str(args.observations.resolve()),
            "--output-dir",
            str(output_dir),
            "--low-quantile",
            str(args.low_quantile),
            "--high-quantile",
            str(args.high_quantile),
            "--z-threshold",
            str(args.z_threshold),
            "--jump-quantile",
            str(args.jump_quantile),
            "--min-points",
            str(args.min_points),
        ],
    )

    run_step(
        "Enrich extreme events with quant context",
        [
            args.python_bin,
            str(enrich_events_py),
            "--events-csv",
            str((output_dir / "tum_asiri_olaylar.csv").resolve()),
            "--context-csv",
            str(context_csv),
            "--output-dir",
            str(output_dir),
            "--max-day-distance",
            str(args.context_max_day_distance),
        ],
    )

    run_step(
        "Scan internet-backed causes",
        [
            args.python_bin,
            str(scan_causes_py),
            "--events-csv",
            str((output_dir / "tum_asiri_olaylar_zengin.csv").resolve()),
            "--quant-context-csv",
            str(context_csv),
            "--output-dir",
            str(output_dir),
            "--context-day-window",
            str(args.context_day_window),
        ],
    )

    run_step(
        "Apply scientific validation",
        [
            args.python_bin,
            str(validate_py),
            "--input-csv",
            str((output_dir / "tum_asiri_olaylar_internet_nedenleri.csv").resolve()),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.seed),
            "--perm-iter",
            str(args.perm_iter),
            "--boot-iter",
            str(args.boot_iter),
        ],
    )

    anomaly_day_output_dir = output_dir / "anomaly_day_data"
    run_step(
        "Extract anomaly-day climate data",
        [
            args.python_bin,
            str(anomaly_day_py),
            "--events-csv",
            str((output_dir / "tum_asiri_olaylar_bilimsel_filtreli.csv").resolve()),
            "--points-csv",
            str((output_dir / "tum_asiri_olay_noktalari.csv").resolve()),
            "--daily-csv",
            str(args.daily_csv.resolve()),
            "--output-dir",
            str(anomaly_day_output_dir),
        ],
    )

    summary = {
        "quant_output_dir": str(quant_output_dir),
        "observations": str(args.observations.resolve()),
        "quant_context_csv": str(context_csv),
        "daily_csv": str(args.daily_csv.resolve()),
        "extreme_events_output_dir": str(output_dir),
        "anomaly_day_output_dir": str(anomaly_day_output_dir),
        "outputs": {
            "events_csv": str((output_dir / "tum_asiri_olaylar.csv").resolve()),
            "points_csv": str((output_dir / "tum_asiri_olay_noktalari.csv").resolve()),
            "enriched_csv": str((output_dir / "tum_asiri_olaylar_zengin.csv").resolve()),
            "internet_csv": str((output_dir / "tum_asiri_olaylar_internet_nedenleri.csv").resolve()),
            "filtered_csv": str((output_dir / "tum_asiri_olaylar_bilimsel_filtreli.csv").resolve()),
            "anomaly_unique_days_csv": str((anomaly_day_output_dir / "anomaly_unique_days_with_daily_climate.csv").resolve()),
            "anomaly_points_csv": str((anomaly_day_output_dir / "anomaly_points_with_daily_climate.csv").resolve()),
            "anomaly_center_days_csv": str((anomaly_day_output_dir / "anomaly_events_center_day_with_daily_climate.csv").resolve()),
            "anomaly_missing_dates_csv": str((anomaly_day_output_dir / "anomaly_days_missing_daily_climate.csv").resolve()),
        },
        "row_counts": {
            "events": csv_row_count(output_dir / "tum_asiri_olaylar.csv"),
            "points": csv_row_count(output_dir / "tum_asiri_olay_noktalari.csv"),
            "filtered_events": csv_row_count(output_dir / "tum_asiri_olaylar_bilimsel_filtreli.csv"),
            "anomaly_unique_days": csv_row_count(anomaly_day_output_dir / "anomaly_unique_days_with_daily_climate.csv"),
        },
    }

    summary_json = output_dir / "quant_extreme_event_pipeline_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[EXTREME_PIPELINE] Summary: {summary_json}")
    print(json.dumps(summary["row_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
