#!/usr/bin/env python3
"""Prepare vapor-pressure observations from new-data CSV and run the quant model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


SERIES_MAP = {
    "buhar_max": "maksimum_buhar_basinci_kpa_(es_tmax)",
    "buhar_anlik": "anlik_buhar_basinci_kpa_(ea)",
    "buhar_fark": "aradaki_fark_kpa_(es_tmax-ea)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run quant model for new-data vapor-pressure series.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/new_data_vapor_pressure/buhar_basinci_new_data.csv"),
        help="Daily vapor-pressure CSV built from new data.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/new_data_vapor_pressure/quant_model_buhar"),
        help="Directory for prepared quant input and quant outputs.",
    )
    p.add_argument(
        "--quant-script",
        type=Path,
        default=Path("scripts/yeni_model_newdata/quant_regime_projection_yeni_model.py"),
        help="Quant model script to execute.",
    )
    p.add_argument("--target-year", type=int, default=2035, help="Forecast horizon year.")
    p.add_argument(
        "--analysis-mode",
        type=str,
        default="full",
        choices=["full", "anomalies_only"],
        help="full: forecast + anomalies, anomalies_only: history only.",
    )
    p.add_argument(
        "--news-catalog",
        type=Path,
        default=None,
        help="Optional news catalog CSV passed to the quant model.",
    )
    p.add_argument(
        "--news-window-days",
        type=int,
        default=75,
        help="News matching window passed to the quant model.",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only prepare the quant input table, do not execute the quant model.",
    )
    return p.parse_args()


def build_long_observations(source_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(source_csv, parse_dates=["date"])
    missing = [col for col in SERIES_MAP.values() if col not in raw.columns]
    if missing:
        raise SystemExit(f"Missing required vapor-pressure columns: {missing}")

    blocks: list[pd.DataFrame] = []
    for variable, source_col in SERIES_MAP.items():
        part = raw[["date", source_col]].copy()
        part = part.rename(columns={"date": "timestamp", source_col: "value"})
        part["variable"] = variable
        part["qc_flag"] = "ok"
        part["unit"] = "kPa"
        part["source_column"] = source_col
        blocks.append(part)

    out = pd.concat(blocks, ignore_index=True)
    out = out.dropna(subset=["timestamp", "value"]).sort_values(["variable", "timestamp"]).reset_index(drop=True)
    return out


def build_quant_command(args: argparse.Namespace, prepared_csv: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.quant_script.resolve()),
        "--observations",
        str(prepared_csv.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--input-kind",
        "long",
        "--timestamp-col",
        "timestamp",
        "--value-col",
        "value",
        "--variable-col",
        "variable",
        "--qc-col",
        "qc_flag",
        "--qc-ok-value",
        "ok",
        "--variables",
        ",".join(SERIES_MAP.keys()),
        "--target-year",
        str(args.target_year),
        "--analysis-mode",
        args.analysis_mode,
        "--backtest-splits",
        "3",
        "--holdout-steps",
        "12",
        "--min-train-steps",
        "36",
        "--vol-model",
        "egarch",
        "--egarch-p",
        "1",
        "--egarch-o",
        "1",
        "--egarch-q",
        "1",
        "--egarch-dist",
        "t",
        "--regime-k",
        "2",
        "--regime-maxiter",
        "200",
        "--interval-alpha",
        "0.10",
        "--anomaly-z",
        "2.5",
        "--anomaly-top",
        "15",
    ]
    if args.news_catalog is not None:
        cmd.extend(["--news-catalog", str(args.news_catalog.resolve())])
    cmd.extend(["--news-window-days", str(args.news_window_days)])
    return cmd


def build_run_summary(args: argparse.Namespace, prepared_csv: Path, observations: pd.DataFrame) -> dict[str, object]:
    index_name = f"quant_index_to_{args.target_year}.csv" if args.analysis_mode == "full" else "quant_index_history_only.csv"
    forecast_dir = args.output_dir / "forecasts"
    chart_dir = args.output_dir / "charts"
    report_dir = args.output_dir / "reports"

    summary: dict[str, object] = {
        "source_csv": str(args.input_csv.resolve()),
        "prepared_quant_csv": str(prepared_csv.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "quant_script": str(args.quant_script.resolve()),
        "analysis_mode": args.analysis_mode,
        "target_year": int(args.target_year),
        "n_input_rows": int(len(observations)),
        "date_min": str(pd.to_datetime(observations["timestamp"]).min().date()),
        "date_max": str(pd.to_datetime(observations["timestamp"]).max().date()),
        "variables": list(SERIES_MAP.keys()),
        "series_map": SERIES_MAP,
        "quant_index_csv": str((args.output_dir / index_name).resolve()),
        "forecast_dir": str(forecast_dir.resolve()),
        "chart_dir": str(chart_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
    }

    idx_path = args.output_dir / index_name
    if idx_path.exists():
        idx = pd.read_csv(idx_path)
        summary["quant_index_rows"] = int(len(idx))
        if {"variable", "frequency", "forecast_csv"}.issubset(idx.columns):
            summary["index_preview"] = idx[["variable", "frequency", "forecast_csv"]].to_dict(orient="records")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quant_input_dir = args.output_dir / "quant_input"
    quant_input_dir.mkdir(parents=True, exist_ok=True)

    observations = build_long_observations(args.input_csv)
    prepared_csv = quant_input_dir / "vapor_pressure_quant_observations.csv"
    observations.to_csv(prepared_csv, index=False)

    summary_json = args.output_dir / "vapor_pressure_quant_run_summary.json"

    if not args.skip_run:
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
        )
        cmd = build_quant_command(args, prepared_csv)
        subprocess.run(cmd, check=True, env=env)

    summary = build_run_summary(args, prepared_csv, observations)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared quant input: {prepared_csv}")
    print(f"Output dir: {args.output_dir}")
    print(f"Summary JSON: {summary_json}")
    if not args.skip_run:
        print(f"Quant index: {summary['quant_index_csv']}")
        print(f"Variables: {', '.join(summary['variables'])}")


if __name__ == "__main__":
    main()
