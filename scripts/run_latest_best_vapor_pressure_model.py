#!/usr/bin/env python3
"""Run the full latest-best vapor-pressure model stack.

Pipeline:
- calibrated quant
- calibrated strong
- calibrated prophet
- best_meta ensemble (fed by a dedicated base-model suite)
- v5 robust arbitration
- v6 stable consensus
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_quant_vapor_pressure_new_data import SERIES_MAP, build_long_observations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the latest best model stack for vapor pressure.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/new_data_vapor_pressure/buhar_basinci_new_data.csv"),
        help="Daily vapor-pressure CSV built from new data.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/new_data_vapor_pressure/latest_best_model_v6"),
        help="Run directory for base models and v5/v6 outputs.",
    )
    p.add_argument("--target-year", type=int, default=2035, help="Forecast horizon year.")
    p.add_argument(
        "--forecast-start-year",
        type=int,
        default=2022,
        help="First forecast year for full-stack members such as walkforward/best_meta.",
    )
    p.add_argument("--skip-run", action="store_true", help="Only prepare observations, do not run models.")
    return p.parse_args()


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    calibrated_dir = out_root / "calibrated"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    observations = build_long_observations(args.input_csv.resolve())
    obs_csv = calibrated_dir / "observations_calibrated_recent_regime.csv"
    obs_parquet = calibrated_dir / "observations_calibrated_recent_regime.parquet"
    observations.to_csv(obs_csv, index=False)
    observations.to_parquet(obs_parquet, index=False)

    summary_json = out_root / "latest_best_vapor_pressure_summary.json"

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    variables = ",".join(SERIES_MAP.keys())

    if not args.skip_run:
        base_jobs = [
            (
                "quant",
                [
                    sys.executable,
                    str((Path(__file__).resolve().parent / "yeni_model_newdata" / "quant_regime_projection_yeni_model.py").resolve()),
                    "--observations",
                    str(obs_csv),
                    "--output-dir",
                    str(out_root / "calibrated_quant"),
                    "--variables",
                    variables,
                    "--input-kind",
                    "long",
                    "--target-year",
                    str(args.target_year),
                    "--analysis-mode",
                    "full",
                    "--vol-model",
                    "egarch",
                    "--anomaly-top",
                    "15",
                ],
            ),
            (
                "strong",
                [
                    sys.executable,
                    str((Path(__file__).resolve().parent / "train_strong_consistent_model.py").resolve()),
                    "--observations",
                    str(obs_csv),
                    "--output-dir",
                    str(out_root / "calibrated_strong"),
                    "--variables",
                    variables,
                    "--input-kind",
                    "auto",
                    "--target-year",
                    str(args.target_year),
                ],
            ),
            (
                "prophet",
                [
                    sys.executable,
                    str((Path(__file__).resolve().parent / "prophet_climate_forecast.py").resolve()),
                    "--observations",
                    str(obs_csv),
                    "--output-dir",
                    str(out_root / "calibrated_prophet"),
                    "--variables",
                    variables,
                    "--input-kind",
                    "auto",
                    "--target-year",
                    str(args.target_year),
                ],
            ),
        ]

        for _, cmd in base_jobs:
            run_command(cmd, env=env)

        best_meta_base_dir = out_root / "best_meta" / "base_models"
        best_meta_jobs = [
            [
                sys.executable,
                str((Path(__file__).resolve().parent / "quant_regime_projection.py").resolve()),
                "--observations",
                str(obs_csv),
                "--output-dir",
                str(best_meta_base_dir / "quant"),
                "--variables",
                variables,
                "--target-year",
                str(args.target_year),
                "--climate-scenario",
                "ssp245",
                "--climate-baseline-year",
                "nan",
                "--climate-temp-rate",
                "nan",
                "--humidity-per-temp-c",
                "-2.0",
                "--climate-adjustment-method",
                "pathway",
            ],
            [
                sys.executable,
                str((Path(__file__).resolve().parent / "train_strong_consistent_model.py").resolve()),
                "--observations",
                str(obs_csv),
                "--output-dir",
                str(best_meta_base_dir / "strong"),
                "--variables",
                variables,
                "--target-year",
                str(args.target_year),
            ],
            [
                sys.executable,
                str((Path(__file__).resolve().parent / "prophet_ultra_500.py").resolve()),
                "--observations",
                str(obs_csv),
                "--output-dir",
                str(best_meta_base_dir / "prophet_ultra"),
                "--variables",
                variables,
                "--target-year",
                str(args.target_year),
                "--input-kind",
                "auto",
            ],
        ]

        for cmd in best_meta_jobs:
            run_command(cmd, env=env)

        run_command(
            [
                sys.executable,
                str((Path(__file__).resolve().parent / "best_climate_meta_ensemble.py").resolve()),
                "--observations",
                str(obs_csv),
                "--output-dir",
                str(out_root / "best_meta"),
                "--variables",
                variables,
                "--target-year",
                str(args.target_year),
                "--walkforward-start-year",
                str(args.forecast_start_year),
                "--forecast-start-year",
                str(args.forecast_start_year),
                "--input-kind",
                "long",
                "--run-base-models",
                "false",
                "--reuse-existing",
                "true",
                "--base-dir-name",
                "base_models",
            ],
            env=env,
        )

        run_command(
            [
                sys.executable,
                str((Path(__file__).resolve().parent / "build_v5_robust_arbitrated_forecast.py").resolve()),
                "--run-dir",
                str(out_root),
                "--obs-parquet",
                str(obs_parquet),
            ],
            env=env,
        )

        run_command(
            [
                sys.executable,
                str((Path(__file__).resolve().parent / "build_v6_stable_consensus_forecast.py").resolve()),
                "--run-dir",
                str(out_root),
                "--obs-parquet",
                str(obs_parquet),
            ],
            env=env,
        )

    v5_csv = out_root / "quant" / "reports" / "v5_final_arbitrated_ozet.csv"
    v6_csv = out_root / "quant" / "reports" / "v6_stable_consensus_ozet.csv"
    v6_dashboard = out_root / "quant" / "reports" / "v6_stable_consensus_dashboard.png"
    v6_summary = out_root / "quant" / "reports" / "v6_stable_consensus_summary.json"
    best_meta_index = out_root / "best_meta" / f"best_meta_index_to_{args.target_year}.csv"

    summary: dict[str, object] = {
        "source_csv": str(args.input_csv.resolve()),
        "output_dir": str(out_root),
        "observations_csv": str(obs_csv),
        "observations_parquet": str(obs_parquet),
        "model_stack": [
            "calibrated_quant",
            "calibrated_strong",
            "calibrated_prophet",
            "best_meta",
            "v5_robust_arbitrated",
            "v6_stable_consensus",
        ],
        "variables": list(SERIES_MAP.keys()),
        "series_map": SERIES_MAP,
        "target_year": int(args.target_year),
        "forecast_start_year": int(args.forecast_start_year),
        "date_min": str(pd.to_datetime(observations["timestamp"]).min().date()),
        "date_max": str(pd.to_datetime(observations["timestamp"]).max().date()),
        "n_input_rows": int(len(observations)),
        "outputs": {
            "best_meta_index_csv": str(best_meta_index),
            "v5_winners_csv": str(v5_csv),
            "v6_summary_csv": str(v6_csv),
            "v6_dashboard_png": str(v6_dashboard),
            "v6_summary_json": str(v6_summary),
        },
    }

    if best_meta_index.exists():
        best_meta = pd.read_csv(best_meta_index)
        summary["best_meta_rows"] = int(len(best_meta))
        if not best_meta.empty:
            summary["best_meta_preview"] = best_meta[
                ["variable", "model_strategy", "forecast_csv", "best_combo_models", "best_combo_model_count"]
            ].to_dict(orient="records")

    if v6_csv.exists():
        v6 = pd.read_csv(v6_csv)
        summary["v6_rows"] = int(len(v6))
        if not v6.empty:
            summary["v6_preview"] = v6[
                ["variable", "forecast_csv", "consensus_score", "confidence_score", "confidence_grade"]
            ].to_dict(orient="records")

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared observations: {obs_csv}")
    print(f"Run dir: {out_root}")
    print(f"Summary JSON: {summary_json}")
    if v6_csv.exists():
        print(f"V6 summary CSV: {v6_csv}")
        print(f"V6 dashboard: {v6_dashboard}")


if __name__ == "__main__":
    main()
