#!/usr/bin/env python3
"""Build early-warning alert JSON from decision forecast outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Istanbul dam alerts JSON")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/alerts_2026_03_2027_02.json"),
    )
    p.add_argument("--threshold-high-risk-months", type=int, default=4)
    p.add_argument("--threshold-medium-risk-months", type=int, default=2)
    p.add_argument("--threshold-high-prob40", type=float, default=60.0)
    p.add_argument("--threshold-medium-prob40", type=float, default=35.0)
    return p.parse_args()


def risk_level(row: pd.Series, args: argparse.Namespace) -> str:
    if (
        int(row["months_lt40"]) >= args.threshold_high_risk_months
        or float(row["mean_prob_below_40_pct"]) >= args.threshold_high_prob40
    ):
        return "high"
    if (
        int(row["months_lt40"]) >= args.threshold_medium_risk_months
        or float(row["mean_prob_below_40_pct"]) >= args.threshold_medium_prob40
    ):
        return "medium"
    return "low"


def main() -> None:
    args = parse_args()
    risk_path = args.input_dir / "risk_summary_2026_03_to_2027_02.csv"
    if not risk_path.exists():
        raise SystemExit("risk_summary_2026_03_to_2027_02.csv not found. Run decision support forecast first.")

    risk_df = pd.read_csv(risk_path)
    alerts = []
    for _, row in risk_df.iterrows():
        level = risk_level(row, args)
        alerts.append(
            {
                "series": str(row["series"]),
                "strategy": str(row["strategy"]),
                "risk_level": level,
                "months_below_40": int(row["months_lt40"]),
                "months_below_30": int(row["months_lt30"]),
                "mean_probability_below_40_pct": float(row["mean_prob_below_40_pct"]),
                "mean_probability_below_30_pct": float(row["mean_prob_below_30_pct"]),
                "mean_forecast_pct": float(row["mean_yhat_pct"]),
                "worst_month": str(row["worst_month"]),
                "worst_forecast_pct": float(row["worst_yhat_pct"]),
            }
        )

    high = [a for a in alerts if a["risk_level"] == "high"]
    medium = [a for a in alerts if a["risk_level"] == "medium"]
    low = [a for a in alerts if a["risk_level"] == "low"]

    payload = {
        "window": {"start": "2026-03-01", "end": "2027-02-01"},
        "rules": {
            "high": {
                "months_below_40_at_least": args.threshold_high_risk_months,
                "or_mean_probability_below_40_pct_at_least": args.threshold_high_prob40,
            },
            "medium": {
                "months_below_40_at_least": args.threshold_medium_risk_months,
                "or_mean_probability_below_40_pct_at_least": args.threshold_medium_prob40,
            },
        },
        "counts": {"high": len(high), "medium": len(medium), "low": len(low), "total": len(alerts)},
        "alerts": alerts,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()

