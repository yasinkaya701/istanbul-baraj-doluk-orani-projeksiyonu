#!/usr/bin/env python3
"""Build multi-scenario alert JSON from scenario risk summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-scenario Istanbul dam alerts")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/alerts_multi_scenario.json"),
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
    risk_path = args.input_dir / "scenario_risk_summary.csv"
    if not risk_path.exists():
        raise SystemExit("scenario_risk_summary.csv not found. Run build_istanbul_dam_scenarios.py first.")

    risk_df = pd.read_csv(risk_path)
    alerts_by_scenario: dict[str, list[dict]] = {}
    counts_by_scenario: dict[str, dict[str, int]] = {}

    for scenario, g in risk_df.groupby("scenario"):
        alerts = []
        for _, row in g.iterrows():
            lvl = risk_level(row, args)
            alerts.append(
                {
                    "series": str(row["series"]),
                    "strategy": str(row["strategy"]),
                    "risk_level": lvl,
                    "months_below_40": int(row["months_lt40"]),
                    "months_below_30": int(row["months_lt30"]),
                    "mean_probability_below_40_pct": float(row["mean_prob_below_40_pct"]),
                    "mean_probability_below_30_pct": float(row["mean_prob_below_30_pct"]),
                    "mean_forecast_pct": float(row["mean_yhat_pct"]),
                    "worst_month": str(row["worst_month"]),
                    "worst_forecast_pct": float(row["worst_yhat_pct"]),
                }
            )

        alerts = sorted(
            alerts,
            key=lambda a: (
                {"high": 0, "medium": 1, "low": 2}[a["risk_level"]],
                -a["months_below_40"],
                -a["mean_probability_below_40_pct"],
                a["worst_forecast_pct"],
            ),
        )
        alerts_by_scenario[str(scenario)] = alerts
        counts_by_scenario[str(scenario)] = {
            "high": sum(1 for a in alerts if a["risk_level"] == "high"),
            "medium": sum(1 for a in alerts if a["risk_level"] == "medium"),
            "low": sum(1 for a in alerts if a["risk_level"] == "low"),
            "total": len(alerts),
        }

    payload = {
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
        "counts_by_scenario": counts_by_scenario,
        "alerts_by_scenario": alerts_by_scenario,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()

