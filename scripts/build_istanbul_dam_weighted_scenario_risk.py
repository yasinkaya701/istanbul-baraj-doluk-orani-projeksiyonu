#!/usr/bin/env python3
"""Build probability-weighted expected risk summary across scenario portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build weighted scenario expected risk summary")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Monthly history CSV path (default: input-dir/istanbul_dam_monthly_history.csv).",
    )
    p.add_argument("--weight-series", default="overall_mean")
    p.add_argument("--laplace-alpha", type=float, default=1.0)
    p.add_argument("--threshold-high-risk-months", type=int, default=4)
    p.add_argument("--threshold-medium-risk-months", type=int, default=2)
    p.add_argument("--threshold-high-prob40", type=float, default=60.0)
    p.add_argument("--threshold-medium-prob40", type=float, default=35.0)
    p.add_argument("--z-dry-severe", type=float, default=-1.5)
    p.add_argument("--z-dry-base", type=float, default=-0.75)
    p.add_argument("--z-dry-mild", type=float, default=-0.25)
    p.add_argument("--z-wet-mild", type=float, default=0.25)
    p.add_argument("--z-wet-base", type=float, default=0.75)
    p.add_argument("--z-wet-severe", type=float, default=1.5)
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


def classify_z_to_label(z: float, args: argparse.Namespace) -> str:
    if z <= float(args.z_dry_severe):
        return "dry_severe"
    if z <= float(args.z_dry_base):
        return "dry_stress"
    if z <= float(args.z_dry_mild):
        return "dry_mild"
    if z < float(args.z_wet_mild):
        return "baseline"
    if z < float(args.z_wet_base):
        return "wet_mild"
    if z < float(args.z_wet_severe):
        return "wet_relief"
    return "wet_severe"


def nearest_available(name: str, available: set[str]) -> str:
    if name in available:
        return name
    # Fallback map for 3-scenario setup or partial grids.
    fallback = {
        "dry_mild": ["dry_stress", "baseline"],
        "dry_severe": ["dry_stress", "baseline"],
        "wet_mild": ["wet_relief", "baseline"],
        "wet_severe": ["wet_relief", "baseline"],
        "dry_stress": ["dry_mild", "baseline"],
        "wet_relief": ["wet_mild", "baseline"],
    }
    for cand in fallback.get(name, []):
        if cand in available:
            return cand
    return "baseline" if "baseline" in available else sorted(available)[0]


def build_scenario_weights(
    hist: pd.DataFrame,
    available_scenarios: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    series_col = str(args.weight_series)
    if series_col not in hist.columns:
        raise SystemExit(f"Weight series '{series_col}' not found in history file.")

    tmp = hist[["ds", series_col]].dropna().copy()
    tmp["month"] = pd.to_datetime(tmp["ds"]).dt.month
    clim = tmp.groupby("month")[series_col].agg(["mean", "std"]).reset_index()
    tmp = tmp.merge(clim, on="month", how="left")
    std = tmp["std"].replace(0.0, np.nan)
    tmp["z"] = (tmp[series_col] - tmp["mean"]) / std
    tmp["z"] = tmp["z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tmp["raw_label"] = tmp["z"].map(lambda z: classify_z_to_label(float(z), args))
    available = set(available_scenarios)
    tmp["scenario"] = tmp["raw_label"].map(lambda s: nearest_available(str(s), available))

    counts = tmp["scenario"].value_counts().to_dict()
    rows = []
    alpha = float(max(args.laplace_alpha, 0.0))
    total = 0.0
    for s in available_scenarios:
        c = float(counts.get(s, 0.0))
        score = c + alpha
        rows.append({"scenario": s, "count": int(c), "weight_score": score})
        total += score
    for r in rows:
        r["weight"] = float(r["weight_score"] / max(total, 1e-8))
    out = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
    return out


def plot_weighted_risk(expected_df: pd.DataFrame, out_png: Path) -> None:
    if expected_df.empty:
        return
    g = expected_df.copy().sort_values("expected_prob_below_40_pct", ascending=True)
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(10.2, max(5.0, len(g) * 0.34)))
    ax.barh(y, g["expected_prob_below_40_pct"], color="#b45309", alpha=0.85, label="Expected P(<40)")
    ax.set_yticks(y)
    ax.set_yticklabels(g["series"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Expected probability / risk (%)")
    ax.grid(axis="x", alpha=0.25)
    ax2 = ax.twiny()
    ax2.plot(g["prob_high_risk"] * 100.0, y, color="#b91c1c", marker="o", linewidth=1.2, label="P(high risk)")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("P(high risk) (%)")
    ax.set_title("Scenario-weighted expected risk by series")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    risk_path = args.input_dir / "scenario_risk_summary.csv"
    catalog_path = args.input_dir / "scenario_catalog.csv"
    hist_path = args.history_file if args.history_file is not None else (args.input_dir / "istanbul_dam_monthly_history.csv")
    if not risk_path.exists():
        raise SystemExit("scenario_risk_summary.csv not found. Run build_istanbul_dam_scenarios.py first.")
    if not hist_path.exists():
        raise SystemExit("monthly history file not found. Needed for empirical scenario weights.")

    risk_df = pd.read_csv(risk_path)
    hist_df = pd.read_csv(hist_path, parse_dates=["ds"])

    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        available = catalog["scenario"].dropna().astype(str).tolist()
    else:
        available = sorted(risk_df["scenario"].dropna().astype(str).unique().tolist())
    if "baseline" not in available:
        available = ["baseline"] + [s for s in available if s != "baseline"]

    weight_df = build_scenario_weights(hist=hist_df, available_scenarios=available, args=args)
    weight_map = {str(r.scenario): float(r.weight) for r in weight_df.itertuples(index=False)}

    risk_df["risk_level"] = risk_df.apply(lambda r: risk_level(r, args), axis=1)
    risk_df["scenario_weight"] = risk_df["scenario"].map(weight_map).fillna(0.0)

    rows = []
    for series, g in risk_df.groupby("series"):
        w = g["scenario_weight"].to_numpy(dtype=float)
        ws = float(w.sum())
        if ws <= 0:
            continue
        wn = w / ws
        rows.append(
            {
                "series": str(series),
                "strategy": str(g["strategy"].iloc[0]),
                "expected_mean_yhat_pct": float(np.sum(wn * g["mean_yhat_pct"].to_numpy(dtype=float))),
                "expected_prob_below_40_pct": float(np.sum(wn * g["mean_prob_below_40_pct"].to_numpy(dtype=float))),
                "expected_prob_below_30_pct": float(np.sum(wn * g["mean_prob_below_30_pct"].to_numpy(dtype=float))),
                "expected_months_lt40": float(np.sum(wn * g["months_lt40"].to_numpy(dtype=float))),
                "expected_months_lt30": float(np.sum(wn * g["months_lt30"].to_numpy(dtype=float))),
                "prob_high_risk": float(np.sum(wn * (g["risk_level"] == "high").astype(float).to_numpy(dtype=float))),
                "prob_medium_or_high_risk": float(
                    np.sum(wn * ((g["risk_level"] == "high") | (g["risk_level"] == "medium")).astype(float).to_numpy(dtype=float))
                ),
            }
        )
    expected_df = pd.DataFrame(rows).sort_values(
        ["expected_prob_below_40_pct", "prob_high_risk", "expected_mean_yhat_pct"],
        ascending=[False, False, True],
    )

    out_w = args.output_dir / "scenario_weights.csv"
    out_e = args.output_dir / "expected_risk_summary.csv"
    out_plot = args.output_dir / "expected_risk_weighted.png"
    out_json = args.output_dir / "expected_risk_summary.json"
    weight_df.to_csv(out_w, index=False)
    expected_df.to_csv(out_e, index=False)
    plot_weighted_risk(expected_df, out_plot)

    overall = expected_df[expected_df["series"] == "overall_mean"]
    payload: dict[str, Any] = {
        "weight_series": str(args.weight_series),
        "laplace_alpha": float(args.laplace_alpha),
        "scenario_count": int(len(available)),
        "available_scenarios": available,
        "weights": weight_df.to_dict(orient="records"),
        "overall": (overall.iloc[0].to_dict() if not overall.empty else {}),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(out_w)
    print(out_e)
    print(out_plot)
    print(out_json)


if __name__ == "__main__":
    main()
