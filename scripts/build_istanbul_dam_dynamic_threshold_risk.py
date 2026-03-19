#!/usr/bin/env python3
"""Build dynamic-threshold risk outputs for Istanbul dam scenarios.

Dynamic thresholds are derived from historical monthly quantiles per series:
- warning threshold: q_warn (default: 0.25)
- critical threshold: q_crit (default: 0.10)

Probabilities are computed using empirical residual pools from CV predictions
with the same strategy logic used in scenario generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dynamic-threshold scenario risk for Istanbul dams")
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
    p.add_argument("--window-start", default="2026-03-01")
    p.add_argument("--window-end", default="2027-02-01")
    p.add_argument("--warn-quantile", type=float, default=0.25)
    p.add_argument("--critical-quantile", type=float, default=0.10)
    p.add_argument("--min-month-samples", type=int, default=8)
    p.add_argument("--ensemble-max-models", type=int, default=3)
    p.add_argument("--threshold-high-risk-months", type=int, default=4)
    p.add_argument("--threshold-medium-risk-months", type=int, default=2)
    p.add_argument("--threshold-high-prob", type=float, default=60.0)
    p.add_argument("--threshold-medium-prob", type=float, default=35.0)
    return p.parse_args()


def clip01(v: np.ndarray | float) -> np.ndarray | float:
    return np.clip(v, 0.0, 1.0)


def _ensemble_cv_selected(
    cv_series: pd.DataFrame,
    metrics_series: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    base_metrics = metrics_series[~metrics_series["model"].isin(["ensemble_topk"])].copy()
    base_metrics = base_metrics.sort_values("rmse").head(max(1, int(top_k)))
    if base_metrics.empty:
        return pd.DataFrame(columns=["split", "ds", "actual", "yhat", "residual"])

    inv = 1.0 / np.maximum(base_metrics["rmse"].to_numpy(dtype=float), 1e-8)
    w = inv / inv.sum()
    weights = dict(zip(base_metrics["model"].tolist(), w.tolist(), strict=False))

    piv = cv_series.pivot_table(index=["split", "ds", "actual"], columns="model", values="yhat").reset_index()
    usable = [m for m in weights if m in piv.columns]
    if not usable:
        return pd.DataFrame(columns=["split", "ds", "actual", "yhat", "residual"])

    pred = np.zeros(len(piv), dtype=float)
    wsum = 0.0
    for m in usable:
        wm = float(weights[m])
        pred += wm * piv[m].to_numpy(dtype=float)
        wsum += wm
    pred = clip01(pred / max(wsum, 1e-8))
    out = pd.DataFrame(
        {
            "split": piv["split"],
            "ds": pd.to_datetime(piv["ds"]),
            "actual": piv["actual"].to_numpy(dtype=float),
            "yhat": pred,
        }
    )
    out["residual"] = out["actual"] - out["yhat"]
    return out


def build_selected_cv(
    cv: pd.DataFrame,
    metrics: pd.DataFrame,
    strategy: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for series_name, g in strategy.groupby("series"):
        strat = str(g["strategy"].iloc[0])
        cv_s = cv[cv["series"] == series_name].copy()
        if cv_s.empty:
            continue

        if strat == "ensemble_topk":
            m_s = metrics[metrics["series"] == series_name].copy()
            sel = _ensemble_cv_selected(cv_s, m_s, top_k=top_k)
        else:
            sel = cv_s[cv_s["model"] == strat][["split", "ds", "actual", "yhat", "residual"]].copy()
            sel["ds"] = pd.to_datetime(sel["ds"])

        if sel.empty:
            continue
        sel["series"] = series_name
        sel["strategy"] = strat
        rows.append(sel)

    if not rows:
        return pd.DataFrame(columns=["split", "ds", "actual", "yhat", "residual", "series", "strategy"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["series", "ds"]).reset_index(drop=True)


def build_residual_pools(selected_cv: pd.DataFrame) -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}
    for series_name, g in selected_cv.groupby("series"):
        by_month: dict[int, np.ndarray] = {}
        for month, gm in g.groupby(g["ds"].dt.month):
            by_month[int(month)] = gm["residual"].dropna().to_numpy(dtype=float)
        pools[series_name] = {
            "global": g["residual"].dropna().to_numpy(dtype=float),
            "by_month": by_month,
        }
    return pools


def row_residual_sample(pool: dict[str, Any], ds: pd.Timestamp) -> np.ndarray:
    m = int(ds.month)
    arr = pool.get("by_month", {}).get(m)
    if arr is not None and len(arr) >= 5:
        return arr
    return pool.get("global", np.array([], dtype=float))


def build_dynamic_threshold_table(
    monthly_history: pd.DataFrame,
    q_warn: float,
    q_crit: float,
    min_month_samples: int,
) -> pd.DataFrame:
    series_cols = [c for c in monthly_history.columns if c != "ds"]
    rows = []
    for s in series_cols:
        gseries = monthly_history[s].dropna().astype(float)
        if gseries.empty:
            continue
        global_warn = float(np.quantile(gseries, q_warn))
        global_crit = float(np.quantile(gseries, q_crit))
        for month in range(1, 13):
            gm = monthly_history.loc[monthly_history["ds"].dt.month == month, s].dropna().astype(float)
            if len(gm) >= max(1, int(min_month_samples)):
                warn = float(np.quantile(gm, q_warn))
                crit = float(np.quantile(gm, q_crit))
                fallback = False
                n = int(len(gm))
            else:
                warn = global_warn
                crit = global_crit
                fallback = True
                n = int(len(gm))
            rows.append(
                {
                    "series": s,
                    "month": month,
                    "warning_threshold": float(np.clip(warn, 0.0, 1.0)),
                    "critical_threshold": float(np.clip(crit, 0.0, 1.0)),
                    "sample_count": n,
                    "used_global_fallback": fallback,
                }
            )
    return pd.DataFrame(rows).sort_values(["series", "month"]).reset_index(drop=True)


def apply_dynamic_thresholds(
    scen_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    pools: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    thr_map = {
        (str(r.series), int(r.month)): (float(r.warning_threshold), float(r.critical_threshold))
        for r in threshold_df.itertuples(index=False)
    }
    out = scen_df.copy()
    warn_vals = []
    crit_vals = []
    p_warn = []
    p_crit = []
    for _, row in out.iterrows():
        s = str(row["series"])
        ds = pd.Timestamp(row["ds"])
        mean = float(row["scenario_yhat"])
        warn, crit = thr_map.get((s, int(ds.month)), (np.nan, np.nan))
        pool = pools.get(s, {"global": np.array([], dtype=float), "by_month": {}})
        sample = row_residual_sample(pool, ds=ds)
        if np.isnan(warn) or np.isnan(crit):
            warn = np.nan
            crit = np.nan
            pw = np.nan
            pc = np.nan
        elif sample.size == 0:
            pw = float(mean < warn)
            pc = float(mean < crit)
        else:
            dist = clip01(mean + sample)
            pw = float(np.mean(dist < warn))
            pc = float(np.mean(dist < crit))
        warn_vals.append(warn)
        crit_vals.append(crit)
        p_warn.append(pw)
        p_crit.append(pc)
    out["warning_threshold"] = warn_vals
    out["critical_threshold"] = crit_vals
    out["scenario_prob_below_warning"] = p_warn
    out["scenario_prob_below_critical"] = p_crit
    out["scenario_gap_to_warning"] = out["scenario_yhat"] - out["warning_threshold"]
    out["scenario_gap_to_critical"] = out["scenario_yhat"] - out["critical_threshold"]
    return out


def summarize_dynamic_risk(
    dynamic_df: pd.DataFrame,
    window_start: str,
    window_end: str,
) -> pd.DataFrame:
    tmp = dynamic_df[(dynamic_df["ds"] >= window_start) & (dynamic_df["ds"] <= window_end)].copy()
    rows = []
    for (scen, s), g in tmp.groupby(["scenario", "series"]):
        i_worst = g["scenario_gap_to_warning"].idxmin()
        rows.append(
            {
                "scenario": scen,
                "series": s,
                "strategy": str(g["strategy"].iloc[0]),
                "months_below_warning": int((g["scenario_yhat"] < g["warning_threshold"]).sum()),
                "months_below_critical": int((g["scenario_yhat"] < g["critical_threshold"]).sum()),
                "mean_prob_below_warning_pct": float(g["scenario_prob_below_warning"].mean() * 100.0),
                "mean_prob_below_critical_pct": float(g["scenario_prob_below_critical"].mean() * 100.0),
                "mean_gap_to_warning_pct": float(g["scenario_gap_to_warning"].mean() * 100.0),
                "mean_gap_to_critical_pct": float(g["scenario_gap_to_critical"].mean() * 100.0),
                "worst_warning_month": str(pd.Timestamp(g.loc[i_worst, "ds"]).date()),
                "worst_gap_to_warning_pct": float(g["scenario_gap_to_warning"].min() * 100.0),
                "worst_warning_threshold_pct": float(g.loc[i_worst, "warning_threshold"] * 100.0),
                "worst_forecast_pct": float(g.loc[i_worst, "scenario_yhat"] * 100.0),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["scenario", "mean_prob_below_warning_pct", "months_below_warning", "worst_gap_to_warning_pct"],
        ascending=[True, False, False, True],
    )
    return out.reset_index(drop=True)


def dynamic_risk_level(row: pd.Series, args: argparse.Namespace) -> str:
    if (
        int(row["months_below_warning"]) >= args.threshold_high_risk_months
        or float(row["mean_prob_below_warning_pct"]) >= args.threshold_high_prob
    ):
        return "high"
    if (
        int(row["months_below_warning"]) >= args.threshold_medium_risk_months
        or float(row["mean_prob_below_warning_pct"]) >= args.threshold_medium_prob
    ):
        return "medium"
    return "low"


def plot_dynamic_counts(dynamic_summary: pd.DataFrame, out_png: Path, args: argparse.Namespace) -> None:
    if dynamic_summary.empty:
        return
    tmp = dynamic_summary.copy()
    tmp["risk_level"] = tmp.apply(lambda r: dynamic_risk_level(r, args), axis=1)
    scenarios = ["baseline", "dry_stress", "wet_relief"]
    scenarios = [s for s in scenarios if s in tmp["scenario"].unique()] + sorted(
        [s for s in tmp["scenario"].unique() if s not in scenarios]
    )
    levels = ["high", "medium", "low"]
    mat = np.zeros((len(levels), len(scenarios)), dtype=float)
    for i, lvl in enumerate(levels):
        for j, scen in enumerate(scenarios):
            mat[i, j] = float(((tmp["scenario"] == scen) & (tmp["risk_level"] == lvl)).sum())

    x = np.arange(len(scenarios))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"high": "#b91c1c", "medium": "#b45309", "low": "#15803d"}
    for i, lvl in enumerate(levels):
        ax.bar(x + (i - 1) * width, mat[i], width=width, label=lvl, color=colors[lvl], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Dam count")
    ax.set_title("Dynamic-threshold risk counts by scenario")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario_fc_path = args.input_dir / "scenario_forecasts.csv"
    history_path = args.input_dir / "istanbul_dam_monthly_history.csv"
    cv_path = args.input_dir / "istanbul_dam_cv_predictions_decision.csv"
    metrics_path = args.input_dir / "istanbul_dam_cv_metrics_decision.csv"
    strategy_path = args.input_dir / "strategy_summary.csv"
    if not all(p.exists() for p in [scenario_fc_path, history_path, cv_path, metrics_path, strategy_path]):
        raise SystemExit("Missing required files. Run decision + scenario scripts first.")

    scen_df = pd.read_csv(scenario_fc_path, parse_dates=["ds"])
    hist_df = pd.read_csv(history_path, parse_dates=["ds"])
    cv_df = pd.read_csv(cv_path, parse_dates=["ds"])
    metrics_df = pd.read_csv(metrics_path)
    strategy_df = pd.read_csv(strategy_path)

    selected_cv = build_selected_cv(cv=cv_df, metrics=metrics_df, strategy=strategy_df, top_k=args.ensemble_max_models)
    pools = build_residual_pools(selected_cv)
    threshold_df = build_dynamic_threshold_table(
        monthly_history=hist_df,
        q_warn=float(args.warn_quantile),
        q_crit=float(args.critical_quantile),
        min_month_samples=int(args.min_month_samples),
    )
    dynamic_df = apply_dynamic_thresholds(scen_df=scen_df, threshold_df=threshold_df, pools=pools)
    dynamic_summary = summarize_dynamic_risk(
        dynamic_df=dynamic_df,
        window_start=args.window_start,
        window_end=args.window_end,
    )
    dynamic_summary["risk_level"] = dynamic_summary.apply(lambda r: dynamic_risk_level(r, args), axis=1)

    out_forecasts = args.output_dir / "scenario_dynamic_threshold_forecasts.csv"
    out_summary = args.output_dir / "scenario_dynamic_risk_summary.csv"
    out_thresholds = args.output_dir / "dynamic_thresholds_by_series_month.csv"
    out_plot = args.output_dir / "dynamic_threshold_risk_counts.png"
    dynamic_df.to_csv(out_forecasts, index=False)
    dynamic_summary.to_csv(out_summary, index=False)
    threshold_df.to_csv(out_thresholds, index=False)
    plot_dynamic_counts(dynamic_summary, out_plot, args)

    out_meta = args.output_dir / "dynamic_threshold_summary.json"
    meta = {
        "window_start": args.window_start,
        "window_end": args.window_end,
        "warn_quantile": float(args.warn_quantile),
        "critical_quantile": float(args.critical_quantile),
        "min_month_samples": int(args.min_month_samples),
        "threshold_high_risk_months": int(args.threshold_high_risk_months),
        "threshold_medium_risk_months": int(args.threshold_medium_risk_months),
        "threshold_high_prob": float(args.threshold_high_prob),
        "threshold_medium_prob": float(args.threshold_medium_prob),
        "record_count": int(len(dynamic_df)),
        "method": "dynamic_monthly_quantile_thresholds_with_empirical_residual_probability",
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(out_forecasts)
    print(out_summary)
    print(out_thresholds)
    print(out_plot)
    print(out_meta)


if __name__ == "__main__":
    main()
