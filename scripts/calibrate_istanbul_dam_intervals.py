#!/usr/bin/env python3
"""Calibrate forecast interval width to reach target empirical coverage on CV.

Supports:
- series-level scaling (single factor per series)
- series+month scaling (factor per series/month with shrinkage to series factor)
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
    p = argparse.ArgumentParser(description="Calibrate Istanbul dam interval width")
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
    p.add_argument("--alpha", type=float, default=0.10, help="Target tail alpha, coverage target=1-alpha")
    p.add_argument("--ensemble-max-models", type=int, default=3)
    p.add_argument("--min-scale", type=float, default=0.6)
    p.add_argument("--max-scale", type=float, default=3.0)
    p.add_argument("--grid-size", type=int, default=241)
    p.add_argument(
        "--scale-mode",
        choices=["series", "series_month"],
        default="series_month",
        help="Calibration mode. series_month uses monthly factors with shrinkage.",
    )
    p.add_argument("--min-month-points", type=int, default=5)
    p.add_argument(
        "--month-shrink-k",
        type=float,
        default=6.0,
        help="Shrinkage strength. Higher value -> monthly factors pulled more toward series factor.",
    )
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


def q_abs_by_month(g: pd.DataFrame, alpha: float) -> tuple[dict[int, float], float]:
    q_level = min(max(1.0 - alpha, 0.5), 0.999)
    arr = np.abs(g["residual"].dropna().to_numpy(dtype=float))
    q_global = float(np.quantile(arr, q_level)) if arr.size else 0.05
    q_map: dict[int, float] = {}
    for month, gm in g.groupby(g["ds"].dt.month):
        a = np.abs(gm["residual"].dropna().to_numpy(dtype=float))
        if a.size >= 5:
            q_map[int(month)] = float(np.quantile(a, q_level))
        else:
            q_map[int(month)] = q_global
    return q_map, q_global


def coverage_with_scale(g: pd.DataFrame, q_map: dict[int, float], q_global: float, scale: float) -> float:
    if g.empty:
        return float("nan")
    y = g["actual"].to_numpy(dtype=float)
    yhat = g["yhat"].to_numpy(dtype=float)
    months = pd.to_datetime(g["ds"]).dt.month.to_numpy(dtype=int)
    qs = np.array([q_map.get(int(m), q_global) for m in months], dtype=float)
    lo = clip01(yhat - scale * qs)
    hi = clip01(yhat + scale * qs)
    covered = (y >= lo) & (y <= hi)
    return float(np.mean(covered))


def find_scale_factor(
    g: pd.DataFrame,
    q_map: dict[int, float],
    q_global: float,
    target_coverage: float,
    min_scale: float,
    max_scale: float,
    grid_size: int,
) -> tuple[float, float, float]:
    scales = np.linspace(min_scale, max_scale, max(5, int(grid_size)))
    covs = np.array([coverage_with_scale(g, q_map, q_global, s) for s in scales], dtype=float)
    base_cov = float(covs[np.argmin(np.abs(scales - 1.0))])
    ok = np.where(covs >= target_coverage)[0]
    if ok.size > 0:
        idx = int(ok[0])
    else:
        idx = int(np.argmax(covs))
    best_scale = float(scales[idx])
    best_cov = float(covs[idx])
    return best_scale, base_cov, best_cov


def calibrate_series_and_month(
    g: pd.DataFrame,
    target: float,
    alpha: float,
    min_scale: float,
    max_scale: float,
    grid_size: int,
    scale_mode: str,
    min_month_points: int,
    month_shrink_k: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    q_map, q_global = q_abs_by_month(g, alpha=alpha)
    series_scale, series_cov_before, series_cov_after = find_scale_factor(
        g=g,
        q_map=q_map,
        q_global=q_global,
        target_coverage=target,
        min_scale=min_scale,
        max_scale=max_scale,
        grid_size=grid_size,
    )
    series_row = {
        "series": str(g["series"].iloc[0]),
        "strategy": str(g["strategy"].iloc[0]),
        "cv_points": int(len(g)),
        "coverage_before_pct": float(series_cov_before * 100.0),
        "coverage_after_pct": float(series_cov_after * 100.0),
        "interval_scale_factor": float(series_scale),
    }

    month_rows = []
    for month in range(1, 13):
        gm = g[g["ds"].dt.month == month].copy()
        n = int(len(gm))
        cov_before = coverage_with_scale(gm, q_map, q_global, scale=1.0)
        if n >= max(1, int(min_month_points)) and scale_mode == "series_month":
            month_raw_scale, _, _ = find_scale_factor(
                g=gm,
                q_map=q_map,
                q_global=q_global,
                target_coverage=target,
                min_scale=min_scale,
                max_scale=max_scale,
                grid_size=grid_size,
            )
        else:
            month_raw_scale = series_scale
        shrink = float(n / (n + max(float(month_shrink_k), 1e-8))) if scale_mode == "series_month" and n > 0 else 0.0
        if scale_mode == "series":
            final_scale = series_scale
            shrink = 0.0
        else:
            final_scale = float(shrink * month_raw_scale + (1.0 - shrink) * series_scale)
        cov_after = coverage_with_scale(gm, q_map, q_global, scale=final_scale)
        month_rows.append(
            {
                "series": str(g["series"].iloc[0]),
                "strategy": str(g["strategy"].iloc[0]),
                "month": month,
                "cv_points_month": n,
                "series_scale_factor": float(series_scale),
                "month_raw_scale_factor": float(month_raw_scale),
                "month_shrink_weight": float(shrink),
                "interval_scale_factor": float(final_scale),
                "coverage_before_pct": (float(cov_before * 100.0) if np.isfinite(cov_before) else np.nan),
                "coverage_after_pct": (float(cov_after * 100.0) if np.isfinite(cov_after) else np.nan),
            }
        )
    return series_row, pd.DataFrame(month_rows)


def plot_scales(summary_df: pd.DataFrame, out_png: Path) -> None:
    if summary_df.empty:
        return
    g = summary_df[summary_df["series"] != "__overall__"].copy().sort_values("interval_scale_factor")
    fig, ax = plt.subplots(figsize=(9.2, max(4.6, len(g) * 0.34)))
    ax.barh(g["series"], g["interval_scale_factor"], color="#2563eb", alpha=0.9)
    ax.axvline(1.0, color="#6b7280", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Scale factor")
    ax.set_title("Interval calibration scale factor by series")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_coverage(summary_df: pd.DataFrame, target_coverage: float, out_png: Path) -> None:
    if summary_df.empty:
        return
    g = summary_df[summary_df["series"] != "__overall__"].copy().sort_values("coverage_before_pct")
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9.2, max(4.8, len(g) * 0.34)))
    ax.barh(y - 0.18, g["coverage_before_pct"], height=0.34, label="Before", color="#f59e0b", alpha=0.9)
    ax.barh(y + 0.18, g["coverage_after_pct"], height=0.34, label="After", color="#16a34a", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(g["series"])
    ax.axvline(target_coverage * 100.0, color="#b91c1c", linestyle="--", linewidth=1.2, label="Target")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Coverage (%)")
    ax.set_title("Interval coverage before/after calibration")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_monthly_scale_heatmap(month_df: pd.DataFrame, out_png: Path) -> None:
    if month_df.empty:
        return
    series_order = (
        month_df.groupby("series", as_index=False)["interval_scale_factor"]
        .mean()
        .sort_values("interval_scale_factor", ascending=False)["series"]
        .tolist()
    )
    mat = np.full((len(series_order), 12), np.nan, dtype=float)
    for i, s in enumerate(series_order):
        g = month_df[month_df["series"] == s]
        for _, r in g.iterrows():
            m = int(r["month"])
            mat[i, m - 1] = float(r["interval_scale_factor"])

    fig, ax = plt.subplots(figsize=(12, max(4.8, len(series_order) * 0.34)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="YlOrRd")
    ax.set_yticks(np.arange(len(series_order)))
    ax.set_yticklabels(series_order, fontsize=9)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([str(m) for m in range(1, 13)])
    ax.set_xlabel("Month")
    ax.set_title("Monthly interval scale factors (series x month)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label("Scale factor", rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def apply_scale_to_forecast(
    fc: pd.DataFrame,
    series_df: pd.DataFrame,
    month_df: pd.DataFrame,
    scale_mode: str,
) -> pd.DataFrame:
    series_scale_map = {
        str(r.series): float(r.interval_scale_factor)
        for r in series_df.itertuples(index=False)
        if str(r.series) != "__overall__"
    }
    month_scale_map = {
        (str(r.series), int(r.month)): float(r.interval_scale_factor)
        for r in month_df.itertuples(index=False)
    }

    out = fc.copy()
    out["month"] = pd.to_datetime(out["ds"]).dt.month
    out["interval_scale_factor_series"] = out["series"].map(series_scale_map).fillna(1.0)
    out["interval_scale_factor_monthly"] = [
        month_scale_map.get((str(s), int(m)), float(ss))
        for s, m, ss in zip(out["series"], out["month"], out["interval_scale_factor_series"], strict=False)
    ]
    out["interval_scale_mode"] = scale_mode
    if scale_mode == "series_month":
        out["interval_scale_factor"] = out["interval_scale_factor_monthly"]
    else:
        out["interval_scale_factor"] = out["interval_scale_factor_series"]

    out["interval_q_abs_base"] = out["interval_q_abs"].to_numpy(dtype=float)
    out["yhat_lower_base"] = out["yhat_lower"].to_numpy(dtype=float)
    out["yhat_upper_base"] = out["yhat_upper"].to_numpy(dtype=float)
    out["interval_q_abs"] = out["interval_q_abs_base"] * out["interval_scale_factor"]
    out["yhat_lower"] = clip01(out["yhat"].to_numpy(dtype=float) - out["interval_q_abs"].to_numpy(dtype=float))
    out["yhat_upper"] = clip01(out["yhat"].to_numpy(dtype=float) + out["interval_q_abs"].to_numpy(dtype=float))
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = args.input_dir / "istanbul_dam_forecasts_decision.csv"
    cv_path = args.input_dir / "istanbul_dam_cv_predictions_decision.csv"
    metrics_path = args.input_dir / "istanbul_dam_cv_metrics_decision.csv"
    strategy_path = args.input_dir / "strategy_summary.csv"
    if not all(p.exists() for p in [forecast_path, cv_path, metrics_path, strategy_path]):
        raise SystemExit("Missing required files. Run forecast_istanbul_dam_decision_support.py first.")

    fc = pd.read_csv(forecast_path, parse_dates=["ds"])
    cv = pd.read_csv(cv_path, parse_dates=["ds"])
    metrics = pd.read_csv(metrics_path)
    strategy = pd.read_csv(strategy_path)

    selected_cv = build_selected_cv(cv=cv, metrics=metrics, strategy=strategy, top_k=args.ensemble_max_models)
    if selected_cv.empty:
        raise SystemExit("No selected CV rows found for interval calibration.")

    target = 1.0 - float(args.alpha)

    series_rows = []
    month_tables = []
    for series_name, g in selected_cv.groupby("series"):
        s_row, m_df = calibrate_series_and_month(
            g=g,
            target=target,
            alpha=float(args.alpha),
            min_scale=float(args.min_scale),
            max_scale=float(args.max_scale),
            grid_size=int(args.grid_size),
            scale_mode=str(args.scale_mode),
            min_month_points=int(args.min_month_points),
            month_shrink_k=float(args.month_shrink_k),
        )
        s_row["series"] = series_name
        series_rows.append(s_row)
        month_tables.append(m_df)

    series_df = pd.DataFrame(series_rows).sort_values("interval_scale_factor").reset_index(drop=True)
    month_df = pd.concat(month_tables, ignore_index=True) if month_tables else pd.DataFrame()

    if not series_df.empty:
        total = float(series_df["cv_points"].sum())
        overall_before = float(np.sum(series_df["coverage_before_pct"] * series_df["cv_points"]) / max(total, 1.0))
        overall_after = float(np.sum(series_df["coverage_after_pct"] * series_df["cv_points"]) / max(total, 1.0))
        overall_scale = float(np.sum(series_df["interval_scale_factor"] * series_df["cv_points"]) / max(total, 1.0))
        series_df = pd.concat(
            [
                series_df,
                pd.DataFrame(
                    [
                        {
                            "series": "__overall__",
                            "strategy": "mixed",
                            "cv_points": int(total),
                            "coverage_before_pct": overall_before,
                            "coverage_after_pct": overall_after,
                            "interval_scale_factor": overall_scale,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    fc_cal = apply_scale_to_forecast(fc=fc, series_df=series_df, month_df=month_df, scale_mode=str(args.scale_mode))

    out_fc = args.output_dir / "istanbul_dam_forecasts_decision_calibrated.csv"
    out_scale = args.output_dir / "interval_calibration_factors.csv"
    out_scale_month = args.output_dir / "interval_calibration_factors_monthly.csv"
    out_summary = args.output_dir / "interval_calibration_summary.json"
    out_plot_scale = args.output_dir / "interval_scale_factors.png"
    out_plot_cov = args.output_dir / "interval_coverage_before_after.png"
    out_plot_heat = args.output_dir / "interval_scale_factors_monthly_heatmap.png"

    fc_cal.to_csv(out_fc, index=False)
    series_df.to_csv(out_scale, index=False)
    month_df.to_csv(out_scale_month, index=False)
    plot_scales(series_df, out_plot_scale)
    plot_coverage(series_df, target_coverage=target, out_png=out_plot_cov)
    plot_monthly_scale_heatmap(month_df, out_plot_heat)

    payload: dict[str, Any] = {
        "alpha": float(args.alpha),
        "target_coverage_pct": float(target * 100.0),
        "grid_size": int(args.grid_size),
        "min_scale": float(args.min_scale),
        "max_scale": float(args.max_scale),
        "scale_mode": str(args.scale_mode),
        "min_month_points": int(args.min_month_points),
        "month_shrink_k": float(args.month_shrink_k),
        "series_count": int((series_df["series"] != "__overall__").sum()),
        "monthly_rows": int(len(month_df)),
    }
    overall_row = series_df[series_df["series"] == "__overall__"]
    if not overall_row.empty:
        payload["overall"] = overall_row.iloc[0].to_dict()
    out_summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(out_fc)
    print(out_scale)
    print(out_scale_month)
    print(out_plot_scale)
    print(out_plot_cov)
    print(out_plot_heat)
    print(out_summary)


if __name__ == "__main__":
    main()
