#!/usr/bin/env python3
"""Evaluate calibration quality of selected Istanbul dam forecasting strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Istanbul dam forecast calibration")
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
    p.add_argument("--threshold-1", type=float, default=0.40)
    p.add_argument("--threshold-2", type=float, default=0.30)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--ensemble-max-models", type=int, default=3)
    p.add_argument(
        "--interval-factors-file",
        type=Path,
        default=None,
        help="Optional interval calibration factors CSV (series, interval_scale_factor).",
    )
    p.add_argument(
        "--interval-factors-monthly-file",
        type=Path,
        default=None,
        help="Optional monthly factors CSV (series, month, interval_scale_factor).",
    )
    return p.parse_args()


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


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


def row_residual_sample(g: pd.DataFrame, ds: pd.Timestamp) -> np.ndarray:
    gm = g[g["ds"].dt.month == int(ds.month)]["residual"].dropna().to_numpy(dtype=float)
    if len(gm) >= 5:
        return gm
    return g["residual"].dropna().to_numpy(dtype=float)


def auc_roc(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(p, dtype=float)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    auc = (ranks[y == 1].sum() - (pos * (pos + 1) / 2.0)) / (pos * neg)
    return float(auc)


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pr = np.asarray(p, dtype=float)
    return float(np.mean((pr - y) ** 2))


def reliability_rows(
    p: np.ndarray,
    y: np.ndarray,
    bins: int,
    series_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, max(2, bins + 1))
    idx = np.digitize(p, edges, right=True)
    idx = np.clip(idx, 1, len(edges) - 1)
    for b in range(1, len(edges)):
        mask = idx == b
        if not np.any(mask):
            continue
        lo = edges[b - 1]
        hi = edges[b]
        rows.append(
            {
                "series": series_name,
                "threshold": threshold,
                "bin_index": b,
                "bin_left": float(lo),
                "bin_right": float(hi),
                "count": int(mask.sum()),
                "mean_pred_prob": float(np.mean(p[mask])),
                "obs_event_rate": float(np.mean(y[mask])),
            }
        )
    return rows


def evaluate_series(
    g: pd.DataFrame,
    threshold_1: float,
    threshold_2: float,
    alpha: float,
    bins: int,
    interval_scale_series: float,
    interval_scale_month: dict[int, float] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    probs_1 = []
    probs_2 = []
    low = []
    high = []
    scale_used = []
    q_level = min(max(1.0 - alpha, 0.5), 0.999)
    abs_err = np.abs(g["residual"].dropna().to_numpy(dtype=float))
    q_global = float(np.quantile(abs_err, q_level)) if abs_err.size else 0.05
    q_by_month: dict[int, float] = {}
    for month, gm in g.groupby(g["ds"].dt.month):
        arr = np.abs(gm["residual"].dropna().to_numpy(dtype=float))
        if arr.size >= 5:
            q_by_month[int(month)] = float(np.quantile(arr, q_level))
        else:
            q_by_month[int(month)] = q_global

    for _, row in g.iterrows():
        ds = pd.Timestamp(row["ds"])
        sample = row_residual_sample(g, ds)
        yhat = float(row["yhat"])
        m = int(ds.month)
        row_scale = float(interval_scale_series)
        if interval_scale_month is not None:
            row_scale = float(interval_scale_month.get(m, row_scale))
        q = float(q_by_month.get(m, q_global)) * row_scale
        if sample.size == 0:
            p1 = float(yhat < threshold_1)
            p2 = float(yhat < threshold_2)
        else:
            dist = clip01(yhat + sample)
            p1 = float(np.mean(dist < threshold_1))
            p2 = float(np.mean(dist < threshold_2))
        lv = float(clip01(yhat - q))
        hv = float(clip01(yhat + q))
        probs_1.append(p1)
        probs_2.append(p2)
        low.append(lv)
        high.append(hv)
        scale_used.append(row_scale)

    eval_df = g.copy()
    eval_df["p_thr1"] = probs_1
    eval_df["p_thr2"] = probs_2
    eval_df["y_thr1"] = (eval_df["actual"] < threshold_1).astype(int)
    eval_df["y_thr2"] = (eval_df["actual"] < threshold_2).astype(int)
    eval_df["pi_low"] = low
    eval_df["pi_high"] = high
    eval_df["interval_scale_factor_row"] = scale_used
    eval_df["covered"] = ((eval_df["actual"] >= eval_df["pi_low"]) & (eval_df["actual"] <= eval_df["pi_high"])).astype(int)

    y1 = eval_df["y_thr1"].to_numpy(dtype=int)
    y2 = eval_df["y_thr2"].to_numpy(dtype=int)
    p1 = eval_df["p_thr1"].to_numpy(dtype=float)
    p2 = eval_df["p_thr2"].to_numpy(dtype=float)

    metrics = {
        "n_points": int(len(eval_df)),
        "event_rate_thr1_pct": float(np.mean(y1) * 100.0),
        "event_rate_thr2_pct": float(np.mean(y2) * 100.0),
        "mean_pred_thr1_pct": float(np.mean(p1) * 100.0),
        "mean_pred_thr2_pct": float(np.mean(p2) * 100.0),
        "calibration_gap_thr1_pct": float((np.mean(p1) - np.mean(y1)) * 100.0),
        "calibration_gap_thr2_pct": float((np.mean(p2) - np.mean(y2)) * 100.0),
        "brier_thr1": brier_score(y1, p1),
        "brier_thr2": brier_score(y2, p2),
        "auc_thr1": auc_roc(y1, p1),
        "auc_thr2": auc_roc(y2, p2),
        "interval_target_coverage_pct": float((1.0 - alpha) * 100.0),
        "interval_empirical_coverage_pct": float(np.mean(eval_df["covered"]) * 100.0),
        "interval_coverage_gap_pct": float((np.mean(eval_df["covered"]) - (1.0 - alpha)) * 100.0),
        "interval_mean_width_pct": float(np.mean(eval_df["pi_high"] - eval_df["pi_low"]) * 100.0),
        "interval_scale_factor": float(np.mean(scale_used) if len(scale_used) else interval_scale_series),
        "interval_scale_factor_series": float(interval_scale_series),
    }

    rel_rows = []
    rel_rows.extend(reliability_rows(p1, y1, bins=bins, series_name=str(g["series"].iloc[0]), threshold=threshold_1))
    rel_rows.extend(reliability_rows(p2, y2, bins=bins, series_name=str(g["series"].iloc[0]), threshold=threshold_2))
    return metrics, rel_rows, eval_df


def plot_reliability(reliability_df: pd.DataFrame, out_png: Path, threshold_1: float, threshold_2: float) -> None:
    if reliability_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for ax, thr in zip(axes, [threshold_1, threshold_2], strict=False):
        g = reliability_df[reliability_df["threshold"] == thr].copy()
        if g.empty:
            ax.set_title(f"Threshold {thr:.2f} (no data)")
            ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280")
            continue
        rows = []
        for (bin_idx, bin_left, bin_right), x in g.groupby(["bin_index", "bin_left", "bin_right"]):
            w = x["count"].to_numpy(dtype=float)
            rows.append(
                {
                    "bin_index": int(bin_idx),
                    "bin_left": float(bin_left),
                    "bin_right": float(bin_right),
                    "count": int(w.sum()),
                    "mean_pred_prob": float(np.average(x["mean_pred_prob"], weights=w)),
                    "obs_event_rate": float(np.average(x["obs_event_rate"], weights=w)),
                }
            )
        agg = pd.DataFrame(rows).sort_values("bin_index")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect calibration")
        ax.plot(agg["mean_pred_prob"], agg["obs_event_rate"], marker="o", color="#1f77b4", label="Observed")
        for _, r in agg.iterrows():
            ax.text(float(r["mean_pred_prob"]), float(r["obs_event_rate"]), str(int(r["count"])), fontsize=7, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability: P(y < {thr:.2f})")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_interval_coverage(metrics_df: pd.DataFrame, out_png: Path) -> None:
    if metrics_df.empty:
        return
    g = metrics_df[metrics_df["series"] != "__overall__"].copy()
    if g.empty:
        return
    g = g.sort_values("interval_empirical_coverage_pct")
    fig, ax = plt.subplots(figsize=(9.5, max(4.4, len(g) * 0.3)))
    ax.barh(g["series"], g["interval_empirical_coverage_pct"], color="#2563eb", alpha=0.85)
    target = float(g["interval_target_coverage_pct"].iloc[0]) if "interval_target_coverage_pct" in g.columns else 90.0
    ax.axvline(target, color="#b91c1c", linestyle="--", linewidth=1.2, label=f"Target {target:.0f}%")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Empirical coverage (%)")
    ax.set_title("Prediction interval coverage by series")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cv_path = args.input_dir / "istanbul_dam_cv_predictions_decision.csv"
    metrics_path = args.input_dir / "istanbul_dam_cv_metrics_decision.csv"
    strategy_path = args.input_dir / "strategy_summary.csv"
    if not all(p.exists() for p in [cv_path, metrics_path, strategy_path]):
        raise SystemExit("Missing required files. Run forecast_istanbul_dam_decision_support.py first.")

    cv_df = pd.read_csv(cv_path, parse_dates=["ds"])
    metrics_df = pd.read_csv(metrics_path)
    strategy_df = pd.read_csv(strategy_path)
    factors_path = args.interval_factors_file
    if factors_path is None:
        auto = args.input_dir / "interval_calibration_factors.csv"
        factors_path = auto if auto.exists() else None
    factors_monthly_path = args.interval_factors_monthly_file
    if factors_monthly_path is None:
        auto_m = args.input_dir / "interval_calibration_factors_monthly.csv"
        factors_monthly_path = auto_m if auto_m.exists() else None

    interval_scale_map: dict[str, float] = {}
    interval_month_map: dict[tuple[str, int], float] = {}
    if factors_path is not None and factors_path.exists():
        fdf = pd.read_csv(factors_path)
        if "series" in fdf.columns and "interval_scale_factor" in fdf.columns:
            interval_scale_map = {
                str(r.series): float(r.interval_scale_factor)
                for r in fdf.itertuples(index=False)
                if str(r.series) != "__overall__"
            }
    if factors_monthly_path is not None and factors_monthly_path.exists():
        fmdf = pd.read_csv(factors_monthly_path)
        if all(c in fmdf.columns for c in ["series", "month", "interval_scale_factor"]):
            interval_month_map = {
                (str(r.series), int(r.month)): float(r.interval_scale_factor)
                for r in fmdf.itertuples(index=False)
            }

    selected_cv = build_selected_cv(
        cv=cv_df,
        metrics=metrics_df,
        strategy=strategy_df,
        top_k=args.ensemble_max_models,
    )
    if selected_cv.empty:
        raise SystemExit("Selected CV table is empty.")

    metric_rows = []
    rel_rows = []
    eval_rows = []
    for series_name, g in selected_cv.groupby("series"):
        scale = float(interval_scale_map.get(str(series_name), 1.0))
        month_map = {m: float(v) for (s, m), v in interval_month_map.items() if s == str(series_name)}
        mt, rel, eval_df = evaluate_series(
            g=g.copy(),
            threshold_1=float(args.threshold_1),
            threshold_2=float(args.threshold_2),
            alpha=float(args.alpha),
            bins=int(args.bins),
            interval_scale_series=scale,
            interval_scale_month=(month_map if month_map else None),
        )
        mt["series"] = series_name
        mt["strategy"] = str(g["strategy"].iloc[0])
        metric_rows.append(mt)
        rel_rows.extend(rel)
        eval_rows.append(eval_df)

    metrics_out = pd.DataFrame(metric_rows).sort_values("brier_thr1").reset_index(drop=True)
    rel_out = pd.DataFrame(rel_rows)
    eval_out = pd.concat(eval_rows, ignore_index=True)

    if not metrics_out.empty:
        overall = {
            "series": "__overall__",
            "strategy": "mixed",
            "n_points": int(metrics_out["n_points"].sum()),
            "event_rate_thr1_pct": float(np.average(metrics_out["event_rate_thr1_pct"], weights=metrics_out["n_points"])),
            "event_rate_thr2_pct": float(np.average(metrics_out["event_rate_thr2_pct"], weights=metrics_out["n_points"])),
            "mean_pred_thr1_pct": float(np.average(metrics_out["mean_pred_thr1_pct"], weights=metrics_out["n_points"])),
            "mean_pred_thr2_pct": float(np.average(metrics_out["mean_pred_thr2_pct"], weights=metrics_out["n_points"])),
            "calibration_gap_thr1_pct": float(np.average(metrics_out["calibration_gap_thr1_pct"], weights=metrics_out["n_points"])),
            "calibration_gap_thr2_pct": float(np.average(metrics_out["calibration_gap_thr2_pct"], weights=metrics_out["n_points"])),
            "brier_thr1": float(np.average(metrics_out["brier_thr1"], weights=metrics_out["n_points"])),
            "brier_thr2": float(np.average(metrics_out["brier_thr2"], weights=metrics_out["n_points"])),
            "auc_thr1": float(np.nanmean(metrics_out["auc_thr1"])),
            "auc_thr2": float(np.nanmean(metrics_out["auc_thr2"])),
            "interval_target_coverage_pct": float(np.average(metrics_out["interval_target_coverage_pct"], weights=metrics_out["n_points"])),
            "interval_empirical_coverage_pct": float(np.average(metrics_out["interval_empirical_coverage_pct"], weights=metrics_out["n_points"])),
            "interval_coverage_gap_pct": float(np.average(metrics_out["interval_coverage_gap_pct"], weights=metrics_out["n_points"])),
            "interval_mean_width_pct": float(np.average(metrics_out["interval_mean_width_pct"], weights=metrics_out["n_points"])),
            "interval_scale_factor": float(np.average(metrics_out["interval_scale_factor"], weights=metrics_out["n_points"])),
            "interval_scale_factor_series": float(np.average(metrics_out["interval_scale_factor_series"], weights=metrics_out["n_points"])),
        }
        metrics_out = pd.concat([metrics_out, pd.DataFrame([overall])], ignore_index=True)

    metrics_path_out = args.output_dir / "calibration_metrics.csv"
    rel_path_out = args.output_dir / "calibration_reliability.csv"
    eval_path_out = args.output_dir / "selected_cv_eval_points.csv"
    fig_rel = args.output_dir / "calibration_reliability_overall.png"
    fig_cov = args.output_dir / "calibration_interval_coverage.png"
    summary_path = args.output_dir / "calibration_summary.json"

    metrics_out.to_csv(metrics_path_out, index=False)
    rel_out.to_csv(rel_path_out, index=False)
    eval_out.to_csv(eval_path_out, index=False)
    plot_reliability(rel_out, fig_rel, threshold_1=float(args.threshold_1), threshold_2=float(args.threshold_2))
    plot_interval_coverage(metrics_out, fig_cov)

    overall_row = metrics_out[metrics_out["series"] == "__overall__"]
    payload = {
        "threshold_1": float(args.threshold_1),
        "threshold_2": float(args.threshold_2),
        "alpha": float(args.alpha),
        "interval_factors_file": str(factors_path) if factors_path is not None else None,
        "interval_factors_monthly_file": str(factors_monthly_path) if factors_monthly_path is not None else None,
        "series_count": int((metrics_out["series"] != "__overall__").sum()),
        "point_count": int(eval_out.shape[0]),
        "overall": (overall_row.iloc[0].to_dict() if not overall_row.empty else {}),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(metrics_path_out)
    print(rel_path_out)
    print(eval_path_out)
    print(fig_rel)
    print(fig_cov)
    print(summary_path)


if __name__ == "__main__":
    main()
