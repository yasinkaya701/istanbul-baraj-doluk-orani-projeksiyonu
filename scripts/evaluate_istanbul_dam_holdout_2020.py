#!/usr/bin/env python3
"""Holdout evaluation: train through 2020-12, test on 2021+ for Istanbul dams."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_istanbul_dam_holdout"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_istanbul_dam_decision_support import forecast_series, rmse, seasonal_naive_forecast, smape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train<=2020 holdout evaluation for Istanbul dam model")
    p.add_argument(
        "--history-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/istanbul_dam_monthly_history.csv"),
    )
    p.add_argument(
        "--config-json",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/run_summary.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/holdout_2020"),
    )
    p.add_argument("--series", default="overall_mean")
    p.add_argument("--train-end", default="2020-12-01")
    p.add_argument("--season-len", type=int, default=12)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-test-months", type=int, default=12)
    p.add_argument("--min-train-months", type=int, default=72)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--ensemble-max-models", type=int, default=3)
    p.add_argument("--ensemble-tie-margin", type=float, default=0.01)
    p.add_argument("--ensemble-shrink", type=float, default=0.00)
    p.add_argument("--enable-stacked-ensemble", action="store_true", default=True)
    p.add_argument("--stack-l2", type=float, default=1.0)
    p.add_argument("--stack-blend-invscore", type=float, default=0.35)
    p.add_argument("--stack-min-weight", type=float, default=0.0)
    p.add_argument("--include-ets-damped", action="store_true")
    p.add_argument("--auto-tune-selection", action="store_true", default=False)
    p.add_argument("--tune-stability-lambda", type=float, default=0.10)
    p.add_argument("--recent-split-weight", type=float, default=0.80)
    p.add_argument("--stability-penalty", type=float, default=0.10)
    p.add_argument("--bias-penalty", type=float, default=0.00)
    p.add_argument("--horizon-damping-start", type=int, default=18)
    p.add_argument("--horizon-damping-strength", type=float, default=0.35)
    p.add_argument("--interval-smoothing", type=float, default=0.35)
    p.add_argument("--lead-bias-correction-strength", type=float, default=0.0)
    p.add_argument("--lead-bias-month-strength", type=float, default=0.0)
    p.add_argument("--lead-bias-max-abs", type=float, default=0.08)
    p.add_argument("--lead-bias-min-samples", type=int, default=3)
    p.add_argument("--seasonal-floor-threshold", type=float, default=1.01)
    p.add_argument("--seasonal-floor-margin", type=float, default=0.0)
    p.add_argument("--seasonal-floor-min-horizon", type=int, default=24)
    return p.parse_args()


def parser_defaults() -> dict[str, Any]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--ensemble-max-models", type=int, default=3)
    p.add_argument("--ensemble-tie-margin", type=float, default=0.01)
    p.add_argument("--ensemble-shrink", type=float, default=0.00)
    p.add_argument("--enable-stacked-ensemble", action="store_true", default=True)
    p.add_argument("--stack-l2", type=float, default=1.0)
    p.add_argument("--stack-blend-invscore", type=float, default=0.35)
    p.add_argument("--stack-min-weight", type=float, default=0.0)
    p.add_argument("--include-ets-damped", action="store_true")
    p.add_argument("--auto-tune-selection", action="store_true", default=False)
    p.add_argument("--tune-stability-lambda", type=float, default=0.10)
    p.add_argument("--recent-split-weight", type=float, default=0.80)
    p.add_argument("--stability-penalty", type=float, default=0.10)
    p.add_argument("--bias-penalty", type=float, default=0.00)
    p.add_argument("--horizon-damping-start", type=int, default=18)
    p.add_argument("--horizon-damping-strength", type=float, default=0.35)
    p.add_argument("--interval-smoothing", type=float, default=0.35)
    p.add_argument("--lead-bias-correction-strength", type=float, default=0.0)
    p.add_argument("--lead-bias-month-strength", type=float, default=0.0)
    p.add_argument("--lead-bias-max-abs", type=float, default=0.08)
    p.add_argument("--lead-bias-min-samples", type=int, default=3)
    p.add_argument("--seasonal-floor-threshold", type=float, default=1.01)
    p.add_argument("--seasonal-floor-margin", type=float, default=0.0)
    p.add_argument("--seasonal-floor-min-horizon", type=int, default=24)
    return vars(p.parse_args([]))


def load_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config_json.exists():
        return args
    try:
        cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    except Exception:
        return args

    defaults = parser_defaults()
    for k in [
        "alpha",
        "ensemble_max_models",
        "ensemble_tie_margin",
        "ensemble_shrink",
        "enable_stacked_ensemble",
        "stack_l2",
        "stack_blend_invscore",
        "stack_min_weight",
        "include_ets_damped",
        "auto_tune_selection",
        "tune_stability_lambda",
        "recent_split_weight",
        "stability_penalty",
        "bias_penalty",
        "horizon_damping_start",
        "horizon_damping_strength",
        "interval_smoothing",
        "lead_bias_correction_strength",
        "lead_bias_month_strength",
        "lead_bias_max_abs",
        "lead_bias_min_samples",
        "seasonal_floor_threshold",
        "seasonal_floor_margin",
        "seasonal_floor_min_horizon",
    ]:
        # CLI argumanlari her zaman config'ten ustun olsun.
        if k in cfg and getattr(args, k) == defaults.get(k):
            setattr(args, k, cfg[k])
    return args


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def save_plot(train: pd.DataFrame, test: pd.DataFrame, pred: pd.DataFrame, series_name: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11.5, 5.8))
    plt.plot(train["ds"], train[series_name] * 100.0, color="#1d4ed8", linewidth=1.7, label="Egitim (<=2020)")
    plt.plot(test["ds"], test[series_name] * 100.0, color="#111827", linewidth=1.8, label="Gercek (2021+)")
    plt.plot(pred["ds"], pred["yhat"] * 100.0, color="#dc2626", linewidth=1.8, linestyle="--", label="Model tahmin")
    if "seasonal_naive_yhat" in pred.columns:
        plt.plot(
            pred["ds"],
            pred["seasonal_naive_yhat"] * 100.0,
            color="#059669",
            linewidth=1.5,
            linestyle=":",
            label="Seasonal naive",
        )
    plt.axvline(pd.Timestamp("2021-01-01"), color="#6b7280", linestyle=":", linewidth=1.0)
    plt.text(pd.Timestamp("2021-01-01"), 99, "  Test baslangici", fontsize=9, color="#374151", va="top")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.xlabel("Tarih")
    plt.ylabel("Doluluk (%)")
    plt.title(f"Holdout Test ({series_name}): Train<=2020, Test>=2021")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    args = load_config_defaults(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(args.history_csv, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    if args.series not in hist.columns:
        raise SystemExit(f"Series not found: {args.series}")

    train_end = pd.Timestamp(args.train_end)
    train = hist[hist["ds"] <= train_end][["ds", args.series]].dropna().copy()
    test = hist[hist["ds"] > train_end][["ds", args.series]].dropna().copy()
    if train.empty or test.empty:
        raise SystemExit("Train/test split is empty. Check train-end date or history file.")

    horizon = len(test)
    fc, stats, cv = forecast_series(
        ds=train["ds"],
        series=train[args.series],
        horizon_months=horizon,
        season_len=int(args.season_len),
        cv_splits=int(args.cv_splits),
        cv_test_months=int(args.cv_test_months),
        min_train_months=int(args.min_train_months),
        alpha=float(args.alpha),
        ensemble_max_models=int(args.ensemble_max_models),
        ensemble_tie_margin=float(args.ensemble_tie_margin),
        ensemble_shrink=float(args.ensemble_shrink),
        enable_stacked_ensemble=bool(args.enable_stacked_ensemble),
        stack_l2=float(args.stack_l2),
        stack_blend_invscore=float(args.stack_blend_invscore),
        stack_min_weight=float(args.stack_min_weight),
        include_ets_damped=bool(args.include_ets_damped),
        auto_tune_selection=bool(args.auto_tune_selection),
        tune_stability_lambda=float(args.tune_stability_lambda),
        recent_split_weight=float(args.recent_split_weight),
        stability_penalty=float(args.stability_penalty),
        bias_penalty=float(args.bias_penalty),
        horizon_damping_start=int(args.horizon_damping_start),
        horizon_damping_strength=float(args.horizon_damping_strength),
        interval_smoothing=float(args.interval_smoothing),
        lead_bias_correction_strength=float(args.lead_bias_correction_strength),
        lead_bias_month_strength=float(args.lead_bias_month_strength),
        lead_bias_max_abs=float(args.lead_bias_max_abs),
        lead_bias_min_samples=int(args.lead_bias_min_samples),
        seasonal_floor_threshold=float(args.seasonal_floor_threshold),
        seasonal_floor_margin=float(args.seasonal_floor_margin),
        seasonal_floor_min_horizon=int(args.seasonal_floor_min_horizon),
    )

    keep_cols = ["ds", "yhat", "strategy", "strategy_rmse", "strategy_score"]
    for c in ["seasonal_floor_applied", "seasonal_floor_trigger_ratio"]:
        if c in fc.columns:
            keep_cols.append(c)
    pred = fc[keep_cols].copy()
    pred = pred.merge(test, on="ds", how="inner")
    pred = pred.rename(columns={args.series: "actual"})
    pred["residual"] = pred["actual"] - pred["yhat"]
    pred["abs_err"] = np.abs(pred["residual"])

    # Baseline: seasonal naive using train-only data.
    sn = seasonal_naive_forecast(train[args.series].to_numpy(dtype=float), horizon=horizon, season_len=int(args.season_len))
    pred["seasonal_naive_yhat"] = np.clip(sn[: len(pred)], 0.0, 1.0)
    pred["seasonal_naive_abs_err"] = np.abs(pred["actual"] - pred["seasonal_naive_yhat"])

    y_true = pred["actual"].to_numpy(dtype=float)
    yhat = pred["yhat"].to_numpy(dtype=float)
    yhat_sn = pred["seasonal_naive_yhat"].to_numpy(dtype=float)

    model_rmse = rmse(y_true, yhat)
    model_mae = mae(y_true, yhat)
    model_smape = smape(y_true, yhat)
    sn_rmse = rmse(y_true, yhat_sn)
    sn_mae = mae(y_true, yhat_sn)
    sn_smape = smape(y_true, yhat_sn)

    metrics = {
        "series": args.series,
        "train_end": str(train_end.date()),
        "test_start": str(test["ds"].min().date()),
        "test_end": str(test["ds"].max().date()),
        "n_train_months": int(len(train)),
        "n_test_months": int(len(test)),
        "selected_strategy": str(pred["strategy"].iloc[0]) if not pred.empty else "unknown",
        "model": {
            "rmse": float(model_rmse),
            "mae": float(model_mae),
            "smape": float(model_smape),
            "approx_accuracy_from_smape_pct": float(100.0 - model_smape),
        },
        "seasonal_naive_baseline": {
            "rmse": float(sn_rmse),
            "mae": float(sn_mae),
            "smape": float(sn_smape),
            "approx_accuracy_from_smape_pct": float(100.0 - sn_smape),
        },
        "improvement_vs_seasonal_naive_pct": {
            "rmse": float((sn_rmse - model_rmse) / max(sn_rmse, 1e-12) * 100.0),
            "mae": float((sn_mae - model_mae) / max(sn_mae, 1e-12) * 100.0),
            "smape": float((sn_smape - model_smape) / max(sn_smape, 1e-12) * 100.0),
        },
        "config_source": str(args.config_json),
        "seasonal_floor": {
            "threshold": float(args.seasonal_floor_threshold),
            "margin": float(args.seasonal_floor_margin),
            "min_horizon": int(args.seasonal_floor_min_horizon),
            "applied_any_month": bool(pred["seasonal_floor_applied"].max()) if "seasonal_floor_applied" in pred.columns else False,
            "trigger_ratio": float(pred["seasonal_floor_trigger_ratio"].iloc[0]) if "seasonal_floor_trigger_ratio" in pred.columns and not pred.empty else 0.0,
        },
    }

    pred.to_csv(args.output_dir / f"{args.series}_holdout_predictions_2021_plus.csv", index=False)
    stats.to_csv(args.output_dir / f"{args.series}_train_cv_metrics.csv", index=False)
    cv.to_csv(args.output_dir / f"{args.series}_train_cv_predictions.csv", index=False)
    (args.output_dir / f"{args.series}_holdout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        f"# Holdout Sonucu ({args.series})",
        "",
        f"- Egitim sonu: `{metrics['train_end']}`",
        f"- Test donemi: `{metrics['test_start']}` - `{metrics['test_end']}`",
        f"- Test ay sayisi: `{metrics['n_test_months']}`",
        f"- Secilen strateji: `{metrics['selected_strategy']}`",
        "",
        "## Model",
        f"- RMSE: `{metrics['model']['rmse']:.6f}`",
        f"- MAE: `{metrics['model']['mae']:.6f}`",
        f"- sMAPE: `{metrics['model']['smape']:.3f}`",
        f"- Yaklasik dogruluk (100-sMAPE): `%{metrics['model']['approx_accuracy_from_smape_pct']:.2f}`",
        "",
        "## Seasonal Naive Baseline",
        f"- RMSE: `{metrics['seasonal_naive_baseline']['rmse']:.6f}`",
        f"- MAE: `{metrics['seasonal_naive_baseline']['mae']:.6f}`",
        f"- sMAPE: `{metrics['seasonal_naive_baseline']['smape']:.3f}`",
        "",
        "## Iyilesme (%)",
        f"- RMSE: `%{metrics['improvement_vs_seasonal_naive_pct']['rmse']:.2f}`",
        f"- MAE: `%{metrics['improvement_vs_seasonal_naive_pct']['mae']:.2f}`",
        f"- sMAPE: `%{metrics['improvement_vs_seasonal_naive_pct']['smape']:.2f}`",
    ]
    (args.output_dir / f"{args.series}_holdout_report.md").write_text("\n".join(lines), encoding="utf-8")

    save_plot(
        train=train,
        test=test,
        pred=pred,
        series_name=args.series,
        out_png=args.output_dir / f"{args.series}_holdout_plot.png",
    )

    print(args.output_dir / f"{args.series}_holdout_metrics.json")
    print(args.output_dir / f"{args.series}_holdout_predictions_2021_plus.csv")
    print(args.output_dir / f"{args.series}_holdout_plot.png")


if __name__ == "__main__":
    main()
