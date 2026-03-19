#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/Users/yasinkaya/Hackhaton")
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
WB_SCRIPT = ROOT / "scripts" / "build_istanbul_water_balance_v4_sourceaware.py"
OUT_DIR = ROOT / "output" / "istanbul_holdout_2015_2020"
TRAIN_END = pd.Timestamp("2015-12-01")
TEST_START = pd.Timestamp("2016-01-01")
TEST_END = pd.Timestamp("2020-12-01")
WEIGHT_HYBRID_PHYS = 0.45


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def mape_pct(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual), 1e-6)
    return float(np.mean(np.abs(pred - actual) / denom) * 100.0)


def smape_pct(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual) + np.abs(pred), 1e-6)
    return float(np.mean(2.0 * np.abs(pred - actual) / denom) * 100.0)


def wape_pct(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.sum(np.abs(actual)), 1e-6)
    return float(np.sum(np.abs(pred - actual)) / denom * 100.0)


def rmse_pp(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, pred)) * 100.0)


def mae_pp(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(mean_absolute_error(actual, pred) * 100.0)


def build_common_frames(forward, wb):
    fwd_df = forward.load_training_frame().copy()
    context = wb.compute_system_context()
    wb_df = wb.load_training_frame(context).copy()
    common_dates = sorted(set(fwd_df["date"]).intersection(set(wb_df["date"])))
    fwd_df = fwd_df[fwd_df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
    wb_df = wb_df[wb_df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
    return fwd_df, wb_df, context


def run_holdout():
    forward = load_module(FORWARD_SCRIPT, "forward_holdout_2015_2020")
    wb = load_module(WB_SCRIPT, "wb_holdout_2015_2020")
    fwd_df, wb_df, context = build_common_frames(forward, wb)

    train_fwd = fwd_df[fwd_df["date"] <= TRAIN_END].copy()
    test_fwd = fwd_df[(fwd_df["date"] >= TEST_START) & (fwd_df["date"] <= TEST_END)].copy()
    train_wb = wb_df[wb_df["date"] <= TRAIN_END].copy()
    test_wb = wb_df[(wb_df["date"] >= TEST_START) & (wb_df["date"] <= TEST_END)].copy()

    if train_fwd.empty or test_fwd.empty:
        raise RuntimeError("Train/test split is empty.")

    # Hybrid ridge
    hybrid_model = forward.fit_model(train_fwd, "hybrid_ridge")
    hybrid_future = test_fwd[
        [
            "date",
            "rain_model_mm",
            "et0_mm_month",
            "consumption_mean_monthly",
            "temp_proxy_c",
            "rh_proxy_pct",
            "vpd_kpa_mean",
            "month_sin",
            "month_cos",
        ]
    ].copy()
    hybrid_pred = forward.simulate_projection(
        train_df=train_fwd,
        future_exog=hybrid_future,
        model=hybrid_model,
        selected_model="hybrid_ridge",
        interval_by_month={},
        global_interval=(0.0, 0.0),
    )[["date", "pred_fill"]].rename(columns={"pred_fill": "pred_fill_hybrid"})

    # Water balance v4
    share_by_year, _ = wb.load_transfer_share_by_year()
    train_comp = wb.component_frame(train_wb, context)
    wb_model, wb_month_bias, wb_fit_df = wb.fit_water_balance_model(train_comp)
    wb_transfer_eff = wb.estimate_transfer_effectiveness(train_wb, wb_fit_df, share_by_year)
    wb_pred = wb.simulate_path(
        history_df=train_wb,
        future_exog=test_wb[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
        model=wb_model,
        month_bias=wb_month_bias,
        context=context,
        transfer_share_anchor_pct=0.0,
        transfer_effectiveness=wb_transfer_eff,
        baseline_transfer_share_pct=0.0,
        transfer_end_pct_2040=0.0,
    )[["date", "pred_fill"]].rename(columns={"pred_fill": "pred_fill_wb"})

    actual = test_fwd[["date", "weighted_total_fill"]].rename(columns={"weighted_total_fill": "actual_fill"})
    pred = actual.merge(hybrid_pred, on="date").merge(wb_pred, on="date")
    pred["pred_fill_ensemble_phys"] = (
        WEIGHT_HYBRID_PHYS * pred["pred_fill_hybrid"] + (1.0 - WEIGHT_HYBRID_PHYS) * pred["pred_fill_wb"]
    )
    return pred


def summarize_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for col, model_name in [
        ("pred_fill_hybrid", "hybrid_ridge"),
        ("pred_fill_wb", "water_balance_v4_sourceaware"),
        ("pred_fill_ensemble_phys", "hybrid_physics_ensemble_phys"),
    ]:
        actual = pred["actual_fill"].to_numpy(dtype=float)
        forecast = pred[col].to_numpy(dtype=float)
        rows.append(
            {
                "model": model_name,
                "train_end": str(TRAIN_END.date()),
                "test_start": str(TEST_START.date()),
                "test_end": str(TEST_END.date()),
                "n_test_months": int(len(pred)),
                "rmse_pp": rmse_pp(actual, forecast),
                "mae_pp": mae_pp(actual, forecast),
                "mape_pct": mape_pct(actual, forecast),
                "smape_pct": smape_pct(actual, forecast),
                "wape_pct": wape_pct(actual, forecast),
                "end_error_pp_2020_12": float((forecast[-1] - actual[-1]) * 100.0),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("mape_pct").reset_index(drop=True)

    yearly_rows = []
    work = pred.copy()
    work["year"] = work["date"].dt.year
    for year, g in work.groupby("year"):
        actual = g["actual_fill"].to_numpy(dtype=float)
        for col, model_name in [
            ("pred_fill_hybrid", "hybrid_ridge"),
            ("pred_fill_wb", "water_balance_v4_sourceaware"),
            ("pred_fill_ensemble_phys", "hybrid_physics_ensemble_phys"),
        ]:
            forecast = g[col].to_numpy(dtype=float)
            yearly_rows.append(
                {
                    "year": int(year),
                    "model": model_name,
                    "rmse_pp": rmse_pp(actual, forecast),
                    "mae_pp": mae_pp(actual, forecast),
                    "mape_pct": mape_pct(actual, forecast),
                    "mean_actual_fill_pct": float(np.mean(actual) * 100.0),
                    "mean_pred_fill_pct": float(np.mean(forecast) * 100.0),
                }
            )
    yearly_df = pd.DataFrame(yearly_rows).sort_values(["year", "mape_pct"]).reset_index(drop=True)
    return summary_df, yearly_df


def plot_holdout(pred: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=170)
    ax.plot(pred["date"], pred["actual_fill"] * 100.0, color="#111827", linewidth=2.4, label="Gerçek")
    ax.plot(pred["date"], pred["pred_fill_hybrid"] * 100.0, color="#2563eb", linewidth=1.8, label="Hybrid Ridge")
    ax.plot(pred["date"], pred["pred_fill_wb"] * 100.0, color="#dc2626", linewidth=1.8, label="Water Balance v4")
    ax.plot(pred["date"], pred["pred_fill_ensemble_phys"] * 100.0, color="#059669", linewidth=2.2, label="Seçilen ensemble")
    ax.set_title("2015'e kadar eğit, 2016-2020 tahmin et - holdout testi")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = run_holdout()
    summary_df, yearly_df = summarize_predictions(pred)
    pred.to_csv(OUT_DIR / "holdout_predictions_2016_2020.csv", index=False)
    summary_df.to_csv(OUT_DIR / "holdout_summary_2015_train_2020_test.csv", index=False)
    yearly_df.to_csv(OUT_DIR / "holdout_yearly_metrics_2016_2020.csv", index=False)
    plot_holdout(pred, OUT_DIR / "holdout_predictions_2016_2020.png")
    summary = {
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "selected_weight_hybrid_phys": WEIGHT_HYBRID_PHYS,
        "best_model_by_mape": str(summary_df.iloc[0]["model"]),
        "best_model_mape_pct": float(summary_df.iloc[0]["mape_pct"]),
    }
    (OUT_DIR / "holdout_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
