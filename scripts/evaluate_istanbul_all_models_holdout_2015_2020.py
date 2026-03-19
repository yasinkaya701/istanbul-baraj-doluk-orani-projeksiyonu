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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path("/Users/yasinkaya/Hackhaton")
BENCHMARK_SCRIPT = ROOT / "scripts" / "benchmark_istanbul_forward_models.py"
WB_SCRIPT = ROOT / "scripts" / "build_istanbul_water_balance_v4_sourceaware.py"
OUT_DIR = ROOT / "output" / "istanbul_all_models_holdout_2015_2020"
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


def corr_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pearson = float(pearsonr(actual, pred).statistic)
    spearman = float(spearmanr(actual, pred).statistic)
    return {
        "pearson_corr": pearson,
        "pearson_corr_pct": pearson * 100.0,
        "spearman_corr": spearman,
        "spearman_corr_pct": spearman * 100.0,
        "r2": float(r2_score(actual, pred)),
    }


def run_benchmark_holdout(bench) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = bench.load_training_frame().copy()
    train = train_df[train_df["date"] <= TRAIN_END].copy()
    test = train_df[(train_df["date"] >= TEST_START) & (train_df["date"] <= TEST_END)].copy().reset_index(drop=True)
    rows = {"date": test["date"].copy(), "actual_fill": test["weighted_total_fill"].copy()}
    for spec in bench.model_specs():
        model = bench.fit_model(spec, train)
        preds = bench.recursive_forecast_known_exog(spec, model, train, test)
        rows[spec.name] = preds
    pred_df = pd.DataFrame(rows)
    return train, pred_df


def run_water_balance_holdout(wb) -> pd.DataFrame:
    context = wb.compute_system_context()
    df = wb.load_training_frame(context).copy()
    train = df[df["date"] <= TRAIN_END].copy()
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
    share_by_year, _ = wb.load_transfer_share_by_year()
    train_comp = wb.component_frame(train, context)
    model, month_bias, fit_df = wb.fit_water_balance_model(train_comp)
    transfer_effectiveness = wb.estimate_transfer_effectiveness(train, fit_df, share_by_year)
    pred = wb.simulate_path(
        history_df=train,
        future_exog=test[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
        model=model,
        month_bias=month_bias,
        context=context,
        transfer_share_anchor_pct=0.0,
        transfer_effectiveness=transfer_effectiveness,
        baseline_transfer_share_pct=0.0,
        transfer_end_pct_2040=0.0,
    )
    return pred[["date", "pred_fill"]].rename(columns={"pred_fill": "water_balance_v4_sourceaware"})


def summarize(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = pred_df["actual_fill"].to_numpy(dtype=float)
    for col in [c for c in pred_df.columns if c not in {"date", "actual_fill"}]:
        pred = pred_df[col].to_numpy(dtype=float)
        corr = corr_metrics(actual, pred)
        rows.append(
            {
                "model": col,
                "train_end": str(TRAIN_END.date()),
                "test_start": str(TEST_START.date()),
                "test_end": str(TEST_END.date()),
                "n_test_months": int(len(pred_df)),
                "rmse_pp": rmse_pp(actual, pred),
                "mae_pp": mae_pp(actual, pred),
                "mape_pct": mape_pct(actual, pred),
                "smape_pct": smape_pct(actual, pred),
                "wape_pct": wape_pct(actual, pred),
                "pearson_corr": corr["pearson_corr"],
                "pearson_corr_pct": corr["pearson_corr_pct"],
                "spearman_corr": corr["spearman_corr"],
                "spearman_corr_pct": corr["spearman_corr_pct"],
                "r2": corr["r2"],
                "end_error_pp_2020_12": float((pred[-1] - actual[-1]) * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["mape_pct", "rmse_pp"]).reset_index(drop=True)


def plot_predictions(pred_df: pd.DataFrame, best_models: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.4), dpi=170)
    ax.plot(pred_df["date"], pred_df["actual_fill"] * 100.0, color="#111827", linewidth=2.6, label="Gerçek")
    colors = ["#2563eb", "#059669", "#dc2626", "#d97706"]
    for color, model in zip(colors, best_models, strict=False):
        ax.plot(pred_df["date"], pred_df[model] * 100.0, linewidth=2.0, color=color, label=model)
    ax.set_title("2015'e kadar eğit, 2016-2020 tahmin et - en iyi modeller")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_correlation_bar(summary_df: pd.DataFrame, out_path: Path) -> None:
    tmp = summary_df.sort_values("pearson_corr", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(11.4, 5.2), dpi=170)
    ax.bar(tmp["model"], tmp["pearson_corr_pct"], color="#2563eb")
    ax.set_title("Holdout testinde model-korelasyon karşılaştırması")
    ax.set_ylabel("Pearson korelasyon (%)")
    ax.grid(True, axis="y", alpha=0.22)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_horizontalalignment("right")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = load_module(BENCHMARK_SCRIPT, "benchmark_holdout_all")
    wb = load_module(WB_SCRIPT, "wb_holdout_all")

    _, pred_df = run_benchmark_holdout(bench)
    wb_pred = run_water_balance_holdout(wb)
    pred_df = pred_df.merge(wb_pred, on="date", how="left")
    pred_df["hybrid_physics_ensemble_phys"] = (
        WEIGHT_HYBRID_PHYS * pred_df["hybrid_ridge"] + (1.0 - WEIGHT_HYBRID_PHYS) * pred_df["water_balance_v4_sourceaware"]
    )

    summary_df = summarize(pred_df)
    pred_df.to_csv(OUT_DIR / "all_models_holdout_predictions_2016_2020.csv", index=False)
    summary_df.to_csv(OUT_DIR / "all_models_holdout_summary_2015_train_2020_test.csv", index=False)
    plot_predictions(pred_df, summary_df["model"].head(4).tolist(), OUT_DIR / "all_models_holdout_best4_2016_2020.png")
    plot_correlation_bar(summary_df, OUT_DIR / "all_models_holdout_correlation_bar.png")

    summary = {
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "best_model_by_mape": str(summary_df.iloc[0]["model"]),
        "best_model_mape_pct": float(summary_df.iloc[0]["mape_pct"]),
        "best_model_pearson_corr_pct": float(summary_df.iloc[0]["pearson_corr_pct"]),
        "best_model_rmse_pp": float(summary_df.iloc[0]["rmse_pp"]),
    }
    (OUT_DIR / "all_models_holdout_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
