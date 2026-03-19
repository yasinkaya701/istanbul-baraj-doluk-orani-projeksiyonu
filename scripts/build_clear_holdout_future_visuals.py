#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/yasinkaya/Hackhaton")
BENCHMARK_SCRIPT = ROOT / "scripts" / "benchmark_istanbul_forward_models.py"
HOLDOUT_PRED_PATH = ROOT / "output" / "istanbul_all_models_holdout_2015_2020" / "all_models_holdout_predictions_2016_2020.csv"
HOLDOUT_SUMMARY_PATH = ROOT / "output" / "istanbul_all_models_holdout_2015_2020" / "all_models_holdout_summary_2015_train_2020_test.csv"
FUTURE_PATH = ROOT / "output" / "istanbul_all_models_holdout_future_plots" / "all_models_future_base_paths_2026_2040.csv"
OUT_DIR = ROOT / "output" / "istanbul_clear_holdout_future_visuals"

TRAIN_END = pd.Timestamp("2015-12-01")
HOLDOUT_START = pd.Timestamp("2016-01-01")
HOLDOUT_END = pd.Timestamp("2020-12-01")
FUTURE_START = pd.Timestamp("2026-01-01")
FUTURE_END = pd.Timestamp("2040-12-01")

MODEL_LABELS = {
    "hybrid_physics_ensemble_phys": "Seçilen ensemble",
    "water_balance_v4_sourceaware": "Water Balance v4",
    "hybrid_ridge": "Hibrit Ridge",
    "extra_trees_full": "Extra Trees",
    "history_only_ridge": "Yalnız tarihsel",
    "hybrid_elastic_net": "Elastic Net",
    "hist_gbm_full": "HistGBM",
    "hist_gbm_monotonic": "Monotonik HistGBM",
    "random_forest_full": "Random Forest",
    "persistence": "Süreklilik",
}

MODEL_COLORS = {
    "hybrid_physics_ensemble_phys": "#0f766e",
    "water_balance_v4_sourceaware": "#2563eb",
    "hybrid_ridge": "#dc2626",
    "extra_trees_full": "#d97706",
    "history_only_ridge": "#7c3aed",
    "hybrid_elastic_net": "#0891b2",
    "hist_gbm_full": "#65a30d",
    "hist_gbm_monotonic": "#be123c",
    "random_forest_full": "#92400e",
    "persistence": "#6b7280",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_actual_history() -> pd.DataFrame:
    bench = load_module(BENCHMARK_SCRIPT, "benchmark_actual_for_clear_plots")
    df = bench.load_training_frame().copy()
    df = df[(df["date"] >= "2010-01-01") & (df["date"] <= "2024-02-01")].copy()
    return df[["date", "weighted_total_fill"]].rename(columns={"weighted_total_fill": "actual_fill"})


def style_time_regions(ax) -> None:
    ax.axvspan(pd.Timestamp("2010-01-01"), TRAIN_END, color="#e5e7eb", alpha=0.35)
    ax.axvspan(HOLDOUT_START, HOLDOUT_END, color="#dbeafe", alpha=0.35)
    ax.axvspan(FUTURE_START, FUTURE_END, color="#fee2e2", alpha=0.30)
    ax.axvline(TRAIN_END, color="#6b7280", linestyle=":", linewidth=1.0)
    ax.axvline(HOLDOUT_END, color="#6b7280", linestyle=":", linewidth=1.0)
    ax.axvline(FUTURE_START, color="#6b7280", linestyle="--", linewidth=1.0)


def plot_best_model_clear(actual: pd.DataFrame, holdout: pd.DataFrame, future: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    best = str(summary.iloc[0]["model"])
    row = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(13.2, 5.6), dpi=180)
    ax.plot(actual["date"], actual["actual_fill"] * 100.0, color="#111827", linewidth=2.2, label="Gerçek doluluk")
    ax.plot(
        holdout["date"],
        holdout[best] * 100.0,
        color=MODEL_COLORS[best],
        linewidth=2.0,
        label="2016-2020 model tahmini",
    )
    fut = future[future["model"] == best].copy()
    ax.plot(
        fut["date"],
        fut["pred_fill"] * 100.0,
        color="#b91c1c",
        linewidth=2.1,
        linestyle="--",
        label="2026-2040 temel projeksiyon",
    )
    style_time_regions(ax)
    ax.text(pd.Timestamp("2012-06-01"), 96, "Eğitim dönemi", fontsize=9, color="#374151")
    ax.text(pd.Timestamp("2017-02-01"), 96, "Holdout test", fontsize=9, color="#1d4ed8")
    ax.text(pd.Timestamp("2029-01-01"), 96, "Gelecek projeksiyon", fontsize=9, color="#b91c1c")
    info = (
        f"Model: {MODEL_LABELS.get(best, best)}\n"
        f"MAPE: %{row['mape_pct']:.2f}\n"
        f"RMSE: {row['rmse_pp']:.2f} yp\n"
        f"Korelasyon: %{row['pearson_corr_pct']:.2f}"
    )
    ax.text(
        0.015,
        0.15,
        info,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db"),
    )
    ax.set_title("En iyi model: geçmiş doğrulama ve gelecek projeksiyon")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_top3_zoom(holdout: pd.DataFrame, future: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    top3 = summary.head(3)["model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), dpi=180, sharey=True)

    actual_holdout = holdout[["date", "actual_fill"]].copy()
    axes[0].plot(actual_holdout["date"], actual_holdout["actual_fill"] * 100.0, color="#111827", linewidth=2.2, label="Gerçek")
    for model in top3:
        axes[0].plot(
            holdout["date"],
            holdout[model] * 100.0,
            color=MODEL_COLORS[model],
            linewidth=1.9,
            label=MODEL_LABELS.get(model, model),
        )
    axes[0].set_title("2016-2020 holdout tahmini")
    axes[0].set_ylabel("Toplam doluluk (%)")
    axes[0].grid(True, axis="y", alpha=0.22)

    for model in top3:
        fut = future[future["model"] == model].copy()
        axes[1].plot(
            fut["date"],
            fut["pred_fill"] * 100.0,
            color=MODEL_COLORS[model],
            linewidth=1.9,
            label=MODEL_LABELS.get(model, model),
        )
    axes[1].set_title("2026-2040 temel gelecek yolu")
    axes[1].grid(True, axis="y", alpha=0.22)

    table_rows = []
    for model in top3:
        r = summary[summary["model"] == model].iloc[0]
        table_rows.append(
            f"{MODEL_LABELS.get(model, model)} | MAPE %{r['mape_pct']:.1f} | r %{r['pearson_corr_pct']:.1f}"
        )
    axes[1].text(
        0.03,
        0.03,
        "\n".join(table_rows),
        transform=axes[1].transAxes,
        fontsize=8.5,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db"),
    )
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_error_corr_scatter(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=180)
    for _, row in summary.iterrows():
        model = row["model"]
        if pd.isna(row["pearson_corr_pct"]):
            continue
        ax.scatter(row["mape_pct"], row["pearson_corr_pct"], s=80, color=MODEL_COLORS.get(model, "#2563eb"))
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (row["mape_pct"], row["pearson_corr_pct"]),
            fontsize=8,
            xytext=(6, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("MAPE (%) - düşük daha iyi")
    ax.set_ylabel("Pearson korelasyon (%) - yüksek daha iyi")
    ax.set_title("Model seçimi: hata ve korelasyon birlikte")
    ax.grid(True, alpha=0.22)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = load_actual_history()
    holdout = pd.read_csv(HOLDOUT_PRED_PATH, parse_dates=["date"])
    future = pd.read_csv(FUTURE_PATH, parse_dates=["date"])
    summary = pd.read_csv(HOLDOUT_SUMMARY_PATH)
    plot_best_model_clear(actual, holdout, future, summary, OUT_DIR / "en_iyi_model_gecmis_ve_gelecek.png")
    plot_top3_zoom(holdout, future, summary, OUT_DIR / "ilk_3_model_holdout_ve_gelecek.png")
    plot_error_corr_scatter(summary, OUT_DIR / "model_secimi_hata_korelasyon.png")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
