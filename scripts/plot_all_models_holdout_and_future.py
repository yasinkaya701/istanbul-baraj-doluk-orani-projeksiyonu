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
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
OUT_DIR = ROOT / "output" / "istanbul_all_models_holdout_future_plots"
HOLDOUT_PRED_PATH = ROOT / "output" / "istanbul_all_models_holdout_2015_2020" / "all_models_holdout_predictions_2016_2020.csv"
HOLDOUT_SUMMARY_PATH = ROOT / "output" / "istanbul_all_models_holdout_2015_2020" / "all_models_holdout_summary_2015_train_2020_test.csv"
WB_FUTURE_PATH = ROOT / "output" / "istanbul_water_balance_v4_sourceaware_2040" / "water_balance_scenario_projection_monthly_2026_2040.csv"
ENSEMBLE_FUTURE_PATH = ROOT / "output" / "istanbul_hybrid_physics_sourceaware_ensemble_2040" / "ensemble_phys_scenario_projection_monthly_2026_2040.csv"
TRAIN_END = pd.Timestamp("2015-12-01")
ACTUAL_START = pd.Timestamp("2010-01-01")
ACTUAL_END = pd.Timestamp("2024-02-01")
FUTURE_START = pd.Timestamp("2026-01-01")
FUTURE_END = pd.Timestamp("2040-12-01")

MODEL_LABELS = {
    "persistence": "Süreklilik",
    "history_only_ridge": "Yalnız tarihsel",
    "hybrid_ridge": "Hibrit Ridge",
    "hybrid_elastic_net": "Elastic Net",
    "extra_trees_full": "Extra Trees",
    "random_forest_full": "Random Forest",
    "hist_gbm_full": "HistGBM",
    "hist_gbm_monotonic": "Monotonik HistGBM",
    "water_balance_v4_sourceaware": "Water Balance v4",
    "hybrid_physics_ensemble_phys": "Seçilen ensemble",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_benchmark_future_base() -> pd.DataFrame:
    bench = load_module(BENCHMARK_SCRIPT, "benchmark_plot_all")
    forward = load_module(FORWARD_SCRIPT, "forward_plot_all")
    train_df = bench.load_training_frame().copy()
    train_df = train_df[train_df["date"] <= ACTUAL_END].copy()
    clim = forward.monthly_climatology(train_df)
    _, demand_relief_pct = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    base_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == "base")
    future_exog = forward.build_future_exog(
        train_df,
        base_cfg,
        clim,
        demand_relief_pct,
        transfer_share_anchor_pct=transfer_share_anchor_pct,
    )
    future_exog = future_exog[(future_exog["date"] >= FUTURE_START) & (future_exog["date"] <= FUTURE_END)].copy()

    rows = []
    for spec in bench.model_specs():
        model = bench.fit_model(spec, train_df)
        pred = bench.recursive_forecast_known_exog(spec, model, train_df, future_exog)
        rows.append(
            pd.DataFrame(
                {
                    "date": future_exog["date"].to_numpy(),
                    "model": spec.name,
                    "pred_fill": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_actual_history() -> pd.DataFrame:
    bench = load_module(BENCHMARK_SCRIPT, "benchmark_plot_all_actual")
    df = bench.load_training_frame().copy()
    df = df[(df["date"] >= ACTUAL_START) & (df["date"] <= ACTUAL_END)].copy()
    return df[["date", "weighted_total_fill"]].rename(columns={"weighted_total_fill": "actual_fill"})


def load_future_base_paths() -> pd.DataFrame:
    benchmark_future = build_benchmark_future_base()
    wb = pd.read_csv(WB_FUTURE_PATH, parse_dates=["date"])
    wb = wb[wb["scenario"] == "base"][["date", "pred_fill"]].copy()
    wb["model"] = "water_balance_v4_sourceaware"
    ensemble = pd.read_csv(ENSEMBLE_FUTURE_PATH, parse_dates=["date"])
    ensemble = ensemble[ensemble["scenario"] == "base"][["date", "pred_fill_ensemble"]].copy()
    ensemble = ensemble.rename(columns={"pred_fill_ensemble": "pred_fill"})
    ensemble["model"] = "hybrid_physics_ensemble_phys"
    return pd.concat([benchmark_future, wb, ensemble], ignore_index=True)


def plot_small_multiples(actual: pd.DataFrame, holdout: pd.DataFrame, future: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    models = summary["model"].tolist()
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), dpi=170, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, model in zip(axes, models, strict=False):
        ax.plot(actual["date"], actual["actual_fill"] * 100.0, color="#111827", linewidth=1.8, label="Gerçek")
        if model in holdout.columns:
            ax.plot(
                holdout["date"],
                holdout[model] * 100.0,
                color="#2563eb",
                linewidth=1.8,
                label="2016-2020 tahmini",
            )
        fut = future[future["model"] == model].copy()
        if not fut.empty:
            ax.plot(
                fut["date"],
                fut["pred_fill"] * 100.0,
                color="#dc2626",
                linewidth=1.8,
                linestyle="--",
                label="2026-2040 temel gelecek",
            )
        row = summary[summary["model"] == model].iloc[0]
        ax.set_title(f"{MODEL_LABELS.get(model, model)} | MAPE %{row['mape_pct']:.1f} | r %{row['pearson_corr_pct']:.1f}")
        ax.grid(True, axis="y", alpha=0.22)
        ax.axvline(TRAIN_END, color="#6b7280", linestyle=":", linewidth=0.9)
        ax.axvline(FUTURE_START, color="#6b7280", linestyle="--", linewidth=0.9)
        ax.set_ylim(0, 100)
    for ax in axes[len(models):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Tüm modeller - 2016-2020 holdout tahmini ve 2026-2040 gelecek yolu", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path)
    plt.close(fig)


def plot_top4(actual: pd.DataFrame, holdout: pd.DataFrame, future: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    top = summary.head(4)["model"].tolist()
    fig, ax = plt.subplots(figsize=(12.4, 5.6), dpi=170)
    ax.plot(actual["date"], actual["actual_fill"] * 100.0, color="#111827", linewidth=2.4, label="Gerçek")
    palette = ["#2563eb", "#059669", "#dc2626", "#d97706"]
    for color, model in zip(palette, top, strict=False):
        ax.plot(holdout["date"], holdout[model] * 100.0, color=color, linewidth=1.8, label=f"{MODEL_LABELS.get(model, model)} holdout")
        fut = future[future["model"] == model].copy()
        ax.plot(fut["date"], fut["pred_fill"] * 100.0, color=color, linewidth=1.8, linestyle="--", label=f"{MODEL_LABELS.get(model, model)} gelecek")
    ax.axvline(TRAIN_END, color="#6b7280", linestyle=":", linewidth=1.0)
    ax.axvline(FUTURE_START, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_title("En iyi 4 model - holdout ve gelecek")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = build_actual_history()
    holdout = pd.read_csv(HOLDOUT_PRED_PATH, parse_dates=["date"])
    summary = pd.read_csv(HOLDOUT_SUMMARY_PATH)
    future = load_future_base_paths()
    future.to_csv(OUT_DIR / "all_models_future_base_paths_2026_2040.csv", index=False)
    plot_small_multiples(actual, holdout, future, summary, OUT_DIR / "all_models_holdout_and_future_small_multiples.png")
    plot_top4(actual, holdout, future, summary, OUT_DIR / "all_models_holdout_and_future_top4.png")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
