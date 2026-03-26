#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/yasinkaya/Hackhaton")
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
SCORECARD_PATH = ROOT / "output" / "istanbul_forward_model_benchmark_round2" / "model_selection_scorecard.csv"
OUT_DIR = ROOT / "output" / "istanbul_dam_forward_projection_2040_multimodel"

PRIMARY_MODELS = ["hybrid_ridge", "extra_trees_full"]
PRIMARY_SCENARIOS = ["base", "wet_mild", "hot_dry_high_demand", "management_improvement"]

MODEL_LABELS = {
    "hybrid_ridge": "Hibrit Ridge",
    "extra_trees_full": "Extra Trees",
    "ensemble": "Ensemble",
}

SCENARIO_LABELS = {
    "base": "Temel",
    "wet_mild": "Ilık-ıslak",
    "hot_dry_high_demand": "Sıcak-kurak-yüksek talep",
    "management_improvement": "Yönetim iyileşme",
    "base_transfer_relief": "Temel + transfer rahatlama",
    "base_transfer_stress": "Temel + transfer stresi",
    "hot_dry_transfer_stress": "Sıcak-kurak + transfer stresi",
}

MODEL_COLORS = {
    "hybrid_ridge": "#2563eb",
    "extra_trees_full": "#dc2626",
    "ensemble": "#111827",
}

SCENARIO_COLORS = {
    "base": "#2563eb",
    "wet_mild": "#059669",
    "hot_dry_high_demand": "#dc2626",
    "management_improvement": "#d97706",
}


def load_forward_module():
    spec = importlib.util.spec_from_file_location("istanbul_forward_projection_multimodel_base", FORWARD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Forward projection script import failed.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model_weights() -> pd.DataFrame:
    score = pd.read_csv(SCORECARD_PATH)
    score = score[score["model"].isin(PRIMARY_MODELS)].copy()
    score["raw_weight"] = 1.0 / score["composite_score"]
    score["weight"] = score["raw_weight"] / score["raw_weight"].sum()
    return score.sort_values("weight", ascending=False).reset_index(drop=True)


def aggregate_ensemble(model_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario", "date"]
    for (scenario, date), g in model_df.groupby(group_cols):
        w = g["model"].map(weights).to_numpy(dtype=float)
        w = w / w.sum()
        pred = float(np.sum(g["pred_fill"].to_numpy(dtype=float) * w))
        low_stat = float(np.sum(g["pred_fill_low"].to_numpy(dtype=float) * w))
        high_stat = float(np.sum(g["pred_fill_high"].to_numpy(dtype=float) * w))
        low_model = float(g["pred_fill"].min())
        high_model = float(g["pred_fill"].max())
        rows.append(
            {
                "scenario": scenario,
                "date": date,
                "pred_fill": pred,
                "pred_fill_low": min(low_stat, low_model),
                "pred_fill_high": max(high_stat, high_model),
                "model_disagreement_low": low_model,
                "model_disagreement_high": high_model,
                "model_spread_pp": (high_model - low_model) * 100.0,
                "prob_below_40": float(np.sum(g["prob_below_40"].to_numpy(dtype=float) * w)),
                "prob_below_30": float(np.sum(g["prob_below_30"].to_numpy(dtype=float) * w)),
                "rain_model_mm": float(g["rain_model_mm"].iloc[0]),
                "et0_mm_month": float(g["et0_mm_month"].iloc[0]),
                "consumption_mean_monthly": float(g["consumption_mean_monthly"].iloc[0]),
                "temp_proxy_c": float(g["temp_proxy_c"].iloc[0]),
                "rh_proxy_pct": float(g["rh_proxy_pct"].iloc[0]),
                "vpd_kpa_mean": float(g["vpd_kpa_mean"].iloc[0]),
                "water_balance_proxy_mm": float(g["water_balance_proxy_mm"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "date"]).reset_index(drop=True)


def build_model_spread_summary(ensemble_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checkpoints = [pd.Timestamp("2030-12-01"), pd.Timestamp("2035-12-01"), pd.Timestamp("2040-12-01")]
    for scenario, g in ensemble_df.groupby("scenario"):
        g = g.set_index("date")
        for cp in checkpoints:
            row = g.loc[cp]
            rows.append(
                {
                    "scenario": scenario,
                    "checkpoint": str(cp.date()),
                    "ensemble_fill_pct": float(row["pred_fill"] * 100.0),
                    "model_spread_pp": float(row["model_spread_pp"]),
                    "ensemble_low_pct": float(row["pred_fill_low"] * 100.0),
                    "ensemble_high_pct": float(row["pred_fill_high"] * 100.0),
                }
            )
    return pd.DataFrame(rows)


def plot_ensemble_paths(history: pd.DataFrame, ensemble_df: pd.DataFrame, out_path: Path) -> None:
    hist = history[history["date"] >= "2018-01-01"].copy()
    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=170)
    ax.plot(hist["date"], hist["weighted_total_fill"] * 100.0, color="#111827", linewidth=2.0, label="Gözlenen toplam doluluk")
    for scenario in PRIMARY_SCENARIOS:
        g = ensemble_df[ensemble_df["scenario"] == scenario].copy()
        ax.fill_between(
            g["date"],
            g["pred_fill_low"] * 100.0,
            g["pred_fill_high"] * 100.0,
            color=SCENARIO_COLORS[scenario],
            alpha=0.10,
        )
        ax.plot(
            g["date"],
            g["pred_fill"] * 100.0,
            color=SCENARIO_COLORS[scenario],
            linewidth=1.9,
            label=SCENARIO_LABELS[scenario],
        )
    ax.axvline(pd.Timestamp("2026-01-01"), color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Doluluk (%)")
    ax.set_title("Çok modelli ensemble ile 2026-2040 toplam doluluk yolları")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_base_model_compare(model_df: pd.DataFrame, ensemble_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=170)
    for model in PRIMARY_MODELS:
        g = model_df[(model_df["scenario"] == "base") & (model_df["model"] == model)].copy()
        ax.plot(g["date"], g["pred_fill"] * 100.0, linewidth=1.8, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    eg = ensemble_df[ensemble_df["scenario"] == "base"].copy()
    ax.fill_between(eg["date"], eg["pred_fill_low"] * 100.0, eg["pred_fill_high"] * 100.0, color="#111827", alpha=0.08)
    ax.plot(eg["date"], eg["pred_fill"] * 100.0, linewidth=2.1, color=MODEL_COLORS["ensemble"], label=MODEL_LABELS["ensemble"])
    ax.set_ylim(0, 100)
    ax.set_title("Temel senaryoda model karşılaştırması")
    ax.set_ylabel("Doluluk (%)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_model_spread(spread_df: pd.DataFrame, out_path: Path) -> None:
    cp = spread_df[spread_df["checkpoint"] == "2040-12-01"].copy()
    cp = cp[cp["scenario"].isin(PRIMARY_SCENARIOS)].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=170)
    x = np.arange(len(cp))
    vals = cp["model_spread_pp"].to_numpy(dtype=float)
    ax.bar(x, vals, color=[SCENARIO_COLORS[s] for s in cp["scenario"]])
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in cp["scenario"]], rotation=10, ha="right")
    ax.set_ylabel("Model ayrışması (yp)")
    ax.set_title("2040 sonunda modeller arası ayrışma")
    ax.grid(True, axis="y", alpha=0.2)
    for i, val in enumerate(vals):
        ax.text(i, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = OUT_DIR / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    forward = load_forward_module()
    weights_df = load_model_weights()
    weights = dict(zip(weights_df["model"], weights_df["weight"]))

    train_df = forward.load_training_frame()
    metrics_df, _, pred_frames = forward.evaluate_models(train_df)
    clim = forward.monthly_climatology(train_df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()

    all_scenarios = forward.build_scenarios() + forward.build_transfer_scenarios()
    model_parts = []
    interval_parts = []
    for model_name in PRIMARY_MODELS:
        interval_by_month, global_interval, interval_df = forward.build_empirical_interval_table(pred_frames[model_name])
        residual_pools, global_pool = forward.build_residual_pools(pred_frames[model_name])
        model = forward.fit_model(train_df, model_name)
        for cfg in all_scenarios:
            future = forward.build_future_exog(
                train_df,
                cfg,
                clim,
                demand_relief,
                transfer_share_anchor_pct=transfer_share_anchor_pct,
            )
            proj = forward.simulate_projection(
                train_df,
                future,
                model,
                selected_model=model_name,
                interval_by_month=interval_by_month,
                global_interval=global_interval,
            )
            proj = forward.apply_threshold_probabilities(proj, residual_pools=residual_pools, global_pool=global_pool)
            proj["scenario"] = cfg.scenario
            proj["model"] = model_name
            model_parts.append(proj)
        interval_df["model"] = model_name
        interval_parts.append(interval_df)

    model_df = pd.concat(model_parts, ignore_index=True)
    interval_df = pd.concat(interval_parts, ignore_index=True)
    ensemble_df = aggregate_ensemble(model_df, weights)
    ensemble_df_2026 = ensemble_df[(ensemble_df["date"] >= "2026-01-01") & (ensemble_df["date"] <= "2040-12-01")].copy()
    model_df_2026 = model_df[(model_df["date"] >= "2026-01-01") & (model_df["date"] <= "2040-12-01")].copy()

    primary_ensemble = ensemble_df_2026[ensemble_df_2026["scenario"].isin(PRIMARY_SCENARIOS)].copy()
    ensemble_summary = forward.build_summary_table(primary_ensemble)
    ensemble_risk = forward.build_threshold_risk_summary(primary_ensemble)
    ensemble_checkpoints = forward.build_checkpoint_table(primary_ensemble)
    spread_df = build_model_spread_summary(ensemble_df_2026)

    weights_df.to_csv(OUT_DIR / "ensemble_model_weights.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "base_forward_model_metrics.csv", index=False)
    interval_df.to_csv(OUT_DIR / "model_interval_by_month.csv", index=False)
    model_df.to_csv(OUT_DIR / "model_projection_monthly_2024_2040.csv", index=False)
    model_df_2026.to_csv(OUT_DIR / "model_projection_monthly_2026_2040.csv", index=False)
    ensemble_df.to_csv(OUT_DIR / "ensemble_projection_monthly_2024_2040.csv", index=False)
    ensemble_df_2026.to_csv(OUT_DIR / "ensemble_projection_monthly_2026_2040.csv", index=False)
    ensemble_summary.to_csv(OUT_DIR / "ensemble_projection_summary_2026_2040.csv", index=False)
    ensemble_risk.to_csv(OUT_DIR / "ensemble_threshold_risk_summary_2026_2040.csv", index=False)
    ensemble_checkpoints.to_csv(OUT_DIR / "ensemble_checkpoints_2030_2035_2040.csv", index=False)
    spread_df.to_csv(OUT_DIR / "ensemble_model_spread_summary_2026_2040.csv", index=False)

    plot_ensemble_paths(train_df, primary_ensemble, figs / "ensemble_primary_paths_2026_2040.png")
    plot_base_model_compare(model_df_2026, ensemble_df_2026, figs / "base_model_compare_2026_2040.png")
    plot_model_spread(spread_df, figs / "ensemble_model_spread_2040.png")

    summary = {
        "models_used": PRIMARY_MODELS,
        "model_weights": weights,
        "primary_model": str(weights_df.iloc[0]["model"]),
        "training_start": str(train_df["date"].min().date()),
        "training_end": str(train_df["date"].max().date()),
        "projection_start": "2026-01-01",
        "projection_end": "2040-12-01",
        "note": "Ensemble mean combines the best statistical model with the main physics-clean challenger model.",
    }
    (OUT_DIR / "multimodel_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(OUT_DIR)


if __name__ == "__main__":
    main()
