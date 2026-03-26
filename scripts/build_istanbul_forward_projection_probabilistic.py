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
OUT_DIR = ROOT / "output" / "istanbul_dam_forward_projection_2040_probabilistic"

PRIMARY_MODELS = ["hybrid_ridge", "extra_trees_full"]
PRIMARY_SCENARIOS = ["base", "wet_mild", "hot_dry_high_demand", "management_improvement"]
SCENARIO_LABELS = {
    "base": "Temel",
    "wet_mild": "Ilık-ıslak",
    "hot_dry_high_demand": "Sıcak-kurak-yüksek talep",
    "management_improvement": "Yönetim iyileşme",
    "base_transfer_relief": "Temel + transfer rahatlama",
    "base_transfer_stress": "Temel + transfer stresi",
    "hot_dry_transfer_stress": "Sıcak-kurak + transfer stresi",
}
SCENARIO_COLORS = {
    "base": "#2563eb",
    "wet_mild": "#059669",
    "hot_dry_high_demand": "#dc2626",
    "management_improvement": "#d97706",
}
N_SIMULATIONS = 4000
BLOCK_SIZE = 12
SEED = 42


def load_forward_module():
    spec = importlib.util.spec_from_file_location("istanbul_forward_projection_probabilistic_base", FORWARD_SCRIPT)
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


def build_deterministic_model_projections(forward, train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    _, _, pred_frames = forward.evaluate_models(train_df)
    clim = forward.monthly_climatology(train_df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    all_scenarios = forward.build_scenarios() + forward.build_transfer_scenarios()

    model_parts: list[pd.DataFrame] = []
    residual_sequences: dict[str, np.ndarray] = {}
    for model_name in PRIMARY_MODELS:
        pred_df = pred_frames[model_name].sort_values("date").copy()
        pred_df["residual"] = pred_df["actual"] - pred_df["pred"]
        residual_sequences[model_name] = pred_df["residual"].to_numpy(dtype=float)
        interval_by_month, global_interval, _ = forward.build_empirical_interval_table(pred_df)
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
            proj["scenario"] = cfg.scenario
            proj["model"] = model_name
            model_parts.append(proj)
    model_df = pd.concat(model_parts, ignore_index=True)
    model_df = model_df[(model_df["date"] >= "2026-01-01") & (model_df["date"] <= "2040-12-01")].copy()
    return model_df, residual_sequences


def sample_residual_blocks(residuals: np.ndarray, horizon: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        return np.zeros(horizon, dtype=float)
    if residuals.size <= block_size:
        idx = rng.integers(0, residuals.size, size=horizon)
        return residuals[idx]
    chunks: list[np.ndarray] = []
    while sum(chunk.size for chunk in chunks) < horizon:
        start = int(rng.integers(0, residuals.size - block_size + 1))
        chunks.append(residuals[start : start + block_size])
    return np.concatenate(chunks)[:horizon]


def simulate_probabilistic_paths(
    deterministic_df: pd.DataFrame,
    residual_sequences: dict[str, np.ndarray],
    weights_df: pd.DataFrame,
    n_simulations: int,
    block_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    model_names = weights_df["model"].tolist()
    weight_values = weights_df["weight"].to_numpy(dtype=float)
    quantile_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    crossing_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []

    checkpoints = [pd.Timestamp("2030-12-01"), pd.Timestamp("2035-12-01"), pd.Timestamp("2040-12-01")]

    for scenario, g in deterministic_df.groupby("scenario"):
        g = g.sort_values(["model", "date"]).copy()
        dates = pd.to_datetime(sorted(g["date"].unique()))
        horizon = len(dates)
        model_paths: dict[str, np.ndarray] = {}
        for model_name, mg in g.groupby("model"):
            model_paths[model_name] = mg.sort_values("date")["pred_fill"].to_numpy(dtype=float)

        sims = np.empty((n_simulations, horizon), dtype=np.float32)
        chosen_models = rng.choice(model_names, size=n_simulations, p=weight_values)
        for i, model_name in enumerate(chosen_models):
            base_path = model_paths[model_name]
            residual_path = sample_residual_blocks(residual_sequences[model_name], horizon, block_size, rng)
            sims[i, :] = np.clip(base_path + residual_path, 0.0, 1.0)

        quants = np.quantile(sims, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], axis=0)
        means = sims.mean(axis=0)
        prob40 = (sims < 0.40).mean(axis=0)
        prob30 = (sims < 0.30).mean(axis=0)
        for idx, date in enumerate(dates):
            quantile_rows.append(
                {
                    "scenario": scenario,
                    "date": date,
                    "mean_fill_pct": float(means[idx] * 100.0),
                    "p05_fill_pct": float(quants[0, idx] * 100.0),
                    "p10_fill_pct": float(quants[1, idx] * 100.0),
                    "p25_fill_pct": float(quants[2, idx] * 100.0),
                    "p50_fill_pct": float(quants[3, idx] * 100.0),
                    "p75_fill_pct": float(quants[4, idx] * 100.0),
                    "p90_fill_pct": float(quants[5, idx] * 100.0),
                    "p95_fill_pct": float(quants[6, idx] * 100.0),
                    "prob_below_40_pct": float(prob40[idx] * 100.0),
                    "prob_below_30_pct": float(prob30[idx] * 100.0),
                }
            )

        years = sorted(pd.DatetimeIndex(dates).year.unique())
        for year in years:
            year_mask = pd.DatetimeIndex(dates).year == year
            year_sims = sims[:, year_mask]
            year_dates = pd.DatetimeIndex(dates[year_mask])
            dec_idx = int(np.flatnonzero(year_dates.month == 12)[0])
            dec_vals = year_sims[:, dec_idx]
            yearly_rows.append(
                {
                    "scenario": scenario,
                    "year": int(year),
                    "mean_annual_min_fill_pct": float(year_sims.min(axis=1).mean() * 100.0),
                    "prob_any_month_below_40_pct": float(((year_sims < 0.40).any(axis=1)).mean() * 100.0),
                    "prob_any_month_below_30_pct": float(((year_sims < 0.30).any(axis=1)).mean() * 100.0),
                    "prob_december_below_40_pct": float((dec_vals < 0.40).mean() * 100.0),
                    "prob_december_below_30_pct": float((dec_vals < 0.30).mean() * 100.0),
                    "p10_december_fill_pct": float(np.quantile(dec_vals, 0.10) * 100.0),
                    "p50_december_fill_pct": float(np.quantile(dec_vals, 0.50) * 100.0),
                    "p90_december_fill_pct": float(np.quantile(dec_vals, 0.90) * 100.0),
                }
            )

        for th, label in [(0.40, "40"), (0.30, "30")]:
            for cp in checkpoints:
                cp_mask = dates <= cp
                cp_idx = int(np.flatnonzero(dates == cp)[0])
                cp_sims = sims[:, cp_mask]
                endpoint = sims[:, cp_idx]
                crossing_rows.append(
                    {
                        "scenario": scenario,
                        "threshold_pct": int(label),
                        "checkpoint": str(cp.date()),
                        "prob_any_cross_pct": float(((cp_sims < th).any(axis=1)).mean() * 100.0),
                        "prob_checkpoint_below_pct": float((endpoint < th).mean() * 100.0),
                        "p10_checkpoint_fill_pct": float(np.quantile(endpoint, 0.10) * 100.0),
                        "p50_checkpoint_fill_pct": float(np.quantile(endpoint, 0.50) * 100.0),
                        "p90_checkpoint_fill_pct": float(np.quantile(endpoint, 0.90) * 100.0),
                    }
                )

        endpoint = sims[:, -1]
        endpoint_rows.append(
            {
                "scenario": scenario,
                "mean_endpoint_2040_12_pct": float(endpoint.mean() * 100.0),
                "p10_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.10) * 100.0),
                "p25_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.25) * 100.0),
                "p50_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.50) * 100.0),
                "p75_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.75) * 100.0),
                "p90_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.90) * 100.0),
                "p95_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.95) * 100.0),
            }
        )

    quantiles_df = pd.DataFrame(quantile_rows).sort_values(["scenario", "date"]).reset_index(drop=True)
    yearly_df = pd.DataFrame(yearly_rows).sort_values(["scenario", "year"]).reset_index(drop=True)
    crossing_df = pd.DataFrame(crossing_rows).sort_values(["threshold_pct", "scenario", "checkpoint"]).reset_index(drop=True)
    endpoints_df = pd.DataFrame(endpoint_rows).sort_values("p50_endpoint_2040_12_pct", ascending=False).reset_index(drop=True)
    return quantiles_df, yearly_df, crossing_df, endpoints_df


def build_probabilistic_summary(yearly_df: pd.DataFrame, endpoints_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in endpoints_df["scenario"]:
        row = endpoints_df[endpoints_df["scenario"] == scenario].iloc[0]
        yr = yearly_df[yearly_df["scenario"] == scenario].copy()
        for threshold in [30, 40]:
            col = f"prob_any_month_below_{threshold}_pct"
            hit = yr[yr[col] >= 50.0]
            rows.append(
                {
                    "scenario": scenario,
                    "threshold_pct": threshold,
                    "first_year_prob_ge_50pct": int(hit.iloc[0]["year"]) if not hit.empty else "",
                    "p50_endpoint_2040_12_pct": float(row["p50_endpoint_2040_12_pct"]),
                    "p10_endpoint_2040_12_pct": float(row["p10_endpoint_2040_12_pct"]),
                    "p90_endpoint_2040_12_pct": float(row["p90_endpoint_2040_12_pct"]),
                }
            )
    return pd.DataFrame(rows)


def plot_probabilistic_fans(history: pd.DataFrame, quantiles_df: pd.DataFrame, out_path: Path) -> None:
    hist = history[history["date"] >= "2018-01-01"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), dpi=170, sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, scenario in zip(axes, PRIMARY_SCENARIOS):
        q = quantiles_df[quantiles_df["scenario"] == scenario].copy()
        ax.plot(hist["date"], hist["weighted_total_fill"] * 100.0, color="#111827", linewidth=1.8, label="Gözlenen")
        ax.fill_between(q["date"], q["p10_fill_pct"], q["p90_fill_pct"], color=SCENARIO_COLORS[scenario], alpha=0.14)
        ax.fill_between(q["date"], q["p25_fill_pct"], q["p75_fill_pct"], color=SCENARIO_COLORS[scenario], alpha=0.25)
        ax.plot(q["date"], q["p50_fill_pct"], color=SCENARIO_COLORS[scenario], linewidth=2.0, label="Olasılıksal medyan")
        ax.axvline(pd.Timestamp("2026-01-01"), color="#6b7280", linestyle="--", linewidth=0.9)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.22)
    axes[0].legend(frameon=False, loc="lower left")
    fig.supylabel("Toplam doluluk (%)")
    fig.suptitle("2026-2040 olasılıksal toplam doluluk fan grafikleri", y=0.98)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_yearly_threshold_risk(yearly_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.8), dpi=170, sharex=True)
    for ax, threshold in zip(axes, [40, 30]):
        col = f"prob_any_month_below_{threshold}_pct"
        for scenario in PRIMARY_SCENARIOS:
            g = yearly_df[yearly_df["scenario"] == scenario].copy()
            ax.plot(g["year"], g[col], linewidth=1.9, color=SCENARIO_COLORS[scenario], label=SCENARIO_LABELS[scenario])
        ax.axhline(50.0, color="#6b7280", linestyle="--", linewidth=0.9)
        ax.set_ylabel(f"Yıl içinde <%{threshold} olasılığı")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.22)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Yıl")
    fig.suptitle("Yıllık eşik-altı risk eğrileri", y=0.98)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_endpoint_ranges(endpoints_df: pd.DataFrame, out_path: Path) -> None:
    ep = endpoints_df[endpoints_df["scenario"].isin(PRIMARY_SCENARIOS)].copy()
    ep["scenario_label"] = ep["scenario"].map(SCENARIO_LABELS)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=170)
    x = np.arange(len(ep))
    p10 = ep["p10_endpoint_2040_12_pct"].to_numpy(dtype=float)
    p50 = ep["p50_endpoint_2040_12_pct"].to_numpy(dtype=float)
    p90 = ep["p90_endpoint_2040_12_pct"].to_numpy(dtype=float)
    colors = [SCENARIO_COLORS[s] for s in ep["scenario"]]
    ax.vlines(x, p10, p90, color=colors, linewidth=5, alpha=0.65)
    ax.scatter(x, p50, color="#111827", s=42, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ep["scenario_label"], rotation=10, ha="right")
    ax.set_ylabel("2040 Aralık doluluk (%)")
    ax.set_title("2040 sonunda olasılıksal aralıklar (P10-P50-P90)")
    ax.grid(True, axis="y", alpha=0.22)
    for i, val in enumerate(p50):
        ax.text(i, val + 0.6, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# İstanbul 2026-2040 Olasılıksal Projeksiyon Özeti",
        "",
        f"- Simülasyon sayısı: `{N_SIMULATIONS}`",
        f"- Artık blok boyu: `{BLOCK_SIZE}` ay",
        "- Yöntem: benchmark ağırlıklı iki model karışımı + tarihsel hata bloklarından olasılıksal yol üretimi",
        "",
        "## Ana okumalar",
    ]
    for scenario in PRIMARY_SCENARIOS:
        tmp30 = summary_df[(summary_df["scenario"] == scenario) & (summary_df["threshold_pct"] == 30)].iloc[0]
        tmp40 = summary_df[(summary_df["scenario"] == scenario) & (summary_df["threshold_pct"] == 40)].iloc[0]
        lines.append(
            f"- `{SCENARIO_LABELS[scenario]}`: 2040 Aralık medyan `{tmp30['p50_endpoint_2040_12_pct']:.2f}%`, "
            f"P10-P90 aralığı `{tmp30['p10_endpoint_2040_12_pct']:.2f}% - {tmp30['p90_endpoint_2040_12_pct']:.2f}%`; "
            f"%40 altı yıllık riskin `%50`yi geçtiği ilk yıl `{tmp40['first_year_prob_ge_50pct'] or 'yok'}`, "
            f"%30 altı için `{tmp30['first_year_prob_ge_50pct'] or 'yok'}`."
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = OUT_DIR / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    forward = load_forward_module()
    train_df = forward.load_training_frame()
    weights_df = load_model_weights()
    deterministic_df, residual_sequences = build_deterministic_model_projections(forward, train_df)
    quantiles_df, yearly_df, crossing_df, endpoints_df = simulate_probabilistic_paths(
        deterministic_df=deterministic_df,
        residual_sequences=residual_sequences,
        weights_df=weights_df,
        n_simulations=N_SIMULATIONS,
        block_size=BLOCK_SIZE,
        seed=SEED,
    )
    summary_df = build_probabilistic_summary(yearly_df, endpoints_df)

    weights_df.to_csv(OUT_DIR / "ensemble_model_weights.csv", index=False)
    deterministic_df.to_csv(OUT_DIR / "deterministic_model_projection_monthly_2026_2040.csv", index=False)
    quantiles_df.to_csv(OUT_DIR / "probabilistic_monthly_quantiles_2026_2040.csv", index=False)
    yearly_df.to_csv(OUT_DIR / "probabilistic_yearly_risk_2026_2040.csv", index=False)
    crossing_df.to_csv(OUT_DIR / "probabilistic_crossing_summary_2026_2040.csv", index=False)
    endpoints_df.to_csv(OUT_DIR / "probabilistic_endpoint_summary_2040.csv", index=False)
    summary_df.to_csv(OUT_DIR / "probabilistic_threshold_summary_2026_2040.csv", index=False)

    plot_probabilistic_fans(train_df, quantiles_df, figs / "probabilistic_fan_paths_2026_2040.png")
    plot_yearly_threshold_risk(yearly_df, figs / "probabilistic_yearly_threshold_risk.png")
    plot_endpoint_ranges(endpoints_df, figs / "probabilistic_endpoint_ranges_2040.png")
    write_summary_markdown(summary_df, OUT_DIR / "probabilistic_summary.md")

    summary = {
        "models_used": PRIMARY_MODELS,
        "model_weights": dict(zip(weights_df["model"], weights_df["weight"])),
        "n_simulations": N_SIMULATIONS,
        "block_size_months": BLOCK_SIZE,
        "projection_start": "2026-01-01",
        "projection_end": "2040-12-01",
        "method_note": "Probabilistic paths are built by sampling one model per path using benchmark-based weights and adding stitched historical residual blocks to that model's deterministic scenario path.",
    }
    (OUT_DIR / "simulation_config.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
