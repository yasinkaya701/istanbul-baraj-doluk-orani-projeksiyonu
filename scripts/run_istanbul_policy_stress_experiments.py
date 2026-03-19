#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/yasinkaya/Hackhaton")
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
WB_SCRIPT = ROOT / "scripts" / "build_istanbul_water_balance_v4_sourceaware.py"
ENSEMBLE_SUMMARY_PATH = ROOT / "output" / "istanbul_hybrid_physics_sourceaware_ensemble_2040" / "ensemble_summary.json"
ENSEMBLE_CALIB_PATH = ROOT / "output" / "istanbul_hybrid_physics_sourceaware_ensemble_2040" / "ensemble_calibration_samples.csv"
CORRECTION_CURVE_PATH = ROOT / "output" / "istanbul_reconciled_projection_2040" / "nearterm_correction_curve_2026_2040.csv"
OUT_DIR = ROOT / "output" / "istanbul_policy_stress_experiments_2026_03_12"
N_SIMULATIONS = 800
BLOCK_SIZE = 12
SPREAD_SCALE = 1.35
SEED = 42

BASE_POLICY_PER_CAPITA = [-0.15, -0.30, -0.45, -0.60, -0.75]
BASE_POLICY_NRW = [1.5, 3.0, 4.5, 6.0, 7.5]
HOT_DRY_PER_CAPITA = [0.15, -0.15, -0.45, -0.75, -1.05]
HOT_DRY_TRANSFER = [0.0, 10.0, 20.0, 30.0, 40.0]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def month_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 12 - diff)


def residual_pool_for_month(residual_df: pd.DataFrame, month: int, min_count: int = 6) -> np.ndarray:
    months = residual_df["date"].dt.month.to_numpy(dtype=int)
    mask = months == month
    pool = residual_df.loc[mask, "residual"].to_numpy(dtype=float)
    if pool.size >= min_count:
        return pool
    for radius in [1, 2]:
        mask = np.array([month_distance(int(m), int(month)) <= radius for m in months], dtype=bool)
        pool = residual_df.loc[mask, "residual"].to_numpy(dtype=float)
        if pool.size >= min_count:
            return pool
    return residual_df["residual"].to_numpy(dtype=float)


def sample_residual_blocks(
    residual_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    residual_df = residual_df.sort_values("date").reset_index(drop=True)
    hist_dates = pd.DatetimeIndex(residual_df["date"])
    hist_residuals = residual_df["residual"].to_numpy(dtype=float)
    valid_starts: dict[int, list[int]] = {}
    for month in range(1, 13):
        starts = []
        for i in range(0, len(residual_df) - block_size + 1):
            if int(hist_dates[i].month) == month:
                starts.append(i)
        valid_starts[month] = starts

    out: list[np.ndarray] = []
    pos = 0
    while pos < len(future_dates):
        month = int(future_dates[pos].month)
        starts = valid_starts.get(month, [])
        take = min(block_size, len(future_dates) - pos)
        if starts:
            start = int(rng.choice(starts))
            out.append(hist_residuals[start:start + take])
        else:
            pool = residual_pool_for_month(residual_df, month)
            idx = rng.integers(0, len(pool), size=take)
            out.append(pool[idx])
        pos += take
    return np.concatenate(out)[: len(future_dates)]


def load_selected_weight() -> float:
    data = json.loads(ENSEMBLE_SUMMARY_PATH.read_text())
    return float(data["weight_hybrid_ridge_phys"])


def load_residuals() -> pd.DataFrame:
    calib = pd.read_csv(ENSEMBLE_CALIB_PATH, parse_dates=["date"])
    calib = calib[calib["horizon_months"] == 1].copy().sort_values("date").reset_index(drop=True)
    calib["residual"] = calib["actual_fill"] - calib["pred_fill_ensemble_phys"]
    return calib[["date", "residual"]].copy()


def load_correction_curve() -> pd.DataFrame:
    corr = pd.read_csv(CORRECTION_CURVE_PATH, parse_dates=["date"])
    return corr[["date", "correction_pp"]].copy()


def simulate_custom_scenarios(configs: list, out_monthly_name: str) -> pd.DataFrame:
    forward = load_module(FORWARD_SCRIPT, "forward_stress_experiments")
    wb = load_module(WB_SCRIPT, "wb_stress_experiments")

    weight_hybrid = load_selected_weight()
    corr = load_correction_curve()

    fwd_df = forward.load_training_frame().copy()
    wb_context = wb.compute_system_context()
    wb_df = wb.load_training_frame(wb_context).copy()
    common_dates = sorted(set(fwd_df["date"]).intersection(set(wb_df["date"])))
    fwd_df = fwd_df[fwd_df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
    wb_df = wb_df[wb_df["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)

    clim = forward.monthly_climatology(fwd_df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    wb_share_by_year, wb_anchor_share_pct = wb.load_transfer_share_by_year()
    wb_comp = wb.component_frame(wb_df, wb_context)
    wb_model, wb_month_bias, wb_fit_df = wb.fit_water_balance_model(wb_comp)
    wb_transfer_eff = wb.estimate_transfer_effectiveness(wb_df, wb_fit_df, wb_share_by_year)
    hybrid_model = forward.fit_model(fwd_df, "hybrid_ridge")

    rows = []
    for cfg in configs:
        neutral_cfg = cfg if float(cfg.transfer_end_pct_2040) == 0.0 else replace(cfg, transfer_end_pct_2040=0.0)
        future = forward.build_future_exog(
            fwd_df,
            neutral_cfg,
            clim,
            demand_relief,
            transfer_share_anchor_pct=0.0,
        )
        sim_h = forward.simulate_projection(
            train_df=fwd_df,
            future_exog=future[
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
            ],
            model=hybrid_model,
            selected_model="hybrid_ridge",
            interval_by_month={},
            global_interval=(0.0, 0.0),
        )
        sim_w = wb.simulate_path(
            history_df=wb_df,
            future_exog=future[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
            model=wb_model,
            month_bias=wb_month_bias,
            context=wb_context,
            transfer_share_anchor_pct=wb_anchor_share_pct,
            transfer_effectiveness=wb_transfer_eff,
            baseline_transfer_share_pct=wb_anchor_share_pct,
            transfer_end_pct_2040=float(cfg.transfer_end_pct_2040),
        )
        ens = pd.DataFrame(
            {
                "date": sim_h["date"],
                "scenario": cfg.scenario,
                "pred_fill_hybrid": sim_h["pred_fill"],
                "pred_fill_wb": sim_w["pred_fill"],
                "pred_fill_core": weight_hybrid * sim_h["pred_fill"] + (1.0 - weight_hybrid) * sim_w["pred_fill"],
                "rain_model_mm": sim_h["rain_model_mm"],
                "et0_mm_month": sim_h["et0_mm_month"],
                "consumption_mean_monthly": future["consumption_mean_monthly"].to_numpy(dtype=float),
                "per_capita_use_pct_per_year": float(cfg.per_capita_use_pct_per_year),
                "nrw_reduction_pp_by_2040": float(cfg.nrw_reduction_pp_by_2040),
                "transfer_end_pct_2040": float(cfg.transfer_end_pct_2040),
            }
        )
        ens = ens.merge(corr, on="date", how="left")
        ens["correction_pp"] = ens["correction_pp"].fillna(0.0)
        ens["pred_fill_reconciled_pct"] = np.clip(ens["pred_fill_core"] * 100.0 + ens["correction_pp"], 0.0, 100.0)
        ens["pred_fill_reconciled"] = ens["pred_fill_reconciled_pct"] / 100.0
        rows.append(ens)

    out = pd.concat(rows, ignore_index=True).sort_values(["scenario", "date"]).reset_index(drop=True)
    out.to_csv(OUT_DIR / out_monthly_name, index=False)
    return out


def summarize_probabilistic(monthly_df: pd.DataFrame, residual_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for scenario, g in monthly_df.groupby("scenario"):
        g = g.sort_values("date").copy()
        g = g[g["date"] >= pd.Timestamp("2026-01-01")].copy()
        dates = pd.DatetimeIndex(g["date"])
        base_path = g["pred_fill_reconciled"].to_numpy(dtype=float)
        sims = np.empty((N_SIMULATIONS, len(base_path)), dtype=np.float32)
        for i in range(N_SIMULATIONS):
            noise = SPREAD_SCALE * sample_residual_blocks(residual_df, dates, BLOCK_SIZE, rng)
            sims[i, :] = np.clip(base_path + noise, 0.0, 1.0)
        endpoint = sims[:, -1]
        yearly_any40 = []
        yearly_any30 = []
        years = [int(y) for y in sorted(dates.year.unique()) if int(y) >= 2026]
        first40 = ""
        first30 = ""
        for year in years:
            mask = dates.year == year
            ys = sims[:, mask]
            p40 = float(((ys < 0.40).any(axis=1)).mean() * 100.0)
            p30 = float(((ys < 0.30).any(axis=1)).mean() * 100.0)
            yearly_any40.append((year, p40))
            yearly_any30.append((year, p30))
        for year, p in yearly_any40:
            if p >= 50.0:
                first40 = int(year)
                break
        for year, p in yearly_any30:
            if p >= 50.0:
                first30 = int(year)
                break
        rows.append(
            {
                "scenario": scenario,
                "mean_fill_2026_2040_pct": float(g["pred_fill_reconciled_pct"].mean()),
                "end_fill_2040_12_pct": float(g.iloc[-1]["pred_fill_reconciled_pct"]),
                "p10_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.10) * 100.0),
                "p50_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.50) * 100.0),
                "p90_endpoint_2040_12_pct": float(np.quantile(endpoint, 0.90) * 100.0),
                "first_year_prob_below_40_ge_50pct": first40,
                "first_year_prob_below_30_ge_50pct": first30,
                "per_capita_use_pct_per_year": float(g["per_capita_use_pct_per_year"].iloc[0]),
                "nrw_reduction_pp_by_2040": float(g["nrw_reduction_pp_by_2040"].iloc[0]),
                "transfer_end_pct_2040": float(g["transfer_end_pct_2040"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def heatmap(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    fmt: str = ".1f",
) -> None:
    pivot = df.pivot(index=index_col, columns=columns_col, values=values_col).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=170)
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, format(float(pivot.iloc[i, j]), fmt), ha="center", va="center", color="#111827", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(values_col.replace("_", " "), rotation=270, labelpad=14)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_key_cases(df: pd.DataFrame, out_path: Path) -> None:
    labels = {
        "base_current": "Temel",
        "base_strong_management": "Temel + güçlü yönetim",
        "base_extreme_management": "Temel + çok güçlü yönetim",
        "hotdry_current": "Sıcak-kurak",
        "hotdry_management_transfer": "Sıcak-kurak + yönetim + transfer",
        "hotdry_full_support": "Sıcak-kurak + tam destek",
    }
    colors = {
        "base_current": "#2563eb",
        "base_strong_management": "#0f766e",
        "base_extreme_management": "#059669",
        "hotdry_current": "#dc2626",
        "hotdry_management_transfer": "#d97706",
        "hotdry_full_support": "#7c3aed",
    }
    fig, ax = plt.subplots(figsize=(11.4, 5.2), dpi=170)
    for scenario, g in df.groupby("scenario"):
        g = g.sort_values("date")
        ax.plot(g["date"], g["pred_fill_reconciled_pct"], label=labels.get(scenario, scenario), linewidth=2.0, color=colors.get(scenario))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Toplam doluluk (%)")
    ax.set_title("Secili deney yolları")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = OUT_DIR / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    forward = load_module(FORWARD_SCRIPT, "forward_stress_configs")
    base_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == "base")
    hot_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == "hot_dry_high_demand")

    # Base-climate management frontier
    base_configs = []
    for pc in BASE_POLICY_PER_CAPITA:
        for nrw in BASE_POLICY_NRW:
            base_configs.append(
                replace(
                    base_cfg,
                    scenario=f"base_pc{abs(pc):.2f}_nrw{nrw:.1f}",
                    per_capita_use_pct_per_year=float(pc),
                    nrw_reduction_pp_by_2040=float(nrw),
                )
            )
    base_monthly = simulate_custom_scenarios(base_configs, "base_policy_frontier_monthly.csv")
    residual_df = load_residuals()
    base_summary = summarize_probabilistic(base_monthly, residual_df)
    base_summary["per_capita_abs_saving_pct_per_year"] = base_summary["per_capita_use_pct_per_year"].abs()
    base_summary.to_csv(OUT_DIR / "base_policy_frontier_summary.csv", index=False)

    # Hot-dry adaptation frontier
    hot_configs = []
    for pc in HOT_DRY_PER_CAPITA:
        for transfer in HOT_DRY_TRANSFER:
            hot_configs.append(
                replace(
                    hot_cfg,
                    scenario=f"hotdry_pc{pc:+.2f}_transfer{transfer:.0f}",
                    per_capita_use_pct_per_year=float(pc),
                    transfer_end_pct_2040=float(transfer),
                )
            )
    hot_monthly = simulate_custom_scenarios(hot_configs, "hotdry_adaptation_frontier_monthly.csv")
    hot_summary = summarize_probabilistic(hot_monthly, residual_df)
    hot_summary["per_capita_abs_saving_pct_per_year"] = hot_summary["per_capita_use_pct_per_year"].abs()
    hot_summary.to_csv(OUT_DIR / "hotdry_adaptation_frontier_summary.csv", index=False)

    # Selected cases for path comparison
    key_cases = [
        replace(base_cfg, scenario="base_current"),
        replace(base_cfg, scenario="base_strong_management", per_capita_use_pct_per_year=-0.60, nrw_reduction_pp_by_2040=6.0),
        replace(base_cfg, scenario="base_extreme_management", per_capita_use_pct_per_year=-0.75, nrw_reduction_pp_by_2040=7.5),
        replace(hot_cfg, scenario="hotdry_current"),
        replace(hot_cfg, scenario="hotdry_management_transfer", per_capita_use_pct_per_year=-0.75, transfer_end_pct_2040=20.0),
        replace(hot_cfg, scenario="hotdry_full_support", per_capita_use_pct_per_year=-1.05, nrw_reduction_pp_by_2040=4.5, transfer_end_pct_2040=40.0),
    ]
    key_monthly = simulate_custom_scenarios(key_cases, "key_cases_monthly.csv")
    key_summary = summarize_probabilistic(key_monthly, residual_df)
    key_summary.to_csv(OUT_DIR / "key_cases_summary.csv", index=False)

    # Heatmaps
    heatmap(
        base_summary,
        index_col="nrw_reduction_pp_by_2040",
        columns_col="per_capita_abs_saving_pct_per_year",
        values_col="p50_endpoint_2040_12_pct",
        title="Temel iklimde yönetim cephesi - 2040 P50 doluluk",
        out_path=figs / "base_policy_frontier_p50_2040.png",
        xlabel="Kisi basi kullanim azalis hizi (%/yil, mutlak)",
        ylabel="NRW iyilesmesi (yp, 2040)",
    )
    heatmap(
        base_summary,
        index_col="nrw_reduction_pp_by_2040",
        columns_col="per_capita_abs_saving_pct_per_year",
        values_col="end_fill_2040_12_pct",
        title="Temel iklimde yönetim cephesi - 2040 deterministik doluluk",
        out_path=figs / "base_policy_frontier_det_2040.png",
        xlabel="Kisi basi kullanim azalis hizi (%/yil, mutlak)",
        ylabel="NRW iyilesmesi (yp, 2040)",
    )
    heatmap(
        hot_summary,
        index_col="transfer_end_pct_2040",
        columns_col="per_capita_use_pct_per_year",
        values_col="p50_endpoint_2040_12_pct",
        title="Sicak-kurak durumda adaptasyon cephesi - 2040 P50 doluluk",
        out_path=figs / "hotdry_adaptation_frontier_p50_2040.png",
        xlabel="Kisi basi kullanim trendi (%/yil)",
        ylabel="Transfer degisimi (%, 2040)",
    )
    heatmap(
        hot_summary,
        index_col="transfer_end_pct_2040",
        columns_col="per_capita_use_pct_per_year",
        values_col="end_fill_2040_12_pct",
        title="Sicak-kurak durumda adaptasyon cephesi - 2040 deterministik doluluk",
        out_path=figs / "hotdry_adaptation_frontier_det_2040.png",
        xlabel="Kisi basi kullanim trendi (%/yil)",
        ylabel="Transfer degisimi (%, 2040)",
    )
    plot_key_cases(key_monthly, figs / "key_cases_paths.png")

    # Decision-oriented extracts
    base_safe40 = base_summary[base_summary["p50_endpoint_2040_12_pct"] >= 40.0].copy()
    hot_safe30 = hot_summary[hot_summary["p50_endpoint_2040_12_pct"] >= 30.0].copy()
    base_safe40.to_csv(OUT_DIR / "base_policy_cases_p50_ge_40.csv", index=False)
    hot_safe30.to_csv(OUT_DIR / "hotdry_cases_p50_ge_30.csv", index=False)

    summary = {
        "n_base_policy_cases": int(len(base_summary)),
        "n_hotdry_cases": int(len(hot_summary)),
        "n_key_cases": int(len(key_summary)),
        "base_cases_with_p50_ge_40": int(len(base_safe40)),
        "hotdry_cases_with_p50_ge_30": int(len(hot_safe30)),
        "selected_weight_hybrid": load_selected_weight(),
        "spread_scale": SPREAD_SCALE,
        "n_simulations": N_SIMULATIONS,
    }
    (OUT_DIR / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
