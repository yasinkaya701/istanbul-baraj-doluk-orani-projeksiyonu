#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/Users/yasinkaya/Hackhaton")
V4_SCRIPT = ROOT / "scripts" / "build_istanbul_water_balance_v4_sourceaware.py"
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
OPS_PROXY_PATH = ROOT / "output" / "newdata_feature_store" / "tables" / "monthly_operational_proxies_2000_2026.csv"
OUT_DIR = ROOT / "output" / "istanbul_water_balance_v7_ops_residual_2040"

TRAIN_END = pd.Timestamp("2015-12-01")
TEST_START = pd.Timestamp("2016-01-01")
TEST_END = pd.Timestamp("2020-12-01")

BASE_FEATURE_ORDER = [
    "source_runoff_now_mcm",
    "source_runoff_lag1_mcm",
    "source_runoff_wetness_mcm",
    "source_lake_rain_mcm",
    "neg_source_openwater_evap_mcm",
    "neg_supply_mcm",
    "neg_storage_mass_mcm",
    "neg_spill_pressure_mcm",
]
OPS_FEATURE_ORDER = [
    "transfer_effective_mcm",
    "neg_nrw_mcm",
    "reclaimed_mcm",
]


@dataclass(frozen=True)
class Params:
    area_exponent: float
    spill_threshold: float
    estimated_transfer_scale: float


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def allocate_source_fills(total_fill: float, context: dict[str, object]) -> np.ndarray:
    src = context["source_table"]
    caps = src["max_storage_million_m3"].to_numpy(dtype=float)
    priors = src["fill_factor_prior"].to_numpy(dtype=float)
    target_storage = float(np.clip(total_fill, 0.0, 1.0) * caps.sum())
    if target_storage <= 0.0:
        return np.zeros_like(caps)
    lo = 0.0
    hi = max(3.0, 1.0 / max(priors.min(), 1e-6))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fills = np.clip(priors * mid, 0.0, 1.0)
        storage = float(np.dot(caps, fills))
        if storage < target_storage:
            lo = mid
        else:
            hi = mid
    return np.clip(priors * hi, 0.0, 1.0)


def source_state(fill: float, context: dict[str, object], params: Params) -> dict[str, np.ndarray]:
    src = context["source_table"]
    fills = allocate_source_fills(fill, context)
    full_lake_area = src["est_lake_area_km2"].to_numpy(dtype=float)
    lake_area = full_lake_area * np.power(np.clip(fills, 0.0, 1.0), params.area_exponent)
    basin_area = src["basin_area_km2"].fillna(0.0).to_numpy(dtype=float)
    land_area = np.maximum(basin_area - lake_area, 0.0)
    storage = fills * src["max_storage_million_m3"].to_numpy(dtype=float)
    spill_pressure = np.maximum(fills - params.spill_threshold, 0.0) * src["max_storage_million_m3"].to_numpy(dtype=float)
    productivity = src["runoff_productivity_weight"].to_numpy(dtype=float)
    return {
        "fills": fills,
        "lake_area_km2": lake_area,
        "land_area_km2": land_area,
        "storage_mcm": storage,
        "spill_pressure_mcm": spill_pressure,
        "runoff_productivity_weight": productivity,
    }


def load_frame(v4, context: dict[str, object]) -> pd.DataFrame:
    df = v4.load_training_frame(context).copy()
    ops = pd.read_csv(OPS_PROXY_PATH, parse_dates=["date"]).copy()
    keep = [
        "date",
        "transfer_share_pct_annual_est",
        "transfer_share_pct_monthly_proxy",
        "transfer_mcm_monthly_proxy",
        "transfer_share_source",
        "nrw_pct_monthly_proxy",
        "nrw_mcm_monthly_proxy",
        "reclaimed_share_pct_monthly_proxy",
        "reclaimed_mcm_monthly_proxy",
    ]
    df = df.merge(ops[keep], on="date", how="left")
    for col in keep:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce") if col != "transfer_share_source" else df[col]
    num_cols = [c for c in keep if c not in {"date", "transfer_share_source"}]
    df[num_cols] = df[num_cols].ffill().bfill()
    df["transfer_share_source"] = df["transfer_share_source"].ffill().bfill().fillna("estimated")
    return df.sort_values("date").reset_index(drop=True)


def component_frame(df: pd.DataFrame, context: dict[str, object], params: Params) -> pd.DataFrame:
    out = df[
        [
            "date",
            "weighted_total_fill",
            "weighted_total_fill_lag1",
            "rain_model_mm",
            "rain_model_mm_lag1",
            "rain_model_mm_roll3",
            "et0_mm_month",
            "et0_mm_month_roll3",
            "supply_mcm",
            "delta_storage_mcm",
            "transfer_mcm_monthly_proxy",
            "transfer_share_source",
            "nrw_mcm_monthly_proxy",
            "reclaimed_mcm_monthly_proxy",
        ]
    ].copy()
    state_rows = [source_state(float(fill), context, params) for fill in out["weighted_total_fill_lag1"]]
    out["lake_area_km2"] = [float(s["lake_area_km2"].sum()) for s in state_rows]
    out["land_area_km2"] = [float(s["land_area_km2"].sum()) for s in state_rows]
    out["runoff_weighted_land_area_km2"] = [
        float(np.dot(s["land_area_km2"], s["runoff_productivity_weight"])) for s in state_rows
    ]
    wetness_mm = np.clip(out["rain_model_mm_roll3"] - out["et0_mm_month_roll3"], 0.0, None)
    out["source_runoff_now_mcm"] = out["runoff_weighted_land_area_km2"] * out["rain_model_mm"] * 0.001
    out["source_runoff_lag1_mcm"] = out["runoff_weighted_land_area_km2"] * out["rain_model_mm_lag1"] * 0.001
    out["source_runoff_wetness_mcm"] = out["runoff_weighted_land_area_km2"] * wetness_mm * 0.001
    out["source_lake_rain_mcm"] = out["lake_area_km2"] * out["rain_model_mm"] * 0.001
    out["source_openwater_evap_mcm"] = out["lake_area_km2"] * out["et0_mm_month"] * 0.001
    out["storage_mass_mcm"] = [float(s["storage_mcm"].sum()) for s in state_rows]
    out["spill_pressure_mcm"] = [float(s["spill_pressure_mcm"].sum()) for s in state_rows]
    out["neg_source_openwater_evap_mcm"] = -out["source_openwater_evap_mcm"]
    out["neg_supply_mcm"] = -out["supply_mcm"]
    out["neg_storage_mass_mcm"] = -out["storage_mass_mcm"]
    out["neg_spill_pressure_mcm"] = -out["spill_pressure_mcm"]
    return out


def fit_models(train_comp: pd.DataFrame, params: Params):
    base_model = LinearRegression(positive=True)
    base_model.fit(train_comp[BASE_FEATURE_ORDER], train_comp["delta_storage_mcm"])
    tmp = train_comp[["date", "delta_storage_mcm"]].copy()
    tmp["base_pred_mcm"] = base_model.predict(train_comp[BASE_FEATURE_ORDER])
    tmp["month"] = tmp["date"].dt.month
    tmp["base_resid_mcm"] = tmp["delta_storage_mcm"] - tmp["base_pred_mcm"]
    month_bias = tmp.groupby("month")["base_resid_mcm"].mean().to_dict()
    tmp["base_plus_month_mcm"] = tmp["base_pred_mcm"] + tmp["month"].map(month_bias).fillna(0.0)

    ops_train = train_comp.copy()
    ops_train["transfer_effective_mcm"] = ops_train["transfer_mcm_monthly_proxy"] * np.where(
        ops_train["transfer_share_source"].eq("official"),
        1.0,
        params.estimated_transfer_scale,
    )
    ops_train["neg_nrw_mcm"] = -ops_train["nrw_mcm_monthly_proxy"]
    ops_train["reclaimed_mcm"] = ops_train["reclaimed_mcm_monthly_proxy"]
    ops_target = train_comp["delta_storage_mcm"].to_numpy(dtype=float) - tmp["base_plus_month_mcm"].to_numpy(dtype=float)
    ops_model = LinearRegression(positive=True)
    ops_model.fit(ops_train[OPS_FEATURE_ORDER], ops_target)
    return base_model, month_bias, ops_model


def simulate_path(history_df: pd.DataFrame, future_df: pd.DataFrame, context: dict[str, object], params: Params, base_model, month_bias, ops_model) -> pd.DataFrame:
    past_fill = history_df["weighted_total_fill"].tolist()
    past_rain = history_df["rain_model_mm"].tolist()
    past_et0 = history_df["et0_mm_month"].tolist()
    total_storage = float(context["total_storage_mcm"])
    rows = []
    for row in future_df.reset_index(drop=True).itertuples(index=False):
        d = row._asdict()
        date = pd.Timestamp(d["date"])
        fill_prev = float(past_fill[-1])
        state = source_state(fill_prev, context, params)
        lake_area_km2 = float(state["lake_area_km2"].sum())
        runoff_land_area = float(np.dot(state["land_area_km2"], state["runoff_productivity_weight"]))
        rain_now = float(d["rain_model_mm"])
        rain_lag1 = float(past_rain[-1])
        et0_now = float(d["et0_mm_month"])
        rain_roll3 = float(np.mean([past_rain[-2], past_rain[-1], rain_now]))
        et0_roll3 = float(np.mean([past_et0[-2], past_et0[-1], et0_now]))
        comp = {
            "source_runoff_now_mcm": runoff_land_area * rain_now * 0.001,
            "source_runoff_lag1_mcm": runoff_land_area * rain_lag1 * 0.001,
            "source_runoff_wetness_mcm": runoff_land_area * max(rain_roll3 - et0_roll3, 0.0) * 0.001,
            "source_lake_rain_mcm": lake_area_km2 * rain_now * 0.001,
            "source_openwater_evap_mcm": lake_area_km2 * et0_now * 0.001,
            "supply_mcm": float(d["supply_mcm"]),
            "storage_mass_mcm": float(state["storage_mcm"].sum()),
            "spill_pressure_mcm": float(state["spill_pressure_mcm"].sum()),
        }
        comp["neg_source_openwater_evap_mcm"] = -comp["source_openwater_evap_mcm"]
        comp["neg_supply_mcm"] = -comp["supply_mcm"]
        comp["neg_storage_mass_mcm"] = -comp["storage_mass_mcm"]
        comp["neg_spill_pressure_mcm"] = -comp["spill_pressure_mcm"]
        base_delta = float(base_model.predict(pd.DataFrame([{k: comp[k] for k in BASE_FEATURE_ORDER}]))[0])
        base_delta += float(month_bias.get(date.month, 0.0))

        transfer_effective = float(d["transfer_mcm_monthly_proxy"])
        if str(d.get("transfer_share_source", "official")) == "estimated":
            transfer_effective *= params.estimated_transfer_scale
        ops_row = {
            "transfer_effective_mcm": transfer_effective,
            "neg_nrw_mcm": -float(d["nrw_mcm_monthly_proxy"]),
            "reclaimed_mcm": float(d["reclaimed_mcm_monthly_proxy"]),
        }
        ops_delta = float(ops_model.predict(pd.DataFrame([{k: ops_row[k] for k in OPS_FEATURE_ORDER}]))[0])
        next_storage = np.clip(fill_prev * total_storage + base_delta + ops_delta, 0.0, total_storage)
        fill_next = float(next_storage / total_storage)
        rows.append(
            {
                "date": date,
                "pred_fill": fill_next,
                "base_delta_mcm": base_delta,
                "ops_delta_mcm": ops_delta,
                "transfer_effective_mcm": transfer_effective,
                "nrw_mcm_monthly_proxy": float(d["nrw_mcm_monthly_proxy"]),
                "reclaimed_mcm_monthly_proxy": float(d["reclaimed_mcm_monthly_proxy"]),
            }
        )
        past_fill.append(fill_next)
        past_rain.append(rain_now)
        past_et0.append(et0_now)
    return pd.DataFrame(rows)


def build_future_ops(df: pd.DataFrame, future: pd.DataFrame, cfg) -> pd.DataFrame:
    ref = df[df["transfer_share_source"].eq("official")].copy()
    if ref.empty:
        ref = df.copy()
    monthly_transfer = ref.groupby(ref["date"].dt.month)["transfer_share_pct_monthly_proxy"].mean()
    annual_transfer_anchor = float(ref.groupby(ref["date"].dt.year)["transfer_share_pct_annual_est"].mean().mean())
    month_factor = (monthly_transfer / annual_transfer_anchor).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    monthly_nrw = df.groupby(df["date"].dt.month)["nrw_pct_monthly_proxy"].mean()
    monthly_reclaimed = df.groupby(df["date"].dt.month)["reclaimed_share_pct_monthly_proxy"].mean()

    out = future.copy()
    progress = np.linspace(0.0, 1.0, len(out))
    out["supply_mcm"] = out["consumption_mean_monthly"] * out["date"].dt.days_in_month / 1e6
    annual_transfer_share = annual_transfer_anchor * (1.0 + (cfg.transfer_end_pct_2040 / 100.0) * progress)
    out["transfer_share_pct_monthly_proxy"] = out["month"].map(month_factor).astype(float) * annual_transfer_share
    out["transfer_share_pct_monthly_proxy"] = out["transfer_share_pct_monthly_proxy"].clip(5.0, 85.0)
    out["transfer_mcm_monthly_proxy"] = out["supply_mcm"] * out["transfer_share_pct_monthly_proxy"] / 100.0
    out["transfer_share_source"] = "official"

    out["nrw_pct_monthly_proxy"] = out["month"].map(monthly_nrw).astype(float) - cfg.nrw_reduction_pp_by_2040 * progress
    out["nrw_pct_monthly_proxy"] = out["nrw_pct_monthly_proxy"].clip(lower=5.0)
    out["nrw_mcm_monthly_proxy"] = out["supply_mcm"] * out["nrw_pct_monthly_proxy"] / 100.0

    out["reclaimed_share_pct_monthly_proxy"] = out["month"].map(monthly_reclaimed).astype(float)
    out["reclaimed_mcm_monthly_proxy"] = out["supply_mcm"] * out["reclaimed_share_pct_monthly_proxy"] / 100.0
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v4 = load_module(V4_SCRIPT, "wb_v4_opsresid")
    forward = load_module(FORWARD_SCRIPT, "forward_opsresid")

    context = v4.compute_system_context()
    df = load_frame(v4, context)

    grid_rows = []
    preds_by_key = {}
    for params in [
        Params(0.64, 0.72, 0.20),
        Params(0.64, 0.72, 0.35),
        Params(0.64, 0.76, 0.35),
        Params(0.68, 0.72, 0.35),
        Params(0.68, 0.76, 0.50),
        Params(0.72, 0.76, 0.50),
        Params(0.72, 0.80, 0.65),
        Params(0.76, 0.80, 0.65),
    ]:
        train = df[df["date"] <= TRAIN_END].copy()
        test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
        train_comp = component_frame(train, context, params)
        test_comp = component_frame(test, context, params)
        base_model, month_bias, ops_model = fit_models(train_comp, params)
        pred = simulate_path(train, test_comp, context, params, base_model, month_bias, ops_model)
        pred["actual_fill"] = test["weighted_total_fill"].to_numpy(dtype=float)
        pred["error_pp"] = (pred["pred_fill"] - pred["actual_fill"]) * 100.0
        row = {
            "area_exponent": params.area_exponent,
            "spill_threshold": params.spill_threshold,
            "estimated_transfer_scale": params.estimated_transfer_scale,
            "rmse_pp": float(np.sqrt(mean_squared_error(pred["actual_fill"], pred["pred_fill"])) * 100.0),
            "mae_pp": float(mean_absolute_error(pred["actual_fill"], pred["pred_fill"]) * 100.0),
            "mape_pct": float((np.abs(pred["actual_fill"] - pred["pred_fill"]) / pred["actual_fill"].clip(lower=1e-6)).mean() * 100.0),
            "pearson_pct": float(pred["actual_fill"].corr(pred["pred_fill"]) * 100.0),
            "spearman_pct": float(pred["actual_fill"].corr(pred["pred_fill"], method="spearman") * 100.0),
        }
        grid_rows.append(row)
        preds_by_key[(params.area_exponent, params.spill_threshold, params.estimated_transfer_scale)] = pred

    grid_df = pd.DataFrame(grid_rows).sort_values(["rmse_pp", "mape_pct"]).reset_index(drop=True)
    best = grid_df.iloc[0]
    best_params = Params(float(best["area_exponent"]), float(best["spill_threshold"]), float(best["estimated_transfer_scale"]))
    best_pred = preds_by_key[(best_params.area_exponent, best_params.spill_threshold, best_params.estimated_transfer_scale)]

    full_comp = component_frame(df, context, best_params)
    base_model, month_bias, ops_model = fit_models(full_comp, best_params)
    clim = forward.monthly_climatology(df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_anchor = forward.load_transfer_dependency_anchor()

    future_parts = []
    for cfg in forward.build_scenarios():
        future = forward.build_future_exog(df, cfg, clim, demand_relief, transfer_share_anchor_pct=transfer_anchor)
        future = future[(future["date"] >= pd.Timestamp("2026-01-01")) & (future["date"] <= pd.Timestamp("2040-12-01"))].copy().reset_index(drop=True)
        future = build_future_ops(df, future, cfg)
        sim = simulate_path(df, future, context, best_params, base_model, month_bias, ops_model)
        sim["scenario"] = cfg.scenario
        future_parts.append(sim)
    future_df = pd.concat(future_parts, ignore_index=True)
    future_summary = (
        future_df.groupby("scenario")
        .agg(
            mean_fill_2026_2040_pct=("pred_fill", lambda s: float(s.mean() * 100.0)),
            end_fill_2040_12_pct=("pred_fill", lambda s: float(s.iloc[-1] * 100.0)),
            min_fill_2026_2040_pct=("pred_fill", lambda s: float(s.min() * 100.0)),
        )
        .reset_index()
        .sort_values("mean_fill_2026_2040_pct", ascending=False)
    )

    grid_df.to_csv(OUT_DIR / "water_balance_v7_ops_residual_grid.csv", index=False)
    best_pred.to_csv(OUT_DIR / "water_balance_v7_ops_residual_holdout_predictions_2016_2020.csv", index=False)
    future_df.to_csv(OUT_DIR / "water_balance_v7_ops_residual_future_2026_2040.csv", index=False)
    future_summary.to_csv(OUT_DIR / "water_balance_v7_ops_residual_future_summary_2026_2040.csv", index=False)
    summary = {
        "best_area_exponent": best_params.area_exponent,
        "best_spill_threshold": best_params.spill_threshold,
        "best_estimated_transfer_scale": best_params.estimated_transfer_scale,
        "holdout_rmse_pp": float(best["rmse_pp"]),
        "holdout_mape_pct": float(best["mape_pct"]),
        "holdout_pearson_pct": float(best["pearson_pct"]),
        "future_base_2040_12_pct": float(
            future_summary.loc[future_summary["scenario"] == "base", "end_fill_2040_12_pct"].iloc[0]
        ),
    }
    (OUT_DIR / "water_balance_v7_ops_residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 5))
    plt.plot(best_pred["date"], best_pred["actual_fill"] * 100.0, color="black", lw=2.0, label="Gerçek")
    plt.plot(best_pred["date"], best_pred["pred_fill"] * 100.0, color="#2563eb", lw=2.0, label="Tahmin")
    plt.title("Water Balance V7 Ops Residual - 2016-2020 holdout")
    plt.ylabel("Doluluk (%)")
    plt.xlabel("Tarih")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "water_balance_v7_ops_residual_holdout.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    for scenario, g in future_df.groupby("scenario"):
        plt.plot(g["date"], g["pred_fill"] * 100.0, lw=2.0, label=scenario)
    plt.title("Water Balance V7 Ops Residual - 2026-2040 senaryolar")
    plt.ylabel("Doluluk (%)")
    plt.xlabel("Tarih")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "water_balance_v7_ops_residual_future.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
