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
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/yasinkaya/Hackhaton")
V4_SCRIPT = ROOT / "scripts" / "build_istanbul_water_balance_v4_sourceaware.py"
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
FEATURE_BLOCKS_SCRIPT = ROOT / "scripts" / "baraj_feature_blocks.py"
OUT_DIR = ROOT / "output" / "istanbul_water_balance_v6_ops_curve_2040"

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
    "transfer_mcm_monthly_proxy",
]
CORR_FEATURES = [
    "weighted_total_fill_lag1",
    "nrw_pct_monthly_proxy",
    "reclaimed_share_pct_monthly_proxy",
    "transfer_share_pct_monthly_proxy",
    "official_supply_m3_month_roll3",
    "month_sin",
    "month_cos",
]


@dataclass(frozen=True)
class CurveParams:
    area_exponent: float
    spill_threshold: float
    corr_clip_quantile: float


def month_sin(month: int) -> float:
    return float(np.sin(2.0 * np.pi * month / 12.0))


def month_cos(month: int) -> float:
    return float(np.cos(2.0 * np.pi * month / 12.0))


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


def source_state_tuned(fill: float, context: dict[str, object], params: CurveParams) -> dict[str, np.ndarray]:
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


def build_monthly_proxy_climatology(exog_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    cols = [
        "src_rain_north",
        "src_rain_west",
        "transfer_share_pct_monthly_proxy",
        "nrw_pct_monthly_proxy",
        "reclaimed_share_pct_monthly_proxy",
        "official_supply_m3_month_roll3",
        "reanalysis_rs_mj_m2_month",
        "reanalysis_wind_speed_10m_max_m_s",
    ]
    out: dict[str, dict[int, float]] = {}
    for col in cols:
        out[col] = exog_df.groupby(exog_df["date"].dt.month)[col].mean().to_dict()
    return out


def openwater_evap_factor(rs: float, wind: float, rs_ref: float, wind_ref: float) -> float:
    factor = 1.0 + 0.00045 * (rs - rs_ref) + 0.020 * (wind - wind_ref)
    return float(np.clip(factor, 0.85, 1.25))


def load_training_frame(v4, feature_blocks, context: dict[str, object]) -> tuple[pd.DataFrame, dict[str, float]]:
    df = v4.load_training_frame(context).copy()
    exog = feature_blocks.load_monthly_exog_table().copy()
    keep = [
        "date",
        "src_rain_north",
        "src_rain_west",
        "reanalysis_rs_mj_m2_month",
        "reanalysis_wind_speed_10m_max_m_s",
        "transfer_share_pct_monthly_proxy",
        "transfer_mcm_monthly_proxy",
        "nrw_pct_monthly_proxy",
        "reclaimed_share_pct_monthly_proxy",
        "official_supply_m3_month_roll3",
    ]
    df = df.merge(exog[keep], on="date", how="left")
    for col in keep:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill()

    df["src_rain_proxy_mm"] = 0.7 * df["rain_model_mm"] + 0.3 * (0.6 * df["src_rain_north"] + 0.4 * df["src_rain_west"])
    df["src_rain_proxy_lag1_mm"] = df["src_rain_proxy_mm"].shift(1)
    df["src_rain_proxy_roll3_mm"] = df["src_rain_proxy_mm"].rolling(3).mean()
    df = df.dropna(subset=["src_rain_proxy_lag1_mm", "src_rain_proxy_roll3_mm"]).reset_index(drop=True)

    refs = {
        "rs_ref": float(df["reanalysis_rs_mj_m2_month"].median()),
        "wind_ref": float(df["reanalysis_wind_speed_10m_max_m_s"].median()),
    }
    return df, refs


def component_frame(df: pd.DataFrame, context: dict[str, object], params: CurveParams, refs: dict[str, float]) -> pd.DataFrame:
    out = df[
        [
            "date",
            "weighted_total_fill",
            "weighted_total_fill_lag1",
            "src_rain_proxy_mm",
            "src_rain_proxy_lag1_mm",
            "src_rain_proxy_roll3_mm",
            "et0_mm_month",
            "et0_mm_month_roll3",
            "supply_mcm",
            "delta_storage_mcm",
            "transfer_share_pct_monthly_proxy",
            "transfer_mcm_monthly_proxy",
            "nrw_pct_monthly_proxy",
            "reclaimed_share_pct_monthly_proxy",
            "official_supply_m3_month_roll3",
            "reanalysis_rs_mj_m2_month",
            "reanalysis_wind_speed_10m_max_m_s",
        ]
    ].copy()
    state_rows = [source_state_tuned(float(fill), context, params) for fill in out["weighted_total_fill_lag1"]]
    out["lake_area_km2"] = [float(s["lake_area_km2"].sum()) for s in state_rows]
    out["land_area_km2"] = [float(s["land_area_km2"].sum()) for s in state_rows]
    out["runoff_weighted_land_area_km2"] = [
        float(np.dot(s["land_area_km2"], s["runoff_productivity_weight"])) for s in state_rows
    ]
    wetness_mm = np.clip(out["src_rain_proxy_roll3_mm"] - out["et0_mm_month_roll3"], 0.0, None)
    out["openwater_evap_factor"] = [
        openwater_evap_factor(rs, wind, refs["rs_ref"], refs["wind_ref"])
        for rs, wind in zip(out["reanalysis_rs_mj_m2_month"], out["reanalysis_wind_speed_10m_max_m_s"])
    ]
    out["source_runoff_now_mcm"] = out["runoff_weighted_land_area_km2"] * out["src_rain_proxy_mm"] * 0.001
    out["source_runoff_lag1_mcm"] = out["runoff_weighted_land_area_km2"] * out["src_rain_proxy_lag1_mm"] * 0.001
    out["source_runoff_wetness_mcm"] = out["runoff_weighted_land_area_km2"] * wetness_mm * 0.001
    out["source_lake_rain_mcm"] = out["lake_area_km2"] * out["src_rain_proxy_mm"] * 0.001
    out["source_openwater_evap_mcm"] = out["lake_area_km2"] * out["et0_mm_month"] * out["openwater_evap_factor"] * 0.001
    out["storage_mass_mcm"] = [float(s["storage_mcm"].sum()) for s in state_rows]
    out["spill_pressure_mcm"] = [float(s["spill_pressure_mcm"].sum()) for s in state_rows]
    out["neg_source_openwater_evap_mcm"] = -out["source_openwater_evap_mcm"]
    out["neg_supply_mcm"] = -out["supply_mcm"]
    out["neg_storage_mass_mcm"] = -out["storage_mass_mcm"]
    out["neg_spill_pressure_mcm"] = -out["spill_pressure_mcm"]
    out["month_sin"] = out["date"].dt.month.map(month_sin)
    out["month_cos"] = out["date"].dt.month.map(month_cos)
    return out


def fit_model(train_comp: pd.DataFrame, params: CurveParams):
    model = LinearRegression(positive=True)
    model.fit(train_comp[BASE_FEATURE_ORDER], train_comp["delta_storage_mcm"])
    tmp = train_comp[["date", "delta_storage_mcm"]].copy()
    tmp["base_pred_mcm"] = model.predict(train_comp[BASE_FEATURE_ORDER])
    tmp["month"] = tmp["date"].dt.month
    tmp["residual_mcm"] = tmp["delta_storage_mcm"] - tmp["base_pred_mcm"]
    month_bias = tmp.groupby("month")["residual_mcm"].mean().to_dict()
    tmp["pred_mcm"] = tmp["base_pred_mcm"] + tmp["month"].map(month_bias).fillna(0.0)

    corr_train = train_comp.copy()
    corr_train["base_pred_mcm"] = tmp["pred_mcm"].to_numpy(dtype=float)
    corr_train["residual_target_mcm"] = corr_train["delta_storage_mcm"] - corr_train["base_pred_mcm"]
    corr = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-4, 4, 41)))])
    corr.fit(corr_train[CORR_FEATURES], corr_train["residual_target_mcm"])
    clip_abs = float(np.quantile(np.abs(corr_train["residual_target_mcm"]), params.corr_clip_quantile))
    return model, month_bias, corr, clip_abs


def simulate_path(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    context: dict[str, object],
    params: CurveParams,
    refs: dict[str, float],
    model,
    month_bias: dict[int, float],
    corr,
    corr_clip_abs: float,
) -> pd.DataFrame:
    past_fill = history_df["weighted_total_fill"].tolist()
    past_src_rain = history_df["src_rain_proxy_mm"].tolist()
    past_et0 = history_df["et0_mm_month"].tolist()
    total_storage = float(context["total_storage_mcm"])
    rows: list[dict[str, float | str]] = []
    for row in future_df.reset_index(drop=True).itertuples(index=False):
        d = row._asdict()
        date = pd.Timestamp(d["date"])
        fill_prev = float(past_fill[-1])
        state = source_state_tuned(fill_prev, context, params)
        lake_area_km2 = float(state["lake_area_km2"].sum())
        runoff_area = float(np.dot(state["land_area_km2"], state["runoff_productivity_weight"]))
        rain_now = float(d["src_rain_proxy_mm"])
        rain_lag1 = float(past_src_rain[-1])
        et0_now = float(d["et0_mm_month"])
        rain_roll3 = float(np.mean([past_src_rain[-2], past_src_rain[-1], rain_now]))
        et0_roll3 = float(np.mean([past_et0[-2], past_et0[-1], et0_now]))
        evap_factor = openwater_evap_factor(
            float(d["reanalysis_rs_mj_m2_month"]),
            float(d["reanalysis_wind_speed_10m_max_m_s"]),
            refs["rs_ref"],
            refs["wind_ref"],
        )
        supply_mcm = float(d["supply_mcm"])
        transfer_mcm = float(d["transfer_mcm_monthly_proxy"])

        comp = {
            "source_runoff_now_mcm": runoff_area * rain_now * 0.001,
            "source_runoff_lag1_mcm": runoff_area * rain_lag1 * 0.001,
            "source_runoff_wetness_mcm": runoff_area * max(rain_roll3 - et0_roll3, 0.0) * 0.001,
            "source_lake_rain_mcm": lake_area_km2 * rain_now * 0.001,
            "source_openwater_evap_mcm": lake_area_km2 * et0_now * evap_factor * 0.001,
            "supply_mcm": supply_mcm,
            "storage_mass_mcm": float(state["storage_mcm"].sum()),
            "spill_pressure_mcm": float(state["spill_pressure_mcm"].sum()),
            "transfer_mcm_monthly_proxy": transfer_mcm,
        }
        comp["neg_source_openwater_evap_mcm"] = -comp["source_openwater_evap_mcm"]
        comp["neg_supply_mcm"] = -comp["supply_mcm"]
        comp["neg_storage_mass_mcm"] = -comp["storage_mass_mcm"]
        comp["neg_spill_pressure_mcm"] = -comp["spill_pressure_mcm"]

        base_delta = float(model.predict(pd.DataFrame([{k: comp[k] for k in BASE_FEATURE_ORDER}]))[0])
        base_delta += float(month_bias.get(date.month, 0.0))
        corr_row = {
            "weighted_total_fill_lag1": fill_prev,
            "nrw_pct_monthly_proxy": float(d["nrw_pct_monthly_proxy"]),
            "reclaimed_share_pct_monthly_proxy": float(d["reclaimed_share_pct_monthly_proxy"]),
            "transfer_share_pct_monthly_proxy": float(d["transfer_share_pct_monthly_proxy"]),
            "official_supply_m3_month_roll3": float(d["official_supply_m3_month_roll3"]),
            "month_sin": month_sin(date.month),
            "month_cos": month_cos(date.month),
        }
        corr_delta = float(corr.predict(pd.DataFrame([{c: corr_row[c] for c in CORR_FEATURES}]))[0])
        corr_delta = float(np.clip(corr_delta, -corr_clip_abs, corr_clip_abs))
        next_storage = np.clip(fill_prev * total_storage + base_delta + corr_delta, 0.0, total_storage)
        fill_next = float(next_storage / total_storage)
        rows.append(
            {
                "date": date,
                "pred_fill": fill_next,
                "base_delta_mcm": float(base_delta),
                "corr_delta_mcm": float(corr_delta),
                "transfer_mcm_monthly_proxy": transfer_mcm,
                "nrw_pct_monthly_proxy": float(d["nrw_pct_monthly_proxy"]),
            }
        )
        past_fill.append(fill_next)
        past_src_rain.append(rain_now)
        past_et0.append(et0_now)
    return pd.DataFrame(rows)


def prepare_future_frame(forward, feature_blocks, history_df: pd.DataFrame, exog_df: pd.DataFrame, cfg) -> pd.DataFrame:
    clim = forward.monthly_climatology(
        history_df[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly", "temp_proxy_c", "rh_proxy_pct", "vpd_kpa_mean"]].copy()
    )
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    future = forward.build_future_exog(history_df, cfg, clim, demand_relief, transfer_share_anchor_pct=transfer_share_anchor_pct)
    future = future[(future["date"] >= pd.Timestamp("2026-01-01")) & (future["date"] <= pd.Timestamp("2040-12-01"))].copy().reset_index(drop=True)
    clim_ops = build_monthly_proxy_climatology(exog_df)
    proxy_models = feature_blocks.fit_future_proxy_models(exog_df)
    future["month"] = future["date"].dt.month
    future["month_sin"] = future["month"].map(month_sin)
    future["month_cos"] = future["month"].map(month_cos)
    future["nao_index"] = 0.0
    future = feature_blocks.add_future_exog_proxies(future, proxy_models)
    future["src_rain_proxy_mm"] = 0.7 * future["rain_model_mm"] + 0.3 * (0.6 * future["src_rain_north"] + 0.4 * future["src_rain_west"])
    future["src_rain_proxy_lag1_mm"] = future["src_rain_proxy_mm"].shift(1).fillna(future["src_rain_proxy_mm"])
    future["src_rain_proxy_roll3_mm"] = future["src_rain_proxy_mm"].rolling(3, min_periods=1).mean()
    future["reanalysis_rs_mj_m2_month"] = future["month"].map(clim_ops["reanalysis_rs_mj_m2_month"]).astype(float)
    future["reanalysis_wind_speed_10m_max_m_s"] = future["month"].map(clim_ops["reanalysis_wind_speed_10m_max_m_s"]).astype(float)
    progress = np.linspace(0.0, 1.0, len(future))
    future["transfer_share_pct_monthly_proxy"] = future["month"].map(clim_ops["transfer_share_pct_monthly_proxy"]).astype(float)
    future["transfer_share_pct_monthly_proxy"] *= 1.0 + (cfg.transfer_end_pct_2040 / 100.0) * progress
    future["nrw_pct_monthly_proxy"] = future["month"].map(clim_ops["nrw_pct_monthly_proxy"]).astype(float) - cfg.nrw_reduction_pp_by_2040 * progress
    future["nrw_pct_monthly_proxy"] = future["nrw_pct_monthly_proxy"].clip(lower=5.0)
    future["reclaimed_share_pct_monthly_proxy"] = future["month"].map(clim_ops["reclaimed_share_pct_monthly_proxy"]).astype(float)
    future["supply_mcm"] = future["consumption_mean_monthly"] * future["date"].dt.days_in_month / 1e6
    future["transfer_mcm_monthly_proxy"] = future["supply_mcm"] * future["transfer_share_pct_monthly_proxy"] / 100.0
    future["official_supply_m3_month_roll3"] = (future["supply_mcm"] * 1e6).rolling(3, min_periods=1).mean()
    return future


def evaluate_holdout(df: pd.DataFrame, context: dict[str, object], params: CurveParams, refs: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= TRAIN_END].copy()
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
    train_comp = component_frame(train, context, params, refs)
    test_comp = component_frame(test, context, params, refs)
    model, month_bias, corr, corr_clip_abs = fit_model(train_comp, params)
    pred = simulate_path(train, test_comp, context, params, refs, model, month_bias, corr, corr_clip_abs)
    pred["actual_fill"] = test["weighted_total_fill"].to_numpy(dtype=float)
    pred["error_pp"] = (pred["pred_fill"] - pred["actual_fill"]) * 100.0
    metrics = pd.DataFrame(
        [
            {
                "model": "water_balance_v6_ops_curve",
                "area_exponent": params.area_exponent,
                "spill_threshold": params.spill_threshold,
                "corr_clip_quantile": params.corr_clip_quantile,
                "rmse_pp": float(np.sqrt(mean_squared_error(pred["actual_fill"], pred["pred_fill"])) * 100.0),
                "mae_pp": float(mean_absolute_error(pred["actual_fill"], pred["pred_fill"]) * 100.0),
                "mape_pct": float((np.abs(pred["actual_fill"] - pred["pred_fill"]) / pred["actual_fill"].clip(lower=1e-6)).mean() * 100.0),
                "pearson_pct": float(pred["actual_fill"].corr(pred["pred_fill"]) * 100.0),
                "spearman_pct": float(pred["actual_fill"].corr(pred["pred_fill"], method="spearman") * 100.0),
                "n_predictions": int(len(pred)),
            }
        ]
    )
    return metrics, pred


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v4 = load_module(V4_SCRIPT, "istanbul_wb_v4_opscurve")
    forward = load_module(FORWARD_SCRIPT, "istanbul_forward_opscurve")
    feature_blocks = load_module(FEATURE_BLOCKS_SCRIPT, "istanbul_feature_blocks_opscurve")

    context = v4.compute_system_context()
    df, refs = load_training_frame(v4, feature_blocks, context)
    exog_df = feature_blocks.load_monthly_exog_table()

    grid = []
    predictions = {}
    for params in [
        CurveParams(0.64, 0.72, 0.90),
        CurveParams(0.64, 0.76, 0.90),
        CurveParams(0.68, 0.72, 0.90),
        CurveParams(0.68, 0.76, 0.90),
        CurveParams(0.72, 0.72, 0.95),
        CurveParams(0.72, 0.76, 0.95),
        CurveParams(0.76, 0.76, 0.95),
        CurveParams(0.76, 0.80, 0.95),
    ]:
        metrics, pred = evaluate_holdout(df, context, params, refs)
        grid.append(metrics.iloc[0].to_dict())
        predictions[(params.area_exponent, params.spill_threshold, params.corr_clip_quantile)] = pred
    grid_df = pd.DataFrame(grid).sort_values(["rmse_pp", "mape_pct"]).reset_index(drop=True)
    best = grid_df.iloc[0]
    best_params = CurveParams(float(best["area_exponent"]), float(best["spill_threshold"]), float(best["corr_clip_quantile"]))
    best_pred = predictions[(best_params.area_exponent, best_params.spill_threshold, best_params.corr_clip_quantile)]

    train_comp = component_frame(df, context, best_params, refs)
    model, month_bias, corr, corr_clip_abs = fit_model(train_comp, best_params)
    future_parts = []
    for cfg in forward.build_scenarios():
        future = prepare_future_frame(forward, feature_blocks, df, exog_df, cfg)
        sim = simulate_path(df, future, context, best_params, refs, model, month_bias, corr, corr_clip_abs)
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

    grid_df.to_csv(OUT_DIR / "water_balance_v6_ops_curve_grid.csv", index=False)
    best_pred.to_csv(OUT_DIR / "water_balance_v6_ops_curve_holdout_predictions_2016_2020.csv", index=False)
    future_df.to_csv(OUT_DIR / "water_balance_v6_ops_curve_future_2026_2040.csv", index=False)
    future_summary.to_csv(OUT_DIR / "water_balance_v6_ops_curve_future_summary_2026_2040.csv", index=False)

    summary = {
        "best_area_exponent": best_params.area_exponent,
        "best_spill_threshold": best_params.spill_threshold,
        "best_corr_clip_quantile": best_params.corr_clip_quantile,
        "holdout_rmse_pp": float(best["rmse_pp"]),
        "holdout_mape_pct": float(best["mape_pct"]),
        "holdout_pearson_pct": float(best["pearson_pct"]),
        "future_base_2040_12_pct": float(
            future_summary.loc[future_summary["scenario"] == "base", "end_fill_2040_12_pct"].iloc[0]
        ),
    }
    (OUT_DIR / "water_balance_v6_ops_curve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 5))
    plt.plot(best_pred["date"], best_pred["actual_fill"] * 100.0, color="black", lw=2.0, label="Gerçek")
    plt.plot(best_pred["date"], best_pred["pred_fill"] * 100.0, color="#2563eb", lw=2.0, label="Tahmin")
    plt.title("Water Balance V6 Ops+Curve - 2016-2020 holdout")
    plt.ylabel("Doluluk (%)")
    plt.xlabel("Tarih")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "water_balance_v6_ops_curve_holdout.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    for scenario, g in future_df.groupby("scenario"):
        plt.plot(g["date"], g["pred_fill"] * 100.0, lw=2.0, label=scenario)
    plt.title("Water Balance V6 Ops+Curve - 2026-2040 senaryolar")
    plt.ylabel("Doluluk (%)")
    plt.xlabel("Tarih")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "water_balance_v6_ops_curve_future.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
