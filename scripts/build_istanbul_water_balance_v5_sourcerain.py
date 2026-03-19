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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("/Users/yasinkaya/Hackhaton")
CORE_PATH = ROOT / "output" / "model_useful_data_bundle" / "tables" / "istanbul_model_core_monthly.csv"
SUPPLY_PATH = ROOT / "output" / "newdata_feature_store" / "tables" / "official_city_supply_monthly_2010_2023.csv"
SOURCE_CONTEXT_PATH = ROOT / "output" / "newdata_feature_store" / "tables" / "official_iski_source_context.csv"
SOURCE_CURRENT_PATH = ROOT / "output" / "model_useful_data_bundle" / "tables" / "istanbul_source_current_context.csv"
REFERENCE_PATH = ROOT / "output" / "reference_tables" / "istanbul_baraj_reference_table.csv"
SOURCE_RAIN_PATH = ROOT / "output" / "source_precip_proxies" / "source_precip_monthly_wide_2000_2026.csv"
BENCHMARK_SCORECARD_PATH = ROOT / "output" / "istanbul_forward_model_benchmark_round2" / "model_selection_scorecard.csv"
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
OUT_DIR = ROOT / "output" / "istanbul_water_balance_v5_sourcerain_2040"

SPILL_THRESHOLD = 0.75
AREA_EXPONENT = 0.72
MIN_TRAIN = 60
PRIMARY_SCENARIOS = ["base", "wet_mild", "hot_dry_high_demand", "management_improvement"]
SCENARIO_COLORS = {
    "base": "#2563eb",
    "wet_mild": "#059669",
    "hot_dry_high_demand": "#dc2626",
    "management_improvement": "#d97706",
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
FEATURE_ORDER = [
    "source_runoff_now_mcm",
    "source_runoff_lag1_mcm",
    "source_runoff_wetness_mcm",
    "source_lake_rain_mcm",
    "neg_source_openwater_evap_mcm",
    "neg_supply_mcm",
    "neg_storage_mass_mcm",
    "neg_spill_pressure_mcm",
]


def load_forward_module():
    spec = importlib.util.spec_from_file_location("istanbul_forward_base_for_wb_v1", FORWARD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Forward projection script import failed.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_transfer_share_by_year() -> tuple[dict[int, float], float]:
    forward = load_forward_module()
    transfer_df, anchor_share_pct = forward.load_transfer_dependency_anchor()
    share_by_year = {int(pd.Timestamp(d).year): float(v) for d, v in zip(transfer_df["tarih"], transfer_df["transfer_share_pct"])}
    return share_by_year, float(anchor_share_pct)


def canonicalize_name(text: str) -> str:
    clean = (
        str(text)
        .replace("Ö", "O")
        .replace("ö", "o")
        .replace("Ü", "U")
        .replace("ü", "u")
        .replace("Ç", "C")
        .replace("ç", "c")
        .replace("Ş", "S")
        .replace("ş", "s")
        .replace("İ", "I")
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("Ğ", "G")
    )
    clean = clean.lower()
    for token in [" baraji", " baraji ", " barajlari", " regulatori", " ve ", "  "]:
        clean = clean.replace(token, " ")
    return "".join(ch for ch in clean if ch.isalnum())


RAIN_CODE_MAP = {
    "omerli": "Omerli",
    "darlik": "Darlik",
    "elmali12": "Elmali",
    "elmali": "Elmali",
    "terkos": "Terkos",
    "alibey": "Alibey",
    "buyukcekmece": "Buyukcekmece",
    "sazlidere": "Sazlidere",
    "kazandere": "Kazandere",
    "pabucdere": "Pabucdere",
    "istrancalar": "Istrancalar",
}


def compute_system_context() -> dict[str, object]:
    src = pd.read_csv(SOURCE_CONTEXT_PATH)
    cur = pd.read_csv(SOURCE_CURRENT_PATH)
    ref = pd.read_csv(REFERENCE_PATH)

    src["key"] = src["source_name"].map(canonicalize_name)
    cur["key"] = cur["source_name"].map(canonicalize_name)
    ref["key"] = ref["reservoir"].map(canonicalize_name)

    merged = src.merge(
        cur[
            [
                "key",
                "source_name",
                "biriktirmeHacmi",
                "dolulukOrani",
                "mevcutSuHacmi",
            ]
        ],
        on="key",
        how="outer",
        suffixes=("_src", "_cur"),
    ).merge(
        ref[["key", "catchment_area_km2", "active_capacity_hm3"]],
        on="key",
        how="left",
    )
    merged["source_name"] = merged["source_name_cur"].fillna(merged["source_name_src"])
    merged["max_storage_million_m3"] = merged["max_storage_million_m3"].fillna(merged["biriktirmeHacmi"]).fillna(merged["active_capacity_hm3"])
    merged["basin_area_km2"] = merged["basin_area_km2"].fillna(merged["catchment_area_km2"])
    merged["annual_yield_million_m3"] = merged["annual_yield_million_m3"].fillna(0.0)
    merged["source_group"] = merged["source_group"].fillna("baraj")
    merged["fill_now"] = merged["dolulukOrani"] / 100.0

    # Keep stored sources and the Istranca aggregate, exclude transfer regulators.
    keep = merged["max_storage_million_m3"].fillna(0.0) > 0.0
    keep &= ~merged["source_group"].isin(["regulator"])
    sources = merged.loc[keep].copy()

    known = sources[sources["normal_lake_area_km2"].notna()].iloc[0]
    area_coeff = float(known["normal_lake_area_km2"] / (known["max_storage_million_m3"] ** (2.0 / 3.0)))
    sources["est_lake_area_km2"] = sources["normal_lake_area_km2"]
    miss = sources["est_lake_area_km2"].isna()
    sources.loc[miss, "est_lake_area_km2"] = area_coeff * (sources.loc[miss, "max_storage_million_m3"] ** (2.0 / 3.0))

    current_total_fill = float(sources["mevcutSuHacmi"].sum() / sources["max_storage_million_m3"].sum())
    sources["fill_factor_prior"] = (sources["fill_now"] / current_total_fill).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    sources["fill_factor_prior"] = sources["fill_factor_prior"].clip(0.55, 1.80)
    sources["rain_code"] = sources["source_name"].map(lambda x: RAIN_CODE_MAP.get(canonicalize_name(x), None))

    ypb = sources["yield_per_basin_mm"].copy()
    fallback = float(ypb.dropna().median())
    ypb = ypb.fillna(fallback)
    sources["runoff_productivity_weight"] = (ypb / fallback).clip(0.60, 1.45)
    sources["est_land_area_full_km2"] = np.maximum(sources["basin_area_km2"].fillna(0.0) - sources["est_lake_area_km2"], 0.0)

    total_storage_mcm = float(sources["max_storage_million_m3"].sum())
    total_lake_area_km2 = float(sources["est_lake_area_km2"].sum())
    total_basin_area_km2 = float(sources["basin_area_km2"].fillna(0.0).sum())
    return {
        "total_storage_mcm": total_storage_mcm,
        "total_lake_area_km2": total_lake_area_km2,
        "total_basin_area_km2": total_basin_area_km2,
        "land_basin_area_km2": max(total_basin_area_km2 - total_lake_area_km2, 0.0),
        "lake_area_coeff": area_coeff,
        "current_total_fill": current_total_fill,
        "n_sources": int(len(sources)),
        "source_table": sources[
            [
                "source_name",
                "source_group",
                "max_storage_million_m3",
                "basin_area_km2",
                "est_lake_area_km2",
                "annual_yield_million_m3",
                "runoff_productivity_weight",
                "fill_factor_prior",
                "rain_code",
            ]
        ].reset_index(drop=True),
    }


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


def source_state(fill: float, context: dict[str, object]) -> dict[str, np.ndarray]:
    src = context["source_table"]
    fills = allocate_source_fills(fill, context)
    full_lake_area = src["est_lake_area_km2"].to_numpy(dtype=float)
    lake_area = full_lake_area * np.power(np.clip(fills, 0.0, 1.0), AREA_EXPONENT)
    basin_area = src["basin_area_km2"].fillna(0.0).to_numpy(dtype=float)
    land_area = np.maximum(basin_area - lake_area, 0.0)
    storage = fills * src["max_storage_million_m3"].to_numpy(dtype=float)
    spill_pressure = np.maximum(fills - SPILL_THRESHOLD, 0.0) * src["max_storage_million_m3"].to_numpy(dtype=float)
    productivity = src["runoff_productivity_weight"].to_numpy(dtype=float)
    return {
        "fills": fills,
        "lake_area_km2": lake_area,
        "land_area_km2": land_area,
        "storage_mcm": storage,
        "spill_pressure_mcm": spill_pressure,
        "runoff_productivity_weight": productivity,
    }


def load_training_frame(context: dict[str, float]) -> pd.DataFrame:
    core = pd.read_csv(CORE_PATH, parse_dates=["date"])
    supply = pd.read_csv(SUPPLY_PATH, parse_dates=["date"])
    source_rain = pd.read_csv(SOURCE_RAIN_PATH, parse_dates=["date"])
    df = core.merge(supply[["date", "city_supply_m3_month_official"]], on="date", how="left")
    df = df.merge(source_rain, on="date", how="left")
    df["days_in_month"] = df["date"].dt.days_in_month
    df["supply_mcm"] = df["city_supply_m3_month_official"] / 1e6
    fallback = df["consumption_mean_monthly"] * df["days_in_month"] / 1e6
    df["supply_mcm"] = df["supply_mcm"].fillna(fallback)
    rain_codes = [c for c in context["source_table"]["rain_code"].dropna().unique().tolist() if c in df.columns]
    for code in rain_codes:
        df[code] = pd.to_numeric(df[code], errors="coerce")
        df[code] = df[code].fillna(df["rain_model_mm"])
        df[f"{code}_lag1"] = df[code].shift(1)
        df[f"{code}_roll3"] = df[code].rolling(3).mean()
    df["delta_fill"] = df["weighted_total_fill"] - df["weighted_total_fill_lag1"]
    df["delta_storage_mcm"] = df["delta_fill"] * context["total_storage_mcm"]
    df = df.dropna(
        subset=[
            "weighted_total_fill",
            "weighted_total_fill_lag1",
            "weighted_total_fill_lag2",
            "rain_model_mm",
            "rain_model_mm_lag1",
            "rain_model_mm_roll3",
            "et0_mm_month",
            "et0_mm_month_roll3",
            "supply_mcm",
            "delta_storage_mcm",
        ]
    ).reset_index(drop=True)
    extra_required = []
    for code in rain_codes:
        extra_required.extend([code, f"{code}_lag1", f"{code}_roll3"])
    if extra_required:
        df = df.dropna(subset=extra_required).reset_index(drop=True)
    df = df[df["date"] >= "2011-01-01"].reset_index(drop=True)
    return df


def component_frame(df: pd.DataFrame, context: dict[str, float]) -> pd.DataFrame:
    keep_cols = [
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
    ]
    rain_codes = [c for c in context["source_table"]["rain_code"].dropna().unique().tolist() if c in df.columns]
    for code in rain_codes:
        keep_cols.extend([code, f"{code}_lag1", f"{code}_roll3"])
    out = df[keep_cols].copy()
    state_rows = [source_state(float(fill), context) for fill in out["weighted_total_fill_lag1"]]
    out["lake_area_km2"] = [float(s["lake_area_km2"].sum()) for s in state_rows]
    out["land_area_km2"] = [float(s["land_area_km2"].sum()) for s in state_rows]
    src_table = context["source_table"].reset_index(drop=True)
    runoff_now_vals = []
    runoff_lag_vals = []
    runoff_wet_vals = []
    lake_rain_vals = []
    for row_idx, (_, row) in enumerate(out.iterrows()):
        state = state_rows[row_idx]
        runoff_now = 0.0
        runoff_lag = 0.0
        runoff_wet = 0.0
        lake_rain = 0.0
        for j, src_row in src_table.iterrows():
            code = src_row["rain_code"]
            rain_now = float(row[code]) if code in out.columns else float(row["rain_model_mm"])
            rain_lag = float(row[f"{code}_lag1"]) if f"{code}_lag1" in out.columns else float(row["rain_model_mm_lag1"])
            rain_roll3 = float(row[f"{code}_roll3"]) if f"{code}_roll3" in out.columns else float(row["rain_model_mm_roll3"])
            wet_mm = max(rain_roll3 - float(row["et0_mm_month_roll3"]), 0.0)
            runoff_now += float(state["land_area_km2"][j] * state["runoff_productivity_weight"][j] * rain_now * 0.001)
            runoff_lag += float(state["land_area_km2"][j] * state["runoff_productivity_weight"][j] * rain_lag * 0.001)
            runoff_wet += float(state["land_area_km2"][j] * state["runoff_productivity_weight"][j] * wet_mm * 0.001)
            lake_rain += float(state["lake_area_km2"][j] * rain_now * 0.001)
        runoff_now_vals.append(runoff_now)
        runoff_lag_vals.append(runoff_lag)
        runoff_wet_vals.append(runoff_wet)
        lake_rain_vals.append(lake_rain)
    out["source_runoff_now_mcm"] = runoff_now_vals
    out["source_runoff_lag1_mcm"] = runoff_lag_vals
    out["source_runoff_wetness_mcm"] = runoff_wet_vals
    out["source_lake_rain_mcm"] = lake_rain_vals
    out["source_openwater_evap_mcm"] = out["lake_area_km2"] * out["et0_mm_month"] * 0.001
    out["storage_mass_mcm"] = [float(s["storage_mcm"].sum()) for s in state_rows]
    out["spill_pressure_mcm"] = [float(s["spill_pressure_mcm"].sum()) for s in state_rows]
    out["neg_source_openwater_evap_mcm"] = -out["source_openwater_evap_mcm"]
    out["neg_supply_mcm"] = -out["supply_mcm"]
    out["neg_storage_mass_mcm"] = -out["storage_mass_mcm"]
    out["neg_spill_pressure_mcm"] = -out["spill_pressure_mcm"]
    return out


def fit_water_balance_model(train_comp: pd.DataFrame) -> tuple[LinearRegression, dict[int, float], pd.DataFrame]:
    model = LinearRegression(positive=True)
    model.fit(train_comp[FEATURE_ORDER], train_comp["delta_storage_mcm"])
    tmp = train_comp[["date", "delta_storage_mcm"]].copy()
    tmp["base_pred_mcm"] = model.predict(train_comp[FEATURE_ORDER])
    tmp["month"] = tmp["date"].dt.month
    tmp["residual_mcm"] = tmp["delta_storage_mcm"] - tmp["base_pred_mcm"]
    month_bias = tmp.groupby("month")["residual_mcm"].mean().to_dict()
    tmp["pred_mcm"] = tmp["base_pred_mcm"] + tmp["month"].map(month_bias).fillna(0.0)
    return model, month_bias, tmp


def source_rain_monthly_climatology(df: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    rain_codes = [c for c in context["source_table"]["rain_code"].dropna().unique().tolist() if c in df.columns]
    cols = ["date"] + rain_codes + ["rain_model_mm"]
    tmp = df[cols].copy()
    tmp["month"] = tmp["date"].dt.month
    clim = tmp.groupby("month")[rain_codes + ["rain_model_mm"]].mean().reset_index()
    return clim


def attach_future_source_rain(future: pd.DataFrame, rain_clim: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    rain_codes = [c for c in context["source_table"]["rain_code"].dropna().unique().tolist() if c in rain_clim.columns]
    tmp = future.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
    tmp = tmp.merge(rain_clim, on="month", how="left", suffixes=("", "_clim"))
    denom = tmp["rain_model_mm_clim"].replace(0.0, np.nan)
    scale = (tmp["rain_model_mm"] / denom).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    for code in rain_codes:
        tmp[code] = tmp[code] * scale
    keep = list(future.columns) + rain_codes
    return tmp[keep]


def future_input_columns(context: dict[str, object]) -> list[str]:
    rain_codes = [c for c in context["source_table"]["rain_code"].dropna().unique().tolist()]
    return ["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"] + rain_codes


def estimate_transfer_effectiveness(train_df: pd.DataFrame, fit_df: pd.DataFrame, share_by_year: dict[int, float]) -> float:
    tmp = train_df[["date", "supply_mcm"]].copy()
    tmp["pred_mcm"] = fit_df["pred_mcm"].to_numpy(dtype=float)
    tmp["actual_delta_mcm"] = fit_df["delta_storage_mcm"].to_numpy(dtype=float)
    tmp["year"] = tmp["date"].dt.year
    tmp["transfer_share_pct"] = tmp["year"].map(share_by_year)
    tmp = tmp.dropna(subset=["transfer_share_pct"]).copy()
    if tmp.empty:
        return 0.0
    tmp["baseline_transfer_proxy_mcm"] = tmp["supply_mcm"] * tmp["transfer_share_pct"] / 100.0
    residual = tmp["actual_delta_mcm"] - tmp["pred_mcm"]
    x = tmp["baseline_transfer_proxy_mcm"].to_numpy(dtype=float)
    y = residual.to_numpy(dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    alpha = float(np.dot(x, y) / denom)
    return float(np.clip(alpha, 0.0, 1.0))


def predict_delta_mcm(component_row: dict[str, float], month: int, model: LinearRegression, month_bias: dict[int, float]) -> float:
    X = pd.DataFrame([{k: component_row[k] for k in FEATURE_ORDER}])
    base = float(model.predict(X)[0])
    return base + float(month_bias.get(month, 0.0))


def simulate_path(
    history_df: pd.DataFrame,
    future_exog: pd.DataFrame,
    model: LinearRegression,
    month_bias: dict[int, float],
    context: dict[str, float],
    transfer_share_anchor_pct: float,
    transfer_effectiveness: float = 0.0,
    baseline_transfer_share_pct: float = 0.0,
    transfer_end_pct_2040: float = 0.0,
) -> pd.DataFrame:
    past_fill = history_df["weighted_total_fill"].tolist()
    past_rain = history_df["rain_model_mm"].tolist()
    past_et0 = history_df["et0_mm_month"].tolist()
    rows: list[dict[str, float | str]] = []
    horizon = len(future_exog)
    progress = np.linspace(0.0, 1.0, horizon) if horizon else np.array([])

    for idx, row in future_exog.reset_index(drop=True).iterrows():
        fill_prev = float(past_fill[-1])
        date = pd.Timestamp(row["date"])
        days = int(date.days_in_month)
        supply_mcm = float(row["consumption_mean_monthly"] * days / 1e6)
        state = source_state(fill_prev, context)
        lake_area_km2 = float(state["lake_area_km2"].sum())
        runoff_weighted_land_area_km2 = float(np.dot(state["land_area_km2"], state["runoff_productivity_weight"]))
        rain_now = float(row["rain_model_mm"])
        rain_lag1 = float(past_rain[-1])
        et0_now = float(row["et0_mm_month"])
        rain_roll3 = float(np.mean([past_rain[-2], past_rain[-1], rain_now]))
        et0_roll3 = float(np.mean([past_et0[-2], past_et0[-1], et0_now]))
        baseline_transfer_mcm = supply_mcm * (baseline_transfer_share_pct / 100.0) * transfer_effectiveness
        transfer_delta_mcm = supply_mcm * (transfer_share_anchor_pct / 100.0) * (transfer_end_pct_2040 / 100.0) * float(progress[idx])
        total_transfer_mcm = baseline_transfer_mcm + transfer_delta_mcm

        comp = {
            "source_openwater_evap_mcm": lake_area_km2 * et0_now * 0.001,
            "supply_mcm": supply_mcm,
            "storage_mass_mcm": float(state["storage_mcm"].sum()),
            "spill_pressure_mcm": float(state["spill_pressure_mcm"].sum()),
        }
        runoff_now = 0.0
        runoff_lag = 0.0
        runoff_wet = 0.0
        lake_rain = 0.0
        src_table = context["source_table"].reset_index(drop=True)
        for j, src_row in src_table.iterrows():
            code = src_row["rain_code"]
            src_rain_now = float(row[code]) if code in future_exog.columns else rain_now
            src_rain_lag = src_rain_now
            if len(rows) > 0 and code in rows[-1]:
                src_rain_lag = float(rows[-1][code])
            src_rain_roll3 = np.mean([
                src_rain_now,
                src_rain_lag,
                src_rain_lag if len(rows) == 0 else float(rows[-1].get(f'{code}_lag1', src_rain_lag)),
            ])
            wet_mm = max(float(src_rain_roll3) - et0_roll3, 0.0)
            runoff_now += float(state["land_area_km2"][j] * state["runoff_productivity_weight"][j] * src_rain_now * 0.001)
            runoff_lag += float(state["land_area_km2"][j] * state["runoff_productivity_weight"][j] * src_rain_lag * 0.001)
            runoff_wet += float(state["land_area_km2"][j] * state["runoff_productivity_weight"][j] * wet_mm * 0.001)
            lake_rain += float(state["lake_area_km2"][j] * src_rain_now * 0.001)
        comp["source_runoff_now_mcm"] = runoff_now
        comp["source_runoff_lag1_mcm"] = runoff_lag
        comp["source_runoff_wetness_mcm"] = runoff_wet
        comp["source_lake_rain_mcm"] = lake_rain
        comp["neg_source_openwater_evap_mcm"] = -comp["source_openwater_evap_mcm"]
        comp["neg_supply_mcm"] = -comp["supply_mcm"]
        comp["neg_storage_mass_mcm"] = -comp["storage_mass_mcm"]
        comp["neg_spill_pressure_mcm"] = -comp["spill_pressure_mcm"]

        delta_storage_mcm = predict_delta_mcm(comp, date.month, model, month_bias) + total_transfer_mcm
        next_storage_mcm = np.clip(fill_prev * context["total_storage_mcm"] + delta_storage_mcm, 0.0, context["total_storage_mcm"])
        fill_next = float(next_storage_mcm / context["total_storage_mcm"])
        rows.append(
            {
                "date": date,
                "pred_fill": fill_next,
                "delta_storage_mcm": float(delta_storage_mcm),
                "supply_mcm": supply_mcm,
                "baseline_transfer_mcm": float(baseline_transfer_mcm),
                "transfer_delta_mcm": float(transfer_delta_mcm),
                "total_transfer_mcm": float(total_transfer_mcm),
                "source_runoff_now_mcm": float(comp["source_runoff_now_mcm"]),
                "source_runoff_lag1_mcm": float(comp["source_runoff_lag1_mcm"]),
                "source_runoff_wetness_mcm": float(comp["source_runoff_wetness_mcm"]),
                "source_lake_rain_mcm": float(comp["source_lake_rain_mcm"]),
                "source_openwater_evap_mcm": float(comp["source_openwater_evap_mcm"]),
                "storage_mass_mcm": float(comp["storage_mass_mcm"]),
                "spill_pressure_mcm": float(comp["spill_pressure_mcm"]),
                "rain_model_mm": rain_now,
                "et0_mm_month": et0_now,
            }
        )
        for code in [c for c in context["source_table"]["rain_code"].dropna().unique().tolist()]:
            if code in future_exog.columns:
                rows[-1][code] = float(row[code])
                rows[-1][f"{code}_lag1"] = float(rows[-2][code]) if len(rows) > 1 and code in rows[-2] else float(row[code])
        past_fill.append(fill_next)
        past_rain.append(rain_now)
        past_et0.append(et0_now)
    return pd.DataFrame(rows)


def historical_transfer_addition(date: pd.Timestamp, supply_mcm: float, share_by_year: dict[int, float], transfer_effectiveness: float) -> float:
    share = float(share_by_year.get(int(date.year), 0.0))
    return supply_mcm * (share / 100.0) * transfer_effectiveness


def one_step_walkforward(df: pd.DataFrame, context: dict[str, float], share_by_year: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = []
    for i in range(MIN_TRAIN, len(df)):
        train = df.iloc[:i].copy()
        test = df.iloc[[i]].copy()
        train_comp = component_frame(train, context)
        test_comp = component_frame(test, context)
        model, month_bias, fit_df = fit_water_balance_model(train_comp)
        transfer_effectiveness = estimate_transfer_effectiveness(train, fit_df, share_by_year)
        row = test_comp.iloc[0]
        delta_pred_mcm = predict_delta_mcm(row.to_dict(), int(pd.Timestamp(row["date"]).month), model, month_bias)
        delta_pred_mcm += historical_transfer_addition(pd.Timestamp(row["date"]), float(row["supply_mcm"]), share_by_year, transfer_effectiveness)
        storage_prev = float(test["weighted_total_fill_lag1"].iloc[0] * context["total_storage_mcm"])
        pred_fill = float(np.clip(storage_prev + delta_pred_mcm, 0.0, context["total_storage_mcm"]) / context["total_storage_mcm"])
        preds.append(
            {
                "date": test["date"].iloc[0],
                "actual_fill": float(test["weighted_total_fill"].iloc[0]),
                "pred_fill": pred_fill,
                "actual_delta_mcm": float(test["delta_storage_mcm"].iloc[0]),
                "pred_delta_mcm": float(delta_pred_mcm),
            }
        )
    pred_df = pd.DataFrame(preds)
    rmse_pp = float(np.sqrt(mean_squared_error(pred_df["actual_fill"], pred_df["pred_fill"])) * 100.0)
    mae_pp = float(mean_absolute_error(pred_df["actual_fill"], pred_df["pred_fill"]) * 100.0)
    direction_accuracy = float(((pred_df["pred_fill"].diff().fillna(0.0) > 0) == (pred_df["actual_fill"].diff().fillna(0.0) > 0)).mean())
    metrics = pd.DataFrame(
        [
            {
                "model": "water_balance_v5_sourcerain",
                "one_step_rmse_pp": rmse_pp,
                "one_step_mae_pp": mae_pp,
                "delta_direction_accuracy": direction_accuracy,
                "n_predictions": int(len(pred_df)),
            }
        ]
    )
    return metrics, pred_df


def recursive_backtest(df: pd.DataFrame, context: dict[str, float], share_by_year: dict[int, float], horizons: tuple[int, ...] = (1, 3, 6, 12)) -> pd.DataFrame:
    rows = []
    input_cols = future_input_columns(context)
    for horizon in horizons:
        actual = []
        pred = []
        origin_fill = []
        for i in range(MIN_TRAIN, len(df) - horizon + 1):
            train = df.iloc[:i].copy()
            future = df.iloc[i : i + horizon].copy()
            train_comp = component_frame(train, context)
            model, month_bias, fit_df = fit_water_balance_model(train_comp)
            transfer_effectiveness = estimate_transfer_effectiveness(train, fit_df, share_by_year)
            sim = simulate_path(
                train,
                future[input_cols],
                model,
                month_bias,
                context,
                transfer_share_anchor_pct=0.0,
                transfer_effectiveness=transfer_effectiveness,
                baseline_transfer_share_pct=0.0,
                transfer_end_pct_2040=0.0,
            )
            actual.append(float(future["weighted_total_fill"].iloc[horizon - 1]))
            pred.append(float(sim["pred_fill"].iloc[-1]))
            origin_fill.append(float(train["weighted_total_fill"].iloc[-1]))
        pred_arr = np.asarray(pred, dtype=float)
        actual_arr = np.asarray(actual, dtype=float)
        origin_arr = np.asarray(origin_fill, dtype=float)
        rows.append(
            {
                "model": "water_balance_v5_sourcerain",
                "horizon_months": int(horizon),
                "rmse_pp": float(np.sqrt(mean_squared_error(actual_arr, pred_arr)) * 100.0),
                "mae_pp": float(mean_absolute_error(actual_arr, pred_arr) * 100.0),
                "direction_accuracy": float(np.mean((pred_arr - origin_arr > 0) == (actual_arr - origin_arr > 0))),
                "n_predictions": int(len(actual_arr)),
            }
        )
    return pd.DataFrame(rows)


def build_component_coefficients(model: LinearRegression, month_bias: dict[int, float], context: dict[str, float]) -> pd.DataFrame:
    coeffs = pd.DataFrame({"feature": FEATURE_ORDER, "coefficient_mcm": model.coef_})
    coeffs.loc[len(coeffs)] = ["intercept_mcm", float(model.intercept_)]
    for month in range(1, 13):
        coeffs.loc[len(coeffs)] = [f"month_bias_{month:02d}_mcm", float(month_bias.get(month, 0.0))]
    coeffs.loc[len(coeffs)] = ["system_total_storage_mcm", context["total_storage_mcm"]]
    coeffs.loc[len(coeffs)] = ["system_total_lake_area_km2", context["total_lake_area_km2"]]
    coeffs.loc[len(coeffs)] = ["system_total_basin_area_km2", context["total_basin_area_km2"]]
    return coeffs


def build_component_contribution_table(proj_df: pd.DataFrame, model: LinearRegression, month_bias: dict[int, float]) -> pd.DataFrame:
    coef_map = dict(zip(FEATURE_ORDER, model.coef_))
    out = proj_df.copy()
    out["contrib_source_runoff_now_mcm"] = coef_map["source_runoff_now_mcm"] * out["source_runoff_now_mcm"]
    out["contrib_source_runoff_lag1_mcm"] = coef_map["source_runoff_lag1_mcm"] * out["source_runoff_lag1_mcm"]
    out["contrib_source_runoff_wetness_mcm"] = coef_map["source_runoff_wetness_mcm"] * out["source_runoff_wetness_mcm"]
    out["contrib_source_lake_rain_mcm"] = coef_map["source_lake_rain_mcm"] * out["source_lake_rain_mcm"]
    out["contrib_source_openwater_evap_mcm"] = coef_map["neg_source_openwater_evap_mcm"] * (-out["source_openwater_evap_mcm"])
    out["contrib_supply_mcm"] = coef_map["neg_supply_mcm"] * (-out["supply_mcm"])
    out["contrib_storage_mass_mcm"] = coef_map["neg_storage_mass_mcm"] * (-out["storage_mass_mcm"])
    out["contrib_spill_pressure_mcm"] = coef_map["neg_spill_pressure_mcm"] * (-out["spill_pressure_mcm"])
    out["contrib_baseline_transfer_mcm"] = out["baseline_transfer_mcm"]
    out["contrib_transfer_delta_mcm"] = out["transfer_delta_mcm"]
    out["contrib_month_bias_mcm"] = pd.to_datetime(out["date"]).dt.month.map(month_bias).fillna(0.0)
    return out


def scenario_projection(context: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forward = load_forward_module()
    df = load_training_frame(context)
    train_comp = component_frame(df, context)
    model, month_bias, fit_df = fit_water_balance_model(train_comp)
    share_by_year, anchor_share_pct = load_transfer_share_by_year()
    transfer_effectiveness = estimate_transfer_effectiveness(df, fit_df, share_by_year)
    clim = forward.monthly_climatology(df)
    rain_clim = source_rain_monthly_climatology(df, context)
    _, demand_relief = forward.latest_policy_anchor()
    all_cfgs = forward.build_scenarios() + forward.build_transfer_scenarios()
    input_cols = future_input_columns(context)
    parts = []
    for cfg in all_cfgs:
        neutral_cfg = replace(cfg, transfer_end_pct_2040=0.0)
        future = forward.build_future_exog(df, neutral_cfg, clim, demand_relief, transfer_share_anchor_pct=0.0)
        future = attach_future_source_rain(future, rain_clim, context)
        sim = simulate_path(
            df,
            future[input_cols],
            model,
            month_bias,
            context,
            transfer_share_anchor_pct=anchor_share_pct,
            transfer_effectiveness=transfer_effectiveness,
            baseline_transfer_share_pct=anchor_share_pct,
            transfer_end_pct_2040=float(cfg.transfer_end_pct_2040),
        )
        sim["scenario"] = cfg.scenario
        parts.append(sim)
    proj = pd.concat(parts, ignore_index=True)
    proj = proj[(proj["date"] >= "2026-01-01") & (proj["date"] <= "2040-12-01")].copy()
    primary = proj[proj["scenario"].isin(PRIMARY_SCENARIOS)].copy()
    summary = []
    for scenario, g in primary.groupby("scenario"):
        g = g.sort_values("date")
        summary.append(
            {
                "scenario": scenario,
                "mean_fill_2026_2040_pct": float(g["pred_fill"].mean() * 100.0),
                "min_fill_2026_2040_pct": float(g["pred_fill"].min() * 100.0),
                "end_fill_2040_12_pct": float(g.iloc[-1]["pred_fill"] * 100.0),
                "first_below_40_date": str(g.loc[g["pred_fill"] < 0.40, "date"].iloc[0].date()) if (g["pred_fill"] < 0.40).any() else "",
                "first_below_30_date": str(g.loc[g["pred_fill"] < 0.30, "date"].iloc[0].date()) if (g["pred_fill"] < 0.30).any() else "",
            }
        )
    summary_df = pd.DataFrame(summary).sort_values("mean_fill_2026_2040_pct", ascending=False).reset_index(drop=True)
    checkpoints = []
    for scenario, g in primary.groupby("scenario"):
        idx = g.set_index("date")
        for cp in [pd.Timestamp("2030-12-01"), pd.Timestamp("2035-12-01"), pd.Timestamp("2040-12-01")]:
            checkpoints.append(
                {
                    "scenario": scenario,
                    "checkpoint": str(cp.date()),
                    "pred_fill_pct": float(idx.loc[cp, "pred_fill"] * 100.0),
                    "supply_mcm": float(idx.loc[cp, "supply_mcm"]),
                    "source_openwater_evap_mcm": float(idx.loc[cp, "source_openwater_evap_mcm"]),
                    "baseline_transfer_mcm": float(idx.loc[cp, "baseline_transfer_mcm"]),
                    "transfer_delta_mcm": float(idx.loc[cp, "transfer_delta_mcm"]),
                    "total_transfer_mcm": float(idx.loc[cp, "total_transfer_mcm"]),
                    "source_runoff_now_mcm": float(idx.loc[cp, "source_runoff_now_mcm"]),
                }
            )
    checkpoints_df = pd.DataFrame(checkpoints)
    return proj, primary, summary_df, checkpoints_df


def physical_sanity(context: dict[str, float]) -> pd.DataFrame:
    forward = load_forward_module()
    df = load_training_frame(context)
    train_comp = component_frame(df, context)
    model, month_bias, fit_df = fit_water_balance_model(train_comp)
    share_by_year, anchor_share_pct = load_transfer_share_by_year()
    transfer_effectiveness = estimate_transfer_effectiveness(df, fit_df, share_by_year)
    clim = forward.monthly_climatology(df)
    rain_clim = source_rain_monthly_climatology(df, context)
    _, demand_relief = forward.latest_policy_anchor()
    base_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == "base")
    future_base = forward.build_future_exog(df, replace(base_cfg, transfer_end_pct_2040=0.0), clim, demand_relief, transfer_share_anchor_pct=0.0)
    future_base = attach_future_source_rain(future_base, rain_clim, context)
    future_base = future_base[future_base["date"].between(pd.Timestamp("2026-01-01"), pd.Timestamp("2040-12-01"))].copy()
    input_cols = future_input_columns(context)
    base_sim = simulate_path(
        df,
        future_base[input_cols],
        model,
        month_bias,
        context,
        transfer_share_anchor_pct=anchor_share_pct,
        transfer_effectiveness=transfer_effectiveness,
        baseline_transfer_share_pct=anchor_share_pct,
        transfer_end_pct_2040=0.0,
    )
    endpoint_base = float(base_sim.iloc[-1]["pred_fill"] * 100.0)

    scenario_inputs = []
    adj = future_base.copy()
    adj["rain_model_mm"] *= 1.10
    for code in [c for c in context["source_table"]["rain_code"].dropna().unique().tolist() if c in adj.columns]:
        adj[code] *= 1.10
    scenario_inputs.append(("rain_plus10", adj, 0.0))
    adj = future_base.copy()
    adj["consumption_mean_monthly"] *= 1.10
    scenario_inputs.append(("demand_plus10", adj, 0.0))
    adj = future_base.copy()
    adj["et0_mm_month"] *= 1.10
    scenario_inputs.append(("et0_plus10", adj, 0.0))
    scenario_inputs.append(("transfer_stress", future_base.copy(), -20.0))

    rows = []
    for name, future, transfer_pct in scenario_inputs:
        sim = simulate_path(
            df,
            future[input_cols],
            model,
            month_bias,
            context,
            transfer_share_anchor_pct=anchor_share_pct,
            transfer_effectiveness=transfer_effectiveness,
            baseline_transfer_share_pct=anchor_share_pct,
            transfer_end_pct_2040=transfer_pct,
        )
        rows.append(
            {
                "model": "water_balance_v5_sourcerain",
                "scenario": name,
                "base_endpoint_2040_pct": endpoint_base,
                "scenario_endpoint_2040_pct": float(sim.iloc[-1]["pred_fill"] * 100.0),
                "delta_pp": float(sim.iloc[-1]["pred_fill"] * 100.0 - endpoint_base),
            }
        )
    return pd.DataFrame(rows)


def plot_one_step(pred_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=170)
    ax.plot(pred_df["date"], pred_df["actual_fill"] * 100.0, color="#111827", linewidth=2.0, label="Gözlenen")
    ax.plot(pred_df["date"], pred_df["pred_fill"] * 100.0, color="#2563eb", linewidth=1.8, label="Su bütçesi v5")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.set_title("Tek adımlı geri test: kaynak + yağış-duyarlı su bütçesi v5")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scenarios(history_df: pd.DataFrame, proj_df: pd.DataFrame, out_path: Path) -> None:
    hist = history_df[history_df["date"] >= "2018-01-01"].copy()
    fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=170)
    ax.plot(hist["date"], hist["weighted_total_fill"] * 100.0, color="#111827", linewidth=2.0, label="Gözlenen")
    for scenario in PRIMARY_SCENARIOS:
        g = proj_df[proj_df["scenario"] == scenario].copy()
        ax.plot(g["date"], g["pred_fill"] * 100.0, color=SCENARIO_COLORS[scenario], linewidth=1.9, label=SCENARIO_LABELS[scenario])
    ax.axvline(pd.Timestamp("2026-01-01"), color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Toplam doluluk (%)")
    ax.set_title("Kaynak + yağış-duyarlı su bütçesi v5 ile 2026-2040 projeksiyon yolları")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_component_mix(contrib_df: pd.DataFrame, out_path: Path) -> None:
    base = contrib_df[contrib_df["scenario"] == "base"].copy()
    annual = (
        base.groupby(base["date"].dt.year)
        .agg(
            inflow_contrib_mcm=("contrib_source_runoff_now_mcm", "sum"),
            inflow_lag_contrib_mcm=("contrib_source_runoff_lag1_mcm", "sum"),
            wetness_contrib_mcm=("contrib_source_runoff_wetness_mcm", "sum"),
            lake_rain_contrib_mcm=("contrib_source_lake_rain_mcm", "sum"),
            openwater_contrib_mcm=("contrib_source_openwater_evap_mcm", "sum"),
            supply_contrib_mcm=("contrib_supply_mcm", "sum"),
            storage_contrib_mcm=("contrib_storage_mass_mcm", "sum"),
            spill_contrib_mcm=("contrib_spill_pressure_mcm", "sum"),
            baseline_transfer_contrib_mcm=("contrib_baseline_transfer_mcm", "sum"),
            transfer_contrib_mcm=("contrib_transfer_delta_mcm", "sum"),
            month_bias_contrib_mcm=("contrib_month_bias_mcm", "sum"),
        )
        .reset_index(names="year")
    )
    annual = annual[annual["year"].between(2026, 2040)].copy()
    inflow_total = annual["inflow_contrib_mcm"] + annual["inflow_lag_contrib_mcm"] + annual["wetness_contrib_mcm"] + annual["lake_rain_contrib_mcm"] + annual["baseline_transfer_contrib_mcm"] + annual["transfer_contrib_mcm"]
    outflow_total = annual["openwater_contrib_mcm"] + annual["supply_contrib_mcm"] + annual["storage_contrib_mcm"] + annual["spill_contrib_mcm"]
    fig, ax = plt.subplots(figsize=(10.8, 4.9), dpi=170)
    ax.plot(annual["year"], inflow_total, color="#059669", linewidth=2.0, label="Pozitif katkılar toplamı")
    ax.plot(annual["year"], outflow_total, color="#dc2626", linewidth=2.0, label="Negatif katkılar toplamı")
    ax.plot(annual["year"], annual["month_bias_contrib_mcm"], color="#2563eb", linewidth=1.8, label="Aylık işletme artığı")
    ax.set_ylabel("Hacim (milyon m3/yıl)")
    ax.set_title("Temel senaryoda kaynak + yağış-duyarlı yıllık katkı bileşenleri")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = OUT_DIR / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    context = compute_system_context()
    df = load_training_frame(context)
    comp = component_frame(df, context)
    model, month_bias, fit_df = fit_water_balance_model(comp)
    share_by_year, anchor_share_pct = load_transfer_share_by_year()
    transfer_effectiveness = estimate_transfer_effectiveness(df, fit_df, share_by_year)

    one_step_metrics, pred_df = one_step_walkforward(df, context, share_by_year)
    recursive_df = recursive_backtest(df, context, share_by_year)
    coeff_df = build_component_coefficients(model, month_bias, context)
    scenario_all, scenario_primary, scenario_summary, checkpoints_df = scenario_projection(context)
    sanity_df = physical_sanity(context)
    contrib_df = build_component_contribution_table(scenario_all, model, month_bias)

    benchmark = pd.read_csv(BENCHMARK_SCORECARD_PATH)
    benchmark = benchmark[["model", "one_step_rmse_pp", "mean_recursive_rmse_pp", "physics_pass_count"]]
    v3_compare_path = ROOT / "output" / "istanbul_water_balance_v3_2040" / "water_balance_vs_benchmark_models.csv"
    v3_compare = pd.read_csv(v3_compare_path)
    v3_compare = v3_compare[v3_compare["model"] == "water_balance_v3"][["model", "one_step_rmse_pp", "mean_recursive_rmse_pp", "physics_pass_count"]]
    v4_compare_path = ROOT / "output" / "istanbul_water_balance_v4_sourceaware_2040" / "water_balance_vs_benchmark_models.csv"
    v4_compare = pd.read_csv(v4_compare_path)
    v4_compare = v4_compare[v4_compare["model"] == "water_balance_v4_sourceaware"][["model", "one_step_rmse_pp", "mean_recursive_rmse_pp", "physics_pass_count"]]
    wb_recursive_mean = float(recursive_df["rmse_pp"].mean())
    physics_pass_count = int(
        sum(
            [
                sanity_df.loc[sanity_df["scenario"] == "rain_plus10", "delta_pp"].iloc[0] > 0,
                sanity_df.loc[sanity_df["scenario"] == "demand_plus10", "delta_pp"].iloc[0] < 0,
                sanity_df.loc[sanity_df["scenario"] == "et0_plus10", "delta_pp"].iloc[0] < 0,
                sanity_df.loc[sanity_df["scenario"] == "transfer_stress", "delta_pp"].iloc[0] < 0,
            ]
        )
    )
    compare = pd.concat(
        [
            benchmark,
            v3_compare,
            v4_compare,
            pd.DataFrame(
                [
                    {
                        "model": "water_balance_v5_sourcerain",
                        "one_step_rmse_pp": float(one_step_metrics.iloc[0]["one_step_rmse_pp"]),
                        "mean_recursive_rmse_pp": wb_recursive_mean,
                        "physics_pass_count": physics_pass_count,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    one_step_metrics.to_csv(OUT_DIR / "water_balance_one_step_metrics.csv", index=False)
    pred_df.to_csv(OUT_DIR / "water_balance_one_step_predictions.csv", index=False)
    recursive_df.to_csv(OUT_DIR / "water_balance_recursive_metrics.csv", index=False)
    coeff_df.to_csv(OUT_DIR / "water_balance_component_coefficients.csv", index=False)
    scenario_all.to_csv(OUT_DIR / "water_balance_scenario_projection_monthly_2026_2040.csv", index=False)
    contrib_df.to_csv(OUT_DIR / "water_balance_component_contributions_monthly_2026_2040.csv", index=False)
    scenario_summary.to_csv(OUT_DIR / "water_balance_scenario_summary_2026_2040.csv", index=False)
    checkpoints_df.to_csv(OUT_DIR / "water_balance_checkpoint_summary_2030_2035_2040.csv", index=False)
    sanity_df.to_csv(OUT_DIR / "water_balance_physical_sanity_checks.csv", index=False)
    compare.to_csv(OUT_DIR / "water_balance_vs_benchmark_models.csv", index=False)

    plot_one_step(pred_df, figs / "water_balance_one_step_backtest.png")
    plot_scenarios(df, scenario_primary, figs / "water_balance_scenarios_2026_2040.png")
    plot_component_mix(contrib_df, figs / "water_balance_component_mix_base.png")

    summary = {
        "model": "water_balance_v5_sourcerain",
        "one_step_rmse_pp": float(one_step_metrics.iloc[0]["one_step_rmse_pp"]),
        "mean_recursive_rmse_pp": wb_recursive_mean,
        "physics_passes": physics_pass_count,
        "transfer_share_anchor_pct": anchor_share_pct,
        "transfer_effectiveness": transfer_effectiveness,
        "system_context": {
            "total_storage_mcm": float(context["total_storage_mcm"]),
            "total_lake_area_km2": float(context["total_lake_area_km2"]),
            "total_basin_area_km2": float(context["total_basin_area_km2"]),
            "current_total_fill": float(context["current_total_fill"]),
            "n_sources": int(context["n_sources"]),
        },
        "source_weights": context["source_table"][
            ["source_name", "max_storage_million_m3", "runoff_productivity_weight", "fill_factor_prior"]
        ].to_dict(orient="records"),
        "notes": [
            "Demand uses official monthly city supply when available and proxy fallback otherwise.",
            "Transfer is modeled as a baseline external inflow anchored to the recent official transfer-share average, scaled by an effectiveness coefficient learned from overlap-year residuals, plus optional scenario deviation.",
            "Source-aware inflow allocates total storage across reservoir groups using current relative fill priors, then aggregates runoff over source-specific basin and lake areas.",
            "Source-specific monthly precipitation proxies are taken from Open-Meteo archive grids at approximate reservoir locations and scaled into future scenarios by monthly scenario rain factors.",
            "NRW is not separately observed monthly; it remains embedded in supply and future demand adjustments.",
        ],
    }
    (OUT_DIR / "water_balance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
