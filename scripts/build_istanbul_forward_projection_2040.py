#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CORE_PATH = Path("/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_core_monthly.csv")
POLICY_PATH = Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_policy_leverage_annual.csv")
TRANSFER_PATH = Path("/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/melen_yesilcay.csv")
SUPPLY_ANNUAL_PATH = Path("/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/son_10_yil_toplam_verilen_su.csv")
OUT_DIR = Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forward_projection_2040")

PRIMARY_SCENARIOS = ["base", "wet_mild", "hot_dry_high_demand", "management_improvement"]
DECOMPOSITION_SCENARIOS = [
    "management_efficiency_only",
    "management_nrw_only",
    "stress_climate_only",
    "stress_demand_only",
]
TRANSFER_SENSITIVITY_SCENARIOS = [
    "base_transfer_relief",
    "base_transfer_stress",
    "hot_dry_transfer_stress",
]
TRANSFER_BASELINE_MAP = {
    "base_transfer_relief": "base",
    "base_transfer_stress": "base",
    "hot_dry_transfer_stress": "hot_dry_high_demand",
}

FEATURES = [
    "weighted_total_fill_lag1",
    "weighted_total_fill_lag2",
    "rain_model_mm",
    "rain_model_mm_lag1",
    "rain_model_mm_roll3",
    "et0_mm_month",
    "et0_mm_month_lag1",
    "et0_mm_month_roll3",
    "consumption_mean_monthly",
    "consumption_mean_monthly_lag1",
    "consumption_mean_monthly_roll3",
    "temp_proxy_c",
    "rh_proxy_pct",
    "vpd_kpa_mean",
    "water_balance_proxy_mm",
    "month_sin",
    "month_cos",
]

HISTORY_ONLY_FEATURES = [
    "weighted_total_fill_lag1",
    "weighted_total_fill_lag2",
    "month_sin",
    "month_cos",
]


@dataclass
class ScenarioConfig:
    scenario: str
    rain_end_pct_2040: float
    et0_end_pct_2040: float
    temp_end_c_2040: float
    rh_end_pp_2040: float
    vpd_end_pct_2040: float
    subscriber_growth_pct_per_year: float
    per_capita_use_pct_per_year: float
    nrw_reduction_pp_by_2040: float
    transfer_end_pct_2040: float
    transfer_assumption: str
    source_ids: str
    rationale: str


def month_sin(month: int) -> float:
    return float(np.sin(2.0 * np.pi * month / 12.0))


def month_cos(month: int) -> float:
    return float(np.cos(2.0 * np.pi * month / 12.0))


def build_month_weight_map(kind: str) -> dict[int, float]:
    if kind == "base":
        return {
            1: 1.02,
            2: 1.02,
            3: 0.97,
            4: 0.96,
            5: 0.96,
            6: 1.00,
            7: 1.01,
            8: 1.01,
            9: 0.99,
            10: 0.98,
            11: 0.98,
            12: 1.02,
        }
    if kind == "wet":
        return {
            1: 1.06,
            2: 1.06,
            3: 1.02,
            4: 1.01,
            5: 1.00,
            6: 1.03,
            7: 1.04,
            8: 1.04,
            9: 1.00,
            10: 1.00,
            11: 1.01,
            12: 1.06,
        }
    if kind == "hot_dry":
        return {
            1: 0.98,
            2: 0.98,
            3: 0.92,
            4: 0.90,
            5: 0.90,
            6: 0.94,
            7: 0.95,
            8: 0.95,
            9: 0.90,
            10: 0.88,
            11: 0.88,
            12: 0.98,
        }
    raise ValueError(f"Unknown month-weight map: {kind}")


def build_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig(
            scenario="base",
            rain_end_pct_2040=-0.5,
            et0_end_pct_2040=4.0,
            temp_end_c_2040=0.8,
            rh_end_pp_2040=-1.0,
            vpd_end_pct_2040=6.0,
            subscriber_growth_pct_per_year=1.10,
            per_capita_use_pct_per_year=-0.15,
            nrw_reduction_pp_by_2040=1.50,
            transfer_end_pct_2040=0.0,
            transfer_assumption="neutral",
            source_ids="SRC-066,SRC-067,SRC-068,SRC-069",
            rationale="Official moderate city-growth path plus mild warming and small NRW improvement.",
        ),
        ScenarioConfig(
            scenario="wet_mild",
            rain_end_pct_2040=4.0,
            et0_end_pct_2040=2.5,
            temp_end_c_2040=0.6,
            rh_end_pp_2040=0.0,
            vpd_end_pct_2040=3.0,
            subscriber_growth_pct_per_year=0.90,
            per_capita_use_pct_per_year=-0.20,
            nrw_reduction_pp_by_2040=2.25,
            transfer_end_pct_2040=0.0,
            transfer_assumption="neutral",
            source_ids="SRC-066,SRC-067,SRC-068,SRC-069",
            rationale="Milder moisture regime, weaker ET0 growth, slower demand growth, slightly better operations.",
        ),
        ScenarioConfig(
            scenario="hot_dry_high_demand",
            rain_end_pct_2040=-8.0,
            et0_end_pct_2040=10.0,
            temp_end_c_2040=1.5,
            rh_end_pp_2040=-3.0,
            vpd_end_pct_2040=16.0,
            subscriber_growth_pct_per_year=1.30,
            per_capita_use_pct_per_year=0.15,
            nrw_reduction_pp_by_2040=0.00,
            transfer_end_pct_2040=0.0,
            transfer_assumption="neutral",
            source_ids="SRC-066,SRC-067,SRC-068,SRC-069,SRC-070,SRC-071",
            rationale="Stress case with drier seasonal redistribution, stronger ET0 rise, faster city pressure, no NRW relief.",
        ),
        ScenarioConfig(
            scenario="management_improvement",
            rain_end_pct_2040=-0.5,
            et0_end_pct_2040=4.0,
            temp_end_c_2040=0.8,
            rh_end_pp_2040=-1.0,
            vpd_end_pct_2040=6.0,
            subscriber_growth_pct_per_year=1.10,
            per_capita_use_pct_per_year=-0.45,
            nrw_reduction_pp_by_2040=5.25,
            transfer_end_pct_2040=0.0,
            transfer_assumption="neutral",
            source_ids="SRC-066,SRC-067,SRC-068,SRC-069",
            rationale="Same climate as base but stronger demand efficiency and larger NRW reduction by 2040.",
        ),
    ]


def build_decomposition_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig(
            scenario="management_efficiency_only",
            rain_end_pct_2040=-0.5,
            et0_end_pct_2040=4.0,
            temp_end_c_2040=0.8,
            rh_end_pp_2040=-1.0,
            vpd_end_pct_2040=6.0,
            subscriber_growth_pct_per_year=1.10,
            per_capita_use_pct_per_year=-0.45,
            nrw_reduction_pp_by_2040=1.50,
            transfer_end_pct_2040=0.0,
            transfer_assumption="neutral",
            source_ids="SRC-066,SRC-067,SRC-068,SRC-069",
            rationale="Base climate plus stronger per-capita efficiency only.",
        ),
        ScenarioConfig(
            scenario="management_nrw_only",
            rain_end_pct_2040=-0.5,
            et0_end_pct_2040=4.0,
            temp_end_c_2040=0.8,
            rh_end_pp_2040=-1.0,
            vpd_end_pct_2040=6.0,
            subscriber_growth_pct_per_year=1.10,
            per_capita_use_pct_per_year=-0.15,
            nrw_reduction_pp_by_2040=5.25,
            transfer_end_pct_2040=0.0,
            transfer_assumption="neutral",
            source_ids="SRC-066,SRC-067,SRC-068,SRC-069",
            rationale="Base climate plus stronger NRW improvement only.",
        ),
        ScenarioConfig(
            scenario="stress_climate_only",
            rain_end_pct_2040=-8.0,
            et0_end_pct_2040=10.0,
            temp_end_c_2040=1.5,
            rh_end_pp_2040=-3.0,
            vpd_end_pct_2040=16.0,
            subscriber_growth_pct_per_year=1.10,
            per_capita_use_pct_per_year=-0.15,
            nrw_reduction_pp_by_2040=1.50,
            transfer_end_pct_2040=0.0,
            transfer_assumption="neutral",
            source_ids="SRC-066,SRC-067,SRC-068,SRC-069,SRC-070,SRC-071",
            rationale="Hot-dry climate but base demand-management path.",
        ),
        ScenarioConfig(
            scenario="stress_demand_only",
            rain_end_pct_2040=-0.5,
            et0_end_pct_2040=4.0,
            temp_end_c_2040=0.8,
            rh_end_pp_2040=-1.0,
            vpd_end_pct_2040=6.0,
            subscriber_growth_pct_per_year=1.30,
            per_capita_use_pct_per_year=0.15,
            nrw_reduction_pp_by_2040=0.00,
            transfer_end_pct_2040=0.0,
            transfer_assumption="neutral",
            source_ids="SRC-066,SRC-067,SRC-068,SRC-069",
            rationale="Base climate but high-demand and no-NRW-relief path.",
        ),
    ]


def build_transfer_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig(
            scenario="base_transfer_relief",
            rain_end_pct_2040=-0.5,
            et0_end_pct_2040=4.0,
            temp_end_c_2040=0.8,
            rh_end_pp_2040=-1.0,
            vpd_end_pct_2040=6.0,
            subscriber_growth_pct_per_year=1.10,
            per_capita_use_pct_per_year=-0.15,
            nrw_reduction_pp_by_2040=1.50,
            transfer_end_pct_2040=10.0,
            transfer_assumption="official_transfer_relief_plus_10pct",
            source_ids="SRC-014,SRC-057,SRC-068,SRC-073,SRC-074",
            rationale="Base path plus stronger external-transfer availability, encoded as demand-equivalent relief from official transfer-share history.",
        ),
        ScenarioConfig(
            scenario="base_transfer_stress",
            rain_end_pct_2040=-0.5,
            et0_end_pct_2040=4.0,
            temp_end_c_2040=0.8,
            rh_end_pp_2040=-1.0,
            vpd_end_pct_2040=6.0,
            subscriber_growth_pct_per_year=1.10,
            per_capita_use_pct_per_year=-0.15,
            nrw_reduction_pp_by_2040=1.50,
            transfer_end_pct_2040=-20.0,
            transfer_assumption="official_transfer_stress_minus_20pct",
            source_ids="SRC-014,SRC-057,SRC-068,SRC-073,SRC-074",
            rationale="Base path plus weaker Melen-Yesilcay availability, encoded as a demand-equivalent burden using official transfer-share history.",
        ),
        ScenarioConfig(
            scenario="hot_dry_transfer_stress",
            rain_end_pct_2040=-8.0,
            et0_end_pct_2040=10.0,
            temp_end_c_2040=1.5,
            rh_end_pp_2040=-3.0,
            vpd_end_pct_2040=16.0,
            subscriber_growth_pct_per_year=1.30,
            per_capita_use_pct_per_year=0.15,
            nrw_reduction_pp_by_2040=0.00,
            transfer_end_pct_2040=-20.0,
            transfer_assumption="hot_dry_plus_transfer_stress_minus_20pct",
            source_ids="SRC-014,SRC-057,SRC-068,SRC-073,SRC-074",
            rationale="Hot-dry-high-demand stress path with an extra external-transfer shortfall encoded as additional demand on internal storage.",
        ),
    ]


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(CORE_PATH, parse_dates=["date"])
    df["delta_fill"] = df["weighted_total_fill"] - df["weighted_total_fill_lag1"]
    keep = ["date", "weighted_total_fill", "delta_fill"] + FEATURES
    return df[keep].dropna().reset_index(drop=True)


def make_model(model_name: str):
    if model_name in {"hybrid_ridge", "history_only_ridge"}:
        return Pipeline(
            [("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 3, 25)))]
        )
    if model_name == "extra_trees_full":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
        )
    raise ValueError(model_name)


def model_features(model_name: str) -> list[str]:
    if model_name == "history_only_ridge":
        return HISTORY_ONLY_FEATURES
    if model_name in {"hybrid_ridge", "extra_trees_full"}:
        return FEATURES
    raise ValueError(model_name)


def evaluate_models(train_df: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, pd.DataFrame]]:
    model_names = ["history_only_ridge", "hybrid_ridge", "extra_trees_full"]
    rows = []
    pred_frames: dict[str, pd.DataFrame] = {}
    min_train = 60
    for name in model_names:
        model = make_model(name)
        cols = model_features(name)
        actual = []
        pred = []
        dates = []
        for i in range(min_train, len(train_df)):
            tr = train_df.iloc[:i]
            te = train_df.iloc[[i]]
            model.fit(tr[cols], tr["delta_fill"])
            delta_hat = float(model.predict(te[cols])[0])
            yhat = float(te["weighted_total_fill_lag1"].iloc[0] + delta_hat)
            pred.append(yhat)
            actual.append(float(te["weighted_total_fill"].iloc[0]))
            dates.append(te["date"].iloc[0])
        rmse_pp = float(np.sqrt(mean_squared_error(actual, pred)) * 100.0)
        rows.append({"model": name, "rmse_pp": rmse_pp})
        pred_frames[name] = pd.DataFrame({"date": dates, "actual": actual, "pred": pred})
    out = pd.DataFrame(rows).sort_values("rmse_pp").reset_index(drop=True)
    return out, str(out.iloc[0]["model"]), pred_frames


def fit_model(train_df: pd.DataFrame, model_name: str):
    model = make_model(model_name)
    model.fit(train_df[model_features(model_name)], train_df["delta_fill"])
    return model


def monthly_climatology(train_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    cols = [
        "rain_model_mm",
        "et0_mm_month",
        "consumption_mean_monthly",
        "temp_proxy_c",
        "rh_proxy_pct",
        "vpd_kpa_mean",
    ]
    return {col: train_df.groupby(train_df["date"].dt.month)[col].mean().to_dict() for col in cols}


def build_empirical_interval_table(pred_df: pd.DataFrame, low_q: float = 0.10, high_q: float = 0.90) -> tuple[dict[int, tuple[float, float]], tuple[float, float], pd.DataFrame]:
    tmp = pred_df.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
    tmp["residual"] = tmp["actual"] - tmp["pred"]
    rows = []
    month_map: dict[int, tuple[float, float]] = {}
    for month, g in tmp.groupby("month"):
        lo = float(np.quantile(g["residual"], low_q))
        hi = float(np.quantile(g["residual"], high_q))
        rows.append({"month": int(month), "low_residual": lo, "high_residual": hi, "n": int(len(g))})
        month_map[int(month)] = (lo, hi)
    global_bounds = (
        float(np.quantile(tmp["residual"], low_q)),
        float(np.quantile(tmp["residual"], high_q)),
    )
    return month_map, global_bounds, pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_residual_pools(pred_df: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray]:
    tmp = pred_df.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
    tmp["residual"] = tmp["actual"] - tmp["pred"]
    pools: dict[int, np.ndarray] = {}
    for month, g in tmp.groupby("month"):
        pools[int(month)] = g["residual"].to_numpy(dtype=float)
    global_pool = tmp["residual"].to_numpy(dtype=float)
    return pools, global_pool


def latest_policy_anchor() -> tuple[pd.Series, float]:
    policy = pd.read_csv(POLICY_PATH)
    policy = policy.sort_values("year")
    row = policy.iloc[-1]
    demand_relief_pct_per_1pp_nrw = float(
        row["delta_volume_1pp_nrw_reduction_m3"] / row["authorized_consumption_m3_year"] * 100.0
    )
    return row, demand_relief_pct_per_1pp_nrw


def load_transfer_dependency_anchor() -> tuple[pd.DataFrame, float]:
    transfer = pd.read_csv(TRANSFER_PATH, parse_dates=["tarih"])
    supply = pd.read_csv(SUPPLY_ANNUAL_PATH, parse_dates=["tarih"])
    df = transfer.merge(supply[["tarih", "verilenTemizsuM3"]], on="tarih", how="inner")
    df = df[df["tarih"].dt.year.between(2021, 2025)].copy()
    df["transfer_share_pct"] = df["toplam"] / df["verilenTemizsuM3"] * 100.0
    df["transfer_demand_equivalent_if_minus20_pct"] = df["transfer_share_pct"] * 0.20
    df["transfer_demand_equivalent_if_plus10_pct"] = df["transfer_share_pct"] * 0.10
    anchor_share_pct = float(df["transfer_share_pct"].mean())
    return df.sort_values("tarih").reset_index(drop=True), anchor_share_pct


def build_future_exog(
    train_df: pd.DataFrame,
    cfg: ScenarioConfig,
    clim: dict[str, dict[int, float]],
    demand_relief_pct_per_1pp_nrw: float,
    transfer_share_anchor_pct: float,
) -> pd.DataFrame:
    start = train_df["date"].max() + pd.offsets.MonthBegin(1)
    future = pd.DataFrame({"date": pd.date_range(start, "2040-12-01", freq="MS")})
    future["month"] = future["date"].dt.month
    horizon = len(future)
    progress = np.linspace(0.0, 1.0, horizon)

    weight_kind = "base"
    if cfg.scenario == "wet_mild":
        weight_kind = "wet"
    elif cfg.scenario == "hot_dry_high_demand":
        weight_kind = "hot_dry"
    weights = build_month_weight_map(weight_kind)

    for col in clim:
        future[col] = future["month"].map(clim[col]).astype(float)

    future["rain_model_mm"] = future.apply(
        lambda r: r["rain_model_mm"]
        * weights[int(r["month"])]
        * (1.0 + (cfg.rain_end_pct_2040 / 100.0) * progress[r.name]),
        axis=1,
    )
    future["et0_mm_month"] *= 1.0 + (cfg.et0_end_pct_2040 / 100.0) * progress
    future["temp_proxy_c"] += cfg.temp_end_c_2040 * progress
    future["rh_proxy_pct"] = np.clip(future["rh_proxy_pct"] + cfg.rh_end_pp_2040 * progress, 25.0, 100.0)
    future["vpd_kpa_mean"] *= 1.0 + (cfg.vpd_end_pct_2040 / 100.0) * progress

    years_since_start = np.arange(horizon) / 12.0
    subscriber_factor = (1.0 + cfg.subscriber_growth_pct_per_year / 100.0) ** years_since_start
    per_capita_factor = (1.0 + cfg.per_capita_use_pct_per_year / 100.0) ** years_since_start
    nrw_relief_end_pct = cfg.nrw_reduction_pp_by_2040 * demand_relief_pct_per_1pp_nrw
    operations_relief = 1.0 - (nrw_relief_end_pct / 100.0) * progress
    transfer_demand_equivalent_end_pct = -(transfer_share_anchor_pct / 100.0) * cfg.transfer_end_pct_2040
    transfer_factor = 1.0 + (transfer_demand_equivalent_end_pct / 100.0) * progress
    future["consumption_mean_monthly"] *= subscriber_factor * per_capita_factor * operations_relief * transfer_factor
    future["month_sin"] = future["month"].apply(month_sin)
    future["month_cos"] = future["month"].apply(month_cos)
    future["nrw_relief_end_pct"] = nrw_relief_end_pct
    future["transfer_end_pct_2040"] = cfg.transfer_end_pct_2040
    future["transfer_demand_equivalent_end_pct"] = transfer_demand_equivalent_end_pct
    future["transfer_factor"] = transfer_factor
    return future


def apply_direct_demand_adjustment(future: pd.DataFrame, demand_end_pct_2040: float) -> pd.DataFrame:
    out = future.copy()
    progress = np.linspace(0.0, 1.0, len(out))
    out["consumption_mean_monthly"] *= 1.0 + (demand_end_pct_2040 / 100.0) * progress
    out["direct_demand_adjustment_end_pct"] = demand_end_pct_2040
    return out


def simulate_projection(
    train_df: pd.DataFrame,
    future_exog: pd.DataFrame,
    model,
    selected_model: str,
    interval_by_month: dict[int, tuple[float, float]],
    global_interval: tuple[float, float],
) -> pd.DataFrame:
    past_fill = train_df["weighted_total_fill"].tolist()
    past_rain = train_df["rain_model_mm"].tolist()
    past_et0 = train_df["et0_mm_month"].tolist()
    past_cons = train_df["consumption_mean_monthly"].tolist()
    rows: list[dict[str, float | str]] = []

    for _, row in future_exog.iterrows():
        rain_now = float(row["rain_model_mm"])
        et0_now = float(row["et0_mm_month"])
        cons_now = float(row["consumption_mean_monthly"])
        feat = {
            "weighted_total_fill_lag1": past_fill[-1],
            "weighted_total_fill_lag2": past_fill[-2],
            "rain_model_mm": rain_now,
            "rain_model_mm_lag1": past_rain[-1],
            "rain_model_mm_roll3": float(np.mean([past_rain[-2], past_rain[-1], rain_now])),
            "et0_mm_month": et0_now,
            "et0_mm_month_lag1": past_et0[-1],
            "et0_mm_month_roll3": float(np.mean([past_et0[-2], past_et0[-1], et0_now])),
            "consumption_mean_monthly": cons_now,
            "consumption_mean_monthly_lag1": past_cons[-1],
            "consumption_mean_monthly_roll3": float(np.mean([past_cons[-2], past_cons[-1], cons_now])),
            "temp_proxy_c": float(row["temp_proxy_c"]),
            "rh_proxy_pct": float(row["rh_proxy_pct"]),
            "vpd_kpa_mean": float(row["vpd_kpa_mean"]),
            "water_balance_proxy_mm": float(rain_now - et0_now),
            "month_sin": float(row["month_sin"]),
            "month_cos": float(row["month_cos"]),
        }
        cols = model_features(selected_model)
        delta_hat = float(model.predict(pd.DataFrame([{k: feat[k] for k in cols}]))[0])
        yhat = float(np.clip(past_fill[-1] + delta_hat, 0.0, 1.0))
        lo_resid, hi_resid = interval_by_month.get(int(pd.Timestamp(row["date"]).month), global_interval)
        ylow = float(np.clip(yhat + lo_resid, 0.0, 1.0))
        yhigh = float(np.clip(yhat + hi_resid, 0.0, 1.0))
        rows.append(
            {
                "date": row["date"],
                "pred_fill": yhat,
                "pred_fill_low": ylow,
                "pred_fill_high": yhigh,
                "rain_model_mm": rain_now,
                "et0_mm_month": et0_now,
                "consumption_mean_monthly": cons_now,
                "temp_proxy_c": float(row["temp_proxy_c"]),
                "rh_proxy_pct": float(row["rh_proxy_pct"]),
                "vpd_kpa_mean": float(row["vpd_kpa_mean"]),
                "water_balance_proxy_mm": float(rain_now - et0_now),
            }
        )
        past_fill.append(yhat)
        past_rain.append(rain_now)
        past_et0.append(et0_now)
        past_cons.append(cons_now)
    return pd.DataFrame(rows)


def apply_threshold_probabilities(
    proj_df: pd.DataFrame,
    residual_pools: dict[int, np.ndarray],
    global_pool: np.ndarray,
    thresholds: tuple[float, ...] = (0.40, 0.30),
) -> pd.DataFrame:
    out = proj_df.copy()
    for th in thresholds:
        probs = []
        col = f"prob_below_{int(th * 100)}"
        for row in out.itertuples(index=False):
            month = int(pd.Timestamp(row.date).month)
            pool = residual_pools.get(month, global_pool)
            if pool.size == 0:
                probs.append(float(row.pred_fill < th))
                continue
            probs.append(float(np.mean((float(row.pred_fill) + pool) < th)))
        out[col] = probs
    return out


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, g in df.groupby("scenario"):
        g = g.sort_values("date")
        target = g[(g["date"] >= "2026-01-01") & (g["date"] <= "2040-12-01")].copy()
        rows.append(
            {
                "scenario": scenario,
                "mean_fill_2026_2040_pct": float(target["pred_fill"].mean() * 100.0),
                "min_fill_2026_2040_pct": float(target["pred_fill"].min() * 100.0),
                "end_fill_2040_12_pct": float(target.iloc[-1]["pred_fill"] * 100.0),
                "mean_fill_2026_2030_pct": float(
                    target[target["date"].dt.year.between(2026, 2030)]["pred_fill"].mean() * 100.0
                ),
                "mean_fill_2031_2035_pct": float(
                    target[target["date"].dt.year.between(2031, 2035)]["pred_fill"].mean() * 100.0
                ),
                "mean_fill_2036_2040_pct": float(
                    target[target["date"].dt.year.between(2036, 2040)]["pred_fill"].mean() * 100.0
                ),
                "first_below_40_date": (
                    str(target.loc[target["pred_fill"] < 0.40, "date"].iloc[0].date())
                    if (target["pred_fill"] < 0.40).any()
                    else ""
                ),
                "first_below_30_date": (
                    str(target.loc[target["pred_fill"] < 0.30, "date"].iloc[0].date())
                    if (target["pred_fill"] < 0.30).any()
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_fill_2026_2040_pct", ascending=False).reset_index(drop=True)


def build_checkpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = [pd.Timestamp("2030-12-01"), pd.Timestamp("2035-12-01"), pd.Timestamp("2040-12-01")]
    rows = []
    base = df[df["scenario"] == "base"].set_index("date")["pred_fill"]
    for scenario, g in df.groupby("scenario"):
        g = g.set_index("date")
        for cp in checkpoints:
            if cp not in g.index:
                continue
            value_pct = float(g.loc[cp, "pred_fill"] * 100.0)
            base_pct = float(base.loc[cp] * 100.0)
            rows.append(
                {
                    "scenario": scenario,
                    "checkpoint": str(cp.date()),
                    "pred_fill_pct": value_pct,
                    "delta_vs_base_pp": value_pct - base_pct,
                }
            )
    return pd.DataFrame(rows)


def first_true_date(mask: pd.Series, dates: pd.Series) -> str:
    idx = np.flatnonzero(mask.to_numpy(dtype=bool))
    if len(idx) == 0:
        return ""
    return str(pd.Timestamp(dates.iloc[idx[0]]).date())


def longest_spell(mask: np.ndarray) -> int:
    best = cur = 0
    for val in mask:
        if bool(val):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def permanent_cross_date(mask: np.ndarray, dates: pd.Series) -> str:
    for i in range(len(mask)):
        if bool(mask[i]) and bool(mask[i:].all()):
            return str(pd.Timestamp(dates.iloc[i]).date())
    return ""


def recovery_info(mask: np.ndarray, dates: pd.Series) -> tuple[str, str, str | int]:
    first_cross = first_true_date(pd.Series(mask), dates)
    if not first_cross:
        return "", "", ""
    first_idx = int(np.flatnonzero(mask)[0])
    for j in range(first_idx + 1, len(mask)):
        if not bool(mask[j]):
            lag = (dates.iloc[j].year - dates.iloc[first_idx].year) * 12 + (dates.iloc[j].month - dates.iloc[first_idx].month)
            return first_cross, str(pd.Timestamp(dates.iloc[j]).date()), int(lag)
    return first_cross, "", ""


def build_threshold_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, g in df.groupby("scenario"):
        g = g.sort_values("date").reset_index(drop=True)
        dates = g["date"]
        for th, label in [(0.40, "40"), (0.30, "30")]:
            point_mask = (g["pred_fill"] < th).to_numpy(dtype=bool)
            prob_col = f"prob_below_{label}"
            prob = g[prob_col].to_numpy(dtype=float)
            first_cross, recovery_date, recovery_lag = recovery_info(point_mask, dates)
            rows.append(
                {
                    "scenario": scenario,
                    "threshold_pct": int(label),
                    "months_point_below_threshold": int(point_mask.sum()),
                    "months_prob_ge_50pct": int((prob >= 0.50).sum()),
                    "mean_prob_below_threshold_pct": float(prob.mean() * 100.0),
                    "max_prob_below_threshold_pct": float(prob.max() * 100.0),
                    "first_cross_date": first_cross,
                    "first_recovery_date": recovery_date,
                    "recovery_lag_months": recovery_lag,
                    "longest_spell_below_threshold_months": longest_spell(point_mask),
                    "permanent_cross_date": permanent_cross_date(point_mask, dates),
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold_pct", "mean_prob_below_threshold_pct"], ascending=[False, False]).reset_index(drop=True)


def build_driver_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = [pd.Timestamp("2030-12-01"), pd.Timestamp("2035-12-01"), pd.Timestamp("2040-12-01")]
    target = df.set_index(["scenario", "date"])["pred_fill"]
    rows = []
    pairs = [
        ("management_improvement", "management_efficiency_only", "management_nrw_only", "base"),
        ("hot_dry_high_demand", "stress_climate_only", "stress_demand_only", "base"),
    ]
    for full_name, part_a, part_b, base_name in pairs:
        for cp in checkpoints:
            full_val = float(target.loc[(full_name, cp)] * 100.0)
            a_val = float(target.loc[(part_a, cp)] * 100.0)
            b_val = float(target.loc[(part_b, cp)] * 100.0)
            base_val = float(target.loc[(base_name, cp)] * 100.0)
            rows.append(
                {
                    "bundle": full_name,
                    "checkpoint": str(cp.date()),
                    "base_fill_pct": base_val,
                    "full_fill_pct": full_val,
                    "full_delta_vs_base_pp": full_val - base_val,
                    "component_a_scenario": part_a,
                    "component_a_delta_vs_base_pp": a_val - base_val,
                    "component_b_scenario": part_b,
                    "component_b_delta_vs_base_pp": b_val - base_val,
                }
            )
    return pd.DataFrame(rows)


def plot_paths(history: pd.DataFrame, projections: pd.DataFrame, out_path: Path) -> None:
    hist = history[history["date"] >= "2018-01-01"].copy()
    fig, ax = plt.subplots(figsize=(11.5, 5.3), dpi=160)
    ax.plot(hist["date"], hist["weighted_total_fill"] * 100.0, color="#111827", linewidth=2.0, label="Gozlenen toplam doluluk")
    colors = {
        "base": "#2563eb",
        "wet_mild": "#059669",
        "hot_dry_high_demand": "#dc2626",
        "management_improvement": "#d97706",
    }
    labels = {
        "base": "Temel senaryo",
        "wet_mild": "Ilik-islak senaryo",
        "hot_dry_high_demand": "Sicak-kurak-yuksek talep",
        "management_improvement": "Yonetim iyilesme",
    }
    for scen in PRIMARY_SCENARIOS:
        if scen not in projections["scenario"].unique():
            continue
        g = projections[projections["scenario"] == scen].copy()
        target = g[g["date"] >= "2026-01-01"].copy()
        ax.fill_between(
            target["date"],
            target["pred_fill_low"] * 100.0,
            target["pred_fill_high"] * 100.0,
            color=colors[scen],
            alpha=0.08,
        )
        ax.plot(target["date"], target["pred_fill"] * 100.0, linewidth=1.8, color=colors[scen], label=labels[scen])
    ax.axvline(pd.Timestamp("2026-01-01"), color="#6b7280", linestyle="--", linewidth=1.0)
    ax.text(pd.Timestamp("2026-01-01"), 98, "2026 baslangici", color="#6b7280", fontsize=9, ha="left", va="top")
    ax.set_ylim(0, 100)
    ax.set_title("Istanbul toplam baraj doluluk projeksiyonu (2026-2040)")
    ax.set_ylabel("Doluluk (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_benchmark(metrics_df: pd.DataFrame, out_path: Path) -> None:
    order = ["history_only_ridge", "hybrid_ridge", "extra_trees_full"]
    labels = ["Yalnız tarihsel", "Hibrit Ridge", "Extra Trees"]
    colors = ["#9ca3af", "#2563eb", "#dc2626"]
    tmp = metrics_df.set_index("model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=160)
    x = np.arange(len(tmp))
    ax.bar(x, tmp["rmse_pp"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8, ha="right")
    ax.set_ylabel("RMSE (yüzde puan)")
    ax.set_title("İleri projeksiyon çekirdeği için model karşılaştırması")
    ax.grid(True, axis="y", alpha=0.2)
    for i, val in enumerate(tmp["rmse_pp"]):
        ax.text(i, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_risk(risk_df: pd.DataFrame, out_path: Path) -> None:
    tmp = risk_df[risk_df["threshold_pct"] == 30].copy()
    order = ["wet_mild", "management_improvement", "base", "hot_dry_high_demand"]
    tmp = tmp.set_index("scenario").loc[order].reset_index()
    labels = ["Ilik-islak", "Yonetim iyilesme", "Temel", "Sicak-kurak-yuksek talep"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=160)
    x = np.arange(len(tmp))
    ax.bar(x, tmp["mean_prob_below_threshold_pct"], color=["#059669", "#d97706", "#2563eb", "#dc2626"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Ortalama %30 alti olasilik (%)")
    ax.set_title("Senaryolara gore %30 alti risk")
    ax.grid(True, axis="y", alpha=0.2)
    for i, val in enumerate(tmp["mean_prob_below_threshold_pct"]):
        ax.text(i, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_driver_decomposition(decomp_df: pd.DataFrame, out_path: Path) -> None:
    tmp = decomp_df[decomp_df["checkpoint"] == "2040-12-01"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=160, sharey=False)
    bundles = [
        ("management_improvement", "Yonetim iyilesme", "#d97706"),
        ("hot_dry_high_demand", "Sicak-kurak baski", "#dc2626"),
    ]
    for ax, (bundle, title, color) in zip(axes, bundles, strict=False):
        row = tmp[tmp["bundle"] == bundle].iloc[0]
        vals = [
            float(row["component_a_delta_vs_base_pp"]),
            float(row["component_b_delta_vs_base_pp"]),
            float(row["full_delta_vs_base_pp"]),
        ]
        labels = ["Bilesen A", "Bilesen B", "Birlesik etki"]
        if bundle == "management_improvement":
            labels = ["Verimlilik", "NRW iyilesmesi", "Birlesik etki"]
        elif bundle == "hot_dry_high_demand":
            labels = ["Iklim baskisi", "Talep baskisi", "Birlesik etki"]
        x = np.arange(3)
        ax.bar(x, vals, color=[color, "#2563eb", "#111827"])
        ax.axhline(0, color="#6b7280", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10, ha="right")
        ax.set_title(title)
        ax.set_ylabel("2040 sonunda baza gore fark (yp)")
        ax.grid(True, axis="y", alpha=0.2)
        for i, val in enumerate(vals):
            va = "bottom" if val >= 0 else "top"
            offset = 0.4 if val >= 0 else -0.4
            ax.text(i, val + offset, f"{val:.1f}", ha="center", va=va, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_future_driver_annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
    rows = []
    for (scenario, year), g in tmp.groupby(["scenario", "year"]):
        rows.append(
            {
                "scenario": scenario,
                "year": int(year),
                "mean_rain_mm": float(g["rain_model_mm"].mean()),
                "mean_et0_mm": float(g["et0_mm_month"].mean()),
                "mean_consumption_m3": float(g["consumption_mean_monthly"].mean()),
                "mean_temp_c": float(g["temp_proxy_c"].mean()),
                "mean_rh_pct": float(g["rh_proxy_pct"].mean()),
                "mean_vpd_kpa": float(g["vpd_kpa_mean"].mean()),
                "mean_water_balance_mm": float(g["water_balance_proxy_mm"].mean()),
                "mean_fill_pct": float(g["pred_fill"].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "year"]).reset_index(drop=True)


def build_future_driver_checkpoint_summary(df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = [pd.Timestamp("2030-12-01"), pd.Timestamp("2035-12-01"), pd.Timestamp("2040-12-01")]
    rows = []
    for scenario, g in df.groupby("scenario"):
        g = g.set_index("date")
        for cp in checkpoints:
            if cp not in g.index:
                continue
            row = g.loc[cp]
            rows.append(
                {
                    "scenario": scenario,
                    "checkpoint": str(cp.date()),
                    "rain_mm": float(row["rain_model_mm"]),
                    "et0_mm": float(row["et0_mm_month"]),
                    "consumption_m3": float(row["consumption_mean_monthly"]),
                    "temp_c": float(row["temp_proxy_c"]),
                    "rh_pct": float(row["rh_proxy_pct"]),
                    "vpd_kpa": float(row["vpd_kpa_mean"]),
                    "water_balance_mm": float(row["water_balance_proxy_mm"]),
                    "fill_pct": float(row["pred_fill"] * 100.0),
                }
            )
    return pd.DataFrame(rows)


def plot_future_driver_paths(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df[df["scenario"].isin(PRIMARY_SCENARIOS)].copy()
    colors = {
        "base": "#2563eb",
        "wet_mild": "#059669",
        "hot_dry_high_demand": "#dc2626",
        "management_improvement": "#d97706",
    }
    labels = {
        "base": "Temel",
        "wet_mild": "Ilık-ıslak",
        "hot_dry_high_demand": "Sıcak-kurak-yüksek talep",
        "management_improvement": "Yönetim iyileşme",
    }
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), dpi=160, sharex=True)
    specs = [
        ("rain_model_mm", "Yağış (mm/ay)"),
        ("et0_mm_month", "Referans evapotranspirasyon - ET0 (mm/ay)"),
        ("consumption_mean_monthly", "Tüketim (m3/ay)"),
        ("temp_proxy_c", "Sıcaklık (°C)"),
        ("rh_proxy_pct", "Bağıl nem (%)"),
        ("vpd_kpa_mean", "Buhar basıncı açığı - VPD (kPa)"),
        ("water_balance_proxy_mm", "Yağış - ET0 (mm/ay)"),
        ("pred_fill", "Toplam doluluk (%)"),
    ]
    for ax, (col, title) in zip(axes.flatten(), specs, strict=False):
        for scen in PRIMARY_SCENARIOS:
            g = plot_df[plot_df["scenario"] == scen].copy()
            y = g[col] * 100.0 if col == "pred_fill" else g[col]
            ax.plot(g["date"], y, color=colors[scen], linewidth=1.7, label=labels[scen])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.22)
        if col == "pred_fill":
            ax.set_ylim(0, 100)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("2026-2040 gelecek sürücü yolları ve toplam doluluk", y=0.998, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.982])
    fig.savefig(out_path)
    plt.close(fig)


def plot_future_driver_deltas(checkpoint_df: pd.DataFrame, out_path: Path) -> None:
    cp = checkpoint_df[checkpoint_df["checkpoint"] == "2040-12-01"].copy()
    base = cp[cp["scenario"] == "base"].iloc[0]
    compare = cp[cp["scenario"].isin(["wet_mild", "management_improvement", "hot_dry_high_demand"])].copy()
    rows = []
    for row in compare.itertuples(index=False):
        rows.append(
            {
                "scenario": row.scenario,
                "rain_delta": float(row.rain_mm - base.rain_mm),
                "et0_delta": float(row.et0_mm - base.et0_mm),
                "consumption_delta_pct": float((row.consumption_m3 / base.consumption_m3 - 1.0) * 100.0),
                "temp_delta": float(row.temp_c - base.temp_c),
                "vpd_delta_pct": float((row.vpd_kpa / base.vpd_kpa - 1.0) * 100.0),
                "fill_delta": float(row.fill_pct - base.fill_pct),
            }
        )
    tmp = pd.DataFrame(rows)
    scenario_labels = {
        "wet_mild": "Ilık-ıslak",
        "management_improvement": "Yönetim iyileşme",
        "hot_dry_high_demand": "Sıcak-kurak-yüksek talep",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), dpi=160)
    specs = [
        ("rain_delta", "2040 yağış farkı (mm/ay)"),
        ("et0_delta", "2040 ET0 farkı (mm/ay)"),
        ("consumption_delta_pct", "2040 tüketim farkı (%)"),
        ("temp_delta", "2040 sıcaklık farkı (°C)"),
        ("vpd_delta_pct", "2040 VPD farkı (%)"),
        ("fill_delta", "2040 doluluk farkı (yp)"),
    ]
    colors = ["#059669", "#d97706", "#dc2626"]
    for ax, (col, title) in zip(axes.flatten(), specs, strict=False):
        vals = tmp[col].to_numpy(dtype=float)
        ax.bar(np.arange(len(tmp)), vals, color=colors)
        ax.axhline(0, color="#6b7280", linewidth=1.0)
        ax.set_xticks(np.arange(len(tmp)))
        ax.set_xticklabels([scenario_labels[s] for s in tmp["scenario"]], rotation=10, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.2)
        for i, val in enumerate(vals):
            va = "bottom" if val >= 0 else "top"
            offset = 0.4 if val >= 0 else -0.4
            ax.text(i, val + offset, f"{val:.1f}", ha="center", va=va, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_transfer_sensitivity_summary(df: pd.DataFrame, assumptions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    assumption_map = assumptions_df.set_index("scenario")
    for scenario in TRANSFER_SENSITIVITY_SCENARIOS:
        g = df[df["scenario"] == scenario].sort_values("date").copy()
        if g.empty:
            continue
        paired = TRANSFER_BASELINE_MAP[scenario]
        base = df[df["scenario"] == paired].sort_values("date").copy()
        end_row = g[g["date"] == pd.Timestamp("2040-12-01")].iloc[0]
        base_end_row = base[base["date"] == pd.Timestamp("2040-12-01")].iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "paired_baseline": paired,
                "transfer_end_pct_2040": float(assumption_map.loc[scenario, "transfer_end_pct_2040"]),
                "transfer_demand_equivalent_end_pct": float(
                    assumption_map.loc[scenario, "transfer_demand_equivalent_end_pct"]
                ),
                "mean_fill_2026_2040_pct": float(g["pred_fill"].mean() * 100.0),
                "end_fill_2040_12_pct": float(end_row["pred_fill"] * 100.0),
                "delta_vs_paired_baseline_2040_12_pp": float(
                    (end_row["pred_fill"] - base_end_row["pred_fill"]) * 100.0
                ),
                "min_fill_2026_2040_pct": float(g["pred_fill"].min() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def plot_transfer_dependency_history(transfer_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=160)
    years = transfer_df["tarih"].dt.year.to_numpy()

    axes[0].bar(years, transfer_df["toplam"], color="#2563eb")
    axes[0].set_title("Melen + Yeşilçay toplamı")
    axes[0].set_ylabel("Milyon m3/yıl")
    axes[0].grid(True, axis="y", alpha=0.2)
    for x, y in zip(years, transfer_df["toplam"], strict=False):
        axes[0].text(x, y + 8, f"{y:.0f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(years, transfer_df["transfer_share_pct"], color="#d97706")
    axes[1].set_title("Şehre verilen su içinde transfer payı")
    axes[1].set_ylabel("Pay (%)")
    axes[1].grid(True, axis="y", alpha=0.2)
    for x, y in zip(years, transfer_df["transfer_share_pct"], strict=False):
        axes[1].text(x, y + 1.0, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Resmî Melen-Yeşilçay bağımlılığı (2021-2025)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def plot_transfer_sensitivity_paths(df: pd.DataFrame, out_path: Path) -> None:
    scenarios = ["base", "base_transfer_relief", "base_transfer_stress", "hot_dry_high_demand", "hot_dry_transfer_stress"]
    labels = {
        "base": "Temel",
        "base_transfer_relief": "Temel + transfer rahatlama",
        "base_transfer_stress": "Temel + transfer stresi",
        "hot_dry_high_demand": "Sıcak-kurak-yüksek talep",
        "hot_dry_transfer_stress": "Sıcak-kurak + transfer stresi",
    }
    colors = {
        "base": "#2563eb",
        "base_transfer_relief": "#059669",
        "base_transfer_stress": "#d97706",
        "hot_dry_high_demand": "#dc2626",
        "hot_dry_transfer_stress": "#7f1d1d",
    }
    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=160)
    for scenario in scenarios:
        g = df[df["scenario"] == scenario].sort_values("date").copy()
        if g.empty:
            continue
        ax.plot(g["date"], g["pred_fill"] * 100.0, color=colors[scenario], linewidth=1.9, label=labels[scenario])
    ax.set_ylim(0, 100)
    ax.set_title("Dış transfer duyarlılığı ile toplam doluluk yolu")
    ax.set_ylabel("Doluluk (%)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_transfer_endpoint_deltas(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    labels = []
    vals = []
    for row in summary_df.itertuples(index=False):
        label = row.scenario.replace("_", " ")
        labels.append(label)
        vals.append(float(row.delta_vs_paired_baseline_2040_12_pp))
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=160)
    x = np.arange(len(labels))
    colors = ["#059669", "#d97706", "#7f1d1d"]
    ax.bar(x, vals, color=colors[: len(vals)])
    ax.axhline(0, color="#6b7280", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("2040 sonu farkı (yp)")
    ax.set_title("Dış transfer değişiminin doluluk etkisi")
    ax.grid(True, axis="y", alpha=0.2)
    for i, val in enumerate(vals):
        va = "bottom" if val >= 0 else "top"
        offset = 0.35 if val >= 0 else -0.35
        ax.text(i, val + offset, f"{val:.1f}", ha="center", va=va, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_variant_projection(
    train_df: pd.DataFrame,
    clim: dict[str, dict[int, float]],
    model,
    selected_model: str,
    interval_by_month: dict[int, tuple[float, float]],
    global_interval: tuple[float, float],
    demand_relief_pct_per_1pp_nrw: float,
    transfer_share_anchor_pct: float,
    base_cfg: ScenarioConfig,
    *,
    rain_end_pct_2040: float | None = None,
    et0_end_pct_2040: float | None = None,
    transfer_end_pct_2040: float | None = None,
    direct_demand_end_pct_2040: float = 0.0,
) -> pd.DataFrame:
    cfg = replace(
        base_cfg,
        rain_end_pct_2040=base_cfg.rain_end_pct_2040 if rain_end_pct_2040 is None else rain_end_pct_2040,
        et0_end_pct_2040=base_cfg.et0_end_pct_2040 if et0_end_pct_2040 is None else et0_end_pct_2040,
        transfer_end_pct_2040=base_cfg.transfer_end_pct_2040 if transfer_end_pct_2040 is None else transfer_end_pct_2040,
    )
    future = build_future_exog(
        train_df,
        cfg,
        clim,
        demand_relief_pct_per_1pp_nrw,
        transfer_share_anchor_pct=transfer_share_anchor_pct,
    )
    future = apply_direct_demand_adjustment(future, demand_end_pct_2040=direct_demand_end_pct_2040)
    return simulate_projection(
        train_df,
        future,
        model,
        selected_model=selected_model,
        interval_by_month=interval_by_month,
        global_interval=global_interval,
    )


def build_sensitivity_grid_tables(
    train_df: pd.DataFrame,
    clim: dict[str, dict[int, float]],
    model,
    selected_model: str,
    interval_by_month: dict[int, tuple[float, float]],
    global_interval: tuple[float, float],
    demand_relief_pct_per_1pp_nrw: float,
    transfer_share_anchor_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cfg = next(cfg for cfg in build_scenarios() if cfg.scenario == "base")
    rain_values = [-10.0, -5.0, 0.0, 5.0, 10.0]
    demand_values = [-10.0, -5.0, 0.0, 5.0, 10.0]
    et0_values = [0.0, 4.0, 8.0, 12.0, 16.0]
    transfer_values = [-20.0, -10.0, 0.0, 10.0]

    rain_demand_rows = []
    for rain_end in rain_values:
        for demand_end in demand_values:
            proj = run_variant_projection(
                train_df,
                clim,
                model,
                selected_model,
                interval_by_month,
                global_interval,
                demand_relief_pct_per_1pp_nrw,
                transfer_share_anchor_pct,
                base_cfg,
                rain_end_pct_2040=rain_end,
                direct_demand_end_pct_2040=demand_end,
            )
            end_row = proj[proj["date"] == pd.Timestamp("2040-12-01")].iloc[0]
            rain_demand_rows.append(
                {
                    "rain_end_pct_2040": rain_end,
                    "direct_demand_end_pct_2040": demand_end,
                    "end_fill_2040_12_pct": float(end_row["pred_fill"] * 100.0),
                    "mean_fill_2026_2040_pct": float(proj["pred_fill"].mean() * 100.0),
                    "end_rain_mm": float(end_row["rain_model_mm"]),
                    "end_consumption_m3": float(end_row["consumption_mean_monthly"]),
                }
            )

    et0_transfer_rows = []
    for et0_end in et0_values:
        for transfer_end in transfer_values:
            proj = run_variant_projection(
                train_df,
                clim,
                model,
                selected_model,
                interval_by_month,
                global_interval,
                demand_relief_pct_per_1pp_nrw,
                transfer_share_anchor_pct,
                base_cfg,
                et0_end_pct_2040=et0_end,
                transfer_end_pct_2040=transfer_end,
            )
            end_row = proj[proj["date"] == pd.Timestamp("2040-12-01")].iloc[0]
            et0_transfer_rows.append(
                {
                    "et0_end_pct_2040": et0_end,
                    "transfer_end_pct_2040": transfer_end,
                    "transfer_demand_equivalent_end_pct": float(-(transfer_share_anchor_pct / 100.0) * transfer_end),
                    "end_fill_2040_12_pct": float(end_row["pred_fill"] * 100.0),
                    "mean_fill_2026_2040_pct": float(proj["pred_fill"].mean() * 100.0),
                    "end_et0_mm": float(end_row["et0_mm_month"]),
                    "end_consumption_m3": float(end_row["consumption_mean_monthly"]),
                }
            )

    return pd.DataFrame(rain_demand_rows), pd.DataFrame(et0_transfer_rows)


def plot_sensitivity_heatmap(
    df: pd.DataFrame,
    index_col: str,
    column_col: str,
    value_col: str,
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    pivot = df.pivot(index=index_col, columns=column_col, values=value_col).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 5.7), dpi=160)
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = float(pivot.iloc[i, j])
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("2040 sonu doluluk (%)")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_endpoints(summary_df: pd.DataFrame, out_path: Path) -> None:
    order = ["wet_mild", "management_improvement", "base", "hot_dry_high_demand"]
    tmp = summary_df.set_index("scenario").loc[order].reset_index()
    labels = ["Ilik-islak", "Yonetim iyilesme", "Temel", "Sicak-kurak-yuksek talep"]
    colors = ["#059669", "#d97706", "#2563eb", "#dc2626"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=160)
    x = np.arange(len(tmp))
    ax.bar(x, tmp["end_fill_2040_12_pct"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("2040 sonu doluluk (%)")
    ax.set_title("Senaryolara gore 2040 sonu toplam doluluk")
    ax.grid(True, axis="y", alpha=0.2)
    for i, val in enumerate(tmp["end_fill_2040_12_pct"]):
        ax.text(i, val + 0.6, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_markdown_summary(
    out_path: Path,
    metrics_df: pd.DataFrame,
    selected_model: str,
    interval_df: pd.DataFrame,
    assumptions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    decomp_df: pd.DataFrame,
    annual_driver_df: pd.DataFrame,
    driver_checkpoint_df: pd.DataFrame,
    transfer_anchor_df: pd.DataFrame,
    transfer_anchor_share_pct: float,
    transfer_summary_df: pd.DataFrame,
    sensitivity_rain_demand_df: pd.DataFrame,
    sensitivity_et0_transfer_df: pd.DataFrame,
    policy_row: pd.Series,
    demand_relief_pct_per_1pp_nrw: float,
) -> None:
    lines = [
        "# 2026-2040 Istanbul Forward Projection Round 1",
        "",
        "## Model selection",
        "",
        f"- Selected model: `{selected_model}`",
        f"- Training window: `2011-02` to `2024-02`",
        f"- Note: actinograph has not yet been integrated; ET0 still uses the current proxy-backed production series.",
        "",
        "Walk-forward RMSE (percentage points):",
        "",
    ]
    for row in metrics_df.itertuples(index=False):
        lines.append(f"- `{row.model}`: `{row.rmse_pp:.2f}`")

    lines += [
        "",
        "## Scenario parameterization basis",
        "",
        f"- Latest official operations anchor year: `{int(policy_row['year'])}`",
        f"- Official demand-relief equivalent of `1 pp` NRW reduction: `{demand_relief_pct_per_1pp_nrw:.2f}%` of authorized demand",
        f"- Official `2021-2025` mean Melen-Yesilcay transfer share: `{transfer_anchor_share_pct:.2f}%` of annual treated water",
        "- Transfer sensitivity is encoded as a demand-equivalent proxy, not as a separate physical inflow model.",
        "",
        "## Scenario assumptions",
        "",
        assumptions_df.to_markdown(index=False),
        "",
        "## Projection summary (2026-2040)",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Source anchors",
        "",
        "- `SRC-066`: official Istanbul population and climate-risk framing",
        "- `SRC-067`: official MGM near-term seasonal climate projection framing",
        "- `SRC-068`: Istanbul WEAP study for climate + demand + external-source scenario logic",
        "- `SRC-069`: Thrace ET0 change signal for ET0 scenario magnitudes",
        "- `SRC-070` and `SRC-071`: CMIP6 / downscaling logic for multi-scenario framing",
        "- `SRC-014`, `SRC-057`, `SRC-073`, `SRC-074`: official İSKİ external-transfer and treated-water anchors used for transfer-share sensitivity encoding",
        "",
        "## History-only vs hybrid",
        "",
        "- `history_only_ridge` only uses lagged occupancy plus seasonality.",
        "- `hybrid_ridge` adds rain, ET0, demand, temperature, humidity, VPD, and water-balance signals.",
        "- In this run the hybrid version improves RMSE from `5.35` to `4.30` percentage points.",
        "",
        "## Empirical uncertainty layer",
        "",
        "Monthly lower/upper residual bands from selected walk-forward model:",
        "",
        interval_df.to_markdown(index=False),
        "",
        "## Checkpoints",
        "",
        checkpoint_df.to_markdown(index=False),
        "",
        "## Threshold risk and recovery",
        "",
        risk_df.to_markdown(index=False),
        "",
        "## Driver decomposition",
        "",
        "These rows are model reruns, not strict causal attribution. They show which block moves the path more when isolated from the combined scenario.",
        "",
        decomp_df.to_markdown(index=False),
        "",
        "## Future drivers first",
        "",
        "Future exogenous paths are now charted explicitly before reading storage results.",
        "",
        "2040 checkpoint values:",
        "",
        driver_checkpoint_df[driver_checkpoint_df["checkpoint"] == "2040-12-01"].to_markdown(index=False),
        "",
        "Annual driver summary tail:",
        "",
        annual_driver_df.groupby("scenario").tail(3).to_markdown(index=False),
        "",
        "## Official transfer dependency anchor",
        "",
        transfer_anchor_df.to_markdown(index=False),
        "",
        "## Transfer sensitivity",
        "",
        transfer_summary_df.to_markdown(index=False),
        "",
        "## Parametre duyarlilik izgarlari",
        "",
        "Yagis ve dogrudan talep ayari icin `2040-12` sonu doluluk tablosu:",
        "",
        sensitivity_rain_demand_df.pivot(
            index="rain_end_pct_2040",
            columns="direct_demand_end_pct_2040",
            values="end_fill_2040_12_pct",
        ).sort_index().sort_index(axis=1).to_markdown(),
        "",
        "ET0 ve dis transfer ayari icin `2040-12` sonu doluluk tablosu:",
        "",
        sensitivity_et0_transfer_df.pivot(
            index="et0_end_pct_2040",
            columns="transfer_end_pct_2040",
            values="end_fill_2040_12_pct",
        ).sort_index().sort_index(axis=1).to_markdown(),
        "",
        "Not:",
        "",
        "- `rain vs demand` yuzeyi yon olarak tutarli calisiyor.",
        "- `ET0` tekil izgarasi ise mevcut hibrit modelde isaret-kararliligi problemi gosteriyor; bu tablo karar ciktisindan cok model tanisi olarak okunmali.",
        "- Aktinograf geldikten sonra ET0 blogu yeniden kurularak bu test tekrar calistirilacak.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = OUT_DIR / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    train_df = load_training_frame()
    metrics_df, selected_model, pred_frames = evaluate_models(train_df)
    model = fit_model(train_df, selected_model)
    interval_by_month, global_interval, interval_df = build_empirical_interval_table(pred_frames[selected_model])
    residual_pools, global_pool = build_residual_pools(pred_frames[selected_model])
    clim = monthly_climatology(train_df)
    policy_row, demand_relief_pct_per_1pp_nrw = latest_policy_anchor()
    transfer_anchor_df, transfer_anchor_share_pct = load_transfer_dependency_anchor()

    assumptions = []
    projections = []
    all_scenarios = build_scenarios() + build_decomposition_scenarios() + build_transfer_scenarios()
    for cfg in all_scenarios:
        future_exog = build_future_exog(
            train_df,
            cfg,
            clim,
            demand_relief_pct_per_1pp_nrw,
            transfer_share_anchor_pct=transfer_anchor_share_pct,
        )
        proj = simulate_projection(
            train_df,
            future_exog,
            model,
            selected_model=selected_model,
            interval_by_month=interval_by_month,
            global_interval=global_interval,
        )
        proj = apply_threshold_probabilities(proj, residual_pools=residual_pools, global_pool=global_pool)
        proj["scenario"] = cfg.scenario
        projections.append(proj)
        assumptions.append(
            {
                "scenario": cfg.scenario,
                "rain_end_pct_2040": cfg.rain_end_pct_2040,
                "et0_end_pct_2040": cfg.et0_end_pct_2040,
                "temp_end_c_2040": cfg.temp_end_c_2040,
                "rh_end_pp_2040": cfg.rh_end_pp_2040,
                "vpd_end_pct_2040": cfg.vpd_end_pct_2040,
                "subscriber_growth_pct_per_year": cfg.subscriber_growth_pct_per_year,
                "per_capita_use_pct_per_year": cfg.per_capita_use_pct_per_year,
                "nrw_reduction_pp_by_2040": cfg.nrw_reduction_pp_by_2040,
                "transfer_end_pct_2040": cfg.transfer_end_pct_2040,
                "transfer_demand_equivalent_end_pct": round(
                    -(transfer_anchor_share_pct / 100.0) * cfg.transfer_end_pct_2040, 3
                ),
                "encoded_demand_relief_pct_by_2040": round(
                    cfg.nrw_reduction_pp_by_2040 * demand_relief_pct_per_1pp_nrw, 3
                ),
                "transfer_assumption": cfg.transfer_assumption,
                "source_ids": cfg.source_ids,
                "rationale": cfg.rationale,
            }
        )

    proj_df = pd.concat(projections, ignore_index=True)
    proj_df_2026_2040 = proj_df[(proj_df["date"] >= "2026-01-01") & (proj_df["date"] <= "2040-12-01")].copy()
    assumptions_df = pd.DataFrame(assumptions)
    primary_proj_df = proj_df[proj_df["scenario"].isin(PRIMARY_SCENARIOS)].copy()
    primary_proj_df_2026_2040 = proj_df_2026_2040[proj_df_2026_2040["scenario"].isin(PRIMARY_SCENARIOS)].copy()
    transfer_proj_df_2026_2040 = proj_df_2026_2040[
        proj_df_2026_2040["scenario"].isin(PRIMARY_SCENARIOS + TRANSFER_SENSITIVITY_SCENARIOS)
    ].copy()
    summary_df = build_summary_table(primary_proj_df)
    checkpoint_df = build_checkpoint_table(primary_proj_df_2026_2040)
    risk_df = build_threshold_risk_summary(primary_proj_df_2026_2040)
    decomp_df = build_driver_decomposition(proj_df_2026_2040)
    annual_driver_df = build_future_driver_annual_summary(primary_proj_df_2026_2040)
    driver_checkpoint_df = build_future_driver_checkpoint_summary(primary_proj_df_2026_2040)
    transfer_summary_df = build_transfer_sensitivity_summary(proj_df_2026_2040, assumptions_df)
    sensitivity_rain_demand_df, sensitivity_et0_transfer_df = build_sensitivity_grid_tables(
        train_df,
        clim,
        model,
        selected_model,
        interval_by_month,
        global_interval,
        demand_relief_pct_per_1pp_nrw,
        transfer_anchor_share_pct,
    )

    train_df.to_csv(OUT_DIR / "training_frame_used.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "model_selection_metrics.csv", index=False)
    pred_frames[selected_model].to_csv(OUT_DIR / "selected_model_walkforward_predictions.csv", index=False)
    interval_df.to_csv(OUT_DIR / "selected_model_empirical_interval_by_month.csv", index=False)
    assumptions_df.to_csv(OUT_DIR / "scenario_parameters_2026_2040.csv", index=False)
    transfer_anchor_df.to_csv(OUT_DIR / "official_transfer_dependency_annual_2021_2025.csv", index=False)
    proj_df.to_csv(OUT_DIR / "scenario_projection_monthly_2024_2040.csv", index=False)
    primary_proj_df_2026_2040.to_csv(OUT_DIR / "scenario_projection_monthly_2026_2040.csv", index=False)
    proj_df_2026_2040.to_csv(OUT_DIR / "scenario_projection_monthly_2026_2040_all.csv", index=False)
    transfer_proj_df_2026_2040.to_csv(OUT_DIR / "transfer_sensitivity_projection_2026_2040.csv", index=False)
    summary_df.to_csv(OUT_DIR / "scenario_projection_summary_2026_2040.csv", index=False)
    checkpoint_df.to_csv(OUT_DIR / "scenario_checkpoints_2030_2035_2040.csv", index=False)
    risk_df.to_csv(OUT_DIR / "scenario_threshold_risk_summary_2026_2040.csv", index=False)
    decomp_df.to_csv(OUT_DIR / "scenario_driver_decomposition_2026_2040.csv", index=False)
    annual_driver_df.to_csv(OUT_DIR / "scenario_driver_annual_summary_2026_2040.csv", index=False)
    driver_checkpoint_df.to_csv(OUT_DIR / "scenario_driver_checkpoints_2026_2040.csv", index=False)
    transfer_summary_df.to_csv(OUT_DIR / "transfer_sensitivity_summary_2026_2040.csv", index=False)
    sensitivity_rain_demand_df.to_csv(OUT_DIR / "sensitivity_rain_demand_grid_2040.csv", index=False)
    sensitivity_et0_transfer_df.to_csv(OUT_DIR / "sensitivity_et0_transfer_grid_2040.csv", index=False)

    plot_paths(train_df, primary_proj_df, figs / "scenario_paths_2026_2040.png")
    plot_endpoints(summary_df, figs / "scenario_endpoints_2040.png")
    plot_benchmark(metrics_df, figs / "benchmark_history_vs_hybrid.png")
    plot_threshold_risk(risk_df, figs / "threshold_risk_below_30.png")
    plot_driver_decomposition(decomp_df, figs / "driver_decomposition_2040.png")
    plot_future_driver_paths(primary_proj_df_2026_2040, figs / "future_driver_paths_2026_2040.png")
    plot_future_driver_deltas(driver_checkpoint_df, figs / "future_driver_deltas_2040.png")
    plot_transfer_dependency_history(transfer_anchor_df, figs / "transfer_dependency_history_2021_2025.png")
    plot_transfer_sensitivity_paths(transfer_proj_df_2026_2040, figs / "transfer_sensitivity_paths_2026_2040.png")
    plot_transfer_endpoint_deltas(transfer_summary_df, figs / "transfer_sensitivity_endpoints_2040.png")
    plot_sensitivity_heatmap(
        sensitivity_rain_demand_df,
        "rain_end_pct_2040",
        "direct_demand_end_pct_2040",
        "end_fill_2040_12_pct",
        figs / "sensitivity_heatmap_rain_demand_2040.png",
        title="Yağış ve talep ayarına göre 2040 sonu doluluk",
        xlabel="Doğrudan talep ayarı, 2040 sonu (%)",
        ylabel="Yağış ayarı, 2040 sonu (%)",
    )
    plot_sensitivity_heatmap(
        sensitivity_et0_transfer_df,
        "et0_end_pct_2040",
        "transfer_end_pct_2040",
        "end_fill_2040_12_pct",
        figs / "sensitivity_heatmap_et0_transfer_2040.png",
        title="ET0 ve dış transfer ayarına göre 2040 sonu doluluk",
        xlabel="Dış transfer ayarı, 2040 sonu (%)",
        ylabel="ET0 ayarı, 2040 sonu (%)",
    )

    write_markdown_summary(
        OUT_DIR / "projection_round1_summary.md",
        metrics_df=metrics_df,
        selected_model=selected_model,
        interval_df=interval_df,
        assumptions_df=assumptions_df,
        summary_df=summary_df,
        checkpoint_df=checkpoint_df,
        risk_df=risk_df,
        decomp_df=decomp_df,
        annual_driver_df=annual_driver_df,
        driver_checkpoint_df=driver_checkpoint_df,
        transfer_anchor_df=transfer_anchor_df,
        transfer_anchor_share_pct=transfer_anchor_share_pct,
        transfer_summary_df=transfer_summary_df,
        sensitivity_rain_demand_df=sensitivity_rain_demand_df,
        sensitivity_et0_transfer_df=sensitivity_et0_transfer_df,
        policy_row=policy_row,
        demand_relief_pct_per_1pp_nrw=demand_relief_pct_per_1pp_nrw,
    )

    summary = {
        "selected_model": selected_model,
        "training_start": str(train_df["date"].min().date()),
        "training_end": str(train_df["date"].max().date()),
        "projection_start": "2026-01-01",
        "projection_end": "2040-12-01",
        "policy_anchor_year": int(policy_row["year"]),
        "demand_relief_pct_per_1pp_nrw": demand_relief_pct_per_1pp_nrw,
        "transfer_anchor_share_pct_2021_2025_mean": transfer_anchor_share_pct,
        "actinograph_status": "pending_integration",
        "source_ids": ["SRC-014", "SRC-057", "SRC-066", "SRC-067", "SRC-068", "SRC-069", "SRC-070", "SRC-071", "SRC-073", "SRC-074"],
    }
    (OUT_DIR / "projection_round1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(OUT_DIR)


if __name__ == "__main__":
    main()
