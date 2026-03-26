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
BENCHMARK_SCORECARD_PATH = ROOT / "output" / "istanbul_forward_model_benchmark_round2" / "model_selection_scorecard.csv"
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
OUT_DIR = ROOT / "output" / "istanbul_water_balance_v2_2040"

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
    "catchment_rain_now_mcm",
    "catchment_rain_lag1_mcm",
    "catchment_wetness_mcm",
    "lake_rain_mcm",
    "neg_openwater_evap_mcm",
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


def compute_system_context() -> dict[str, float]:
    src = pd.read_csv(SOURCE_CONTEXT_PATH)
    baraj = src[src["source_group"].eq("baraj")].copy()
    total_storage_mcm = float(baraj["max_storage_million_m3"].sum())
    known = baraj[baraj["normal_lake_area_km2"].notna()].iloc[0]
    area_coeff = float(known["normal_lake_area_km2"] / (known["max_storage_million_m3"] ** (2.0 / 3.0)))
    baraj["est_lake_area_km2"] = area_coeff * (baraj["max_storage_million_m3"] ** (2.0 / 3.0))
    total_lake_area_km2 = float(baraj["est_lake_area_km2"].sum())
    total_basin_area_km2 = float(baraj["basin_area_km2"].fillna(0.0).sum())
    return {
        "total_storage_mcm": total_storage_mcm,
        "total_lake_area_km2": total_lake_area_km2,
        "total_basin_area_km2": total_basin_area_km2,
        "land_basin_area_km2": max(total_basin_area_km2 - total_lake_area_km2, 0.0),
        "lake_area_coeff": area_coeff,
    }


def effective_lake_area(fill: float, context: dict[str, float]) -> float:
    return context["total_lake_area_km2"] * float(np.clip(fill, 0.0, 1.0) ** AREA_EXPONENT)


def load_training_frame(context: dict[str, float]) -> pd.DataFrame:
    core = pd.read_csv(CORE_PATH, parse_dates=["date"])
    supply = pd.read_csv(SUPPLY_PATH, parse_dates=["date"])
    df = core.merge(supply[["date", "city_supply_m3_month_official"]], on="date", how="left")
    df["days_in_month"] = df["date"].dt.days_in_month
    df["supply_mcm"] = df["city_supply_m3_month_official"] / 1e6
    fallback = df["consumption_mean_monthly"] * df["days_in_month"] / 1e6
    df["supply_mcm"] = df["supply_mcm"].fillna(fallback)
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
    df = df[df["date"] >= "2011-01-01"].reset_index(drop=True)
    return df


def component_frame(df: pd.DataFrame, context: dict[str, float]) -> pd.DataFrame:
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
        ]
    ].copy()
    out["lake_area_km2"] = out["weighted_total_fill_lag1"].apply(lambda x: effective_lake_area(x, context))
    out["land_area_km2"] = np.maximum(context["total_basin_area_km2"] - out["lake_area_km2"], 0.0)
    out["catchment_rain_now_mcm"] = out["land_area_km2"] * out["rain_model_mm"] * 0.001
    out["catchment_rain_lag1_mcm"] = out["land_area_km2"] * out["rain_model_mm_lag1"] * 0.001
    wetness_mm = np.clip(out["rain_model_mm_roll3"] - out["et0_mm_month_roll3"], 0.0, None)
    out["catchment_wetness_mcm"] = out["land_area_km2"] * wetness_mm * 0.001
    out["lake_rain_mcm"] = out["lake_area_km2"] * out["rain_model_mm"] * 0.001
    out["openwater_evap_mcm"] = out["lake_area_km2"] * out["et0_mm_month"] * 0.001
    out["storage_mass_mcm"] = out["weighted_total_fill_lag1"] * context["total_storage_mcm"]
    out["spill_pressure_mcm"] = np.clip(out["weighted_total_fill_lag1"] - SPILL_THRESHOLD, 0.0, None) * context["total_storage_mcm"]
    out["neg_openwater_evap_mcm"] = -out["openwater_evap_mcm"]
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
        lake_area_km2 = effective_lake_area(fill_prev, context)
        land_area_km2 = max(context["total_basin_area_km2"] - lake_area_km2, 0.0)
        rain_now = float(row["rain_model_mm"])
        rain_lag1 = float(past_rain[-1])
        et0_now = float(row["et0_mm_month"])
        rain_roll3 = float(np.mean([past_rain[-2], past_rain[-1], rain_now]))
        et0_roll3 = float(np.mean([past_et0[-2], past_et0[-1], et0_now]))
        baseline_transfer_mcm = supply_mcm * (baseline_transfer_share_pct / 100.0)
        transfer_delta_mcm = supply_mcm * (transfer_share_anchor_pct / 100.0) * (transfer_end_pct_2040 / 100.0) * float(progress[idx])
        total_transfer_mcm = baseline_transfer_mcm + transfer_delta_mcm

        comp = {
            "catchment_rain_now_mcm": land_area_km2 * rain_now * 0.001,
            "catchment_rain_lag1_mcm": land_area_km2 * rain_lag1 * 0.001,
            "catchment_wetness_mcm": land_area_km2 * max(rain_roll3 - et0_roll3, 0.0) * 0.001,
            "lake_rain_mcm": lake_area_km2 * rain_now * 0.001,
            "openwater_evap_mcm": lake_area_km2 * et0_now * 0.001,
            "supply_mcm": supply_mcm,
            "storage_mass_mcm": fill_prev * context["total_storage_mcm"],
            "spill_pressure_mcm": max(fill_prev - SPILL_THRESHOLD, 0.0) * context["total_storage_mcm"],
        }
        comp["neg_openwater_evap_mcm"] = -comp["openwater_evap_mcm"]
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
                "catchment_rain_now_mcm": float(comp["catchment_rain_now_mcm"]),
                "catchment_rain_lag1_mcm": float(comp["catchment_rain_lag1_mcm"]),
                "catchment_wetness_mcm": float(comp["catchment_wetness_mcm"]),
                "lake_rain_mcm": float(comp["lake_rain_mcm"]),
                "openwater_evap_mcm": float(comp["openwater_evap_mcm"]),
                "storage_mass_mcm": float(comp["storage_mass_mcm"]),
                "spill_pressure_mcm": float(comp["spill_pressure_mcm"]),
                "rain_model_mm": rain_now,
                "et0_mm_month": et0_now,
            }
        )
        past_fill.append(fill_next)
        past_rain.append(rain_now)
        past_et0.append(et0_now)
    return pd.DataFrame(rows)


def one_step_walkforward(df: pd.DataFrame, context: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = []
    for i in range(MIN_TRAIN, len(df)):
        train = df.iloc[:i].copy()
        test = df.iloc[[i]].copy()
        train_comp = component_frame(train, context)
        test_comp = component_frame(test, context)
        model, month_bias, _ = fit_water_balance_model(train_comp)
        row = test_comp.iloc[0]
        delta_pred_mcm = predict_delta_mcm(row.to_dict(), int(pd.Timestamp(row["date"]).month), model, month_bias)
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
                "model": "water_balance_v2",
                "one_step_rmse_pp": rmse_pp,
                "one_step_mae_pp": mae_pp,
                "delta_direction_accuracy": direction_accuracy,
                "n_predictions": int(len(pred_df)),
            }
        ]
    )
    return metrics, pred_df


def recursive_backtest(df: pd.DataFrame, context: dict[str, float], horizons: tuple[int, ...] = (1, 3, 6, 12)) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        actual = []
        pred = []
        origin_fill = []
        for i in range(MIN_TRAIN, len(df) - horizon + 1):
            train = df.iloc[:i].copy()
            future = df.iloc[i : i + horizon].copy()
            train_comp = component_frame(train, context)
            model, month_bias, _ = fit_water_balance_model(train_comp)
            sim = simulate_path(
                train,
                future[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
                model,
                month_bias,
                context,
                transfer_share_anchor_pct=0.0,
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
                "model": "water_balance_v2",
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
    out["contrib_catchment_rain_now_mcm"] = coef_map["catchment_rain_now_mcm"] * out["catchment_rain_now_mcm"]
    out["contrib_catchment_rain_lag1_mcm"] = coef_map["catchment_rain_lag1_mcm"] * out["catchment_rain_lag1_mcm"]
    out["contrib_catchment_wetness_mcm"] = coef_map["catchment_wetness_mcm"] * out["catchment_wetness_mcm"]
    out["contrib_lake_rain_mcm"] = coef_map["lake_rain_mcm"] * out["lake_rain_mcm"]
    out["contrib_openwater_evap_mcm"] = coef_map["neg_openwater_evap_mcm"] * (-out["openwater_evap_mcm"])
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
    model, month_bias, _ = fit_water_balance_model(train_comp)
    clim = forward.monthly_climatology(df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    all_cfgs = forward.build_scenarios() + forward.build_transfer_scenarios()
    parts = []
    for cfg in all_cfgs:
        neutral_cfg = replace(cfg, transfer_end_pct_2040=0.0)
        future = forward.build_future_exog(df, neutral_cfg, clim, demand_relief, transfer_share_anchor_pct=0.0)
        sim = simulate_path(
            df,
            future[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
            model,
            month_bias,
            context,
            transfer_share_anchor_pct=transfer_share_anchor_pct,
            baseline_transfer_share_pct=transfer_share_anchor_pct,
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
                    "openwater_evap_mcm": float(idx.loc[cp, "openwater_evap_mcm"]),
                    "baseline_transfer_mcm": float(idx.loc[cp, "baseline_transfer_mcm"]),
                    "transfer_delta_mcm": float(idx.loc[cp, "transfer_delta_mcm"]),
                    "total_transfer_mcm": float(idx.loc[cp, "total_transfer_mcm"]),
                    "catchment_rain_now_mcm": float(idx.loc[cp, "catchment_rain_now_mcm"]),
                }
            )
    checkpoints_df = pd.DataFrame(checkpoints)
    return proj, primary, summary_df, checkpoints_df


def physical_sanity(context: dict[str, float]) -> pd.DataFrame:
    forward = load_forward_module()
    df = load_training_frame(context)
    train_comp = component_frame(df, context)
    model, month_bias, _ = fit_water_balance_model(train_comp)
    clim = forward.monthly_climatology(df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    base_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == "base")
    future_base = forward.build_future_exog(df, replace(base_cfg, transfer_end_pct_2040=0.0), clim, demand_relief, transfer_share_anchor_pct=0.0)
    future_base = future_base[future_base["date"].between(pd.Timestamp("2026-01-01"), pd.Timestamp("2040-12-01"))].copy()
    base_sim = simulate_path(
        df,
        future_base[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
        model,
        month_bias,
        context,
        transfer_share_anchor_pct=transfer_share_anchor_pct,
        baseline_transfer_share_pct=transfer_share_anchor_pct,
        transfer_end_pct_2040=0.0,
    )
    endpoint_base = float(base_sim.iloc[-1]["pred_fill"] * 100.0)

    scenario_inputs = []
    adj = future_base.copy()
    adj["rain_model_mm"] *= 1.10
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
            future[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
            model,
            month_bias,
            context,
            transfer_share_anchor_pct=transfer_share_anchor_pct,
            baseline_transfer_share_pct=transfer_share_anchor_pct,
            transfer_end_pct_2040=transfer_pct,
        )
        rows.append(
            {
                "model": "water_balance_v2",
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
    ax.plot(pred_df["date"], pred_df["pred_fill"] * 100.0, color="#2563eb", linewidth=1.8, label="Su bütçesi v2")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.set_title("Tek adımlı geri test: su bütçesi v2")
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
    ax.set_title("Su bütçesi v2 ile 2026-2040 projeksiyon yolları")
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
            inflow_contrib_mcm=("contrib_catchment_rain_now_mcm", "sum"),
            inflow_lag_contrib_mcm=("contrib_catchment_rain_lag1_mcm", "sum"),
            wetness_contrib_mcm=("contrib_catchment_wetness_mcm", "sum"),
            lake_rain_contrib_mcm=("contrib_lake_rain_mcm", "sum"),
            openwater_contrib_mcm=("contrib_openwater_evap_mcm", "sum"),
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
    ax.set_title("Temel senaryoda yıllık katkı bileşenleri")
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
    model, month_bias, _ = fit_water_balance_model(comp)

    one_step_metrics, pred_df = one_step_walkforward(df, context)
    recursive_df = recursive_backtest(df, context)
    coeff_df = build_component_coefficients(model, month_bias, context)
    scenario_all, scenario_primary, scenario_summary, checkpoints_df = scenario_projection(context)
    sanity_df = physical_sanity(context)
    contrib_df = build_component_contribution_table(scenario_all, model, month_bias)

    benchmark = pd.read_csv(BENCHMARK_SCORECARD_PATH)
    benchmark = benchmark[["model", "one_step_rmse_pp", "mean_recursive_rmse_pp", "physics_pass_count"]]
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
            pd.DataFrame(
                [
                    {
                        "model": "water_balance_v2",
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
        "model": "water_balance_v2",
        "one_step_rmse_pp": float(one_step_metrics.iloc[0]["one_step_rmse_pp"]),
        "mean_recursive_rmse_pp": wb_recursive_mean,
        "physics_passes": physics_pass_count,
        "system_context": context,
        "notes": [
            "Demand uses official monthly city supply when available and proxy fallback otherwise.",
            "Transfer is modeled as a baseline external inflow anchored to the recent official transfer-share average, plus optional scenario deviation.",
            "NRW is not separately observed monthly; it remains embedded in supply and future demand adjustments.",
        ],
    }
    (OUT_DIR / "water_balance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
