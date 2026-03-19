#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/yasinkaya/Hackhaton")
BENCHMARK_SCRIPT = ROOT / "scripts" / "benchmark_istanbul_forward_models.py"
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"
WB_SCRIPT = ROOT / "scripts" / "build_istanbul_water_balance_v4_sourceaware.py"
OUT_DIR = ROOT / "output" / "istanbul_top3_developed_plus_external_2026_03_12"

TRAIN_END = pd.Timestamp("2015-12-01")
TEST_START = pd.Timestamp("2016-01-01")
TEST_END = pd.Timestamp("2020-12-01")
FUTURE_START = pd.Timestamp("2026-01-01")
FUTURE_END = pd.Timestamp("2040-12-01")
STACK_MIN_TRAIN = 36

BASE_FEATURES = [
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

ENHANCED_EXTRA = [
    "rain_et0_ratio",
    "rain_roll3_minus_et0_roll3",
    "fill_change_lag",
    "fill_rain_interaction",
    "fill_et0_interaction",
    "fill_cons_interaction",
    "dry_balance_flag",
    "wet_season_flag",
    "warm_season_flag",
    "trend_index",
]

ENHANCED_FEATURES = BASE_FEATURES + ENHANCED_EXTRA
PROPHET_REGRESSORS = [
    "y_lag1",
    "y_lag2",
    "rain_model_mm",
    "et0_mm_month",
    "consumption_mean_monthly",
    "temp_proxy_c",
    "vpd_kpa_mean",
]
WB_CORR_FEATURES = [
    "weighted_total_fill_lag1",
    "rain_model_mm",
    "rain_model_mm_roll3",
    "et0_mm_month",
    "supply_mcm",
    "source_runoff_now_mcm",
    "source_runoff_wetness_mcm",
    "source_lake_rain_mcm",
    "source_openwater_evap_mcm",
    "spill_pressure_mcm",
    "month_sin",
    "month_cos",
]

MODEL_LABELS = {
    "hybrid_ridge_plus": "Hibrit Ridge+",
    "water_balance_v4_corrected": "Water Balance v4+",
    "hybrid_physics_stacked_plus": "Stacked Hybrid+",
    "quantile_regressor_plus": "Quantile Reg.",
    "prophet_regressor_plus": "Prophet",
    "hybrid_physics_ensemble_phys_old": "Eski seçilen ensemble",
}

MODEL_COLORS = {
    "hybrid_ridge_plus": "#2563eb",
    "water_balance_v4_corrected": "#dc2626",
    "hybrid_physics_stacked_plus": "#059669",
    "quantile_regressor_plus": "#7c3aed",
    "prophet_regressor_plus": "#d97706",
    "hybrid_physics_ensemble_phys_old": "#111827",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def month_sin(month: int) -> float:
    return float(np.sin(2.0 * np.pi * month / 12.0))


def month_cos(month: int) -> float:
    return float(np.cos(2.0 * np.pi * month / 12.0))


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rain_et0_ratio"] = out["rain_model_mm"] / (out["et0_mm_month"] + 1.0)
    out["rain_roll3_minus_et0_roll3"] = out["rain_model_mm_roll3"] - out["et0_mm_month_roll3"]
    out["fill_change_lag"] = out["weighted_total_fill_lag1"] - out["weighted_total_fill_lag2"]
    out["fill_rain_interaction"] = out["weighted_total_fill_lag1"] * out["rain_model_mm"]
    out["fill_et0_interaction"] = out["weighted_total_fill_lag1"] * out["et0_mm_month"]
    out["fill_cons_interaction"] = out["weighted_total_fill_lag1"] * out["consumption_mean_monthly"]
    out["dry_balance_flag"] = (out["water_balance_proxy_mm"] < 0.0).astype(int)
    month = pd.to_datetime(out["date"]).dt.month
    out["wet_season_flag"] = month.isin([11, 12, 1, 2, 3, 4]).astype(int)
    out["warm_season_flag"] = month.isin([5, 6, 7, 8, 9, 10]).astype(int)
    out["trend_index"] = np.arange(len(out), dtype=float)
    return out


def build_feature_row(past_fill: list[float], past_rain: list[float], past_et0: list[float], past_cons: list[float], row, step_idx: int) -> dict[str, float]:
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
    feat["rain_et0_ratio"] = feat["rain_model_mm"] / (feat["et0_mm_month"] + 1.0)
    feat["rain_roll3_minus_et0_roll3"] = feat["rain_model_mm_roll3"] - feat["et0_mm_month_roll3"]
    feat["fill_change_lag"] = feat["weighted_total_fill_lag1"] - feat["weighted_total_fill_lag2"]
    feat["fill_rain_interaction"] = feat["weighted_total_fill_lag1"] * feat["rain_model_mm"]
    feat["fill_et0_interaction"] = feat["weighted_total_fill_lag1"] * feat["et0_mm_month"]
    feat["fill_cons_interaction"] = feat["weighted_total_fill_lag1"] * feat["consumption_mean_monthly"]
    feat["dry_balance_flag"] = float(feat["water_balance_proxy_mm"] < 0.0)
    date = pd.Timestamp(row["date"])
    feat["wet_season_flag"] = float(date.month in [11, 12, 1, 2, 3, 4])
    feat["warm_season_flag"] = float(date.month in [5, 6, 7, 8, 9, 10])
    feat["trend_index"] = float(step_idx)
    return feat


def fit_hybrid_plus(train_df: pd.DataFrame):
    model = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-4, 4, 41)))])
    model.fit(train_df[ENHANCED_FEATURES], train_df["delta_fill"])
    return model


def fit_quantile_plus(train_df: pd.DataFrame):
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.5,
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.9,
        random_state=42,
    )
    model.fit(train_df[ENHANCED_FEATURES], train_df["delta_fill"])
    return model


def recursive_predict_delta_model(model, history_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
    past_fill = history_df["weighted_total_fill"].tolist()
    past_rain = history_df["rain_model_mm"].tolist()
    past_et0 = history_df["et0_mm_month"].tolist()
    past_cons = history_df["consumption_mean_monthly"].tolist()
    preds = []
    step_offset = len(history_df)
    for i, row in enumerate(future_df.itertuples(index=False)):
        feat = build_feature_row(past_fill, past_rain, past_et0, past_cons, row._asdict(), step_idx=step_offset + i)
        delta_hat = float(model.predict(pd.DataFrame([{k: feat[k] for k in ENHANCED_FEATURES}]))[0])
        yhat = float(np.clip(past_fill[-1] + delta_hat, 0.0, 1.0))
        preds.append(yhat)
        past_fill.append(yhat)
        past_rain.append(float(row.rain_model_mm))
        past_et0.append(float(row.et0_mm_month))
        past_cons.append(float(row.consumption_mean_monthly))
    return np.asarray(preds, dtype=float)


def prophet_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"date": "ds", "weighted_total_fill": "y"})
    out["y_lag1"] = out["weighted_total_fill_lag1"]
    out["y_lag2"] = out["weighted_total_fill_lag2"]
    return out[["ds", "y"] + PROPHET_REGRESSORS].copy()


def fit_prophet(train_df: pd.DataFrame) -> Prophet:
    prop = Prophet(
        growth="flat",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.03,
        seasonality_mode="multiplicative",
    )
    for reg in PROPHET_REGRESSORS:
        prop.add_regressor(reg, standardize=True)
    prop.fit(prophet_train_frame(train_df))
    return prop


def recursive_predict_prophet(model: Prophet, history_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
    past_fill = history_df["weighted_total_fill"].tolist()
    preds = []
    for row in future_df.itertuples(index=False):
        frame = pd.DataFrame(
            [
                {
                    "ds": pd.Timestamp(row.date),
                    "y_lag1": past_fill[-1],
                    "y_lag2": past_fill[-2],
                    "rain_model_mm": float(row.rain_model_mm),
                    "et0_mm_month": float(row.et0_mm_month),
                    "consumption_mean_monthly": float(row.consumption_mean_monthly),
                    "temp_proxy_c": float(row.temp_proxy_c),
                    "vpd_kpa_mean": float(row.vpd_kpa_mean),
                }
            ]
        )
        yhat = float(np.clip(model.predict(frame)["yhat"].iloc[0], 0.0, 1.0))
        preds.append(yhat)
        past_fill.append(yhat)
    return np.asarray(preds, dtype=float)


def fit_wb_corrected(wb, train_df: pd.DataFrame, context: dict[str, object], share_by_year: dict[int, float]):
    train_comp = wb.component_frame(train_df, context).copy()
    train_comp["month_sin"] = train_comp["date"].dt.month.map(month_sin)
    train_comp["month_cos"] = train_comp["date"].dt.month.map(month_cos)
    base_model, month_bias, fit_df = wb.fit_water_balance_model(train_comp)
    transfer_effectiveness = wb.estimate_transfer_effectiveness(train_df, fit_df, share_by_year)
    fit_df = fit_df.copy()
    fit_df["hist_transfer_mcm"] = [
        wb.historical_transfer_addition(pd.Timestamp(d), float(s), share_by_year, transfer_effectiveness)
        for d, s in zip(train_df["date"], train_df["supply_mcm"])
    ]
    residual_target = fit_df["delta_storage_mcm"] - (fit_df["pred_mcm"] + fit_df["hist_transfer_mcm"])
    corr_model = Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-4, 4, 31)))])
    corr_model.fit(train_comp[WB_CORR_FEATURES], residual_target)
    return base_model, month_bias, transfer_effectiveness, corr_model


def recursive_predict_wb_corrected(
    wb,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    context: dict[str, object],
    base_model,
    month_bias: dict[int, float],
    transfer_effectiveness: float,
    corr_model,
    share_by_year: dict[int, float],
    baseline_transfer_share_pct: float = 0.0,
    transfer_end_pct_2040: float = 0.0,
    transfer_share_anchor_pct: float = 0.0,
) -> np.ndarray:
    past_fill = history_df["weighted_total_fill"].tolist()
    past_rain = history_df["rain_model_mm"].tolist()
    past_et0 = history_df["et0_mm_month"].tolist()
    horizon = len(future_df)
    progress = np.linspace(0.0, 1.0, horizon) if horizon else np.array([])
    preds = []
    total_storage = float(context["total_storage_mcm"])
    for idx, row in enumerate(future_df.itertuples(index=False)):
        fill_prev = float(past_fill[-1])
        date = pd.Timestamp(row.date)
        days = int(date.days_in_month)
        supply_mcm = float(row.consumption_mean_monthly * days / 1e6)
        state = wb.source_state(fill_prev, context)
        lake_area_km2 = float(state["lake_area_km2"].sum())
        runoff_weighted_land_area_km2 = float(np.dot(state["land_area_km2"], state["runoff_productivity_weight"]))
        rain_now = float(row.rain_model_mm)
        rain_lag1 = float(past_rain[-1])
        et0_now = float(row.et0_mm_month)
        rain_roll3 = float(np.mean([past_rain[-2], past_rain[-1], rain_now]))
        et0_roll3 = float(np.mean([past_et0[-2], past_et0[-1], et0_now]))
        comp = {
            "source_runoff_now_mcm": runoff_weighted_land_area_km2 * rain_now * 0.001,
            "source_runoff_lag1_mcm": runoff_weighted_land_area_km2 * rain_lag1 * 0.001,
            "source_runoff_wetness_mcm": runoff_weighted_land_area_km2 * max(rain_roll3 - et0_roll3, 0.0) * 0.001,
            "source_lake_rain_mcm": lake_area_km2 * rain_now * 0.001,
            "source_openwater_evap_mcm": lake_area_km2 * et0_now * 0.001,
            "supply_mcm": supply_mcm,
            "storage_mass_mcm": float(state["storage_mcm"].sum()),
            "spill_pressure_mcm": float(state["spill_pressure_mcm"].sum()),
        }
        comp["neg_source_openwater_evap_mcm"] = -comp["source_openwater_evap_mcm"]
        comp["neg_supply_mcm"] = -comp["supply_mcm"]
        comp["neg_storage_mass_mcm"] = -comp["storage_mass_mcm"]
        comp["neg_spill_pressure_mcm"] = -comp["spill_pressure_mcm"]
        base_delta = wb.predict_delta_mcm(comp, date.month, base_model, month_bias)

        corr_feats = {
            "weighted_total_fill_lag1": fill_prev,
            "rain_model_mm": rain_now,
            "rain_model_mm_roll3": rain_roll3,
            "et0_mm_month": et0_now,
            "supply_mcm": supply_mcm,
            "source_runoff_now_mcm": comp["source_runoff_now_mcm"],
            "source_runoff_wetness_mcm": comp["source_runoff_wetness_mcm"],
            "source_lake_rain_mcm": comp["source_lake_rain_mcm"],
            "source_openwater_evap_mcm": comp["source_openwater_evap_mcm"],
            "spill_pressure_mcm": comp["spill_pressure_mcm"],
            "month_sin": month_sin(date.month),
            "month_cos": month_cos(date.month),
        }
        corr_delta = float(corr_model.predict(pd.DataFrame([corr_feats]))[0])
        baseline_transfer_mcm = supply_mcm * (baseline_transfer_share_pct / 100.0) * transfer_effectiveness
        transfer_delta_mcm = supply_mcm * (transfer_share_anchor_pct / 100.0) * (transfer_end_pct_2040 / 100.0) * float(progress[idx])
        total_delta = base_delta + corr_delta + baseline_transfer_mcm + transfer_delta_mcm
        next_storage = np.clip(fill_prev * total_storage + total_delta, 0.0, total_storage)
        fill_next = float(next_storage / total_storage)
        preds.append(fill_next)
        past_fill.append(fill_next)
        past_rain.append(rain_now)
        past_et0.append(et0_now)
    return np.asarray(preds, dtype=float)


def fit_stacker(train_df: pd.DataFrame, pred_h: np.ndarray, pred_w: np.ndarray) -> LinearRegression:
    work = train_df.iloc[-len(pred_h):].copy()
    X = pd.DataFrame(
        {
            "pred_h": pred_h,
            "pred_w": pred_w,
            "month_sin": work["month_sin"].to_numpy(dtype=float),
            "month_cos": work["month_cos"].to_numpy(dtype=float),
        }
    )
    y = work["weighted_total_fill"].to_numpy(dtype=float)
    model = LinearRegression(positive=True)
    model.fit(X, y)
    return model


def internal_stacking_dataset(
    train_df: pd.DataFrame,
    train_wb_df: pd.DataFrame,
    wb,
    context: dict[str, object],
    share_by_year: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    preds_h = []
    preds_w = []
    actual_rows = []
    for idx in range(STACK_MIN_TRAIN, len(train_df)):
        tr = train_df.iloc[:idx].copy()
        tr_wb = train_wb_df[train_wb_df["date"].isin(tr["date"])].copy().reset_index(drop=True)
        te = train_df.iloc[[idx]].copy()
        h_model = fit_hybrid_plus(tr)
        pred_h = recursive_predict_delta_model(h_model, tr, te)[0]
        wb_base, wb_bias, wb_transfer_eff, wb_corr = fit_wb_corrected(wb, tr_wb, context, share_by_year)
        pred_w = recursive_predict_wb_corrected(
            wb,
            tr_wb,
            te[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
            context,
            wb_base,
            wb_bias,
            wb_transfer_eff,
            wb_corr,
            share_by_year,
        )[0]
        preds_h.append(pred_h)
        preds_w.append(pred_w)
        actual_rows.append(te.iloc[0])
    actual_df = pd.DataFrame(actual_rows).reset_index(drop=True)
    return np.asarray(preds_h, dtype=float), np.asarray(preds_w, dtype=float), actual_df


def evaluate_model_set() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bench = load_module(BENCHMARK_SCRIPT, "benchmark_dev_top3")
    forward = load_module(FORWARD_SCRIPT, "forward_dev_top3")
    wb = load_module(WB_SCRIPT, "wb_dev_top3")

    base_df = bench.load_training_frame().copy()
    base_df = add_enhanced_features(base_df)
    train = base_df[base_df["date"] <= TRAIN_END].copy().reset_index(drop=True)
    test = base_df[(base_df["date"] >= TEST_START) & (base_df["date"] <= TEST_END)].copy().reset_index(drop=True)

    h_model = fit_hybrid_plus(train)
    pred_h = recursive_predict_delta_model(h_model, train, test)

    q_model = fit_quantile_plus(train)
    pred_q = recursive_predict_delta_model(q_model, train, test)

    p_model = fit_prophet(train)
    pred_p = recursive_predict_prophet(p_model, train, test)

    context = wb.compute_system_context()
    wb_df = wb.load_training_frame(context).copy()
    wb_df = wb_df[wb_df["date"].isin(base_df["date"])].sort_values("date").reset_index(drop=True)
    train_wb = wb_df[wb_df["date"] <= TRAIN_END].copy().reset_index(drop=True)
    test_wb = wb_df[(wb_df["date"] >= TEST_START) & (wb_df["date"] <= TEST_END)].copy().reset_index(drop=True)
    share_by_year, _ = wb.load_transfer_share_by_year()
    wb_base, wb_bias, wb_transfer_eff, wb_corr = fit_wb_corrected(wb, train_wb, context, share_by_year)
    pred_w = recursive_predict_wb_corrected(
        wb,
        train_wb,
        test_wb,
        context,
        wb_base,
        wb_bias,
        wb_transfer_eff,
        wb_corr,
        share_by_year,
    )

    # old selected ensemble for direct comparison
    pred_old = 0.45 * recursive_predict_delta_model(fit_hybrid_plus(train.assign()), train, test)  # placeholder replaced below
    # Use old original components instead of enhanced hybrid
    orig_train = bench.load_training_frame().copy()
    orig_train = orig_train[orig_train["date"] <= TRAIN_END].copy().reset_index(drop=True)
    orig_test = bench.load_training_frame().copy()
    orig_test = orig_test[(orig_test["date"] >= TEST_START) & (orig_test["date"] <= TEST_END)].copy().reset_index(drop=True)
    orig_h_model = bench.fit_model(bench.ModelSpec("hybrid_ridge", bench.FEATURES, "ridge"), orig_train)
    orig_h_pred = bench.recursive_forecast_known_exog(bench.ModelSpec("hybrid_ridge", bench.FEATURES, "ridge"), orig_h_model, orig_train, orig_test)
    old_wb_pred = pred_w  # corrected wb is closer; for old selected compare use raw wb v4
    wb_base_old, wb_bias_old, wb_transfer_eff_old, _ = fit_wb_corrected(wb, train_wb, context, share_by_year)
    old_wb_raw = wb.simulate_path(
        history_df=train_wb,
        future_exog=test_wb[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
        model=wb_base_old,
        month_bias=wb_bias_old,
        context=context,
        transfer_share_anchor_pct=0.0,
        transfer_effectiveness=wb_transfer_eff_old,
        baseline_transfer_share_pct=0.0,
        transfer_end_pct_2040=0.0,
    )["pred_fill"].to_numpy(dtype=float)
    pred_old = 0.45 * orig_h_pred + 0.55 * old_wb_raw

    stack_h, stack_w, stack_actual = internal_stacking_dataset(train, train_wb, wb, context, share_by_year)
    stack_model = fit_stacker(stack_actual, stack_h, stack_w)
    stack_X_test = pd.DataFrame(
        {
            "pred_h": pred_h,
            "pred_w": pred_w,
            "month_sin": test["month_sin"].to_numpy(dtype=float),
            "month_cos": test["month_cos"].to_numpy(dtype=float),
        }
    )
    pred_stack = np.clip(stack_model.predict(stack_X_test), 0.0, 1.0)

    pred_df = pd.DataFrame(
        {
            "date": test["date"],
            "actual_fill": test["weighted_total_fill"],
            "hybrid_ridge_plus": pred_h,
            "water_balance_v4_corrected": pred_w,
            "hybrid_physics_stacked_plus": pred_stack,
            "quantile_regressor_plus": pred_q,
            "prophet_regressor_plus": pred_p,
            "hybrid_physics_ensemble_phys_old": pred_old,
        }
    )
    stack_meta = pd.DataFrame(
        {
            "feature": ["pred_h", "pred_w", "month_sin", "month_cos", "intercept"],
            "value": list(stack_model.coef_) + [float(stack_model.intercept_)],
        }
    )
    return pred_df, stack_meta, pd.DataFrame(
        {
            "date": stack_actual["date"],
            "actual_fill": stack_actual["weighted_total_fill"],
            "pred_h": stack_h,
            "pred_w": stack_w,
        }
    )


def metric_row(model_name: str, actual: np.ndarray, pred: np.ndarray) -> dict[str, float | str]:
    pearson = float(pearsonr(actual, pred).statistic)
    spearman = float(spearmanr(actual, pred).statistic)
    return {
        "model": model_name,
        "rmse_pp": float(np.sqrt(mean_squared_error(actual, pred)) * 100.0),
        "mae_pp": float(mean_absolute_error(actual, pred) * 100.0),
        "mape_pct": float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-6)) * 100.0),
        "smape_pct": float(np.mean(2.0 * np.abs(pred - actual) / np.maximum(np.abs(actual) + np.abs(pred), 1e-6)) * 100.0),
        "pearson_corr_pct": pearson * 100.0,
        "spearman_corr_pct": spearman * 100.0,
        "r2": float(r2_score(actual, pred)),
        "end_error_pp_2020_12": float((pred[-1] - actual[-1]) * 100.0),
    }


def summarize_holdout(pred_df: pd.DataFrame) -> pd.DataFrame:
    actual = pred_df["actual_fill"].to_numpy(dtype=float)
    rows = []
    for col in [c for c in pred_df.columns if c not in {"date", "actual_fill"}]:
        rows.append(metric_row(col, actual, pred_df[col].to_numpy(dtype=float)))
    return pd.DataFrame(rows).sort_values(["mape_pct", "rmse_pp"]).reset_index(drop=True)


def build_future_base_paths(stack_meta: pd.DataFrame) -> pd.DataFrame:
    bench = load_module(BENCHMARK_SCRIPT, "benchmark_future_dev_top3")
    forward = load_module(FORWARD_SCRIPT, "forward_future_dev_top3")
    wb = load_module(WB_SCRIPT, "wb_future_dev_top3")

    base_df = add_enhanced_features(bench.load_training_frame().copy())
    train = base_df.copy().reset_index(drop=True)
    clim = forward.monthly_climatology(train)
    _, demand_relief_pct = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    base_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == "base")
    future_exog = forward.build_future_exog(train, base_cfg, clim, demand_relief_pct, transfer_share_anchor_pct=transfer_share_anchor_pct)
    future_exog = future_exog[(future_exog["date"] >= FUTURE_START) & (future_exog["date"] <= FUTURE_END)].copy().reset_index(drop=True)
    future_exog["trend_index"] = np.arange(len(train), len(train) + len(future_exog), dtype=float)

    pred_h = recursive_predict_delta_model(fit_hybrid_plus(train), train, future_exog)
    pred_q = recursive_predict_delta_model(fit_quantile_plus(train), train, future_exog)
    pred_p = recursive_predict_prophet(fit_prophet(train), train, future_exog)

    context = wb.compute_system_context()
    wb_df = wb.load_training_frame(context).copy()
    wb_df = wb_df[wb_df["date"].isin(train["date"])].sort_values("date").reset_index(drop=True)
    share_by_year, anchor_share_pct = wb.load_transfer_share_by_year()
    wb_base, wb_bias, wb_transfer_eff, wb_corr = fit_wb_corrected(wb, wb_df, context, share_by_year)
    pred_w = recursive_predict_wb_corrected(
        wb,
        wb_df,
        future_exog[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]].merge(
            future_exog[["date", "month_sin", "month_cos"]], on="date", how="left"
        ).merge(future_exog[["date"]], on="date", how="left"),
        context,
        wb_base,
        wb_bias,
        wb_transfer_eff,
        wb_corr,
        share_by_year,
        baseline_transfer_share_pct=anchor_share_pct,
        transfer_end_pct_2040=0.0,
        transfer_share_anchor_pct=anchor_share_pct,
    )

    orig_train = bench.load_training_frame().copy().reset_index(drop=True)
    orig_h_model = bench.fit_model(bench.ModelSpec("hybrid_ridge", bench.FEATURES, "ridge"), orig_train)
    orig_h_pred = bench.recursive_forecast_known_exog(bench.ModelSpec("hybrid_ridge", bench.FEATURES, "ridge"), orig_h_model, orig_train, future_exog)
    old_wb_raw = wb.simulate_path(
        history_df=wb_df,
        future_exog=future_exog[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
        model=wb_base,
        month_bias=wb_bias,
        context=context,
        transfer_share_anchor_pct=anchor_share_pct,
        transfer_effectiveness=wb_transfer_eff,
        baseline_transfer_share_pct=anchor_share_pct,
        transfer_end_pct_2040=0.0,
    )["pred_fill"].to_numpy(dtype=float)
    pred_old = 0.45 * orig_h_pred + 0.55 * old_wb_raw

    stack_coef = dict(zip(stack_meta["feature"], stack_meta["value"]))
    pred_stack = np.clip(
        stack_coef["intercept"]
        + stack_coef["pred_h"] * pred_h
        + stack_coef["pred_w"] * pred_w
        + stack_coef["month_sin"] * future_exog["month_sin"].to_numpy(dtype=float)
        + stack_coef["month_cos"] * future_exog["month_cos"].to_numpy(dtype=float),
        0.0,
        1.0,
    )

    rows = []
    for model_name, arr in [
        ("hybrid_ridge_plus", pred_h),
        ("water_balance_v4_corrected", pred_w),
        ("hybrid_physics_stacked_plus", pred_stack),
        ("quantile_regressor_plus", pred_q),
        ("prophet_regressor_plus", pred_p),
        ("hybrid_physics_ensemble_phys_old", pred_old),
    ]:
        rows.append(pd.DataFrame({"date": future_exog["date"], "model": model_name, "pred_fill": arr}))
    return pd.concat(rows, ignore_index=True)


def plot_holdout(pred_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    top = summary_df.head(5)["model"].tolist()
    fig, ax = plt.subplots(figsize=(12.4, 5.6), dpi=170)
    ax.plot(pred_df["date"], pred_df["actual_fill"] * 100.0, color="#111827", linewidth=2.3, label="Gerçek")
    for model in top:
        ax.plot(pred_df["date"], pred_df[model] * 100.0, linewidth=1.8, color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
    ax.set_title("Geliştirilmiş ve dış modeller - 2016-2020 holdout")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_future(future_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    top = summary_df.head(5)["model"].tolist()
    fig, ax = plt.subplots(figsize=(12.4, 5.6), dpi=170)
    for model in top:
        g = future_df[future_df["model"] == model].copy()
        ax.plot(g["date"], g["pred_fill"] * 100.0, linewidth=2.0, color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
    ax.set_title("Geliştirilmiş ve dış modeller - 2026-2040 temel gelecek")
    ax.set_ylabel("Toplam doluluk (%)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_error_corr(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.2), dpi=170)
    for _, row in summary_df.iterrows():
        model = row["model"]
        ax.scatter(row["mape_pct"], row["pearson_corr_pct"], s=90, color=MODEL_COLORS.get(model, "#2563eb"))
        ax.annotate(MODEL_LABELS.get(model, model), (row["mape_pct"], row["pearson_corr_pct"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("Pearson korelasyon (%)")
    ax.set_title("Geliştirilmiş model seçimi: hata ve korelasyon")
    ax.grid(True, alpha=0.22)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df, stack_meta, stack_calib = evaluate_model_set()
    summary_df = summarize_holdout(pred_df)
    future_df = build_future_base_paths(stack_meta)

    pred_df.to_csv(OUT_DIR / "developed_models_holdout_predictions_2016_2020.csv", index=False)
    summary_df.to_csv(OUT_DIR / "developed_models_holdout_summary_2015_train_2020_test.csv", index=False)
    stack_meta.to_csv(OUT_DIR / "stacked_model_coefficients.csv", index=False)
    stack_calib.to_csv(OUT_DIR / "stacked_model_internal_calibration.csv", index=False)
    future_df.to_csv(OUT_DIR / "developed_models_future_base_2026_2040.csv", index=False)

    plot_holdout(pred_df, summary_df, OUT_DIR / "developed_models_holdout_top5.png")
    plot_future(future_df, summary_df, OUT_DIR / "developed_models_future_top5.png")
    plot_error_corr(summary_df, OUT_DIR / "developed_models_error_vs_corr.png")

    summary = {
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "best_model_by_mape": str(summary_df.iloc[0]["model"]),
        "best_model_mape_pct": float(summary_df.iloc[0]["mape_pct"]),
        "best_model_corr_pct": float(summary_df.iloc[0]["pearson_corr_pct"]),
        "external_models": ["quantile_regressor_plus", "prophet_regressor_plus"],
    }
    (OUT_DIR / "developed_models_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
