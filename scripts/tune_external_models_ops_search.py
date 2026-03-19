#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path("/Users/yasinkaya/Hackhaton")
BASE_DEV_SCRIPT = ROOT / "scripts" / "develop_models_with_extra_params.py"
TOP3_SCRIPT = ROOT / "scripts" / "develop_top3_plus_external_models.py"
BENCH_SCRIPT = ROOT / "scripts" / "benchmark_istanbul_forward_models.py"
OUT_DIR = ROOT / "output" / "istanbul_external_models_ops_search_2026_03_12"

TRAIN_END = pd.Timestamp("2015-12-01")
TEST_START = pd.Timestamp("2016-01-01")
TEST_END = pd.Timestamp("2020-12-01")

BASE_PROPHET_REGS = [
    "weighted_total_fill_lag1",
    "weighted_total_fill_lag2",
    "rain_model_mm",
    "et0_mm_month",
    "consumption_mean_monthly",
    "temp_proxy_c",
    "vpd_kpa_mean",
]
OPTIONAL_PROPHET_REGS = [
    "src_rain_north",
    "src_rain_west",
    "official_supply_m3_month_roll3",
    "transfer_share_pct_monthly_proxy",
    "nrw_pct_monthly_proxy",
    "reclaimed_share_pct_monthly_proxy",
]

PROPHET_SUBSETS = [
    ["src_rain_north"],
    ["src_rain_north", "src_rain_west"],
    ["src_rain_north", "official_supply_m3_month_roll3"],
    ["src_rain_north", "transfer_share_pct_monthly_proxy"],
    ["src_rain_north", "nrw_pct_monthly_proxy"],
    ["src_rain_north", "src_rain_west", "official_supply_m3_month_roll3"],
    ["src_rain_north", "src_rain_west", "transfer_share_pct_monthly_proxy"],
    ["src_rain_north", "src_rain_west", "nrw_pct_monthly_proxy"],
    ["src_rain_north", "src_rain_west", "official_supply_m3_month_roll3", "transfer_share_pct_monthly_proxy", "nrw_pct_monthly_proxy"],
]

QUANT_EXTRA_SETS = {
    "base": [],
    "rain_ops_light": ["src_rain_north", "src_rain_west", "transfer_share_pct_monthly_proxy"],
    "rain_ops_full": ["src_rain_north", "src_rain_west", "transfer_share_pct_monthly_proxy", "nrw_pct_monthly_proxy", "reclaimed_share_pct_monthly_proxy"],
}
QUANT_PARAM_GRID = [
    {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.9},
    {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.02, "subsample": 0.9},
    {"n_estimators": 240, "max_depth": 3, "learning_rate": 0.025, "subsample": 0.85},
    {"n_estimators": 400, "max_depth": 2, "learning_rate": 0.015, "subsample": 0.9},
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics_row(model_name: str, actual: np.ndarray, pred: np.ndarray, **extra) -> dict[str, float | str]:
    row = {
        "model": model_name,
        "rmse_pp": float(np.sqrt(mean_squared_error(actual, pred)) * 100.0),
        "mae_pp": float(mean_absolute_error(actual, pred)) * 100.0,
        "mape_pct": float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-6)) * 100.0),
        "smape_pct": float(np.mean(2.0 * np.abs(pred - actual) / np.maximum(np.abs(actual) + np.abs(pred), 1e-6)) * 100.0),
        "pearson_corr_pct": float(pearsonr(actual, pred).statistic * 100.0),
        "spearman_corr_pct": float(spearmanr(actual, pred).statistic * 100.0),
        "r2": float(r2_score(actual, pred)),
        "end_error_pp_2020_12": float((pred[-1] - actual[-1]) * 100.0),
    }
    row.update(extra)
    return row


def prophet_train_frame(df: pd.DataFrame, regs: list[str]) -> pd.DataFrame:
    out = df.rename(columns={"date": "ds", "weighted_total_fill": "y"}).copy()
    return out[["ds", "y"] + regs].copy()


def fit_prophet_subset(train_df: pd.DataFrame, regs: list[str]) -> Prophet:
    model = Prophet(
        growth="flat",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.03,
        seasonality_mode="multiplicative",
    )
    for reg in regs:
        model.add_regressor(reg, standardize=True)
    model.fit(prophet_train_frame(train_df, regs))
    return model


def recursive_predict_prophet_subset(model: Prophet, history_df: pd.DataFrame, future_df: pd.DataFrame, regs: list[str]) -> np.ndarray:
    past_fill = history_df["weighted_total_fill"].tolist()
    preds = []
    for row in future_df.itertuples(index=False):
        d = row._asdict()
        frame = {"ds": pd.Timestamp(d["date"])}
        for reg in regs:
            if reg == "weighted_total_fill_lag1":
                frame[reg] = past_fill[-1]
            elif reg == "weighted_total_fill_lag2":
                frame[reg] = past_fill[-2]
            else:
                frame[reg] = float(d[reg])
        yhat = float(np.clip(model.predict(pd.DataFrame([frame]))["yhat"].iloc[0], 0.0, 1.0))
        preds.append(yhat)
        past_fill.append(yhat)
    return np.asarray(preds, dtype=float)


def fit_quantile_custom(train_df: pd.DataFrame, features: list[str], params: dict[str, float]) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.5,
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        random_state=42,
    )
    model.fit(train_df[features], train_df["delta_fill"])
    return model


def recursive_predict_quantile_custom(base, model, features: list[str], history_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
    past_fill = history_df["weighted_total_fill"].tolist()
    past_rain = history_df["rain_model_mm"].tolist()
    past_et0 = history_df["et0_mm_month"].tolist()
    past_cons = history_df["consumption_mean_monthly"].tolist()
    preds = []
    step_offset = len(history_df)
    for i, row in enumerate(future_df.itertuples(index=False)):
        row_dict = row._asdict()
        feat = base.build_feature_row(past_fill, past_rain, past_et0, past_cons, row_dict, step_idx=step_offset + i)
        for col in features:
            if col not in feat:
                feat[col] = float(row_dict[col])
        delta_hat = float(model.predict(pd.DataFrame([{k: feat[k] for k in features}]))[0])
        yhat = float(np.clip(past_fill[-1] + delta_hat, 0.0, 1.0))
        preds.append(yhat)
        past_fill.append(yhat)
        past_rain.append(float(row.rain_model_mm))
        past_et0.append(float(row.et0_mm_month))
        past_cons.append(float(row.consumption_mean_monthly))
    return np.asarray(preds, dtype=float)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = load_module(BASE_DEV_SCRIPT, "dev_external_search")
    base = load_module(TOP3_SCRIPT, "top3_external_search")
    bench = load_module(BENCH_SCRIPT, "bench_external_search")

    feature_df = base.add_enhanced_features(bench.load_training_frame().copy())
    exog_df = dev.load_exog_frame()
    merge_cols = [
        "date",
        "src_rain_north",
        "src_rain_west",
        "official_supply_m3_month_roll3",
        "transfer_share_pct_monthly_proxy",
        "nrw_pct_monthly_proxy",
        "reclaimed_share_pct_monthly_proxy",
    ]
    feature_df = feature_df.merge(exog_df[merge_cols], on="date", how="left")
    train = feature_df[feature_df["date"] <= TRAIN_END].copy().reset_index(drop=True)
    test = feature_df[(feature_df["date"] >= TEST_START) & (feature_df["date"] <= TEST_END)].copy().reset_index(drop=True)
    actual = test["weighted_total_fill"].to_numpy(dtype=float)

    prophet_rows = []
    prophet_preds = []
    for subset in PROPHET_SUBSETS:
        regs = BASE_PROPHET_REGS + subset
        try:
            model = fit_prophet_subset(train, regs)
            pred = recursive_predict_prophet_subset(model, train, test, regs)
            name = "prophet_" + "_".join(subset)
            prophet_rows.append(metrics_row(name, actual, pred, regressor_count=len(regs), regressor_set=",".join(subset)))
            prophet_preds.append((name, pred))
        except Exception as exc:
            prophet_rows.append({"model": "prophet_" + "_".join(subset), "error": str(exc), "regressor_set": ",".join(subset)})

    prophet_df = pd.DataFrame(prophet_rows)
    prophet_ok = prophet_df.dropna(subset=["rmse_pp"]).sort_values(["mape_pct", "rmse_pp"]).reset_index(drop=True)

    quant_rows = []
    quant_preds = []
    for extra_name, extra_features in QUANT_EXTRA_SETS.items():
        features = base.ENHANCED_FEATURES + extra_features
        for params in QUANT_PARAM_GRID:
            name = f"quant_{extra_name}_ne{params['n_estimators']}_d{params['max_depth']}_lr{params['learning_rate']}"
            model = fit_quantile_custom(train, features, params)
            pred = recursive_predict_quantile_custom(base, model, features, train, test)
            quant_rows.append(metrics_row(name, actual, pred, feature_set=extra_name, **params))
            quant_preds.append((name, pred))

    quant_df = pd.DataFrame(quant_rows).sort_values(["mape_pct", "rmse_pp"]).reset_index(drop=True)

    prophet_df.to_csv(OUT_DIR / "prophet_ops_search_summary.csv", index=False)
    quant_df.to_csv(OUT_DIR / "quantile_ops_search_summary.csv", index=False)

    best_prophet = prophet_ok.iloc[0].to_dict() if not prophet_ok.empty else {}
    best_quant = quant_df.iloc[0].to_dict() if not quant_df.empty else {}
    summary = {
        "best_prophet": best_prophet,
        "best_quantile": best_quant,
    }
    (OUT_DIR / "external_search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not prophet_ok.empty:
        best_prophet_name = prophet_ok.iloc[0]["model"]
        best_prophet_pred = next(arr for name, arr in prophet_preds if name == best_prophet_name)
        pd.DataFrame({"date": test["date"], "actual_fill": actual, "pred_fill": best_prophet_pred}).to_csv(
            OUT_DIR / "best_prophet_predictions_2016_2020.csv", index=False
        )
    best_quant_name = quant_df.iloc[0]["model"]
    best_quant_pred = next(arr for name, arr in quant_preds if name == best_quant_name)
    pd.DataFrame({"date": test["date"], "actual_fill": actual, "pred_fill": best_quant_pred}).to_csv(
        OUT_DIR / "best_quantile_predictions_2016_2020.csv", index=False
    )

    print(OUT_DIR)


if __name__ == "__main__":
    main()
