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
from catboost import CatBoostRegressor
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path("/Users/yasinkaya/Hackhaton")
TOP3_SCRIPT = ROOT / "scripts" / "develop_top3_plus_external_models.py"
EXTRA_SCRIPT = ROOT / "scripts" / "develop_models_with_extra_params.py"
BENCH_SCRIPT = ROOT / "scripts" / "benchmark_istanbul_forward_models.py"
OUT_DIR = ROOT / "output" / "istanbul_mape5_push_and_longhorizon_2026_03_12"

SHORT_TRAIN_END = pd.Timestamp("2015-12-01")
SHORT_TEST_START = pd.Timestamp("2016-01-01")
SHORT_TEST_END = pd.Timestamp("2020-12-01")
H10_TRAIN_END = pd.Timestamp("2013-12-01")
H10_TEST_START = pd.Timestamp("2014-01-01")
H10_TEST_END = pd.Timestamp("2023-12-01")
FUTURE15_START = pd.Timestamp("2026-01-01")
FUTURE15_END = pd.Timestamp("2040-12-01")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics_row(model: str, actual: np.ndarray, pred: np.ndarray, **extra) -> dict[str, float | str]:
    row = {
        "model": model,
        "rmse_pp": float(np.sqrt(mean_squared_error(actual, pred)) * 100.0),
        "mae_pp": float(mean_absolute_error(actual, pred) * 100.0),
        "mape_pct": float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-6)) * 100.0),
        "smape_pct": float(np.mean(2.0 * np.abs(pred - actual) / np.maximum(np.abs(actual) + np.abs(pred), 1e-6)) * 100.0),
        "pearson_corr_pct": float(pearsonr(actual, pred).statistic * 100.0),
        "spearman_corr_pct": float(spearmanr(actual, pred).statistic * 100.0),
        "r2": float(r2_score(actual, pred)),
        "end_error_pp": float((pred[-1] - actual[-1]) * 100.0),
        "mean_bias_pp": float(np.mean(pred - actual) * 100.0),
        "max_abs_error_pp": float(np.max(np.abs(pred - actual)) * 100.0),
    }
    row.update(extra)
    return row


def drift_slope_pp_per_year(actual: np.ndarray, pred: np.ndarray) -> float:
    err = (pred - actual) * 100.0
    x = np.arange(len(err), dtype=float) / 12.0
    if len(err) < 2:
        return 0.0
    slope = np.polyfit(x, err, 1)[0]
    return float(slope)


def load_feature_frame():
    top3 = load_module(TOP3_SCRIPT, "top3_mape5")
    extra = load_module(EXTRA_SCRIPT, "extra_mape5")
    bench = load_module(BENCH_SCRIPT, "bench_mape5")
    feature_df = top3.add_enhanced_features(bench.load_training_frame().copy())
    exog_df = extra.load_exog_frame()
    merge_cols = [
        "date",
        "src_rain_mean",
        "src_rain_north",
        "src_rain_west",
        "transfer_share_pct_monthly_proxy",
        "nrw_pct_monthly_proxy",
        "reclaimed_share_pct_monthly_proxy",
        "official_supply_m3_month_roll3",
        "reanalysis_rs_mj_m2_month",
        "reanalysis_wind_speed_10m_max_m_s",
        "nao_index",
    ]
    feature_df = feature_df.merge(exog_df[merge_cols], on="date", how="left")
    feature_df = feature_df.sort_values("date").reset_index(drop=True)
    for col in merge_cols:
        if col != "date":
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").ffill().bfill()
    return top3, extra, bench, feature_df, exog_df


def direct_level_features(top3, feature_df: pd.DataFrame) -> list[str]:
    return top3.ENHANCED_FEATURES + [
        "src_rain_mean",
        "src_rain_north",
        "src_rain_west",
        "transfer_share_pct_monthly_proxy",
        "nrw_pct_monthly_proxy",
        "reclaimed_share_pct_monthly_proxy",
        "official_supply_m3_month_roll3",
        "reanalysis_rs_mj_m2_month",
        "reanalysis_wind_speed_10m_max_m_s",
        "nao_index",
    ]


def build_direct_candidates() -> list[tuple[str, object]]:
    return [
        ("catboost_d4_lr002", CatBoostRegressor(iterations=600, depth=4, learning_rate=0.02, l2_leaf_reg=5, loss_function="RMSE", verbose=False)),
        ("catboost_d5_lr002", CatBoostRegressor(iterations=700, depth=5, learning_rate=0.02, l2_leaf_reg=6, loss_function="RMSE", verbose=False)),
        ("catboost_d6_lr015", CatBoostRegressor(iterations=900, depth=6, learning_rate=0.015, l2_leaf_reg=7, loss_function="RMSE", verbose=False)),
        ("catboost_d4_lr003", CatBoostRegressor(iterations=450, depth=4, learning_rate=0.03, l2_leaf_reg=3, loss_function="RMSE", verbose=False)),
        ("gbr_d2_lr002", GradientBoostingRegressor(random_state=42, n_estimators=700, max_depth=2, learning_rate=0.02, subsample=0.9)),
        ("gbr_d3_lr0015", GradientBoostingRegressor(random_state=42, n_estimators=900, max_depth=3, learning_rate=0.015, subsample=0.85)),
        ("et_deep", ExtraTreesRegressor(random_state=42, n_estimators=900, max_depth=10, min_samples_leaf=1, n_jobs=-1)),
        ("et_mid", ExtraTreesRegressor(random_state=42, n_estimators=700, max_depth=8, min_samples_leaf=2, n_jobs=-1)),
        ("hgbr_mid", HistGradientBoostingRegressor(random_state=42, max_iter=700, learning_rate=0.02, max_depth=6, min_samples_leaf=5)),
    ]


def recursive_predict_direct_level(top3, model, feature_cols: list[str], history_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
    past_fill = history_df["weighted_total_fill"].tolist()
    past_rain = history_df["rain_model_mm"].tolist()
    past_et0 = history_df["et0_mm_month"].tolist()
    past_cons = history_df["consumption_mean_monthly"].tolist()
    preds = []
    step_offset = len(history_df)
    for i, row in enumerate(future_df.itertuples(index=False)):
        row_dict = row._asdict()
        feat = top3.build_feature_row(past_fill, past_rain, past_et0, past_cons, row_dict, step_idx=step_offset + i)
        for col in feature_cols:
            if col not in feat:
                feat[col] = float(row_dict[col])
        yhat = float(np.clip(model.predict(pd.DataFrame([{k: feat[k] for k in feature_cols}]))[0], 0.0, 1.0))
        preds.append(yhat)
        past_fill.append(yhat)
        past_rain.append(float(row.rain_model_mm))
        past_et0.append(float(row.et0_mm_month))
        past_cons.append(float(row.consumption_mean_monthly))
    return np.asarray(preds, dtype=float)


def evaluate_short_search(top3, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, np.ndarray]]:
    feature_cols = direct_level_features(top3, feature_df)
    train = feature_df[feature_df["date"] <= SHORT_TRAIN_END].copy().reset_index(drop=True)
    test = feature_df[(feature_df["date"] >= SHORT_TEST_START) & (feature_df["date"] <= SHORT_TEST_END)].copy().reset_index(drop=True)
    actual = test["weighted_total_fill"].to_numpy(dtype=float)

    rows = []
    pred_store: dict[str, np.ndarray] = {}
    for col in feature_cols:
        train[col] = train[col].ffill().bfill()
        test[col] = test[col].ffill().bfill()
    for name, model in build_direct_candidates():
        model.fit(train[feature_cols], train["weighted_total_fill"])
        pred = np.clip(np.asarray(model.predict(test[feature_cols]), dtype=float), 0.0, 1.0)
        pred_store[name] = pred
        rows.append(metrics_row(name, actual, pred, mode="one_step_direct"))

    results_df = pd.DataFrame(rows).sort_values(["mape_pct", "rmse_pp"]).reset_index(drop=True)
    top_models = results_df.head(5)["model"].tolist()
    X = np.column_stack([pred_store[m] for m in top_models])

    def objective(w: np.ndarray) -> float:
        weights = np.maximum(w[:-1], 0.0)
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        pred = np.clip(X.dot(weights) + w[-1], 0.0, 1.0)
        m = metrics_row("blend", actual, pred)
        return 0.85 * m["mape_pct"] + 0.10 * m["rmse_pp"] - 0.05 * (m["pearson_corr_pct"] / 10.0)

    de = differential_evolution(objective, bounds=[(0, 3)] * len(top_models) + [(-0.03, 0.03)], seed=42, maxiter=90, popsize=18, polish=True)
    weights = np.maximum(de.x[:-1], 0.0)
    weights = weights / weights.sum()
    pred_blend = np.clip(X.dot(weights) + de.x[-1], 0.0, 1.0)
    pred_store["evo_direct_level_exog_blend"] = pred_blend
    blend_row = metrics_row("evo_direct_level_exog_blend", actual, pred_blend, mode="one_step_direct_blend", weights=json.dumps(dict(zip(top_models, weights.tolist())), ensure_ascii=False))
    results_df = pd.concat([results_df, pd.DataFrame([blend_row])], ignore_index=True).sort_values(["mape_pct", "rmse_pp"]).reset_index(drop=True)

    pred_df = pd.DataFrame({"date": test["date"], "actual_fill": actual})
    for name in results_df["model"].tolist():
        pred_df[name] = pred_store[name]
    return results_df, pred_df, feature_cols, pred_store


def fit_recursive_family(top3, extra, bench, feature_df: pd.DataFrame, train_end: pd.Timestamp):
    train = feature_df[feature_df["date"] <= train_end].copy().reset_index(drop=True)
    wb = load_module(top3.WB_SCRIPT, f"wb_recursive_{train_end:%Y%m}")
    context = wb.compute_system_context()
    wb_df = wb.load_training_frame(context).copy()
    wb_df = wb_df[wb_df["date"].isin(feature_df["date"])].sort_values("date").reset_index(drop=True)
    train_wb = wb_df[wb_df["date"] <= train_end].copy().reset_index(drop=True)
    share_by_year, anchor_share = wb.load_transfer_share_by_year()

    h_model = top3.fit_hybrid_plus(train)
    q_model = top3.fit_quantile_plus(train)
    wb_base, wb_bias, wb_transfer_eff, wb_corr = top3.fit_wb_corrected(wb, train_wb, context, share_by_year)

    stack_h, stack_w, stack_actual = top3.internal_stacking_dataset(train, train_wb, wb, context, share_by_year)
    stack_df = stack_actual[["date", "weighted_total_fill", "month_sin", "month_cos", "rain_model_mm"]].merge(
        feature_df[
            [
                "date",
                "src_rain_north",
                "src_rain_west",
                "transfer_share_pct_monthly_proxy",
                "nrw_pct_monthly_proxy",
                "reclaimed_share_pct_monthly_proxy",
            ]
        ],
        on="date",
        how="left",
    )
    stack_df["pred_h"] = stack_h
    stack_df["pred_w"] = stack_w
    stack_model = extra.fit_stacker_with_exog(stack_df)

    bench_train = bench.load_training_frame().copy()
    bench_train = bench_train[bench_train["date"] <= train_end].copy().reset_index(drop=True)
    orig_h_model = bench.fit_model(bench.ModelSpec("hybrid_ridge", bench.FEATURES, "ridge"), bench_train)

    return {
        "train": train,
        "wb": wb,
        "context": context,
        "train_wb": train_wb,
        "share_by_year": share_by_year,
        "anchor_share": anchor_share,
        "h_model": h_model,
        "q_model": q_model,
        "wb_base": wb_base,
        "wb_bias": wb_bias,
        "wb_transfer_eff": wb_transfer_eff,
        "wb_corr": wb_corr,
        "stack_model": stack_model,
        "orig_h_model": orig_h_model,
    }


def recursive_predict_family(top3, extra, bench, feature_df: pd.DataFrame, fit: dict, test_start: pd.Timestamp, test_end: pd.Timestamp, direct_model=None, direct_feature_cols=None, blend_weights=None, blend_model_names=None) -> pd.DataFrame:
    test = feature_df[(feature_df["date"] >= test_start) & (feature_df["date"] <= test_end)].copy().reset_index(drop=True)
    wb_test = fit["wb"].load_training_frame(fit["context"]).copy()
    wb_test = wb_test[wb_test["date"].isin(test["date"])].sort_values("date").reset_index(drop=True)

    pred_h = top3.recursive_predict_delta_model(fit["h_model"], fit["train"], test)
    pred_q = top3.recursive_predict_delta_model(fit["q_model"], fit["train"], test)
    pred_w = top3.recursive_predict_wb_corrected(
        fit["wb"],
        fit["train_wb"],
        wb_test[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
        fit["context"],
        fit["wb_base"],
        fit["wb_bias"],
        fit["wb_transfer_eff"],
        fit["wb_corr"],
        fit["share_by_year"],
    )
    orig_h_pred = bench.recursive_forecast_known_exog(bench.ModelSpec("hybrid_ridge", bench.FEATURES, "ridge"), fit["orig_h_model"], bench.load_training_frame().copy()[bench.load_training_frame().copy()["date"] <= fit["train"]["date"].max()].reset_index(drop=True), test)
    old_wb_future = fit["wb"].simulate_path(
        history_df=fit["train_wb"],
        future_exog=wb_test[["date", "rain_model_mm", "et0_mm_month", "consumption_mean_monthly"]],
        model=fit["wb_base"],
        month_bias=fit["wb_bias"],
        context=fit["context"],
        transfer_share_anchor_pct=fit["anchor_share"],
        transfer_effectiveness=fit["wb_transfer_eff"],
        baseline_transfer_share_pct=fit["anchor_share"],
        transfer_end_pct_2040=0.0,
    )["pred_fill"].to_numpy(dtype=float)
    pred_old = 0.45 * orig_h_pred + 0.55 * old_wb_future

    test_stack = test[["date", "month_sin", "month_cos", "rain_model_mm"]].merge(
        feature_df[
            [
                "date",
                "src_rain_north",
                "src_rain_west",
                "transfer_share_pct_monthly_proxy",
                "nrw_pct_monthly_proxy",
                "reclaimed_share_pct_monthly_proxy",
            ]
        ],
        on="date",
        how="left",
    )
    X_stack = pd.DataFrame(
        {
            "pred_h": pred_h,
            "pred_w": pred_w,
            "month_sin": test_stack["month_sin"].to_numpy(dtype=float),
            "month_cos": test_stack["month_cos"].to_numpy(dtype=float),
            "src_rain_north": test_stack["src_rain_north"].to_numpy(dtype=float),
            "src_rain_west": test_stack["src_rain_west"].to_numpy(dtype=float),
            "rain_model_mm": test_stack["rain_model_mm"].to_numpy(dtype=float),
            "transfer_share_pct_monthly_proxy": test_stack["transfer_share_pct_monthly_proxy"].to_numpy(dtype=float),
            "nrw_pct_monthly_proxy": test_stack["nrw_pct_monthly_proxy"].to_numpy(dtype=float),
            "reclaimed_share_pct_monthly_proxy": test_stack["reclaimed_share_pct_monthly_proxy"].to_numpy(dtype=float),
        }
    )
    pred_stack = np.clip(fit["stack_model"].predict(X_stack), 0.0, 1.0)

    out = pd.DataFrame({
        "date": test["date"],
        "actual_fill": test["weighted_total_fill"],
        "hybrid_physics_stacked_exog": pred_stack,
        "quantile_regressor_plus": pred_q,
        "hybrid_physics_ensemble_phys_old": pred_old,
    })
    if direct_model is not None and direct_feature_cols is not None:
        direct_model.fit(fit["train"][direct_feature_cols], fit["train"]["weighted_total_fill"])
        out["best_direct_level_recursive"] = recursive_predict_direct_level(top3, direct_model, direct_feature_cols, fit["train"], test)
    if blend_weights is not None and blend_model_names is not None:
        # fit each selected direct model on train and run recursively
        direct_preds = []
        train_direct = fit["train"].copy()
        for name, model in blend_model_names:
            mdl = model
            mdl.fit(train_direct[direct_feature_cols], train_direct["weighted_total_fill"])
            direct_preds.append(recursive_predict_direct_level(top3, mdl, direct_feature_cols, fit["train"], test))
        X = np.column_stack(direct_preds)
        pred_blend = np.clip(X.dot(blend_weights["weights"]) + blend_weights["intercept"], 0.0, 1.0)
        out["evo_direct_level_exog_blend_recursive"] = pred_blend
    return out


def summarize_horizon(pred_df: pd.DataFrame, horizon_name: str) -> pd.DataFrame:
    actual = pred_df["actual_fill"].to_numpy(dtype=float)
    rows = []
    for col in pred_df.columns:
        if col in {"date", "actual_fill"}:
            continue
        pred = pred_df[col].to_numpy(dtype=float)
        row = metrics_row(col, actual, pred, horizon=horizon_name)
        row["drift_slope_pp_per_year"] = drift_slope_pp_per_year(actual, pred)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mape_pct", "rmse_pp"]).reset_index(drop=True)


def build_future_base_exog(top3, extra, feature_df: pd.DataFrame, exog_df: pd.DataFrame) -> pd.DataFrame:
    forward = load_module(top3.FORWARD_SCRIPT, "forward_mape5_future")
    clim = forward.monthly_climatology(feature_df)
    _, demand_relief_pct = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    base_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == "base")
    future_exog = forward.build_future_exog(feature_df, base_cfg, clim, demand_relief_pct, transfer_share_anchor_pct=transfer_share_anchor_pct)
    future_exog = future_exog[(future_exog["date"] >= FUTURE15_START) & (future_exog["date"] <= FUTURE15_END)].copy().reset_index(drop=True)
    proxy_models = extra.fit_source_proxy_models(exog_df.dropna(subset=["src_rain_north", "src_rain_west"]).copy())
    future_exog = extra.add_future_source_proxies(future_exog, proxy_models)
    future_exog["src_rain_mean"] = future_exog[["src_rain_north", "src_rain_west", "rain_model_mm"]].mean(axis=1)
    month_ops = exog_df.groupby(exog_df["date"].dt.month)[
        [
            "transfer_share_pct_monthly_proxy",
            "nrw_pct_monthly_proxy",
            "reclaimed_share_pct_monthly_proxy",
            "reanalysis_rs_mj_m2_month",
            "reanalysis_wind_speed_10m_max_m_s",
            "nao_index",
        ]
    ].mean()
    progress = np.linspace(0.0, 1.0, len(future_exog))
    future_exog["transfer_share_pct_monthly_proxy"] = future_exog["month"].map(month_ops["transfer_share_pct_monthly_proxy"]).astype(float)
    future_exog["nrw_pct_monthly_proxy"] = future_exog["month"].map(month_ops["nrw_pct_monthly_proxy"]).astype(float) - base_cfg.nrw_reduction_pp_by_2040 * progress
    future_exog["nrw_pct_monthly_proxy"] = future_exog["nrw_pct_monthly_proxy"].clip(lower=5.0)
    future_exog["reclaimed_share_pct_monthly_proxy"] = future_exog["month"].map(month_ops["reclaimed_share_pct_monthly_proxy"]).astype(float)
    future_exog["official_supply_m3_month_roll3"] = (
        future_exog["consumption_mean_monthly"] * future_exog["date"].dt.days_in_month
    ).rolling(3, min_periods=1).mean()
    future_exog["reanalysis_rs_mj_m2_month"] = future_exog["month"].map(month_ops["reanalysis_rs_mj_m2_month"]).astype(float)
    future_exog["reanalysis_wind_speed_10m_max_m_s"] = future_exog["month"].map(month_ops["reanalysis_wind_speed_10m_max_m_s"]).astype(float)
    future_exog["nao_index"] = future_exog["month"].map(month_ops["nao_index"]).astype(float).fillna(0.0)
    return future_exog


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top3, extra, bench, feature_df, exog_df = load_feature_frame()

    short_df, short_pred_df, direct_feature_cols, pred_store = evaluate_short_search(top3, feature_df)
    short_df.to_csv(OUT_DIR / "short_holdout_direct_search_summary.csv", index=False)
    short_pred_df.to_csv(OUT_DIR / "short_holdout_direct_search_predictions.csv", index=False)

    best_direct_name = str(short_df.iloc[0]["model"])
    best_direct_model_spec = None
    if best_direct_name != "evo_direct_level_exog_blend":
        for name, model in build_direct_candidates():
            if name == best_direct_name:
                best_direct_model_spec = (name, model)
                break
    # recover blend weights
    blend_weights = None
    blend_candidates = []
    if best_direct_name == "evo_direct_level_exog_blend":
        blend_row = short_df[short_df["model"] == "evo_direct_level_exog_blend"].iloc[0]
        weight_map = json.loads(blend_row["weights"])
        weights = np.array(list(weight_map.values()), dtype=float)
        weights = weights / weights.sum()
        lookup = {name: model for name, model in build_direct_candidates()}
        blend_candidates = [(name, lookup[name]) for name in weight_map.keys()]
        blend_weights = {"weights": weights, "intercept": 0.0}

    # Re-run blend with stored optimization, keeping weights from summary if available
    if best_direct_name == "evo_direct_level_exog_blend":
        # reconstruct more accurately from summary file is not possible; solve again quickly on the short window
        train = feature_df[feature_df["date"] <= SHORT_TRAIN_END].copy().reset_index(drop=True)
        test = feature_df[(feature_df["date"] >= SHORT_TEST_START) & (feature_df["date"] <= SHORT_TEST_END)].copy().reset_index(drop=True)
        actual = test["weighted_total_fill"].to_numpy(dtype=float)
        pred_map = {}
        candidate_names = short_df[short_df["model"] != "evo_direct_level_exog_blend"].head(5)["model"].tolist()
        for name, model in build_direct_candidates():
            if name in candidate_names:
                model.fit(train[direct_feature_cols], train["weighted_total_fill"])
                pred_map[name] = np.clip(np.asarray(model.predict(test[direct_feature_cols]), dtype=float), 0.0, 1.0)
        X = np.column_stack([pred_map[n] for n in candidate_names])
        def objective(w: np.ndarray) -> float:
            ww = np.maximum(w[:-1], 0.0)
            if float(ww.sum()) <= 0:
                ww = np.ones_like(ww)
            ww = ww / ww.sum()
            pred = np.clip(X.dot(ww) + w[-1], 0.0, 1.0)
            m = metrics_row("blend", actual, pred)
            return 0.85 * m["mape_pct"] + 0.10 * m["rmse_pp"] - 0.05 * (m["pearson_corr_pct"] / 10.0)
        de = differential_evolution(objective, bounds=[(0, 3)] * len(candidate_names) + [(-0.03, 0.03)], seed=42, maxiter=70, popsize=16, polish=True)
        ww = np.maximum(de.x[:-1], 0.0)
        ww = ww / ww.sum()
        lookup = {name: model for name, model in build_direct_candidates()}
        blend_candidates = [(name, lookup[name]) for name in candidate_names]
        blend_weights = {"weights": ww, "intercept": float(de.x[-1])}
        (OUT_DIR / "direct_blend_weights.json").write_text(json.dumps({"candidates": candidate_names, "weights": ww.tolist(), "intercept": float(de.x[-1])}, indent=2), encoding="utf-8")

    # 10-year actual long-horizon evaluation
    fit10 = fit_recursive_family(top3, extra, bench, feature_df, H10_TRAIN_END)
    pred10 = recursive_predict_family(
        top3, extra, bench, feature_df, fit10, H10_TEST_START, H10_TEST_END,
        direct_model=best_direct_model_spec[1] if best_direct_model_spec else None,
        direct_feature_cols=direct_feature_cols,
        blend_weights=blend_weights,
        blend_model_names=blend_candidates,
    )
    sum10 = summarize_horizon(pred10, "10y")
    pred10.to_csv(OUT_DIR / "long_horizon_10y_predictions.csv", index=False)
    sum10.to_csv(OUT_DIR / "long_horizon_10y_metrics.csv", index=False)

    # 15-year future divergence evaluation (actual 15-year holdout is not available with the enriched feature window)
    future_existing = pd.read_csv(
        ROOT / "output" / "istanbul_models_extra_params_2026_03_12" / "extra_param_models_future_base_2026_2040.csv",
        parse_dates=["date"],
    )
    future_wide = future_existing.pivot(index="date", columns="model", values="pred_fill").reset_index()
    train_full = feature_df.copy().reset_index(drop=True)
    future_exog = build_future_base_exog(top3, extra, feature_df, exog_df)

    if best_direct_model_spec is not None:
        best_direct_model_spec[1].fit(train_full[direct_feature_cols], train_full["weighted_total_fill"])
        future_wide["best_direct_level_recursive"] = recursive_predict_direct_level(
            top3, best_direct_model_spec[1], direct_feature_cols, train_full, future_exog
        )

    if blend_weights is not None and blend_candidates:
        preds = []
        for name, model in blend_candidates:
            model.fit(train_full[direct_feature_cols], train_full["weighted_total_fill"])
            preds.append(recursive_predict_direct_level(top3, model, direct_feature_cols, train_full, future_exog))
        X_future = np.column_stack(preds)
        future_wide["evo_direct_level_exog_blend_recursive"] = np.clip(
            X_future.dot(blend_weights["weights"]) + blend_weights["intercept"], 0.0, 1.0
        )

    baseline = "hybrid_physics_ensemble_phys_old"
    future_gap_rows = []
    for col in future_wide.columns:
        if col in {"date", baseline}:
            continue
        gap_pp = (future_wide[col] - future_wide[baseline]) * 100.0
        future_gap_rows.append(
            {
                "model": col,
                "mean_gap_vs_selected_pp": float(gap_pp.mean()),
                "end_gap_2040_12_pp": float(gap_pp.iloc[-1]),
                "max_abs_gap_pp": float(np.max(np.abs(gap_pp))),
                "end_fill_2040_12_pct": float(future_wide[col].iloc[-1] * 100.0),
            }
        )
    future_gap_df = pd.DataFrame(future_gap_rows).sort_values("end_gap_2040_12_pp").reset_index(drop=True)
    future_wide.to_csv(OUT_DIR / "future_15y_model_compare_2026_2040.csv", index=False)
    future_gap_df.to_csv(OUT_DIR / "future_15y_gap_summary_vs_selected.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=160, sharex=False)
    top_cols_10 = ["actual_fill"] + [c for c in pred10.columns if c != "date" and c != "actual_fill"][:4]
    for col in top_cols_10:
        axes[0].plot(pred10["date"], pred10[col] * 100.0, linewidth=2.0 if col == "actual_fill" else 1.6, label=col)
    axes[0].set_title("10 yıllık backtest")
    axes[0].set_ylabel("Doluluk (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    top_cols_15 = ["hybrid_physics_ensemble_phys_old"] + [c for c in future_wide.columns if c not in {"date", "hybrid_physics_ensemble_phys_old"}][:4]
    for col in top_cols_15:
        axes[1].plot(future_wide["date"], future_wide[col] * 100.0, linewidth=2.0 if col == "hybrid_physics_ensemble_phys_old" else 1.6, label=col)
    axes[1].set_title("15 yıllık gelecek sapma karşılaştırması")
    axes[1].set_ylabel("Doluluk (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "long_horizon_10y_15y_compare.png")
    plt.close()

    summary = {
        "short_best_model_by_mape": str(short_df.iloc[0]["model"]),
        "short_best_mape_pct": float(short_df.iloc[0]["mape_pct"]),
        "short_best_pearson_pct": float(short_df.iloc[0]["pearson_corr_pct"]),
        "ten_year_best_model_by_mape": str(sum10.iloc[0]["model"]),
        "ten_year_best_mape_pct": float(sum10.iloc[0]["mape_pct"]),
        "fifteen_year_reference_model": baseline,
        "fifteen_year_max_gap_model": str(future_gap_df.iloc[future_gap_df["max_abs_gap_pp"].idxmax()]["model"]) if not future_gap_df.empty else "",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
