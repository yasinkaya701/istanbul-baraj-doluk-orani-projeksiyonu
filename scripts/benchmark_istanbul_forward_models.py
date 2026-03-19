#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
import importlib.util
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


ROOT = Path("/Users/yasinkaya/Hackhaton")
CORE_PATH = ROOT / "output" / "model_useful_data_bundle" / "tables" / "istanbul_model_core_monthly.csv"
OUT_DIR = ROOT / "output" / "istanbul_forward_model_benchmark_round2"
FORWARD_SCRIPT = ROOT / "scripts" / "build_istanbul_forward_projection_2040.py"

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

PLOT_LABELS = {
    "persistence": "Süreklilik",
    "history_only_ridge": "Yalnız tarihsel",
    "hybrid_ridge": "Hibrit Ridge",
    "hybrid_elastic_net": "Elastic Net",
    "extra_trees_full": "Extra Trees",
    "random_forest_full": "Random Forest",
    "hist_gbm_full": "HistGBM",
    "hist_gbm_monotonic": "Monotonik HistGBM",
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: list[str]
    family: str


def load_forward_module():
    spec = importlib.util.spec_from_file_location("istanbul_forward_projection_2040", FORWARD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Forward projection script could not be imported.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(CORE_PATH, parse_dates=["date"])
    df["delta_fill"] = df["weighted_total_fill"] - df["weighted_total_fill_lag1"]
    keep = ["date", "weighted_total_fill", "delta_fill"] + FEATURES
    return df[keep].dropna().reset_index(drop=True)


def monotonic_constraints() -> list[int]:
    mapping = {
        "weighted_total_fill_lag1": 1,
        "weighted_total_fill_lag2": 1,
        "rain_model_mm": 1,
        "rain_model_mm_lag1": 1,
        "rain_model_mm_roll3": 1,
        "et0_mm_month": -1,
        "et0_mm_month_lag1": -1,
        "et0_mm_month_roll3": -1,
        "consumption_mean_monthly": -1,
        "consumption_mean_monthly_lag1": -1,
        "consumption_mean_monthly_roll3": -1,
        "temp_proxy_c": -1,
        "rh_proxy_pct": 1,
        "vpd_kpa_mean": -1,
        "water_balance_proxy_mm": 1,
        "month_sin": 0,
        "month_cos": 0,
    }
    return [mapping[col] for col in FEATURES]


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("persistence", HISTORY_ONLY_FEATURES, "persistence"),
        ModelSpec("history_only_ridge", HISTORY_ONLY_FEATURES, "ridge"),
        ModelSpec("hybrid_ridge", FEATURES, "ridge"),
        ModelSpec("hybrid_elastic_net", FEATURES, "elastic_net"),
        ModelSpec("extra_trees_full", FEATURES, "extra_trees"),
        ModelSpec("random_forest_full", FEATURES, "random_forest"),
        ModelSpec("hist_gbm_full", FEATURES, "hist_gbm"),
        ModelSpec("hist_gbm_monotonic", FEATURES, "hist_gbm_monotonic"),
    ]


def make_model(spec: ModelSpec):
    if spec.family == "persistence":
        return None
    if spec.family == "ridge":
        return Pipeline(
            [("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 3, 25)))]
        )
    if spec.family == "elastic_net":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        alphas=np.logspace(-4, 1, 25),
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        )
    if spec.family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    if spec.family == "random_forest":
        return RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    if spec.family == "hist_gbm":
        return HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=6,
            random_state=42,
        )
    if spec.family == "hist_gbm_monotonic":
        return HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=6,
            monotonic_cst=monotonic_constraints(),
            random_state=42,
        )
    raise ValueError(spec.family)


def fit_model(spec: ModelSpec, train_df: pd.DataFrame):
    model = make_model(spec)
    if model is None:
        return None
    model.fit(train_df[spec.features], train_df["delta_fill"])
    return model


def predict_delta(spec: ModelSpec, model, row_df: pd.DataFrame) -> float:
    if spec.family == "persistence":
        return 0.0
    return float(model.predict(row_df[spec.features])[0])


def rmse_pp(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) * 100.0)


def mae_pp(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred) * 100.0)


def sign_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.sign(actual) == np.sign(pred)))


def one_step_walkforward(train_df: pd.DataFrame, specs: list[ModelSpec], min_train: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_parts = []
    for spec in specs:
        actual = []
        pred = []
        delta_actual = []
        delta_pred = []
        dates = []
        for idx in range(min_train, len(train_df)):
            tr = train_df.iloc[:idx]
            te = train_df.iloc[[idx]]
            model = fit_model(spec, tr)
            delta_hat = predict_delta(spec, model, te)
            yhat = float(np.clip(te["weighted_total_fill_lag1"].iloc[0] + delta_hat, 0.0, 1.0))
            pred.append(yhat)
            actual.append(float(te["weighted_total_fill"].iloc[0]))
            delta_actual.append(float(te["delta_fill"].iloc[0]))
            delta_pred.append(delta_hat)
            dates.append(te["date"].iloc[0])
        actual_arr = np.asarray(actual, dtype=float)
        pred_arr = np.asarray(pred, dtype=float)
        metric_rows.append(
            {
                "model": spec.name,
                "one_step_rmse_pp": rmse_pp(actual_arr, pred_arr),
                "one_step_mae_pp": mae_pp(actual_arr, pred_arr),
                "delta_direction_accuracy": sign_accuracy(np.asarray(delta_actual), np.asarray(delta_pred)),
                "n_predictions": len(actual_arr),
            }
        )
        pred_parts.append(pd.DataFrame({"date": dates, "actual": actual_arr, "pred": pred_arr, "model": spec.name}))
    return pd.DataFrame(metric_rows).sort_values("one_step_rmse_pp").reset_index(drop=True), pd.concat(pred_parts, ignore_index=True)


def recursive_forecast_known_exog(spec: ModelSpec, model, history_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
    past_fill = history_df["weighted_total_fill"].tolist()
    past_rain = history_df["rain_model_mm"].tolist()
    past_et0 = history_df["et0_mm_month"].tolist()
    past_cons = history_df["consumption_mean_monthly"].tolist()
    preds = []
    for _, row in future_df.iterrows():
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
        feat_df = pd.DataFrame([feat])
        delta_hat = predict_delta(spec, model, feat_df)
        yhat = float(np.clip(past_fill[-1] + delta_hat, 0.0, 1.0))
        preds.append(yhat)
        past_fill.append(yhat)
        past_rain.append(rain_now)
        past_et0.append(et0_now)
        past_cons.append(cons_now)
    return np.asarray(preds, dtype=float)


def recursive_backtest(
    train_df: pd.DataFrame,
    specs: list[ModelSpec],
    min_train: int,
    horizons: list[int],
    origin_step: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_h = max(horizons)
    pred_rows = []
    metric_rows = []
    for spec in specs:
        for origin in range(min_train, len(train_df) - max_h + 1, origin_step):
            history = train_df.iloc[:origin].copy()
            future = train_df.iloc[origin : origin + max_h].copy().reset_index(drop=True)
            model = fit_model(spec, history)
            preds = recursive_forecast_known_exog(spec, model, history, future)
            origin_fill = float(history["weighted_total_fill"].iloc[-1])
            for h in horizons:
                actual = float(future["weighted_total_fill"].iloc[h - 1])
                pred = float(preds[h - 1])
                pred_rows.append(
                    {
                        "model": spec.name,
                        "origin_date": str(history["date"].iloc[-1].date()),
                        "target_date": str(future["date"].iloc[h - 1].date()),
                        "horizon_month": h,
                        "origin_fill": origin_fill,
                        "actual_fill": actual,
                        "pred_fill": pred,
                        "actual_change": actual - origin_fill,
                        "pred_change": pred - origin_fill,
                        "actual_below_40": int(actual < 0.40),
                        "pred_below_40": int(pred < 0.40),
                        "actual_below_30": int(actual < 0.30),
                        "pred_below_30": int(pred < 0.30),
                    }
                )
        pred_df = pd.DataFrame([row for row in pred_rows if row["model"] == spec.name])
        for h in horizons:
            g = pred_df[pred_df["horizon_month"] == h].copy()
            metric_rows.append(
                {
                    "model": spec.name,
                    "horizon_month": h,
                    "rmse_pp": rmse_pp(g["actual_fill"].to_numpy(), g["pred_fill"].to_numpy()),
                    "mae_pp": mae_pp(g["actual_fill"].to_numpy(), g["pred_fill"].to_numpy()),
                    "direction_accuracy": sign_accuracy(g["actual_change"].to_numpy(), g["pred_change"].to_numpy()),
                    "below_40_accuracy": float(np.mean(g["actual_below_40"] == g["pred_below_40"])),
                    "below_30_accuracy": float(np.mean(g["actual_below_30"] == g["pred_below_30"])),
                    "n_predictions": int(len(g)),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def simulate_endpoint(
    spec: ModelSpec,
    model,
    train_df: pd.DataFrame,
    future_exog: pd.DataFrame,
) -> float:
    proj = recursive_forecast_known_exog(spec, model, train_df, future_exog)
    return float(proj[-1])


def physics_sanity_checks(train_df: pd.DataFrame, specs: list[ModelSpec]) -> pd.DataFrame:
    forward = load_forward_module()
    clim = forward.monthly_climatology(train_df)
    _, demand_relief_pct = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    base_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == "base")
    rows = []
    for spec in specs:
        model = fit_model(spec, train_df)
        base_exog = forward.build_future_exog(
            train_df,
            base_cfg,
            clim,
            demand_relief_pct,
            transfer_share_anchor_pct=transfer_share_anchor_pct,
        )
        base_endpoint = simulate_endpoint(spec, model, train_df, base_exog)

        rain_cfg = replace(base_cfg, rain_end_pct_2040=base_cfg.rain_end_pct_2040 + 10.0)
        rain_exog = forward.build_future_exog(
            train_df,
            rain_cfg,
            clim,
            demand_relief_pct,
            transfer_share_anchor_pct=transfer_share_anchor_pct,
        )
        rain_endpoint = simulate_endpoint(spec, model, train_df, rain_exog)

        demand_exog = forward.build_future_exog(
            train_df,
            base_cfg,
            clim,
            demand_relief_pct,
            transfer_share_anchor_pct=transfer_share_anchor_pct,
        )
        demand_exog = forward.apply_direct_demand_adjustment(demand_exog, demand_end_pct_2040=10.0)
        demand_endpoint = simulate_endpoint(spec, model, train_df, demand_exog)

        et0_cfg = replace(base_cfg, et0_end_pct_2040=base_cfg.et0_end_pct_2040 + 10.0)
        et0_exog = forward.build_future_exog(
            train_df,
            et0_cfg,
            clim,
            demand_relief_pct,
            transfer_share_anchor_pct=transfer_share_anchor_pct,
        )
        et0_endpoint = simulate_endpoint(spec, model, train_df, et0_exog)

        transfer_cfg = replace(base_cfg, transfer_end_pct_2040=-20.0)
        transfer_exog = forward.build_future_exog(
            train_df,
            transfer_cfg,
            clim,
            demand_relief_pct,
            transfer_share_anchor_pct=transfer_share_anchor_pct,
        )
        transfer_endpoint = simulate_endpoint(spec, model, train_df, transfer_exog)

        rain_delta_pp = (rain_endpoint - base_endpoint) * 100.0
        demand_delta_pp = (demand_endpoint - base_endpoint) * 100.0
        et0_delta_pp = (et0_endpoint - base_endpoint) * 100.0
        transfer_delta_pp = (transfer_endpoint - base_endpoint) * 100.0
        pass_rain = int(rain_delta_pp > 0)
        pass_demand = int(demand_delta_pp < 0)
        pass_et0 = int(et0_delta_pp < 0)
        pass_transfer = int(transfer_delta_pp < 0)
        rows.append(
            {
                "model": spec.name,
                "base_endpoint_2040_pct": base_endpoint * 100.0,
                "rain_plus10_delta_pp": rain_delta_pp,
                "demand_plus10_delta_pp": demand_delta_pp,
                "et0_plus10_delta_pp": et0_delta_pp,
                "transfer_stress_delta_pp": transfer_delta_pp,
                "pass_rain_positive": pass_rain,
                "pass_demand_negative": pass_demand,
                "pass_et0_negative": pass_et0,
                "pass_transfer_negative": pass_transfer,
                "physics_pass_count": pass_rain + pass_demand + pass_et0 + pass_transfer,
            }
        )
    return pd.DataFrame(rows).sort_values(["physics_pass_count", "base_endpoint_2040_pct"], ascending=[False, False]).reset_index(drop=True)


def build_scorecard(one_step_df: pd.DataFrame, recursive_df: pd.DataFrame, sanity_df: pd.DataFrame) -> pd.DataFrame:
    avg_h = recursive_df.groupby("model", as_index=False).agg(
        mean_recursive_rmse_pp=("rmse_pp", "mean"),
        mean_direction_accuracy=("direction_accuracy", "mean"),
    )
    h12 = (
        recursive_df[recursive_df["horizon_month"] == 12][["model", "rmse_pp"]]
        .rename(columns={"rmse_pp": "rmse_h12_pp"})
        .reset_index(drop=True)
    )
    avg_h = avg_h.merge(h12, on="model", how="left")
    out = one_step_df.merge(avg_h, on="model", how="left").merge(sanity_df[["model", "physics_pass_count"]], on="model", how="left")
    out["rank_one_step"] = out["one_step_rmse_pp"].rank(method="min")
    out["rank_recursive"] = out["mean_recursive_rmse_pp"].rank(method="min")
    out["rank_h12"] = out["rmse_h12_pp"].rank(method="min")
    out["composite_score"] = out["rank_one_step"] + out["rank_recursive"] + out["rank_h12"] + (4 - out["physics_pass_count"]) * 0.75
    return out.sort_values(["composite_score", "mean_recursive_rmse_pp", "one_step_rmse_pp"]).reset_index(drop=True)


def plot_one_step_bar(df: pd.DataFrame, out_path: Path) -> None:
    tmp = df.sort_values("one_step_rmse_pp").copy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    ax.bar([PLOT_LABELS.get(m, m) for m in tmp["model"]], tmp["one_step_rmse_pp"], color="#2563eb")
    ax.set_ylabel("RMSE (yüzde puan)")
    ax.set_title("Tek adımlı yürüyen test karşılaştırması")
    ax.grid(True, axis="y", alpha=0.2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(18)
        tick.set_horizontalalignment("right")
    for i, val in enumerate(tmp["one_step_rmse_pp"]):
        ax.text(i, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_recursive_lines(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.0), dpi=180)
    for model, g in df.groupby("model"):
        g = g.sort_values("horizon_month")
        ax.plot(g["horizon_month"], g["rmse_pp"], marker="o", linewidth=1.8, label=PLOT_LABELS.get(model, model))
    ax.set_xlabel("Tahmin ufku (ay)")
    ax.set_ylabel("RMSE (yüzde puan)")
    ax.set_title("Çok adımlı geri test: ufka göre hata")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sanity(df: pd.DataFrame, out_path: Path) -> None:
    cols = ["rain_plus10_delta_pp", "demand_plus10_delta_pp", "et0_plus10_delta_pp", "transfer_stress_delta_pp"]
    labels = ["Yağış +%10", "Talep +%10", "ET0 +%10", "Transfer stresi"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), dpi=180, sharex=True)
    for ax, col, label in zip(axes.flatten(), cols, labels, strict=False):
        tmp = df.sort_values(col, ascending=False).copy()
        ax.bar([PLOT_LABELS.get(m, m) for m in tmp["model"]], tmp[col], color="#0f766e")
        ax.axhline(0, color="#6b7280", linewidth=1.0)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.2)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_horizontalalignment("right")
    fig.suptitle("Fiziksel işaret testi", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = OUT_DIR / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    train_df = load_training_frame()
    specs = model_specs()
    min_train = 60
    horizons = [1, 3, 6, 12]

    one_step_df, one_step_pred_df = one_step_walkforward(train_df, specs, min_train=min_train)
    recursive_df, recursive_pred_df = recursive_backtest(train_df, specs, min_train=min_train, horizons=horizons, origin_step=6)
    sanity_df = physics_sanity_checks(train_df, specs)
    scorecard_df = build_scorecard(one_step_df, recursive_df, sanity_df)

    one_step_df.to_csv(OUT_DIR / "one_step_metrics.csv", index=False)
    one_step_pred_df.to_csv(OUT_DIR / "one_step_predictions.csv", index=False)
    recursive_df.to_csv(OUT_DIR / "recursive_horizon_metrics.csv", index=False)
    recursive_pred_df.to_csv(OUT_DIR / "recursive_origin_predictions.csv", index=False)
    sanity_df.to_csv(OUT_DIR / "physical_sanity_checks.csv", index=False)
    scorecard_df.to_csv(OUT_DIR / "model_selection_scorecard.csv", index=False)

    plot_one_step_bar(one_step_df, figs / "one_step_rmse_bar.png")
    plot_recursive_lines(recursive_df, figs / "recursive_rmse_by_horizon.png")
    plot_sanity(sanity_df, figs / "physical_sanity_checks.png")

    best_stat = scorecard_df.iloc[0]
    physics_viable = scorecard_df[scorecard_df["physics_pass_count"] >= 3].copy()
    summary = {
        "training_start": str(train_df["date"].min().date()),
        "training_end": str(train_df["date"].max().date()),
        "min_train": min_train,
        "horizons_tested": horizons,
        "best_composite_model": str(best_stat["model"]),
        "best_composite_score": float(best_stat["composite_score"]),
        "best_statistical_one_step_model": str(one_step_df.iloc[0]["model"]),
        "best_recursive_mean_model": str(
            recursive_df.groupby("model", as_index=False)["rmse_pp"].mean().sort_values("rmse_pp").iloc[0]["model"]
        ),
        "best_physics_viable_model": (
            str(physics_viable.iloc[0]["model"]) if not physics_viable.empty else None
        ),
    }
    (OUT_DIR / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
