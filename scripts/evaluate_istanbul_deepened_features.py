#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    features: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deepened monthly climate features for Istanbul dam forecasting.")
    parser.add_argument(
        "--driver-panel",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/istanbul_dam_driver_panel.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store"),
    )
    parser.add_argument("--start-date", type=str, default="2011-01-01")
    parser.add_argument("--end-date", type=str, default="2021-12-01")
    parser.add_argument("--min-train", type=int, default=60)
    return parser.parse_args()


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def add_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def build_frame(driver_panel: Path, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_csv(driver_panel, parse_dates=["date"])
    df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))].copy()
    keep = [
        "date",
        "weighted_total_fill",
        "rain_model_mm",
        "consumption_mean_monthly",
        "et0_mm_month",
        "vpd_kpa_mean",
        "temp_proxy_c",
        "rh_proxy_pct",
        "water_balance_proxy_mm",
        "month_sin",
        "month_cos",
    ]
    df = df[keep].sort_values("date").reset_index(drop=True)
    df = add_lags(df, ["weighted_total_fill"], (1, 2))
    for base in [
        "rain_model_mm",
        "et0_mm_month",
        "consumption_mean_monthly",
        "vpd_kpa_mean",
        "temp_proxy_c",
        "rh_proxy_pct",
        "water_balance_proxy_mm",
    ]:
        roll = f"{base}_roll2"
        anom = f"{base}_roll2_anom"
        df[roll] = df[base].rolling(2).mean()
        df[anom] = df[roll] - df.groupby(df["date"].dt.month)[roll].transform("mean")
    df["delta_fill"] = df["weighted_total_fill"] - df["weighted_total_fill_lag1"]
    return df.dropna().reset_index(drop=True)


def specs() -> list[ModelSpec]:
    baseline = [
        "weighted_total_fill_lag1",
        "weighted_total_fill_lag2",
        "month_sin",
        "month_cos",
        "rain_model_mm_roll2_anom",
        "et0_mm_month_roll2_anom",
        "consumption_mean_monthly_roll2_anom",
    ]
    plus_vpd_balance = baseline + [
        "vpd_kpa_mean_roll2_anom",
        "water_balance_proxy_mm_roll2_anom",
    ]
    plus_temp_humidity = baseline + [
        "temp_proxy_c_roll2_anom",
        "rh_proxy_pct_roll2_anom",
    ]
    deep_all = plus_vpd_balance + [
        "temp_proxy_c_roll2_anom",
        "rh_proxy_pct_roll2_anom",
    ]
    return [
        ModelSpec("baseline_full", baseline),
        ModelSpec("plus_vpd_balance", plus_vpd_balance),
        ModelSpec("plus_temp_humidity", plus_temp_humidity),
        ModelSpec("deep_all", deep_all),
    ]


def fit_predict_walkforward(df: pd.DataFrame, spec: ModelSpec, min_train: int) -> tuple[pd.DataFrame, Pipeline]:
    preds: list[dict[str, object]] = []
    pipeline: Pipeline | None = None
    for idx in range(min_train, len(df)):
        train = df.iloc[:idx]
        test = df.iloc[[idx]]
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]
        )
        pipeline.fit(train[spec.features], train["delta_fill"])
        delta_hat = float(pipeline.predict(test[spec.features])[0])
        yhat = float(test["weighted_total_fill_lag1"].iloc[0] + delta_hat)
        preds.append(
            {
                "date": test["date"].iloc[0],
                "actual": float(test["weighted_total_fill"].iloc[0]),
                "pred": yhat,
                "model": spec.name,
            }
        )
    assert pipeline is not None
    return pd.DataFrame(preds), pipeline


def metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = pred_df["actual"].to_numpy(dtype=float)
    y_pred = pred_df["pred"].to_numpy(dtype=float)
    return {
        "rmse_pp": float(np.sqrt(mean_squared_error(y_true, y_pred)) * 100.0),
        "mae_pp": float(mean_absolute_error(y_true, y_pred) * 100.0),
        "smape": smape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def extract_coefficients(pipeline: Pipeline, spec: ModelSpec) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    coef = np.asarray(model.coef_, dtype=float)
    return pd.DataFrame({"feature": spec.features, "coefficient": coef})


def plot_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = ["#111827", "#0f766e", "#2563eb", "#b45309"]
    ax.bar(metrics_df["model"], metrics_df["rmse_pp"], color=colors[: len(metrics_df)])
    ax.set_title("Walk-forward RMSE: derinlestirilmis ozellikler")
    ax.set_ylabel("RMSE (yuzde puan)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_figures = args.out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    frame = build_frame(args.driver_panel, args.start_date, args.end_date)

    metric_rows: list[dict[str, object]] = []
    pred_parts: list[pd.DataFrame] = []
    coef_parts: list[pd.DataFrame] = []
    best_model = None
    best_rmse = None

    for spec in specs():
        pred_df, pipeline = fit_predict_walkforward(frame, spec, args.min_train)
        result = metrics(pred_df)
        result["model"] = spec.name
        result["n_predictions"] = int(len(pred_df))
        metric_rows.append(result)
        pred_parts.append(pred_df)
        coef_df = extract_coefficients(pipeline, spec)
        coef_df["model"] = spec.name
        coef_parts.append(coef_df)
        if best_rmse is None or result["rmse_pp"] < best_rmse:
            best_rmse = result["rmse_pp"]
            best_model = spec.name

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse_pp").reset_index(drop=True)
    preds_df = pd.concat(pred_parts, ignore_index=True)
    coef_df = pd.concat(coef_parts, ignore_index=True)

    plot_metrics(metrics_df, out_figures / "deepened_feature_model_rmse.png")
    metrics_df.to_csv(out_tables / "deepened_feature_model_metrics.csv", index=False)
    preds_df.to_csv(out_tables / "deepened_feature_predictions.csv", index=False)
    coef_df.to_csv(out_tables / "deepened_feature_coefficients.csv", index=False)

    summary = {
        "window": {
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
            "rows": int(len(frame)),
            "min_train": int(args.min_train),
        },
        "best_model": best_model,
        "best_rmse_pp": float(best_rmse) if best_rmse is not None else None,
        "models": metrics_df.to_dict(orient="records"),
    }
    (args.out_dir / "deepened_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(out_tables / "deepened_feature_model_metrics.csv")
    print(out_tables / "deepened_feature_predictions.csv")
    print(out_tables / "deepened_feature_coefficients.csv")
    print(out_figures / "deepened_feature_model_rmse.png")
    print(args.out_dir / "deepened_feature_summary.json")


if __name__ == "__main__":
    main()
