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
    parser = argparse.ArgumentParser(description="Evaluate annual official context as a monthly model block.")
    parser.add_argument(
        "--driver-panel",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/istanbul_dam_driver_panel.csv"),
    )
    parser.add_argument(
        "--annual-context",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_operational_context_annual.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store"),
    )
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument("--end-date", type=str, default="2023-12-01")
    parser.add_argument("--min-train", type=int, default=18)
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


def build_frame(driver_panel: Path, annual_context: Path, start_date: str, end_date: str) -> pd.DataFrame:
    monthly = pd.read_csv(driver_panel, parse_dates=["date"])
    annual = pd.read_csv(annual_context)

    context_cols = [
        "year",
        "nrw_pct",
        "reclaimed_share_of_system_input_pct",
        "authorized_consumption_l_per_active_subscriber_day",
        "active_subscribers",
    ]
    monthly["year"] = monthly["date"].dt.year
    df = monthly.merge(annual[context_cols], on="year", how="left", suffixes=("", "_annual"))
    df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))].copy()

    keep = [
        "date",
        "weighted_total_fill",
        "rain_model_mm",
        "consumption_mean_monthly",
        "et0_mm_month",
        "temp_proxy_c",
        "rh_proxy_pct",
        "month_sin",
        "month_cos",
        "nrw_pct",
        "reclaimed_share_of_system_input_pct",
        "authorized_consumption_l_per_active_subscriber_day",
        "active_subscribers",
    ]
    df = df[keep].sort_values("date").reset_index(drop=True)
    df = add_lags(df, ["weighted_total_fill"], (1, 2))

    roll_inputs = [
        "rain_model_mm",
        "consumption_mean_monthly",
        "et0_mm_month",
        "temp_proxy_c",
        "rh_proxy_pct",
    ]
    for base in roll_inputs:
        roll = f"{base}_roll2"
        anom = f"{base}_roll2_anom"
        df[roll] = df[base].rolling(2).mean()
        df[anom] = df[roll] - df.groupby(df["date"].dt.month)[roll].transform("mean")

    df["delta_fill"] = df["weighted_total_fill"] - df["weighted_total_fill_lag1"]
    return df.dropna().reset_index(drop=True)


def specs() -> list[ModelSpec]:
    base = [
        "weighted_total_fill_lag1",
        "weighted_total_fill_lag2",
        "month_sin",
        "month_cos",
        "rain_model_mm_roll2_anom",
        "consumption_mean_monthly_roll2_anom",
        "et0_mm_month_roll2_anom",
        "temp_proxy_c_roll2_anom",
        "rh_proxy_pct_roll2_anom",
    ]
    nrw = base + ["nrw_pct"]
    reuse = base + ["reclaimed_share_of_system_input_pct", "authorized_consumption_l_per_active_subscriber_day"]
    all_ctx = nrw + ["reclaimed_share_of_system_input_pct", "authorized_consumption_l_per_active_subscriber_day", "active_subscribers"]
    return [
        ModelSpec("baseline_temp_humidity", base),
        ModelSpec("plus_nrw", nrw),
        ModelSpec("plus_reuse_intensity", reuse),
        ModelSpec("plus_all_annual_context", all_ctx),
    ]


def fit_predict_walkforward(df: pd.DataFrame, spec: ModelSpec, min_train: int) -> pd.DataFrame:
    preds: list[dict[str, object]] = []
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
    return pd.DataFrame(preds)


def metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = pred_df["actual"].to_numpy(dtype=float)
    y_pred = pred_df["pred"].to_numpy(dtype=float)
    return {
        "rmse_pp": float(np.sqrt(mean_squared_error(y_true, y_pred)) * 100.0),
        "mae_pp": float(mean_absolute_error(y_true, y_pred) * 100.0),
        "smape": smape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.3, 4.7))
    colors = ["#111827", "#0f766e", "#2563eb", "#b45309"]
    ax.bar(metrics_df["model"], metrics_df["rmse_pp"], color=colors[: len(metrics_df)])
    ax.set_ylabel("RMSE (yuzde puan)")
    ax.set_title("Yillik resmi baglam blogu: kisa pencere walk-forward")
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

    frame = build_frame(args.driver_panel, args.annual_context, args.start_date, args.end_date)
    metric_rows: list[dict[str, object]] = []
    pred_parts: list[pd.DataFrame] = []

    for spec in specs():
        pred_df = fit_predict_walkforward(frame, spec, args.min_train)
        result = metrics(pred_df)
        result["model"] = spec.name
        result["n_predictions"] = int(len(pred_df))
        metric_rows.append(result)
        pred_parts.append(pred_df)

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse_pp").reset_index(drop=True)
    preds_df = pd.concat(pred_parts, ignore_index=True)

    metrics_df.to_csv(out_tables / "annual_context_monthly_model_metrics.csv", index=False)
    preds_df.to_csv(out_tables / "annual_context_monthly_predictions.csv", index=False)
    plot_metrics(metrics_df, out_figures / "annual_context_monthly_model_rmse.png")

    summary = {
        "window": {
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
            "rows": int(len(frame)),
            "min_train": int(args.min_train),
        },
        "sample_warning": "Short window because annual official context is only public for 2020-2023 in machine-readable form.",
        "models": metrics_df.to_dict(orient="records"),
    }
    (args.out_dir / "annual_context_monthly_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(out_tables / "annual_context_monthly_model_metrics.csv")
    print(out_tables / "annual_context_monthly_predictions.csv")
    print(out_figures / "annual_context_monthly_model_rmse.png")
    print(args.out_dir / "annual_context_monthly_model_summary.json")


if __name__ == "__main__":
    main()
