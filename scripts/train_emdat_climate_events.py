#!/usr/bin/env python3
"""
Train yearly climate-event models from EM-DAT Turkey event data.

Outputs:
- annual aggregated dataset
- model metrics
- holdout predictions
- 5-year recursive forecast
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CLIMATE_SUBGROUPS = {"Hydrological", "Meteorological", "Climatological"}
TARGETS = ["event_count_total", "total_deaths_sum", "total_affected_sum"]


@dataclass
class ModelResult:
    target: str
    model_name: str
    rmse: float
    mae: float
    r2: float
    train_rows: int
    test_rows: int
    model_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train models with Turkey climate-event EM-DAT dataset")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/spreadsheet/turkiye_iklim_olaylari_1900_2023_emdat_archive.csv"),
        help="Event-level CSV input path",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/spreadsheet/emdat_model_training"),
        help="Output directory",
    )
    p.add_argument("--lag", type=int, default=5, help="Number of yearly lag features")
    p.add_argument("--forecast-horizon", type=int, default=5, help="Forecast horizon in years")
    return p.parse_args()


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def prepare_annual_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # keep climate-related natural events if caller passed broader data.
    if {"disaster_group", "disaster_subgroup"}.issubset(df.columns):
        mask = (df["disaster_group"] == "Natural") & (df["disaster_subgroup"].isin(CLIMATE_SUBGROUPS))
        df = df[mask].copy()

    df["year"] = _to_numeric(df["start_date"].astype(str).str[:4])
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    df["total_deaths"] = _to_numeric(df.get("total_deaths", pd.Series(dtype=float))).fillna(0.0)
    df["total_affected"] = _to_numeric(df.get("total_affected", pd.Series(dtype=float))).fillna(0.0)

    grp = df.groupby("year", as_index=False).agg(
        event_count_total=("event_id", "count"),
        total_deaths_sum=("total_deaths", "sum"),
        total_affected_sum=("total_affected", "sum"),
    )

    # Ensure continuous year index (helps autoregressive training)
    full_years = pd.DataFrame({"year": np.arange(grp["year"].min(), grp["year"].max() + 1)})
    out = full_years.merge(grp, on="year", how="left").fillna(0.0)
    out = out.sort_values("year").reset_index(drop=True)
    return out


def make_supervised(df: pd.DataFrame, target: str, lag: int) -> pd.DataFrame:
    out = df[["year", target]].copy()
    for i in range(1, lag + 1):
        out[f"lag_{i}"] = out[target].shift(i)
    out["rolling_mean_3"] = out[target].shift(1).rolling(3).mean()
    out["rolling_std_3"] = out[target].shift(1).rolling(3).std().fillna(0.0)
    out = out.dropna().reset_index(drop=True)
    return out


def get_models() -> dict[str, Callable[[], object]]:
    return {
        "LinearRegression": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
    }


def train_one_target(
    annual_df: pd.DataFrame,
    target: str,
    lag: int,
    out_dir: Path,
) -> tuple[list[dict], list[dict], ModelResult]:
    sup = make_supervised(annual_df, target=target, lag=lag)
    if len(sup) < 18:
        raise SystemExit(f"Not enough rows for training target={target}. rows={len(sup)}")

    test_rows = max(8, int(round(len(sup) * 0.2)))
    test_rows = min(test_rows, len(sup) - 5)
    train_rows = len(sup) - test_rows

    feats = [c for c in sup.columns if c.startswith("lag_")] + ["rolling_mean_3", "rolling_std_3"]
    X = sup[feats]
    y = sup[target]

    X_train, X_test = X.iloc[:train_rows], X.iloc[train_rows:]
    y_train, y_test = y.iloc[:train_rows], y.iloc[train_rows:]
    years_test = sup["year"].iloc[train_rows:]

    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []

    best_name = ""
    best_rmse = float("inf")
    best_model = None

    for model_name, ctor in get_models().items():
        model = ctor()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))

        metrics_rows.append(
            {
                "target": target,
                "model": model_name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "train_rows": int(train_rows),
                "test_rows": int(test_rows),
            }
        )

        for yr, yt, yp in zip(years_test, y_test, pred):
            pred_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "year": int(yr),
                    "actual": float(yt),
                    "pred": float(yp),
                    "abs_error": float(abs(yt - yp)),
                }
            )

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = model_name
            best_model = model

    assert best_model is not None
    model_path = out_dir / f"best_model_{target}.joblib"
    joblib.dump({"model": best_model, "features": feats, "target": target, "lag": lag}, model_path)

    chosen = [m for m in metrics_rows if m["model"] == best_name][0]
    result = ModelResult(
        target=target,
        model_name=best_name,
        rmse=float(chosen["rmse"]),
        mae=float(chosen["mae"]),
        r2=float(chosen["r2"]),
        train_rows=int(chosen["train_rows"]),
        test_rows=int(chosen["test_rows"]),
        model_path=model_path,
    )
    return metrics_rows, pred_rows, result


def recursive_forecast(
    annual_df: pd.DataFrame,
    target: str,
    lag: int,
    model_bundle: dict,
    horizon: int,
) -> pd.DataFrame:
    model = model_bundle["model"]

    series = annual_df[["year", target]].copy()
    last_year = int(series["year"].max())
    history = list(series[target].astype(float).values)

    rows: list[dict] = []
    for step in range(1, horizon + 1):
        if len(history) < lag:
            break

        lags = history[-lag:][::-1]  # lag_1 first
        prev_vals = history[-3:]
        rolling_mean_3 = float(np.mean(prev_vals))
        rolling_std_3 = float(np.std(prev_vals))

        feat_values = lags + [rolling_mean_3, rolling_std_3]
        feat = pd.DataFrame([feat_values], columns=model_bundle["features"])
        yhat = float(model.predict(feat)[0])

        forecast_year = last_year + step
        rows.append(
            {
                "target": target,
                "year": forecast_year,
                "forecast": yhat,
                "model": model_bundle.get("model_name", ""),
            }
        )
        history.append(yhat)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    raw = pd.read_csv(args.input_csv, dtype=str)
    annual = prepare_annual_dataset(raw)
    annual_out = args.out_dir / "annual_climate_events_dataset.csv"
    annual.to_csv(annual_out, index=False)

    all_metrics: list[dict] = []
    all_preds: list[dict] = []
    chosen_rows: list[dict] = []
    all_fc: list[pd.DataFrame] = []

    for target in TARGETS:
        metrics_rows, pred_rows, result = train_one_target(
            annual_df=annual,
            target=target,
            lag=args.lag,
            out_dir=args.out_dir,
        )
        all_metrics.extend(metrics_rows)
        all_preds.extend(pred_rows)
        chosen_rows.append(
            {
                "target": result.target,
                "best_model": result.model_name,
                "rmse": result.rmse,
                "mae": result.mae,
                "r2": result.r2,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "model_path": str(result.model_path),
            }
        )

        bundle = joblib.load(result.model_path)
        bundle["model_name"] = result.model_name
        fc = recursive_forecast(
            annual_df=annual,
            target=target,
            lag=args.lag,
            model_bundle=bundle,
            horizon=args.forecast_horizon,
        )
        all_fc.append(fc)

    metrics_df = pd.DataFrame(all_metrics).sort_values(["target", "rmse"])
    preds_df = pd.DataFrame(all_preds).sort_values(["target", "model", "year"])
    best_df = pd.DataFrame(chosen_rows).sort_values("target")
    fc_df = pd.concat(all_fc, ignore_index=True).sort_values(["target", "year"])

    metrics_path = args.out_dir / "emdat_model_metrics.csv"
    preds_path = args.out_dir / "emdat_model_predictions.csv"
    best_path = args.out_dir / "emdat_best_models.csv"
    fc_path = args.out_dir / "emdat_forecast_next_years.csv"

    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    best_df.to_csv(best_path, index=False)
    fc_df.to_csv(fc_path, index=False)

    print("Training completed.")
    print(f"- Annual dataset: {annual_out}")
    print(f"- Metrics: {metrics_path}")
    print(f"- Predictions: {preds_path}")
    print(f"- Best models: {best_path}")
    print(f"- Forecast: {fc_path}")
    print("\nBest models:")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
