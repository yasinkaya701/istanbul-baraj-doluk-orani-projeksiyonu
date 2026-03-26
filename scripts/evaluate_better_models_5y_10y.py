#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path("/Users/yasinkaya/Hackhaton")


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(1e-6, np.abs(y_true))
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(1e-6, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


def corr(a, b):
    if len(a) < 2:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def eval_metrics(y_true, y_pred):
    return {
        "rmse_pp": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_pp": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mape(y_true, y_pred)),
        "smape_pct": float(smape(y_true, y_pred)),
        "pearson_corr_pct": float(corr(y_true, y_pred) * 100.0),
    }


def load_driver_panel(panel_path: Path) -> pd.DataFrame:
    path = panel_path
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["fill_pct"] = df["weighted_total_fill"] * 100.0
    df["lag1_fill_pct"] = df["fill_pct"].shift(1)
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def build_features(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    cols = [
        "rain_mm",
        "et0_mm_month",
        "t_mean_c",
        "rh_mean_pct",
        "pressure_kpa",
        "vpd_kpa_mean",
        "month_sin",
        "month_cos",
        "lag1_fill_pct",
    ]
    cols = [c for c in cols if c not in drop_cols]
    out = df[["date", "fill_pct"] + cols].copy()
    out = out.dropna(subset=cols + ["fill_pct"]).copy()
    return out


def split_window(df, train_end, test_start, test_end):
    train = df[df["date"] <= train_end].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    return train, test


def run_models(train, test):
    X_train = train.drop(columns=["date", "fill_pct"]).values
    y_train = train["fill_pct"].values
    X_test = test.drop(columns=["date", "fill_pct"]).values
    y_test = test["fill_pct"].values

    models = {
        "ridge": Ridge(alpha=1.0),
        "gbr": GradientBoostingRegressor(random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "etr": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }

    rows = []
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({"model": name, **eval_metrics(y_test, y_pred)})
        preds[name] = y_pred

    return pd.DataFrame(rows), preds, y_test


def yearly_comparison(test_df, y_test, preds):
    out = []
    years = pd.to_datetime(test_df["date"]).dt.year
    for name, y_pred in preds.items():
        df = pd.DataFrame({
            "year": years.values,
            "actual": y_test,
            "pred": y_pred,
        })
        g = df.groupby("year", as_index=False).agg(actual_mean=("actual", "mean"), pred_mean=("pred", "mean"))
        g["model"] = name
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/istanbul_better_models_5y_10y")
    p.add_argument("--panel", default="output/newdata_feature_store/tables/istanbul_dam_driver_panel.csv")
    p.add_argument("--drop-cols", default="")
    args = p.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = ROOT / args.panel
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    df = load_driver_panel(panel_path)
    df = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2024-12-01")].copy()
    feat = build_features(df, drop_cols)

    # Determine windows based on available data
    end_date = feat["date"].max()
    # 5-year window (60 months) ending at end_date
    test5_start = (end_date - pd.DateOffset(months=59)).normalize()
    test5_end = end_date
    train5_end = (test5_start - pd.DateOffset(months=1)).normalize()

    # 10-year window (120 months) ending at end_date
    test10_start = (end_date - pd.DateOffset(months=119)).normalize()
    test10_end = end_date
    train10_end = (test10_start - pd.DateOffset(months=1)).normalize()

    # 5-year window
    train5, test5 = split_window(feat, train5_end, test5_start, test5_end)
    res5, preds5, y5 = run_models(train5, test5)
    res5["window"] = f"{test5_start.date()} -> {test5_end.date()}"
    res5.to_csv(out_dir / "metrics_5y.csv", index=False)
    yearly5 = yearly_comparison(test5, y5, preds5)
    yearly5.to_csv(out_dir / "yearly_compare_5y.csv", index=False)

    # 10-year window
    train10, test10 = split_window(feat, train10_end, test10_start, test10_end)
    res10, preds10, y10 = run_models(train10, test10)
    res10["window"] = f"{test10_start.date()} -> {test10_end.date()}"
    res10.to_csv(out_dir / "metrics_10y.csv", index=False)
    yearly10 = yearly_comparison(test10, y10, preds10)
    yearly10.to_csv(out_dir / "yearly_compare_10y.csv", index=False)

    summary = pd.concat([res5, res10], ignore_index=True)
    summary.to_csv(out_dir / "metrics_5y_10y.csv", index=False)


if __name__ == "__main__":
    main()
