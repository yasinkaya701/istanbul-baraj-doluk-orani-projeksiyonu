#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

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


def load_driver_panel():
    path = ROOT / "output" / "newdata_feature_store" / "tables" / "istanbul_dam_driver_panel.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["fill_pct"] = df["weighted_total_fill"] * 100.0
    df["wb"] = df["rain_mm"] - df["et0_mm_month"]
    return df


def split_window(df, train_end: str, test_start: str, test_end: str):
    train = df[df["date"] <= train_end].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    return train, test


def model_sarimax(train, test):
    exog_cols = ["rain_mm", "et0_mm_month"]
    y_train = train["fill_pct"].values
    x_train = train[exog_cols].values
    y_test = test["fill_pct"].values
    x_test = test[exog_cols].values

    model = SARIMAX(
        y_train,
        exog=x_train,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred_test = res.forecast(steps=len(y_test), exog=x_test)
    return pred_test


def model_stl_ar(train, test):
    y = train["fill_pct"].values
    stl = STL(y, period=12, robust=True)
    res = stl.fit()
    trend = res.trend
    seasonal = res.seasonal
    resid = res.resid

    t = np.arange(len(trend))
    a, b = np.polyfit(t, trend, 1)

    if len(resid) > 2:
        r1 = np.corrcoef(resid[1:], resid[:-1])[0, 1]
    else:
        r1 = 0.0

    def forecast(n):
        t_future = np.arange(len(trend), len(trend) + n)
        trend_f = a * t_future + b
        seasonal_f = np.tile(seasonal[-12:], int(np.ceil(n / 12)))[:n]
        res_f = []
        prev = resid[-1] if len(resid) > 0 else 0.0
        for _ in range(n):
            prev = r1 * prev
            res_f.append(prev)
        res_f = np.array(res_f)
        return trend_f + seasonal_f + res_f

    return forecast(len(test))


def model_runoff_routing(train, test):
    data = train.copy()
    data["lag_fill"] = data["fill_pct"].shift(1)
    data["lag_wb"] = data["wb"].shift(1)
    data["delta"] = data["fill_pct"].shift(-1) - data["fill_pct"]
    fit = data.dropna(subset=["delta", "lag_fill", "lag_wb", "wb"]).copy()

    X = fit[["wb", "lag_fill", "lag_wb"]].values
    y = fit["delta"].values
    reg = Ridge(alpha=1.0)
    reg.fit(X, y)

    last_fill = train["fill_pct"].iloc[-1]
    last_wb = train["wb"].iloc[-1]
    pred_test = []
    for _, row in test.iterrows():
        Xr = np.array([[row["wb"], last_fill, last_wb]])
        delta = reg.predict(Xr)[0]
        last_fill = float(np.clip(last_fill + delta, 0.0, 100.0))
        last_wb = row["wb"]
        pred_test.append(last_fill)
    return np.array(pred_test)


def run_window(df, label, train_end, test_start, test_end):
    train, test = split_window(df, train_end, test_start, test_end)
    models = {
        "sarimax_exog": model_sarimax,
        "stl_ar": model_stl_ar,
        "runoff_routing": model_runoff_routing,
    }
    rows = []
    for name, fn in models.items():
        pred = fn(train, test)
        metrics = eval_metrics(test["fill_pct"].values, pred)
        rows.append({"window": label, "model": name, **metrics})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/istanbul_extra_models_research_2026_03_18")
    args = p.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_driver_panel()
    df = df.dropna(subset=["fill_pct", "rain_mm", "et0_mm_month"]).copy()
    df = df[(df["date"] >= "2001-01-01") & (df["date"] <= "2024-12-01")].copy()

    # 5-year window: 2016-2020
    res5 = run_window(df, "2016-2020", "2015-12-01", "2016-01-01", "2020-12-01")

    # 10-year window: 2014-2023 (train through 2013-12)
    res10 = run_window(df, "2014-2023", "2013-12-01", "2014-01-01", "2023-12-01")

    result = pd.concat([res5, res10], ignore_index=True)
    result.to_csv(out_dir / "extra_models_holdout_5y_10y_accuracy.csv", index=False)


if __name__ == "__main__":
    main()
