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


def load_driver_panel():
    path = ROOT / "output" / "newdata_feature_store" / "tables" / "istanbul_dam_driver_panel.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["fill_pct"] = df["weighted_total_fill"] * 100.0
    df["wb"] = df["rain_mm"] - df["et0_mm_month"]
    return df


def load_future_exog():
    path = ROOT / "output" / "scientific_climate_projection_2026_2040" / "climate_projection_2010_2040_monthly.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2026-01-01"].copy()
    df = df.rename(columns={"precip_mm_month": "rain_mm", "et0_mm_month": "et0_mm_month"})
    df["wb"] = df["rain_mm"] - df["et0_mm_month"]
    return df[["date", "rain_mm", "et0_mm_month", "wb"]]


def split_holdout(df, train_end="2015-12-01", test_start="2016-01-01", test_end="2020-12-01"):
    train = df[df["date"] <= train_end].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    return train, test


def eval_metrics(y_true, y_pred):
    return {
        "rmse_pp": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_pp": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mape(y_true, y_pred)),
        "smape_pct": float(smape(y_true, y_pred)),
        "pearson_corr_pct": float(corr(y_true, y_pred) * 100.0),
    }


def model_sarimax(train, test, future):
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

    # Future
    x_future = future[exog_cols].values
    pred_future = res.forecast(steps=len(x_future), exog=x_future)

    return pred_test, pred_future


def model_stl_ar(train, test, future):
    y = train["fill_pct"].values
    dates = train["date"].values

    stl = STL(y, period=12, robust=True)
    res = stl.fit()
    trend = res.trend
    seasonal = res.seasonal
    resid = res.resid

    # Trend forecast by linear regression on time index
    t = np.arange(len(trend))
    a, b = np.polyfit(t, trend, 1)

    # Residual AR(1)
    if len(resid) > 2:
        r1 = np.corrcoef(resid[1:], resid[:-1])[0, 1]
    else:
        r1 = 0.0

    def forecast(n):
        t_future = np.arange(len(trend), len(trend) + n)
        trend_f = a * t_future + b
        seasonal_f = np.tile(seasonal[-12:], int(np.ceil(n / 12)))[:n]
        # AR(1) residual forecast
        res_f = []
        prev = resid[-1] if len(resid) > 0 else 0.0
        for _ in range(n):
            prev = r1 * prev
            res_f.append(prev)
        res_f = np.array(res_f)
        return trend_f + seasonal_f + res_f

    pred_test = forecast(len(test))
    pred_future = forecast(len(future))
    return pred_test, pred_future


def model_runoff_routing(train, test, future):
    # delta fill = a*wb + b*lag1_fill + c*lag1_wb + d
    data = train.copy()
    data["lag_fill"] = data["fill_pct"].shift(1)
    data["lag_wb"] = data["wb"].shift(1)
    data["delta"] = data["fill_pct"].shift(-1) - data["fill_pct"]
    fit = data.dropna(subset=["delta", "lag_fill", "lag_wb", "wb"]).copy()

    X = fit[["wb", "lag_fill", "lag_wb"]].values
    y = fit["delta"].values
    reg = Ridge(alpha=1.0)
    reg.fit(X, y)

    # Predict test recursively
    last_fill = train["fill_pct"].iloc[-1]
    last_wb = train["wb"].iloc[-1]
    pred_test = []
    for _, row in test.iterrows():
        Xr = np.array([[row["wb"], last_fill, last_wb]])
        delta = reg.predict(Xr)[0]
        last_fill = float(np.clip(last_fill + delta, 0.0, 100.0))
        last_wb = row["wb"]
        pred_test.append(last_fill)

    # Future recursively
    pred_future = []
    for _, row in future.iterrows():
        Xr = np.array([[row["wb"], last_fill, last_wb]])
        delta = reg.predict(Xr)[0]
        last_fill = float(np.clip(last_fill + delta, 0.0, 100.0))
        last_wb = row["wb"]
        pred_future.append(last_fill)

    return np.array(pred_test), np.array(pred_future)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/istanbul_extra_models_research_2026_03_18")
    args = p.parse_args()
    out_dir = ROOT / args.out
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_driver_panel()
    df = df.dropna(subset=["fill_pct", "rain_mm", "et0_mm_month"]).copy()
    df = df[(df["date"] >= "2001-01-01") & (df["date"] <= "2024-12-01")].copy()

    train, test = split_holdout(df)
    future = load_future_exog()

    results = []
    preds = {}

    # SARIMAX
    pred_test, pred_future = model_sarimax(train, test, future)
    results.append({"model": "sarimax_exog", **eval_metrics(test["fill_pct"], pred_test)})
    preds["sarimax_exog"] = (pred_test, pred_future)

    # STL-AR
    pred_test, pred_future = model_stl_ar(train, test, future)
    results.append({"model": "stl_ar", **eval_metrics(test["fill_pct"], pred_test)})
    preds["stl_ar"] = (pred_test, pred_future)

    # Runoff-routing hybrid
    pred_test, pred_future = model_runoff_routing(train, test, future)
    results.append({"model": "runoff_routing", **eval_metrics(test["fill_pct"], pred_test)})
    preds["runoff_routing"] = (pred_test, pred_future)

    results_df = pd.DataFrame(results).sort_values("mape_pct")
    results_df.to_csv(out_dir / "extra_models_holdout_summary_2015_train_2020_test.csv", index=False)

    # Save predictions
    pred_df = pd.DataFrame({"date": test["date"].values, "actual": test["fill_pct"].values})
    for m, (pt, _) in preds.items():
        pred_df[m] = pt
    pred_df.to_csv(out_dir / "extra_models_holdout_predictions_2016_2020.csv", index=False)

    fut_df = pd.DataFrame({"date": future["date"].values})
    for m, (_, pf) in preds.items():
        fut_df[m] = pf
    fut_df.to_csv(out_dir / "extra_models_future_base_2026_2040.csv", index=False)

    # Plot holdout
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(test["date"], test["fill_pct"], color="#222222", label="Gozlenen")
    for m in preds:
        plt.plot(test["date"], preds[m][0], label=m)
    plt.title("Extra Modeller - Holdout 2016-2020")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "extra_models_holdout.png", dpi=160)
    plt.close()

    # Plot future
    plt.figure(figsize=(12, 5))
    for m in preds:
        plt.plot(future["date"], preds[m][1], label=m)
    plt.title("Extra Modeller - 2026-2040 Baz Projeksiyon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "extra_models_future.png", dpi=160)
    plt.close()

    summary = {
        "train_end": "2015-12",
        "test_window": "2016-01 -> 2020-12",
        "future_window": "2026-01 -> 2040-12",
        "best_model": results_df.iloc[0]["model"],
    }
    (out_dir / "extra_models_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
