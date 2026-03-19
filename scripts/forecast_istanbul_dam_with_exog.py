#!/usr/bin/env python3
"""Istanbul overall dam occupancy forecast with exogenous variables.

Uses:
- Occupancy resource (daily dam fill ratios)
- Rain + daily consumption resource

Method:
- Monthly aggregation
- Rolling-origin CV (12-month test windows)
- Candidate models: seasonal naive, SARIMA, SARIMAX(exog)
- Inverse-RMSE weighted ensemble
- Conformal absolute-error interval
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OCC_RESOURCE = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"
EXOG_RESOURCE = "762b802e-c5f9-4175-a5c1-78b892d9764b"
API_URL = "https://data.ibb.gov.tr/api/3/action/datastore_search"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Istanbul occupancy forecast with exogenous rainfall/consumption")
    p.add_argument("--api-url", default=API_URL)
    p.add_argument("--occupancy-resource-id", default=OCC_RESOURCE)
    p.add_argument("--exog-resource-id", default=EXOG_RESOURCE)
    p.add_argument("--output-dir", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_exog"))
    p.add_argument("--horizon-months", type=int, default=36)
    p.add_argument("--season-len", type=int, default=12)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-test-months", type=int, default=12)
    p.add_argument("--min-train-months", type=int, default=72)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--lag-rain-months", type=int, default=4, help="Rainfall lag (months) for occupancy response")
    p.add_argument("--lag-consumption-months", type=int, default=3, help="Consumption lag (months) for occupancy response")
    return p.parse_args()


def fetch_records(api_url: str, resource_id: str, limit: int = 5000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode({"resource_id": resource_id, "limit": str(limit), "offset": str(offset)})
        with urllib.request.urlopen(f"{api_url}?{query}", timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError(f"API returned success=false for resource_id={resource_id}")
        result = payload.get("result", {})
        chunk = result.get("records", [])
        records.extend(chunk)
        total = int(result.get("total", len(records)))
        if len(records) >= total or not chunk:
            break
        offset += len(chunk)
    return records


def to_numeric(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    return float(s.replace(",", "."))


def seasonal_naive(values: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.full(horizon, np.nan, dtype=float)
    if len(v) < season_len:
        return np.repeat(v[-1], horizon)
    base = v[-season_len:]
    rep = int(math.ceil(horizon / season_len))
    return np.tile(base, rep)[:horizon]


def build_occupancy_monthly(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    dam_cols = [c for c in df.columns if c not in {"_id", "Tarih"}]
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Tarih"], errors="coerce")
    for c in dam_cols:
        out[c] = df[c].map(to_numeric)
        mask = out[c] > 1.2
        out.loc[mask, c] = out.loc[mask, c] / 100.0
        out[c] = out[c].clip(lower=0.0, upper=1.0)
    out["overall_mean"] = out[dam_cols].mean(axis=1, skipna=True)
    out = out.dropna(subset=["date"]).sort_values("date")
    out["ds"] = out["date"].dt.to_period("M").dt.to_timestamp()
    monthly = out.groupby("ds", as_index=False)["overall_mean"].mean()
    return monthly


def build_exog_monthly(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Tarih"], errors="coerce")

    cons_col = "Istanbul gunluk tuketim(m3/gun)"
    if cons_col not in df.columns:
        raise ValueError("Consumption column not found in exogenous resource.")

    ex_cols = [c for c in df.columns if c not in {"_id", "Tarih", cons_col}]
    for c in ex_cols:
        out[c] = df[c].map(to_numeric)
    out["consumption_m3_day"] = df[cons_col].map(to_numeric)

    out["rain_mean_daily"] = out[ex_cols].mean(axis=1, skipna=True)
    out = out.dropna(subset=["date"]).sort_values("date")
    out["ds"] = out["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        out.groupby("ds", as_index=False)
        .agg(
            rain_sum_monthly=("rain_mean_daily", "sum"),
            rain_mean_monthly=("rain_mean_daily", "mean"),
            consumption_mean_monthly=("consumption_m3_day", "mean"),
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return monthly


def sarima_forecast(train_y: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    if len(train_y) < max(3 * season_len, 36):
        return np.full(horizon, np.nan, dtype=float)
    orders = [(1, 1, 1), (2, 1, 1), (1, 0, 1)]
    sorders = [(1, 0, 1, season_len), (0, 1, 1, season_len), (1, 1, 0, season_len)]
    best_aic = float("inf")
    best_fit = None
    for o in orders:
        for so in sorders:
            try:
                fit = SARIMAX(
                    train_y,
                    order=o,
                    seasonal_order=so,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic = float(fit.aic)
                    best_fit = fit
            except Exception:
                continue
    if best_fit is None:
        return np.full(horizon, np.nan, dtype=float)
    return np.asarray(best_fit.get_forecast(steps=horizon).predicted_mean, dtype=float)


def build_lagged_exog_future(
    train_x_raw: np.ndarray,
    horizon: int,
    season_len: int,
    lag_rain: int,
    lag_cons: int,
) -> np.ndarray:
    rain_hist = train_x_raw[:, 0]
    cons_hist = train_x_raw[:, 1]
    future_raw = exog_future_naive(train_x_raw, horizon=horizon, season_len=season_len)
    rain_full = np.concatenate([rain_hist, future_raw[:, 0]])
    cons_full = np.concatenate([cons_hist, future_raw[:, 1]])

    n = len(train_x_raw)
    rain_slice = rain_full[n - lag_rain : n - lag_rain + horizon]
    cons_slice = cons_full[n - lag_cons : n - lag_cons + horizon]
    return np.column_stack([rain_slice, cons_slice])


def build_lagged_exog_train(train_x_raw: np.ndarray, lag_rain: int, lag_cons: int) -> tuple[np.ndarray, int]:
    max_lag = max(lag_rain, lag_cons)
    rain = train_x_raw[:, 0]
    cons = train_x_raw[:, 1]
    ex = np.column_stack(
        [
            rain[max_lag - lag_rain : len(rain) - lag_rain],
            cons[max_lag - lag_cons : len(cons) - lag_cons],
        ]
    )
    return ex, max_lag


def sarimax_exog_forecast(
    train_y: np.ndarray,
    train_x_raw: np.ndarray,
    horizon: int,
    season_len: int,
    lag_rain: int,
    lag_cons: int,
) -> np.ndarray:
    max_lag = max(lag_rain, lag_cons)
    if len(train_y) < max(3 * season_len, 36, max_lag + 24):
        return np.full(horizon, np.nan, dtype=float)

    x_train, max_lag = build_lagged_exog_train(train_x_raw, lag_rain=lag_rain, lag_cons=lag_cons)
    y_train = train_y[max_lag:]
    x_future = build_lagged_exog_future(
        train_x_raw=train_x_raw,
        horizon=horizon,
        season_len=season_len,
        lag_rain=lag_rain,
        lag_cons=lag_cons,
    )

    orders = [(1, 1, 1), (2, 1, 1), (1, 0, 1)]
    sorders = [(1, 0, 1, season_len), (0, 1, 1, season_len), (1, 1, 0, season_len)]
    best_aic = float("inf")
    best_fit = None
    for o in orders:
        for so in sorders:
            try:
                fit = SARIMAX(
                    y_train,
                    exog=x_train,
                    order=o,
                    seasonal_order=so,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic = float(fit.aic)
                    best_fit = fit
            except Exception:
                continue
    if best_fit is None:
        return np.full(horizon, np.nan, dtype=float)
    return np.asarray(best_fit.get_forecast(steps=horizon, exog=x_future).predicted_mean, dtype=float)


def clip_ratio(v: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=float), 0.0, 1.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true) + np.abs(y_pred)
    den = np.where(den == 0, np.nan, den)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / den) * 100.0)


def exog_future_naive(train_x: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    cols = []
    for j in range(train_x.shape[1]):
        cols.append(seasonal_naive(train_x[:, j], horizon=horizon, season_len=season_len))
    return np.column_stack(cols)


def rolling_cv(
    df: pd.DataFrame,
    season_len: int,
    cv_splits: int,
    cv_test_months: int,
    min_train: int,
    lag_rain: int,
    lag_cons: int,
) -> pd.DataFrame:
    y = df["overall_mean"].to_numpy(dtype=float)
    x = df[["rain_sum_monthly", "consumption_mean_monthly"]].to_numpy(dtype=float)
    ds = pd.to_datetime(df["ds"])
    n = len(df)
    rows: list[dict[str, Any]] = []
    for split in range(cv_splits):
        test_end = n - (cv_splits - split - 1) * cv_test_months
        test_start = test_end - cv_test_months
        train_end = test_start
        if train_end < min_train:
            continue
        y_train = y[:train_end]
        y_test = y[test_start:test_end]
        x_train = x[:train_end]

        h = len(y_test)

        preds = {
            "seasonal_naive": clip_ratio(seasonal_naive(y_train, horizon=h, season_len=season_len)),
            "sarima": clip_ratio(sarima_forecast(y_train, horizon=h, season_len=season_len)),
            "sarimax_exog": clip_ratio(
                sarimax_exog_forecast(
                    train_y=y_train,
                    train_x_raw=x_train,
                    horizon=h,
                    season_len=season_len,
                    lag_rain=lag_rain,
                    lag_cons=lag_cons,
                )
            ),
        }

        for model_name, p in preds.items():
            for i in range(h):
                rows.append(
                    {
                        "split": split + 1,
                        "ds": ds.iloc[test_start + i],
                        "model": model_name,
                        "actual": float(y_test[i]),
                        "yhat": float(p[i]),
                        "abs_err": float(abs(y_test[i] - p[i])),
                    }
                )
    return pd.DataFrame(rows)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_weights(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in cv.groupby("model"):
        y = g["actual"].to_numpy(dtype=float)
        p = g["yhat"].to_numpy(dtype=float)
        r = rmse(y, p)
        rows.append({"model": m, "rmse": r, "smape": smape(y, p), "mae": float(np.mean(np.abs(y - p))), "n": len(g)})
    s = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    inv = 1.0 / np.maximum(s["rmse"].to_numpy(dtype=float), 1e-8)
    w = inv / inv.sum()
    s["weight"] = w
    return s


def cv_ensemble_errors(cv: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    piv = cv.pivot_table(index=["split", "ds", "actual"], columns="model", values="yhat").reset_index()
    used = [m for m in weights if m in piv.columns]
    p = np.zeros(len(piv), dtype=float)
    wsum = 0.0
    for m in used:
        p += float(weights[m]) * piv[m].to_numpy(dtype=float)
        wsum += float(weights[m])
    p = p / max(wsum, 1e-8)
    return np.abs(piv["actual"].to_numpy(dtype=float) - p)


def run_forecast(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv = rolling_cv(
        df=df,
        season_len=args.season_len,
        cv_splits=args.cv_splits,
        cv_test_months=args.cv_test_months,
        min_train=args.min_train_months,
        lag_rain=args.lag_rain_months,
        lag_cons=args.lag_consumption_months,
    )
    metrics = compute_weights(cv)
    weights = {r["model"]: float(r["weight"]) for _, r in metrics.iterrows()}

    y = df["overall_mean"].to_numpy(dtype=float)
    x = df[["rain_sum_monthly", "consumption_mean_monthly"]].to_numpy(dtype=float)
    h = args.horizon_months

    pred_map = {
        "seasonal_naive": clip_ratio(seasonal_naive(y, horizon=h, season_len=args.season_len)),
        "sarima": clip_ratio(sarima_forecast(y, horizon=h, season_len=args.season_len)),
        "sarimax_exog": clip_ratio(
            sarimax_exog_forecast(
                train_y=y,
                train_x_raw=x,
                horizon=h,
                season_len=args.season_len,
                lag_rain=args.lag_rain_months,
                lag_cons=args.lag_consumption_months,
            )
        ),
    }

    future_ds = pd.date_range(pd.to_datetime(df["ds"]).max() + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    out = pd.DataFrame({"ds": future_ds})
    for m, p in pred_map.items():
        out[f"pred_{m}"] = p

    ens = np.zeros(h, dtype=float)
    wsum = 0.0
    for m, w in weights.items():
        c = f"pred_{m}"
        if c in out.columns:
            ens += float(w) * out[c].to_numpy(dtype=float)
            wsum += float(w)
    ens = ens / max(wsum, 1e-8)
    out["yhat"] = clip_ratio(ens)

    abs_err = cv_ensemble_errors(cv, weights=weights)
    q = float(np.nanquantile(abs_err, min(max(1.0 - args.alpha, 0.5), 0.999)))
    out["yhat_lower"] = clip_ratio(out["yhat"].to_numpy(dtype=float) - q)
    out["yhat_upper"] = clip_ratio(out["yhat"].to_numpy(dtype=float) + q)
    out["interval_q_abs"] = q

    return out, metrics, cv


def save_plot(df: pd.DataFrame, fc: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(11, 5.5))
    plt.plot(df["ds"], df["overall_mean"] * 100.0, color="#1f77b4", linewidth=1.8, label="Gerceklesen")
    plt.plot(fc["ds"], fc["yhat"] * 100.0, color="#d62728", linestyle="--", linewidth=1.8, label="Tahmin (exog)")
    plt.fill_between(
        fc["ds"],
        fc["yhat_lower"] * 100.0,
        fc["yhat_upper"] * 100.0,
        color="#d62728",
        alpha=0.15,
        label="90% aralik",
    )
    plt.title("Istanbul Overall Dam Occupancy Forecast (with exogenous)")
    plt.ylabel("Doluluk (%)")
    plt.xlabel("Tarih")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    occ_records = fetch_records(args.api_url, args.occupancy_resource_id)
    exog_records = fetch_records(args.api_url, args.exog_resource_id)

    occ_m = build_occupancy_monthly(occ_records)
    exog_m = build_exog_monthly(exog_records)

    model_df = occ_m.merge(exog_m, on="ds", how="inner").sort_values("ds").reset_index(drop=True)
    model_df.to_csv(out / "model_input_monthly.csv", index=False)

    fc, metrics, cv = run_forecast(model_df, args=args)
    fc.to_csv(out / "forecast_overall_with_exog.csv", index=False)
    metrics.to_csv(out / "cv_metrics_with_exog.csv", index=False)
    cv.to_csv(out / "cv_predictions_with_exog.csv", index=False)
    save_plot(model_df, fc, out / "forecast_overall_with_exog.png")

    summary = {
        "history_start": str(model_df["ds"].min().date()),
        "history_end": str(model_df["ds"].max().date()),
        "history_rows": int(len(model_df)),
        "forecast_start": str(fc["ds"].min().date()),
        "forecast_end": str(fc["ds"].max().date()),
        "horizon_months": int(args.horizon_months),
        "alpha": float(args.alpha),
        "lag_rain_months": int(args.lag_rain_months),
        "lag_consumption_months": int(args.lag_consumption_months),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved outputs to: {out}")
    print(metrics[["model", "rmse", "smape", "weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
