#!/usr/bin/env python3
"""Advanced overall Istanbul dam occupancy forecast.

Features:
- Uses IBB open data (occupancy + rainfall/consumption)
- Leakage-safe rolling CV
- Candidate models: seasonal naive, ETS, SARIMA, SARIMAX(exog with auto lags)
- Automatic choose best single vs weighted ensemble
- Month-adaptive conformal intervals
- Threshold risk probabilities (%40 and %30)
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=ConvergenceWarning)

API_URL = "https://data.ibb.gov.tr/api/3/action/datastore_search"
OCC_RESOURCE = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"
EXOG_RESOURCE = "762b802e-c5f9-4175-a5c1-78b892d9764b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced forecast for Istanbul overall dam occupancy")
    p.add_argument("--api-url", default=API_URL)
    p.add_argument("--occupancy-resource-id", default=OCC_RESOURCE)
    p.add_argument("--exog-resource-id", default=EXOG_RESOURCE)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_advanced"),
    )
    p.add_argument("--horizon-months", type=int, default=60)
    p.add_argument("--season-len", type=int, default=12)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-test-months", type=int, default=12)
    p.add_argument("--min-train-months", type=int, default=72)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--max-exog-lag", type=int, default=6)
    p.add_argument("--ensemble-max-models", type=int, default=3)
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
    monthly = out.groupby("ds", as_index=False)["overall_mean"].mean().sort_values("ds").reset_index(drop=True)
    return monthly


def build_exog_monthly(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Tarih"], errors="coerce")
    cons_col = "Istanbul gunluk tuketim(m3/gun)"
    if cons_col not in df.columns:
        raise ValueError("Consumption column not found in exogenous resource.")
    rain_cols = [c for c in df.columns if c not in {"_id", "Tarih", cons_col}]
    for c in rain_cols:
        out[c] = df[c].map(to_numeric)
    out["consumption_m3_day"] = df[cons_col].map(to_numeric)
    out["rain_mean_daily"] = out[rain_cols].mean(axis=1, skipna=True)
    out = out.dropna(subset=["date"]).sort_values("date")
    out["ds"] = out["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        out.groupby("ds", as_index=False)
        .agg(
            rain_sum_monthly=("rain_mean_daily", "sum"),
            consumption_mean_monthly=("consumption_m3_day", "mean"),
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return monthly


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true) + np.abs(y_pred)
    den = np.where(den == 0.0, np.nan, den)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / den) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def clip_ratio(v: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=float), 0.0, 1.0)


def seasonal_naive(values: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.full(horizon, np.nan, dtype=float)
    if len(v) < season_len:
        return np.repeat(v[-1], horizon)
    base = v[-season_len:]
    rep = int(math.ceil(horizon / season_len))
    return np.tile(base, rep)[:horizon]


def ets_forecast(values: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    try:
        if len(v) >= max(2 * season_len, 24):
            fit = ExponentialSmoothing(
                v,
                trend="add",
                seasonal="add",
                seasonal_periods=season_len,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
        elif len(v) >= 8:
            fit = ExponentialSmoothing(v, trend="add", initialization_method="estimated").fit(
                optimized=True,
                use_brute=False,
            )
        else:
            return np.repeat(v[-1], horizon)
        return np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:
        return seasonal_naive(v, horizon=horizon, season_len=season_len)


def sarima_forecast(values: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) < max(3 * season_len, 36):
        return np.full(horizon, np.nan, dtype=float)
    orders = [(1, 1, 1), (2, 1, 1), (1, 0, 1)]
    sorders = [(1, 0, 1, season_len), (0, 1, 1, season_len), (1, 1, 0, season_len)]
    best_aic = float("inf")
    best_fit = None
    for o in orders:
        for so in sorders:
            try:
                fit = SARIMAX(
                    v,
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


def choose_best_lag(y: np.ndarray, x: np.ndarray, max_lag: int) -> int:
    s_y = pd.Series(np.asarray(y, dtype=float))
    s_x = pd.Series(np.asarray(x, dtype=float))
    best_lag = 0
    best_abs = -1.0
    for lag in range(0, max_lag + 1):
        corr = s_y.corr(s_x.shift(lag))
        if pd.isna(corr):
            continue
        val = abs(float(corr))
        if val > best_abs:
            best_abs = val
            best_lag = lag
    return int(best_lag)


def exog_future_naive(train_x_raw: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    cols = []
    for j in range(train_x_raw.shape[1]):
        cols.append(seasonal_naive(train_x_raw[:, j], horizon=horizon, season_len=season_len))
    return np.column_stack(cols)


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


def build_lagged_exog_future(
    train_x_raw: np.ndarray,
    horizon: int,
    season_len: int,
    lag_rain: int,
    lag_cons: int,
) -> np.ndarray:
    raw_future = exog_future_naive(train_x_raw, horizon=horizon, season_len=season_len)
    rain_full = np.concatenate([train_x_raw[:, 0], raw_future[:, 0]])
    cons_full = np.concatenate([train_x_raw[:, 1], raw_future[:, 1]])
    n = len(train_x_raw)
    rain_slice = rain_full[n - lag_rain : n - lag_rain + horizon]
    cons_slice = cons_full[n - lag_cons : n - lag_cons + horizon]
    return np.column_stack([rain_slice, cons_slice])


def sarimax_forecast_auto_lag(
    train_y: np.ndarray,
    train_x_raw: np.ndarray,
    horizon: int,
    season_len: int,
    max_exog_lag: int,
) -> tuple[np.ndarray, dict[str, int]]:
    lag_rain = choose_best_lag(train_y, train_x_raw[:, 0], max_lag=max_exog_lag)
    lag_cons = choose_best_lag(train_y, train_x_raw[:, 1], max_lag=max_exog_lag)
    max_lag = max(lag_rain, lag_cons)
    if len(train_y) < max(3 * season_len, 36, max_lag + 24):
        return np.full(horizon, np.nan, dtype=float), {"lag_rain": lag_rain, "lag_cons": lag_cons}

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
        return np.full(horizon, np.nan, dtype=float), {"lag_rain": lag_rain, "lag_cons": lag_cons}
    pred = np.asarray(best_fit.get_forecast(steps=horizon, exog=x_future).predicted_mean, dtype=float)
    return pred, {"lag_rain": lag_rain, "lag_cons": lag_cons}


def candidate_predictions(
    train_y: np.ndarray,
    train_x_raw: np.ndarray,
    horizon: int,
    season_len: int,
    max_exog_lag: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    pred_sarimax, lags = sarimax_forecast_auto_lag(
        train_y=train_y,
        train_x_raw=train_x_raw,
        horizon=horizon,
        season_len=season_len,
        max_exog_lag=max_exog_lag,
    )
    preds = {
        "seasonal_naive": clip_ratio(seasonal_naive(train_y, horizon=horizon, season_len=season_len)),
        "ets": clip_ratio(ets_forecast(train_y, horizon=horizon, season_len=season_len)),
        "sarima": clip_ratio(sarima_forecast(train_y, horizon=horizon, season_len=season_len)),
        "sarimax_exog": clip_ratio(pred_sarimax),
    }
    return preds, lags


def rolling_cv(
    ds: pd.Series,
    y: np.ndarray,
    x: np.ndarray,
    season_len: int,
    cv_splits: int,
    cv_test_months: int,
    min_train_months: int,
    max_exog_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(y)
    rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    for split in range(cv_splits):
        test_end = n - (cv_splits - split - 1) * cv_test_months
        test_start = test_end - cv_test_months
        train_end = test_start
        if train_end < min_train_months:
            continue
        y_train = y[:train_end]
        x_train = x[:train_end]
        y_test = y[test_start:test_end]
        h = len(y_test)

        preds, lags = candidate_predictions(
            train_y=y_train,
            train_x_raw=x_train,
            horizon=h,
            season_len=season_len,
            max_exog_lag=max_exog_lag,
        )
        lag_rows.append({"split": split + 1, "lag_rain": lags["lag_rain"], "lag_cons": lags["lag_cons"]})

        for model_name, pred in preds.items():
            for i in range(h):
                rows.append(
                    {
                        "split": split + 1,
                        "ds": pd.Timestamp(ds.iloc[test_start + i]),
                        "model": model_name,
                        "actual": float(y_test[i]),
                        "yhat": float(pred[i]),
                        "abs_err": float(abs(y_test[i] - pred[i])),
                    }
                )
    cv_df = pd.DataFrame(rows).sort_values(["split", "model", "ds"]).reset_index(drop=True)
    lag_df = pd.DataFrame(lag_rows)
    return cv_df, lag_df


def model_metrics(cv_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, g in cv_df.groupby("model"):
        y_true = g["actual"].to_numpy(dtype=float)
        y_pred = g["yhat"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model_name,
                "rmse": rmse(y_true, y_pred),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "smape": smape(y_true, y_pred),
                "n_points": len(g),
            }
        )
    out = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    inv = 1.0 / np.maximum(out["rmse"].to_numpy(dtype=float), 1e-8)
    out["weight"] = inv / inv.sum()
    return out


def add_ensemble_oof(cv_df: pd.DataFrame, metrics_df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, dict[str, float]]:
    selected = metrics_df.head(max(1, top_k)).copy()
    weights = {r["model"]: float(r["weight"]) for _, r in selected.iterrows()}
    piv = cv_df.pivot_table(index=["split", "ds", "actual"], columns="model", values="yhat").reset_index()
    usable = [m for m in weights if m in piv.columns]
    pred = np.zeros(len(piv), dtype=float)
    wsum = 0.0
    for m in usable:
        w = float(weights[m])
        pred += w * piv[m].to_numpy(dtype=float)
        wsum += w
    pred = pred / max(wsum, 1e-8)
    out = piv[["split", "ds", "actual"]].copy()
    out["pred_ensemble"] = clip_ratio(pred)
    out["abs_err_ensemble"] = np.abs(out["actual"].to_numpy(dtype=float) - out["pred_ensemble"].to_numpy(dtype=float))
    return out, weights


def select_strategy(metrics_df: pd.DataFrame, ens_oof: pd.DataFrame) -> tuple[str, float]:
    best_single = float(metrics_df["rmse"].min())
    ens_rmse = rmse(ens_oof["actual"].to_numpy(dtype=float), ens_oof["pred_ensemble"].to_numpy(dtype=float))
    # Prefer ensemble only if not worse than 2% tolerance.
    if ens_rmse <= best_single * 1.02:
        return "ensemble", ens_rmse
    return str(metrics_df.iloc[0]["model"]), best_single


def monthwise_quantiles(errors_df: pd.DataFrame, alpha: float) -> tuple[dict[int, float], float]:
    global_q = float(np.nanquantile(errors_df["abs_err"].to_numpy(dtype=float), min(max(1.0 - alpha, 0.5), 0.999)))
    q_map: dict[int, float] = {}
    for month, g in errors_df.groupby("month"):
        arr = g["abs_err"].dropna().to_numpy(dtype=float)
        if len(arr) >= 5:
            q_map[int(month)] = float(np.nanquantile(arr, min(max(1.0 - alpha, 0.5), 0.999)))
        else:
            q_map[int(month)] = global_q
    return q_map, global_q


def monthwise_residuals(errors_df: pd.DataFrame) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for month, g in errors_df.groupby("month"):
        out[int(month)] = g["residual"].dropna().to_numpy(dtype=float)
    return out


def run_advanced(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    occ = build_occupancy_monthly(fetch_records(args.api_url, args.occupancy_resource_id))
    exog = build_exog_monthly(fetch_records(args.api_url, args.exog_resource_id))
    df = occ.merge(exog, on="ds", how="inner").dropna().sort_values("ds").reset_index(drop=True)
    df.to_csv(out / "model_input_monthly.csv", index=False)

    ds = pd.to_datetime(df["ds"])
    y = df["overall_mean"].to_numpy(dtype=float)
    x = df[["rain_sum_monthly", "consumption_mean_monthly"]].to_numpy(dtype=float)

    cv_df, lag_df = rolling_cv(
        ds=ds,
        y=y,
        x=x,
        season_len=args.season_len,
        cv_splits=args.cv_splits,
        cv_test_months=args.cv_test_months,
        min_train_months=args.min_train_months,
        max_exog_lag=args.max_exog_lag,
    )
    cv_df.to_csv(out / "cv_predictions_models.csv", index=False)
    lag_df.to_csv(out / "cv_selected_lags.csv", index=False)

    metrics = model_metrics(cv_df)
    ens_oof, ens_weights = add_ensemble_oof(cv_df, metrics_df=metrics, top_k=args.ensemble_max_models)
    strategy, strategy_rmse = select_strategy(metrics_df=metrics, ens_oof=ens_oof)

    metrics_out = metrics.copy()
    metrics_out.loc[len(metrics_out)] = {
        "model": "ensemble_topk",
        "rmse": rmse(ens_oof["actual"].to_numpy(dtype=float), ens_oof["pred_ensemble"].to_numpy(dtype=float)),
        "mae": float(np.mean(np.abs(ens_oof["actual"].to_numpy(dtype=float) - ens_oof["pred_ensemble"].to_numpy(dtype=float)))),
        "smape": smape(ens_oof["actual"].to_numpy(dtype=float), ens_oof["pred_ensemble"].to_numpy(dtype=float)),
        "n_points": len(ens_oof),
        "weight": np.nan,
    }
    metrics_out.to_csv(out / "cv_metrics.csv", index=False)

    h = args.horizon_months
    final_preds, final_lags = candidate_predictions(
        train_y=y,
        train_x_raw=x,
        horizon=h,
        season_len=args.season_len,
        max_exog_lag=args.max_exog_lag,
    )
    future_ds = pd.date_range(ds.max() + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fc = pd.DataFrame({"ds": future_ds})
    for name, pred in final_preds.items():
        fc[f"pred_{name}"] = pred

    if strategy == "ensemble":
        pred = np.zeros(len(fc), dtype=float)
        wsum = 0.0
        for m, w in ens_weights.items():
            c = f"pred_{m}"
            if c in fc.columns:
                pred += float(w) * fc[c].to_numpy(dtype=float)
                wsum += float(w)
        fc["yhat"] = clip_ratio(pred / max(wsum, 1e-8))
        err_base = ens_oof[["ds", "actual", "pred_ensemble"]].copy()
        err_base["residual"] = err_base["actual"] - err_base["pred_ensemble"]
        err_base["abs_err"] = np.abs(err_base["residual"])
    else:
        fc["yhat"] = clip_ratio(fc[f"pred_{strategy}"].to_numpy(dtype=float))
        err_base = cv_df[cv_df["model"] == strategy][["ds", "actual", "yhat"]].copy()
        err_base["residual"] = err_base["actual"] - err_base["yhat"]
        err_base["abs_err"] = np.abs(err_base["residual"])

    err_base["month"] = pd.to_datetime(err_base["ds"]).dt.month
    q_map, global_q = monthwise_quantiles(err_base, alpha=args.alpha)
    resid_map = monthwise_residuals(err_base)
    global_resid = err_base["residual"].dropna().to_numpy(dtype=float)

    q_values = []
    probs_40 = []
    probs_30 = []
    for _, r in fc.iterrows():
        month = int(pd.Timestamp(r["ds"]).month)
        q = q_map.get(month, global_q)
        yhat = float(r["yhat"])
        q_values.append(q)

        samples = resid_map.get(month)
        if samples is None or len(samples) < 5:
            samples = global_resid
        if samples is None or len(samples) == 0:
            probs_40.append(float(yhat < 0.40))
            probs_30.append(float(yhat < 0.30))
        else:
            probs_40.append(float(np.mean((yhat + samples) < 0.40)))
            probs_30.append(float(np.mean((yhat + samples) < 0.30)))

    fc["interval_q_abs"] = q_values
    fc["yhat_lower"] = clip_ratio(fc["yhat"].to_numpy(dtype=float) - fc["interval_q_abs"].to_numpy(dtype=float))
    fc["yhat_upper"] = clip_ratio(fc["yhat"].to_numpy(dtype=float) + fc["interval_q_abs"].to_numpy(dtype=float))
    fc["prob_below_40"] = probs_40
    fc["prob_below_30"] = probs_30
    fc.to_csv(out / "forecast_overall_advanced.csv", index=False)

    plt.figure(figsize=(11, 5.6))
    plt.plot(df["ds"], df["overall_mean"] * 100.0, color="#1f77b4", linewidth=1.8, label="Gerceklesen")
    plt.plot(fc["ds"], fc["yhat"] * 100.0, color="#d62728", linestyle="--", linewidth=1.8, label=f"Tahmin ({strategy})")
    plt.fill_between(fc["ds"], fc["yhat_lower"] * 100.0, fc["yhat_upper"] * 100.0, color="#d62728", alpha=0.15, label="90% aralik")
    plt.ylabel("Doluluk (%)")
    plt.xlabel("Tarih")
    plt.title("Istanbul Overall Dam Occupancy - Advanced Forecast")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "forecast_overall_advanced.png", dpi=160)
    plt.close()

    next12_start = pd.Timestamp("2026-03-01")
    next12_end = pd.Timestamp("2027-02-01")
    n12 = fc[(fc["ds"] >= next12_start) & (fc["ds"] <= next12_end)].copy()

    summary = {
        "history_start": str(df["ds"].min().date()),
        "history_end": str(df["ds"].max().date()),
        "history_rows": int(len(df)),
        "forecast_start": str(fc["ds"].min().date()),
        "forecast_end": str(fc["ds"].max().date()),
        "horizon_months": int(args.horizon_months),
        "strategy": strategy,
        "strategy_rmse": float(strategy_rmse),
        "selected_lag_rain_final": int(final_lags["lag_rain"]),
        "selected_lag_cons_final": int(final_lags["lag_cons"]),
        "mean_2026_03_to_2027_02_pct": float(n12["yhat"].mean() * 100.0) if not n12.empty else None,
        "min_2026_03_to_2027_02_pct": float(n12["yhat"].min() * 100.0) if not n12.empty else None,
        "max_2026_03_to_2027_02_pct": float(n12["yhat"].max() * 100.0) if not n12.empty else None,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Istanbul Overall Dam Forecast - Gelistirilmis Ozet")
    lines.append("")
    lines.append(f"- Kullanilan strateji: `{strategy}`")
    lines.append(f"- CV RMSE: `{strategy_rmse:.4f}`")
    lines.append(f"- Son secilen exog laglar: yagis `{final_lags['lag_rain']}` ay, tuketim `{final_lags['lag_cons']}` ay")
    lines.append(f"- Veri donemi: `{df['ds'].min().date()}` - `{df['ds'].max().date()}`")
    lines.append("")
    lines.append("## 2026-03 to 2027-02 Tahmin Ozeti")
    if not n12.empty:
        lines.append(
            f"- Ortalama: `{n12['yhat'].mean()*100:.1f}%` | Min: `{n12['yhat'].min()*100:.1f}%` | Max: `{n12['yhat'].max()*100:.1f}%`"
        )
        lines.append(f"- `%40 alti olasilik` ortalamasi: `{n12['prob_below_40'].mean()*100:.1f}%`")
        lines.append(f"- `%30 alti olasilik` ortalamasi: `{n12['prob_below_30'].mean()*100:.1f}%`")
    lines.append("")
    lines.append("## Not")
    lines.append("- Son gozlem 2024-02 oldugu icin 2026 sonrasi degerler model projeksiyonudur.")
    (out / "HALK_DILI_OZET.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved outputs to: {out}")
    print(metrics_out[["model", "rmse", "smape", "weight"]].to_string(index=False))
    print(f"Selected strategy: {strategy} (rmse={strategy_rmse:.4f})")
    print(f"Final lags -> rain: {final_lags['lag_rain']}, consumption: {final_lags['lag_cons']}")


def main() -> None:
    args = parse_args()
    run_advanced(args)


if __name__ == "__main__":
    main()

