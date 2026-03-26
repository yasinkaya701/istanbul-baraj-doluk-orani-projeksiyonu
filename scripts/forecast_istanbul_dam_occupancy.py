#!/usr/bin/env python3
"""Forecast Istanbul dam occupancy from IBB open data.

Pipeline:
- Download daily records from CKAN datastore API
- Build monthly mean series for each dam + overall mean
- Compare candidate models via rolling-origin backtests
- Build inverse-RMSE weighted ensemble
- Calibrate uncertainty with split-conformal absolute residuals
- Export forecasts and diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_istanbul_dam_forecast"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


DEFAULT_RESOURCE_ID = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"
DEFAULT_API = "https://data.ibb.gov.tr/api/3/action/datastore_search"

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class CVStats:
    model: str
    mae: float
    rmse: float
    smape: float
    n_points: int
    weight: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Istanbul dam occupancy forecast")
    p.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID)
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--output-dir", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast"))
    p.add_argument("--horizon-months", type=int, default=36, help="Forecast horizon in months")
    p.add_argument("--season-len", type=int, default=12, help="Monthly season length")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-test-months", type=int, default=12)
    p.add_argument("--min-train-months", type=int, default=72)
    p.add_argument("--alpha", type=float, default=0.10, help="Conformal alpha (0.10 -> 90%% interval)")
    p.add_argument("--plot-series", default="overall_mean", help="Series name to visualize")
    return p.parse_args()


def fetch_records(api_url: str, resource_id: str, limit: int = 5000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode({"resource_id": resource_id, "limit": str(limit), "offset": str(offset)})
        url = f"{api_url}?{query}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError(f"API returned success=false at offset={offset}")
        result = payload["result"]
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


def build_daily_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if "Tarih" not in df.columns:
        raise ValueError("Expected 'Tarih' column in API records.")

    dam_cols = [c for c in df.columns if c not in {"_id", "Tarih"}]
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Tarih"], errors="coerce")
    for c in dam_cols:
        out[c] = df[c].map(to_numeric)
        # Dataset contains mixed scales across periods (ratio 0-1 and percent 0-100).
        # Convert percent-like entries to ratio for physical consistency.
        mask_percent_like = out[c] > 1.2
        out.loc[mask_percent_like, c] = out.loc[mask_percent_like, c] / 100.0
        out[c] = out[c].clip(lower=0.0, upper=1.0)
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["overall_mean"] = out[dam_cols].mean(axis=1, skipna=True)
    return out


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = daily.copy()
    monthly["ds"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    value_cols = [c for c in monthly.columns if c not in {"date", "ds"}]
    monthly = monthly.groupby("ds", as_index=False)[value_cols].mean()
    return monthly.sort_values("ds").reset_index(drop=True)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, np.nan, denom)
    v = 2.0 * np.abs(y_pred - y_true) / denom
    return float(np.nanmean(v) * 100.0)


def seasonal_naive_forecast(train: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    if y.size == 0:
        return np.full(horizon, np.nan, dtype=float)
    if y.size < season_len:
        return np.repeat(y[-1], horizon)
    base = y[-season_len:]
    reps = int(math.ceil(horizon / season_len))
    return np.tile(base, reps)[:horizon]


def ets_forecast(train: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    try:
        if y.size >= max(2 * season_len, 24):
            fit = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=season_len,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
        elif y.size >= 8:
            fit = ExponentialSmoothing(y, trend="add", initialization_method="estimated").fit(
                optimized=True,
                use_brute=False,
            )
        else:
            return np.repeat(y[-1], horizon)
        pred = fit.forecast(horizon)
        return np.asarray(pred, dtype=float)
    except Exception:
        return seasonal_naive_forecast(y, horizon, season_len=season_len)


def sarima_forecast(train: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(train, dtype=float)
    if y.size < max(3 * season_len, 36):
        return np.full(horizon, np.nan, dtype=float)

    orders = [(1, 1, 1), (2, 1, 1), (1, 0, 1)]
    sorders = [(1, 0, 1, season_len), (0, 1, 1, season_len), (1, 1, 0, season_len)]

    best_aic = float("inf")
    best_fit = None
    for order in orders:
        for sorder in sorders:
            try:
                fit = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=sorder,
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
    pred = best_fit.get_forecast(steps=horizon).predicted_mean
    return np.asarray(pred, dtype=float)


def _clip_ratio(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def model_forecasts(train: np.ndarray, horizon: int, season_len: int) -> dict[str, np.ndarray]:
    fc = {
        "seasonal_naive": seasonal_naive_forecast(train, horizon=horizon, season_len=season_len),
        "ets": ets_forecast(train, horizon=horizon, season_len=season_len),
        "sarima": sarima_forecast(train, horizon=horizon, season_len=season_len),
    }
    return {k: _clip_ratio(v) for k, v in fc.items()}


def rolling_backtest(
    series: pd.Series,
    season_len: int,
    cv_splits: int,
    cv_test_months: int,
    min_train_months: int,
) -> pd.DataFrame:
    y = series.dropna().astype(float).copy()
    n = len(y)
    rows: list[dict[str, Any]] = []
    for split in range(cv_splits):
        test_end = n - (cv_splits - split - 1) * cv_test_months
        test_start = test_end - cv_test_months
        train_end = test_start
        if train_end < min_train_months:
            continue
        train = y.iloc[:train_end].to_numpy()
        test = y.iloc[test_start:test_end]
        preds = model_forecasts(train, horizon=len(test), season_len=season_len)
        for model_name, pred in preds.items():
            for ds, actual, yhat in zip(test.index, test.to_numpy(), pred, strict=False):
                rows.append(
                    {
                        "split": split + 1,
                        "ds": pd.Timestamp(ds),
                        "model": model_name,
                        "actual": float(actual),
                        "yhat": float(yhat),
                        "abs_err": float(abs(actual - yhat)),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["split", "ds", "model", "actual", "yhat", "abs_err"])
    return pd.DataFrame(rows).sort_values(["split", "model", "ds"]).reset_index(drop=True)


def compute_weights(cv_df: pd.DataFrame) -> tuple[list[CVStats], dict[str, float]]:
    stats: list[CVStats] = []
    if cv_df.empty:
        return stats, {}
    for model_name, g in cv_df.groupby("model"):
        y_true = g["actual"].to_numpy(dtype=float)
        y_pred = g["yhat"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        s = smape(y_true, y_pred)
        stats.append(CVStats(model=model_name, mae=mae, rmse=rmse, smape=s, n_points=len(g), weight=0.0))

    inv = np.array([1.0 / max(s.rmse, 1e-8) for s in stats], dtype=float)
    inv = inv / inv.sum()
    for i, st in enumerate(stats):
        st.weight = float(inv[i])
    weight_map = {s.model: s.weight for s in stats}
    return sorted(stats, key=lambda z: z.rmse), weight_map


def conformal_q(cv_df: pd.DataFrame, weights: dict[str, float], alpha: float) -> float:
    if cv_df.empty or not weights:
        return float("nan")
    pivot = cv_df.pivot_table(index=["split", "ds", "actual"], columns="model", values="yhat").reset_index()
    used_models = [m for m in weights if m in pivot.columns]
    if not used_models:
        return float("nan")
    preds = np.zeros(len(pivot), dtype=float)
    wsum = 0.0
    for m in used_models:
        w = float(weights[m])
        preds += w * pivot[m].to_numpy(dtype=float)
        wsum += w
    if wsum <= 0:
        return float("nan")
    preds /= wsum
    abs_err = np.abs(pivot["actual"].to_numpy(dtype=float) - preds)
    q = np.nanquantile(abs_err, min(max(1.0 - alpha, 0.5), 0.999))
    return float(q)


def forecast_series(
    ds: pd.Series,
    series: pd.Series,
    horizon_months: int,
    season_len: int,
    cv_splits: int,
    cv_test_months: int,
    min_train_months: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = pd.Series(series.to_numpy(dtype=float), index=pd.to_datetime(ds)).dropna().sort_index()
    cv_df = rolling_backtest(
        series=ts,
        season_len=season_len,
        cv_splits=cv_splits,
        cv_test_months=cv_test_months,
        min_train_months=min_train_months,
    )
    cv_stats, weights = compute_weights(cv_df)
    q = conformal_q(cv_df, weights=weights, alpha=alpha)

    final_preds = model_forecasts(ts.to_numpy(dtype=float), horizon=horizon_months, season_len=season_len)
    future_ds = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")
    out = pd.DataFrame({"ds": future_ds})
    for model_name, pred in final_preds.items():
        out[f"pred_{model_name}"] = pred

    used_models = [m for m in weights if f"pred_{m}" in out.columns]
    if used_models:
        ens = np.zeros(len(out), dtype=float)
        wsum = 0.0
        for m in used_models:
            w = float(weights[m])
            ens += w * out[f"pred_{m}"].to_numpy(dtype=float)
            wsum += w
        ens = ens / max(wsum, 1e-8)
    else:
        ens = out.filter(like="pred_").mean(axis=1).to_numpy(dtype=float)

    out["yhat"] = _clip_ratio(ens)
    if np.isnan(q):
        q = float(np.nanstd(cv_df["abs_err"].to_numpy(dtype=float))) if not cv_df.empty else 0.05
    out["yhat_lower"] = _clip_ratio(out["yhat"].to_numpy(dtype=float) - q)
    out["yhat_upper"] = _clip_ratio(out["yhat"].to_numpy(dtype=float) + q)
    out["interval_q_abs"] = q

    stats_df = pd.DataFrame(
        [
            {
                "model": s.model,
                "mae": s.mae,
                "rmse": s.rmse,
                "smape": s.smape,
                "n_points": s.n_points,
                "weight": s.weight,
            }
            for s in cv_stats
        ]
    )
    return out, stats_df, cv_df


def save_plot(monthly: pd.DataFrame, fc: pd.DataFrame, series_name: str, out_png: Path) -> None:
    hist = monthly[["ds", series_name]].dropna().copy()
    hist = hist.rename(columns={series_name: "y"})
    plt.figure(figsize=(11, 5.5))
    plt.plot(hist["ds"], hist["y"] * 100.0, color="#1f77b4", linewidth=1.8, label="Gerceklesen")
    plt.plot(fc["ds"], fc["yhat"] * 100.0, color="#d62728", linestyle="--", linewidth=1.8, label="Tahmin")
    plt.fill_between(
        fc["ds"],
        fc["yhat_lower"] * 100.0,
        fc["yhat_upper"] * 100.0,
        color="#d62728",
        alpha=0.15,
        label="90% aralik",
    )
    plt.title(f"Istanbul Dam Occupancy Forecast ({series_name})")
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

    records = fetch_records(api_url=args.api_url, resource_id=args.resource_id)
    daily = build_daily_frame(records)
    monthly = aggregate_monthly(daily)
    monthly.to_csv(out / "istanbul_dam_monthly_history.csv", index=False)

    series_cols = [c for c in monthly.columns if c != "ds"]
    all_fc = []
    all_stats = []
    all_cv = []

    for col in series_cols:
        fc, stats, cv = forecast_series(
            ds=monthly["ds"],
            series=monthly[col],
            horizon_months=args.horizon_months,
            season_len=args.season_len,
            cv_splits=args.cv_splits,
            cv_test_months=args.cv_test_months,
            min_train_months=args.min_train_months,
            alpha=args.alpha,
        )
        fc["series"] = col
        stats["series"] = col
        cv["series"] = col
        all_fc.append(fc)
        all_stats.append(stats)
        all_cv.append(cv)

    forecast_all = pd.concat(all_fc, ignore_index=True)
    stats_all = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    cv_all = pd.concat(all_cv, ignore_index=True) if all_cv else pd.DataFrame()

    forecast_all.to_csv(out / "istanbul_dam_forecasts_monthly.csv", index=False)
    stats_all.to_csv(out / "istanbul_dam_cv_metrics.csv", index=False)
    cv_all.to_csv(out / "istanbul_dam_cv_predictions.csv", index=False)

    plot_series = args.plot_series if args.plot_series in series_cols else "overall_mean"
    fc_plot = forecast_all[forecast_all["series"] == plot_series].copy()
    save_plot(monthly=monthly, fc=fc_plot, series_name=plot_series, out_png=out / f"forecast_{plot_series}.png")

    summary = {
        "resource_id": args.resource_id,
        "record_count_daily": int(len(daily)),
        "record_count_monthly": int(len(monthly)),
        "monthly_start": str(monthly["ds"].min().date()),
        "monthly_end": str(monthly["ds"].max().date()),
        "forecast_horizon_months": int(args.horizon_months),
        "forecast_start": str(fc_plot["ds"].min().date()) if not fc_plot.empty else None,
        "forecast_end": str(fc_plot["ds"].max().date()) if not fc_plot.empty else None,
        "alpha": float(args.alpha),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top = stats_all[stats_all["series"] == "overall_mean"].sort_values("rmse")
    print(f"Saved outputs to: {out}")
    print("overall_mean model ranking:")
    if not top.empty:
        print(top[["model", "rmse", "smape", "weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
