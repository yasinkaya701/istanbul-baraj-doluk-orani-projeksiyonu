#!/usr/bin/env python3
"""Generate clean, continuous forecast charts up to 2035 and replace old chart set."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


VARIABLES = ["humidity", "temp", "pressure", "precip"]
FREQS = {
    "monthly": {"freq": "MS", "seasonal": 12, "title": "Aylik"},
    "yearly": {"freq": "YS", "seasonal": 5, "title": "Yillik"},
}
UNITS = {"humidity": "%", "temp": "C", "pressure": "unknown", "precip": "mm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean continuous forecast charts to 2035.")
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
        help="Unified observations parquet path",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/charts"),
        help="Charts directory to replace",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/clean_2035_datasets"),
        help="Output datasets directory for clean 2035 series",
    )
    parser.add_argument("--target-year", type=int, default=2035, help="Forecast target end year")
    return parser.parse_args()


def seasonal_trend_forecast(series: pd.Series, horizon: int, seasonal_period: int) -> pd.DataFrame:
    y = np.asarray(series.values, dtype=float)
    n = len(y)
    if n == 0 or horizon <= 0:
        return pd.DataFrame(columns=["yhat", "low", "high"])
    if n == 1:
        pred = np.repeat(y[0], horizon)
        return pd.DataFrame({"yhat": pred, "low": pred, "high": pred})

    # For short histories, avoid unstable long-horizon linear trends.
    if n < max(8, seasonal_period * 2):
        p = max(2, min(seasonal_period, n))
        profile = np.array([np.nanmean(y[i::p]) for i in range(p)], dtype=float)
        if np.isnan(profile).any():
            m = float(np.nanmean(y))
            profile = np.where(np.isnan(profile), m, profile)
        yhat_hist = np.array([profile[i % p] for i in range(n)], dtype=float)
        sigma = float(np.nanstd(y - yhat_hist))
        if not np.isfinite(sigma):
            sigma = 0.0
        yhat = np.array([profile[i % p] for i in range(horizon)], dtype=float)
        low = yhat - 1.96 * sigma
        high = yhat + 1.96 * sigma
        return pd.DataFrame({"yhat": yhat, "low": low, "high": high})

    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    trend = intercept + slope * t

    p = max(2, min(seasonal_period, n // 2 if n >= 4 else 2))
    resid = y - trend
    season = np.zeros(p, dtype=float)
    for i in range(p):
        vals = resid[i::p]
        season[i] = float(np.nanmean(vals)) if len(vals) else 0.0

    fit = trend + season[(t.astype(int) % p)]
    sigma = float(np.nanstd(y - fit))
    if not np.isfinite(sigma):
        sigma = 0.0

    tf = np.arange(n, n + horizon, dtype=float)
    yhat = intercept + slope * tf + season[(tf.astype(int) % p)]
    low = yhat - 1.96 * sigma
    high = yhat + 1.96 * sigma
    return pd.DataFrame({"yhat": yhat, "low": low, "high": high})


def apply_bounds(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    out = df.copy()
    if variable == "humidity":
        for c in ["value", "low", "high"]:
            if c in out:
                out[c] = out[c].clip(0, 100)
    elif variable in {"precip", "pressure"}:
        for c in ["value", "low", "high"]:
            if c in out:
                out[c] = out[c].clip(lower=0)
    return out


def make_clean_series(obs: pd.DataFrame, variable: str, freq_code: str, target_year: int, seasonal: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    sub = obs[(obs["variable"] == variable) & (obs["qc_flag"] == "ok")].copy()
    if sub.empty:
        return pd.DataFrame(), pd.Timestamp("1970-01-01")

    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    raw = sub.groupby("timestamp")["value"].mean()

    if variable == "precip":
        base = raw.resample(freq_code).sum(min_count=1)
    else:
        base = raw.resample(freq_code).mean()
    observed = base.notna()

    base = base.interpolate("time").ffill().bfill()
    if variable == "precip":
        base = base.fillna(0.0)

    if base.empty:
        return pd.DataFrame(), pd.Timestamp("1970-01-01")

    last_obs = observed[observed].index.max()
    target_end = pd.Timestamp(year=target_year, month=12, day=1 if freq_code == "MS" else 31)
    if freq_code == "YS":
        target_end = pd.Timestamp(year=target_year, month=1, day=1)

    if last_obs >= target_end:
        horizon = 0
    else:
        future_idx = pd.date_range(last_obs + to_offset(freq_code), target_end, freq=freq_code)
        horizon = len(future_idx)

    fc = seasonal_trend_forecast(base, horizon=horizon, seasonal_period=seasonal)
    if horizon > 0:
        fidx = pd.date_range(last_obs + to_offset(freq_code), periods=horizon, freq=freq_code)
        fc_df = pd.DataFrame({"timestamp": fidx, "value": fc["yhat"], "low": fc["low"], "high": fc["high"], "is_forecast": True})
    else:
        fc_df = pd.DataFrame(columns=["timestamp", "value", "low", "high", "is_forecast"])

    hist_df = pd.DataFrame(
        {
            "timestamp": base.index,
            "value": base.values,
            "low": np.nan,
            "high": np.nan,
            "is_forecast": False,
        }
    )
    out = pd.concat([hist_df, fc_df], ignore_index=True)
    out["variable"] = variable
    out["frequency"] = freq_code
    out["unit"] = UNITS.get(variable, "unknown")
    out = apply_bounds(out, variable)
    return out, pd.Timestamp(last_obs)


def plot_clean(df: pd.DataFrame, last_obs: pd.Timestamp, variable: str, freq_label: str, out_path: Path) -> None:
    hist = df[df["is_forecast"] == False]  # noqa: E712
    fc = df[df["is_forecast"] == True]  # noqa: E712

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(hist["timestamp"], hist["value"], color="#1f77b4", linewidth=1.4, label="Tarihsel (islenmis)")
    if not fc.empty:
        # Forecast starts exactly after historical for continuous look.
        ax.plot(fc["timestamp"], fc["value"], color="#ff7f0e", linewidth=1.9, label="Tahmin (2035'e kadar)")
        ax.fill_between(fc["timestamp"], fc["low"], fc["high"], color="#ff7f0e", alpha=0.18, label="Guven bandi")

    ax.axvline(last_obs, color="#666666", linestyle="--", linewidth=1)
    ax.text(last_obs, ax.get_ylim()[1], " Son gozlem", va="top", ha="left", fontsize=8, color="#444444")
    ax.set_title(f"{freq_label} Kesintisiz Tahmin - {variable}")
    ax.set_xlabel("Tarih")
    ax.set_ylabel(f"Deger ({UNITS.get(variable, 'unit')})")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    obs = pd.read_parquet(args.observations)

    args.datasets_dir.mkdir(parents=True, exist_ok=True)
    args.charts_dir.mkdir(parents=True, exist_ok=True)

    # Remove old charts and replace with clean set.
    for old in args.charts_dir.glob("*.png"):
        old.unlink()

    index_rows = []
    for var in VARIABLES:
        for key, cfg in FREQS.items():
            series_df, last_obs = make_clean_series(
                obs=obs,
                variable=var,
                freq_code=cfg["freq"],
                target_year=args.target_year,
                seasonal=cfg["seasonal"],
            )
            if series_df.empty:
                continue

            stem = f"{var}_{key}_continuous_to_{args.target_year}"
            csv_p = args.datasets_dir / f"{stem}.csv"
            pq_p = args.datasets_dir / f"{stem}.parquet"
            series_df.to_csv(csv_p, index=False)
            series_df.to_parquet(pq_p, index=False)

            chart_p = args.charts_dir / f"{stem}.png"
            plot_clean(series_df, last_obs=last_obs, variable=var, freq_label=cfg["title"], out_path=chart_p)

            index_rows.append(
                {
                    "variable": var,
                    "resolution": key,
                    "last_observation": str(last_obs),
                    "target_year": args.target_year,
                    "rows_total": len(series_df),
                    "forecast_rows": int((series_df["is_forecast"] == True).sum()),  # noqa: E712
                    "dataset_csv": str(csv_p),
                    "dataset_parquet": str(pq_p),
                    "chart_png": str(chart_p),
                }
            )

    idx = pd.DataFrame(index_rows).sort_values(["variable", "resolution"])
    idx.to_csv(args.datasets_dir / f"clean_charts_index_{args.target_year}.csv", index=False)
    idx.to_parquet(args.datasets_dir / f"clean_charts_index_{args.target_year}.parquet", index=False)

    print("Clean charts generated and old charts replaced.")
    print(f"Charts dir: {args.charts_dir}")
    print(f"Datasets dir: {args.datasets_dir}")
    print(idx[["variable", "resolution", "forecast_rows", "chart_png"]].to_string(index=False))


if __name__ == "__main__":
    main()
