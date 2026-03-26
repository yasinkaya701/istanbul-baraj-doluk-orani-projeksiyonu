#!/usr/bin/env python3
"""Cursor long-term forecast: build historical series and forecasts from unified observations.

Bu script:
- output/cursor_forecast_package/cursor_observations_with_graph.parquet dosyasını okur,
- Saatlik/günlük/aylık/yıllık tarihsel seriler üretir,
- Yaklaşık 10 yıllık ufukla (özellikle aylık/yıllık) basit trend+sezonsallık tahminleri üretir,
- Sonuçları output/cursor_forecast_package/ altına yazar.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/cursor_mpl_config").absolute()))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


FREQ_CONFIG = {
    "hourly": {"freq": "1h", "horizon": 24 * 30, "seasonal_period": 24, "title": "Saatlik"},
    "daily": {"freq": "1D", "horizon": 365, "seasonal_period": 7, "title": "Gunluk"},
    "monthly": {"freq": "MS", "horizon": 120, "seasonal_period": 12, "title": "Aylik"},  # 10 yıl
    "yearly": {"freq": "YS", "horizon": 10, "seasonal_period": 5, "title": "Yillik"},  # 10 yıl
}

VARIABLES = ["humidity", "temp", "pressure", "precip"]
UNITS = {"humidity": "pct", "temp": "degC", "pressure": "unknown", "precip": "mm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cursor: build multi-frequency historical datasets and long-term forecasts."
    )
    parser.add_argument(
        "--unified-parquet",
        type=Path,
        default=Path("output/cursor_forecast_package/cursor_observations_with_graph.parquet"),
        help="Unified observations parquet from cursor_unified_observations.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cursor_forecast_package"),
        help="Root output directory for datasets, forecasts and charts.",
    )
    parser.add_argument(
        "--monthly-horizon",
        type=int,
        default=120,
        help="Forecast horizon (steps) for monthly frequency (default: 120=10 years).",
    )
    parser.add_argument(
        "--yearly-horizon",
        type=int,
        default=10,
        help="Forecast horizon (steps) for yearly frequency (default: 10 years).",
    )
    return parser.parse_args()


def qc_flag(variable: str, value: float) -> str:
    if pd.isna(value):
        return "missing"
    if variable == "humidity" and not (0 <= value <= 100):
        return "range_fail"
    if variable == "temp" and not (-60 <= value <= 70):
        return "range_fail"
    if variable == "pressure" and not (0 <= value <= 2000):
        return "range_fail"
    if variable == "precip" and value < 0:
        return "range_fail"
    return "ok"


def aggregate_series(raw: pd.Series, variable: str, freq_key: str, freq_code: str) -> pd.DataFrame:
    """Create regular historical dataset for one variable and one frequency."""
    raw = raw.sort_index()
    agg_method = "sum" if variable == "precip" and freq_key in {"daily", "monthly", "yearly"} else "mean"

    if agg_method == "sum":
        rs = raw.resample(freq_code).sum(min_count=1)
    else:
        rs = raw.resample(freq_code).mean()

    observed = rs.notna()
    filled = rs.copy()

    if variable == "precip":
        filled = filled.fillna(0.0)
    else:
        filled = filled.interpolate("time").ffill().bfill()

    if filled.isna().all():
        filled = pd.Series(np.zeros(len(filled)), index=filled.index)

    out = pd.DataFrame(
        {
            "timestamp": filled.index,
            "value": filled.values.astype(float),
            "observed": observed.values,
            "observed_ratio": float(observed.mean()) if len(observed) else 0.0,
        }
    )
    out["variable"] = variable
    out["frequency"] = freq_key
    out["unit"] = UNITS.get(variable, "unknown")
    return out


def seasonal_trend_forecast(series: pd.Series, horizon: int, seasonal_period: int) -> pd.DataFrame:
    y = np.asarray(series.values, dtype=float)
    n = len(y)
    if n == 0:
        return pd.DataFrame(columns=["step", "yhat", "low", "high"])
    if n == 1:
        pred = np.repeat(y[0], horizon)
        return pd.DataFrame({"step": np.arange(1, horizon + 1), "yhat": pred, "low": pred, "high": pred})

    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    trend_fit = intercept + slope * t

    p = max(2, min(seasonal_period, n // 2 if n >= 4 else 2))
    resid = y - trend_fit
    season = np.zeros(p, dtype=float)
    for i in range(p):
        vals = resid[i::p]
        season[i] = float(np.nanmean(vals)) if len(vals) else 0.0

    fit = trend_fit + season[(t.astype(int) % p)]
    err = y - fit
    sigma = float(np.nanstd(err))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(y - trend_fit))
    if not np.isfinite(sigma):
        sigma = 0.0

    future_t = np.arange(n, n + horizon, dtype=float)
    yhat = intercept + slope * future_t + season[(future_t.astype(int) % p)]
    low = yhat - 1.96 * sigma
    high = yhat + 1.96 * sigma
    return pd.DataFrame(
        {"step": np.arange(1, horizon + 1), "yhat": yhat.astype(float), "low": low.astype(float), "high": high.astype(float)}
    )


def apply_physical_bounds(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    out = df.copy()
    if variable == "humidity":
        for c in ["yhat", "low", "high"]:
            out[c] = out[c].clip(lower=0, upper=100)
    elif variable == "precip":
        for c in ["yhat", "low", "high"]:
            out[c] = out[c].clip(lower=0)
    elif variable == "pressure":
        for c in ["yhat", "low", "high"]:
            out[c] = out[c].clip(lower=0)
    return out


def plot_forecast(series_df: pd.DataFrame, variable: str, freq_key: str, out_path: Path) -> None:
    hist = series_df[series_df["is_forecast"] == False]  # noqa: E712
    fc = series_df[series_df["is_forecast"] == True]  # noqa: E712

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist["timestamp"], hist["yhat"], label="Gercek/Islenmis", linewidth=1.2)
    if not fc.empty:
        ax.plot(fc["timestamp"], fc["yhat"], label="Tahmin", linewidth=1.6, color="tab:orange")
        if fc["low"].notna().any() and fc["high"].notna().any():
            ax.fill_between(fc["timestamp"], fc["low"], fc["high"], alpha=0.2, color="tab:orange", label="Guven bandi")
    ax.set_title(f"{FREQ_CONFIG[freq_key]['title']} Tahmin - {variable}")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Deger")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=145)
    plt.close(fig)


def save_dataframe(df: pd.DataFrame, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)


def main() -> None:
    args = parse_args()
    out_root = args.output_dir
    datasets_dir = out_root / "datasets"
    forecasts_dir = out_root / "forecasts"
    charts_dir = out_root / "charts"
    for d in [datasets_dir, forecasts_dir, charts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not args.unified_parquet.exists():
        raise SystemExit(f"Unified observations parquet not found: {args.unified_parquet}")

    obs = pd.read_parquet(args.unified_parquet)
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs["variable"] = obs["variable"].astype(str)
    obs = obs.dropna(subset=["timestamp", "value", "variable"])

    # Ensure qc_flag exists; if not, compute quickly.
    if "qc_flag" not in obs.columns:
        obs["qc_flag"] = [qc_flag(v, x) for v, x in zip(obs["variable"], obs["value"])]

    forecasts_index_rows: list[dict[str, object]] = []

    # Override horizons from args
    FREQ_CONFIG["monthly"]["horizon"] = int(args.monthly_horizon)
    FREQ_CONFIG["yearly"]["horizon"] = int(args.yearly_horizon)

    for variable in VARIABLES:
        sub = obs[(obs["variable"] == variable) & (obs["qc_flag"] == "ok")].copy()
        if sub.empty:
            continue
        raw_series = sub.groupby("timestamp")["value"].mean().sort_index()

        for freq_key, cfg in FREQ_CONFIG.items():
            hist = aggregate_series(raw_series, variable=variable, freq_key=freq_key, freq_code=cfg["freq"])
            if hist.empty:
                continue

            hist_base = datasets_dir / f"{variable}_{freq_key}_historical"
            save_dataframe(hist, hist_base)

            fc_core = seasonal_trend_forecast(
                hist["value"],
                horizon=cfg["horizon"],
                seasonal_period=cfg["seasonal_period"],
            )
            if fc_core.empty:
                continue

            last_ts = pd.Timestamp(hist["timestamp"].iloc[-1])
            future_idx = pd.date_range(last_ts + to_offset(cfg["freq"]), periods=cfg["horizon"], freq=cfg["freq"])
            fc = pd.DataFrame(
                {
                    "timestamp": future_idx,
                    "yhat": fc_core["yhat"].values,
                    "low": fc_core["low"].values,
                    "high": fc_core["high"].values,
                    "is_forecast": True,
                }
            )
            hist_out = pd.DataFrame(
                {
                    "timestamp": hist["timestamp"],
                    "yhat": hist["value"],
                    "low": np.nan,
                    "high": np.nan,
                    "is_forecast": False,
                }
            )
            fc_all = pd.concat([hist_out, fc], ignore_index=True)
            fc_all = apply_physical_bounds(fc_all, variable)
            fc_all["variable"] = variable
            fc_all["frequency"] = freq_key
            fc_all["unit"] = UNITS.get(variable, "unknown")

            fc_base = forecasts_dir / f"{variable}_{freq_key}_forecast"
            save_dataframe(fc_all, fc_base)

            chart_path = charts_dir / f"{variable}_{freq_key}_forecast.png"
            plot_forecast(fc_all, variable=variable, freq_key=freq_key, out_path=chart_path)

            forecasts_index_rows.append(
                {
                    "variable": variable,
                    "frequency": freq_key,
                    "historical_rows": len(hist),
                    "observed_ratio": float(hist["observed_ratio"].iloc[0]) if len(hist) else np.nan,
                    "forecast_rows": int((fc_all["is_forecast"] == True).sum()),  # noqa: E712
                    "historical_start": hist["timestamp"].min(),
                    "historical_end": hist["timestamp"].max(),
                    "forecast_end": fc_all["timestamp"].max(),
                    "historical_file_csv": str(hist_base.with_suffix(".csv")),
                    "forecast_file_csv": str(fc_base.with_suffix(".csv")),
                    "chart_file": str(chart_path),
                }
            )

    index_df = pd.DataFrame(forecasts_index_rows).sort_values(["variable", "frequency"])
    index_csv = out_root / "cursor_forecast_index.csv"
    index_parquet = out_root / "cursor_forecast_index.parquet"
    index_df.to_csv(index_csv, index=False)
    index_df.to_parquet(index_parquet, index=False)

    print("Forecast index:")
    cols = ["variable", "frequency", "historical_rows", "forecast_rows", "historical_end", "forecast_end"]
    print(index_df[cols].to_string(index=False))
    print(f"\nIndex files:\n- {index_csv}\n- {index_parquet}")


if __name__ == "__main__":
    main()

