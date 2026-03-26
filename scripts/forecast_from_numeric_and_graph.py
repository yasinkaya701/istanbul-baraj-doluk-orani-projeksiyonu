#!/usr/bin/env python3
"""Forecast climate variables at hourly/daily/monthly/yearly frequencies.

Inputs:
- Numeric observations parquet (from ingest_numeric_and_plot.py)
- Humidity trace CSV files digitized from graph papers

Outputs:
- Unified observations containing both numeric + graph-paper sources
- Per-variable and per-frequency historical datasets
- Per-variable and per-frequency forecasts
- Forecast plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


FREQ_CONFIG = {
    "hourly": {"freq": "1h", "horizon": 24 * 30, "seasonal_period": 24, "title": "Saatlik"},
    "daily": {"freq": "1D", "horizon": 365, "seasonal_period": 7, "title": "Gunluk"},
    "monthly": {"freq": "MS", "horizon": 24, "seasonal_period": 12, "title": "Aylik"},
    "yearly": {"freq": "YS", "horizon": 10, "seasonal_period": 5, "title": "Yillik"},
}

VARIABLES = ["humidity", "temp", "pressure", "precip"]
UNITS = {"humidity": "pct", "temp": "degC", "pressure": "unknown", "precip": "mm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast climate variables with multi-frequency outputs.")
    parser.add_argument(
        "--numeric-parquet",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/sample/observations_numeric.parquet"),
        help="Path to numeric observations parquet",
    )
    parser.add_argument(
        "--graph-csv-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output"),
        help="Directory containing *_humidity_trace.csv graph-paper outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package"),
        help="Output directory",
    )
    return parser.parse_args()


def parse_date_from_filename(path: Path) -> pd.Timestamp | None:
    m = re.search(r"(19|20)\d{2}[_-](\d{2})[_-](\d{2})", path.name)
    if not m:
        return None
    y = int(path.name[m.start() : m.start() + 4])
    mm = int(m.group(2))
    dd = int(m.group(3))
    try:
        return pd.Timestamp(year=y, month=mm, day=dd)
    except ValueError:
        return None


def load_graph_humidity(graph_csv_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for p in sorted(graph_csv_dir.glob("*_humidity_trace.csv")):
        df = pd.read_csv(p)
        ts = pd.to_datetime(df.get("timestamp", pd.Series(dtype=object)), errors="coerce")

        if ts.isna().all():
            base_date = parse_date_from_filename(p)
            elapsed = pd.to_numeric(df.get("elapsed_hour"), errors="coerce")
            if base_date is not None and elapsed.notna().any():
                ts = base_date + pd.to_timedelta(8, unit="h") + pd.to_timedelta(elapsed.fillna(0), unit="h")

        value = pd.to_numeric(df.get("humidity_pct"), errors="coerce")
        out = pd.DataFrame(
            {
                "timestamp": ts,
                "variable": "humidity",
                "value": value,
                "unit": "pct",
                "station_id": "KRDAE_KLIMA",
                "source_kind": "graph_paper",
                "source_file": str(p),
                "method": "trace_digitize",
                "confidence": 0.78,
            }
        )
        rows.append(out)

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "variable",
                "value",
                "unit",
                "station_id",
                "source_kind",
                "source_file",
                "method",
                "confidence",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def load_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
            "variable": df["variable"].astype(str),
            "value": pd.to_numeric(df["value"], errors="coerce"),
            "unit": df["unit"].astype(str),
            "station_id": df.get("station_id", "KRDAE_KLIMA"),
            "source_kind": "numeric",
            "source_file": df["source_file"].astype(str),
            "method": df["method"].astype(str),
            "confidence": pd.to_numeric(df["confidence"], errors="coerce").fillna(0.9),
        }
    )
    return out


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


def build_unified_observations(numeric_parquet: Path, graph_csv_dir: Path) -> pd.DataFrame:
    num = load_numeric(numeric_parquet)
    graph = load_graph_humidity(graph_csv_dir)
    all_df = pd.concat([num, graph], ignore_index=True)
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], errors="coerce")
    all_df["value"] = pd.to_numeric(all_df["value"], errors="coerce")
    all_df = all_df.dropna(subset=["timestamp", "value", "variable"])
    all_df["qc_flag"] = [qc_flag(v, x) for v, x in zip(all_df["variable"], all_df["value"])]
    all_df["is_missing"] = False
    all_df["year"] = all_df["timestamp"].dt.year
    all_df["month"] = all_df["timestamp"].dt.month
    all_df["day"] = all_df["timestamp"].dt.day
    all_df["hour"] = all_df["timestamp"].dt.hour
    return all_df.sort_values(["timestamp", "variable", "source_kind"]).reset_index(drop=True)


def aggregate_series(raw: pd.Series, variable: str, freq_key: str, freq_code: str) -> pd.DataFrame:
    """Create regular historical dataset for one variable and one frequency."""
    raw = raw.sort_index()
    agg_method = "sum" if variable == "precip" and freq_key in {"daily", "monthly", "yearly"} else "mean"

    # Resample to target frequency.
    if agg_method == "sum":
        rs = raw.resample(freq_code).sum(min_count=1)
    else:
        rs = raw.resample(freq_code).mean()

    observed = rs.notna()
    filled = rs.copy()

    # Keep precipitation missing as 0 on model grid; others time-interpolate.
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


def make_forecast_dataset(hist: pd.DataFrame, freq_code: str, horizon: int, seasonal_period: int) -> pd.DataFrame:
    hist = hist.sort_values("timestamp").reset_index(drop=True)
    fc = seasonal_trend_forecast(hist["value"], horizon=horizon, seasonal_period=seasonal_period)
    if fc.empty:
        return pd.DataFrame(columns=["timestamp", "yhat", "low", "high", "is_forecast"])

    last_ts = pd.Timestamp(hist["timestamp"].iloc[-1])
    future_idx = pd.date_range(last_ts + to_offset(freq_code), periods=horizon, freq=freq_code)
    out_fc = pd.DataFrame(
        {
            "timestamp": future_idx,
            "yhat": fc["yhat"].values,
            "low": fc["low"].values,
            "high": fc["high"].values,
            "is_forecast": True,
        }
    )
    hist_out = pd.DataFrame(
        {
            "timestamp": hist["timestamp"],
            "yhat": hist["value"].values,
            "low": np.nan,
            "high": np.nan,
            "is_forecast": False,
        }
    )
    return pd.concat([hist_out, out_fc], ignore_index=True)


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
    fig.savefig(out_path, dpi=145)
    plt.close(fig)


def save_dataframe(df: pd.DataFrame, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    datasets_dir = out_dir / "datasets"
    forecasts_dir = out_dir / "forecasts"
    charts_dir = out_dir / "charts"
    for d in [datasets_dir, forecasts_dir, charts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    obs = build_unified_observations(args.numeric_parquet, args.graph_csv_dir)
    save_dataframe(obs, out_dir / "observations_with_graph")

    # Summary proving numeric + graph-paper inclusion.
    src_summary = (
        obs.groupby(["variable", "source_kind"])
        .agg(rows=("value", "size"), min_ts=("timestamp", "min"), max_ts=("timestamp", "max"))
        .reset_index()
    )
    src_summary.to_csv(out_dir / "source_summary.csv", index=False)

    forecasts_index_rows: list[dict[str, object]] = []

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

            fc_all = make_forecast_dataset(
                hist=hist[["timestamp", "value"]],
                freq_code=cfg["freq"],
                horizon=cfg["horizon"],
                seasonal_period=cfg["seasonal_period"],
            )
            fc_all = apply_physical_bounds(fc_all, variable)
            fc_all["variable"] = variable
            fc_all["frequency"] = freq_key
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
    index_df.to_csv(out_dir / "forecast_index.csv", index=False)
    index_df.to_parquet(out_dir / "forecast_index.parquet", index=False)

    print("Wrote unified observations:")
    print(f"- {out_dir / 'observations_with_graph.parquet'}")
    print(f"- {out_dir / 'observations_with_graph.csv'}")
    print("\nSource summary:")
    print(src_summary.to_string(index=False))
    print("\nForecast index:")
    print(index_df[["variable", "frequency", "historical_rows", "forecast_rows", "historical_end", "forecast_end"]].to_string(index=False))


if __name__ == "__main__":
    main()
