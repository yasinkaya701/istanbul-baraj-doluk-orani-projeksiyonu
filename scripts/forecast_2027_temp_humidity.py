#!/usr/bin/env python3
"""Forecast monthly temperature/humidity for year 2027."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast 2027 monthly temp/humidity from cleaned time series."
    )
    parser.add_argument(
        "--temp-csv",
        type=Path,
        required=True,
        help="CSV from extract_ods_hourly.py (must include temp_c + timestamp/date/hour)",
    )
    parser.add_argument(
        "--humidity-csv",
        type=Path,
        default=None,
        help="Optional humidity CSV from digitize_humidity_tif.py or merged humidity data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/forecast_2027.csv"),
        help="Output forecast CSV",
    )
    return parser.parse_args()


def _to_timestamp(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns and df["timestamp"].astype(str).str.strip().ne("").any():
        return pd.to_datetime(df["timestamp"], errors="coerce")
    if {"date", "hour"}.issubset(df.columns):
        base = pd.to_datetime(df["date"], errors="coerce")
        return base + pd.to_timedelta(pd.to_numeric(df["hour"], errors="coerce") - 1, unit="h")
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce")
    raise ValueError("No timestamp/date columns found.")


def _monthly_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    ts = _to_timestamp(df)
    s = pd.to_numeric(df[value_col], errors="coerce")
    m = (
        pd.DataFrame({"ts": ts, "value": s})
        .dropna()
        .set_index("ts")["value"]
        .resample("MS")
        .mean()
        .sort_index()
    )
    return m


def _seasonal_naive(series: pd.Series, months: Iterable[pd.Timestamp]) -> pd.DataFrame:
    month_mean = series.groupby(series.index.month).mean()
    values = [float(month_mean.get(m.month, series.mean())) for m in months]
    return pd.DataFrame({"forecast": values, "low": [pd.NA] * len(values), "high": [pd.NA] * len(values)})


def _sarimax_or_fallback(series: pd.Series, months: pd.DatetimeIndex) -> pd.DataFrame:
    series = series.dropna().asfreq("MS")
    if len(series) < 24:
        return _seasonal_naive(series, months)
    try:
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        steps = len(months)
        fc = fit.get_forecast(steps=steps)
        ci = fc.conf_int()
        out = pd.DataFrame(
            {
                "forecast": fc.predicted_mean.values,
                "low": ci.iloc[:, 0].values,
                "high": ci.iloc[:, 1].values,
            },
            index=months,
        )
        return out.reset_index(drop=True)
    except Exception:
        return _seasonal_naive(series, months)


def main() -> None:
    args = parse_args()
    months_2027 = pd.date_range("2027-01-01", "2027-12-01", freq="MS")

    temp_df = pd.read_csv(args.temp_csv)
    temp_monthly = _monthly_series(temp_df, "temp_c")
    temp_fc = _sarimax_or_fallback(temp_monthly, months_2027)

    out = pd.DataFrame({"month": months_2027.strftime("%Y-%m")})
    out["temp_forecast_c"] = temp_fc["forecast"]
    out["temp_low"] = temp_fc["low"]
    out["temp_high"] = temp_fc["high"]

    if args.humidity_csv is not None and args.humidity_csv.exists():
        hum_df = pd.read_csv(args.humidity_csv)
        hum_col = "humidity_pct" if "humidity_pct" in hum_df.columns else None
        if hum_col:
            hum_monthly = _monthly_series(hum_df, hum_col)
            if len(hum_monthly) >= 12:
                hum_fc = _sarimax_or_fallback(hum_monthly, months_2027)
                out["humidity_forecast_pct"] = hum_fc["forecast"]
                out["humidity_low"] = hum_fc["low"]
                out["humidity_high"] = hum_fc["high"]
            else:
                print(
                    "Humidity series has <12 monthly points; skipping humidity forecast."
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote forecast to {args.output}")


if __name__ == "__main__":
    main()
