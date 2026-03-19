#!/usr/bin/env python3
"""15-year projection for Istanbul dam occupancy (Elmali + overall mean).

Uses IBB CKAN datastore API, aggregates to monthly means, then fits ETS models.
Outputs long-format forecast CSV plus a small summary JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning


DEFAULT_RESOURCE_ID = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"
DEFAULT_API = "https://data.ibb.gov.tr/api/3/action/datastore_search"

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="15-year projection for Istanbul dam occupancy")
    p.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID)
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--output-dir", type=Path, default=Path("output/istanbul_dam_forecast_15y"))
    p.add_argument("--horizon-months", type=int, default=180)
    p.add_argument("--season-len", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0.10, help="Two-sided interval alpha (0.10 -> 90%%)")
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


_TRANSLIT = str.maketrans(
    {
        "\u0131": "i",
        "\u0130": "i",
        "\u015f": "s",
        "\u015e": "s",
        "\u011f": "g",
        "\u011e": "g",
        "\u00fc": "u",
        "\u00dc": "u",
        "\u00f6": "o",
        "\u00d6": "o",
        "\u00e7": "c",
        "\u00c7": "c",
    }
)


def normalize_name(value: str) -> str:
    s = value.strip().lower().translate(_TRANSLIT)
    return "".join(ch for ch in s if ch.isalnum())


def pick_column(columns: Iterable[str], target: str) -> str | None:
    target_norm = normalize_name(target)
    for col in columns:
        if normalize_name(col) == target_norm:
            return col
    return None


def build_daily_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if "Tarih" not in df.columns:
        raise ValueError("Expected 'Tarih' column in API records.")

    dam_cols = [c for c in df.columns if c not in {"_id", "Tarih"}]
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Tarih"], errors="coerce")
    for c in dam_cols:
        out[c] = df[c].map(to_numeric)
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


def seasonal_naive(values: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if y.size == 0:
        return np.full(horizon, np.nan, dtype=float)
    if y.size < season_len:
        return np.repeat(y[-1], horizon)
    base = y[-season_len:]
    reps = int(math.ceil(horizon / season_len))
    return np.tile(base, reps)[:horizon]


def fit_ets(values: np.ndarray, season_len: int) -> ExponentialSmoothing | None:
    y = np.asarray(values, dtype=float)
    if y.size < max(2 * season_len, 24):
        return None
    model = ExponentialSmoothing(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=season_len,
        initialization_method="estimated",
    )
    return model.fit(optimized=True, use_brute=False)


def forecast_series(
    series: pd.Series,
    horizon_months: int,
    season_len: int,
    alpha: float,
) -> pd.DataFrame:
    series = series.dropna()
    y = series.to_numpy(dtype=float)
    fit = None
    model_name = "seasonal_naive"
    try:
        fit = fit_ets(y, season_len=season_len)
    except Exception:
        fit = None

    if fit is not None:
        model_name = "ets"
        yhat = np.asarray(fit.forecast(horizon_months), dtype=float)
        resid = y - np.asarray(fit.fittedvalues, dtype=float)
    else:
        yhat = seasonal_naive(y, horizon=horizon_months, season_len=season_len)
        base = seasonal_naive(y, horizon=min(season_len, y.size), season_len=season_len)
        resid = y[-base.size :] - base

    q = np.nanquantile(np.abs(resid), 1.0 - alpha)
    yhat = np.clip(yhat, 0.0, 1.0)
    yhat_lower = np.clip(yhat - q, 0.0, 1.0)
    yhat_upper = np.clip(yhat + q, 0.0, 1.0)

    start = series.index.max() + pd.offsets.MonthBegin(1)
    ds = pd.date_range(start, periods=horizon_months, freq="MS")
    return pd.DataFrame(
        {
            "ds": ds,
            "yhat": yhat,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper,
            "model": model_name,
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = fetch_records(args.api_url, args.resource_id)
    daily = build_daily_frame(records)
    monthly = aggregate_monthly(daily)
    monthly.to_csv(args.output_dir / "istanbul_dam_monthly_history.csv", index=False)

    elmali_col = pick_column(monthly.columns, "Elmali")
    if elmali_col is None:
        raise RuntimeError("Elmali column not found in dataset.")

    targets = {
        "Elmali": elmali_col,
        "overall_mean": "overall_mean",
    }

    forecasts = []
    for label, col in targets.items():
        fc = forecast_series(
            series=monthly.set_index("ds")[col],
            horizon_months=args.horizon_months,
            season_len=args.season_len,
            alpha=args.alpha,
        )
        fc.insert(1, "series", label)
        forecasts.append(fc)

    forecast_df = pd.concat(forecasts, ignore_index=True)
    forecast_df.to_csv(args.output_dir / "istanbul_dam_15y_forecast.csv", index=False)

    summary = {
        "history_start": str(monthly["ds"].min().date()),
        "history_end": str(monthly["ds"].max().date()),
        "forecast_start": str(forecast_df["ds"].min().date()),
        "forecast_end": str(forecast_df["ds"].max().date()),
        "horizon_months": int(args.horizon_months),
        "series": list(targets.keys()),
        "alpha": float(args.alpha),
    }
    (args.output_dir / "istanbul_dam_15y_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
