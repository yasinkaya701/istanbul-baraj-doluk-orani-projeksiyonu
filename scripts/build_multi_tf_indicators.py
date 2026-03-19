#!/usr/bin/env python3
"""Build multi-timeframe indicator features from canonical observations dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TIMEFRAMES = {
    "1H": "1h",
    "8H": "8h",
    "1D": "1D",
    "1W": "1W-MON",
    "1M": "MS",
    "1Y": "YS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create finance-style indicators for climate variables across timeframes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/ml/observations.parquet"),
        help="Input canonical observations parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ml"),
        help="Output directory",
    )
    return parser.parse_args()


def _build_indicators_for_series(series: pd.Series, freq: str) -> pd.DataFrame:
    agg = series.resample(freq).agg(["first", "max", "min", "last", "mean", "std", "count"])
    agg = agg.rename(
        columns={
            "first": "open",
            "max": "high",
            "min": "low",
            "last": "close",
            "mean": "mean",
            "std": "std",
            "count": "n_obs",
        }
    )
    agg = agg.dropna(subset=["close"])

    close = agg["close"]
    returns = close.pct_change()
    agg["ret_1"] = returns
    agg["ret_3"] = close.pct_change(3)
    agg["sma_3"] = close.rolling(3, min_periods=1).mean()
    agg["sma_6"] = close.rolling(6, min_periods=1).mean()
    agg["sma_12"] = close.rolling(12, min_periods=1).mean()
    agg["ema_6"] = close.ewm(span=6, adjust=False).mean()
    agg["ema_12"] = close.ewm(span=12, adjust=False).mean()
    agg["momentum_3"] = close - close.shift(3)
    agg["range_hl"] = agg["high"] - agg["low"]
    agg["atr_6"] = agg["range_hl"].rolling(6, min_periods=1).mean()
    agg["vol_6"] = returns.rolling(6, min_periods=2).std()
    roll_mean = close.rolling(12, min_periods=3).mean()
    roll_std = close.rolling(12, min_periods=3).std()
    agg["zscore_12"] = (close - roll_mean) / roll_std

    return agg


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    clean = df[
        df["timestamp"].notna()
        & df["value"].notna()
        & (df["is_missing"] == False)  # noqa: E712
        & (df["qc_flag"] == "ok")
    ].copy()
    if clean.empty:
        raise SystemExit("No clean observations to build indicators from.")

    frames: list[pd.DataFrame] = []
    for variable in sorted(clean["variable"].dropna().unique()):
        sub = clean[clean["variable"] == variable].sort_values("timestamp")
        s = sub.set_index("timestamp")["value"]
        for tf_name, tf_freq in TIMEFRAMES.items():
            out = _build_indicators_for_series(s, tf_freq).reset_index()
            out["variable"] = variable
            out["timeframe"] = tf_name
            frames.append(out)

    indicators_long = pd.concat(frames, ignore_index=True)
    indicators_long = indicators_long[
        [
            "timestamp",
            "variable",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "mean",
            "std",
            "n_obs",
            "ret_1",
            "ret_3",
            "sma_3",
            "sma_6",
            "sma_12",
            "ema_6",
            "ema_12",
            "momentum_3",
            "range_hl",
            "atr_6",
            "vol_6",
            "zscore_12",
        ]
    ].sort_values(["timestamp", "variable", "timeframe"])

    # Wide feature table for ML training (timestamp index, flattened columns).
    wide = indicators_long.copy()
    wide["var_tf"] = wide["variable"] + "__" + wide["timeframe"]
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "mean",
        "std",
        "n_obs",
        "ret_1",
        "ret_3",
        "sma_3",
        "sma_6",
        "sma_12",
        "ema_6",
        "ema_12",
        "momentum_3",
        "range_hl",
        "atr_6",
        "vol_6",
        "zscore_12",
    ]
    pivot = wide.pivot_table(index="timestamp", columns="var_tf", values=feature_cols, aggfunc="last")
    pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index().sort_values("timestamp")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_pq = args.output_dir / "indicators_multitf_long.parquet"
    long_csv = args.output_dir / "indicators_multitf_long.csv"
    wide_pq = args.output_dir / "indicators_multitf_wide.parquet"
    wide_csv = args.output_dir / "indicators_multitf_wide.csv"

    indicators_long.to_parquet(long_pq, index=False)
    indicators_long.to_csv(long_csv, index=False)
    pivot.to_parquet(wide_pq, index=False)
    pivot.to_csv(wide_csv, index=False)

    print(f"Wrote {len(indicators_long)} rows to {long_pq}")
    print(f"Wrote {len(pivot)} rows to {wide_pq}")
    print(f"Also exported CSV: {long_csv} and {wide_csv}")


if __name__ == "__main__":
    main()

