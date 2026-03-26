#!/usr/bin/env python3
"""Extract all extreme climate events into a single file.

Rules (per variable):
- tail extremes: value <= q_low or value >= q_high
- robust outlier: |robust_z| >= z_threshold
- abrupt jump: |delta_prev| >= jump_quantile threshold

Outputs:
- tum_asiri_olay_noktalari.csv  (point-level)
- tum_asiri_olaylar.csv         (event-level, grouped)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tum asiri iklim olaylarini tek dosyada topla.")
    p.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/quant_all_visuals_input/observations_with_all_visuals_for_quant.csv"),
        help="timestamp,variable,value iceren veri dosyasi",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events"),
    )
    p.add_argument("--low-quantile", type=float, default=0.01)
    p.add_argument("--high-quantile", type=float, default=0.99)
    p.add_argument("--z-threshold", type=float, default=3.5)
    p.add_argument("--jump-quantile", type=float, default=0.99)
    p.add_argument("--min-points", type=int, default=25)
    return p.parse_args()


@dataclass
class VarCfg:
    name: str
    unit: str


UNITS = {
    "temp": "C",
    "humidity": "%",
    "pressure": "hPa",
    "precip": "mm",
}


def read_table(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s in {".csv", ".txt"}:
        return pd.read_csv(path)
    if s in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if s in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if s in {".xlsx", ".xls", ".ods"}:
        return pd.read_excel(path)
    raise SystemExit(f"Unsupported extension: {path.suffix}")


def robust_z(x: pd.Series) -> pd.Series:
    med = float(x.median())
    mad = float((x - med).abs().median())
    if mad <= 1e-12:
        std = float(x.std(ddof=0))
        if std <= 1e-12:
            return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
        return (x - float(x.mean())) / std
    return 0.67448975 * (x - med) / mad


def severity_label(score: float) -> str:
    if score >= 3.0:
        return "kritik"
    if score >= 2.0:
        return "cok_yuksek"
    if score >= 1.2:
        return "yuksek"
    return "orta"


def gap_seconds(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 3:
        return 86400.0
    d = idx.to_series().diff().dropna().dt.total_seconds()
    if d.empty:
        return 86400.0
    g = float(d.median())
    return g if np.isfinite(g) and g > 0 else 86400.0


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df = read_table(args.observations).copy()
    needed = {"timestamp", "variable", "value"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"Missing required columns: {needed - set(df.columns)}")

    if "qc_flag" in df.columns:
        ok = df["qc_flag"].astype(str).str.lower().eq("ok")
        if ok.any():
            df = df[ok].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["variable"] = df["variable"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["timestamp", "value", "variable"]).sort_values(["variable", "timestamp"])

    point_rows: list[pd.DataFrame] = []

    for var in sorted(df["variable"].unique()):
        sub = df[df["variable"] == var].copy()
        if len(sub) < args.min_points:
            continue

        s = sub.groupby("timestamp", as_index=True)["value"].mean().sort_index()
        if len(s) < args.min_points:
            continue

        # STL Decomposition for robust seasonality/trend handling
        from statsmodels.tsa.seasonal import STL
        
        # Ensure we have a strictly periodic frequency for STL (daily)
        s_resampled = s.resample('D').mean().interpolate(method='linear')
        period = 365 # Annual seasonality
        
        try:
            res = STL(s_resampled, period=period, robust=True).fit()
            # We care about the 'resid' (residual) and 'seasonal' (for extreme seasonal shifts)
            residual = res.resid.reindex(s.index)
            seasonal = res.seasonal.reindex(s.index)
            trend = res.trend.reindex(s.index)
        except Exception as e:
            print(f"STL failed for {var}, falling back to simple residuals: {e}")
            residual = s - s.rolling(window=30, center=True).median().fillna(s.mean())
            seasonal = pd.Series(0, index=s.index)
            trend = s - residual

        rz = robust_z(residual)
        
        # Detection logic
        # 1. Residual Outliers (Short term shocks)
        is_rob = rz.abs() >= float(args.z_threshold)
        
        # 2. Tail Extremes on raw values (Physical limits)
        q_low = float(s.quantile(args.low_quantile))
        q_high = float(s.quantile(args.high_quantile))
        is_low = s <= q_low
        is_high = s >= q_high
        
        # 3. Abrupt Jump (First derivative)
        d_prev = s.diff()
        d_abs = d_prev.abs()
        jump_thr = float(d_abs.quantile(args.jump_quantile)) if len(d_abs.dropna()) else np.inf
        is_jump = d_abs >= jump_thr
        
        is_ext = is_low | is_high | is_rob | is_jump

        ext = pd.DataFrame(
            {
                "timestamp": s.index,
                "variable": var,
                "value": s.values,
                "unit": UNITS.get(var, "unknown"),
                "residual": residual.values,
                "seasonal": seasonal.values,
                "trend": trend.values,
                "robust_z": rz.values,
                "q_low": q_low,
                "q_high": q_high,
                "jump_threshold": jump_thr,
                "is_low_tail": is_low.values,
                "is_high_tail": is_high.values,
                "is_robust_outlier": is_rob.values,
                "is_jump": is_jump.values,
                "is_extreme": is_ext.values,
            }
        )
        ext = ext[ext["is_extreme"]].copy()
        if ext.empty:
            continue

        # Severity score: combine robust-z, tail distance and jump intensity.
        iqr = float(s.quantile(0.75) - s.quantile(0.25))
        if not np.isfinite(iqr) or iqr <= 1e-12:
            iqr = float(s.std(ddof=0))
        if not np.isfinite(iqr) or iqr <= 1e-12:
            iqr = 1.0

        tail_dist = np.maximum(
            (ext["value"] - q_high) / iqr,
            (q_low - ext["value"]) / iqr,
        )
        tail_dist = np.maximum(tail_dist, 0.0)
        z_score = ext["robust_z"].abs() / max(float(args.z_threshold), 1e-6)
        
        # d_abs and jump_thr are already defined in the outer scope
        jump_err_safe = max(jump_thr, 1e-6)
        jump_score = (ext["value"].diff().abs().fillna(0)) / jump_err_safe
        
        sev = np.maximum.reduce([tail_dist.values, z_score.values, jump_score.values])
        ext["severity_score"] = sev
        ext["severity_level"] = [severity_label(float(x)) for x in sev]
        ext["direction"] = np.where(ext["value"] >= q_high, "yuksek", np.where(ext["value"] <= q_low, "dusuk", "sicrama"))
        ext["reason_tags"] = (
            ext["is_low_tail"].map({True: "low_tail", False: ""})
            + "|"
            + ext["is_high_tail"].map({True: "high_tail", False: ""})
            + "|"
            + ext["is_robust_outlier"].map({True: "robust_outlier", False: ""})
            + "|"
            + ext["is_jump"].map({True: "jump", False: ""})
        ).str.strip("|")

        # Group nearby extremes into event blocks.
        ext = ext.sort_values("timestamp").reset_index(drop=True)
        gsec = gap_seconds(pd.DatetimeIndex(ext["timestamp"]))
        max_gap = pd.to_timedelta(max(60.0, gsec * 1.5), unit="s")
        ext["gap"] = ext["timestamp"].diff()
        ext["event_break"] = (ext["gap"].isna()) | (ext["gap"] > max_gap)
        ext["event_local_id"] = ext["event_break"].cumsum().astype(int)
        ext["event_id"] = ext["variable"] + "_" + ext["event_local_id"].astype(str)
        point_rows.append(ext.drop(columns=["gap", "event_break"]))

    if not point_rows:
        raise SystemExit("No extreme events found with current thresholds.")

    points = pd.concat(point_rows, ignore_index=True).sort_values(["timestamp", "variable"])

    events = (
        points.groupby(["event_id", "variable"], as_index=False)
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            duration_points=("timestamp", "size"),
            peak_severity_score=("severity_score", "max"),
            severity_level=("severity_level", "max"),
            max_value=("value", "max"),
            min_value=("value", "min"),
            dominant_direction=("direction", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
            reason_tags=("reason_tags", lambda x: "|".join(sorted(set("|".join(x.astype(str)).split("|")) - {""}))),
            q_low=("q_low", "first"),
            q_high=("q_high", "first"),
            jump_threshold=("jump_threshold", "first"),
            unit=("unit", "first"),
        )
        .sort_values(["start_time", "peak_severity_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    events.insert(0, "event_rank", np.arange(1, len(events) + 1))

    points_csv = out / "tum_asiri_olay_noktalari.csv"
    events_csv = out / "tum_asiri_olaylar.csv"
    summary_csv = out / "tum_asiri_olay_ozet.csv"

    points.to_csv(points_csv, index=False)
    events.to_csv(events_csv, index=False)

    summary = (
        events.groupby("variable", as_index=False)
        .agg(
            event_count=("event_id", "size"),
            first_event=("start_time", "min"),
            last_event=("end_time", "max"),
            max_peak_severity=("peak_severity_score", "max"),
        )
        .sort_values("event_count", ascending=False)
    )
    summary.to_csv(summary_csv, index=False)

    print(f"Wrote: {events_csv}")
    print(f"Wrote: {points_csv}")
    print(f"Wrote: {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
