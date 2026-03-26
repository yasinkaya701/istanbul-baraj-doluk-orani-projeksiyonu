#!/usr/bin/env python3
"""Calibrate mixed-source observations and extract recent contiguous regimes.

Purpose:
- Fix known scale issues (e.g., pressure values missing leading 9xx).
- Remove large historical gaps that destabilize long-horizon training.
- Produce a model-ready calibrated table with provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate observations and keep recent contiguous regimes.")
    p.add_argument(
        "--input-observations",
        type=Path,
        default=Path(
            "/Users/yasinkaya/Hackhaton/output/data_factory/run_20260306_000419/prepared/observations_with_all_visuals_for_quant.parquet"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/data_factory/run_20260306_000419/calibrated"),
    )
    p.add_argument(
        "--gap-years",
        type=float,
        default=5.0,
        help="If consecutive observations have a gap larger than this, keep only the last segment.",
    )
    p.add_argument(
        "--pressure-offset",
        type=float,
        default=900.0,
        help="Offset added to low-scale numeric pressure values (e.g., 63.7 -> 963.7).",
    )
    p.add_argument(
        "--pressure-low-min",
        type=float,
        default=40.0,
        help="Lower bound for low-scale pressure detection.",
    )
    p.add_argument(
        "--pressure-low-max",
        type=float,
        default=120.0,
        help="Upper bound for low-scale pressure detection.",
    )
    return p.parse_args()


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def calibrate_pressure(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    out = ensure_columns(out, ["source_kind", "method"])
    m = (
        out["variable"].astype(str).eq("pressure")
        & out["source_kind"].astype(str).str.lower().eq("numeric")
        & pd.to_numeric(out["value"], errors="coerce").between(args.pressure_low_min, args.pressure_low_max, inclusive="both")
    )
    n = int(m.sum())
    if n > 0:
        out.loc[m, "value"] = pd.to_numeric(out.loc[m, "value"], errors="coerce") + float(args.pressure_offset)
        out.loc[m, "method"] = out.loc[m, "method"].astype(str) + f"+pressure_offset_{int(args.pressure_offset)}"
    return out, n


def find_recent_regime_start(ts: pd.Series, gap_days: float) -> pd.Timestamp | None:
    s = pd.to_datetime(ts, errors="coerce").dropna().sort_values().drop_duplicates()
    if s.empty:
        return None
    dif = s.diff().dt.total_seconds().div(86400.0)
    breaks = np.where(dif.to_numpy() > gap_days)[0]
    if len(breaks) == 0:
        return s.iloc[0]
    last_break_idx = int(breaks[-1])
    return s.iloc[last_break_idx]


def summarize_by_var(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["variable", "rows", "min_ts", "max_ts", "median_value"])
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    return (
        x.groupby("variable", dropna=False)
        .agg(
            rows=("value", "size"),
            min_ts=("timestamp", "min"),
            max_ts=("timestamp", "max"),
            median_value=("value", "median"),
        )
        .reset_index()
        .sort_values("variable")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_observations.exists():
        raise SystemExit(f"input not found: {args.input_observations}")

    if args.input_observations.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(args.input_observations)
    else:
        raw = pd.read_csv(args.input_observations)

    required = ["timestamp", "variable", "value"]
    for c in required:
        if c not in raw.columns:
            raise SystemExit(f"missing required column: {c}")

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "variable", "value"]).sort_values(["timestamp", "variable"]).reset_index(drop=True)
    df = ensure_columns(df, ["qc_flag", "source_kind", "source_file", "method", "confidence"])
    before_summary = summarize_by_var(df)

    calibrated, pressure_fix_count = calibrate_pressure(df, args)
    calibrated = calibrated.sort_values(["timestamp", "variable"]).reset_index(drop=True)
    after_cal_summary = summarize_by_var(calibrated)

    gap_days = float(args.gap_years) * 365.0
    keep_rows: list[pd.DataFrame] = []
    regime_rows: list[dict[str, Any]] = []
    for var, sub in calibrated.groupby("variable", dropna=False):
        start_ts = find_recent_regime_start(sub["timestamp"], gap_days=gap_days)
        if start_ts is None:
            continue
        kept = sub[sub["timestamp"] >= start_ts].copy()
        keep_rows.append(kept)
        regime_rows.append(
            {
                "variable": str(var),
                "recent_regime_start": str(pd.Timestamp(start_ts)),
                "rows_before": int(len(sub)),
                "rows_after": int(len(kept)),
            }
        )

    recent = pd.concat(keep_rows, ignore_index=True) if keep_rows else calibrated.iloc[0:0].copy()
    recent = recent.sort_values(["timestamp", "variable"]).reset_index(drop=True)
    recent_summary = summarize_by_var(recent)
    regime_df = pd.DataFrame(regime_rows).sort_values("variable").reset_index(drop=True) if regime_rows else pd.DataFrame()

    full_pq = args.output_dir / "observations_calibrated_full.parquet"
    full_csv = args.output_dir / "observations_calibrated_full.csv"
    recent_pq = args.output_dir / "observations_calibrated_recent_regime.parquet"
    recent_csv = args.output_dir / "observations_calibrated_recent_regime.csv"

    calibrated.to_parquet(full_pq, index=False)
    calibrated.to_csv(full_csv, index=False)
    recent.to_parquet(recent_pq, index=False)
    recent.to_csv(recent_csv, index=False)

    before_summary.to_csv(args.output_dir / "summary_before.csv", index=False)
    after_cal_summary.to_csv(args.output_dir / "summary_after_calibration.csv", index=False)
    recent_summary.to_csv(args.output_dir / "summary_recent_regime.csv", index=False)
    regime_df.to_csv(args.output_dir / "recent_regime_boundaries.csv", index=False)

    summary = {
        "input_observations": str(args.input_observations),
        "rows_input": int(len(df)),
        "pressure_fix_count": int(pressure_fix_count),
        "gap_years": float(args.gap_years),
        "rows_calibrated_full": int(len(calibrated)),
        "rows_recent_regime": int(len(recent)),
        "outputs": {
            "full_parquet": str(full_pq),
            "full_csv": str(full_csv),
            "recent_parquet": str(recent_pq),
            "recent_csv": str(recent_csv),
            "summary_before_csv": str(args.output_dir / "summary_before.csv"),
            "summary_after_calibration_csv": str(args.output_dir / "summary_after_calibration.csv"),
            "summary_recent_regime_csv": str(args.output_dir / "summary_recent_regime.csv"),
            "recent_regime_boundaries_csv": str(args.output_dir / "recent_regime_boundaries.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

