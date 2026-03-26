#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot real observation-only climate charts with physical QC.")
    p.add_argument("--observations", type=Path, required=True, help="Input long-form observations file (parquet/csv).")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory where charts and monthly CSVs will be written.")
    p.add_argument("--start", type=str, default="1912-01-01", help="Lower date bound (inclusive).")
    p.add_argument("--end", type=str, default="2023-12-31", help="Upper date bound (inclusive).")
    p.add_argument(
        "--anomalies-dir",
        type=Path,
        default=None,
        help="Optional anomalies directory from quant output. If given, anomaly points are overlaid.",
    )
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if s == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported observations format: {path}")


def physical_qc(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["variable"] = out["variable"].astype(str).str.strip().str.lower()
    out = out.dropna(subset=["timestamp", "value", "variable"])

    p = out["variable"].eq("pressure")
    out.loc[p, "value"] = np.where(out.loc[p, "value"] <= 120.0, out.loc[p, "value"] + 700.0, out.loc[p, "value"])

    keep = (
        ((out["variable"] != "humidity") | out["value"].between(0, 100))
        & ((out["variable"] != "precip") | out["value"].between(0, 500))
        & ((out["variable"] != "pressure") | out["value"].between(650, 820))
        & ((out["variable"] != "temp") | out["value"].between(-40, 55))
    )
    return out[keep].copy()


def unit_for(var: str) -> str:
    return {"humidity": "%", "precip": "mm", "pressure": "mmHg", "temp": "C"}.get(var, "unknown")


def label_tr(var: str) -> str:
    return {"humidity": "Nem", "precip": "Yağış", "pressure": "Basınç", "temp": "Sıcaklık"}.get(var, var)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_table(args.observations)
    need = {"timestamp", "variable", "value"}
    if not need.issubset(raw.columns):
        raise SystemExit(f"Missing columns: expected {need}, got {set(raw.columns)}")

    obs = physical_qc(raw[list(need)])
    obs = obs[(obs["timestamp"] >= pd.Timestamp(args.start)) & (obs["timestamp"] <= pd.Timestamp(args.end))]

    variables = ["humidity", "precip", "pressure", "temp"]
    for var in variables:
        sub = obs[obs["variable"] == var].copy()
        if sub.empty:
            continue
        monthly = sub.set_index("timestamp")["value"].resample("MS").mean().dropna()
        if monthly.empty:
            continue

        monthly_df = monthly.reset_index()
        monthly_df.columns = ["ds", "value"]
        monthly_csv = out_dir / f"{var}_monthly_real_qc_{args.start}_to_{args.end}.csv"
        monthly_df.to_csv(monthly_csv, index=False)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(monthly_df["ds"], monthly_df["value"], color="#145da0", linewidth=1.2, label="Gerçek gözlem (aylık ortalama)")
        trend = monthly_df["value"].rolling(24, min_periods=8).mean().interpolate(limit_direction="both")
        ax.plot(monthly_df["ds"], trend, color="#111111", linestyle="--", linewidth=1.5, alpha=0.9, label="24 ay trend")

        if args.anomalies_dir is not None:
            cand = sorted(args.anomalies_dir.glob(f"{var}_*_anomalies_*.csv"))
            if cand:
                an = pd.read_csv(cand[0])
                if {"ds", "actual"}.issubset(an.columns):
                    an["ds"] = pd.to_datetime(an["ds"], errors="coerce")
                    an = an.dropna(subset=["ds"]).copy()
                    an = an[(an["ds"] >= pd.Timestamp(args.start)) & (an["ds"] <= pd.Timestamp(args.end))]
                    if not an.empty:
                        y_anom = an["actual"] if "actual" in an.columns else np.nan
                        ax.scatter(an["ds"], y_anom, color="#d62728", s=26, alpha=0.85, label=f"Anomali ({len(an)})", zorder=5)

        ax.set_title(f"Gerçek Gözlem Serisi (QC) - {label_tr(var)}")
        ax.set_xlabel("Tarih")
        ax.set_ylabel(f"Değer ({unit_for(var)})")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{var}_monthly_real_qc_{args.start}_to_{args.end}.png", dpi=180)
        plt.close(fig)

    print(f"Charts written: {out_dir}")


if __name__ == "__main__":
    main()
