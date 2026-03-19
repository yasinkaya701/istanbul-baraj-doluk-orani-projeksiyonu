#!/usr/bin/env python3
"""Enrich extreme events with quant anomaly context and build report files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asiri olaylari neden baglamiyla zenginlestir.")
    p.add_argument(
        "--events-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar.csv"),
    )
    p.add_argument(
        "--context-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/quant_all_visuals_package/reports/top_anomalies_global_context_input.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events"),
    )
    p.add_argument(
        "--max-day-distance",
        type=int,
        default=45,
        help="Ayni ay/gun eslesmesi yoksa merkeze en yakin baglam kaydi icin gun esigi.",
    )
    return p.parse_args()


def match_context_row(ev: pd.Series, ctx_var: pd.DataFrame, max_day_distance: int) -> tuple[str, pd.Series | None]:
    if ctx_var.empty:
        return "none", None

    start = pd.Timestamp(ev["start_time"])
    end = pd.Timestamp(ev["end_time"])
    center = start + (end - start) / 2

    # Priority 1: same year-month for temp (monthly model).
    if str(ev["variable"]) == "temp":
        ym = center.strftime("%Y-%m")
        m = ctx_var[ctx_var["ds"].dt.strftime("%Y-%m") == ym]
        if not m.empty:
            return "month", m.iloc[0]

    # Priority 2: same year for yearly modeled variables.
    y = int(center.year)
    m = ctx_var[ctx_var["ds"].dt.year == y]
    if not m.empty:
        # pick closest in days within same year
        dd = (m["ds"] - center).abs().dt.days
        return "year", m.iloc[int(dd.argmin())]

    # Priority 3: nearest date within threshold.
    dd = (ctx_var["ds"] - center).abs().dt.days
    i = int(dd.argmin())
    if int(dd.iloc[i]) <= int(max_day_distance):
        return "nearest", ctx_var.iloc[i]

    return "none", None


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(args.events_csv).copy()
    ev["variable"] = ev["variable"].astype(str).str.lower().str.strip()
    ev["start_time"] = pd.to_datetime(ev["start_time"], errors="coerce")
    ev["end_time"] = pd.to_datetime(ev["end_time"], errors="coerce")
    ev = ev.dropna(subset=["start_time", "end_time", "variable"]).copy()

    ctx = pd.read_csv(args.context_csv).copy()
    ctx["variable"] = ctx["variable"].astype(str).str.lower().str.strip()
    ctx["ds"] = pd.to_datetime(ctx["ds"], errors="coerce")
    ctx = ctx.dropna(subset=["ds", "variable"]).copy()

    # Ensure columns exist in output.
    extra_cols = [
        "context_match_type",
        "context_ds",
        "context_zscore",
        "context_robust_zscore",
        "context_anomaly_type_tr",
        "context_cause_primary",
        "context_cause_details_tr",
        "context_local_pattern_hint",
        "context_local_event_match",
        "context_global_event_match",
        "context_cause_confidence",
    ]
    for c in extra_cols:
        ev[c] = ""

    for i in ev.index:
        row = ev.loc[i]
        ctx_var = ctx[ctx["variable"] == row["variable"]]
        mtype, hit = match_context_row(row, ctx_var, max_day_distance=int(args.max_day_distance))
        if hit is None:
            ev.at[i, "context_match_type"] = "none"
            continue

        ev.at[i, "context_match_type"] = mtype
        ev.at[i, "context_ds"] = str(pd.Timestamp(hit["ds"]).date())
        ev.at[i, "context_zscore"] = hit.get("zscore", "")
        ev.at[i, "context_robust_zscore"] = hit.get("robust_zscore", "")
        ev.at[i, "context_anomaly_type_tr"] = hit.get("anomaly_type_tr", "")
        ev.at[i, "context_cause_primary"] = hit.get("cause_primary", "")
        ev.at[i, "context_cause_details_tr"] = hit.get("cause_details_tr", "")
        ev.at[i, "context_local_pattern_hint"] = hit.get("local_pattern_hint", "")
        ev.at[i, "context_local_event_match"] = hit.get("local_event_match", "")
        ev.at[i, "context_global_event_match"] = hit.get("global_event_match", "")
        ev.at[i, "context_cause_confidence"] = hit.get("cause_confidence", "")

    ev["has_context"] = ev["context_match_type"].astype(str).str.lower().ne("none")
    ev = ev.sort_values(["peak_severity_score", "duration_points"], ascending=[False, False]).reset_index(drop=True)
    ev.insert(0, "rank_by_severity", np.arange(1, len(ev) + 1))

    z_csv = out / "tum_asiri_olaylar_zengin.csv"
    ev.to_csv(z_csv, index=False)

    coverage = (
        ev.groupby("variable", as_index=False)
        .agg(
            event_count=("event_id", "size"),
            context_count=("has_context", "sum"),
        )
        .sort_values("event_count", ascending=False)
    )
    coverage["context_ratio"] = coverage["context_count"] / coverage["event_count"]
    cov_csv = out / "tum_asiri_olay_baglam_kapsami.csv"
    coverage.to_csv(cov_csv, index=False)

    top = ev.head(40).copy()
    md_path = out / "tum_asiri_olaylar_rapor.md"
    lines = [
        "# Tüm Aşırı Olaylar Raporu",
        "",
        f"- Toplam olay: **{len(ev)}**",
        f"- Bağlam eşleşmesi bulunan olay: **{int(ev['has_context'].sum())}**",
        "",
        "## Değişken Bazlı Özet",
    ]
    for _, r in coverage.iterrows():
        lines.append(
            f"- {r['variable']}: olay={int(r['event_count'])}, bağlam={int(r['context_count'])}, oran={r['context_ratio']:.2%}"
        )
    lines += [
        "",
        "## En Şiddetli 40 Olay",
        "",
        "|Sıra|Event ID|Değişken|Başlangıç|Bitiş|Şiddet|Seviye|Muhtemel Sebep|Bağlam|",
        "|---:|---|---|---|---|---:|---|---|---|",
    ]
    for _, r in top.iterrows():
        reason = str(r.get("context_cause_primary", "")).strip() or "-"
        mtype = str(r.get("context_match_type", "none"))
        lines.append(
            f"|{int(r['rank_by_severity'])}|{r['event_id']}|{r['variable']}|{pd.Timestamp(r['start_time']).date()}|{pd.Timestamp(r['end_time']).date()}|{float(r['peak_severity_score']):.2f}|{r['severity_level']}|{reason}|{mtype}|"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {z_csv}")
    print(f"Wrote: {cov_csv}")
    print(f"Wrote: {md_path}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
