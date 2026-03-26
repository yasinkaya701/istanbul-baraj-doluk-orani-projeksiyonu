#!/usr/bin/env python3
"""Higher-quality static charts for anomaly events + matched news headlines."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# Matplotlib cache under writable temp.
_MPL_CACHE = Path(tempfile.gettempdir()) / "anomaly_news_v2_mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build improved anomaly+news charts (v2).")
    p.add_argument(
        "--events-news-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/tum_asiri_olaylar_haber_enriched.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/charts_v2"),
    )
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--min-news-score", type=float, default=0.50)
    p.add_argument("--top-headlines", type=int, default=8)
    return p.parse_args()


def short_text(x: str, n: int = 78) -> str:
    s = str(x).strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def plot_variable_v2(
    var: str,
    events_all: pd.DataFrame,
    events_news: pd.DataFrame,
    out_path: Path,
    start_year: int,
    end_year: int,
    top_headlines: int,
) -> None:
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")
    ea = events_all[(events_all["variable"] == var)].copy()
    ea = ea[(ea["center_time"] >= lo) & (ea["center_time"] <= hi)].sort_values("center_time")
    en = events_news[(events_news["variable"] == var)].copy()
    en = en[(en["center_time"] >= lo) & (en["center_time"] <= hi)].sort_values("center_time")
    if ea.empty:
        return

    # Monthly aggregation for smoother context.
    monthly = (
        ea.set_index("center_time")["peak_severity_score"].resample("MS").max().to_frame("monthly_max").reset_index()
    )
    monthly["roll_med"] = monthly["monthly_max"].rolling(12, min_periods=3).median()
    monthly["roll_p90"] = monthly["monthly_max"].rolling(18, min_periods=6).quantile(0.90)
    monthly["roll_p10"] = monthly["monthly_max"].rolling(18, min_periods=6).quantile(0.10)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.4, 1.9], hspace=0.22)

    # Panel 1: Event severity timeline + matched news markers.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(ea["center_time"], ea["peak_severity_score"], s=16, color="#9a9a9a", alpha=0.45, label="Tum olaylar")
    ax1.plot(monthly["center_time"], monthly["roll_med"], color="#1f5c8d", lw=1.8, label="12-ay rolling median")
    ax1.fill_between(
        monthly["center_time"],
        monthly["roll_p10"],
        monthly["roll_p90"],
        color="#7cb5d6",
        alpha=0.20,
        label="18-ay rolling p10-p90",
    )
    if not en.empty:
        sc = en["top_headline_match_score"].clip(0, 1).fillna(0.5)
        size = 70 + 110 * sc
        ax1.scatter(
            en["center_time"],
            en["peak_severity_score"],
            s=size,
            color="#d1495b",
            alpha=0.86,
            edgecolor="white",
            linewidth=0.7,
            label="Haber eslesmis olay",
            zorder=5,
        )
    ax1.set_title(f"{var} | Anomali olay siddeti + haber eslesmeleri (v2)")
    ax1.set_ylabel("Peak severity score")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: News match score by time.
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if not en.empty:
        ax2.plot(en["center_time"], en["top_headline_match_score"], color="#d1495b", lw=1.2, alpha=0.8)
        ax2.scatter(
            en["center_time"],
            en["top_headline_match_score"],
            c=en["top_headline_match_score"],
            cmap="YlOrRd",
            s=55,
            vmin=0.5,
            vmax=1.0,
            edgecolor="#6b1f1f",
            linewidth=0.5,
        )
    ax2.axhline(0.5, color="#666666", lw=0.8, ls="--", alpha=0.6)
    ax2.set_ylim(0.45, 1.02)
    ax2.set_ylabel("News match score")
    ax2.grid(alpha=0.22)

    # Panel 3: Top headline list.
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    if en.empty:
        txt = "Bu degisken icin haber eslesmesi yok."
    else:
        top = en.sort_values(["top_headline_match_score", "peak_severity_score"], ascending=[False, False]).head(top_headlines)
        lines = ["Top headline baglantilari:"]
        for _, r in top.iterrows():
            lines.append(
                f"- {pd.Timestamp(r['top_headline_date']).date()} | {r['top_headline_source']} | "
                f"score={float(r['top_headline_match_score']):.3f} | {short_text(r['top_headline'], 92)}"
            )
        txt = "\n".join(lines)
    ax3.text(
        0.01,
        0.98,
        txt,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "fc": "#f6f8fa", "ec": "#d0d7de"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dashboard_v2(
    events_all: pd.DataFrame,
    events_news: pd.DataFrame,
    out_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    vars_order = [v for v in ["temp", "humidity", "pressure", "precip"] if v in set(events_all["variable"])]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    axes = axes.ravel()
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")

    for i, var in enumerate(vars_order):
        ax = axes[i]
        ea = events_all[(events_all["variable"] == var)].copy()
        ea = ea[(ea["center_time"] >= lo) & (ea["center_time"] <= hi)].sort_values("center_time")
        en = events_news[(events_news["variable"] == var)].copy()
        en = en[(en["center_time"] >= lo) & (en["center_time"] <= hi)].sort_values("center_time")
        if ea.empty:
            ax.set_title(f"{var}: veri yok")
            continue
        monthly = ea.set_index("center_time")["peak_severity_score"].resample("MS").max().reset_index()
        monthly["roll_med"] = monthly["peak_severity_score"].rolling(12, min_periods=3).median()
        ax.scatter(ea["center_time"], ea["peak_severity_score"], s=11, color="#9a9a9a", alpha=0.35)
        ax.plot(monthly["center_time"], monthly["roll_med"], color="#1f5c8d", lw=1.4)
        if not en.empty:
            ax.scatter(en["center_time"], en["peak_severity_score"], s=45, color="#d1495b", alpha=0.88)
        ax.set_title(f"{var} | matched={len(en)}")
        ax.grid(alpha=0.22)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for j in range(len(vars_order), 4):
        axes[j].axis("off")
    fig.suptitle("Anomali-Haber Birlesik Dashboard v2", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.events_news_csv).copy()
    df["variable"] = df["variable"].astype(str).str.lower().str.strip()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["center_time"] = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
    df["peak_severity_score"] = pd.to_numeric(df["peak_severity_score"], errors="coerce")
    df["top_headline_match_score"] = pd.to_numeric(df["top_headline_match_score"], errors="coerce")
    df = df.dropna(subset=["center_time", "peak_severity_score"]).copy()

    events_all = df.copy()
    events_news = df[df["top_headline"].notna()].copy()
    events_news = events_news[events_news["top_headline_match_score"] >= float(args.min_news_score)].copy()

    vars_order = [v for v in ["temp", "humidity", "pressure", "precip"] if v in set(events_all["variable"])]
    for var in vars_order:
        p = args.output_dir / f"anomali_haber_v2_{var}.png"
        plot_variable_v2(
            var=var,
            events_all=events_all,
            events_news=events_news,
            out_path=p,
            start_year=int(args.start_year),
            end_year=int(args.end_year),
            top_headlines=int(args.top_headlines),
        )

    dash = args.output_dir / "anomali_haber_v2_dashboard.png"
    plot_dashboard_v2(
        events_all=events_all,
        events_news=events_news,
        out_path=dash,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )

    # Monthly export for reuse.
    monthly = (
        events_all.set_index("center_time")
        .groupby("variable")["peak_severity_score"]
        .resample("MS")
        .max()
        .reset_index()
        .rename(columns={"peak_severity_score": "monthly_max_severity"})
    )
    monthly_csv = args.output_dir / "anomali_haber_v2_monthly_max.csv"
    monthly.to_csv(monthly_csv, index=False)

    summary = (
        events_all.groupby("variable", as_index=False)
        .agg(
            event_count=("event_id", "size"),
            matched_news=("top_headline", lambda s: int(s.notna().sum())),
            matched_news_strict=(
                "top_headline_match_score",
                lambda s: int((pd.to_numeric(s, errors="coerce") >= float(args.min_news_score)).sum()),
            ),
            median_severity=("peak_severity_score", "median"),
        )
        .sort_values("event_count", ascending=False)
    )
    sum_csv = args.output_dir / "anomali_haber_v2_ozet.csv"
    summary.to_csv(sum_csv, index=False)

    report = args.output_dir / "anomali_haber_v2_rapor.md"
    lines = [
        "# Anomali-Haber Birlesik Grafik v2 Raporu",
        "",
        f"- Input: `{args.events_news_csv}`",
        f"- Dashboard: `{dash}`",
        f"- News strict threshold: `{float(args.min_news_score):.2f}`",
        "",
        "## Degisken Ozet",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['variable']}: olay={int(r['event_count'])}, haber={int(r['matched_news'])}, "
            f"haber_strict={int(r['matched_news_strict'])}, medyan_siddet={float(r['median_severity']):.3f}"
        )
    lines += [
        "",
        "## Ciktilar",
        f"- `{dash}`",
        f"- `{monthly_csv}`",
        f"- `{sum_csv}`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {dash}")
    print(f"Wrote: {monthly_csv}")
    print(f"Wrote: {sum_csv}")
    print(f"Wrote: {report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
