#!/usr/bin/env python3
"""Build continuous (monthly) anomaly-news visual report."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_MPL_CACHE = Path(tempfile.gettempdir()) / "continuous_anomaly_news_mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous anomaly-news report generator")
    p.add_argument(
        "--events-news-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/tum_asiri_olaylar_haber_enriched.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/continuous"),
    )
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--news-score-threshold", type=float, default=0.50)
    p.add_argument("--ema-span", type=int, default=12)
    p.add_argument("--top-annotations", type=int, default=10)
    return p.parse_args()


def short_text(x: str, n: int = 70) -> str:
    s = str(x).strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["variable"] = df["variable"].astype(str).str.lower().str.strip()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["center_time"] = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
    df["peak_severity_score"] = pd.to_numeric(df["peak_severity_score"], errors="coerce")
    df["top_headline_match_score"] = pd.to_numeric(df["top_headline_match_score"], errors="coerce")
    return df.dropna(subset=["variable", "center_time", "peak_severity_score"]).copy()


def build_monthly_continuous(
    events: pd.DataFrame,
    start_year: int,
    end_year: int,
    news_score_threshold: float,
    ema_span: int,
) -> pd.DataFrame:
    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")
    out = []

    for var in sorted(events["variable"].unique()):
        sub = events[events["variable"] == var].copy()
        monthly = (
            sub.set_index("center_time")
            .groupby(pd.Grouper(freq="MS"))
            .agg(
                monthly_event_count=("event_id", "size"),
                monthly_max_severity=("peak_severity_score", "max"),
                monthly_mean_severity=("peak_severity_score", "mean"),
                news_count=("top_headline", lambda s: int(s.notna().sum())),
                news_max_score=("top_headline_match_score", "max"),
            )
            .reset_index()
            .rename(columns={"center_time": "month"})
        )
        monthly = monthly.rename(columns={monthly.columns[0]: "month"})
        full = pd.DataFrame({"month": idx}).merge(monthly, on="month", how="left")
        full["variable"] = var
        full["monthly_event_count"] = full["monthly_event_count"].fillna(0).astype(int)
        full["monthly_max_severity"] = full["monthly_max_severity"].fillna(0.0)
        full["monthly_mean_severity"] = full["monthly_mean_severity"].fillna(0.0)
        full["news_count"] = full["news_count"].fillna(0).astype(int)
        full["news_max_score"] = full["news_max_score"].fillna(0.0)
        full["severity_ema"] = (
            full["monthly_max_severity"]
            .ewm(span=max(2, int(ema_span)), adjust=False, min_periods=1)
            .mean()
        )
        full["news_strict"] = (full["news_max_score"] >= float(news_score_threshold)).astype(int)

        # robust normalization for multi-variable unified chart
        q95 = float(np.nanquantile(full["monthly_max_severity"], 0.95)) if len(full) else 1.0
        q95 = max(q95, 1e-6)
        full["severity_norm"] = (full["monthly_max_severity"] / q95).clip(0, 1.5)
        out.append(full)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_overview(monthly: pd.DataFrame, out_png: Path, start_year: int, end_year: int) -> None:
    vars_order = [v for v in ["temp", "humidity", "pressure", "precip"] if v in set(monthly["variable"])]
    if not vars_order:
        vars_order = sorted(monthly["variable"].unique().tolist())
    colors = {
        "temp": "#c0392b",
        "humidity": "#2471a3",
        "pressure": "#7d3c98",
        "precip": "#117a65",
    }

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    for var in vars_order:
        sub = monthly[monthly["variable"] == var].copy()
        ax.plot(
            sub["month"],
            sub["severity_norm"],
            lw=1.6,
            alpha=0.92,
            label=f"{var} (normalized)",
            color=colors.get(var, None),
        )
        # news markers
        n = sub[sub["news_strict"] == 1]
        if not n.empty:
            ax.scatter(
                n["month"],
                n["severity_norm"],
                s=28,
                alpha=0.8,
                color=colors.get(var, "#333333"),
                edgecolor="white",
                linewidth=0.6,
            )

    ax.set_title("SUREKLI (Aylik) Anomali Sinyali + Haber Eslesme Noktalari")
    ax.set_ylabel("Normalized monthly max severity")
    ax.set_xlabel("Tarih")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    ax.set_xlim(pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{end_year}-12-31"))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190)
    plt.close(fig)


def plot_variable(
    variable: str,
    monthly: pd.DataFrame,
    events_news: pd.DataFrame,
    out_png: Path,
    start_year: int,
    end_year: int,
    top_annotations: int,
) -> None:
    sub = monthly[monthly["variable"] == variable].copy()
    if sub.empty:
        return
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")
    sub = sub[(sub["month"] >= lo) & (sub["month"] <= hi)].copy()
    if sub.empty:
        return

    ev = events_news[events_news["variable"] == variable].copy()
    ev = ev[(ev["center_time"] >= lo) & (ev["center_time"] <= hi)].copy()

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.8, 1.6, 1.5], hspace=0.15)

    # Panel 1: continuous severity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sub["month"], sub["monthly_max_severity"], color="#606060", lw=1.0, alpha=0.7, label="Monthly max severity")
    ax1.plot(sub["month"], sub["severity_ema"], color="#1f5c8d", lw=2.0, label="EMA trend")
    news_months = sub[sub["news_strict"] == 1]
    if not news_months.empty:
        ax1.scatter(
            news_months["month"],
            news_months["monthly_max_severity"],
            s=48,
            color="#d1495b",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.7,
            label="News matched month",
            zorder=5,
        )
    ax1.set_title(f"{variable} | Surekli aylik anomali-seri + haber eslesmeleri")
    ax1.set_ylabel("Severity")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: events and news score
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.bar(sub["month"], sub["monthly_event_count"], width=25, color="#9fb3c8", alpha=0.55, label="Event count")
    ax2b = ax2.twinx()
    ax2b.plot(sub["month"], sub["news_max_score"], color="#d1495b", lw=1.5, alpha=0.85, label="News max score")
    ax2.set_ylabel("Event count")
    ax2b.set_ylabel("News score")
    ax2.set_ylim(bottom=0)
    ax2b.set_ylim(0, 1.02)
    ax2.grid(alpha=0.2)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left")

    # Panel 3: top matched headlines
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    if ev.empty:
        txt = "Bu degisken icin haber eslesmesi yok."
    else:
        top = ev.sort_values(["top_headline_match_score", "peak_severity_score"], ascending=[False, False]).head(top_annotations)
        lines = ["Top haber basliklari (skora gore):"]
        for _, r in top.iterrows():
            lines.append(
                f"- {pd.Timestamp(r['center_time']).date()} | score={float(r['top_headline_match_score']):.3f} | "
                f"{r['top_headline_source']} | {short_text(r['top_headline'])}"
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

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=185)
    plt.close(fig)


def build_pdf(output_dir: Path, overview_png: Path, per_var_pngs: list[Path], pdf_path: Path) -> None:
    with PdfPages(pdf_path) as pdf:
        for p in [overview_png] + per_var_pngs:
            if not p.exists():
                continue
            img = plt.imread(p)
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(args.events_news_csv)
    events = events[
        (events["center_time"] >= pd.Timestamp(f"{args.start_year}-01-01"))
        & (events["center_time"] <= pd.Timestamp(f"{args.end_year}-12-31"))
    ].copy()
    if events.empty:
        raise SystemExit("Event window empty for selected year range.")

    events_news = events[events["top_headline"].notna()].copy()
    events_news = events_news[
        pd.to_numeric(events_news["top_headline_match_score"], errors="coerce") >= float(args.news_score_threshold)
    ].copy()

    monthly = build_monthly_continuous(
        events=events,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        news_score_threshold=float(args.news_score_threshold),
        ema_span=int(args.ema_span),
    )
    if monthly.empty:
        raise SystemExit("Monthly continuous table is empty.")

    overview_png = args.output_dir / "surekli_anomali_haber_overview.png"
    plot_overview(
        monthly=monthly,
        out_png=overview_png,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )

    per_var = []
    vars_order = [v for v in ["temp", "humidity", "pressure", "precip"] if v in set(monthly["variable"])]
    for var in vars_order:
        p = args.output_dir / f"surekli_anomali_haber_{var}.png"
        plot_variable(
            variable=var,
            monthly=monthly,
            events_news=events_news,
            out_png=p,
            start_year=int(args.start_year),
            end_year=int(args.end_year),
            top_annotations=int(args.top_annotations),
        )
        per_var.append(p)

    monthly_csv = args.output_dir / "surekli_anomali_haber_aylik.csv"
    monthly.to_csv(monthly_csv, index=False)

    ev_csv = args.output_dir / "surekli_anomali_haber_news_events.csv"
    events_news.to_csv(ev_csv, index=False)

    summary = (
        events.groupby("variable", as_index=False)
        .agg(
            event_count=("event_id", "size"),
            news_matched=("top_headline", lambda s: int(s.notna().sum())),
            news_matched_strict=(
                "top_headline_match_score",
                lambda s: int((pd.to_numeric(s, errors="coerce") >= float(args.news_score_threshold)).sum()),
            ),
            median_severity=("peak_severity_score", "median"),
        )
        .sort_values("event_count", ascending=False)
    )
    summary_csv = args.output_dir / "surekli_anomali_haber_ozet.csv"
    summary.to_csv(summary_csv, index=False)

    pdf_path = args.output_dir / "surekli_anomali_haber_rapor.pdf"
    build_pdf(args.output_dir, overview_png, per_var, pdf_path)

    md_path = args.output_dir / "surekli_anomali_haber_rapor.md"
    lines = [
        "# Surekli Anomali-Haber Raporu",
        "",
        f"- Input: `{args.events_news_csv}`",
        f"- Zaman araligi: {args.start_year}-{args.end_year}",
        f"- News score threshold: {float(args.news_score_threshold):.2f}",
        f"- EMA span: {int(args.ema_span)} ay",
        "",
        "## Ciktilar",
        f"- Overview PNG: `{overview_png}`",
        f"- PDF rapor: `{pdf_path}`",
        f"- Aylik tablo: `{monthly_csv}`",
        f"- News event tablo: `{ev_csv}`",
        f"- Ozet: `{summary_csv}`",
        "",
        "## Degisken Ozet",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['variable']}: olay={int(r['event_count'])}, haber={int(r['news_matched'])}, "
            f"haber_strict={int(r['news_matched_strict'])}, medyan_siddet={float(r['median_severity']):.3f}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {overview_png}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {monthly_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {md_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
