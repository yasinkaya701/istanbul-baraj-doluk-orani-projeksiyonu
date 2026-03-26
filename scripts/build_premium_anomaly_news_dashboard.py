#!/usr/bin/env python3
"""Build a higher-quality interactive anomaly+news dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive premium dashboard for anomaly-news fusion.")
    p.add_argument(
        "--events-news-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/tum_asiri_olaylar_haber_enriched.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/premium_dashboard"),
    )
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument("--min-news-score", type=float, default=0.50)
    p.add_argument("--top-news-table", type=int, default=120)
    return p.parse_args()


def variable_order(df: pd.DataFrame) -> list[str]:
    pref = ["temp", "humidity", "pressure", "precip"]
    out = [v for v in pref if v in set(df["variable"].astype(str).str.lower())]
    if not out:
        out = sorted(df["variable"].astype(str).str.lower().unique().tolist())
    return out


def build_subplot(df: pd.DataFrame, start_year: int, end_year: int, min_news_score: float) -> go.Figure:
    vars_ = variable_order(df)
    rows, cols = 2, 2
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{v} | event severity + news" for v in vars_[:4]],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")

    for i, var in enumerate(vars_[:4]):
        r = i // 2 + 1
        c = i % 2 + 1
        sub = df[(df["variable"] == var)].copy()
        sub = sub[(sub["center_time"] >= lo) & (sub["center_time"] <= hi)].copy()
        if sub.empty:
            continue

        # Monthly max severity + smoothed median for trend readability.
        monthly = (
            sub.set_index("center_time")["peak_severity_score"]
            .resample("MS")
            .max()
            .reset_index()
            .rename(columns={"peak_severity_score": "monthly_max"})
        )
        monthly["roll_med"] = monthly["monthly_max"].rolling(12, min_periods=3).median()

        # All events.
        fig.add_trace(
            go.Scattergl(
                x=sub["center_time"],
                y=sub["peak_severity_score"],
                mode="markers",
                marker=dict(size=6, color="rgba(120,120,120,0.35)"),
                name=f"{var} all events",
                hovertemplate=(
                    "event=%{customdata[0]}<br>"
                    "time=%{x|%Y-%m-%d}<br>"
                    "severity=%{y:.3f}<br>"
                    "tier=%{customdata[1]}<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        sub["event_id"].astype(str).to_numpy(),
                        sub["scientific_tier"].fillna("").astype(str).to_numpy(),
                    ],
                    axis=1,
                ),
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # Smoothed trend.
        fig.add_trace(
            go.Scatter(
                x=monthly["center_time"],
                y=monthly["roll_med"],
                mode="lines",
                line=dict(color="#1f5c8d", width=2),
                name=f"{var} rolling median",
                hovertemplate="month=%{x|%Y-%m}<br>roll_med=%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # News-matched events.
        news = sub[sub["top_headline"].notna()].copy()
        news = news[pd.to_numeric(news["top_headline_match_score"], errors="coerce") >= float(min_news_score)].copy()
        if not news.empty:
            score = pd.to_numeric(news["top_headline_match_score"], errors="coerce").fillna(0.5)
            size = 9 + 14 * np.clip(score.to_numpy(), 0, 1)
            fig.add_trace(
                go.Scattergl(
                    x=news["center_time"],
                    y=news["peak_severity_score"],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=score,
                        colorscale="YlOrRd",
                        cmin=0.5,
                        cmax=1.0,
                        line=dict(width=0.5, color="#5a1a1a"),
                        showscale=False,
                    ),
                    name=f"{var} news matched",
                    hovertemplate=(
                        "event=%{customdata[0]}<br>"
                        "time=%{x|%Y-%m-%d}<br>"
                        "severity=%{y:.3f}<br>"
                        "score=%{customdata[1]:.3f}<br>"
                        "source=%{customdata[2]}<br>"
                        "headline=%{customdata[3]}<br>"
                        "url=%{customdata[4]}<extra></extra>"
                    ),
                    customdata=np.stack(
                        [
                            news["event_id"].astype(str).to_numpy(),
                            pd.to_numeric(news["top_headline_match_score"], errors="coerce").fillna(0.0).to_numpy(),
                            news["top_headline_source"].fillna("").astype(str).to_numpy(),
                            news["top_headline"].fillna("").astype(str).to_numpy(),
                            news["top_headline_url"].fillna("").astype(str).to_numpy(),
                        ],
                        axis=1,
                    ),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(range=[lo, hi], row=r, col=c, showgrid=True, gridcolor="rgba(200,200,200,0.2)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.2)", row=r, col=c)

    fig.update_layout(
        title="Premium Anomaly-News Fusion Dashboard (Interactive)",
        template="plotly_white",
        height=900,
        width=1500,
        margin=dict(l=40, r=20, t=80, b=40),
    )
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.events_news_csv).copy()
    df["variable"] = df["variable"].astype(str).str.lower().str.strip()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["center_time"] = df["start_time"] + (df["end_time"] - df["start_time"]) / 2
    df["peak_severity_score"] = pd.to_numeric(df["peak_severity_score"], errors="coerce")
    df = df.dropna(subset=["center_time", "peak_severity_score"]).copy()

    fig = build_subplot(
        df=df,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        min_news_score=float(args.min_news_score),
    )
    html_path = args.output_dir / "premium_anomali_haber_dashboard.html"
    fig.write_html(html_path, include_plotlyjs="cdn")

    # Data extracts for table/review.
    top_news = (
        df[df["top_headline"].notna()]
        .copy()
        .assign(top_headline_match_score=pd.to_numeric(df["top_headline_match_score"], errors="coerce"))
        .sort_values(["top_headline_match_score", "peak_severity_score"], ascending=[False, False])
        .head(int(args.top_news_table))
    )
    cols = [
        "event_id",
        "variable",
        "center_time",
        "peak_severity_score",
        "scientific_tier",
        "internet_confidence",
        "top_headline_match_score",
        "top_headline_date",
        "top_headline_source",
        "top_headline",
        "top_headline_url",
    ]
    top_news = top_news[cols]
    top_csv = args.output_dir / "premium_anomali_haber_top_news_table.csv"
    top_news.to_csv(top_csv, index=False)

    summary = (
        df.groupby("variable", as_index=False)
        .agg(
            event_count=("event_id", "size"),
            news_matched=("top_headline", lambda s: int(s.notna().sum())),
            median_severity=("peak_severity_score", "median"),
        )
        .sort_values("event_count", ascending=False)
    )
    sum_csv = args.output_dir / "premium_anomali_haber_summary.csv"
    summary.to_csv(sum_csv, index=False)

    md_path = args.output_dir / "premium_anomali_haber_dashboard_rapor.md"
    lines = [
        "# Premium Anomali-Haber Dashboard Raporu",
        "",
        f"- Input: `{args.events_news_csv}`",
        f"- Dashboard HTML: `{html_path}`",
        f"- News score filtresi: `>= {float(args.min_news_score):.2f}`",
        "",
        "## Degisken Ozet",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['variable']}: olay={int(r['event_count'])}, news_eslesme={int(r['news_matched'])}, "
            f"medyan_siddet={float(r['median_severity']):.3f}"
        )
    lines += [
        "",
        "## Cikti Dosyalari",
        f"- `{html_path}`",
        f"- `{top_csv}`",
        f"- `{sum_csv}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {html_path}")
    print(f"Wrote: {top_csv}")
    print(f"Wrote: {sum_csv}")
    print(f"Wrote: {md_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

