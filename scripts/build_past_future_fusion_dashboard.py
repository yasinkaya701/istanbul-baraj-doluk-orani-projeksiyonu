#!/usr/bin/env python3
"""Past+Future unified dashboard: anomalies + news + forecast to 2035."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_MPL_CACHE = Path(tempfile.gettempdir()) / "past_future_fusion_mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Past+Future anomaly-news-forecast fusion dashboard")
    p.add_argument(
        "--continuous-monthly-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/continuous/surekli_anomali_haber_aylik.csv"),
    )
    p.add_argument(
        "--events-news-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/tum_asiri_olaylar_haber_enriched.csv"),
    )
    p.add_argument(
        "--temp-forecast-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/best_meta_auto_combo_v15_consistency3/forecasts/temp_monthly_best_meta_to_2035.csv"),
    )
    p.add_argument(
        "--humidity-forecast-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/best_meta_auto_combo_v15_consistency3/forecasts/humidity_monthly_best_meta_to_2035.csv"),
    )
    p.add_argument(
        "--precip-forecast-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/best_meta_auto_combo_v15_consistency3/forecasts/precip_monthly_best_meta_to_2035.csv"),
    )
    p.add_argument(
        "--pressure-forecast-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/prophet_package/forecasts/pressure_monthly_prophet_to_2035.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/past_future_fusion"),
    )
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2035)
    p.add_argument("--news-score-threshold", type=float, default=0.50)
    p.add_argument("--top-annotations", type=int, default=8)
    return p.parse_args()


UNITS = {"temp": "C", "humidity": "%", "pressure": "hPa", "precip": "mm"}


def short_text(x: str, n: int = 72) -> str:
    s = str(x).strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def load_forecast(path: Path, variable: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "ds" not in df.columns:
        raise SystemExit(f"Forecast file missing ds: {path}")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["yhat"] = pd.to_numeric(df.get("yhat"), errors="coerce")
    df["yhat_lower"] = pd.to_numeric(df.get("yhat_lower"), errors="coerce")
    df["yhat_upper"] = pd.to_numeric(df.get("yhat_upper"), errors="coerce")
    if "is_forecast" not in df.columns:
        # fallback infer by missing actual
        if "actual" in df.columns:
            df["is_forecast"] = df["actual"].isna()
        else:
            df["is_forecast"] = False
    else:
        df["is_forecast"] = df["is_forecast"].astype(bool)
    df["variable"] = variable
    df = df.dropna(subset=["ds", "yhat"]).sort_values("ds").reset_index(drop=True)
    # Monthly regularization for yearly series.
    if df["ds"].dt.to_period("M").nunique() < max(24, len(df) // 2):
        # likely yearly
        full_idx = pd.date_range(df["ds"].min(), df["ds"].max(), freq="MS")
        tmp = df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper", "is_forecast"]].copy()
        tmp = tmp.reindex(full_idx).ffill()
        tmp.index.name = "ds"
        df = tmp.reset_index()
        df["variable"] = variable
    return df


def load_news_events(path: Path, threshold: float) -> pd.DataFrame:
    ev = pd.read_csv(path).copy()
    ev["variable"] = ev["variable"].astype(str).str.lower().str.strip()
    ev["start_time"] = pd.to_datetime(ev["start_time"], errors="coerce")
    ev["end_time"] = pd.to_datetime(ev["end_time"], errors="coerce")
    ev["center_time"] = ev["start_time"] + (ev["end_time"] - ev["start_time"]) / 2
    ev["peak_severity_score"] = pd.to_numeric(ev["peak_severity_score"], errors="coerce")
    ev["top_headline_match_score"] = pd.to_numeric(ev["top_headline_match_score"], errors="coerce")
    ev = ev.dropna(subset=["center_time", "variable", "peak_severity_score"]).copy()
    ev_news = ev[ev["top_headline"].notna()].copy()
    ev_news = ev_news[ev_news["top_headline_match_score"] >= float(threshold)].copy()
    return ev, ev_news


def load_continuous(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["variable"] = df["variable"].astype(str).str.lower().str.strip()
    for c in ["monthly_max_severity", "severity_ema", "news_max_score", "monthly_event_count"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    return df.dropna(subset=["month", "variable"]).copy()


def build_joined_monthly(continuous: pd.DataFrame, forecasts: dict[str, pd.DataFrame], end_year: int) -> pd.DataFrame:
    parts = []
    for var, fc in forecasts.items():
        c = continuous[continuous["variable"] == var].copy()
        if c.empty and fc.empty:
            continue
        c = c.sort_values("month")
        fc = fc.sort_values("ds")
        min_month = min(
            c["month"].min() if not c.empty else pd.Timestamp(f"{end_year}-01-01"),
            fc["ds"].min() if not fc.empty else pd.Timestamp(f"{end_year}-01-01"),
        )
        max_month = max(
            c["month"].max() if not c.empty else pd.Timestamp(f"{end_year}-12-01"),
            fc["ds"].max() if not fc.empty else pd.Timestamp(f"{end_year}-12-01"),
            pd.Timestamp(f"{end_year}-12-01"),
        )
        full_idx = pd.date_range(min_month, max_month, freq="MS")
        base = pd.DataFrame({"month": full_idx})
        j = base.merge(c, on="month", how="left")
        j = j.merge(
            fc[["ds", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]].rename(columns={"ds": "month"}),
            on="month",
            how="left",
        )
        j["variable"] = var
        # Keep anomaly fields as NaN outside observed coverage; fill explicit counters only.
        if "monthly_event_count" in j.columns:
            j["monthly_event_count"] = pd.to_numeric(j["monthly_event_count"], errors="coerce").fillna(0).astype(int)
        if "news_count" in j.columns:
            j["news_count"] = pd.to_numeric(j["news_count"], errors="coerce").fillna(0).astype(int)
        if "news_max_score" in j.columns:
            j["news_max_score"] = pd.to_numeric(j["news_max_score"], errors="coerce").fillna(0.0)
        if "news_strict" in j.columns:
            j["news_strict"] = pd.to_numeric(j["news_strict"], errors="coerce").fillna(0).astype(int)
        if "is_forecast" in j.columns:
            j["is_forecast"] = pd.Series(j["is_forecast"], dtype="boolean").fillna(False).astype(bool)
        parts.append(j)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["variable", "month"]).reset_index(drop=True)


def plot_variable(
    variable: str,
    joined: pd.DataFrame,
    news_events: pd.DataFrame,
    out_png: Path,
    start_year: int,
    end_year: int,
    top_ann: int,
) -> None:
    sub = joined[joined["variable"] == variable].copy()
    if sub.empty:
        return
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")
    sub = sub[(sub["month"] >= lo) & (sub["month"] <= hi)].copy()
    if sub.empty:
        return
    ne = news_events[(news_events["variable"] == variable)].copy()
    ne = ne[(ne["center_time"] >= lo) & (ne["center_time"] <= hi)].copy()

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.0, 1.6], hspace=0.15)

    # Panel 1: anomaly signal + news score.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sub["month"], sub["monthly_max_severity"], color="#9a9a9a", lw=1.0, alpha=0.6, label="monthly max severity")
    ax1.plot(sub["month"], sub["severity_ema"], color="#1f5c8d", lw=2.0, label="severity EMA")
    ax1.set_ylabel("Anomaly severity")
    ax1.grid(alpha=0.22)
    ax1b = ax1.twinx()
    ax1b.plot(sub["month"], sub["news_max_score"], color="#d1495b", lw=1.2, alpha=0.8, label="news max score")
    ax1b.set_ylabel("News score")
    ax1b.set_ylim(0, 1.02)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title(f"{variable} | Gecmis anomali + haber + gelecek tahmin (2035)")

    # Panel 2: forecast trajectory.
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    hist = sub[sub["is_forecast"] == False]  # noqa: E712
    fut = sub[sub["is_forecast"] == True]  # noqa: E712
    if not hist.empty:
        ax2.plot(hist["month"], hist["yhat"], color="#2f6f3e", lw=1.6, label="historical yhat")
    if not fut.empty:
        ax2.plot(fut["month"], fut["yhat"], color="#f39c12", lw=2.0, ls="--", label="forecast yhat")
        if fut["yhat_lower"].notna().any() and fut["yhat_upper"].notna().any():
            ax2.fill_between(
                fut["month"],
                fut["yhat_lower"],
                fut["yhat_upper"],
                color="#f39c12",
                alpha=0.18,
                label="forecast interval",
            )
        ax2.axvline(fut["month"].min(), color="#f39c12", lw=1.0, ls=":", alpha=0.8)
    ax2.set_ylabel(f"Forecast value ({UNITS.get(variable,'unit')})")
    ax2.grid(alpha=0.22)
    ax2.legend(loc="upper left")

    # Panel 3: top news annotations
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    if ne.empty:
        txt = "Bu degiskende haber eslesmesi yok."
    else:
        top = ne.sort_values(["top_headline_match_score", "peak_severity_score"], ascending=[False, False]).head(top_ann)
        lines = ["Top haber anotasyonlari:"]
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

    ax2.set_xlim(lo, hi)
    ax2.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=185)
    plt.close(fig)


def plot_overview(joined: pd.DataFrame, out_png: Path, start_year: int, end_year: int) -> None:
    vars_order = [v for v in ["temp", "humidity", "pressure", "precip"] if v in set(joined["variable"])]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False, constrained_layout=True)
    axes = axes.ravel()
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")

    for i, var in enumerate(vars_order):
        ax = axes[i]
        sub = joined[joined["variable"] == var].copy()
        sub = sub[(sub["month"] >= lo) & (sub["month"] <= hi)].copy()
        if sub.empty:
            ax.set_title(f"{var}: veri yok")
            continue
        hist = sub[sub["is_forecast"] == False]  # noqa: E712
        fut = sub[sub["is_forecast"] == True]  # noqa: E712
        # normalized yhat for comparability
        base = pd.to_numeric(hist["yhat"], errors="coerce") if not hist.empty else pd.to_numeric(sub["yhat"], errors="coerce")
        q95 = float(np.nanquantile(base, 0.95)) if len(base) else 1.0
        q95 = max(q95, 1e-6)
        sub["yhat_norm"] = (pd.to_numeric(sub["yhat"], errors="coerce") / q95).clip(0, None)
        hist = sub[sub["is_forecast"] == False]  # noqa: E712
        fut = sub[sub["is_forecast"] == True]  # noqa: E712

        ax.plot(sub["month"], sub["severity_ema"], color="#1f5c8d", lw=1.2, label="severity EMA")
        if not hist.empty:
            ax.plot(hist["month"], hist["yhat_norm"], color="#2f6f3e", lw=1.1, label="hist yhat (norm)")
        if not fut.empty:
            ax.plot(fut["month"], fut["yhat_norm"], color="#f39c12", lw=1.4, ls="--", label="forecast yhat (norm)")
            ax.axvline(fut["month"].min(), color="#f39c12", lw=0.9, ls=":", alpha=0.8)
        ax.set_title(var)
        ax.grid(alpha=0.22)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    for j in range(len(vars_order), 4):
        axes[j].axis("off")
    fig.suptitle("Gecmis + Gelecek (2035) Tek Dashboard: Anomali-Haber-Tahmin Fusion", fontsize=15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190)
    plt.close(fig)


def build_pdf(overview_png: Path, var_pngs: list[Path], out_pdf: Path) -> None:
    with PdfPages(out_pdf) as pdf:
        for p in [overview_png] + var_pngs:
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

    continuous = load_continuous(args.continuous_monthly_csv)
    _, news_events = load_news_events(args.events_news_csv, threshold=float(args.news_score_threshold))

    forecasts = {
        "temp": load_forecast(args.temp_forecast_csv, "temp"),
        "humidity": load_forecast(args.humidity_forecast_csv, "humidity"),
        "precip": load_forecast(args.precip_forecast_csv, "precip"),
        "pressure": load_forecast(args.pressure_forecast_csv, "pressure"),
    }

    joined = build_joined_monthly(continuous=continuous, forecasts=forecasts, end_year=int(args.end_year))
    if joined.empty:
        raise SystemExit("Joined monthly table is empty.")

    lo = pd.Timestamp(f"{args.start_year}-01-01")
    hi = pd.Timestamp(f"{args.end_year}-12-31")
    joined = joined[(joined["month"] >= lo) & (joined["month"] <= hi)].copy()

    overview_png = args.output_dir / "gecmis_gelecek_fusion_dashboard.png"
    plot_overview(joined=joined, out_png=overview_png, start_year=int(args.start_year), end_year=int(args.end_year))

    vars_order = [v for v in ["temp", "humidity", "pressure", "precip"] if v in set(joined["variable"])]
    var_pngs: list[Path] = []
    for var in vars_order:
        p = args.output_dir / f"gecmis_gelecek_fusion_{var}.png"
        plot_variable(
            variable=var,
            joined=joined,
            news_events=news_events,
            out_png=p,
            start_year=int(args.start_year),
            end_year=int(args.end_year),
            top_ann=int(args.top_annotations),
        )
        var_pngs.append(p)

    out_csv = args.output_dir / "gecmis_gelecek_fusion_aylik_birlesik.csv"
    joined.to_csv(out_csv, index=False)

    out_pdf = args.output_dir / "gecmis_gelecek_fusion_rapor.pdf"
    build_pdf(overview_png, var_pngs, out_pdf)

    summary = (
        joined.groupby("variable", as_index=False)
        .agg(
            rows=("month", "size"),
            forecast_months=("is_forecast", lambda s: int(pd.Series(s).astype(bool).sum())),
            anomaly_ema_median=("severity_ema", "median"),
            forecast_yhat_median=("yhat", "median"),
        )
        .sort_values("rows", ascending=False)
    )
    sum_csv = args.output_dir / "gecmis_gelecek_fusion_ozet.csv"
    summary.to_csv(sum_csv, index=False)

    md = args.output_dir / "gecmis_gelecek_fusion_rapor.md"
    lines = [
        "# Gecmis-Gelecek Fusion Raporu",
        "",
        f"- Anomali serisi: `{args.continuous_monthly_csv}`",
        f"- Haber olay dosyasi: `{args.events_news_csv}`",
        f"- Donem: {args.start_year}-{args.end_year}",
        "",
        "## Ciktilar",
        f"- Dashboard: `{overview_png}`",
        f"- PDF: `{out_pdf}`",
        f"- Aylik birlesik CSV: `{out_csv}`",
        f"- Ozet CSV: `{sum_csv}`",
        "",
        "## Degisken Ozet",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['variable']}: rows={int(r['rows'])}, forecast_months={int(r['forecast_months'])}, "
            f"median_anomaly_ema={float(r['anomaly_ema_median']):.3f}, median_forecast_yhat={float(r['forecast_yhat_median']):.3f}"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {overview_png}")
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {sum_csv}")
    print(f"Wrote: {md}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
