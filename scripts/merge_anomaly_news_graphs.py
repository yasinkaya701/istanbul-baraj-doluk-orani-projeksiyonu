#!/usr/bin/env python3
"""Merge anomaly timelines with matched newspaper/news headlines on charts."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

# Matplotlib cache in writable temp dir.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "merge_anomaly_news_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge anomaly charts with news-headline matched events.")
    p.add_argument(
        "--anomaly-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/quant_all_visuals_package/anomalies"),
    )
    p.add_argument(
        "--events-news-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/tum_asiri_olaylar_haber_enriched.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news/charts"),
    )
    p.add_argument("--max-labels-per-variable", type=int, default=10)
    p.add_argument("--start-year", type=int, default=1980)
    p.add_argument("--end-year", type=int, default=2026)
    return p.parse_args()


def clip_score(x: pd.Series) -> pd.Series:
    z = pd.to_numeric(x, errors="coerce")
    return z.clip(lower=-25, upper=25)


def infer_tolerance_days(ds: pd.Series) -> int:
    d = pd.to_datetime(ds, errors="coerce").dropna().sort_values()
    if len(d) < 3:
        return 120
    med = np.median(np.diff(d.values).astype("timedelta64[D]").astype(float))
    if med >= 300:
        return 370
    if med >= 25:
        return 70
    if med >= 6:
        return 20
    return 5


def short_headline(h: str, max_len: int = 64) -> str:
    s = str(h).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def load_anomalies(anomaly_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(anomaly_dir.glob("*_anomalies_to_*.csv")):
        df = pd.read_csv(p).copy()
        if "variable" not in df.columns or "ds" not in df.columns:
            continue
        var = str(df["variable"].iloc[0]).strip().lower()
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["robust_z_plot"] = clip_score(df.get("robust_zscore"))
        df["zscore_plot"] = clip_score(df.get("zscore"))
        df["is_anomaly"] = df.get("is_anomaly", False).astype(bool)
        df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
        out[var] = df
    return out


def attach_nearest_anomaly(events: pd.DataFrame, anomalies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, ev in events.iterrows():
        var = str(ev["variable"]).strip().lower()
        if var not in anomalies:
            continue
        a = anomalies[var]
        if a.empty:
            continue
        center = pd.Timestamp(ev["center_time"])
        tol = infer_tolerance_days(a["ds"])
        dd = (a["ds"] - center).abs().dt.days
        i = int(dd.argmin())
        if int(dd.iloc[i]) > tol:
            continue
        r = a.iloc[i]
        rows.append(
            {
                **ev.to_dict(),
                "anomaly_ds": r["ds"],
                "anomaly_day_diff": int(dd.iloc[i]),
                "anomaly_zscore": float(r.get("zscore", np.nan)),
                "anomaly_robust_zscore": float(r.get("robust_zscore", np.nan)),
                "anomaly_robust_z_plot": float(r.get("robust_z_plot", np.nan)),
                "anomaly_type_tr": str(r.get("anomaly_type_tr", "")),
            }
        )
    return pd.DataFrame(rows)


def plot_variable(
    var: str,
    a: pd.DataFrame,
    matched: pd.DataFrame,
    out_path: Path,
    max_labels: int,
    start_year: int,
    end_year: int,
) -> None:
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")
    aa = a[(a["ds"] >= lo) & (a["ds"] <= hi)].copy()
    mm = matched[(matched["variable"] == var)].copy()
    if aa.empty:
        return

    plt.figure(figsize=(16, 6))
    plt.plot(aa["ds"], aa["robust_z_plot"], color="#255f85", lw=1.7, label="Robust Z (clipped)")
    an = aa[aa["is_anomaly"]]
    if not an.empty:
        plt.scatter(an["ds"], an["robust_z_plot"], color="#8db6cd", s=22, alpha=0.7, label="Anomaly points")

    if not mm.empty:
        mm = mm.sort_values("top_headline_match_score", ascending=False)
        plt.scatter(
            mm["anomaly_ds"],
            mm["anomaly_robust_z_plot"],
            color="#d1495b",
            s=70,
            alpha=0.9,
            label="News-matched events",
            zorder=5,
        )
        # Label top-k only to keep readability.
        for _, r in mm.head(max_labels).iterrows():
            text = f"{pd.Timestamp(r['top_headline_date']).date()} | {r['top_headline_source']} | {short_headline(r['top_headline'])}"
            plt.annotate(
                text,
                xy=(r["anomaly_ds"], r["anomaly_robust_z_plot"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#d1495b", "alpha": 0.75},
                arrowprops={"arrowstyle": "-", "color": "#d1495b", "lw": 0.6},
            )

    plt.axhline(0, color="#666666", lw=0.8, alpha=0.8)
    plt.axhline(3, color="#999999", lw=0.6, ls="--", alpha=0.6)
    plt.axhline(-3, color="#999999", lw=0.6, ls="--", alpha=0.6)
    plt.title(f"Anomali + Haber Basliklari | {var}")
    plt.ylabel("Robust Z (clipped to [-25, 25])")
    plt.xlabel("Tarih")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_dashboard(anomalies: dict[str, pd.DataFrame], matched: pd.DataFrame, out_path: Path, start_year: int, end_year: int) -> None:
    vars_order = [v for v in ["temp", "humidity", "pressure", "precip"] if v in anomalies]
    if not vars_order:
        return
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    axes = axes.ravel()

    for i, var in enumerate(vars_order):
        ax = axes[i]
        lo = pd.Timestamp(f"{start_year}-01-01")
        hi = pd.Timestamp(f"{end_year}-12-31")
        aa = anomalies[var]
        aa = aa[(aa["ds"] >= lo) & (aa["ds"] <= hi)].copy()
        mm = matched[matched["variable"] == var].copy()
        if aa.empty:
            ax.set_title(f"{var}: veri yok")
            continue
        ax.plot(aa["ds"], aa["robust_z_plot"], color="#255f85", lw=1.2)
        an = aa[aa["is_anomaly"]]
        if not an.empty:
            ax.scatter(an["ds"], an["robust_z_plot"], color="#8db6cd", s=12, alpha=0.6)
        if not mm.empty:
            ax.scatter(mm["anomaly_ds"], mm["anomaly_robust_z_plot"], color="#d1495b", s=40, alpha=0.9)
        ax.axhline(0, color="#666666", lw=0.6, alpha=0.8)
        ax.set_title(f"{var} | matched news={len(mm)}")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for j in range(len(vars_order), 4):
        axes[j].axis("off")

    fig.suptitle("Anomali Zaman Serileri + Haber Eslesen Olaylar", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_event_severity_variable(
    var: str,
    events_all: pd.DataFrame,
    events_news: pd.DataFrame,
    out_path: Path,
    max_labels: int,
    start_year: int,
    end_year: int,
) -> None:
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")
    ea = events_all[(events_all["variable"] == var)].copy()
    ea = ea[(ea["center_time"] >= lo) & (ea["center_time"] <= hi)].copy()
    en = events_news[(events_news["variable"] == var)].copy()
    en = en[(en["center_time"] >= lo) & (en["center_time"] <= hi)].copy()
    if ea.empty:
        return

    plt.figure(figsize=(16, 6))
    plt.scatter(
        ea["center_time"],
        ea["peak_severity_score"],
        color="#8f8f8f",
        s=26,
        alpha=0.45,
        label="Tum anomali olaylari",
    )
    # Running median line for trend readability.
    e2 = ea.sort_values("center_time").copy()
    e2["sev_med"] = e2["peak_severity_score"].rolling(15, min_periods=5).median()
    plt.plot(e2["center_time"], e2["sev_med"], color="#255f85", lw=1.6, label="Siddet rolling median")

    if not en.empty:
        plt.scatter(
            en["center_time"],
            en["peak_severity_score"],
            color="#d1495b",
            s=80,
            alpha=0.9,
            label="Haber eslesmis olay",
            zorder=5,
        )
        en = en.sort_values(["top_headline_match_score", "peak_severity_score"], ascending=[False, False])
        for _, r in en.head(max_labels).iterrows():
            text = f"{pd.Timestamp(r['top_headline_date']).date()} | {r['top_headline_source']} | {short_headline(r['top_headline'])}"
            plt.annotate(
                text,
                xy=(r["center_time"], r["peak_severity_score"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#d1495b", "alpha": 0.75},
                arrowprops={"arrowstyle": "-", "color": "#d1495b", "lw": 0.6},
            )

    plt.title(f"Olay Siddeti + Haber Basliklari | {var}")
    plt.ylabel("Peak severity score")
    plt.xlabel("Tarih")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_event_severity_dashboard(
    events_all: pd.DataFrame,
    events_news: pd.DataFrame,
    out_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    vars_order = [v for v in ["temp", "humidity", "pressure", "precip"] if v in set(events_all["variable"].astype(str))]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    axes = axes.ravel()
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")

    for i, var in enumerate(vars_order):
        ax = axes[i]
        ea = events_all[(events_all["variable"] == var)].copy()
        ea = ea[(ea["center_time"] >= lo) & (ea["center_time"] <= hi)].copy()
        en = events_news[(events_news["variable"] == var)].copy()
        en = en[(en["center_time"] >= lo) & (en["center_time"] <= hi)].copy()
        if ea.empty:
            ax.set_title(f"{var}: veri yok")
            continue
        ax.scatter(ea["center_time"], ea["peak_severity_score"], color="#8f8f8f", s=14, alpha=0.45)
        e2 = ea.sort_values("center_time").copy()
        e2["sev_med"] = e2["peak_severity_score"].rolling(15, min_periods=5).median()
        ax.plot(e2["center_time"], e2["sev_med"], color="#255f85", lw=1.2)
        if not en.empty:
            ax.scatter(en["center_time"], en["peak_severity_score"], color="#d1495b", s=34, alpha=0.9)
        ax.set_title(f"{var} | news-matched={len(en)}")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for j in range(len(vars_order), 4):
        axes[j].axis("off")
    fig.suptitle("Olay Siddeti Anomalileri + Haber Eslesmeleri", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anomalies = load_anomalies(args.anomaly_dir)
    if not anomalies:
        raise SystemExit(f"No anomaly csv found in: {args.anomaly_dir}")

    ev = pd.read_csv(args.events_news_csv).copy()
    ev["start_time"] = pd.to_datetime(ev["start_time"], errors="coerce")
    ev["end_time"] = pd.to_datetime(ev["end_time"], errors="coerce")
    ev["center_time"] = ev["start_time"] + (ev["end_time"] - ev["start_time"]) / 2
    ev["variable"] = ev["variable"].astype(str).str.lower().str.strip()
    ev = ev.dropna(subset=["center_time"]).copy()
    ev_news = ev[ev["top_headline"].notna()].copy()
    if ev_news.empty:
        raise SystemExit("No events with news headline found in events-news csv.")

    matched = attach_nearest_anomaly(ev_news, anomalies)
    if matched.empty:
        raise SystemExit("No event-news rows could be aligned to anomaly timestamps.")

    # Per-variable charts + dashboard.
    for var, a in anomalies.items():
        out_png = args.output_dir / f"anomali_haber_birlesik_{var}.png"
        plot_variable(
            var=var,
            a=a,
            matched=matched,
            out_path=out_png,
            max_labels=int(args.max_labels_per_variable),
            start_year=int(args.start_year),
            end_year=int(args.end_year),
        )

    dash_png = args.output_dir / "anomali_haber_birlesik_dashboard.png"
    plot_dashboard(
        anomalies=anomalies,
        matched=matched,
        out_path=dash_png,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )

    # Event-severity based merged charts (full anomaly-event coverage).
    sev_dir = args.output_dir / "event_severity"
    sev_dir.mkdir(parents=True, exist_ok=True)
    for var in sorted(set(ev["variable"].astype(str))):
        out_png = sev_dir / f"olay_siddet_haber_birlesik_{var}.png"
        plot_event_severity_variable(
            var=var,
            events_all=ev,
            events_news=ev_news,
            out_path=out_png,
            max_labels=int(args.max_labels_per_variable),
            start_year=int(args.start_year),
            end_year=int(args.end_year),
        )
    sev_dash = sev_dir / "olay_siddet_haber_birlesik_dashboard.png"
    plot_event_severity_dashboard(
        events_all=ev,
        events_news=ev_news,
        out_path=sev_dash,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )

    out_csv = args.output_dir / "anomali_haber_birlesik_nokta_tablosu.csv"
    matched = matched.sort_values(["variable", "anomaly_ds", "top_headline_match_score"], ascending=[True, True, False])
    matched.to_csv(out_csv, index=False)

    summary = (
        matched.groupby("variable", as_index=False)
        .agg(
            matched_points=("event_id", "size"),
            uniq_events=("event_id", pd.Series.nunique),
            mean_match_score=("top_headline_match_score", "mean"),
            median_day_diff=("anomaly_day_diff", "median"),
        )
        .sort_values("matched_points", ascending=False)
    )
    out_sum = args.output_dir / "anomali_haber_birlesik_ozet.csv"
    summary.to_csv(out_sum, index=False)

    out_md = args.output_dir / "anomali_haber_birlesik_rapor.md"
    lines = [
        "# Anomali + Haber Birlesik Grafik Raporu",
        "",
        f"- Anomali kaynak klasoru: `{args.anomaly_dir}`",
        f"- Olay+haber dosyasi: `{args.events_news_csv}`",
        f"- Toplam eslesen nokta: **{len(matched)}**",
        f"- Eslesen benzersiz olay: **{matched['event_id'].nunique()}**",
        "",
        "## Degisken Ozet",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['variable']}: nokta={int(r['matched_points'])}, olay={int(r['uniq_events'])}, "
            f"ort_match_skoru={float(r['mean_match_score']):.3f}, medyan_gun_farki={float(r['median_day_diff']):.1f}"
        )
    lines += [
        "",
        "## Uretilen Grafikler",
        f"- Dashboard: `{dash_png}`",
        f"- Event-severity dashboard: `{sev_dash}`",
    ]
    for var in sorted(anomalies.keys()):
        lines.append(f"- {var}: `{args.output_dir / f'anomali_haber_birlesik_{var}.png'}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_sum}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {dash_png}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
