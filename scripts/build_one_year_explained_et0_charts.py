#!/usr/bin/env python3
"""Build one-year ET0 charts with a left-side explanation panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from et0_visual_style import render_et0_panel, style_axes, theme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one-year explained ET0 charts.")
    parser.add_argument(
        "--daily-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/tables/tarim_et0_daily_radiation_complete.csv"),
        help="Daily ET0 CSV.",
    )
    parser.add_argument(
        "--monthly-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/tables/tarim_et0_monthly_radiation_complete.csv"),
        help="Monthly ET0 CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/charts"),
        help="Output chart directory.",
    )
    parser.add_argument("--label", type=str, default="Tarimsal", help="Context label (e.g., Baraj / Tarimsal).")
    parser.add_argument("--prefix", type=str, default="tarim_et0", help="Filename prefix for outputs.")
    parser.add_argument("--year", type=int, default=2004, help="Target year.")
    return parser.parse_args()


def add_season_bands(ax: plt.Axes, year: int, colors: dict[str, str]) -> None:
    seasons = [
        ("Kış", f"{year}-01-01", f"{year}-03-01"),
        ("İlkbahar", f"{year}-03-01", f"{year}-06-01"),
        ("Yaz", f"{year}-06-01", f"{year}-09-01"),
        ("Sonbahar", f"{year}-09-01", f"{year}-12-01"),
        ("Kış", f"{year}-12-01", f"{year+1}-01-01"),
    ]
    for i, (label, start, end) in enumerate(seasons):
        alpha = 0.22 if i % 2 == 0 else 0.10
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=colors["season_fill"], alpha=alpha, zorder=0)
        center = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(center, 0.985, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=colors["muted"])


def make_daily_chart(df: pd.DataFrame, year: int, out_path: Path, label: str) -> None:
    colors = theme()
    year_df = df[df["date"].dt.year == year].copy()
    if year_df.empty:
        raise ValueError(f"No daily ET0 rows found for year={year}")
    year_df["et0_30d"] = year_df["et0_mm_day"].rolling(30, min_periods=10).mean()
    real_days = int((year_df["rs_data_source"] == "real_extracted").sum())
    synthetic_days = int((year_df["rs_data_source"] == "synthetic").sum())
    peak_row = year_df.loc[year_df["et0_mm_day"].idxmax()]
    low_row = year_df.loc[year_df["et0_mm_day"].idxmin()]
    stats_lines = [
        f"ortalama ET0: {year_df['et0_mm_day'].mean():.2f}",
        f"zirve gün: {peak_row['date']:%d %b}",
        f"zirve ET0: {peak_row['et0_mm_day']:.2f}",
        f"dip ET0: {low_row['et0_mm_day']:.2f}",
    ]

    fig = plt.figure(figsize=(16.2, 9.4), facecolor=colors["fig_bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 2.18], wspace=0.06)

    ax_text = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    render_et0_panel(
        ax_text,
        context_title=f"Gunluk {label} ET0 | {year}",
        context_lines=[
            "Zaman adımı: günlük seri",
            "Amaç: mevsim içi ET0 ritmini görmek",
        ],
        assumption_lines=[
            "Tmean = (Tmax + Tmin) / 2 → tutarlı sıcaklık özeti.",
            "Delta = f(Tmean) → fiziksel olarak doğru eğim.",
            "G = 0 → günlük ET0 için standart kabul.",
            "u2 = 2.0 m/s → eksik rüzgar için fallback.",
            "Rs = radiation CSV → net radyasyon girdisi.",
        ],
        summary_lines=stats_lines,
        source_lines=[
            f"gerçek radyasyon: {real_days} gün",
            f"sentetik radyasyon: {synthetic_days} gün",
        ],
    )

    style_axes(ax_plot, colors)
    add_season_bands(ax_plot, year, colors)
    ax_plot.plot(year_df["date"], year_df["et0_mm_day"], color=colors["daily"], linewidth=0.9, alpha=0.28, label="Günlük ET0")
    ax_plot.plot(year_df["date"], year_df["et0_30d"], color=colors["daily_ma"], linewidth=2.3, label="30 gün ortalama")
    ax_plot.scatter(peak_row["date"], peak_row["et0_mm_day"], s=46, color=colors["accent"], zorder=4, label="Zirve gün")
    ax_plot.scatter(low_row["date"], low_row["et0_mm_day"], s=42, color=colors["real_marker"], zorder=4, label="Dip gün")
    ax_plot.annotate(
        f"Zirve {peak_row['date']:%d %b}\n{peak_row['et0_mm_day']:.2f} mm/gün",
        xy=(peak_row["date"], peak_row["et0_mm_day"]),
        xytext=(18, 12),
        textcoords="offset points",
        fontsize=9.2,
        color=colors["text"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff7ec", edgecolor=colors["panel_edge"]),
        arrowprops=dict(arrowstyle="-", color=colors["accent"], lw=1.1),
    )
    real_mask = year_df["rs_data_source"].eq("real_extracted")
    if real_mask.any():
        base_y = float(year_df["et0_mm_day"].min()) - 0.16
        ax_plot.scatter(
            year_df.loc[real_mask, "date"],
            np.full(real_mask.sum(), base_y),
            s=10,
            color=colors["real_marker"],
            alpha=0.7,
            label="Gerçek radyasyon günleri",
        )
    ax_plot.set_title(f"Günlük {label} ET0 - {year}", fontsize=16, color=colors["text"], pad=14)
    ax_plot.set_xlabel("Tarih", fontsize=11, color=colors["text"])
    ax_plot.set_ylabel("ET0 (mm/gün)", fontsize=11, color=colors["text"])
    ax_plot.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_plot.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_plot.set_xlim(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
    ax_plot.text(
        0.985,
        0.04,
        f"Gerçek radyasyon payı: {real_days / max(real_days + synthetic_days, 1):.1%}",
        transform=ax_plot.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.4,
        color=colors["muted"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#fff9f0", edgecolor=colors["panel_edge"]),
    )
    ax_plot.legend(loc="upper right", frameon=True, facecolor="#fff9f0", edgecolor=colors["panel_edge"])

    ymin = min(base_y if real_mask.any() else year_df["et0_mm_day"].min(), year_df["et0_mm_day"].min()) - 0.08
    ymax = year_df["et0_mm_day"].max() * 1.10
    ax_plot.set_ylim(ymin, ymax)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.94, bottom=0.08, wspace=0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_monthly_chart(df: pd.DataFrame, year: int, out_path: Path, label: str) -> None:
    colors = theme()
    year_df = df[df["date"].dt.year == year].copy().reset_index(drop=True)
    if year_df.empty:
        raise ValueError(f"No monthly ET0 rows found for year={year}")
    real_days = int(year_df["real_extracted_days"].sum())
    synthetic_days = int(year_df["synthetic_days"].sum())
    peak_month = year_df.loc[year_df["et0_mm_month"].idxmax()]
    low_month = year_df.loc[year_df["et0_mm_month"].idxmin()]
    stats_lines = [
        f"yıllık toplam: {year_df['et0_mm_month'].sum():.1f}",
        f"zirve ay: {peak_month['date']:%b}",
        f"zirve ET0: {peak_month['et0_mm_month']:.1f}",
        f"ortalama Rs: {year_df['rs_mj_m2_day'].mean():.1f}",
    ]

    fig = plt.figure(figsize=(16.2, 9.4), facecolor=colors["fig_bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 2.18], wspace=0.06)

    ax_text = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    render_et0_panel(
        ax_text,
        context_title=f"Aylik {label} ET0 | {year}",
        context_lines=[
            "Zaman adımı: aylık toplamlama",
            "Amaç: tepe ay ve düşük ay desenini görmek",
        ],
        assumption_lines=[
            "Günlük ET0 önce hesaplandı, sonra aya toplandı.",
            "G = 0 → günlük denklem aynen korundu.",
            "Rs dosyadan geldi → ET0 ile birlikte okunur.",
            "u2 = 2.0 m/s → tüm aylar için aynı fallback.",
            "Grafik ET0 ile radyasyonu birlikte gösterir.",
        ],
        summary_lines=stats_lines,
        source_lines=[
            f"gerçek radyasyon: {real_days} gün",
            f"sentetik radyasyon: {synthetic_days} gün",
        ],
    )

    style_axes(ax_plot, colors)
    month_labels = pd.to_datetime(year_df["date"]).dt.strftime("%b")
    x = np.arange(len(year_df))
    bars = ax_plot.bar(x, year_df["et0_mm_month"], color=colors["monthly_bar"], width=0.72, edgecolor="none", label="Aylık ET0", zorder=2)
    ax_plot.plot(x, year_df["et0_mm_month"], color=colors["monthly_line"], linewidth=2.2, marker="o", markersize=5, label="Aylık desen", zorder=3)
    ax_plot.scatter(int(peak_month.name), peak_month["et0_mm_month"], s=46, color=colors["accent"], zorder=4)
    ax_plot.annotate(
        f"Zirve {peak_month['date']:%b}\n{peak_month['et0_mm_month']:.1f} mm/ay",
        xy=(int(peak_month.name), peak_month["et0_mm_month"]),
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        fontsize=9.2,
        color=colors["text"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff7ec", edgecolor=colors["panel_edge"]),
        arrowprops=dict(arrowstyle="-", color=colors["accent"], lw=1.1),
    )
    ax_aux = ax_plot.twinx()
    ax_aux.plot(x, year_df["rs_mj_m2_day"], color=colors["monthly_aux"], linewidth=1.8, linestyle=":", marker="s", markersize=4, label="Rs")
    ax_aux.set_ylabel("Rs (MJ/m²/gün)", fontsize=10.5, color=colors["monthly_aux"])
    ax_aux.tick_params(axis="y", colors=colors["monthly_aux"])
    ax_aux.spines["top"].set_visible(False)
    ax_aux.spines["left"].set_visible(False)
    ax_aux.spines["right"].set_color(colors["spine"])
    ax_plot.set_xticks(x, month_labels)
    ax_plot.set_title(f"Aylık {label} ET0 - {year}", fontsize=16, color=colors["text"], pad=14)
    ax_plot.set_xlabel("Ay", fontsize=11, color=colors["text"])
    ax_plot.set_ylabel("ET0 (mm/ay)", fontsize=11, color=colors["text"])
    ax_plot.text(
        0.985,
        0.04,
        f"En düşük ay: {low_month['date']:%b} | {low_month['et0_mm_month']:.1f} mm/ay",
        transform=ax_plot.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        color=colors["muted"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#fff9f0", edgecolor=colors["panel_edge"]),
    )
    handles1, labels1 = ax_plot.get_legend_handles_labels()
    handles2, labels2 = ax_aux.get_legend_handles_labels()
    ax_plot.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True, facecolor="#fff9f0", edgecolor=colors["panel_edge"])

    fig.subplots_adjust(left=0.03, right=0.985, top=0.94, bottom=0.08, wspace=0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    daily = pd.read_csv(args.daily_csv, parse_dates=["date"])
    monthly = pd.read_csv(args.monthly_csv, parse_dates=["date"])

    daily_chart = args.out_dir / f"{args.prefix}_daily_explained_{args.year}.png"
    monthly_chart = args.out_dir / f"{args.prefix}_monthly_explained_{args.year}.png"

    make_daily_chart(daily, args.year, daily_chart, args.label)
    make_monthly_chart(monthly, args.year, monthly_chart, args.label)

    print(f"Wrote: {daily_chart}")
    print(f"Wrote: {monthly_chart}")


if __name__ == "__main__":
    main()
