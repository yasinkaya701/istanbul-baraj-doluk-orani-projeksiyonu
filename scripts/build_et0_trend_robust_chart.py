#!/usr/bin/env python3
"""Create an improved ET0 trend chart with a left-side formula panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess

from et0_visual_style import render_et0_panel, style_axes, theme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build robust ET0 trend chart.")
    parser.add_argument(
        "--yearly-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/tables/tarim_et0_yearly_radiation_complete.csv"),
        help="Yearly ET0 CSV.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/charts/tarim_et0_yearly_trend_robust_explained.png"),
        help="Output chart path.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    colors = theme()

    df = pd.read_csv(args.yearly_csv, parse_dates=["date"]).sort_values("year").reset_index(drop=True)
    df["et0_mm_year"] = pd.to_numeric(df["et0_mm_year"], errors="coerce")
    df = df.dropna(subset=["year", "et0_mm_year"]).copy()
    if df.empty:
        raise ValueError(f"No yearly ET0 rows found in {args.yearly_csv}")

    x = df["year"].to_numpy(dtype=float)
    y = df["et0_mm_year"].to_numpy(dtype=float)

    robust = TheilSenRegressor(random_state=42)
    robust.fit(x.reshape(-1, 1), y)
    robust_line = robust.predict(x.reshape(-1, 1))
    slope_decade = float(robust.coef_[0] * 10.0)

    smooth = lowess(y, x, frac=0.28, return_sorted=False)
    max_row = df.loc[df["et0_mm_year"].idxmax()]
    min_row = df.loc[df["et0_mm_year"].idxmin()]
    obs_mask = df["real_extracted_days"].fillna(0).gt(0)
    start_row = df.iloc[0]
    end_row = df.iloc[-1]

    fig = plt.figure(figsize=(16.2, 9.3), facecolor=colors["fig_bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 2.18], wspace=0.06)
    ax_text = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    direction = "artan" if slope_decade > 0 else "azalan"
    render_et0_panel(
        ax_text,
        context_title=f"Genel Trend | {int(start_row['year'])}-{int(end_row['year'])}",
        context_lines=[
            "Zaman adimi: yillik toplam ET0",
            "Amac: uzun donem yonu gormek",
        ],
        assumption_lines=[
            "Theil-Sen -> asiri yillara daha dayanikli.",
            "LOWESS -> lineer olmayan deseni gosterir.",
            "R kare ve p grafikten kaldirildi.",
            "G = 0, Delta = f(Tmean), u2 = 2.0 m/s sabit tutuldu.",
            "Gercek radiation gunu olan yillar ayri katmanda.",
        ],
        summary_lines=[
            f"genel yon: {direction}",
            f"robust egim: {slope_decade:+.1f} mm/10y",
            f"maksimum yil: {int(max_row['year'])} | {max_row['et0_mm_year']:.1f} mm/yil",
            f"minimum yil: {int(min_row['year'])} | {min_row['et0_mm_year']:.1f} mm/yil",
        ],
        model_lines=[
            "trend modeli: Theil-Sen",
            "yumusatma: LOWESS",
        ],
        source_lines=[
            f"gercek radiation gunu olan yil sayisi: {int(obs_mask.sum())}",
        ],
    )

    style_axes(ax_plot, colors)
    if obs_mask.any():
        obs_start = int(df.loc[obs_mask, "year"].min())
        obs_end = int(df.loc[obs_mask, "year"].max())
        ax_plot.axvspan(obs_start - 0.5, obs_end + 0.5, color=colors["card_blue"], alpha=0.60, zorder=0)
        ax_plot.text((obs_start + obs_end) / 2, 0.985, "Gercek radiation gunleri olan yillar", transform=ax_plot.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=colors["muted"])
    ax_plot.bar(
        df["year"],
        df["et0_mm_year"],
        width=0.82,
        color=colors["card_gold"],
        edgecolor=colors["accent_soft"],
        linewidth=0.4,
        alpha=0.95,
        label="Yillik ET0",
    )
    ax_plot.plot(df["year"], smooth, color=colors["accent_2"], linewidth=2.8, label="LOWESS desen")
    ax_plot.plot(df["year"], robust_line, color=colors["accent"], linewidth=2.4, linestyle="--", label="Theil-Sen robust trend")
    ax_plot.scatter(max_row["year"], max_row["et0_mm_year"], s=54, color=colors["accent"], zorder=4)
    ax_plot.scatter(min_row["year"], min_row["et0_mm_year"], s=54, color=colors["accent_2"], zorder=4)
    ax_plot.annotate(
        f"Maksimum\n{int(max_row['year'])}: {max_row['et0_mm_year']:.1f}",
        xy=(max_row["year"], max_row["et0_mm_year"]),
        xytext=(18, 14),
        textcoords="offset points",
        fontsize=9.2,
        color=colors["text"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff7ec", edgecolor=colors["panel_edge"]),
        arrowprops=dict(arrowstyle="-", color=colors["accent"], lw=1.1),
    )
    ax_plot.annotate(
        f"Minimum\n{int(min_row['year'])}: {min_row['et0_mm_year']:.1f}",
        xy=(min_row["year"], min_row["et0_mm_year"]),
        xytext=(-68, -6),
        textcoords="offset points",
        fontsize=9.2,
        color=colors["text"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#eef7f5", edgecolor=colors["panel_edge"]),
        arrowprops=dict(arrowstyle="-", color=colors["accent_2"], lw=1.1),
    )
    change = float(end_row["et0_mm_year"] - start_row["et0_mm_year"])
    ax_plot.text(
        0.985,
        0.04,
        f"Donem farki: {change:+.1f} mm/yil",
        transform=ax_plot.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.3,
        color=colors["muted"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff9f0", edgecolor=colors["panel_edge"]),
    )
    ax_plot.set_title("Tarimsal ET0 Genel Trend", fontsize=16.5, color=colors["text"], pad=14)
    ax_plot.set_xlabel("Yil", fontsize=11, color=colors["text"])
    ax_plot.set_ylabel("ET0 (mm/yil)", fontsize=11, color=colors["text"])
    ax_aux = ax_plot.twinx()
    ax_aux.bar(
        df["year"],
        df["real_extracted_days"].fillna(0),
        width=0.32,
        color="#7ea8a3",
        alpha=0.7,
        label="Gercek radiation gunu",
        zorder=1,
    )
    ax_aux.set_ylabel("Gercek radiation gunu", fontsize=10.5, color="#7ea8a3")
    ax_aux.tick_params(axis="y", colors="#7ea8a3")
    ax_aux.spines["top"].set_visible(False)
    ax_aux.spines["left"].set_visible(False)
    ax_aux.spines["right"].set_color(colors["spine"])
    handles1, labels1 = ax_plot.get_legend_handles_labels()
    handles2, labels2 = ax_aux.get_legend_handles_labels()
    ax_plot.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True, facecolor="#fff9f0", edgecolor=colors["panel_edge"])

    fig.subplots_adjust(left=0.03, right=0.985, top=0.94, bottom=0.08, wspace=0.06)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Wrote: {args.out_png}")


if __name__ == "__main__":
    main()
