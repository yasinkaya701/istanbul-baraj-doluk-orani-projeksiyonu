#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly/yearly/10y ET0 charts.")
    parser.add_argument("--monthly-csv", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/tables/tarim_et0_monthly_radiation_complete.csv"))
    parser.add_argument("--yearly-csv", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/tables/tarim_et0_yearly_radiation_complete.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_real_radiation/charts"))
    parser.add_argument("--label", type=str, default="ET0", help="Context label for titles.")
    parser.add_argument("--prefix", type=str, default="tarim_et0", help="Filename prefix for outputs.")
    return parser.parse_args()


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")


def add_note(ax: plt.Axes, text: str, loc=(0.02, 0.98)) -> None:
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, ha="left", va="top", fontsize=10.5, color="#333333", bbox=dict(boxstyle="round,pad=0.3", facecolor="#faf6ef", edgecolor="#d2c2aa"))


def build_monthly(monthly: pd.DataFrame, out_png: Path, label: str) -> None:
    df = monthly.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df["ma_60m"] = df["et0_mm_month"].rolling(60, min_periods=24).mean()
    peak = df.loc[df["et0_mm_month"].idxmax()]

    fig, ax = plt.subplots(figsize=(14.5, 7.0), facecolor="white")
    style(ax)
    ax.plot(df["date"], df["et0_mm_month"], color="#d3b287", linewidth=1.4, alpha=0.85, label="Aylik ET0")
    ax.plot(df["date"], df["ma_60m"], color="#7c3419", linewidth=2.8, label="5 yillik ortalama (60 ay)")
    ax.scatter(peak["date"], peak["et0_mm_month"], color="#ad4f34", s=46, zorder=4)
    ax.annotate(f"Tepe ay\n{peak['date']:%Y-%m} | {peak['et0_mm_month']:.1f} mm/ay", xy=(peak["date"], peak["et0_mm_month"]), xytext=(18, 10), textcoords="offset points", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8f0", edgecolor="#d2c2aa"), arrowprops=dict(arrowstyle="-", color="#ad4f34"))
    ax.set_title(f"Aylik {label} Serisi (1995-2004)", fontsize=19, pad=16)
    ax.set_ylabel("ET0 (mm/ay)", fontsize=12)
    ax.set_xlabel("Tarih", fontsize=12)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    add_note(ax, "5 yillik ortalama = 60 aylik hareketli ortalama\nIslevi: mevsimsel ve kisa donem oynakligi bastirir, ana egilimi gosterir.")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#d2c2aa")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_yearly(yearly: pd.DataFrame, out_png: Path, label: str) -> None:
    df = yearly.copy().sort_values("year")
    df["ma_5y"] = df["et0_mm_year"].rolling(5, min_periods=3).mean()
    coef = np.polyfit(df["year"], df["et0_mm_year"], 1)
    trend = np.polyval(coef, df["year"])

    fig, ax = plt.subplots(figsize=(14.5, 7.0), facecolor="white")
    style(ax)
    ax.bar(df["year"], df["et0_mm_year"], width=0.72, color="#eadfcf", edgecolor="#ccb89c", label="Yillik ET0")
    ax.plot(df["year"], df["ma_5y"], color="#1f5c5b", linewidth=3.0, marker="o", label="5 yillik ortalama")
    ax.plot(df["year"], trend, color="#b14d3b", linewidth=2.0, linestyle="--", label="Dogrusal yon")
    ax.set_title(f"Yillik {label} ve 5 Yillik Ortalama", fontsize=19, pad=16)
    ax.set_ylabel("ET0 (mm/yil)", fontsize=12)
    ax.set_xlabel("Yil", fontsize=12)
    add_note(ax, "5 yillik ortalama yillik serideki sicrama ve dususleri yumusatir.\nIslevi: ana yonu tek yillik sapmalardan ayirmaktir.")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#d2c2aa")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_ten_year_summary(monthly: pd.DataFrame, out_png: Path, label: str) -> None:
    df = monthly.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    clim = df.groupby("month", as_index=False)["et0_mm_month"].mean()
    month_names = ["Oca", "Sub", "Mar", "Nis", "May", "Haz", "Tem", "Agu", "Eyl", "Eki", "Kas", "Ara"]
    peak = clim.loc[clim["et0_mm_month"].idxmax()]

    fig, ax = plt.subplots(figsize=(14.5, 7.0), facecolor="white")
    style(ax)
    ax.bar(clim["month"], clim["et0_mm_month"], width=0.72, color="#d9b382", edgecolor="#c29457", label="1995-2004 ortalama")
    ax.plot(clim["month"], clim["et0_mm_month"], color="#7c3419", linewidth=2.4, marker="o")
    ax.scatter(peak["month"], peak["et0_mm_month"], s=50, color="#ad4f34", zorder=4)
    ax.annotate(f"En yuksek ay\n{month_names[int(peak['month'])-1]} | {peak['et0_mm_month']:.1f} mm/ay", xy=(peak["month"], peak["et0_mm_month"]), xytext=(10, 14), textcoords="offset points", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8f0", edgecolor="#d2c2aa"), arrowprops=dict(arrowstyle="-", color="#ad4f34"))
    ax.set_xticks(clim["month"], month_names)
    ax.set_title(f"10 Yillik Ozet: Ortalama Aylik {label} Deseni (1995-2004)", fontsize=19, pad=16)
    ax.set_ylabel("ET0 (mm/ay)", fontsize=12)
    ax.set_xlabel("Ay", fontsize=12)
    add_note(ax, "Bu grafik zaten 10 yillik toplulastirilmis ozet oldugu icin\nayrica 5 yillik ortalama eklenmedi.")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#d2c2aa")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    monthly = pd.read_csv(args.monthly_csv, parse_dates=["date"])
    yearly = pd.read_csv(args.yearly_csv, parse_dates=["date"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out1 = args.out_dir / f"{args.prefix}_monthly_5y_mean.png"
    out2 = args.out_dir / f"{args.prefix}_yearly_5y_mean.png"
    out3 = args.out_dir / f"{args.prefix}_10year_summary.png"
    build_monthly(monthly, out1, args.label)
    build_yearly(yearly, out2, args.label)
    build_ten_year_summary(monthly, out3, args.label)
    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print(f"Wrote: {out3}")


if __name__ == "__main__":
    main()
