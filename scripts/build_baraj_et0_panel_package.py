#!/usr/bin/env python3
"""Build ET0 summary + charts from climate panel ET0 (monthly)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "assets/data/climate_baseline.js"
OUT_DIR = BASE / "assets/et0"
OUT_DIR_WEB = BASE / "baraj_web/assets/et0"
REPORT_DIR = OUT_DIR / "reports"
REPORT_DIR_WEB = OUT_DIR_WEB / "reports"


def load_panel() -> pd.DataFrame:
    raw = DATA_PATH.read_text()
    prefix = "window.CLIMATE_BASELINE = "
    if not raw.strip().startswith(prefix):
        raise ValueError("Unexpected climate_baseline.js format")
    payload = raw.strip()[len(prefix):].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    data = json.loads(payload)
    rows = []
    for k, v in data.items():
        row = {"date": pd.to_datetime(k)}
        row.update(v)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def linear_trend(years: np.ndarray, values: np.ndarray) -> dict[str, float]:
    # Use scipy if available for p-value, otherwise compute slope + r2 only.
    slope = float(np.polyfit(years, values, 1)[0])
    intercept = float(values.mean() - slope * years.mean())
    fit = slope * years + intercept
    ss_res = float(np.sum((values - fit) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p_value = None
    try:
        from scipy import stats  # type: ignore

        res = stats.linregress(years, values)
        slope = float(res.slope)
        intercept = float(res.intercept)
        r2 = float(res.rvalue ** 2)
        p_value = float(res.pvalue)
    except Exception:
        pass

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "p_value": p_value,
    }


def build_summary(df: pd.DataFrame) -> tuple[dict, dict]:
    hist = df[df["scenario"] == "historical"].copy()
    future = df[df["scenario"] == "future_projection"].copy()

    yearly = hist.groupby("year")["et0_mm_month"].sum()
    years = yearly.index.to_numpy()
    values = yearly.values

    trend = linear_trend(years, values)

    mean_et0 = float(values.mean())
    min_et0 = float(values.min())
    max_et0 = float(values.max())

    baseline = hist[(hist["year"] >= 2015) & (hist["year"] <= 2024)]
    baseline_yearly = baseline.groupby("year")["et0_mm_month"].sum()
    baseline_mean = float(baseline_yearly.mean())

    future_2031_2035 = future[(future["year"] >= 2031) & (future["year"] <= 2035)]
    future_yearly = future_2031_2035.groupby("year")["et0_mm_month"].sum()
    future_mean = float(future_yearly.mean())

    summary = {
        "coverage": {
            "history_rows_daily": 0,
            "history_rows_monthly": int(hist.shape[0]),
            "history_rows_yearly": int(len(yearly)),
            "daily_start": hist["date"].min().strftime("%Y-%m-%d"),
            "daily_end": hist["date"].max().strftime("%Y-%m-%d"),
            "model_start": df["date"].min().strftime("%Y-%m-%d"),
            "model_end": df["date"].max().strftime("%Y-%m-%d"),
        },
        "radiation_input": {
            "file": "assets/data/climate_baseline.js",
            "real_extracted_days": 0,
            "synthetic_days": 0,
            "days_clipped_to_rso": 0,
            "note": "ET0 aylik degerleri iklim panelinden alinmistir; ham radyasyon dosyasi bu repoda yoktur.",
        },
        "historical_stats": {
            "et0_mm_year_mean": mean_et0,
            "et0_mm_year_min": min_et0,
            "et0_mm_year_max": max_et0,
            "trend_mm_per_decade": float(trend["slope"] * 10.0),
        },
        "forecast_stats": {
            "baseline_year_range": "2015-2024",
            "baseline_mm_year": baseline_mean,
            "forecast_2031_2035_mm_year": future_mean,
            "delta_2031_2035_vs_baseline_mm_year": float(future_mean - baseline_mean),
        },
        "assumptions": {
            "et0_source": "Iklim paneli aylik ET0 (2010-2024 gozlem, 2026-2040 projeksiyon)",
            "monthly_coverage_rule": "Tumu aylik panelden alinmistir",
            "forecast_method": "Panelde verilen projeksiyon kullanildi",
        },
    }

    trend_stats = {
        "year_start": float(years.min()),
        "year_end": float(years.max()),
        "n_years": float(len(years)),
        "mean_et0_mm_year": mean_et0,
        "trend_mm_per_year": float(trend["slope"]),
        "trend_mm_per_decade": float(trend["slope"] * 10.0),
        "r_squared": float(trend["r2"]),
        "p_value": trend["p_value"],
        "min_et0_mm_year": min_et0,
        "max_et0_mm_year": max_et0,
    }

    return summary, trend_stats


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def build_charts(df: pd.DataFrame, summary: dict, trend_stats: dict) -> None:
    hist = df[df["scenario"] == "historical"].copy()

    yearly = hist.groupby("year")["et0_mm_month"].sum()
    years = yearly.index.to_numpy()
    yvals = yearly.values

    mean_et0 = summary["historical_stats"]["et0_mm_year_mean"]
    min_et0 = summary["historical_stats"]["et0_mm_year_min"]
    max_et0 = summary["historical_stats"]["et0_mm_year_max"]
    trend_decade = summary["historical_stats"]["trend_mm_per_decade"]

    baseline = summary["forecast_stats"]["baseline_mm_year"]
    forecast = summary["forecast_stats"]["forecast_2031_2035_mm_year"]
    delta = summary["forecast_stats"]["delta_2031_2035_vs_baseline_mm_year"]

    slope = trend_stats["trend_mm_per_year"]
    r2 = trend_stats.get("r_squared", None)
    pval = trend_stats.get("p_value", None)
    intercept = float(yvals.mean() - slope * years.mean())
    fit = slope * years + intercept

    # Yearly trend
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.plot(years, yvals, marker="o", lw=2, color="#1f77b4", label="Yillik ET0")
    ax.plot(years, fit, lw=2, color="#d62728", linestyle="--", label="Dogrusal trend")
    ax.set_title("Yillik ET0 (2010-2024) ve lineer trend egimi", fontsize=18)
    ax.set_xlabel("Yil")
    ax.set_ylabel("ET0 (mm/yil)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    info_lines = [
        f"Ortalama: {mean_et0:.1f} mm/yil",
        f"Min/Max: {min_et0:.1f} / {max_et0:.1f} mm/yil",
        f"Trend egimi: {trend_decade:+.1f} mm/10 yil (2010-2024)",
        "Not: Trend egimi, baz-tahmin seviye farki degildir.",
    ]
    if r2 is not None and pval is not None:
        info_lines.append(f"R^2: {r2:.2f} | p: {pval:.3f}")

    ax.text(
        0.02,
        0.95,
        "\n".join(info_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "baraj_et0_yearly_trend_robust_explained.png")
    plt.close(fig)

    # Monthly explained chart
    sample_year = 2014 if 2014 in hist["year"].unique() else int(hist["year"].min())
    month_df = hist[hist["year"] == sample_year].sort_values("month")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.bar(month_df["month"], month_df["et0_mm_month"], color="#2ca02c")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Ay")
    ax.set_ylabel("ET0 (mm/ay)")
    ax.set_title(f"Ornek Yil Aylik ET0: {sample_year}", fontsize=18)
    ax.grid(True, axis="y", alpha=0.25)
    annual_total = month_df["et0_mm_month"].sum()
    ax.text(
        0.02,
        0.95,
        f"Yillik toplam: {annual_total:.1f} mm/yil",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baraj_et0_monthly_explained_2004.png")
    plt.close(fig)

    # 10-year summary chart
    fig, ax = plt.subplots(figsize=(15.5, 7.5), dpi=150)
    labels = ["2015-2024 (Baz)", "2031-2035 (Tahmin)"]
    values = [baseline, forecast]
    colors = ["#4c78a8", "#f58518"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("ET0 (mm/yil)")
    ax.set_title("Baz (2015-2024) vs Tahmin (2031-2035) ET0 Seviye Farki", fontsize=17)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.5,
        0.9,
        f"Seviye farki (tahmin - baz): {delta:+.1f} mm/yil\n(Not: bu deger trend egimi degildir)",
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baraj_et0_10year_summary.png")
    plt.close(fig)

    # Monthly 5-year means
    periods = [
        (2010, 2014, "2010-2014"),
        (2015, 2019, "2015-2019"),
        (2020, 2024, "2020-2024"),
        (2031, 2035, "2031-2035 (proj.)"),
    ]
    fig, ax = plt.subplots(figsize=(15.5, 7.5), dpi=150)
    for start, end, label in periods:
        subset = df[(df["year"] >= start) & (df["year"] <= end)]
        if subset.empty:
            continue
        monthly_mean = subset.groupby("month")["et0_mm_month"].mean()
        style = "--" if "proj" in label else "-"
        ax.plot(monthly_mean.index, monthly_mean.values, marker="o", lw=2, linestyle=style, label=label)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Ay")
    ax.set_ylabel("ET0 (mm/ay)")
    ax.set_title("Aylik ET0 Ortalama Deseni (5 yillik)", fontsize=17)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baraj_et0_monthly_5y_mean.png")
    plt.close(fig)

    # Formula / summary card
    fig = plt.figure(figsize=(11, 12), dpi=300)
    fig.patch.set_facecolor("white")

    fig.text(0.06, 0.94, "ET0 Ozet ve Veri Kaynagi", fontsize=20, fontweight="bold")

    summary_lines = [
        "Veri kaynagi: assets/data/climate_baseline.js",
        "Gozlem: 2010-2024 | Projeksiyon: 2026-2040",
        f"Ortalama yillik ET0 (2010-2024): {mean_et0:.1f} mm/yil",
        f"Trend: {trend_decade:+.1f} mm/10 yil",
        f"Baz (2015-2024): {baseline:.1f} mm/yil",
        f"2031-2035 ortalama: {forecast:.1f} mm/yil",
        f"Beklenen fark: {delta:+.1f} mm/yil",
    ]

    fig.text(0.06, 0.86, "\n".join(summary_lines), fontsize=12, va="top")

    note_lines = [
        "Notlar:",
        "- ET0 degerleri iklim panelinden aylik olarak alinmistir.",
        "- Bu rapor ET0 degerlerini yeniden hesaplamaz.",
        "- ET0 referans evapotranspirasyondur; acik su buharlasmasi degildir.",
    ]
    fig.text(0.06, 0.64, "\n".join(note_lines), fontsize=12, va="top")

    formula_title = "FAO-56 Penman-Monteith (referans)"
    formula = "ET0 = [0.408*Delta*(Rn-G) + gamma*(900/(T+273))*u2*(es-ea)] / [Delta + gamma*(1 + 0.34*u2)]"

    fig.text(0.06, 0.48, formula_title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.06, 0.44, formula, fontsize=11, va="top")

    fig.text(
        0.06,
        0.36,
        "Formul, ET0 kavraminin fiziksel temelini gosterir; burada hesaplama yapilmamistir.",
        fontsize=11,
        va="top",
    )

    fig.savefig(OUT_DIR / "baraj_et0_formula_explained.png")
    plt.close(fig)


def copy_outputs() -> None:
    OUT_DIR_WEB.mkdir(parents=True, exist_ok=True)
    REPORT_DIR_WEB.mkdir(parents=True, exist_ok=True)
    for name in [
        "baraj_et0_yearly_trend_robust_explained.png",
        "baraj_et0_monthly_explained_2004.png",
        "baraj_et0_10year_summary.png",
        "baraj_et0_monthly_5y_mean.png",
        "baraj_et0_formula_explained.png",
    ]:
        (OUT_DIR_WEB / name).write_bytes((OUT_DIR / name).read_bytes())

    for name in [
        "baraj_et0_real_radiation_summary.json",
        "baraj_et0_yearly_trend_stats.json",
    ]:
        (REPORT_DIR_WEB / name).write_bytes((REPORT_DIR / name).read_bytes())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_panel()
    summary, trend_stats = build_summary(df)

    write_json(summary, REPORT_DIR / "baraj_et0_real_radiation_summary.json")
    write_json(trend_stats, REPORT_DIR / "baraj_et0_yearly_trend_stats.json")

    build_charts(df, summary, trend_stats)
    copy_outputs()

    print("ET0 panel package updated.")


if __name__ == "__main__":
    main()
