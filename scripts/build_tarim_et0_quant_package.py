#!/usr/bin/env python3
"""Build a monthly agricultural ET0 package with tables, charts, forecast, and report.

Outputs under output/tarim_et0_quant:
  - tables/tarim_et0_monthly_history.csv
  - tables/tarim_et0_yearly_history.csv
  - tables/tarim_et0_quant_forecast_to_<year>.csv
  - charts/*.png
  - reports/tarim_et0_model_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_et0_fao56 import calc_ra_mj_m2_day


SIGMA_MJ_K4_M2_DAY = 4.903e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly agricultural ET0 package.")
    parser.add_argument(
        "--daily-input",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/spreadsheet/es_ea_newdata_daily.csv"),
        help="Daily temperature/humidity source CSV.",
    )
    parser.add_argument(
        "--solar-input",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_sunshine_proxy/istanbul_monthly_sunshine_radiation_1911_2025.csv"),
        help="Monthly radiation proxy CSV.",
    )
    parser.add_argument(
        "--quant-script",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/scripts/quant_regime_projection.py"),
        help="Quant forecast script path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_quant"),
        help="Output package directory.",
    )
    parser.add_argument("--latitude", type=float, default=41.01, help="Latitude in degrees.")
    parser.add_argument("--elevation-m", type=float, default=39.0, help="Elevation in meters.")
    parser.add_argument("--u2", type=float, default=2.0, help="Constant 2 m wind speed in m/s.")
    parser.add_argument("--target-year", type=int, default=2035, help="Forecast horizon year.")
    return parser.parse_args()


def saturation_vapor_pressure_kpa(temp_c: pd.Series | np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))


def delta_svp_curve_kpa_c(temp_c: pd.Series | np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    return 4098.0 * (0.6108 * np.exp((17.27 * temp) / (temp + 237.3))) / ((temp + 237.3) ** 2)


def pressure_from_elevation_kpa(elevation_m: float) -> float:
    return 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26


def build_monthly_history(
    daily_input: Path,
    solar_input: Path,
    latitude: float,
    elevation_m: float,
    u2_m_s: float,
) -> pd.DataFrame:
    daily = pd.read_csv(
        daily_input,
        usecols=["date", "t_max_c", "t_min_c", "t_mean_c", "rh_mean_pct"],
        parse_dates=["date"],
    )
    daily = daily.dropna(subset=["date", "t_max_c", "t_min_c", "rh_mean_pct"]).copy()
    daily["t_mean_formula_c"] = (pd.to_numeric(daily["t_max_c"], errors="coerce") + pd.to_numeric(daily["t_min_c"], errors="coerce")) / 2.0
    daily["t_mean_c"] = daily["t_mean_formula_c"]
    daily["rh_mean_pct"] = pd.to_numeric(daily["rh_mean_pct"], errors="coerce").clip(lower=1.0, upper=100.0)
    daily = daily.dropna(subset=["t_mean_c", "rh_mean_pct"]).sort_values("date")
    daily = daily.set_index("date")

    monthly = (
        daily.resample("MS")
        .agg(
            t_mean_c=("t_mean_c", "mean"),
            t_max_c=("t_max_c", "mean"),
            t_min_c=("t_min_c", "mean"),
            rh_mean_pct=("rh_mean_pct", "mean"),
        )
        .dropna(subset=["t_mean_c", "t_max_c", "t_min_c", "rh_mean_pct"])
    )
    monthly["days_in_month"] = monthly.index.days_in_month.astype(int)

    solar = pd.read_csv(
        solar_input,
        usecols=["timestamp", "radiation_mj_m2_day_proxy"],
        parse_dates=["timestamp"],
    )
    solar = solar.rename(columns={"timestamp": "date", "radiation_mj_m2_day_proxy": "rs_mj_m2_day"})
    solar = solar.dropna(subset=["date", "rs_mj_m2_day"]).copy()
    solar["date"] = solar["date"].dt.to_period("M").dt.to_timestamp()
    solar = solar.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    solar = solar.set_index("date")

    monthly = monthly.join(solar, how="left")
    monthly["month"] = monthly.index.month
    solar_climo = solar.assign(month=solar.index.month).groupby("month")["rs_mj_m2_day"].mean()
    monthly["rs_source"] = np.where(monthly["rs_mj_m2_day"].notna(), "proxy_observed", "monthly_climatology_fill")
    monthly["rs_mj_m2_day"] = monthly["rs_mj_m2_day"].fillna(monthly["month"].map(solar_climo))
    monthly = monthly.dropna(subset=["rs_mj_m2_day"]).copy()

    all_days = pd.DataFrame({"date": pd.date_range(monthly.index.min(), monthly.index.max() + pd.offsets.MonthEnd(0), freq="D")})
    all_days["ra_mj_m2_day"] = calc_ra_mj_m2_day(all_days["date"].dt.dayofyear.astype(int), latitude).to_numpy(dtype=float)
    ra_monthly = all_days.set_index("date")["ra_mj_m2_day"].resample("MS").mean()
    monthly = monthly.join(ra_monthly, how="left")

    monthly["u2_m_s"] = float(u2_m_s)
    monthly["u2_source"] = "constant_fao56_fallback"
    monthly["p_kpa"] = pressure_from_elevation_kpa(elevation_m)
    monthly["p_source"] = "elevation_constant"
    monthly["gamma_kpa_c"] = 0.000665 * monthly["p_kpa"]
    monthly["delta_kpa_c"] = delta_svp_curve_kpa_c(monthly["t_mean_c"])
    monthly["es_tmax_kpa"] = saturation_vapor_pressure_kpa(monthly["t_max_c"])
    monthly["es_tmin_kpa"] = saturation_vapor_pressure_kpa(monthly["t_min_c"])
    monthly["es_kpa"] = 0.5 * (monthly["es_tmax_kpa"] + monthly["es_tmin_kpa"])
    monthly["ea_kpa"] = (monthly["rh_mean_pct"] / 100.0) * monthly["es_kpa"]
    monthly["vpd_kpa"] = np.clip(monthly["es_kpa"] - monthly["ea_kpa"], 0.0, None)
    monthly["rso_mj_m2_day"] = (0.75 + 2.0e-5 * elevation_m) * monthly["ra_mj_m2_day"]
    monthly["rs_mj_m2_day"] = np.minimum(monthly["rs_mj_m2_day"], monthly["rso_mj_m2_day"])
    monthly["rns_mj_m2_day"] = 0.77 * monthly["rs_mj_m2_day"]

    tmax_k = monthly["t_max_c"] + 273.16
    tmin_k = monthly["t_min_c"] + 273.16
    rs_rso = np.where(monthly["rso_mj_m2_day"] > 0, monthly["rs_mj_m2_day"] / monthly["rso_mj_m2_day"], np.nan)
    rs_rso = np.clip(rs_rso, 0.0, 1.0)
    monthly["rs_rso_ratio"] = rs_rso
    monthly["rnl_mj_m2_day"] = (
        SIGMA_MJ_K4_M2_DAY
        * ((tmax_k**4 + tmin_k**4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.maximum(monthly["ea_kpa"], 0.0)))
        * (1.35 * monthly["rs_rso_ratio"] - 0.35)
    )
    monthly["rn_mj_m2_day"] = monthly["rns_mj_m2_day"] - monthly["rnl_mj_m2_day"]
    monthly["g_mj_m2_day"] = 0.0
    numerator = (
        0.408 * monthly["delta_kpa_c"] * (monthly["rn_mj_m2_day"] - monthly["g_mj_m2_day"])
        + monthly["gamma_kpa_c"] * (900.0 / (monthly["t_mean_c"] + 273.0)) * monthly["u2_m_s"] * monthly["vpd_kpa"]
    )
    denominator = monthly["delta_kpa_c"] + monthly["gamma_kpa_c"] * (1.0 + 0.34 * monthly["u2_m_s"])
    monthly["et0_mm_day"] = np.where(denominator > 0, numerator / denominator, np.nan)
    monthly["et0_mm_day"] = monthly["et0_mm_day"].clip(lower=0.0)
    monthly["et0_mm_month"] = monthly["et0_mm_day"] * monthly["days_in_month"]
    monthly["date"] = monthly.index

    cols = [
        "date",
        "days_in_month",
        "t_mean_c",
        "t_max_c",
        "t_min_c",
        "rh_mean_pct",
        "rs_mj_m2_day",
        "rs_source",
        "ra_mj_m2_day",
        "rso_mj_m2_day",
        "rn_mj_m2_day",
        "g_mj_m2_day",
        "u2_m_s",
        "u2_source",
        "p_kpa",
        "p_source",
        "delta_kpa_c",
        "gamma_kpa_c",
        "es_kpa",
        "ea_kpa",
        "vpd_kpa",
        "et0_mm_day",
        "et0_mm_month",
    ]
    return monthly[cols].reset_index(drop=True)


def build_yearly_history(monthly: pd.DataFrame) -> pd.DataFrame:
    yearly = monthly.copy()
    yearly["year"] = pd.to_datetime(yearly["date"]).dt.year
    out = (
        yearly.groupby("year", as_index=False)
        .agg(
            months_present=("et0_mm_month", "count"),
            et0_mm_year=("et0_mm_month", "sum"),
            et0_mm_day_mean=("et0_mm_day", "mean"),
            t_mean_c=("t_mean_c", "mean"),
            rs_mj_m2_day=("rs_mj_m2_day", "mean"),
            vpd_kpa=("vpd_kpa", "mean"),
        )
    )
    out = out[out["months_present"] == 12].copy()
    out["date"] = pd.to_datetime(out["year"].astype(str) + "-01-01")
    return out[["date", "year", "months_present", "et0_mm_year", "et0_mm_day_mean", "t_mean_c", "rs_mj_m2_day", "vpd_kpa"]]


def plot_history(monthly: pd.DataFrame, yearly: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    hist = monthly.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.set_index("date").sort_index()

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(hist.index, hist["et0_mm_month"], color="#c55a11", linewidth=0.8, alpha=0.35, label="Aylik ET0")
    ax.plot(hist.index, hist["et0_mm_month"].rolling(12, min_periods=6).mean(), color="#7a2e00", linewidth=2.0, label="12 ay ort.")
    ax.set_title("Tarihsel Tarimsal ET0 (Aylik)")
    ax.set_ylabel("ET0 (mm/ay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "tarim_et0_monthly_history.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    annual = yearly.copy()
    annual["date"] = pd.to_datetime(annual["date"])
    coef = np.polyfit(annual["year"], annual["et0_mm_year"], 1)
    trend = np.polyval(coef, annual["year"])

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(annual["year"], annual["et0_mm_year"], color="#d99843", width=0.9, alpha=0.85)
    ax.plot(annual["year"], trend, color="#264653", linewidth=2.0, label=f"Trend: {coef[0] * 10.0:+.1f} mm/10y")
    ax.set_title("Yillik Tarimsal ET0")
    ax.set_ylabel("ET0 (mm/yil)")
    ax.set_xlabel("Yil")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "tarim_et0_yearly_trend.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    climatology = hist.groupby(hist.index.month)["et0_mm_month"].mean()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(range(1, 13), climatology.values, marker="o", color="#2a9d8f", linewidth=2.0)
    ax.set_xticks(range(1, 13))
    ax.set_title("Aylik ET0 Klimatolojisi")
    ax.set_xlabel("Ay")
    ax.set_ylabel("Ortalama ET0 (mm/ay)")
    fig.tight_layout()
    fig.savefig(charts_dir / "tarim_et0_monthly_climatology.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_quant_forecast(monthly_history_csv: Path, quant_script: Path, quant_dir: Path, target_year: int) -> Path:
    quant_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = quant_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(tmp_dir))
    env.setdefault("MPLCONFIGDIR", str(quant_dir / ".mpl"))
    env.setdefault("XDG_CACHE_HOME", str(quant_dir / ".cache"))
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        "python3",
        str(quant_script),
        "--observations",
        str(monthly_history_csv),
        "--output-dir",
        str(quant_dir),
        "--input-kind",
        "single",
        "--timestamp-col",
        "date",
        "--value-col",
        "et0_mm_month",
        "--single-variable",
        "et0",
        "--variables",
        "et0",
        "--target-year",
        str(target_year),
        "--disable-climate-adjustment",
        "--holdout-steps",
        "24",
        "--backtest-splits",
        "4",
        "--min-train-steps",
        "60",
        "--vol-model",
        "ewma",
    ]
    subprocess.run(cmd, check=True, env=env)

    matches = sorted((quant_dir / "forecasts").glob(f"et0*_quant_to_{target_year}.csv"))
    if not matches:
        raise FileNotFoundError(f"Forecast CSV not found in {quant_dir / 'forecasts'}")
    return matches[0]


def clean_forecast_csv(raw_forecast_csv: Path, out_csv: Path) -> pd.DataFrame:
    fc = pd.read_csv(raw_forecast_csv, parse_dates=["ds"])
    for col in ["actual", "yhat", "yhat_lower", "yhat_upper"]:
        if col in fc.columns:
            fc[col] = pd.to_numeric(fc[col], errors="coerce")
            fc[col] = fc[col].clip(lower=0.0)
    fc = fc.rename(columns={"ds": "date"})
    fc.to_csv(out_csv, index=False)
    return fc


def plot_forecast(forecast_df: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    hist = df[df["is_forecast"] == False].copy()
    fc = df[df["is_forecast"] == True].copy()

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(hist["date"], hist["actual"], color="#4c6a92", linewidth=0.8, alpha=0.35, label="Tarihsel ET0")
    ax.plot(hist["date"], hist["yhat"], color="#1d3557", linewidth=1.6, label="Model uyumu")
    if not fc.empty:
        ax.plot(fc["date"], fc["yhat"], color="#d1495b", linewidth=2.0, label="Quant oengoru")
        if {"yhat_lower", "yhat_upper"}.issubset(fc.columns):
            ax.fill_between(fc["date"], fc["yhat_lower"], fc["yhat_upper"], color="#d1495b", alpha=0.18, label="Guven bandi")
    ax.set_title("Tarimsal ET0 Quant Oengoru")
    ax.set_ylabel("ET0 (mm/ay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "tarim_et0_quant_forecast.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_summary(monthly: pd.DataFrame, yearly: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    annual = yearly.copy()
    annual["year"] = annual["year"].astype(int)
    slope_per_year = np.polyfit(annual["year"], annual["et0_mm_year"], 1)[0]
    recent_base = annual[annual["year"].between(2011, 2025)]["et0_mm_year"].mean()

    fc = forecast.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    fc["year"] = fc["date"].dt.year
    fc_future = fc[fc["is_forecast"] == True].copy()
    fc_yearly = fc_future.groupby("year", as_index=False)["yhat"].sum().rename(columns={"yhat": "et0_mm_year"})
    far_future = fc_yearly[fc_yearly["year"].between(2031, 2035)]["et0_mm_year"].mean()

    month_climo = monthly.copy()
    month_climo["month"] = pd.to_datetime(month_climo["date"]).dt.month
    peak_month = (
        month_climo.groupby("month", as_index=False)["et0_mm_month"].mean().sort_values("et0_mm_month", ascending=False).iloc[0]
    )

    return {
        "coverage": {
            "history_rows_monthly": int(len(monthly)),
            "history_rows_yearly": int(len(yearly)),
            "history_start": str(pd.to_datetime(monthly["date"]).min().date()),
            "history_end": str(pd.to_datetime(monthly["date"]).max().date()),
        },
        "historical_stats": {
            "et0_mm_year_mean": float(yearly["et0_mm_year"].mean()),
            "et0_mm_year_min": float(yearly["et0_mm_year"].min()),
            "et0_mm_year_max": float(yearly["et0_mm_year"].max()),
            "trend_mm_per_decade": float(slope_per_year * 10.0),
            "peak_climatology_month": int(peak_month["month"]),
            "peak_climatology_month_et0_mm": float(peak_month["et0_mm_month"]),
        },
        "forecast_stats": {
            "baseline_2011_2025_mm_year": float(recent_base) if pd.notna(recent_base) else None,
            "forecast_2031_2035_mm_year": float(far_future) if pd.notna(far_future) else None,
            "delta_2031_2035_vs_2011_2025_mm_year": (
                float(far_future - recent_base) if pd.notna(far_future) and pd.notna(recent_base) else None
            ),
        },
        "assumptions": {
            "tmean_method": "Tmean = (Tmax + Tmin) / 2",
            "delta_method": "Delta from Tmean with FAO-56 slope equation",
            "soil_heat_flux": "G = 0",
            "wind_speed": "u2 = 2.0 m/s constant fallback",
            "pressure": "elevation-derived constant pressure",
            "radiation": "monthly proxy series, climatology fill when unavailable",
            "forecast_method": "quant regime forecast run directly on historical monthly ET0",
        },
    }


def write_report(
    report_path: Path,
    monthly_csv: Path,
    yearly_csv: Path,
    forecast_csv: Path,
    charts_dir: Path,
    summary: dict,
) -> None:
    hist = summary["historical_stats"]
    fc = summary["forecast_stats"]

    lines = [
        "# Tarimsal ET0 Model Paketi",
        "",
        "## Kabuller ve Nedenleri",
        "",
        "1. `Tmean = (Tmax + Tmin) / 2` kullanildi.",
        "Nedeni: kullanicinin talebiyle uyumlu ve gunluk seri boyunca en tutarli sicaklik temsili bu.",
        "",
        "2. `Delta`, Tmean uzerinden FAO-56 egri egimi denklemiyle hesaplandi.",
        "Nedeni: Delta dogrudan sicaklik ortalamasi degil, doymus buhar basinci egiminin turevidir; fiziksel dogruluk icin bu sekilde hesaplandi.",
        "",
        "3. `G = 0` alindi.",
        "Nedeni: tarimsal ET0 icin aylik olcekte toprak isi akisinin net etkisi ikincil oldugu icin standart yaklasim tercih edildi.",
        "",
        "4. `u2 = 2.0 m/s` sabit ruzgar kullanildi.",
        "Nedeni: uzun donem, kesintisiz ve ayni frekansta ruzgar serisi bulunmuyor. FAO-56 icin en yaygin operasyonel fallback budur.",
        "",
        "5. Basinc, rakimdan turetilen sabit deger olarak kullanildi.",
        "Nedeni: psikrometrik sabit uzerindeki etkisi sinirlidir; uzun seri boyunca girdi tutarliligi saglar.",
        "",
        "6. Radyasyon icin aylik proxy seri kullanildi; eksik aylarda ayni ay klimatolojisi ile dolduruldu.",
        "Nedeni: fiziksel ET0 hesabinda radyasyon zorunlu girdidir. Eksik yerlerde mevsimsel kalip korundu.",
        "",
        "7. Gelecek oengorusu, girdi bazli degil, ET0 serisinin kendisi uzerinden quant model ile yapildi.",
        "Nedeni: ileriye donuk ruzgar ve radyasyon belirsizligi yuksek. Dogrudan ET0 serisini modellemek daha kontrollu ve daha savunulabilir.",
        "",
        "## Hesaplama Ozeti",
        "",
        "- Yontem: aylik FAO-56 Penman-Monteith",
        "- Cikti birimleri: `mm/gun` ve `mm/ay`",
        "- Tarihsel kapsama: "
        f"`{summary['coverage']['history_start']}` -> `{summary['coverage']['history_end']}`",
        "",
        "## Temel Bulgular",
        "",
        f"- Ortalama yillik ET0: `{hist['et0_mm_year_mean']:.1f} mm/yil`",
        f"- Yillik ET0 trendi: `{hist['trend_mm_per_decade']:+.1f} mm/10y`",
        f"- Klimatolojik zirve ay: `{hist['peak_climatology_month']}`. ay (`{hist['peak_climatology_month_et0_mm']:.1f} mm/ay`)",
    ]

    if fc["baseline_2011_2025_mm_year"] is not None and fc["forecast_2031_2035_mm_year"] is not None:
        lines.extend(
            [
                f"- 2011-2025 ortalama yillik ET0: `{fc['baseline_2011_2025_mm_year']:.1f} mm/yil`",
                f"- 2031-2035 quant oengoru ortalama yillik ET0: `{fc['forecast_2031_2035_mm_year']:.1f} mm/yil`",
                f"- Beklenen fark: `{fc['delta_2031_2035_vs_2011_2025_mm_year']:+.1f} mm/yil`",
            ]
        )

    lines.extend(
        [
            "",
            "## Faydalari ve Kullanimlari",
            "",
            "Bu ET0 paketi sulama planlamasi, kuraklik izleme, urun su ihtiyaci tahmini ve operasyonel tarim karar destegi icin kullanilabilir.",
            "",
            "Sulama tarafinda ana faydasi, su ihtiyacini dogrudan meteorolojik talep tarafindan okumayi saglamasidir. Yagis tek basina yeterli degildir; ET0 yuksekse toprak ve bitki su acigi hizla buyuyebilir. Bu nedenle ET0 serisi sulama periyodu, su verme sikligi ve kritik aylarin tespiti icin temel gostergedir.",
            "",
            "Kuraklik tarafinda ET0, sadece yagis eksikligini degil atmosferik talep baskisini da gosterir. Sicaklik ve radyasyon arttiginda, yagis normal kalsa bile tarimsal stres buyuyebilir. Bu yuzden ET0, yagisla birlestirildiginde daha gercekci bir su dengesi gorunumu verir.",
            "",
            "Urun yonetiminde ET0, `ETc = Kc * ET0` yapisina dogrudan girer. Yani bugun burada urettigimiz referans seri, daha sonra bugday, misir, aycicegi, sebze gibi urunlere ozel Kc katsayilariyla urun su tuketimine cevrilebilir.",
            "",
            "Havza ve baraj yonetiminde bu paket dolayli fayda da saglar. Her ne kadar burada tarimsal ET0 hesapliyor olsak da, yuksek ET0 donemleri tarimsal su talebini buyuterek havza uzerindeki baskiyi artirir. Bu nedenle tarim ET0 serisi, baraj isletmesi ve sulama suyu tahsisi tarafinda erken uyari katmani olarak kullanilabilir.",
            "",
            "Sigortacilik ve risk yonetiminde ET0 trendi, sicak-yuksek buharlasmanin urun verimi uzerindeki baskisini gosteren nicel bir degiskendir. Tarihsel trend ve quant forecast birlikte kullanildiginda, gelecekte daha fazla su stresi beklenen donemler icin risk senaryolari kurulabilir.",
            "",
            "Enerji ve saha operasyonlarinda ET0 dolayli olarak is gucu, sulama pompa calisma suresi, enerji tuketimi ve saha ziyaret takvimini etkiler. Yuksek ET0 donemleri operasyonel maliyetleri buyutebilir.",
            "",
            "Akademik ve teknik raporlama tarafinda bu paket tekrar uretilebilir bir hat sunar: kabuller acik, ara terimler kayitli, tarihsel seri sakli ve oengoru ayni dosya yapisinda verilir. Bu, savunulabilirlik icin onemlidir.",
            "",
            "## Uretilen Dosyalar",
            "",
            f"- Tarihsel aylik tablo: `{monthly_csv}`",
            f"- Tarihsel yillik tablo: `{yearly_csv}`",
            f"- Quant forecast tablo: `{forecast_csv}`",
            f"- Grafik klasoru: `{charts_dir}`",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir = args.out_dir / "tables"
    charts_dir = args.out_dir / "charts"
    reports_dir = args.out_dir / "reports"
    quant_dir = args.out_dir / "quant"
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    monthly = build_monthly_history(
        daily_input=args.daily_input,
        solar_input=args.solar_input,
        latitude=args.latitude,
        elevation_m=args.elevation_m,
        u2_m_s=args.u2,
    )
    yearly = build_yearly_history(monthly)

    monthly_csv = tables_dir / "tarim_et0_monthly_history.csv"
    yearly_csv = tables_dir / "tarim_et0_yearly_history.csv"
    monthly.to_csv(monthly_csv, index=False)
    yearly.to_csv(yearly_csv, index=False)

    plot_history(monthly, yearly, charts_dir)

    quant_source_csv = tables_dir / "tarim_et0_quant_source.csv"
    monthly[["date", "et0_mm_month"]].to_csv(quant_source_csv, index=False)
    raw_forecast_csv = run_quant_forecast(
        monthly_history_csv=quant_source_csv,
        quant_script=args.quant_script,
        quant_dir=quant_dir,
        target_year=args.target_year,
    )
    forecast_csv = tables_dir / f"tarim_et0_quant_forecast_to_{args.target_year}.csv"
    forecast_df = clean_forecast_csv(raw_forecast_csv, forecast_csv)
    plot_forecast(forecast_df, charts_dir)

    summary = build_summary(monthly, yearly, forecast_df)
    summary_path = reports_dir / "tarim_et0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = reports_dir / "tarim_et0_model_report.md"
    write_report(report_path, monthly_csv, yearly_csv, forecast_csv, charts_dir, summary)

    print(f"Wrote: {monthly_csv}")
    print(f"Wrote: {yearly_csv}")
    print(f"Wrote: {forecast_csv}")
    print(f"Wrote: {charts_dir}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
