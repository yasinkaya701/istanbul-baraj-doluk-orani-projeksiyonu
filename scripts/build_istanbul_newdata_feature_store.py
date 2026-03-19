#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_es_ea_newdata_csv import load_wide_daily_sheet


CAPACITIES_MCM = {
    "Omerli": 235.371,
    "Darlik": 107.500,
    "Elmali": 9.600,
    "Terkos": 162.241,
    "Alibey": 34.143,
    "Buyukcekmece": 148.943,
    "Sazlidere": 88.730,
    "Kazandere": 17.424,
    "Pabucdere": 58.500,
    "Istrancalar": 6.231,
}

MMHG_TO_KPA = 0.133322368


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a professional monthly feature store from Istanbul new data.")
    parser.add_argument(
        "--temp-xlsx",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/new data/Veriler_H-3/Sçcaklçk/Uzunyçl-Max-Min-Ort-orj.xlsx"),
    )
    parser.add_argument(
        "--humidity-xlsx",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/new data/Veriler_H-3/Nem/1911-2022-Nem.xlsx"),
    )
    parser.add_argument(
        "--pressure-xlsx",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/new data/Veriler_H-3/Basçná/Basçná 1912-2018Kasim.xlsx"),
    )
    parser.add_argument(
        "--auto-table1",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/new data/Veriler_H-3/Otomatik òstasyon/CR800Series_Table1.dat"),
    )
    parser.add_argument(
        "--auto-table2",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/new data/Veriler_H-3/Otomatik òstasyon/CR800Series_Table2.dat"),
    )
    parser.add_argument(
        "--vpd-daily-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/spreadsheet/es_ea_newdata_daily.csv"),
    )
    parser.add_argument(
        "--rain-monthly-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/spreadsheet/rainfall_monthly_newdata.csv"),
    )
    parser.add_argument(
        "--et0-monthly-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/tarim_et0_quant/tables/tarim_et0_monthly_history.csv"),
    )
    parser.add_argument(
        "--occupancy-history",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast/istanbul_dam_monthly_history.csv"),
    )
    parser.add_argument(
        "--exog-monthly-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_quant_exog/tables/istanbul_dam_model_input_monthly.csv"),
    )
    parser.add_argument(
        "--wind-daily-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/kandilli_et0_project/data/derived/daily/kandilli_wind_daily_1949_2013.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_historical_daily(temp_xlsx: Path, humidity_xlsx: Path, pressure_xlsx: Path) -> pd.DataFrame:
    tmax = load_wide_daily_sheet(temp_xlsx, "Max", year_header_row=0, value_name="t_max_c")
    tmin = load_wide_daily_sheet(temp_xlsx, "Min", year_header_row=0, value_name="t_min_c")
    tmean = load_wide_daily_sheet(temp_xlsx, "Ort", year_header_row=0, value_name="t_mean_c")
    rh = load_wide_daily_sheet(humidity_xlsx, "Nem 1911-", year_header_row=1, value_name="rh_mean_pct")
    pressure = load_wide_daily_sheet(pressure_xlsx, "Basinc", year_header_row=0, value_name="pressure_mmhg")

    df = tmax.merge(tmin, on="date", how="outer")
    df = df.merge(tmean, on="date", how="outer")
    df = df.merge(rh, on="date", how="outer")
    df = df.merge(pressure, on="date", how="outer")

    df["t_max_c"] = pd.to_numeric(df["t_max_c"], errors="coerce")
    df["t_min_c"] = pd.to_numeric(df["t_min_c"], errors="coerce")
    df["t_mean_c"] = pd.to_numeric(df["t_mean_c"], errors="coerce")
    df["rh_mean_pct"] = pd.to_numeric(df["rh_mean_pct"], errors="coerce")
    df["pressure_mmhg"] = pd.to_numeric(df["pressure_mmhg"], errors="coerce")

    df.loc[(df["t_max_c"] < -40) | (df["t_max_c"] > 60), "t_max_c"] = np.nan
    df.loc[(df["t_min_c"] < -40) | (df["t_min_c"] > 40), "t_min_c"] = np.nan
    df.loc[(df["t_mean_c"] < -40) | (df["t_mean_c"] > 50), "t_mean_c"] = np.nan
    df.loc[(df["rh_mean_pct"] < 0) | (df["rh_mean_pct"] > 100), "rh_mean_pct"] = np.nan
    df.loc[(df["pressure_mmhg"] < 650) | (df["pressure_mmhg"] > 800), "pressure_mmhg"] = np.nan

    df["pressure_kpa"] = df["pressure_mmhg"] * MMHG_TO_KPA
    return df.sort_values("date").reset_index(drop=True)


def load_auto_daily(table1_path: Path, table2_path: Path) -> pd.DataFrame:
    t1 = pd.read_csv(table1_path, skiprows=[0, 2, 3], low_memory=False)
    t1["TIMESTAMP"] = pd.to_datetime(t1["TIMESTAMP"], errors="coerce")
    t1 = t1.dropna(subset=["TIMESTAMP"]).copy()
    for col in [
        "WS_ms_S_WVT",
        "WS_ms_Max",
        "WS_ms_Min",
        "AirTCee181_Avg",
        "RHee181_Avg",
        "BP_mbar_Avg",
    ]:
        t1[col] = pd.to_numeric(t1[col], errors="coerce")
    t1["date"] = t1["TIMESTAMP"].dt.floor("D")
    auto_daily = (
        t1.groupby("date", as_index=False)
        .agg(
            auto_t_mean_c=("AirTCee181_Avg", "mean"),
            auto_rh_mean_pct=("RHee181_Avg", "mean"),
            auto_pressure_hpa=("BP_mbar_Avg", "mean"),
            auto_wind_m_s_mean=("WS_ms_S_WVT", "mean"),
            auto_wind_m_s_max_10min=("WS_ms_Max", "max"),
            auto_wind_m_s_min_10min=("WS_ms_Min", "min"),
            auto_obs_count=("AirTCee181_Avg", "count"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    t2 = pd.read_csv(table2_path, skiprows=[0, 2, 3], low_memory=False)
    t2["TIMESTAMP"] = pd.to_datetime(t2["TIMESTAMP"], errors="coerce")
    t2 = t2.dropna(subset=["TIMESTAMP"]).copy()
    for col in ["AirTCee181_Max", "AirTCee181_Min", "WS_ms_Max"]:
        t2[col] = pd.to_numeric(t2[col], errors="coerce")
    t2["date"] = t2["TIMESTAMP"].dt.floor("D") - pd.Timedelta(days=1)
    auto_dailymax = (
        t2.groupby("date", as_index=False)
        .agg(
            auto_t_max_c=("AirTCee181_Max", "max"),
            auto_t_min_c=("AirTCee181_Min", "min"),
            auto_wind_m_s_dailymax=("WS_ms_Max", "max"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    auto = auto_daily.merge(auto_dailymax, on="date", how="outer")
    auto["auto_pressure_kpa"] = auto["auto_pressure_hpa"] * 0.1
    return auto.sort_values("date").reset_index(drop=True)


def monthly_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def aggregate_monthly_daily_observed(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = monthly_start(df["date"])
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            t_mean_c=("t_mean_c", "mean"),
            t_max_daily_mean_c=("t_max_c", "mean"),
            t_min_daily_mean_c=("t_min_c", "mean"),
            rh_mean_pct=("rh_mean_pct", "mean"),
            pressure_mmhg=("pressure_mmhg", "mean"),
            pressure_kpa=("pressure_kpa", "mean"),
            temp_obs_days=("t_mean_c", lambda s: int(s.notna().sum())),
            rh_obs_days=("rh_mean_pct", lambda s: int(s.notna().sum())),
            pressure_obs_days=("pressure_mmhg", lambda s: int(s.notna().sum())),
        )
        .rename(columns={"month": "date"})
    )
    return monthly


def aggregate_monthly_vpd(vpd_daily_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(vpd_daily_csv, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = monthly_start(df["date"])
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            vpd_kpa_mean=("vpd_kpa", "mean"),
            vpd_kpa_sum=("vpd_kpa", "sum"),
            es_kpa_mean=("es_kpa", "mean"),
            ea_kpa_mean=("ea_kpa", "mean"),
            vpd_obs_days=("vpd_kpa", lambda s: int(s.notna().sum())),
        )
        .rename(columns={"month": "date"})
    )
    return monthly


def aggregate_monthly_auto(auto_daily: pd.DataFrame) -> pd.DataFrame:
    df = auto_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = monthly_start(df["date"])
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            auto_t_mean_c=("auto_t_mean_c", "mean"),
            auto_t_max_c=("auto_t_max_c", "mean"),
            auto_t_min_c=("auto_t_min_c", "mean"),
            auto_rh_mean_pct=("auto_rh_mean_pct", "mean"),
            auto_pressure_hpa=("auto_pressure_hpa", "mean"),
            auto_pressure_kpa=("auto_pressure_kpa", "mean"),
            auto_wind_m_s_mean=("auto_wind_m_s_mean", "mean"),
            auto_wind_m_s_dailymax=("auto_wind_m_s_dailymax", "mean"),
            auto_days=("auto_obs_count", lambda s: int(s.notna().sum())),
        )
        .rename(columns={"month": "date"})
    )
    return monthly


def aggregate_monthly_historical_wind(wind_daily_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(wind_daily_csv, parse_dates=["date"])
    df["month"] = monthly_start(df["date"])
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            wind_historical_index_mean=("wind_speed_mean", "mean"),
            wind_historical_obs_days=("wind_speed_count", lambda s: int(s.notna().sum())),
            wind_historical_obs_per_day=("wind_speed_count", "mean"),
        )
        .rename(columns={"month": "date"})
    )
    return monthly


def load_monthly_rain(rain_monthly_csv: Path) -> pd.DataFrame:
    rain = pd.read_csv(rain_monthly_csv, parse_dates=["date"]).rename(columns={"rain_mm": "rain_mm"})
    rain["date"] = monthly_start(rain["date"])
    return rain[["date", "rain_mm"]].sort_values("date").reset_index(drop=True)


def load_monthly_et0(et0_monthly_csv: Path) -> pd.DataFrame:
    et0 = pd.read_csv(et0_monthly_csv, parse_dates=["date"])
    et0["date"] = monthly_start(et0["date"])
    keep = [
        "date",
        "et0_mm_month",
        "t_mean_c",
        "t_max_c",
        "t_min_c",
        "rh_mean_pct",
        "p_kpa",
        "vpd_kpa",
        "rs_source",
        "u2_source",
    ]
    et0 = et0[keep].rename(
        columns={
            "t_mean_c": "et0_t_mean_c",
            "t_max_c": "et0_t_max_c",
            "t_min_c": "et0_t_min_c",
            "rh_mean_pct": "et0_rh_mean_pct",
            "p_kpa": "et0_pressure_kpa",
            "vpd_kpa": "et0_vpd_kpa",
        }
    )
    return et0.sort_values("date").reset_index(drop=True)


def build_official_water_loss_table() -> pd.DataFrame:
    rows = [
        {
            "year": 2014,
            "system_input_m3_year": 924_448_577.0,
            "authorized_consumption_m3_year": 702_513_223.0,
            "nrw_m3_year": 221_935_354.0,
            "nrw_pct": 24.01,
            "administrative_loss_m3_year": 19_158_774.0,
            "administrative_loss_pct": 2.07,
            "physical_loss_m3_year": 202_776_580.0,
            "physical_loss_pct": 21.93,
            "source_pdf": "https://cdn.iski.istanbul/uploads/Su_Kayiplari_2014_2399e81de3.pdf",
        },
        {
            "year": 2016,
            "system_input_m3_year": 998_622_627.0,
            "authorized_consumption_m3_year": 758_279_029.0,
            "nrw_m3_year": 240_343_588.0,
            "nrw_pct": 24.07,
            "administrative_loss_m3_year": 20_410_808.0,
            "administrative_loss_pct": 2.04,
            "physical_loss_m3_year": 219_932_780.0,
            "physical_loss_pct": 22.03,
            "source_pdf": "https://cdn.iski.istanbul/uploads/Icmesuyu_Temini_ve_Dagitim_Sistemlerindeki_Su_Kayiplari_Yillik_Raporlari_2016_f790d8c849.pdf",
        },
        {
            "year": 2017,
            "system_input_m3_year": 1_020_647_179.0,
            "authorized_consumption_m3_year": 779_668_722.0,
            "nrw_m3_year": 240_978_457.0,
            "nrw_pct": 23.60,
            "administrative_loss_m3_year": 22_802_169.0,
            "administrative_loss_pct": 2.20,
            "physical_loss_m3_year": np.nan,
            "physical_loss_pct": np.nan,
            "source_pdf": "https://cdn.iski.istanbul/uploads/Icmesuyu_Temini_ve_Dagitim_Sistemlerindeki_Su_Kayiplari_Yillik_Raporlari_2017_3bb7d1f40d.pdf",
        },
        {
            "year": 2020,
            "system_input_m3_year": 1_074_133_977.0,
            "authorized_consumption_m3_year": 852_032_565.0,
            "nrw_m3_year": 222_101_412.0,
            "nrw_pct": 20.68,
            "administrative_loss_m3_year": 22_556_814.0,
            "administrative_loss_pct": 2.10,
            "physical_loss_m3_year": 199_544_598.0,
            "physical_loss_pct": 18.57,
            "source_pdf": "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2020_9a984f0ba7.pdf",
        },
        {
            "year": 2021,
            "system_input_m3_year": 1_073_990_361.0,
            "authorized_consumption_m3_year": 853_584_040.0,
            "nrw_m3_year": 220_406_321.0,
            "nrw_pct": 20.52,
            "administrative_loss_m3_year": 22_553_798.0,
            "administrative_loss_pct": 2.10,
            "physical_loss_m3_year": 197_852_523.0,
            "physical_loss_pct": 18.42,
            "source_pdf": "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2021_9e4b97ee29.pdf",
        },
        {
            "year": 2022,
            "system_input_m3_year": 1_103_672_069.0,
            "authorized_consumption_m3_year": 889_046_470.0,
            "nrw_m3_year": 214_625_600.0,
            "nrw_pct": 19.45,
            "administrative_loss_m3_year": 20_969_769.0,
            "administrative_loss_pct": 1.90,
            "physical_loss_m3_year": 193_655_830.0,
            "physical_loss_pct": 17.55,
            "source_pdf": "https://cdn.iski.istanbul/uploads/Su_Denge_Tablosu_2022_46b2a9477c.pdf",
        },
        {
            "year": 2023,
            "system_input_m3_year": 1_117_064_116.0,
            "authorized_consumption_m3_year": 905_522_094.0,
            "nrw_m3_year": 211_542_022.0,
            "nrw_pct": 18.94,
            "administrative_loss_m3_year": 20_762_457.0,
            "administrative_loss_pct": 1.86,
            "physical_loss_m3_year": 190_779_565.0,
            "physical_loss_pct": 17.08,
            "source_pdf": "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2023_4c07821536.pdf",
        },
        {
            "year": 2024,
            "system_input_m3_year": 1_161_020_209.0,
            "authorized_consumption_m3_year": 944_695_142.0,
            "nrw_m3_year": 216_325_067.0,
            "nrw_pct": 18.63,
            "administrative_loss_m3_year": 24_840_404.0,
            "administrative_loss_pct": 2.14,
            "physical_loss_m3_year": 191_484_663.0,
            "physical_loss_pct": 16.49,
            "source_pdf": "https://cdn.iski.istanbul/uploads/Su_Kayiplari_2024_cdd097df1e.pdf",
        },
    ]
    df = pd.DataFrame(rows)
    df["authorized_share_pct"] = 100.0 - df["nrw_pct"]
    return df


def load_weighted_occupancy(history_csv: Path) -> pd.DataFrame:
    hist = pd.read_csv(history_csv)
    hist["date"] = pd.to_datetime(hist["ds"])
    dam_cols = [col for col in CAPACITIES_MCM if col in hist.columns]
    weights = np.array([CAPACITIES_MCM[c] for c in dam_cols], dtype=float)
    hist["weighted_total_fill"] = hist[dam_cols].mul(weights, axis=1).sum(axis=1) / weights.sum()
    return hist[["date", "overall_mean", "weighted_total_fill"]].sort_values("date").reset_index(drop=True)


def load_consumption(exog_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(exog_csv)
    df["date"] = pd.to_datetime(df["ds"])
    return df[["date", "rain_sum_monthly", "rain_mean_monthly", "consumption_mean_monthly"]].sort_values("date").reset_index(drop=True)


def build_monthly_climate_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_daily = load_historical_daily(args.temp_xlsx, args.humidity_xlsx, args.pressure_xlsx)
    hist_monthly = aggregate_monthly_daily_observed(hist_daily)
    vpd_monthly = aggregate_monthly_vpd(args.vpd_daily_csv)
    auto_daily = load_auto_daily(args.auto_table1, args.auto_table2)
    auto_monthly = aggregate_monthly_auto(auto_daily)
    wind_hist_monthly = aggregate_monthly_historical_wind(args.wind_daily_csv)
    rain_monthly = load_monthly_rain(args.rain_monthly_csv)
    et0_monthly = load_monthly_et0(args.et0_monthly_csv)

    panel = hist_monthly.merge(vpd_monthly, on="date", how="outer")
    panel = panel.merge(wind_hist_monthly, on="date", how="outer")
    panel = panel.merge(auto_monthly, on="date", how="outer")
    panel = panel.merge(rain_monthly, on="date", how="outer")
    panel = panel.merge(et0_monthly, on="date", how="outer")
    panel = panel.sort_values("date").reset_index(drop=True)

    panel["month"] = panel["date"].dt.month
    panel["year"] = panel["date"].dt.year
    panel["climate_water_balance_mm"] = panel["rain_mm"] - panel["et0_mm_month"]
    panel["dryness_deficit_mm"] = panel["et0_mm_month"] - panel["rain_mm"]
    panel["rain_to_et0_ratio"] = np.where(
        panel["et0_mm_month"].gt(0),
        panel["rain_mm"] / panel["et0_mm_month"],
        np.nan,
    )
    panel["vpd_et0_interaction"] = panel["vpd_kpa_mean"] * panel["et0_mm_month"]
    panel["core_obs_score"] = panel[
        ["t_mean_c", "rh_mean_pct", "pressure_kpa", "vpd_kpa_mean", "rain_mm", "et0_mm_month"]
    ].notna().sum(axis=1)

    water_loss_annual = build_official_water_loss_table()
    panel = panel.merge(water_loss_annual, on="year", how="left")
    return panel, water_loss_annual


def build_driver_panel(monthly_climate: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    occupancy = load_weighted_occupancy(args.occupancy_history)
    exog = load_consumption(args.exog_monthly_csv)

    driver = occupancy.merge(exog, on="date", how="outer")
    driver = driver.merge(monthly_climate, on="date", how="left")
    driver = driver.sort_values("date").reset_index(drop=True)

    driver["date"] = pd.to_datetime(driver["date"])
    driver["month"] = driver["date"].dt.month
    driver["month_sin"] = np.sin(2 * np.pi * driver["month"] / 12.0)
    driver["month_cos"] = np.cos(2 * np.pi * driver["month"] / 12.0)
    driver["rain_model_mm"] = driver["rain_mm"].fillna(driver["rain_sum_monthly"])
    driver["temp_proxy_c"] = driver["t_mean_c"].combine_first(driver["auto_t_mean_c"]).combine_first(driver["et0_t_mean_c"])
    driver["rh_proxy_pct"] = driver["rh_mean_pct"].combine_first(driver["auto_rh_mean_pct"]).combine_first(driver["et0_rh_mean_pct"])
    driver["pressure_proxy_kpa"] = driver["pressure_kpa"].combine_first(driver["auto_pressure_kpa"])
    driver["water_balance_proxy_mm"] = driver["rain_model_mm"] - driver["et0_mm_month"]
    driver["dam_core_available"] = driver[
        ["weighted_total_fill", "consumption_mean_monthly", "rain_model_mm", "et0_mm_month"]
    ].notna().all(axis=1)
    driver["deep_climate_available"] = driver[
        ["weighted_total_fill", "consumption_mean_monthly", "rain_model_mm", "et0_mm_month", "vpd_kpa_mean", "temp_proxy_c"]
    ].notna().all(axis=1)
    return driver


def segment_spans(values: pd.Series, dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    mask = values.notna().to_numpy(dtype=bool)
    if mask.size == 0:
        return []
    dts = pd.to_datetime(dates).reset_index(drop=True)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start_idx: int | None = None
    for idx, present in enumerate(mask):
        if present and start_idx is None:
            start_idx = idx
        if start_idx is not None and (idx == len(mask) - 1 or not mask[idx + 1]):
            spans.append((dts.iloc[start_idx], dts.iloc[idx]))
            start_idx = None
    return spans


def plot_feature_coverage(monthly_climate: pd.DataFrame, driver: pd.DataFrame, out_path: Path) -> None:
    coverage_df = driver.merge(
        monthly_climate[["date", "rain_mm", "t_mean_c", "rh_mean_pct", "pressure_kpa", "vpd_kpa_mean", "et0_mm_month", "wind_historical_index_mean", "auto_wind_m_s_mean", "nrw_pct"]],
        on="date",
        how="outer",
        suffixes=("", "_dup"),
    )
    coverage_df = coverage_df.sort_values("date").reset_index(drop=True)
    coverage_df = coverage_df.loc[coverage_df["date"].notna()].copy()

    variables = [
        ("rain_mm", "Yagis"),
        ("t_mean_c", "Sicaklik"),
        ("rh_mean_pct", "Nem"),
        ("pressure_kpa", "Basinc"),
        ("vpd_kpa_mean", "VPD"),
        ("et0_mm_month", "ET0"),
        ("wind_historical_index_mean", "Ruzgar tarihsel"),
        ("auto_wind_m_s_mean", "Ruzgar otomatik"),
        ("weighted_total_fill", "Doluluk"),
        ("consumption_mean_monthly", "Tuketim"),
        ("nrw_pct", "Resmi kayip"),
    ]

    colors = {
        "rain_mm": "#2563eb",
        "t_mean_c": "#dc2626",
        "rh_mean_pct": "#0891b2",
        "pressure_kpa": "#7c3aed",
        "vpd_kpa_mean": "#f97316",
        "et0_mm_month": "#059669",
        "wind_historical_index_mean": "#6b7280",
        "auto_wind_m_s_mean": "#0f766e",
        "weighted_total_fill": "#111827",
        "consumption_mean_monthly": "#be123c",
        "nrw_pct": "#92400e",
    }

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    base = pd.Timestamp("1910-01-01")
    for y, (col, label) in enumerate(variables):
        for start, end in segment_spans(coverage_df[col], coverage_df["date"]):
            x0 = (start - base).days
            width = (end - start).days + 30
            ax.broken_barh([(x0, width)], (y - 0.35, 0.7), facecolors=colors[col])

    tick_dates = pd.date_range("1912-01-01", coverage_df["date"].max(), freq="10YS")
    ax.set_xticks([(d - base).days for d in tick_dates])
    ax.set_xticklabels([str(d.year) for d in tick_dates], rotation=0)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels([label for _, label in variables])
    ax.set_title("Aylik veri kapsami: gozlem ve resmi yardimci seriler")
    ax.set_xlabel("Yil")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_annual_water_balance(monthly_climate: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    annual = (
        monthly_climate.loc[monthly_climate["rain_mm"].notna() & monthly_climate["et0_mm_month"].notna(), ["date", "rain_mm", "et0_mm_month"]]
        .assign(year=lambda d: pd.to_datetime(d["date"]).dt.year)
        .groupby("year", as_index=False)
        .agg(rain_mm_year=("rain_mm", "sum"), et0_mm_year=("et0_mm_month", "sum"))
    )
    annual["climate_water_balance_mm_year"] = annual["rain_mm_year"] - annual["et0_mm_year"]
    annual = annual[annual["year"] <= 2021].copy()

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    colors = np.where(annual["climate_water_balance_mm_year"] >= 0, "#0f766e", "#b91c1c")
    ax.bar(annual["year"], annual["climate_water_balance_mm_year"], color=colors, width=0.85, alpha=0.9)
    ax.plot(annual["year"], annual["climate_water_balance_mm_year"].rolling(10, min_periods=3).mean(), color="#111827", linewidth=2.2, label="10-yil hareketli ortalama")
    ax.axhline(0, color="#475569", linewidth=1.0)
    ax.set_title("Yillik iklim su dengesi: yagis - ET0 (1912-2021)")
    ax.set_xlabel("Yil")
    ax.set_ylabel("mm/yil")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return annual


def plot_water_loss_trend(water_loss_annual: pd.DataFrame, out_path: Path) -> None:
    df = water_loss_annual.sort_values("year").copy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(df["year"], df["nrw_pct"], marker="o", linewidth=2.0, color="#1d4ed8", label="Toplam su kaybi")
    ax.plot(df["year"], df["physical_loss_pct"], marker="o", linewidth=1.8, color="#dc2626", label="Fiziki kayip")
    ax.plot(df["year"], df["administrative_loss_pct"], marker="o", linewidth=1.8, color="#7c3aed", label="Idari kayip")
    ax.set_title("ISKI resmi su kaybi serisi (yillik, secili yillar)")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Yuzde")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(monthly_climate: pd.DataFrame, driver: pd.DataFrame, water_loss_annual: pd.DataFrame) -> dict[str, object]:
    def variable_summary(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for col in cols:
            ser = df[col]
            valid = df.loc[ser.notna(), "date"]
            out[col] = {
                "non_null_rows": int(ser.notna().sum()),
                "date_min": str(valid.min().date()) if not valid.empty else None,
                "date_max": str(valid.max().date()) if not valid.empty else None,
            }
        return out

    common_deep = driver.loc[driver["deep_climate_available"], "date"]
    common_base = driver.loc[driver["dam_core_available"], "date"]

    summary = {
        "generated_at": pd.Timestamp.now(tz="Europe/Istanbul").isoformat(),
        "monthly_climate_rows": int(len(monthly_climate)),
        "monthly_climate_date_min": str(monthly_climate["date"].min().date()),
        "monthly_climate_date_max": str(monthly_climate["date"].max().date()),
        "driver_rows": int(len(driver)),
        "driver_date_min": str(driver["date"].min().date()),
        "driver_date_max": str(driver["date"].max().date()),
        "dam_core_window": {
            "rows": int(driver["dam_core_available"].sum()),
            "date_min": str(common_base.min().date()) if not common_base.empty else None,
            "date_max": str(common_base.max().date()) if not common_base.empty else None,
        },
        "deep_climate_window": {
            "rows": int(driver["deep_climate_available"].sum()),
            "date_min": str(common_deep.min().date()) if not common_deep.empty else None,
            "date_max": str(common_deep.max().date()) if not common_deep.empty else None,
        },
        "variable_coverage": variable_summary(
            driver,
            [
                "weighted_total_fill",
                "consumption_mean_monthly",
                "rain_model_mm",
                "et0_mm_month",
                "t_mean_c",
                "temp_proxy_c",
                "rh_mean_pct",
                "rh_proxy_pct",
                "pressure_kpa",
                "pressure_proxy_kpa",
                "vpd_kpa_mean",
                "wind_historical_index_mean",
                "auto_wind_m_s_mean",
                "nrw_pct",
            ],
        ),
        "water_loss_years": water_loss_annual["year"].astype(int).tolist(),
        "notes": [
            "Tarihsel ruzgar serisi onceki Kandilli ET0 paketinden alinmistir ve birim belirsizligi nedeniyle ayri tutulmustur.",
            "Otomatik istasyon ruzgari m/s birimindedir; 2021-09 sonrasini guclendirir ancak tarihsel ruzgarla ayni kolonda birlestirilmemistir.",
            "Resmi ISKI su kaybi serisi secili yillarda yillik form olarak mevcuttur; aylik modele ancak yillik proxy olarak eklenebilir.",
            "Rain modeli icin once new data yagis kullanildi, yoksa exogenous joined table fallback olarak tutuldu.",
        ],
    }
    return summary


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    tables_dir = ensure_dir(out_dir / "tables")
    figures_dir = ensure_dir(out_dir / "figures")

    monthly_climate, water_loss_annual = build_monthly_climate_panel(args)
    driver_panel = build_driver_panel(monthly_climate, args)
    annual_water_balance = plot_annual_water_balance(monthly_climate, figures_dir / "annual_water_balance_1912_2021.png")
    plot_feature_coverage(monthly_climate, driver_panel, figures_dir / "feature_coverage_timeline.png")
    plot_water_loss_trend(water_loss_annual, figures_dir / "official_water_loss_trend.png")

    monthly_climate.to_csv(tables_dir / "istanbul_newdata_monthly_climate_panel.csv", index=False)
    driver_panel.to_csv(tables_dir / "istanbul_dam_driver_panel.csv", index=False)
    water_loss_annual.to_csv(tables_dir / "official_iski_water_loss_annual.csv", index=False)
    annual_water_balance.to_csv(tables_dir / "annual_climate_water_balance_1912_2021.csv", index=False)

    summary = build_summary(monthly_climate, driver_panel, water_loss_annual)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(tables_dir / "istanbul_newdata_monthly_climate_panel.csv")
    print(tables_dir / "istanbul_dam_driver_panel.csv")
    print(tables_dir / "official_iski_water_loss_annual.csv")
    print(figures_dir / "feature_coverage_timeline.png")
    print(figures_dir / "annual_water_balance_1912_2021.png")
    print(figures_dir / "official_water_loss_trend.png")
    print(out_dir / "summary.json")


if __name__ == "__main__":
    main()
