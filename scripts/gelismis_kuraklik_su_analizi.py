#!/usr/bin/env python3
"""
Gelismis yagis-sicaklik kuraklik ve su kaynaklari analizi.

Urettigi ciktilar:
- merged_monthly_with_risk_flags.csv
- annual_metrics.csv
- period_summary.csv
- early_warning_dashboard.csv
- drought_episodes_spi12.csv
- future_alert_calendar_monthly.csv
- future_alert_calendar_yearly.csv
- data_quality_summary.csv
- trend_diagnostics.csv
- spi_reliability_diagnostics.csv
- spi_sensitivity_baseline_windows.csv
- meteo_hydro_lag_correlation.csv
- meteo_hydro_lag_summary.csv
- literature_consistency_check.csv
- halk_dili_ozet.md
- seasonal_comparison.csv
- top_risk_years_wsi.csv
- gelismis_kuraklik_su_analizi.md
- charts/*.png (matplotlib mevcutsa)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import gamma as gamma_dist
    from scipy.stats import fisk as fisk_dist, kendalltau, norm as norm_dist, theilslopes

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    gamma_dist = None
    fisk_dist = None
    kendalltau = None
    norm_dist = None
    theilslopes = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gelismis kuraklik-su kaynaklari analizi")
    parser.add_argument(
        "--precip",
        default="output/best_meta_auto_combo_v12_allvars/base_models/quant/forecasts/precip_monthly_quant_to_2035.csv",
        help="Aylik yagis (quant) CSV yolu",
    )
    parser.add_argument(
        "--temp",
        default="output/best_meta_auto_combo_v12_allvars/base_models/quant/forecasts/temp_monthly_quant_to_2035.csv",
        help="Aylik sicaklik (quant) CSV yolu",
    )
    parser.add_argument(
        "--output-dir",
        default="output/analysis_gelismis",
        help="Cikti klasoru",
    )
    parser.add_argument("--baseline-start", type=int, default=1988, help="Referans baslangic yili")
    parser.add_argument("--baseline-end", type=int, default=2018, help="Referans bitis yili")
    parser.add_argument("--future-start", type=int, default=2026, help="Karsilastirma gelecek baslangic yili")
    parser.add_argument("--future-end", type=int, default=2035, help="Karsilastirma gelecek bitis yili")
    parser.add_argument(
        "--latitude",
        type=float,
        default=39.0,
        help="Thornthwaite PET icin enlem (derece, Kuzey +)",
    )
    parser.add_argument(
        "--max-lag-months",
        type=int,
        default=24,
        help="Meteo-hidro lag analizinde test edilecek maksimum gecikme (ay)",
    )
    return parser.parse_args()


def load_quant_series(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"])
    required = {"ds", "actual", "yhat", "yhat_lower", "yhat_upper", "is_forecast"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolon(lar): {sorted(missing)} in {path}")

    out = df[["ds", "actual", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]].copy()
    out = out.rename(
        columns={
            "ds": "timestamp",
            "actual": f"actual_{value_name}",
            "yhat": f"{value_name}_yhat",
            "yhat_lower": f"{value_name}_lower",
            "yhat_upper": f"{value_name}_upper",
            "is_forecast": f"is_forecast_{value_name}",
        }
    )
    return out


def first_non_null(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.where(a.notna(), b)


def build_monthly_frame(precip_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    df = precip_df.merge(temp_df, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)

    df["precip"] = first_non_null(df["actual_precip"], df["precip_yhat"])
    df["temp"] = first_non_null(df["actual_temp"], df["temp_yhat"])

    precip_low_fallback = first_non_null(df["precip_lower"], df["precip_yhat"])
    precip_high_fallback = first_non_null(df["precip_upper"], df["precip_yhat"])
    temp_low_fallback = first_non_null(df["temp_lower"], df["temp_yhat"])
    temp_high_fallback = first_non_null(df["temp_upper"], df["temp_yhat"])

    # Gozlem varsa belirsizlik yok kabul edilir (low/high = actual).
    df["precip_low"] = first_non_null(df["actual_precip"], precip_low_fallback)
    df["precip_high"] = first_non_null(df["actual_precip"], precip_high_fallback)
    df["temp_low"] = first_non_null(df["actual_temp"], temp_low_fallback)
    df["temp_high"] = first_non_null(df["actual_temp"], temp_high_fallback)

    df["all_observed"] = df["actual_precip"].notna() & df["actual_temp"].notna()
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    season_map = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }
    df["season"] = df["month"].map(season_map)
    return df


def safe_div(a: float, b: float) -> float:
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    p = float(np.clip(p, 1e-10, 1.0 - 1e-10))
    if HAS_SCIPY and norm_dist is not None:
        return float(norm_dist.ppf(p))
    return float(NormalDist().inv_cdf(p))


def interval_to_sigma(mean: float, low: float, high: float, z_value: float = 1.96) -> float:
    if np.isnan(mean) or np.isnan(low) or np.isnan(high):
        return np.nan
    width = high - low
    if width <= 0:
        return 0.0
    return max(width / (2.0 * z_value), 1e-9)


def prob_leq(threshold: float, mean: float, sigma: float) -> float:
    if np.isnan(threshold) or np.isnan(mean):
        return np.nan
    if np.isnan(sigma) or sigma <= 0:
        return float(mean <= threshold)
    z = (threshold - mean) / sigma
    return float(np.clip(normal_cdf(z), 0.0, 1.0))


def prob_geq(threshold: float, mean: float, sigma: float) -> float:
    p = prob_leq(threshold, mean, sigma)
    if np.isnan(p):
        return np.nan
    return float(np.clip(1.0 - p, 0.0, 1.0))


def percentile_ranks(reference: pd.Series, values: pd.Series) -> np.ndarray:
    ref = np.sort(reference.dropna().to_numpy(dtype=float))
    vals = values.to_numpy(dtype=float)
    if ref.size == 0:
        return np.full(vals.shape, np.nan, dtype=float)
    left = np.searchsorted(ref, vals, side="left")
    right = np.searchsorted(ref, vals, side="right")
    # mid-rank percentile in [0, 1]
    return (left + right) / (2.0 * ref.size)


def to_markdown_safe(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def classify_wsi(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value >= 0.35:
        return "cok_yuksek"
    if value >= 0.15:
        return "yuksek"
    if value >= -0.15:
        return "orta"
    return "dusuk"


def classify_de_martonne(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value < 10:
        return "kurak"
    if value < 20:
        return "yari_kurak"
    if value < 24:
        return "akdeniz"
    if value < 28:
        return "yari_nemli"
    if value < 35:
        return "nemli"
    if value < 55:
        return "cok_nemli"
    return "asiri_nemli"


def classify_spi_like(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value <= -2.0:
        return "asiri_kurak"
    if value <= -1.5:
        return "siddetli_kurak"
    if value <= -1.0:
        return "orta_kurak"
    if value <= -0.5:
        return "hafif_kurak"
    if value < 0.5:
        return "normal"
    if value < 1.0:
        return "hafif_nemli"
    if value < 1.5:
        return "orta_nemli"
    return "cok_nemli"


def empirical_spi(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = reference[np.isfinite(reference)]
    vals = values.astype(float)
    out = np.full(vals.shape, np.nan, dtype=float)
    if ref.size < 6:
        return out
    ref_sorted = np.sort(ref)
    n = ref_sorted.size
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        rank = np.searchsorted(ref_sorted, v, side="right")
        p = (rank + 0.5) / (n + 1.0)
        out[i] = normal_ppf(p)
    return out


def gamma_spi(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = reference[np.isfinite(reference)]
    vals = values.astype(float)
    out = np.full(vals.shape, np.nan, dtype=float)
    if ref.size < 8:
        return empirical_spi(ref, vals)

    q0 = float(np.mean(ref <= 0))
    positive = ref[ref > 0]
    if positive.size < 6:
        return empirical_spi(ref, vals)

    mean_pos = float(np.mean(positive))
    var_pos = float(np.var(positive, ddof=0))
    if mean_pos <= 0 or var_pos <= 0:
        return empirical_spi(ref, vals)

    shape = (mean_pos * mean_pos) / var_pos
    scale = var_pos / mean_pos
    if shape <= 0 or scale <= 0:
        return empirical_spi(ref, vals)

    for i, x in enumerate(vals):
        if not np.isfinite(x):
            continue
        if x <= 0:
            h = q0
        else:
            if HAS_SCIPY and gamma_dist is not None:
                g = float(gamma_dist.cdf(x, a=shape, loc=0.0, scale=scale))
            else:
                return empirical_spi(ref, vals)
            h = q0 + (1.0 - q0) * g
        out[i] = normal_ppf(float(np.clip(h, 1e-10, 1.0 - 1e-10)))
    return out


MONTH_DAYS = np.array([31.0, 28.25, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0], dtype=float)
MONTH_MID_DOY = np.array([15.0, 46.0, 74.0, 105.0, 135.0, 166.0, 196.0, 227.0, 258.0, 288.0, 319.0, 349.0], dtype=float)


def loglogistic_standard_index(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = reference[np.isfinite(reference)]
    vals = values.astype(float)
    if ref.size < 12 or np.std(ref) <= 0:
        return empirical_spi(ref, vals)
    if not HAS_SCIPY or fisk_dist is None:
        return empirical_spi(ref, vals)

    out = np.full(vals.shape, np.nan, dtype=float)
    try:
        shape, loc, scale = fisk_dist.fit(ref)
        if shape <= 0 or scale <= 0:
            return empirical_spi(ref, vals)
        for i, x in enumerate(vals):
            if not np.isfinite(x):
                continue
            h = float(fisk_dist.cdf(x, c=shape, loc=loc, scale=scale))
            out[i] = normal_ppf(float(np.clip(h, 1e-10, 1.0 - 1e-10)))
        return out
    except Exception:
        return empirical_spi(ref, vals)


def mean_daylight_hours(latitude_deg: float, month: int) -> float:
    lat = float(np.clip(latitude_deg, -66.0, 66.0))
    phi = math.radians(lat)
    j = MONTH_MID_DOY[month - 1]
    delta = 0.409 * math.sin((2.0 * math.pi * j / 365.0) - 1.39)
    arg = float(np.clip(-math.tan(phi) * math.tan(delta), -1.0, 1.0))
    ws = math.acos(arg)
    return float((24.0 / math.pi) * ws)


def thornthwaite_pet_series(
    monthly: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    latitude_deg: float,
) -> np.ndarray:
    baseline = monthly[(monthly["year"] >= baseline_start) & (monthly["year"] <= baseline_end)]
    month_index = list(range(1, 13))
    temp_clim = baseline.groupby("month")["temp"].mean().reindex(month_index)
    fallback_clim = monthly.groupby("month")["temp"].mean().reindex(month_index)
    temp_clim = temp_clim.fillna(fallback_clim).fillna(0.0)

    tpos_clim = np.maximum(temp_clim.to_numpy(dtype=float), 0.0)
    heat_index = float(np.sum((tpos_clim / 5.0) ** 1.514))
    if heat_index <= 0:
        return np.zeros(len(monthly), dtype=float)

    a = (6.75e-7 * (heat_index**3)) - (7.71e-5 * (heat_index**2)) + (1.792e-2 * heat_index) + 0.49239
    daylight = np.array([mean_daylight_hours(latitude_deg, m) for m in month_index], dtype=float)
    k_month = (daylight / 12.0) * (MONTH_DAYS / 30.0)

    pet = np.full(len(monthly), np.nan, dtype=float)
    temp_vals = monthly["temp"].to_numpy(dtype=float)
    month_vals = monthly["month"].to_numpy(dtype=int)
    for i in range(len(monthly)):
        t = temp_vals[i]
        if not np.isfinite(t):
            continue
        t = max(t, 0.0)
        if t <= 0:
            pet[i] = 0.0
            continue
        m = int(month_vals[i])
        pet[i] = 16.0 * k_month[m - 1] * ((10.0 * t / heat_index) ** a)
    return np.clip(pet, 0.0, None)


def monthly_baseline_anomalies(
    monthly: pd.DataFrame, baseline_start: int, baseline_end: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = monthly[(monthly["year"] >= baseline_start) & (monthly["year"] <= baseline_end)].copy()
    normals = (
        baseline.groupby("month")
        .agg(
            precip_norm_mm=("precip", "mean"),
            temp_norm_c=("temp", "mean"),
        )
        .reset_index()
    )
    out = monthly.merge(normals, on="month", how="left")
    out["precip_anom_mm"] = out["precip"] - out["precip_norm_mm"]
    out["temp_anom_c"] = out["temp"] - out["temp_norm_c"]
    out["precip_anom_pct"] = np.where(
        out["precip_norm_mm"] > 0,
        (out["precip_anom_mm"] / out["precip_norm_mm"]) * 100.0,
        np.nan,
    )
    return out, normals


def add_spi_indices(monthly: pd.DataFrame, baseline_start: int, baseline_end: int) -> pd.DataFrame:
    out = monthly.sort_values("timestamp").reset_index(drop=True).copy()
    out["precip_3m_sum"] = out["precip"].rolling(3, min_periods=3).sum()
    out["precip_12m_sum"] = out["precip"].rolling(12, min_periods=12).sum()
    out["spi3"] = np.nan
    out["spi12"] = np.nan

    baseline = out[(out["year"] >= baseline_start) & (out["year"] <= baseline_end)]
    for m in range(1, 13):
        mask_all = out["month"] == m
        mask_ref = baseline["month"] == m

        ref3 = baseline.loc[mask_ref, "precip_3m_sum"].to_numpy(dtype=float)
        val3 = out.loc[mask_all, "precip_3m_sum"].to_numpy(dtype=float)
        out.loc[mask_all, "spi3"] = gamma_spi(ref3, val3)

        ref12 = baseline.loc[mask_ref, "precip_12m_sum"].to_numpy(dtype=float)
        val12 = out.loc[mask_all, "precip_12m_sum"].to_numpy(dtype=float)
        out.loc[mask_all, "spi12"] = gamma_spi(ref12, val12)

    out["spi3_class"] = out["spi3"].map(classify_spi_like)
    out["spi12_class"] = out["spi12"].map(classify_spi_like)
    out["spi3_drought"] = out["spi3"] <= -1.0
    out["spi12_drought"] = out["spi12"] <= -1.0
    out["spi12_severe_drought"] = out["spi12"] <= -1.5
    return out


def add_spei_indices(
    monthly: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    latitude_deg: float,
) -> pd.DataFrame:
    out = monthly.sort_values("timestamp").reset_index(drop=True).copy()
    out["pet_thornthwaite_mm"] = thornthwaite_pet_series(out, baseline_start, baseline_end, latitude_deg)
    out["water_balance_mm"] = out["precip"] - out["pet_thornthwaite_mm"]
    out["water_balance_3m_sum"] = out["water_balance_mm"].rolling(3, min_periods=3).sum()
    out["water_balance_12m_sum"] = out["water_balance_mm"].rolling(12, min_periods=12).sum()
    out["water_balance_24m_sum"] = out["water_balance_mm"].rolling(24, min_periods=24).sum()
    out["spei3"] = np.nan
    out["spei12"] = np.nan
    out["hydro_proxy_index"] = np.nan

    baseline = out[(out["year"] >= baseline_start) & (out["year"] <= baseline_end)]
    for m in range(1, 13):
        mask_all = out["month"] == m
        mask_ref = baseline["month"] == m

        ref3 = baseline.loc[mask_ref, "water_balance_3m_sum"].to_numpy(dtype=float)
        val3 = out.loc[mask_all, "water_balance_3m_sum"].to_numpy(dtype=float)
        out.loc[mask_all, "spei3"] = loglogistic_standard_index(ref3, val3)

        ref12 = baseline.loc[mask_ref, "water_balance_12m_sum"].to_numpy(dtype=float)
        val12 = out.loc[mask_all, "water_balance_12m_sum"].to_numpy(dtype=float)
        out.loc[mask_all, "spei12"] = loglogistic_standard_index(ref12, val12)

        ref24 = baseline.loc[mask_ref, "water_balance_24m_sum"].to_numpy(dtype=float)
        val24 = out.loc[mask_all, "water_balance_24m_sum"].to_numpy(dtype=float)
        out.loc[mask_all, "hydro_proxy_index"] = empirical_spi(ref24, val24)

    out["spei3_class"] = out["spei3"].map(classify_spi_like)
    out["spei12_class"] = out["spei12"].map(classify_spi_like)
    out["spei3_drought"] = out["spei3"] <= -1.0
    out["spei12_drought"] = out["spei12"] <= -1.0
    out["spei12_severe_drought"] = out["spei12"] <= -1.5
    return out


def extract_drought_episodes(
    monthly: pd.DataFrame,
    min_len_months: int = 3,
    spi_col: str = "spi12",
    threshold: float = -1.0,
) -> pd.DataFrame:
    m = monthly.sort_values("timestamp").reset_index(drop=True)
    active = False
    start_idx = 0
    episodes = []

    for i in range(len(m)):
        v = m.loc[i, spi_col]
        is_drought = (not pd.isna(v)) and (v <= threshold)
        if is_drought and not active:
            active = True
            start_idx = i
        elif (not is_drought) and active:
            end_idx = i - 1
            seg = m.loc[start_idx:end_idx]
            length = len(seg)
            if length >= min_len_months:
                episodes.append(
                    {
                        "start": seg["timestamp"].iloc[0],
                        "end": seg["timestamp"].iloc[-1],
                        "months": length,
                        "min_spi": float(seg[spi_col].min()),
                        "mean_spi": float(seg[spi_col].mean()),
                        "mean_precip_mm": float(seg["precip"].mean()),
                        "mean_temp_c": float(seg["temp"].mean()),
                        "severity_score": float(abs(seg[spi_col].min()) * length),
                    }
                )
            active = False

    if active:
        seg = m.loc[start_idx : len(m) - 1]
        length = len(seg)
        if length >= min_len_months:
            episodes.append(
                {
                    "start": seg["timestamp"].iloc[0],
                    "end": seg["timestamp"].iloc[-1],
                    "months": length,
                    "min_spi": float(seg[spi_col].min()),
                    "mean_spi": float(seg[spi_col].mean()),
                    "mean_precip_mm": float(seg["precip"].mean()),
                    "mean_temp_c": float(seg["temp"].mean()),
                    "severity_score": float(abs(seg[spi_col].min()) * length),
                }
            )

    if not episodes:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "months",
                "min_spi",
                "mean_spi",
                "mean_precip_mm",
                "mean_temp_c",
                "severity_score",
            ]
        )
    out = pd.DataFrame(episodes).sort_values(
        ["severity_score", "months", "start"], ascending=[False, False, True]
    )
    out["start_year"] = pd.to_datetime(out["start"]).dt.year
    out["end_year"] = pd.to_datetime(out["end"]).dt.year
    return out.reset_index(drop=True)


def run_length_stats(flags: Iterable[bool]) -> tuple[float, int, int]:
    runs = []
    cur = 0
    for f in flags:
        if f:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    if not runs:
        return 0.0, 0, 0
    return float(np.mean(runs)), int(np.max(runs)), int(len(runs))


def build_annual_metrics(monthly: pd.DataFrame, baseline_start: int, baseline_end: int) -> pd.DataFrame:
    annual = (
        monthly.groupby("year", as_index=False)
        .agg(
            precip_total_mm=("precip", "sum"),
            precip_total_low_mm=("precip_low", "sum"),
            precip_total_high_mm=("precip_high", "sum"),
            temp_mean_c=("temp", "mean"),
            temp_mean_low_c=("temp_low", "mean"),
            temp_mean_high_c=("temp_high", "mean"),
            pet_total_mm=("pet_thornthwaite_mm", "sum"),
            water_balance_total_mm=("water_balance_mm", "sum"),
            spi3_drought_rate=("spi3_drought", "mean"),
            spi12_drought_rate=("spi12_drought", "mean"),
            spi12_severe_rate=("spi12_severe_drought", "mean"),
            spei3_drought_rate=("spei3_drought", "mean"),
            spei12_drought_rate=("spei12_drought", "mean"),
            spei12_severe_rate=("spei12_severe_drought", "mean"),
            hydro_proxy_mean=("hydro_proxy_index", "mean"),
            precip_observed_months=("actual_precip", lambda s: int(s.notna().sum())),
            temp_observed_months=("actual_temp", lambda s: int(s.notna().sum())),
            observed_months=("all_observed", "sum"),
            n_months=("timestamp", "size"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    annual["observed_share"] = annual["observed_months"] / annual["n_months"]
    annual["precip_observed_share"] = annual["precip_observed_months"] / annual["n_months"]
    annual["temp_observed_share"] = annual["temp_observed_months"] / annual["n_months"]

    annual["de_martonne"] = annual["precip_total_mm"] / (annual["temp_mean_c"] + 10.0)
    annual["de_martonne_low"] = annual["precip_total_low_mm"] / (annual["temp_mean_high_c"] + 10.0)
    annual["de_martonne_high"] = annual["precip_total_high_mm"] / (annual["temp_mean_low_c"] + 10.0)

    baseline = annual[(annual["year"] >= baseline_start) & (annual["year"] <= baseline_end)]
    p_mu, p_std = baseline["precip_total_mm"].mean(), baseline["precip_total_mm"].std(ddof=0)
    t_mu, t_std = baseline["temp_mean_c"].mean(), baseline["temp_mean_c"].std(ddof=0)
    dm_mu, dm_std = baseline["de_martonne"].mean(), baseline["de_martonne"].std(ddof=0)

    annual["precip_z_baseline"] = (annual["precip_total_mm"] - p_mu) / p_std if p_std > 0 else np.nan
    annual["temp_z_baseline"] = (annual["temp_mean_c"] - t_mu) / t_std if t_std > 0 else np.nan
    annual["de_martonne_z_baseline"] = (annual["de_martonne"] - dm_mu) / dm_std if dm_std > 0 else np.nan

    # Daha stabil su stresi indeksi:
    # referans donemine gore sicaklik ve yagis yuzdelik farki.
    annual["precip_pct_baseline"] = percentile_ranks(baseline["precip_total_mm"], annual["precip_total_mm"])
    annual["temp_pct_baseline"] = percentile_ranks(baseline["temp_mean_c"], annual["temp_mean_c"])
    annual["water_stress_index"] = annual["temp_pct_baseline"] - annual["precip_pct_baseline"]

    annual["wsi_class"] = annual["water_stress_index"].map(classify_wsi)
    annual["de_martonne_class"] = annual["de_martonne"].map(classify_de_martonne)
    return annual


def build_quality_summary(
    monthly_flags: pd.DataFrame, baseline_start: int, baseline_end: int, future_start: int, future_end: int
) -> pd.DataFrame:
    all_df = monthly_flags.copy()
    base_df = monthly_flags[(monthly_flags["year"] >= baseline_start) & (monthly_flags["year"] <= baseline_end)]
    fut_df = monthly_flags[(monthly_flags["year"] >= future_start) & (monthly_flags["year"] <= future_end)]

    rows = []
    for label, df in [("all_period", all_df), ("baseline_period", base_df), ("future_period", fut_df)]:
        if df.empty:
            continue
        rows.append(
            {
                "period": label,
                "n_months": int(len(df)),
                "precip_observed_share": float(df["actual_precip"].notna().mean()),
                "temp_observed_share": float(df["actual_temp"].notna().mean()),
                "joint_observed_share": float(df["all_observed"].mean()),
                "mean_precip_mm": float(df["precip"].mean()),
                "mean_temp_c": float(df["temp"].mean()),
                "mean_pet_mm": float(df["pet_thornthwaite_mm"].mean()),
                "mean_water_balance_mm": float(df["water_balance_mm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _trend_single(y: np.ndarray, x: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(x)
    yv = y[mask]
    xv = x[mask]
    if yv.size < 8:
        return {
            "n_years": int(yv.size),
            "kendall_tau": np.nan,
            "p_value": np.nan,
            "sen_slope_per_year": np.nan,
            "sen_slope_per_decade": np.nan,
            "slope_low_per_year": np.nan,
            "slope_high_per_year": np.nan,
            "trend_label": "veri_yetersiz",
        }

    if HAS_SCIPY and kendalltau is not None and theilslopes is not None:
        tau, p = kendalltau(xv, yv)
        slope, _intercept, low, high = theilslopes(yv, xv, 0.95)
    else:
        tau, p = np.nan, np.nan
        slope = np.polyfit(xv, yv, 1)[0]
        low, high = np.nan, np.nan

    if np.isnan(p):
        label = "egilim_hesaplandi_p_yok"
    elif p <= 0.05 and slope > 0:
        label = "anlamli_artan"
    elif p <= 0.05 and slope < 0:
        label = "anlamli_azalan"
    else:
        label = "anlamsiz"

    return {
        "n_years": int(yv.size),
        "kendall_tau": float(tau) if not np.isnan(tau) else np.nan,
        "p_value": float(p) if not np.isnan(p) else np.nan,
        "sen_slope_per_year": float(slope),
        "sen_slope_per_decade": float(slope * 10.0),
        "slope_low_per_year": float(low) if not np.isnan(low) else np.nan,
        "slope_high_per_year": float(high) if not np.isnan(high) else np.nan,
        "trend_label": label,
    }


def build_trend_diagnostics(
    annual: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    future_start: int,
    future_end: int,
) -> pd.DataFrame:
    metrics = [
        ("precip_total_mm", "mm/yil"),
        ("temp_mean_c", "C"),
        ("pet_total_mm", "mm/yil"),
        ("water_balance_total_mm", "mm/yil"),
        ("de_martonne", "index"),
        ("water_stress_index", "index"),
        ("spi12_drought_rate", "oran"),
        ("spei12_drought_rate", "oran"),
        ("hydro_proxy_mean", "index"),
    ]
    periods = [
        (f"{baseline_start}-{baseline_end}", baseline_start, baseline_end),
        (f"{future_start}-{future_end}", future_start, future_end),
        (f"{baseline_start}-{future_end}", baseline_start, future_end),
    ]
    rows = []
    for period_label, start, end in periods:
        sub = annual[(annual["year"] >= start) & (annual["year"] <= end)].copy()
        if sub.empty:
            continue
        for metric, unit in metrics:
            if metric not in sub.columns:
                continue
            trend = _trend_single(sub[metric].to_numpy(dtype=float), sub["year"].to_numpy(dtype=float))
            if metric in {"precip_total_mm", "spi12_drought_rate", "spei12_drought_rate"}:
                obs_share = float(sub["precip_observed_share"].mean())
            elif metric in {"temp_mean_c"}:
                obs_share = float(sub["temp_observed_share"].mean())
            else:
                obs_share = float(sub["observed_share"].mean())
            rows.append(
                {
                    "period": period_label,
                    "metric": metric,
                    "unit": unit,
                    "observed_share_mean": obs_share,
                    **trend,
                }
            )
    return pd.DataFrame(rows)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 3:
        return np.nan
    av = a[mask]
    bv = b[mask]
    if np.std(av) <= 0 or np.std(bv) <= 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def build_spi_reliability_diagnostics(
    monthly_flags: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    future_start: int,
    future_end: int,
) -> pd.DataFrame:
    baseline = monthly_flags[(monthly_flags["year"] >= baseline_start) & (monthly_flags["year"] <= baseline_end)]
    future = monthly_flags[(monthly_flags["year"] >= future_start) & (monthly_flags["year"] <= future_end)]
    rows = []
    scale_defs = [
        ("SPI", 3, "precip_3m_sum", "spi3"),
        ("SPI", 12, "precip_12m_sum", "spi12"),
        ("SPEI", 3, "water_balance_3m_sum", "spei3"),
        ("SPEI", 12, "water_balance_12m_sum", "spei12"),
    ]

    for index_type, scale, sum_col, spi_col in scale_defs:
        if sum_col not in baseline.columns or spi_col not in monthly_flags.columns:
            continue
        for month in range(1, 13):
            ref = baseline.loc[baseline["month"] == month, sum_col].to_numpy(dtype=float)
            ref_valid = ref[np.isfinite(ref)]
            n_ref = int(ref_valid.size)
            if index_type == "SPI":
                pos_ref = int(np.sum(ref_valid > 0))
                zero_share = float(np.mean(ref_valid <= 0)) if n_ref > 0 else np.nan
                gamma_fit_possible = bool(
                    n_ref >= 8 and pos_ref >= 6 and np.var(ref_valid[ref_valid > 0], ddof=0) > 0
                )
            else:
                pos_ref = int(np.sum(ref_valid > 0))
                zero_share = np.nan
                gamma_fit_possible = bool(n_ref >= 12 and np.std(ref_valid) > 0)

            score = 100.0
            if n_ref < 20:
                score -= 45.0
            elif n_ref < 30:
                score -= 25.0
            elif n_ref < 50:
                score -= 10.0

            if index_type == "SPI" and not np.isnan(zero_share):
                if zero_share >= 0.50:
                    score -= 20.0
                elif zero_share >= 0.30:
                    score -= 10.0

            if not gamma_fit_possible:
                score -= 20.0
            score = float(np.clip(score, 0.0, 100.0))

            if score >= 80:
                reliability_label = "yuksek"
            elif score >= 60:
                reliability_label = "orta"
            elif score >= 40:
                reliability_label = "sinirli"
            else:
                reliability_label = "dusuk"

            notes = []
            if n_ref < 30:
                notes.append("wmo_30yil_referans_altinda")
            if n_ref < 50:
                notes.append("uzun_seri_tercih_edilir_50_60yil")
            if not gamma_fit_possible:
                if index_type == "SPI":
                    notes.append("gamma_fit_zayif_empirik_fallback")
                else:
                    notes.append("loglogistic_fit_zayif_empirik_fallback")
            if index_type == "SPI" and not np.isnan(zero_share) and zero_share >= 0.30:
                notes.append("sifir_yagis_orani_yuksek")
            note_text = "|".join(notes) if notes else "uygun"

            all_month = monthly_flags["month"] == month
            fut_month = future["month"] == month
            rows.append(
                {
                    "index_type": index_type,
                    "scale_months": scale,
                    "month": month,
                    "baseline_start": baseline_start,
                    "baseline_end": baseline_end,
                    "baseline_sample_n": n_ref,
                    "baseline_positive_n": pos_ref,
                    "baseline_zero_share": zero_share,
                    "gamma_fit_possible": gamma_fit_possible,
                    "spi_valid_share_all": float(monthly_flags.loc[all_month, spi_col].notna().mean()),
                    "spi_valid_share_future": float(future.loc[fut_month, spi_col].notna().mean()) if not future.empty else np.nan,
                    "reliability_score_0_100": score,
                    "reliability_label": reliability_label,
                    "notes": note_text,
                }
            )

    out = pd.DataFrame(rows).sort_values(["index_type", "scale_months", "month"]).reset_index(drop=True)
    return out


def build_spi_sensitivity_baseline_windows(
    monthly_flags: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    future_start: int,
    future_end: int,
    latitude_deg: float,
) -> pd.DataFrame:
    monthly_sorted = monthly_flags.sort_values("timestamp").reset_index(drop=True)
    if monthly_sorted.empty:
        return pd.DataFrame(
            columns=[
                "baseline_window",
                "baseline_start",
                "baseline_end",
                "window_role",
                "n_eval_spi12",
                "mae_spi12",
                "corr_spi12",
                "max_abs_diff_spi12",
                "drought_agreement_spi12",
                "severe_agreement_spi12",
                "drought_rate_pp_delta_spi12",
                "future_drought_rate_pp_delta_spi12",
                "n_eval_spei12",
                "mae_spei12",
                "corr_spei12",
                "max_abs_diff_spei12",
                "drought_agreement_spei12",
                "severe_agreement_spei12",
                "drought_rate_pp_delta_spei12",
                "future_drought_rate_pp_delta_spei12",
                "sensitivity_label_spi",
                "sensitivity_label_spei",
                "sensitivity_label",
            ]
        )

    default_spi3 = monthly_sorted["spi3"].to_numpy(dtype=float)
    default_spi12 = monthly_sorted["spi12"].to_numpy(dtype=float)
    default_spei3 = monthly_sorted["spei3"].to_numpy(dtype=float)
    default_spei12 = monthly_sorted["spei12"].to_numpy(dtype=float)

    window_len = baseline_end - baseline_start + 1
    if window_len <= 0:
        raise ValueError("baseline_end baseline_start'tan buyuk veya esit olmali")

    min_year = int(monthly_sorted["year"].min())
    max_hist_end = min(int(monthly_sorted["year"].max()), future_start - 1)

    candidate_windows = []
    if max_hist_end - min_year + 1 >= window_len:
        for start in range(min_year, max_hist_end - window_len + 2):
            candidate_windows.append((start, start + window_len - 1))

    if (baseline_start, baseline_end) not in candidate_windows:
        candidate_windows.append((baseline_start, baseline_end))

    candidate_windows = sorted(set(candidate_windows))
    earliest = candidate_windows[0]
    latest = candidate_windows[-1]

    future_mask = (monthly_sorted["year"] >= future_start) & (monthly_sorted["year"] <= future_end)

    def sensitivity_label_from_metrics(mae: float, agreement: float) -> str:
        if np.isnan(mae) or np.isnan(agreement):
            return "hesaplanamadi"
        if mae <= 0.10 and agreement >= 0.90:
            return "dusuk_duyarlilik"
        if mae <= 0.20 and agreement >= 0.80:
            return "orta_duyarlilik"
        return "yuksek_duyarlilik"

    rows = []
    for start, end in candidate_windows:
        alt = add_spi_indices(monthly_sorted, start, end)
        alt = add_spei_indices(alt, start, end, latitude_deg).sort_values("timestamp").reset_index(drop=True)
        alt_spi3 = alt["spi3"].to_numpy(dtype=float)
        alt_spi12 = alt["spi12"].to_numpy(dtype=float)
        alt_spei3 = alt["spei3"].to_numpy(dtype=float)
        alt_spei12 = alt["spei12"].to_numpy(dtype=float)

        mask12 = np.isfinite(default_spi12) & np.isfinite(alt_spi12)
        n_eval12 = int(np.sum(mask12))
        if n_eval12 > 0:
            diff12 = alt_spi12[mask12] - default_spi12[mask12]
            mae12 = float(np.mean(np.abs(diff12)))
            max_abs12 = float(np.max(np.abs(diff12)))
            corr12 = _safe_corr(default_spi12, alt_spi12)
            default_dry = default_spi12[mask12] <= -1.0
            alt_dry = alt_spi12[mask12] <= -1.0
            default_severe = default_spi12[mask12] <= -1.5
            alt_severe = alt_spi12[mask12] <= -1.5
            drought_agreement = float(np.mean(default_dry == alt_dry))
            severe_agreement = float(np.mean(default_severe == alt_severe))
            drought_rate_pp_delta = float((np.mean(alt_dry) - np.mean(default_dry)) * 100.0)
        else:
            mae12 = np.nan
            max_abs12 = np.nan
            corr12 = np.nan
            drought_agreement = np.nan
            severe_agreement = np.nan
            drought_rate_pp_delta = np.nan

        fut_mask12 = mask12 & future_mask.to_numpy(dtype=bool)
        if int(np.sum(fut_mask12)) > 0:
            fut_default_dry = default_spi12[fut_mask12] <= -1.0
            fut_alt_dry = alt_spi12[fut_mask12] <= -1.0
            fut_delta_pp = float((np.mean(fut_alt_dry) - np.mean(fut_default_dry)) * 100.0)
        else:
            fut_delta_pp = np.nan

        mask3 = np.isfinite(default_spi3) & np.isfinite(alt_spi3)
        if int(np.sum(mask3)) > 0:
            mae3 = float(np.mean(np.abs(alt_spi3[mask3] - default_spi3[mask3])))
            corr3 = _safe_corr(default_spi3, alt_spi3)
        else:
            mae3 = np.nan
            corr3 = np.nan

        mask_spei12 = np.isfinite(default_spei12) & np.isfinite(alt_spei12)
        n_eval_spei12 = int(np.sum(mask_spei12))
        if n_eval_spei12 > 0:
            diff_spei12 = alt_spei12[mask_spei12] - default_spei12[mask_spei12]
            mae_spei12 = float(np.mean(np.abs(diff_spei12)))
            max_abs_spei12 = float(np.max(np.abs(diff_spei12)))
            corr_spei12 = _safe_corr(default_spei12, alt_spei12)
            default_spei_dry = default_spei12[mask_spei12] <= -1.0
            alt_spei_dry = alt_spei12[mask_spei12] <= -1.0
            default_spei_severe = default_spei12[mask_spei12] <= -1.5
            alt_spei_severe = alt_spei12[mask_spei12] <= -1.5
            drought_agreement_spei12 = float(np.mean(default_spei_dry == alt_spei_dry))
            severe_agreement_spei12 = float(np.mean(default_spei_severe == alt_spei_severe))
            drought_rate_pp_delta_spei12 = float((np.mean(alt_spei_dry) - np.mean(default_spei_dry)) * 100.0)
        else:
            mae_spei12 = np.nan
            max_abs_spei12 = np.nan
            corr_spei12 = np.nan
            drought_agreement_spei12 = np.nan
            severe_agreement_spei12 = np.nan
            drought_rate_pp_delta_spei12 = np.nan

        fut_mask_spei12 = mask_spei12 & future_mask.to_numpy(dtype=bool)
        if int(np.sum(fut_mask_spei12)) > 0:
            fut_default_spei_dry = default_spei12[fut_mask_spei12] <= -1.0
            fut_alt_spei_dry = alt_spei12[fut_mask_spei12] <= -1.0
            fut_delta_pp_spei12 = float((np.mean(fut_alt_spei_dry) - np.mean(fut_default_spei_dry)) * 100.0)
        else:
            fut_delta_pp_spei12 = np.nan

        mask_spei3 = np.isfinite(default_spei3) & np.isfinite(alt_spei3)
        if int(np.sum(mask_spei3)) > 0:
            mae_spei3 = float(np.mean(np.abs(alt_spei3[mask_spei3] - default_spei3[mask_spei3])))
            corr_spei3 = _safe_corr(default_spei3, alt_spei3)
        else:
            mae_spei3 = np.nan
            corr_spei3 = np.nan

        sensitivity_label_spi = sensitivity_label_from_metrics(mae12, drought_agreement)
        sensitivity_label_spei = sensitivity_label_from_metrics(mae_spei12, drought_agreement_spei12)
        if "yuksek_duyarlilik" in {sensitivity_label_spi, sensitivity_label_spei}:
            sensitivity_label = "yuksek_duyarlilik"
        elif "orta_duyarlilik" in {sensitivity_label_spi, sensitivity_label_spei}:
            sensitivity_label = "orta_duyarlilik"
        elif "hesaplanamadi" in {sensitivity_label_spi, sensitivity_label_spei}:
            sensitivity_label = "kismi_hesaplanamadi"
        else:
            sensitivity_label = "dusuk_duyarlilik"

        if (start, end) == (baseline_start, baseline_end):
            role = "user_baseline"
        elif (start, end) == earliest:
            role = "earliest_window"
        elif (start, end) == latest:
            role = "latest_window"
        else:
            role = "alternative_window"

        rows.append(
            {
                "baseline_window": f"{start}-{end}",
                "baseline_start": start,
                "baseline_end": end,
                "window_role": role,
                "n_eval_spi3": int(np.sum(mask3)),
                "mae_spi3": mae3,
                "corr_spi3": corr3,
                "n_eval_spi12": n_eval12,
                "mae_spi12": mae12,
                "corr_spi12": corr12,
                "max_abs_diff_spi12": max_abs12,
                "drought_agreement_spi12": drought_agreement,
                "severe_agreement_spi12": severe_agreement,
                "drought_rate_pp_delta_spi12": drought_rate_pp_delta,
                "future_drought_rate_pp_delta_spi12": fut_delta_pp,
                "n_eval_spei3": int(np.sum(mask_spei3)),
                "mae_spei3": mae_spei3,
                "corr_spei3": corr_spei3,
                "n_eval_spei12": n_eval_spei12,
                "mae_spei12": mae_spei12,
                "corr_spei12": corr_spei12,
                "max_abs_diff_spei12": max_abs_spei12,
                "drought_agreement_spei12": drought_agreement_spei12,
                "severe_agreement_spei12": severe_agreement_spei12,
                "drought_rate_pp_delta_spei12": drought_rate_pp_delta_spei12,
                "future_drought_rate_pp_delta_spei12": fut_delta_pp_spei12,
                "sensitivity_label_spi": sensitivity_label_spi,
                "sensitivity_label_spei": sensitivity_label_spei,
                "sensitivity_label": sensitivity_label,
            }
        )

    return pd.DataFrame(rows).sort_values(["baseline_start", "baseline_end"]).reset_index(drop=True)


def build_meteo_hydro_lag_diagnostics(
    monthly_flags: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    future_start: int,
    future_end: int,
    max_lag_months: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = {"hydro_proxy_index", "spi3", "spi12", "spei3", "spei12"}
    if any(c not in monthly_flags.columns for c in needed):
        empty_detail = pd.DataFrame(
            columns=["period", "meteo_index", "lag_months", "n_pairs", "corr_meteo_vs_hydro_proxy"]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "period",
                "meteo_index",
                "best_lag_months",
                "best_corr",
                "corr_lag0",
                "n_pairs_at_best_lag",
                "lag_strength_label",
            ]
        )
        return empty_detail, empty_summary

    periods = [
        (f"{baseline_start}-{baseline_end}", baseline_start, baseline_end),
        (f"{future_start}-{future_end}", future_start, future_end),
        (f"{baseline_start}-{future_end}", baseline_start, future_end),
    ]
    meteo_indices = ["spi3", "spi12", "spei3", "spei12"]

    detail_rows = []
    summary_rows = []
    for period_label, start, end in periods:
        sub = monthly_flags[(monthly_flags["year"] >= start) & (monthly_flags["year"] <= end)].copy()
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        if sub.empty:
            continue

        hydro = sub["hydro_proxy_index"].to_numpy(dtype=float)
        for met in meteo_indices:
            met_arr = sub[met].to_numpy(dtype=float)
            best_lag = np.nan
            best_corr = np.nan
            best_n = 0
            corr_lag0 = np.nan

            for lag in range(0, max_lag_months + 1):
                if lag == 0:
                    a = met_arr
                    b = hydro
                else:
                    if len(met_arr) - lag < 8:
                        continue
                    a = met_arr[:-lag]
                    b = hydro[lag:]

                mask = np.isfinite(a) & np.isfinite(b)
                n_pairs = int(np.sum(mask))
                if n_pairs < 8:
                    corr = np.nan
                else:
                    corr = _safe_corr(a, b)

                detail_rows.append(
                    {
                        "period": period_label,
                        "meteo_index": met,
                        "lag_months": lag,
                        "n_pairs": n_pairs,
                        "corr_meteo_vs_hydro_proxy": corr,
                    }
                )
                if lag == 0:
                    corr_lag0 = corr
                if np.isfinite(corr) and (np.isnan(best_corr) or corr > best_corr):
                    best_corr = float(corr)
                    best_lag = int(lag)
                    best_n = n_pairs

            if np.isnan(best_corr):
                lag_strength = "hesaplanamadi"
            elif best_corr >= 0.70:
                lag_strength = "cok_guclu"
            elif best_corr >= 0.50:
                lag_strength = "guclu"
            elif best_corr >= 0.30:
                lag_strength = "orta"
            elif best_corr >= 0.10:
                lag_strength = "zayif"
            else:
                lag_strength = "cok_zayif_veya_ters"

            summary_rows.append(
                {
                    "period": period_label,
                    "meteo_index": met,
                    "best_lag_months": best_lag,
                    "best_corr": best_corr,
                    "corr_lag0": corr_lag0,
                    "n_pairs_at_best_lag": best_n,
                    "lag_strength_label": lag_strength,
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["period", "meteo_index", "lag_months"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["period", "meteo_index"]).reset_index(drop=True)
    return detail_df, summary_df


def _get_period_row(df: pd.DataFrame, period_label: str, fallback: str = "head") -> pd.Series:
    hit = df[df["period"] == period_label]
    if not hit.empty:
        return hit.iloc[0]
    if fallback == "tail":
        return df.tail(1).iloc[0]
    return df.head(1).iloc[0]


def _get_trend_slope_per_decade(trend_diagnostics: pd.DataFrame, period: str, metric: str) -> float:
    hit = trend_diagnostics[(trend_diagnostics["period"] == period) & (trend_diagnostics["metric"] == metric)]
    if hit.empty:
        return np.nan
    return float(hit["sen_slope_per_decade"].iloc[0])


def _status_from_value(value: float, up_good: bool = True, strong: float = 0.0, weak: float = 0.0) -> str:
    if np.isnan(value):
        return "hesaplanamadi"
    x = value if up_good else -value
    if x >= strong:
        return "uyumlu"
    if x >= weak:
        return "kismi_uyumlu"
    return "celiski_olasi"


def build_literature_consistency_check(
    period_summary: pd.DataFrame,
    trend_diagnostics: pd.DataFrame,
    lag_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    spi_sensitivity: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    future_start: int,
    future_end: int,
) -> pd.DataFrame:
    if period_summary.empty:
        return pd.DataFrame(
            columns=["topic", "literature_expectation", "model_result", "status", "comment", "priority"]
        )

    base_label = f"{baseline_start}-{baseline_end}"
    fut_label = f"{future_start}-{future_end}"
    all_label = f"{baseline_start}-{future_end}"

    base_row = _get_period_row(period_summary, base_label, fallback="head")
    fut_row = _get_period_row(period_summary, fut_label, fallback="tail")

    temp_slope = _get_trend_slope_per_decade(trend_diagnostics, all_label, "temp_mean_c")
    pet_slope = _get_trend_slope_per_decade(trend_diagnostics, all_label, "pet_total_mm")
    wb_slope = _get_trend_slope_per_decade(trend_diagnostics, all_label, "water_balance_total_mm")

    spei_delta = float(fut_row.get("spei12_drought_rate", np.nan) - base_row.get("spei12_drought_rate", np.nan))
    spei_delta_pp = spei_delta * 100.0 if np.isfinite(spei_delta) else np.nan

    lag_hit = lag_summary[(lag_summary["period"] == all_label) & (lag_summary["meteo_index"] == "spi12")]
    best_lag = float(lag_hit["best_lag_months"].iloc[0]) if not lag_hit.empty else np.nan
    best_corr = float(lag_hit["best_corr"].iloc[0]) if not lag_hit.empty else np.nan

    alt = spi_sensitivity[spi_sensitivity["window_role"] != "user_baseline"].copy()
    if alt.empty:
        max_mae_spi12 = np.nan
        max_mae_spei12 = np.nan
    else:
        max_mae_spi12 = float(np.nanmax(alt["mae_spi12"].to_numpy(dtype=float)))
        max_mae_spei12 = float(np.nanmax(alt["mae_spei12"].to_numpy(dtype=float)))

    q_base = quality_summary[quality_summary["period"] == "baseline_period"]
    temp_obs_share = float(q_base["temp_observed_share"].iloc[0]) if not q_base.empty else np.nan

    rows = []

    temp_status = _status_from_value(temp_slope, up_good=True, strong=0.05, weak=0.0)
    rows.append(
        {
            "topic": "Sicaklik egilimi",
            "literature_expectation": "Genel literaturde uzun donemde sicaklik artisi beklenir.",
            "model_result": f"Sen slope={temp_slope:+.3f} C/10yil" if np.isfinite(temp_slope) else "hesaplanamadi",
            "status": temp_status,
            "comment": "Negatif veya sifira yakin trend, genel isinma literaturu ile celisebilir.",
            "priority": 1,
        }
    )

    pet_status = _status_from_value(pet_slope, up_good=True, strong=1.0, weak=0.0)
    rows.append(
        {
            "topic": "PET egilimi",
            "literature_expectation": "Isinma ile PET genelde artar.",
            "model_result": f"Sen slope={pet_slope:+.2f} mm/10yil" if np.isfinite(pet_slope) else "hesaplanamadi",
            "status": pet_status,
            "comment": "PET azalimi cikiyorsa enerji/nem tabanli PET ile tekrar dogrulama gerekir.",
            "priority": 1,
        }
    )

    if np.isnan(spei_delta_pp):
        spei_status = "hesaplanamadi"
    elif spei_delta_pp >= -1.0:
        spei_status = "uyumlu"
    elif spei_delta_pp >= -3.0:
        spei_status = "kismi_uyumlu"
    else:
        spei_status = "celiski_olasi"
    rows.append(
        {
            "topic": "SPEI12 kuraklik orani degisimi",
            "literature_expectation": "Bolgede kuraklik baskisinin artmasi sik gorulur; yerel yagis artislari dengeleyebilir.",
            "model_result": f"Gelecek-baseline farki={spei_delta_pp:+.1f} puan"
            if np.isfinite(spei_delta_pp)
            else "hesaplanamadi",
            "status": spei_status,
            "comment": "Guclu azalis cikarsa, modeldeki yagis artisinin fiziksel tutarliligi ayrica test edilmelidir.",
            "priority": 1,
        }
    )

    if np.isnan(best_lag) or np.isnan(best_corr):
        lag_status = "hesaplanamadi"
    elif 1 <= best_lag <= 12 and best_corr >= 0.30:
        lag_status = "uyumlu"
    elif best_corr >= 0.10:
        lag_status = "kismi_uyumlu"
    else:
        lag_status = "celiski_olasi"
    rows.append(
        {
            "topic": "Meteo-hidro gecikme",
            "literature_expectation": "Meteorolojik sinyalin hidrolojik etkiye 1-12 ay gecikmeli yansimasi beklenir.",
            "model_result": f"En iyi lag={best_lag:.0f} ay, corr={best_corr:.3f}"
            if np.isfinite(best_lag) and np.isfinite(best_corr)
            else "hesaplanamadi",
            "status": lag_status,
            "comment": "Pozitif ve gecikmeli baglanti, literaturdeki gecikme davranisiyla uyumludur.",
            "priority": 2,
        }
    )

    if np.isnan(max_mae_spi12) and np.isnan(max_mae_spei12):
        sens_status = "hesaplanamadi"
    else:
        max_mae = float(np.nanmax(np.array([max_mae_spi12, max_mae_spei12], dtype=float)))
        if max_mae >= 0.10:
            sens_status = "uyumlu"
        elif max_mae > 0:
            sens_status = "kismi_uyumlu"
        else:
            sens_status = "celiski_olasi"
    rows.append(
        {
            "topic": "Baseline pencere duyarliligi",
            "literature_expectation": "SPI/SPEI sonuclari referans pencereye duyarlidir.",
            "model_result": (
                f"max(MAE SPI12,SPEI12)=({max_mae_spi12:.3f},{max_mae_spei12:.3f})"
                if np.isfinite(max_mae_spi12) or np.isfinite(max_mae_spei12)
                else "hesaplanamadi"
            ),
            "status": sens_status,
            "comment": "Duyarlilik gorulmesi beklenen bir durumdur; tek pencere ile karar verilmemelidir.",
            "priority": 2,
        }
    )

    if np.isnan(temp_obs_share):
        quality_status = "hesaplanamadi"
    elif temp_obs_share < 0.20:
        quality_status = "uyumlu"
    elif temp_obs_share < 0.60:
        quality_status = "kismi_uyumlu"
    else:
        quality_status = "celiski_olasi"
    rows.append(
        {
            "topic": "Veri kapsami uyarisi",
            "literature_expectation": "Dusuk gozlem kapsami varsa projeksiyon belirsizligi yuksektir.",
            "model_result": f"Baseline sicaklik gozlem payi={temp_obs_share*100:.1f}%"
            if np.isfinite(temp_obs_share)
            else "hesaplanamadi",
            "status": quality_status,
            "comment": "Bu analizde sicaklik gozlemi cok dusukse fiziksel yorum dikkatli yapilmalidir.",
            "priority": 2,
        }
    )

    out = pd.DataFrame(rows).sort_values(["priority", "topic"]).reset_index(drop=True)
    return out


def build_plain_language_summary(
    period_summary: pd.DataFrame,
    consistency_check: pd.DataFrame,
    lag_summary: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    future_start: int,
    future_end: int,
) -> str:
    base_label = f"{baseline_start}-{baseline_end}"
    fut_label = f"{future_start}-{future_end}"
    all_label = f"{baseline_start}-{future_end}"

    base = _get_period_row(period_summary, base_label, fallback="head")
    fut = _get_period_row(period_summary, fut_label, fallback="tail")

    temp_delta = float(fut.get("annual_temp_c_mean", np.nan) - base.get("annual_temp_c_mean", np.nan))
    precip_pct = np.nan
    if np.isfinite(base.get("annual_precip_mm_mean", np.nan)) and base.get("annual_precip_mm_mean", 0) != 0:
        precip_pct = (
            (fut.get("annual_precip_mm_mean", np.nan) - base.get("annual_precip_mm_mean", np.nan))
            / base.get("annual_precip_mm_mean", np.nan)
            * 100.0
        )
    wb_delta = float(fut.get("annual_water_balance_mm_mean", np.nan) - base.get("annual_water_balance_mm_mean", np.nan))
    spi_delta_pp = float((fut.get("spi12_drought_rate", np.nan) - base.get("spi12_drought_rate", np.nan)) * 100.0)
    spei_delta_pp = float((fut.get("spei12_drought_rate", np.nan) - base.get("spei12_drought_rate", np.nan)) * 100.0)

    lag_hit = lag_summary[(lag_summary["period"] == all_label) & (lag_summary["meteo_index"] == "spi12")]
    best_lag = float(lag_hit["best_lag_months"].iloc[0]) if not lag_hit.empty else np.nan
    best_corr = float(lag_hit["best_corr"].iloc[0]) if not lag_hit.empty else np.nan

    n_conflict = int((consistency_check["status"] == "celiski_olasi").sum()) if not consistency_check.empty else 0
    n_partial = int((consistency_check["status"] == "kismi_uyumlu").sum()) if not consistency_check.empty else 0
    n_ok = int((consistency_check["status"] == "uyumlu").sum()) if not consistency_check.empty else 0

    lines = []
    lines.append("# Halk Dili Ozet")
    lines.append("")
    lines.append("## Kisa Cevap")
    if n_conflict == 0:
        lines.append("- Tahminlerin ana yonu genel literaturle buyuk olcude uyumlu gorunuyor.")
    elif n_conflict <= 2:
        lines.append("- Tahminlerin bir kismi literaturle uyumlu, bir kismi ise celiski riski tasiyor.")
    else:
        lines.append("- Tahminlerde literaturle celiski riski yuksek; sonuclar dikkatli yorumlanmali.")
    lines.append(f"- Uyum kontrolu ozet: uyumlu={n_ok}, kismi={n_partial}, celiski_olasi={n_conflict}.")
    lines.append("")
    lines.append("## Bu Rapor Ne Diyor? (Sade)")
    lines.append(
        f"- Yagis: gelecek donemde yaklasik {precip_pct:+.1f}% degisim."
        if np.isfinite(precip_pct)
        else "- Yagis degisimi hesaplanamadi."
    )
    lines.append(
        f"- Sicaklik: gelecek donemde {temp_delta:+.2f} C degisim."
        if np.isfinite(temp_delta)
        else "- Sicaklik degisimi hesaplanamadi."
    )
    lines.append(
        f"- Su dengesi (P-PET): {wb_delta:+.1f} mm degisim."
        if np.isfinite(wb_delta)
        else "- Su dengesi degisimi hesaplanamadi."
    )
    lines.append(
        f"- SPI12 kurak ay orani: {spi_delta_pp:+.1f} puan degisim."
        if np.isfinite(spi_delta_pp)
        else "- SPI12 degisimi hesaplanamadi."
    )
    lines.append(
        f"- SPEI12 kurak ay orani: {spei_delta_pp:+.1f} puan degisim."
        if np.isfinite(spei_delta_pp)
        else "- SPEI12 degisimi hesaplanamadi."
    )
    lines.append(
        f"- Meteo->hidro gecikme: en guclu sinyal yaklasik {best_lag:.0f} ay gecikmeli (corr={best_corr:.2f})."
        if np.isfinite(best_lag) and np.isfinite(best_corr)
        else "- Meteo->hidro gecikme hesaplanamadi."
    )
    lines.append("")
    lines.append("## Neden Dikkatli Olmaliyiz?")
    lines.append("- Sicaklik gozlem verisi cok dusuk oldugu icin uzun donem fiziksel yorum belirsizdir.")
    lines.append("- Sonuclar model senaryosudur; operasyonel karar icin akim/baraj/yeraltisuyu ile birlikte okunmalidir.")
    lines.append("")
    lines.append("## Pratik Yorum")
    lines.append("- Kisa vadede risk dusuk gorunse bile, kuraklik izleme sistemini (SPI+SPEI+hidro veri) birlikte surdur.")
    lines.append("- Yagis artisi tahminleri tek basina guvence degildir; sicaklik/PET yonu literaturle yeniden dogrulanmali.")
    lines.append("")
    return "\n".join(lines)


def build_monthly_risk_flags(
    monthly: pd.DataFrame, baseline_start: int, baseline_end: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = monthly[(monthly["year"] >= baseline_start) & (monthly["year"] <= baseline_end)].copy()
    month_thresholds = (
        baseline.groupby("month")
        .agg(
            precip_p20=("precip", lambda s: float(s.quantile(0.20))),
            precip_p50=("precip", "median"),
            temp_p80=("temp", lambda s: float(s.quantile(0.80))),
            temp_p50=("temp", "median"),
        )
        .reset_index()
    )
    out = monthly.merge(month_thresholds, on="month", how="left")
    out["dry_month"] = out["precip"] <= out["precip_p20"]
    out["hot_month"] = out["temp"] >= out["temp_p80"]
    out["hot_dry_month"] = out["dry_month"] & out["hot_month"]
    return out, month_thresholds


def summarize_period(
    label: str,
    start: int,
    end: int,
    annual: pd.DataFrame,
    monthly_flags: pd.DataFrame,
) -> dict:
    a = annual[(annual["year"] >= start) & (annual["year"] <= end)].copy()
    m = monthly_flags[(monthly_flags["year"] >= start) & (monthly_flags["year"] <= end)].copy()
    if a.empty or m.empty:
        return {
            "period": label,
            "years": 0,
        }

    dry_mean_run, dry_max_run, dry_runs = run_length_stats(m["dry_month"].tolist())
    hot_dry_mean_run, hot_dry_max_run, hot_dry_runs = run_length_stats(m["hot_dry_month"].tolist())
    spi12_valid = m["spi12"].dropna()
    spi3_valid = m["spi3"].dropna()
    spei12_valid = m["spei12"].dropna()
    spei3_valid = m["spei3"].dropna()

    if len(spi12_valid) > 0:
        spi12_drought_rate = float((spi12_valid <= -1.0).mean())
        spi12_severe_rate = float((spi12_valid <= -1.5).mean())
    else:
        spi12_drought_rate = np.nan
        spi12_severe_rate = np.nan

    if len(spi3_valid) > 0:
        spi3_drought_rate = float((spi3_valid <= -1.0).mean())
    else:
        spi3_drought_rate = np.nan

    if len(spei12_valid) > 0:
        spei12_drought_rate = float((spei12_valid <= -1.0).mean())
        spei12_severe_rate = float((spei12_valid <= -1.5).mean())
    else:
        spei12_drought_rate = np.nan
        spei12_severe_rate = np.nan

    if len(spei3_valid) > 0:
        spei3_drought_rate = float((spei3_valid <= -1.0).mean())
    else:
        spei3_drought_rate = np.nan

    return {
        "period": label,
        "years": len(a),
        "annual_precip_mm_mean": a["precip_total_mm"].mean(),
        "annual_precip_mm_low_mean": a["precip_total_low_mm"].mean(),
        "annual_precip_mm_high_mean": a["precip_total_high_mm"].mean(),
        "annual_temp_c_mean": a["temp_mean_c"].mean(),
        "annual_temp_c_low_mean": a["temp_mean_low_c"].mean(),
        "annual_temp_c_high_mean": a["temp_mean_high_c"].mean(),
        "annual_pet_mm_mean": a["pet_total_mm"].mean(),
        "annual_water_balance_mm_mean": a["water_balance_total_mm"].mean(),
        "de_martonne_mean": a["de_martonne"].mean(),
        "de_martonne_low_mean": a["de_martonne_low"].mean(),
        "de_martonne_high_mean": a["de_martonne_high"].mean(),
        "water_stress_index_mean": a["water_stress_index"].mean(),
        "hydro_proxy_mean": a["hydro_proxy_mean"].mean(),
        "precip_anom_pct_mean": m["precip_anom_pct"].mean(),
        "temp_anom_c_mean": m["temp_anom_c"].mean(),
        "dry_month_rate": m["dry_month"].mean(),
        "hot_month_rate": m["hot_month"].mean(),
        "hot_dry_month_rate": m["hot_dry_month"].mean(),
        "spi3_drought_rate": spi3_drought_rate,
        "spi12_drought_rate": spi12_drought_rate,
        "spi12_severe_rate": spi12_severe_rate,
        "spei3_drought_rate": spei3_drought_rate,
        "spei12_drought_rate": spei12_drought_rate,
        "spei12_severe_rate": spei12_severe_rate,
        "dry_spell_mean_len_month": dry_mean_run,
        "dry_spell_max_len_month": dry_max_run,
        "dry_spell_count": dry_runs,
        "hot_dry_spell_mean_len_month": hot_dry_mean_run,
        "hot_dry_spell_max_len_month": hot_dry_max_run,
        "hot_dry_spell_count": hot_dry_runs,
        "observed_month_share_mean": a["observed_share"].mean(),
    }


def level_by_threshold(value: float, t_high: float, t_medium: float, invert: bool = False) -> str:
    if np.isnan(value):
        return "unknown"
    if invert:
        if value <= t_high:
            return "yuksek"
        if value <= t_medium:
            return "orta"
        return "dusuk"
    if value >= t_high:
        return "yuksek"
    if value >= t_medium:
        return "orta"
    return "dusuk"


def build_early_warning_dashboard(
    baseline_period: dict,
    future_period: dict,
    future_episode_count: int,
    future_episode_max_months: int,
) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "indicator": "water_stress_index_mean",
            "baseline": baseline_period["water_stress_index_mean"],
            "future": future_period["water_stress_index_mean"],
            "delta": future_period["water_stress_index_mean"] - baseline_period["water_stress_index_mean"],
            "risk_level": level_by_threshold(future_period["water_stress_index_mean"], 0.15, -0.15),
            "note": "Pozitif artarsa su stresi artar.",
        }
    )
    rows.append(
        {
            "indicator": "de_martonne_mean",
            "baseline": baseline_period["de_martonne_mean"],
            "future": future_period["de_martonne_mean"],
            "delta": future_period["de_martonne_mean"] - baseline_period["de_martonne_mean"],
            "risk_level": level_by_threshold(future_period["de_martonne_mean"], 24.0, 28.0, invert=True),
            "note": "Dusuk deger daha kurak kosul.",
        }
    )
    rows.append(
        {
            "indicator": "dry_month_rate",
            "baseline": baseline_period["dry_month_rate"],
            "future": future_period["dry_month_rate"],
            "delta": future_period["dry_month_rate"] - baseline_period["dry_month_rate"],
            "risk_level": level_by_threshold(future_period["dry_month_rate"], 0.35, 0.20),
            "note": "P20 altindaki yagisli ay orani.",
        }
    )
    rows.append(
        {
            "indicator": "hot_dry_month_rate",
            "baseline": baseline_period["hot_dry_month_rate"],
            "future": future_period["hot_dry_month_rate"],
            "delta": future_period["hot_dry_month_rate"] - baseline_period["hot_dry_month_rate"],
            "risk_level": level_by_threshold(future_period["hot_dry_month_rate"], 0.15, 0.07),
            "note": "Ayni ay hem sicak hem kuru kosulu.",
        }
    )
    rows.append(
        {
            "indicator": "spi12_severe_rate",
            "baseline": baseline_period["spi12_severe_rate"],
            "future": future_period["spi12_severe_rate"],
            "delta": future_period["spi12_severe_rate"] - baseline_period["spi12_severe_rate"],
            "risk_level": level_by_threshold(future_period["spi12_severe_rate"], 0.10, 0.04),
            "note": "SPI12 <= -1.5 oran.",
        }
    )
    rows.append(
        {
            "indicator": "spei12_severe_rate",
            "baseline": baseline_period["spei12_severe_rate"],
            "future": future_period["spei12_severe_rate"],
            "delta": future_period["spei12_severe_rate"] - baseline_period["spei12_severe_rate"],
            "risk_level": level_by_threshold(future_period["spei12_severe_rate"], 0.10, 0.04),
            "note": "SPEI12 <= -1.5 oran.",
        }
    )
    rows.append(
        {
            "indicator": "future_drought_episode_count",
            "baseline": np.nan,
            "future": future_episode_count,
            "delta": np.nan,
            "risk_level": level_by_threshold(float(future_episode_count), 4.0, 2.0),
            "note": "Gelecek donemde 3+ ay SPI12 kurak donem sayisi.",
        }
    )
    rows.append(
        {
            "indicator": "future_drought_episode_max_len",
            "baseline": np.nan,
            "future": future_episode_max_months,
            "delta": np.nan,
            "risk_level": level_by_threshold(float(future_episode_max_months), 8.0, 4.0),
            "note": "Gelecek donemde en uzun SPI12 kurak donemi (ay).",
        }
    )
    return pd.DataFrame(rows)


def risk_level_from_score(score: float) -> str:
    if np.isnan(score):
        return "unknown"
    if score >= 70:
        return "kritik"
    if score >= 50:
        return "yuksek"
    if score >= 30:
        return "orta"
    return "dusuk"


def build_future_alert_calendar(
    monthly_flags: pd.DataFrame,
    month_thresholds: pd.DataFrame,
    future_start: int,
    future_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fut = monthly_flags[
        (monthly_flags["year"] >= future_start) & (monthly_flags["year"] <= future_end)
    ].copy()
    if fut.empty:
        empty = pd.DataFrame(
            columns=[
                "timestamp",
                "year",
                "month",
                "precip",
                "temp",
                "dry_prob",
                "hot_prob",
                "hot_dry_prob",
                "spi3",
                "spi12",
                "spei3",
                "spei12",
                "risk_score",
                "risk_level",
            ]
        )
        return empty, pd.DataFrame(columns=["year", "avg_risk_score", "max_risk_score", "high_risk_months"])

    thr = month_thresholds.set_index("month")
    fut["precip_threshold_dry"] = fut["month"].map(thr["precip_p20"])
    fut["temp_threshold_hot"] = fut["month"].map(thr["temp_p80"])

    fut["precip_sigma"] = [
        interval_to_sigma(m, lo, hi)
        for m, lo, hi in zip(fut["precip"], fut["precip_low"], fut["precip_high"])
    ]
    fut["temp_sigma"] = [
        interval_to_sigma(m, lo, hi)
        for m, lo, hi in zip(fut["temp"], fut["temp_low"], fut["temp_high"])
    ]

    fut["dry_prob"] = [
        prob_leq(t, m, s)
        for t, m, s in zip(fut["precip_threshold_dry"], fut["precip"], fut["precip_sigma"])
    ]
    fut["hot_prob"] = [
        prob_geq(t, m, s)
        for t, m, s in zip(fut["temp_threshold_hot"], fut["temp"], fut["temp_sigma"])
    ]
    fut["hot_dry_prob"] = fut["dry_prob"] * fut["hot_prob"]

    spi3_penalty = np.where(fut["spi3"] <= -1.5, 20.0, np.where(fut["spi3"] <= -1.0, 10.0, 0.0))
    spi12_penalty = np.where(
        fut["spi12"] <= -1.5,
        25.0,
        np.where(fut["spi12"] <= -1.0, 12.0, 0.0),
    )
    spei3_penalty = np.where(fut["spei3"] <= -1.5, 20.0, np.where(fut["spei3"] <= -1.0, 10.0, 0.0))
    spei12_penalty = np.where(
        fut["spei12"] <= -1.5,
        25.0,
        np.where(fut["spei12"] <= -1.0, 12.0, 0.0),
    )
    base_score = 100.0 * (
        0.45 * fut["dry_prob"].fillna(0.0)
        + 0.25 * fut["hot_prob"].fillna(0.0)
        + 0.30 * fut["hot_dry_prob"].fillna(0.0)
    )
    fut["risk_score"] = np.clip(
        base_score + spi3_penalty + spi12_penalty + 0.5 * spei3_penalty + 0.5 * spei12_penalty,
        0.0,
        100.0,
    )
    fut["risk_level"] = fut["risk_score"].map(risk_level_from_score)

    monthly_alerts = fut[
        [
            "timestamp",
            "year",
            "month",
            "precip",
            "temp",
            "precip_threshold_dry",
            "temp_threshold_hot",
            "dry_prob",
            "hot_prob",
            "hot_dry_prob",
            "spi3",
            "spi12",
            "spei3",
            "spei12",
            "risk_score",
            "risk_level",
        ]
    ].copy()

    yearly_alerts = (
        monthly_alerts.groupby("year", as_index=False)
        .agg(
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            high_risk_months=("risk_level", lambda s: int(np.sum(np.isin(s, ["kritik", "yuksek"])))),
        )
        .sort_values("year")
    )
    return monthly_alerts, yearly_alerts


def plot_outputs(
    annual: pd.DataFrame,
    monthly_flags: pd.DataFrame,
    drought_episodes: pd.DataFrame,
    seasonal_compare: pd.DataFrame,
    period_summary: pd.DataFrame,
    monthly_alerts: pd.DataFrame,
    chart_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    chart_dir.mkdir(parents=True, exist_ok=True)

    # 1) Yillik trendler (yagis + sicaklik + su stresi)
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    ax1.bar(annual["year"], annual["precip_total_mm"], color="#3b82f6", alpha=0.45, label="Yillik yagis (mm)")
    ax2.plot(annual["year"], annual["temp_mean_c"], color="#ef4444", linewidth=2.2, label="Yillik ort. sicaklik (C)")
    ax2.plot(annual["year"], annual["water_stress_index"], color="#111827", linewidth=1.8, linestyle="--", label="Su stresi indeksi")
    ax1.set_xlabel("Yil")
    ax1.set_ylabel("Yagis (mm)")
    ax2.set_ylabel("Sicaklik (C) / WSI")
    ax1.set_title("Yagis, Sicaklik ve Su Stresi")
    ax1.grid(alpha=0.2)
    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    ax1.legend(lines, labels, loc="upper left")
    fig.tight_layout()
    fig.savefig(chart_dir / "yillik_trendler_yagis_sicaklik_su_stresi.png", dpi=150)
    plt.close(fig)

    # 2) Mevsimsel degisimler
    if not seasonal_compare.empty:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        x = np.arange(len(seasonal_compare))
        width = 0.35
        ax.bar(
            x - width / 2,
            seasonal_compare["precip_change_pct"],
            width=width,
            label="Yagis degisimi (%)",
            color="#0ea5e9",
        )
        ax.bar(
            x + width / 2,
            seasonal_compare["temp_change_c"],
            width=width,
            label="Sicaklik degisimi (C)",
            color="#f97316",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(seasonal_compare["season"])
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title("Mevsimsel Degisim (Gelecek vs Referans)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(chart_dir / "mevsimsel_degisimler.png", dpi=150)
        plt.close(fig)

    # 3) Donemsel risk oranlari
    if not period_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(period_summary))
        ax.plot(x, period_summary["dry_month_rate"] * 100.0, marker="o", linewidth=2, label="Kuru ay orani (%)")
        ax.plot(
            x,
            period_summary["hot_dry_month_rate"] * 100.0,
            marker="o",
            linewidth=2,
            label="Sicak-kuru ay orani (%)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(period_summary["period"], rotation=0)
        ax.set_ylim(bottom=0)
        ax.set_title("Donemsel Kuru ve Sicak-Kuru Ay Sikliklari")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(chart_dir / "donemsel_risk_oranlari.png", dpi=150)
        plt.close(fig)

    # 4) SPI12 iz ve kurak donemler
    if "spi12" in monthly_flags.columns:
        m = monthly_flags.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(m["timestamp"], m["spi12"], color="#0f172a", linewidth=1.8, label="SPI12")
        ax.axhline(-1.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Kurak esik (-1.0)")
        ax.axhline(-1.5, color="#7f1d1d", linestyle=":", linewidth=1.2, label="Siddetli kurak esik (-1.5)")
        if not drought_episodes.empty:
            for _, r in drought_episodes.iterrows():
                ax.axvspan(pd.to_datetime(r["start"]), pd.to_datetime(r["end"]), color="#fecaca", alpha=0.35)
        ax.set_title("SPI12 Kuraklik Izleme")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("SPI12")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(chart_dir / "spi12_kuraklik_izleme.png", dpi=150)
        plt.close(fig)

    # 5) Gelecek aylik alarm skoru
    if not monthly_alerts.empty:
        al = monthly_alerts.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(al["timestamp"], al["risk_score"], color="#b91c1c", linewidth=2, label="Risk skoru")
        ax.fill_between(al["timestamp"], 70, 100, color="#ef4444", alpha=0.12, label="Kritik")
        ax.fill_between(al["timestamp"], 50, 70, color="#f59e0b", alpha=0.12, label="Yuksek")
        ax.fill_between(al["timestamp"], 30, 50, color="#fde68a", alpha=0.20, label="Orta")
        ax.axhline(70, color="#ef4444", linestyle="--", linewidth=1.0)
        ax.axhline(50, color="#f59e0b", linestyle="--", linewidth=1.0)
        ax.axhline(30, color="#a16207", linestyle="--", linewidth=1.0)
        ax.set_ylim(0, 100)
        ax.set_title("Gelecek Aylik Kuraklik Risk Skoru")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("Risk skoru (0-100)")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(chart_dir / "gelecek_aylik_risk_skoru.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    precip_path = Path(args.precip)
    temp_path = Path(args.temp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precip_df = load_quant_series(precip_path, "precip")
    temp_df = load_quant_series(temp_path, "temp")
    monthly = build_monthly_frame(precip_df, temp_df)
    monthly, monthly_normals = monthly_baseline_anomalies(monthly, args.baseline_start, args.baseline_end)
    monthly_flags, monthly_thresholds = build_monthly_risk_flags(
        monthly, args.baseline_start, args.baseline_end
    )
    monthly_flags = add_spi_indices(monthly_flags, args.baseline_start, args.baseline_end)
    monthly_flags = add_spei_indices(monthly_flags, args.baseline_start, args.baseline_end, args.latitude)
    annual = build_annual_metrics(monthly_flags, args.baseline_start, args.baseline_end)
    drought_episodes = extract_drought_episodes(monthly_flags, min_len_months=3, spi_col="spi12", threshold=-1.0)
    quality_summary = build_quality_summary(
        monthly_flags, args.baseline_start, args.baseline_end, args.future_start, args.future_end
    )
    trend_diagnostics = build_trend_diagnostics(
        annual, args.baseline_start, args.baseline_end, args.future_start, args.future_end
    )
    spi_reliability = build_spi_reliability_diagnostics(
        monthly_flags, args.baseline_start, args.baseline_end, args.future_start, args.future_end
    )
    spi_sensitivity = build_spi_sensitivity_baseline_windows(
        monthly_flags, args.baseline_start, args.baseline_end, args.future_start, args.future_end, args.latitude
    )
    lag_detail, lag_summary = build_meteo_hydro_lag_diagnostics(
        monthly_flags,
        args.baseline_start,
        args.baseline_end,
        args.future_start,
        args.future_end,
        max_lag_months=max(0, int(args.max_lag_months)),
    )

    periods = [
        (f"{args.baseline_start}-{args.baseline_end}", args.baseline_start, args.baseline_end),
        ("1988-2000", 1988, 2000),
        ("2001-2010", 2001, 2010),
        ("2011-2020", 2011, 2020),
        ("2021-2035", 2021, 2035),
        (f"{args.future_start}-{args.future_end}", args.future_start, args.future_end),
    ]
    period_rows = [
        summarize_period(label, start, end, annual, monthly_flags) for label, start, end in periods
    ]
    period_summary = pd.DataFrame(period_rows)

    baseline_period = period_summary[
        period_summary["period"] == f"{args.baseline_start}-{args.baseline_end}"
    ]
    compare_period = period_summary[period_summary["period"] == f"{args.future_start}-{args.future_end}"]
    if baseline_period.empty:
        baseline_period = period_summary.head(1)
    if compare_period.empty:
        compare_period = period_summary.tail(1)

    # Mevsimsel karsilastirma: referans (baseline-start to baseline-end) vs future-start to future-end
    ref_monthly = monthly_flags[
        (monthly_flags["year"] >= args.baseline_start) & (monthly_flags["year"] <= args.baseline_end)
    ]
    fut_monthly = monthly_flags[
        (monthly_flags["year"] >= args.future_start) & (monthly_flags["year"] <= args.future_end)
    ]
    ref_season = (
        ref_monthly.groupby("season")
        .agg(
            precip_ref_mm=("precip", "mean"),
            temp_ref_c=("temp", "mean"),
            dry_rate_ref=("dry_month", "mean"),
            hot_dry_rate_ref=("hot_dry_month", "mean"),
        )
        .reindex(["DJF", "MAM", "JJA", "SON"])
        .reset_index()
    )
    fut_season = (
        fut_monthly.groupby("season")
        .agg(
            precip_fut_mm=("precip", "mean"),
            temp_fut_c=("temp", "mean"),
            dry_rate_fut=("dry_month", "mean"),
            hot_dry_rate_fut=("hot_dry_month", "mean"),
        )
        .reindex(["DJF", "MAM", "JJA", "SON"])
        .reset_index()
    )
    seasonal_compare = ref_season.merge(fut_season, on="season", how="outer")
    seasonal_compare["precip_change_pct"] = (
        (seasonal_compare["precip_fut_mm"] - seasonal_compare["precip_ref_mm"])
        / seasonal_compare["precip_ref_mm"]
        * 100.0
    )
    seasonal_compare["temp_change_c"] = seasonal_compare["temp_fut_c"] - seasonal_compare["temp_ref_c"]
    seasonal_compare["dry_rate_change_pp"] = (
        seasonal_compare["dry_rate_fut"] - seasonal_compare["dry_rate_ref"]
    ) * 100.0
    seasonal_compare["hot_dry_rate_change_pp"] = (
        seasonal_compare["hot_dry_rate_fut"] - seasonal_compare["hot_dry_rate_ref"]
    ) * 100.0

    top_risk_years = annual.sort_values(["water_stress_index", "year"], ascending=[False, True]).head(12)
    de_martonne_driest = annual.sort_values(["de_martonne", "year"], ascending=[True, True]).head(12)
    future_episodes = drought_episodes[
        (drought_episodes["start_year"] >= args.future_start) & (drought_episodes["start_year"] <= args.future_end)
    ].copy()
    monthly_alerts, yearly_alerts = build_future_alert_calendar(
        monthly_flags, monthly_thresholds, args.future_start, args.future_end
    )
    early_warning_dashboard = build_early_warning_dashboard(
        baseline_period.iloc[0].to_dict(),
        compare_period.iloc[0].to_dict(),
        future_episode_count=len(future_episodes),
        future_episode_max_months=int(future_episodes["months"].max()) if not future_episodes.empty else 0,
    )
    consistency_check = build_literature_consistency_check(
        period_summary,
        trend_diagnostics,
        lag_summary,
        quality_summary,
        spi_sensitivity,
        args.baseline_start,
        args.baseline_end,
        args.future_start,
        args.future_end,
    )
    plain_summary_text = build_plain_language_summary(
        period_summary,
        consistency_check,
        lag_summary,
        args.baseline_start,
        args.baseline_end,
        args.future_start,
        args.future_end,
    )
    plain_summary_path = output_dir / "halk_dili_ozet.md"
    plain_summary_path.write_text(plain_summary_text, encoding="utf-8")

    # Ciktilar
    monthly_flags.to_csv(output_dir / "merged_monthly_with_risk_flags.csv", index=False)
    monthly_normals.to_csv(output_dir / "monthly_normals_baseline.csv", index=False)
    monthly_thresholds.to_csv(output_dir / "monthly_thresholds_baseline.csv", index=False)
    drought_episodes.to_csv(output_dir / "drought_episodes_spi12.csv", index=False)
    early_warning_dashboard.to_csv(output_dir / "early_warning_dashboard.csv", index=False)
    monthly_alerts.to_csv(output_dir / "future_alert_calendar_monthly.csv", index=False)
    yearly_alerts.to_csv(output_dir / "future_alert_calendar_yearly.csv", index=False)
    quality_summary.to_csv(output_dir / "data_quality_summary.csv", index=False)
    trend_diagnostics.to_csv(output_dir / "trend_diagnostics.csv", index=False)
    spi_reliability.to_csv(output_dir / "spi_reliability_diagnostics.csv", index=False)
    spi_sensitivity.to_csv(output_dir / "spi_sensitivity_baseline_windows.csv", index=False)
    lag_detail.to_csv(output_dir / "meteo_hydro_lag_correlation.csv", index=False)
    lag_summary.to_csv(output_dir / "meteo_hydro_lag_summary.csv", index=False)
    consistency_check.to_csv(output_dir / "literature_consistency_check.csv", index=False)
    annual.to_csv(output_dir / "annual_metrics.csv", index=False)
    period_summary.to_csv(output_dir / "period_summary.csv", index=False)
    seasonal_compare.to_csv(output_dir / "seasonal_comparison.csv", index=False)
    top_risk_years.to_csv(output_dir / "top_risk_years_wsi.csv", index=False)
    de_martonne_driest.to_csv(output_dir / "driest_years_de_martonne.csv", index=False)

    plot_outputs(
        annual,
        monthly_flags,
        drought_episodes,
        seasonal_compare,
        period_summary,
        monthly_alerts,
        output_dir / "charts",
    )

    # Rapor
    b = baseline_period.iloc[0].to_dict()
    f = compare_period.iloc[0].to_dict()

    def pchg(k: str) -> float:
        return (f[k] - b[k]) / b[k] * 100.0

    report_lines = []
    report_lines.append("# Gelismis Yagis-Sicaklik Kuraklik ve Su Kaynaklari Analizi")
    report_lines.append("")
    report_lines.append("## Kapsam")
    report_lines.append(f"- Veri: `{precip_path}` ve `{temp_path}`")
    report_lines.append(f"- Referans donem: {args.baseline_start}-{args.baseline_end}")
    report_lines.append(f"- Gelecek donem: {args.future_start}-{args.future_end}")
    report_lines.append("- Birlesik seri: varsa `actual`, yoksa `yhat`")
    report_lines.append("- SPI hesaplama: ay-bazli gamma dagilimi + sifir-yagis duzeltmesi (WMO yaklasimina uyumlu)")
    report_lines.append(f"- SPEI hesaplama: Thornthwaite PET (enlem={args.latitude:.2f}) + aylik log-logistic/empirik standardizasyon")
    report_lines.append(f"- Meteo-hidro lag analizi: hydro-proxy (24 aylik su dengesi indeksi), max lag={int(args.max_lag_months)} ay")
    report_lines.append("- Belirsizlik: yillik metriklerde monthly `yhat_lower/yhat_upper` kullanildi")
    report_lines.append("")
    report_lines.append("## Ana Bulgular")
    report_lines.append(
        f"- Yillik yagis ortalamasi: {b['annual_precip_mm_mean']:.1f} -> {f['annual_precip_mm_mean']:.1f} mm ({pchg('annual_precip_mm_mean'):+.1f}%)"
    )
    report_lines.append(
        f"- Yillik ortalama sicaklik: {b['annual_temp_c_mean']:.2f} -> {f['annual_temp_c_mean']:.2f} C ({f['annual_temp_c_mean'] - b['annual_temp_c_mean']:+.2f} C)"
    )
    report_lines.append(
        f"- De Martonne ortalamasi: {b['de_martonne_mean']:.2f} -> {f['de_martonne_mean']:.2f} ({pchg('de_martonne_mean'):+.1f}%)"
    )
    report_lines.append(
        f"- Su stresi indeksi: {b['water_stress_index_mean']:.2f} -> {f['water_stress_index_mean']:.2f} ({f['water_stress_index_mean'] - b['water_stress_index_mean']:+.2f})"
    )
    report_lines.append(
        f"- Kuru ay orani: {b['dry_month_rate']*100:.1f}% -> {f['dry_month_rate']*100:.1f}% ({(f['dry_month_rate'] - b['dry_month_rate'])*100:+.1f} puan)"
    )
    report_lines.append(
        f"- Sicak-kuru ay orani: {b['hot_dry_month_rate']*100:.1f}% -> {f['hot_dry_month_rate']*100:.1f}% ({(f['hot_dry_month_rate'] - b['hot_dry_month_rate'])*100:+.1f} puan)"
    )
    report_lines.append(
        f"- Yillik su dengesi (P-PET): {b['annual_water_balance_mm_mean']:.1f} -> {f['annual_water_balance_mm_mean']:.1f} mm ({f['annual_water_balance_mm_mean'] - b['annual_water_balance_mm_mean']:+.1f} mm)"
    )
    report_lines.append(
        f"- SPEI12 kurak ay orani: {b['spei12_drought_rate']*100:.1f}% -> {f['spei12_drought_rate']*100:.1f}% ({(f['spei12_drought_rate'] - b['spei12_drought_rate'])*100:+.1f} puan)"
    )
    report_lines.append("")
    report_lines.append("## Donem Ozeti")
    report_lines.append("")
    report_lines.append(to_markdown_safe(period_summary))
    report_lines.append("")
    report_lines.append("## Veri Kalitesi")
    report_lines.append("")
    report_lines.append(to_markdown_safe(quality_summary))
    report_lines.append("")
    report_lines.append("## Trend Testleri (Kendall + Sen)")
    report_lines.append("")
    report_lines.append(to_markdown_safe(trend_diagnostics))
    report_lines.append("")
    report_lines.append("## SPI Guvenilirlik Tani Ozeti")
    report_lines.append("")
    report_lines.append(
        to_markdown_safe(
            spi_reliability[
                [
                    "index_type",
                    "scale_months",
                    "month",
                    "baseline_sample_n",
                    "baseline_zero_share",
                    "gamma_fit_possible",
                    "reliability_score_0_100",
                    "reliability_label",
                    "notes",
                ]
            ]
        )
    )
    report_lines.append("")
    report_lines.append("## SPI Baz Donem Duyarlilik Analizi")
    report_lines.append("")
    report_lines.append(
        to_markdown_safe(
            spi_sensitivity[
                [
                    "baseline_window",
                    "window_role",
                    "n_eval_spi12",
                    "mae_spi12",
                    "corr_spi12",
                    "drought_agreement_spi12",
                    "drought_rate_pp_delta_spi12",
                    "future_drought_rate_pp_delta_spi12",
                    "n_eval_spei12",
                    "mae_spei12",
                    "corr_spei12",
                    "drought_agreement_spei12",
                    "drought_rate_pp_delta_spei12",
                    "future_drought_rate_pp_delta_spei12",
                    "sensitivity_label_spi",
                    "sensitivity_label_spei",
                    "sensitivity_label",
                ]
            ]
        )
    )
    report_lines.append("")
    report_lines.append("## Meteo-Hidro Lag (Proxy) Ozeti")
    report_lines.append("")
    if lag_summary.empty:
        report_lines.append("Lag analizi hesaplanamadi.")
    else:
        report_lines.append(
            to_markdown_safe(
                lag_summary[
                    [
                        "period",
                        "meteo_index",
                        "best_lag_months",
                        "best_corr",
                        "corr_lag0",
                        "n_pairs_at_best_lag",
                        "lag_strength_label",
                    ]
                ]
            )
        )
    report_lines.append("")
    report_lines.append("## Literaturle Tutarlilik Kontrolu")
    report_lines.append("")
    if consistency_check.empty:
        report_lines.append("Literatur tutarlilik kontrolu hesaplanamadi.")
    else:
        n_conflict = int((consistency_check["status"] == "celiski_olasi").sum())
        n_partial = int((consistency_check["status"] == "kismi_uyumlu").sum())
        n_ok = int((consistency_check["status"] == "uyumlu").sum())
        report_lines.append(
            f"- Ozet: uyumlu={n_ok}, kismi_uyumlu={n_partial}, celiski_olasi={n_conflict}."
        )
        if n_conflict > 0:
            report_lines.append(
                "- Dikkat: celiski_olasi maddeler fiziksel tutarlilik icin ek veriyle tekrar dogrulanmalidir."
            )
        report_lines.append("")
        report_lines.append(
            to_markdown_safe(
                consistency_check[
                    [
                        "topic",
                        "literature_expectation",
                        "model_result",
                        "status",
                        "comment",
                    ]
                ]
            )
        )
    report_lines.append("")
    report_lines.append("## Herkes Icin Kisa Ozet")
    report_lines.append(f"- Sade ozet dosyasi: `{plain_summary_path}`")
    report_lines.append("")
    report_lines.append("## Literatur Uyum Notu")
    report_lines.append("- SPI hesaplamasi WMO SPI User Guide (WMO-No.1090) yaklasimina uygun sekilde gamma dagilimi + sifir-yagis duzeltmesi ile yapildi.")
    report_lines.append("- SPEI icin aylik su dengesi (P-PET) yaklasimi kullanildi; PET, Thornthwaite (1948) sicaklik tabanli formul ile tahmin edildi.")
    report_lines.append("- Referans periyot guvenilirligi icin ay-bazli orneklem sayisi ve sifir-yagis payi raporlandi; 30 yil alti durumlar not edildi.")
    report_lines.append("- Baseline secimine duyarlilik analizi eklenerek SPI12 ve SPEI12 kuraklik siniflamasinin pencereye bagli degisimi olculdu.")
    report_lines.append("- Hidrolojik veri olmadiginda meteo->hidro gecikme, 24 aylik su dengesi tabanli hydro-proxy indeks ile yaklasiklandi.")
    report_lines.append("")
    report_lines.append("## Kullanilan Literatur (Kisa)")
    report_lines.append(
        "- WMO (2012), SPI User Guide (WMO-No.1090): https://www.droughtmanagement.info/literature/StandardizedPrecipitationIndexUserGuide-McKee-UserGuide-WMO1090-2012.pdf"
    )
    report_lines.append(
        "- WMO/GWP (2016), Handbook of Drought Indicators and Indices: https://www.droughtmanagement.info/literature/GWP_Handbook_of_Drought_Indicators_and_Indices_2016.pdf"
    )
    report_lines.append(
        "- McKee et al. (1993), The relationship of drought frequency and duration to time scales: https://climate.colostate.edu/pdfs/relationshipofdroughtfrequency.pdf"
    )
    report_lines.append(
        "- WMO Climatological Normals (30-year standard): https://community.wmo.int/en/activity-areas/climate-services/climate-products-and-initiatives/wmo-climatological-normals"
    )
    report_lines.append(
        "- Thornthwaite (1948), climate classification and potential evapotranspiration: https://doi.org/10.1007/BF02288874"
    )
    report_lines.append(
        "- Vicente-Serrano et al. (2010), SPEI formulation: https://doi.org/10.1175/2009JCLI2909.1"
    )
    report_lines.append("")
    report_lines.append("## Erken Uyari Dashboard")
    report_lines.append("")
    report_lines.append(to_markdown_safe(early_warning_dashboard))
    report_lines.append("")
    report_lines.append("## Gelecek Alarm Takvimi")
    report_lines.append("")
    if monthly_alerts.empty:
        report_lines.append("Gelecek alarm takvimi olusturulamadi.")
    else:
        report_lines.append("### Aylik (ilk 24 ay)")
        report_lines.append("")
        report_lines.append(
            to_markdown_safe(
                monthly_alerts[
                    [
                        "timestamp",
                        "dry_prob",
                        "hot_prob",
                        "hot_dry_prob",
                        "spi12",
                        "spei12",
                        "risk_score",
                        "risk_level",
                    ]
                ].head(24)
            )
        )
        report_lines.append("")
        report_lines.append("### Yillik ozet")
        report_lines.append("")
        report_lines.append(to_markdown_safe(yearly_alerts))
    report_lines.append("")
    report_lines.append("## Mevsimsel Karsilastirma (Gelecek vs Referans)")
    report_lines.append("")
    report_lines.append(
        to_markdown_safe(
            seasonal_compare[
                [
                    "season",
                    "precip_ref_mm",
                    "precip_fut_mm",
                    "precip_change_pct",
                    "temp_ref_c",
                    "temp_fut_c",
                    "temp_change_c",
                    "dry_rate_change_pp",
                    "hot_dry_rate_change_pp",
                ]
            ]
        )
    )
    report_lines.append("")
    report_lines.append("## Kurak Donemler (SPI12, 3+ ay)")
    report_lines.append("")
    if drought_episodes.empty:
        report_lines.append("Kurak donem tespit edilmedi.")
    else:
        report_lines.append(
            to_markdown_safe(
                drought_episodes[
                    [
                        "start",
                        "end",
                        "months",
                        "min_spi",
                        "mean_spi",
                        "severity_score",
                        "mean_precip_mm",
                        "mean_temp_c",
                    ]
                ].head(12)
            )
        )
    report_lines.append("")
    report_lines.append("## En Riskli Yillar (WSI yuksek)")
    report_lines.append("")
    report_lines.append(
        to_markdown_safe(
            top_risk_years[
                [
                    "year",
                    "water_stress_index",
                    "wsi_class",
                    "precip_total_mm",
                    "temp_mean_c",
                    "de_martonne",
                    "de_martonne_class",
                ]
            ]
        )
    )
    report_lines.append("")
    report_lines.append("## En Dusuk De Martonne Yillari (Daha kuru)")
    report_lines.append("")
    report_lines.append(
        to_markdown_safe(
            de_martonne_driest[
                [
                    "year",
                    "de_martonne",
                    "de_martonne_class",
                    "precip_total_mm",
                    "temp_mean_c",
                    "water_stress_index",
                    "wsi_class",
                ]
            ]
        )
    )
    report_lines.append("")
    report_lines.append("## Notlar")
    report_lines.append("- Sicaklikta gozlem kapsami cok sinirli (1987); sonraki yillar model tabanlidir.")
    report_lines.append("- Bu nedenle bulgular operasyonel karar icin degil, senaryo-karsilastirma amacli yorumlanmalidir.")
    report_lines.append("- Trend testleri projeksiyon doneminde model cikisina dayalidir; fiziksel gozlem trendi olarak yorumlanmamali.")
    report_lines.append("- Olasilik tabanli aylik risk skoru, model intervalinden turetilmis yaklasik bir alarm endeksidir.")
    report_lines.append("- Daha guvenilir kuraklik degerlendirmesi icin uzun donem sicaklik gozlemi, akim/baraj verisi ve ET0 gerekir.")
    report_lines.append("- SPEI/PET sonuclari Thornthwaite tabanlidir; enerji/ruzgar/nem tabanli Penman-Monteith PET mevcut degildir.")
    report_lines.append("- Meteo-hidro lag ciktilari hydro-proxy indeks uzerindendir; gercek akim/yeralti suyu serisi ile yeniden dogrulanmalidir.")
    report_lines.append("")

    report_path = output_dir / "gelismis_kuraklik_su_analizi.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"OK report={report_path}")
    print(f"OK annual={output_dir / 'annual_metrics.csv'}")
    print(f"OK periods={output_dir / 'period_summary.csv'}")
    print(f"OK seasonal={output_dir / 'seasonal_comparison.csv'}")
    print(f"OK dashboard={output_dir / 'early_warning_dashboard.csv'}")
    print(f"OK episodes={output_dir / 'drought_episodes_spi12.csv'}")
    print(f"OK alerts_monthly={output_dir / 'future_alert_calendar_monthly.csv'}")
    print(f"OK alerts_yearly={output_dir / 'future_alert_calendar_yearly.csv'}")
    print(f"OK quality={output_dir / 'data_quality_summary.csv'}")
    print(f"OK trends={output_dir / 'trend_diagnostics.csv'}")
    print(f"OK spi_reliability={output_dir / 'spi_reliability_diagnostics.csv'}")
    print(f"OK spi_sensitivity={output_dir / 'spi_sensitivity_baseline_windows.csv'}")
    print(f"OK lag_detail={output_dir / 'meteo_hydro_lag_correlation.csv'}")
    print(f"OK lag_summary={output_dir / 'meteo_hydro_lag_summary.csv'}")
    print(f"OK consistency={output_dir / 'literature_consistency_check.csv'}")
    print(f"OK plain_summary={plain_summary_path}")
    print(f"OK risk={output_dir / 'top_risk_years_wsi.csv'}")
    print(f"OK driest={output_dir / 'driest_years_de_martonne.csv'}")
    if (output_dir / "charts").exists():
        print(f"OK charts={output_dir / 'charts'}")


if __name__ == "__main__":
    main()
