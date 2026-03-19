#!/usr/bin/env python3
"""Analyze human health impact proxies from climate temperature/humidity time series.

This script combines monthly temperature and humidity forecasts and derives
heat-stress indicators:
- Heat Index (NOAA approximation)
- Thom Discomfort Index (THI)
- Threshold-based risk categories
- A simple relative-risk proxy index (for comparison across periods)

Outputs:
- health_monthly_metrics.csv
- health_annual_summary.csv
- health_period_comparison.csv
- health_impact_summary.json
- health_impact_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Human health impact proxy analysis from temp+humidity series")
    p.add_argument("--temp-csv", type=Path, required=True, help="Temperature forecast CSV (Celsius)")
    p.add_argument("--humidity-csv", type=Path, required=True, help="Humidity forecast CSV (relative humidity %)")
    p.add_argument("--output-dir", type=Path, required=True, help="Output folder for reports")
    p.add_argument("--temp-col", default="yhat", help="Temperature value column (default: yhat)")
    p.add_argument("--humidity-col", default="yhat", help="Humidity value column (default: yhat)")
    p.add_argument("--date-col", default="ds", help="Date column name (default: ds)")
    p.add_argument("--baseline-start", type=int, default=1991, help="Baseline period start year")
    p.add_argument("--baseline-end", type=int, default=2020, help="Baseline period end year")
    p.add_argument("--future-start", type=int, default=2026, help="Future comparison period start year")
    p.add_argument("--future-end", type=int, default=2035, help="Future comparison period end year")
    p.add_argument(
        "--realism-mode",
        choices=["none", "conservative", "strict"],
        default="conservative",
        help="Apply climatology-aware realism adjustment to future temp/humidity series",
    )
    p.add_argument(
        "--realism-blend",
        type=float,
        default=np.nan,
        help="Optional override: blend factor for capped anomalies (0-1). Uses mode default when NaN.",
    )
    p.add_argument(
        "--realism-temp-sigma",
        type=float,
        default=np.nan,
        help="Optional override: monthly temp anomaly cap multiplier in sigma units.",
    )
    p.add_argument(
        "--realism-humidity-sigma",
        type=float,
        default=np.nan,
        help="Optional override: monthly humidity anomaly cap multiplier in sigma units.",
    )
    p.add_argument(
        "--realism-temp-trend-per-year",
        type=float,
        default=np.nan,
        help="Optional override: extra positive temp anomaly allowance per year after baseline end (C/year).",
    )
    p.add_argument(
        "--analysis-scope",
        choices=["all", "forecast_only", "historical_only"],
        default="all",
        help="Restrict analysis to all rows, only forecast rows, or only historical rows",
    )
    p.add_argument(
        "--prefer-actual-when-available",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 'actual' values on non-forecast rows when present",
    )
    p.add_argument(
        "--risk-beta-per-c",
        type=float,
        default=np.log(1.05),
        help="Log-risk increase per +1C apparent heat above threshold (default ln(1.05))",
    )
    p.add_argument(
        "--epi-mode",
        choices=[
            "custom",
            "meta_urban_mortality_yang2024",
            "meta_urban_morbidity_yang2024",
            "meta_urban_heatwave_mortality_yang2024",
            "legacy_mortality_working_age",
            "legacy_morbidity_working_age",
            "legacy_heatwave_mortality",
        ],
        default="meta_urban_mortality_yang2024",
        help="Literature-based epidemiology preset for risk beta",
    )
    p.add_argument(
        "--epi-ci-bound",
        choices=["point", "lower", "upper"],
        default="point",
        help="Use point estimate or 95%% CI bound from selected epidemiology preset",
    )
    p.add_argument(
        "--risk-threshold-mode",
        choices=["absolute", "baseline_quantile", "baseline_p90"],
        default="baseline_quantile",
        help="Threshold mode for proxy risk (default: baseline_quantile)",
    )
    p.add_argument(
        "--risk-threshold-c",
        type=float,
        default=26.0,
        help="Apparent heat threshold for risk proxy in absolute mode (default 26C)",
    )
    p.add_argument(
        "--risk-threshold-quantile",
        type=float,
        default=0.85,
        help="Quantile used when threshold mode is baseline_quantile (default 0.85)",
    )
    p.add_argument(
        "--enable-humidity-interaction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply dry-hot / wet-hot interaction multipliers for threshold-exceed months",
    )
    p.add_argument(
        "--dry-hot-rh-max",
        type=float,
        default=60.0,
        help="Max RH%% threshold for dry-hot interaction regime (default 60)",
    )
    p.add_argument(
        "--wet-hot-rh-min",
        type=float,
        default=80.0,
        help="Min RH%% threshold for wet-hot interaction regime (default 80)",
    )
    p.add_argument(
        "--dry-hot-multiplier",
        type=float,
        default=1.1018,
        help="Risk multiplier for dry-hot regime (default from Fang 2023 ER=10.18%%)",
    )
    p.add_argument(
        "--wet-hot-multiplier",
        type=float,
        default=1.0321,
        help="Risk multiplier for wet-hot regime (default from Fang 2023 ER=3.21%%)",
    )
    p.add_argument(
        "--adaptation-mode",
        choices=["none", "moderate", "strong", "custom_linear"],
        default="moderate",
        help="Adaptation scenario applied to excess heat before risk transform",
    )
    p.add_argument(
        "--adaptation-start-year",
        type=int,
        default=2026,
        help="Adaptation ramp start year (default 2026)",
    )
    p.add_argument(
        "--adaptation-end-year",
        type=int,
        default=2035,
        help="Adaptation ramp end year (default 2035)",
    )
    p.add_argument(
        "--adaptation-max-reduction",
        type=float,
        default=0.30,
        help="Max fractional reduction in excess heat for custom_linear adaptation",
    )
    return p.parse_args()


def _pick_numeric_series(df: pd.DataFrame, requested_col: str, fallbacks: list[str]) -> pd.Series:
    candidates = [requested_col] + fallbacks
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    raise ValueError(f"No usable numeric column found. Tried: {candidates}")


def load_series(
    path: Path,
    variable_prefix: str,
    value_col: str,
    date_col: str,
    fallback_cols: list[str],
    lower_cols: list[str],
    upper_cols: list[str],
    prefer_actual_when_available: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"{path}: date column '{date_col}' not found")

    out = pd.DataFrame()
    out["ds"] = pd.to_datetime(df[date_col], errors="coerce")

    if "is_forecast" in df.columns:
        is_fc = df["is_forecast"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    else:
        is_fc = pd.Series(False, index=df.index)
    out[f"{variable_prefix}_is_forecast"] = is_fc

    yhat = _pick_numeric_series(df, value_col, fallback_cols)
    actual = pd.to_numeric(df.get("actual"), errors="coerce") if "actual" in df.columns else None
    if prefer_actual_when_available and (actual is not None):
        out[f"{variable_prefix}_value"] = np.where((~is_fc) & actual.notna(), actual, yhat)
    else:
        out[f"{variable_prefix}_value"] = yhat

    lower = None
    for c in lower_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                lower = s
                break
    if lower is None:
        lower = pd.Series(np.nan, index=df.index)

    upper = None
    for c in upper_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                upper = s
                break
    if upper is None:
        upper = pd.Series(np.nan, index=df.index)

    out[f"{variable_prefix}_lower"] = lower
    out[f"{variable_prefix}_upper"] = upper

    out = out.dropna(subset=["ds", f"{variable_prefix}_value"]).copy()
    out = out.sort_values("ds")

    # Defensive aggregation if there are duplicates in source.
    out = (
        out.groupby("ds", as_index=False)
        .agg(
            **{
                f"{variable_prefix}_value": (f"{variable_prefix}_value", "mean"),
                f"{variable_prefix}_is_forecast": (f"{variable_prefix}_is_forecast", "max"),
                f"{variable_prefix}_lower": (f"{variable_prefix}_lower", "mean"),
                f"{variable_prefix}_upper": (f"{variable_prefix}_upper", "mean"),
            }
        )
        .sort_values("ds")
    )

    return out


def _c_to_f(temp_c: np.ndarray) -> np.ndarray:
    return (temp_c * 9.0 / 5.0) + 32.0


def _f_to_c(temp_f: np.ndarray) -> np.ndarray:
    return (temp_f - 32.0) * 5.0 / 9.0


def heat_index_c(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """NOAA heat index approximation, returned in Celsius.

    Note: Formula is defined for warm conditions and is used here as a proxy
    metric in monthly aggregates.
    """
    t_f = _c_to_f(temp_c)
    r = np.clip(rh_pct, 0.0, 100.0)

    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (r * 0.094))

    hi_reg = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * r
        - 0.22475541 * t_f * r
        - 6.83783e-3 * (t_f**2)
        - 5.481717e-2 * (r**2)
        + 1.22874e-3 * (t_f**2) * r
        + 8.5282e-4 * t_f * (r**2)
        - 1.99e-6 * (t_f**2) * (r**2)
    )

    use_simple = ((hi_simple + t_f) / 2.0) < 80.0
    hi_f = np.where(use_simple, hi_simple, hi_reg)

    low_humidity_adj_mask = (r < 13.0) & (t_f >= 80.0) & (t_f <= 112.0)
    # Guard sqrt domain; NOAA correction is only valid in a bounded temperature window.
    low_humidity_core = np.clip((17.0 - np.abs(t_f - 95.0)) / 17.0, 0.0, None)
    low_humidity_adj = ((13.0 - r) / 4.0) * np.sqrt(low_humidity_core)
    hi_f = np.where(low_humidity_adj_mask, hi_f - low_humidity_adj, hi_f)

    high_humidity_adj_mask = (r > 85.0) & (t_f >= 80.0) & (t_f <= 87.0)
    high_humidity_adj = ((r - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    hi_f = np.where(high_humidity_adj_mask, hi_f + high_humidity_adj, hi_f)

    return _f_to_c(hi_f)


def discomfort_index_thi(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    # Thom's discomfort index (C-based version)
    return temp_c - (0.55 - 0.0055 * rh_pct) * (temp_c - 14.5)


def categorize_heat_index(hi_c: pd.Series) -> pd.Series:
    conditions = [
        hi_c < 27.0,
        (hi_c >= 27.0) & (hi_c < 32.0),
        (hi_c >= 32.0) & (hi_c < 41.0),
        (hi_c >= 41.0) & (hi_c < 54.0),
        hi_c >= 54.0,
    ]
    labels = ["dusuk", "orta", "yuksek", "cok_yuksek", "asiri"]
    return pd.Series(np.select(conditions, labels, default="bilinmiyor"), index=hi_c.index)


def categorize_discomfort(di_c: pd.Series) -> pd.Series:
    conditions = [
        di_c < 21.0,
        (di_c >= 21.0) & (di_c < 24.0),
        (di_c >= 24.0) & (di_c < 27.0),
        (di_c >= 27.0) & (di_c < 29.0),
        (di_c >= 29.0) & (di_c < 32.0),
        di_c >= 32.0,
    ]
    labels = [
        "rahat",
        "hafif_rahatsizlik",
        "belirgin_rahatsizlik",
        "yuksek_rahatsizlik",
        "ciddi_isi_stresi",
        "acil_risk",
    ]
    return pd.Series(np.select(conditions, labels, default="bilinmiyor"), index=di_c.index)


def apply_analysis_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "forecast_only":
        return df[df["is_forecast"]].copy()
    if scope == "historical_only":
        return df[~df["is_forecast"]].copy()
    return df.copy()


def period_stats(df: pd.DataFrame, start_year: int, end_year: int, label: str) -> dict:
    d = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    if d.empty:
        return {
            "period": label,
            "start_year": start_year,
            "end_year": end_year,
            "n_months": 0,
            "mean_temp_c": np.nan,
            "mean_humidity_pct": np.nan,
            "mean_heat_index_c": np.nan,
            "mean_discomfort_index": np.nan,
            "high_risk_month_share": np.nan,
            "very_high_risk_month_share": np.nan,
            "mean_proxy_relative_risk": np.nan,
            "mean_proxy_relative_risk_lower": np.nan,
            "mean_proxy_relative_risk_upper": np.nan,
            "p95_proxy_relative_risk": np.nan,
            "mean_attributable_fraction": np.nan,
            "threshold_exceed_month_share": np.nan,
            "dry_hot_share": np.nan,
            "wet_hot_share": np.nan,
            "mean_adaptation_factor": np.nan,
            "out_of_distribution_share": np.nan,
            "forecast_share": np.nan,
        }

    n = len(d)
    high_or_above = d["hi_risk"].isin(["yuksek", "cok_yuksek", "asiri"]).sum()
    very_high_or_above = d["hi_risk"].isin(["cok_yuksek", "asiri"]).sum()

    return {
        "period": label,
        "start_year": start_year,
        "end_year": end_year,
        "n_months": int(n),
        "mean_temp_c": float(d["temp_c"].mean()),
        "mean_humidity_pct": float(d["humidity_pct"].mean()),
        "mean_heat_index_c": float(d["heat_index_c"].mean()),
        "mean_discomfort_index": float(d["discomfort_index"].mean()),
        "high_risk_month_share": float(high_or_above / n),
        "very_high_risk_month_share": float(very_high_or_above / n),
        "mean_proxy_relative_risk": float(d["proxy_relative_risk"].mean()),
        "mean_proxy_relative_risk_lower": float(d["proxy_relative_risk_lower"].mean()),
        "mean_proxy_relative_risk_upper": float(d["proxy_relative_risk_upper"].mean()),
        "p95_proxy_relative_risk": float(d["proxy_relative_risk"].quantile(0.95)),
        "mean_attributable_fraction": float(d["attributable_fraction"].mean()),
        "threshold_exceed_month_share": float(d["threshold_exceed"].mean()),
        "dry_hot_share": float((d["humid_heat_regime"] == "dry_hot").mean()),
        "wet_hot_share": float((d["humid_heat_regime"] == "wet_hot").mean()),
        "mean_adaptation_factor": float(d["adaptation_factor"].mean()),
        "out_of_distribution_share": float(d["above_baseline_p99"].mean()),
        "forecast_share": float(d["is_forecast"].mean()),
    }


def safe_pct_change(new_val: float, old_val: float) -> float:
    if pd.isna(new_val) or pd.isna(old_val):
        return np.nan
    if np.isclose(old_val, 0.0):
        if np.isclose(new_val, 0.0):
            return 0.0
        return np.inf
    return float((new_val - old_val) / old_val)


def resolve_epi_beta(args: argparse.Namespace) -> tuple[float, str, float]:
    # Meta-analysis presets from Yang et al. (Sci Total Environ, 2024; PMID: 38986701).
    # Values are RRs and 95% CIs reported by the review.
    presets = {
        "meta_urban_mortality_yang2024": {
            "point": 1.021,
            "lower": 1.018,
            "upper": 1.023,
            "ref": "Yang 2024 urban mortality RR=1.021 (95% CI 1.018-1.023) per +1C",
        },
        "meta_urban_morbidity_yang2024": {
            "point": 1.011,
            "lower": 1.007,
            "upper": 1.016,
            "ref": "Yang 2024 urban morbidity RR=1.011 (95% CI 1.007-1.016) per +1C",
        },
        "meta_urban_heatwave_mortality_yang2024": {
            "point": 1.224,
            "lower": 1.176,
            "upper": 1.274,
            "ref": "Yang 2024 urban heatwave mortality RR=1.224 (95% CI 1.176-1.274)",
        },
        # Legacy modes retained for backward compatibility.
        "legacy_mortality_working_age": {
            "point": 1.016,
            "lower": 1.016,
            "upper": 1.016,
            "ref": "Legacy preset RR=1.016 per +1C",
        },
        "legacy_morbidity_working_age": {
            "point": 1.023,
            "lower": 1.023,
            "upper": 1.023,
            "ref": "Legacy preset RR=1.023 per +1C",
        },
        "legacy_heatwave_mortality": {
            "point": 1.038,
            "lower": 1.038,
            "upper": 1.038,
            "ref": "Legacy preset RR=1.038",
        },
    }

    if args.epi_mode == "custom":
        return float(args.risk_beta_per_c), "custom", 1.0

    if args.epi_mode not in presets:
        return float(args.risk_beta_per_c), "custom_unknown_mode", 1.0

    if args.epi_mode == "meta_urban_heatwave_mortality_yang2024":
        # Heatwave RR is event-level, not per +1C. Use mortality beta per +1C
        # and apply heatwave RR as a threshold-exceed multiplier.
        base_rr = {
            "point": 1.021,
            "lower": 1.018,
            "upper": 1.023,
        }[args.epi_ci_bound]
        heatwave_rr = float(presets[args.epi_mode][args.epi_ci_bound])
        return (
            float(np.log(max(base_rr, 1e-9))),
            "Yang 2024 heatwave mode: base mortality RR per +1C + heatwave event RR multiplier",
            max(heatwave_rr, 1.0),
        )

    rr = float(presets[args.epi_mode][args.epi_ci_bound])
    rr = max(rr, 1e-9)
    return float(np.log(rr)), str(presets[args.epi_mode]["ref"]), 1.0


def resolve_adaptation_max_reduction(args: argparse.Namespace) -> float:
    if args.adaptation_mode == "none":
        return 0.0
    if args.adaptation_mode == "moderate":
        return 0.30
    if args.adaptation_mode == "strong":
        return 0.50
    return float(np.clip(args.adaptation_max_reduction, 0.0, 0.95))


def adaptation_factor_for_years(years: pd.Series, args: argparse.Namespace) -> pd.Series:
    max_red = resolve_adaptation_max_reduction(args)
    if max_red <= 0.0:
        return pd.Series(1.0, index=years.index, dtype=float)

    y0 = int(args.adaptation_start_year)
    y1 = int(args.adaptation_end_year)
    if y1 <= y0:
        # Step adaptation: no reduction before start year, full reduction from start year onward.
        ramp = pd.Series(0.0, index=years.index, dtype=float)
        ramp = ramp.where(years < y0, 1.0)
        reduction = max_red * ramp
        return (1.0 - reduction).clip(lower=0.05, upper=1.0)

    t = ((years.astype(float) - float(y0)) / float(y1 - y0)).clip(lower=0.0, upper=1.0)
    reduction = max_red * t
    return (1.0 - reduction).clip(lower=0.05, upper=1.0)


def humidity_regime_and_multiplier(
    humidity_pct: pd.Series,
    threshold_exceed: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.Series]:
    regime = pd.Series("neutral", index=humidity_pct.index, dtype=object)
    regime = regime.where(~((humidity_pct <= float(args.dry_hot_rh_max)) & threshold_exceed), "dry_hot")
    regime = regime.where(~((humidity_pct >= float(args.wet_hot_rh_min)) & threshold_exceed), "wet_hot")

    mult = pd.Series(1.0, index=humidity_pct.index, dtype=float)
    if bool(args.enable_humidity_interaction):
        mult = mult.where(regime != "dry_hot", float(args.dry_hot_multiplier))
        mult = mult.where(regime != "wet_hot", float(args.wet_hot_multiplier))
    return regime, mult


def resolve_realism_params(args: argparse.Namespace) -> dict:
    if args.realism_mode == "none":
        base = {"blend": 1.0, "temp_sigma": 1e9, "humidity_sigma": 1e9, "temp_trend": 1e9}
    elif args.realism_mode == "strict":
        base = {"blend": 0.45, "temp_sigma": 2.0, "humidity_sigma": 2.0, "temp_trend": 0.035}
    else:
        base = {"blend": 0.65, "temp_sigma": 2.5, "humidity_sigma": 2.5, "temp_trend": 0.05}

    if not pd.isna(args.realism_blend):
        base["blend"] = float(np.clip(args.realism_blend, 0.0, 1.0))
    if not pd.isna(args.realism_temp_sigma):
        base["temp_sigma"] = float(max(args.realism_temp_sigma, 0.1))
    if not pd.isna(args.realism_humidity_sigma):
        base["humidity_sigma"] = float(max(args.realism_humidity_sigma, 0.1))
    if not pd.isna(args.realism_temp_trend_per_year):
        base["temp_trend"] = float(max(args.realism_temp_trend_per_year, 0.0))
    return base


def apply_realism_adjustment(monthly: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    d = monthly.copy()
    params = resolve_realism_params(args)

    d["temp_c_raw"] = d["temp_c"].astype(float)
    d["humidity_pct_raw"] = d["humidity_pct"].astype(float)

    if args.realism_mode == "none":
        d["temp_adjustment_c"] = 0.0
        d["humidity_adjustment_pct"] = 0.0
        return d, {
            "realism_mode": args.realism_mode,
            "blend": params["blend"],
            "temp_sigma": params["temp_sigma"],
            "humidity_sigma": params["humidity_sigma"],
            "temp_trend_per_year": params["temp_trend"],
            "future_rows_adjusted_share": 0.0,
            "mean_abs_temp_adjustment_c": 0.0,
            "max_abs_temp_adjustment_c": 0.0,
            "mean_abs_humidity_adjustment_pct": 0.0,
            "max_abs_humidity_adjustment_pct": 0.0,
        }

    base = d[(d["year"] >= int(args.baseline_start)) & (d["year"] <= int(args.baseline_end))].copy()
    if base.empty:
        d["temp_adjustment_c"] = 0.0
        d["humidity_adjustment_pct"] = 0.0
        return d, {
            "realism_mode": args.realism_mode,
            "blend": params["blend"],
            "temp_sigma": params["temp_sigma"],
            "humidity_sigma": params["humidity_sigma"],
            "temp_trend_per_year": params["temp_trend"],
            "future_rows_adjusted_share": 0.0,
            "mean_abs_temp_adjustment_c": 0.0,
            "max_abs_temp_adjustment_c": 0.0,
            "mean_abs_humidity_adjustment_pct": 0.0,
            "max_abs_humidity_adjustment_pct": 0.0,
            "warning": "baseline window empty; realism adjustment skipped",
        }

    clim = (
        base.groupby("month", as_index=False)
        .agg(
            base_temp_mean=("temp_c_raw", "mean"),
            base_temp_std=("temp_c_raw", "std"),
            base_hum_mean=("humidity_pct_raw", "mean"),
            base_hum_std=("humidity_pct_raw", "std"),
        )
        .fillna(0.0)
    )
    d = d.merge(clim, on="month", how="left")
    global_temp_mean = float(base["temp_c_raw"].mean())
    global_hum_mean = float(base["humidity_pct_raw"].mean())
    global_temp_std = float(base["temp_c_raw"].std(ddof=0))
    global_hum_std = float(base["humidity_pct_raw"].std(ddof=0))
    if not np.isfinite(global_temp_mean):
        global_temp_mean = float(d["temp_c_raw"].mean())
    if not np.isfinite(global_hum_mean):
        global_hum_mean = float(d["humidity_pct_raw"].mean())
    if (not np.isfinite(global_temp_std)) or np.isclose(global_temp_std, 0.0):
        global_temp_std = 1.0
    if (not np.isfinite(global_hum_std)) or np.isclose(global_hum_std, 0.0):
        global_hum_std = 1.0

    d["base_temp_mean"] = d["base_temp_mean"].fillna(global_temp_mean)
    d["base_hum_mean"] = d["base_hum_mean"].fillna(global_hum_mean)
    d["base_temp_std"] = d["base_temp_std"].replace(0.0, np.nan).fillna(global_temp_std)
    d["base_hum_std"] = d["base_hum_std"].replace(0.0, np.nan).fillna(global_hum_std)

    is_future = d["year"] >= int(args.future_start)
    years_ahead = (d["year"] - int(args.baseline_end)).clip(lower=0).astype(float)

    temp_anom_raw = d["temp_c_raw"] - d["base_temp_mean"]
    hum_anom_raw = d["humidity_pct_raw"] - d["base_hum_mean"]

    temp_cap_pos = params["temp_sigma"] * d["base_temp_std"] + params["temp_trend"] * years_ahead
    temp_cap_neg = -params["temp_sigma"] * d["base_temp_std"]
    hum_cap_pos = params["humidity_sigma"] * d["base_hum_std"]
    hum_cap_neg = -params["humidity_sigma"] * d["base_hum_std"]

    temp_anom_capped = temp_anom_raw.clip(lower=temp_cap_neg, upper=temp_cap_pos)
    hum_anom_capped = hum_anom_raw.clip(lower=hum_cap_neg, upper=hum_cap_pos)

    temp_target = d["base_temp_mean"] + params["blend"] * temp_anom_capped
    hum_target = (d["base_hum_mean"] + params["blend"] * hum_anom_capped).clip(0.0, 100.0)

    d["temp_c"] = d["temp_c_raw"].where(~is_future, temp_target)
    d["humidity_pct"] = d["humidity_pct_raw"].where(~is_future, hum_target)

    d["temp_adjustment_c"] = d["temp_c"] - d["temp_c_raw"]
    d["humidity_adjustment_pct"] = d["humidity_pct"] - d["humidity_pct_raw"]

    future_n = int(is_future.sum())
    if future_n > 0:
        future_temp_adj = np.abs(d.loc[is_future, "temp_adjustment_c"])
        future_hum_adj = np.abs(d.loc[is_future, "humidity_adjustment_pct"])
        changed = (future_temp_adj > 1e-9) | (future_hum_adj > 1e-9)
        future_rows_adjusted_share = float(changed.mean())
        mean_abs_temp_adjustment = float(future_temp_adj.mean())
        max_abs_temp_adjustment = float(future_temp_adj.max())
        mean_abs_humidity_adjustment = float(future_hum_adj.mean())
        max_abs_humidity_adjustment = float(future_hum_adj.max())
    else:
        future_rows_adjusted_share = 0.0
        mean_abs_temp_adjustment = 0.0
        max_abs_temp_adjustment = 0.0
        mean_abs_humidity_adjustment = 0.0
        max_abs_humidity_adjustment = 0.0

    return d, {
        "realism_mode": args.realism_mode,
        "blend": params["blend"],
        "temp_sigma": params["temp_sigma"],
        "humidity_sigma": params["humidity_sigma"],
        "temp_trend_per_year": params["temp_trend"],
        "future_rows_adjusted_share": future_rows_adjusted_share,
        "mean_abs_temp_adjustment_c": mean_abs_temp_adjustment,
        "max_abs_temp_adjustment_c": max_abs_temp_adjustment,
        "mean_abs_humidity_adjustment_pct": mean_abs_humidity_adjustment,
        "max_abs_humidity_adjustment_pct": max_abs_humidity_adjustment,
    }


def build_report_markdown(
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    baseline: dict,
    future: dict,
    delta: dict,
    args: argparse.Namespace,
    risk_threshold_c: float,
    risk_beta_per_c: float,
    epi_reference: str,
    heatwave_event_multiplier: float,
    realism_summary: dict,
) -> str:
    top_months = monthly.sort_values("proxy_relative_risk", ascending=False).head(10).copy()
    top_future = monthly[
        (monthly["year"] >= int(args.future_start)) & (monthly["year"] <= int(args.future_end))
    ].sort_values("proxy_relative_risk", ascending=False).head(10).copy()
    top_months_lines = []
    for _, row in top_months.iterrows():
        top_months_lines.append(
            f"- {row['ds'].date()}: HI={row['heat_index_c']:.2f}C, DI={row['discomfort_index']:.2f}, risk={row['hi_risk']}, rr={row['proxy_relative_risk']:.3f}"
        )
    top_future_lines = []
    for _, row in top_future.iterrows():
        top_future_lines.append(
            f"- {row['ds'].date()}: HI={row['heat_index_c']:.2f}C, DI={row['discomfort_index']:.2f}, risk={row['hi_risk']}, rr={row['proxy_relative_risk']:.3f}"
        )

    hi_trend = annual[["year", "mean_heat_index_c"]].dropna()
    hi_slope = np.nan
    if len(hi_trend) >= 2:
        hi_slope = float(np.polyfit(hi_trend["year"], hi_trend["mean_heat_index_c"], 1)[0])

    lines = [
        "# İnsan Sağlığı Etki Analizi (Proxy)",
        "",
        "Bu rapor sıcaklık ve nemden türetilen ısı stresi göstergelerine dayanır.",
        "Sonuçlar medikal tanı değildir; iklim kaynaklı göreli risk karşılaştırması için kullanılmalıdır.",
        "",
        "## Kullanılan metrikler",
        "",
        "- Heat Index (NOAA): sıcaklık+nemi birlikte ele alan görünür sıcaklık göstergesi.",
        "- Discomfort Index (THI): nüfus düzeyinde ısı rahatsızlığına dair klasik gösterge.",
        f"- Proxy relative risk: max(HI-{risk_threshold_c:.2f}C,0) üzerinden exp(beta*x), beta={risk_beta_per_c:.6f}.",
        f"- Epidemiolojik mod: `{args.epi_mode}` / ci_bound=`{args.epi_ci_bound}` ({epi_reference}).",
        f"- Heatwave olay carpani (esik asimi aylari): x{heatwave_event_multiplier:.3f}.",
        f"- Nem etkileşimi: `enable={bool(args.enable_humidity_interaction)}` dry_hot<=%{float(args.dry_hot_rh_max):.1f} x{float(args.dry_hot_multiplier):.4g}, wet_hot>=%{float(args.wet_hot_rh_min):.1f} x{float(args.wet_hot_multiplier):.4g}.",
        f"- Adaptasyon: `{args.adaptation_mode}` (start={int(args.adaptation_start_year)}, end={int(args.adaptation_end_year)}).",
        f"- Gerceklik ayari: mode=`{realism_summary.get('realism_mode', 'none')}`, blend={realism_summary.get('blend', np.nan):.3f}, temp_sigma={realism_summary.get('temp_sigma', np.nan):.3f}, humidity_sigma={realism_summary.get('humidity_sigma', np.nan):.3f}, temp_trend={realism_summary.get('temp_trend_per_year', np.nan):.4f} C/yil.",
        f"- Gerceklik ayari etkisi (future): duzeltilen satir payi={realism_summary.get('future_rows_adjusted_share', np.nan):.2%}, |dT| ort/max={realism_summary.get('mean_abs_temp_adjustment_c', np.nan):.3f}/{realism_summary.get('max_abs_temp_adjustment_c', np.nan):.3f} C, |dRH| ort/max={realism_summary.get('mean_abs_humidity_adjustment_pct', np.nan):.3f}/{realism_summary.get('max_abs_humidity_adjustment_pct', np.nan):.3f} puan.",
        "",
        "## Dönem karşılaştırması",
        "",
        f"- Baz dönem: {int(baseline['start_year'])}-{int(baseline['end_year'])}",
        f"- Gelecek dönem: {int(future['start_year'])}-{int(future['end_year'])}",
        f"- Analiz kapsamı: `{args.analysis_scope}`",
        f"- Baz dönem forecast payı: {baseline.get('forecast_share', np.nan):.2%}",
        f"- Gelecek dönem forecast payı: {future.get('forecast_share', np.nan):.2%}",
        f"- Ortalama Heat Index değişimi: {delta.get('mean_heat_index_c_delta', np.nan):.3f}C ({delta.get('mean_heat_index_c_pct', np.nan):.2%})",
        f"- Ortalama Discomfort Index değişimi: {delta.get('mean_discomfort_index_delta', np.nan):.3f} ({delta.get('mean_discomfort_index_pct', np.nan):.2%})",
        f"- Yüksek+ risk ay oranı değişimi: {delta.get('high_risk_month_share_delta', np.nan):.3f} ({delta.get('high_risk_month_share_pct', np.nan):.2%})",
        f"- Çok yüksek+ risk ay oranı değişimi: {delta.get('very_high_risk_month_share_delta', np.nan):.3f} ({delta.get('very_high_risk_month_share_pct', np.nan):.2%})",
        f"- Eşik aşımı ay oranı değişimi: {delta.get('threshold_exceed_month_share_delta', np.nan):.3f} ({delta.get('threshold_exceed_month_share_pct', np.nan):.2%})",
        f"- Dry-hot ay oranı değişimi: {delta.get('dry_hot_share_delta', np.nan):.3f} ({delta.get('dry_hot_share_pct', np.nan):.2%})",
        f"- Wet-hot ay oranı değişimi: {delta.get('wet_hot_share_delta', np.nan):.3f} ({delta.get('wet_hot_share_pct', np.nan):.2%})",
        f"- Dagilim-disi ay orani degisimi (HI > baz p99): {delta.get('out_of_distribution_share_delta', np.nan):.3f} ({delta.get('out_of_distribution_share_pct', np.nan):.2%})",
        f"- Ortalama proxy göreli risk değişimi: {delta.get('mean_proxy_relative_risk_delta', np.nan):.3f} ({delta.get('mean_proxy_relative_risk_pct', np.nan):.2%})",
        f"- Ortalama proxy RR alt/üst: {future.get('mean_proxy_relative_risk_lower', np.nan):.3f} / {future.get('mean_proxy_relative_risk_upper', np.nan):.3f}",
        f"- Ortalama atfedilebilir fraksiyon (AF) değişimi: {delta.get('mean_attributable_fraction_delta', np.nan):.3f} ({delta.get('mean_attributable_fraction_pct', np.nan):.2%})",
        f"- Ortalama adaptasyon faktörü (gelecek): {future.get('mean_adaptation_factor', np.nan):.3f}",
        "",
        "## Eğilim",
        "",
        f"- Yıllık ortalama Heat Index doğrusal eğilim: {hi_slope:.4f} C/yıl",
        "",
        "## En riskli 10 ay",
        "",
        *top_months_lines,
        "",
        "## Gelecek dönemde en riskli 10 ay",
        "",
        *top_future_lines,
        "",
        "## Notlar",
        "",
        "- Aylık ortalamalarda Heat Index, günlük/saatlik pik etkileri bastırabilir.",
        "- Sağlık etkisini daha doğru kalibre etmek için hastane başvurusu/ölüm verisiyle eşleştirme önerilir.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temp = load_series(
        path=args.temp_csv,
        variable_prefix="temp",
        value_col=args.temp_col,
        date_col=args.date_col,
        fallback_cols=["value", "actual", "y"],
        lower_cols=["yhat_lower", "lower", "lo", "value_lower", "actual_lower"],
        upper_cols=["yhat_upper", "upper", "hi", "value_upper", "actual_upper"],
        prefer_actual_when_available=bool(args.prefer_actual_when_available),
    )
    hum = load_series(
        path=args.humidity_csv,
        variable_prefix="humidity",
        value_col=args.humidity_col,
        date_col=args.date_col,
        fallback_cols=["value", "actual", "y"],
        lower_cols=["yhat_lower", "lower", "lo", "value_lower", "actual_lower"],
        upper_cols=["yhat_upper", "upper", "hi", "value_upper", "actual_upper"],
        prefer_actual_when_available=bool(args.prefer_actual_when_available),
    )

    monthly = temp.merge(hum, on="ds", how="inner")
    if monthly.empty:
        raise SystemExit("Temperature and humidity series do not overlap on date axis.")

    monthly = monthly.sort_values("ds").copy()
    monthly["temp_c"] = monthly["temp_value"].astype(float)
    monthly["humidity_pct"] = monthly["humidity_value"].astype(float).clip(0.0, 100.0)
    monthly["is_forecast"] = monthly[["temp_is_forecast", "humidity_is_forecast"]].any(axis=1)
    monthly["year"] = monthly["ds"].dt.year
    monthly["month"] = monthly["ds"].dt.month

    monthly, realism_summary = apply_realism_adjustment(monthly, args)

    monthly["heat_index_c"] = heat_index_c(monthly["temp_c"].to_numpy(), monthly["humidity_pct"].to_numpy())
    monthly["discomfort_index"] = discomfort_index_thi(monthly["temp_c"].to_numpy(), monthly["humidity_pct"].to_numpy())

    temp_shift = monthly["temp_adjustment_c"].fillna(0.0).to_numpy()
    hum_shift = monthly["humidity_adjustment_pct"].fillna(0.0).to_numpy()
    temp_low_raw = monthly["temp_lower"].fillna(monthly["temp_c_raw"]).to_numpy()
    temp_up_raw = monthly["temp_upper"].fillna(monthly["temp_c_raw"]).to_numpy()
    hum_low_raw = monthly["humidity_lower"].fillna(monthly["humidity_pct_raw"]).to_numpy()
    hum_up_raw = monthly["humidity_upper"].fillna(monthly["humidity_pct_raw"]).to_numpy()

    temp_low = temp_low_raw + temp_shift
    temp_up = temp_up_raw + temp_shift
    hum_low = np.clip(hum_low_raw + hum_shift, 0.0, 100.0)
    hum_up = np.clip(hum_up_raw + hum_shift, 0.0, 100.0)

    hi_ll = heat_index_c(temp_low, hum_low)
    hi_lu = heat_index_c(temp_low, hum_up)
    hi_ul = heat_index_c(temp_up, hum_low)
    hi_uu = heat_index_c(temp_up, hum_up)
    hi_stack = np.vstack([hi_ll, hi_lu, hi_ul, hi_uu, monthly["heat_index_c"].to_numpy()])
    monthly["heat_index_c_lower"] = np.nanmin(hi_stack, axis=0)
    monthly["heat_index_c_upper"] = np.nanmax(hi_stack, axis=0)

    monthly["hi_risk"] = categorize_heat_index(monthly["heat_index_c"])
    monthly["di_risk"] = categorize_discomfort(monthly["discomfort_index"])

    risk_beta_per_c, epi_reference, heatwave_event_multiplier = resolve_epi_beta(args)

    threshold_mode = args.risk_threshold_mode
    if threshold_mode == "baseline_p90":
        threshold_mode = "baseline_quantile"
        if np.isclose(float(args.risk_threshold_quantile), 0.85):
            args.risk_threshold_quantile = 0.90

    if threshold_mode == "absolute":
        risk_threshold_c = float(args.risk_threshold_c)
        baseline_hi_p99 = np.nan
    else:
        baseline_slice = monthly[
            (monthly["year"] >= int(args.baseline_start)) & (monthly["year"] <= int(args.baseline_end))
        ]["heat_index_c"].dropna()
        if baseline_slice.empty:
            risk_threshold_c = float(args.risk_threshold_c)
            baseline_hi_p99 = np.nan
        else:
            q = float(np.clip(args.risk_threshold_quantile, 0.01, 0.99))
            risk_threshold_c = float(baseline_slice.quantile(q))
            baseline_hi_p99 = float(baseline_slice.quantile(0.99))

    monthly["threshold_exceed"] = monthly["heat_index_c"] > risk_threshold_c
    monthly["above_baseline_p99"] = (
        monthly["heat_index_c"] > float(baseline_hi_p99) if not pd.isna(baseline_hi_p99) else False
    )
    monthly["heatwave_event_multiplier"] = np.where(
        monthly["threshold_exceed"], float(heatwave_event_multiplier), 1.0
    )
    monthly["humid_heat_regime"], monthly["humidity_interaction_multiplier"] = humidity_regime_and_multiplier(
        monthly["humidity_pct"], monthly["threshold_exceed"], args
    )
    monthly["adaptation_factor"] = adaptation_factor_for_years(monthly["year"], args)

    monthly["excess_heat_c"] = (monthly["heat_index_c"] - risk_threshold_c).clip(lower=0.0)
    monthly["excess_heat_c_lower"] = (monthly["heat_index_c_lower"] - risk_threshold_c).clip(lower=0.0)
    monthly["excess_heat_c_upper"] = (monthly["heat_index_c_upper"] - risk_threshold_c).clip(lower=0.0)
    monthly["excess_heat_c_adj"] = monthly["excess_heat_c"] * monthly["adaptation_factor"]
    monthly["excess_heat_c_adj_lower"] = monthly["excess_heat_c_lower"] * monthly["adaptation_factor"]
    monthly["excess_heat_c_adj_upper"] = monthly["excess_heat_c_upper"] * monthly["adaptation_factor"]

    base_rr = np.exp(float(risk_beta_per_c) * monthly["excess_heat_c_adj"])
    base_rr_lo = np.exp(float(risk_beta_per_c) * monthly["excess_heat_c_adj_lower"])
    base_rr_hi = np.exp(float(risk_beta_per_c) * monthly["excess_heat_c_adj_upper"])

    rr_mult = monthly["humidity_interaction_multiplier"] * monthly["heatwave_event_multiplier"]
    monthly["proxy_relative_risk"] = base_rr * rr_mult
    monthly["proxy_relative_risk_lower"] = base_rr_lo * rr_mult
    monthly["proxy_relative_risk_upper"] = base_rr_hi * rr_mult
    monthly["attributable_fraction"] = (monthly["proxy_relative_risk"] - 1.0) / monthly["proxy_relative_risk"].clip(
        lower=1e-9
    )

    monthly = apply_analysis_scope(monthly, args.analysis_scope)
    if monthly.empty:
        raise SystemExit(f"No rows left after analysis scope filter: {args.analysis_scope}")

    annual = (
        monthly.groupby("year", as_index=False)
        .agg(
            mean_temp_c=("temp_c", "mean"),
            mean_humidity_pct=("humidity_pct", "mean"),
            mean_heat_index_c=("heat_index_c", "mean"),
            p95_heat_index_c=("heat_index_c", lambda s: float(np.nanpercentile(s, 95))),
            mean_discomfort_index=("discomfort_index", "mean"),
            high_risk_months=("hi_risk", lambda s: int(pd.Series(s).isin(["yuksek", "cok_yuksek", "asiri"]).sum())),
            very_high_risk_months=("hi_risk", lambda s: int(pd.Series(s).isin(["cok_yuksek", "asiri"]).sum())),
            mean_proxy_relative_risk=("proxy_relative_risk", "mean"),
            mean_proxy_relative_risk_lower=("proxy_relative_risk_lower", "mean"),
            mean_proxy_relative_risk_upper=("proxy_relative_risk_upper", "mean"),
            mean_attributable_fraction=("attributable_fraction", "mean"),
            max_proxy_relative_risk=("proxy_relative_risk", "max"),
            threshold_exceed_months=("threshold_exceed", "sum"),
            above_baseline_p99_months=("above_baseline_p99", "sum"),
            dry_hot_months=("humid_heat_regime", lambda s: int((pd.Series(s) == "dry_hot").sum())),
            wet_hot_months=("humid_heat_regime", lambda s: int((pd.Series(s) == "wet_hot").sum())),
            mean_adaptation_factor=("adaptation_factor", "mean"),
            n_months=("ds", "count"),
        )
        .sort_values("year")
    )
    annual["threshold_exceed_month_share"] = annual["threshold_exceed_months"] / annual["n_months"].replace(0, np.nan)
    annual["above_baseline_p99_month_share"] = annual["above_baseline_p99_months"] / annual["n_months"].replace(0, np.nan)
    annual["dry_hot_month_share"] = annual["dry_hot_months"] / annual["n_months"].replace(0, np.nan)
    annual["wet_hot_month_share"] = annual["wet_hot_months"] / annual["n_months"].replace(0, np.nan)

    baseline_clim = (
        monthly[(monthly["year"] >= int(args.baseline_start)) & (monthly["year"] <= int(args.baseline_end))]
        .groupby("month", as_index=False)
        .agg(
            baseline_temp_c=("temp_c", "mean"),
            baseline_humidity_pct=("humidity_pct", "mean"),
            baseline_heat_index_c=("heat_index_c", "mean"),
            baseline_discomfort_index=("discomfort_index", "mean"),
        )
    )

    monthly_anom = monthly.merge(baseline_clim, on="month", how="left")
    monthly_anom["temp_anomaly_c"] = monthly_anom["temp_c"] - monthly_anom["baseline_temp_c"]
    monthly_anom["humidity_anomaly_pct"] = monthly_anom["humidity_pct"] - monthly_anom["baseline_humidity_pct"]
    monthly_anom["heat_index_anomaly_c"] = monthly_anom["heat_index_c"] - monthly_anom["baseline_heat_index_c"]
    monthly_anom["discomfort_anomaly"] = monthly_anom["discomfort_index"] - monthly_anom["baseline_discomfort_index"]

    baseline = period_stats(monthly, args.baseline_start, args.baseline_end, "baseline")
    future = period_stats(monthly, args.future_start, args.future_end, "future")

    metric_keys = [
        "mean_temp_c",
        "mean_humidity_pct",
        "mean_heat_index_c",
        "mean_discomfort_index",
        "high_risk_month_share",
        "very_high_risk_month_share",
        "threshold_exceed_month_share",
        "dry_hot_share",
        "wet_hot_share",
        "mean_adaptation_factor",
        "out_of_distribution_share",
        "mean_proxy_relative_risk",
        "mean_proxy_relative_risk_lower",
        "mean_proxy_relative_risk_upper",
        "mean_attributable_fraction",
        "p95_proxy_relative_risk",
    ]
    delta = {}
    for k in metric_keys:
        b = baseline.get(k, np.nan)
        f = future.get(k, np.nan)
        delta[f"{k}_delta"] = float(f - b) if (not pd.isna(f) and not pd.isna(b)) else np.nan
        delta[f"{k}_pct"] = safe_pct_change(f, b)

    period_comp = pd.DataFrame([baseline, future, {"period": "delta", **delta}])

    monthly_out = monthly[
        [
            "ds",
            "year",
            "month",
            "temp_c",
            "humidity_pct",
            "temp_c_raw",
            "humidity_pct_raw",
            "temp_adjustment_c",
            "humidity_adjustment_pct",
            "heat_index_c",
            "heat_index_c_lower",
            "heat_index_c_upper",
            "discomfort_index",
            "hi_risk",
            "di_risk",
            "excess_heat_c",
            "excess_heat_c_lower",
            "excess_heat_c_upper",
            "excess_heat_c_adj",
            "excess_heat_c_adj_lower",
            "excess_heat_c_adj_upper",
            "proxy_relative_risk",
            "proxy_relative_risk_lower",
            "proxy_relative_risk_upper",
            "attributable_fraction",
            "threshold_exceed",
            "above_baseline_p99",
            "humid_heat_regime",
            "humidity_interaction_multiplier",
            "heatwave_event_multiplier",
            "adaptation_factor",
            "temp_is_forecast",
            "humidity_is_forecast",
            "is_forecast",
        ]
    ].copy()

    monthly_csv = args.output_dir / "health_monthly_metrics.csv"
    monthly_anom_csv = args.output_dir / "health_monthly_anomalies_vs_baseline.csv"
    annual_csv = args.output_dir / "health_annual_summary.csv"
    period_csv = args.output_dir / "health_period_comparison.csv"
    summary_json = args.output_dir / "health_impact_summary.json"
    report_md = args.output_dir / "health_impact_report.md"

    monthly_out.to_csv(monthly_csv, index=False)
    monthly_anom[
        [
            "ds",
            "year",
            "month",
            "temp_c",
            "humidity_pct",
            "temp_c_raw",
            "humidity_pct_raw",
            "temp_adjustment_c",
            "humidity_adjustment_pct",
            "heat_index_c",
            "discomfort_index",
            "baseline_temp_c",
            "baseline_humidity_pct",
            "baseline_heat_index_c",
            "baseline_discomfort_index",
            "temp_anomaly_c",
            "humidity_anomaly_pct",
            "heat_index_anomaly_c",
            "discomfort_anomaly",
            "proxy_relative_risk",
            "attributable_fraction",
            "threshold_exceed",
            "above_baseline_p99",
            "humid_heat_regime",
            "is_forecast",
        ]
    ].to_csv(monthly_anom_csv, index=False)
    annual.to_csv(annual_csv, index=False)
    period_comp.to_csv(period_csv, index=False)

    summary = {
        "inputs": {
            "temp_csv": str(args.temp_csv),
            "humidity_csv": str(args.humidity_csv),
            "baseline_years": [args.baseline_start, args.baseline_end],
            "future_years": [args.future_start, args.future_end],
            "risk_threshold_mode": args.risk_threshold_mode,
            "risk_threshold_c": risk_threshold_c,
            "baseline_hi_p99_c": baseline_hi_p99,
            "risk_threshold_quantile": args.risk_threshold_quantile,
            "risk_beta_per_c": args.risk_beta_per_c,
            "risk_beta_per_c_effective": risk_beta_per_c,
            "epi_mode": args.epi_mode,
            "epi_ci_bound": args.epi_ci_bound,
            "epi_reference": epi_reference,
            "heatwave_event_multiplier": heatwave_event_multiplier,
            "enable_humidity_interaction": bool(args.enable_humidity_interaction),
            "dry_hot_rh_max": args.dry_hot_rh_max,
            "wet_hot_rh_min": args.wet_hot_rh_min,
            "dry_hot_multiplier": args.dry_hot_multiplier,
            "wet_hot_multiplier": args.wet_hot_multiplier,
            "adaptation_mode": args.adaptation_mode,
            "adaptation_start_year": args.adaptation_start_year,
            "adaptation_end_year": args.adaptation_end_year,
            "adaptation_max_reduction_effective": resolve_adaptation_max_reduction(args),
            "realism_mode": args.realism_mode,
            "realism_blend_override": None if pd.isna(args.realism_blend) else float(args.realism_blend),
            "realism_temp_sigma_override": None
            if pd.isna(args.realism_temp_sigma)
            else float(args.realism_temp_sigma),
            "realism_humidity_sigma_override": None
            if pd.isna(args.realism_humidity_sigma)
            else float(args.realism_humidity_sigma),
            "realism_temp_trend_per_year_override": None
            if pd.isna(args.realism_temp_trend_per_year)
            else float(args.realism_temp_trend_per_year),
            "analysis_scope": args.analysis_scope,
            "prefer_actual_when_available": bool(args.prefer_actual_when_available),
        },
        "realism": realism_summary,
        "coverage": {
            "start_date": str(monthly_out["ds"].min().date()),
            "end_date": str(monthly_out["ds"].max().date()),
            "n_months": int(len(monthly_out)),
            "n_years": int(monthly_out["year"].nunique()),
        },
        "baseline": baseline,
        "future": future,
        "delta": delta,
        "top_5_riskiest_months": monthly_out.sort_values("proxy_relative_risk", ascending=False)
        .head(5)
        .assign(ds=lambda x: x["ds"].astype(str))
        .to_dict(orient="records"),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_text = build_report_markdown(
        monthly_out,
        annual,
        baseline,
        future,
        delta,
        args,
        risk_threshold_c,
        risk_beta_per_c,
        epi_reference,
        heatwave_event_multiplier,
        realism_summary,
    )
    report_md.write_text(report_text, encoding="utf-8")

    print(f"Wrote: {monthly_csv}")
    print(f"Wrote: {monthly_anom_csv}")
    print(f"Wrote: {annual_csv}")
    print(f"Wrote: {period_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()
