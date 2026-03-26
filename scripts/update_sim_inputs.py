#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/yasinkaya/Hackhaton")
PANEL = ROOT / "output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_extended.csv"
CLIMATE_PATH = ROOT / "assets/data/climate_baseline.js"
ISKI_ANNUAL_SUPPLY_PATH = ROOT / "output/iski_baraj_api_snapshot/tables/son_10_yil_toplam_verilen_su.csv"

USAGE_PROFILE_OUT = [
    ROOT / "assets/data/usage_monthly_profile.js",
    ROOT / "baraj_web/assets/data/usage_monthly_profile.js",
]
USAGE_TREND_OUT = [
    ROOT / "assets/data/usage_trend_stats.js",
    ROOT / "baraj_web/assets/data/usage_trend_stats.js",
]
SIM_COEFFS_OUT = [
    ROOT / "assets/data/sim_coeffs.js",
    ROOT / "baraj_web/assets/data/sim_coeffs.js",
]
BASELINE_OUT = [
    ROOT / "assets/data/evap_usage_baseline.js",
    ROOT / "baraj_web/assets/data/evap_usage_baseline.js",
]


def load_js_payload(path: Path, prefix: str) -> dict:
    raw = path.read_text().strip()
    if not raw.startswith(prefix):
        raise ValueError(f"Unexpected JS format in {path}")
    payload = raw[len(prefix):].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    return json.loads(payload)


def load_reservoir_reference_simple(path: Path) -> dict:
    text = path.read_text()
    def find_num(key: str) -> float | None:
        match = re.search(rf"{key}\\s*:\\s*([0-9.]+)", text)
        return float(match.group(1)) if match else None
    def find_str(key: str) -> str | None:
        match = re.search(rf"{key}\\s*:\\s*\\\"([^\\\"]+)\\\"", text)
        return match.group(1) if match else None
    return {
        "total_catchment_km2": find_num("total_catchment_km2") or 3150.0,
        "total_active_capacity_mcm": find_num("total_active_capacity_mcm") or 868.683,
        "source": find_str("source") or "",
        "source_page": find_str("source_page") or "",
        "updated_at": find_str("updated_at") or "",
    }


def write_js_payload(path: Path, var_name: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"window.{var_name} = " + json.dumps(payload, ensure_ascii=False) + ";"
    path.write_text(text, encoding="utf-8")


def build_usage_stats(panel: pd.DataFrame) -> tuple[dict, dict]:
    df = panel[["date", "consumption_mean_monthly"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["days"] = df["date"].dt.days_in_month
    df["month_total"] = df["consumption_mean_monthly"] * df["days"]

    counts = df.groupby("year")["month"].nunique()
    full_years = counts[counts == 12].index
    df_full = df[df["year"].isin(full_years)].copy()
    annual = df_full.groupby("year")["month_total"].sum().sort_index()
    yoy = annual.pct_change().dropna()

    year_min = int(annual.index.min())
    year_max = int(annual.index.max())
    n_years = int(len(annual))
    span_years = max(1, year_max - year_min)
    cagr_all = float((annual.iloc[-1] / annual.iloc[0]) ** (1 / span_years) - 1) if n_years > 1 else 0.0

    cagr_2019_2023 = None
    if 2019 in annual.index and 2023 in annual.index:
        cagr_2019_2023 = float((annual.loc[2023] / annual.loc[2019]) ** (1 / 4) - 1)

    trend_payload = {
        "year_min": year_min,
        "year_max": year_max,
        "n_years": n_years,
        "cagr_all": cagr_all,
        "cagr_2019_2023": cagr_2019_2023 if cagr_2019_2023 is not None else cagr_all,
        "yoy_mean": float(yoy.mean()) if len(yoy) else 0.0,
        "yoy_median": float(yoy.median()) if len(yoy) else 0.0,
        "yoy_p10": float(np.percentile(yoy, 10)) if len(yoy) else 0.0,
        "yoy_p90": float(np.percentile(yoy, 90)) if len(yoy) else 0.0,
        "trend_source": "panel_consumption_mean_monthly_full_years",
    }

    # Prefer official annual total-supply trend when available.
    if ISKI_ANNUAL_SUPPLY_PATH.exists():
        try:
            off = pd.read_csv(ISKI_ANNUAL_SUPPLY_PATH)
            if {"tarih", "verilenTemizsuM3"}.issubset(set(off.columns)):
                off_year = pd.to_datetime(off["tarih"], errors="coerce").dt.year
                off_val = pd.to_numeric(off["verilenTemizsuM3"], errors="coerce")
                annual_off = (
                    pd.DataFrame({"year": off_year, "value": off_val})
                    .dropna()
                    .groupby("year")["value"]
                    .mean()
                    .sort_index()
                )
                # Drop likely partial latest year (typically much smaller than previous full year)
                if len(annual_off) >= 2 and annual_off.iloc[-1] < annual_off.iloc[-2] * 0.6:
                    annual_off = annual_off.iloc[:-1]
                yoy_off = annual_off.pct_change().dropna()
                if len(annual_off) >= 4 and len(yoy_off):
                    y_min = int(annual_off.index.min())
                    y_max = int(annual_off.index.max())
                    span = max(1, y_max - y_min)
                    cagr_off = float((annual_off.iloc[-1] / annual_off.iloc[0]) ** (1 / span) - 1)
                    cagr_2019_2023_off = cagr_off
                    if 2019 in annual_off.index and 2023 in annual_off.index:
                        cagr_2019_2023_off = float((annual_off.loc[2023] / annual_off.loc[2019]) ** (1 / 4) - 1)
                    trend_payload.update(
                        {
                            "year_min": y_min,
                            "year_max": y_max,
                            "n_years": int(len(annual_off)),
                            "cagr_all": cagr_off,
                            "cagr_2019_2023": cagr_2019_2023_off,
                            "yoy_mean": float(yoy_off.mean()),
                            "yoy_median": float(yoy_off.median()),
                            "yoy_p10": float(np.percentile(yoy_off, 10)),
                            "yoy_p90": float(np.percentile(yoy_off, 90)),
                            "trend_source": "iski_baraj_api_son_10_yil_toplam_verilen_su",
                        }
                    )
        except Exception:
            pass

    monthly_share = (
        df_full.groupby(["year", "month"])["month_total"]
        .sum()
        .div(annual, level=0)
        .groupby("month")
        .mean()
    )
    profile = [float(monthly_share.loc[m]) for m in range(1, 13)]
    profile_payload = {
        "year_min": year_min,
        "year_max": year_max,
        "months": list(range(1, 13)),
        "profile": profile,
        "note": f"Aylik tuketim dagilimi ({year_min}-{year_max} tam yil).",
    }

    return trend_payload, profile_payload


def compute_adjustment_scale(panel: pd.DataFrame, baseline: dict, reservoir: dict, runoff_coeff: float) -> float:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"weighted_total_fill": "fill_pct"})
    df = df[["date", "fill_pct", "rain_mm", "et0_mm_month", "consumption_mean_monthly"]].dropna().copy()
    df = df.sort_values("date")
    df = df[(df["date"] >= "2010-01-01") & (df["date"] <= "2024-02-01")]
    # Source panel keeps fill as 0..1 ratio; convert to 0..100 percentage points.
    if float(df["fill_pct"].max()) <= 1.5:
        df["fill_pct"] = df["fill_pct"] * 100.0
    df["delta_fill"] = df["fill_pct"].diff(1)
    df = df.dropna()

    area_km2 = float(baseline.get("area_km2_total", 99.69))
    kc = float(baseline.get("kc_open_water", 1.05))
    catchment_km2 = float(reservoir.get("total_catchment_km2", 3150.0))
    capacity_mcm = float(reservoir.get("total_active_capacity_mcm", 868.683))
    if capacity_mcm <= 0:
        return 1.0

    inflow = df["rain_mm"] * catchment_km2 * 0.001
    lake_rain = df["rain_mm"] * area_km2 * 0.001
    evap = df["et0_mm_month"] * kc * area_km2 * 0.001
    use = df["consumption_mean_monthly"] / 1e6
    pred = (runoff_coeff * inflow + lake_rain - evap - use) / capacity_mcm * 100
    obs = df["delta_fill"]

    pred_abs = np.median(np.abs(pred))
    obs_abs = np.median(np.abs(obs))
    if pred_abs <= 0:
        return 1.0
    scale = float(obs_abs / pred_abs)
    # Keep within a realistic, visible band (monthly scenario effect calibration).
    return float(np.clip(scale, 0.05, 0.8))


def build_sim_coeffs(panel: pd.DataFrame, climate: dict, existing: dict, baseline: dict, reservoir: dict) -> dict:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"weighted_total_fill": "fill_pct"})
    df = df[["date", "fill_pct", "rain_mm", "et0_mm_month", "consumption_mean_monthly"]].dropna().copy()
    df = df.sort_values("date")
    # Source panel keeps fill as 0..1 ratio; convert to 0..100 percentage points.
    if float(df["fill_pct"].max()) <= 1.5:
        df["fill_pct"] = df["fill_pct"] * 100.0
    df["delta_fill"] = df["fill_pct"].diff(1)
    df = df.dropna()

    X = df[["rain_mm", "et0_mm_month", "consumption_mean_monthly"]].values
    X = np.column_stack([X, np.ones(len(X))])
    Y = df["delta_fill"].values
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    a, b, c, intercept = [float(v) for v in coef.tolist()]

    # Standardized coefficients: used to keep rain/ET0 slider effects balanced.
    z = df[["rain_mm", "et0_mm_month", "consumption_mean_monthly", "delta_fill"]].copy()
    for col in z.columns:
        std = float(z[col].std(ddof=0))
        if std > 0:
            z[col] = (z[col] - float(z[col].mean())) / std
        else:
            z[col] = 0.0
    Xz = z[["rain_mm", "et0_mm_month", "consumption_mean_monthly"]].values
    Yz = z["delta_fill"].values
    zcoef, *_ = np.linalg.lstsq(np.column_stack([Xz, np.ones(len(Xz))]), Yz, rcond=None)
    beta_rain = abs(float(zcoef[0]))
    beta_et0 = abs(float(zcoef[1]))
    beta_use = abs(float(zcoef[2]))

    rows = []
    for date_str, vals in climate.items():
        year = int(date_str.split("-")[0])
        if 2010 <= year <= 2024:
            rows.append(vals)
    mean_rain = float(np.mean([r["precip_mm_month"] for r in rows])) if rows else float(df["rain_mm"].mean())
    mean_et0 = float(np.mean([r["et0_mm_month"] for r in rows])) if rows else float(df["et0_mm_month"].mean())
    mean_use = float(df["consumption_mean_monthly"].mean())

    # Keep scenario shocks in a realistic monthly band using observed fill deltas.
    obs = panel.copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs[(obs["date"] >= "2010-01-01") & (obs["date"] <= "2024-02-01")]
    obs = obs[["date", "weighted_total_fill"]].dropna().sort_values("date")
    # Source panel keeps fill as 0..1 ratio; convert to 0..100 percentage points.
    fill_series = obs["weighted_total_fill"] * 100.0 if float(obs["weighted_total_fill"].max()) <= 1.5 else obs["weighted_total_fill"]
    obs["delta_fill"] = fill_series.diff(1)
    abs_delta = obs["delta_fill"].abs().dropna()
    if len(abs_delta):
        # Keep monthly scenario increment realistic but visible for policy stress tests.
        scenario_delta_cap_pp = float(np.clip(np.percentile(abs_delta, 75) * 0.22, 1.6, 2.8))
    else:
        scenario_delta_cap_pp = 2.0

    updated = dict(existing)
    runoff_coeff = float(existing.get("runoff_coeff", 0.6))
    adjustment_scale = compute_adjustment_scale(panel, baseline, reservoir, runoff_coeff)

    # Calibrated gains (increased to keep sliders decision-relevant while bounded).
    rain_sensitivity_gain = float(np.clip(0.58 + 0.52 * beta_rain, 0.62, 0.88))
    et0_sensitivity_gain = float(np.clip(1.12 + 0.95 * beta_et0, 1.20, 1.60))
    et0_effect_gain = float(np.clip(1.05 + 0.45 * beta_et0, 1.15, 1.45))
    use_sensitivity_gain = float(np.clip(1.60 + 1.20 * beta_use, 1.55, 2.00))
    scenario_response_gain = float(np.clip(0.95 + 0.75 * adjustment_scale, 1.10, 1.45))

    updated.update(
        {
            "a_rain": a,
            "b_et0": b,
            "c_use": c,
            "intercept": intercept,
            "mean_rain": mean_rain,
            "mean_et0": mean_et0,
            "mean_use_monthly": mean_use,
            "adjustment_cap_scale": adjustment_scale,
            "scenario_delta_cap_pp": scenario_delta_cap_pp,
            "scenario_damping_floor": 0.25,
            "scenario_offset_cap_pp": 35.0,
            "scenario_offset_damping_floor": 0.15,
            "scenario_response_gain": scenario_response_gain,
            "rain_sensitivity_gain": rain_sensitivity_gain,
            "et0_sensitivity_gain": et0_sensitivity_gain,
            "et0_effect_gain": et0_effect_gain,
            "use_sensitivity_gain": use_sensitivity_gain,
            "et0_runoff_beta": 0.70,
            "k_source": "observed_fill_pct_2010_2024_vs_balance_median_scale",
        }
    )
    return updated


def build_baseline(climate: dict, baseline: dict) -> dict:
    year = int(baseline.get("year", 2023))
    area_km2_total = float(baseline.get("area_km2_total", 99.69))
    kc = float(baseline.get("kc_open_water", 1.05))
    usage_m3 = float(baseline.get("usage_baraj_m3", 275_104_161))

    rows = []
    for date_str, vals in climate.items():
        if date_str.startswith(f"{year}-"):
            rows.append(vals)
    if not rows:
        rows = list(climate.values())

    area_m2 = area_km2_total * 1_000_000.0
    evap_total_m3 = float(
        sum((r["et0_mm_month"] / 1000.0) * kc * area_m2 for r in rows)
    )
    total_loss = evap_total_m3 + usage_m3
    evap_share = evap_total_m3 / total_loss if total_loss else 0.0
    usage_share = usage_m3 / total_loss if total_loss else 0.0

    updated = dict(baseline)
    updated.update(
        {
            "evap_total_m3": evap_total_m3,
            "usage_baraj_m3": usage_m3,
            "evap_share": evap_share,
            "usage_share": usage_share,
        }
    )
    return updated


def main() -> None:
    panel = pd.read_csv(PANEL)
    climate = load_js_payload(CLIMATE_PATH, "window.CLIMATE_BASELINE = ")
    reservoir = load_reservoir_reference_simple(ROOT / "assets/data/reservoir_reference.js")

    trend_payload, profile_payload = build_usage_stats(panel)
    for out in USAGE_TREND_OUT:
        write_js_payload(out, "USAGE_TREND", trend_payload)
    for out in USAGE_PROFILE_OUT:
        write_js_payload(out, "USAGE_PROFILE", profile_payload)

    existing_coeffs = load_js_payload(SIM_COEFFS_OUT[0], "window.SIM_COEFFS = ")
    baseline_existing = load_js_payload(BASELINE_OUT[0], "window.BASELINE = ")
    updated_coeffs = build_sim_coeffs(panel, climate, existing_coeffs, baseline_existing, reservoir)
    for out in SIM_COEFFS_OUT:
        write_js_payload(out, "SIM_COEFFS", updated_coeffs)

    updated_baseline = build_baseline(climate, baseline_existing)
    for out in BASELINE_OUT:
        write_js_payload(out, "BASELINE", updated_baseline)

    print("Updated usage trend/profile, sim coeffs, and baseline.")


if __name__ == "__main__":
    main()
