#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path("/Users/yasinkaya/Hackhaton")
EXTENDED_PATH = ROOT / "output" / "model_useful_data_bundle" / "tables" / "istanbul_model_extended_monthly.csv"
SOURCE_PRECIP_PATH = ROOT / "output" / "source_precip_proxies" / "source_precip_monthly_wide_2000_2026.csv"
OPS_ANNUAL_PATH = ROOT / "output" / "newdata_feature_store" / "tables" / "official_iski_operational_context_annual.csv"
TRANSFER_ANNUAL_PATH = ROOT / "output" / "istanbul_dam_forward_projection_2040" / "official_transfer_dependency_annual_2021_2025.csv"
OUT_DIR = ROOT / "output" / "newdata_feature_store"
OUT_TABLE = OUT_DIR / "tables" / "monthly_operational_proxies_2000_2026.csv"
OUT_SUMMARY = OUT_DIR / "monthly_operational_proxies_summary.json"


def _safe_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().any():
        return values
    return pd.Series(np.zeros(len(values), dtype=float), index=values.index)


def _zscore(values: pd.Series) -> pd.Series:
    values = _safe_series(values)
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - float(values.mean())) / std


def _positive(values: pd.Series) -> pd.Series:
    return np.maximum(_safe_series(values), 0.0)


def load_monthly_base() -> pd.DataFrame:
    ext = pd.read_csv(EXTENDED_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    src = pd.read_csv(SOURCE_PRECIP_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    src["src_rain_mean"] = src[[c for c in src.columns if c != "date"]].mean(axis=1)
    src["src_rain_north"] = src[["Terkos", "Kazandere", "Pabucdere", "Istrancalar"]].mean(axis=1)
    src["src_rain_west"] = src[["Alibey", "Buyukcekmece", "Sazlidere"]].mean(axis=1)
    df = ext.merge(
        src[["date", "src_rain_mean", "src_rain_north", "src_rain_west"]],
        on="date",
        how="left",
    )
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["days_in_month"] = df["date"].dt.days_in_month
    fallback_supply = df["consumption_mean_monthly"] * df["days_in_month"]
    df["official_supply_m3_month"] = df["city_supply_m3_month_official"].fillna(fallback_supply)
    month_supply_clim = df.groupby("month")["official_supply_m3_month"].transform("mean")
    df["official_supply_m3_month"] = df["official_supply_m3_month"].fillna(month_supply_clim).ffill().bfill()
    df["official_supply_mcm"] = df["official_supply_m3_month"] / 1e6
    df["drought_stress_mm"] = np.maximum(df["et0_mm_month"] - df["rain_model_mm"], 0.0)
    src_median = float(df["src_rain_mean"].median())
    df["source_rain_deficit_mm"] = np.maximum(src_median - df["src_rain_mean"], 0.0)
    return df


def estimate_annual_transfer_share(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    transfer = pd.read_csv(TRANSFER_ANNUAL_PATH, parse_dates=["tarih"]).copy()
    transfer["year"] = transfer["tarih"].dt.year
    transfer = transfer.rename(columns={"transfer_share_pct": "transfer_share_pct_official"})

    annual = (
        df.groupby("year")
        .agg(
            annual_supply_mcm=("official_supply_mcm", "sum"),
            annual_drought_mm=("drought_stress_mm", "mean"),
            annual_source_deficit_mm=("source_rain_deficit_mm", "mean"),
            annual_et0_mm=("et0_mm_month", "mean"),
            annual_rain_mm=("rain_model_mm", "mean"),
        )
        .reset_index()
    )
    annual = annual.merge(transfer[["year", "transfer_share_pct_official"]], on="year", how="left")

    known = annual.dropna(subset=["transfer_share_pct_official"]).copy()
    predictors = ["annual_supply_mcm", "annual_drought_mm", "annual_source_deficit_mm"]
    if len(known) >= 3:
        x_mean = known[predictors].mean()
        x_std = known[predictors].std(ddof=0).replace(0.0, 1.0)
        model = RidgeCV(alphas=np.logspace(-4, 4, 41))
        model.fit((known[predictors] - x_mean) / x_std, known["transfer_share_pct_official"])
        annual["transfer_share_pct_est"] = model.predict((annual[predictors] - x_mean) / x_std)
    else:
        annual["transfer_share_pct_est"] = float(known["transfer_share_pct_official"].mean()) if not known.empty else 0.0

    annual["transfer_share_pct_est"] = annual["transfer_share_pct_est"].clip(10.0, 80.0)
    annual.loc[annual["transfer_share_pct_official"].notna(), "transfer_share_pct_est"] = annual["transfer_share_pct_official"]
    annual["transfer_share_source"] = np.where(
        annual["transfer_share_pct_official"].notna(),
        "official",
        "estimated",
    )
    share_map = dict(zip(annual["year"], annual["transfer_share_pct_est"]))
    return annual, share_map


def add_monthly_transfer_proxy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_transfer, share_map = estimate_annual_transfer_share(df)
    out = df.copy()
    out["transfer_share_pct_annual_est"] = out["year"].map(share_map).astype(float)
    out["transfer_share_source"] = out["year"].map(
        dict(zip(annual_transfer["year"], annual_transfer["transfer_share_source"]))
    )

    monthly_parts = []
    for year, g in out.groupby("year", sort=True):
        g = g.copy()
        demand_z = _positive(_zscore(g["official_supply_mcm"]))
        drought_z = _positive(_zscore(g["drought_stress_mm"]))
        source_deficit_z = _positive(_zscore(g["source_rain_deficit_mm"]))
        summer = g["month"].isin([6, 7, 8, 9]).astype(float)
        shoulder = g["month"].isin([5, 10]).astype(float)
        raw = 1.0 + 0.45 * drought_z + 0.25 * demand_z + 0.20 * source_deficit_z + 0.15 * summer + 0.05 * shoulder
        raw = np.clip(raw, 0.25, None)
        annual_transfer_mcm = float(g["transfer_share_pct_annual_est"].iloc[0] / 100.0 * g["official_supply_mcm"].sum())
        g["transfer_mcm_monthly_proxy"] = annual_transfer_mcm * raw / float(raw.sum())
        g["transfer_share_pct_monthly_proxy"] = 100.0 * g["transfer_mcm_monthly_proxy"] / g["official_supply_mcm"].replace(0.0, np.nan)
        g["transfer_share_pct_monthly_proxy"] = g["transfer_share_pct_monthly_proxy"].clip(0.0, 95.0).fillna(0.0)
        g["transfer_mcm_monthly_proxy"] = g["official_supply_mcm"] * g["transfer_share_pct_monthly_proxy"] / 100.0
        monthly_parts.append(g)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    return monthly, annual_transfer


def add_monthly_nrw_proxy(df: pd.DataFrame) -> pd.DataFrame:
    ops = pd.read_csv(OPS_ANNUAL_PATH)
    annual = (
        ops[["year", "nrw_pct", "reclaimed_share_of_system_input_pct"]]
        .rename(columns={"reclaimed_share_of_system_input_pct": "reclaimed_share_pct_annual"})
        .sort_values("year")
    )
    year_range = pd.DataFrame({"year": np.arange(int(df["year"].min()), int(df["year"].max()) + 1)})
    annual = year_range.merge(annual, on="year", how="left").sort_values("year")
    annual["nrw_pct_annual_est"] = annual["nrw_pct"].interpolate(limit_direction="both").ffill().bfill()
    annual["reclaimed_share_pct_annual_est"] = annual["reclaimed_share_pct_annual"].interpolate(limit_direction="both").ffill().bfill()
    annual["nrw_source"] = np.where(annual["nrw_pct"].notna(), "official", "interpolated")

    out = df.copy()
    out = out.merge(
        annual[["year", "nrw_pct_annual_est", "reclaimed_share_pct_annual_est", "nrw_source"]],
        on="year",
        how="left",
    )
    parts = []
    for _, g in out.groupby("year", sort=True):
        g = g.copy()
        demand_z = _positive(_zscore(g["official_supply_mcm"]))
        temp_z = _positive(_zscore(g["temp_proxy_c"]))
        vpd_z = _positive(_zscore(g["vpd_kpa_mean"]))
        raw = np.exp(0.18 * demand_z + 0.15 * temp_z + 0.10 * vpd_z)
        mean_raw = float(raw.mean()) if float(raw.mean()) > 0 else 1.0
        g["nrw_pct_monthly_proxy"] = g["nrw_pct_annual_est"] * raw / mean_raw
        g["nrw_mcm_monthly_proxy"] = g["official_supply_mcm"] * g["nrw_pct_monthly_proxy"] / 100.0
        g["reclaimed_share_pct_monthly_proxy"] = g["reclaimed_share_pct_annual_est"]
        g["reclaimed_mcm_monthly_proxy"] = g["official_supply_mcm"] * g["reclaimed_share_pct_monthly_proxy"] / 100.0
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

    df = load_monthly_base()
    df, annual_transfer = add_monthly_transfer_proxy(df)
    df = add_monthly_nrw_proxy(df)

    keep = [
        "date",
        "year",
        "month",
        "official_supply_m3_month",
        "official_supply_mcm",
        "transfer_share_pct_annual_est",
        "transfer_share_pct_monthly_proxy",
        "transfer_mcm_monthly_proxy",
        "transfer_share_source",
        "nrw_pct_annual_est",
        "nrw_pct_monthly_proxy",
        "nrw_mcm_monthly_proxy",
        "nrw_source",
        "reclaimed_share_pct_annual_est",
        "reclaimed_share_pct_monthly_proxy",
        "reclaimed_mcm_monthly_proxy",
        "drought_stress_mm",
        "source_rain_deficit_mm",
        "src_rain_mean",
        "src_rain_north",
        "src_rain_west",
    ]
    out = df[keep].sort_values("date").reset_index(drop=True)
    out.to_csv(OUT_TABLE, index=False)

    summary = {
        "row_count": int(len(out)),
        "date_min": str(out["date"].min().date()),
        "date_max": str(out["date"].max().date()),
        "official_transfer_years": [int(y) for y in annual_transfer.loc[annual_transfer["transfer_share_source"] == "official", "year"].tolist()],
        "annual_transfer_share_mean_pct": float(out["transfer_share_pct_annual_est"].mean()),
        "monthly_transfer_share_mean_pct": float(out["transfer_share_pct_monthly_proxy"].mean()),
        "annual_nrw_pct_mean": float(out["nrw_pct_annual_est"].mean()),
        "monthly_nrw_pct_mean": float(out["nrw_pct_monthly_proxy"].mean()),
    }
    OUT_SUMMARY.write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
