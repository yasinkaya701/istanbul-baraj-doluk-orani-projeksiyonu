#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/yasinkaya/Hackhaton")
FEATURE_STORE = ROOT / "output" / "newdata_feature_store" / "tables"
API_SNAPSHOT = ROOT / "output" / "iski_baraj_api_snapshot" / "tables"
OUT_DIR = ROOT / "output" / "model_useful_data_bundle"


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(FEATURE_STORE / name, **kwargs)


def prepare_monthly_bundle() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    driver = load_csv("istanbul_dam_driver_panel.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    official_supply = load_csv("official_supply_vs_model_consumption_monthly.csv", parse_dates=["date"])
    official_supply = official_supply[
        [
            "date",
            "city_supply_m3_day_avg_official",
            "city_supply_m3_month_official",
            "recorded_water_m3_official",
            "recorded_share_pct",
            "model_vs_supply_ratio_pct",
            "model_vs_recorded_ratio_pct",
            "supply_minus_model_m3",
            "recorded_minus_model_m3",
        ]
    ].copy()

    reanalysis = load_csv("kandilli_openmeteo_monthly_1940_present.csv", parse_dates=["date"])
    reanalysis = reanalysis.rename(
        columns={
            "t_mean_c": "reanalysis_t_mean_c",
            "t_max_c": "reanalysis_t_max_c",
            "t_min_c": "reanalysis_t_min_c",
            "precip_mm_month": "reanalysis_precip_mm_month",
            "rs_mj_m2_month": "reanalysis_rs_mj_m2_month",
            "et0_openmeteo_mm_month": "reanalysis_et0_mm_month",
            "wind_speed_10m_max_m_s": "reanalysis_wind_speed_10m_max_m_s",
            "sunshine_duration_h_month": "reanalysis_sunshine_duration_h_month",
            "is_full_month": "reanalysis_full_month",
        }
    )[
        [
            "date",
            "reanalysis_t_mean_c",
            "reanalysis_t_max_c",
            "reanalysis_t_min_c",
            "reanalysis_precip_mm_month",
            "reanalysis_rs_mj_m2_month",
            "reanalysis_et0_mm_month",
            "reanalysis_wind_speed_10m_max_m_s",
            "reanalysis_sunshine_duration_h_month",
            "reanalysis_full_month",
        ]
    ].copy()

    reanalysis_compare = load_csv("kandilli_openmeteo_vs_local_et0_monthly.csv", parse_dates=["date"])
    reanalysis_compare = reanalysis_compare[
        [
            "date",
            "et0_month_delta_mm",
            "rs_month_delta_mj_m2",
        ]
    ].rename(
        columns={
            "et0_month_delta_mm": "reanalysis_et0_minus_local_mm",
            "rs_month_delta_mj_m2": "reanalysis_rs_minus_local_mj_m2",
        }
    )

    nao = load_csv("noaa_cpc_nao_monthly_1950_present.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    nao["nao_lag1"] = nao["nao_index"].shift(1)
    nao["nao_lag2"] = nao["nao_index"].shift(2)
    nao["nao_roll3"] = nao["nao_index"].rolling(3, min_periods=2).mean()
    nao["nao_positive_flag"] = (nao["nao_index"] > 0).astype(int)

    annual_ops = load_csv("official_iski_operational_context_annual.csv")
    annual_ops = annual_ops[
        [
            "year",
            "active_subscribers",
            "service_population",
            "city_supply_m3_year_report",
            "city_supply_m3_day_avg",
            "reclaimed_water_m3_day",
            "reclaimed_water_m3_year",
            "reclaimed_share_of_system_input_pct",
            "authorized_consumption_m3_per_active_subscriber_year",
            "authorized_consumption_l_per_active_subscriber_day",
            "nrw_m3_per_active_subscriber_year",
        ]
    ].rename(
        columns={
            "city_supply_m3_day_avg": "city_supply_m3_day_avg_report",
            "reclaimed_share_of_system_input_pct": "reclaimed_share_pct",
            "authorized_consumption_l_per_active_subscriber_day": "consumption_liters_per_active_subscriber_day",
        }
    )

    bundle = driver.merge(official_supply, on="date", how="left")
    bundle = bundle.merge(reanalysis, on="date", how="left")
    bundle = bundle.merge(reanalysis_compare, on="date", how="left")
    bundle = bundle.merge(nao, on="date", how="left")
    bundle = bundle.merge(annual_ops, on="year", how="left")

    bundle["official_supply_available"] = bundle["city_supply_m3_month_official"].notna().astype(int)
    bundle["reanalysis_available"] = bundle["reanalysis_et0_mm_month"].notna().astype(int)
    bundle["nao_available"] = bundle["nao_index"].notna().astype(int)

    bundle["weighted_total_fill_lag1"] = bundle["weighted_total_fill"].shift(1)
    bundle["weighted_total_fill_lag2"] = bundle["weighted_total_fill"].shift(2)
    bundle["rain_model_mm_lag1"] = bundle["rain_model_mm"].shift(1)
    bundle["et0_mm_month_lag1"] = bundle["et0_mm_month"].shift(1)
    bundle["consumption_mean_monthly_lag1"] = bundle["consumption_mean_monthly"].shift(1)
    bundle["rain_model_mm_roll3"] = bundle["rain_model_mm"].rolling(3, min_periods=2).mean()
    bundle["et0_mm_month_roll3"] = bundle["et0_mm_month"].rolling(3, min_periods=2).mean()
    bundle["consumption_mean_monthly_roll3"] = bundle["consumption_mean_monthly"].rolling(3, min_periods=2).mean()
    bundle["official_supply_m3_month_roll3"] = bundle["city_supply_m3_month_official"].rolling(3, min_periods=2).mean()

    core_cols = [
        "date",
        "weighted_total_fill",
        "weighted_total_fill_lag1",
        "weighted_total_fill_lag2",
        "rain_model_mm",
        "rain_model_mm_lag1",
        "rain_model_mm_roll3",
        "et0_mm_month",
        "et0_mm_month_lag1",
        "et0_mm_month_roll3",
        "consumption_mean_monthly",
        "consumption_mean_monthly_lag1",
        "consumption_mean_monthly_roll3",
        "temp_proxy_c",
        "rh_proxy_pct",
        "pressure_proxy_kpa",
        "vpd_kpa_mean",
        "water_balance_proxy_mm",
        "month_sin",
        "month_cos",
        "core_obs_score",
    ]

    extended_extra_cols = [
        "city_supply_m3_day_avg_official",
        "city_supply_m3_month_official",
        "official_supply_m3_month_roll3",
        "recorded_water_m3_official",
        "recorded_share_pct",
        "model_vs_supply_ratio_pct",
        "model_vs_recorded_ratio_pct",
        "supply_minus_model_m3",
        "recorded_minus_model_m3",
        "system_input_m3_year",
        "authorized_consumption_m3_year",
        "nrw_m3_year",
        "nrw_pct",
        "administrative_loss_pct",
        "physical_loss_pct",
        "active_subscribers",
        "reclaimed_water_m3_year",
        "reclaimed_share_pct",
        "consumption_liters_per_active_subscriber_day",
        "reanalysis_t_mean_c",
        "reanalysis_t_max_c",
        "reanalysis_t_min_c",
        "reanalysis_precip_mm_month",
        "reanalysis_rs_mj_m2_month",
        "reanalysis_et0_mm_month",
        "reanalysis_wind_speed_10m_max_m_s",
        "reanalysis_sunshine_duration_h_month",
        "reanalysis_full_month",
        "reanalysis_et0_minus_local_mm",
        "reanalysis_rs_minus_local_mj_m2",
        "nao_index",
        "nao_lag1",
        "nao_lag2",
        "nao_roll3",
        "nao_positive_flag",
        "official_supply_available",
        "reanalysis_available",
        "nao_available",
    ]

    core = bundle[core_cols].copy()
    extended = bundle[core_cols + extended_extra_cols].copy()

    coverage_rows: list[dict[str, object]] = []
    blocks = {
        "core": core.columns.drop("date"),
        "official_supply": [
            "city_supply_m3_month_official",
            "recorded_water_m3_official",
            "recorded_share_pct",
        ],
        "annual_operations": [
            "nrw_pct",
            "active_subscribers",
            "reclaimed_share_pct",
        ],
        "reanalysis": [
            "reanalysis_precip_mm_month",
            "reanalysis_rs_mj_m2_month",
            "reanalysis_et0_mm_month",
        ],
        "nao": [
            "nao_index",
            "nao_lag1",
            "nao_roll3",
        ],
    }
    for block_name, cols in blocks.items():
        cols = list(cols)
        available_rows = int(extended[cols].notna().all(axis=1).sum())
        coverage_rows.append(
            {
                "block_name": block_name,
                "row_count": int(len(extended)),
                "rows_with_full_block": available_rows,
                "coverage_pct": float(100.0 * available_rows / len(extended)),
                "start_with_full_block": str(extended.loc[extended[cols].notna().all(axis=1), "date"].min()) if available_rows else None,
                "end_with_full_block": str(extended.loc[extended[cols].notna().all(axis=1), "date"].max()) if available_rows else None,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    return core, extended, coverage


def build_source_current_context() -> pd.DataFrame:
    static = load_csv("official_iski_source_context.csv")
    current = pd.read_csv(API_SNAPSHOT / "baraj_listesi.csv")
    mapping = {
        "Omerli": "Ömerli Barajı",
        "Darlik": "Darlık Barajı",
        "Elmali": "Elmalı 1 ve 2 Barajları",
        "Terkos": "Terkos Barajı",
        "Alibey": "Alibey Barajı",
        "BCekmece": "Büyükçekmece Barajı",
        "Sazlidere": "Sazlıdere Barajı",
        "Istrancalar": "Istrancalar",
        "Kazandere": "Kazandere Barajı",
        "Pabucdere": "Pabuçdere Barajı",
    }
    current["source_name"] = current["kaynakAdi"].map(mapping)
    out = current.merge(static, on="source_name", how="left")
    out["current_storage_to_yield_ratio"] = out["mevcutSuHacmi"] / out["annual_yield_million_m3"]
    out["current_storage_to_max_storage_ratio"] = out["mevcutSuHacmi"] / out["max_storage_million_m3"]
    return out


def build_summary(core: pd.DataFrame, extended: pd.DataFrame, coverage: pd.DataFrame, source_current: pd.DataFrame) -> dict:
    latest = extended.dropna(subset=["weighted_total_fill"]).sort_values("date").iloc[-1]
    return {
        "core_rows": int(len(core)),
        "core_start": str(core["date"].min().date()),
        "core_end": str(core["date"].max().date()),
        "extended_rows": int(len(extended)),
        "official_supply_rows": int(extended["city_supply_m3_month_official"].notna().sum()),
        "reanalysis_rows": int(extended["reanalysis_et0_mm_month"].notna().sum()),
        "nao_rows": int(extended["nao_index"].notna().sum()),
        "latest_training_month": str(latest["date"].date()),
        "latest_weighted_total_fill": float(latest["weighted_total_fill"]),
        "source_current_rows": int(len(source_current)),
        "notes": [
            "Core bundle is the immediate monthly training matrix.",
            "Extended bundle adds official urban-supply, reanalysis proxy, and NAO regime context.",
            "Source-current context is intended for source-aware nowcasting and scenario analysis rather than the aggregate monthly model.",
        ],
    }


def build_readme(summary: dict) -> str:
    return f"""# Istanbul Model Useful Data Bundle

- Core monthly rows: `{summary["core_rows"]}`
- Core window: `{summary["core_start"]}` -> `{summary["core_end"]}`
- Extended monthly rows: `{summary["extended_rows"]}`
- Official supply rows inside extended bundle: `{summary["official_supply_rows"]}`
- Reanalysis rows inside extended bundle: `{summary["reanalysis_rows"]}`
- NAO rows inside extended bundle: `{summary["nao_rows"]}`

Main files:
- `tables/istanbul_model_core_monthly.csv`
- `tables/istanbul_model_extended_monthly.csv`
- `tables/istanbul_model_feature_block_coverage.csv`
- `tables/istanbul_source_current_context.csv`

Use guidance:
- Core monthly: first model runs and ablation.
- Extended monthly: richer feature experiments.
- Source current context: source-aware stress scoring and nowcast explainability.
"""


def main() -> None:
    out_tables = OUT_DIR / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    core, extended, coverage = prepare_monthly_bundle()
    source_current = build_source_current_context()

    core_csv = out_tables / "istanbul_model_core_monthly.csv"
    extended_csv = out_tables / "istanbul_model_extended_monthly.csv"
    coverage_csv = out_tables / "istanbul_model_feature_block_coverage.csv"
    source_current_csv = out_tables / "istanbul_source_current_context.csv"

    core.to_csv(core_csv, index=False)
    extended.to_csv(extended_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    source_current.to_csv(source_current_csv, index=False)

    summary = build_summary(core, extended, coverage, source_current)
    summary_path = OUT_DIR / "model_useful_data_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(build_readme(summary), encoding="utf-8")

    print(core_csv)
    print(extended_csv)
    print(coverage_csv)
    print(source_current_csv)
    print(summary_path)
    print(OUT_DIR / "README.md")


if __name__ == "__main__":
    main()
