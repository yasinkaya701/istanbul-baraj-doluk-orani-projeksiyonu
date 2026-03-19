#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build annual official ISKI operational-context table for the dam project.")
    parser.add_argument(
        "--water-loss-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_water_loss_annual.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store"),
    )
    return parser.parse_args()


def build_activity_report_context() -> pd.DataFrame:
    rows = [
        {
            "year": 2020,
            "active_subscribers": 6_607_981,
            "service_population": np.nan,
            "city_supply_m3_year_report": 1_074_133_977.0,
            "city_supply_m3_day_avg": 2_934_792.0,
            "reclaimed_water_m3_day": 68_136.0,
            "reclaimed_water_m3_year": 24_869_460.0,
            "source_report_pdf": "https://cdn.iski.istanbul/uploads/2020_FAALIYET_RAPORU_903efe0267.pdf",
        },
        {
            "year": 2021,
            "active_subscribers": 6_723_824,
            "service_population": 15_840_900.0,
            "city_supply_m3_year_report": 1_073_990_361.0,
            "city_supply_m3_day_avg": 2_942_439.0,
            "reclaimed_water_m3_day": 85_262.0,
            "reclaimed_water_m3_year": 31_120_667.0,
            "source_report_pdf": "https://cdn.iski.istanbul/uploads/2021_FAALIYET_RAPORU_64bf206f27.pdf",
        },
        {
            "year": 2022,
            "active_subscribers": 6_818_930,
            "service_population": np.nan,
            "city_supply_m3_year_report": 1_103_672_069.0,
            "city_supply_m3_day_avg": 3_023_759.0,
            "reclaimed_water_m3_day": 81_160.0,
            "reclaimed_water_m3_year": 29_623_315.0,
            "source_report_pdf": "https://cdn.iski.istanbul/uploads/2022_Faaliyet_Raporu_c65c8a733d.pdf",
        },
        {
            "year": 2023,
            "active_subscribers": 6_891_231,
            "service_population": 15_655_924.0,
            "city_supply_m3_year_report": 1_117_064_116.0,
            "city_supply_m3_day_avg": 3_060_450.0,
            "reclaimed_water_m3_day": 80_235.0,
            "reclaimed_water_m3_year": 29_285_760.0,
            "source_report_pdf": "https://iskiapi.iski.gov.tr/uploads/2023_Yili_Faaliyet_Raporu_24309dd9dd.pdf",
        },
    ]
    return pd.DataFrame(rows)


def build_context(loss_csv: Path) -> pd.DataFrame:
    loss = pd.read_csv(loss_csv)
    context = build_activity_report_context()
    df = loss.merge(context, on="year", how="left")
    df["city_supply_gap_m3"] = df["city_supply_m3_year_report"] - df["system_input_m3_year"]
    df["reclaimed_share_of_system_input_pct"] = 100.0 * df["reclaimed_water_m3_year"] / df["system_input_m3_year"]
    df["authorized_consumption_m3_per_active_subscriber_year"] = (
        df["authorized_consumption_m3_year"] / df["active_subscribers"]
    )
    df["authorized_consumption_l_per_active_subscriber_day"] = (
        df["authorized_consumption_m3_year"] * 1000.0 / df["active_subscribers"] / 365.0
    )
    df["nrw_m3_per_active_subscriber_year"] = df["nrw_m3_year"] / df["active_subscribers"]
    return df


def plot_context(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.4), sharex=True)

    plot_df = df.sort_values("year").copy()
    plot_df = plot_df[plot_df["year"] >= 2020]

    ax = axes[0]
    ax.plot(plot_df["year"], plot_df["active_subscribers"] / 1e6, marker="o", linewidth=2.2, color="#1d4ed8", label="Aktif abone (milyon)")
    ax2 = ax.twinx()
    ax2.plot(plot_df["year"], plot_df["authorized_consumption_l_per_active_subscriber_day"], marker="o", linewidth=2.0, color="#0f766e", label="Yetkili tuketim / aktif abone")
    ax.set_ylabel("Aktif abone (milyon)")
    ax2.set_ylabel("L / aktif abone / gun")
    ax.set_title("ISKI resmi operasyonel baglam: abone ve tuketim yogunlugu")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(plot_df["year"], plot_df["nrw_pct"], marker="o", linewidth=2.0, color="#b91c1c", label="NRW yuzdesi")
    ax.plot(plot_df["year"], plot_df["reclaimed_share_of_system_input_pct"], marker="o", linewidth=2.0, color="#7c3aed", label="Geri kazanilmis su payi")
    ax.set_ylabel("Yuzde")
    ax.set_title("ISKI resmi operasyonel baglam: kayip ve geri kazanilmis su")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("Yil")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    sub = df[df["active_subscribers"].notna()].sort_values("year")
    return {
        "years_with_activity_context": sub["year"].astype(int).tolist(),
        "years_with_loss_context": df["year"].astype(int).tolist(),
        "active_subscriber_growth_2020_2023_pct": float(
            100.0 * (sub["active_subscribers"].iloc[-1] - sub["active_subscribers"].iloc[0]) / sub["active_subscribers"].iloc[0]
        ),
        "authorized_consumption_l_per_active_subscriber_day_2020": float(
            sub.loc[sub["year"] == 2020, "authorized_consumption_l_per_active_subscriber_day"].iloc[0]
        ),
        "authorized_consumption_l_per_active_subscriber_day_2023": float(
            sub.loc[sub["year"] == 2023, "authorized_consumption_l_per_active_subscriber_day"].iloc[0]
        ),
        "reclaimed_share_pct_2020": float(sub.loc[sub["year"] == 2020, "reclaimed_share_of_system_input_pct"].iloc[0]),
        "reclaimed_share_pct_2023": float(sub.loc[sub["year"] == 2023, "reclaimed_share_of_system_input_pct"].iloc[0]),
        "notes": [
            "Active subscriber count is an official system-wide count; it is not a household-only metric.",
            "Authorized consumption per active subscriber is an intensity proxy, not a direct household liters-per-capita metric.",
            "Reclaimed-water share is small relative to total system input, but it is explicit and growing enough to keep as a future substitution variable.",
        ],
    }


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_figures = args.out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    df = build_context(args.water_loss_csv)
    plot_context(df, out_figures / "official_operational_context_trends.png")
    df.to_csv(out_tables / "official_iski_operational_context_annual.csv", index=False)
    summary = build_summary(df)
    (args.out_dir / "official_operational_context_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(out_tables / "official_iski_operational_context_annual.csv")
    print(out_figures / "official_operational_context_trends.png")
    print(args.out_dir / "official_operational_context_summary.json")


if __name__ == "__main__":
    main()
