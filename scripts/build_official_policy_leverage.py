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


TOTAL_ACTIVE_STORAGE_MCM = 868.683
TOTAL_ACTIVE_STORAGE_M3 = TOTAL_ACTIVE_STORAGE_MCM * 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build annual policy leverage analysis from official ISKI context.")
    parser.add_argument(
        "--context-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_operational_context_annual.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store"),
    )
    return parser.parse_args()


def build_leverage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["active_subscribers"].notna()].copy()

    out["delta_volume_1pp_nrw_reduction_m3"] = out["system_input_m3_year"] * 0.01
    out["delta_fill_1pp_nrw_reduction_pp"] = 100.0 * out["delta_volume_1pp_nrw_reduction_m3"] / TOTAL_ACTIVE_STORAGE_M3

    out["delta_volume_1pct_authorized_demand_reduction_m3"] = out["authorized_consumption_m3_year"] * 0.01
    out["delta_fill_1pct_authorized_demand_reduction_pp"] = 100.0 * out["delta_volume_1pct_authorized_demand_reduction_m3"] / TOTAL_ACTIVE_STORAGE_M3

    out["delta_volume_10pct_reclaimed_increase_m3"] = out["reclaimed_water_m3_year"] * 0.10
    out["delta_fill_10pct_reclaimed_increase_pp"] = 100.0 * out["delta_volume_10pct_reclaimed_increase_m3"] / TOTAL_ACTIVE_STORAGE_M3

    out["delta_volume_100k_subscriber_growth_m3"] = 100_000.0 * out["authorized_consumption_m3_per_active_subscriber_year"]
    out["delta_fill_100k_subscriber_growth_pp"] = -100.0 * out["delta_volume_100k_subscriber_growth_m3"] / TOTAL_ACTIVE_STORAGE_M3

    out["delta_volume_5pct_reclaimed_share_target_m3"] = np.maximum(
        0.0,
        0.05 * out["system_input_m3_year"] - out["reclaimed_water_m3_year"],
    )
    out["delta_fill_5pct_reclaimed_share_target_pp"] = 100.0 * out["delta_volume_5pct_reclaimed_share_target_m3"] / TOTAL_ACTIVE_STORAGE_M3
    return out


def plot_latest_year(leverage: pd.DataFrame, out_path: Path) -> None:
    latest = leverage.sort_values("year").iloc[-1]
    labels = [
        "NRW -1 yp",
        "Talep -1%",
        "Geri kazanim +10%",
        "100k abone artis",
        "Geri kazanim %5 hedef",
    ]
    values = [
        latest["delta_fill_1pp_nrw_reduction_pp"],
        latest["delta_fill_1pct_authorized_demand_reduction_pp"],
        latest["delta_fill_10pct_reclaimed_increase_pp"],
        latest["delta_fill_100k_subscriber_growth_pp"],
        latest["delta_fill_5pct_reclaimed_share_target_pp"],
    ]
    colors = ["#0f766e", "#2563eb", "#7c3aed", "#b91c1c", "#b45309"]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#475569", linewidth=1.0)
    ax.set_ylabel("Toplam doluluk esdegeri (yuzde puan)")
    ax.set_title(f"Resmi yillik kaldirac etkisi ({int(latest['year'])})")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_horizontalalignment("right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(leverage: pd.DataFrame) -> dict[str, object]:
    latest = leverage.sort_values("year").iloc[-1]
    return {
        "storage_basis_mcm": TOTAL_ACTIVE_STORAGE_MCM,
        "years": leverage["year"].astype(int).tolist(),
        "latest_year": int(latest["year"]),
        "latest_year_levers": {
            "delta_fill_1pp_nrw_reduction_pp": float(latest["delta_fill_1pp_nrw_reduction_pp"]),
            "delta_fill_1pct_authorized_demand_reduction_pp": float(latest["delta_fill_1pct_authorized_demand_reduction_pp"]),
            "delta_fill_10pct_reclaimed_increase_pp": float(latest["delta_fill_10pct_reclaimed_increase_pp"]),
            "delta_fill_100k_subscriber_growth_pp": float(latest["delta_fill_100k_subscriber_growth_pp"]),
            "delta_fill_5pct_reclaimed_share_target_pp": float(latest["delta_fill_5pct_reclaimed_share_target_pp"]),
        },
        "notes": [
            "These are upper-bound annual occupancy-equivalent calculations relative to total active storage capacity.",
            "They assume one-for-one reduction or substitution against potable-system withdrawals over the annual horizon.",
            "They are useful for policy leverage ranking, not as direct month-by-month causal estimates.",
        ],
    }


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_figures = args.out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    context = pd.read_csv(args.context_csv)
    leverage = build_leverage(context)
    leverage.to_csv(out_tables / "official_policy_leverage_annual.csv", index=False)
    plot_latest_year(leverage, out_figures / "official_policy_leverage_latest_year.png")
    summary = build_summary(leverage)
    (args.out_dir / "official_policy_leverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(out_tables / "official_policy_leverage_annual.csv")
    print(out_figures / "official_policy_leverage_latest_year.png")
    print(args.out_dir / "official_policy_leverage_summary.json")


if __name__ == "__main__":
    main()
