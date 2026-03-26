#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MONTH_MAP = {
    "Ocak": 1,
    "Şubat": 2,
    "Mart": 3,
    "Nisan": 4,
    "Mayıs": 5,
    "Haziran": 6,
    "Temmuz": 7,
    "Ağustos": 8,
    "Eylül": 9,
    "Ekim": 10,
    "Kasım": 11,
    "Aralık": 12,
}


CITY_SUPPLY_DAILY_AVG = {
    2010: {
        "Ocak": 1955690,
        "Şubat": 1985977,
        "Mart": 1957753,
        "Nisan": 2001995,
        "Mayıs": 2197763,
        "Haziran": 2193174,
        "Temmuz": 2203668,
        "Ağustos": 2438510,
        "Eylül": 2224480,
        "Ekim": 2142555,
        "Kasım": 2136245,
        "Aralık": 2128035,
    },
    2011: {
        "Ocak": 2102517,
        "Şubat": 2112808,
        "Mart": 2125656,
        "Nisan": 2105965,
        "Mayıs": 2234926,
        "Haziran": 2404202,
        "Temmuz": 2481266,
        "Ağustos": 2450216,
        "Eylül": 2396856,
        "Ekim": 2260305,
        "Kasım": 2193234,
        "Aralık": 2202631,
    },
    2012: {
        "Ocak": 2165725,
        "Şubat": 2193560,
        "Mart": 2184742,
        "Nisan": 2225646,
        "Mayıs": 2313025,
        "Haziran": 2583560,
        "Temmuz": 2695931,
        "Ağustos": 2583806,
        "Eylül": 2546543,
        "Ekim": 2451321,
        "Kasım": 2348539,
        "Aralık": 2321405,
    },
    2013: {
        "Ocak": 2307690,
        "Şubat": 2299244,
        "Mart": 2311708,
        "Nisan": 2361974,
        "Mayıs": 2596652,
        "Haziran": 2647782,
        "Temmuz": 2716907,
        "Ağustos": 2676992,
        "Eylül": 2672105,
        "Ekim": 2454085,
        "Kasım": 2441531,
        "Aralık": 2399628,
    },
    2014: {
        "Ocak": 2404118,
        "Şubat": 2387281,
        "Mart": 2380197,
        "Nisan": 2430295,
        "Mayıs": 2567238,
        "Haziran": 2667333,
        "Temmuz": 2738872,
        "Ağustos": 2685855,
        "Eylül": 2607174,
        "Ekim": 2508619,
        "Kasım": 2511462,
        "Aralık": 2493061,
    },
    2015: {
        "Ocak": 2485572,
        "Şubat": 2425431,
        "Mart": 2434161,
        "Nisan": 2472557,
        "Mayıs": 2674352,
        "Haziran": 2749059,
        "Temmuz": 2853334,
        "Ağustos": 2898667,
        "Eylül": 2802772,
        "Ekim": 2653650,
        "Kasım": 2646228,
        "Aralık": 2617195,
    },
    2016: {
        "Ocak": 2587305,
        "Şubat": 2555664,
        "Mart": 2553124,
        "Nisan": 2678012,
        "Mayıs": 2740361,
        "Haziran": 2934439,
        "Temmuz": 2935875,
        "Ağustos": 2960434,
        "Eylül": 2825553,
        "Ekim": 2756585,
        "Kasım": 2693407,
        "Aralık": 2698591,
    },
    2017: {
        "Ocak": 2637306,
        "Şubat": 2625530,
        "Mart": 2619834,
        "Nisan": 2602678,
        "Mayıs": 2820665,
        "Haziran": 2900266,
        "Temmuz": 3005715,
        "Ağustos": 2961113,
        "Eylül": 2920997,
        "Ekim": 2810140,
        "Kasım": 2805426,
        "Aralık": 2747258,
    },
    2018: {
        "Ocak": 2716501,
        "Şubat": 2703272,
        "Mart": 2718509,
        "Nisan": 2807971,
        "Mayıs": 2915116,
        "Haziran": 3027697,
        "Temmuz": 3067409,
        "Ağustos": 3012977,
        "Eylül": 2935100,
        "Ekim": 2802055,
        "Kasım": 2773769,
        "Aralık": 2733162,
    },
    2019: {
        "Ocak": 2695987,
        "Şubat": 2671326,
        "Mart": 2731157,
        "Nisan": 2832268,
        "Mayıs": 3022314,
        "Haziran": 3104501,
        "Temmuz": 3097687,
        "Ağustos": 3037772,
        "Eylül": 3066233,
        "Ekim": 2917398,
        "Kasım": 2891581,
        "Aralık": 2824627,
    },
    2020: {
        "Ocak": 2741225,
        "Şubat": 2762787,
        "Mart": 2853472,
        "Nisan": 2811262,
        "Mayıs": 2904522,
        "Haziran": 3030796,
        "Temmuz": 3167260,
        "Ağustos": 3172405,
        "Eylül": 3189802,
        "Ekim": 2966960,
        "Kasım": 2848354,
        "Aralık": 2762116,
    },
    2021: {
        "Ocak": 2714857,
        "Şubat": 2752429,
        "Mart": 2755484,
        "Nisan": 2750022,
        "Mayıs": 2948254,
        "Haziran": 2983085,
        "Temmuz": 3109270,
        "Ağustos": 3336472,
        "Eylül": 3167974,
        "Ekim": 2966132,
        "Kasım": 2921908,
        "Aralık": 2886714,
    },
    2022: {
        "Ocak": 2864917,
        "Şubat": 2858684,
        "Mart": 2838319,
        "Nisan": 2905168,
        "Mayıs": 3074857,
        "Haziran": 3202007,
        "Temmuz": 3127789,
        "Ağustos": 3239696,
        "Eylül": 3172793,
        "Ekim": 3067241,
        "Kasım": 2988895,
        "Aralık": 2934376,
    },
    2023: {
        "Ocak": 2911036,
        "Şubat": 2873538,
        "Mart": 2892018,
        "Nisan": 2860134,
        "Mayıs": 2956110,
        "Haziran": 3145789,
        "Temmuz": 3345475,
        "Ağustos": 3337323,
        "Eylül": 3285013,
        "Ekim": 3086386,
        "Kasım": 2995016,
        "Aralık": 3020893,
    },
}


RECORDED_VS_SUPPLY_TOTALS = {
    2020: {
        "Ocak": {"recorded_m3": 66354248, "supplied_m3": 84977984},
        "Şubat": {"recorded_m3": 64183773, "supplied_m3": 80120814},
        "Mart": {"recorded_m3": 71668460, "supplied_m3": 88457622},
        "Nisan": {"recorded_m3": 55957867, "supplied_m3": 84337847},
        "Mayıs": {"recorded_m3": 62264986, "supplied_m3": 90040181},
        "Haziran": {"recorded_m3": 98524801, "supplied_m3": 90923882},
        "Temmuz": {"recorded_m3": 70573065, "supplied_m3": 98185057},
        "Ağustos": {"recorded_m3": 77095400, "supplied_m3": 98344547},
        "Eylül": {"recorded_m3": 74375679, "supplied_m3": 95694061},
        "Ekim": {"recorded_m3": 72273427, "supplied_m3": 91975770},
        "Kasım": {"recorded_m3": 69716659, "supplied_m3": 85450631},
        "Aralık": {"recorded_m3": 69044200, "supplied_m3": 85625581},
    },
    2021: {
        "Ocak": {"recorded_m3": 64109173, "supplied_m3": 84160554},
        "Şubat": {"recorded_m3": 61002480, "supplied_m3": 77068001},
        "Mart": {"recorded_m3": 74320264, "supplied_m3": 85420000},
        "Nisan": {"recorded_m3": 68491539, "supplied_m3": 82500656},
        "Mayıs": {"recorded_m3": 57267886, "supplied_m3": 91395884},
        "Haziran": {"recorded_m3": 84756553, "supplied_m3": 89492561},
        "Temmuz": {"recorded_m3": 61948614, "supplied_m3": 96490898},
        "Ağustos": {"recorded_m3": 83860638, "supplied_m3": 103397386},
        "Eylül": {"recorded_m3": 83747863, "supplied_m3": 94941769},
        "Ekim": {"recorded_m3": 69296226, "supplied_m3": 92084962},
        "Kasım": {"recorded_m3": 75600249, "supplied_m3": 87549545},
        "Aralık": {"recorded_m3": 69182560, "supplied_m3": 89488145},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official monthly supply context from ISKI activity reports.")
    parser.add_argument(
        "--model-monthly-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_quant_exog/tables/istanbul_dam_model_input_monthly.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store"),
    )
    return parser.parse_args()


def build_city_supply_monthly() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, months in CITY_SUPPLY_DAILY_AVG.items():
        for month_name, avg_val in months.items():
            month = MONTH_MAP[month_name]
            days = calendar.monthrange(year, month)[1]
            rows.append(
                {
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "year": year,
                    "month": month,
                    "month_name_tr": month_name,
                    "city_supply_m3_day_avg_official": float(avg_val),
                    "city_supply_m3_month_official": float(avg_val * days),
                    "days_in_month": days,
                    "source_type": "official_activity_report_daily_avg",
                }
            )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_recorded_monthly() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, months in RECORDED_VS_SUPPLY_TOTALS.items():
        for month_name, vals in months.items():
            month = MONTH_MAP[month_name]
            rows.append(
                {
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "year": year,
                    "month": month,
                    "month_name_tr": month_name,
                    "recorded_water_m3_official": float(vals["recorded_m3"]),
                    "city_supply_m3_month_official_from_recorded_table": float(vals["supplied_m3"]),
                }
            )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["recorded_share_pct"] = 100.0 * df["recorded_water_m3_official"] / df["city_supply_m3_month_official_from_recorded_table"]
    return df


def build_comparison(city_supply: pd.DataFrame, recorded: pd.DataFrame, model_monthly_csv: Path) -> pd.DataFrame:
    model = pd.read_csv(model_monthly_csv, parse_dates=["ds"]).rename(columns={"ds": "date"})
    model["days_in_month"] = pd.to_datetime(model["date"]).dt.days_in_month
    model["model_consumption_m3_month"] = model["consumption_mean_monthly"] * model["days_in_month"]
    model = model[["date", "model_consumption_m3_month", "consumption_mean_monthly"]]

    out = city_supply.merge(model, on="date", how="left")
    out = out.merge(recorded[["date", "recorded_water_m3_official", "recorded_share_pct"]], on="date", how="left")
    out["model_vs_supply_ratio_pct"] = 100.0 * out["model_consumption_m3_month"] / out["city_supply_m3_month_official"]
    out["model_vs_recorded_ratio_pct"] = 100.0 * out["model_consumption_m3_month"] / out["recorded_water_m3_official"]
    out["supply_minus_model_m3"] = out["city_supply_m3_month_official"] - out["model_consumption_m3_month"]
    out["recorded_minus_model_m3"] = out["recorded_water_m3_official"] - out["model_consumption_m3_month"]
    return out.sort_values("date").reset_index(drop=True)


def plot_supply_vs_model(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2023-12-01")].copy()
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.plot(sub["date"], sub["city_supply_m3_month_official"] / 1e6, label="Resmi sehre verilen su", color="#1d4ed8", linewidth=2.0)
    ax.plot(sub["date"], sub["model_consumption_m3_month"] / 1e6, label="Model aylik tuketim", color="#0f766e", linewidth=1.8)
    ax.plot(sub["date"], sub["recorded_water_m3_official"] / 1e6, label="Resmi kayit altina alinan su", color="#b45309", linewidth=1.6)
    ax.set_ylabel("Milyon m3 / ay")
    ax.set_title("Resmi aylik su hacmi ile model tuketim karsilastirmasi")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_recorded_share(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["recorded_share_pct"].notna()].copy()
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.bar(sub["date"].dt.strftime("%Y-%m"), sub["recorded_share_pct"], color="#b91c1c")
    ax.set_ylabel("Yuzde")
    ax.set_title("Resmi kayit altina alinan su payi (2020-2021)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(70)
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(comparison: pd.DataFrame) -> dict[str, object]:
    overlap_recorded = comparison[comparison["recorded_water_m3_official"].notna() & comparison["model_consumption_m3_month"].notna()].copy()
    overlap_supply = comparison[comparison["model_consumption_m3_month"].notna()].copy()

    def corr(a: pd.Series, b: pd.Series) -> float | None:
        if len(a.dropna()) < 2 or len(b.dropna()) < 2:
            return None
        return float(a.corr(b))

    def mape(actual: pd.Series, pred: pd.Series) -> float | None:
        actual = actual.astype(float)
        pred = pred.astype(float)
        mask = actual.ne(0) & actual.notna() & pred.notna()
        if int(mask.sum()) == 0:
            return None
        return float((np.abs((pred[mask] - actual[mask]) / actual[mask])).mean() * 100.0)

    return {
        "official_city_supply_window": {
            "start": str(comparison["date"].min().date()),
            "end": str(comparison["date"].max().date()),
            "rows": int(len(comparison)),
        },
        "official_recorded_window": {
            "start": str(overlap_recorded["date"].min().date()) if not overlap_recorded.empty else None,
            "end": str(overlap_recorded["date"].max().date()) if not overlap_recorded.empty else None,
            "rows": int(len(overlap_recorded)),
        },
        "model_vs_supply": {
            "corr": corr(overlap_supply["city_supply_m3_month_official"], overlap_supply["model_consumption_m3_month"]),
            "mean_ratio_pct": float(overlap_supply["model_vs_supply_ratio_pct"].mean()),
        },
        "model_vs_recorded": {
            "corr": corr(overlap_recorded["recorded_water_m3_official"], overlap_recorded["model_consumption_m3_month"]),
            "mape_pct": mape(overlap_recorded["recorded_water_m3_official"], overlap_recorded["model_consumption_m3_month"]),
            "mean_ratio_pct": float(overlap_recorded["model_vs_recorded_ratio_pct"].mean()) if not overlap_recorded.empty else None,
        },
        "recorded_share_pct_2020_2021_mean": float(overlap_recorded["recorded_share_pct"].mean()) if not overlap_recorded.empty else None,
        "notes": [
            "Official city-supply series is reconstructed from ISKI activity-report monthly daily averages for 2010-2023.",
            "Official recorded-water monthly series is directly available for 2020-2021 in the activity reports used here.",
            "This package is useful for validating whether the project consumption proxy behaves more like supplied water or recorded water.",
        ],
    }


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_figures = args.out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    city_supply = build_city_supply_monthly()
    recorded = build_recorded_monthly()
    comparison = build_comparison(city_supply, recorded, args.model_monthly_csv)

    city_supply.to_csv(out_tables / "official_city_supply_monthly_2010_2023.csv", index=False)
    recorded.to_csv(out_tables / "official_recorded_water_monthly_2020_2021.csv", index=False)
    comparison.to_csv(out_tables / "official_supply_vs_model_consumption_monthly.csv", index=False)

    plot_supply_vs_model(comparison, out_figures / "official_supply_vs_model_consumption.png")
    plot_recorded_share(comparison, out_figures / "official_recorded_share_2020_2021.png")

    summary = build_summary(comparison)
    (args.out_dir / "official_monthly_supply_context_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(out_tables / "official_city_supply_monthly_2010_2023.csv")
    print(out_tables / "official_recorded_water_monthly_2020_2021.csv")
    print(out_tables / "official_supply_vs_model_consumption_monthly.csv")
    print(out_figures / "official_supply_vs_model_consumption.png")
    print(out_figures / "official_recorded_share_2020_2021.png")
    print(args.out_dir / "official_monthly_supply_context_summary.json")


if __name__ == "__main__":
    main()
