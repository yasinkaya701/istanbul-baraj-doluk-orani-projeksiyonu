#!/usr/bin/env python3
"""Build integrated ET0/ETc irrigation, crop-shift comparison, and ML bundle.

Outputs (default: output/spreadsheet):
  - irrigation_daily_<year>.csv
  - irrigation_weekly_<year>.csv
  - crop_shift_comparison_<year>.csv
  - ml_et0_leaderboard_<year>.csv
  - ml_et0_predictions_<year>.csv
  - parameter_source_map_<year>.csv
  - et0_formula_source_map_<year>.md
  - et0_kaynak_haritasi_<year>.png
  - irrigation_crop_ml_bundle_<year>.xlsx
  - irrigation_crop_ml_validation_<year>.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from compute_et0_fao56 import calc_ra_mj_m2_day


SOURCE_CODE_MAP = {
    "local": 0.0,
    "nasa": 1.0,
    "fallback": 2.0,
    "estimated": 3.0,
}


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    title: str
    feature_builder: Callable[[pd.DataFrame, float], pd.DataFrame]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build integrated ET0/ETc irrigation + crop + ML outputs.")
    p.add_argument("--year", type=int, default=1987, help="Target year.")
    p.add_argument(
        "--et0-csv",
        type=Path,
        default=Path("output/spreadsheet/et0_inputs_completed_1987.csv"),
        help="Completed ET0 input CSV.",
    )
    p.add_argument(
        "--water-balance-csv",
        type=Path,
        default=Path("output/spreadsheet/water_balance_partial_1987.csv"),
        help="Monthly partial water-balance CSV for effective rainfall proxy.",
    )
    p.add_argument("--area-ha", type=float, default=1.0, help="Irrigated area (hectare).")
    p.add_argument("--efficiency", type=float, default=0.75, help="Irrigation efficiency (0-1].")
    p.add_argument(
        "--crop-calendar-csv",
        type=Path,
        default=None,
        help="Optional crop calendar override CSV with schema: "
        "crop,planting_date,season_length_days,l_ini,l_dev,l_mid,l_late,kc_ini,kc_mid,kc_end",
    )
    p.add_argument(
        "--sowing-shifts",
        type=str,
        default="-15,0,15",
        help="Comma-separated sowing day shifts (e.g. -15,0,15).",
    )
    p.add_argument(
        "--models",
        type=str,
        default="linear,rf,lgbm",
        help="Comma-separated model set for ML benchmark. Options: linear,rf,lgbm",
    )
    p.add_argument("--latitude", type=float, default=41.01, help="Latitude for fallback Rs estimate.")
    p.add_argument("--elevation-m", type=float, default=39.0, help="Elevation for fallback pressure.")
    p.add_argument("--krs", type=float, default=0.19, help="Hargreaves kRs for fallback Rs.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/spreadsheet"),
        help="Output directory.",
    )
    return p.parse_args()


def parse_int_list(text: str) -> list[int]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    out = [int(v) for v in vals]
    if not out:
        raise ValueError("At least one sowing shift is required.")
    return out


def load_et0(et0_csv: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(et0_csv)
    if "date" not in df.columns:
        raise ValueError(f"Missing date column in {et0_csv}")
    if "et0_completed_mm_day" not in df.columns:
        raise ValueError(f"Missing et0_completed_mm_day in {et0_csv}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df[df["date"].dt.year == year].sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No ET0 rows for year={year} in {et0_csv}")
    if len(df) < 120:
        raise ValueError("ET0 dataset is too short for train/val/test + rolling validation.")
    return df


def load_precip_proxy(water_balance_csv: Path, dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(dates)})
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["days_in_month"] = out["date"].dt.days_in_month.astype(int)

    if not water_balance_csv.exists():
        out["effective_precip_mm"] = 0.0
        out["coverage_flag"] = "missing"
    else:
        wb = pd.read_csv(water_balance_csv)
        req = {"month", "precip_obs_mm"}
        miss = req - set(wb.columns)
        if miss:
            raise ValueError(f"{water_balance_csv} missing columns: {sorted(miss)}")
        wb = wb.copy()
        wb["effective_precip_mm"] = 0.8 * pd.to_numeric(wb["precip_obs_mm"], errors="coerce").fillna(0.0)
        if "coverage_flag" not in wb.columns:
            wb["coverage_flag"] = "unknown"
        wb = wb[["month", "effective_precip_mm", "coverage_flag"]]
        out = out.merge(wb, on="month", how="left")
        out["effective_precip_mm"] = out["effective_precip_mm"].fillna(0.0)
        out["coverage_flag"] = out["coverage_flag"].fillna("missing")

    out["peff_proxy_mm_day"] = np.where(out["days_in_month"] > 0, out["effective_precip_mm"] / out["days_in_month"], 0.0)
    return out[["date", "month", "peff_proxy_mm_day", "coverage_flag"]]


def default_crop_calendar(year: int) -> pd.DataFrame:
    rows = [
        {
            "crop": "bugday_kislik",
            "planting_date": f"{year-1}-11-01",
            "season_length_days": 240,
            "l_ini": 30,
            "l_dev": 60,
            "l_mid": 90,
            "l_late": 60,
            "kc_ini": 0.40,
            "kc_mid": 1.15,
            "kc_end": 0.25,
        },
        {
            "crop": "misir",
            "planting_date": f"{year}-05-01",
            "season_length_days": 150,
            "l_ini": 25,
            "l_dev": 30,
            "l_mid": 55,
            "l_late": 40,
            "kc_ini": 0.35,
            "kc_mid": 1.20,
            "kc_end": 0.60,
        },
        {
            "crop": "domates",
            "planting_date": f"{year}-04-15",
            "season_length_days": 170,
            "l_ini": 30,
            "l_dev": 40,
            "l_mid": 60,
            "l_late": 40,
            "kc_ini": 0.60,
            "kc_mid": 1.15,
            "kc_end": 0.80,
        },
        {
            "crop": "aycicegi",
            "planting_date": f"{year}-05-01",
            "season_length_days": 140,
            "l_ini": 20,
            "l_dev": 30,
            "l_mid": 50,
            "l_late": 40,
            "kc_ini": 0.40,
            "kc_mid": 1.15,
            "kc_end": 0.35,
        },
        {
            "crop": "bag_uzum",
            "planting_date": f"{year}-04-01",
            "season_length_days": 210,
            "l_ini": 30,
            "l_dev": 50,
            "l_mid": 90,
            "l_late": 40,
            "kc_ini": 0.30,
            "kc_mid": 0.85,
            "kc_end": 0.45,
        },
    ]
    return pd.DataFrame(rows)


def load_crop_calendar(path: Path | None, year: int, out_dir: Path) -> pd.DataFrame:
    cols = [
        "crop",
        "planting_date",
        "season_length_days",
        "l_ini",
        "l_dev",
        "l_mid",
        "l_late",
        "kc_ini",
        "kc_mid",
        "kc_end",
    ]
    if path is None:
        df = default_crop_calendar(year)
        df.to_csv(out_dir / f"crop_calendar_default_{year}.csv", index=False)
    else:
        df = pd.read_csv(path)

    miss = set(cols) - set(df.columns)
    if miss:
        raise ValueError(f"Crop calendar missing columns: {sorted(miss)}")

    out = df[cols].copy()
    out["crop"] = out["crop"].astype(str).str.strip()
    out["planting_date"] = pd.to_datetime(out["planting_date"], errors="coerce")
    if out["planting_date"].isna().any():
        raise ValueError("Invalid planting_date detected in crop calendar.")

    int_cols = ["season_length_days", "l_ini", "l_dev", "l_mid", "l_late"]
    for c in int_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
        if (out[c] < 0).any():
            raise ValueError(f"Negative value in crop calendar column: {c}")
    for c in ["kc_ini", "kc_mid", "kc_end"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].isna().any():
            raise ValueError(f"Invalid Kc value in column: {c}")

    # Harmonize season length if phase totals differ.
    phase_total = out["l_ini"] + out["l_dev"] + out["l_mid"] + out["l_late"]
    mismatch = phase_total != out["season_length_days"]
    out.loc[mismatch, "season_length_days"] = phase_total[mismatch]
    return out.reset_index(drop=True)


def kc_stage_and_value(day_after_plant: int, row: pd.Series) -> tuple[str, float]:
    if day_after_plant < 1 or day_after_plant > int(row["season_length_days"]):
        return "off_season", 0.0

    l_ini = int(row["l_ini"])
    l_dev = int(row["l_dev"])
    l_mid = int(row["l_mid"])
    l_late = int(row["l_late"])
    kc_ini = float(row["kc_ini"])
    kc_mid = float(row["kc_mid"])
    kc_end = float(row["kc_end"])

    b1 = l_ini
    b2 = l_ini + l_dev
    b3 = l_ini + l_dev + l_mid
    b4 = l_ini + l_dev + l_mid + l_late

    if day_after_plant <= b1:
        return "initial", kc_ini
    if day_after_plant <= b2:
        frac = (day_after_plant - b1) / max(l_dev, 1)
        return "development", kc_ini + frac * (kc_mid - kc_ini)
    if day_after_plant <= b3:
        return "mid", kc_mid
    if day_after_plant <= b4:
        frac = (day_after_plant - b3) / max(l_late, 1)
        return "late", kc_mid + frac * (kc_end - kc_mid)
    return "off_season", 0.0


def build_irrigation_outputs(
    et0: pd.DataFrame,
    precip: pd.DataFrame,
    crop_cal: pd.DataFrame,
    shifts: list[int],
    area_ha: float,
    efficiency: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = et0.copy()
    base = base.merge(precip, on="date", how="left")
    base["peff_proxy_mm_day"] = base["peff_proxy_mm_day"].fillna(0.0)
    base["coverage_flag"] = base["coverage_flag"].fillna("missing")
    base["et0_mm_day"] = pd.to_numeric(base["et0_completed_mm_day"], errors="coerce").fillna(0.0)
    base["iso_year"] = base["date"].dt.isocalendar().year.astype(int)
    base["iso_week"] = base["date"].dt.isocalendar().week.astype(int)

    daily_rows: list[dict] = []
    summary_rows: list[dict] = []

    for _, crop in crop_cal.iterrows():
        for shift in shifts:
            planting = pd.Timestamp(crop["planting_date"]) + pd.Timedelta(days=int(shift))
            sub = base.copy()
            sub["crop"] = crop["crop"]
            sub["sowing_shift_days"] = int(shift)
            sub["planting_date_shifted"] = planting
            sub["area_ha"] = float(area_ha)
            sub["efficiency"] = float(efficiency)
            sub["day_after_plant"] = (sub["date"] - planting).dt.days + 1

            stage_vals = [kc_stage_and_value(int(d), crop) for d in sub["day_after_plant"].to_numpy()]
            sub["kc_stage"] = [sv[0] for sv in stage_vals]
            sub["kc_day"] = np.array([sv[1] for sv in stage_vals], dtype=float)
            sub["in_season"] = sub["kc_stage"] != "off_season"

            sub["etc_mm_day"] = sub["kc_day"] * sub["et0_mm_day"]
            sub["net_mm_day"] = np.maximum(sub["etc_mm_day"] - sub["peff_proxy_mm_day"], 0.0)
            sub["gross_mm_day"] = np.where(efficiency > 0, sub["net_mm_day"] / efficiency, np.nan)
            sub["dose_m3_day"] = sub["gross_mm_day"] * area_ha * 10.0

            sub["net_mm_day_no_rain"] = sub["etc_mm_day"]
            sub["gross_mm_day_no_rain"] = np.where(efficiency > 0, sub["etc_mm_day"] / efficiency, np.nan)
            sub["dose_m3_day_no_rain"] = sub["gross_mm_day_no_rain"] * area_ha * 10.0

            # Save all days for full calendar trace.
            daily_rows.extend(sub.to_dict(orient="records"))

            s = sub[sub["in_season"]].copy()
            if s.empty:
                continue
            weeks = (
                s.groupby(["iso_year", "iso_week"], as_index=False)
                .agg(weekly_gross_m3=("dose_m3_day", "sum"))
                .sort_values(["iso_year", "iso_week"])
            )
            peak_weekly = float(weeks["weekly_gross_m3"].max()) if not weeks.empty else 0.0
            peak_week = (
                f"{int(weeks.loc[weeks['weekly_gross_m3'].idxmax(), 'iso_year'])}-W"
                f"{int(weeks.loc[weeks['weekly_gross_m3'].idxmax(), 'iso_week']):02d}"
                if not weeks.empty
                else ""
            )

            precip_good = bool((s["coverage_flag"] == "good").all())
            wind_fallback = bool((s["source_wind"] == "fallback").any()) if "source_wind" in s.columns else False
            rad_est = bool((s["source_radiation"] == "estimated").any()) if "source_radiation" in s.columns else False
            data_quality = "high" if precip_good and not wind_fallback and not rad_est else "partial"

            summary_rows.append(
                {
                    "crop": crop["crop"],
                    "sowing_shift_days": int(shift),
                    "planting_date_shifted": planting.date().isoformat(),
                    "season_days_in_year": int(s["in_season"].sum()),
                    "season_etc_mm": float(s["etc_mm_day"].sum()),
                    "season_net_irrigation_mm": float(s["net_mm_day"].sum()),
                    "season_gross_irrigation_mm": float(s["gross_mm_day"].sum()),
                    "season_gross_irrigation_m3": float(s["dose_m3_day"].sum()),
                    "season_gross_no_rain_m3": float(s["dose_m3_day_no_rain"].sum()),
                    "peak_weekly_gross_m3": peak_weekly,
                    "peak_week_id": peak_week,
                    "precip_coverage_all_good": precip_good,
                    "data_quality_flag": data_quality,
                }
            )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise ValueError("Daily irrigation output is empty. Check crop calendar configuration.")

    weekly = (
        daily[daily["in_season"]]
        .groupby(["crop", "sowing_shift_days", "iso_year", "iso_week"], as_index=False)
        .agg(
            week_start=("date", "min"),
            week_end=("date", "max"),
            et0_mm_week=("et0_mm_day", "sum"),
            etc_mm_week=("etc_mm_day", "sum"),
            net_mm_week=("net_mm_day", "sum"),
            gross_mm_week=("gross_mm_day", "sum"),
            dose_m3_week=("dose_m3_day", "sum"),
            gross_mm_week_no_rain=("gross_mm_day_no_rain", "sum"),
            dose_m3_week_no_rain=("dose_m3_day_no_rain", "sum"),
            mean_kc_week=("kc_day", "mean"),
            mean_et0_mm_day=("et0_mm_day", "mean"),
        )
        .sort_values(["crop", "sowing_shift_days", "iso_year", "iso_week"])
        .reset_index(drop=True)
    )
    weekly["week_start"] = pd.to_datetime(weekly["week_start"]).dt.date
    weekly["week_end"] = pd.to_datetime(weekly["week_end"]).dt.date

    comparison = pd.DataFrame(summary_rows).sort_values(
        ["season_gross_irrigation_m3", "peak_weekly_gross_m3"],
        ascending=False,
    )
    return daily, weekly, comparison


def build_source_map_table() -> pd.DataFrame:
    rows = [
        {
            "symbol": "Tmean, Tmin, Tmax",
            "dataset_column_or_term": "t_mean_c, t_min_c, t_max_c",
            "source_primary": "Local hourly temperature workbook",
            "source_fallback_or_alt": "NASA POWER T2M/T2M_MAX/T2M_MIN",
            "used_in": "ET0 FAO-56",
        },
        {
            "symbol": "RHmean",
            "dataset_column_or_term": "rh_mean_pct",
            "source_primary": "Local humidity workbook",
            "source_fallback_or_alt": "NASA POWER RH2M",
            "used_in": "ET0 FAO-56",
        },
        {
            "symbol": "u2",
            "dataset_column_or_term": "u2_m_s",
            "source_primary": "NASA POWER WS2M",
            "source_fallback_or_alt": "u2=2.0 m/s constant fallback",
            "used_in": "ET0 FAO-56",
        },
        {
            "symbol": "Rs",
            "dataset_column_or_term": "rs_mj_m2_day",
            "source_primary": "NASA POWER ALLSKY_SFC_SW_DWN",
            "source_fallback_or_alt": "Hargreaves estimate (kRs*sqrt(Tmax-Tmin)*Ra)",
            "used_in": "ET0 FAO-56",
        },
        {
            "symbol": "p",
            "dataset_column_or_term": "p_kpa",
            "source_primary": "NASA POWER PS",
            "source_fallback_or_alt": "Elevation-based pressure formula",
            "used_in": "ET0 FAO-56",
        },
        {
            "symbol": "Ra,Rso,es,ea,Delta,gamma",
            "dataset_column_or_term": "derived internal terms",
            "source_primary": "FAO-56 equations",
            "source_fallback_or_alt": "Same equations in fallback scenario",
            "used_in": "ET0 FAO-56",
        },
        {
            "symbol": "Kc(d)",
            "dataset_column_or_term": "kc_day",
            "source_primary": "Crop calendar config + FAO stage lengths",
            "source_fallback_or_alt": "CSV override for crop params",
            "used_in": "ETc=Kc*ET0",
        },
        {
            "symbol": "Peff(d)",
            "dataset_column_or_term": "peff_proxy_mm_day",
            "source_primary": "0.8*monthly precip / days_in_month",
            "source_fallback_or_alt": "0 if monthly data missing",
            "used_in": "Net irrigation",
        },
    ]
    return pd.DataFrame(rows)


def build_source_refs_table() -> pd.DataFrame:
    rows = [
        {"topic": "FAO-56 Manual", "url": "https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/fao56.pdf"},
        {"topic": "FAO Single Kc", "url": "https://www.fao.org/3/X0490E/x0490e0b.htm"},
        {"topic": "FAO Dual Kc", "url": "https://www.fao.org/3/x0490E/x0490e0c.htm"},
        {"topic": "Water 2023 ET0 ML", "url": "https://doi.org/10.3390/w15223954"},
        {"topic": "PubMed 2025 ET0 limited inputs", "url": "https://pubmed.ncbi.nlm.nih.gov/41026596/"},
        {"topic": "AI Technology 2023 ET0", "url": "https://doi.org/10.1016/j.atech.2022.100115"},
    ]
    return pd.DataFrame(rows)


def source_map_markdown(map_df: pd.DataFrame, year: int) -> str:
    lines = [
        f"# ET0/ETc Formul ve Kaynak Haritasi ({year})",
        "",
        "## Formul",
        "",
        r"\[ ET_c = K_c \times ET_0 \]",
        "",
        r"\[ ET_0 = \frac{0.408\Delta(R_n-G)+\gamma\frac{900}{T_{mean}+273}u_2(e_s-e_a)}{\Delta+\gamma(1+0.34u_2)} \]",
        "",
        "## Akis",
        "",
        "```mermaid",
        "flowchart TD",
        '    A["T, RH, u2, Rs, p"] --> B["FAO-56 ET0"]',
        '    C["Kc stage motoru"] --> D["ETc = Kc * ET0"]',
        '    E["Aylik efektif yagis -> gunluk proxy"] --> F["Net/Gross sulama"]',
        '    B --> D',
        '    D --> F',
        "```",
        "",
        "## Parametre Kaynak Tablosu",
        "",
        "| symbol | dataset_column_or_term | source_primary | source_fallback_or_alt | used_in |",
        "|---|---|---|---|---|",
    ]
    for _, r in map_df.iterrows():
        lines.append(
            f"| {r['symbol']} | {r['dataset_column_or_term']} | {r['source_primary']} | "
            f"{r['source_fallback_or_alt']} | {r['used_in']} |"
        )
    lines.extend(
        [
            "",
            "## Sources",
            "- https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/fao56.pdf",
            "- https://www.fao.org/3/X0490E/x0490e0b.htm",
            "- https://www.fao.org/3/x0490E/x0490e0c.htm",
            "- https://doi.org/10.3390/w15223954",
            "- https://pubmed.ncbi.nlm.nih.gov/41026596/",
            "- https://doi.org/10.1016/j.atech.2022.100115",
        ]
    )
    return "\n".join(lines) + "\n"


def render_source_map_png(map_df: pd.DataFrame, out_png: Path, year: int) -> None:
    fig = plt.figure(figsize=(19, 10), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.9])

    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.axis("off")
    text = (
        f"ET0/ETc Kaynak Haritasi ({year})\n\n"
        "ETc = Kc * ET0\n\n"
        "ET0 (FAO-56 Penman-Monteith):\n"
        "0.408*Delta*(Rn-G) + gamma*(900/(Tmean+273))*u2*(es-ea)\n"
        "------------------------------------------------------\n"
        "          Delta + gamma*(1+0.34*u2)\n\n"
        "Akis:\n"
        "1) T/RH/u2/Rs/p -> ET0\n"
        "2) Kc stage motoru -> Kc(d)\n"
        "3) ETc=Kc*ET0\n"
        "4) Peff proxy -> Net/Gross sulama\n\n"
        "Not:\n"
        "- Gunluk yagis tam degilse aylik efektif yagis proxy dagitimi kullanilir.\n"
        "- ML benchmark ciktilari ET0 tahmini icin ayri tabloda verilir."
    )
    ax_left.text(0.01, 0.99, text, va="top", ha="left", fontsize=11, family="monospace")

    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.axis("off")
    col_labels = [
        "symbol",
        "dataset_column_or_term",
        "source_primary",
        "source_fallback_or_alt",
        "used_in",
    ]
    table_data = map_df[col_labels].to_numpy()
    table = ax_right.table(cellText=table_data, colLabels=col_labels, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.08, 1.65)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DCE6F2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FAFAFA" if row % 2 == 0 else "#FFFFFF")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_model(model_key: str) -> tuple[Pipeline, str]:
    model_key = model_key.strip().lower()
    if model_key == "linear":
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        return pipe, "LinearRegression"
    if model_key == "rf":
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
            ]
        )
        return pipe, "RandomForestRegressor"
    if model_key == "lgbm":
        try:
            from lightgbm import LGBMRegressor  # type: ignore

            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMRegressor(
                            random_state=42,
                            n_estimators=300,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                        ),
                    ),
                ]
            )
            return pipe, "LightGBMRegressor"
        except Exception:
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingRegressor(random_state=42)),
                ]
            )
            return pipe, "HistGradientBoostingRegressor_fallback_for_lgbm"
    raise ValueError(f"Unsupported model key: {model_key}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    denom_mask = np.abs(y_true) > 1e-8
    if denom_mask.any():
        mape = float(np.mean(np.abs((y_true[denom_mask] - y_pred[denom_mask]) / y_true[denom_mask])) * 100.0)
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape_pct": mape}


def add_common_features(df: pd.DataFrame, elevation_m: float, latitude: float, krs: float) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["doy"] = out["date"].dt.dayofyear.astype(int)
    out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"] / 365.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"] / 365.0)

    for c in ["source_temp", "source_humidity", "source_wind", "source_radiation"]:
        if c in out.columns:
            out[f"{c}_code"] = out[c].map(SOURCE_CODE_MAP).fillna(9.0)
        else:
            out[f"{c}_code"] = 9.0

    # Fallback fields for NASA-missing scenario.
    out["u2_fallback_m_s"] = 2.0
    out["p_fallback_kpa"] = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    ra = calc_ra_mj_m2_day(out["doy"], latitude)
    delta_t = np.maximum(pd.to_numeric(out["t_max_c"], errors="coerce") - pd.to_numeric(out["t_min_c"], errors="coerce"), 0.0)
    out["rs_fallback_mj_m2_day"] = krs * np.sqrt(delta_t) * ra
    return out


def scenario_full(df: pd.DataFrame, _elevation: float) -> pd.DataFrame:
    cols = [
        "date",
        "et0_completed_mm_day",
        "t_mean_c",
        "t_min_c",
        "t_max_c",
        "rh_mean_pct",
        "u2_m_s",
        "rs_mj_m2_day",
        "p_kpa",
        "doy_sin",
        "doy_cos",
        "source_temp_code",
        "source_humidity_code",
        "source_wind_code",
        "source_radiation_code",
    ]
    return df[cols].copy()


def scenario_missing_wind(df: pd.DataFrame, _elevation: float) -> pd.DataFrame:
    out = scenario_full(df, _elevation)
    return out.drop(columns=["u2_m_s"])


def scenario_missing_radiation(df: pd.DataFrame, _elevation: float) -> pd.DataFrame:
    out = scenario_full(df, _elevation)
    return out.drop(columns=["rs_mj_m2_day"])


def scenario_missing_humidity(df: pd.DataFrame, _elevation: float) -> pd.DataFrame:
    out = scenario_full(df, _elevation)
    return out.drop(columns=["rh_mean_pct"])


def scenario_missing_wind_radiation(df: pd.DataFrame, _elevation: float) -> pd.DataFrame:
    out = scenario_full(df, _elevation)
    return out.drop(columns=["u2_m_s", "rs_mj_m2_day"])


def scenario_local_missing_nasa(df: pd.DataFrame, _elevation: float) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": df["date"],
            "et0_completed_mm_day": df["et0_completed_mm_day"],
            "t_mean_nasa_c": pd.to_numeric(df["t_mean_nasa_c"], errors="coerce"),
            "rh_nasa_pct": pd.to_numeric(df["rh_nasa_pct"], errors="coerce"),
            "u2_m_s": pd.to_numeric(df["u2_m_s"], errors="coerce"),
            "rs_mj_m2_day": pd.to_numeric(df["rs_mj_m2_day"], errors="coerce"),
            "p_kpa": pd.to_numeric(df["p_kpa"], errors="coerce"),
            "doy_sin": df["doy_sin"],
            "doy_cos": df["doy_cos"],
            "source_temp_code": SOURCE_CODE_MAP["nasa"],
            "source_humidity_code": SOURCE_CODE_MAP["nasa"],
            "source_wind_code": SOURCE_CODE_MAP["nasa"],
            "source_radiation_code": SOURCE_CODE_MAP["nasa"],
        }
    )
    return out


def scenario_nasa_missing_local_fallback(df: pd.DataFrame, _elevation: float) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": df["date"],
            "et0_completed_mm_day": df["et0_completed_mm_day"],
            "t_mean_c": pd.to_numeric(df["t_mean_c"], errors="coerce"),
            "t_min_c": pd.to_numeric(df["t_min_c"], errors="coerce"),
            "t_max_c": pd.to_numeric(df["t_max_c"], errors="coerce"),
            "rh_mean_pct": pd.to_numeric(df["rh_mean_pct"], errors="coerce"),
            "u2_fallback_m_s": pd.to_numeric(df["u2_fallback_m_s"], errors="coerce"),
            "rs_fallback_mj_m2_day": pd.to_numeric(df["rs_fallback_mj_m2_day"], errors="coerce"),
            "p_fallback_kpa": pd.to_numeric(df["p_fallback_kpa"], errors="coerce"),
            "doy_sin": df["doy_sin"],
            "doy_cos": df["doy_cos"],
            "source_temp_code": SOURCE_CODE_MAP["local"],
            "source_humidity_code": SOURCE_CODE_MAP["local"],
            "source_wind_code": SOURCE_CODE_MAP["fallback"],
            "source_radiation_code": SOURCE_CODE_MAP["estimated"],
        }
    )
    return out


def build_ml_scenarios() -> list[ScenarioSpec]:
    return [
        ScenarioSpec("missing_wind", "Only wind sensor missing", scenario_missing_wind),
        ScenarioSpec("missing_radiation", "Only radiation sensor missing", scenario_missing_radiation),
        ScenarioSpec("missing_humidity", "Only humidity sensor missing", scenario_missing_humidity),
        ScenarioSpec("missing_wind_radiation", "Wind + radiation sensors missing", scenario_missing_wind_radiation),
        ScenarioSpec("local_missing_nasa_weighted", "Local sensors missing, NASA-weighted", scenario_local_missing_nasa),
        ScenarioSpec("nasa_missing_local_fallback", "NASA missing, local + fallback", scenario_nasa_missing_local_fallback),
    ]


def split_idx(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    if n_train < 60 or n_val < 20 or n_test < 20:
        raise ValueError("Insufficient rows for 70/15/15 time split.")
    idx = np.arange(n)
    tr = idx[:n_train]
    va = idx[n_train : n_train + n_val]
    te = idx[n_train + n_val :]
    return tr, va, te


def run_ml_benchmark(df: pd.DataFrame, model_keys: list[str], elevation_m: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.sort_values("date").reset_index(drop=True)
    scenarios = build_ml_scenarios()
    leaderboard_rows: list[dict] = []
    pred_rows: list[dict] = []

    for spec in scenarios:
        scen = spec.feature_builder(data, elevation_m).copy()
        scen["date"] = pd.to_datetime(scen["date"])
        scen = scen.dropna(subset=["date", "et0_completed_mm_day"]).reset_index(drop=True)
        feature_cols = [c for c in scen.columns if c not in {"date", "et0_completed_mm_day"}]
        if len(feature_cols) < 2:
            continue

        X = scen[feature_cols].copy()
        y = pd.to_numeric(scen["et0_completed_mm_day"], errors="coerce").to_numpy(dtype=float)
        dates = scen["date"].copy()
        tr, va, te = split_idx(len(scen))

        X_train, y_train = X.iloc[tr], y[tr]
        X_val, y_val = X.iloc[va], y[va]
        X_test, y_test = X.iloc[te], y[te]
        train_val_cut = np.concatenate([tr, va])
        X_trainval, y_trainval = X.iloc[train_val_cut], y[train_val_cut]

        # Rolling validation on pre-test period.
        tscv = TimeSeriesSplit(n_splits=4)

        for model_key in model_keys:
            model, backend = make_model(model_key)
            fitted = clone(model).fit(X_train, y_train)
            p_val = fitted.predict(X_val)
            p_test = fitted.predict(X_test)
            m_val = regression_metrics(y_val, p_val)
            m_test = regression_metrics(y_test, p_test)

            cv_metrics = []
            for fold_id, (cv_tr, cv_va) in enumerate(tscv.split(X_trainval), start=1):
                cv_model = clone(model).fit(X_trainval.iloc[cv_tr], y_trainval[cv_tr])
                cv_pred = cv_model.predict(X_trainval.iloc[cv_va])
                m = regression_metrics(y_trainval[cv_va], cv_pred)
                cv_metrics.append(
                    {
                        "fold": fold_id,
                        "mae": m["mae"],
                        "rmse": m["rmse"],
                        "r2": m["r2"],
                        "mape_pct": m["mape_pct"],
                    }
                )

            cv_df = pd.DataFrame(cv_metrics)
            leaderboard_rows.append(
                {
                    "scenario": spec.key,
                    "scenario_title": spec.title,
                    "model_key": model_key,
                    "model_backend": backend,
                    "feature_count": len(feature_cols),
                    "features": ",".join(feature_cols),
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    "val_mae": m_val["mae"],
                    "val_rmse": m_val["rmse"],
                    "val_r2": m_val["r2"],
                    "val_mape_pct": m_val["mape_pct"],
                    "test_mae": m_test["mae"],
                    "test_rmse": m_test["rmse"],
                    "test_r2": m_test["r2"],
                    "test_mape_pct": m_test["mape_pct"],
                    "cv_mae_mean": float(cv_df["mae"].mean()),
                    "cv_rmse_mean": float(cv_df["rmse"].mean()),
                    "cv_r2_mean": float(cv_df["r2"].mean()),
                    "cv_mape_pct_mean": float(cv_df["mape_pct"].mean()),
                }
            )

            val_rows = pd.DataFrame(
                {
                    "date": dates.iloc[va].to_numpy(),
                    "split": "val",
                    "scenario": spec.key,
                    "scenario_title": spec.title,
                    "model_key": model_key,
                    "model_backend": backend,
                    "y_true_et0_mm_day": y_val,
                    "y_pred_et0_mm_day": p_val,
                    "abs_error_mm_day": np.abs(y_val - p_val),
                }
            )
            test_rows = pd.DataFrame(
                {
                    "date": dates.iloc[te].to_numpy(),
                    "split": "test",
                    "scenario": spec.key,
                    "scenario_title": spec.title,
                    "model_key": model_key,
                    "model_backend": backend,
                    "y_true_et0_mm_day": y_test,
                    "y_pred_et0_mm_day": p_test,
                    "abs_error_mm_day": np.abs(y_test - p_test),
                }
            )
            pred_rows.extend(val_rows.to_dict(orient="records"))
            pred_rows.extend(test_rows.to_dict(orient="records"))

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(["scenario", "test_rmse"]).reset_index(drop=True)
    preds = pd.DataFrame(pred_rows).sort_values(["scenario", "model_key", "split", "date"]).reset_index(drop=True)
    return leaderboard, preds


def validation_report(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    comparison: pd.DataFrame,
    leaderboard: pd.DataFrame,
    preds: pd.DataFrame,
) -> dict:
    checks = {
        "daily_non_empty": bool(not daily.empty),
        "weekly_non_empty": bool(not weekly.empty),
        "comparison_non_empty": bool(not comparison.empty),
        "ml_leaderboard_non_empty": bool(not leaderboard.empty),
        "ml_predictions_non_empty": bool(not preds.empty),
        "formula_check_etc_equals_kc_times_et0": bool(
            np.allclose(daily["etc_mm_day"].to_numpy(), (daily["kc_day"] * daily["et0_mm_day"]).to_numpy(), atol=1e-9)
        ),
        "formula_check_gross_equals_net_over_efficiency": bool(
            (daily["gross_mm_day"] >= daily["net_mm_day"] - 1e-9).all()
        ),
        "formula_check_dose_m3_conversion": bool(
            np.allclose(daily["dose_m3_day"].to_numpy(), (daily["gross_mm_day"] * daily["area_ha"] * 10.0).to_numpy(), atol=1e-9)
        ),
    }
    return {"checks": checks, "all_passed": bool(all(checks.values()))}


def write_outputs(
    out_dir: Path,
    year: int,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    comparison: pd.DataFrame,
    leaderboard: pd.DataFrame,
    preds: pd.DataFrame,
    source_map: pd.DataFrame,
    source_refs: pd.DataFrame,
    source_md: str,
    validation: dict,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "daily_csv": out_dir / f"irrigation_daily_{year}.csv",
        "weekly_csv": out_dir / f"irrigation_weekly_{year}.csv",
        "comparison_csv": out_dir / f"crop_shift_comparison_{year}.csv",
        "ml_leaderboard_csv": out_dir / f"ml_et0_leaderboard_{year}.csv",
        "ml_predictions_csv": out_dir / f"ml_et0_predictions_{year}.csv",
        "source_map_csv": out_dir / f"parameter_source_map_{year}.csv",
        "source_refs_csv": out_dir / f"method_sources_{year}.csv",
        "source_map_md": out_dir / f"et0_formula_source_map_{year}.md",
        "source_map_png": out_dir / f"et0_kaynak_haritasi_{year}.png",
        "bundle_xlsx": out_dir / f"irrigation_crop_ml_bundle_{year}.xlsx",
        "validation_json": out_dir / f"irrigation_crop_ml_validation_{year}.json",
    }

    daily.to_csv(paths["daily_csv"], index=False)
    weekly.to_csv(paths["weekly_csv"], index=False)
    comparison.to_csv(paths["comparison_csv"], index=False)
    leaderboard.to_csv(paths["ml_leaderboard_csv"], index=False)
    preds.to_csv(paths["ml_predictions_csv"], index=False)
    source_map.to_csv(paths["source_map_csv"], index=False)
    source_refs.to_csv(paths["source_refs_csv"], index=False)
    paths["source_map_md"].write_text(source_md, encoding="utf-8")
    render_source_map_png(source_map, paths["source_map_png"], year)
    paths["validation_json"].write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(paths["bundle_xlsx"], engine="openpyxl") as writer:
        daily.to_excel(writer, sheet_name="daily_irrigation", index=False)
        weekly.to_excel(writer, sheet_name="weekly_irrigation", index=False)
        comparison.to_excel(writer, sheet_name="crop_shift_comparison", index=False)
        leaderboard.to_excel(writer, sheet_name="ml_leaderboard", index=False)
        preds.to_excel(writer, sheet_name="ml_predictions", index=False)
        source_map.to_excel(writer, sheet_name="parameter_source_map", index=False)
        source_refs.to_excel(writer, sheet_name="method_sources", index=False)

    return paths


def main() -> None:
    args = parse_args()
    if args.efficiency <= 0 or args.efficiency > 1:
        raise SystemExit("--efficiency must be in (0,1].")
    if args.area_ha <= 0:
        raise SystemExit("--area-ha must be positive.")

    shifts = parse_int_list(args.sowing_shifts)
    model_keys = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not model_keys:
        raise SystemExit("At least one model key is required.")

    et0 = load_et0(args.et0_csv, args.year)
    et0_common = add_common_features(et0, args.elevation_m, args.latitude, args.krs)
    precip = load_precip_proxy(args.water_balance_csv, et0_common["date"])
    crop_cal = load_crop_calendar(args.crop_calendar_csv, args.year, args.out_dir)

    daily, weekly, comparison = build_irrigation_outputs(
        et0=et0_common,
        precip=precip,
        crop_cal=crop_cal,
        shifts=shifts,
        area_ha=float(args.area_ha),
        efficiency=float(args.efficiency),
    )

    leaderboard, preds = run_ml_benchmark(et0_common, model_keys, args.elevation_m)
    source_map = build_source_map_table()
    source_refs = build_source_refs_table()
    source_md = source_map_markdown(source_map, args.year)
    validation = validation_report(daily, weekly, comparison, leaderboard, preds)
    paths = write_outputs(
        out_dir=args.out_dir,
        year=args.year,
        daily=daily,
        weekly=weekly,
        comparison=comparison,
        leaderboard=leaderboard,
        preds=preds,
        source_map=source_map,
        source_refs=source_refs,
        source_md=source_md,
        validation=validation,
    )

    print("Integrated ET0/ETc + ML bundle created:")
    for k, v in paths.items():
        print(f"- {k}: {v}")
    print(f"- validation_all_passed: {validation['all_passed']}")


if __name__ == "__main__":
    main()
