#!/usr/bin/env python3
"""Independent verification for vapor-pressure outputs built from new-data workbooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_VP_COL = "maksimum_buhar_basinci_kpa_(es_tmax)"
ACTUAL_VP_COL = "anlik_buhar_basinci_kpa_(ea)"
DIFF_VP_COL = "aradaki_fark_kpa_(es_tmax-ea)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independently verify vapor-pressure CSV against source workbooks.")
    p.add_argument(
        "--new-data-dir",
        type=Path,
        default=Path("new data/Veriler_H-3"),
        help="Folder containing the source temperature and humidity workbooks.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("output/new_data_vapor_pressure/buhar_basinci_new_data.csv"),
        help="Produced vapor-pressure CSV to verify.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/new_data_vapor_pressure"),
        help="Directory for verification artifacts.",
    )
    return p.parse_args()


def resolve_workbook(base_dir: Path, suffix: str) -> Path:
    candidates = [base_dir]
    if not base_dir.is_absolute():
        candidates.extend([Path.cwd() / base_dir, Path.cwd().parent / base_dir])
    for candidate in candidates:
        if not candidate.exists():
            continue
        matches = sorted(candidate.rglob(suffix))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Workbook not found for pattern {suffix} under {base_dir}")


def detect_year_columns(row: pd.Series) -> list[tuple[int, int]]:
    year_cols: list[tuple[int, int]] = []
    started = False
    for idx, value in row.items():
        try:
            year = int(round(float(value)))
        except Exception:
            if started:
                break
            continue
        if 1800 <= year <= 2100:
            year_cols.append((int(idx), year))
            started = True
            continue
        if started:
            break
    if not year_cols:
        raise ValueError("No year columns detected.")
    return year_cols


def safe_date(year: int, month: int, day: int) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def build_long_matrix(raw: pd.DataFrame, year_row_idx: int, data_start_row: int, value_name: str) -> pd.DataFrame:
    year_cols = detect_year_columns(raw.iloc[year_row_idx])
    year_col_ids = [col_idx for col_idx, _ in year_cols]
    year_map = {col_idx: year for col_idx, year in year_cols}

    base = raw.iloc[data_start_row:, [0, *year_col_ids]].copy()
    base = base.rename(columns={0: "template_date"})
    base["template_date"] = pd.to_datetime(base["template_date"], errors="coerce")
    base = base.dropna(subset=["template_date"])

    long = base.melt(id_vars=["template_date"], var_name="year_col", value_name=value_name)
    long["year"] = long["year_col"].map(year_map)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long = long.dropna(subset=["year", value_name]).copy()
    long["month"] = long["template_date"].dt.month.astype(int)
    long["day"] = long["template_date"].dt.day.astype(int)
    long["date"] = [safe_date(int(y), int(m), int(d)) for y, m, d in zip(long["year"], long["month"], long["day"], strict=False)]
    long = long.dropna(subset=["date"]).copy()
    return long[["date", value_name]].reset_index(drop=True)


def load_independent_temperature(temp_path: Path) -> pd.DataFrame:
    tmax_raw = pd.read_excel(temp_path, sheet_name="Max", header=None)
    tmin_raw = pd.read_excel(temp_path, sheet_name="Min", header=None)
    tmean_raw = pd.read_excel(temp_path, sheet_name="Ort", header=None)

    tmax = build_long_matrix(tmax_raw, 0, 2, "t_max_c").groupby("date", as_index=False)["t_max_c"].mean()
    tmin = build_long_matrix(tmin_raw, 0, 2, "t_min_c").groupby("date", as_index=False)["t_min_c"].mean()
    tmean = build_long_matrix(tmean_raw, 0, 2, "t_mean_c").groupby("date", as_index=False)["t_mean_c"].mean()

    out = tmax.merge(tmin, on="date", how="inner").merge(tmean, on="date", how="left")
    out["t_mean_c"] = out["t_mean_c"].combine_first(0.5 * (out["t_max_c"] + out["t_min_c"]))
    return out.sort_values("date").reset_index(drop=True)


def load_independent_humidity(humidity_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(humidity_path, sheet_name=0, header=None)
    long = build_long_matrix(raw, 1, 3, "raw_rh_mean_pct")

    quality = (
        long.groupby("date", as_index=False)
        .agg(
            raw_values_count=("raw_rh_mean_pct", "size"),
            raw_values=("raw_rh_mean_pct", lambda s: "|".join(str(v) for v in s)),
            raw_mean_before_clean=("raw_rh_mean_pct", "mean"),
        )
    )
    quality["clean_mean_used"] = quality["raw_mean_before_clean"].clip(lower=0.0, upper=100.0)

    def classify(row: pd.Series) -> str:
        issues: list[str] = []
        if int(row["raw_values_count"]) > 1:
            issues.append("duplicate_rows_averaged")
        if float(row["raw_mean_before_clean"]) != float(row["clean_mean_used"]):
            issues.append("clipped_to_0_100")
        return ",".join(issues)

    quality["issue_type"] = quality.apply(classify, axis=1)
    humidity = quality[["date", "clean_mean_used"]].rename(columns={"clean_mean_used": "rh_mean_pct"})
    quality = quality.loc[quality["issue_type"] != ""].copy()
    return humidity.sort_values("date").reset_index(drop=True), quality.sort_values("date").reset_index(drop=True)


def compute_vapor_pressure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["es_tmax_kpa"] = 0.6108 * np.exp((17.27 * out["t_max_c"]) / (out["t_max_c"] + 237.3))
    out["es_tmin_kpa"] = 0.6108 * np.exp((17.27 * out["t_min_c"]) / (out["t_min_c"] + 237.3))
    out["es_kpa"] = 0.5 * (out["es_tmax_kpa"] + out["es_tmin_kpa"])
    out["ea_kpa"] = (out["rh_mean_pct"] / 100.0) * out["es_kpa"]
    out["ea_hpa"] = out["ea_kpa"] * 10.0
    out[MAX_VP_COL] = out["es_tmax_kpa"]
    out[ACTUAL_VP_COL] = out["ea_kpa"]
    out[DIFF_VP_COL] = out["es_tmax_kpa"] - out["ea_kpa"]
    return out.sort_values("date").reset_index(drop=True)


def compare_series(produced: pd.DataFrame, independent: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged = produced.merge(independent, on="date", how="outer", suffixes=("_prod", "_ind"), indicator=True)
    compare_cols = [
        "t_max_c",
        "t_min_c",
        "t_mean_c",
        "rh_mean_pct",
        "es_tmax_kpa",
        "es_tmin_kpa",
        "es_kpa",
        "ea_kpa",
        "ea_hpa",
        MAX_VP_COL,
        ACTUAL_VP_COL,
        DIFF_VP_COL,
    ]
    for name in compare_cols:
        prod = f"{name}_prod"
        ind = f"{name}_ind"
        if prod in merged.columns and ind in merged.columns:
            merged[f"{name}_diff"] = merged[prod] - merged[ind]

    summary = {
        "produced_rows": int(len(produced)),
        "independent_rows": int(len(independent)),
        "left_only_dates": int((merged["_merge"] == "left_only").sum()),
        "right_only_dates": int((merged["_merge"] == "right_only").sum()),
    }
    for name in compare_cols:
        diff = f"{name}_diff"
        if diff in merged.columns:
            summary[f"max_abs_{name}_diff"] = float(merged[diff].abs().max())
    return merged, summary


def main() -> None:
    args = parse_args()
    temp_path = resolve_workbook(args.new_data_dir, "*Max-Min-Ort-orj.xlsx")
    humidity_path = resolve_workbook(args.new_data_dir, "*1911-2022-Nem.xlsx")

    produced = pd.read_csv(args.csv, parse_dates=["date"])
    indep_temp = load_independent_temperature(temp_path)
    indep_hum, indep_quality = load_independent_humidity(humidity_path)
    independent = compute_vapor_pressure(indep_temp.merge(indep_hum, on="date", how="inner"))

    compare_df, summary = compare_series(produced, independent)
    summary["temp_workbook"] = str(temp_path.resolve())
    summary["humidity_workbook"] = str(humidity_path.resolve())
    summary["independent_quality_rows"] = int(len(indep_quality))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compare_csv = args.out_dir / "buhar_basinci_new_data_bagimsiz_karsilastirma.csv"
    compare_df.to_csv(compare_csv, index=False)
    indep_quality_csv = args.out_dir / "buhar_basinci_new_data_bagimsiz_veri_kalite.csv"
    indep_quality.to_csv(indep_quality_csv, index=False)
    summary_json = args.out_dir / "buhar_basinci_new_data_bagimsiz_dogrulama_ozet.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote comparison CSV: {compare_csv}")
    print(f"Wrote independent quality CSV: {indep_quality_csv}")
    print(f"Wrote summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
