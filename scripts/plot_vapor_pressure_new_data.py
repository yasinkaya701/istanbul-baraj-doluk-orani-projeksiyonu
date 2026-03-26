#!/usr/bin/env python3
"""Build a vapor-pressure graph from the new temperature and humidity workbooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

MAX_VP_COL = "maksimum_buhar_basinci_kpa_(es_tmax)"
ACTUAL_VP_COL = "anlik_buhar_basinci_kpa_(ea)"
DIFF_VP_COL = "aradaki_fark_kpa_(es_tmax-ea)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute and plot vapor pressure from new data workbooks.")
    p.add_argument(
        "--new-data-dir",
        type=Path,
        default=Path("new data/Veriler_H-3"),
        help="Root folder that contains the new temperature and humidity workbooks.",
    )
    p.add_argument(
        "--temp-xlsx",
        type=Path,
        default=None,
        help="Temperature workbook path. If omitted, auto-detected under --new-data-dir.",
    )
    p.add_argument(
        "--humidity-xlsx",
        type=Path,
        default=None,
        help="Humidity workbook path. If omitted, auto-detected under --new-data-dir.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/new_data_vapor_pressure"),
        help="Output directory for CSV, JSON, and PNG files.",
    )
    return p.parse_args()


def to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_year(value: object) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    year = int(round(number))
    if 1800 <= year <= 2100:
        return year
    return None


def safe_date(year: int, month: int, day: int) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def detect_workbook(base_dir: Path, patterns: list[str], label: str) -> Path:
    candidates = [base_dir]
    if not base_dir.is_absolute():
        candidates.append(Path.cwd() / base_dir)
        candidates.append(Path.cwd().parent / base_dir)
    for candidate in candidates:
        if not candidate.exists():
            continue
        for pattern in patterns:
            matches = sorted(candidate.rglob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Could not find {label} workbook under {base_dir}")


def extract_year_columns(header_row: pd.Series) -> list[tuple[int, int]]:
    year_cols: list[tuple[int, int]] = []
    started = False
    for idx, value in header_row.items():
        year = to_year(value)
        if year is not None:
            year_cols.append((int(idx), year))
            started = True
            continue
        if started:
            break
    if not year_cols:
        raise ValueError("No year columns detected in workbook header.")
    return year_cols


def load_daily_matrix(
    workbook: Path,
    sheet_name: str,
    year_row_idx: int,
    data_start_row: int,
    value_name: str,
) -> pd.DataFrame:
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    year_cols = extract_year_columns(raw.iloc[year_row_idx])

    records: list[dict[str, object]] = []
    for row_idx in range(data_start_row, raw.shape[0]):
        template_date = pd.to_datetime(raw.iat[row_idx, 0], errors="coerce")
        if pd.isna(template_date):
            continue
        month = int(template_date.month)
        day = int(template_date.day)
        for col_idx, year in year_cols:
            value = to_float(raw.iat[row_idx, col_idx])
            if value is None:
                continue
            real_date = safe_date(year, month, day)
            if real_date is None:
                continue
            records.append({"date": real_date, value_name: float(value)})

    if not records:
        raise ValueError(f"No daily records parsed from {workbook} [{sheet_name}]")

    out = pd.DataFrame.from_records(records)
    out = out.groupby("date", as_index=False)[value_name].mean()
    return out.sort_values("date").reset_index(drop=True)


def load_temperature_daily(workbook: Path) -> pd.DataFrame:
    tmax = load_daily_matrix(workbook, sheet_name="Max", year_row_idx=0, data_start_row=2, value_name="t_max_c")
    tmin = load_daily_matrix(workbook, sheet_name="Min", year_row_idx=0, data_start_row=2, value_name="t_min_c")
    tmean = load_daily_matrix(workbook, sheet_name="Ort", year_row_idx=0, data_start_row=2, value_name="t_mean_c")
    merged = tmax.merge(tmin, on="date", how="inner").merge(tmean, on="date", how="outer")
    merged["t_mean_c"] = merged["t_mean_c"].combine_first(0.5 * (merged["t_max_c"] + merged["t_min_c"]))
    return merged.sort_values("date").reset_index(drop=True)


def load_humidity_daily(workbook: Path) -> pd.DataFrame:
    out = load_daily_matrix(
        workbook,
        sheet_name=0,
        year_row_idx=1,
        data_start_row=3,
        value_name="rh_mean_pct",
    )
    out["rh_mean_pct"] = out["rh_mean_pct"].clip(lower=0.0, upper=100.0)
    return out


def saturation_vapor_pressure_kpa(temp_c: pd.Series | np.ndarray) -> pd.Series:
    index = temp_c.index if isinstance(temp_c, pd.Series) else None
    temp = np.asarray(temp_c, dtype=float)
    values = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    return pd.Series(values, index=index)


def compute_vapor_pressure(temp_df: pd.DataFrame, humidity_df: pd.DataFrame) -> pd.DataFrame:
    df = temp_df.merge(humidity_df, on="date", how="inner")
    df = df.dropna(subset=["t_max_c", "t_min_c", "rh_mean_pct"]).copy()
    if df.empty:
        raise ValueError("Temperature and humidity datasets do not overlap after merge.")

    df["es_tmax_kpa"] = saturation_vapor_pressure_kpa(df["t_max_c"])
    df["es_tmin_kpa"] = saturation_vapor_pressure_kpa(df["t_min_c"])
    df["es_kpa"] = 0.5 * (df["es_tmax_kpa"] + df["es_tmin_kpa"])
    df["ea_kpa"] = (df["rh_mean_pct"] / 100.0) * df["es_kpa"]
    df["ea_hpa"] = df["ea_kpa"] * 10.0
    df["vpd_kpa"] = df["es_kpa"] - df["ea_kpa"]
    df[MAX_VP_COL] = df["es_tmax_kpa"]
    df[ACTUAL_VP_COL] = df["ea_kpa"]
    df[DIFF_VP_COL] = df["es_tmax_kpa"] - df["ea_kpa"]
    return df.sort_values("date").reset_index(drop=True)


def build_aggregates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_cols = [MAX_VP_COL, ACTUAL_VP_COL, DIFF_VP_COL]
    monthly = df.set_index("date")[focus_cols].resample("ME").mean().reset_index()
    annual = df.assign(year=df["date"].dt.year).groupby("year", as_index=False)[focus_cols].mean()

    for col in focus_cols:
        monthly[f"{col}_12ay_hareketli_ortalama"] = monthly[col].rolling(12, min_periods=6).mean()
        annual[f"{col}_10y_hareketli_ortalama"] = annual[col].rolling(10, min_periods=5).mean()

    return monthly, annual


def build_missing_days_report(temp_df: pd.DataFrame, humidity_df: pd.DataFrame) -> pd.DataFrame:
    temp_ready = temp_df.loc[temp_df["t_max_c"].notna() & temp_df["t_min_c"].notna(), "date"]
    humidity_ready = humidity_df.loc[humidity_df["rh_mean_pct"].notna(), "date"]
    start = max(temp_ready.min(), humidity_ready.min())
    end = min(temp_ready.max(), humidity_ready.max())
    full = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})

    temp_status = temp_df[["date", "t_max_c", "t_min_c"]].copy()
    temp_status["has_tmax"] = temp_status["t_max_c"].notna()
    temp_status["has_tmin"] = temp_status["t_min_c"].notna()
    temp_status = temp_status[["date", "has_tmax", "has_tmin"]]

    humidity_status = humidity_df[["date", "rh_mean_pct"]].copy()
    humidity_status["has_humidity"] = humidity_status["rh_mean_pct"].notna()
    humidity_status = humidity_status[["date", "has_humidity"]]

    merged = full.merge(temp_status, on="date", how="left").merge(humidity_status, on="date", how="left")
    merged["has_tmax"] = merged["has_tmax"].eq(True)
    merged["has_tmin"] = merged["has_tmin"].eq(True)
    merged["has_humidity"] = merged["has_humidity"].eq(True)
    merged["in_output"] = merged["has_tmax"] & merged["has_tmin"] & merged["has_humidity"]

    def classify(row: pd.Series) -> str:
        missing: list[str] = []
        if not bool(row["has_tmax"]):
            missing.append("tmax")
        if not bool(row["has_tmin"]):
            missing.append("tmin")
        if not bool(row["has_humidity"]):
            missing.append("humidity")
        return ",".join(missing)

    missing = merged.loc[~merged["in_output"], ["date", "has_tmax", "has_tmin", "has_humidity"]].copy()
    missing["missing_fields"] = missing.apply(classify, axis=1)
    return missing.reset_index(drop=True)


def load_raw_matrix(workbook: Path, sheet_name: str | int) -> pd.DataFrame:
    return pd.read_excel(workbook, sheet_name=sheet_name, header=None)


def build_year_col_map(raw: pd.DataFrame, year_row_idx: int) -> dict[int, int]:
    return {year: col_idx for col_idx, year in extract_year_columns(raw.iloc[year_row_idx])}


def build_monthday_row_map(raw: pd.DataFrame, data_start_row: int) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for row_idx in range(data_start_row, raw.shape[0]):
        template_date = pd.to_datetime(raw.iat[row_idx, 0], errors="coerce")
        if pd.isna(template_date):
            continue
        mapping[(int(template_date.month), int(template_date.day))] = int(row_idx)
    return mapping


def lookup_raw_value(
    raw: pd.DataFrame,
    year_col_map: dict[int, int],
    monthday_row_map: dict[tuple[int, int], int],
    date: pd.Timestamp,
) -> float | None:
    row_idx = monthday_row_map.get((int(date.month), int(date.day)))
    col_idx = year_col_map.get(int(date.year))
    if row_idx is None or col_idx is None:
        return None
    return to_float(raw.iat[row_idx, col_idx])


def lookup_raw_values(
    raw: pd.DataFrame,
    year_col_map: dict[int, int],
    monthday: tuple[int, int],
    year: int,
    data_start_row: int,
) -> list[float]:
    col_idx = year_col_map.get(int(year))
    if col_idx is None:
        return []
    values: list[float] = []
    for row_idx in range(data_start_row, raw.shape[0]):
        template_date = pd.to_datetime(raw.iat[row_idx, 0], errors="coerce")
        if pd.isna(template_date):
            continue
        if (int(template_date.month), int(template_date.day)) != monthday:
            continue
        value = to_float(raw.iat[row_idx, col_idx])
        if value is not None:
            values.append(float(value))
    return values


def compute_formula_verification(df: pd.DataFrame) -> dict[str, float]:
    es_tmax = 0.6108 * np.exp((17.27 * df["t_max_c"]) / (df["t_max_c"] + 237.3))
    es_tmin = 0.6108 * np.exp((17.27 * df["t_min_c"]) / (df["t_min_c"] + 237.3))
    es_check = 0.5 * (es_tmax + es_tmin)
    ea_check = (df["rh_mean_pct"] / 100.0) * es_check
    return {
        "formula_max_abs_es_diff": float((df["es_kpa"] - es_check).abs().max()),
        "formula_max_abs_ea_diff": float((df["ea_kpa"] - ea_check).abs().max()),
    }


def build_raw_validation_audit(vapor_df: pd.DataFrame, temp_path: Path, humidity_path: Path) -> pd.DataFrame:
    audit = vapor_df[["date", "t_max_c", "t_min_c", "rh_mean_pct", "es_kpa", "ea_kpa"]].copy()
    raw_tmax = load_daily_matrix(temp_path, "Max", 0, 2, "raw_t_max_c")
    raw_tmin = load_daily_matrix(temp_path, "Min", 0, 2, "raw_t_min_c")
    raw_hum = load_daily_matrix(humidity_path, 0, 1, 3, "raw_rh_mean_pct_before_clip")
    raw_hum["raw_rh_mean_pct"] = raw_hum["raw_rh_mean_pct_before_clip"].clip(lower=0.0, upper=100.0)

    audit = (
        audit.merge(raw_tmax, on="date", how="left")
        .merge(raw_tmin, on="date", how="left")
        .merge(raw_hum, on="date", how="left")
    )
    audit["t_max_diff"] = audit["t_max_c"] - audit["raw_t_max_c"]
    audit["t_min_diff"] = audit["t_min_c"] - audit["raw_t_min_c"]
    audit["rh_diff"] = audit["rh_mean_pct"] - audit["raw_rh_mean_pct"]

    es_tmax = 0.6108 * np.exp((17.27 * audit["raw_t_max_c"]) / (audit["raw_t_max_c"] + 237.3))
    es_tmin = 0.6108 * np.exp((17.27 * audit["raw_t_min_c"]) / (audit["raw_t_min_c"] + 237.3))
    audit["recomputed_es_kpa_from_raw"] = 0.5 * (es_tmax + es_tmin)
    audit["recomputed_ea_kpa_from_raw"] = (audit["raw_rh_mean_pct"] / 100.0) * audit["recomputed_es_kpa_from_raw"]
    audit["es_diff_vs_raw_recompute"] = audit["es_kpa"] - audit["recomputed_es_kpa_from_raw"]
    audit["ea_diff_vs_raw_recompute"] = audit["ea_kpa"] - audit["recomputed_ea_kpa_from_raw"]
    return audit


def build_humidity_quality_report(humidity_path: Path) -> pd.DataFrame:
    hum_raw = load_raw_matrix(humidity_path, 0)
    hum_cols = extract_year_columns(hum_raw.iloc[1])
    records: list[dict[str, object]] = []

    for row_idx in range(3, hum_raw.shape[0]):
        template_date = pd.to_datetime(hum_raw.iat[row_idx, 0], errors="coerce")
        if pd.isna(template_date):
            continue
        month = int(template_date.month)
        day = int(template_date.day)
        for col_idx, year in hum_cols:
            value = to_float(hum_raw.iat[row_idx, col_idx])
            if value is None:
                continue
            date = safe_date(year, month, day)
            if date is None:
                continue
            records.append(
                {
                    "date": date,
                    "raw_value": float(value),
                }
            )

    if not records:
        return pd.DataFrame(columns=["date", "raw_values_count", "raw_values", "raw_mean_before_clean", "clean_mean_used", "issue_type"])

    df = pd.DataFrame.from_records(records)
    grouped = (
        df.groupby("date", as_index=False)
        .agg(
            raw_values_count=("raw_value", "size"),
            raw_values=("raw_value", lambda s: "|".join(str(v) for v in s)),
            raw_mean_before_clean=("raw_value", "mean"),
        )
    )
    grouped["clean_mean_used"] = grouped["raw_mean_before_clean"].clip(lower=0.0, upper=100.0)

    def issue_type(row: pd.Series) -> str:
        issues: list[str] = []
        if int(row["raw_values_count"]) > 1:
            issues.append("duplicate_rows_averaged")
        if float(row["raw_mean_before_clean"]) != float(row["clean_mean_used"]):
            issues.append("clipped_to_0_100")
        return ",".join(issues)

    grouped["issue_type"] = grouped.apply(issue_type, axis=1)
    out = grouped.loc[
        grouped["issue_type"] != "",
        ["date", "raw_values_count", "raw_values", "raw_mean_before_clean", "clean_mean_used", "issue_type"],
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=["date", "raw_values_count", "raw_values", "raw_mean_before_clean", "clean_mean_used", "issue_type"])
    return out.sort_values("date").reset_index(drop=True)


def build_yearly_coverage_report(vapor_df: pd.DataFrame, missing_df: pd.DataFrame) -> pd.DataFrame:
    years = pd.DataFrame({"year": sorted(vapor_df["date"].dt.year.unique())})
    produced = vapor_df.assign(year=vapor_df["date"].dt.year).groupby("year", as_index=False).agg(
        output_days=("date", "count"),
        ea_mean_kpa=("ea_kpa", "mean"),
        ea_min_kpa=("ea_kpa", "min"),
        ea_max_kpa=("ea_kpa", "max"),
        maksimum_buhar_basinci_mean_kpa=(MAX_VP_COL, "mean"),
        anlik_buhar_basinci_mean_kpa=(ACTUAL_VP_COL, "mean"),
        fark_mean_kpa=(DIFF_VP_COL, "mean"),
    )
    missing = missing_df.assign(year=missing_df["date"].dt.year).groupby("year", as_index=False).agg(
        missing_days=("date", "count")
    )
    report = years.merge(produced, on="year", how="left").merge(missing, on="year", how="left")
    report["missing_days"] = report["missing_days"].fillna(0).astype(int)
    report["expected_days_if_complete"] = report["output_days"] + report["missing_days"]
    return report


def plot_vapor_pressure(daily: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    specs = [
        (MAX_VP_COL, "Maksimum buhar basıncı (es_tmax)", "#C62828"),
        (ACTUAL_VP_COL, "Anlık buhar basıncı (ea)", "#1565C0"),
        (DIFF_VP_COL, "Aradaki fark (es_tmax - ea)", "#F9A825"),
    ]
    daily_plot = daily[["date", *[col for col, _, _ in specs]]].copy()
    for col, _, _ in specs:
        daily_plot[f"{col}_30gun_hareketli_ortalama"] = daily_plot[col].rolling(30, min_periods=15).mean()

    ax = axes[0]
    for col, label, color in specs:
        ax.plot(daily_plot["date"], daily_plot[col], color=color, linewidth=0.7, alpha=0.35, label=label)
    ax.set_title("Yeni veriden maksimum, anlık ve fark buhar basıncı - günlük seri", fontsize=14)
    ax.set_ylabel("kPa")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    for col, label, color in specs:
        ax.plot(
            daily_plot["date"],
            daily_plot[f"{col}_30gun_hareketli_ortalama"],
            color=color,
            linewidth=2.2,
            label=f"{label} - 30 günlük ort.",
        )
    ax.set_title("Yeni veriden maksimum, anlık ve fark buhar basıncı - 30 günlük ortalama", fontsize=14)
    ax.set_xlabel("Tarih")
    ax.set_ylabel("kPa")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Buhar basıncı grafiği - new data (günlük)", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    missing_df: pd.DataFrame,
    verification: dict[str, float],
    audit_df: pd.DataFrame,
    humidity_quality_df: pd.DataFrame,
    temp_path: Path,
    humidity_path: Path,
) -> dict:
    return {
        "source_temp_xlsx": str(temp_path.resolve()),
        "source_humidity_xlsx": str(humidity_path.resolve()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_daily_rows": int(len(df)),
        "n_months": int(len(monthly)),
        "n_years": int(len(annual)),
        "n_missing_days_within_overlap": int(len(missing_df)),
        "n_data_quality_adjustments": int(len(humidity_quality_df)),
        "max_buhar_basinci_mean_kpa": float(df[MAX_VP_COL].mean()),
        "ea_kpa_mean": float(df["ea_kpa"].mean()),
        "fark_mean_kpa": float(df[DIFF_VP_COL].mean()),
        "ea_kpa_min": float(df["ea_kpa"].min()),
        "ea_kpa_max": float(df["ea_kpa"].max()),
        "audit_max_abs_t_max_diff": float(audit_df["t_max_diff"].abs().max()),
        "audit_max_abs_t_min_diff": float(audit_df["t_min_diff"].abs().max()),
        "audit_max_abs_rh_diff": float(audit_df["rh_diff"].abs().max()),
        "audit_max_abs_es_diff_from_raw": float(audit_df["es_diff_vs_raw_recompute"].abs().max()),
        "audit_max_abs_ea_diff_from_raw": float(audit_df["ea_diff_vs_raw_recompute"].abs().max()),
        **verification,
    }


def main() -> None:
    args = parse_args()

    temp_path = args.temp_xlsx or detect_workbook(args.new_data_dir, ["*Max-Min-Ort-orj.xlsx"], "temperature")
    humidity_path = args.humidity_xlsx or detect_workbook(args.new_data_dir, ["*1911-2022-Nem.xlsx"], "humidity")

    temp_df = load_temperature_daily(temp_path)
    humidity_df = load_humidity_daily(humidity_path)
    vapor_df = compute_vapor_pressure(temp_df, humidity_df)
    monthly_df, annual_df = build_aggregates(vapor_df)
    missing_df = build_missing_days_report(temp_df, humidity_df)
    verification = compute_formula_verification(vapor_df)
    audit_df = build_raw_validation_audit(vapor_df, temp_path, humidity_path)
    coverage_df = build_yearly_coverage_report(vapor_df, missing_df)
    humidity_quality_df = build_humidity_quality_report(humidity_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily_csv = args.out_dir / "vapor_pressure_daily_new_data.csv"
    monthly_csv = args.out_dir / "vapor_pressure_monthly_new_data.csv"
    annual_csv = args.out_dir / "vapor_pressure_annual_new_data.csv"
    daily_csv_tr = args.out_dir / "buhar_basinci_new_data.csv"
    monthly_csv_tr = args.out_dir / "buhar_basinci_new_data_aylik.csv"
    annual_csv_tr = args.out_dir / "buhar_basinci_new_data_yillik.csv"
    missing_csv_tr = args.out_dir / "buhar_basinci_new_data_eksik_gunler.csv"
    audit_csv_tr = args.out_dir / "buhar_basinci_new_data_dogrulama.csv"
    coverage_csv_tr = args.out_dir / "buhar_basinci_new_data_yillik_kapsam.csv"
    humidity_quality_csv_tr = args.out_dir / "buhar_basinci_new_data_veri_kalite_kayitlari.csv"
    summary_json = args.out_dir / "vapor_pressure_summary_new_data.json"
    chart_png = args.out_dir / "buhar_basinci_new_data.png"

    preferred_cols = [
        "date",
        MAX_VP_COL,
        ACTUAL_VP_COL,
        DIFF_VP_COL,
        "t_max_c",
        "t_min_c",
        "t_mean_c",
        "rh_mean_pct",
        "es_tmax_kpa",
        "es_tmin_kpa",
        "es_kpa",
        "ea_kpa",
        "ea_hpa",
        "vpd_kpa",
    ]
    remaining_cols = [c for c in vapor_df.columns if c not in preferred_cols]
    vapor_df_export = vapor_df[preferred_cols + remaining_cols].copy()

    vapor_df_export.to_csv(daily_csv, index=False)
    monthly_df.to_csv(monthly_csv, index=False)
    annual_df.to_csv(annual_csv, index=False)
    vapor_df_export.to_csv(daily_csv_tr, index=False)
    monthly_df.to_csv(monthly_csv_tr, index=False)
    annual_df.to_csv(annual_csv_tr, index=False)
    missing_df.to_csv(missing_csv_tr, index=False)
    audit_df.to_csv(audit_csv_tr, index=False)
    coverage_df.to_csv(coverage_csv_tr, index=False)
    humidity_quality_df.to_csv(humidity_quality_csv_tr, index=False)
    summary = build_summary(
        vapor_df,
        monthly_df,
        annual_df,
        missing_df,
        verification,
        audit_df,
        humidity_quality_df,
        temp_path,
        humidity_path,
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_vapor_pressure(vapor_df, chart_png)

    print(f"Temperature workbook: {temp_path}")
    print(f"Humidity workbook: {humidity_path}")
    print(f"Daily rows: {len(vapor_df)}")
    print(f"Date range: {summary['date_min']} -> {summary['date_max']}")
    print(f"Mean maximum vapor pressure: {summary['max_buhar_basinci_mean_kpa']:.3f} kPa")
    print(f"Mean actual vapor pressure: {summary['ea_kpa_mean']:.3f} kPa")
    print(f"Mean difference: {summary['fark_mean_kpa']:.3f} kPa")
    print(f"Missing days within overlap: {summary['n_missing_days_within_overlap']}")
    print(f"Data-quality adjustments: {summary['n_data_quality_adjustments']}")
    print(f"Formula max abs diff (ea): {summary['formula_max_abs_ea_diff']:.3e}")
    print(f"Raw-cell audit max abs diff (ea): {summary['audit_max_abs_ea_diff_from_raw']:.3e}")
    print(f"Wrote chart: {chart_png}")
    print(f"Wrote daily CSV: {daily_csv}")
    print(f"Wrote Turkish daily CSV: {daily_csv_tr}")
    print(f"Wrote missing-days CSV: {missing_csv_tr}")
    print(f"Wrote validation CSV: {audit_csv_tr}")
    print(f"Wrote yearly coverage CSV: {coverage_csv_tr}")
    print(f"Wrote data-quality CSV: {humidity_quality_csv_tr}")
    print(f"Wrote summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
