#!/usr/bin/env python3
"""Build a daily es-ea (vapor pressure deficit) CSV from the new-data folder.

Method selection by data quality:
  - Historical wide workbooks: es from Tmax/Tmin, ea from RHmean (FAO-56 Eq.19 fallback)
  - Automatic station: es from Tmax/Tmin, ea from RHmax/RHmin when available

Outputs:
  - output/spreadsheet/es_ea_newdata_daily.csv
  - output/spreadsheet/es_ea_newdata_summary.json
"""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create daily es-ea CSV from new data files.")
    parser.add_argument(
        "--temp-xlsx",
        type=Path,
        default=Path("../Downloads/Veriler_H-3/Sçcaklçk/Uzunyçl-Max-Min-Ort-orj.xlsx"),
        help="Temperature workbook with Max/Min/Ort sheets.",
    )
    parser.add_argument(
        "--humidity-xlsx",
        type=Path,
        default=Path("../Downloads/Veriler_H-3/Nem/1911-2022-Nem.xlsx"),
        help="Humidity workbook with daily RH values.",
    )
    parser.add_argument(
        "--auto-table1",
        type=Path,
        default=Path("../Downloads/Veriler_H-3/Otomatik òstasyon/CR800Series_Table1.dat"),
        help="Automatic station 10-minute table (RH/T mean).",
    )
    parser.add_argument(
        "--auto-table2",
        type=Path,
        default=Path("../Downloads/Veriler_H-3/Otomatik òstasyon/CR800Series_Table2.dat"),
        help="Automatic station daily Tmax/Tmin table.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/spreadsheet/es_ea_newdata_daily.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path("output/spreadsheet/es_ea_newdata_summary.json"),
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--out-series-csv",
        type=Path,
        default=Path("output/spreadsheet/es_ea_newdata_series_only.csv"),
        help="Slim output CSV with only date and es-ea.",
    )
    return parser.parse_args()


def saturation_vapor_pressure_kpa(temp_c: pd.Series | np.ndarray) -> pd.Series:
    temp = np.asarray(temp_c, dtype=float)
    return pd.Series(0.6108 * np.exp((17.27 * temp) / (temp + 237.3)))


def _extract_year_cols(header_row: pd.Series) -> dict[int, int]:
    year_cols: dict[int, int] = {}
    for idx, value in header_row.items():
        year = pd.to_numeric(value, errors="coerce")
        if pd.isna(year):
            continue
        year = int(year)
        if 1800 <= year <= 2100:
            year_cols[int(idx)] = year
    return year_cols


def _build_dates_from_template_pos(template_pos: pd.Series, years: pd.Series) -> pd.Series:
    leap_template = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    if int(template_pos.max()) >= len(leap_template):
        raise ValueError(f"Template position exceeds leap-year length: max={int(template_pos.max())}")

    month = leap_template.month.to_numpy()[template_pos.to_numpy(dtype=int)]
    day = leap_template.day.to_numpy()[template_pos.to_numpy(dtype=int)]
    valid = ~((month == 2) & (day == 29) & ~years.map(calendar.isleap).to_numpy(dtype=bool))
    out = pd.Series(pd.NaT, index=template_pos.index, dtype="datetime64[ns]")
    out.loc[valid] = pd.to_datetime(
        {
            "year": years.loc[valid].to_numpy(dtype=int),
            "month": month[valid],
            "day": day[valid],
        },
        errors="coerce",
    ).to_numpy()
    return out


def load_wide_daily_sheet(path: Path, sheet_name: str, year_header_row: int, value_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    if raw.empty:
        return pd.DataFrame(columns=["date", value_name])

    year_cols = _extract_year_cols(raw.iloc[year_header_row])
    if not year_cols:
        raise ValueError(f"No year columns found in {path} [{sheet_name}]")

    keep_idxs = [0, *year_cols.keys()]
    rows = raw.iloc[:, keep_idxs].copy()
    rows.columns = ["template_raw", *[year_cols[i] for i in year_cols]]
    rows["template_dt"] = pd.to_datetime(rows["template_raw"], errors="coerce", format="mixed")
    rows = rows[rows["template_dt"].notna()].copy().reset_index(drop=True)
    rows["template_pos"] = np.arange(len(rows), dtype=int)

    melted = rows.melt(
        id_vars=["template_pos"],
        value_vars=[c for c in rows.columns if isinstance(c, int)],
        var_name="year",
        value_name=value_name,
    )
    melted["year"] = melted["year"].astype(int)
    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    melted["date"] = _build_dates_from_template_pos(melted["template_pos"], melted["year"])
    melted = melted.dropna(subset=["date", value_name]).copy()
    return melted[["date", value_name]].sort_values("date").reset_index(drop=True)


def _clip_plausible(df: pd.DataFrame, col: str, lower: float, upper: float) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan


def load_historical(temp_xlsx: Path, humidity_xlsx: Path) -> pd.DataFrame:
    tmax = load_wide_daily_sheet(temp_xlsx, "Max", year_header_row=0, value_name="t_max_c")
    tmin = load_wide_daily_sheet(temp_xlsx, "Min", year_header_row=0, value_name="t_min_c")
    tmean = load_wide_daily_sheet(temp_xlsx, "Ort", year_header_row=0, value_name="t_mean_c")
    rh = load_wide_daily_sheet(humidity_xlsx, "Nem 1911-", year_header_row=1, value_name="rh_mean_pct")

    hist = tmax.merge(tmin, on="date", how="inner").merge(tmean, on="date", how="left").merge(rh, on="date", how="inner")
    _clip_plausible(hist, "t_max_c", -40.0, 60.0)
    _clip_plausible(hist, "t_min_c", -40.0, 40.0)
    _clip_plausible(hist, "t_mean_c", -40.0, 50.0)
    _clip_plausible(hist, "rh_mean_pct", 0.0, 100.0)
    hist = hist.dropna(subset=["t_max_c", "t_min_c", "rh_mean_pct"]).copy()
    hist = hist[hist["t_max_c"] >= hist["t_min_c"]].copy()
    hist["source"] = "historical_wide"
    hist["source_temp"] = "Uzunyil-Max-Min-Ort"
    hist["source_humidity"] = "1911-2022-Nem"
    hist["ea_formula"] = "rhmean"
    hist["rh_max_pct"] = np.nan
    hist["rh_min_pct"] = np.nan
    hist["auto_obs_count"] = np.nan
    return hist.sort_values("date").reset_index(drop=True)


def load_auto_station(table1_path: Path, table2_path: Path) -> pd.DataFrame:
    t1 = pd.read_csv(table1_path, skiprows=[0, 2, 3], low_memory=False)
    t1["TIMESTAMP"] = pd.to_datetime(t1["TIMESTAMP"], errors="coerce")
    t1 = t1.dropna(subset=["TIMESTAMP"]).copy()
    for col in ["AirTCee181_Avg", "RHee181", "RHee181_Avg"]:
        t1[col] = pd.to_numeric(t1[col], errors="coerce")
    _clip_plausible(t1, "AirTCee181_Avg", -40.0, 60.0)
    _clip_plausible(t1, "RHee181", 0.0, 100.0)
    _clip_plausible(t1, "RHee181_Avg", 0.0, 100.0)
    t1["date"] = t1["TIMESTAMP"].dt.floor("D")
    auto_rh = (
        t1.groupby("date", as_index=False)
        .agg(
            t_mean_c=("AirTCee181_Avg", "mean"),
            rh_mean_pct=("RHee181_Avg", "mean"),
            rh_max_pct=("RHee181", "max"),
            rh_min_pct=("RHee181", "min"),
            auto_obs_count=("RHee181_Avg", "count"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    t2 = pd.read_csv(table2_path, skiprows=[0, 2, 3], low_memory=False)
    t2["TIMESTAMP"] = pd.to_datetime(t2["TIMESTAMP"], errors="coerce")
    t2 = t2.dropna(subset=["TIMESTAMP"]).copy()
    for col in ["AirTCee181_Max", "AirTCee181_Min"]:
        t2[col] = pd.to_numeric(t2[col], errors="coerce")
    _clip_plausible(t2, "AirTCee181_Max", -40.0, 60.0)
    _clip_plausible(t2, "AirTCee181_Min", -40.0, 40.0)
    t2["date"] = t2["TIMESTAMP"].dt.floor("D") - pd.Timedelta(days=1)
    t2 = t2[t2["AirTCee181_Max"] >= t2["AirTCee181_Min"]].copy()
    auto_temp = (
        t2.groupby("date", as_index=False)
        .agg(
            t_max_c=("AirTCee181_Max", "max"),
            t_min_c=("AirTCee181_Min", "min"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    auto = auto_temp.merge(auto_rh, on="date", how="left")
    auto = auto.dropna(subset=["t_max_c", "t_min_c"]).copy()
    auto["source"] = "auto_station"
    auto["source_temp"] = "CR800_Table2"
    auto["source_humidity"] = "CR800_Table1"
    auto["ea_formula"] = np.where(auto["rh_max_pct"].notna() & auto["rh_min_pct"].notna(), "rhmax_rhmin", "rhmean")
    return auto.sort_values("date").reset_index(drop=True)


def compute_es_ea(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["es_tmax_kpa"] = saturation_vapor_pressure_kpa(out["t_max_c"]).to_numpy(dtype=float)
    out["es_tmin_kpa"] = saturation_vapor_pressure_kpa(out["t_min_c"]).to_numpy(dtype=float)
    out["es_kpa"] = 0.5 * (out["es_tmax_kpa"] + out["es_tmin_kpa"])

    ea_rhmean = (out["rh_mean_pct"].to_numpy(dtype=float) / 100.0) * out["es_kpa"].to_numpy(dtype=float)
    ea_rhmaxmin = 0.5 * (
        out["es_tmin_kpa"].to_numpy(dtype=float) * (out["rh_max_pct"].to_numpy(dtype=float) / 100.0)
        + out["es_tmax_kpa"].to_numpy(dtype=float) * (out["rh_min_pct"].to_numpy(dtype=float) / 100.0)
    )
    use_rhmaxmin = out["ea_formula"].eq("rhmax_rhmin").to_numpy(dtype=bool)
    out["ea_kpa"] = np.where(use_rhmaxmin, ea_rhmaxmin, ea_rhmean)
    out["ea_kpa"] = np.clip(out["ea_kpa"], 0.0, out["es_kpa"].to_numpy(dtype=float))
    out["es_minus_ea_kpa"] = np.clip(out["es_kpa"].to_numpy(dtype=float) - out["ea_kpa"].to_numpy(dtype=float), 0.0, None)
    out["vpd_kpa"] = out["es_minus_ea_kpa"]
    return out


def build_output(historical: pd.DataFrame, auto: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([historical, auto], ignore_index=True, sort=False)
    combined["source_priority"] = combined["source"].map({"historical_wide": 1, "auto_station": 2}).fillna(0).astype(int)
    combined = combined.sort_values(["date", "source_priority"], ascending=[True, False]).drop_duplicates("date", keep="first")
    combined = combined.sort_values("date").reset_index(drop=True)
    out = compute_es_ea(combined)

    out_cols = [
        "date",
        "t_max_c",
        "t_min_c",
        "t_mean_c",
        "rh_mean_pct",
        "rh_max_pct",
        "rh_min_pct",
        "es_tmax_kpa",
        "es_tmin_kpa",
        "es_kpa",
        "ea_kpa",
        "es_minus_ea_kpa",
        "vpd_kpa",
        "source",
        "source_temp",
        "source_humidity",
        "ea_formula",
        "auto_obs_count",
    ]
    return out[out_cols].copy()


def build_summary(out_df: pd.DataFrame, historical: pd.DataFrame, auto: pd.DataFrame) -> dict:
    return {
        "coverage": {
            "rows_total": int(len(out_df)),
            "date_min": str(out_df["date"].min().date()) if not out_df.empty else None,
            "date_max": str(out_df["date"].max().date()) if not out_df.empty else None,
            "historical_candidate_rows": int(len(historical)),
            "auto_candidate_rows": int(len(auto)),
            "rows_selected_historical": int((out_df["source"] == "historical_wide").sum()),
            "rows_selected_auto": int((out_df["source"] == "auto_station").sum()),
        },
        "method_selection": {
            "historical": "es from Tmax/Tmin, ea from RHmean",
            "automatic_station": "es from Tmax/Tmin, ea from RHmax/RHmin when available",
        },
        "stats": {
            "es_kpa_mean": float(out_df["es_kpa"].mean()),
            "ea_kpa_mean": float(out_df["ea_kpa"].mean()),
            "es_minus_ea_kpa_mean": float(out_df["es_minus_ea_kpa"].mean()),
            "es_minus_ea_kpa_max": float(out_df["es_minus_ea_kpa"].max()),
            "es_minus_ea_kpa_min": float(out_df["es_minus_ea_kpa"].min()),
        },
        "notes": [
            "Historical wide sheets are mapped by row order onto a leap-year template to avoid leap-day drift.",
            "Humidity workbook duplicate 1900-02-28 rows are treated as Feb 28 and Feb 29 via row position.",
            "Automatic station Table2 daily summaries are shifted back by one day because midnight rows summarize the previous day.",
            "Automatic station RH/T values outside plausible physical ranges are discarded before aggregation.",
        ],
    }


def main() -> None:
    args = parse_args()
    historical = load_historical(args.temp_xlsx, args.humidity_xlsx)
    auto = load_auto_station(args.auto_table1, args.auto_table2)
    out_df = build_output(historical, auto)
    out_series = out_df[["date", "es_minus_ea_kpa"]].copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_series_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    out_series.to_csv(args.out_series_csv, index=False)

    summary = build_summary(out_df, historical, auto)
    args.out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_series_csv}")
    print(f"Wrote: {args.out_summary_json}")
    print(f"Rows: {len(out_df)} | Range: {summary['coverage']['date_min']} -> {summary['coverage']['date_max']}")


if __name__ == "__main__":
    main()
