#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/yasinkaya/Hackhaton")


def parse_matrix_xlsx(path: Path, sheet: str, value_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    # find date column (first column)
    date_col = df.columns[0]
    base_dates = pd.to_datetime(df[date_col], errors="coerce")
    # year columns: numeric headers
    year_cols = []
    for c in df.columns[1:]:
        try:
            y = int(str(c).strip())
            if 1800 <= y <= 2100:
                year_cols.append((c, y))
        except Exception:
            continue
    rows = []
    for col, year in year_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        for bd, v in zip(base_dates, vals):
            if pd.isna(bd) or pd.isna(v):
                continue
            try:
                dt = pd.Timestamp(year=year, month=bd.month, day=bd.day)
            except Exception:
                continue
            rows.append({"date": dt, value_name: float(v)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.groupby(out["date"].dt.to_period("M").dt.to_timestamp()).mean(numeric_only=True)
    out = out.reset_index().rename(columns={"date": "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def parse_precip_monthly(path: Path, sheet: str = "Uzun Yıllar") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    year_col = df.columns[0]
    years = pd.to_numeric(df[year_col], errors="coerce")
    # Month column candidates (multiple sets)
    month_names = [
        "ocak", "subat", "mart", "nisan", "mayis", "haziran",
        "temmuz", "agustos", "eylul", "ekim", "kasim", "aralik",
    ]

    def norm(s: str) -> str:
        s = str(s).lower()
        s = s.replace("ş", "s").replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")
        s = "".join([ch for ch in s if ch.isalpha() or ch == "."])
        return s

    # group columns by suffix (.1, .2, .3) or base
    col_map = {}
    for c in df.columns:
        ns = norm(c)
        for m in month_names:
            if ns.startswith(m):
                # detect suffix
                suffix = ""
                if "." in ns:
                    suffix = ns.split(".")[1]
                key = suffix if suffix != "" else "base"
                col_map.setdefault(key, {}).setdefault(m, []).append(c)
    # choose best set by non-null coverage
    best_key = None
    best_score = -1
    for key, months in col_map.items():
        score = 0
        for m in month_names:
            if m in months:
                col = months[m][0]
                score += pd.to_numeric(df[col], errors="coerce").notna().sum()
        if score > best_score:
            best_score = score
            best_key = key
    if best_key is None:
        return pd.DataFrame()

    month_cols = {}
    for m in month_names:
        if m in col_map.get(best_key, {}):
            month_cols[m] = col_map[best_key][m][0]

    rows = []
    for y, row in df.iterrows():
        year = years.iloc[y]
        if pd.isna(year):
            continue
        year = int(year)
        for idx, m in enumerate(month_names, start=1):
            col = month_cols.get(m)
            if col is None:
                continue
            v = pd.to_numeric(df.loc[y, col], errors="coerce")
            if pd.isna(v):
                continue
            rows.append({"date": pd.Timestamp(year=year, month=idx, day=1), "rain_mm_newdata": float(v)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.groupby("date", as_index=False).mean(numeric_only=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/newdata_feature_store/tables/newdata_meteo_monthly_from_xlsx.csv")
    args = p.parse_args()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = ROOT / "new data" / "Veriler_H-3"
    temp_path = base / "Sçcaklçk" / "Uzunyçl-Max-Min-Ort-orj.xlsx"
    hum_path = base / "Nem" / "1911-2022-Nem.xlsx"
    pres_path = base / "Basçná" / "Basçná 1912-2018Kasim.xlsx"
    precip_path = base / "Yaßçü" / "1911-2023.xlsx"

    temp = parse_matrix_xlsx(temp_path, "Ort", "t_mean_c")
    hum = parse_matrix_xlsx(hum_path, "Nem 1911-", "rh_mean_pct")
    pres = parse_matrix_xlsx(pres_path, "Basinc", "pressure_mmhg")
    precip = parse_precip_monthly(precip_path)
    if not pres.empty:
        pres["pressure_kpa_newdata"] = pres["pressure_mmhg"] * 0.133322
        pres = pres.drop(columns=["pressure_mmhg"])

    # merge
    df = None
    for part in [temp, hum, pres, precip]:
        if part is None or part.empty:
            continue
        if df is None:
            df = part.copy()
        else:
            df = df.merge(part, on="date", how="outer")
    if df is None:
        df = pd.DataFrame(columns=["date", "t_mean_c", "rh_mean_pct", "pressure_kpa_newdata"])
    df = df.sort_values("date")
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
