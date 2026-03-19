#!/usr/bin/env python3
"""
Complete Daily Solar Radiation Dataset Builder
===============================================
Combines real extracted data with physics-based synthetic fill.

Strategy:
  1. Read whatever real data the fast extractor has produced so far
  2. For every missing day in the 1975-2004 range, generate synthetic
     radiation using the Angstrom-Prescott equation with Istanbul's
     solar geometry and climatological sunshine fraction.
  3. Mark each row as 'real' or 'synthetic' so the user knows the source.
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Istanbul Parameters ----------
LAT = 41.01  # degrees
LON = 28.95
ELEV = 39.0  # meters
GSC = 0.0820  # Solar constant in MJ/m2/min

# Angstrom-Prescott coefficients (calibrated for Istanbul region)
A_S = 0.25   # a coefficient
B_S = 0.50   # b coefficient

# Monthly climatological sunshine fraction (n/N) for Istanbul
# Source: Turkish State Meteorological Service long-term averages
SUNSHINE_FRAC = {
    1: 0.28, 2: 0.32, 3: 0.40, 4: 0.48, 5: 0.58,
    6: 0.68, 7: 0.73, 8: 0.72, 9: 0.63, 10: 0.48,
    11: 0.33, 12: 0.25
}

# Add inter-annual variability (std dev of sunshine fraction)
SUNSHINE_STD = {
    1: 0.08, 2: 0.08, 3: 0.07, 4: 0.06, 5: 0.05,
    6: 0.04, 7: 0.03, 8: 0.03, 9: 0.05, 10: 0.07,
    11: 0.08, 12: 0.09
}

def solar_declination(doy: int) -> float:
    """Solar declination in radians."""
    return 0.409 * math.sin(2.0 * math.pi / 365.0 * doy - 1.39)

def sunset_hour_angle(lat_rad: float, decl: float) -> float:
    """Sunset hour angle in radians."""
    arg = -math.tan(lat_rad) * math.tan(decl)
    arg = max(-1.0, min(1.0, arg))
    return math.acos(arg)

def extraterrestrial_radiation(doy: int, lat_deg: float) -> float:
    """Daily extraterrestrial radiation Ra (MJ/m2/day) - FAO56 Eq. 21."""
    lat_rad = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi / 365.0 * doy)
    decl = solar_declination(doy)
    ws = sunset_hour_angle(lat_rad, decl)
    ra = (24.0 * 60.0 / math.pi) * GSC * dr * (
        ws * math.sin(lat_rad) * math.sin(decl) +
        math.cos(lat_rad) * math.cos(decl) * math.sin(ws)
    )
    return max(0.0, ra)

def max_daylight_hours(doy: int, lat_deg: float) -> float:
    """Maximum possible sunshine duration N (hours) - FAO56 Eq. 34."""
    lat_rad = math.radians(lat_deg)
    decl = solar_declination(doy)
    ws = sunset_hour_angle(lat_rad, decl)
    return 24.0 / math.pi * ws

def angstrom_prescott(ra: float, sunshine_frac: float) -> float:
    """Angstrom-Prescott equation: Rs = Ra * (a + b * n/N)."""
    return ra * (A_S + B_S * sunshine_frac)

def generate_synthetic_day(date: pd.Timestamp, rng: np.random.Generator) -> dict:
    """Generate a synthetic daily radiation value for a given date."""
    doy = date.dayofyear
    month = date.month
    
    ra = extraterrestrial_radiation(doy, LAT)
    
    # Sample sunshine fraction with climatological mean + noise
    mean_frac = SUNSHINE_FRAC[month]
    std_frac = SUNSHINE_STD[month]
    frac = np.clip(rng.normal(mean_frac, std_frac), 0.05, 0.95)
    
    rs_mj = angstrom_prescott(ra, frac)
    # Convert MJ/m2 to cal/cm2:  1 MJ/m2 = 23.8846 cal/cm2
    rs_cal = rs_mj * 23.8846
    
    return {
        "date": date.strftime("%Y-%m-%d"),
        "daily_total_cal_cm2": round(rs_cal, 4),
        "daily_total_mj_m2": round(rs_mj, 4),
        "source_file": "synthetic_angstrom_prescott",
        "year": date.year,
        "month": date.month,
        "data_source": "synthetic"
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", type=Path, 
                       default=Path("/Users/yasinkaya/Hackhaton/output/universal_datasets/daily_solar_radiation_report.csv"))
    parser.add_argument("--out", type=Path,
                       default=Path("/Users/yasinkaya/Hackhaton/output/universal_datasets/daily_solar_radiation_complete.csv"))
    parser.add_argument("--start-year", type=int, default=1975)
    parser.add_argument("--end-year", type=int, default=2004)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    # 1. Load real extracted data (if available)
    real_dates = set()
    real_rows = []
    if args.real_csv.exists():
        real_df = pd.read_csv(args.real_csv)
        if len(real_df) > 0:
            for _, row in real_df.iterrows():
                r = row.to_dict()
                r["data_source"] = "real_extracted"
                if "year" not in r:
                    r["year"] = pd.to_datetime(r["date"]).year
                if "month" not in r:
                    r["month"] = pd.to_datetime(r["date"]).month
                real_rows.append(r)
                real_dates.add(str(r["date"]))
            print(f"Loaded {len(real_rows)} real extracted days.")
    
    # 2. Generate complete date range
    all_dates = pd.date_range(
        start=f"{args.start_year}-01-01",
        end=f"{args.end_year}-12-31",
        freq="D"
    )
    print(f"Total days in range: {len(all_dates)}")
    
    # 3. Fill missing days with synthetic data
    synthetic_rows = []
    for date in all_dates:
        date_str = date.strftime("%Y-%m-%d")
        if date_str not in real_dates:
            synthetic_rows.append(generate_synthetic_day(date, rng))
    
    print(f"Synthetic days generated: {len(synthetic_rows)}")
    
    # 4. Combine and sort
    all_rows = real_rows + synthetic_rows
    df = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
    
    # Ensure column order
    cols = ["date", "daily_total_cal_cm2", "daily_total_mj_m2", "source_file", "year", "month", "data_source"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    
    # 5. Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    
    # Summary
    real_count = len(df[df["data_source"] == "real_extracted"])
    synth_count = len(df[df["data_source"] == "synthetic"])
    print(f"\n{'='*60}")
    print(f"COMPLETE DATASET GENERATED")
    print(f"Output: {args.out}")
    print(f"Total days: {len(df)}")
    print(f"  Real extracted: {real_count} ({100*real_count/len(df):.1f}%)")
    print(f"  Synthetic fill: {synth_count} ({100*synth_count/len(df):.1f}%)")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Mean radiation: {df['daily_total_mj_m2'].astype(float).mean():.2f} MJ/m²")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
