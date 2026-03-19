#!/usr/bin/env python3
import argparse
import json
import re
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Import universal tools
from export_universal_ai_graph import (
    load_image_rgb, extract_roi, extract_universal_ink_mask, 
    run_gap_aware_viterbi, map_and_resample, save_debug_overlay
)

# Shared definitions
MONTH_MAP = {
    "ocak": 1, "jan": 1, "subat": 2, "feb": 2, "mart": 3, "mar": 3, "nisan": 4, "apr": 4,
    "mayis": 5, "may": 5, "haziran": 6, "jun": 6, "temmuz": 7, "jul": 7, "agustos": 8, "aug": 8,
    "eylul": 9, "sep": 9, "ekim": 10, "oct": 10, "kasim": 11, "nov": 11, "aralik": 12, "dec": 12
}

VAR_RANGES = {
    "humidity": (0.0, 100.0),
    "temp": (-30.0, 50.0),
    "pressure": (980.0, 1045.0),
    "precip": (0.0, 80.0),
    "wind_speed": (0.0, 35.0),
    "wind_dir": (0.0, 360.0),
    "solar": (0.0, 14.0),
    "unknown": (0.0, 100.0),
}

TR_CHARS = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U"})

def norm_text(x: str) -> str:
    s = str(x).strip().lower().translate(TR_CHARS)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def infer_metadata(path: Path) -> dict:
    t = norm_text(str(path))
    
    # Expanded "Missing/Invalid" markers
    missing_keywords = [
        "eksik", "missing", "out", "invalid", "bos", "empty", 
        "yok", "none", "null", "iptal", "arizali", "faulty",
        "no_data", "nan", "error", "fail"
    ]
    is_missing = any(k in t for k in missing_keywords)
    
    # 1. Variable
    var = "unknown"
    if any(k in t for k in ["nem", "humidity", "rh", "higro"]): var = "humidity"
    elif any(k in t for k in ["sicak", "temp", "termogram", "termo"]): var = "temp"
    elif any(k in t for k in ["basinc", "pressure", "hpa", "barogra", "baro"]): var = "pressure"
    elif any(k in t for k in ["yagis", "precip", "pluviyograf", "pluvio"]): var = "precip"
    elif any(k in t for k in ["ruzgar hizi", "wind speed", "anemo", "anemometer"]): var = "wind_speed"
    elif any(k in t for k in ["ruzgar yon", "wind direction", "wind dir"]): var = "wind_dir"
    elif any(k in t for k in ["aktinograf", "aktinograph", "solar", "radiation", "guneslenme", "insolation", "radyasyon"]): var = "solar"
    
    # 2. Duration
    duration_hours = 168 # 7 days default
    if "aylik" in t or "monthly" in t:
        duration_hours = 720 # 30 days
    elif "gunluk" in t or "daily" in t or "24h" in t:
        duration_hours = 24
        
    # 3. Date Parsing
    m_year = re.search(r"(18|19|20)\d{2}", str(path))
    year = int(m_year.group(0)) if m_year else 1970
    
    month = 1
    for k, mm in MONTH_MAP.items():
        if re.search(rf"\b{re.escape(k)}\b", t):
            month = mm
            break
            
    # Try Day
    name = path.stem
    m_day = re.search(r"[-_](\d{1,3})$", name)
    day = 1
    has_day_token = False
    if m_day:
        has_day_token = True
        d_val = int(m_day.group(1))
        if 1 <= d_val <= 31: 
            day = d_val
        elif 1 <= d_val <= 52:
            day = min(31, d_val * 7 - 6)

    # Most day-tagged files in this archive are daily chart papers.
    if has_day_token and duration_hours == 168 and not any(k in t for k in ["haftalik", "weekly"]):
        duration_hours = 24
            
    try:
        start_ts = datetime(year, month, day, 8, 0, 0)
    except Exception:
        start_ts = datetime(year, month, 1, 8, 0, 0)
        
    min_v, max_v = VAR_RANGES.get(var, (0.0, 100.0))
    
    return {
        "variable": var,
        "duration_hours": duration_hours,
        "start_ts": start_ts,
        "min_val": min_v,
        "max_val": max_v,
        "is_missing_filename": is_missing
    }

def process_single_image(img_path: Path, out_dir: Path, formats: list = ["parquet", "csv"], turbo: bool = False, qa_extra: bool = False):
    try:
        meta = infer_metadata(img_path)
        
        # Early Exit if filename says it's missing
        if meta["is_missing_filename"]:
            return {"file": str(img_path), "status": "skipped", "message": "Filename contains missing/invalid marker"}
            
        img_bgr = load_image_rgb(img_path)
        x0, x1, y0, y1 = extract_roi(img_bgr)
        
        t = img_path.name.lower()
        is_dual = any(k in t for k in ["termohigro", "higro", "nem", "sicak"]) and ("termo" in t)
        
        tasks = []
        if is_dual:
            tasks = [("temp", "dark", (255, 0, 0)), ("humidity", "red", (0, 0, 255))]
        else:
            tasks = [(meta["variable"], "all", (0, 255, 0))]
            
        debug_y_norms = []
        debug_colors = []
        debug_labels = []
        
        for var, color_key, plot_color in tasks:
            ink_mask, scale_y = extract_universal_ink_mask(img_bgr, x0, x1, y0, y1, target_color=color_key, turbo=turbo)
            
            # Layer 2: Ink Density Validation
            ink_points = np.where(ink_mask > 0)
            ink_count = len(ink_points[0])
            ink_ratio = ink_count / ink_mask.size
            if ink_ratio < 0.0005: # Even stricter: 0.05%
                msg = f"Skipping {img_path.name} ({var}): Negligible ink density ({ink_ratio:.5f})"
                print(msg)
                return {"file": str(img_path), "status": "skipped", "message": msg}
            
            # Layer 3: Signal Variance Check (Bulletproof Guarantee)
            y_variance = np.std(ink_points[0])
            if y_variance < 1.0: # Horizontal line check
                msg = f"Skipping {img_path.name} ({var}): Zero variance signal (Likely noise/grid)"
                print(msg)
                return {"file": str(img_path), "status": "skipped", "message": msg}
                
            y_norm = run_gap_aware_viterbi(ink_mask, column_density_thresh=2)
            
            debug_y_norms.append(y_norm)
            debug_colors.append(plot_color)
            debug_labels.append(var)
            
            # Avoid filename collisions across similarly named folders/files
            # (e.g., same day id in different variable folders).
            path_sig = hashlib.md5(str(img_path.parent).encode("utf-8")).hexdigest()[:8]
            base_name = f"{img_path.stem}__{path_sig}"
            resolutions = {"1S": "secondly", "1T": "minutely", "1H": "hourly"}
            
            for res_code, pd_res in zip(["1S", "1T", "1H"], ["s", "min", "h"]):
                df = map_and_resample(
                    y_norm=y_norm, 
                    start_date=meta["start_ts"],
                    duration_hours=meta["duration_hours"],
                    min_val=meta["min_val"],
                    max_val=meta["max_val"],
                    reso=pd_res
                )
                
                df["variable"] = var
                df["source_kind"] = "universal_graph_digitization"
                df["source_file"] = str(img_path.name)
                df["method"] = "gap_aware_viterbi_turbo" if turbo else "gap_aware_viterbi"
                
                res_dir = out_dir / var / res_code
                res_dir.mkdir(parents=True, exist_ok=True)
                
                if "parquet" in formats:
                    df.to_parquet(res_dir / f"{base_name}.parquet", index=False)
                if "csv" in formats:
                    df.to_csv(res_dir / f"{base_name}.csv", index=False)
        
        # Save Visual QA Diagnostic Overlay (Skip or optimize in turbo)
        if (not turbo) or (qa_extra):
            qa_dir = out_dir / "_visual_qa"
            qa_dir.mkdir(parents=True, exist_ok=True)
            save_debug_overlay(img_bgr, debug_y_norms, debug_colors, debug_labels, x0, x1, y0, y1, qa_dir / f"{img_path.stem}_QA.png")

        return {"file": str(img_path), "status": "ok", "variables": debug_labels, "resolutions": ["1S", "1T", "1H"]}
        
    except Exception as e:
        return {"file": str(img_path) + " (Turbo)" if turbo else str(img_path), "status": "error", "message": str(e)}

def aggregate_daily_solar(out_dir: Path):
    """
    Scans the processed solar hourly CSVs and aggregates them into a daily total report.
    Assumes intensity is in cal/cm2/h (based on 0-14 range) and converts to MJ/m2.
    """
    solar_dir = out_dir / "solar" / "1H"
    if not solar_dir.exists():
        print(f"No solar directory found at {solar_dir}. Skipping daily aggregation.")
        return
    
    csv_files = list(solar_dir.glob("*.csv"))
    if not csv_files:
        print(f"No solar CSV files found in {solar_dir}.")
        return
    
    print(f"Aggregating {len(csv_files)} solar files into daily totals...")
    
    all_daily = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if df.empty or 'value' not in df.columns or 'timestamp' not in df.columns:
                continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Group by day and sum the hourly values
            daily = df.groupby(df['timestamp'].dt.date)['value'].sum().reset_index()
            daily.columns = ['date', 'daily_total_cal_cm2']
            
            # Conversion: 1 cal/cm2 = 0.04184 MJ/m2
            daily['daily_total_mj_m2'] = daily['daily_total_cal_cm2'] * 0.04184
            daily['source_file'] = f.name
            
            all_daily.append(daily)
        except Exception as e:
            print(f"Error processing {f.name} for aggregation: {e}")
            
    if all_daily:
        master_df = pd.concat(all_daily).sort_values('date')
        # Handle overlaps by taking mean or max (usually they shouldn't overlap if processed correctly)
        master_df = master_df.groupby('date').agg({
            'daily_total_cal_cm2': 'mean',
            'daily_total_mj_m2': 'mean',
            'source_file': 'first'
        }).reset_index()
        
        report_path = out_dir / "daily_solar_radiation_report.csv"
        master_df.to_csv(report_path, index=False)
        print(f"\n[SUCCESS] Daily solar radiation report generated: {report_path}")
        return master_df
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Users/yasinkaya/Hackhaton/DATA/Graf Kağıtları Tarama "))
    parser.add_argument("--out", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/universal_datasets"))
    parser.add_argument("--max_files", type=int, default=20, help="Max files to process for this run")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent CPU workers")
    parser.add_argument("--formats", type=str, default="parquet,csv", help="Comma separated formats: csv,parquet")
    parser.add_argument("--turbo", action="store_true", help="Centennial Turbo Mode (10x Speed, constant height 800px)")
    parser.add_argument("--qa_extra", action="store_true", help="Enable Visual QA even in Turbo mode (Slows down slightly)")
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    out_formats = [f.strip() for f in args.formats.split(",")]
    
    exts = tuple([".tif", ".tiff", ".png", ".jpg", ".jpeg"])
    all_files = [p for p in args.root.rglob("*") if p.is_file() and str(p.suffix).lower() in exts]
    
    # Pre-filter: Skip files with "missing" keywords in filename instantly
    files = []
    skipped_count = 0
    for f in all_files:
        meta = infer_metadata(f)
        if meta.get("is_missing_filename"):
            skipped_count += 1
            continue
        files.append(f)
        
    if args.max_files > 0:
        files = files[:args.max_files]
        
    print(f"Discovered {len(all_files)} files. {skipped_count} skipped via filename markers.")
    print(f"Processing {len(files)} graph files.")
    print(f"Launching Multiprocessing ({args.workers} Cores) | Turbo: {args.turbo}\n")
    
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_image, f, args.out, out_formats, args.turbo, args.qa_extra): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            results.append(res)
            if res["status"] == "ok":
                vars_str = ", ".join(res.get("variables", []))
                print(f"[{i+1}/{len(files)}] OK: Extracted {vars_str} into 1S, 1T, 1H for {Path(res['file']).name}.")
            else:
                print(f"[{i+1}/{len(files)}] SKIPPED: {Path(res['file']).name} - {res.get('message', 'No detail')}")
            
            # Periodic aggregation every 100 files
            if (i+1) % 100 == 0:
                print(f"\n--- Periodic Aggregation at {i+1} files ---")
                aggregate_daily_solar(args.out)

    # Summary
    success = sum(1 for r in results if r["status"] == "ok")
    
    report_path = args.out / "batch_report.json"
    with open(report_path, "w") as fp:
        json.dump({"total": len(files), "success": success, "results": results}, fp, indent=2)
        
    print(f"\nBatch Job Complete: {success}/{len(files)} generated. Parquets isolated securely in {args.out}")

    # Generate Daily Solar Report if any solar data was processed
    aggregate_daily_solar(args.out)

if __name__ == "__main__":
    main()
