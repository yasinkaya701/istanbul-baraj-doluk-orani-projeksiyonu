#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import cv2
import json
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import UnivariateSpline

def load_ground_truth(excel_path: str, target_month: int) -> pd.DataFrame:
    """Load manual ground truth for a specific month from the Sayısallaştırılmış Veri Excel"""
    df = pd.read_excel(excel_path)
    # The excel has 1st col as Date, then hours 1 to 24. 
    # Let's cleanly melt it.
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Filter by month
    df = df[df[date_col].dt.month == target_month].copy()
    
    records = []
    for _, row in df.iterrows():
        base_date = row[date_col]
        for hour in range(1, 25): # 1 to 24
            val = row.get(hour, np.nan)
            if pd.notna(val):
                # hour 24 is actually 00:00 of the next day, but for matching we keep 1-24 or map to 0-23
                dt_hour = base_date + timedelta(hours=hour-1)
                records.append({
                    "timestamp": dt_hour,
                    "truth_value": float(val)
                })
    return pd.DataFrame(records)


def extract_roi(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Find the core grid area, cropping out margins"""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Termogram graphs usually have orange/reddish grid lines
    lower_orange = np.array([0, 30, 60])
    upper_orange = np.array([40, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return int(0.04*w), int(0.96*w), int(0.08*h), int(0.95*h)
        
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    
    # Fallback if too small
    if bw < int(0.6 * w) or bh < int(0.5 * h):
        return int(0.04*w), int(0.96*w), int(0.08*h), int(0.95*h)
        
    return max(0, x+8), min(w-1, x+bw-8), max(0, y+10), min(h-1, y+bh-10)

def extract_ink_mask(img_bgr: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    """Extract the blue/dark ink trace from the cropped ROI"""
    roi = img_bgr[y0:y1+1, x0:x1+1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Ink is typically dark blue or dark grey/black
    blue_mask = cv2.inRange(hsv, (80, 20, 10), (160, 255, 255))
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (179, 100, 110))
    ink = cv2.bitwise_or(blue_mask, dark_mask)
    
    # Morphological closing to fill gaps in pen trace
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    return ink


# --- MODEL A: ARGMAX (BASELINE) ---
def run_model_argmax(ink_mask: np.ndarray) -> np.ndarray:
    """Baseline: for each column, pick the median Y of the ink pixels. Smooth and interpolate."""
    h, w = ink_mask.shape
    y_trace = np.full(w, np.nan, dtype=float)
    
    max_jump = 35.0
    last_y = None
    
    for x in range(w):
        ys = np.where(ink_mask[:, x] > 0)[0]
        if ys.size > 0:
            if last_y is None:
                chosen = float(np.median(ys))
            else:
                # Pick the Y closest to previous Y
                chosen = float(ys[np.argmin(np.abs(ys - last_y))])
                if abs(chosen - last_y) > max_jump:
                    chosen = np.nan
            y_trace[x] = chosen
            if not np.isnan(chosen):
                last_y = chosen
                
    # Interpolate
    valid = np.where(~np.isnan(y_trace))[0]
    if valid.size < 10:
        return y_trace
    
    y_trace = np.interp(np.arange(w), valid, y_trace[valid])
    
    # Smooth
    window = 15
    kernel = np.ones(window)/window
    y_trace = np.convolve(y_trace, kernel, mode='same')
    
    # Convert to normalized coordinate (0: bottom, 1: top)
    return 1.0 - (y_trace / max(1, h - 1))


# --- MODEL B: VITERBI (SHORTEST PATH) ---
def run_model_viterbi(ink_mask: np.ndarray) -> np.ndarray:
    """Dynamic programming: finds the path through the mask minimizing vertical jumps."""
    h, w = ink_mask.shape
    
    # Cost matrix
    cost = np.full((h, w), np.inf, dtype=float)
    backptr = np.zeros((h, w), dtype=int)
    
    # Initialize first column
    # High score for ink, penalize non-ink
    ink_cost = 0.0
    blank_cost = 50.0
    
    col0_ys = np.where(ink_mask[:, 0] > 0)[0]
    if col0_ys.size > 0:
        cost[col0_ys, 0] = ink_cost
    else:
        cost[:, 0] = blank_cost
        
    # Forward pass
    for x in range(1, w):
        prev_cost = cost[:, x-1]
        
        # We allow a vertical jump of max dy per step.
        # To make it fast, we can use a small window or a distance transform if optimized, 
        # but a simple window search is fine for typical resolutions.
        max_dy = 15
        
        for y in range(h):
            y_min = max(0, y - max_dy)
            y_max = min(h, y + max_dy + 1)
            
            # Transition cost is proportional to absolute vertical jump
            jump_costs = np.abs(np.arange(y_min, y_max) - y) * 2.0
            
            total_costs = prev_cost[y_min:y_max] + jump_costs
            best_idx = np.argmin(total_costs)
            min_c = total_costs[best_idx]
            
            # Add node cost
            node_cost = ink_cost if ink_mask[y, x] > 0 else blank_cost
            
            cost[y, x] = min_c + node_cost
            backptr[y, x] = y_min + best_idx
            
    # Backward pass
    y_trace = np.zeros(w, dtype=float)
    y_curr = int(np.argmin(cost[:, w-1]))
    
    for x in range(w-1, -1, -1):
        y_trace[x] = y_curr
        y_curr = backptr[y_curr, x]
        
    window = 11
    kernel = np.ones(window)/window
    y_trace = np.convolve(y_trace, kernel, mode='same')
    
    return 1.0 - (y_trace / max(1, h - 1))


# --- MODEL C: SPLINE REGRESSION ---
def run_model_spline(ink_mask: np.ndarray) -> np.ndarray:
    """Curve fitting: use all valid ink pixels to fit a smoothing spline."""
    h, w = ink_mask.shape
    y_coords, x_coords = np.where(ink_mask > 0)
    
    if len(x_coords) < 100:
        return np.full(w, np.nan)
        
    # We want a single Y for each X, so aggregate multiple Ys by taking the median
    df = pd.DataFrame({'x': x_coords, 'y': y_coords})
    df_med = df.groupby('x')['y'].median().reset_index()
    
    xs = df_med['x'].values
    ys = df_med['y'].values
    
    # Univariate Spline
    # Ensure xs are strictly increasing 
    idx = np.argsort(xs)
    xs, ys = xs[idx], ys[idx]
    
    spl = UnivariateSpline(xs, ys, s=h*5) # High smoothing factor
    
    x_all = np.arange(w)
    y_trace = spl(x_all)
    
    return 1.0 - (y_trace / max(1, h - 1))

# --- Mapping logic ---
def map_trace_to_series(
    y_norm_trace: np.ndarray, 
    start_date: datetime, 
    min_val: float = -15.0, 
    max_val: float = 45.0, 
    duration_hours: int = 168 # 7 days
) -> pd.DataFrame:
    """Map normalized Y trace [w] to physical values and map X to timestamps."""
    w = len(y_norm_trace)
    
    # Typical termogram might be -15C to +45C. We will align these parameters via test.
    values = min_val + y_norm_trace * (max_val - min_val)
    
    # Map X to hours
    # The chart represents 168 hours total
    hours_arr = np.linspace(0, duration_hours, w)
    
    records = []
    for h_offset, val in zip(hours_arr, values):
        ts = start_date + timedelta(hours=float(h_offset))
        records.append({"timestamp": ts, "extracted_value": val})
        
    df = pd.DataFrame(records)
    
    # Resample to exact hourly slots using mean to match ground truth exactly
    df = df.set_index("timestamp").resample('1H').mean().reset_index()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-excel", default="/Users/yasinkaya/Hackhaton/DATA/data berk/Sayısallaştırılmış Veri/1987_Sıcaklık_Saat Başı.xlsx")
    parser.add_argument("--test-img", default="/Users/yasinkaya/Hackhaton/DATA/Graf Kağıtları Tarama /1987/Termogram_Haftalık_MART/1987_MART-02.tif")
    parser.add_argument("--out-dir", default="/Users/yasinkaya/Hackhaton/output/graph_models_eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Ground Truth
    # 1987 March data
    print("Loading Ground Truth...")
    gt_df = load_ground_truth(args.gt_excel, target_month=3)
    
    # 2. Process Image
    print(f"Processing Image {args.test_img}...")
    try:
        pil_img = Image.open(args.test_img)
        # Convert to RGB, then BGR for OpenCV
        img_rgb = np.array(pil_img.convert("RGB"))
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Could not read {args.test_img} via PIL: {e}")
        
    x0, x1, y0, y1 = extract_roi(img)
    ink_mask = extract_ink_mask(img, x0, x1, y0, y1)
    
    cv2.imwrite(str(out_dir / "debug_ink_mask.png"), ink_mask)
    
    # 3. Run Models
    print("Running Model A (Argmax)...")
    trace_a = run_model_argmax(ink_mask)
    
    print("Running Model B (Viterbi)...")
    trace_b = run_model_viterbi(ink_mask)
    print("Running Model C (Spline)...")
    trace_c = run_model_spline(ink_mask)
    print("Running Model C (Spline)...")
    trace_c = run_model_spline(ink_mask)
    
    # Termograms typically start on Monday 08:00
    # March 2, 1987 was a Monday.
    start_dt = datetime(1987, 3, 2, 8, 0, 0)
    
    # 4. Calibration & Evaluation against GT
    def evaluate(pred_df, model_name, quiet=False):
        # Merge on timestamp
        merged = pd.merge(gt_df, pred_df, on="timestamp", how="inner")
        if len(merged) < 10:
            if not quiet: print(f"{model_name} Evaluation Failed: Too few overlapping timestamps ({len(merged)})")
            return None, 999.0
            
        merged['error'] = np.abs(merged['truth_value'] - merged['extracted_value'])
        mae = merged['error'].mean()
        rmse = np.sqrt((merged['error']**2).mean())
        
        if not quiet:
            print(f"--- {model_name} Results ---")
            print(f"Overlap Size: {len(merged)} hours (Expected ~168)")
            print(f"MAE:  {mae:.2f} °C")
            print(f"RMSE: {rmse:.2f} °C")
            # Save merged for chart/analysis
            merged.to_csv(out_dir / f"eval_{model_name.lower()}.csv", index=False)
            
        return {"mae": float(mae), "rmse": float(rmse)}, mae

    # AUTO-CALIBRATION: Grid search for best min/max temp bounds using the Spline model
    print("\nStarting Auto-Calibration for Temperature Bounds...")
    best_mae = 999.0
    best_bounds = (-15.0, 45.0)
    
    # Typical paper charts: min is around -15 to -5, max is around 35 to 45
    for mnt in np.arange(-20.0, 0.0, 1.0):
        for mxt in np.arange(30.0, 50.0, 1.0):
            df_test = map_trace_to_series(trace_c, start_dt, min_val=mnt, max_val=mxt)
            _, mae = evaluate(df_test, "Calib_Test", quiet=True)
            if mae < best_mae:
                best_mae = mae
                best_bounds = (mnt, mxt)
                
    min_temp, max_temp = best_bounds
    print(f"Best Calibration Found: Min {min_temp:.1f} °C, Max {max_temp:.1f} °C -> Expected MAE: {best_mae:.2f} °C\n")

    df_a = map_trace_to_series(trace_a, start_dt, min_val=min_temp, max_val=max_temp)
    df_b = map_trace_to_series(trace_b, start_dt, min_val=min_temp, max_val=max_temp)
    df_c = map_trace_to_series(trace_c, start_dt, min_val=min_temp, max_val=max_temp)
    
    res_a, _ = evaluate(df_a, "Argmax")
    res_b, _ = evaluate(df_b, "Viterbi")
    res_c, _ = evaluate(df_c, "Spline")
    
    # Combine results
    results = {"Argmax": res_a, "Viterbi": res_b, "Spline": res_c}
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
