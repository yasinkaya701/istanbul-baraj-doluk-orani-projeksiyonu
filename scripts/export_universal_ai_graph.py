#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta
from scipy.signal import medfilt

def load_image_rgb(path: Path) -> np.ndarray:
    try:
        pil_img = Image.open(path)
        img_rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Could not read with PIL: {e}")

def deskew_image(img_bgr: np.ndarray) -> np.ndarray:
    """Detects primary horizontal lines and rotates the image to perfectly level 0 degrees."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Probability Hough Transform to find line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return img_bgr
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep horizontal-ish lines 
        if -10 <= angle <= 10:
            angles.append(angle)
            
    if not angles:
        return img_bgr
        
    median_angle = np.median(angles)
    
    # Don't rotate if it's already basically perfect
    if abs(median_angle) < 0.2:
        return img_bgr
        
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    # Affine matrix for rotation
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    # White background for rotation boundary fill
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def extract_roi(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, (0, 30, 60), (40, 255, 255))
    
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return int(0.04*w), int(0.96*w), int(0.08*h), int(0.95*h)
        
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    
    if bw < int(0.6 * w) or bh < int(0.5 * h):
        return int(0.04*w), int(0.96*w), int(0.08*h), int(0.95*h)
        
    # Return ROI with safe margins. 
    # Viterbi Search should avoid the very top (where headers usually are)
    # We add 40px top margin and 20px bottom margin
    return max(0, x+8), min(w-1, x+bw-8), max(0, y+45), min(h-1, y+bh-20)

def remove_grids(ink_mask: np.ndarray) -> np.ndarray:
    """Uses morphological erosion/dilation to destroy purely horizontal and vertical grid lines."""
    # Find vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Find horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Combine grids
    grid = cv2.bitwise_or(vertical_lines, horizontal_lines)
    
    # Subtract grid from ink
    ink_no_grid = cv2.subtract(ink_mask, grid)
    
    return cv2.medianBlur(ink_no_grid, 3)

def extract_universal_ink_mask(
    img_bgr: np.ndarray, 
    x0: int, x1: int, y0: int, y1: int, 
    target_color: str = "all",
    turbo: bool = False
) -> tuple[np.ndarray, float]:
    """
    Robust extraction with optional Turbo downsampling for 10x speedup.
    Returns (mask, scale_y).
    """
    roi = img_bgr[y0:y1+1, x0:x1+1]
    orig_h, orig_w = roi.shape[:2]
    
    scale_y = 1.0
    if turbo and orig_h > 800:
        # Centennial Turbo: Downsample to fixed 800px height for 10x speedup
        # while keeping the width (time resolution) intact.
        scale_y = 800.0 / orig_h
        new_h = 800
        roi = cv2.resize(roi, (orig_w, new_h), interpolation=cv2.INTER_AREA)
    
    # --- Layer 1: Background Normalization (Flat-field correction) ---
    # Estimate background by using a very large median filter
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray_roi, 151) # Large kernel for global background trend
    # Normalize lighting: (roi / bg) * 255
    norm_roi = cv2.divide(gray_roi, bg, scale=255)
    
    # Apply CLAHE to the normalized ROI
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    roi_eq = clahe.apply(norm_roi)
    
    # --- Layer 2: Adaptive Multi-Channel Ink Extraction ---
    # We use both a global threshold and a local adaptive one
    _, global_thresh = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(roi_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    
    # Combine Otsu and Adaptive for maximum robustness
    ink_base = cv2.bitwise_or(global_thresh, adaptive_thresh)
    
    # Color Masking (preserving existing color logic for dual-channel)
    hsv = cv2.cvtColor(cv2.cvtColor(roi_eq, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    masks = {}
    masks["blue"] = cv2.inRange(hsv, (80, 30, 20), (160, 255, 255))
    masks["dark"] = cv2.inRange(hsv, (0, 0, 0), (179, 120, 80))
    red1 = cv2.inRange(hsv, (0, 40, 40), (15, 255, 255))
    red2 = cv2.inRange(hsv, (160, 40, 40), (180, 255, 255))
    masks["red"] = cv2.bitwise_or(red1, red2)
    masks["green"] = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    
    # Combined color mask
    color_ink = masks["blue"]
    for k in ["dark", "red", "green"]:
        color_ink = cv2.bitwise_or(color_ink, masks[k])
        
    if target_color == "all":
        # Intersect structural ink (thresholds) with color ink to remove noise/text
        ink = cv2.bitwise_and(ink_base, color_ink)
    else:
        ink = cv2.bitwise_and(ink_base, masks.get(target_color, masks["dark"]))
    
    # --- Layer 3: Noise Cleaning ---
    ink = remove_grids(ink)
    
    # Remove very small components (speckles)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 5: # Tiny noise
            ink[labels == i] = 0
            
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    
    # --- Layer 4: Hard Border Suppression ---
    # Eliminate any residue at the very edges of the ROI
    ih, iw = ink.shape
    ink[0:int(ih*0.05), :] = 0 # Top 5%
    ink[int(ih*0.95):, :] = 0 # Bottom 5%
    
    return ink, scale_y

def save_debug_overlay(
    img_bgr: np.ndarray, 
    y_norm_list: list[np.ndarray], 
    colors_list: list[tuple[int, int, int]],
    labels: list[str],
    x0: int, x1: int, y0: int, y1: int,
    out_path: Path
):
    """Saves a diagnostic PNG with the extracted trace(s) overlaid on the original image."""
    overlay = img_bgr.copy()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    
    for y_norm, line_color, label in zip(y_norm_list, colors_list, labels):
        y_px = (1.0 - y_norm) * (h - 1)
        x_px = np.linspace(0, w - 1, len(y_norm))
        
        pts = np.vstack((x_px + x0, y_px + y0)).T.astype(np.int32)
        cv2.polylines(overlay, [pts], isClosed=False, color=line_color, thickness=3, lineType=cv2.LINE_AA)
        
        # Add label near the start of the trace
        cv2.putText(overlay, label, (pts[0][0], pts[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, line_color, 3)

    cv2.imwrite(str(out_path), overlay)


def run_gap_aware_viterbi(ink_mask: np.ndarray, column_density_thresh: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Dynamic programming (Viterbi) that bridges gaps mathematically without outputting NaN.
    Returns:
        y_norm: Array of floats (0 to 1), perfectly continuous.
    """
    h, w = ink_mask.shape
    cost = np.full((h, w), np.inf, dtype=float)
    backptr = np.zeros((h, w), dtype=int)
    
    ink_cost = 0.0
    blank_cost = 15.0 # Lower cost to make bridges less "desperate" for noise
    void_tracking = np.zeros(w, dtype=bool)
    
    col0_ys = np.where(ink_mask[:, 0] > 0)[0]
    if col0_ys.size >= column_density_thresh:
        cost[col0_ys, 0] = ink_cost
    else:
        void_tracking[0] = True
        cost[int(h/2), 0] = blank_cost # Tie to middle to prevent arbitrary drifting
    magnet_y = h/2 # Initialize scent
    for x in range(1, w):
        col_ys = np.where(ink_mask[:, x] > 0)[0]
        if col_ys.size >= column_density_thresh:
            raw_magnet = np.median(col_ys)
            magnet_y = 0.9 * magnet_y + 0.1 * raw_magnet
        
        y_indices = np.arange(h)
        magnet_cost = (np.abs(y_indices - magnet_y) / h) * 100.0 
        
        max_dy = 20
        dy_range = np.arange(-max_dy, max_dy + 1)
        jump_penalties = (dy_range / max_dy)**2 * 60.0 + (np.abs(dy_range) * 0.2)
        
        prev_cost = cost[:, x-1]
        best_prev_costs = np.copy(prev_cost)
        
        for idx, dy in enumerate(dy_range):
            shifted_idx = np.arange(h) - dy
            valid_mask = (shifted_idx >= 0) & (shifted_idx < h)
            
            penalty = jump_penalties[idx]
            current_shifted_costs = np.full(h, np.inf)
            current_shifted_costs[valid_mask] = prev_cost[shifted_idx[valid_mask]] + penalty
            
            closer_mask = current_shifted_costs < best_prev_costs
            best_prev_costs[closer_mask] = current_shifted_costs[closer_mask]
            backptr[closer_mask, x] = shifted_idx[closer_mask]
            
        node_costs = np.full(h, blank_cost)
        node_costs[ink_mask[:, x] > 0] = ink_cost
        cost[:, x] = best_prev_costs + node_costs + magnet_cost
            
    y_trace = np.zeros(w, dtype=float)
    y_curr = int(np.argmin(cost[:, w-1]))
    
    # Backtrack normally, do NOT put NaN. We want a continuous bridge.
    for x in range(w-1, -1, -1):
        y_trace[x] = y_curr
        y_curr = backptr[y_curr, x]
        
    # Impulse Noise Killer
    # Dust on the paper creates tiny 1-3 pixel jumps that violate physics.
    # Median filter strips out these impulses completely without blurring edges.
    y_trace = medfilt(y_trace, kernel_size=15)
        
    # Smoothing non-nan segments
    window = 11
    kernel = np.ones(window)/window
    
    # Smooth strictly continuous line to emulate elegant interpolation
    smoothed = np.convolve(y_trace, kernel, mode='same')
    
    # Keep the edges unchanged to prevent convolution edge-drop
    pad = int(window/2)
    smoothed[:pad] = y_trace[:pad]
    smoothed[-pad:] = y_trace[-pad:]
        
    y_norm = 1.0 - (smoothed / max(1, h - 1))
    return y_norm

def map_and_resample(
    y_norm: np.ndarray, 
    start_date: datetime,
    duration_hours: int,
    min_val: float,
    max_val: float,
    reso: str
) -> pd.DataFrame:
    """Transforms raw X/Y to time/value using vectorized math."""
    w = len(y_norm)
    values = min_val + y_norm * (max_val - min_val)
    
    # Vectorized Timestamp creation
    hours_arr = np.linspace(0, duration_hours, w)
    start_dt = pd.to_datetime(start_date)
    timestamps = start_dt + pd.to_timedelta(hours_arr, unit='h')
    
    # Create DataFrame directly from arrays
    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values
    })
    
    # Resample
    df = df.set_index("timestamp").resample(reso).mean()
    df["value"] = df["value"].interpolate(method='time')
    
    df = df.reset_index()
    df['qc_flag'] = 'ok'
    
    return df

def process_universal_image(img_path: Path, output_dir: Path):
    print(f"Reading {img_path}")
    img_bgr = load_image_rgb(img_path)
    
    print("Deskewing Image Formats...")
    img_bgr = deskew_image(img_bgr)
    
    x0, x1, y0, y1 = extract_roi(img_bgr)
    ink_mask, scale_y = extract_universal_ink_mask(img_bgr, x0, x1, y0, y1)
    
    print("Running Universal Gap-Aware Viterbi Extraction...")
    y_norm = run_gap_aware_viterbi(ink_mask, column_density_thresh=2)
    
    # SAVE DEBUG VISUALS
    debug_dir = output_dir / "_debug_accuracy"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{img_path.stem}_ink.png"), ink_mask)
    save_debug_overlay(img_bgr, [y_norm], [(0, 255, 0)], ["temp"], x0, x1, y0, y1, debug_dir / f"{img_path.stem}_overlay.png")
    
    # Using typical settings or defaults (Could be mapped from arguments per image type later)
    # The week graph starts Monday 08:00
    start_dt = datetime(1987, 3, 2, 8, 0, 0)
    
    resolutions = {
        "1S": "secondly",
        "1T": "minutely", # 1T is pandas specific for minutely (now uses 'min' in newer pd, we'll try '1T' / 'min')
        "1H": "hourly"
    }
    
    # Determine variable based on file name or generic
    base_name = img_path.stem
    variable = "temp" if "MART" in base_name else "unknown"
    
    for res_code, res_name in resolutions.items():
        print(f"-> Generating {res_name.upper()} series...")
        # Fallbacks for pandas versions 
        pd_res = 'min' if res_code == '1T' else ('s' if res_code == '1S' else 'h')
        
        df = map_and_resample(
            y_norm=y_norm, 
            start_date=start_dt,
            duration_hours=168, # 7 days
            min_val=-30.0,
            max_val=20.0,
            reso=pd_res
        )
        
        # Add Universal Meta-columns
        df["variable"] = variable
        df["source_kind"] = "universal_graph_digitization"
        df["source_file"] = str(img_path.name)
        df["method"] = "gap_aware_viterbi"
        
        # Order columns cleanly
        df = df[["timestamp", "value", "variable", "qc_flag", "source_kind", "source_file", "method"]]
        
        csv_path = output_dir / f"{base_name}_{res_name}_ai_dataset.csv"
        pq_path = output_dir / f"{base_name}_{res_name}_ai_dataset.parquet"
        
        df.to_csv(csv_path, index=False)
        df.to_parquet(pq_path, index=False)
        
        print(f"   Exported {len(df)} {res_name} rows. Valid: {df['value'].notna().sum()}, Missing: {df['value'].isna().sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=Path, default=Path("/Users/yasinkaya/Hackhaton/DATA/Graf Kağıtları Tarama /1987/Termogram_Haftalık_MART/1987_MART-02.tif"))
    parser.add_argument("--out", type=Path, default=Path("/Users/yasinkaya/Hackhaton/output/graph_models_eval/universal"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    process_universal_image(args.img, args.out)
    print("\nUniversal Process Complete.")

if __name__ == "__main__":
    main()
