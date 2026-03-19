#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def load_ground_truth(excel_path: Path, target_month: int) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df[df[date_col].dt.month == target_month].copy()
    records = []
    for _, row in df.iterrows():
        base_date = row[date_col]
        for hour in range(1, 25):
            val = row.get(hour, np.nan)
            if pd.notna(val):
                records.append({"timestamp": base_date + timedelta(hours=hour-1), "truth": float(val)})
    return pd.DataFrame(records)

def main():
    root = Path("/Users/yasinkaya/Hackhaton")
    pred_path = root / "output/accuracy_test/1987_MART-02_hourly_ai_dataset.csv"
    gt_path = root / "DATA/data berk/Sayısallaştırılmış Veri/1987_Sıcaklık_Saat Başı.xlsx"
    
    gt = load_ground_truth(gt_path, 3)
    # Get current AI values which were mapped to -15 / 45
    # Value = -15 + y_norm * 60 => y_norm = (Value + 15) / 60
    pred_df = pd.read_csv(pred_path)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    
    merged = pd.merge(gt, pred_df, on="timestamp", how="inner")
    y_norms = (merged["value"] + 15.0) / 60.0
    
    best_mae = 999
    best_range = (-15, 45)
    
    print("Sweeping ranges...")
    for mn in np.arange(-30, 10, 2):
        for mx in np.arange(20, 60, 2):
            test_vals = mn + y_norms * (mx - mn)
            mae = (merged["truth"] - test_vals).abs().mean()
            if mae < best_mae:
                best_mae = mae
                best_range = (mn, mx)
                
    print(f"🏆 Best Range Found: Min {best_range[0]}, Max {best_range[1]} | MAE: {best_mae:.3f}")

if __name__ == "__main__":
    main()
