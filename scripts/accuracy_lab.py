#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def load_ground_truth(excel_path: Path, target_month: int) -> pd.DataFrame:
    """Load manual ground truth for a specific month from the Sayısallaştırılmış Veri Excel"""
    df = pd.read_excel(excel_path)
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
                dt_hour = base_date + timedelta(hours=hour-1)
                records.append({
                    "timestamp": dt_hour,
                    "truth_value": float(val)
                })
    return pd.DataFrame(records)

def evaluate_accuracy(gt_df: pd.DataFrame, pred_df: pd.DataFrame):
    # Ensure pred_df has timestamp column
    if "timestamp" not in pred_df.columns:
        if "time" in pred_df.columns:
            pred_df = pred_df.rename(columns={"time": "timestamp"})
    
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    
    # Merge on timestamp
    merged = pd.merge(gt_df, pred_df, on="timestamp", how="inner")
    
    if merged.empty:
        return None
        
    merged['error'] = merged['truth_value'] - merged['value']
    mae = merged['error'].abs().mean()
    rmse = np.sqrt((merged['error']**2).mean())
    bias = merged['error'].mean()
    
    # MAPE (Mean Absolute Percentage Error)
    # Using 0.1 threshold to avoid division by zero
    mape = (merged['error'].abs() / merged['truth_value'].abs().replace(0, 0.1)).mean() * 100
    
    # NMAE (Normalized Mean Absolute Error) - Percentage of the range
    truth_range = merged['truth_value'].max() - merged['truth_value'].min()
    nmae = (mae / truth_range * 100) if truth_range > 0 else 0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape,
        "nmae": nmae,
        "count": len(merged)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default=Path("/Users/yasinkaya/Hackhaton/DATA/data berk/Sayısallaştırılmış Veri/1987_Sıcaklık_Saat Başı.xlsx"))
    parser.add_argument("--pred", type=Path, help="Path to AI-generated CSV or Parquet")
    parser.add_argument("--month", type=int, default=3)
    args = parser.parse_args()
    
    if not args.pred:
        print("Please provide --pred path.")
        return
        
    print(f"Loading Ground Truth (Month {args.month})...")
    gt = load_ground_truth(args.gt, args.month)
    
    print(f"Loading AI Predictions from {args.pred.name}...")
    if args.pred.suffix == ".parquet":
        pred = pd.read_parquet(args.pred)
    else:
        pred = pd.read_csv(args.pred)
        
    stats = evaluate_accuracy(gt, pred)
    
    if stats:
        print("\n📊 Accuracy Report:")
        print(f"Comparison Points: {stats['count']}")
        print(f"MAE (Mean Absolute Error): {stats['mae']:.3f}")
        print(f"RMSE (Root Mean Square Error): {stats['rmse']:.3f}")
        print(f"Bias (Mean Error): {stats['bias']:.3f}")
        print(f"MAPE (Mean Abs Percentage Error): {stats['mape']:.2f}%")
        print(f"NMAE (Normalized MAE / Range): {stats['nmae']:.2f}%")
    else:
        print("❌ Data mismatch. Please ensure timestamps and variables align.")

if __name__ == "__main__":
    main()
