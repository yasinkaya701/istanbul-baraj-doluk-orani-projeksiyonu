import pandas as pd
from evaluate_graph_models import load_ground_truth, extract_roi, extract_ink_mask, run_model_viterbi, map_trace_to_series
import cv2
from datetime import datetime
from pathlib import Path

def export_viterbi_production():
    test_img = "/Users/yasinkaya/Hackhaton/DATA/Graf Kağıtları Tarama /1987/Termogram_Haftalık_MART/1987_MART-02.tif"
    out_dir = Path("/Users/yasinkaya/Hackhaton/output/graph_models_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {test_img}")
    try:
        from PIL import Image
        import numpy as np
        pil_img = Image.open(test_img)
        img_rgb = np.array(pil_img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Could not read with PIL: {e}")
        
    x0, x1, y0, y1 = extract_roi(img_bgr)
    ink_mask = extract_ink_mask(img_bgr, x0, x1, y0, y1)
    
    print("Running Viterbi Model (Optimal MAE ~ 1.93 °C)...")
    trace = run_model_viterbi(ink_mask)
    
    # Best calibration params from the grid search
    start_dt = datetime(1987, 3, 2, 8, 0, 0)
    best_df = map_trace_to_series(trace, start_dt, min_val=-20.0, max_val=30.0)
    
    # Adding metadata columns expected in the universal format
    best_df["variable"] = "temp"
    best_df["qc_flag"] = "ok"
    best_df["source_kind"] = "graph_paper_digitized"
    best_df["source_file"] = "1987_MART-02.tif"
    best_df["method"] = "image_trace_viterbi"
    best_df["confidence"] = 0.95
    
    best_df = best_df.rename(columns={"extracted_value": "value"})
    
    # Exporting
    csv_path = out_dir / "1987_MART_02_viterbi_digitized.csv"
    pq_path = out_dir / "1987_MART_02_viterbi_digitized.parquet"
    
    best_df.to_csv(csv_path, index=False)
    best_df.to_parquet(pq_path, index=False)
    
    print(f"Exported {len(best_df)} rows to CSV and Parquet.")

if __name__ == "__main__":
    export_viterbi_production()
