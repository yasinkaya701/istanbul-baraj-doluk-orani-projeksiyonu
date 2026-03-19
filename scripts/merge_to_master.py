#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import sys
import tqdm

def aggregate_universal_datalake(root_dir: Path, output_root: Path):
    """
    Scans the entire output structure and creates unified master files 
    for each variable and resolution.
    """
    if not root_dir.exists():
        print(f"Error: {root_dir} not found.")
        return

    # Discovery
    variables = [d.name for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    
    for var in variables:
        var_path = root_dir / var
        resolutions = [d.name for d in var_path.iterdir() if d.is_dir()]
        
        for res in resolutions:
            target_dir = var_path / res
            files = sorted(list(target_dir.glob("*.parquet")))
            
            if not files:
                continue
                
            print(f"Aggregating {var} @ {res} ({len(files)} files)...")
            
            # Use chunks or combined read for large datasets
            dfs = []
            for f in tqdm.tqdm(files, desc=f"Reading {var}/{res}"):
                dfs.append(pd.read_parquet(f))
            
            master_df = pd.concat(dfs, ignore_index=True)
            master_df = master_df.sort_values("timestamp").drop_duplicates("timestamp")
            
            # Statistics
            start = master_df["timestamp"].min()
            end = master_df["timestamp"].max()
            rows = len(master_df)
            
            out_file_base = output_root / f"MASTER_{var.upper()}_{res}"
            output_root.mkdir(parents=True, exist_ok=True)
            
            # Save in multiple formats for different AI needs
            master_df.to_parquet(out_file_base.with_suffix(".parquet"), index=False, compression="snappy")
            # Only save CSV for lower resolutions (1H) or smaller samples to save disk
            if res == "1H" or rows < 500000:
                master_df.to_csv(out_file_base.with_suffix(".csv"), index=False)
            
            print(f"DONE: {var}/{res} | {start} to {end} | {rows} rows.")
            print(f"SAVED: {out_file_base}.parquet\n")

if __name__ == "__main__":
    root = Path("/Users/yasinkaya/Hackhaton/output/universal_datasets")
    out = Path("/Users/yasinkaya/Hackhaton/output/MASTER_DATASETS")
    aggregate_universal_datalake(root, out)
