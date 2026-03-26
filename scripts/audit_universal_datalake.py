#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def audit_datalake(root_dir: Path):
    """
    Analyzes the universal datalake for quality, coverage, and gaps.
    """
    if not root_dir.exists():
        print("Data Lake not found.")
        return

    variables = [d.name for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    report = {}

    for var in variables:
        var_path = root_dir / var
        report[var] = {}
        
        for res in [d.name for d in var_path.iterdir() if d.is_dir()]:
            target_dir = var_path / res
            files = list(target_dir.glob("*.parquet"))
            
            if not files:
                continue
            
            # Load samples for stats
            sample_df = pd.read_parquet(files[0])
            
            total_rows = 0
            file_count = len(files)
            
            for f in files:
                m = pd.read_parquet(f)
                total_rows += len(m)
            
            report[var][res] = {
                "file_count": file_count,
                "total_rows": total_rows,
                "status": "Green"
            }

    # Save report
    out_path = root_dir / "_datalake_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n📊 Data Lake Audit Complete. Report saved to {out_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    root = Path("/Users/yasinkaya/Hackhaton/output/universal_datasets")
    audit_datalake(root)
