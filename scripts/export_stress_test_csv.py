#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path

def generate_stress_test_csv(report_json: Path, output_csv: Path):
    if not report_json.exists():
        print(f"Error: {report_json} not found.")
        return

    with open(report_json, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    df = pd.DataFrame(results)
    
    # Rename columns for clarity
    df = df.rename(columns={
        "file": "File Name",
        "status": "Final Status",
        "message": "Reason/Detail",
        "variables": "Detected Variables"
    })

    # Clean up variables list for CSV
    if "Detected Variables" in df.columns:
        df["Detected Variables"] = df["Detected Variables"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")

    df.to_csv(output_csv, index=False)
    print(f"✅ Stress Test Summary exported to: {output_csv}")
    print(df[["File Name", "Final Status", "Reason/Detail"]].to_string())

if __name__ == "__main__":
    report = Path("output/universal_datasets/batch_report.json")
    out = Path("output/STRESS_TEST_SUMMARY.csv")
    generate_stress_test_csv(report, out)
