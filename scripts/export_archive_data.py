#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path

INPUT_CSV = "/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar.csv"
OUTPUT_JSON = "/Users/yasinkaya/Hackhaton/output/archive/events_archive.json"

def export():
    df = pd.read_csv(INPUT_CSV)
    # Filter for last 25 years and get more points for the timeline
    df['start_time'] = pd.to_datetime(df['start_time'])
    timeline_df = df[df['start_time'] >= '2000-01-01'].sort_values('start_time')
    
    data = timeline_df.to_dict(orient='records')
    
    with open(OUTPUT_JSON, 'w') as f:
        # Convert date to string for JSON
        json.dump(data, f, indent=2, default=str)
    print(f"Exported {len(data)} events to {OUTPUT_JSON}")

if __name__ == "__main__":
    export()
