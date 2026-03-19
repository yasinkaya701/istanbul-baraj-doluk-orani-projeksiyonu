#!/usr/bin/env python3
import pandas as pd
import json
import time
from pathlib import Path

# NOTE: This script is intended to be used by the Antigravity Agent.
# It prepares search queries for the agent to execute and then processes the results.

INPUT_CSV = "/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar.csv"
OUTPUT_JSON = "/Users/yasinkaya/Hackhaton/output/archive/news_proofs.json"
Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)

def generate_queries():
    df = pd.read_csv(INPUT_CSV)
    # Focus on Tier 1 (Kritik/Cok Yuksek) events from the last 20 years for better search results
    df['start_time'] = pd.to_datetime(df['start_time'])
    top_events = df[
        (df['severity_level'].isin(['kritik', 'cok_yuksek'])) & 
        (df['start_time'] >= '2000-01-01')
    ].sort_values('peak_severity_score', ascending=False).head(20)
    
    queries = []
    for _, row in top_events.iterrows():
        date_str = row['start_time'].strftime('%Y-%m-%d')
        var_tr = {
            'precip': 'yağış sel fırtına',
            'temp': 'sıcaklık rekoru sıcak dalgası',
            'humidity': 'nem oranı sis',
            'pressure': 'fırtına hava basıncı'
        }.get(row['variable'], 'meteoroloji olay')
        
        query = f"Türkiye {date_str} {var_tr} haberleri"
        queries.append({
            "event_id": row['event_id'],
            "date": date_str,
            "variable": row['variable'],
            "severity": row['peak_severity_score'],
            "query": query
        })
    return queries

if __name__ == "__main__":
    qs = generate_queries()
    print(json.dumps(qs, indent=2))
