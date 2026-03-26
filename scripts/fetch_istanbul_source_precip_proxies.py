#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path('/Users/yasinkaya/Hackhaton')
OUT_DIR = ROOT / 'output' / 'source_precip_proxies'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {'source': 'Omerli', 'lat': 41.0607514, 'lon': 29.3571826, 'method': 'nominatim'},
    {'source': 'Darlik', 'lat': 41.1155742, 'lon': 29.5683502, 'method': 'nominatim'},
    {'source': 'Elmali', 'lat': 41.0746897, 'lon': 29.1032266, 'method': 'nominatim'},
    {'source': 'Terkos', 'lat': 41.3363679, 'lon': 28.6179291, 'method': 'nominatim'},
    {'source': 'Alibey', 'lat': 41.1006085, 'lon': 28.9208551, 'method': 'nominatim'},
    {'source': 'Buyukcekmece', 'lat': 41.0248506, 'lon': 28.5729016, 'method': 'nominatim'},
    {'source': 'Sazlidere', 'lat': 41.1059148, 'lon': 28.7185740, 'method': 'nominatim'},
    {'source': 'Kazandere', 'lat': 41.6160468, 'lon': 28.0733846, 'method': 'nominatim'},
    {'source': 'Pabucdere', 'lat': 41.6397126, 'lon': 28.0627980, 'method': 'nominatim'},
    {'source': 'Istrancalar', 'lat': 41.5837425, 'lon': 28.1182905, 'method': 'centroid_from_sub-reservoir geocodes'},
]

START = '2000-01-01'
END = '2026-03-12'
URL = 'https://archive-api.open-meteo.com/v1/archive'


def fetch_one(src: dict) -> pd.DataFrame:
    params = {
        'latitude': src['lat'],
        'longitude': src['lon'],
        'start_date': START,
        'end_date': END,
        'daily': 'precipitation_sum',
        'timezone': 'Europe/Istanbul',
    }
    last_err = None
    for attempt in range(6):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            break
        last_err = f"429 for {src['source']} attempt={attempt + 1}"
        time.sleep(2 + 2 * attempt)
    else:
        raise RuntimeError(last_err or f"Failed for {src['source']}")
    data = r.json()
    daily = pd.DataFrame(data['daily'])
    daily['date'] = pd.to_datetime(daily['time'])
    daily['precipitation_sum'] = pd.to_numeric(daily['precipitation_sum'], errors='coerce').fillna(0.0)
    daily['source'] = src['source']
    daily['latitude'] = float(data.get('latitude', src['lat']))
    daily['longitude'] = float(data.get('longitude', src['lon']))
    daily['grid_elevation_m'] = data.get('elevation')
    return daily[['date', 'source', 'latitude', 'longitude', 'grid_elevation_m', 'precipitation_sum']]


def main() -> None:
    daily_parts = []
    failed_sources = []
    for src in SOURCES:
        try:
            daily_parts.append(fetch_one(src))
        except Exception:
            failed_sources.append(src['source'])
    daily = pd.concat(daily_parts, ignore_index=True)

    if 'Istrancalar' in failed_sources and {'Kazandere', 'Pabucdere'}.issubset(set(daily['source'].unique())):
        fallback = (
            daily[daily['source'].isin(['Kazandere', 'Pabucdere'])]
            .groupby('date', as_index=False)
            .agg(
                latitude=('latitude', 'mean'),
                longitude=('longitude', 'mean'),
                grid_elevation_m=('grid_elevation_m', 'mean'),
                precipitation_sum=('precipitation_sum', 'mean'),
            )
        )
        fallback['source'] = 'Istrancalar'
        daily = pd.concat([daily, fallback[['date', 'source', 'latitude', 'longitude', 'grid_elevation_m', 'precipitation_sum']]], ignore_index=True)

    monthly = (
        daily.assign(month=lambda x: x['date'].values.astype('datetime64[M]'))
        .groupby(['source', 'month', 'latitude', 'longitude', 'grid_elevation_m'], as_index=False)['precipitation_sum']
        .sum()
        .rename(columns={'month': 'date', 'precipitation_sum': 'rain_mm_month'})
    )
    wide = monthly.pivot(index='date', columns='source', values='rain_mm_month').reset_index().sort_values('date')
    wide.columns.name = None

    meta = pd.DataFrame(SOURCES)
    meta.to_csv(OUT_DIR / 'source_precip_proxy_metadata.csv', index=False)
    daily.to_csv(OUT_DIR / 'source_precip_daily_2000_2026.csv', index=False)
    monthly.to_csv(OUT_DIR / 'source_precip_monthly_long_2000_2026.csv', index=False)
    wide.to_csv(OUT_DIR / 'source_precip_monthly_wide_2000_2026.csv', index=False)
    (OUT_DIR / 'source_precip_summary.json').write_text(json.dumps({
        'n_sources': len(SOURCES),
        'start': START,
        'end': END,
        'rows_daily': int(len(daily)),
        'rows_monthly_long': int(len(monthly)),
        'rows_monthly_wide': int(len(wide)),
        'failed_sources': failed_sources,
        'source_methods': {s['source']: s['method'] for s in SOURCES},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT_DIR)


if __name__ == '__main__':
    main()
