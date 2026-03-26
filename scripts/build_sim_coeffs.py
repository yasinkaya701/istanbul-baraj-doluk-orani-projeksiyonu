#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('/Users/yasinkaya/Hackhaton')
PANEL = ROOT / 'output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_extended.csv'
OUT_JS = ROOT / 'baraj_web/assets/data/sim_coeffs.js'

_df = pd.read_csv(PANEL)
_df['date'] = pd.to_datetime(_df['date'])
_df = _df.sort_values('date')

# Use weighted_total_fill as target
_df = _df.rename(columns={'weighted_total_fill':'fill_pct'})

# Keep needed cols
_df = _df[['date','fill_pct','rain_mm','et0_mm_month','consumption_mean_monthly']].dropna()

_df['delta_fill'] = _df['fill_pct'].diff(1)
_df = _df.dropna()

# Regression: delta_fill = a*rain + b*et0 + c*consumption + intercept
X = _df[['rain_mm','et0_mm_month','consumption_mean_monthly']].values
X = np.column_stack([X, np.ones(len(X))])
Y = _df['delta_fill'].values

coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
a, b, c, intercept = coef.tolist()

payload = {
    'a_rain': a,
    'b_et0': b,
    'c_use': c,
    'intercept': intercept,
    'mean_rain': float(_df['rain_mm'].mean()),
    'mean_et0': float(_df['et0_mm_month'].mean()),
    'mean_use_monthly': float(_df['consumption_mean_monthly'].mean()),
}

OUT_JS.write_text('window.SIM_COEFFS = ' + json.dumps(payload, ensure_ascii=False) + ';', encoding='utf-8')
print('wrote', OUT_JS)
print(payload)
