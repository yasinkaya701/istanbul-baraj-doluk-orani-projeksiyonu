#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('/Users/yasinkaya/Hackhaton')
FORECAST_PATH = ROOT / 'output' / 'istanbul_dam_forecast' / 'istanbul_dam_forecasts_monthly.csv'
CURRENT_PATH = ROOT / 'output' / 'model_useful_data_bundle' / 'tables' / 'istanbul_source_current_context.csv'
TOPDOWN_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_sourceaware_ensemble_2040' / 'ensemble_phys_scenario_projection_monthly_2026_2040.csv'
OUT_DIR = ROOT / 'output' / 'nearterm_bottom_up_reconciliation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP = {
    'Omerli': 'Ömerli Barajı',
    'Darlik': 'Darlık Barajı',
    'Elmali': 'Elmalı 1 ve 2 Barajları',
    'Terkos': 'Terkos Barajı',
    'Alibey': 'Alibey Barajı',
    'Buyukcekmece': 'Büyükçekmece Barajı',
    'Sazlidere': 'Sazlıdere Barajı',
    'Kazandere': 'Kazandere Barajı',
    'Pabucdere': 'Pabuçdere Barajı',
    'Istrancalar': 'Istrancalar',
}


def main() -> None:
    f = pd.read_csv(FORECAST_PATH, parse_dates=['ds'])
    c = pd.read_csv(CURRENT_PATH)
    cap = c[['source_name','biriktirmeHacmi']].copy()
    cap['series'] = cap['source_name'].map({v:k for k,v in MAP.items()})
    cap = cap.dropna(subset=['series'])[['series','biriktirmeHacmi']]

    base = f[f['series'].isin(cap['series'])].copy()
    base = base.merge(cap, on='series', how='left')
    base['weighted_storage_pred'] = base['yhat'] * base['biriktirmeHacmi']
    agg = base.groupby('ds', as_index=False).agg(
        weighted_storage_pred=('weighted_storage_pred','sum'),
        total_capacity=('biriktirmeHacmi','sum'),
    )
    agg['bottom_up_fill_pct'] = agg['weighted_storage_pred'] / agg['total_capacity'] * 100.0

    overall = f[f['series']=='overall_mean'][['ds','yhat']].copy()
    overall['overall_mean_fill_pct'] = overall['yhat'] * 100.0
    agg = agg.merge(overall, on='ds', how='left')

    td = pd.read_csv(TOPDOWN_PATH, parse_dates=['date'])
    td = td[td['scenario']=='base'][['date','pred_fill_ensemble']].rename(columns={'date':'ds'})
    td['topdown_base_fill_pct'] = td['pred_fill_ensemble'] * 100.0
    agg = agg.merge(td, on='ds', how='left')

    agg['bottom_up_minus_topdown_pp'] = agg['bottom_up_fill_pct'] - agg['topdown_base_fill_pct']
    agg['bottom_up_minus_overall_mean_pp'] = agg['bottom_up_fill_pct'] - agg['overall_mean_fill_pct']
    agg.to_csv(OUT_DIR / 'nearterm_bottom_up_vs_topdown_2024_2027.csv', index=False)

    summary = {
        'start': str(agg['ds'].min().date()),
        'end': str(agg['ds'].max().date()),
        'mean_bottom_up_fill_pct': float(agg['bottom_up_fill_pct'].mean()),
        'mean_topdown_base_fill_pct': float(agg['topdown_base_fill_pct'].mean()),
        'mean_gap_bottom_up_minus_topdown_pp': float(agg['bottom_up_minus_topdown_pp'].mean()),
        'max_gap_bottom_up_minus_topdown_pp': float(agg['bottom_up_minus_topdown_pp'].max()),
        'min_gap_bottom_up_minus_topdown_pp': float(agg['bottom_up_minus_topdown_pp'].min()),
        'end_bottom_up_fill_pct': float(agg.iloc[-1]['bottom_up_fill_pct']),
        'end_topdown_base_fill_pct': float(agg.iloc[-1]['topdown_base_fill_pct']),
    }
    (OUT_DIR / 'nearterm_bottom_up_reconciliation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=170)
    ax.plot(agg['ds'], agg['bottom_up_fill_pct'], label='Baraj bazlı ağırlıklı alt-toplam', color='#111827', linewidth=2.2)
    ax.plot(agg['ds'], agg['topdown_base_fill_pct'], label='Top-down temel senaryo', color='#2563eb', linewidth=1.9)
    ax.plot(agg['ds'], agg['overall_mean_fill_pct'], label='Baraj bazlı eşit ortalama', color='#d97706', linewidth=1.6, linestyle='--')
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('2024-2027 yakın vadede bottom-up ve top-down karşılaştırması')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'nearterm_bottom_up_vs_topdown.png')
    plt.close(fig)

    print(OUT_DIR)

if __name__ == '__main__':
    main()
