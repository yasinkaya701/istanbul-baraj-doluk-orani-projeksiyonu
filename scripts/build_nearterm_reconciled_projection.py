#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/Users/yasinkaya/Hackhaton')
TOPDOWN_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_sourceaware_ensemble_2040' / 'ensemble_phys_scenario_projection_monthly_2026_2040.csv'
BOTTOMUP_PATH = ROOT / 'output' / 'nearterm_bottom_up_reconciliation' / 'nearterm_bottom_up_vs_topdown_2024_2027.csv'
OUT_DIR = ROOT / 'output' / 'istanbul_reconciled_projection_2040'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRIMARY_SCENARIOS = ['base','wet_mild','hot_dry_high_demand','management_improvement']
COLORS = {'base':'#2563eb','wet_mild':'#059669','hot_dry_high_demand':'#dc2626','management_improvement':'#d97706'}
LABELS = {'base':'Temel','wet_mild':'Ilık-ıslak','hot_dry_high_demand':'Sıcak-kurak-yüksek talep','management_improvement':'Yönetim iyileşme'}


def build_correction_curve(td: pd.DataFrame, bu: pd.DataFrame) -> pd.DataFrame:
    ov = bu.dropna(subset=['topdown_base_fill_pct']).copy()
    ov = ov[['ds','bottom_up_minus_topdown_pp']].rename(columns={'ds':'date','bottom_up_minus_topdown_pp':'correction_pp'})
    full_dates = pd.DataFrame({'date': pd.date_range('2026-01-01','2040-12-01',freq='MS')})
    corr = full_dates.merge(ov, on='date', how='left')

    last_anchor_date = pd.Timestamp('2027-02-01')
    last_anchor = float(ov.loc[ov['date']==last_anchor_date, 'correction_pp'].iloc[0])
    decay_end = pd.Timestamp('2030-01-01')
    for i, row in corr.iterrows():
        date = row['date']
        if pd.notna(row['correction_pp']):
            continue
        if date <= decay_end:
            months_from_anchor = (date.year - last_anchor_date.year) * 12 + date.month - last_anchor_date.month
            total_months = (decay_end.year - last_anchor_date.year) * 12 + decay_end.month - last_anchor_date.month
            frac = max(0.0, 1.0 - months_from_anchor / total_months)
            corr.loc[i, 'correction_pp'] = last_anchor * frac
        else:
            corr.loc[i, 'correction_pp'] = 0.0
    return corr


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for scenario,g in df[df['scenario'].isin(PRIMARY_SCENARIOS)].groupby('scenario'):
        g=g.sort_values('date')
        rows.append({
            'scenario': scenario,
            'mean_fill_2026_2040_pct': float(g['pred_fill_reconciled_pct'].mean()),
            'min_fill_2026_2040_pct': float(g['pred_fill_reconciled_pct'].min()),
            'end_fill_2040_12_pct': float(g.iloc[-1]['pred_fill_reconciled_pct']),
            'first_below_40_date': str(g.loc[g['pred_fill_reconciled_pct'] < 40.0, 'date'].iloc[0].date()) if (g['pred_fill_reconciled_pct'] < 40.0).any() else '',
            'first_below_30_date': str(g.loc[g['pred_fill_reconciled_pct'] < 30.0, 'date'].iloc[0].date()) if (g['pred_fill_reconciled_pct'] < 30.0).any() else '',
        })
    return pd.DataFrame(rows).sort_values('mean_fill_2026_2040_pct', ascending=False).reset_index(drop=True)


def main() -> None:
    td = pd.read_csv(TOPDOWN_PATH, parse_dates=['date'])
    td['pred_fill_pct'] = td['pred_fill_ensemble'] * 100.0
    bu = pd.read_csv(BOTTOMUP_PATH, parse_dates=['ds'])
    corr = build_correction_curve(td, bu)
    out = td.merge(corr, on='date', how='left')
    out['pred_fill_reconciled_pct'] = (out['pred_fill_pct'] + out['correction_pp']).clip(0.0, 100.0)
    out['pred_fill_reconciled'] = out['pred_fill_reconciled_pct'] / 100.0
    out.to_csv(OUT_DIR / 'reconciled_scenario_projection_monthly_2026_2040.csv', index=False)
    corr.to_csv(OUT_DIR / 'nearterm_correction_curve_2026_2040.csv', index=False)
    summary = summarize(out)
    summary.to_csv(OUT_DIR / 'reconciled_scenario_summary_2026_2040.csv', index=False)

    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=170)
    base = out[out['scenario']=='base'].copy()
    ax.plot(base['date'], base['pred_fill_pct'], color='#94a3b8', linewidth=1.8, label='Top-down temel')
    ax.plot(base['date'], base['pred_fill_reconciled_pct'], color='#111827', linewidth=2.2, label='Uzlaştırılmış temel')
    ov = bu.dropna(subset=['topdown_base_fill_pct']).copy()
    ax.plot(ov['ds'], ov['bottom_up_fill_pct'], color='#059669', linewidth=1.8, label='Bottom-up yakın vade')
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('Yakın-vade mutabakat düzeltmesi')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'reconciled_base_projection.png')
    plt.close(fig)

    meta = {
        'overlap_start': str(corr['date'].min().date()),
        'overlap_end': str(corr[corr['correction_pp'] != 0]['date'].max().date()),
        'mean_correction_pp': float(corr['correction_pp'].mean()),
        'min_correction_pp': float(corr['correction_pp'].min()),
        'max_correction_pp': float(corr['correction_pp'].max()),
    }
    (OUT_DIR / 'reconciled_summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT_DIR)

if __name__ == '__main__':
    main()
