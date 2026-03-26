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
DET_PATH = ROOT / 'output' / 'istanbul_reconciled_projection_2040' / 'reconciled_scenario_projection_monthly_2026_2040.csv'
BASE_PROB_DIR = ROOT / 'output' / 'istanbul_hybrid_physics_ensemble_probabilistic_2040'
CALIB_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_ensemble_2040' / 'ensemble_calibration_samples.csv'
CORE_PATH = ROOT / 'output' / 'model_useful_data_bundle' / 'tables' / 'istanbul_model_core_monthly.csv'
OUT_DIR = ROOT / 'output' / 'istanbul_reconciled_probabilistic_projection_2040'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRIMARY_SCENARIOS = ['base', 'wet_mild', 'hot_dry_high_demand', 'management_improvement']
SCENARIO_LABELS = {
    'base': 'Temel',
    'wet_mild': 'Ilık-ıslak',
    'hot_dry_high_demand': 'Sıcak-kurak-yüksek talep',
    'management_improvement': 'Yönetim iyileşme',
}
SCENARIO_COLORS = {
    'base': '#2563eb',
    'wet_mild': '#059669',
    'hot_dry_high_demand': '#dc2626',
    'management_improvement': '#d97706',
}
N_SIMULATIONS = 5000
BLOCK_SIZE = 12
SEED = 42
MIN_MONTH_POOL = 6


def month_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 12 - diff)


def residual_pool_for_month(residual_df: pd.DataFrame, month: int, min_count: int = MIN_MONTH_POOL) -> np.ndarray:
    month_series = residual_df['date'].dt.month.to_numpy(dtype=int)
    mask = month_series == month
    pool = residual_df.loc[mask, 'residual'].to_numpy(dtype=float)
    if pool.size >= min_count:
        return pool
    for radius in [1, 2]:
        mask = np.array([month_distance(int(m), int(month)) <= radius for m in month_series], dtype=bool)
        pool = residual_df.loc[mask, 'residual'].to_numpy(dtype=float)
        if pool.size >= min_count:
            return pool
    return residual_df['residual'].to_numpy(dtype=float)


def sample_seasonal_residual_blocks(residual_df: pd.DataFrame, future_dates: pd.DatetimeIndex, block_size: int, rng: np.random.Generator) -> np.ndarray:
    residual_df = residual_df.sort_values('date').reset_index(drop=True)
    hist_dates = pd.DatetimeIndex(residual_df['date'])
    hist_residuals = residual_df['residual'].to_numpy(dtype=float)
    valid_starts = {}
    for month in range(1, 13):
        starts = []
        for i in range(0, len(residual_df) - block_size + 1):
            if int(hist_dates[i].month) == month:
                starts.append(i)
        valid_starts[month] = starts
    out = []
    pos = 0
    while pos < len(future_dates):
        month = int(future_dates[pos].month)
        take = min(block_size, len(future_dates) - pos)
        starts = valid_starts.get(month, [])
        if starts:
            start = int(rng.choice(starts))
            out.append(hist_residuals[start:start + take])
        else:
            pool = residual_pool_for_month(residual_df, month)
            idx = rng.integers(0, len(pool), size=take)
            out.append(pool[idx])
        pos += take
    return np.concatenate(out)[:len(future_dates)]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    det = pd.read_csv(DET_PATH, parse_dates=['date'])
    det = det[['date', 'scenario', 'pred_fill_reconciled']].rename(columns={'pred_fill_reconciled': 'pred_fill'})
    det = det.sort_values(['scenario', 'date']).reset_index(drop=True)

    calib = pd.read_csv(CALIB_PATH, parse_dates=['date'])
    calib = calib[calib['horizon_months'] == 1].copy().sort_values('date').reset_index(drop=True)
    calib['residual'] = calib['actual_fill'] - calib['pred_fill_ensemble_phys']
    residual_df = calib[['date', 'residual']].copy()

    prob_summary = json.loads((BASE_PROB_DIR / 'probabilistic_summary.json').read_text(encoding='utf-8'))
    spread_scale = float(prob_summary['spread_scale'])
    return det, residual_df, spread_scale


def simulate_probabilistic_paths(det: pd.DataFrame, residual_df: pd.DataFrame, spread_scale: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    quant_rows = []
    yearly_rows = []
    endpoint_rows = []
    for scenario, g in det.groupby('scenario'):
        g = g.sort_values('date').copy()
        dates = pd.DatetimeIndex(g['date'])
        base_path = g['pred_fill'].to_numpy(dtype=float)
        sims = np.empty((N_SIMULATIONS, len(g)), dtype=np.float32)
        for i in range(N_SIMULATIONS):
            noise = spread_scale * sample_seasonal_residual_blocks(residual_df, dates, BLOCK_SIZE, rng)
            sims[i, :] = np.clip(base_path + noise, 0.0, 1.0)
        qs = np.quantile(sims, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
        prob40 = (sims < 0.40).mean(axis=0)
        prob30 = (sims < 0.30).mean(axis=0)
        for idx, date in enumerate(dates):
            quant_rows.append({
                'scenario': scenario,
                'date': date,
                'p10_fill_pct': float(qs[0, idx] * 100.0),
                'p25_fill_pct': float(qs[1, idx] * 100.0),
                'p50_fill_pct': float(qs[2, idx] * 100.0),
                'p75_fill_pct': float(qs[3, idx] * 100.0),
                'p90_fill_pct': float(qs[4, idx] * 100.0),
                'prob_below_40_pct': float(prob40[idx] * 100.0),
                'prob_below_30_pct': float(prob30[idx] * 100.0),
            })
        for year in sorted(dates.year.unique()):
            mask = dates.year == year
            year_sims = sims[:, mask]
            dec_vals = year_sims[:, -1]
            yearly_rows.append({
                'scenario': scenario,
                'year': int(year),
                'prob_any_month_below_40_pct': float(((year_sims < 0.40).any(axis=1)).mean() * 100.0),
                'prob_any_month_below_30_pct': float(((year_sims < 0.30).any(axis=1)).mean() * 100.0),
                'p10_december_fill_pct': float(np.quantile(dec_vals, 0.10) * 100.0),
                'p50_december_fill_pct': float(np.quantile(dec_vals, 0.50) * 100.0),
                'p90_december_fill_pct': float(np.quantile(dec_vals, 0.90) * 100.0),
            })
        endpoint = sims[:, -1]
        endpoint_rows.append({
            'scenario': scenario,
            'p10_endpoint_2040_12_pct': float(np.quantile(endpoint, 0.10) * 100.0),
            'p50_endpoint_2040_12_pct': float(np.quantile(endpoint, 0.50) * 100.0),
            'p90_endpoint_2040_12_pct': float(np.quantile(endpoint, 0.90) * 100.0),
        })
    return pd.DataFrame(quant_rows), pd.DataFrame(yearly_rows), pd.DataFrame(endpoint_rows)


def build_summary(yearly_df: pd.DataFrame, endpoint_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in PRIMARY_SCENARIOS:
        ep = endpoint_df[endpoint_df['scenario'] == scenario].iloc[0]
        yr = yearly_df[yearly_df['scenario'] == scenario].copy()
        row = {
            'scenario': scenario,
            'p10_endpoint_2040_12_pct': float(ep['p10_endpoint_2040_12_pct']),
            'p50_endpoint_2040_12_pct': float(ep['p50_endpoint_2040_12_pct']),
            'p90_endpoint_2040_12_pct': float(ep['p90_endpoint_2040_12_pct']),
        }
        for th in [40, 30]:
            hit = yr[yr[f'prob_any_month_below_{th}_pct'] >= 50.0]
            row[f'first_year_prob_below_{th}_ge_50pct'] = int(hit.iloc[0]['year']) if not hit.empty else ''
        rows.append(row)
    return pd.DataFrame(rows).sort_values('p50_endpoint_2040_12_pct', ascending=False).reset_index(drop=True)


def compare_with_base(reconciled_summary: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASE_PROB_DIR / 'probabilistic_projection_summary_2026_2040.csv')
    merged = reconciled_summary.merge(base, on='scenario', suffixes=('_reconciled', '_baseprob'))
    merged['delta_p50_endpoint_pp'] = merged['p50_endpoint_2040_12_pct_reconciled'] - merged['p50_endpoint_2040_12_pct_baseprob']
    merged['delta_p10_endpoint_pp'] = merged['p10_endpoint_2040_12_pct_reconciled'] - merged['p10_endpoint_2040_12_pct_baseprob']
    merged['delta_p90_endpoint_pp'] = merged['p90_endpoint_2040_12_pct_reconciled'] - merged['p90_endpoint_2040_12_pct_baseprob']
    return merged


def plot_base_compare(quant_df: pd.DataFrame, out_path: Path) -> None:
    base_old = pd.read_csv(BASE_PROB_DIR / 'probabilistic_monthly_quantiles_2026_2040.csv', parse_dates=['date'])
    g_new = quant_df[quant_df['scenario'] == 'base'].copy()
    g_old = base_old[base_old['scenario'] == 'base'].copy()
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=170)
    ax.fill_between(g_old['date'], g_old['p10_fill_pct'], g_old['p90_fill_pct'], color='#93c5fd', alpha=0.18)
    ax.plot(g_old['date'], g_old['p50_fill_pct'], color='#2563eb', linewidth=1.8, label='Eski P50')
    ax.fill_between(g_new['date'], g_new['p10_fill_pct'], g_new['p90_fill_pct'], color='#86efac', alpha=0.18)
    ax.plot(g_new['date'], g_new['p50_fill_pct'], color='#059669', linewidth=2.0, label='Uzlaştırılmış P50')
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('Temel senaryoda eski ve uzlaştırılmış olasılıksal yol')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fans(quant_df: pd.DataFrame, out_path: Path) -> None:
    hist = pd.read_csv(CORE_PATH, parse_dates=['date'])
    hist = hist[hist['date'] >= '2018-01-01'].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), dpi=170, sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, scenario in zip(axes, PRIMARY_SCENARIOS):
        q = quant_df[quant_df['scenario'] == scenario].copy()
        ax.plot(hist['date'], hist['weighted_total_fill'] * 100.0, color='#111827', linewidth=1.8)
        ax.fill_between(q['date'], q['p10_fill_pct'], q['p90_fill_pct'], color=SCENARIO_COLORS[scenario], alpha=0.16)
        ax.fill_between(q['date'], q['p25_fill_pct'], q['p75_fill_pct'], color=SCENARIO_COLORS[scenario], alpha=0.25)
        ax.plot(q['date'], q['p50_fill_pct'], color=SCENARIO_COLORS[scenario], linewidth=2.0)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.grid(True, axis='y', alpha=0.22)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel('Toplam doluluk (%)')
    axes[2].set_ylabel('Toplam doluluk (%)')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    det, residual_df, spread_scale = load_inputs()
    quant_df, yearly_df, endpoint_df = simulate_probabilistic_paths(det, residual_df, spread_scale)
    summary_df = build_summary(yearly_df, endpoint_df)
    compare_df = compare_with_base(summary_df)

    quant_df.to_csv(OUT_DIR / 'reconciled_probabilistic_monthly_quantiles_2026_2040.csv', index=False)
    yearly_df.to_csv(OUT_DIR / 'reconciled_probabilistic_yearly_risk_2026_2040.csv', index=False)
    endpoint_df.to_csv(OUT_DIR / 'reconciled_probabilistic_endpoint_summary_2040.csv', index=False)
    summary_df.to_csv(OUT_DIR / 'reconciled_probabilistic_projection_summary_2026_2040.csv', index=False)
    compare_df.to_csv(OUT_DIR / 'reconciled_vs_baseprob_summary_2026_2040.csv', index=False)

    plot_base_compare(quant_df, OUT_DIR / 'reconciled_vs_baseprob_base.png')
    plot_fans(quant_df, OUT_DIR / 'reconciled_probabilistic_fan_paths_2026_2040.png')

    meta = {
        'inherits_probabilistic_calibration_from': str(BASE_PROB_DIR / 'probabilistic_summary.json'),
        'spread_scale': spread_scale,
        'n_simulations': N_SIMULATIONS,
        'block_size_months': BLOCK_SIZE,
        'seed': SEED,
    }
    (OUT_DIR / 'reconciled_probabilistic_summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT_DIR)

if __name__ == '__main__':
    main()
