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
CALIB_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_ensemble_2040' / 'ensemble_calibration_samples.csv'
PROB_SUMMARY_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_ensemble_probabilistic_2040' / 'probabilistic_summary.json'
PREFERRED_DIR = ROOT / 'output' / 'istanbul_preferred_projection_2040'
OUT_DIR = ROOT / 'output' / 'istanbul_preferred_operational_risk_2030'
OUT_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [50, 40, 30, 20]
N_SIMULATIONS = 5000
BLOCK_SIZE = 12
SEED = 42
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
SUMMER_MONTHS = [6, 7, 8, 9]


def month_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 12 - diff)


def residual_pool_for_month(residual_df: pd.DataFrame, month: int, min_count: int = 6) -> np.ndarray:
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


def longest_run(mask: np.ndarray) -> np.ndarray:
    # mask shape: (n_sims, n_months)
    n_sims, n_months = mask.shape
    out = np.zeros(n_sims, dtype=int)
    for i in range(n_sims):
        best = 0
        cur = 0
        row = mask[i]
        for j in range(n_months):
            if row[j]:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        out[i] = best
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    det = pd.read_csv(DET_PATH, parse_dates=['date'])
    det = det[['date', 'scenario', 'pred_fill_reconciled']].rename(columns={'pred_fill_reconciled': 'pred_fill'})
    det = det.sort_values(['scenario', 'date']).reset_index(drop=True)

    calib = pd.read_csv(CALIB_PATH, parse_dates=['date'])
    calib = calib[calib['horizon_months'] == 1].copy().sort_values('date').reset_index(drop=True)
    calib['residual'] = calib['actual_fill'] - calib['pred_fill_ensemble_phys']
    residual_df = calib[['date', 'residual']].copy()

    spread_scale = float(json.loads(PROB_SUMMARY_PATH.read_text(encoding='utf-8'))['spread_scale'])
    return det, residual_df, spread_scale


def build_simulations(det: pd.DataFrame, residual_df: pd.DataFrame, spread_scale: float) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    out = {}
    for scenario, g in det.groupby('scenario'):
        g = g.sort_values('date').copy()
        dates = pd.DatetimeIndex(g['date'])
        base_path = g['pred_fill'].to_numpy(dtype=float)
        sims = np.empty((N_SIMULATIONS, len(g)), dtype=np.float32)
        for i in range(N_SIMULATIONS):
            noise = spread_scale * sample_seasonal_residual_blocks(residual_df, dates, BLOCK_SIZE, rng)
            sims[i, :] = np.clip(base_path + noise, 0.0, 1.0)
        out[scenario] = (dates, base_path, sims)
    return out


def monthly_threshold_tables(sim_map: dict[str, tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for scenario in PRIMARY_SCENARIOS:
        dates, base_path, sims = sim_map[scenario]
        for i, date in enumerate(dates):
            row = {
                'scenario': scenario,
                'date': date,
                'deterministic_fill_pct': float(base_path[i] * 100.0),
                'p10_fill_pct': float(np.quantile(sims[:, i], 0.10) * 100.0),
                'p50_fill_pct': float(np.quantile(sims[:, i], 0.50) * 100.0),
                'p90_fill_pct': float(np.quantile(sims[:, i], 0.90) * 100.0),
            }
            for th in THRESHOLDS:
                row[f'prob_below_{th}_pct'] = float((sims[:, i] < (th / 100.0)).mean() * 100.0)
            rows.append(row)
    return pd.DataFrame(rows)


def yearly_consecutive_tables(sim_map: dict[str, tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for scenario in PRIMARY_SCENARIOS:
        dates, _, sims = sim_map[scenario]
        years = sorted(set(dates.year))
        for year in years:
            mask_year = dates.year == year
            year_sims = sims[:, mask_year]
            summer_mask = np.isin(dates[mask_year].month, SUMMER_MONTHS)
            summer_sims = year_sims[:, summer_mask]
            row = {'scenario': scenario, 'year': int(year)}
            for th in [40, 30]:
                below = year_sims < (th / 100.0)
                below_summer = summer_sims < (th / 100.0)
                row[f'prob_any_month_below_{th}_pct'] = float(below.any(axis=1).mean() * 100.0)
                row[f'prob_3consec_below_{th}_pct'] = float((longest_run(below) >= 3).mean() * 100.0)
                row[f'prob_6consec_below_{th}_pct'] = float((longest_run(below) >= 6).mean() * 100.0)
                row[f'prob_any_summer_month_below_{th}_pct'] = float(below_summer.any(axis=1).mean() * 100.0) if summer_sims.shape[1] else 0.0
            if summer_sims.shape[1]:
                summer_mean = summer_sims.mean(axis=1)
                row['p10_summer_mean_fill_pct'] = float(np.quantile(summer_mean, 0.10) * 100.0)
                row['p50_summer_mean_fill_pct'] = float(np.quantile(summer_mean, 0.50) * 100.0)
                row['p90_summer_mean_fill_pct'] = float(np.quantile(summer_mean, 0.90) * 100.0)
            dec_vals = year_sims[:, -1]
            row['p10_december_fill_pct'] = float(np.quantile(dec_vals, 0.10) * 100.0)
            row['p50_december_fill_pct'] = float(np.quantile(dec_vals, 0.50) * 100.0)
            row['p90_december_fill_pct'] = float(np.quantile(dec_vals, 0.90) * 100.0)
            rows.append(row)
    return pd.DataFrame(rows)


def build_nearterm_summary(monthly_df: pd.DataFrame, yearly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    near_monthly = monthly_df[monthly_df['date'] <= '2030-12-01'].copy()
    near_yearly = yearly_df[yearly_df['year'] <= 2030].copy()
    for scenario in PRIMARY_SCENARIOS:
        m = near_monthly[near_monthly['scenario'] == scenario].copy()
        y = near_yearly[near_yearly['scenario'] == scenario].copy()
        row = {'scenario': scenario}
        for th in THRESHOLDS:
            row[f'max_monthly_prob_below_{th}_2026_2030_pct'] = float(m[f'prob_below_{th}_pct'].max())
            peak = m.loc[m[f'prob_below_{th}_pct'].idxmax(), 'date'] if not m.empty else pd.NaT
            row[f'peak_month_prob_below_{th}_date'] = peak.strftime('%Y-%m') if pd.notna(peak) else ''
        for th in [40, 30]:
            hit3 = y[y[f'prob_3consec_below_{th}_pct'] >= 50.0]
            row[f'first_year_prob_3consec_below_{th}_ge_50pct'] = int(hit3.iloc[0]['year']) if not hit3.empty else ''
            hit6 = y[y[f'prob_6consec_below_{th}_pct'] >= 50.0]
            row[f'first_year_prob_6consec_below_{th}_ge_50pct'] = int(hit6.iloc[0]['year']) if not hit6.empty else ''
        summer = y[['year', 'p50_summer_mean_fill_pct']]
        if not summer.empty:
            min_row = summer.loc[summer['p50_summer_mean_fill_pct'].idxmin()]
            row['lowest_p50_summer_year_2026_2030'] = int(min_row['year'])
            row['lowest_p50_summer_fill_pct_2026_2030'] = float(min_row['p50_summer_mean_fill_pct'])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_monthly_risk(monthly_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), dpi=170, sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, scenario in zip(axes, PRIMARY_SCENARIOS):
        g = monthly_df[(monthly_df['scenario'] == scenario) & (monthly_df['date'] <= '2030-12-01')].copy()
        ax.plot(g['date'], g['prob_below_40_pct'], color='#b45309', linewidth=2.0, label='%40 altı olasılık')
        ax.plot(g['date'], g['prob_below_30_pct'], color='#dc2626', linewidth=2.0, label='%30 altı olasılık')
        ax.plot(g['date'], g['prob_below_20_pct'], color='#7f1d1d', linewidth=1.8, label='%20 altı olasılık')
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.grid(True, axis='y', alpha=0.22)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel('Olasılık (%)')
    axes[2].set_ylabel('Olasılık (%)')
    axes[0].legend(frameon=False, loc='upper left')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_consecutive_risk(yearly_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=170, sharey=True)
    for ax, th in zip(axes, [40, 30]):
        for scenario in PRIMARY_SCENARIOS:
            g = yearly_df[(yearly_df['scenario'] == scenario) & (yearly_df['year'] <= 2030)].copy()
            ax.plot(g['year'], g[f'prob_3consec_below_{th}_pct'], color=SCENARIO_COLORS[scenario], linewidth=2.0, label=SCENARIO_LABELS[scenario])
        ax.set_title(f'En az 3 ay arka arkaya %{th} altı')
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', alpha=0.22)
        ax.set_xlabel('Yıl')
    axes[0].set_ylabel('Olasılık (%)')
    axes[0].legend(frameon=False, loc='upper left')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def update_preferred_manifest() -> None:
    manifest_path = PREFERRED_DIR / 'PREFERRED_PACKAGE.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    manifest['preferred_operational_risk_pack'] = str(OUT_DIR)
    manifest['preferred_operational_risk_files'] = {
        'monthly_threshold_risk': str(OUT_DIR / 'preferred_monthly_threshold_risk_2026_2040.csv'),
        'yearly_consecutive_risk': str(OUT_DIR / 'preferred_yearly_consecutive_risk_2026_2040.csv'),
        'nearterm_summary': str(OUT_DIR / 'preferred_nearterm_risk_summary_2026_2030.csv'),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    det, residual_df, spread_scale = load_inputs()
    sim_map = build_simulations(det, residual_df, spread_scale)
    monthly_df = monthly_threshold_tables(sim_map)
    yearly_df = yearly_consecutive_tables(sim_map)
    nearterm_df = build_nearterm_summary(monthly_df, yearly_df)

    monthly_df.to_csv(OUT_DIR / 'preferred_monthly_threshold_risk_2026_2040.csv', index=False)
    yearly_df.to_csv(OUT_DIR / 'preferred_yearly_consecutive_risk_2026_2040.csv', index=False)
    nearterm_df.to_csv(OUT_DIR / 'preferred_nearterm_risk_summary_2026_2030.csv', index=False)

    plot_monthly_risk(monthly_df, OUT_DIR / 'preferred_monthly_threshold_risk_2026_2030.png')
    plot_consecutive_risk(yearly_df, OUT_DIR / 'preferred_consecutive_risk_2026_2030.png')

    summary = {
        'deterministic_source': str(DET_PATH),
        'spread_scale': spread_scale,
        'n_simulations': N_SIMULATIONS,
        'block_size_months': BLOCK_SIZE,
        'seed': SEED,
        'scenarios': PRIMARY_SCENARIOS,
    }
    (OUT_DIR / 'preferred_operational_risk_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    update_preferred_manifest()


if __name__ == '__main__':
    main()
