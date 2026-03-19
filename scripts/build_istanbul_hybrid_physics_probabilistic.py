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
ENSEMBLE_DIR = ROOT / 'output' / 'istanbul_hybrid_physics_ensemble_2040'
OUT_DIR = ROOT / 'output' / 'istanbul_hybrid_physics_ensemble_probabilistic_2040'
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
MIN_RESIDUAL_HISTORY = 24
MIN_MONTH_POOL = 6
NOMINAL_COVERAGE_P10_P90 = 80.0
SPREAD_GRID = np.round(np.arange(0.80, 1.81, 0.05), 2)


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


def sample_seasonal_residual_blocks(
    residual_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    residual_df = residual_df.sort_values('date').reset_index(drop=True)
    if residual_df.empty:
        return np.zeros(len(future_dates), dtype=float)

    hist_dates = pd.DatetimeIndex(residual_df['date'])
    hist_residuals = residual_df['residual'].to_numpy(dtype=float)
    valid_starts = {}
    for month in range(1, 13):
        starts = []
        for i in range(0, len(residual_df) - block_size + 1):
            if int(hist_dates[i].month) == month:
                starts.append(i)
        valid_starts[month] = starts

    out: list[np.ndarray] = []
    pos = 0
    while pos < len(future_dates):
        month = int(future_dates[pos].month)
        starts = valid_starts.get(month, [])
        take = min(block_size, len(future_dates) - pos)
        if starts:
            start = int(rng.choice(starts))
            out.append(hist_residuals[start:start + take])
        else:
            pool = residual_pool_for_month(residual_df, month)
            idx = rng.integers(0, len(pool), size=take)
            out.append(pool[idx])
        pos += take
    return np.concatenate(out)[:len(future_dates)]


def empirical_crps(samples: np.ndarray, obs: float) -> float:
    x = np.asarray(samples, dtype=float)
    term1 = np.mean(np.abs(x - obs))
    diffs = np.abs(x[:, None] - x[None, :])
    term2 = 0.5 * np.mean(diffs)
    return float(term1 - term2)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(
        ROOT / 'output' / 'model_useful_data_bundle' / 'tables' / 'istanbul_model_core_monthly.csv',
        parse_dates=['date'],
    )
    history = history.rename(columns={'weighted_total_fill': 'observed_fill'}).sort_values('date').reset_index(drop=True)

    future = pd.read_csv(
        ENSEMBLE_DIR / 'ensemble_phys_scenario_projection_monthly_2026_2040.csv',
        parse_dates=['date'],
    ).sort_values(['scenario', 'date']).reset_index(drop=True)

    calib = pd.read_csv(
        ENSEMBLE_DIR / 'ensemble_calibration_samples.csv',
        parse_dates=['date'],
    )
    calib = calib[calib['horizon_months'] == 1].copy().sort_values('date').reset_index(drop=True)
    calib['residual'] = calib['actual_fill'] - calib['pred_fill_ensemble_phys']
    return history, future, calib


def evaluate_backtest_with_scale(calib: pd.DataFrame, spread_scale: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_rows = []
    for i in range(MIN_RESIDUAL_HISTORY, len(calib)):
        history_pool = calib.iloc[:i].copy()
        pool = residual_pool_for_month(history_pool, int(calib.iloc[i]['date'].month))
        pred = float(calib.iloc[i]['pred_fill_ensemble_phys'])
        actual = float(calib.iloc[i]['actual_fill'])
        samples = np.clip(pred + spread_scale * pool, 0.0, 1.0)
        q10, q50, q90 = np.quantile(samples, [0.10, 0.50, 0.90])
        p30 = float((samples < 0.30).mean())
        p40 = float((samples < 0.40).mean())
        pit = float((samples <= actual).mean())
        sample_rows.append({
            'date': calib.iloc[i]['date'],
            'month': int(pd.Timestamp(calib.iloc[i]['date']).month),
            'actual_fill_pct': actual * 100.0,
            'pred_median_fill_pct': q50 * 100.0,
            'pred_p10_fill_pct': q10 * 100.0,
            'pred_p90_fill_pct': q90 * 100.0,
            'crps_pct': empirical_crps(samples, actual) * 100.0,
            'inside_p10_p90': int(q10 <= actual <= q90),
            'pred_prob_below_40_pct': p40 * 100.0,
            'pred_prob_below_30_pct': p30 * 100.0,
            'obs_below_40': int(actual < 0.40),
            'obs_below_30': int(actual < 0.30),
            'pit': pit,
        })

    samples_df = pd.DataFrame(sample_rows)
    monthly_diag = (
        samples_df.groupby('month')
        .agg(
            n_predictions=('month', 'size'),
            mean_crps_pct=('crps_pct', 'mean'),
            p10_p90_coverage_pct=('inside_p10_p90', lambda x: float(np.mean(x) * 100.0)),
        )
        .reset_index()
    )

    reliability_rows = []
    for threshold in [40, 30]:
        prob_col = f'pred_prob_below_{threshold}_pct'
        obs_col = f'obs_below_{threshold}'
        bins = np.arange(0, 110, 10)
        cats = pd.cut(samples_df[prob_col], bins=bins, right=False, include_lowest=True)
        grouped = samples_df.groupby(cats, observed=False)
        for interval, g in grouped:
            if g.empty:
                continue
            reliability_rows.append({
                'threshold_pct': threshold,
                'bin_left_pct': float(interval.left),
                'bin_right_pct': float(interval.right),
                'bin_mid_pct': float((interval.left + interval.right) / 2.0),
                'n_predictions': int(len(g)),
                'mean_predicted_prob_pct': float(g[prob_col].mean()),
                'observed_frequency_pct': float(g[obs_col].mean() * 100.0),
            })
    reliability_df = pd.DataFrame(reliability_rows)

    metrics = pd.DataFrame([
        {
            'forecast': 'hybrid_physics_ensemble_phys',
            'spread_scale': float(spread_scale),
            'backtest_start': str(samples_df['date'].min().date()),
            'backtest_end': str(samples_df['date'].max().date()),
            'n_predictions': int(len(samples_df)),
            'mean_crps_pct': float(samples_df['crps_pct'].mean()),
            'p10_p90_coverage_pct': float(samples_df['inside_p10_p90'].mean() * 100.0),
            'median_rmse_pct': float(np.sqrt(np.mean((samples_df['actual_fill_pct'] - samples_df['pred_median_fill_pct']) ** 2))),
        }
    ])
    return metrics, samples_df, monthly_diag, reliability_df


def build_backtest_probabilistic_metrics(calib: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows = []
    best = None
    best_score = None
    best_pack = None
    for scale in SPREAD_GRID:
        metrics, samples_df, monthly_diag, reliability_df = evaluate_backtest_with_scale(calib, float(scale))
        coverage = float(metrics.iloc[0]['p10_p90_coverage_pct'])
        crps = float(metrics.iloc[0]['mean_crps_pct'])
        score = (abs(coverage - NOMINAL_COVERAGE_P10_P90), crps)
        grid_rows.append({
            'spread_scale': float(scale),
            'p10_p90_coverage_pct': coverage,
            'mean_crps_pct': crps,
            'median_rmse_pct': float(metrics.iloc[0]['median_rmse_pct']),
            'coverage_gap_pct': abs(coverage - NOMINAL_COVERAGE_P10_P90),
        })
        if best_score is None or score < best_score:
            best = float(scale)
            best_score = score
            best_pack = (metrics, samples_df, monthly_diag, reliability_df)
    metrics, samples_df, monthly_diag, reliability_df = best_pack
    grid_df = pd.DataFrame(grid_rows).sort_values('spread_scale').reset_index(drop=True)
    return metrics, samples_df, monthly_diag, reliability_df, grid_df


def simulate_probabilistic_paths(future: pd.DataFrame, residual_df: pd.DataFrame, spread_scale: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    quantile_rows = []
    yearly_rows = []
    endpoint_rows = []
    checkpoints = [pd.Timestamp('2030-12-01'), pd.Timestamp('2035-12-01'), pd.Timestamp('2040-12-01')]
    crossing_rows = []

    for scenario, g in future.groupby('scenario'):
        g = g.sort_values('date').copy()
        dates = pd.DatetimeIndex(g['date'])
        base_path = g['pred_fill_ensemble'].to_numpy(dtype=float)
        horizon = len(base_path)
        sims = np.empty((N_SIMULATIONS, horizon), dtype=np.float32)
        for i in range(N_SIMULATIONS):
            noise = spread_scale * sample_seasonal_residual_blocks(residual_df, dates, BLOCK_SIZE, rng)
            sims[i, :] = np.clip(base_path + noise, 0.0, 1.0)

        qs = np.quantile(sims, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], axis=0)
        probs40 = (sims < 0.40).mean(axis=0)
        probs30 = (sims < 0.30).mean(axis=0)
        means = sims.mean(axis=0)
        for idx, date in enumerate(dates):
            quantile_rows.append({
                'scenario': scenario,
                'date': pd.Timestamp(date),
                'mean_fill_pct': float(means[idx] * 100.0),
                'p05_fill_pct': float(qs[0, idx] * 100.0),
                'p10_fill_pct': float(qs[1, idx] * 100.0),
                'p25_fill_pct': float(qs[2, idx] * 100.0),
                'p50_fill_pct': float(qs[3, idx] * 100.0),
                'p75_fill_pct': float(qs[4, idx] * 100.0),
                'p90_fill_pct': float(qs[5, idx] * 100.0),
                'p95_fill_pct': float(qs[6, idx] * 100.0),
                'prob_below_40_pct': float(probs40[idx] * 100.0),
                'prob_below_30_pct': float(probs30[idx] * 100.0),
            })

        years = sorted(pd.DatetimeIndex(dates).year.unique())
        for year in years:
            year_mask = pd.DatetimeIndex(dates).year == year
            year_sims = sims[:, year_mask]
            year_dates = pd.DatetimeIndex(dates[year_mask])
            dec_idx = int(np.flatnonzero(year_dates.month == 12)[0])
            dec_vals = year_sims[:, dec_idx]
            yearly_rows.append({
                'scenario': scenario,
                'year': int(year),
                'prob_any_month_below_40_pct': float(((year_sims < 0.40).any(axis=1)).mean() * 100.0),
                'prob_any_month_below_30_pct': float(((year_sims < 0.30).any(axis=1)).mean() * 100.0),
                'p10_december_fill_pct': float(np.quantile(dec_vals, 0.10) * 100.0),
                'p50_december_fill_pct': float(np.quantile(dec_vals, 0.50) * 100.0),
                'p90_december_fill_pct': float(np.quantile(dec_vals, 0.90) * 100.0),
            })

        for threshold in [40, 30]:
            th = threshold / 100.0
            for cp in checkpoints:
                cp_idx = int(np.flatnonzero(pd.DatetimeIndex(dates) == cp)[0])
                subset = sims[:, :cp_idx + 1]
                endpoint = sims[:, cp_idx]
                crossing_rows.append({
                    'scenario': scenario,
                    'threshold_pct': threshold,
                    'checkpoint': str(cp.date()),
                    'prob_any_cross_pct': float(((subset < th).any(axis=1)).mean() * 100.0),
                    'prob_checkpoint_below_pct': float((endpoint < th).mean() * 100.0),
                    'p10_checkpoint_fill_pct': float(np.quantile(endpoint, 0.10) * 100.0),
                    'p50_checkpoint_fill_pct': float(np.quantile(endpoint, 0.50) * 100.0),
                    'p90_checkpoint_fill_pct': float(np.quantile(endpoint, 0.90) * 100.0),
                })

        endpoint = sims[:, -1]
        endpoint_rows.append({
            'scenario': scenario,
            'mean_endpoint_2040_12_pct': float(endpoint.mean() * 100.0),
            'p10_endpoint_2040_12_pct': float(np.quantile(endpoint, 0.10) * 100.0),
            'p25_endpoint_2040_12_pct': float(np.quantile(endpoint, 0.25) * 100.0),
            'p50_endpoint_2040_12_pct': float(np.quantile(endpoint, 0.50) * 100.0),
            'p75_endpoint_2040_12_pct': float(np.quantile(endpoint, 0.75) * 100.0),
            'p90_endpoint_2040_12_pct': float(np.quantile(endpoint, 0.90) * 100.0),
        })

    return (
        pd.DataFrame(quantile_rows).sort_values(['scenario', 'date']).reset_index(drop=True),
        pd.DataFrame(yearly_rows).sort_values(['scenario', 'year']).reset_index(drop=True),
        pd.DataFrame(crossing_rows).sort_values(['threshold_pct', 'scenario', 'checkpoint']).reset_index(drop=True),
        pd.DataFrame(endpoint_rows).sort_values('p50_endpoint_2040_12_pct', ascending=False).reset_index(drop=True),
    )


def build_summary(yearly_df: pd.DataFrame, endpoint_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in PRIMARY_SCENARIOS:
        ep = endpoint_df[endpoint_df['scenario'] == scenario].iloc[0]
        yr = yearly_df[yearly_df['scenario'] == scenario].copy()
        row = {
            'scenario': scenario,
            'p50_endpoint_2040_12_pct': float(ep['p50_endpoint_2040_12_pct']),
            'p10_endpoint_2040_12_pct': float(ep['p10_endpoint_2040_12_pct']),
            'p90_endpoint_2040_12_pct': float(ep['p90_endpoint_2040_12_pct']),
        }
        for th in [40, 30]:
            hit = yr[yr[f'prob_any_month_below_{th}_pct'] >= 50.0]
            row[f'first_year_prob_below_{th}_ge_50pct'] = int(hit.iloc[0]['year']) if not hit.empty else ''
        rows.append(row)
    return pd.DataFrame(rows).sort_values('p50_endpoint_2040_12_pct', ascending=False).reset_index(drop=True)


def plot_fan(history: pd.DataFrame, quantiles_df: pd.DataFrame, out_path: Path) -> None:
    hist = history[history['date'] >= '2018-01-01'].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), dpi=170, sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, scenario in zip(axes, PRIMARY_SCENARIOS):
        q = quantiles_df[quantiles_df['scenario'] == scenario].copy()
        ax.plot(hist['date'], hist['observed_fill'] * 100.0, color='#111827', linewidth=1.8)
        ax.fill_between(q['date'], q['p10_fill_pct'], q['p90_fill_pct'], color=SCENARIO_COLORS[scenario], alpha=0.16)
        ax.fill_between(q['date'], q['p25_fill_pct'], q['p75_fill_pct'], color=SCENARIO_COLORS[scenario], alpha=0.24)
        ax.plot(q['date'], q['p50_fill_pct'], color=SCENARIO_COLORS[scenario], linewidth=2.0)
        ax.axvline(pd.Timestamp('2026-01-01'), color='#6b7280', linestyle='--', linewidth=1.0)
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', alpha=0.22)
    axes[0].set_ylabel('Toplam doluluk (%)')
    axes[2].set_ylabel('Toplam doluluk (%)')
    fig.suptitle('Fizik-kısıtlı ensemble ile olasılıksal projeksiyon (P10-P50-P90)', y=0.98)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_yearly_risk(yearly_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=170, sharey=True)
    for ax, threshold in zip(axes, [40, 30]):
        for scenario in PRIMARY_SCENARIOS:
            g = yearly_df[yearly_df['scenario'] == scenario].copy()
            ax.plot(g['year'], g[f'prob_any_month_below_{threshold}_pct'], linewidth=2.0, color=SCENARIO_COLORS[scenario], label=SCENARIO_LABELS[scenario])
        ax.set_title(f'Yıllık %{threshold} altı risk')
        ax.set_xlabel('Yıl')
        ax.grid(True, axis='y', alpha=0.22)
    axes[0].set_ylabel('Olasılık (%)')
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_endpoint_ranges(endpoint_df: pd.DataFrame, out_path: Path) -> None:
    target = endpoint_df[endpoint_df['scenario'].isin(PRIMARY_SCENARIOS)].copy()
    target['scenario_tr'] = target['scenario'].map(SCENARIO_LABELS)
    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=170)
    y = np.arange(len(target))
    ax.hlines(y, target['p10_endpoint_2040_12_pct'], target['p90_endpoint_2040_12_pct'], color='#94a3b8', linewidth=5)
    ax.scatter(target['p50_endpoint_2040_12_pct'], y, color='#111827', s=36, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(target['scenario_tr'])
    ax.set_xlabel('2040 Aralık toplam doluluk (%)')
    ax.set_title('2040 sonu belirsizlik aralığı')
    ax.grid(True, axis='x', alpha=0.22)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_monthly_coverage(monthly_diag: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.8), dpi=170, sharex=True)
    axes[0].bar(monthly_diag['month'], monthly_diag['p10_p90_coverage_pct'], color='#2563eb')
    axes[0].axhline(80.0, color='#dc2626', linestyle='--', linewidth=1.2)
    axes[0].set_ylabel('Kapsama (%)')
    axes[0].set_title('Aylara göre P10-P90 kapsama')
    axes[0].grid(True, axis='y', alpha=0.22)
    axes[1].bar(monthly_diag['month'], monthly_diag['mean_crps_pct'], color='#059669')
    axes[1].set_ylabel('CRPS (yp)')
    axes[1].set_xlabel('Ay')
    axes[1].set_title('Aylara göre CRPS')
    axes[1].grid(True, axis='y', alpha=0.22)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reliability(reliability_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=170, sharey=True)
    for ax, threshold in zip(axes, [40, 30]):
        g = reliability_df[reliability_df['threshold_pct'] == threshold].copy()
        ax.plot([0, 100], [0, 100], color='#6b7280', linestyle='--', linewidth=1.0)
        ax.plot(g['mean_predicted_prob_pct'], g['observed_frequency_pct'], marker='o', color='#2563eb', linewidth=1.8)
        for _, row in g.iterrows():
            ax.annotate(str(int(row['n_predictions'])), (row['mean_predicted_prob_pct'], row['observed_frequency_pct']), fontsize=7, xytext=(3, 3), textcoords='offset points')
        ax.set_title(f'%{threshold} altı güvenilirlik')
        ax.set_xlabel('Tahmin edilen olasılık (%)')
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel('Gözlenen sıklık (%)')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pit_histogram(samples_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=170)
    bins = np.linspace(0.0, 1.0, 11)
    ax.hist(samples_df['pit'], bins=bins, color='#7c3aed', edgecolor='white')
    ax.axhline(len(samples_df) / 10.0, color='#dc2626', linestyle='--', linewidth=1.0)
    ax.set_title('PIT histogramı')
    ax.set_xlabel('Tahmini dağılım içindeki konum')
    ax.set_ylabel('Gözlem sayısı')
    ax.grid(True, axis='y', alpha=0.22)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = OUT_DIR / 'figures'
    figs.mkdir(parents=True, exist_ok=True)

    history, future, calib = load_inputs()
    backtest_metrics, backtest_samples, monthly_diag, reliability_df, scale_grid = build_backtest_probabilistic_metrics(calib)
    spread_scale = float(backtest_metrics.iloc[0]['spread_scale'])
    quantiles_df, yearly_df, crossing_df, endpoint_df = simulate_probabilistic_paths(
        future,
        calib[['date', 'residual']].copy(),
        spread_scale,
    )
    summary_df = build_summary(yearly_df, endpoint_df)

    backtest_metrics.to_csv(OUT_DIR / 'probabilistic_backtest_metrics.csv', index=False)
    backtest_samples.to_csv(OUT_DIR / 'probabilistic_backtest_samples.csv', index=False)
    monthly_diag.to_csv(OUT_DIR / 'probabilistic_monthly_diagnostics.csv', index=False)
    reliability_df.to_csv(OUT_DIR / 'probabilistic_reliability_by_threshold.csv', index=False)
    scale_grid.to_csv(OUT_DIR / 'probabilistic_spread_scale_grid.csv', index=False)
    quantiles_df.to_csv(OUT_DIR / 'probabilistic_monthly_quantiles_2026_2040.csv', index=False)
    yearly_df.to_csv(OUT_DIR / 'probabilistic_yearly_risk_2026_2040.csv', index=False)
    crossing_df.to_csv(OUT_DIR / 'probabilistic_crossing_summary_2026_2040.csv', index=False)
    endpoint_df.to_csv(OUT_DIR / 'probabilistic_endpoint_summary_2040.csv', index=False)
    summary_df.to_csv(OUT_DIR / 'probabilistic_projection_summary_2026_2040.csv', index=False)

    plot_fan(history, quantiles_df, figs / 'probabilistic_fan_paths_2026_2040.png')
    plot_yearly_risk(yearly_df, figs / 'probabilistic_yearly_threshold_risk.png')
    plot_endpoint_ranges(endpoint_df, figs / 'probabilistic_endpoint_ranges_2040.png')
    plot_monthly_coverage(monthly_diag, figs / 'probabilistic_monthly_coverage_crps.png')
    plot_reliability(reliability_df, figs / 'probabilistic_reliability_thresholds.png')
    plot_pit_histogram(backtest_samples, figs / 'probabilistic_pit_histogram.png')

    summary = {
        'model': 'hybrid_physics_ensemble_phys',
        'n_simulations': N_SIMULATIONS,
        'block_size_months': BLOCK_SIZE,
        'sampling_mode': 'season_aware_block_bootstrap',
        'spread_scale': spread_scale,
        'seed': SEED,
        'backtest_start': str(backtest_metrics.iloc[0]['backtest_start']),
        'backtest_end': str(backtest_metrics.iloc[0]['backtest_end']),
        'mean_crps_pct': float(backtest_metrics.iloc[0]['mean_crps_pct']),
        'p10_p90_coverage_pct': float(backtest_metrics.iloc[0]['p10_p90_coverage_pct']),
        'median_rmse_pct': float(backtest_metrics.iloc[0]['median_rmse_pct']),
    }
    (OUT_DIR / 'probabilistic_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT_DIR)


if __name__ == '__main__':
    main()
