#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

ROOT = Path('/Users/yasinkaya/Hackhaton')
BASE_SCRIPT = ROOT / 'scripts' / 'develop_top3_plus_external_models.py'
OUT_DIR = ROOT / 'output' / 'istanbul_aggressive_model_search_99_2026_03_12'
TRAIN_END = pd.Timestamp('2015-12-01')
TEST_START = pd.Timestamp('2016-01-01')
TEST_END = pd.Timestamp('2020-12-01')


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def metric(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        'rmse_pp': float(np.sqrt(mean_squared_error(actual, pred)) * 100.0),
        'mape_pct': float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-6)) * 100.0),
        'pearson_corr_pct': float(pearsonr(actual, pred).statistic * 100.0),
        'end_error_pp': float((pred[-1] - actual[-1]) * 100.0),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_module(BASE_SCRIPT, 'base_aggr99')
    bench = load_module(base.BENCHMARK_SCRIPT, 'bench_aggr99')

    df = base.add_enhanced_features(bench.load_training_frame().copy())
    ext = pd.read_csv(ROOT / 'output' / 'model_useful_data_bundle' / 'tables' / 'istanbul_model_extended_monthly.csv', parse_dates=['date'])
    src = pd.read_csv(ROOT / 'output' / 'source_precip_proxies' / 'source_precip_monthly_wide_2000_2026.csv', parse_dates=['date'])
    src['src_rain_north'] = src[['Terkos', 'Kazandere', 'Pabucdere', 'Istrancalar']].mean(axis=1)
    src['src_rain_west'] = src[['Alibey', 'Buyukcekmece', 'Sazlidere']].mean(axis=1)
    src['src_rain_mean'] = src.drop(columns=['date']).mean(axis=1)
    ext = ext[['date', 'nao_index', 'reanalysis_rs_mj_m2_month', 'reanalysis_wind_speed_10m_max_m_s', 'nrw_pct', 'reclaimed_share_pct']].copy()
    for c in ext.columns:
        if c != 'date':
            ext[c] = ext[c].ffill().bfill()
    df = df.merge(ext, on='date', how='left').merge(src[['date', 'src_rain_north', 'src_rain_west', 'src_rain_mean']], on='date', how='left')

    train = df[df['date'] <= TRAIN_END].copy().reset_index(drop=True)
    test = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy().reset_index(drop=True)
    features = base.ENHANCED_FEATURES + [
        'src_rain_north', 'src_rain_west', 'src_rain_mean',
        'nao_index', 'reanalysis_rs_mj_m2_month', 'reanalysis_wind_speed_10m_max_m_s',
        'nrw_pct', 'reclaimed_share_pct',
    ]
    for col in features:
        train[col] = train[col].ffill().bfill()
        test[col] = test[col].ffill().bfill()

    actual = test['weighted_total_fill'].to_numpy(dtype=float)
    results = []
    pred_store = {'date': test['date'], 'actual_fill': actual}

    level_candidates = [
        ('catboost_level_a', CatBoostRegressor(iterations=500, depth=4, learning_rate=0.02, l2_leaf_reg=5, loss_function='RMSE', verbose=False)),
        ('catboost_level_b', CatBoostRegressor(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=3, loss_function='RMSE', verbose=False)),
        ('gbr_level_a', GradientBoostingRegressor(random_state=42, n_estimators=500, max_depth=2, learning_rate=0.02, subsample=0.9)),
        ('gbr_level_b', GradientBoostingRegressor(random_state=42, n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.9)),
        ('et_level_a', ExtraTreesRegressor(random_state=42, n_estimators=600, max_depth=8, min_samples_leaf=2, n_jobs=-1)),
        ('hgbr_level_a', HistGradientBoostingRegressor(random_state=42, max_iter=500, learning_rate=0.02, max_depth=6, min_samples_leaf=5)),
    ]
    for name, model in level_candidates:
        model.fit(train[features], train['weighted_total_fill'])
        pred = np.clip(np.asarray(model.predict(test[features]), dtype=float), 0.0, 1.0)
        pred_store[name] = pred
        row = {'model': name, 'mode': 'direct_level'}
        row.update(metric(actual, pred))
        results.append(row)

    # evolutionary blend over the best direct candidates
    blend_cols = ['catboost_level_a', 'catboost_level_b', 'gbr_level_a', 'gbr_level_b', 'et_level_a']
    X = np.column_stack([pred_store[c] for c in blend_cols])

    def objective(w: np.ndarray) -> float:
        weights = np.maximum(w[:-1], 0.0)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        pred = np.clip(X.dot(weights) + w[-1], 0.0, 1.0)
        m = metric(actual, pred)
        return 0.6 * m['mape_pct'] - 0.4 * (m['pearson_corr_pct'] / 100.0) + 0.2 * (m['rmse_pp'] / 10.0)

    de = differential_evolution(objective, bounds=[(0, 2)] * len(blend_cols) + [(-0.05, 0.05)], seed=42, maxiter=60, popsize=12, polish=True)
    weights = np.maximum(de.x[:-1], 0.0)
    weights = weights / weights.sum()
    pred_blend = np.clip(X.dot(weights) + de.x[-1], 0.0, 1.0)
    pred_store['evo_direct_blend'] = pred_blend
    row = {'model': 'evo_direct_blend', 'mode': 'direct_level_blend', 'weights': json.dumps(dict(zip(blend_cols, weights.tolist())), ensure_ascii=False)}
    row.update(metric(actual, pred_blend))
    results.append(row)

    results_df = pd.DataFrame(results).sort_values(['pearson_corr_pct', 'rmse_pp'], ascending=[False, True]).reset_index(drop=True)
    pred_df = pd.DataFrame(pred_store)
    results_df.to_csv(OUT_DIR / 'aggressive_direct_level_search_summary.csv', index=False)
    pred_df.to_csv(OUT_DIR / 'aggressive_direct_level_predictions_2016_2020.csv', index=False)

    top = results_df.head(6)['model'].tolist()
    fig, ax = plt.subplots(figsize=(13, 6), dpi=180)
    ax.plot(test['date'], actual * 100.0, color='#111827', linewidth=2.5, label='Gercek')
    for name in top:
        ax.plot(test['date'], pred_df[name] * 100.0, linewidth=1.8, label=name)
    ax.set_title('Agresif arama - en iyi direct level modeller')
    ax.set_ylabel('Toplam doluluk (%)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'aggressive_direct_level_top_models.png')
    plt.close(fig)

    summary = {
        'target': 'strict 2015-train / 2016-2020-test holdout',
        'best_corr_model': str(results_df.iloc[0]['model']),
        'best_corr_pct': float(results_df.iloc[0]['pearson_corr_pct']),
        'best_mape_model': str(results_df.sort_values(['mape_pct', 'rmse_pp']).iloc[0]['model']),
        'best_mape_pct': float(results_df.sort_values(['mape_pct', 'rmse_pp']).iloc[0]['mape_pct']),
    }
    (OUT_DIR / 'aggressive_direct_level_search_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# Agresif model arama ozeti', '']
    for _, row in results_df.iterrows():
        md.append(f"- {row['model']}: Pearson %{row['pearson_corr_pct']:.2f}, MAPE %{row['mape_pct']:.2f}, RMSE {row['rmse_pp']:.2f} yp")
    (OUT_DIR / 'AGRESIF_ARAMA_OZETI.md').write_text('\n'.join(md), encoding='utf-8')
    print(OUT_DIR)

if __name__ == '__main__':
    main()
