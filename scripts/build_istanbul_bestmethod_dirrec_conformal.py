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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/yasinkaya/Hackhaton')
FORWARD_SCRIPT = ROOT / 'scripts' / 'build_istanbul_forward_projection_2040.py'
BASE_CALIB_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_sourceaware_ensemble_2040' / 'ensemble_calibration_samples.csv'
BASE_SCENARIO_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_sourceaware_ensemble_2040' / 'ensemble_phys_scenario_projection_monthly_2026_2040.csv'
CORRECTION_PATH = ROOT / 'output' / 'istanbul_reconciled_projection_2040' / 'nearterm_correction_curve_2026_2040.csv'
CURRENT_PREF_SUMMARY_PATH = ROOT / 'output' / 'istanbul_preferred_projection_2040' / 'probabilistic_summary.csv'
OUT_DIR = ROOT / 'output' / 'istanbul_bestmethod_dirrec_conformal_2040'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_HORIZONS = [1, 3, 6, 12]
ALL_HORIZONS = list(range(1, 13))
MIN_TRAIN = 60
BACKTEST_STEP = 2
WEIGHT_GRID = np.linspace(0.0, 1.0, 21)  # weight on base recursive path
ALPHA = 0.20  # 80% symmetric conformal interval
PRIMARY_SCENARIOS = ['base', 'wet_mild', 'management_improvement', 'hot_dry_high_demand']
SCENARIO_LABELS = {
    'base': 'Temel',
    'wet_mild': 'Ilık-ıslak',
    'management_improvement': 'Yönetim iyileşme',
    'hot_dry_high_demand': 'Sıcak-kurak-yüksek talep',
}
SCENARIO_COLORS = {
    'base': '#2563eb',
    'wet_mild': '#059669',
    'management_improvement': '#d97706',
    'hot_dry_high_demand': '#dc2626',
}

DIRECT_FEATURES = [
    'origin_fill',
    'origin_fill_lag1',
    'origin_delta_fill',
    'horizon_months',
    'target_rain_model_mm',
    'target_rain_model_mm_lag1',
    'target_rain_model_mm_roll3',
    'target_et0_mm_month',
    'target_et0_mm_month_lag1',
    'target_et0_mm_month_roll3',
    'target_consumption_mean_monthly',
    'target_consumption_mean_monthly_lag1',
    'target_consumption_mean_monthly_roll3',
    'target_temp_proxy_c',
    'target_rh_proxy_pct',
    'target_vpd_kpa_mean',
    'target_water_balance_proxy_mm',
    'target_month_sin',
    'target_month_cos',
]

DIRECT_MONOTONIC = [
    1,  # origin_fill
    1,  # origin_fill_lag1
    0,  # origin_delta_fill
    0,  # horizon_months
    1,  # rain
    1,  # rain lag1
    1,  # rain roll3
    -1, # et0
    -1, # et0 lag1
    -1, # et0 roll3
    -1, # cons
    -1, # cons lag1
    -1, # cons roll3
    -1, # temp
    1,  # rh
    -1, # vpd
    1,  # water balance
    0,  # month sin
    0,  # month cos
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def rmse_pp(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) * 100.0)


def load_training_frame(forward) -> pd.DataFrame:
    return forward.load_training_frame().copy().reset_index(drop=True)


def make_direct_model(family: str):
    if family == 'ridge':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ])
    if family == 'hist_gbm_monotonic':
        return HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=180,
            min_samples_leaf=6,
            monotonic_cst=DIRECT_MONOTONIC,
            random_state=42,
        )
    raise ValueError(family)


def augment_future_exog(train_df: pd.DataFrame, future_exog: pd.DataFrame) -> pd.DataFrame:
    tail = train_df[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']].tail(2).copy()
    combined = pd.concat([
        tail,
        future_exog[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']].copy(),
    ], ignore_index=True)
    for col in ['rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']:
        combined[f'{col}_lag1'] = combined[col].shift(1)
        combined[f'{col}_roll3'] = combined[col].rolling(3).mean()
    out = future_exog.copy().reset_index(drop=True)
    derived = combined.iloc[2:].reset_index(drop=True)
    for col in [
        'rain_model_mm_lag1', 'rain_model_mm_roll3',
        'et0_mm_month_lag1', 'et0_mm_month_roll3',
        'consumption_mean_monthly_lag1', 'consumption_mean_monthly_roll3',
    ]:
        out[col] = derived[col].to_numpy(dtype=float)
    out['water_balance_proxy_mm'] = out['rain_model_mm'] - out['et0_mm_month']
    return out


def direct_feature_dict(origin_row: pd.Series, target_row: pd.Series, horizon: int) -> dict[str, float]:
    return {
        'origin_fill': float(origin_row['weighted_total_fill']),
        'origin_fill_lag1': float(origin_row['weighted_total_fill_lag1']),
        'origin_delta_fill': float(origin_row['delta_fill']),
        'horizon_months': float(horizon),
        'target_rain_model_mm': float(target_row['rain_model_mm']),
        'target_rain_model_mm_lag1': float(target_row['rain_model_mm_lag1']),
        'target_rain_model_mm_roll3': float(target_row['rain_model_mm_roll3']),
        'target_et0_mm_month': float(target_row['et0_mm_month']),
        'target_et0_mm_month_lag1': float(target_row['et0_mm_month_lag1']),
        'target_et0_mm_month_roll3': float(target_row['et0_mm_month_roll3']),
        'target_consumption_mean_monthly': float(target_row['consumption_mean_monthly']),
        'target_consumption_mean_monthly_lag1': float(target_row['consumption_mean_monthly_lag1']),
        'target_consumption_mean_monthly_roll3': float(target_row['consumption_mean_monthly_roll3']),
        'target_temp_proxy_c': float(target_row['temp_proxy_c']),
        'target_rh_proxy_pct': float(target_row['rh_proxy_pct']),
        'target_vpd_kpa_mean': float(target_row['vpd_kpa_mean']),
        'target_water_balance_proxy_mm': float(target_row['water_balance_proxy_mm']),
        'target_month_sin': float(target_row['month_sin']),
        'target_month_cos': float(target_row['month_cos']),
    }


def build_direct_training_dataset(df: pd.DataFrame, horizon: int, end_origin_idx: int | None = None) -> pd.DataFrame:
    rows = []
    limit = len(df) - horizon
    if end_origin_idx is not None:
        limit = min(limit, end_origin_idx)
    for i in range(0, limit):
        target_idx = i + horizon
        origin_row = df.iloc[i]
        target_row = df.iloc[target_idx]
        feat = direct_feature_dict(origin_row, target_row, horizon)
        feat['origin_date'] = origin_row['date']
        feat['date'] = target_row['date']
        feat['actual_fill'] = float(target_row['weighted_total_fill'])
        rows.append(feat)
    return pd.DataFrame(rows)


def precompute_direct_datasets(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {h: build_direct_training_dataset(df, h) for h in ALL_HORIZONS}


def backtest_direct_family(df: pd.DataFrame, family: str, ds_map: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    for horizon in ANCHOR_HORIZONS:
        ds = ds_map[horizon]
        actual, pred, dates = [], [], []
        for origin_idx in range(MIN_TRAIN, len(df) - horizon, BACKTEST_STEP):
            cutoff_date = df.iloc[origin_idx]['date']
            train_pairs = ds[ds['origin_date'] < cutoff_date]
            if len(train_pairs) < 30:
                continue
            model = make_direct_model(family)
            model.fit(train_pairs[DIRECT_FEATURES], train_pairs['actual_fill'])
            feat = pd.DataFrame([direct_feature_dict(df.iloc[origin_idx], df.iloc[origin_idx + horizon], horizon)])
            yhat = float(np.clip(model.predict(feat[DIRECT_FEATURES])[0], 0.0, 1.0))
            pred.append(yhat)
            actual.append(float(df.iloc[origin_idx + horizon]['weighted_total_fill']))
            dates.append(df.iloc[origin_idx + horizon]['date'])
            pred_rows.append({
                'family': family,
                'horizon_months': horizon,
                'date': df.iloc[origin_idx + horizon]['date'],
                'actual_fill': float(df.iloc[origin_idx + horizon]['weighted_total_fill']),
                'pred_fill_direct': yhat,
            })
        metric_rows.append({
            'family': family,
            'horizon_months': horizon,
            'rmse_pp': rmse_pp(np.asarray(actual), np.asarray(pred)),
            'n_predictions': len(actual),
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def choose_best_direct_family(df: pd.DataFrame, ds_map: dict[int, pd.DataFrame]) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    all_metrics = []
    all_preds = []
    for family in ['ridge', 'hist_gbm_monotonic']:
        m, p = backtest_direct_family(df, family, ds_map)
        all_metrics.append(m)
        all_preds.append(p)
    metrics = pd.concat(all_metrics, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    summary = metrics.groupby('family', as_index=False).agg(mean_anchor_rmse_pp=('rmse_pp', 'mean'))
    best_family = str(summary.sort_values('mean_anchor_rmse_pp').iloc[0]['family'])
    return best_family, metrics, preds[preds['family'] == best_family].copy()


def optimize_blend_weights(best_direct_preds: pd.DataFrame, base_calib: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = base_calib[base_calib['horizon_months'].isin(ANCHOR_HORIZONS)].copy()
    merged = base.merge(
        best_direct_preds[['horizon_months', 'date', 'actual_fill', 'pred_fill_direct']],
        on=['horizon_months', 'date', 'actual_fill'],
        how='inner',
    )
    rows = []
    best_rows = []
    for horizon, g in merged.groupby('horizon_months'):
        best_rmse = None
        best_w = None
        best_pred = None
        for w in WEIGHT_GRID:
            pred = w * g['pred_fill_ensemble_phys'].to_numpy(dtype=float) + (1.0 - w) * g['pred_fill_direct'].to_numpy(dtype=float)
            rmse = rmse_pp(g['actual_fill'].to_numpy(dtype=float), pred)
            rows.append({
                'horizon_months': int(horizon),
                'weight_base_recursive': float(w),
                'weight_direct': float(1.0 - w),
                'rmse_pp': rmse,
            })
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_w = float(w)
                best_pred = pred
        best_rows.append({
            'horizon_months': int(horizon),
            'weight_base_recursive': best_w,
            'weight_direct': 1.0 - best_w,
            'base_rmse_pp': rmse_pp(g['actual_fill'].to_numpy(dtype=float), g['pred_fill_ensemble_phys'].to_numpy(dtype=float)),
            'direct_rmse_pp': rmse_pp(g['actual_fill'].to_numpy(dtype=float), g['pred_fill_direct'].to_numpy(dtype=float)),
            'blend_rmse_pp': best_rmse,
            'n_predictions': int(len(g)),
        })
        g = g.copy()
        g['pred_fill_blend'] = best_pred
        best_rows[-1]['mean_blend_fill'] = float(np.mean(best_pred))
    return pd.DataFrame(rows), pd.DataFrame(best_rows)


def interpolate_anchor_map(anchor_df: pd.DataFrame, value_col: str) -> dict[int, float]:
    xs = anchor_df['horizon_months'].to_numpy(dtype=float)
    ys = anchor_df[value_col].to_numpy(dtype=float)
    out = {}
    for h in ALL_HORIZONS:
        out[h] = float(np.interp(h, xs, ys))
    return out


def fit_full_direct_models(ds_map: dict[int, pd.DataFrame], family: str) -> dict[int, object]:
    models = {}
    for horizon in ALL_HORIZONS:
        ds = ds_map[horizon]
        model = make_direct_model(family)
        model.fit(ds[DIRECT_FEATURES], ds['actual_fill'])
        models[horizon] = model
    return models


def compute_conformal_halfwidths(best_direct_preds: pd.DataFrame, base_calib: pd.DataFrame, best_weights: pd.DataFrame) -> pd.DataFrame:
    base = base_calib[base_calib['horizon_months'].isin(ANCHOR_HORIZONS)].copy()
    merged = base.merge(
        best_direct_preds[['horizon_months', 'date', 'actual_fill', 'pred_fill_direct']],
        on=['horizon_months', 'date', 'actual_fill'],
        how='inner',
    ).merge(best_weights[['horizon_months', 'weight_base_recursive']], on='horizon_months', how='left')
    merged['pred_fill_blend'] = merged['weight_base_recursive'] * merged['pred_fill_ensemble_phys'] + (1.0 - merged['weight_base_recursive']) * merged['pred_fill_direct']
    rows = []
    for horizon, g in merged.groupby('horizon_months'):
        abs_resid = np.abs(g['actual_fill'].to_numpy(dtype=float) - g['pred_fill_blend'].to_numpy(dtype=float))
        q = float(np.quantile(abs_resid, 1.0 - ALPHA))
        cov = float(np.mean((g['actual_fill'] >= (g['pred_fill_blend'] - q)) & (g['actual_fill'] <= (g['pred_fill_blend'] + q))) * 100.0)
        rows.append({
            'horizon_months': int(horizon),
            'conformal_halfwidth': q,
            'empirical_coverage_pct': cov,
            'n_predictions': int(len(g)),
        })
    return pd.DataFrame(rows).sort_values('horizon_months').reset_index(drop=True)


def build_future_paths(forward, train_df: pd.DataFrame, direct_models: dict[int, object], weight_map: dict[int, float], q_map: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_proj = pd.read_csv(BASE_SCENARIO_PATH, parse_dates=['date'])
    correction = pd.read_csv(CORRECTION_PATH, parse_dates=['date'])
    corr_map = dict(zip(correction['date'], correction['correction_pp']))

    clim = forward.monthly_climatology(train_df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    cfg_map = {cfg.scenario: cfg for cfg in (forward.build_scenarios() + forward.build_transfer_scenarios())}

    path_rows = []
    summary_rows = []
    for scenario in sorted(base_proj['scenario'].unique()):
        if scenario not in cfg_map:
            continue
        cfg = cfg_map[scenario]
        future_raw = forward.build_future_exog(train_df, cfg, clim, demand_relief, transfer_share_anchor_pct)
        future_raw = future_raw[future_raw['date'] >= pd.Timestamp('2026-01-01')].copy().reset_index(drop=True)
        future = augment_future_exog(train_df, future_raw)
        base_s = base_proj[base_proj['scenario'] == scenario].sort_values('date').reset_index(drop=True)
        hist_fill = train_df['weighted_total_fill'].tolist()
        rows_s = []
        for block_start in range(0, len(future), 12):
            block = future.iloc[block_start:block_start + 12].copy().reset_index(drop=True)
            origin_fill = float(hist_fill[-1])
            origin_fill_lag1 = float(hist_fill[-2])
            origin_delta = origin_fill - origin_fill_lag1
            block_preds = []
            for h in range(1, len(block) + 1):
                target_row = block.iloc[h - 1]
                feat = pd.DataFrame([{
                    'origin_fill': origin_fill,
                    'origin_fill_lag1': origin_fill_lag1,
                    'origin_delta_fill': origin_delta,
                    'horizon_months': float(h),
                    'target_rain_model_mm': float(target_row['rain_model_mm']),
                    'target_rain_model_mm_lag1': float(target_row['rain_model_mm_lag1']),
                    'target_rain_model_mm_roll3': float(target_row['rain_model_mm_roll3']),
                    'target_et0_mm_month': float(target_row['et0_mm_month']),
                    'target_et0_mm_month_lag1': float(target_row['et0_mm_month_lag1']),
                    'target_et0_mm_month_roll3': float(target_row['et0_mm_month_roll3']),
                    'target_consumption_mean_monthly': float(target_row['consumption_mean_monthly']),
                    'target_consumption_mean_monthly_lag1': float(target_row['consumption_mean_monthly_lag1']),
                    'target_consumption_mean_monthly_roll3': float(target_row['consumption_mean_monthly_roll3']),
                    'target_temp_proxy_c': float(target_row['temp_proxy_c']),
                    'target_rh_proxy_pct': float(target_row['rh_proxy_pct']),
                    'target_vpd_kpa_mean': float(target_row['vpd_kpa_mean']),
                    'target_water_balance_proxy_mm': float(target_row['water_balance_proxy_mm']),
                    'target_month_sin': float(target_row['month_sin']),
                    'target_month_cos': float(target_row['month_cos']),
                }])
                direct_pred = float(np.clip(direct_models[h].predict(feat[DIRECT_FEATURES])[0], 0.0, 1.0))
                base_pred = float(base_s.iloc[block_start + h - 1]['pred_fill_ensemble'])
                w_base = weight_map[h]
                blend = float(np.clip(w_base * base_pred + (1.0 - w_base) * direct_pred, 0.0, 1.0))
                q = q_map[h]
                corr_pp = float(corr_map.get(pd.Timestamp(target_row['date']), 0.0))
                blend_reconciled = float(np.clip(blend + corr_pp / 100.0, 0.0, 1.0))
                rows_s.append({
                    'date': target_row['date'],
                    'scenario': scenario,
                    'pred_fill_base': base_pred,
                    'pred_fill_direct': direct_pred,
                    'pred_fill_blend': blend,
                    'pred_fill_blend_reconciled': blend_reconciled,
                    'pred_fill_low_conf80': float(np.clip(blend_reconciled - q, 0.0, 1.0)),
                    'pred_fill_high_conf80': float(np.clip(blend_reconciled + q, 0.0, 1.0)),
                    'rain_model_mm': float(target_row['rain_model_mm']),
                    'et0_mm_month': float(target_row['et0_mm_month']),
                    'consumption_mean_monthly': float(target_row['consumption_mean_monthly']),
                    'correction_pp': corr_pp,
                    'horizon_in_block': int(h),
                    'weight_base_recursive': w_base,
                    'weight_direct': 1.0 - w_base,
                    'conformal_halfwidth': q,
                })
                block_preds.append(blend_reconciled)
            hist_fill.extend(block_preds)
        scen_df = pd.DataFrame(rows_s)
        path_rows.append(scen_df)
        target = scen_df[(scen_df['date'] >= '2026-01-01') & (scen_df['date'] <= '2040-12-01')].copy()
        summary_rows.append({
            'scenario': scenario,
            'mean_fill_2026_2040_pct': float(target['pred_fill_blend_reconciled'].mean() * 100.0),
            'min_fill_2026_2040_pct': float(target['pred_fill_blend_reconciled'].min() * 100.0),
            'end_fill_2040_12_pct': float(target.iloc[-1]['pred_fill_blend_reconciled'] * 100.0),
            'end_p10_2040_12_pct': float(target.iloc[-1]['pred_fill_low_conf80'] * 100.0),
            'end_p90_2040_12_pct': float(target.iloc[-1]['pred_fill_high_conf80'] * 100.0),
        })
    return pd.concat(path_rows, ignore_index=True), pd.DataFrame(summary_rows)


def plot_anchor_metrics(best_weights: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=170)
    x = np.arange(len(best_weights))
    ax.bar(x - 0.22, best_weights['base_rmse_pp'], width=0.22, color='#94a3b8', label='Mevcut recursive')
    ax.bar(x, best_weights['direct_rmse_pp'], width=0.22, color='#60a5fa', label='Direct model')
    ax.bar(x + 0.22, best_weights['blend_rmse_pp'], width=0.22, color='#0f766e', label='Direct-recursive blend')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(h)) for h in best_weights['horizon_months']])
    ax.set_xlabel('Ufuk (ay)')
    ax.set_ylabel('RMSE (yüzde puan)')
    ax.set_title('Anchor ufuklarda yöntem karşılaştırması')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_weight_map(weight_map: dict[int, float], out_path: Path) -> None:
    xs = list(weight_map.keys())
    ys = [weight_map[x] for x in xs]
    fig, ax = plt.subplots(figsize=(7.8, 4.1), dpi=170)
    ax.plot(xs, ys, marker='o', color='#2563eb', linewidth=2.0)
    ax.set_xlabel('Ufuk (ay)')
    ax.set_ylabel('Base recursive ağırlığı')
    ax.set_title('Direct-recursive blend ağırlık haritası')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.22)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scenarios(paths: pd.DataFrame, train_df: pd.DataFrame, out_path: Path) -> None:
    hist = train_df[train_df['date'] >= '2018-01-01'].copy()
    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=170)
    ax.plot(hist['date'], hist['weighted_total_fill'] * 100.0, color='#111827', linewidth=2.0, label='Gözlenen')
    for scenario in PRIMARY_SCENARIOS:
        g = paths[paths['scenario'] == scenario].copy()
        ax.plot(g['date'], g['pred_fill_blend_reconciled'] * 100.0, color=SCENARIO_COLORS[scenario], linewidth=2.0, label=SCENARIO_LABELS[scenario])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('Direct-recursive blend ile 2026-2040 projeksiyon yolları')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_base_compare(paths: pd.DataFrame, out_path: Path) -> None:
    g = paths[paths['scenario'] == 'base'].copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=170)
    ax.plot(g['date'], g['pred_fill_base'] * 100.0, color='#94a3b8', linewidth=1.6, label='Mevcut recursive temel yol')
    ax.plot(g['date'], g['pred_fill_direct'] * 100.0, color='#60a5fa', linewidth=1.6, label='Direct yol')
    ax.plot(g['date'], g['pred_fill_blend_reconciled'] * 100.0, color='#0f766e', linewidth=2.2, label='Direct-recursive blend')
    ax.fill_between(g['date'], g['pred_fill_low_conf80'] * 100.0, g['pred_fill_high_conf80'] * 100.0, color='#99f6e4', alpha=0.20)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('Temel senaryoda direct-recursive blend ve conformal band')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    forward = load_module(FORWARD_SCRIPT, 'forward_for_bestmethod_dirrec')
    train_df = load_training_frame(forward)
    ds_map = precompute_direct_datasets(train_df)

    best_family, family_metrics, best_direct_preds = choose_best_direct_family(train_df, ds_map)
    base_calib = pd.read_csv(BASE_CALIB_PATH, parse_dates=['date'])
    weight_grid_df, best_weights_df = optimize_blend_weights(best_direct_preds, base_calib)
    weight_map = interpolate_anchor_map(best_weights_df.sort_values('horizon_months'), 'weight_base_recursive')
    direct_models = fit_full_direct_models(ds_map, best_family)
    conformal_df = compute_conformal_halfwidths(best_direct_preds, base_calib, best_weights_df)
    q_map = interpolate_anchor_map(conformal_df, 'conformal_halfwidth')
    future_paths, future_summary = build_future_paths(forward, train_df, direct_models, weight_map, q_map)

    current_pref = pd.read_csv(CURRENT_PREF_SUMMARY_PATH)
    compare = future_summary.merge(current_pref[['scenario', 'p50_endpoint_2040_12_pct']], on='scenario', how='left')
    compare['delta_vs_current_preferred_p50_pp'] = compare['end_fill_2040_12_pct'] - compare['p50_endpoint_2040_12_pct']

    family_metrics.to_csv(OUT_DIR / 'direct_family_anchor_metrics.csv', index=False)
    best_direct_preds.to_csv(OUT_DIR / 'best_direct_anchor_predictions.csv', index=False)
    weight_grid_df.to_csv(OUT_DIR / 'dirrec_weight_grid.csv', index=False)
    best_weights_df.to_csv(OUT_DIR / 'dirrec_best_weights_by_anchor.csv', index=False)
    conformal_df.to_csv(OUT_DIR / 'dirrec_conformal_anchor_intervals.csv', index=False)
    future_paths.to_csv(OUT_DIR / 'dirrec_blended_scenario_projection_monthly_2026_2040.csv', index=False)
    future_summary.to_csv(OUT_DIR / 'dirrec_blended_scenario_summary_2026_2040.csv', index=False)
    compare.to_csv(OUT_DIR / 'dirrec_compare_vs_current_preferred.csv', index=False)

    plot_anchor_metrics(best_weights_df, FIG_DIR / 'dirrec_anchor_method_comparison.png')
    plot_weight_map(weight_map, FIG_DIR / 'dirrec_weight_map.png')
    plot_scenarios(future_paths, train_df, FIG_DIR / 'dirrec_scenarios_2026_2040.png')
    plot_base_compare(future_paths, FIG_DIR / 'dirrec_base_compare_conformal.png')

    summary = {
        'best_direct_family': best_family,
        'mean_anchor_rmse_best_family_pp': float(family_metrics[family_metrics['family'] == best_family]['rmse_pp'].mean()),
        'mean_blend_rmse_anchor_pp': float(best_weights_df['blend_rmse_pp'].mean()),
        'mean_base_rmse_anchor_pp': float(best_weights_df['base_rmse_pp'].mean()),
        'mean_direct_rmse_anchor_pp': float(best_weights_df['direct_rmse_pp'].mean()),
        'method_note': 'Direct-recursive forecast averaging with horizon-specific conformal intervals.',
    }
    (OUT_DIR / 'dirrec_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
