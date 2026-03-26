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
PREFERRED_MONTHLY_PATH = ROOT / 'output' / 'istanbul_preferred_projection_2040' / 'deterministic_monthly.csv'
PREFERRED_SUMMARY_PATH = ROOT / 'output' / 'istanbul_preferred_projection_2040' / 'deterministic_summary.csv'
PREFERRED_PROB_PATH = ROOT / 'output' / 'istanbul_preferred_projection_2040' / 'probabilistic_summary.csv'
OUT_DIR = ROOT / 'output' / 'istanbul_bestmethod_rectify_conformal_2040'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_HORIZONS = [1, 3, 6, 12]
ALL_HORIZONS = list(range(1, 13))
MIN_RESID_TRAIN = 36
ALPHA = 0.20
WEIGHT_GRID = np.linspace(0.0, 1.0, 21)
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

RECTIFY_FEATURES = [
    'base_pred',
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


def make_model(family: str):
    if family == 'ridge':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', RidgeCV(alphas=np.logspace(-4, 4, 33))),
        ])
    if family == 'hist_gbm':
        return HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=160,
            min_samples_leaf=8,
            l2_regularization=0.1,
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


def build_anchor_residual_dataset(train_df: pd.DataFrame) -> pd.DataFrame:
    calib = pd.read_csv(BASE_CALIB_PATH, parse_dates=['date'])
    train_idx = train_df.set_index('date')
    rows = []
    for rec in calib.itertuples(index=False):
        horizon = int(rec.horizon_months)
        if horizon not in ANCHOR_HORIZONS:
            continue
        target_date = pd.Timestamp(rec.date)
        origin_date = target_date - pd.DateOffset(months=horizon)
        if target_date not in train_idx.index or origin_date not in train_idx.index:
            continue
        target_row = train_idx.loc[target_date]
        origin_row = train_idx.loc[origin_date]
        base_pred = float(rec.pred_fill_ensemble_phys)
        actual_fill = float(rec.actual_fill)
        rows.append({
            'date': target_date,
            'origin_date': origin_date,
            'horizon_months': horizon,
            'base_pred': base_pred,
            'actual_fill': actual_fill,
            'residual_target': actual_fill - base_pred,
            'origin_fill': float(origin_row['weighted_total_fill']),
            'origin_fill_lag1': float(origin_row['weighted_total_fill_lag1']),
            'origin_delta_fill': float(origin_row['delta_fill']),
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
        })
    return pd.DataFrame(rows).sort_values(['date', 'horizon_months']).reset_index(drop=True)


def backtest_family(anchor_ds: pd.DataFrame, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = []
    for row in anchor_ds.itertuples(index=False):
        train_mask = anchor_ds['date'] < row.date
        train_part = anchor_ds.loc[train_mask]
        if len(train_part) < MIN_RESID_TRAIN:
            continue
        model = make_model(family)
        model.fit(train_part[RECTIFY_FEATURES], train_part['residual_target'])
        feat = pd.DataFrame([{col: getattr(row, col) for col in RECTIFY_FEATURES}])
        resid_hat = float(model.predict(feat[RECTIFY_FEATURES])[0])
        rectified = float(np.clip(row.base_pred + resid_hat, 0.0, 1.0))
        preds.append({
            'family': family,
            'date': row.date,
            'horizon_months': int(row.horizon_months),
            'actual_fill': float(row.actual_fill),
            'base_pred': float(row.base_pred),
            'pred_residual': resid_hat,
            'pred_fill_rectified': rectified,
        })
    pred_df = pd.DataFrame(preds)
    metrics = []
    for horizon, g in pred_df.groupby('horizon_months'):
        metrics.append({
            'family': family,
            'horizon_months': int(horizon),
            'base_rmse_pp': rmse_pp(g['actual_fill'].to_numpy(dtype=float), g['base_pred'].to_numpy(dtype=float)),
            'rectified_rmse_pp': rmse_pp(g['actual_fill'].to_numpy(dtype=float), g['pred_fill_rectified'].to_numpy(dtype=float)),
            'n_predictions': int(len(g)),
        })
    return pd.DataFrame(metrics), pred_df


def choose_best_family(anchor_ds: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    all_metrics = []
    all_preds = []
    for family in ['ridge', 'hist_gbm']:
        metric_df, pred_df = backtest_family(anchor_ds, family)
        all_metrics.append(metric_df)
        all_preds.append(pred_df)
    metrics = pd.concat(all_metrics, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    summary = metrics.groupby('family', as_index=False).agg(
        mean_base_rmse_pp=('base_rmse_pp', 'mean'),
        mean_rectified_rmse_pp=('rectified_rmse_pp', 'mean'),
    )
    best_family = str(summary.sort_values('mean_rectified_rmse_pp').iloc[0]['family'])
    return best_family, metrics, preds[preds['family'] == best_family].copy()


def fit_final_model(anchor_ds: pd.DataFrame, family: str):
    model = make_model(family)
    model.fit(anchor_ds[RECTIFY_FEATURES], anchor_ds['residual_target'])
    return model


def interpolate_anchor_map(anchor_df: pd.DataFrame, value_col: str) -> dict[int, float]:
    xs = anchor_df['horizon_months'].to_numpy(dtype=float)
    ys = anchor_df[value_col].to_numpy(dtype=float)
    return {h: float(np.interp(h, xs, ys)) for h in ALL_HORIZONS}


def compute_conformal(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, g in pred_df.groupby('horizon_months'):
        abs_resid = np.abs(g['actual_fill'].to_numpy(dtype=float) - g['pred_fill_rectified_blend'].to_numpy(dtype=float))
        q = float(np.quantile(abs_resid, 1.0 - ALPHA))
        cov = float(np.mean((g['actual_fill'] >= (g['pred_fill_rectified_blend'] - q)) & (g['actual_fill'] <= (g['pred_fill_rectified_blend'] + q))) * 100.0)
        rows.append({
            'horizon_months': int(horizon),
            'conformal_halfwidth': q,
            'empirical_coverage_pct': cov,
            'n_predictions': int(len(g)),
        })
    return pd.DataFrame(rows).sort_values('horizon_months').reset_index(drop=True)


def optimize_correction_weights(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, g in pred_df.groupby('horizon_months'):
        y = g['actual_fill'].to_numpy(dtype=float)
        base = g['base_pred'].to_numpy(dtype=float)
        rect = g['pred_fill_rectified'].to_numpy(dtype=float)
        best_rmse = None
        best_w = None
        best_pred = None
        for w in WEIGHT_GRID:
            pred = np.clip(base + w * (rect - base), 0.0, 1.0)
            rmse = rmse_pp(y, pred)
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_w = float(w)
                best_pred = pred
        rows.append({
            'horizon_months': int(horizon),
            'base_rmse_pp': rmse_pp(y, base),
            'rectified_rmse_pp': rmse_pp(y, rect),
            'blended_rmse_pp': best_rmse,
            'correction_weight': best_w,
            'n_predictions': int(len(g)),
            'mean_blended_fill': float(np.mean(best_pred)),
        })
    return pd.DataFrame(rows).sort_values('horizon_months').reset_index(drop=True)


def apply_correction_weights(pred_df: pd.DataFrame, weight_map: dict[int, float]) -> pd.DataFrame:
    out = pred_df.copy()
    out['correction_weight'] = out['horizon_months'].map(weight_map)
    out['pred_fill_rectified_blend'] = np.clip(
        out['base_pred'] + out['correction_weight'] * (out['pred_fill_rectified'] - out['base_pred']),
        0.0,
        1.0,
    )
    return out


def build_future_paths(forward, train_df: pd.DataFrame, model, q_map: dict[int, float], weight_map: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_proj = pd.read_csv(PREFERRED_MONTHLY_PATH, parse_dates=['date'])
    preferred_summary = pd.read_csv(PREFERRED_SUMMARY_PATH)
    preferred_prob = pd.read_csv(PREFERRED_PROB_PATH)

    clim = forward.monthly_climatology(train_df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    cfg_map = {cfg.scenario: cfg for cfg in (forward.build_scenarios() + forward.build_transfer_scenarios())}

    path_rows = []
    for scenario in sorted(base_proj['scenario'].unique()):
        if scenario not in cfg_map:
            continue
        cfg = cfg_map[scenario]
        future_raw = forward.build_future_exog(train_df, cfg, clim, demand_relief, transfer_share_anchor_pct)
        future_raw = future_raw[future_raw['date'] >= pd.Timestamp('2026-01-01')].copy().reset_index(drop=True)
        future = augment_future_exog(train_df, future_raw)
        base_s = base_proj[base_proj['scenario'] == scenario].sort_values('date').reset_index(drop=True)
        hist_fill = train_df['weighted_total_fill'].tolist()
        scen_rows = []
        for block_start in range(0, len(future), 12):
            block = future.iloc[block_start:block_start + 12].copy().reset_index(drop=True)
            origin_fill = float(hist_fill[-1])
            origin_fill_lag1 = float(hist_fill[-2])
            origin_delta = origin_fill - origin_fill_lag1
            block_preds = []
            for h in range(1, len(block) + 1):
                target_row = block.iloc[h - 1]
                base_row = base_s.iloc[block_start + h - 1]
                base_pred = float(base_row['pred_fill_ensemble'])
                feat = pd.DataFrame([{
                    'base_pred': base_pred,
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
                resid_hat = float(model.predict(feat[RECTIFY_FEATURES])[0])
                full_rectified_base = float(np.clip(base_pred + resid_hat, 0.0, 1.0))
                corr_w = weight_map[h]
                rectified_base = float(np.clip(base_pred + corr_w * (full_rectified_base - base_pred), 0.0, 1.0))
                corr_pp = float(base_row['correction_pp'])
                rectified_reconciled = float(np.clip(rectified_base + corr_pp / 100.0, 0.0, 1.0))
                q = q_map[h]
                scen_rows.append({
                    'date': target_row['date'],
                    'scenario': scenario,
                    'pred_fill_base': base_pred,
                    'pred_residual_correction': resid_hat,
                    'pred_fill_rectified_full': full_rectified_base,
                    'pred_fill_rectified': rectified_base,
                    'pred_fill_rectified_reconciled': rectified_reconciled,
                    'pred_fill_low_conf80': float(np.clip(rectified_reconciled - q, 0.0, 1.0)),
                    'pred_fill_high_conf80': float(np.clip(rectified_reconciled + q, 0.0, 1.0)),
                    'rain_model_mm': float(target_row['rain_model_mm']),
                    'et0_mm_month': float(target_row['et0_mm_month']),
                    'consumption_mean_monthly': float(target_row['consumption_mean_monthly']),
                    'correction_pp': corr_pp,
                    'horizon_in_block': h,
                    'conformal_halfwidth': q,
                    'correction_weight': corr_w,
                })
                block_preds.append(rectified_reconciled)
            hist_fill.extend(block_preds)
        path_rows.append(pd.DataFrame(scen_rows))

    paths = pd.concat(path_rows, ignore_index=True)
    summary = (
        paths.groupby('scenario', as_index=False)
        .agg(
            mean_fill_2026_2040_pct=('pred_fill_rectified_reconciled', lambda s: float(s.mean() * 100.0)),
            min_fill_2026_2040_pct=('pred_fill_rectified_reconciled', lambda s: float(s.min() * 100.0)),
            end_fill_2040_12_pct=('pred_fill_rectified_reconciled', lambda s: float(s.iloc[-1] * 100.0)),
            end_p10_2040_12_pct=('pred_fill_low_conf80', lambda s: float(s.iloc[-1] * 100.0)),
            end_p90_2040_12_pct=('pred_fill_high_conf80', lambda s: float(s.iloc[-1] * 100.0)),
        )
    )
    compare = summary.merge(
        preferred_summary[['scenario', 'end_fill_2040_12_pct']],
        on='scenario',
        how='left',
        suffixes=('', '_current_det'),
    ).merge(
        preferred_prob[['scenario', 'p50_endpoint_2040_12_pct']],
        on='scenario',
        how='left',
    )
    compare['delta_vs_current_det_pp'] = compare['end_fill_2040_12_pct'] - compare['end_fill_2040_12_pct_current_det']
    compare['delta_vs_current_prob_p50_pp'] = compare['end_fill_2040_12_pct'] - compare['p50_endpoint_2040_12_pct']
    return paths, compare


def plot_anchor_metrics(metric_df: pd.DataFrame, best_family: str, out_path: Path) -> None:
    g = metric_df[metric_df['family'] == best_family].sort_values('horizon_months')
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=170)
    x = np.arange(len(g))
    ax.bar(x - 0.16, g['base_rmse_pp'], width=0.32, color='#94a3b8', label='Mevcut tercih edilen yol')
    ax.bar(x + 0.16, g['rectified_rmse_pp'], width=0.32, color='#0f766e', label='Rectify düzeltmeli yol')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in g['horizon_months']])
    ax.set_xlabel('Ufuk (ay)')
    ax.set_ylabel('RMSE (yüzde puan)')
    ax.set_title('Rectify ile anchor horizon hata karşılaştırması')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scenarios(paths: pd.DataFrame, train_df: pd.DataFrame, out_path: Path) -> None:
    hist = train_df[train_df['date'] >= '2018-01-01'].copy()
    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=170)
    ax.plot(hist['date'], hist['weighted_total_fill'] * 100.0, color='#111827', linewidth=2.0, label='Gözlenen')
    for scenario in PRIMARY_SCENARIOS:
        g = paths[paths['scenario'] == scenario].copy()
        ax.plot(g['date'], g['pred_fill_rectified_reconciled'] * 100.0, color=SCENARIO_COLORS[scenario], linewidth=2.0, label=SCENARIO_LABELS[scenario])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('Rectify + conformal ile 2026-2040 projeksiyon yolları')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_base_compare(paths: pd.DataFrame, base_proj: pd.DataFrame, out_path: Path) -> None:
    old = base_proj[base_proj['scenario'] == 'base'].sort_values('date').copy()
    new = paths[paths['scenario'] == 'base'].sort_values('date').copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=170)
    ax.plot(old['date'], old['pred_fill_reconciled'] * 100.0, color='#94a3b8', linewidth=1.6, label='Mevcut tercih edilen yol')
    ax.plot(new['date'], new['pred_fill_rectified_reconciled'] * 100.0, color='#0f766e', linewidth=2.1, label='Rectify düzeltmeli yol')
    ax.fill_between(new['date'], new['pred_fill_low_conf80'] * 100.0, new['pred_fill_high_conf80'] * 100.0, color='#99f6e4', alpha=0.20)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('Temel senaryoda rectify düzeltmesi ve conformal band')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    forward = load_module(FORWARD_SCRIPT, 'forward_for_bestmethod_rectify')
    train_df = load_training_frame(forward)
    anchor_ds = build_anchor_residual_dataset(train_df)
    best_family, metric_df, best_pred_df = choose_best_family(anchor_ds)
    blend_df = optimize_correction_weights(best_pred_df)
    weight_map = interpolate_anchor_map(blend_df, 'correction_weight')
    best_pred_df = apply_correction_weights(best_pred_df, weight_map)
    final_model = fit_final_model(anchor_ds, best_family)
    conformal_df = compute_conformal(best_pred_df)
    q_map = interpolate_anchor_map(conformal_df, 'conformal_halfwidth')
    paths, compare = build_future_paths(forward, train_df, final_model, q_map, weight_map)

    metric_df.to_csv(OUT_DIR / 'rectify_anchor_metrics.csv', index=False)
    anchor_ds.to_csv(OUT_DIR / 'rectify_anchor_dataset.csv', index=False)
    best_pred_df.to_csv(OUT_DIR / 'rectify_anchor_predictions.csv', index=False)
    blend_df.to_csv(OUT_DIR / 'rectify_correction_weight_grid.csv', index=False)
    conformal_df.to_csv(OUT_DIR / 'rectify_conformal_anchor_intervals.csv', index=False)
    paths.to_csv(OUT_DIR / 'rectify_scenario_projection_monthly_2026_2040.csv', index=False)
    compare.to_csv(OUT_DIR / 'rectify_compare_vs_current_preferred.csv', index=False)

    summary = {
        'best_family': best_family,
        'mean_base_rmse_anchor_pp': float(metric_df[metric_df['family'] == best_family]['base_rmse_pp'].mean()),
        'mean_rectified_rmse_anchor_pp': float(metric_df[metric_df['family'] == best_family]['rectified_rmse_pp'].mean()),
        'mean_blended_rmse_anchor_pp': float(blend_df['blended_rmse_pp'].mean()),
        'method_note': 'Rectify residual correction on preferred ensemble path with horizon-specific conformal intervals.',
    }
    (OUT_DIR / 'rectify_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    base_proj = pd.read_csv(PREFERRED_MONTHLY_PATH, parse_dates=['date'])
    plot_anchor_metrics(metric_df, best_family, FIG_DIR / 'rectify_anchor_method_comparison.png')
    plot_scenarios(paths, train_df, FIG_DIR / 'rectify_scenarios_2026_2040.png')
    plot_base_compare(paths, base_proj, FIG_DIR / 'rectify_base_compare_conformal.png')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
