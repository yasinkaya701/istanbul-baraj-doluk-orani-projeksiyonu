#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path('/Users/yasinkaya/Hackhaton')
FORWARD_SCRIPT = ROOT / 'scripts' / 'build_istanbul_forward_projection_2040.py'
WB_SCRIPT = ROOT / 'scripts' / 'build_istanbul_water_balance_v5_sourcerain.py'
OUT_DIR = ROOT / 'output' / 'istanbul_hybrid_physics_sourcerain_ensemble_2040'
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
MIN_TRAIN = 60
FUTURE_START = pd.Timestamp('2026-01-01')


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


def mae_pp(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred) * 100.0)


def sign_accuracy(origin: np.ndarray, actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((pred - origin > 0) == (actual - origin > 0)))


def build_hybrid_recursive_predictions(forward, df: pd.DataFrame, horizons=(3, 6, 12)) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        for i in range(MIN_TRAIN, len(df) - horizon + 1):
            train = df.iloc[:i].copy()
            future = df.iloc[i:i + horizon].copy()
            model = forward.fit_model(train, 'hybrid_ridge')
            sim = forward.simulate_projection(
                train_df=train,
                future_exog=future[[
                    'date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly',
                    'temp_proxy_c', 'rh_proxy_pct', 'vpd_kpa_mean', 'month_sin', 'month_cos'
                ]],
                model=model,
                selected_model='hybrid_ridge',
                interval_by_month={},
                global_interval=(0.0, 0.0),
            )
            rows.append({
                'date': future['date'].iloc[horizon - 1],
                'origin_date': train['date'].iloc[-1],
                'horizon_months': int(horizon),
                'actual_fill': float(future['weighted_total_fill'].iloc[horizon - 1]),
                'pred_fill_hybrid': float(sim['pred_fill'].iloc[-1]),
                'origin_fill': float(train['weighted_total_fill'].iloc[-1]),
            })
    return pd.DataFrame(rows)


def build_wb_recursive_predictions(wb, df: pd.DataFrame, horizons=(3, 6, 12)) -> pd.DataFrame:
    context = wb.compute_system_context()
    share_by_year, _ = wb.load_transfer_share_by_year()
    rows = []
    for horizon in horizons:
        for i in range(MIN_TRAIN, len(df) - horizon + 1):
            train = df.iloc[:i].copy()
            future = df.iloc[i:i + horizon].copy()
            train_comp = wb.component_frame(train, context)
            model, month_bias, fit_df = wb.fit_water_balance_model(train_comp)
            transfer_effectiveness = wb.estimate_transfer_effectiveness(train, fit_df, share_by_year)
            sim = wb.simulate_path(
                history_df=train,
                future_exog=future[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']],
                model=model,
                month_bias=month_bias,
                context=context,
                transfer_share_anchor_pct=0.0,
                transfer_effectiveness=transfer_effectiveness,
                baseline_transfer_share_pct=0.0,
                transfer_end_pct_2040=0.0,
            )
            rows.append({
                'date': future['date'].iloc[horizon - 1],
                'origin_date': train['date'].iloc[-1],
                'horizon_months': int(horizon),
                'actual_fill': float(future['weighted_total_fill'].iloc[horizon - 1]),
                'pred_fill_wb': float(sim['pred_fill'].iloc[-1]),
                'origin_fill': float(train['weighted_total_fill'].iloc[-1]),
            })
    return pd.DataFrame(rows)


def physics_from_weight(weight_hybrid: float, fwd_phys: pd.DataFrame, wb_phys: pd.DataFrame) -> dict[str, float]:
    fwd = fwd_phys[fwd_phys['model'] == 'hybrid_ridge'].copy()
    mapping = {
        'rain_plus10_delta_pp': 'rain_plus10',
        'demand_plus10_delta_pp': 'demand_plus10',
        'et0_plus10_delta_pp': 'et0_plus10',
        'transfer_stress_delta_pp': 'transfer_stress',
    }
    rows = {}
    for f_col, w_scenario in mapping.items():
        rows[w_scenario] = (
            weight_hybrid * float(fwd[f_col].iloc[0])
            + (1.0 - weight_hybrid) * float(wb_phys.loc[wb_phys['scenario'] == w_scenario, 'delta_pp'].iloc[0])
        )
    return rows


def select_weight(calib: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    fwd_phys = pd.read_csv(ROOT / 'output' / 'istanbul_forward_model_benchmark_round2' / 'physical_sanity_checks.csv')
    wb_phys = pd.read_csv(ROOT / 'output' / 'istanbul_water_balance_v5_sourcerain_2040' / 'water_balance_physical_sanity_checks.csv')
    weights = np.linspace(0.0, 1.0, 101)
    rows = []
    best = None
    best_score = None
    best_phys = None
    best_phys_score = None
    one = calib[calib['horizon_months'] == 1].copy()
    rec = calib[calib['horizon_months'] > 1].copy()
    for w in weights:
        pred = w * calib['pred_fill_hybrid'].to_numpy(dtype=float) + (1.0 - w) * calib['pred_fill_wb'].to_numpy(dtype=float)
        actual = calib['actual_fill'].to_numpy(dtype=float)
        rmse = rmse_pp(actual, pred)
        mae = mae_pp(actual, pred)
        one_pred = w * one['pred_fill_hybrid'].to_numpy(dtype=float) + (1.0 - w) * one['pred_fill_wb'].to_numpy(dtype=float)
        one_rmse = rmse_pp(one['actual_fill'].to_numpy(dtype=float), one_pred)
        rec_rmses = []
        for _, g in rec.groupby('horizon_months'):
            g_pred = w * g['pred_fill_hybrid'].to_numpy(dtype=float) + (1.0 - w) * g['pred_fill_wb'].to_numpy(dtype=float)
            rec_rmses.append(rmse_pp(g['actual_fill'].to_numpy(dtype=float), g_pred))
        mean_recursive_rmse = float(np.mean(rec_rmses))
        physics = physics_from_weight(float(w), fwd_phys, wb_phys)
        physics_pass_count = int(
            (physics['rain_plus10'] > 0.0)
            + (physics['demand_plus10'] < 0.0)
            + (physics['et0_plus10'] < 0.0)
            + (physics['transfer_stress'] < 0.0)
        )
        rows.append({
            'weight_hybrid': float(w),
            'weight_wb': float(1.0 - w),
            'rmse_pp': rmse,
            'mae_pp': mae,
            'one_step_rmse_pp': one_rmse,
            'mean_recursive_rmse_pp': mean_recursive_rmse,
            'physics_pass_count': physics_pass_count,
            'rain_plus10_delta_pp': float(physics['rain_plus10']),
            'demand_plus10_delta_pp': float(physics['demand_plus10']),
            'et0_plus10_delta_pp': float(physics['et0_plus10']),
            'transfer_stress_delta_pp': float(physics['transfer_stress']),
        })
        if best_score is None or rmse < best_score:
            best = float(w)
            best_score = rmse
        if physics_pass_count == 4 and (best_phys_score is None or rmse < best_phys_score):
            best_phys = float(w)
            best_phys_score = rmse
    return float(best), float(best_phys), pd.DataFrame(rows)


def build_calibration_dataset(forward, wb) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    fwd_df = forward.load_training_frame().copy()
    wb_df = wb.load_training_frame(wb.compute_system_context()).copy()
    common_dates = sorted(set(fwd_df['date']).intersection(set(wb_df['date'])))
    fwd_df = fwd_df[fwd_df['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    wb_df = wb_df[wb_df['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)

    _, _, pred_frames = forward.evaluate_models(fwd_df)
    hybrid_one = pred_frames['hybrid_ridge'].rename(columns={'actual': 'actual_fill', 'pred': 'pred_fill_hybrid'})[['date', 'actual_fill', 'pred_fill_hybrid']]
    _, wb_one = wb.one_step_walkforward(wb_df, wb.compute_system_context(), wb.load_transfer_share_by_year()[0])
    wb_one = wb_one[['date', 'actual_fill', 'pred_fill']].rename(columns={'pred_fill': 'pred_fill_wb'})
    one = hybrid_one.merge(wb_one, on=['date', 'actual_fill'], how='inner')
    one['horizon_months'] = 1
    one['origin_fill'] = np.nan

    hybrid_rec = build_hybrid_recursive_predictions(forward, fwd_df)
    wb_rec = build_wb_recursive_predictions(wb, wb_df)
    rec = hybrid_rec.merge(wb_rec, on=['date', 'origin_date', 'horizon_months', 'actual_fill', 'origin_fill'], how='inner')

    calib = pd.concat([
        one[['date', 'horizon_months', 'actual_fill', 'pred_fill_hybrid', 'pred_fill_wb', 'origin_fill']],
        rec[['date', 'horizon_months', 'actual_fill', 'pred_fill_hybrid', 'pred_fill_wb', 'origin_fill']],
    ], ignore_index=True)
    meta = {
        'train_start': str(fwd_df['date'].min().date()),
        'train_end': str(fwd_df['date'].max().date()),
        'n_one_step': int(len(one)),
        'n_recursive': int(len(rec)),
    }
    return calib, one, rec, fwd_df, meta


def build_metrics(calib: pd.DataFrame, weight_hybrid: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for horizon, g in calib.groupby('horizon_months'):
        pred = weight_hybrid * g['pred_fill_hybrid'].to_numpy(dtype=float) + (1.0 - weight_hybrid) * g['pred_fill_wb'].to_numpy(dtype=float)
        actual = g['actual_fill'].to_numpy(dtype=float)
        if horizon == 1:
            origin = np.r_[actual[0], actual[:-1]]
        else:
            origin = g['origin_fill'].to_numpy(dtype=float)
        parts.append({
            'model': 'hybrid_physics_ensemble',
            'horizon_months': int(horizon),
            'rmse_pp': rmse_pp(actual, pred),
            'mae_pp': mae_pp(actual, pred),
            'direction_accuracy': sign_accuracy(origin, actual, pred),
            'n_predictions': int(len(g)),
        })
    metrics = pd.DataFrame(parts).sort_values('horizon_months').reset_index(drop=True)
    one = metrics[metrics['horizon_months'] == 1].rename(columns={'rmse_pp': 'one_step_rmse_pp', 'mae_pp': 'one_step_mae_pp', 'direction_accuracy': 'delta_direction_accuracy'})[['model', 'one_step_rmse_pp', 'one_step_mae_pp', 'delta_direction_accuracy', 'n_predictions']]
    return one, metrics


def scenario_paths(forward, wb, weight_hybrid: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fwd_df = forward.load_training_frame().copy()
    wb_context = wb.compute_system_context()
    wb_df = wb.load_training_frame(wb_context).copy()
    common_dates = sorted(set(fwd_df['date']).intersection(set(wb_df['date'])))
    fwd_df = fwd_df[fwd_df['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
    wb_df = wb_df[wb_df['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)

    clim = forward.monthly_climatology(fwd_df)
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    wb_share_by_year, wb_anchor_share_pct = wb.load_transfer_share_by_year()
    wb_comp = wb.component_frame(wb_df, wb_context)
    wb_model, wb_month_bias, wb_fit_df = wb.fit_water_balance_model(wb_comp)
    wb_transfer_eff = wb.estimate_transfer_effectiveness(wb_df, wb_fit_df, wb_share_by_year)
    hybrid_model = forward.fit_model(fwd_df, 'hybrid_ridge')

    rows_h = []
    rows_w = []
    rows_e = []
    for cfg in forward.build_scenarios() + forward.build_transfer_scenarios():
        neutral_cfg = cfg if float(cfg.transfer_end_pct_2040) == 0.0 else type(cfg)(**{**cfg.__dict__, 'transfer_end_pct_2040': 0.0})
        future = forward.build_future_exog(fwd_df, neutral_cfg, clim, demand_relief, transfer_share_anchor_pct=0.0)
        sim_h = forward.simulate_projection(
            train_df=fwd_df,
            future_exog=future[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly', 'temp_proxy_c', 'rh_proxy_pct', 'vpd_kpa_mean', 'month_sin', 'month_cos']],
            model=hybrid_model,
            selected_model='hybrid_ridge',
            interval_by_month={},
            global_interval=(0.0, 0.0),
        )
        sim_w = wb.simulate_path(
            history_df=wb_df,
            future_exog=future[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']],
            model=wb_model,
            month_bias=wb_month_bias,
            context=wb_context,
            transfer_share_anchor_pct=wb_anchor_share_pct,
            transfer_effectiveness=wb_transfer_eff,
            baseline_transfer_share_pct=wb_anchor_share_pct,
            transfer_end_pct_2040=float(cfg.transfer_end_pct_2040),
        )
        sim_h['scenario'] = cfg.scenario
        sim_w['scenario'] = cfg.scenario
        rows_h.append(sim_h[['date', 'scenario', 'pred_fill', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']])
        rows_w.append(sim_w[['date', 'scenario', 'pred_fill', 'rain_model_mm', 'et0_mm_month']].assign(consumption_mean_monthly=future['consumption_mean_monthly'].to_numpy(dtype=float)))
        ens = pd.DataFrame({
            'date': sim_h['date'],
            'scenario': cfg.scenario,
            'pred_fill_hybrid': sim_h['pred_fill'],
            'pred_fill_wb': sim_w['pred_fill'],
            'pred_fill_ensemble': weight_hybrid * sim_h['pred_fill'] + (1.0 - weight_hybrid) * sim_w['pred_fill'],
            'rain_model_mm': sim_h['rain_model_mm'],
            'et0_mm_month': sim_h['et0_mm_month'],
            'consumption_mean_monthly': future['consumption_mean_monthly'].to_numpy(dtype=float),
        })
        rows_e.append(ens)
    future_mask_h = pd.concat(rows_h, ignore_index=True)['date'] >= FUTURE_START
    future_mask_w = pd.concat(rows_w, ignore_index=True)['date'] >= FUTURE_START
    future_mask_e = pd.concat(rows_e, ignore_index=True)['date'] >= FUTURE_START
    return (
        pd.concat(rows_h, ignore_index=True).loc[future_mask_h].reset_index(drop=True),
        pd.concat(rows_w, ignore_index=True).loc[future_mask_w].reset_index(drop=True),
        pd.concat(rows_e, ignore_index=True).loc[future_mask_e].reset_index(drop=True),
    )


def build_summary(ensemble_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = ensemble_df[
        ensemble_df['scenario'].isin(PRIMARY_SCENARIOS) & (ensemble_df['date'] >= FUTURE_START)
    ].copy()
    for scenario, g in target.groupby('scenario'):
        g = g.sort_values('date')
        rows.append({
            'scenario': scenario,
            'mean_fill_2026_2040_pct': float(g['pred_fill_ensemble'].mean() * 100.0),
            'min_fill_2026_2040_pct': float(g['pred_fill_ensemble'].min() * 100.0),
            'end_fill_2040_12_pct': float(g.iloc[-1]['pred_fill_ensemble'] * 100.0),
            'first_below_40_date': str(g.loc[g['pred_fill_ensemble'] < 0.40, 'date'].iloc[0].date()) if (g['pred_fill_ensemble'] < 0.40).any() else '',
            'first_below_30_date': str(g.loc[g['pred_fill_ensemble'] < 0.30, 'date'].iloc[0].date()) if (g['pred_fill_ensemble'] < 0.30).any() else '',
        })
    return pd.DataFrame(rows).sort_values('mean_fill_2026_2040_pct', ascending=False).reset_index(drop=True)


def build_physical_sanity(forward, wb, weight_hybrid: float, model_name: str) -> pd.DataFrame:
    fwd_phys = pd.read_csv(ROOT / 'output' / 'istanbul_forward_model_benchmark_round2' / 'physical_sanity_checks.csv')
    wb_phys = pd.read_csv(ROOT / 'output' / 'istanbul_water_balance_v5_sourcerain_2040' / 'water_balance_physical_sanity_checks.csv')
    fwd = fwd_phys[fwd_phys['model'] == 'hybrid_ridge'].copy()
    base_endpoint = float(fwd['base_endpoint_2040_pct'].iloc[0])
    rows = []
    for w_scenario, delta in physics_from_weight(weight_hybrid, fwd_phys, wb_phys).items():
        rows.append({
            'model': model_name,
            'scenario': w_scenario,
            'base_endpoint_2040_pct': base_endpoint,
            'scenario_endpoint_2040_pct': base_endpoint + delta,
            'delta_pp': delta,
        })
    return pd.DataFrame(rows)


def plot_scenarios(history_df: pd.DataFrame, ensemble_df: pd.DataFrame, out_path: Path) -> None:
    hist = history_df[history_df['date'] >= '2018-01-01'].copy()
    fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=170)
    ax.plot(hist['date'], hist['weighted_total_fill'] * 100.0, color='#111827', linewidth=2.0, label='Gözlenen')
    for scenario in PRIMARY_SCENARIOS:
        g = ensemble_df[ensemble_df['scenario'] == scenario].copy()
        ax.plot(g['date'], g['pred_fill_ensemble'] * 100.0, color=SCENARIO_COLORS[scenario], linewidth=2.0, label=SCENARIO_LABELS[scenario])
    ax.axvline(pd.Timestamp('2026-01-01'), color='#6b7280', linestyle='--', linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('Hibrit-fizik ensemble ile 2026-2040 projeksiyon yolları')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_base_compare(ensemble_df: pd.DataFrame, out_path: Path) -> None:
    g = ensemble_df[ensemble_df['scenario'] == 'base'].copy()
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=170)
    ax.plot(g['date'], g['pred_fill_hybrid'] * 100.0, color='#2563eb', linewidth=1.8, label='Hibrit Ridge')
    ax.plot(g['date'], g['pred_fill_wb'] * 100.0, color='#dc2626', linewidth=1.8, label='Su bütçesi v3')
    ax.plot(g['date'], g['pred_fill_ensemble'] * 100.0, color='#111827', linewidth=2.2, label='Ensemble')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Toplam doluluk (%)')
    ax.set_title('Temel senaryoda model birleşimi')
    ax.grid(True, axis='y', alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = OUT_DIR / 'figures'
    figs.mkdir(parents=True, exist_ok=True)

    forward = load_module(FORWARD_SCRIPT, 'forward_for_hp_ensemble')
    wb = load_module(WB_SCRIPT, 'wb_v5_sourcerain_for_hp_ensemble')

    calib, one_df, rec_df, history_df, meta = build_calibration_dataset(forward, wb)
    weight_hybrid, weight_hybrid_phys, weight_grid = select_weight(calib)
    one_metrics, recursive_metrics = build_metrics(calib, weight_hybrid)
    phys_one_metrics, phys_recursive_metrics = build_metrics(calib, weight_hybrid_phys)
    scenario_h, scenario_w, scenario_e = scenario_paths(forward, wb, weight_hybrid)
    _, _, scenario_e_phys = scenario_paths(forward, wb, weight_hybrid_phys)
    summary = build_summary(scenario_e)
    summary_phys = build_summary(scenario_e_phys)
    sanity = build_physical_sanity(forward, wb, weight_hybrid, 'hybrid_physics_ensemble')
    sanity_phys = build_physical_sanity(forward, wb, weight_hybrid_phys, 'hybrid_physics_ensemble_phys')

    calib['pred_fill_ensemble'] = weight_hybrid * calib['pred_fill_hybrid'] + (1.0 - weight_hybrid) * calib['pred_fill_wb']
    calib['pred_fill_ensemble_phys'] = weight_hybrid_phys * calib['pred_fill_hybrid'] + (1.0 - weight_hybrid_phys) * calib['pred_fill_wb']
    weight_grid.to_csv(OUT_DIR / 'ensemble_weight_grid.csv', index=False)
    calib.to_csv(OUT_DIR / 'ensemble_calibration_samples.csv', index=False)
    one_metrics.to_csv(OUT_DIR / 'ensemble_one_step_metrics.csv', index=False)
    recursive_metrics.to_csv(OUT_DIR / 'ensemble_recursive_metrics.csv', index=False)
    phys_one_metrics.to_csv(OUT_DIR / 'ensemble_phys_one_step_metrics.csv', index=False)
    phys_recursive_metrics.to_csv(OUT_DIR / 'ensemble_phys_recursive_metrics.csv', index=False)
    scenario_h.to_csv(OUT_DIR / 'hybrid_ridge_scenario_projection_monthly_2026_2040.csv', index=False)
    scenario_w.to_csv(OUT_DIR / 'water_balance_v5_sourcerain_scenario_projection_monthly_2026_2040.csv', index=False)
    scenario_e.to_csv(OUT_DIR / 'ensemble_scenario_projection_monthly_2026_2040.csv', index=False)
    scenario_e_phys.to_csv(OUT_DIR / 'ensemble_phys_scenario_projection_monthly_2026_2040.csv', index=False)
    summary.to_csv(OUT_DIR / 'ensemble_scenario_summary_2026_2040.csv', index=False)
    summary_phys.to_csv(OUT_DIR / 'ensemble_phys_scenario_summary_2026_2040.csv', index=False)
    sanity.to_csv(OUT_DIR / 'ensemble_physical_sanity_checks.csv', index=False)
    sanity_phys.to_csv(OUT_DIR / 'ensemble_phys_physical_sanity_checks.csv', index=False)

    benchmark = pd.read_csv(ROOT / 'output' / 'istanbul_forward_model_benchmark_round2' / 'model_selection_scorecard.csv')
    benchmark = benchmark[['model', 'one_step_rmse_pp', 'mean_recursive_rmse_pp', 'physics_pass_count']]
    wb_compare = pd.read_csv(ROOT / 'output' / 'istanbul_water_balance_v5_sourcerain_2040' / 'water_balance_vs_benchmark_models.csv')
    wb_compare = wb_compare[wb_compare['model'] == 'water_balance_v5_sourcerain'][['model', 'one_step_rmse_pp', 'mean_recursive_rmse_pp', 'physics_pass_count']]
    ensemble_compare = pd.DataFrame([{
        'model': 'hybrid_physics_ensemble',
        'one_step_rmse_pp': float(one_metrics.iloc[0]['one_step_rmse_pp']),
        'mean_recursive_rmse_pp': float(recursive_metrics['rmse_pp'].mean()),
        'physics_pass_count': int(((sanity['scenario'] == 'rain_plus10') & (sanity['delta_pp'] > 0)).sum() + ((sanity['scenario'] == 'demand_plus10') & (sanity['delta_pp'] < 0)).sum() + ((sanity['scenario'] == 'et0_plus10') & (sanity['delta_pp'] < 0)).sum() + ((sanity['scenario'] == 'transfer_stress') & (sanity['delta_pp'] < 0)).sum()),
    }, {
        'model': 'hybrid_physics_ensemble_phys',
        'one_step_rmse_pp': float(phys_one_metrics.iloc[0]['one_step_rmse_pp']),
        'mean_recursive_rmse_pp': float(phys_recursive_metrics['rmse_pp'].mean()),
        'physics_pass_count': int(((sanity_phys['scenario'] == 'rain_plus10') & (sanity_phys['delta_pp'] > 0)).sum() + ((sanity_phys['scenario'] == 'demand_plus10') & (sanity_phys['delta_pp'] < 0)).sum() + ((sanity_phys['scenario'] == 'et0_plus10') & (sanity_phys['delta_pp'] < 0)).sum() + ((sanity_phys['scenario'] == 'transfer_stress') & (sanity_phys['delta_pp'] < 0)).sum()),
    }])
    compare = pd.concat([benchmark, wb_compare, ensemble_compare], ignore_index=True)
    compare.to_csv(OUT_DIR / 'ensemble_vs_benchmark_models.csv', index=False)

    plot_scenarios(history_df, scenario_e[scenario_e['scenario'].isin(PRIMARY_SCENARIOS)].copy(), figs / 'ensemble_scenarios_2026_2040.png')
    plot_base_compare(scenario_e, figs / 'ensemble_base_compare_2026_2040.png')
    plot_scenarios(history_df, scenario_e_phys[scenario_e_phys['scenario'].isin(PRIMARY_SCENARIOS)].copy(), figs / 'ensemble_phys_scenarios_2026_2040.png')
    plot_base_compare(scenario_e_phys, figs / 'ensemble_phys_base_compare_2026_2040.png')

    meta.update({
        'weight_hybrid_ridge': weight_hybrid,
        'weight_water_balance_v5_sourcerain': float(1.0 - weight_hybrid),
        'one_step_rmse_pp': float(one_metrics.iloc[0]['one_step_rmse_pp']),
        'mean_recursive_rmse_pp': float(recursive_metrics['rmse_pp'].mean()),
        'weight_hybrid_ridge_phys': weight_hybrid_phys,
        'weight_water_balance_v5_sourcerain_phys': float(1.0 - weight_hybrid_phys),
        'one_step_rmse_pp_phys': float(phys_one_metrics.iloc[0]['one_step_rmse_pp']),
        'mean_recursive_rmse_pp_phys': float(phys_recursive_metrics['rmse_pp'].mean()),
    })
    (OUT_DIR / 'ensemble_summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT_DIR)


if __name__ == '__main__':
    main()
