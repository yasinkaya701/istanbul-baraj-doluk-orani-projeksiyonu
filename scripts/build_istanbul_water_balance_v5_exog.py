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
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/yasinkaya/Hackhaton')
V4_SCRIPT = ROOT / 'scripts' / 'build_istanbul_water_balance_v4_sourceaware.py'
FORWARD_SCRIPT = ROOT / 'scripts' / 'build_istanbul_forward_projection_2040.py'
FEATURE_BLOCKS_SCRIPT = ROOT / 'scripts' / 'baraj_feature_blocks.py'
OUT_DIR = ROOT / 'output' / 'istanbul_water_balance_v5_exog_2040'

TRAIN_END = pd.Timestamp('2015-12-01')
TEST_START = pd.Timestamp('2016-01-01')
TEST_END = pd.Timestamp('2020-12-01')

BASE_FEATURE_ORDER = [
    'source_runoff_now_mcm',
    'source_runoff_lag1_mcm',
    'source_runoff_wetness_mcm',
    'source_lake_rain_mcm',
    'neg_source_openwater_evap_mcm',
    'neg_supply_mcm',
    'neg_storage_mass_mcm',
    'neg_spill_pressure_mcm',
]
EXTRA_CORR_FEATURES = [
    'src_rain_north',
    'src_rain_west',
    'src_rain_proxy_mm',
    'reanalysis_rs_mj_m2_month',
    'reanalysis_wind_speed_10m_max_m_s',
    'openwater_evap_factor',
    'nrw_pct',
    'reclaimed_share_pct',
    'official_supply_m3_month_roll3',
]
BASE_CORR_FEATURES = [
    'weighted_total_fill_lag1',
    'rain_model_mm',
    'rain_model_mm_roll3',
    'et0_mm_month',
    'supply_mcm',
    'source_runoff_now_mcm',
    'source_runoff_wetness_mcm',
    'source_lake_rain_mcm',
    'source_openwater_evap_mcm',
    'spill_pressure_mcm',
    'month_sin',
    'month_cos',
]


def month_sin(month: int) -> float:
    return float(np.sin(2.0 * np.pi * month / 12.0))


def month_cos(month: int) -> float:
    return float(np.cos(2.0 * np.pi * month / 12.0))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_exog_climatology(exog_df: pd.DataFrame) -> dict[str, dict[int, float]]:
    month = exog_df['date'].dt.month
    cols = [
        'src_rain_north',
        'src_rain_west',
        'reanalysis_rs_mj_m2_month',
        'reanalysis_wind_speed_10m_max_m_s',
        'nrw_pct',
        'reclaimed_share_pct',
        'official_supply_m3_month_roll3',
        'nao_index',
    ]
    out = {}
    for c in cols:
        out[c] = exog_df.groupby(month)[c].mean().to_dict()
    return out


def load_training_frame_exog(v4, feature_blocks, context: dict[str, object]) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    df = v4.load_training_frame(context).copy()
    exog = feature_blocks.load_monthly_exog_table().copy()
    keep = [
        'date', 'src_rain_north', 'src_rain_west', 'src_rain_mean', 'src_rain_north_west_gap',
        'reanalysis_rs_mj_m2_month', 'reanalysis_wind_speed_10m_max_m_s',
        'nrw_pct', 'reclaimed_share_pct', 'official_supply_m3_month_roll3', 'nao_index'
    ]
    df = df.merge(exog[keep], on='date', how='left')
    for c in keep:
        if c != 'date':
            df[c] = df[c].ffill().bfill()
    df['src_rain_proxy_mm'] = 0.60 * df['src_rain_north'] + 0.40 * df['src_rain_west']
    df['src_rain_proxy_lag1_mm'] = df['src_rain_proxy_mm'].shift(1)
    df['src_rain_proxy_roll3_mm'] = df['src_rain_proxy_mm'].rolling(3).mean()
    df = df.dropna(subset=['src_rain_proxy_lag1_mm', 'src_rain_proxy_roll3_mm']).reset_index(drop=True)
    evap_refs = {
        'rs_ref': float(df['reanalysis_rs_mj_m2_month'].median()),
        'wind_ref': float(df['reanalysis_wind_speed_10m_max_m_s'].median()),
    }
    return df, evap_refs, exog


def openwater_evap_factor(rs: float, wind: float, refs: dict[str, float]) -> float:
    factor = 1.0 + 0.0007 * (rs - refs['rs_ref']) + 0.045 * (wind - refs['wind_ref'])
    return float(np.clip(factor, 0.75, 1.45))


def component_frame_exog(v4, df: pd.DataFrame, context: dict[str, object], refs: dict[str, float]) -> pd.DataFrame:
    out = df[[
        'date', 'weighted_total_fill', 'weighted_total_fill_lag1', 'rain_model_mm',
        'rain_model_mm_roll3', 'src_rain_proxy_mm',
        'src_rain_proxy_lag1_mm', 'src_rain_proxy_roll3_mm', 'et0_mm_month',
        'et0_mm_month_roll3', 'supply_mcm', 'delta_storage_mcm',
        'src_rain_north', 'src_rain_west', 'reanalysis_rs_mj_m2_month',
        'reanalysis_wind_speed_10m_max_m_s', 'nrw_pct', 'reclaimed_share_pct',
        'official_supply_m3_month_roll3'
    ]].copy()
    state_rows = [v4.source_state(float(fill), context) for fill in out['weighted_total_fill_lag1']]
    out['lake_area_km2'] = [float(s['lake_area_km2'].sum()) for s in state_rows]
    out['land_area_km2'] = [float(s['land_area_km2'].sum()) for s in state_rows]
    out['runoff_weighted_land_area_km2'] = [float(np.dot(s['land_area_km2'], s['runoff_productivity_weight'])) for s in state_rows]
    wetness_mm = np.clip(out['src_rain_proxy_roll3_mm'] - out['et0_mm_month_roll3'], 0.0, None)
    out['openwater_evap_factor'] = [openwater_evap_factor(rs, wind, refs) for rs, wind in zip(out['reanalysis_rs_mj_m2_month'], out['reanalysis_wind_speed_10m_max_m_s'])]
    out['source_runoff_now_mcm'] = out['runoff_weighted_land_area_km2'] * out['src_rain_proxy_mm'] * 0.001
    out['source_runoff_lag1_mcm'] = out['runoff_weighted_land_area_km2'] * out['src_rain_proxy_lag1_mm'] * 0.001
    out['source_runoff_wetness_mcm'] = out['runoff_weighted_land_area_km2'] * wetness_mm * 0.001
    out['source_lake_rain_mcm'] = out['lake_area_km2'] * out['src_rain_proxy_mm'] * 0.001
    out['source_openwater_evap_mcm'] = out['lake_area_km2'] * out['et0_mm_month'] * out['openwater_evap_factor'] * 0.001
    out['storage_mass_mcm'] = [float(s['storage_mcm'].sum()) for s in state_rows]
    out['spill_pressure_mcm'] = [float(s['spill_pressure_mcm'].sum()) for s in state_rows]
    out['neg_source_openwater_evap_mcm'] = -out['source_openwater_evap_mcm']
    out['neg_supply_mcm'] = -out['supply_mcm']
    out['neg_storage_mass_mcm'] = -out['storage_mass_mcm']
    out['neg_spill_pressure_mcm'] = -out['spill_pressure_mcm']
    out['month_sin'] = out['date'].dt.month.map(month_sin)
    out['month_cos'] = out['date'].dt.month.map(month_cos)
    return out


def fit_model(train_comp: pd.DataFrame):
    model = LinearRegression(positive=True)
    model.fit(train_comp[BASE_FEATURE_ORDER], train_comp['delta_storage_mcm'])
    tmp = train_comp[['date', 'delta_storage_mcm']].copy()
    tmp['base_pred_mcm'] = model.predict(train_comp[BASE_FEATURE_ORDER])
    tmp['month'] = tmp['date'].dt.month
    tmp['residual_mcm'] = tmp['delta_storage_mcm'] - tmp['base_pred_mcm']
    month_bias = tmp.groupby('month')['residual_mcm'].mean().to_dict()
    tmp['pred_mcm'] = tmp['base_pred_mcm'] + tmp['month'].map(month_bias).fillna(0.0)

    train_corr = train_comp.copy()
    train_corr['base_pred_mcm'] = tmp['pred_mcm'].to_numpy(dtype=float)
    train_corr['residual_target_mcm'] = train_comp['delta_storage_mcm'].to_numpy(dtype=float) - train_corr['base_pred_mcm'].to_numpy(dtype=float)
    corr_features = BASE_CORR_FEATURES + EXTRA_CORR_FEATURES
    corr = Pipeline([('scaler', StandardScaler()), ('model', RidgeCV(alphas=np.logspace(-4, 4, 31)))])
    corr.fit(train_corr[corr_features], train_corr['residual_target_mcm'])
    return model, month_bias, corr, corr_features


def simulate_path(v4, history_df: pd.DataFrame, future_df: pd.DataFrame, context: dict[str, object], refs: dict[str, float], model, month_bias, corr, corr_features, share_by_year: dict[int, float], transfer_effectiveness: float) -> pd.DataFrame:
    past_fill = history_df['weighted_total_fill'].tolist()
    past_src_rain = history_df['src_rain_proxy_mm'].tolist()
    past_et0 = history_df['et0_mm_month'].tolist()
    rows = []
    total_storage = float(context['total_storage_mcm'])
    for row in future_df.reset_index(drop=True).itertuples(index=False):
        d = row._asdict()
        fill_prev = float(past_fill[-1])
        date = pd.Timestamp(d['date'])
        days = int(date.days_in_month)
        supply_mcm = float(d['consumption_mean_monthly'] * days / 1e6)
        state = v4.source_state(fill_prev, context)
        lake_area_km2 = float(state['lake_area_km2'].sum())
        runoff_area = float(np.dot(state['land_area_km2'], state['runoff_productivity_weight']))
        rain_now = float(d['src_rain_proxy_mm'])
        rain_lag1 = float(past_src_rain[-1])
        et0_now = float(d['et0_mm_month'])
        rain_roll3 = float(np.mean([past_src_rain[-2], past_src_rain[-1], rain_now]))
        et0_roll3 = float(np.mean([past_et0[-2], past_et0[-1], et0_now]))
        evap_factor = openwater_evap_factor(float(d['reanalysis_rs_mj_m2_month']), float(d['reanalysis_wind_speed_10m_max_m_s']), refs)
        comp = {
            'source_runoff_now_mcm': runoff_area * rain_now * 0.001,
            'source_runoff_lag1_mcm': runoff_area * rain_lag1 * 0.001,
            'source_runoff_wetness_mcm': runoff_area * max(rain_roll3 - et0_roll3, 0.0) * 0.001,
            'source_lake_rain_mcm': lake_area_km2 * rain_now * 0.001,
            'source_openwater_evap_mcm': lake_area_km2 * et0_now * evap_factor * 0.001,
            'supply_mcm': supply_mcm,
            'storage_mass_mcm': float(state['storage_mcm'].sum()),
            'spill_pressure_mcm': float(state['spill_pressure_mcm'].sum()),
        }
        comp['neg_source_openwater_evap_mcm'] = -comp['source_openwater_evap_mcm']
        comp['neg_supply_mcm'] = -comp['supply_mcm']
        comp['neg_storage_mass_mcm'] = -comp['storage_mass_mcm']
        comp['neg_spill_pressure_mcm'] = -comp['spill_pressure_mcm']
        base_delta = v4.predict_delta_mcm(comp, date.month, model, month_bias)
        corr_row = {
            'weighted_total_fill_lag1': fill_prev,
            'rain_model_mm': float(d['rain_model_mm']),
            'rain_model_mm_roll3': float(d['src_rain_proxy_roll3_mm']),
            'et0_mm_month': et0_now,
            'supply_mcm': supply_mcm,
            'source_runoff_now_mcm': comp['source_runoff_now_mcm'],
            'source_runoff_wetness_mcm': comp['source_runoff_wetness_mcm'],
            'source_lake_rain_mcm': comp['source_lake_rain_mcm'],
            'source_openwater_evap_mcm': comp['source_openwater_evap_mcm'],
            'spill_pressure_mcm': comp['spill_pressure_mcm'],
            'month_sin': month_sin(date.month),
            'month_cos': month_cos(date.month),
            'src_rain_north': float(d['src_rain_north']),
            'src_rain_west': float(d['src_rain_west']),
            'src_rain_proxy_mm': rain_now,
            'reanalysis_rs_mj_m2_month': float(d['reanalysis_rs_mj_m2_month']),
            'reanalysis_wind_speed_10m_max_m_s': float(d['reanalysis_wind_speed_10m_max_m_s']),
            'openwater_evap_factor': evap_factor,
            'nrw_pct': float(d['nrw_pct']),
            'reclaimed_share_pct': float(d['reclaimed_share_pct']),
            'official_supply_m3_month_roll3': float(d['official_supply_m3_month_roll3']),
        }
        corr_delta = float(corr.predict(pd.DataFrame([{c: corr_row[c] for c in corr_features}]))[0])
        transfer_delta = v4.historical_transfer_addition(date, supply_mcm, share_by_year, transfer_effectiveness)
        next_storage = np.clip(fill_prev * total_storage + base_delta + corr_delta + transfer_delta, 0.0, total_storage)
        fill_next = float(next_storage / total_storage)
        rows.append({'date': date, 'pred_fill': fill_next, 'source_runoff_now_mcm': comp['source_runoff_now_mcm'], 'source_openwater_evap_mcm': comp['source_openwater_evap_mcm'], 'corr_delta_mcm': corr_delta})
        past_fill.append(fill_next)
        past_src_rain.append(rain_now)
        past_et0.append(et0_now)
    return pd.DataFrame(rows)


def future_frame(forward, feature_blocks, history_df: pd.DataFrame, exog_df: pd.DataFrame) -> pd.DataFrame:
    clim = forward.monthly_climatology(history_df[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly', 'temp_proxy_c', 'rh_proxy_pct', 'vpd_kpa_mean']].copy())
    _, demand_relief = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == 'base')
    future = forward.build_future_exog(history_df, cfg, clim, demand_relief, transfer_share_anchor_pct=transfer_share_anchor_pct)
    future = future[(future['date'] >= TEST_END + pd.offsets.MonthBegin(73)) & (future['date'] <= pd.Timestamp('2040-12-01'))].copy().reset_index(drop=True)
    # explicit horizon start 2026-01
    future = future[future['date'] >= pd.Timestamp('2026-01-01')].copy().reset_index(drop=True)
    exog_clim = build_exog_climatology(exog_df)
    future['nao_index'] = future['month'].map(exog_clim['nao_index']).fillna(0.0)
    future['reanalysis_rs_mj_m2_month'] = future['month'].map(exog_clim['reanalysis_rs_mj_m2_month']).astype(float)
    future['reanalysis_wind_speed_10m_max_m_s'] = future['month'].map(exog_clim['reanalysis_wind_speed_10m_max_m_s']).astype(float)
    future['nrw_pct'] = future['month'].map(exog_clim['nrw_pct']).astype(float)
    future['reclaimed_share_pct'] = future['month'].map(exog_clim['reclaimed_share_pct']).astype(float)
    future['official_supply_m3_month_roll3'] = future['consumption_mean_monthly'] * future['date'].dt.days_in_month
    proxy_models = feature_blocks.fit_future_proxy_models(exog_df)
    future = feature_blocks.add_future_exog_proxies(future, proxy_models)
    future['src_rain_proxy_mm'] = 0.60 * future['src_rain_north'] + 0.40 * future['src_rain_west']
    future['src_rain_proxy_roll3_mm'] = future['src_rain_proxy_mm'].rolling(3, min_periods=1).mean()
    return future


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global v4
    v4 = load_module(V4_SCRIPT, 'wb_v4_exog_base')
    forward = load_module(FORWARD_SCRIPT, 'forward_v5_exog')
    feature_blocks = load_module(FEATURE_BLOCKS_SCRIPT, 'feature_blocks_v5_exog')

    context = v4.compute_system_context()
    df, refs, exog_df = load_training_frame_exog(v4, feature_blocks, context)
    train = df[df['date'] <= TRAIN_END].copy().reset_index(drop=True)
    test = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy().reset_index(drop=True)
    share_by_year, _ = v4.load_transfer_share_by_year()

    train_comp = component_frame_exog(v4, train, context, refs)
    model, month_bias, corr, corr_features = fit_model(train_comp)
    fit_df = v4.fit_water_balance_model(train_comp)[2]
    transfer_effectiveness = v4.estimate_transfer_effectiveness(train, fit_df, share_by_year)

    holdout = simulate_path(v4, train, test, context, refs, model, month_bias, corr, corr_features, share_by_year, transfer_effectiveness)
    actual = test[['date', 'weighted_total_fill']].rename(columns={'weighted_total_fill': 'actual_fill'})
    holdout = actual.merge(holdout, on='date', how='left')
    summary = {
        'model': 'water_balance_v5_exog',
        'rmse_pp': float(np.sqrt(mean_squared_error(holdout['actual_fill'], holdout['pred_fill'])) * 100.0),
        'mae_pp': float(mean_absolute_error(holdout['actual_fill'], holdout['pred_fill'])) * 100.0,
        'mape_pct': float(np.mean(np.abs(holdout['pred_fill'] - holdout['actual_fill']) / np.maximum(np.abs(holdout['actual_fill']), 1e-6)) * 100.0),
        'end_error_pp_2020_12': float((holdout['pred_fill'].iloc[-1] - holdout['actual_fill'].iloc[-1]) * 100.0),
    }
    full_comp = component_frame_exog(v4, df, context, refs)
    full_model, full_month_bias, full_corr, full_corr_features = fit_model(full_comp)
    full_fit_df = v4.fit_water_balance_model(full_comp)[2]
    full_transfer_eff = v4.estimate_transfer_effectiveness(df, full_fit_df, share_by_year)
    future = future_frame(forward, feature_blocks, df, exog_df)
    future_sim = simulate_path(v4, df, future, context, refs, full_model, full_month_bias, full_corr, full_corr_features, share_by_year, full_transfer_eff)

    holdout.to_csv(OUT_DIR / 'water_balance_v5_exog_holdout_predictions_2016_2020.csv', index=False)
    pd.DataFrame([summary]).to_csv(OUT_DIR / 'water_balance_v5_exog_holdout_summary.csv', index=False)
    future_sim.to_csv(OUT_DIR / 'water_balance_v5_exog_future_base_2026_2040.csv', index=False)
    (OUT_DIR / 'water_balance_v5_exog_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=180)
    ax.plot(holdout['date'], holdout['actual_fill'] * 100.0, color='#111827', linewidth=2.2, label='Gercek')
    ax.plot(holdout['date'], holdout['pred_fill'] * 100.0, color='#0f766e', linewidth=2.0, label='WB v5 exog')
    ax.set_title('Water balance v5 exog - 2016-2020 holdout')
    ax.set_ylabel('Toplam doluluk (%)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'water_balance_v5_exog_holdout.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=180)
    ax.plot(future_sim['date'], future_sim['pred_fill'] * 100.0, color='#0f766e', linewidth=2.0)
    ax.set_title('Water balance v5 exog - 2026-2040 temel gelecek')
    ax.set_ylabel('Toplam doluluk (%)')
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'water_balance_v5_exog_future.png')
    plt.close(fig)

    print(OUT_DIR)


if __name__ == '__main__':
    main()
