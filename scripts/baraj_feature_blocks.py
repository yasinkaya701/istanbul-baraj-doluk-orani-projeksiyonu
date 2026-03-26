#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path('/Users/yasinkaya/Hackhaton')
EXTENDED_PATH = ROOT / 'output' / 'model_useful_data_bundle' / 'tables' / 'istanbul_model_extended_monthly.csv'
SOURCE_PRECIP_PATH = ROOT / 'output' / 'source_precip_proxies' / 'source_precip_monthly_wide_2000_2026.csv'
MONTHLY_OPS_PROXY_PATH = ROOT / 'output' / 'newdata_feature_store' / 'tables' / 'monthly_operational_proxies_2000_2026.csv'

ANNUAL_FILL_COLS = [
    'nrw_pct',
    'physical_loss_pct',
    'administrative_loss_pct',
    'active_subscribers',
    'reclaimed_share_pct',
    'consumption_liters_per_active_subscriber_day',
]


def _safe_ffill_bfill(series: pd.Series) -> pd.Series:
    if series.notna().any():
        return series.ffill().bfill()
    return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)


def load_source_precip_features() -> pd.DataFrame:
    src = pd.read_csv(SOURCE_PRECIP_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    source_cols = [c for c in src.columns if c != 'date']
    src['src_rain_mean'] = src[source_cols].mean(axis=1)
    src['src_rain_std'] = src[source_cols].std(axis=1)
    src['src_rain_spread'] = src[source_cols].max(axis=1) - src[source_cols].min(axis=1)
    src['src_rain_north'] = src[['Terkos', 'Kazandere', 'Pabucdere', 'Istrancalar']].mean(axis=1)
    src['src_rain_west'] = src[['Alibey', 'Buyukcekmece', 'Sazlidere']].mean(axis=1)
    src['src_rain_east'] = src[['Darlik', 'Omerli', 'Elmali']].mean(axis=1)
    src['src_rain_north_west_gap'] = src['src_rain_north'] - src['src_rain_west']
    src['src_rain_east_west_gap'] = src['src_rain_east'] - src['src_rain_west']
    return src


def load_monthly_exog_table() -> pd.DataFrame:
    ext = pd.read_csv(EXTENDED_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    src = load_source_precip_features()
    ops = pd.read_csv(MONTHLY_OPS_PROXY_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True) if MONTHLY_OPS_PROXY_PATH.exists() else pd.DataFrame(columns=['date'])
    df = ext.merge(
        src[
            [
                'date',
                'src_rain_mean',
                'src_rain_std',
                'src_rain_spread',
                'src_rain_north',
                'src_rain_west',
                'src_rain_east',
                'src_rain_north_west_gap',
                'src_rain_east_west_gap',
            ]
        ],
        on='date',
        how='left',
    )
    if not ops.empty:
        df = df.merge(
            ops[
                [
                    'date',
                    'transfer_share_pct_annual_est',
                    'transfer_share_pct_monthly_proxy',
                    'transfer_mcm_monthly_proxy',
                    'nrw_pct_annual_est',
                    'nrw_pct_monthly_proxy',
                    'nrw_mcm_monthly_proxy',
                    'reclaimed_share_pct_annual_est',
                    'reclaimed_share_pct_monthly_proxy',
                    'reclaimed_mcm_monthly_proxy',
                ]
            ],
            on='date',
            how='left',
        )
        if 'nrw_pct_monthly_proxy' in df.columns:
            df['nrw_pct'] = df['nrw_pct_monthly_proxy'].combine_first(df['nrw_pct'])
        if 'reclaimed_share_pct_monthly_proxy' in df.columns:
            df['reclaimed_share_pct'] = df['reclaimed_share_pct_monthly_proxy'].combine_first(df['reclaimed_share_pct'])

    for col in ANNUAL_FILL_COLS + ['nao_index', 'nao_lag1', 'nao_lag2', 'nao_roll3', 'reanalysis_rs_mj_m2_month', 'reanalysis_wind_speed_10m_max_m_s', 'reanalysis_sunshine_duration_h_month', 'official_supply_m3_month_roll3', 'recorded_share_pct']:
        if col in df.columns:
            df[col] = _safe_ffill_bfill(df[col])

    # availability flags before hard fills are useful as model features
    for col in ['nrw_pct', 'active_subscribers', 'reclaimed_share_pct', 'official_supply_m3_month_roll3', 'recorded_share_pct']:
        if col in ext.columns:
            df[f'{col}_flag'] = ext[col].notna().astype(int)

    df['rain_reanalysis_gap'] = df['rain_model_mm'] - df['reanalysis_precip_mm_month']
    df['et0_reanalysis_gap'] = df['et0_mm_month'] - df['reanalysis_et0_mm_month']
    df['rs_wind_interaction'] = df['reanalysis_rs_mj_m2_month'] * df['reanalysis_wind_speed_10m_max_m_s']
    df['nao_rain_interaction'] = df['nao_index'] * df['rain_model_mm']
    df['nrw_cons_interaction'] = df['nrw_pct'] * df['consumption_mean_monthly']
    df['supply_cons_gap_m3'] = df['city_supply_m3_month_official'] - (df['consumption_mean_monthly'] * df['date'].dt.days_in_month)
    df['supply_cons_gap_m3'] = df['supply_cons_gap_m3'].fillna(0.0)
    df['subscriber_trend_idx'] = np.arange(len(df), dtype=float)
    df['rs_et0_interaction'] = df['reanalysis_rs_mj_m2_month'] * df['et0_mm_month']
    df['north_rain_nao_interaction'] = df['src_rain_north'] * df['nao_index']
    for col in ['transfer_share_pct_annual_est', 'transfer_share_pct_monthly_proxy', 'transfer_mcm_monthly_proxy', 'nrw_mcm_monthly_proxy', 'reclaimed_mcm_monthly_proxy']:
        if col in df.columns:
            df[col] = _safe_ffill_bfill(df[col])
    return df


def attach_exog(base_df: pd.DataFrame) -> pd.DataFrame:
    exog = load_monthly_exog_table()
    out = base_df.merge(exog.drop(columns=[c for c in base_df.columns if c in exog.columns and c != 'date']), on='date', how='left')
    return out.sort_values('date').reset_index(drop=True)


def fit_future_proxy_models(exog_df: pd.DataFrame) -> dict[str, RidgeCV]:
    train = exog_df.dropna(subset=['src_rain_north', 'src_rain_west', 'src_rain_east']).copy()
    feats = ['rain_model_mm', 'month_sin', 'month_cos', 'nao_index']
    models: dict[str, RidgeCV] = {}
    for target in ['src_rain_north', 'src_rain_west', 'src_rain_east']:
        m = RidgeCV(alphas=np.logspace(-4, 4, 41))
        m.fit(train[feats], train[target])
        models[target] = m
    return models


def add_future_exog_proxies(future_df: pd.DataFrame, fitted_models: dict[str, RidgeCV]) -> pd.DataFrame:
    out = future_df.copy()
    if 'nao_index' not in out.columns:
        out['nao_index'] = 0.0
    feats = out[['rain_model_mm', 'month_sin', 'month_cos', 'nao_index']]
    for target in ['src_rain_north', 'src_rain_west', 'src_rain_east']:
        out[target] = fitted_models[target].predict(feats)
    out['src_rain_mean'] = out[['src_rain_north', 'src_rain_west', 'src_rain_east']].mean(axis=1)
    out['src_rain_north_west_gap'] = out['src_rain_north'] - out['src_rain_west']
    out['src_rain_east_west_gap'] = out['src_rain_east'] - out['src_rain_west']
    return out
