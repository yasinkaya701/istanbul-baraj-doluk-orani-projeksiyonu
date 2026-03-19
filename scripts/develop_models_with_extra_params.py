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
from prophet import Prophet
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path('/Users/yasinkaya/Hackhaton')
BASE_SCRIPT = ROOT / 'scripts' / 'develop_top3_plus_external_models.py'
FEATURE_BLOCKS_SCRIPT = ROOT / 'scripts' / 'baraj_feature_blocks.py'
OUT_DIR = ROOT / 'output' / 'istanbul_models_extra_params_2026_03_12'

TRAIN_END = pd.Timestamp('2015-12-01')
TEST_START = pd.Timestamp('2016-01-01')
TEST_END = pd.Timestamp('2020-12-01')
FUTURE_START = pd.Timestamp('2026-01-01')
FUTURE_END = pd.Timestamp('2040-12-01')

MODEL_LABELS = {
    'hybrid_physics_stacked_exog': 'Stacked Hybrid Exog',
    'hybrid_physics_ensemble_phys_old': 'Secilen Ensemble',
    'water_balance_v4_corrected': 'Water Balance v4+',
    'quantile_regressor_plus': 'Quantile Boost',
    'prophet_regressor_exog': 'Prophet Exog',
}
MODEL_COLORS = {
    'hybrid_physics_stacked_exog': '#047857',
    'hybrid_physics_ensemble_phys_old': '#111827',
    'water_balance_v4_corrected': '#b91c1c',
    'quantile_regressor_plus': '#7c3aed',
    'prophet_regressor_exog': '#d97706',
}
STACK_EXOG = [
    'src_rain_north',
    'src_rain_west',
    'rain_model_mm',
    'transfer_share_pct_monthly_proxy',
    'nrw_pct_monthly_proxy',
    'reclaimed_share_pct_monthly_proxy',
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exog_frame() -> pd.DataFrame:
    feature_blocks = load_module(FEATURE_BLOCKS_SCRIPT, 'feature_blocks_extra_params')
    exog = feature_blocks.load_monthly_exog_table().copy().sort_values('date').reset_index(drop=True)
    for col in [
        'nrw_pct',
        'reclaimed_share_pct',
        'active_subscribers',
        'nao_index',
        'reanalysis_rs_mj_m2_month',
        'transfer_share_pct_monthly_proxy',
        'nrw_pct_monthly_proxy',
        'reclaimed_share_pct_monthly_proxy',
        'official_supply_m3_month_roll3',
    ]:
        if col in exog.columns:
            exog[col] = exog[col].ffill().bfill()
    return exog.sort_values('date').reset_index(drop=True)


def fit_source_proxy_models(train_exog: pd.DataFrame) -> dict[str, RidgeCV]:
    feats = ['rain_model_mm', 'month_sin', 'month_cos']
    models = {}
    for target in ['src_rain_north', 'src_rain_west']:
        m = RidgeCV(alphas=np.logspace(-4, 4, 41))
        m.fit(train_exog[feats], train_exog[target])
        models[target] = m
    return models


def add_future_source_proxies(future_df: pd.DataFrame, proxy_models: dict[str, RidgeCV]) -> pd.DataFrame:
    out = future_df.copy()
    feats = out[['rain_model_mm', 'month_sin', 'month_cos']]
    out['src_rain_north'] = proxy_models['src_rain_north'].predict(feats)
    out['src_rain_west'] = proxy_models['src_rain_west'].predict(feats)
    return out


def prophet_regressors() -> list[str]:
    return [
        'weighted_total_fill_lag1',
        'weighted_total_fill_lag2',
        'rain_model_mm',
        'et0_mm_month',
        'consumption_mean_monthly',
        'temp_proxy_c',
        'vpd_kpa_mean',
        'src_rain_north',
        'src_rain_west',
        'transfer_share_pct_monthly_proxy',
        'nrw_pct_monthly_proxy',
        'reclaimed_share_pct_monthly_proxy',
        'official_supply_m3_month_roll3',
    ]


def prophet_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    regs = prophet_regressors()
    out = df.rename(columns={'date': 'ds', 'weighted_total_fill': 'y'}).copy()
    return out[['ds', 'y'] + regs]


def fit_prophet_exog(train_df: pd.DataFrame) -> Prophet:
    model = Prophet(
        growth='flat',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.03,
        seasonality_mode='multiplicative',
    )
    for reg in prophet_regressors():
        model.add_regressor(reg, standardize=True)
    model.fit(prophet_train_frame(train_df))
    return model


def recursive_predict_prophet_exog(model: Prophet, history_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
    past_fill = history_df['weighted_total_fill'].tolist()
    preds = []
    regs = prophet_regressors()
    for row in future_df.itertuples(index=False):
        d = row._asdict()
        frame = {'ds': pd.Timestamp(d['date'])}
        for reg in regs:
            if reg == 'weighted_total_fill_lag1':
                frame[reg] = past_fill[-1]
            elif reg == 'weighted_total_fill_lag2':
                frame[reg] = past_fill[-2]
            else:
                frame[reg] = float(d[reg])
        yhat = float(np.clip(model.predict(pd.DataFrame([frame]))['yhat'].iloc[0], 0.0, 1.0))
        preds.append(yhat)
        past_fill.append(yhat)
    return np.asarray(preds, dtype=float)


def fit_stacker_with_exog(stack_df: pd.DataFrame) -> RidgeCV:
    feats = ['pred_h', 'pred_w', 'month_sin', 'month_cos'] + STACK_EXOG
    model = RidgeCV(alphas=np.logspace(-4, 4, 41))
    model.fit(stack_df[feats], stack_df['weighted_total_fill'])
    return model


def metric_row(model_name: str, actual: np.ndarray, pred: np.ndarray) -> dict[str, float | str]:
    return {
        'model': model_name,
        'rmse_pp': float(np.sqrt(mean_squared_error(actual, pred)) * 100.0),
        'mae_pp': float(mean_absolute_error(actual, pred) * 100.0),
        'mape_pct': float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-6)) * 100.0),
        'smape_pct': float(np.mean(2.0 * np.abs(pred - actual) / np.maximum(np.abs(actual) + np.abs(pred), 1e-6)) * 100.0),
        'pearson_corr_pct': float(pearsonr(actual, pred).statistic * 100.0),
        'spearman_corr_pct': float(spearmanr(actual, pred).statistic * 100.0),
        'r2': float(r2_score(actual, pred)),
        'end_error_pp_2020_12': float((pred[-1] - actual[-1]) * 100.0),
    }


def plot_holdout(pred_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    keep = summary_df['model'].tolist()
    fig, ax = plt.subplots(figsize=(13, 6), dpi=180)
    ax.plot(pred_df['date'], pred_df['actual_fill'] * 100.0, color='#111827', linewidth=2.4, label='Gercek')
    for model in keep:
        ax.plot(pred_df['date'], pred_df[model] * 100.0, color=MODEL_COLORS[model], linewidth=1.9, label=MODEL_LABELS[model])
    ax.set_title('Ekstra parametrelerle holdout: 2016-2020')
    ax.set_ylabel('Toplam doluluk (%)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_future(future_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6), dpi=180)
    for model in future_df['model'].unique():
        g = future_df[future_df['model'] == model]
        ax.plot(g['date'], g['pred_fill'] * 100.0, color=MODEL_COLORS[model], linewidth=2.0, label=MODEL_LABELS[model])
    ax.set_title('Ekstra parametrelerle temel gelecek: 2026-2040')
    ax.set_ylabel('Toplam doluluk (%)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_module(BASE_SCRIPT, 'dev_base_extra')
    bench = load_module(base.BENCHMARK_SCRIPT, 'bench_extra')
    wb = load_module(base.WB_SCRIPT, 'wb_extra')
    forward = load_module(base.FORWARD_SCRIPT, 'forward_extra')

    feature_df = base.add_enhanced_features(bench.load_training_frame().copy())
    exog_df = load_exog_frame()
    feature_df = feature_df.merge(
        exog_df[
            [
                'date',
                'src_rain_north',
                'src_rain_west',
                'transfer_share_pct_monthly_proxy',
                'nrw_pct_monthly_proxy',
                'reclaimed_share_pct_monthly_proxy',
                'official_supply_m3_month_roll3',
            ]
        ],
        on='date',
        how='left',
    )

    train = feature_df[feature_df['date'] <= TRAIN_END].copy().reset_index(drop=True)
    test = feature_df[(feature_df['date'] >= TEST_START) & (feature_df['date'] <= TEST_END)].copy().reset_index(drop=True)

    # internal models from current best script
    h_model = base.fit_hybrid_plus(train)
    pred_h = base.recursive_predict_delta_model(h_model, train, test)

    context = wb.compute_system_context()
    wb_df = wb.load_training_frame(context).copy()
    wb_df = wb_df[wb_df['date'].isin(feature_df['date'])].sort_values('date').reset_index(drop=True)
    train_wb = wb_df[wb_df['date'] <= TRAIN_END].copy().reset_index(drop=True)
    test_wb = wb_df[(wb_df['date'] >= TEST_START) & (wb_df['date'] <= TEST_END)].copy().reset_index(drop=True)
    share_by_year, _ = wb.load_transfer_share_by_year()
    wb_base, wb_bias, wb_transfer_eff, wb_corr = base.fit_wb_corrected(wb, train_wb, context, share_by_year)
    pred_w = base.recursive_predict_wb_corrected(
        wb,
        train_wb,
        test_wb[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']],
        context,
        wb_base,
        wb_bias,
        wb_transfer_eff,
        wb_corr,
        share_by_year,
    )

    # old selected ensemble
    orig_train = bench.load_training_frame().copy()
    orig_train = orig_train[orig_train['date'] <= TRAIN_END].copy().reset_index(drop=True)
    orig_test = bench.load_training_frame().copy()
    orig_test = orig_test[(orig_test['date'] >= TEST_START) & (orig_test['date'] <= TEST_END)].copy().reset_index(drop=True)
    orig_h_model = bench.fit_model(bench.ModelSpec('hybrid_ridge', bench.FEATURES, 'ridge'), orig_train)
    orig_h_pred = bench.recursive_forecast_known_exog(bench.ModelSpec('hybrid_ridge', bench.FEATURES, 'ridge'), orig_h_model, orig_train, orig_test)
    wb_base_old, wb_bias_old, wb_transfer_eff_old, _ = base.fit_wb_corrected(wb, train_wb, context, share_by_year)
    old_wb_raw = wb.simulate_path(
        history_df=train_wb,
        future_exog=test_wb[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']],
        model=wb_base_old,
        month_bias=wb_bias_old,
        context=context,
        transfer_share_anchor_pct=0.0,
        transfer_effectiveness=wb_transfer_eff_old,
        baseline_transfer_share_pct=0.0,
        transfer_end_pct_2040=0.0,
    )['pred_fill'].to_numpy(dtype=float)
    pred_old = 0.45 * orig_h_pred + 0.55 * old_wb_raw

    # quantile: keep best median-quantile boosted variant from prior search
    q_model = base.fit_quantile_plus(train)
    pred_q = base.recursive_predict_delta_model(q_model, train, test)

    # prophet with source-rain regressor
    prophet_model = fit_prophet_exog(train)
    pred_prophet = recursive_predict_prophet_exog(prophet_model, train, test)

    # stacked exog model
    stack_h, stack_w, stack_actual = base.internal_stacking_dataset(train, train_wb, wb, context, share_by_year)
    stack_df = stack_actual[['date', 'weighted_total_fill', 'month_sin', 'month_cos', 'rain_model_mm']].merge(
        exog_df[
            [
                'date',
                'src_rain_north',
                'src_rain_west',
                'transfer_share_pct_monthly_proxy',
                'nrw_pct_monthly_proxy',
                'reclaimed_share_pct_monthly_proxy',
            ]
        ],
        on='date',
        how='left'
    )
    stack_df['pred_h'] = stack_h
    stack_df['pred_w'] = stack_w
    stack_model = fit_stacker_with_exog(stack_df)
    test_stack = test[['date', 'month_sin', 'month_cos', 'rain_model_mm']].merge(
        exog_df[
            [
                'date',
                'src_rain_north',
                'src_rain_west',
                'transfer_share_pct_monthly_proxy',
                'nrw_pct_monthly_proxy',
                'reclaimed_share_pct_monthly_proxy',
            ]
        ],
        on='date',
        how='left'
    )
    X_test_stack = pd.DataFrame(
        {
            'pred_h': pred_h,
            'pred_w': pred_w,
            'month_sin': test_stack['month_sin'].to_numpy(dtype=float),
            'month_cos': test_stack['month_cos'].to_numpy(dtype=float),
            'src_rain_north': test_stack['src_rain_north'].to_numpy(dtype=float),
            'src_rain_west': test_stack['src_rain_west'].to_numpy(dtype=float),
            'rain_model_mm': test_stack['rain_model_mm'].to_numpy(dtype=float),
            'transfer_share_pct_monthly_proxy': test_stack['transfer_share_pct_monthly_proxy'].to_numpy(dtype=float),
            'nrw_pct_monthly_proxy': test_stack['nrw_pct_monthly_proxy'].to_numpy(dtype=float),
            'reclaimed_share_pct_monthly_proxy': test_stack['reclaimed_share_pct_monthly_proxy'].to_numpy(dtype=float),
        }
    )
    pred_stack = np.clip(stack_model.predict(X_test_stack), 0.0, 1.0)

    pred_df = pd.DataFrame(
        {
            'date': test['date'],
            'actual_fill': test['weighted_total_fill'],
            'hybrid_physics_stacked_exog': pred_stack,
            'hybrid_physics_ensemble_phys_old': pred_old,
            'water_balance_v4_corrected': pred_w,
            'quantile_regressor_plus': pred_q,
            'prophet_regressor_exog': pred_prophet,
        }
    )
    actual = pred_df['actual_fill'].to_numpy(dtype=float)
    summary_rows = [metric_row(col, actual, pred_df[col].to_numpy(dtype=float)) for col in pred_df.columns if col not in {'date', 'actual_fill'}]
    summary_df = pd.DataFrame(summary_rows).sort_values(['mape_pct', 'rmse_pp']).reset_index(drop=True)

    # future: use full available history for projection fit
    full_train = feature_df.copy().reset_index(drop=True)
    full_wb = wb_df.copy().reset_index(drop=True)
    h_model_future = base.fit_hybrid_plus(full_train)
    q_model_future = base.fit_quantile_plus(full_train)
    prophet_model_future = fit_prophet_exog(full_train)
    wb_base_future, wb_bias_future, wb_transfer_eff_future, wb_corr_future = base.fit_wb_corrected(
        wb, full_wb, context, share_by_year
    )
    stack_h_full, stack_w_full, stack_actual_full = base.internal_stacking_dataset(
        full_train, full_wb, wb, context, share_by_year
    )
    stack_df_full = stack_actual_full[['date', 'weighted_total_fill', 'month_sin', 'month_cos', 'rain_model_mm']].merge(
        exog_df[
            [
                'date',
                'src_rain_north',
                'src_rain_west',
                'transfer_share_pct_monthly_proxy',
                'nrw_pct_monthly_proxy',
                'reclaimed_share_pct_monthly_proxy',
            ]
        ],
        on='date',
        how='left'
    )
    stack_df_full['pred_h'] = stack_h_full
    stack_df_full['pred_w'] = stack_w_full
    stack_model_future = fit_stacker_with_exog(stack_df_full)

    clim = forward.monthly_climatology(feature_df)
    _, demand_relief_pct = forward.latest_policy_anchor()
    _, transfer_share_anchor_pct = forward.load_transfer_dependency_anchor()
    base_cfg = next(cfg for cfg in forward.build_scenarios() if cfg.scenario == 'base')
    future_exog = forward.build_future_exog(feature_df, base_cfg, clim, demand_relief_pct, transfer_share_anchor_pct=transfer_share_anchor_pct)
    future_exog = future_exog[(future_exog['date'] >= FUTURE_START) & (future_exog['date'] <= FUTURE_END)].copy().reset_index(drop=True)
    proxy_models = fit_source_proxy_models(exog_df[exog_df['date'] <= pd.Timestamp('2026-03-01')].dropna(subset=['src_rain_north', 'src_rain_west']).copy())
    future_exog = add_future_source_proxies(future_exog, proxy_models)
    month_ops = exog_df.groupby(exog_df['date'].dt.month)[
        ['transfer_share_pct_monthly_proxy', 'nrw_pct_monthly_proxy', 'reclaimed_share_pct_monthly_proxy']
    ].mean()
    progress = np.linspace(0.0, 1.0, len(future_exog))
    future_exog['transfer_share_pct_monthly_proxy'] = future_exog['month'].map(month_ops['transfer_share_pct_monthly_proxy']).astype(float)
    future_exog['transfer_share_pct_monthly_proxy'] *= 1.0 + (base_cfg.transfer_end_pct_2040 / 100.0) * progress
    future_exog['nrw_pct_monthly_proxy'] = future_exog['month'].map(month_ops['nrw_pct_monthly_proxy']).astype(float) - base_cfg.nrw_reduction_pp_by_2040 * progress
    future_exog['nrw_pct_monthly_proxy'] = future_exog['nrw_pct_monthly_proxy'].clip(lower=5.0)
    future_exog['reclaimed_share_pct_monthly_proxy'] = future_exog['month'].map(month_ops['reclaimed_share_pct_monthly_proxy']).astype(float)
    future_exog['official_supply_m3_month_roll3'] = (
        future_exog['consumption_mean_monthly'] * future_exog['date'].dt.days_in_month
    ).rolling(3, min_periods=1).mean()

    pred_h_future = base.recursive_predict_delta_model(h_model_future, full_train, future_exog)
    pred_q_future = base.recursive_predict_delta_model(q_model_future, full_train, future_exog)
    pred_prophet_future = recursive_predict_prophet_exog(prophet_model_future, full_train, future_exog)
    pred_w_future = base.recursive_predict_wb_corrected(
        wb,
        full_wb,
        future_exog[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']],
        context,
        wb_base_future,
        wb_bias_future,
        wb_transfer_eff_future,
        wb_corr_future,
        share_by_year,
    )
    orig_train_full = bench.load_training_frame().copy().reset_index(drop=True)
    orig_h_model_full = bench.fit_model(bench.ModelSpec('hybrid_ridge', bench.FEATURES, 'ridge'), orig_train_full)
    orig_h_pred_future = bench.recursive_forecast_known_exog(bench.ModelSpec('hybrid_ridge', bench.FEATURES, 'ridge'), orig_h_model_full, orig_train_full, future_exog)
    share_by_year_full, anchor_share_pct = wb.load_transfer_share_by_year()
    wb_base_old_full, wb_bias_old_full, wb_transfer_eff_old_full, _ = base.fit_wb_corrected(wb, full_wb, context, share_by_year_full)
    old_wb_future = wb.simulate_path(
        history_df=full_wb,
        future_exog=future_exog[['date', 'rain_model_mm', 'et0_mm_month', 'consumption_mean_monthly']],
        model=wb_base_old_full,
        month_bias=wb_bias_old_full,
        context=context,
        transfer_share_anchor_pct=anchor_share_pct,
        transfer_effectiveness=wb_transfer_eff_old_full,
        baseline_transfer_share_pct=anchor_share_pct,
        transfer_end_pct_2040=0.0,
    )['pred_fill'].to_numpy(dtype=float)
    pred_old_future = 0.45 * orig_h_pred_future + 0.55 * old_wb_future

    X_future_stack = pd.DataFrame(
        {
            'pred_h': pred_h_future,
            'pred_w': pred_w_future,
            'month_sin': future_exog['month_sin'].to_numpy(dtype=float),
            'month_cos': future_exog['month_cos'].to_numpy(dtype=float),
            'src_rain_north': future_exog['src_rain_north'].to_numpy(dtype=float),
            'src_rain_west': future_exog['src_rain_west'].to_numpy(dtype=float),
            'rain_model_mm': future_exog['rain_model_mm'].to_numpy(dtype=float),
            'transfer_share_pct_monthly_proxy': future_exog['transfer_share_pct_monthly_proxy'].to_numpy(dtype=float),
            'nrw_pct_monthly_proxy': future_exog['nrw_pct_monthly_proxy'].to_numpy(dtype=float),
            'reclaimed_share_pct_monthly_proxy': future_exog['reclaimed_share_pct_monthly_proxy'].to_numpy(dtype=float),
        }
    )
    pred_stack_future = np.clip(stack_model_future.predict(X_future_stack), 0.0, 1.0)

    future_rows = []
    for name, arr in [
        ('hybrid_physics_stacked_exog', pred_stack_future),
        ('hybrid_physics_ensemble_phys_old', pred_old_future),
        ('water_balance_v4_corrected', pred_w_future),
        ('quantile_regressor_plus', pred_q_future),
        ('prophet_regressor_exog', pred_prophet_future),
    ]:
        future_rows.append(pd.DataFrame({'date': future_exog['date'], 'model': name, 'pred_fill': arr}))
    future_df = pd.concat(future_rows, ignore_index=True)

    stack_coef = pd.DataFrame(
        {
            'feature': ['pred_h', 'pred_w', 'month_sin', 'month_cos', 'src_rain_north', 'src_rain_west', 'rain_model_mm', 'transfer_share_pct_monthly_proxy', 'nrw_pct_monthly_proxy', 'reclaimed_share_pct_monthly_proxy', 'alpha'],
            'value': list(stack_model.coef_) + [float(stack_model.alpha_)],
        }
    )

    pred_df.to_csv(OUT_DIR / 'extra_param_models_holdout_predictions_2016_2020.csv', index=False)
    summary_df.to_csv(OUT_DIR / 'extra_param_models_holdout_summary_2015_train_2020_test.csv', index=False)
    future_df.to_csv(OUT_DIR / 'extra_param_models_future_base_2026_2040.csv', index=False)
    stack_coef.to_csv(OUT_DIR / 'extra_param_stacker_coefficients.csv', index=False)
    plot_holdout(pred_df, summary_df, OUT_DIR / 'extra_param_models_holdout.png')
    plot_future(future_df, OUT_DIR / 'extra_param_models_future.png')

    endpoint = future_df[future_df['date'] == future_df['date'].max()].copy()
    endpoint['pred_fill_pct'] = endpoint['pred_fill'] * 100.0
    endpoint[['model', 'pred_fill', 'pred_fill_pct']].to_csv(OUT_DIR / 'extra_param_models_2040_endpoints.csv', index=False)

    summary = {
        'train_end': str(TRAIN_END.date()),
        'test_period': [str(TEST_START.date()), str(TEST_END.date())],
        'best_model_by_mape': str(summary_df.iloc[0]['model']),
        'best_mape_pct': float(summary_df.iloc[0]['mape_pct']),
        'best_rmse_pp': float(summary_df.iloc[0]['rmse_pp']),
        'extra_data_used': ['source_precip_north', 'source_precip_west', 'rain_model_mm', 'monthly_transfer_proxy', 'monthly_nrw_proxy', 'monthly_reclaimed_proxy'],
    }
    (OUT_DIR / 'extra_param_models_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# Ekstra parametreli model gelistirme ozeti', '']
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {MODEL_LABELS[row['model']]}: MAPE %{row['mape_pct']:.2f}, RMSE {row['rmse_pp']:.2f} yp, Pearson %{row['pearson_corr_pct']:.2f}"
        )
    (OUT_DIR / 'SONUC_OZETI.md').write_text('\n'.join(lines), encoding='utf-8')
    print(OUT_DIR)


if __name__ == '__main__':
    main()
