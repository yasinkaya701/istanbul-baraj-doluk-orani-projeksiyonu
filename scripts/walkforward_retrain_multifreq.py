#!/usr/bin/env python3
"""Expanding-window recursive retrain forecasts for multiple frequencies.

Mantık:
- Her hedef dönem için model o ana kadar (t-1) olan veriyle yeniden eğitilir.
- Sonraki tek adım (t) tahmin edilir.
- Tahmin edilen değer veri setine eklenir ve bir sonraki adımda yeniden fit edilir.

Desteklenen frekanslar:
- `YS` (yıllık), `MS` (aylık), `W-MON` (haftalık), `D` (günlük)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from climate_scenario_adjustment import (
    climate_delta_scalar,
    climate_strategy_suffix,
    from_args as climate_cfg_from_args,
    with_series_baseline as climate_cfg_with_series_baseline,
)

from quant_regime_projection import (
    apply_bounds,
    canonical_variable_name,
    ensure_dirs,
    fit_forecast_quant,
    infer_unit,
    is_precip,
    normalize_observations,
    read_table,
    select_variables,
    variable_tr,
    winsorize,
)


@dataclass
class FrequencyCfg:
    code: str
    label_tr: str
    min_train: int
    max_train_points_default: int


FREQ_MAP: dict[str, FrequencyCfg] = {
    "YS": FrequencyCfg(code="YS", label_tr="yıllık", min_train=8, max_train_points_default=300),
    "MS": FrequencyCfg(code="MS", label_tr="aylık", min_train=36, max_train_points_default=1200),
    "W": FrequencyCfg(code="W-MON", label_tr="haftalık", min_train=104, max_train_points_default=2500),
    "W-MON": FrequencyCfg(code="W-MON", label_tr="haftalık", min_train=104, max_train_points_default=2500),
    "D": FrequencyCfg(code="D", label_tr="günlük", min_train=365, max_train_points_default=3000),
    "YEARLY": FrequencyCfg(code="YS", label_tr="yıllık", min_train=8, max_train_points_default=300),
    "MONTHLY": FrequencyCfg(code="MS", label_tr="aylık", min_train=36, max_train_points_default=1200),
    "WEEKLY": FrequencyCfg(code="W-MON", label_tr="haftalık", min_train=104, max_train_points_default=2500),
    "DAILY": FrequencyCfg(code="D", label_tr="günlük", min_train=365, max_train_points_default=3000),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adım adım yeniden eğitimli çoklu frekans tahmin")
    p.add_argument(
        "--observations",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/forecast_package/observations_with_graph.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/walkforward_retrain_package"),
    )
    p.add_argument("--variables", type=str, default="*")
    p.add_argument("--frequencies", type=str, default="YS,MS,W,D")
    p.add_argument("--start-year", type=int, default=2026, help="Walk-forward başlangıç yılı")
    p.add_argument("--target-year", type=int, default=2035, help="Walk-forward bitiş yılı")

    p.add_argument("--input-kind", type=str, default="auto", choices=["auto", "long", "single"])
    p.add_argument("--timestamp-col", type=str, default="timestamp")
    p.add_argument("--value-col", type=str, default="value")
    p.add_argument("--variable-col", type=str, default="variable")
    p.add_argument("--qc-col", type=str, default="qc_flag")
    p.add_argument("--qc-ok-value", type=str, default="ok")
    p.add_argument("--single-variable", type=str, default="target")

    p.add_argument("--winsor-lower", type=float, default=0.005)
    p.add_argument("--winsor-upper", type=float, default=0.995)
    p.add_argument("--interval-alpha", type=float, default=0.10)
    p.add_argument("--vol-model", type=str, default="egarch", choices=["auto", "egarch", "ewma"])
    p.add_argument("--ewma-lambda", type=float, default=0.94)
    p.add_argument("--egarch-p", type=int, default=1)
    p.add_argument("--egarch-o", type=int, default=1)
    p.add_argument("--egarch-q", type=int, default=1)
    p.add_argument("--egarch-dist", type=str, default="t", choices=["normal", "t"])
    p.add_argument("--regime-k", type=int, default=2)
    p.add_argument("--regime-maxiter", type=int, default=200)

    p.add_argument(
        "--max-train-points",
        type=int,
        default=0,
        help="0 ise frekans varsayılanı kullanılır; >0 ise tüm frekanslarda bu pencere boyutu kullanılır",
    )
    p.add_argument(
        "--max-future-steps",
        type=int,
        default=0,
        help="0 ise hedef yıla kadar tamamı; >0 ise seri başına adım sayısı limiti",
    )
    p.add_argument(
        "--climate-scenario",
        type=str,
        default="ssp245",
        help="İklim senaryosu: none, ssp126, ssp245, ssp370, ssp585",
    )
    p.add_argument(
        "--climate-baseline-year",
        type=float,
        default=float("nan"),
        help="Senaryo düzeltmesi için baz yıl; NaN ise seri son gözlem yılı otomatik alınır.",
    )
    p.add_argument(
        "--climate-temp-rate",
        type=float,
        default=float("nan"),
        help="Sıcaklık trend override (C/yıl). NaN ise senaryo varsayılanı.",
    )
    p.add_argument(
        "--humidity-per-temp-c",
        type=float,
        default=-2.0,
        help="Nem düzeltme katsayısı (yüzde puan / C).",
    )
    p.add_argument(
        "--climate-adjustment-method",
        type=str,
        default="pathway",
        help="Düzeltme metodu: pathway (IPCC AR6 SSP eğrisi) veya linear.",
    )
    p.add_argument(
        "--disable-climate-adjustment",
        action="store_true",
        help="Senaryo katsayısı düzeltmesini kapat.",
    )
    return p.parse_args()


def normalize_freq_list(freqs: str) -> list[FrequencyCfg]:
    out: list[FrequencyCfg] = []
    seen: set[str] = set()
    for tok in [x.strip().upper() for x in str(freqs).split(",") if x.strip()]:
        if tok not in FREQ_MAP:
            continue
        cfg = FREQ_MAP[tok]
        if cfg.code in seen:
            continue
        seen.add(cfg.code)
        out.append(cfg)
    return out


def aggregate_series_freq(obs: pd.DataFrame, variable: str, freq: str, ok_value: str) -> pd.Series:
    sub = obs[obs["variable"] == variable].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    ok_mask = sub["qc_flag"].astype(str).str.lower().eq(str(ok_value).lower())
    if ok_mask.any():
        sub = sub[ok_mask]
    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return pd.Series(dtype=float)

    raw = sub.groupby("timestamp")["value"].mean().sort_index()
    if raw.empty:
        return pd.Series(dtype=float)

    if is_precip(variable):
        s = raw.resample(freq).sum(min_count=1).fillna(0.0)
    else:
        s = raw.resample(freq).mean().interpolate("time").ffill().bfill()
    return s.astype(float)


def make_future_index(last_obs: pd.Timestamp, start_year: int, target_year: int, freq: str) -> pd.DatetimeIndex:
    if target_year < start_year:
        return pd.DatetimeIndex([])

    start_floor = pd.Timestamp(year=int(start_year), month=1, day=1)
    end_day = pd.Timestamp(year=int(target_year), month=12, day=31)

    if freq == "YS":
        cand = max(last_obs + pd.offsets.YearBegin(1), start_floor)
        end = pd.Timestamp(year=int(target_year), month=1, day=1)
        if cand > end:
            return pd.DatetimeIndex([])
        return pd.date_range(cand, end=end, freq="YS")

    if freq == "MS":
        cand = max(last_obs + pd.offsets.MonthBegin(1), start_floor)
        end = pd.Timestamp(year=int(target_year), month=12, day=1)
        if cand > end:
            return pd.DatetimeIndex([])
        return pd.date_range(cand, end=end, freq="MS")

    if freq == "W-MON":
        cand = max(last_obs + pd.Timedelta(days=1), start_floor)
        idx = pd.date_range(cand, end=end_day, freq="W-MON")
        return pd.DatetimeIndex(idx)

    if freq == "D":
        cand = max(last_obs + pd.Timedelta(days=1), start_floor)
        idx = pd.date_range(cand, end=end_day, freq="D")
        return pd.DatetimeIndex(idx)

    return pd.DatetimeIndex([])


def calibration_stats(
    ds: pd.Series,
    y: np.ndarray,
    freq: str,
    variable: str,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray]:
    fit0 = fit_forecast_quant(
        ds=ds,
        y=y,
        future_ds=pd.Series(pd.DatetimeIndex([])),
        freq=freq,
        regime_k=int(args.regime_k),
        regime_maxiter=int(args.regime_maxiter),
        vol_model=str(args.vol_model),
        ewma_lam=float(args.ewma_lambda),
        egarch_p=int(args.egarch_p),
        egarch_o=int(args.egarch_o),
        egarch_q=int(args.egarch_q),
        egarch_dist=str(args.egarch_dist),
        variable=variable,
    )

    yhat_hist = np.asarray(fit0["yhat_hist"], dtype=float)

    resid = np.asarray(fit0["resid"], dtype=float)
    if len(resid) == 0:
        resid = np.asarray(y, dtype=float) - float(np.nanmean(y))

    alpha = float(args.interval_alpha)
    if is_precip(variable):
        alpha_eff = max(alpha, 0.24)
    elif freq == "YS":
        alpha_eff = max(alpha, 0.18)
    else:
        alpha_eff = alpha

    sigma_hist = np.sqrt(np.maximum(np.asarray(fit0["var_hist"], dtype=float), 1e-9))
    if len(sigma_hist):
        score_hist = np.abs(resid) / np.maximum(sigma_hist, 1e-6)
        q_norm = float(np.quantile(score_hist, np.clip(1.0 - alpha_eff, 0.50, 0.99)))
    else:
        q_norm = 1.64
    if is_precip(variable):
        q_norm = float(np.clip(q_norm, 0.50, 1.90))
    else:
        q_norm = float(np.clip(q_norm, 0.70, 2.80))

    q_abs = float(np.quantile(np.abs(resid), np.clip(1.0 - alpha_eff, 0.50, 0.99))) if len(resid) else float(np.nanstd(y) * 0.1)
    hist_abs = np.abs(resid)
    q90 = float(np.quantile(hist_abs, 0.90)) if len(hist_abs) else q_abs
    q95 = float(np.quantile(hist_abs, 0.95)) if len(hist_abs) else q_abs

    if is_precip(variable):
        band_cap = max(q90 * 1.0, q_abs * 1.02, 1e-6)
        band_floor = max(q_abs * 0.08, 1e-6)
    else:
        band_cap = max(q95 * 1.18, q_abs * 1.10, 1e-6)
        band_floor = max(q_abs * 0.10, 1e-6)

    # Bias calibration from historical fit.
    hist_err = np.asarray(y, dtype=float) - yhat_hist
    global_bias = float(np.nanmean(hist_err)) if len(hist_err) else 0.0
    month_bias_map: dict[int, float] = {}
    month_level_map: dict[int, float] = {}
    if freq == "MS" and len(ds) == len(hist_err):
        bdf = pd.DataFrame({"ds": pd.to_datetime(ds), "err": hist_err})
        grp = bdf.groupby(bdf["ds"].dt.month)["err"].agg(["mean", "count"]).reset_index()
        for r in grp.itertuples(index=False):
            m = int(r.ds)
            mean_m = float(r.mean)
            n = int(r.count)
            w = float(np.clip(n / (n + 3.0), 0.0, 1.0))
            month_bias_map[m] = float(global_bias + w * (mean_m - global_bias))
        lvl = pd.DataFrame({"ds": pd.to_datetime(ds), "y": np.asarray(y, dtype=float)})
        lvl_grp = lvl.groupby(lvl["ds"].dt.month)["y"].mean().reset_index()
        for r in lvl_grp.itertuples(index=False):
            month_level_map[int(r.ds)] = float(r.y)

    return {
        "alpha_eff": float(alpha_eff),
        "q_norm": float(q_norm),
        "q_abs": float(q_abs),
        "band_cap": float(band_cap),
        "band_floor": float(band_floor),
        "global_bias": float(global_bias),
        "month_bias_map": month_bias_map,
        "month_level_map": month_level_map,
    }, yhat_hist


def run_recursive_retrain(
    ds_hist: pd.Series,
    y_hist: np.ndarray,
    future_idx: pd.DatetimeIndex,
    freq: str,
    variable: str,
    args: argparse.Namespace,
    climate_cfg: Any,
    max_train_points: int,
    max_future_steps: int,
) -> pd.DataFrame:
    train_ds_full = pd.Series(pd.to_datetime(ds_hist).reset_index(drop=True))
    train_y_full = np.asarray(y_hist, dtype=float).copy()

    cal, yhat_hist = calibration_stats(ds=train_ds_full, y=train_y_full, freq=freq, variable=variable, args=args)

    hist_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(ds_hist),
            "actual": np.asarray(y_hist, dtype=float),
            "yhat": yhat_hist,
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
            "is_forecast": False,
            "bias_correction": 0.0,
            "mean_reversion": 0.0,
            "climate_delta": 0.0,
            "train_until": pd.NaT,
            "train_size": len(y_hist),
            "step_no": 0,
        }
    )

    rows: list[dict[str, Any]] = []
    limit = int(max_future_steps) if int(max_future_steps) > 0 else len(future_idx)

    for step_no, t in enumerate(future_idx[:limit], start=1):
        if max_train_points > 0 and len(train_y_full) > max_train_points:
            ds_win = train_ds_full.iloc[-max_train_points:].reset_index(drop=True)
            y_win = train_y_full[-max_train_points:]
        else:
            ds_win = train_ds_full.reset_index(drop=True)
            y_win = train_y_full

        fit = fit_forecast_quant(
            ds=ds_win,
            y=y_win,
            future_ds=pd.Series(pd.DatetimeIndex([t])),
            freq=freq,
            regime_k=int(args.regime_k),
            regime_maxiter=int(args.regime_maxiter),
            vol_model=str(args.vol_model),
            ewma_lam=float(args.ewma_lambda),
            egarch_p=int(args.egarch_p),
            egarch_o=int(args.egarch_o),
            egarch_q=int(args.egarch_q),
            egarch_dist=str(args.egarch_dist),
            variable=variable,
        )

        yhat = float(fit["yhat_fc"][0]) if len(fit["yhat_fc"]) else float(np.nanmean(y_win))
        bias_adj = float(cal.get("global_bias", 0.0))
        if freq == "MS":
            mm = int(pd.Timestamp(t).month)
            bias_adj = float(cal.get("month_bias_map", {}).get(mm, bias_adj))
        yhat = yhat + bias_adj
        climate_delta = climate_delta_scalar(t, variable=variable, cfg=climate_cfg)
        yhat = yhat + float(climate_delta)
        mean_rev_adj = 0.0
        if freq == "MS":
            mm = int(pd.Timestamp(t).month)
            month_level = cal.get("month_level_map", {}).get(mm)
            if month_level is not None and np.isfinite(float(month_level)):
                # Reduce recursive long-horizon drift by shrinking toward monthly climatology.
                horizon_frac = float(step_no / max(limit, 1))
                rev_w = float(np.clip(0.10 + 0.25 * horizon_frac, 0.10, 0.35))
                yhat_prev = float(yhat)
                yhat = (1.0 - rev_w) * float(yhat_prev) + rev_w * float(month_level)
                mean_rev_adj = float(yhat - yhat_prev)
        yhat = float(apply_bounds(np.array([yhat]), variable)[0])

        sigma_fc = float(np.sqrt(max(float(fit["var_fc"][0]), 1e-9))) if len(fit["var_fc"]) else float(np.nanstd(y_win) * 0.1)
        band = float(np.clip(cal["q_norm"] * sigma_fc, cal["band_floor"], cal["band_cap"]))

        lo = float(apply_bounds(np.array([yhat - band]), variable)[0])
        hi = float(apply_bounds(np.array([yhat + band]), variable)[0])

        rows.append(
            {
                "ds": pd.Timestamp(t),
                "actual": np.nan,
                "yhat": yhat,
                "yhat_lower": lo,
                "yhat_upper": hi,
                "is_forecast": True,
                "bias_correction": float(bias_adj),
                "mean_reversion": float(mean_rev_adj),
                "climate_delta": float(climate_delta),
                "train_until": pd.Timestamp(ds_win.iloc[-1]),
                "train_size": int(len(y_win)),
                "step_no": int(step_no),
            }
        )

        train_ds_full = pd.concat([train_ds_full, pd.Series([pd.Timestamp(t)])], ignore_index=True)
        train_y_full = np.append(train_y_full, yhat)

    fc_df = pd.DataFrame(rows)
    if fc_df.empty:
        return hist_df
    return pd.concat([hist_df, fc_df], ignore_index=True)


def plot_walkforward(df: pd.DataFrame, last_obs: pd.Timestamp, variable: str, freq_label: str, chart_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5))
    hist = df[df["is_forecast"] == False]
    fc = df[df["is_forecast"] == True]

    if not hist.empty:
        ax.plot(hist["ds"], hist["actual"], color="#17becf", linewidth=1.0, alpha=0.6, label="Gözlem")
        ax.plot(hist["ds"], hist["yhat"], color="#1f77b4", linewidth=1.3, alpha=0.9, label="Geçmiş uyum")
    if not fc.empty:
        ax.plot(fc["ds"], fc["yhat"], color="#d62728", linewidth=1.9, label="Adım adım yeniden eğitim tahmini")
        if fc["yhat_lower"].notna().any() and fc["yhat_upper"].notna().any():
            ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], color="#d62728", alpha=0.14, label="Güven bandı")

    ax.axvline(last_obs, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_title(f"Walk-Forward Yeniden Eğitim - {variable_tr(variable)} ({freq_label})")
    ax.set_xlabel("Tarih")
    ax.set_ylabel(f"Değer ({infer_unit(variable)})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    climate_cfg = climate_cfg_from_args(args)

    raw = read_table(args.observations)
    norm_args = SimpleNamespace(
        input_kind=args.input_kind,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
        variable_col=args.variable_col,
        qc_col=args.qc_col,
        qc_ok_value=args.qc_ok_value,
        single_variable=args.single_variable,
    )
    obs, input_kind = normalize_observations(raw, norm_args)
    vars_use = select_variables(obs, args.variables)
    freqs = normalize_freq_list(args.frequencies)

    if not vars_use:
        raise SystemExit("Seçili değişken bulunamadı")
    if not freqs:
        raise SystemExit("Seçili frekans bulunamadı")

    out = args.output_dir
    fc_dir = out / "forecasts"
    ch_dir = out / "charts"
    rep_dir = out / "reports"
    ensure_dirs([out, fc_dir, ch_dir, rep_dir])

    print(f"Girdi: {args.observations} | tür={input_kind} | satır={len(obs)}")
    print(f"Değişkenler: {','.join(vars_use)} | Frekanslar: {','.join([f.code for f in freqs])}")

    index_rows: list[dict[str, Any]] = []

    for variable in vars_use:
        for fcfg in freqs:
            s = aggregate_series_freq(obs=obs, variable=variable, freq=fcfg.code, ok_value=args.qc_ok_value)
            if s.empty:
                continue

            s = winsorize(s, ql=float(args.winsor_lower), qu=float(args.winsor_upper))
            if len(s) < int(fcfg.min_train):
                continue

            ds_hist = pd.Series(pd.to_datetime(s.index))
            y_hist = s.values.astype(float)
            last_obs = pd.Timestamp(ds_hist.iloc[-1])
            climate_cfg_var = climate_cfg_with_series_baseline(climate_cfg, last_obs)

            future_idx = make_future_index(
                last_obs=last_obs,
                start_year=int(args.start_year),
                target_year=int(args.target_year),
                freq=fcfg.code,
            )
            if len(future_idx) == 0:
                continue

            max_train_points = int(args.max_train_points) if int(args.max_train_points) > 0 else int(fcfg.max_train_points_default)
            max_future_steps = int(args.max_future_steps)

            out_df = run_recursive_retrain(
                ds_hist=ds_hist,
                y_hist=y_hist,
                future_idx=future_idx,
                freq=fcfg.code,
                variable=variable,
                args=args,
                climate_cfg=climate_cfg_var,
                max_train_points=max_train_points,
                max_future_steps=max_future_steps,
            )

            out_df["variable"] = canonical_variable_name(variable)
            out_df["variable_tr"] = variable_tr(variable)
            out_df["frequency"] = fcfg.code
            out_df["frequency_tr"] = fcfg.label_tr
            out_df["model_strategy"] = f"walkforward_retrain_quant{climate_strategy_suffix(variable, climate_cfg_var)}"

            freq_tag = fcfg.code.lower().replace("-", "_")
            csv_path = fc_dir / f"{canonical_variable_name(variable)}_{freq_tag}_walkforward_{args.start_year}_{args.target_year}.csv"
            pq_path = fc_dir / f"{canonical_variable_name(variable)}_{freq_tag}_walkforward_{args.start_year}_{args.target_year}.parquet"
            chart_path = ch_dir / f"{canonical_variable_name(variable)}_{freq_tag}_walkforward_{args.start_year}_{args.target_year}.png"

            out_df.to_csv(csv_path, index=False)
            out_df.to_parquet(pq_path, index=False)
            plot_walkforward(out_df, last_obs=last_obs, variable=variable, freq_label=fcfg.label_tr, chart_path=chart_path)

            n_fc = int((out_df["is_forecast"] == True).sum())
            fc_part = out_df[out_df["is_forecast"] == True]
            rep = {
                "variable": canonical_variable_name(variable),
                "variable_tr": variable_tr(variable),
                "frequency": fcfg.code,
                "frequency_tr": fcfg.label_tr,
                "start_year": int(args.start_year),
                "target_year": int(args.target_year),
                "history_points": int(len(y_hist)),
                "forecast_steps": int(n_fc),
                "climate_adjustment_enabled": bool(climate_cfg_var.enabled),
                "climate_scenario": str(climate_cfg_var.scenario),
                "climate_adjustment_method": str(climate_cfg_var.method),
                "climate_baseline_year": float(climate_cfg_var.baseline_year),
                "climate_temp_rate_c_per_year": float(climate_cfg_var.temp_rate_c_per_year),
                "humidity_per_temp_c": float(climate_cfg_var.humidity_per_temp_c),
                "bias_correction_mode": "seasonal_monthly" if fcfg.code == "MS" else "global_only",
                "bias_correction_mean": float(fc_part["bias_correction"].mean()) if not fc_part.empty else 0.0,
                "mean_reversion_mean": float(fc_part["mean_reversion"].mean()) if not fc_part.empty else 0.0,
                "last_observed": str(last_obs.date()),
                "first_forecast": str(pd.Timestamp(future_idx[0]).date()) if len(future_idx) else None,
                "last_forecast": str(pd.Timestamp(future_idx[min(len(future_idx), max(n_fc, 1)) - 1]).date()) if len(future_idx) else None,
                "max_train_points_used": int(max_train_points),
                "max_future_steps": int(max_future_steps),
                "forecast_csv": str(csv_path),
                "forecast_parquet": str(pq_path),
                "chart_png": str(chart_path),
            }
            rep_path = rep_dir / f"{canonical_variable_name(variable)}_{freq_tag}_walkforward_{args.start_year}_{args.target_year}.json"
            rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

            index_rows.append(
                {
                    "variable": canonical_variable_name(variable),
                    "variable_tr": variable_tr(variable),
                    "frequency": fcfg.code,
                    "frequency_tr": fcfg.label_tr,
                    "history_points": len(y_hist),
                    "forecast_steps": n_fc,
                    "last_observed": str(last_obs.date()),
                    "model_strategy": f"walkforward_retrain_quant{climate_strategy_suffix(variable, climate_cfg_var)}",
                    "climate_adjustment_enabled": bool(climate_cfg_var.enabled),
                    "climate_scenario": str(climate_cfg_var.scenario),
                    "climate_adjustment_method": str(climate_cfg_var.method),
                    "climate_baseline_year": float(climate_cfg_var.baseline_year),
                    "bias_correction_mode": "seasonal_monthly" if fcfg.code == "MS" else "global_only",
                    "bias_correction_mean": float(fc_part["bias_correction"].mean()) if not fc_part.empty else 0.0,
                    "mean_reversion_mean": float(fc_part["mean_reversion"].mean()) if not fc_part.empty else 0.0,
                    "forecast_csv": str(csv_path),
                    "forecast_parquet": str(pq_path),
                    "chart_png": str(chart_path),
                    "report_json": str(rep_path),
                }
            )

    idx = pd.DataFrame(index_rows).sort_values(["variable", "frequency"]) if index_rows else pd.DataFrame()
    idx_csv = out / f"walkforward_index_{args.start_year}_{args.target_year}.csv"
    idx_pq = out / f"walkforward_index_{args.start_year}_{args.target_year}.parquet"
    idx.to_csv(idx_csv, index=False)
    idx.to_parquet(idx_pq, index=False)

    print("Walk-forward yeniden eğitim tamamlandı.")
    print(f"İndeks: {idx_csv}")
    if not idx.empty:
        cols = ["variable", "frequency", "history_points", "forecast_steps", "forecast_csv"]
        print(idx[cols].to_string(index=False))


if __name__ == "__main__":
    main()
