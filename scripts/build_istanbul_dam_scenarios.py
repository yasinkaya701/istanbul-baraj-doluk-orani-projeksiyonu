#!/usr/bin/env python3
"""Build scenario variants using empirical CV residuals.

Scenarios:
- baseline
- dry_stress
- wet_relief

Supports two shift modes:
- static: fixed dry/wet multipliers
- series_month: seasonal multipliers per series+month derived from historical
  monthly variability (q90-q10 amplitude), with clipping and smoothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Istanbul dam scenario forecasts (empirical residual method)")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision"),
    )
    p.add_argument("--dry-shift-k", type=float, default=0.8, help="Base shift multiplier for dry scenario")
    p.add_argument("--wet-shift-k", type=float, default=0.8, help="Base shift multiplier for wet scenario")
    p.add_argument(
        "--max-shift-abs",
        type=float,
        default=0.12,
        help="Maximum absolute shift on occupancy ratio scale (0.12 -> 12 points)",
    )
    p.add_argument("--window-start", default="2026-03-01")
    p.add_argument("--window-end", default="2027-02-01")
    p.add_argument("--alpha", type=float, default=0.10, help="Interval alpha, default 0.10 -> 90% interval")
    p.add_argument("--ensemble-max-models", type=int, default=3)
    p.add_argument(
        "--forecast-file",
        type=Path,
        default=None,
        help="Optional forecast file. Default: calibrated file if exists, else base decision forecast.",
    )
    p.add_argument(
        "--shift-mode",
        choices=["static", "series_month"],
        default="series_month",
        help="Scenario shift mode.",
    )
    p.add_argument(
        "--expand-severity-grid",
        action="store_true",
        help="If set, generate dry/wet scenario families using severity levels.",
    )
    p.add_argument(
        "--severity-levels",
        default="mild:0.6,base:1.0,severe:1.4",
        help="Comma-separated label:multiplier list used when --expand-severity-grid is set.",
    )
    p.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Monthly history CSV path. Required for series_month mode (auto: input-dir/istanbul_dam_monthly_history.csv).",
    )
    p.add_argument("--seasonal-min-ratio", type=float, default=0.70, help="Lower clip for seasonal ratio")
    p.add_argument("--seasonal-max-ratio", type=float, default=1.60, help="Upper clip for seasonal ratio")
    p.add_argument(
        "--seasonal-smooth",
        type=float,
        default=0.35,
        help="Smoothing toward neighboring months. 0=no smoothing, 1=full neighbor average.",
    )
    return p.parse_args()


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def parse_labeled_floats(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for token in str(spec).split(","):
        t = token.strip()
        if not t:
            continue
        if ":" not in t:
            raise ValueError(f"Invalid label:value token: {t}")
        k, v = t.split(":", 1)
        key = k.strip().lower()
        val = float(v.strip())
        out[key] = val
    if not out:
        raise ValueError("No valid severity levels parsed.")
    return out


def _ensemble_residuals_from_cv(
    cv_series: pd.DataFrame,
    metrics_series: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    base_metrics = metrics_series[~metrics_series["model"].isin(["ensemble_topk"])].copy()
    base_metrics = base_metrics.sort_values("rmse").head(max(1, int(top_k)))
    if base_metrics.empty:
        return pd.DataFrame(columns=["ds", "residual"])

    inv = 1.0 / np.maximum(base_metrics["rmse"].to_numpy(dtype=float), 1e-8)
    w = inv / inv.sum()
    weights = dict(zip(base_metrics["model"].tolist(), w.tolist(), strict=False))

    piv = cv_series.pivot_table(index=["split", "ds", "actual"], columns="model", values="yhat").reset_index()
    usable = [m for m in weights if m in piv.columns]
    if not usable:
        return pd.DataFrame(columns=["ds", "residual"])

    pred = np.zeros(len(piv), dtype=float)
    wsum = 0.0
    for m in usable:
        wm = float(weights[m])
        pred += wm * piv[m].to_numpy(dtype=float)
        wsum += wm
    pred = clip01(pred / max(wsum, 1e-8))
    out = pd.DataFrame({"ds": pd.to_datetime(piv["ds"]), "residual": piv["actual"].to_numpy(dtype=float) - pred})
    return out


def build_residual_pools(
    cv: pd.DataFrame,
    metrics: pd.DataFrame,
    strategy: pd.DataFrame,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}
    for series_name, g in strategy.groupby("series"):
        strat = str(g["strategy"].iloc[0])
        cv_s = cv[cv["series"] == series_name].copy()
        if cv_s.empty:
            pools[series_name] = {"global": np.array([], dtype=float), "by_month": {}}
            continue

        if strat == "ensemble_topk":
            m_s = metrics[metrics["series"] == series_name].copy()
            res_df = _ensemble_residuals_from_cv(cv_s, m_s, top_k=top_k)
        else:
            res_df = cv_s[cv_s["model"] == strat][["ds", "residual"]].copy()
            res_df["ds"] = pd.to_datetime(res_df["ds"])

        if res_df.empty:
            pools[series_name] = {"global": np.array([], dtype=float), "by_month": {}}
            continue

        by_month: dict[int, np.ndarray] = {}
        for month, gm in res_df.groupby(res_df["ds"].dt.month):
            by_month[int(month)] = gm["residual"].dropna().to_numpy(dtype=float)
        pools[series_name] = {
            "global": res_df["residual"].dropna().to_numpy(dtype=float),
            "by_month": by_month,
        }
    return pools


def row_residual_sample(pool: dict[str, Any], ds: pd.Timestamp) -> np.ndarray:
    month = int(ds.month)
    by_month = pool.get("by_month", {})
    arr = by_month.get(month)
    if arr is not None and len(arr) >= 5:
        return arr
    return pool.get("global", np.array([], dtype=float))


def smooth_circular_month(values: dict[int, float], smooth: float) -> dict[int, float]:
    if smooth <= 0.0:
        return values
    out = {}
    a = float(np.clip(smooth, 0.0, 1.0))
    for m in range(1, 13):
        prev_m = 12 if m == 1 else m - 1
        next_m = 1 if m == 12 else m + 1
        self_v = float(values.get(m, 1.0))
        neigh = 0.5 * (float(values.get(prev_m, self_v)) + float(values.get(next_m, self_v)))
        out[m] = (1.0 - a) * self_v + a * neigh
    return out


def build_seasonal_ratio_table(
    history_df: pd.DataFrame,
    series_names: list[str],
    min_ratio: float,
    max_ratio: float,
    smooth: float,
) -> pd.DataFrame:
    rows = []
    hist = history_df.copy()
    hist["month"] = pd.to_datetime(hist["ds"]).dt.month
    min_r = float(min(min_ratio, max_ratio))
    max_r = float(max(min_ratio, max_ratio))

    for s in series_names:
        if s not in hist.columns:
            continue
        gm = (
            hist.groupby("month")[s]
            .agg(q10=lambda x: float(np.nanquantile(x, 0.10)), q90=lambda x: float(np.nanquantile(x, 0.90)))
            .reset_index()
        )
        gm["amp"] = (gm["q90"] - gm["q10"]).clip(lower=1e-6)
        global_amp = float(np.nanmedian(gm["amp"])) if not gm["amp"].isna().all() else float(np.nan)
        if not np.isfinite(global_amp) or global_amp <= 0:
            global_amp = float(np.nanmean(gm["amp"])) if not gm["amp"].isna().all() else 1.0
        if not np.isfinite(global_amp) or global_amp <= 0:
            global_amp = 1.0
        raw = {int(r.month): float(r.amp / global_amp) for r in gm.itertuples(index=False)}
        clipped = {m: float(np.clip(v, min_r, max_r)) for m, v in raw.items()}
        smooth_map = smooth_circular_month(clipped, smooth=smooth)

        for m in range(1, 13):
            rows.append(
                {
                    "series": s,
                    "month": m,
                    "raw_ratio": float(raw.get(m, 1.0)),
                    "clipped_ratio": float(clipped.get(m, 1.0)),
                    "seasonal_ratio": float(smooth_map.get(m, 1.0)),
                    "global_amp": global_amp,
                }
            )
    out = pd.DataFrame(rows).sort_values(["series", "month"]).reset_index(drop=True)
    return out


def get_shift_multiplier_vector(
    df: pd.DataFrame,
    scenario_base_mult: float,
    shift_mode: str,
    ratio_map: dict[tuple[str, int], float],
) -> np.ndarray:
    base = float(scenario_base_mult)
    if shift_mode == "static" or abs(base) < 1e-12:
        return np.full(len(df), base, dtype=float)
    vals = []
    for s, ds in zip(df["series"], df["ds"], strict=False):
        ratio = float(ratio_map.get((str(s), int(pd.Timestamp(ds).month)), 1.0))
        vals.append(base * ratio)
    return np.asarray(vals, dtype=float)


def make_scenario_mean(
    base_yhat: np.ndarray,
    interval_q: np.ndarray,
    shift_mult: np.ndarray | float,
    max_shift_abs: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(base_yhat, dtype=float)
    q = np.asarray(interval_q, dtype=float)
    m = np.asarray(shift_mult, dtype=float)
    if m.ndim == 0:
        m = np.full_like(y, float(m))
    raw_shift = np.abs(m) * q
    shift = np.minimum(raw_shift, float(max_shift_abs))
    signed_shift = np.sign(m) * shift
    return clip01(y + signed_shift), signed_shift


def apply_empirical_prob_and_interval(
    scenario_df: pd.DataFrame,
    pools: dict[str, dict[str, Any]],
    alpha: float,
) -> pd.DataFrame:
    out = scenario_df.copy()
    p40, p30, low, high = [], [], [], []
    q_low = float(alpha / 2.0)
    q_high = float(1.0 - alpha / 2.0)

    for _, row in out.iterrows():
        series_name = str(row["series"])
        ds = pd.Timestamp(row["ds"])
        mean = float(row["scenario_yhat"])
        pool = pools.get(series_name, {"global": np.array([], dtype=float), "by_month": {}})
        resid = row_residual_sample(pool, ds=ds)

        if resid.size == 0:
            qabs = float(row.get("interval_q_abs", 0.05))
            low_val = float(clip01(mean - qabs))
            high_val = float(clip01(mean + qabs))
            p40_val = float(mean < 0.40)
            p30_val = float(mean < 0.30)
        else:
            low_val = float(clip01(mean + float(np.quantile(resid, q_low))))
            high_val = float(clip01(mean + float(np.quantile(resid, q_high))))
            p40_val = float(np.mean((mean + resid) < 0.40))
            p30_val = float(np.mean((mean + resid) < 0.30))

        low.append(low_val)
        high.append(high_val)
        p40.append(p40_val)
        p30.append(p30_val)

    out["scenario_yhat_lower"] = low
    out["scenario_yhat_upper"] = high
    out["scenario_prob_below_40"] = p40
    out["scenario_prob_below_30"] = p30
    return out


def build_risk_summary(scen_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    tmp = scen_df[(scen_df["ds"] >= start) & (scen_df["ds"] <= end)].copy()
    rows = []
    for (scenario, series), g in tmp.groupby(["scenario", "series"]):
        i_min = g["scenario_yhat"].idxmin()
        rows.append(
            {
                "scenario": scenario,
                "series": series,
                "strategy": str(g["strategy"].iloc[0]),
                "scenario_group": str(g["scenario_group"].iloc[0]) if "scenario_group" in g.columns else "unknown",
                "severity_label": str(g["severity_label"].iloc[0]) if "severity_label" in g.columns else "base",
                "severity_multiplier": float(g["severity_multiplier"].iloc[0]) if "severity_multiplier" in g.columns else 1.0,
                "months_lt40": int((g["scenario_yhat"] < 0.40).sum()),
                "months_lt30": int((g["scenario_yhat"] < 0.30).sum()),
                "mean_prob_below_40_pct": float(g["scenario_prob_below_40"].mean() * 100.0),
                "mean_prob_below_30_pct": float(g["scenario_prob_below_30"].mean() * 100.0),
                "mean_yhat_pct": float(g["scenario_yhat"].mean() * 100.0),
                "worst_month": str(pd.Timestamp(g.loc[i_min, "ds"]).date()),
                "worst_yhat_pct": float(g["scenario_yhat"].min() * 100.0),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["scenario", "months_lt40", "months_lt30", "mean_prob_below_40_pct", "worst_yhat_pct"],
        ascending=[True, False, False, False, True],
    )
    return out.reset_index(drop=True)


def plot_shift_heatmap(shift_factors: pd.DataFrame, out_png: Path) -> None:
    if shift_factors.empty:
        return
    # show dry multiplier magnitude as seasonal structure reference.
    if "scenario_group" in shift_factors.columns:
        g = shift_factors[shift_factors["scenario_group"] == "dry"].copy()
    else:
        g = shift_factors[shift_factors["scenario"].astype(str).str.startswith("dry")].copy()
    if g.empty:
        return
    # If multiple dry scenarios exist, keep the strongest absolute multiplier per series-month.
    g["abs_k"] = g["effective_shift_k"].abs()
    g = (
        g.sort_values("abs_k", ascending=False)
        .drop_duplicates(["series", "month"])
        .copy()
    )
    order = (
        g.groupby("series", as_index=False)["effective_shift_k"]
        .mean()
        .sort_values("effective_shift_k", ascending=False)["series"]
        .tolist()
    )
    mat = np.full((len(order), 12), np.nan, dtype=float)
    for i, s in enumerate(order):
        gs = g[g["series"] == s]
        for _, r in gs.iterrows():
            m = int(r["month"])
            mat[i, m - 1] = abs(float(r["effective_shift_k"]))

    fig, ax = plt.subplots(figsize=(12, max(4.8, len(order) * 0.34)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="YlOrRd")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([str(m) for m in range(1, 13)])
    ax.set_xlabel("Month")
    ax.set_title("Scenario shift multipliers |dry| (series x month)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label("|effective shift k|", rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.forecast_file is not None:
        forecast_path = args.forecast_file
    else:
        calibrated = args.input_dir / "istanbul_dam_forecasts_decision_calibrated.csv"
        base = args.input_dir / "istanbul_dam_forecasts_decision.csv"
        forecast_path = calibrated if calibrated.exists() else base
    cv_path = args.input_dir / "istanbul_dam_cv_predictions_decision.csv"
    metrics_path = args.input_dir / "istanbul_dam_cv_metrics_decision.csv"
    strategy_path = args.input_dir / "strategy_summary.csv"
    required = [forecast_path, cv_path, metrics_path, strategy_path]
    if not all(p.exists() for p in required):
        raise SystemExit("Missing required decision output files. Run forecast_istanbul_dam_decision_support.py first.")

    base = pd.read_csv(forecast_path, parse_dates=["ds"])
    cv = pd.read_csv(cv_path, parse_dates=["ds"])
    metrics = pd.read_csv(metrics_path)
    strategy = pd.read_csv(strategy_path)

    needed_cols = [
        "ds",
        "series",
        "strategy",
        "strategy_rmse",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "interval_q_abs",
    ]
    miss = [c for c in needed_cols if c not in base.columns]
    if miss:
        raise SystemExit(f"Missing required columns in decision forecast: {miss}")
    base = base[needed_cols].copy()
    base["month"] = pd.to_datetime(base["ds"]).dt.month

    ratio_table = pd.DataFrame(columns=["series", "month", "seasonal_ratio"])
    ratio_map: dict[tuple[str, int], float] = {}
    history_source = None
    if args.shift_mode == "series_month":
        hist_path = args.history_file if args.history_file is not None else (args.input_dir / "istanbul_dam_monthly_history.csv")
        if not hist_path.exists():
            raise SystemExit("series_month mode requires monthly history file.")
        hist = pd.read_csv(hist_path, parse_dates=["ds"])
        series_names = sorted(base["series"].dropna().astype(str).unique().tolist())
        ratio_table = build_seasonal_ratio_table(
            history_df=hist,
            series_names=series_names,
            min_ratio=float(args.seasonal_min_ratio),
            max_ratio=float(args.seasonal_max_ratio),
            smooth=float(args.seasonal_smooth),
        )
        ratio_map = {(str(r.series), int(r.month)): float(r.seasonal_ratio) for r in ratio_table.itertuples(index=False)}
        history_source = str(hist_path)

    pools = build_residual_pools(cv=cv, metrics=metrics, strategy=strategy, top_k=args.ensemble_max_models)

    scenario_defs: list[dict[str, Any]] = []
    scenario_defs.append(
        {
            "scenario": "baseline",
            "group": "baseline",
            "severity_label": "base",
            "severity_multiplier": 1.0,
            "base_shift_k": 0.0,
        }
    )
    if args.expand_severity_grid:
        sev_map = parse_labeled_floats(args.severity_levels)
        if "base" not in sev_map:
            sev_map["base"] = 1.0
        for label, sev_mult in sorted(sev_map.items(), key=lambda x: x[1]):
            dry_name = "dry_stress" if label == "base" else f"dry_{label}"
            wet_name = "wet_relief" if label == "base" else f"wet_{label}"
            scenario_defs.append(
                {
                    "scenario": dry_name,
                    "group": "dry",
                    "severity_label": label,
                    "severity_multiplier": float(sev_mult),
                    "base_shift_k": -abs(float(args.dry_shift_k)) * float(sev_mult),
                }
            )
            scenario_defs.append(
                {
                    "scenario": wet_name,
                    "group": "wet",
                    "severity_label": label,
                    "severity_multiplier": float(sev_mult),
                    "base_shift_k": abs(float(args.wet_shift_k)) * float(sev_mult),
                }
            )
    else:
        scenario_defs.extend(
            [
                {
                    "scenario": "dry_stress",
                    "group": "dry",
                    "severity_label": "base",
                    "severity_multiplier": 1.0,
                    "base_shift_k": -abs(float(args.dry_shift_k)),
                },
                {
                    "scenario": "wet_relief",
                    "group": "wet",
                    "severity_label": "base",
                    "severity_multiplier": 1.0,
                    "base_shift_k": abs(float(args.wet_shift_k)),
                },
            ]
        )

    scenario_frames = []
    shift_rows = []
    for scen in scenario_defs:
        scenario_name = str(scen["scenario"])
        base_mult = float(scen["base_shift_k"])
        df = base.copy()
        mult_vec = get_shift_multiplier_vector(
            df=df,
            scenario_base_mult=float(base_mult),
            shift_mode=str(args.shift_mode),
            ratio_map=ratio_map,
        )
        scenario_mean, signed_shift = make_scenario_mean(
            base_yhat=df["yhat"].to_numpy(dtype=float),
            interval_q=df["interval_q_abs"].to_numpy(dtype=float),
            shift_mult=mult_vec,
            max_shift_abs=float(args.max_shift_abs),
        )
        df["scenario"] = scenario_name
        df["scenario_group"] = str(scen["group"])
        df["severity_label"] = str(scen["severity_label"])
        df["severity_multiplier"] = float(scen["severity_multiplier"])
        df["base_shift_k"] = float(base_mult)
        df["effective_shift_k"] = mult_vec
        df["scenario_shift_applied"] = signed_shift
        df["scenario_yhat"] = scenario_mean
        df = apply_empirical_prob_and_interval(df, pools=pools, alpha=float(args.alpha))
        scenario_frames.append(df)

        # Monthly factor transparency table
        for (series_name, month), gm in df.groupby(["series", "month"]):
            shift_rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_group": str(scen["group"]),
                    "series": str(series_name),
                    "month": int(month),
                    "base_shift_k": float(base_mult),
                    "severity_label": str(scen["severity_label"]),
                    "severity_multiplier": float(scen["severity_multiplier"]),
                    "seasonal_ratio": float(ratio_map.get((str(series_name), int(month)), 1.0)),
                    "effective_shift_k": float(gm["effective_shift_k"].mean()),
                }
            )

    scen_df = pd.concat(scenario_frames, ignore_index=True).drop(columns=["month"])
    scen_out = args.output_dir / "scenario_forecasts.csv"
    scen_df.to_csv(scen_out, index=False)

    risk = build_risk_summary(scen_df, start=args.window_start, end=args.window_end)
    risk_out = args.output_dir / "scenario_risk_summary.csv"
    risk.to_csv(risk_out, index=False)

    shift_df = pd.DataFrame(shift_rows).sort_values(["scenario", "series", "month"]).reset_index(drop=True)
    shift_out = args.output_dir / "scenario_shift_factors.csv"
    shift_df.to_csv(shift_out, index=False)
    catalog_out = args.output_dir / "scenario_catalog.csv"
    pd.DataFrame(scenario_defs).to_csv(catalog_out, index=False)

    ratio_out = args.output_dir / "scenario_seasonal_ratios.csv"
    if not ratio_table.empty:
        ratio_table.to_csv(ratio_out, index=False)

    shift_heatmap = args.output_dir / "scenario_shift_factors_heatmap.png"
    plot_shift_heatmap(shift_df, shift_heatmap)

    summary = {
        "window_start": args.window_start,
        "window_end": args.window_end,
        "dry_shift_k": float(args.dry_shift_k),
        "wet_shift_k": float(args.wet_shift_k),
        "max_shift_abs": float(args.max_shift_abs),
        "alpha": float(args.alpha),
        "method": "empirical_cv_residual_bootstrap",
        "shift_mode": str(args.shift_mode),
        "seasonal_min_ratio": float(args.seasonal_min_ratio),
        "seasonal_max_ratio": float(args.seasonal_max_ratio),
        "seasonal_smooth": float(args.seasonal_smooth),
        "expand_severity_grid": bool(args.expand_severity_grid),
        "severity_levels": str(args.severity_levels),
        "forecast_source": str(forecast_path),
        "history_source": history_source,
        "scenario_count": int(scen_df["scenario"].nunique()),
        "scenario_names": sorted(scen_df["scenario"].dropna().astype(str).unique().tolist()),
        "record_count": int(len(scen_df)),
    }
    (args.output_dir / "scenario_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(scen_out)
    print(risk_out)
    print(shift_out)
    print(catalog_out)
    if not ratio_table.empty:
        print(ratio_out)
    print(shift_heatmap)


if __name__ == "__main__":
    main()
