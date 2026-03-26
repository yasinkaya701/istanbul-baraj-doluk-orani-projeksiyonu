#!/usr/bin/env python3
"""Build v4 arbitrated final forecast from calibrated model candidates.

Selection logic per variable:
1) Start with model holdout metric (RMSE-like).
2) Apply sanity penalties:
   - non-finite / extreme forecast
   - physical range violation
   - abrupt jump at forecast start
   - excessive volatility vs recent history
3) Pick lowest total score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Candidate:
    variable: str
    model_name: str
    source: str
    metric_name: str
    metric_value: float
    forecast_csv: Path


RANGES = {
    "temp": (-60.0, 60.0),
    "humidity": (0.0, 100.0),
    "pressure": (850.0, 1100.0),
    "precip": (0.0, 500.0),  # monthly precipitation plausibility guard
}

# Domain guards for final horizon values and trajectory.
END_BOUNDS = {
    "temp": (-20.0, 20.0),
    "humidity": (65.0, 95.0),
    "pressure": (930.0, 1035.0),
    "precip": (2.0, 200.0),
}
DELTA_ABS_MAX = {
    "temp": 10.0,
    "humidity": 8.0,
    "pressure": 20.0,
    "precip": 120.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v4 arbitrated final forecast")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/data_factory/run_20260306_000419"),
    )
    return p.parse_args()


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def confidence_from_score(score_total: float, penalty: float) -> tuple[float, str]:
    s = safe_float(score_total, default=np.inf)
    p = safe_float(penalty, default=0.0)
    if not np.isfinite(s):
        return 0.0, "D"
    conf = 100.0 - (12.0 * s) - (1.8 * p)
    conf = float(np.clip(conf, 0.0, 100.0))
    if conf >= 85.0:
        grade = "A"
    elif conf >= 70.0:
        grade = "B"
    elif conf >= 55.0:
        grade = "C"
    else:
        grade = "D"
    return conf, grade


def add_quant_candidates(run_dir: Path, out: list[Candidate]) -> None:
    idx = run_dir / "calibrated_quant" / "quant_index_to_2035.csv"
    if not idx.exists():
        return
    d = pd.read_csv(idx)
    for _, r in d.iterrows():
        var = str(r.get("variable", "")).strip()
        if var not in {"humidity", "precip"}:
            continue
        out.append(
            Candidate(
                variable=var,
                model_name="quant_calibrated",
                source="calibrated_quant",
                metric_name="cv_rmse",
                metric_value=safe_float(r.get("cv_rmse")),
                forecast_csv=Path(str(r.get("forecast_csv", ""))),
            )
        )


def add_strong_candidates(run_dir: Path, out: list[Candidate]) -> None:
    strong_sets = [
        ("calibrated_strong", run_dir / "calibrated_strong" / "strong_ensemble_index_to_2035.csv"),
        ("calibrated_strong_hp", run_dir / "calibrated_strong_hp" / "strong_ensemble_index_to_2035.csv"),
        ("calibrated_strong_t", run_dir / "calibrated_strong_t" / "strong_ensemble_index_to_2035.csv"),
    ]
    for source_name, idx in strong_sets:
        if not idx.exists():
            continue
        d = pd.read_csv(idx)
        for _, r in d.iterrows():
            var = str(r.get("variable", "")).strip()
            if var not in {"temp", "pressure", "humidity", "precip"}:
                continue
            out.append(
                Candidate(
                    variable=var,
                    model_name=f"strong_{source_name}",
                    source=source_name,
                    metric_name="best_cv_rmse",
                    metric_value=safe_float(r.get("best_cv_rmse")),
                    forecast_csv=Path(str(r.get("forecast_csv", ""))),
                )
            )


def add_prophet_candidates(run_dir: Path, out: list[Candidate]) -> None:
    prophet_sets = [
        ("calibrated_prophet", run_dir / "calibrated_prophet" / "prophet_index_to_2035.csv"),
        ("calibrated_prophet_hp", run_dir / "calibrated_prophet_hp" / "prophet_index_to_2035.csv"),
        ("calibrated_prophet_p", run_dir / "calibrated_prophet_p" / "prophet_index_to_2035.csv"),
    ]
    for source_name, idx in prophet_sets:
        if not idx.exists():
            continue
        d = pd.read_csv(idx)
        for _, r in d.iterrows():
            var = str(r.get("variable", "")).strip()
            if var not in {"temp", "pressure", "humidity", "precip"}:
                continue
            out.append(
                Candidate(
                    variable=var,
                    model_name=f"prophet_{source_name}",
                    source=source_name,
                    metric_name="holdout_rmse",
                    metric_value=safe_float(r.get("holdout_rmse")),
                    forecast_csv=Path(str(r.get("forecast_csv", ""))),
                )
            )


def load_fc(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
    d = d.dropna(subset=["ds"]).sort_values("ds")
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "is_forecast" not in d.columns:
        d["is_forecast"] = False
    return d


def candidate_score(c: Candidate) -> dict[str, Any]:
    if not c.forecast_csv.exists():
        return {
            "ok": False,
            "score_total": np.inf,
            "penalty": 1000.0,
            "note": "forecast_missing",
        }
    d = load_fc(c.forecast_csv)
    hist = d[d["is_forecast"] == False].copy()
    fut = d[d["is_forecast"] == True].copy()
    if fut.empty:
        return {
            "ok": False,
            "score_total": np.inf,
            "penalty": 1000.0,
            "note": "forecast_empty",
        }

    y_hist = hist["yhat"].dropna().to_numpy(dtype=float)
    y_fut = fut["yhat"].dropna().to_numpy(dtype=float)
    y0 = float(y_fut[0]) if len(y_fut) else np.nan
    y1 = float(y_fut[-1]) if len(y_fut) else np.nan

    spread = float(np.nanpercentile(y_hist, 90) - np.nanpercentile(y_hist, 10)) if len(y_hist) >= 5 else float(np.nanstd(y_hist))
    spread = max(spread, 1e-6)
    metric = c.metric_value if np.isfinite(c.metric_value) else (np.nanstd(y_hist) * 10.0)
    base = float(metric / spread)

    penalty = 0.0
    notes: list[str] = []
    lo, hi = RANGES.get(c.variable, (-np.inf, np.inf))

    if not np.isfinite(y0) or not np.isfinite(y1):
        penalty += 100.0
        notes.append("nan_forecast")
    if np.isfinite(y_fut).any():
        if np.nanmin(y_fut) < lo or np.nanmax(y_fut) > hi:
            penalty += 30.0
            notes.append("range_violation")

    # End-of-horizon guard.
    e_lo, e_hi = END_BOUNDS.get(c.variable, (-np.inf, np.inf))
    if np.isfinite(y1) and (y1 < e_lo or y1 > e_hi):
        penalty += 12.0
        notes.append("end_bound")

    if len(y_hist) and len(y_fut):
        last = float(y_hist[-1])
        jump = abs(y0 - last) / spread
        if jump > 4.0:
            penalty += 8.0
            notes.append("jump")
        dmax = DELTA_ABS_MAX.get(c.variable)
        if dmax is not None and abs(y1 - y0) > dmax:
            penalty += 6.0
            notes.append("delta_excess")

    if len(y_hist) >= 12 and len(y_fut) >= 12:
        hist_vol = float(np.nanstd(np.diff(y_hist[-24:])))
        fut_vol = float(np.nanstd(np.diff(y_fut[:24])))
        if hist_vol > 1e-9 and fut_vol / hist_vol > 6.0:
            penalty += 6.0
            notes.append("volatile")

    # Level consistency: forecast central level should stay near recent historical regime.
    if len(y_hist) >= 24 and len(y_fut) >= 24:
        h_ref = float(np.nanmedian(y_hist[-60:])) if len(y_hist) >= 60 else float(np.nanmedian(y_hist))
        f_ref = float(np.nanmedian(y_fut))
        if np.isfinite(h_ref) and abs(h_ref) > 1e-9 and np.isfinite(f_ref):
            ratio = f_ref / h_ref
            if c.variable == "pressure" and (ratio < 0.95 or ratio > 1.05):
                penalty += 8.0
                notes.append("level_shift")
            elif c.variable == "humidity" and (ratio < 0.85 or ratio > 1.15):
                penalty += 6.0
                notes.append("level_shift")
            elif c.variable == "temp" and (ratio < 0.60 or ratio > 1.60):
                penalty += 6.0
                notes.append("level_shift")
            elif c.variable == "precip" and (ratio < 0.50 or ratio > 2.00):
                penalty += 8.0
                notes.append("level_shift")

    # Trend-drift consistency: avoid models that diverge far faster than recent history.
    if len(y_hist) >= 36 and len(y_fut) >= 36:
        hh = y_hist[-120:] if len(y_hist) >= 120 else y_hist
        ff = y_fut
        h_mid = len(hh) // 2
        f_mid = len(ff) // 2
        h1 = float(np.nanmedian(hh[:h_mid]))
        h2 = float(np.nanmedian(hh[h_mid:]))
        f1 = float(np.nanmedian(ff[:f_mid]))
        f2 = float(np.nanmedian(ff[f_mid:]))
        # approximate per-window drift
        drift_hist = abs(h2 - h1)
        drift_fut = abs(f2 - f1)
        # variable-aware tolerance floor
        drift_floor = {"temp": 1.5, "humidity": 2.5, "pressure": 3.0, "precip": 15.0}.get(c.variable, 1.0)
        if np.isfinite(drift_fut) and np.isfinite(drift_hist):
            if drift_fut > max(drift_floor, 3.0 * drift_hist):
                penalty += 7.0
                notes.append("trend_drift")

    score_total = base + penalty
    return {
        "ok": True,
        "score_base": base,
        "penalty": penalty,
        "score_total": score_total,
        "note": ";".join(notes),
        "forecast_start": str(pd.to_datetime(fut["ds"].iloc[0]).date()),
        "forecast_end": str(pd.to_datetime(fut["ds"].iloc[-1]).date()),
        "forecast_start_yhat": y0,
        "forecast_end_yhat_2035": y1,
        "frequency": str(d["frequency"].dropna().iloc[0]) if "frequency" in d.columns and d["frequency"].notna().any() else "-",
    }


def main() -> None:
    args = parse_args()
    out_dir = args.run_dir / "quant" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []
    add_quant_candidates(args.run_dir, candidates)
    add_strong_candidates(args.run_dir, candidates)
    add_prophet_candidates(args.run_dir, candidates)

    scored_rows: list[dict[str, Any]] = []
    for c in candidates:
        sc = candidate_score(c)
        row = {
            "variable": c.variable,
            "model_name": c.model_name,
            "source": c.source,
            "metric_name": c.metric_name,
            "metric_value": c.metric_value,
            "forecast_csv": str(c.forecast_csv),
            **sc,
        }
        scored_rows.append(row)

    scored = pd.DataFrame(scored_rows)
    cand_csv = out_dir / "v4_candidate_scores.csv"
    scored.to_csv(cand_csv, index=False)

    sorted_ok = scored[scored["ok"] == True].sort_values(
        ["variable", "score_total", "penalty", "metric_value"], ascending=[True, True, True, True]
    )
    winners = sorted_ok.drop_duplicates(subset=["variable"], keep="first").reset_index(drop=True)
    conf_vals = winners.apply(lambda r: confidence_from_score(r.get("score_total"), r.get("penalty")), axis=1)
    winners["confidence_score"] = [x[0] for x in conf_vals]
    winners["confidence_grade"] = [x[1] for x in conf_vals]
    winners_csv = out_dir / "v4_final_arbitrated_ozet.csv"
    winners.to_csv(winners_csv, index=False)

    # Build dashboard from winners.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    order = ["temp", "humidity", "pressure", "precip"]
    for ax, var in zip(axes, order):
        w = winners[winners["variable"] == var]
        if w.empty:
            ax.set_axis_off()
            continue
        row = w.iloc[0]
        d = load_fc(Path(str(row["forecast_csv"])))
        hist = d[d["is_forecast"] == False].copy()
        fut = d[d["is_forecast"] == True].copy()
        ax.plot(pd.to_datetime(hist["ds"]), pd.to_numeric(hist["yhat"], errors="coerce"), color="#4c78a8", lw=1.2, label="Gecmis")
        ax.plot(pd.to_datetime(fut["ds"]), pd.to_numeric(fut["yhat"], errors="coerce"), color="#f58518", lw=1.4, label="Tahmin")
        lo = pd.to_numeric(fut["yhat_lower"], errors="coerce")
        hi = pd.to_numeric(fut["yhat_upper"], errors="coerce")
        if lo.notna().any() and hi.notna().any():
            ax.fill_between(pd.to_datetime(fut["ds"]), lo, hi, color="#f58518", alpha=0.18, label="Guven bandi")
        ax.set_title(
            f"{var} | {row['model_name']} | 2035={safe_float(row['forecast_end_yhat_2035']):.3f} | "
            f"score={safe_float(row['score_total']):.2f} | conf={safe_float(row['confidence_score']):.1f} ({row['confidence_grade']})"
        )
        ax.grid(alpha=0.25)
        if str(row.get("note", "")).strip():
            ax.text(0.01, 0.04, f"not: {row['note']}", transform=ax.transAxes, fontsize=8, color="#a33")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Final v4: Otomatik Model Secimi (Arbitration)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    dash_png = out_dir / "v4_final_arbitrated_dashboard.png"
    fig.savefig(dash_png, dpi=160)
    plt.close(fig)

    # Confidence panel.
    cfig, cax = plt.subplots(figsize=(10, 4.8))
    w2 = winners.sort_values("variable").copy()
    cax.bar(w2["variable"], w2["confidence_score"], color="#4c78a8", alpha=0.85)
    for i, (_, r) in enumerate(w2.iterrows()):
        cax.text(i, safe_float(r["confidence_score"]) + 1.2, f"{safe_float(r['confidence_score']):.1f}\n{r['confidence_grade']}",
                 ha="center", va="bottom", fontsize=9)
    cax.set_ylim(0, 105)
    cax.set_ylabel("Confidence Score")
    cax.set_title("v4 Model Secimi: Guven Skorlari")
    cax.grid(axis="y", alpha=0.25)
    conf_png = out_dir / "v4_confidence_panel.png"
    cfig.tight_layout()
    cfig.savefig(conf_png, dpi=160)
    plt.close(cfig)

    md_lines = []
    md_lines.append("# Final v4 Arbitration Ozeti")
    md_lines.append("")
    md_lines.append(f"- Candidate table: `{cand_csv}`")
    md_lines.append(f"- Final winners: `{winners_csv}`")
    md_lines.append(f"- Dashboard: `{dash_png}`")
    md_lines.append(f"- Confidence panel: `{conf_png}`")
    md_lines.append("")
    for _, r in winners.sort_values("variable").iterrows():
        md_lines.append(
            f"- {r['variable']}: {r['model_name']} | 2035={safe_float(r['forecast_end_yhat_2035']):.3f} | "
            f"score={safe_float(r['score_total']):.3f} | metric={r['metric_name']}={safe_float(r['metric_value']):.3f} | "
            f"confidence={safe_float(r['confidence_score']):.1f} ({r['confidence_grade']})"
        )
    md_path = out_dir / "v4_final_arbitrated_yorum.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary = {
        "run_dir": str(args.run_dir),
        "candidate_count": int(len(scored)),
        "winner_count": int(len(winners)),
        "outputs": {
            "candidate_scores_csv": str(cand_csv),
            "final_winners_csv": str(winners_csv),
            "dashboard_png": str(dash_png),
            "confidence_panel_png": str(conf_png),
            "comment_md": str(md_path),
        },
    }
    (out_dir / "v4_final_arbitrated_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
