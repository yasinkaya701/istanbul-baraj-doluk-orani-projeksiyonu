#!/usr/bin/env python3
"""Build stability-calibrated health model summary.

This script improves forecast stability by blending point estimates with
robust sensitivity-center statistics (median/quantiles) and OOD-aware shrinkage.
Outputs are written under output/health_impact by default.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stability-calibrated health model summary.")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--comparison-csv", type=Path, default=None)
    p.add_argument(
        "--quant-sensitivity-csv",
        type=Path,
        default=Path("output/health_impact/quant_duzenlenmis_run/sensitivity_genis/sensitivity_summary.csv"),
    )
    p.add_argument(
        "--strong-sensitivity-csv",
        type=Path,
        default=Path("output/health_impact/strong_duzenlenmis_run/sensitivity_genis/sensitivity_summary.csv"),
    )
    p.add_argument("--date-label", type=str, default=str(date.today()))
    return p.parse_args()


def _find_comparison_csv(root: Path, override: Path | None) -> Path:
    if override is not None and override.exists():
        return override
    candidates = [
        root / "model_comparison_summary_duzenlenmis_run.csv",
        root / "model_comparison_summary.csv",
        root / "model_comparison_summary_stable_calibrated.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("Missing model comparison summary CSV.")


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}")
    return pd.read_csv(path)


def _model_key(name: str) -> str:
    n = str(name).lower()
    if "strong" in n:
        return "strong"
    if "quant" in n:
        return "quant"
    return n


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _as_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or weights.size == 0:
        return float("nan")
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if total <= 0:
        return float("nan")
    target = float(np.clip(q, 0.0, 1.0)) * total
    return float(np.interp(target, cw, v))


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if values.size == 0 or weights.size == 0:
        return float("nan"), float("nan")
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan"), float("nan")
    mean = float(np.sum(values * weights) / total)
    var = float(np.sum(weights * (values - mean) ** 2) / total)
    return mean, float(np.sqrt(max(0.0, var)))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, x: float) -> float:
    if values.size == 0 or weights.size == 0 or not np.isfinite(x):
        return float("nan")
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    cw = np.cumsum(w)
    total = float(cw[-1])
    if total <= 0:
        return float("nan")
    return float(np.interp(float(x), v, cw / total, left=0.0, right=1.0))


def _finite_weighted_arrays(series: pd.Series, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return v[m], w[m]


def _effective_n(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    s1 = float(np.sum(weights))
    s2 = float(np.sum(weights**2))
    if s1 <= 0 or s2 <= 0:
        return 0.0
    return float((s1**2) / s2)


def _build_similarity_weights(ok: pd.DataFrame, row_cfg: pd.Series | None) -> np.ndarray:
    w = np.ones(len(ok), dtype=float)
    if row_cfg is None or len(ok) == 0:
        return w

    for sc, rc, boost in [
        ("epi_mode", "epi_mode", 1.20),
        ("epi_ci_bound", "epi_ci_bound", 1.85),
        ("adaptation_mode", "adaptation_mode", 1.70),
    ]:
        if sc in ok.columns and rc in row_cfg.index and pd.notna(row_cfg[rc]):
            target = str(row_cfg[rc]).strip().lower()
            m = ok[sc].astype(str).str.strip().str.lower().eq(target).to_numpy(dtype=bool)
            w *= np.where(m, boost, 1.0)

    if "humidity_interaction" in ok.columns and "enable_humidity_interaction" in row_cfg.index:
        target_h = _as_bool(row_cfg["enable_humidity_interaction"])
        if target_h is not None:
            hum = ok["humidity_interaction"].map(_as_bool).to_numpy()
            m = hum == target_h
            w *= np.where(m, 1.55, 1.0)

    if "risk_threshold_c" in ok.columns and "risk_threshold_c" in row_cfg.index and pd.notna(row_cfg["risk_threshold_c"]):
        target_thr = float(row_cfg["risk_threshold_c"])
        rr = pd.to_numeric(ok["risk_threshold_c"], errors="coerce").to_numpy(dtype=float)
        sigma = 1.35
        g = np.exp(-0.5 * ((rr - target_thr) / sigma) ** 2)
        g = np.where(np.isfinite(g), 0.35 + 0.65 * g, 1.0)
        w *= g

    if (
        "risk_beta_effective" in ok.columns
        and "risk_beta_per_c_effective" in row_cfg.index
        and pd.notna(row_cfg["risk_beta_per_c_effective"])
    ):
        target_beta = float(row_cfg["risk_beta_per_c_effective"])
        bb = pd.to_numeric(ok["risk_beta_effective"], errors="coerce").to_numpy(dtype=float)
        sigma_b = max(0.0025, abs(target_beta) * 0.20)
        g = np.exp(-0.5 * ((bb - target_beta) / sigma_b) ** 2)
        g = np.where(np.isfinite(g), 0.50 + 0.50 * g, 1.0)
        w *= g

    if "future_out_of_distribution_share" in ok.columns:
        ood = pd.to_numeric(ok["future_out_of_distribution_share"], errors="coerce").to_numpy(dtype=float)
        ood_penalty = np.exp(-0.80 * np.clip(ood, 0.0, None))
        ood_penalty = np.where(np.isfinite(ood_penalty), ood_penalty, 1.0)
        w *= ood_penalty

    return np.clip(w, 1e-6, None)


def _sensitivity_stats(df: pd.DataFrame, row_cfg: pd.Series | None = None, orig_rr: float = np.nan) -> dict[str, float]:
    ok = df.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"].astype(str).str.lower() == "ok"].copy()
    if ok.empty:
        return {
            "n": 0.0,
            "rr_median": np.nan,
            "rr_q10": np.nan,
            "rr_q90": np.nan,
            "rr_iqr": np.nan,
            "rr_cv": np.nan,
            "af_median": np.nan,
            "thr_median": np.nan,
            "wet_median": np.nan,
            "ood_mean": np.nan,
            "pos_delta_share": np.nan,
            "effective_n": 0.0,
            "orig_rr_percentile": np.nan,
        }

    w = _build_similarity_weights(ok, row_cfg=row_cfg)

    rr_v, rr_w = _finite_weighted_arrays(ok["future_rr_mean"], w)
    af_v, af_w = _finite_weighted_arrays(ok["future_af_mean"], w)
    thr_v, thr_w = _finite_weighted_arrays(ok["future_threshold_exceed_share"], w)
    wet_v, wet_w = _finite_weighted_arrays(ok["future_wet_hot_share"], w)
    ood_v, ood_w = _finite_weighted_arrays(ok["future_out_of_distribution_share"], w)
    drr_v, drr_w = _finite_weighted_arrays(ok["delta_rr_mean"], w)

    rr_mean, rr_std = _weighted_mean_std(rr_v, rr_w)
    rr_cv = rr_std / abs(rr_mean) if np.isfinite(rr_mean) and abs(rr_mean) > 1e-9 else 0.0
    rr_q25 = _weighted_quantile(rr_v, rr_w, 0.25)
    rr_q75 = _weighted_quantile(rr_v, rr_w, 0.75)
    rr_q10 = _weighted_quantile(rr_v, rr_w, 0.10)
    rr_q90 = _weighted_quantile(rr_v, rr_w, 0.90)
    rr_median = _weighted_quantile(rr_v, rr_w, 0.50)
    af_median = _weighted_quantile(af_v, af_w, 0.50)
    thr_median = _weighted_quantile(thr_v, thr_w, 0.50)
    wet_median = _weighted_quantile(wet_v, wet_w, 0.50)
    ood_mean, _ = _weighted_mean_std(ood_v, ood_w)
    pos_delta_share = float(np.sum(drr_w[drr_v > 0.0]) / np.sum(drr_w)) if drr_v.size else np.nan
    rr_percentile = _weighted_percentile(rr_v, rr_w, float(orig_rr)) if np.isfinite(orig_rr) else np.nan

    return {
        "n": float(len(ok)),
        "effective_n": _effective_n(rr_w),
        "rr_median": rr_median,
        "rr_q10": rr_q10,
        "rr_q90": rr_q90,
        "rr_iqr": (rr_q75 - rr_q25) if np.isfinite(rr_q75) and np.isfinite(rr_q25) else np.nan,
        "rr_cv": float(rr_cv),
        "af_median": af_median,
        "thr_median": thr_median,
        "wet_median": wet_median,
        "ood_mean": ood_mean,
        "pos_delta_share": pos_delta_share,
        "orig_rr_percentile": rr_percentile,
    }


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _calibration_strength(stats: dict[str, float], orig_rr: float) -> tuple[float, float, float]:
    ood_mean = float(stats.get("ood_mean", np.nan))
    rr_cv = float(stats.get("rr_cv", np.nan))
    rr_iqr = float(stats.get("rr_iqr", np.nan))
    pos_share = float(stats.get("pos_delta_share", np.nan))
    eff_n = float(stats.get("effective_n", np.nan))
    q10 = float(stats.get("rr_q10", np.nan))
    q90 = float(stats.get("rr_q90", np.nan))
    med = float(stats.get("rr_median", np.nan))

    ood_term = _clip01((ood_mean if np.isfinite(ood_mean) else 0.0) / 0.60)
    cv_term = _clip01((rr_cv if np.isfinite(rr_cv) else 0.0) / 0.25)
    disp_term = _clip01((rr_iqr if np.isfinite(rr_iqr) else 0.0) / 0.10)
    consistency_term = _clip01(1.0 - (pos_share if np.isfinite(pos_share) else 0.5))
    sample_term = _clip01((12.0 - (eff_n if np.isfinite(eff_n) else 0.0)) / 12.0)

    outside_term = 0.0
    plausibility = 0.5
    if np.isfinite(orig_rr) and np.isfinite(q10) and np.isfinite(q90) and q90 > q10:
        width = max(1e-6, q90 - q10)
        if orig_rr < q10:
            outside_term = _clip01((q10 - orig_rr) / width)
        elif orig_rr > q90:
            outside_term = _clip01((orig_rr - q90) / width)
        z = abs(orig_rr - med) / max(width * 0.50, 1e-6) if np.isfinite(med) else 0.0
        plausibility = float(np.exp(-0.5 * (z**2)))

    base_shrink = float(
        np.clip(0.06 + 0.38 * ood_term + 0.22 * cv_term + 0.16 * disp_term + 0.08 * consistency_term + 0.10 * sample_term, 0.04, 0.78)
    )
    shrink = base_shrink * (0.72 + 0.55 * outside_term) * (0.92 + 0.25 * (1.0 - plausibility))
    if outside_term < 1e-9:
        shrink *= 0.88 + 0.22 * (1.0 - plausibility)
    shrink = float(np.clip(shrink, 0.05, 0.82))

    stability = float(np.clip(1.0 - (0.43 * ood_term + 0.29 * cv_term + 0.18 * disp_term + 0.10 * outside_term), 0.0, 1.0))
    accuracy_proxy = float(
        np.clip(1.0 - (0.40 * disp_term + 0.28 * ood_term + 0.17 * outside_term + 0.10 * (1.0 - plausibility) + 0.05 * sample_term), 0.0, 1.0)
    )
    return shrink, stability, accuracy_proxy


def _confidence_label(stability: float, accuracy_proxy: float) -> str:
    c = min(float(stability), float(accuracy_proxy))
    if c >= 0.75:
        return "yuksek"
    if c >= 0.55:
        return "orta"
    return "dusuk"


def build_calibrated_summary(
    cmp_df: pd.DataFrame,
    quant_sens_df: pd.DataFrame,
    strong_sens_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sens_df_map = {
        "quant": quant_sens_df,
        "strong": strong_sens_df,
    }

    out = cmp_df.copy()
    if "model" not in out.columns:
        raise SystemExit("model column missing in comparison summary.")
    out["model_key"] = out["model"].map(_model_key)

    for col in [
        "future_rr_mean",
        "delta_rr_mean",
        "baseline_rr_mean",
        "future_af_mean",
        "delta_af_mean",
        "future_threshold_exceed_share",
        "future_wet_hot_share",
        "delta_ood_share",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    rows: list[dict[str, float | str]] = []
    for i, row in out.iterrows():
        key = str(row["model_key"])
        orig_future_rr = float(row.get("future_rr_mean", np.nan))
        sens_df = sens_df_map.get(key, pd.DataFrame())
        stats = _sensitivity_stats(sens_df, row_cfg=row, orig_rr=orig_future_rr)
        shrink, stability, accuracy_proxy = _calibration_strength(stats if stats else {}, orig_rr=orig_future_rr)

        base_rr = float(row.get("baseline_rr_mean", np.nan))
        if not np.isfinite(base_rr):
            base_rr = 1.0
        target_rr = float(stats.get("rr_median", np.nan))
        if not np.isfinite(target_rr):
            target_rr = orig_future_rr if np.isfinite(orig_future_rr) else base_rr

        q10 = float(stats.get("rr_q10", np.nan))
        q90 = float(stats.get("rr_q90", np.nan))
        if np.isfinite(orig_future_rr) and np.isfinite(q10) and np.isfinite(q90) and q90 > q10:
            anchored = float(np.clip(orig_future_rr, q10, q90))
            target_rr = 0.70 * target_rr + 0.30 * anchored
        cal_rr = (orig_future_rr * (1.0 - shrink) + target_rr * shrink) if np.isfinite(orig_future_rr) else target_rr
        if np.isfinite(q10) and np.isfinite(q90) and q90 > q10:
            cal_rr = float(np.clip(cal_rr, q10, q90))
        cal_rr = max(float(base_rr) + 0.0002, float(cal_rr))
        cal_delta_rr = cal_rr - float(base_rr)

        orig_future_af = float(row.get("future_af_mean", np.nan))
        target_af = float(stats.get("af_median", np.nan))
        if not np.isfinite(target_af):
            target_af = orig_future_af if np.isfinite(orig_future_af) else 0.0
        cal_af = (orig_future_af * (1.0 - shrink) + target_af * shrink) if np.isfinite(orig_future_af) else target_af
        cal_af = float(np.clip(cal_af, 0.0, 1.0))

        cal_thr = row.get("future_threshold_exceed_share", np.nan)
        thr_med = float(stats.get("thr_median", np.nan))
        if np.isfinite(cal_thr) and np.isfinite(thr_med):
            cal_thr = float(np.clip(cal_thr * (1.0 - shrink) + thr_med * shrink, 0.0, 1.0))
        elif np.isfinite(thr_med):
            cal_thr = float(np.clip(thr_med, 0.0, 1.0))

        cal_wet = row.get("future_wet_hot_share", np.nan)
        wet_med = float(stats.get("wet_median", np.nan))
        if np.isfinite(cal_wet) and np.isfinite(wet_med):
            cal_wet = float(np.clip(cal_wet * (1.0 - shrink) + wet_med * shrink, 0.0, 1.0))
        elif np.isfinite(wet_med):
            cal_wet = float(np.clip(wet_med, 0.0, 1.0))

        cal_ood = row.get("delta_ood_share", np.nan)
        ood_mean = float(stats.get("ood_mean", np.nan))
        if np.isfinite(cal_ood) and np.isfinite(ood_mean):
            cal_ood = float(np.clip(cal_ood * (1.0 - shrink) + ood_mean * shrink, 0.0, 1.0))
        elif np.isfinite(ood_mean):
            cal_ood = float(np.clip(ood_mean, 0.0, 1.0))

        out.loc[i, "orig_future_rr_mean"] = orig_future_rr
        out.loc[i, "orig_delta_rr_mean"] = float(row.get("delta_rr_mean", np.nan))
        out.loc[i, "orig_future_af_mean"] = orig_future_af
        out.loc[i, "future_rr_mean"] = cal_rr
        out.loc[i, "delta_rr_mean"] = cal_delta_rr
        out.loc[i, "future_af_mean"] = cal_af
        out.loc[i, "delta_af_mean"] = cal_af - float(row.get("baseline_af_mean", 0.0) if np.isfinite(row.get("baseline_af_mean", np.nan)) else 0.0)
        out.loc[i, "future_threshold_exceed_share"] = cal_thr
        out.loc[i, "future_wet_hot_share"] = cal_wet
        out.loc[i, "delta_ood_share"] = cal_ood
        out.loc[i, "stability_calibration_shrink"] = shrink
        out.loc[i, "stability_score"] = stability
        out.loc[i, "accuracy_proxy_score"] = accuracy_proxy
        out.loc[i, "prediction_confidence"] = _confidence_label(stability, accuracy_proxy)
        out.loc[i, "sens_rr_median"] = float(stats.get("rr_median", np.nan))
        out.loc[i, "sens_rr_iqr"] = float(stats.get("rr_iqr", np.nan))
        out.loc[i, "sens_ood_mean"] = float(stats.get("ood_mean", np.nan))
        out.loc[i, "sens_effective_n"] = float(stats.get("effective_n", np.nan))
        out.loc[i, "sens_orig_rr_percentile"] = float(stats.get("orig_rr_percentile", np.nan))

        rows.append(
            {
                "model": str(row["model"]),
                "model_key": key,
                "n_sensitivity": float(stats.get("n", 0.0)),
                "effective_n": float(stats.get("effective_n", 0.0)),
                "orig_future_rr_mean": orig_future_rr,
                "calibrated_future_rr_mean": cal_rr,
                "orig_delta_rr_mean": float(row.get("delta_rr_mean", np.nan)),
                "calibrated_delta_rr_mean": cal_delta_rr,
                "sens_rr_median": float(stats.get("rr_median", np.nan)),
                "sens_rr_q10": float(stats.get("rr_q10", np.nan)),
                "sens_rr_q90": float(stats.get("rr_q90", np.nan)),
                "sens_rr_iqr": float(stats.get("rr_iqr", np.nan)),
                "sens_rr_cv": float(stats.get("rr_cv", np.nan)),
                "sens_ood_mean": float(stats.get("ood_mean", np.nan)),
                "sens_pos_delta_share": float(stats.get("pos_delta_share", np.nan)),
                "sens_orig_rr_percentile": float(stats.get("orig_rr_percentile", np.nan)),
                "stability_calibration_shrink": shrink,
                "stability_score": stability,
                "accuracy_proxy_score": accuracy_proxy,
                "prediction_confidence": _confidence_label(stability, accuracy_proxy),
            }
        )

    out["calibration_version"] = "stable_calibrated_v2"
    out["model_key"] = out["model_key"].astype(str)
    diag = pd.DataFrame(rows).sort_values("model_key")
    return out, diag


def build_notes(diag: pd.DataFrame, out_path: Path, date_label: str) -> None:
    lines = [
        f"# Stabilite ve Kalibrasyon Notu ({date_label})",
        "",
        "Bu dosya model tahminlerini, senaryo-benzerlik agirlikli dagilim ve OOD baskisina gore stabilize eder.",
        "- Amac: asiri oynakligi azaltmak ve daha dengeli bir tahmin merkezi kullanmak.",
        "- Yontem: point tahmini + agirlikli robust merkez (duyarlilik medyani) + uyum kontrollu shrinkage (OOD/cv/iqr).",
        "",
        "## Model Bazli Sonuc",
    ]
    for _, r in diag.iterrows():
        lines.append(
            f"- {r['model_key']}: RR {float(r['orig_future_rr_mean']):.4f} -> {float(r['calibrated_future_rr_mean']):.4f}, "
            f"shrink={float(r['stability_calibration_shrink']):.2f}, stability={float(r['stability_score']):.2f}, "
            f"confidence={r['prediction_confidence']}"
        )
    lines.extend(
        [
            "",
            "## Not",
            "- Bu kalibrasyon klinik nedensellik iddiasi degil; tahmin stabilizasyon katmanidir.",
            "- Yeni gozlem geldikce yeniden calistirilmalidir.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_dashboard(diag: pd.DataFrame, out_path: Path) -> None:
    if diag.empty:
        raise SystemExit("No diagnostic rows to plot.")

    d = diag.copy()
    d["orig_future_rr_mean"] = pd.to_numeric(d["orig_future_rr_mean"], errors="coerce")
    d["calibrated_future_rr_mean"] = pd.to_numeric(d["calibrated_future_rr_mean"], errors="coerce")
    d["sens_rr_q10"] = pd.to_numeric(d["sens_rr_q10"], errors="coerce")
    d["sens_rr_q90"] = pd.to_numeric(d["sens_rr_q90"], errors="coerce")
    d["stability_score"] = pd.to_numeric(d["stability_score"], errors="coerce")
    d["accuracy_proxy_score"] = pd.to_numeric(d["accuracy_proxy_score"], errors="coerce")
    d["stability_calibration_shrink"] = pd.to_numeric(d["stability_calibration_shrink"], errors="coerce")
    d["sens_ood_mean"] = pd.to_numeric(d["sens_ood_mean"], errors="coerce")
    d["sens_rr_cv"] = pd.to_numeric(d["sens_rr_cv"], errors="coerce")
    d["effective_n"] = pd.to_numeric(d.get("effective_n", np.nan), errors="coerce")
    d["sens_orig_rr_percentile"] = pd.to_numeric(d.get("sens_orig_rr_percentile", np.nan), errors="coerce")

    x = np.arange(len(d))
    labels = d["model_key"].astype(str).tolist()

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.8))

    # 1) Original vs calibrated RR
    ax1 = axes[0, 0]
    w = 0.35
    ax1.bar(x - w / 2.0, d["orig_future_rr_mean"].to_numpy(dtype=float), width=w, color="#d95f02", label="orijinal_rr")
    ax1.bar(x + w / 2.0, d["calibrated_future_rr_mean"].to_numpy(dtype=float), width=w, color="#1b9e77", label="kalibre_rr")
    ax1.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Future RR mean")
    ax1.set_title("Tahmin Stabilizasyonu")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8, frameon=True)

    # 2) Sensitivity interval + points
    ax2 = axes[0, 1]
    q10 = d["sens_rr_q10"].to_numpy(dtype=float)
    q90 = d["sens_rr_q90"].to_numpy(dtype=float)
    med = d["sens_rr_median"].to_numpy(dtype=float)
    ax2.vlines(x, q10, q90, color="#888888", linewidth=4, alpha=0.7, label="sens_q10_q90")
    ax2.scatter(x, med, color="#2c3e50", s=65, label="sens_median", zorder=3)
    ax2.scatter(x, d["orig_future_rr_mean"].to_numpy(dtype=float), color="#d95f02", marker="x", s=70, label="orijinal_rr", zorder=3)
    ax2.scatter(x, d["calibrated_future_rr_mean"].to_numpy(dtype=float), color="#1b9e77", marker="o", s=60, label="kalibre_rr", zorder=3)
    ax2.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("RR")
    ax2.set_title("Duyarlilik Araligi Uyum Kontrolu")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8, frameon=True)

    # 3) Stability metrics
    ax3 = axes[0, 2]
    ax3.bar(x - w, d["stability_score"].to_numpy(dtype=float), width=w, color="#1f78b4", label="stability")
    ax3.bar(x, d["accuracy_proxy_score"].to_numpy(dtype=float), width=w, color="#33a02c", label="accuracy_proxy")
    ax3.bar(x + w, d["stability_calibration_shrink"].to_numpy(dtype=float), width=w, color="#ff7f00", label="shrink")
    ax3.set_xticks(x, labels)
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("0-1 skor")
    ax3.set_title("Stabilite ve Kalibrasyon Skorlari")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(loc="upper right", fontsize=8, frameon=True)

    # 4) OOD vs CV
    ax4 = axes[1, 0]
    ax4.scatter(
        d["sens_ood_mean"].to_numpy(dtype=float),
        d["sens_rr_cv"].to_numpy(dtype=float),
        s=180,
        c=d["stability_score"].to_numpy(dtype=float),
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5,
    )
    for _, r in d.iterrows():
        ax4.text(float(r["sens_ood_mean"]) + 0.005, float(r["sens_rr_cv"]) + 0.002, str(r["model_key"]), fontsize=8)
    ax4.set_xlabel("OOD ortalamasi")
    ax4.set_ylabel("RR varyasyon katsayisi")
    ax4.set_title("Oynaklik-Kaynak Analizi")
    ax4.grid(alpha=0.25)

    # 5) Deviation from robust center before/after
    ax5 = axes[1, 1]
    dev_orig = np.abs(d["orig_future_rr_mean"].to_numpy(dtype=float) - d["sens_rr_median"].to_numpy(dtype=float))
    dev_cal = np.abs(d["calibrated_future_rr_mean"].to_numpy(dtype=float) - d["sens_rr_median"].to_numpy(dtype=float))
    ax5.bar(x - w / 2.0, dev_orig, width=w, color="#d95f02", label="|orig-medyan|")
    ax5.bar(x + w / 2.0, dev_cal, width=w, color="#1b9e77", label="|kalibre-medyan|")
    ax5.set_xticks(x, labels)
    ax5.set_ylabel("Mutlak sapma")
    ax5.set_title("Kalibrasyon Sonrasi Sapma Azalimi")
    ax5.grid(axis="y", alpha=0.25)
    ax5.legend(loc="upper right", fontsize=8, frameon=True)

    # 6) Effective sample + original percentile support
    ax6 = axes[1, 2]
    eff_n = d["effective_n"].to_numpy(dtype=float)
    eff_scale = max(1.0, float(np.nanmax(eff_n)))
    eff_norm = np.clip(eff_n / eff_scale, 0.0, 1.0)
    rr_pct = np.clip(d["sens_orig_rr_percentile"].to_numpy(dtype=float), 0.0, 1.0)
    ax6.bar(x - w / 2.0, eff_norm, width=w, color="#6a3d9a", label="normalize_eff_n")
    ax6.bar(x + w / 2.0, rr_pct, width=w, color="#b15928", label="orig_percentile")
    ax6.set_xticks(x, labels)
    ax6.set_ylim(0.0, 1.05)
    ax6.set_ylabel("0-1 oran")
    ax6.set_title("Destek Gucu ve Konum")
    ax6.grid(axis="y", alpha=0.25)
    ax6.legend(loc="upper right", fontsize=8, frameon=True)
    for xi, raw in zip(x, eff_n):
        if np.isfinite(raw):
            ax6.text(xi - w / 2.0, 1.02, f"n~{raw:.0f}", ha="center", va="bottom", fontsize=7, color="#4b2c6f")

    fig.suptitle("Model Stabilite ve Kalibrasyon Panosu", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    comparison_csv = _find_comparison_csv(root, args.comparison_csv.resolve() if args.comparison_csv is not None else None)
    cmp_df = _safe_read(comparison_csv)
    q_df = _safe_read(args.quant_sensitivity_csv.resolve())
    s_df = _safe_read(args.strong_sensitivity_csv.resolve())

    calibrated, diag = build_calibrated_summary(cmp_df=cmp_df, quant_sens_df=q_df, strong_sens_df=s_df)

    dated_summary = root / f"model_comparison_summary_stable_calibrated_{args.date_label}.csv"
    base_summary = root / "model_comparison_summary_stable_calibrated.csv"
    diag_csv = root / f"model_stability_diagnostics_{args.date_label}.csv"
    diag_md = root / f"model_stability_notes_{args.date_label}.md"
    diag_png = root / f"model_stability_dashboard_{args.date_label}.png"

    calibrated.to_csv(dated_summary, index=False)
    calibrated.to_csv(base_summary, index=False)
    diag.to_csv(diag_csv, index=False)
    build_notes(diag=diag, out_path=diag_md, date_label=args.date_label)
    build_dashboard(diag=diag, out_path=diag_png)

    print(f"Wrote: {dated_summary}")
    print(f"Wrote: {base_summary}")
    print(f"Wrote: {diag_csv}")
    print(f"Wrote: {diag_md}")
    print(f"Wrote: {diag_png}")


if __name__ == "__main__":
    main()
