#!/usr/bin/env python3
"""Build a human-readable report + visuals for Istanbul dam scenario outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Istanbul dam public report and visuals")
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
    p.add_argument("--window-start", default="2026-03-01")
    p.add_argument("--window-end", default="2027-02-01")
    p.add_argument("--threshold-high-risk-months", type=int, default=4)
    p.add_argument("--threshold-medium-risk-months", type=int, default=2)
    p.add_argument("--threshold-high-prob40", type=float, default=60.0)
    p.add_argument("--threshold-medium-prob40", type=float, default=35.0)
    p.add_argument("--top-n", type=int, default=6)
    return p.parse_args()


def risk_level(row: pd.Series, args: argparse.Namespace) -> str:
    if (
        int(row["months_lt40"]) >= args.threshold_high_risk_months
        or float(row["mean_prob_below_40_pct"]) >= args.threshold_high_prob40
    ):
        return "high"
    if (
        int(row["months_lt40"]) >= args.threshold_medium_risk_months
        or float(row["mean_prob_below_40_pct"]) >= args.threshold_medium_prob40
    ):
        return "medium"
    return "low"


def scenario_order(values: list[str]) -> list[str]:
    def _key(s: str) -> tuple[int, int, str]:
        if s == "baseline":
            return (0, 0, s)
        if s.startswith("dry_"):
            label = s.split("_", 1)[1]
            ord_map = {"mild": 1, "base": 2, "stress": 2, "severe": 3, "extreme": 4}
            return (1, ord_map.get(label, 9), s)
        if s.startswith("wet_"):
            label = s.split("_", 1)[1]
            ord_map = {"mild": 1, "base": 2, "relief": 2, "severe": 3, "extreme": 4}
            return (2, ord_map.get(label, 9), s)
        return (3, 99, s)

    uniq = sorted({str(v) for v in values}, key=_key)
    return uniq


def plot_prob_heatmaps(
    scen_fc: pd.DataFrame,
    risk_df: pd.DataFrame,
    out_png: Path,
    window_start: str,
    window_end: str,
) -> None:
    tmp = scen_fc[(scen_fc["ds"] >= window_start) & (scen_fc["ds"] <= window_end)].copy()
    if tmp.empty:
        return

    all_scenarios = scenario_order(tmp["scenario"].dropna().astype(str).unique().tolist())
    months = sorted(pd.to_datetime(tmp["ds"]).dt.to_period("M").dt.to_timestamp().unique())
    month_labels = [pd.Timestamp(m).strftime("%Y-%m") for m in months]

    baseline = risk_df[risk_df["scenario"] == "baseline"].sort_values(
        ["mean_prob_below_40_pct", "months_lt40"],
        ascending=[False, False],
    )
    series_order = baseline["series"].tolist()
    if not series_order:
        series_order = sorted(tmp["series"].dropna().astype(str).unique().tolist())

    fig, axes = plt.subplots(1, len(all_scenarios), figsize=(5.8 * len(all_scenarios), 8.2), constrained_layout=True)
    if len(all_scenarios) == 1:
        axes = [axes]

    for ax, scen in zip(axes, all_scenarios, strict=False):
        g = tmp[tmp["scenario"] == scen].copy()
        mat = np.full((len(series_order), len(months)), np.nan, dtype=float)
        for i, series_name in enumerate(series_order):
            gg = g[g["series"] == series_name].copy()
            if gg.empty:
                continue
            mp = {
                pd.Timestamp(d).to_period("M").to_timestamp(): float(p)
                for d, p in zip(gg["ds"], gg["scenario_prob_below_40"], strict=False)
            }
            for j, m in enumerate(months):
                if m in mp:
                    mat[i, j] = mp[m]

        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="RdYlGn_r")
        ax.set_title(f"{scen} | P(y < 40%)")
        ax.set_yticks(np.arange(len(series_order)))
        ax.set_yticklabels(series_order, fontsize=9)
        ax.set_xticks(np.arange(len(months)))
        ax.set_xticklabels(month_labels, rotation=60, ha="right", fontsize=8)

    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("Probability", rotation=270, labelpad=15)
    fig.suptitle("Istanbul Dams - Monthly Risk Heatmaps (2026-03 to 2027-02)", fontsize=14)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_top_risk_bars(risk_df: pd.DataFrame, out_png: Path, top_n: int) -> None:
    if risk_df.empty:
        return

    scenarios = scenario_order(risk_df["scenario"].dropna().astype(str).unique().tolist())
    baseline = risk_df[risk_df["scenario"] == "baseline"].sort_values(
        ["mean_prob_below_40_pct", "months_lt40"],
        ascending=[False, False],
    )
    if baseline.empty:
        baseline = risk_df.sort_values(["mean_prob_below_40_pct", "months_lt40"], ascending=[False, False])
    selected = baseline.head(max(1, int(top_n)))["series"].tolist()

    x = np.arange(len(selected))
    width = 0.8 / max(len(scenarios), 1)

    fig, ax = plt.subplots(figsize=(max(9, len(selected) * 1.2), 5.5))
    for i, scen in enumerate(scenarios):
        vals = []
        g = risk_df[risk_df["scenario"] == scen].set_index("series")
        for s in selected:
            vals.append(float(g.loc[s, "mean_prob_below_40_pct"]) if s in g.index else np.nan)
        offset = (i - (len(scenarios) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=scen, alpha=0.9)

    ax.axhline(60.0, color="#b91c1c", linestyle="--", linewidth=1.1, label="High-risk ref (60%)")
    ax.axhline(35.0, color="#b45309", linestyle=":", linewidth=1.1, label="Medium-risk ref (35%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean P(y < 40%)")
    ax.set_xticks(x)
    ax.set_xticklabels(selected, rotation=25, ha="right")
    ax.set_title("Top Risky Dams by Scenario (Window Mean Probability)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def top_rows_md(risk_df: pd.DataFrame, top_n: int) -> str:
    lines = []
    for scen in scenario_order(risk_df["scenario"].dropna().astype(str).unique().tolist()):
        g = risk_df[risk_df["scenario"] == scen].copy()
        g = g.sort_values(["mean_prob_below_40_pct", "months_lt40", "worst_yhat_pct"], ascending=[False, False, True]).head(
            max(1, int(top_n))
        )
        lines.append(f"### {scen}")
        if g.empty:
            lines.append("- No data.")
            lines.append("")
            continue
        for _, r in g.iterrows():
            lines.append(
                "- "
                f"{r['series']}: ort. P(<40)={float(r['mean_prob_below_40_pct']):.1f}%, "
                f"<40 ay={int(r['months_lt40'])}, en kotu ay={r['worst_month']}, "
                f"en kotu tahmin={float(r['worst_yhat_pct']):.1f}%."
            )
        lines.append("")
    return "\n".join(lines)


def literature_consistency_section(
    run_summary: dict[str, Any],
    scenario_summary: dict[str, Any],
    cv_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> str:
    cv_splits = int(cv_df["split"].nunique()) if "split" in cv_df.columns else 0
    model_count = int(metrics_df["model"].nunique()) if "model" in metrics_df.columns else 0
    has_ensemble = bool((metrics_df["model"] == "ensemble_topk").any()) if "model" in metrics_df.columns else False
    has_interval = all(c in forecast_df.columns for c in ["yhat_lower", "yhat_upper"])
    uses_bootstrap = str(scenario_summary.get("method", "")).strip() == "empirical_cv_residual_bootstrap"

    rows = [
        (
            "Rolling-origin CV",
            "TSCV and rolling forecast origin are recommended for time-series model selection.",
            "Used (splits={})".format(cv_splits),
            "Uyumlu" if cv_splits >= 3 else "Zayif",
            "https://otexts.com/fpp3/tscv.html",
        ),
        (
            "Forecast combination",
            "Combining forecasts often improves robustness and accuracy.",
            "Used (models={}, ensemble={})".format(model_count, "yes" if has_ensemble else "no"),
            "Uyumlu" if (model_count >= 3 and has_ensemble) else "Kismen",
            "https://otexts.com/fpp3/combinations.html",
        ),
        (
            "Uncertainty intervals",
            "Point forecasts should be accompanied by prediction intervals.",
            "Used ({})".format("yes" if has_interval else "no"),
            "Uyumlu" if has_interval else "Zayif",
            "https://otexts.com/fpp3/prediction-intervals.html",
        ),
        (
            "Residual bootstrap/scenario",
            "Residual-based bootstrap approaches are valid for non-parametric uncertainty workflows.",
            "Used ({})".format("yes" if uses_bootstrap else "no"),
            "Uyumlu" if uses_bootstrap else "Kismen",
            "https://otexts.com/fpp3/bootstrap.html",
        ),
        (
            "Threshold policy (30/40)",
            "Reservoir warning thresholds are generally basin- and demand-specific, often dynamic by season.",
            "Fixed reference thresholds (30/40) used for screening",
            "Kismen (policy tuning needed)",
            "https://www.mdpi.com/2077-0472/15/13/1408",
        ),
    ]

    lines = []
    lines.append("## Literatur Uyum Kontrolu")
    lines.append("")
    lines.append("| Ilke | Literaturluk Ozet | Modelde Durum | Sonuc | Kaynak |")
    lines.append("|---|---|---|---|---|")
    for ilke, ozet, durum, sonuc, link in rows:
        lines.append(f"| {ilke} | {ozet} | {durum} | {sonuc} | [link]({link}) |")
    lines.append("")
    lines.append(
        "Degerlendirme: Cekirdek modelleme yaklasimi literaturle uyumlu; "
        "esas acik nokta, 30/40 esiklerinin ISKI operasyon kurallari ve talep-sunum dengesine gore "
        "yerel/dinamik olarak tekrar kalibre edilmesi gerekliligidir."
    )
    lines.append("")
    return "\n".join(lines)


def risk_counts_md(risk_df: pd.DataFrame) -> str:
    if risk_df.empty:
        return "No risk rows."
    rows = []
    for scen in scenario_order(risk_df["scenario"].dropna().astype(str).unique().tolist()):
        g = risk_df[risk_df["scenario"] == scen]
        rows.append(
            {
                "scenario": scen,
                "high": int((g["risk_level"] == "high").sum()),
                "medium": int((g["risk_level"] == "medium").sum()),
                "low": int((g["risk_level"] == "low").sum()),
                "total": int(len(g)),
            }
        )
    out = pd.DataFrame(rows)
    return out.to_markdown(index=False)


def dynamic_risk_counts_md(dynamic_df: pd.DataFrame) -> str:
    if dynamic_df is None or dynamic_df.empty:
        return "Dinamik esik ozeti henuz uretilmedi."
    rows = []
    for scen in scenario_order(dynamic_df["scenario"].dropna().astype(str).unique().tolist()):
        g = dynamic_df[dynamic_df["scenario"] == scen]
        rows.append(
            {
                "scenario": scen,
                "high": int((g["risk_level"] == "high").sum()),
                "medium": int((g["risk_level"] == "medium").sum()),
                "low": int((g["risk_level"] == "low").sum()),
                "total": int(len(g)),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def dynamic_top_rows_md(dynamic_df: pd.DataFrame, top_n: int) -> str:
    if dynamic_df is None or dynamic_df.empty:
        return "Dinamik esik ozeti henuz uretilmedi."
    lines = []
    for scen in scenario_order(dynamic_df["scenario"].dropna().astype(str).unique().tolist()):
        g = dynamic_df[dynamic_df["scenario"] == scen].copy()
        g = g.sort_values(
            ["mean_prob_below_warning_pct", "months_below_warning", "worst_gap_to_warning_pct"],
            ascending=[False, False, True],
        ).head(max(1, int(top_n)))
        lines.append(f"### {scen}")
        if g.empty:
            lines.append("- No data.")
            lines.append("")
            continue
        for _, r in g.iterrows():
            lines.append(
                "- "
                f"{r['series']}: ort. P(warn alti)={float(r['mean_prob_below_warning_pct']):.1f}%, "
                f"warn alti ay={int(r['months_below_warning'])}, en kotu gap={float(r['worst_gap_to_warning_pct']):.1f} puan."
            )
        lines.append("")
    return "\n".join(lines)


def scenario_shift_md(shift_df: pd.DataFrame | None, top_n: int) -> str:
    if shift_df is None or shift_df.empty:
        return "Senaryo shift faktorleri henuz uretilmedi."
    lines = []
    candidates = []
    if "scenario_group" in shift_df.columns:
        for grp in ["dry", "wet"]:
            ss = scenario_order(shift_df[shift_df["scenario_group"] == grp]["scenario"].astype(str).unique().tolist())
            candidates.extend(ss)
    else:
        candidates.extend(scenario_order(shift_df["scenario"].astype(str).unique().tolist()))
    for scen in candidates:
        if scen == "baseline":
            continue
        g = shift_df[shift_df["scenario"] == scen].copy()
        if g.empty:
            continue
        agg = g.groupby("series", as_index=False).agg(
            mean_k=("effective_shift_k", "mean"),
            std_k=("effective_shift_k", "std"),
            max_abs_k=("effective_shift_k", lambda x: float(np.max(np.abs(x)))),
        )
        agg = agg.sort_values("max_abs_k", ascending=False).head(max(1, int(top_n)))
        lines.append(f"### {scen}")
        for _, r in agg.iterrows():
            lines.append(
                "- "
                f"{r['series']}: ort.k={float(r['mean_k']):+.2f}, "
                f"max|k|={float(r['max_abs_k']):.2f}, "
                f"aylik std={float(r['std_k'] if pd.notna(r['std_k']) else 0.0):.2f}"
            )
        lines.append("")
    return "\n".join(lines) if lines else "Senaryo shift faktorleri henuz uretilmedi."


def expected_risk_md(
    weights_df: pd.DataFrame | None,
    expected_df: pd.DataFrame | None,
    top_n: int,
) -> str:
    if weights_df is None or weights_df.empty or expected_df is None or expected_df.empty:
        return "Beklenen risk ozeti henuz uretilmedi."

    lines = []
    lines.append("### Senaryo Olasilik Agirliklari")
    w = weights_df.copy().sort_values("weight", ascending=False)
    for _, r in w.iterrows():
        lines.append(
            "- "
            f"{r['scenario']}: agirlik={float(r['weight'])*100.0:.1f}%, "
            f"sayim={int(r['count'])}"
        )
    lines.append("")

    lines.append("### Beklenen Riskte En Kritik Seriler")
    g = expected_df.copy().sort_values(
        ["expected_prob_below_40_pct", "prob_high_risk", "expected_mean_yhat_pct"],
        ascending=[False, False, True],
    ).head(max(1, int(top_n)))
    for _, r in g.iterrows():
        lines.append(
            "- "
            f"{r['series']}: Beklenen P(<40)={float(r['expected_prob_below_40_pct']):.1f}%, "
            f"P(high)={float(r['prob_high_risk'])*100.0:.1f}%, "
            f"beklenen ort. doluluk={float(r['expected_mean_yhat_pct']):.1f}%."
        )
    return "\n".join(lines)


def calibration_summary_md(calibration_metrics: pd.DataFrame | None) -> str:
    if calibration_metrics is None or calibration_metrics.empty:
        return "Kalibrasyon degerlendirmesi henuz uretilmedi."
    overall = calibration_metrics[calibration_metrics["series"] == "__overall__"]
    if overall.empty:
        overall = calibration_metrics.copy()
    o = overall.iloc[0]
    lines = []
    lines.append(
        "- Olasilik kalibrasyon boslugu: "
        f"P(<40) `{float(o['calibration_gap_thr1_pct']):+.1f}` puan, "
        f"P(<30) `{float(o['calibration_gap_thr2_pct']):+.1f}` puan."
    )
    lines.append(
        "- Brier skor (dusuk daha iyi): "
        f"P(<40) `{float(o['brier_thr1']):.3f}`, P(<30) `{float(o['brier_thr2']):.3f}`."
    )
    lines.append(
        "- AUC (yuksek daha iyi): "
        f"P(<40) `{float(o['auc_thr1']):.3f}`, P(<30) `{float(o['auc_thr2']):.3f}`."
    )
    lines.append(
        f"- Aralik kapsama: hedef `{float(o['interval_target_coverage_pct']):.1f}%`, "
        f"gercek `{float(o['interval_empirical_coverage_pct']):.1f}%` "
        f"(fark `{float(o['interval_coverage_gap_pct']):+.1f}` puan)."
    )
    return "\n".join(lines)


def interval_calibration_md(
    factor_df: pd.DataFrame | None,
    factor_summary: dict[str, Any] | None,
    factor_monthly_df: pd.DataFrame | None,
) -> str:
    if factor_df is None or factor_df.empty:
        return "Aralik kalibrasyon adimi henuz calistirilmadi."
    g = factor_df[factor_df["series"] != "__overall__"].copy()
    if g.empty:
        return "Aralik kalibrasyon adimi henuz calistirilmadi."
    o = factor_df[factor_df["series"] == "__overall__"]
    lines = []
    if factor_summary:
        lines.append(
            "- Hedef kapsama: `{:.1f}%`".format(float(factor_summary.get("target_coverage_pct", 90.0)))
        )
        lines.append(
            "- Kalibrasyon modu: `{}`".format(str(factor_summary.get("scale_mode", "series")))
        )
    if not o.empty:
        r = o.iloc[0]
        lines.append(
            "- Ortalama kapsama (once -> sonra): "
            f"`{float(r['coverage_before_pct']):.1f}%` -> `{float(r['coverage_after_pct']):.1f}%`."
        )
        lines.append(
            "- Ortalama aralik carpani: "
            f"`{float(r['interval_scale_factor']):.2f}`."
        )
    lines.append(
        "- Seri bazinda en yuksek carpana ihtiyac duyanlar: "
        + ", ".join(
            f"{r.series} ({float(r.interval_scale_factor):.2f}x)"
            for r in g.sort_values("interval_scale_factor", ascending=False).head(3).itertuples(index=False)
        )
        + "."
    )
    if factor_monthly_df is not None and not factor_monthly_df.empty:
        agg = factor_monthly_df.groupby("series", as_index=False).agg(
            mean_scale=("interval_scale_factor", "mean"),
            std_scale=("interval_scale_factor", "std"),
        )
        agg = agg.sort_values("std_scale", ascending=False)
        top_var = agg.head(3)
        lines.append(
            "- Aylik degiskenligi en yuksek seriler: "
            + ", ".join(
                f"{r.series} (std={float(r.std_scale):.2f})" for r in top_var.itertuples(index=False)
            )
            + "."
        )
    return "\n".join(lines)


def build_markdown(
    run_summary: dict[str, Any],
    scenario_summary: dict[str, Any],
    risk_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    dynamic_risk_df: pd.DataFrame | None,
    dynamic_meta: dict[str, Any] | None,
    calibration_metrics: pd.DataFrame | None,
    interval_factor_df: pd.DataFrame | None,
    interval_factor_summary: dict[str, Any] | None,
    interval_factor_monthly_df: pd.DataFrame | None,
    scenario_shift_df: pd.DataFrame | None,
    scenario_weights_df: pd.DataFrame | None,
    expected_risk_df: pd.DataFrame | None,
    window_start: str,
    window_end: str,
    heatmap_png: Path,
    bars_png: Path,
    scenario_shift_heatmap_png: Path | None,
    expected_risk_png: Path | None,
    dynamic_plot_png: Path | None,
    calibration_rel_png: Path | None,
    calibration_cov_png: Path | None,
    interval_scale_png: Path | None,
    interval_cov_png: Path | None,
    interval_month_heatmap_png: Path | None,
    story_pack_png: Path | None,
    story_timeline_png: Path | None,
    story_gap_heatmap_png: Path | None,
    story_md_path: Path | None,
    top_n: int,
) -> str:
    lines = []
    lines.append("# Istanbul Baraj Tahminleri - Halk Dili Ozet + Bilimsel Uyum Kontrolu")
    lines.append("")
    lines.append("## Kisa Ozet")
    lines.append("")
    lines.append(
        "- Veri kaynagi: IBB acik veri (baraj doluluklari), aylik ortalamaya cevrildi."
    )
    lines.append(
        "- Gozlem donemi: `{}` - `{}`".format(run_summary.get("monthly_start"), run_summary.get("monthly_end"))
    )
    lines.append(
        "- Tahmin ufku: {} ay, odak pencere: `{}` - `{}`".format(
            run_summary.get("forecast_horizon_months"),
            window_start,
            window_end,
        )
    )
    lines.append(
        "- Senaryo metodu: `{}`".format(scenario_summary.get("method", "unknown"))
    )
    if "shift_mode" in scenario_summary:
        lines.append("- Senaryo shift modu: `{}`".format(scenario_summary.get("shift_mode")))
    lines.append("")

    overall = risk_df[risk_df["series"] == "overall_mean"].copy()
    if not overall.empty:
        o = overall.set_index("scenario")
        base = float(o.loc["baseline", "mean_yhat_pct"]) if "baseline" in o.index else np.nan
        dry = float(o.loc["dry_stress", "mean_yhat_pct"]) if "dry_stress" in o.index else np.nan
        wet = float(o.loc["wet_relief", "mean_yhat_pct"]) if "wet_relief" in o.index else np.nan
        lines.append(
            "- Sistem ortalamasi (overall_mean) bu pencerede: "
            f"baz `{base:.1f}%`, kurak stres `{dry:.1f}%`, islak rahatlama `{wet:.1f}%`."
        )
        if np.isfinite(base) and np.isfinite(dry):
            lines.append(f"- Kurak senaryo baz senaryoya gore yaklasik `{base - dry:.1f}` puan daha dusuk.")
        if np.isfinite(base) and np.isfinite(wet):
            lines.append(f"- Islak senaryo baz senaryoya gore yaklasik `{wet - base:.1f}` puan daha yuksek.")
        lines.append("")

    lines.append("## Risk Seviyesi Sayimi")
    lines.append("")
    lines.append(risk_counts_md(risk_df))
    lines.append("")

    lines.append("## Senaryo Bazli En Riskli Barajlar")
    lines.append("")
    lines.append(top_rows_md(risk_df, top_n=top_n))

    lines.append("## Bu Sonuclar Nasil Yorumlanmali?")
    lines.append("")
    lines.append("- Bu model, olasilikli bir erken uyari aracidir; kesin gelecek bildirimi degildir.")
    lines.append("- Yuzde 30/40 alti olasiliklari, operasyonel karar onceliklendirmesi icin kullanilmistir.")
    lines.append("- En kritik cikan barajlarda talep yonetimi, havza koruma ve alternatif kaynak planlari once alinmalidir.")
    lines.append("- Sonuclari aylik guncellemek, yeni gozlem geldikce hatayi azaltir.")
    lines.append("")

    lines.append("## Dinamik Esik (Mevsimsel) Analizi")
    lines.append("")
    if dynamic_meta:
        lines.append(
            "- Dinamik esik metodu: `{}`".format(
                dynamic_meta.get("method", "dynamic_monthly_quantile_thresholds")
            )
        )
        lines.append(
            "- Uyari esigi kuantili: `{:.2f}`, kritik esik kuantili: `{:.2f}`.".format(
                float(dynamic_meta.get("warn_quantile", 0.25)),
                float(dynamic_meta.get("critical_quantile", 0.10)),
            )
        )
        lines.append("")
    lines.append(dynamic_risk_counts_md(dynamic_risk_df))
    lines.append("")
    lines.append("### Dinamik Esige Gore En Riskli Barajlar")
    lines.append("")
    lines.append(dynamic_top_rows_md(dynamic_risk_df, top_n=top_n))

    lines.append("## Senaryo Shift Ozet")
    lines.append("")
    lines.append(scenario_shift_md(scenario_shift_df, top_n=top_n))

    lines.append("## Olasilik Agirlikli Beklenen Risk")
    lines.append("")
    lines.append(expected_risk_md(scenario_weights_df, expected_risk_df, top_n=top_n))
    lines.append("")

    lines.append("## Hikaye Paketi (Halk Dili)")
    lines.append("")
    if story_md_path is not None:
        lines.append(f"- Kisa anlatim dosyasi: `{story_md_path.name}`")
    else:
        lines.append("- Kisa anlatim dosyasi: uretilmedi.")
    if story_pack_png is not None:
        lines.append(f"- Ozet gorsel paketi: `{story_pack_png.name}`")
    else:
        lines.append("- Ozet gorsel paketi: uretilmedi.")
    if story_timeline_png is not None:
        lines.append(f"- Agirlikli timeline: `{story_timeline_png.name}`")
    if story_gap_heatmap_png is not None:
        lines.append(f"- Beklenen uyari-gap isi haritasi: `{story_gap_heatmap_png.name}`")
    lines.append("")

    lines.append("## Kalibrasyon Sonuclari (CV)")
    lines.append("")
    lines.append(calibration_summary_md(calibration_metrics))
    lines.append("")

    lines.append("## Aralik Kalibrasyonu (Coverage Tuning)")
    lines.append("")
    lines.append(interval_calibration_md(interval_factor_df, interval_factor_summary, interval_factor_monthly_df))
    lines.append("")

    lines.append("## Gorseller")
    lines.append("")
    lines.append(f"- Risk isi haritasi: `{heatmap_png.name}`")
    lines.append(f"- Top risk karsilastirma: `{bars_png.name}`")
    if scenario_shift_heatmap_png is not None:
        lines.append(f"- Senaryo shift isi haritasi: `{scenario_shift_heatmap_png.name}`")
    if expected_risk_png is not None:
        lines.append(f"- Agirlikli beklenen risk grafigi: `{expected_risk_png.name}`")
    if dynamic_plot_png is not None:
        lines.append(f"- Dinamik esik risk sayimi: `{dynamic_plot_png.name}`")
    if calibration_rel_png is not None:
        lines.append(f"- Kalibrasyon guvenilirlik egirisi: `{calibration_rel_png.name}`")
    if calibration_cov_png is not None:
        lines.append(f"- Aralik kapsama grafigi: `{calibration_cov_png.name}`")
    if interval_scale_png is not None:
        lines.append(f"- Aralik carpani (seri bazli): `{interval_scale_png.name}`")
    if interval_cov_png is not None:
        lines.append(f"- Kalibrasyon once/sonra kapsama: `{interval_cov_png.name}`")
    if interval_month_heatmap_png is not None:
        lines.append(f"- Aylik aralik carpani isi haritasi: `{interval_month_heatmap_png.name}`")
    if story_pack_png is not None:
        lines.append(f"- Hikaye gorsel paketi: `{story_pack_png.name}`")
    if story_timeline_png is not None:
        lines.append(f"- Hikaye timeline grafigi: `{story_timeline_png.name}`")
    if story_gap_heatmap_png is not None:
        lines.append(f"- Hikaye gap isi haritasi: `{story_gap_heatmap_png.name}`")
    lines.append("")

    lines.append(
        literature_consistency_section(
            run_summary=run_summary,
            scenario_summary=scenario_summary,
            cv_df=cv_df,
            metrics_df=metrics_df,
            forecast_df=forecast_df,
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    req = {
        "scenario_forecasts": args.input_dir / "scenario_forecasts.csv",
        "scenario_risk": args.input_dir / "scenario_risk_summary.csv",
        "cv_predictions": args.input_dir / "istanbul_dam_cv_predictions_decision.csv",
        "cv_metrics": args.input_dir / "istanbul_dam_cv_metrics_decision.csv",
        "forecasts": args.input_dir / "istanbul_dam_forecasts_decision.csv",
        "run_summary": args.input_dir / "run_summary.json",
        "scenario_summary": args.input_dir / "scenario_summary.json",
    }
    missing = [k for k, p in req.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")

    scen_fc = pd.read_csv(req["scenario_forecasts"], parse_dates=["ds"])
    risk_df = pd.read_csv(req["scenario_risk"])
    cv_df = pd.read_csv(req["cv_predictions"], parse_dates=["ds"])
    metrics_df = pd.read_csv(req["cv_metrics"])
    forecast_df = pd.read_csv(req["forecasts"], parse_dates=["ds"])
    run_summary = json.loads(req["run_summary"].read_text(encoding="utf-8"))
    scenario_summary = json.loads(req["scenario_summary"].read_text(encoding="utf-8"))

    dynamic_risk_path = args.input_dir / "scenario_dynamic_risk_summary.csv"
    dynamic_meta_path = args.input_dir / "dynamic_threshold_summary.json"
    scenario_shift_path = args.input_dir / "scenario_shift_factors.csv"
    scenario_shift_heatmap_png = args.output_dir / "scenario_shift_factors_heatmap.png"
    scenario_weights_path = args.input_dir / "scenario_weights.csv"
    expected_risk_path = args.input_dir / "expected_risk_summary.csv"
    expected_risk_png = args.output_dir / "expected_risk_weighted.png"
    dynamic_plot_png = args.output_dir / "dynamic_threshold_risk_counts.png"
    calibration_metrics_path = args.input_dir / "calibration_metrics.csv"
    calibration_rel_png = args.output_dir / "calibration_reliability_overall.png"
    calibration_cov_png = args.output_dir / "calibration_interval_coverage.png"
    interval_factor_path = args.input_dir / "interval_calibration_factors.csv"
    interval_factor_monthly_path = args.input_dir / "interval_calibration_factors_monthly.csv"
    interval_factor_summary_path = args.input_dir / "interval_calibration_summary.json"
    interval_scale_png = args.output_dir / "interval_scale_factors.png"
    interval_cov_png = args.output_dir / "interval_coverage_before_after.png"
    interval_month_heatmap_png = args.output_dir / "interval_scale_factors_monthly_heatmap.png"
    story_pack_png = args.output_dir / "story_visual_pack.png"
    story_timeline_png = args.output_dir / "story_overall_timeline_weighted.png"
    story_gap_heatmap_png = args.output_dir / "story_expected_gap_heatmap.png"
    story_md_path = args.output_dir / "HIKAYE_OZETI.md"

    dynamic_risk_df = pd.read_csv(dynamic_risk_path) if dynamic_risk_path.exists() else None
    dynamic_meta = json.loads(dynamic_meta_path.read_text(encoding="utf-8")) if dynamic_meta_path.exists() else None
    scenario_shift_df = pd.read_csv(scenario_shift_path) if scenario_shift_path.exists() else None
    scenario_weights_df = pd.read_csv(scenario_weights_path) if scenario_weights_path.exists() else None
    expected_risk_df = pd.read_csv(expected_risk_path) if expected_risk_path.exists() else None
    calibration_metrics = pd.read_csv(calibration_metrics_path) if calibration_metrics_path.exists() else None
    interval_factor_df = pd.read_csv(interval_factor_path) if interval_factor_path.exists() else None
    interval_factor_monthly_df = (
        pd.read_csv(interval_factor_monthly_path) if interval_factor_monthly_path.exists() else None
    )
    interval_factor_summary = (
        json.loads(interval_factor_summary_path.read_text(encoding="utf-8")) if interval_factor_summary_path.exists() else None
    )

    risk_df["risk_level"] = risk_df.apply(lambda r: risk_level(r, args), axis=1)

    heatmap_png = args.output_dir / "risk_heatmap_prob_below_40.png"
    bars_png = args.output_dir / "top_risk_compare_by_scenario.png"
    plot_prob_heatmaps(
        scen_fc=scen_fc,
        risk_df=risk_df,
        out_png=heatmap_png,
        window_start=args.window_start,
        window_end=args.window_end,
    )
    plot_top_risk_bars(risk_df=risk_df, out_png=bars_png, top_n=max(4, args.top_n))

    report_md = build_markdown(
        run_summary=run_summary,
        scenario_summary=scenario_summary,
        risk_df=risk_df,
        cv_df=cv_df,
        metrics_df=metrics_df,
        forecast_df=forecast_df,
        dynamic_risk_df=dynamic_risk_df,
        dynamic_meta=dynamic_meta,
        calibration_metrics=calibration_metrics,
        interval_factor_df=interval_factor_df,
        interval_factor_summary=interval_factor_summary,
        interval_factor_monthly_df=interval_factor_monthly_df,
        scenario_shift_df=scenario_shift_df,
        scenario_weights_df=scenario_weights_df,
        expected_risk_df=expected_risk_df,
        window_start=args.window_start,
        window_end=args.window_end,
        heatmap_png=heatmap_png,
        bars_png=bars_png,
        scenario_shift_heatmap_png=(scenario_shift_heatmap_png if scenario_shift_heatmap_png.exists() else None),
        expected_risk_png=(expected_risk_png if expected_risk_png.exists() else None),
        dynamic_plot_png=(dynamic_plot_png if dynamic_plot_png.exists() else None),
        calibration_rel_png=(calibration_rel_png if calibration_rel_png.exists() else None),
        calibration_cov_png=(calibration_cov_png if calibration_cov_png.exists() else None),
        interval_scale_png=(interval_scale_png if interval_scale_png.exists() else None),
        interval_cov_png=(interval_cov_png if interval_cov_png.exists() else None),
        interval_month_heatmap_png=(interval_month_heatmap_png if interval_month_heatmap_png.exists() else None),
        story_pack_png=(story_pack_png if story_pack_png.exists() else None),
        story_timeline_png=(story_timeline_png if story_timeline_png.exists() else None),
        story_gap_heatmap_png=(story_gap_heatmap_png if story_gap_heatmap_png.exists() else None),
        story_md_path=(story_md_path if story_md_path.exists() else None),
        top_n=args.top_n,
    )
    report_path = args.output_dir / "SONUC_OZETI_VE_LITERATUR_KONTROLU.md"
    report_path.write_text(report_md, encoding="utf-8")

    payload = {
        "window_start": args.window_start,
        "window_end": args.window_end,
        "report_path": str(report_path),
        "heatmap_path": str(heatmap_png),
        "bars_path": str(bars_png),
        "dynamic_risk_path": str(dynamic_risk_path) if dynamic_risk_path.exists() else None,
        "scenario_shift_path": str(scenario_shift_path) if scenario_shift_path.exists() else None,
        "scenario_weights_path": str(scenario_weights_path) if scenario_weights_path.exists() else None,
        "expected_risk_path": str(expected_risk_path) if expected_risk_path.exists() else None,
        "calibration_metrics_path": str(calibration_metrics_path) if calibration_metrics_path.exists() else None,
        "interval_calibration_factors_path": str(interval_factor_path) if interval_factor_path.exists() else None,
        "interval_calibration_factors_monthly_path": (
            str(interval_factor_monthly_path) if interval_factor_monthly_path.exists() else None
        ),
        "story_visual_pack_path": str(story_pack_png) if story_pack_png.exists() else None,
        "story_timeline_path": str(story_timeline_png) if story_timeline_png.exists() else None,
        "story_gap_heatmap_path": str(story_gap_heatmap_png) if story_gap_heatmap_png.exists() else None,
        "story_summary_markdown_path": str(story_md_path) if story_md_path.exists() else None,
        "top_n": int(args.top_n),
    }
    (args.output_dir / "public_report_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(report_path)
    print(heatmap_png)
    print(bars_png)


if __name__ == "__main__":
    main()
