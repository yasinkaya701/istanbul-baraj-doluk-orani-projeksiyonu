#!/usr/bin/env python3
"""Build skin-cancer focused summary and visuals from existing health outputs.

Scientific guardrail:
- UV variable is not present in the current pipeline.
- Therefore this pack does NOT produce numeric skin-cancer incidence increase.
- It reports directional risk pressure using climate proxy indicators only.
- It also provides a per-10k *proxy scenario* table anchored to a transparent
  baseline (default: GLOBOCAN 2022 Turkiye melanoma new cases/population).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build skin-cancer focused summary and dashboard.")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--date-label", type=str, default=str(date.today()))
    p.add_argument(
        "--solar-forecast-csv",
        type=Path,
        default=Path("output/forecast_package/forecasts/solar_potential_monthly_forecast.csv"),
        help="Monthly solar forecast CSV used for baseline/future solar-energy context.",
    )
    p.add_argument("--solar-baseline-start", type=int, default=2013)
    p.add_argument("--solar-baseline-end", type=int, default=2022)
    p.add_argument("--solar-future-start", type=int, default=2026)
    p.add_argument("--solar-future-end", type=int, default=2035)
    p.add_argument(
        "--baseline-population",
        type=float,
        default=85561976.0,
        help="Baseline population for per-10k conversion (default: Turkiye 2022 population from GLOBOCAN factsheet).",
    )
    p.add_argument(
        "--baseline-new-cases",
        type=float,
        default=1783.0,
        help="Baseline annual new skin-cancer cases used for per-10k anchor (default: melanoma 2022, GLOBOCAN factsheet).",
    )
    return p.parse_args()


def _find_existing(paths: list[Path], label: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise SystemExit(f"Missing required {label}. Tried: {', '.join(str(x) for x in paths)}")


def _model_key(value: object) -> str:
    s = str(value).strip().lower()
    if "strong" in s:
        return "strong"
    if "quant" in s:
        return "quant"
    return s


def _safe_num(v: object, default: float = np.nan) -> float:
    try:
        out = float(v)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _proxy_level(idx: float) -> str:
    if not np.isfinite(idx):
        return "bilinmiyor"
    if idx >= 70:
        return "yuksek_proxy"
    if idx >= 35:
        return "orta_proxy"
    return "dusuk_proxy"


def _conf_score(label: str) -> float:
    m = {"yuksek": 0.80, "orta": 0.58, "dusuk": 0.35}
    return float(m.get(str(label).strip().lower(), 0.45))


def _yn(v: bool) -> str:
    return "var" if bool(v) else "yok"


def _resolve_inputs(root: Path, date_label: str) -> dict[str, Path]:
    return {
        "all_disease_summary": _find_existing(
            [root / "health_all_disease_summary.csv"],
            "health_all_disease_summary.csv",
        ),
        "etiology_matrix": _find_existing(
            [root / "health_all_disease_etiology_matrix.csv"],
            "health_all_disease_etiology_matrix.csv",
        ),
        "consensus_skin": _find_existing(
            [
                root / f"halk_dili_hastalik_ozet_{date_label}.csv",
                root / "halk_dili_hastalik_ozet_latest.csv",
                root / "halk_dili_hastalik_ozet_2026-03-06.csv",
            ],
            "halk_dili_hastalik_ozet_<DATE>.csv",
        ),
        "comparison": _find_existing(
            [
                root / "model_comparison_summary_stable_calibrated.csv",
                root / "model_comparison_summary_duzenlenmis_run.csv",
                root / "model_comparison_summary.csv",
            ],
            "model comparison summary",
        ),
        "stability_diag": _find_existing(
            [
                root / f"model_stability_diagnostics_{date_label}.csv",
                root / "model_stability_diagnostics_latest.csv",
            ],
            "model_stability_diagnostics",
        ),
    }


def _build_model_rows(
    skin_model_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    etiology_df: pd.DataFrame,
) -> pd.DataFrame:
    comp = comparison_df.copy()
    comp["model_key"] = comp["model"].map(_model_key)
    stab = stability_df.copy()
    stab["model_key"] = stab.get("model_key", stab.get("model", "")).map(_model_key)
    skin_eti = etiology_df.copy()
    if "model_key" not in skin_eti.columns and "model" in skin_eti.columns:
        skin_eti["model_key"] = skin_eti["model"].map(_model_key)

    out_rows: list[dict[str, object]] = []
    for _, r in skin_model_df.iterrows():
        mk = _model_key(r["model"])
        comp_row = comp[comp["model_key"] == mk]
        stab_row = stab[stab["model_key"] == mk]
        eti_rows = skin_eti[skin_eti["model_key"] == mk]

        delta_hi = _safe_num(comp_row.iloc[0]["delta_hi_mean_c"]) if not comp_row.empty else np.nan
        thr_share = _safe_num(comp_row.iloc[0].get("future_threshold_exceed_share", np.nan)) if not comp_row.empty else np.nan
        wet_share = _safe_num(comp_row.iloc[0].get("future_wet_hot_share", np.nan)) if not comp_row.empty else np.nan
        future_rr = _safe_num(comp_row.iloc[0].get("future_rr_mean", np.nan)) if not comp_row.empty else np.nan

        hi_norm = float(np.clip(delta_hi / 12.0, 0.0, 1.0)) if np.isfinite(delta_hi) else np.nan
        thr_norm = float(np.clip(thr_share, 0.0, 1.0)) if np.isfinite(thr_share) else np.nan
        wet_norm = float(np.clip(wet_share, 0.0, 1.0)) if np.isfinite(wet_share) else np.nan
        if np.isfinite(hi_norm) and np.isfinite(thr_norm) and np.isfinite(wet_norm):
            proxy_idx = float(100.0 * np.clip((0.55 * hi_norm) + (0.30 * thr_norm) + (0.15 * wet_norm), 0.0, 1.0))
        else:
            proxy_idx = np.nan

        uv_q = eti_rows[
            eti_rows["etiology_id"].astype(str).str.strip().str.lower().eq("uv_radiation")
        ]["quantifiable_with_current_data"]
        occ_q = eti_rows[
            eti_rows["etiology_id"].astype(str).str.strip().str.lower().eq("occupational_exposure")
        ]["quantifiable_with_current_data"]

        conf_label = str(stab_row.iloc[0].get("prediction_confidence", "bilinmiyor")) if not stab_row.empty else "bilinmiyor"
        conf_score = _conf_score(conf_label)

        out_rows.append(
            {
                "model_key": mk,
                "direct_signal_score": _safe_num(r["direct_signal_score"], np.nan),
                "direct_signal_level": str(r.get("direct_signal_level", "bilinmiyor")),
                "future_rr_mean": future_rr,
                "delta_hi_mean_c": delta_hi,
                "future_threshold_exceed_share": thr_norm,
                "future_wet_hot_share": wet_norm,
                "proxy_component_hi_norm": hi_norm,
                "proxy_component_threshold": thr_norm,
                "proxy_component_wet_hot": wet_norm,
                "climate_uv_proxy_index_0_100": proxy_idx,
                "climate_uv_proxy_level": _proxy_level(proxy_idx),
                "prediction_confidence": conf_label,
                "prediction_confidence_score_0_1": conf_score,
                "uv_quantifiable_with_current_data": bool(uv_q.astype(bool).any()) if not uv_q.empty else False,
                "occupational_quantifiable_with_current_data": bool(occ_q.astype(bool).any()) if not occ_q.empty else False,
                "numeric_skin_incidence_estimation_allowed": False,
            }
        )

    out = pd.DataFrame(out_rows)
    order = pd.Categorical(out["model_key"], categories=["quant", "strong"], ordered=True)
    out = out.assign(_ord=order).sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return out


def _resolve_solar_csv(path: Path) -> Path:
    cands = [path, Path("/Users/yasinkaya/Hackhaton") / path]
    for p in cands:
        if p.exists():
            return p.resolve()
    raise SystemExit(f"Solar forecast CSV not found: {path}")


def _solar_column(df: pd.DataFrame) -> str:
    for c in [
        "solar_potential_kwh_m2_day",
        "solar_potential_expected_kwh_m2_day",
        "solar_potential_p50_kwh_m2_day",
        "global_horizontal_kwh_m2_day",
        "shortwave_internet_kwh_m2_day",
    ]:
        if c in df.columns:
            return c
    raise SystemExit("No usable solar column found in solar forecast CSV.")


def _solar_stats(
    solar_df: pd.DataFrame,
    col: str,
    baseline_start: int,
    baseline_end: int,
    future_start: int,
    future_end: int,
) -> dict[str, float]:
    x = solar_df.copy()
    if "timestamp" not in x.columns:
        raise SystemExit("Solar forecast CSV requires 'timestamp' column.")
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["timestamp", col]).copy()
    x["year"] = x["timestamp"].dt.year

    base = x[x["year"].between(int(baseline_start), int(baseline_end), inclusive="both")]
    fut = x[x["year"].between(int(future_start), int(future_end), inclusive="both")]
    if base.empty or fut.empty:
        raise SystemExit(
            f"Solar periods empty. baseline={baseline_start}-{baseline_end} rows={len(base)}, "
            f"future={future_start}-{future_end} rows={len(fut)}"
        )
    base_mean = float(base[col].mean())
    fut_mean = float(fut[col].mean())
    delta = float(fut_mean - base_mean)
    pct = float((delta / base_mean) * 100.0) if abs(base_mean) > 1e-9 else np.nan
    return {
        "baseline_mean_kwh_m2_day": base_mean,
        "future_mean_kwh_m2_day": fut_mean,
        "delta_kwh_m2_day": delta,
        "delta_pct": pct,
        "baseline_n": float(len(base)),
        "future_n": float(len(fut)),
    }


def _build_cases_per_10k_proxy_table(
    model_rows: pd.DataFrame,
    solar_stats: dict[str, float],
    baseline_population: float,
    baseline_new_cases: float,
) -> pd.DataFrame:
    if baseline_population <= 0:
        raise SystemExit("baseline_population must be > 0.")
    if baseline_new_cases < 0:
        raise SystemExit("baseline_new_cases must be >= 0.")

    base_per10k = float((baseline_new_cases / baseline_population) * 10000.0)
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "scenario": "baseline_reference",
            "model_key": "baseline",
            "solar_baseline_mean_kwh_m2_day": float(solar_stats["baseline_mean_kwh_m2_day"]),
            "solar_future_mean_kwh_m2_day": float(solar_stats["future_mean_kwh_m2_day"]),
            "solar_delta_pct": float(solar_stats["delta_pct"]),
            "baseline_population": float(baseline_population),
            "baseline_new_cases": float(baseline_new_cases),
            "baseline_cases_per_10000": base_per10k,
            "future_rr_mean": 1.0,
            "climate_uv_proxy_index_0_100": np.nan,
            "proxy_rr_multiplier": 1.0,
            "projected_cases_per_10000": base_per10k,
            "additional_cases_per_10000": 0.0,
            "note": "Referans: GLOBOCAN 2022 yillik yeni melanoma vaka / nufus",
        }
    )

    for _, r in model_rows.iterrows():
        mk = str(r["model_key"])
        rr = _safe_num(r.get("future_rr_mean", np.nan), np.nan)
        if not np.isfinite(rr):
            rr = 1.0
        rr_uplift = max(0.0, rr - 1.0)
        uv_proxy = np.clip(_safe_num(r.get("climate_uv_proxy_index_0_100", np.nan), 0.0) / 100.0, 0.0, 1.0)
        # Conservative proxy: only the uplift part is scaled by UV-climate proxy.
        proxy_rr = float(1.0 + (rr_uplift * uv_proxy))
        projected = float(base_per10k * proxy_rr)
        rows.append(
            {
                "scenario": f"{mk}_proxy_rr",
                "model_key": mk,
                "solar_baseline_mean_kwh_m2_day": float(solar_stats["baseline_mean_kwh_m2_day"]),
                "solar_future_mean_kwh_m2_day": float(solar_stats["future_mean_kwh_m2_day"]),
                "solar_delta_pct": float(solar_stats["delta_pct"]),
                "baseline_population": float(baseline_population),
                "baseline_new_cases": float(baseline_new_cases),
                "baseline_cases_per_10000": base_per10k,
                "future_rr_mean": rr,
                "climate_uv_proxy_index_0_100": float(_safe_num(r.get("climate_uv_proxy_index_0_100", np.nan), np.nan)),
                "proxy_rr_multiplier": proxy_rr,
                "projected_cases_per_10000": projected,
                "additional_cases_per_10000": float(projected - base_per10k),
                "note": "Proxy senaryo: sayisal cilt-kanseri insidansi degil, iklim-baski modullu RR",
            }
        )

    out = pd.DataFrame(rows)
    order = pd.Categorical(out["model_key"], categories=["baseline", "quant", "strong"], ordered=True)
    out = out.assign(_ord=order).sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return out


def _build_etiology_table(etiology_df: pd.DataFrame) -> pd.DataFrame:
    skin = etiology_df[
        etiology_df["disease_group_id"].astype(str).str.strip().str.lower().eq("skin_cancer")
        | etiology_df["disease_group_tr"].astype(str).str.lower().str.contains("cilt kanseri", regex=False)
    ].copy()
    if skin.empty:
        return pd.DataFrame(
            columns=[
                "model_key",
                "etiology_id",
                "etiology_tr",
                "quantifiable_with_current_data",
                "signal_level",
                "primary_references",
            ]
        )
    skin["model_key"] = skin["model"].map(_model_key)
    keep = [
        "model_key",
        "etiology_id",
        "etiology_tr",
        "quantifiable_with_current_data",
        "signal_level",
        "primary_references",
    ]
    out = skin[keep].copy()
    order_m = pd.Categorical(out["model_key"], categories=["quant", "strong"], ordered=True)
    order_e = pd.Categorical(out["etiology_id"], categories=["uv_radiation", "occupational_exposure"], ordered=True)
    out = out.assign(_m=order_m, _e=order_e).sort_values(["_m", "_e"]).drop(columns=["_m", "_e"]).reset_index(drop=True)
    return out


def _write_markdown(
    out_path: Path,
    date_label: str,
    consensus_skin_row: pd.Series | None,
    model_rows: pd.DataFrame,
    etiology_rows: pd.DataFrame,
    cases_per10k_df: pd.DataFrame,
    solar_stats: dict[str, float],
) -> None:
    lines = [
        f"# Cilt Kanseri Odak Notu ({date_label})",
        "",
        "## Net Sonuc",
        "- Cilt kanseri icin ana surucu UV maruziyetidir.",
        "- Mevcut veri boru hattinda UV degiskeni yok.",
        "- Bu nedenle cilt kanseri icin sayisal artis yuzdesi / insidans tahmini verilmez.",
        "- Raporlanan sayilar yalnizca iklim baskisi proksileridir (insidans degil).",
        "",
    ]

    if consensus_skin_row is not None:
        lines.extend(
            [
                "## Konsensus Satiri",
                f"- Risk duzeyi: {str(consensus_skin_row.get('risk_duzeyi', 'bilinmiyor'))}",
                f"- Sayisal kanit guveni: {str(consensus_skin_row.get('sayisal_kanit_guveni', 'bilinmiyor'))}",
                f"- Model uyumu: {str(consensus_skin_row.get('model_uyumu', 'bilinmiyor'))}",
                f"- Dogrudan olculebilir oran: %{100.0 * _safe_num(consensus_skin_row.get('dogrudan_olculebilir_oran', np.nan), 0.0):.1f}",
                f"- Literaturlle celiski: {str(consensus_skin_row.get('literatur_celiski_durumu', 'bilinmiyor'))}",
                "",
            ]
        )

    lines.extend(["## Model Bazli Iklim-Baski Proksisi", ""])
    for _, r in model_rows.iterrows():
        lines.append(
            f"- {r['model_key']}: proxy_index={_safe_num(r['climate_uv_proxy_index_0_100'], np.nan):.1f}/100 "
            f"({r['climate_uv_proxy_level']}), "
            f"delta_hi={_safe_num(r['delta_hi_mean_c'], np.nan):+.2f}C, "
            f"threshold_share={100.0 * _safe_num(r['future_threshold_exceed_share'], 0.0):.1f}%, "
            f"wet_hot_share={100.0 * _safe_num(r['future_wet_hot_share'], 0.0):.1f}%, "
            f"tahmin_guveni={r['prediction_confidence']}"
        )

    lines.extend(
        [
            "",
            "## Gunes Enerjisi (Solar Potential) Ozeti",
            f"- Baseline ortalama ({int(solar_stats.get('baseline_n', 0))} ay): "
            f"{float(solar_stats['baseline_mean_kwh_m2_day']):.3f} kWh/m2/gun",
            f"- Gelecek ortalama ({int(solar_stats.get('future_n', 0))} ay): "
            f"{float(solar_stats['future_mean_kwh_m2_day']):.3f} kWh/m2/gun",
            f"- Degisim: {float(solar_stats['delta_kwh_m2_day']):+.3f} kWh/m2/gun "
            f"({float(solar_stats['delta_pct']):+.2f}%)",
            "",
            "## 10.000 Kiside Kanserli Insan Sayisi (Proxy Senaryo)",
        ]
    )
    for _, r in cases_per10k_df.iterrows():
        lines.append(
            f"- {r['model_key']}: baseline={float(r['baseline_cases_per_10000']):.4f}/10k, "
            f"projeksiyon={float(r['projected_cases_per_10000']):.4f}/10k, "
            f"ek={float(r['additional_cases_per_10000']):+.4f}/10k, "
            f"proxy_rr={float(r['proxy_rr_multiplier']):.4f}"
        )

    uv_q = bool(model_rows["uv_quantifiable_with_current_data"].any()) if not model_rows.empty else False
    occ_q = bool(model_rows["occupational_quantifiable_with_current_data"].any()) if not model_rows.empty else False
    lines.extend(
        [
            "",
            "## Etiyoloji Kapsam Durumu",
            f"- UV maruziyeti dogrudan olculebilir mi?: {_yn(uv_q)}",
            f"- Mesleki maruziyet (dis ortam sicaklik+UV) dogrudan olculebilir mi?: {_yn(occ_q)}",
            "- Her iki etiyoloji icin su an nitel baski yorumu yapilir.",
            "",
            "## Bilimsel Kural",
            "- UV katmani eklenmeden cilt kanseri icin sayisal artis yuzdesi raporlanmamalidir.",
            "",
            "## 90 Gunluk Veri-Oncelik Onerisi",
            "1. Ilce-gunluk UV index / UV doz katmanini modele ekle.",
            "2. Mesleki acik alan maruziyetini (saat, sektor, bolge) birlestir.",
            "3. Hastane/kanser kayit verisiyle gecikmeli (lag) kalibrasyon yap.",
            "4. UV + sicaklik + nem ile skin_cancer icin ayri RR modeli kur (duzey + belirsizlik).",
            "",
        ]
    )

    refs = []
    if not etiology_rows.empty and "primary_references" in etiology_rows.columns:
        for chunk in etiology_rows["primary_references"].dropna().astype(str):
            refs.extend([x.strip() for x in chunk.split(";") if x.strip()])
    uniq_refs = sorted(set(refs))
    if uniq_refs:
        lines.append("## Birincil Referanslar")
        for ref in uniq_refs:
            lines.append(f"- {ref}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_dashboard(model_rows: pd.DataFrame, etiology_rows: pd.DataFrame, out_path: Path) -> None:
    if model_rows.empty:
        raise SystemExit("No skin-cancer model rows available to plot.")

    d = model_rows.copy()
    x = np.arange(len(d))
    labels = d["model_key"].astype(str).tolist()

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    # Panel-1: proxy index
    ax1 = axes[0, 0]
    proxy = d["climate_uv_proxy_index_0_100"].to_numpy(dtype=float)
    colors = ["#1f77b4" if k == "quant" else "#d95f02" for k in labels]
    ax1.bar(x, proxy, color=colors, width=0.55)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Proxy index (0-100)")
    ax1.set_title("Cilt Kanseri Icin Iklim-Baski Proksisi")
    ax1.grid(axis="y", alpha=0.25)
    for xi, yi in zip(x, proxy):
        if np.isfinite(yi):
            ax1.text(xi, yi + 2.0, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)

    # Panel-2: proxy components
    ax2 = axes[0, 1]
    c_hi = (100.0 * d["proxy_component_hi_norm"].to_numpy(dtype=float)).clip(min=0.0)
    c_thr = (100.0 * d["proxy_component_threshold"].to_numpy(dtype=float)).clip(min=0.0)
    c_wet = (100.0 * d["proxy_component_wet_hot"].to_numpy(dtype=float)).clip(min=0.0)
    ax2.bar(x, c_hi, width=0.56, color="#e66101", label="HI degisimi (norm)")
    ax2.bar(x, c_thr, width=0.56, color="#5e3c99", bottom=c_hi, label="Threshold exceed")
    ax2.bar(x, c_wet, width=0.56, color="#1b9e77", bottom=(c_hi + c_thr), label="Wet-hot payi")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Bilesen skorlari (0-100 toplamsiz)")
    ax2.set_title("Proxy Bilesenleri (Insidans Degil)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel-3: direct signal + confidence
    ax3 = axes[1, 0]
    direct = d["direct_signal_score"].to_numpy(dtype=float)
    conf = d["prediction_confidence_score_0_1"].to_numpy(dtype=float)
    w = 0.35
    ax3.bar(x - w / 2.0, direct, width=w, color="#4daf4a", label="direct_signal_score")
    ax3b = ax3.twinx()
    ax3b.plot(x + w / 2.0, conf, color="#377eb8", marker="o", linewidth=2, label="prediction_confidence")
    ax3.set_xticks(x, labels)
    ax3.set_ylabel("Direct signal score")
    ax3b.set_ylabel("Guven skoru (0-1)")
    ax3b.set_ylim(0, 1)
    ax3.set_title("Model Sinyali ve Tahmin Guveni")
    ax3.grid(axis="y", alpha=0.25)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=True)

    # Panel-4: coverage matrix
    ax4 = axes[1, 1]
    if etiology_rows.empty:
        mat = np.zeros((1, 2), dtype=float)
        row_labels = ["veri_yok"]
    else:
        tmp = etiology_rows.copy()
        tmp["model_key"] = tmp["model_key"].astype(str)
        tmp["etiology_id"] = tmp["etiology_id"].astype(str)
        tmp["row_label"] = tmp["model_key"] + " | " + tmp["etiology_id"]
        direct_var = tmp["quantifiable_with_current_data"].astype(bool).astype(float).to_numpy(dtype=float)
        lit_support = np.ones(len(tmp), dtype=float)
        mat = np.column_stack([direct_var, lit_support])
        row_labels = tmp["row_label"].tolist()
    im = ax4.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax4.set_xticks([0, 1], ["dogrudan_degisken", "literatur_destegi"])
    ax4.set_yticks(np.arange(len(row_labels)), row_labels)
    ax4.set_title("Etiyoloji Kapsama Matrisi")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = "var" if mat[i, j] >= 0.5 else "yok"
            ax4.text(j, i, txt, ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=ax4, fraction=0.045, pad=0.03)

    fig.suptitle("Cilt Kanseri Odak Panosu (Sayisal Insidans Tahmini Yok)", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_cases_per10k(
    cases_df: pd.DataFrame,
    solar_stats: dict[str, float],
    out_path: Path,
) -> None:
    if cases_df.empty:
        raise SystemExit("No per-10k rows to plot.")

    x_df = cases_df[cases_df["model_key"].isin(["baseline", "quant", "strong"])].copy()
    x_df["model_key"] = pd.Categorical(x_df["model_key"], categories=["baseline", "quant", "strong"], ordered=True)
    x_df = x_df.sort_values("model_key")
    labels = x_df["model_key"].astype(str).tolist()
    vals = x_df["projected_cases_per_10000"].to_numpy(dtype=float)
    add = x_df["additional_cases_per_10000"].to_numpy(dtype=float)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))

    # Panel-1: solar baseline/future
    ax1 = axes[0]
    solar_vals = [
        float(solar_stats["baseline_mean_kwh_m2_day"]),
        float(solar_stats["future_mean_kwh_m2_day"]),
    ]
    ax1.bar(["baseline", "gelecek"], solar_vals, color=["#4daf4a", "#e66101"], width=0.58)
    ax1.set_ylabel("kWh/m2/gun")
    ax1.set_title("Gunes Enerjisi Ortalamasi")
    ax1.grid(axis="y", alpha=0.25)
    delta_pct = float(solar_stats["delta_pct"])
    ax1.text(
        0.02,
        0.94,
        f"Degisim: {delta_pct:+.2f}%",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#f3f3f3", "ec": "#aaaaaa"},
    )

    # Panel-2: cases per 10k
    ax2 = axes[1]
    colors = ["#7f7f7f", "#1f78b4", "#d95f02"]
    ax2.bar(labels, vals, color=colors, width=0.58)
    ax2.set_ylabel("Kisi / 10.000")
    ax2.set_title("Cilt Kanseri Vaka Senaryosu (10.000 kiside)")
    ax2.grid(axis="y", alpha=0.25)
    for xi, (v, a) in enumerate(zip(vals, add)):
        ax2.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        if labels[xi] != "baseline":
            ax2.text(xi, max(v * 0.55, 0.004), f"ek {a:+.4f}", ha="center", va="center", fontsize=8, color="#202020")

    fig.suptitle("Gunes Enerjisi ve Cilt Kanseri (10.000 Kiside Proxy Senaryo)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _copy_latest(dated: Path, latest: Path) -> None:
    if dated.exists():
        latest.write_bytes(dated.read_bytes())


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ins = _resolve_inputs(root, args.date_label)

    all_dis = pd.read_csv(ins["all_disease_summary"])
    etiology = pd.read_csv(ins["etiology_matrix"])
    consensus = pd.read_csv(ins["consensus_skin"])
    comparison = pd.read_csv(ins["comparison"])
    stability = pd.read_csv(ins["stability_diag"])
    solar_csv = _resolve_solar_csv(args.solar_forecast_csv)
    solar_df = pd.read_csv(solar_csv)
    solar_col = _solar_column(solar_df)
    solar_stats = _solar_stats(
        solar_df=solar_df,
        col=solar_col,
        baseline_start=int(args.solar_baseline_start),
        baseline_end=int(args.solar_baseline_end),
        future_start=int(args.solar_future_start),
        future_end=int(args.solar_future_end),
    )

    skin_model = all_dis[all_dis["disease_group_id"].astype(str).str.lower().eq("skin_cancer")].copy()
    if skin_model.empty:
        skin_model = all_dis[all_dis["disease_group_tr"].astype(str).str.lower().str.contains("cilt kanseri", regex=False)].copy()
    if skin_model.empty:
        raise SystemExit("Skin-cancer row not found in health_all_disease_summary.")

    cons_skin = consensus[consensus["disease_group_tr"].astype(str).str.lower().str.contains("cilt kanseri", regex=False)]
    cons_row = cons_skin.iloc[0] if not cons_skin.empty else None

    etio_skin = _build_etiology_table(etiology_df=etiology)
    model_rows = _build_model_rows(
        skin_model_df=skin_model,
        comparison_df=comparison,
        stability_df=stability,
        etiology_df=etio_skin,
    )
    cases_per10k_df = _build_cases_per_10k_proxy_table(
        model_rows=model_rows,
        solar_stats=solar_stats,
        baseline_population=float(args.baseline_population),
        baseline_new_cases=float(args.baseline_new_cases),
    )

    out_model_csv = root / f"cilt_kanseri_odak_model_tablosu_{args.date_label}.csv"
    out_eti_csv = root / f"cilt_kanseri_odak_etioloji_tablosu_{args.date_label}.csv"
    out_md = root / f"cilt_kanseri_odak_ozet_{args.date_label}.md"
    out_png = root / f"cilt_kanseri_odak_pano_{args.date_label}.png"
    out_10k_csv = root / f"cilt_kanseri_10000_kisi_senaryo_{args.date_label}.csv"
    out_10k_png = root / f"cilt_kanseri_10000_kisi_gorsel_{args.date_label}.png"

    model_rows.to_csv(out_model_csv, index=False)
    etio_skin.to_csv(out_eti_csv, index=False)
    cases_per10k_df.to_csv(out_10k_csv, index=False)
    _write_markdown(
        out_path=out_md,
        date_label=args.date_label,
        consensus_skin_row=cons_row,
        model_rows=model_rows,
        etiology_rows=etio_skin,
        cases_per10k_df=cases_per10k_df,
        solar_stats=solar_stats,
    )
    _plot_dashboard(model_rows=model_rows, etiology_rows=etio_skin, out_path=out_png)
    _plot_cases_per10k(cases_df=cases_per10k_df, solar_stats=solar_stats, out_path=out_10k_png)

    _copy_latest(out_model_csv, root / "cilt_kanseri_odak_model_tablosu_latest.csv")
    _copy_latest(out_eti_csv, root / "cilt_kanseri_odak_etioloji_tablosu_latest.csv")
    _copy_latest(out_md, root / "cilt_kanseri_odak_ozet_latest.md")
    _copy_latest(out_png, root / "cilt_kanseri_odak_pano_latest.png")
    _copy_latest(out_10k_csv, root / "cilt_kanseri_10000_kisi_senaryo_latest.csv")
    _copy_latest(out_10k_png, root / "cilt_kanseri_10000_kisi_gorsel_latest.png")

    print(f"Wrote: {out_model_csv}")
    print(f"Wrote: {out_eti_csv}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_10k_csv}")
    print(f"Wrote: {out_10k_png}")
    print(f"Wrote: {root / 'cilt_kanseri_odak_model_tablosu_latest.csv'}")
    print(f"Wrote: {root / 'cilt_kanseri_odak_etioloji_tablosu_latest.csv'}")
    print(f"Wrote: {root / 'cilt_kanseri_odak_ozet_latest.md'}")
    print(f"Wrote: {root / 'cilt_kanseri_odak_pano_latest.png'}")
    print(f"Wrote: {root / 'cilt_kanseri_10000_kisi_senaryo_latest.csv'}")
    print(f"Wrote: {root / 'cilt_kanseri_10000_kisi_gorsel_latest.png'}")


if __name__ == "__main__":
    main()
