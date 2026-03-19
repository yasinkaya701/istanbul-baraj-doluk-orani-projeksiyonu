#!/usr/bin/env python3
"""Build a simple and transparent summary pack for yearly skin-cancer evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create simple quality-focused summary outputs")
    p.add_argument(
        "--root-dir",
        type=Path,
        default=Path("output/yearly_skin_cancer_evidence"),
        help="Existing yearly evidence output directory",
    )
    return p.parse_args()


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if np.isfinite(x) else default


def _label(score: float) -> str:
    if score >= 0.75:
        return "Yuksek"
    if score >= 0.55:
        return "Orta"
    return "Dusuk"


def _copy_latest(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    annual = pd.read_csv(root / "annual_climate_and_skin_cancer_proxy.csv")
    trend = pd.read_csv(root / "trend_proof_table.csv")
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))

    hist = annual[annual["is_projected"] == False].copy()  # noqa: E712
    if hist.empty:
        raise SystemExit("No historical rows found")
    history_end = int(hist["year"].max())
    proj_end = int(annual["year"].max())

    row_h = annual[annual["year"] == history_end].iloc[0]
    row_p = annual[annual["year"] == proj_end].iloc[0]

    p_uv = _safe_float(
        trend.loc[trend["variable"].eq("effective_uv_kwh_m2_day"), "p_value"].iloc[0]
    )
    p_case = _safe_float(
        trend.loc[trend["variable"].eq("cases_per10k_mid"), "p_value"].iloc[0]
    )
    slope_uv = _safe_float(
        trend.loc[trend["variable"].eq("effective_uv_kwh_m2_day"), "slope_per_decade"].iloc[0]
    )
    slope_case = _safe_float(
        trend.loc[trend["variable"].eq("cases_per10k_mid"), "slope_per_decade"].iloc[0]
    )

    cov_temp = float(hist["temp_local_c"].notna().mean())
    cov_hum = float(hist["humidity_local_pct"].notna().mean())
    cov_prec = float(hist["precip_local_mm"].notna().mean())
    coverage_score = float(np.mean([cov_temp, cov_hum, cov_prec, 1.0, 1.0]))

    stat_score = float(np.mean([1.0 if p_uv < 0.01 else 0.0, 1.0 if p_case < 0.01 else 0.0]))
    lit_score = 1.0 if str(meta.get("literature_consistency", "")).strip().lower() == "uyumlu" else 0.6

    climate_conf = float(np.clip((0.5 * coverage_score) + (0.4 * stat_score) + (0.1 * lit_score), 0.0, 1.0))
    # Skin-cancer row is a proxy scenario (not direct incidence): apply explicit penalty.
    skin_proxy_conf = float(np.clip(climate_conf * 0.65, 0.0, 1.0))

    annual_csv = root / "sade_ozet_sayilar.csv"
    pd.DataFrame(
        [
            {
                "history_end_year": history_end,
                "projection_end_year": proj_end,
                "cases_per10k_history_end": _safe_float(row_h["cases_per10k_mid"]),
                "cases_per10k_projection_end": _safe_float(row_p["cases_per10k_mid"]),
                "cases_per10k_projection_low": _safe_float(row_p["cases_per10k_low"]),
                "cases_per10k_projection_high": _safe_float(row_p["cases_per10k_high"]),
                "effective_uv_history_end": _safe_float(row_h["effective_uv_kwh_m2_day"]),
                "effective_uv_projection_end": _safe_float(row_p["effective_uv_kwh_m2_day"]),
                "p_uv_trend": p_uv,
                "p_skin_proxy_trend": p_case,
                "slope_uv_per_decade": slope_uv,
                "slope_skin_proxy_per_decade": slope_case,
                "coverage_temp_local": cov_temp,
                "coverage_humidity_local": cov_hum,
                "coverage_precip_local": cov_prec,
                "confidence_climate_score_0_1": climate_conf,
                "confidence_climate_label": _label(climate_conf),
                "confidence_skin_proxy_score_0_1": skin_proxy_conf,
                "confidence_skin_proxy_label": _label(skin_proxy_conf),
                "literature_consistency": meta.get("literature_consistency"),
            }
        ]
    ).to_csv(annual_csv, index=False)

    md = root / "sade_ozet_tr.md"
    md_lines = [
        "# Sade Sonuc",
        "",
        "## 1) Ana mesaj",
        f"- 10.000 kiside cilt kanseri proxy degeri {history_end} yilinda {_safe_float(row_h['cases_per10k_mid']):.4f}, {proj_end} yilinda {_safe_float(row_p['cases_per10k_mid']):.4f}.",
        f"- {proj_end} icin aralik: [{_safe_float(row_p['cases_per10k_low']):.4f}, {_safe_float(row_p['cases_per10k_high']):.4f}] / 10.000.",
        "",
        "## 2) Dogruluk durumu (acik)",
        f"- Iklim egilimi guveni: {_label(climate_conf)} ({climate_conf:.2f}/1.00)",
        f"- Cilt kanseri sayi guveni (proxy): {_label(skin_proxy_conf)} ({skin_proxy_conf:.2f}/1.00)",
        "- Not: Cilt kanseri satiri dogrudan klinik insidans degil, iklim-baski proxy senaryosudur.",
        "",
        "## 3) Kanit (sayisal)",
        f"- Etkili UV trend p-degeri: {p_uv:.4g}",
        f"- Cilt kanseri proxy trend p-degeri: {p_case:.4g}",
        f"- Etkili UV egimi: {slope_uv:+.4f} / dekad",
        f"- Cilt kanseri proxy egimi: {slope_case:+.4f} / dekad",
        f"- Yerel veri kapsami: sicaklik %{100*cov_temp:.1f}, nem %{100*cov_hum:.1f}, yagis %{100*cov_prec:.1f}",
        f"- Literatur uyumu: {meta.get('literature_consistency')}",
        "",
        "## 4) Ne dogrudan, ne dolayli?",
        "- Dogrudan: sicaklik, nem, yagis, gunes, bulutluluk yillik serileri.",
        "- Dolayli: cilt kanseri sayisi (UV etkisine dayali proxy senaryo).",
        "",
        "## Kaynaklar (kisa)",
        "- WHO UV Q&A, WHO UV fact sheet, IARC UV notu, WHO-ILO gunes maruziyeti, NASA POWER.",
        "",
    ]
    md.write_text("\n".join(md_lines), encoding="utf-8")

    # Simple visual panel
    years = annual["year"].to_numpy(dtype=int)
    eff = annual["effective_uv_kwh_m2_day"].to_numpy(dtype=float)
    cases = annual["cases_per10k_mid"].to_numpy(dtype=float)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(years, eff, color="#E67E22", linewidth=2.4, label="Etkili UV")
    ax2.plot(years, cases, color="#2E86C1", linewidth=2.4, label="Cilt proxy (/10k)")
    ax.axvline(history_end + 0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Yillik Gelisim (Sade)")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Etkili UV")
    ax2.set_ylabel("Proxy vaka /10k")
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = ["Iklim egilimi", "Cilt kanseri proxy"]
    vals = [climate_conf, skin_proxy_conf]
    cols = ["#27AE60" if v >= 0.75 else "#F39C12" if v >= 0.55 else "#C0392B" for v in vals]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=cols, height=0.55)
    ax.set_yticks(y_pos, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Guven skoru (0-1)")
    ax.set_title("Dogruluk Seviyesi")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.2f} ({_label(v)})", va="center", fontsize=10)

    fig.suptitle("Sade Ozet Panosu", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = fig_dir / "sade_ozet_pano.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    _copy_latest(md, root / "sade_ozet_tr_latest.md")
    _copy_latest(annual_csv, root / "sade_ozet_sayilar_latest.csv")
    _copy_latest(out_png, fig_dir / "sade_ozet_pano_latest.png")

    print(f"Wrote: {annual_csv}")
    print(f"Wrote: {md}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {root / 'sade_ozet_tr_latest.md'}")
    print(f"Wrote: {root / 'sade_ozet_sayilar_latest.csv'}")
    print(f"Wrote: {fig_dir / 'sade_ozet_pano_latest.png'}")


if __name__ == "__main__":
    main()

