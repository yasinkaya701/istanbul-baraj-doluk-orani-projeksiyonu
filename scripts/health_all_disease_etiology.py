#!/usr/bin/env python3
"""Build an all-disease x etiology climate-health risk matrix.

Goal:
- Translate model outputs into human-readable disease-group risk signals.
- Separate directly quantifiable pathways (heat/humidity) from qualitative ones
  requiring extra data (UV, PM2.5, pollen, flood exposure, etc.).

Important:
- This is a population-level screening summary, not a clinical diagnosis tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="All disease x etiology climate-health matrix")
    p.add_argument(
        "--strong-summary",
        type=Path,
        default=Path("output/health_impact/strong/health_impact_summary.json"),
        help="Path to strong model health_impact_summary.json",
    )
    p.add_argument(
        "--quant-summary",
        type=Path,
        default=Path("output/health_impact/quant/health_impact_summary.json"),
        help="Path to quant model health_impact_summary.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/health_impact"),
        help="Output directory for matrix and report",
    )
    return p.parse_args()


def load_summary(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "baseline": d.get("baseline", {}),
        "future": d.get("future", {}),
        "delta": d.get("delta", {}),
        "inputs": d.get("inputs", {}),
        "realism": d.get("realism", {}),
    }


def safe_float(x: object, fallback: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return fallback
        return v
    except Exception:
        return fallback


def classify_score(score: float) -> str:
    if score >= 1.5:
        return "yüksek"
    if score >= 0.6:
        return "orta"
    if score >= 0.15:
        return "düşük"
    return "çok düşük"


def climate_pressure_level(heat_signal: float) -> str:
    if heat_signal >= 1.5:
        return "yüksek (nitel)"
    if heat_signal >= 0.6:
        return "orta (nitel)"
    if heat_signal >= 0.15:
        return "düşük (nitel)"
    return "çok düşük (nitel)"


def build_signals(summary: dict) -> dict:
    d = summary["delta"]
    delta_hi = max(0.0, safe_float(d.get("mean_heat_index_c_delta"), 0.0))
    delta_rr = max(0.0, safe_float(d.get("mean_proxy_relative_risk_delta"), 0.0))
    delta_af = max(0.0, safe_float(d.get("mean_attributable_fraction_delta"), 0.0))
    delta_thr = max(0.0, safe_float(d.get("threshold_exceed_month_share_delta"), 0.0))
    delta_hum = safe_float(d.get("mean_humidity_pct_delta"), 0.0)

    # Composite signals:
    # - heat_signal: uses multiple heat-related indicators for robustness.
    # - wet/dry signals: humidity direction split.
    heat_signal = (delta_hi / 5.0) + (delta_thr * 2.0) + (delta_rr * 10.0) + (delta_af * 10.0)
    wet_signal = max(0.0, delta_hum / 5.0)
    dry_signal = max(0.0, -delta_hum / 5.0)

    return {
        "delta_hi": delta_hi,
        "delta_rr": delta_rr,
        "delta_af": delta_af,
        "delta_thr": delta_thr,
        "delta_hum": delta_hum,
        "heat_signal": heat_signal,
        "wet_signal": wet_signal,
        "dry_signal": dry_signal,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_inputs = {
        "strong": load_summary(args.strong_summary),
        "quant": load_summary(args.quant_summary),
    }

    etiologies = {
        "heat_extreme": {"title": "Aşırı sıcak / ısı dalgası", "quantifiable": True},
        "humidity_change": {"title": "Nem değişimi (kuru-nemli uçlar)", "quantifiable": True},
        "uv_radiation": {"title": "UV maruziyeti", "quantifiable": False},
        "air_pollution_smoke": {"title": "Hava kirliliği / duman (PM2.5, ozon)", "quantifiable": False},
        "allergens_pollen": {"title": "Polen / alerjen yükü", "quantifiable": False},
        "vector_ecology": {"title": "Vektör ekolojisi (sivrisinek/kene)", "quantifiable": False},
        "water_food_sanitation": {"title": "Su-gıda hijyen zinciri", "quantifiable": False},
        "extreme_events_disruption": {"title": "Sel/fırtına/afet kaynaklı sağlık kesintisi", "quantifiable": False},
        "occupational_exposure": {"title": "Mesleki maruziyet (dış ortam sıcaklık + UV)", "quantifiable": False},
    }

    diseases = [
        {
            "id": "all_cause_heat_mortality_morbidity",
            "name_tr": "Genel ısıya bağlı ölüm-hastalık yükü",
            "heat_w": 1.00,
            "wet_w": 0.20,
            "dry_w": 0.10,
            "etiologies": list(etiologies.keys()),
            "primary_refs": ["PMID:38986701", "PMID:26003380", "WHO:heat-health-2024"],
        },
        {
            "id": "cardiovascular",
            "name_tr": "Kardiyovasküler hastalıklar",
            "heat_w": 0.95,
            "wet_w": 0.15,
            "dry_w": 0.10,
            "etiologies": ["heat_extreme", "air_pollution_smoke", "extreme_events_disruption", "occupational_exposure"],
            "primary_refs": ["PMID:31376629", "PMID:39649149", "WHO:heat-health-2024"],
        },
        {
            "id": "respiratory",
            "name_tr": "Solunum hastalıkları (astım/KOAH vb.)",
            "heat_w": 0.75,
            "wet_w": 0.20,
            "dry_w": 0.20,
            "etiologies": ["heat_extreme", "humidity_change", "air_pollution_smoke", "allergens_pollen"],
            "primary_refs": ["PMID:31376629", "PMID:39649149", "PMID:29331087", "CDC:pollen-health"],
        },
        {
            "id": "kidney",
            "name_tr": "Böbrek hastalıkları (AKI/CKD riski)",
            "heat_w": 0.90,
            "wet_w": 0.05,
            "dry_w": 0.35,
            "etiologies": ["heat_extreme", "humidity_change", "air_pollution_smoke", "occupational_exposure"],
            "primary_refs": ["PMID:34467930", "PMID:31869688", "WHO:heat-health-2024"],
        },
        {
            "id": "mental_health",
            "name_tr": "Ruh sağlığı (anksiyete, acil başvuru, stres)",
            "heat_w": 0.60,
            "wet_w": 0.10,
            "dry_w": 0.10,
            "etiologies": ["heat_extreme", "air_pollution_smoke", "extreme_events_disruption"],
            "primary_refs": ["PMID:34144243", "PMID:33799230", "PMID:37437999"],
        },
        {
            "id": "maternal_neonatal",
            "name_tr": "Anne-bebek sağlığı (erken doğum vb.)",
            "heat_w": 0.80,
            "wet_w": 0.10,
            "dry_w": 0.05,
            "etiologies": ["heat_extreme", "humidity_change", "extreme_events_disruption"],
            "primary_refs": ["PMID:33148618", "WHO:climate-health-2023"],
        },
        {
            "id": "vector_borne_infectious",
            "name_tr": "Vektörle bulaşan enfeksiyonlar (dengue vb.)",
            "heat_w": 0.45,
            "wet_w": 0.70,
            "dry_w": 0.00,
            "etiologies": ["vector_ecology", "humidity_change", "heat_extreme", "water_food_sanitation"],
            "primary_refs": ["PMID:37088034", "WHO:climate-health-2023"],
        },
        {
            "id": "water_food_borne_infectious",
            "name_tr": "Su-gıda kaynaklı enfeksiyonlar (ishal vb.)",
            "heat_w": 0.35,
            "wet_w": 0.65,
            "dry_w": 0.10,
            "etiologies": ["water_food_sanitation", "humidity_change", "extreme_events_disruption", "heat_extreme"],
            "primary_refs": ["PMID:26567313", "PMID:38962364", "PMID:40047003", "WHO:climate-health-2023"],
        },
        {
            "id": "allergic_asthma",
            "name_tr": "Alerji ve astım alevlenmeleri",
            "heat_w": 0.25,
            "wet_w": 0.20,
            "dry_w": 0.20,
            "etiologies": ["allergens_pollen", "air_pollution_smoke", "humidity_change", "heat_extreme"],
            "primary_refs": ["PMID:31106285", "CDC:allergens-pollen", "WHO:air-pollution-pollen-2025"],
        },
        {
            "id": "skin_cancer",
            "name_tr": "Cilt kanseri (özellikle UV ilişkili)",
            "heat_w": 0.05,
            "wet_w": 0.00,
            "dry_w": 0.00,
            "etiologies": ["uv_radiation", "occupational_exposure"],
            "primary_refs": ["WHO:uv-radiation-2022", "PMID:40421619", "WHO-ILO:occupational-uv-2023"],
        },
        {
            "id": "injury_accident",
            "name_tr": "Yaralanmalar ve iş kazaları",
            "heat_w": 0.55,
            "wet_w": 0.10,
            "dry_w": 0.05,
            "etiologies": ["heat_extreme", "extreme_events_disruption", "occupational_exposure"],
            "primary_refs": ["WHO:heat-health-2024", "WHO:climate-health-2023"],
        },
    ]

    ref_map = {
        "PMID:38986701": "https://pubmed.ncbi.nlm.nih.gov/38986701/",
        "PMID:26003380": "https://pubmed.ncbi.nlm.nih.gov/26003380/",
        "PMID:31376629": "https://pubmed.ncbi.nlm.nih.gov/31376629/",
        "PMID:39649149": "https://pubmed.ncbi.nlm.nih.gov/39649149/",
        "PMID:34467930": "https://pubmed.ncbi.nlm.nih.gov/34467930/",
        "PMID:31869688": "https://pubmed.ncbi.nlm.nih.gov/31869688/",
        "PMID:33799230": "https://pubmed.ncbi.nlm.nih.gov/33799230/",
        "PMID:37437999": "https://pubmed.ncbi.nlm.nih.gov/37437999/",
        "PMID:34144243": "https://pubmed.ncbi.nlm.nih.gov/34144243/",
        "PMID:33148618": "https://pubmed.ncbi.nlm.nih.gov/33148618/",
        "PMID:37088034": "https://pubmed.ncbi.nlm.nih.gov/37088034/",
        "PMID:26567313": "https://pubmed.ncbi.nlm.nih.gov/26567313/",
        "PMID:38962364": "https://pubmed.ncbi.nlm.nih.gov/38962364/",
        "PMID:40047003": "https://pubmed.ncbi.nlm.nih.gov/40047003/",
        "PMID:31106285": "https://pubmed.ncbi.nlm.nih.gov/31106285/",
        "PMID:40421619": "https://pubmed.ncbi.nlm.nih.gov/40421619/",
        "WHO:heat-health-2024": "https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health",
        "WHO:climate-health-2023": "https://www.who.int/news-room/fact-sheets/detail/climate-change-and-health",
        "WHO:uv-radiation-2022": "https://www.who.int/news-room/fact-sheets/detail/ultraviolet-radiation",
        "WHO:air-pollution-pollen-2025": "https://www.who.int/publications/i/item/B09412",
        "WHO-ILO:occupational-uv-2023": "https://www.who.int/news/item/08-11-2023-working-under-the-sun-causes-1-in-3-deaths-from-non-melanoma-skin-cancer--say-who-and-ilo",
        "CDC:pollen-health": "https://www.cdc.gov/climate-health/php/effects/pollen-health.html",
        "CDC:allergens-pollen": "https://www.cdc.gov/climate-health/php/effects/allergens-and-pollen.html",
    }

    rows: list[dict] = []
    disease_summary_rows: list[dict] = []

    for model_name, summary in model_inputs.items():
        s = build_signals(summary)
        pressure = climate_pressure_level(s["heat_signal"])

        for d in diseases:
            direct_score = (d["heat_w"] * s["heat_signal"]) + (d["wet_w"] * s["wet_signal"]) + (d["dry_w"] * s["dry_signal"])
            disease_summary_rows.append(
                {
                    "model": model_name,
                    "disease_group_id": d["id"],
                    "disease_group_tr": d["name_tr"],
                    "direct_signal_score": float(direct_score),
                    "direct_signal_level": classify_score(float(direct_score)),
                    "heat_signal": float(s["heat_signal"]),
                    "wet_signal": float(s["wet_signal"]),
                    "dry_signal": float(s["dry_signal"]),
                    "delta_mean_heat_index_c": float(s["delta_hi"]),
                    "delta_mean_humidity_pct": float(s["delta_hum"]),
                    "delta_threshold_exceed_share": float(s["delta_thr"]),
                    "delta_rr": float(s["delta_rr"]),
                    "delta_af": float(s["delta_af"]),
                    "primary_references": "; ".join(d["primary_refs"]),
                }
            )

            for etiology_id in d["etiologies"]:
                et = etiologies[etiology_id]
                is_direct = bool(et["quantifiable"])

                if etiology_id == "heat_extreme":
                    etiology_score = d["heat_w"] * s["heat_signal"]
                elif etiology_id == "humidity_change":
                    etiology_score = (d["wet_w"] * s["wet_signal"]) + (d["dry_w"] * s["dry_signal"])
                else:
                    etiology_score = np.nan

                if is_direct:
                    level = classify_score(float(etiology_score))
                    interpretation = (
                        "Bu etiyoloji mevcut model verisiyle (sıcaklık/nem) doğrudan hesaplandı."
                    )
                else:
                    level = pressure
                    interpretation = (
                        "Bu etiyoloji için modelde doğrudan değişken yok; nitel risk yorumu literatür ve genel iklim baskısına göre verildi."
                    )

                rows.append(
                    {
                        "model": model_name,
                        "disease_group_id": d["id"],
                        "disease_group_tr": d["name_tr"],
                        "etiology_id": etiology_id,
                        "etiology_tr": et["title"],
                        "quantifiable_with_current_data": is_direct,
                        "signal_score": None if np.isnan(etiology_score) else float(etiology_score),
                        "signal_level": level,
                        "interpretation_tr": interpretation,
                        "primary_references": "; ".join(d["primary_refs"]),
                    }
                )

    matrix = pd.DataFrame(rows)
    disease_summary = pd.DataFrame(disease_summary_rows)

    matrix_csv = args.output_dir / "health_all_disease_etiology_matrix.csv"
    summary_csv = args.output_dir / "health_all_disease_summary.csv"
    report_md = args.output_dir / "health_all_disease_report.md"
    refs_md = args.output_dir / "health_all_disease_references.md"

    matrix.to_csv(matrix_csv, index=False)
    disease_summary.to_csv(summary_csv, index=False)

    strong_top = (
        disease_summary[disease_summary["model"] == "strong"]
        .sort_values("direct_signal_score", ascending=False)
        .head(8)[["disease_group_tr", "direct_signal_score", "direct_signal_level"]]
    )
    quant_top = (
        disease_summary[disease_summary["model"] == "quant"]
        .sort_values("direct_signal_score", ascending=False)
        .head(8)[["disease_group_tr", "direct_signal_score", "direct_signal_level"]]
    )

    direct_share = float(matrix["quantifiable_with_current_data"].mean())
    lines = [
        "# Tum Hastaliklar x Tum Etiyolojiler: Iklim-Saglik Etki Ozeti",
        "",
        "Bu cikti klinik tanı aracı degildir; nufus duzeyi iklim-risk taramasidir.",
        "",
        "## Kapsam",
        "",
        f"- Hastalik grubu sayisi: {disease_summary['disease_group_id'].nunique()}",
        f"- Etiyoloji sayisi: {matrix['etiology_id'].nunique()}",
        f"- Toplam hucre (model x hastalik x etiyoloji): {len(matrix)}",
        f"- Mevcut veriyle dogrudan hesaplanabilen hucre orani: {direct_share:.1%}",
        "- Dogrudan hesaplanan etiyolojiler: asiri sicak / nem degisimi",
        "- Dogrudan hesaplanamayan ama nitel degerlendirilen etiyolojiler: UV, PM2.5/ozon, polen, vektor, su-gida zinciri, afet kesintileri, mesleki maruziyet",
        "",
        "## Strong modelde en yuksek dogrudan risk sinyali",
        "",
    ]
    for _, r in strong_top.iterrows():
        lines.append(f"- {r['disease_group_tr']}: skor={r['direct_signal_score']:.3f}, duzey={r['direct_signal_level']}")

    lines += ["", "## Quant modelde en yuksek dogrudan risk sinyali", ""]
    for _, r in quant_top.iterrows():
        lines.append(f"- {r['disease_group_tr']}: skor={r['direct_signal_score']:.3f}, duzey={r['direct_signal_level']}")

    lines += [
        "",
        "## Cilt kanseri notu",
        "",
        "- Cilt kanseri icin ana etiyoloji UV maruziyetidir.",
        "- Mevcut modelde UV girdisi olmadigi icin cilt kanseri artisini sayisal olarak tahmin etmiyoruz; nitel risk baskisi olarak raporluyoruz.",
        "",
        "## Kuruluk notu",
        "",
        "- Bu projeksiyonda bagil nem iki modelde de baz doneme gore artiyor; genel kurulasma sinyali yok.",
        "",
        "## Sinirlar",
        "",
        "- Tum hastaliklarin bireysel insidansini bu veriyle dogrudan tahmin etmek mumkun degildir.",
        "- Sayisal tahmin icin hastalik sonlanim verisi + UV/PM2.5/polen/hidrometeorolojik maruziyet veri katmanlari gerekir.",
        "",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ref_lines = [
        "# Birincil Kaynaklar",
        "",
        "Asagidaki kaynaklar hastalik-etiyoloji baglantilarinin bilimsel dayanaklari icin kullanildi.",
        "",
    ]
    for key in sorted(ref_map):
        ref_lines.append(f"- {key}: {ref_map[key]}")
    refs_md.write_text("\n".join(ref_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {matrix_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {report_md}")
    print(f"Wrote: {refs_md}")


if __name__ == "__main__":
    main()
