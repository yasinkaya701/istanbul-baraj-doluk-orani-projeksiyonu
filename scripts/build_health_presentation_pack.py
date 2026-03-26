#!/usr/bin/env python3
"""Build a compact presentation pack (6-slide markdown + 1-page PDF brief)."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build health presentation pack")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--fig-dir", type=Path, default=Path("output/health_impact/figures"))
    p.add_argument(
        "--model-summary-csv",
        type=Path,
        default=None,
        help="Optional model summary CSV path. If omitted, prefers model_comparison_summary_duzenlenmis_run.csv.",
    )
    p.add_argument("--date-label", type=str, default=str(date.today()))
    return p.parse_args()


def resolve_model_summary_path(root: Path, override_path: Path | None) -> Path:
    if override_path is not None:
        return override_path
    for p in [
        root / "model_comparison_summary_stable_calibrated.csv",
        root / "model_comparison_summary_duzenlenmis_run.csv",
    ]:
        if p.exists():
            return p
    return root / "model_comparison_summary.csv"


def load_inputs(root: Path, model_summary_csv: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv(resolve_model_summary_path(root, model_summary_csv))
    disease = pd.read_csv(root / "health_all_disease_summary.csv")
    return model, disease


def model_row(model: pd.DataFrame, name: str) -> pd.Series:
    needle = name.lower().strip()
    labels = model["model"].astype(str).str.lower()
    d = model[labels == needle].copy()
    if d.empty:
        d = model[labels.str.contains(needle, regex=False)].copy()
    if d.empty:
        raise ValueError(f"model row not found: {name}")
    return d.iloc[0]


def top_diseases(disease: pd.DataFrame, model_name: str, n: int = 5) -> pd.DataFrame:
    d = disease[disease["model"].str.lower() == model_name.lower()].copy()
    d = d.sort_values("direct_signal_score", ascending=False).head(n)
    return d


def _metric(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col in row.index and pd.notna(row[col]):
        try:
            return float(row[col])
        except Exception:
            return default
    return default


def _disease_model_metric(disease: pd.DataFrame, model_name: str, col: str, default: float = 0.0) -> float:
    if col not in disease.columns:
        return default
    d = disease[disease["model"].astype(str).str.lower().str.contains(model_name.lower(), regex=False)].copy()
    if d.empty:
        return default
    try:
        return float(d.iloc[0][col])
    except Exception:
        return default


def write_slide_markdown(
    out_path: Path,
    strong: pd.Series,
    quant: pd.Series,
    strong_top: pd.DataFrame,
    quant_top: pd.DataFrame,
    date_label: str,
) -> None:
    lines: list[str] = []
    lines.append("# 6-Slide Sunum - Iklim ve Saglik Riski")
    lines.append(f"Tarih: {date_label}")
    lines.append("")
    lines.append("## Slide 1 - Yonetici Mesaji")
    lines.append("- Isi kaynakli saglik riski iki senaryoda ayrisiyor: strong belirgin, quant sinirli.")
    lines.append("- Operasyon onceligi: isi uyum programi + erken uyari sistemi.")
    lines.append("")
    lines.append("## Slide 2 - Sayilarla Ozet")
    lines.append(
        f"- Strong gelecekte ort. RR: {float(strong['future_rr_mean']):.4f} (delta {float(strong['delta_rr_mean']):+.4f})"
    )
    lines.append(
        f"- Quant gelecekte ort. RR: {float(quant['future_rr_mean']):.4f} (delta {float(quant['delta_rr_mean']):+.4f})"
    )
    lines.append(
        f"- Esik asimi ay payi: strong {float(strong['future_threshold_exceed_share']):.1%}, quant {float(quant['future_threshold_exceed_share']):.1%}"
    )
    lines.append(
        f"- OOD ay payi: strong {float(strong['delta_ood_share']):.1%} artim, quant {float(quant['delta_ood_share']):.1%} artim"
    )
    lines.append("")
    lines.append("## Slide 3 - Hastalik Gruplari (Strong)")
    for _, r in strong_top.iterrows():
        lines.append(
            f"- {r['disease_group_tr']}: skor {float(r['direct_signal_score']):.2f} ({r['direct_signal_level']})"
        )
    lines.append("")
    lines.append("## Slide 4 - Hastalik Gruplari (Quant)")
    for _, r in quant_top.iterrows():
        lines.append(
            f"- {r['disease_group_tr']}: skor {float(r['direct_signal_score']):.2f} ({r['direct_signal_level']})"
        )
    lines.append("")
    lines.append("## Slide 5 - Etiyoloji Kapsam ve Sinirlar")
    lines.append("- Dogrudan olculebilir: isi ve nem etkisi.")
    lines.append("- Nitel izlenen: UV, PM2.5/ozon, polen, vektor, su-gida zinciri, afet kesintileri.")
    lines.append("- Not: Cilt kanseri icin UV ana etkendir; UV verisi olmadan sayisal artis verilmez.")
    lines.append("")
    lines.append("## Slide 6 - 90 Gunluk Aksiyon Plani")
    lines.append("- Gunluk saglik sonlanim verisi + DLNM kalibrasyonu.")
    lines.append("- UV/PM2.5/O3/polen katmanlarini modele ekleme.")
    lines.append("- Isi-esik bazli erken uyari paneli.")
    lines.append("- Kirmizi gruplar icin hedefli koruma protokolu.")
    lines.append("- Aylik model performans ve literatur uyum denetimi.")
    lines.append("")
    lines.append("## Kullanilacak Grafikler")
    lines.append("- output/health_impact/figures/fig01_model_overview.png")
    lines.append("- output/health_impact/figures/fig06_all_disease_direct_scores.png")
    lines.append("- output/health_impact/figures/fig07_all_disease_etiology_coverage.png")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_imread(path: Path):
    if path.exists():
        try:
            return plt.imread(path)
        except Exception:
            return None
    return None


def write_onepage_pdf(
    out_pdf: Path,
    strong: pd.Series,
    quant: pd.Series,
    disease: pd.DataFrame,
    fig_dir: Path,
    date_label: str,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150)  # A4 landscape
    fig.patch.set_facecolor("white")

    ax_text = fig.add_axes([0.04, 0.05, 0.56, 0.90])
    ax_text.axis("off")

    title = "Yonetici Brifi: Iklim Degisimi ve Saglik Riski"
    subtitle = f"Tarih: {date_label} | Donem: 1991-2020 baz vs 2026-2035 gelecek"
    strong_hum = _metric(strong, "realism_mean_abs_humidity_adjustment_pct", None)
    quant_hum = _metric(quant, "realism_mean_abs_humidity_adjustment_pct", None)
    if strong_hum is None:
        strong_hum = _disease_model_metric(disease, "strong", "delta_mean_humidity_pct", 0.0)
    if quant_hum is None:
        quant_hum = _disease_model_metric(disease, "quant", "delta_mean_humidity_pct", 0.0)

    lines = [
        "Ana mesaj:",
        "Strong senaryoda isi kaynakli saglik baskisi belirgin; quant senaryoda artis sinirli.",
        "",
        "Kilit metrikler:",
        f"- Strong RR: {_metric(strong, 'future_rr_mean'):.4f} (delta {_metric(strong, 'delta_rr_mean'):+.4f})",
        f"- Quant RR: {_metric(quant, 'future_rr_mean'):.4f} (delta {_metric(quant, 'delta_rr_mean'):+.4f})",
        f"- Esik asimi ay payi: strong {_metric(strong, 'future_threshold_exceed_share'):.1%}, quant {_metric(quant, 'future_threshold_exceed_share'):.1%}",
        f"- OOD ay payi artis: strong {_metric(strong, 'delta_ood_share'):.1%}, quant {_metric(quant, 'delta_ood_share'):.1%}",
        f"- Nem degisimi (delta RH puan): strong {strong_hum:.2f}, quant {quant_hum:.2f}",
        "",
        "Yorum:",
        "- Model sonucuna gore genel kurulasma sinyali yok; bagil nem artisi goruluyor.",
        "- Cilt kanseri artisini sayisal vermek icin UV katmani gerekir.",
        "",
        "90 gun aksiyon:",
        "1) Gunluk saglik verisi + DLNM",
        "2) UV ve hava kirliligi katmani",
        "3) Erken uyari paneli",
        "4) Kirmizi grup koruma protokolu",
        "5) Aylik performans ve literatur denetimi",
    ]

    ax_text.text(0.0, 0.98, title, fontsize=17, fontweight="bold", va="top")
    ax_text.text(0.0, 0.93, subtitle, fontsize=10, color="#444444", va="top")
    ax_text.text(0.0, 0.88, "\n".join(lines), fontsize=11, va="top", linespacing=1.35)

    # Right side charts
    img1 = _safe_imread(fig_dir / "fig01_model_overview.png")
    img2 = _safe_imread(fig_dir / "fig06_all_disease_direct_scores.png")
    img3 = _safe_imread(fig_dir / "fig07_all_disease_etiology_coverage.png")

    if img1 is not None:
        ax1 = fig.add_axes([0.62, 0.68, 0.35, 0.25])
        ax1.imshow(img1)
        ax1.axis("off")
        ax1.set_title("Model Ozeti", fontsize=9)
    if img2 is not None:
        ax2 = fig.add_axes([0.62, 0.36, 0.35, 0.25])
        ax2.imshow(img2)
        ax2.axis("off")
        ax2.set_title("Hastalik Risk Skorlari", fontsize=9)
    if img3 is not None:
        ax3 = fig.add_axes([0.62, 0.04, 0.35, 0.25])
        ax3.imshow(img3)
        ax3.axis("off")
        ax3.set_title("Etiyoloji Kapsama", fontsize=9)

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.root_dir.mkdir(parents=True, exist_ok=True)

    model, disease = load_inputs(args.root_dir, args.model_summary_csv)
    strong = model_row(model, "strong")
    quant = model_row(model, "quant")
    strong_top = top_diseases(disease, "strong", 5)
    quant_top = top_diseases(disease, "quant", 5)

    slide_md = args.root_dir / f"sunum_6_slayt_{args.date_label}.md"
    onepage_pdf = args.root_dir / f"yonetici_brif_tek_sayfa_{args.date_label}.pdf"

    write_slide_markdown(slide_md, strong, quant, strong_top, quant_top, args.date_label)
    write_onepage_pdf(onepage_pdf, strong, quant, disease, args.fig_dir, args.date_label)

    print(f"Wrote: {slide_md}")
    print(f"Wrote: {onepage_pdf}")


if __name__ == "__main__":
    main()
