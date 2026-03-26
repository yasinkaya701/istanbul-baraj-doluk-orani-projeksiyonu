#!/usr/bin/env python3
"""Build a presentation-ready PDF for skin-cancer climate evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build skin-cancer academic PDF report")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/yearly_skin_cancer_evidence"),
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("output/pdf/cilt_kanseri_akademik_rapor.pdf"),
    )
    return p.parse_args()


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _fmt(v: Any, digits: int = 4) -> str:
    x = _safe_float(v)
    if not np.isfinite(x):
        return "-"
    return f"{x:.{digits}f}"


def _fmt_pct(v: Any, digits: int = 2) -> str:
    x = _safe_float(v)
    if not np.isfinite(x):
        return "-"
    return f"{x:.{digits}f}%"


def _add_page_number(canvas, doc) -> None:  # noqa: ANN001
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def _img(path: Path, width_cm: float) -> Image:
    if not path.exists():
        raise SystemExit(f"Missing figure: {path}")
    img = Image(str(path))
    w = width_cm * cm
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = w
    img.drawHeight = w * ratio
    return img


def build_pdf(input_dir: Path, output_pdf: Path) -> None:
    annual_csv = input_dir / "annual_climate_and_skin_cancer_proxy.csv"
    checks_csv = input_dir / "literature_alignment_checks.csv"
    lit_csv = input_dir / "academic_literature_table.csv"
    meta_json = input_dir / "run_meta.json"
    fig_dir = input_dir / "figures"

    required = [annual_csv, checks_csv, lit_csv, meta_json]
    for p in required:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    annual = pd.read_csv(annual_csv).sort_values("year")
    checks = pd.read_csv(checks_csv)
    literature = pd.read_csv(lit_csv)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    assumptions = dict(meta.get("assumptions", {}))
    inputs = dict(meta.get("inputs", {}))

    hist = annual[annual["is_projected"] == False].copy()  # noqa: E712
    proj = annual[annual["is_projected"] == True].copy()  # noqa: E712
    if hist.empty:
        raise SystemExit("Historical rows are missing in annual table")

    y_hist_end = int(hist["year"].max())
    y_proj_end = int(proj["year"].max()) if not proj.empty else y_hist_end
    row_hist = annual.loc[annual["year"] == y_hist_end].iloc[0]
    row_proj = annual.loc[annual["year"] == y_proj_end].iloc[0]

    case_hist = _safe_float(row_hist["cases_per10k_mid"])
    case_proj = _safe_float(row_proj["cases_per10k_mid"])
    case_delta_pct = (case_proj / case_hist - 1.0) * 100.0 if case_hist > 0 else float("nan")
    uv_hist = _safe_float(row_hist["effective_uv_kwh_m2_day"])
    uv_proj = _safe_float(row_proj["effective_uv_kwh_m2_day"])
    uv_delta_pct = (uv_proj / uv_hist - 1.0) * 100.0 if uv_hist > 0 else float("nan")

    aligned = int((checks["alignment_label"].astype(str) == "uyumlu").sum())
    total_checks = int(len(checks))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Cilt Kanseri Iklim Etki Raporu",
        author="Hackhaton Pipeline",
    )

    styles = getSampleStyleSheet()
    s_title = ParagraphStyle(
        "title_custom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
    )
    s_h1 = ParagraphStyle(
        "h1_custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=6,
    )
    s_p = ParagraphStyle(
        "p_custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=4,
    )
    s_small = ParagraphStyle(
        "small_custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=12,
        textColor=colors.HexColor("#334155"),
        spaceAfter=3,
    )

    story: list[Any] = []
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M")

    story.append(Paragraph("Cilt Kanseri Iklim Etki Raporu", s_title))
    story.append(
        Paragraph(
            f"Tarihsel: {int(hist['year'].min())}-{y_hist_end} | Projeksiyon: {y_hist_end + 1}-{y_proj_end} | Uretim: {now_text}",
            s_small,
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    summary_data = [
        ["Gostergeler", "Deger"],
        [f"{y_hist_end} vaka/10.000 (merkez)", _fmt(case_hist, 4)],
        [f"{y_proj_end} vaka/10.000 (merkez)", _fmt(case_proj, 4)],
        [f"{y_hist_end}->{y_proj_end} vaka degisimi", _fmt_pct(case_delta_pct, 2)],
        [f"{y_hist_end}->{y_proj_end} etkili UV degisimi", _fmt_pct(uv_delta_pct, 2)],
        ["Literatur uyumu", f"{aligned}/{total_checks}"],
    ]
    t = Table(summary_data, colWidths=[10.3 * cm, 6.2 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)

    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Aciklama", s_h1))
    story.append(
        Paragraph(
            "Bu rapor, rasathane ve internet tabanli iklim suruculerini birlestirerek cilt kanseri risk-baski senaryosunu 10.000 kisi bazinda sunar. Klinik tani araci degildir; toplumsal risk yonu ve buyukluk sinifi icin karar destek raporudur.",
            s_p,
        )
    )
    story.append(
        Paragraph(
            "Gercekcilik ayarlari: son yillara odakli fit penceresi, damping, UV gecikmeli ortalama, kontrollu AR(1) gurultu, yillik artis/dusus guardrail.",
            s_p,
        )
    )

    realism_data = [
        ["Ayar", "Deger"],
        ["Fit penceresi (yil)", str(int(assumptions.get("projection_fit_window_years", 20)))],
        ["Damping", _fmt(assumptions.get("projection_damping"), 2)],
        ["UV gecikme (yil)", str(int(assumptions.get("uv_lag_years", 1)))],
        ["Gurultu aktif", "Evet" if bool(assumptions.get("projection_noise_enable", False)) else "Hayir"],
        ["Gurultu gucu", _fmt(assumptions.get("projection_noise_strength"), 2)],
        ["Gurultu AR(1)", _fmt(assumptions.get("projection_noise_ar1"), 2)],
        ["Yillik artis tavani", _fmt_pct(assumptions.get("projected_case_growth_cap_pct"), 2)],
        ["Yillik dusus tabani", _fmt_pct(assumptions.get("projected_case_growth_floor_pct"), 2)],
    ]
    tr = Table(realism_data, colWidths=[10.3 * cm, 6.2 * cm])
    tr.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ]
        )
    )
    story.append(tr)

    story.append(PageBreak())
    story.append(Paragraph("Grafikler", s_h1))
    story.append(Paragraph("1) Yillik UV ve vaka projeksiyon seridi", s_small))
    story.append(_img(fig_dir / "academic_projection_strip.png", width_cm=17.8))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("2) Yillik iklim suruculeri", s_small))
    story.append(_img(fig_dir / "yearly_climate_drivers.png", width_cm=17.8))

    story.append(PageBreak())
    story.append(Paragraph("Ek Grafikler", s_h1))
    story.append(Paragraph("3) Cilt kanseri proxy trendi (10.000 kisi)", s_small))
    story.append(_img(fig_dir / "skin_cancer_per10k_trend.png", width_cm=17.8))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("4) Istatistiksel kanit panosu", s_small))
    story.append(_img(fig_dir / "evidence_dashboard.png", width_cm=17.8))

    story.append(PageBreak())
    story.append(Paragraph("Kaynakca", s_h1))
    story.append(Paragraph("Veri Kaynaklari", s_small))

    data_items = [
        ("Rasathane/Otomatik Istasyon (CR800 Table1)", str(inputs.get("station_table1", "-"))),
        ("Yerel sicaklik tarihsel tablo", str(inputs.get("temp_local", "-"))),
        ("Yerel nem tarihsel tablo", str(inputs.get("humidity_local", "-"))),
        ("Yerel yagis tarihsel tablo", str(inputs.get("precip_local", "-"))),
        ("NASA POWER API", "https://power.larc.nasa.gov/api/temporal/monthly/point"),
    ]
    for label, val in data_items:
        story.append(Paragraph(f"- <b>{label}:</b> {val}", s_small))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Akademik ve Kurumsal Kaynaklar", s_small))
    for i, row in literature.iterrows():
        src = str(row.get("source_url", "-"))
        txt = str(row.get("quantitative_finding", "-"))
        eid = str(row.get("evidence_id", f"SRC-{i+1}"))
        story.append(Paragraph(f"{i+1}. <b>{eid}</b> - {txt} ({src})", s_small))

    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("Not: Bu rapor toplum duzeyi modelleme ve literatur uyum kontrolu sunar; klinik tani amacli degildir.", s_small))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)


def main() -> None:
    args = parse_args()
    build_pdf(input_dir=args.input_dir.resolve(), output_pdf=args.output_pdf.resolve())
    print(f"Wrote: {args.output_pdf.resolve()}")


if __name__ == "__main__":
    main()

