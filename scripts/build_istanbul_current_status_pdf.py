#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path("/Users/yasinkaya/Hackhaton")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _table_from_df(df: pd.DataFrame, max_rows: int = 10) -> list[list[str]]:
    if df.empty:
        return [["No data"]]
    head = df.head(max_rows)
    cols = list(head.columns)
    rows = [cols]
    for _, row in head.iterrows():
        rows.append([str(row.get(c, "")) for c in cols])
    return rows


def build_pdf(out_path: Path, report_date: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    s_title = styles["Title"]
    s_h1 = styles["Heading2"]
    s_body = styles["BodyText"]
    s_small = ParagraphStyle(
        "Small",
        parent=s_body,
        fontSize=9,
        leading=12,
    )

    # Paths
    final_pkg = ROOT / "output" / "final_delivery" / "istanbul_baraj_final_paket"
    extra_params_dir = ROOT / "output" / "istanbul_models_extra_params_2026_03_12"
    mape5_dir = ROOT / "output" / "istanbul_mape5_push_and_longhorizon_2026_03_12"
    wb_retry = final_pkg / "water_balance_retry_compare_round5.csv"

    holdout_summary = extra_params_dir / "extra_param_models_holdout_summary_2015_train_2020_test.csv"
    endpoints_2040 = extra_params_dir / "extra_param_models_2040_endpoints.csv"
    direct_search = mape5_dir / "short_holdout_direct_search_summary.csv"

    # Read tables
    df_holdout = _safe_read_csv(holdout_summary)
    df_end = _safe_read_csv(endpoints_2040)
    df_direct = _safe_read_csv(direct_search)
    df_wb = _safe_read_csv(wb_retry)

    # Pick key metrics if present
    top_models_text = ""
    if not df_holdout.empty and "model" in df_holdout.columns:
        mape_col = "mape_pct" if "mape_pct" in df_holdout.columns else "mape"
        rmse_col = "rmse_pp" if "rmse_pp" in df_holdout.columns else "rmse"
        pearson_col = "pearson_corr_pct" if "pearson_corr_pct" in df_holdout.columns else "pearson"
        df_sorted = df_holdout.sort_values(mape_col, ascending=True).head(5)
        lines = [
            f"- {r['model']}: MAPE {r.get(mape_col, 'na'):.2f}%, RMSE {r.get(rmse_col, 'na'):.2f}, Pearson {r.get(pearson_col, 'na'):.2f}%"
            for _, r in df_sorted.iterrows()
        ]
        top_models_text = "<br/>".join(lines)
    else:
        top_models_text = "- Holdout ozet tablosu bulunamadi."

    story: list = []
    story.append(Paragraph(f"Istanbul Baraj Projesi - Guncel Durum ve Cikti Ozeti ({report_date})", s_title))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Ozet", s_h1))
    story.append(Paragraph(
        "Bu rapor, mevcut model gelistirme durumunu, son test sonuclarini, "
        "uretmis oldugumuz ana ciktilari ve guncel deneyleri tek belgede toplar. "
        "Egitim-test kurgu su sekildedir: 2015 sonuna kadar egitim, 2016-2020 holdout test.",
        s_body,
    ))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Yeni odak: iklim ciktilarini (sicaklik, yagis, nem) netlestirip ET0 (FAO Penman-Monteith) "
        "ve basit su dengesi (giren-cikan su) ile proje omurgasini kurmak. "
        "Insan davranisi ve isletme kararlari bu turda modelin disinda tutuluyor.",
        s_body,
    ))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("En Iyi Modeller (Holdout 2016-2020)", s_h1))
    story.append(Paragraph(top_models_text, s_body))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Holdout Ozet Tablosu (ilk 10 satir)", s_h1))
    t1 = Table(_table_from_df(df_holdout, max_rows=10), hAlign="LEFT")
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(t1)
    story.append(Spacer(1, 0.2 * cm))

    if not df_end.empty:
        story.append(Paragraph("2040 Sonu Baz Senaryo Uc Degerleri", s_h1))
        t2 = Table(_table_from_df(df_end, max_rows=10), hAlign="LEFT")
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.2 * cm))

    if not df_direct.empty:
        story.append(Paragraph("MAPE 5 Hedefi Icın Kisa Holdout Direkt Arama", s_h1))
        story.append(Paragraph(
            "Bu arama, kisa vade one-step dogruluk icin agresif blend aramasidir. "
            "Uzun vadeli projeksiyon omurgasi degildir.",
            s_body,
        ))
        t3 = Table(_table_from_df(df_direct, max_rows=10), hAlign="LEFT")
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(t3)
        story.append(Spacer(1, 0.2 * cm))

    if not df_wb.empty:
        story.append(Paragraph("Water-Balance Denemeleri (Round 5)", s_h1))
        t4 = Table(_table_from_df(df_wb, max_rows=10), hAlign="LEFT")
        t4.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(t4)
        story.append(Spacer(1, 0.2 * cm))

    story.append(PageBreak())

    story.append(Paragraph("Ana Cikti Klasorleri", s_h1))
    story.append(Paragraph(
        "- output/final_delivery/istanbul_baraj_final_paket\n"
        "- output/istanbul_models_extra_params_2026_03_12\n"
        "- output/istanbul_mape5_push_and_longhorizon_2026_03_12\n"
        "- output/istanbul_hybrid_physics_sourceaware_ensemble_2040\n"
        "- output/istanbul_preferred_projection_2040",
        s_body,
    ))
    story.append(Spacer(1, 0.2 * cm))

    # Include key visuals if present
    visuals = [
        extra_params_dir / "extra_param_models_holdout.png",
        extra_params_dir / "extra_param_models_future.png",
        extra_params_dir / "extra_param_improvement_vs_previous.png",
    ]
    for img_path in visuals:
        if img_path.exists():
            story.append(Paragraph(f"Gorsel: {img_path.name}", s_small))
            story.append(Image(str(img_path), width=16*cm, height=9*cm))
            story.append(Spacer(1, 0.3 * cm))

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    doc.build(story)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/pdf/istanbul_baraj_guncel_durum_ozet_raporu_2026_03_18.pdf")
    p.add_argument("--date", default="2026-03-18")
    args = p.parse_args()
    out_path = ROOT / args.out
    build_pdf(out_path, args.date)
    print(out_path)


if __name__ == "__main__":
    main()
