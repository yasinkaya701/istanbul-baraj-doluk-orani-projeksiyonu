#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path("/Users/yasinkaya/Hackhaton")


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _table_from_df(df: pd.DataFrame, max_rows: int = 12) -> list[list[str]]:
    if df.empty:
        return [["No data"]]
    head = df.head(max_rows)
    cols = list(head.columns)
    rows = [cols]
    for _, row in head.iterrows():
        rows.append([str(row.get(c, "")) for c in cols])
    return rows


def _add_table(story, df: pd.DataFrame, max_rows: int = 12):
    t = Table(_table_from_df(df, max_rows=max_rows), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)


def build_pdf(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    s_title = styles["Title"]
    s_h1 = styles["Heading2"]
    s_h2 = styles["Heading3"]
    s_body = styles["BodyText"]
    s_small = ParagraphStyle(
        "Small",
        parent=s_body,
        fontSize=9,
        leading=12,
    )

    final_pkg = ROOT / "output" / "final_delivery" / "istanbul_baraj_final_paket"
    extra_params_dir = ROOT / "output" / "istanbul_models_extra_params_2026_03_12"
    mape5_dir = ROOT / "output" / "istanbul_mape5_push_and_longhorizon_2026_03_12"
    aggressive_dir = ROOT / "output" / "istanbul_aggressive_model_search_99_2026_03_12"
    newdata_store = ROOT / "output" / "newdata_feature_store"

    df_holdout = _safe_read_csv(extra_params_dir / "extra_param_models_holdout_summary_2015_train_2020_test.csv")
    df_end = _safe_read_csv(extra_params_dir / "extra_param_models_2040_endpoints.csv")
    df_improve = _safe_read_csv(extra_params_dir / "extra_param_improvement_vs_previous.csv")
    df_direct = _safe_read_csv(mape5_dir / "short_holdout_direct_search_summary.csv")
    df_aggr = _safe_read_csv(aggressive_dir / "aggressive_direct_level_search_summary.csv")
    df_wb_retry = _safe_read_csv(final_pkg / "water_balance_retry_compare_round5.csv")
    df_round3 = _safe_read_csv(final_pkg / "model_gelistirme_round3_ozet.csv")
    df_round4 = _safe_read_csv(final_pkg / "model_gelistirme_round4_ozet.csv")

    summary_newdata = _safe_read_json(newdata_store / "summary.json")
    summary_reanalysis = _safe_read_json(newdata_store / "kandilli_openmeteo_reanalysis_summary.json")
    summary_supply = _safe_read_json(newdata_store / "official_monthly_supply_context_summary.json")
    summary_ops = _safe_read_json(newdata_store / "monthly_operational_proxies_summary.json")

    story: list = []
    story.append(Paragraph("Istanbul Baraj Projesi - Guncel Durum ve Detayli Cikti Raporu", s_title))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Kapsam ve Amac", s_h1))
    story.append(Paragraph(
        "Bu rapor, Istanbul baraj doluluk projesinin guncel durumunu, veri katmanlarini, "
        "model gelistirme turlarini, test sonuclarini, projeksiyon ciktilarini ve acik veri bosluklarini "
        "detayli sekilde ozetler.",
        s_body,
    ))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Veri Katmanlari (Secili Ozetler)", s_h1))
    if summary_newdata:
        story.append(Paragraph(f"- newdata_feature_store: {summary_newdata.get('core_start', 'na')} -> {summary_newdata.get('core_end', 'na')} ({summary_newdata.get('core_rows', 'na')} ay)", s_body))
    if summary_supply:
        story.append(Paragraph(f"- Resmi sehir suyu: {summary_supply.get('official_city_supply_window', {}).get('start', 'na')} -> {summary_supply.get('official_city_supply_window', {}).get('end', 'na')}", s_body))
    if summary_reanalysis:
        story.append(Paragraph(f"- Reanalysis proxy: {summary_reanalysis.get('request_window', {}).get('start', 'na')} -> {summary_reanalysis.get('request_window', {}).get('end', 'na')}", s_body))
    if summary_ops:
        story.append(Paragraph(f"- Aylik operasyon proxy (transfer/NRW/geri kazanım): {summary_ops.get('window', {}).get('start', 'na')} -> {summary_ops.get('window', {}).get('end', 'na')}", s_body))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Model Gelistirme Turlari", s_h1))
    story.append(Paragraph("Round 3 Ozet", s_h2))
    if not df_round3.empty:
        _add_table(story, df_round3, max_rows=12)
    else:
        story.append(Paragraph("Round 3 ozet tablosu bulunamadi.", s_small))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Round 4 Ozet", s_h2))
    if not df_round4.empty:
        _add_table(story, df_round4, max_rows=12)
    else:
        story.append(Paragraph("Round 4 ozet tablosu bulunamadi.", s_small))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Holdout Test Sonuclari (2016-2020)", s_h1))
    if not df_holdout.empty:
        _add_table(story, df_holdout, max_rows=10)
    else:
        story.append(Paragraph("Holdout ozet tablosu bulunamadi.", s_small))
    story.append(Spacer(1, 0.2 * cm))

    if not df_end.empty:
        story.append(Paragraph("2040 Sonu Baz Senaryo Uc Degerleri", s_h1))
        _add_table(story, df_end, max_rows=10)
        story.append(Spacer(1, 0.2 * cm))

    story.append(PageBreak())

    story.append(Paragraph("MAPE 5 ve Agresif Aramalar", s_h1))
    if not df_direct.empty:
        story.append(Paragraph("Kisa vadeli direct arama (MAPE 5 hedefi) ozeti", s_h2))
        _add_table(story, df_direct, max_rows=10)
        story.append(Spacer(1, 0.2 * cm))
    if not df_aggr.empty:
        story.append(Paragraph("Agresif direct-level arama ozeti", s_h2))
        _add_table(story, df_aggr, max_rows=10)
        story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Water-Balance Retry (Round 5)", s_h1))
    if not df_wb_retry.empty:
        _add_table(story, df_wb_retry, max_rows=12)
    else:
        story.append(Paragraph("Water-balance retry tablosu bulunamadi.", s_small))
    story.append(Spacer(1, 0.2 * cm))

    if not df_improve.empty:
        story.append(Paragraph("Model Ilerleme Ozeti", s_h1))
        _add_table(story, df_improve, max_rows=12)
        story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Ana Cikti Klasorleri", s_h1))
    story.append(Paragraph(
        "- output/final_delivery/istanbul_baraj_final_paket\n"
        "- output/istanbul_models_extra_params_2026_03_12\n"
        "- output/istanbul_mape5_push_and_longhorizon_2026_03_12\n"
        "- output/istanbul_aggressive_model_search_99_2026_03_12\n"
        "- output/istanbul_preferred_projection_2040\n"
        "- output/istanbul_preferred_operational_risk_2030",
        s_body,
    ))

    story.append(PageBreak())

    story.append(Paragraph("Gorseller", s_h1))
    visuals = [
        extra_params_dir / "extra_param_models_holdout.png",
        extra_params_dir / "extra_param_models_future.png",
        extra_params_dir / "extra_param_improvement_vs_previous.png",
        aggressive_dir / "aggressive_direct_level_top_models.png",
    ]
    for img_path in visuals:
        if img_path.exists():
            story.append(Paragraph(f"Gorsel: {img_path.name}", s_small))
            story.append(Image(str(img_path), width=16*cm, height=9*cm))
            story.append(Spacer(1, 0.3 * cm))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )
    doc.build(story)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/pdf/istanbul_baraj_guncel_durum_detayli_raporu.pdf")
    args = p.parse_args()
    out_path = ROOT / args.out
    build_pdf(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
