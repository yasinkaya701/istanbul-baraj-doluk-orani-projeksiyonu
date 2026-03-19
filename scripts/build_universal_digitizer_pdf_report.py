#!/usr/bin/env python3
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, 
    Spacer, Table, TableStyle
)

def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=26, leading=32, textColor=colors.HexColor("#1A365D"), spaceAfter=20),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#2C5282"), spaceBefore=12, spaceAfter=10, borderPadding=5, thickness=1),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#2B6CB0"), spaceBefore=10, spaceAfter=8),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=11, leading=15, spaceAfter=8),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12, textColor=colors.grey),
        "stats": ParagraphStyle("stats", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=colors.HexColor("#2F855A"), spaceAfter=10)
    }
    return styles

def fit_image(path: Path, max_w: float, max_h: float) -> Image:
    img = PILImage.open(path)
    w, h = img.size
    ratio = min(max_w / float(w), max_h / float(h))
    return Image(str(path), width=w * ratio, height=h * ratio)

def build_summary_table(df: pd.DataFrame, max_rows: int = 15) -> Table:
    data = [df.columns.to_list()] + df.head(max_rows).values.tolist()
    tbl = Table(data, repeatRows=1, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C5282")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return tbl

def main():
    root = Path("/Users/yasinkaya/Hackhaton")
    out_pdf = root / "output/pdf/Universal_Digitization_Project_Summary.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    batch_report_path = root / "output/universal_datasets/batch_report.json"
    stress_test_path = root / "output/STRESS_TEST_SUMMARY.csv"
    
    styles = build_styles()
    story = []
    
    # --- PAGE 1: COVER ---
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("Universal Graph Digitizer", styles["title"]))
    story.append(Paragraph("115 Yillik Iklim Arşivi Sayisallastirma Projesi", styles["h1"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(f"Rapor Tarihi: {datetime.now().strftime('%d %B %Y')}", styles["body"]))
    story.append(Paragraph("Durum: Tamamlandi & Battle-Tested", styles["stats"]))
    story.append(Spacer(1, 2 * cm))
    
    story.append(Paragraph("<b>Proje Özeti:</b>", styles["h2"]))
    summary_text = (
        "Bu rapor, 115 yillik tarihsel grafik kağıdı arşivinin (sıcaklık, nem, yağış, basınç vb.) "
        "modüler ve yüksek performanslı bir 'Direct-to-AI' pipeline ile sayısallaştırılmasını özetler. "
        "Sistem, Viterbi algoritması kullanarak fiziksel mürekkep izlerini yüksek hassasiyetle yakalar."
    )
    story.append(Paragraph(summary_text, styles["body"]))
    
    # High-level metrics
    if batch_report_path.exists():
        with open(batch_report_path, "r") as f:
            data = json.load(f)
        total = data.get("total", 0)
        success = data.get("success", 0)
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"Taramadaki Toplam Dosya: {total}", styles["body"]))
        story.append(Paragraph(f"Başarıyla İşlenen: {success}", styles["body"]))
        story.append(Paragraph(f"Başarı Oranı: %{ (success/total*100) if total > 0 else 0:.1f}", styles["stats"]))

    story.append(PageBreak())
    
    # --- PAGE 2: FILTRATION & STRESS TEST ---
    story.append(Paragraph("1. Bulletproof Veri Filtreleme Sistemi", styles["h1"]))
    story.append(Paragraph(
        "Veri kalitesini garanti altına almak için 3 katmanlı bir filtrasyon mekanizması uygulanmıştır:",
        styles["body"]
    ))
    story.append(Paragraph("- <b>Katman 1: Dosya Adi Analizi:</b> 'Eksik', 'Bozuk', 'Null' gibi anahtar kelimeler anında skip edilir.", styles["body"]))
    story.append(Paragraph("- <b>Katman 2: Mürekkep Yoğunluğu (Density):</b> Sayfada yeterli mürekkep yoksa (boş sayfa) işlem iptal edilir.", styles["body"]))
    story.append(Paragraph("- <b>Katman 3: Sinyal Varyans Kontrolü:</b> Sinyali olmayan (sadece ızgara çizgisi içeren) görseller elenir.", styles["body"]))
    
    if stress_test_path.exists():
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Stres Testi Özeti (Adversarial Results):", styles["h2"]))
        sdf = pd.read_csv(stress_test_path)
        # Select key columns
        cols = ["File Name", "Final Status", "Reason/Detail"]
        story.append(build_summary_table(sdf[cols], max_rows=10))
        
    story.append(PageBreak())
    
    # --- PAGE 3: DATA VISUALIZATION ---
    story.append(Paragraph("2. Örnek Veri ve Çiktilar", styles["h1"]))
    
    # Add Trend Plot if exists
    trend_png = root / "output/visuals/trend_temp_1H.png"
    if trend_png.exists():
        story.append(Paragraph("Sayılaştırılmış Veri Trendi (Örnek - Sıcaklık):", styles["h2"]))
        story.append(fit_image(trend_png, max_w=17 * cm, max_h=10 * cm))
        story.append(Paragraph("Sekil 1: 1H çözünürlükte elde edilen kesintisiz zaman serisi.", styles["small"]))
    else:
        story.append(Paragraph("Zaman serisi grafiği bulunamadı (Önce visualize_climate_trends.py çalıştırılmalı).", styles["body"]))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("3. Direct-to-AI Altyapısı", styles["h1"]))
    ai_text = (
        "Üretilen tüm Parquet ve CSV çıktıları, yapay zeka eğitimine uygun olarak hazırdır. "
        "Gereksiz tüm 'confidence' metrikleri kaldırılarak, AI modellerinin (LSTM, Transformer vb.) "
        "doğrudan veri beslemesi alabileceği temiz bir yapı (`Direct-to-AI`) oluşturulmuştur."
    )
    story.append(Paragraph(ai_text, styles["body"]))
    
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("Raporun Sonu", styles["small"]))
    
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    doc.build(story)
    print(f"🏆 Final PDF Raporu Hazir: {out_pdf}")

if __name__ == "__main__":
    main()
