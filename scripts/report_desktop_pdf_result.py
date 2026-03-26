#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, 
    Spacer, Table, TableStyle
)
from PIL import Image as PILImage

def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, textColor=colors.HexColor("#1A365D")),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, textColor=colors.HexColor("#2C5282")),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=11, leading=14),
        "highlight": ParagraphStyle("highlight", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#2F855A"))
    }
    return styles

def fit_image(path: Path, max_w: float, max_h: float) -> Image:
    img = PILImage.open(path)
    w, h = img.size
    ratio = min(max_w / float(w), max_h / float(h))
    return Image(str(path), width=w * ratio, height=h * ratio)

def main():
    root = Path("/Users/yasinkaya/Hackhaton")
    out_pdf = root / "output/pdf/Desktop_PDF_Digitization_Report.pdf"
    
    styles = build_styles()
    story = []
    
    story.append(Paragraph("Masaüstü PDF Analiz Raporu", styles["title"]))
    story.append(Paragraph(f"Dosya: bunuyap.pdf", styles["h1"]))
    story.append(Paragraph(f"Analiz Tarihi: {datetime.now().strftime('%d %B %Y %H:%M')}", styles["body"]))
    story.append(Spacer(1, 1 * cm))
    
    story.append(Paragraph("1. Sayisallastirma Sonucu", styles["h1"]))
    story.append(Paragraph(
        "Masaüstünde bulunan 'bunuyap.pdf' dosyası başarıyla TIFF formatına dönüştürüldü ve "
        "Universal Graph Digitizer modeli ile işlendi. Sistem, görseldeki ana sinyali yakalayarak "
        "zaman serisi verisine dönüştürdü.", 
        styles["body"]
    ))
    
    # Show the extracted plot
    plot_path = root / "output/visuals/trend_unknown_1H.png"
    if plot_path.exists():
        story.append(Spacer(1, 0.5 * cm))
        story.append(fit_image(plot_path, max_w=17 * cm, max_h=10 * cm))
        story.append(Paragraph("Sekil 1: bunuyap.pdf dosyasindan cikarilan 1H (Saatlik) veri trendi.", styles["body"]))
    
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("2. Veri Konumu", styles["h1"]))
    story.append(Paragraph("Elde edilen veriler asagidaki konumlarda saklanmaktadir:", styles["body"]))
    story.append(Paragraph(f"- CSV: output/universal_datasets/unknown/1H/page-1.csv", styles["highlight"]))
    story.append(Paragraph(f"- Parquet: output/universal_datasets/unknown/1H/page-1.parquet", styles["highlight"]))
    
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("Rapor otomatik olarak Antigravity AI tarafından üretilmiştir.", styles["body"]))
    
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4)
    doc.build(story)
    print(f"✅ Desktop PDF Raporu Hazir: {out_pdf}")

if __name__ == "__main__":
    main()
