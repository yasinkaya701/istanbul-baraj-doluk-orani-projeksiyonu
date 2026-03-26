#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path("/Users/yasinkaya/Hackhaton")
OUT_PDF = ROOT / "output" / "pdf" / "istanbul_baraj_model_veri_ozeti.pdf"
FONT_DIR = Path("/opt/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf")
REGULAR_FONT = FONT_DIR / "DejaVuSans.ttf"
BOLD_FONT = FONT_DIR / "DejaVuSans-Bold.ttf"


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(REGULAR_FONT)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(BOLD_FONT)))
    pdfmetrics.registerFontFamily(
        "DejaVuSansFamily",
        normal="DejaVuSans",
        bold="DejaVuSans-Bold",
        italic="DejaVuSans",
        boldItalic="DejaVuSans-Bold",
    )


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="DejaVuSans-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#334155"),
            spaceAfter=12,
        ),
        "h": ParagraphStyle(
            "h",
            parent=base["Heading2"],
            fontName="DejaVuSans-Bold",
            fontSize=12.5,
            leading=15,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#111827"),
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=9.5,
            leading=13,
            leftIndent=12,
            bulletIndent=0,
            textColor=colors.HexColor("#111827"),
            spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#475569"),
            spaceAfter=2,
        ),
        "box_num": ParagraphStyle(
            "box_num",
            parent=base["BodyText"],
            fontName="DejaVuSans-Bold",
            fontSize=18,
            leading=20,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "box_label": ParagraphStyle(
            "box_label",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=8.5,
            leading=10,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
    }


def load_inputs():
    bundle_summary = json.loads((ROOT / "output" / "model_useful_data_bundle" / "model_useful_data_summary.json").read_text(encoding="utf-8"))
    supply_summary = json.loads((ROOT / "output" / "newdata_feature_store" / "official_monthly_supply_context_summary.json").read_text(encoding="utf-8"))
    reanalysis_summary = json.loads((ROOT / "output" / "newdata_feature_store" / "kandilli_openmeteo_reanalysis_summary.json").read_text(encoding="utf-8"))
    nao_summary = json.loads((ROOT / "output" / "newdata_feature_store" / "noaa_nao_context_summary.json").read_text(encoding="utf-8"))
    api_genel = pd.read_csv(ROOT / "output" / "iski_baraj_api_snapshot" / "tables" / "genel_oran.csv")
    source_current = pd.read_csv(ROOT / "output" / "model_useful_data_bundle" / "tables" / "istanbul_source_current_context.csv")
    coverage = pd.read_csv(ROOT / "output" / "model_useful_data_bundle" / "tables" / "istanbul_model_feature_block_coverage.csv")
    return bundle_summary, supply_summary, reanalysis_summary, nao_summary, api_genel, source_current, coverage


def metric_boxes(s):
    data = [
        [
            Paragraph("281", s["box_num"]),
            Paragraph("168", s["box_num"]),
            Paragraph("0.988", s["box_num"]),
            Paragraph("45.9%", s["box_num"]),
        ],
        [
            Paragraph("Aylık core model satırı", s["box_label"]),
            Paragraph("Resmi aylık şehir suyu satırı", s["box_label"]),
            Paragraph("Reanalysis ET0 ile yerel ET0 korelasyonu", s["box_label"]),
            Paragraph("11 Mart 2026 resmi toplam doluluk snapshot'ı", s["box_label"]),
        ],
    ]
    table = Table(data, colWidths=[43 * mm] * 4, rowHeights=[10 * mm, 13 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f766e")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#115e59")),
                ("INNERGRID", (0, 0), (-1, -1), 0.8, colors.HexColor("#115e59")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def source_table(df, s):
    view = df[
        [
            "baslikAdi",
            "dolulukOrani",
            "annual_yield_million_m3",
            "max_storage_million_m3",
            "current_storage_to_yield_ratio",
        ]
    ].copy()
    view = view.sort_values("dolulukOrani", ascending=False).head(6)
    rows = [[
        Paragraph("<b>Kaynak</b>", s["small"]),
        Paragraph("<b>Doluluk</b>", s["small"]),
        Paragraph("<b>Yıllık verim</b>", s["small"]),
        Paragraph("<b>Maks. depolama</b>", s["small"]),
        Paragraph("<b>Mevcut su / yıllık verim</b>", s["small"]),
    ]]
    for _, r in view.iterrows():
        rows.append(
            [
                Paragraph(str(r["baslikAdi"]), s["small"]),
                Paragraph(f"{r['dolulukOrani']:.2f}%", s["small"]),
                Paragraph(f"{r['annual_yield_million_m3']:.0f} milyon m3", s["small"]),
                Paragraph(f"{r['max_storage_million_m3']:.1f} milyon m3", s["small"]),
                Paragraph(f"{r['current_storage_to_yield_ratio']:.2f}", s["small"]),
            ]
        )
    table = Table(rows, colWidths=[35 * mm, 20 * mm, 32 * mm, 32 * mm, 42 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def block_coverage_table(df, s):
    rows = [[
        Paragraph("<b>Blok</b>", s["small"]),
        Paragraph("<b>Tam kapsama</b>", s["small"]),
        Paragraph("<b>Aralık</b>", s["small"]),
    ]]
    for _, r in df.iterrows():
        start = str(r["start_with_full_block"]).split(" ")[0] if pd.notna(r["start_with_full_block"]) else "-"
        end = str(r["end_with_full_block"]).split(" ")[0] if pd.notna(r["end_with_full_block"]) else "-"
        rows.append(
            [
                Paragraph(str(r["block_name"]), s["small"]),
                Paragraph(f"{r['coverage_pct']:.1f}%", s["small"]),
                Paragraph(f"{start} -> {end}", s["small"]),
            ]
        )
    table = Table(rows, colWidths=[35 * mm, 25 * mm, 90 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_pdf():
    register_fonts()
    s = styles()
    bundle_summary, supply_summary, reanalysis_summary, nao_summary, api_genel, source_current, coverage = load_inputs()

    story = []
    story.append(Paragraph("İstanbul Baraj Doluluk Modeli - Veri Özeti", s["title"]))
    story.append(
        Paragraph(
            "Bu PDF, proje için şimdiye kadar toplanan ve gerçekten modelde işe yarayan veri bloklarını tek bakışta özetler. "
            "Amaç, dağınık veri kümeleri yerine savunulabilir ve tekrar üretilebilir bir model altyapısı göstermektir.",
            s["subtitle"],
        )
    )
    story.append(metric_boxes(s))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Ne Kuruldu", s["h"]))
    for text in [
        "Aylık tahmin modeli için tek giriş noktası oluşturuldu: <b>core monthly</b> ve <b>extended monthly</b> bundle.",
        "İSKİ faaliyet raporlarından resmi şehir suyu arz serisi 2010-2023 aralığına uzatıldı.",
        "İSKİ baraj sayfasından resmi frontend API snapshot paketi çıkarıldı; güncel doluluk, baraj bazlı özet, yıllık yağış ve verilen su serileri alındı.",
        "Kandilli yakınında reanalysis tabanlı radyasyon, ET0, rüzgar ve güneşlenme süresi serisi çekildi.",
        "NOAA CPC aylık NAO indeksi dış rejim değişkeni olarak eklendi.",
        "Kaynak bazlı baraj kapasitesi, verim ve havza alanı bilgileri resmi İSKİ sayfasından tabloya döküldü.",
    ]:
        story.append(Paragraph(text, s["bullet"], bulletText="-"))

    story.append(Paragraph("2. Modelde Kullanılacak Veri Blokları", s["h"]))
    for text in [
        "Core monthly tablo: hedef doluluk, yağış, ET0, tüketim, sıcaklık-nem-basınç proxy'leri, VPD, su dengesi, mevsimsellik ve lag değişkenleri.",
        "Extended monthly tablo: core bloklara ek olarak resmi arz, resmi kayıtlı su, yıllık operasyon bağlamı, reanalysis radyasyon ve ET0, NAO rejim değişkeni.",
        "Source current context: güncel kaynak bazlı durum, kapasite ve verim bağlamı ile baraj bazlı stres anlatımı.",
    ]:
        story.append(Paragraph(text, s["bullet"], bulletText="-"))

    story.append(Paragraph("3. Hızlı Teknik Bulgular", s["h"]))
    live_occ = float(api_genel["oran"].iloc[0])
    for text in [
        f"Resmi şehir suyu ile mevcut tüketim proxy'si çok yüksek uyumlu: <b>corr = {supply_summary['model_vs_supply']['corr']:.3f}</b>.",
        f"Kandilli reanalysis ET0 ile yerel ET0 tarihi güçlü uyumlu: <b>corr = {reanalysis_summary['compare_to_local_et0']['et0_corr']:.3f}</b>.",
        f"Reanalysis radyasyon ile yerel radyasyon türevi de güçlü uyumlu: <b>corr = {reanalysis_summary['compare_to_local_et0']['radiation_month_corr']:.3f}</b>.",
        f"NAO verisi doğrudan ana tahminci değil; fakat DJF yağış rejimi için anlamlı bir dış bağlam veriyor: <b>corr = {nao_summary['long_climate_summary']['seasonal_djf_rain_corr']:.3f}</b>.",
        f"11 Mart 2026 tarihli resmi İSKİ snapshot'ına göre toplam doluluk <b>{live_occ:.1f}%</b>.",
    ]:
        story.append(Paragraph(text, s["bullet"], bulletText="-"))

    story.append(Paragraph("4. Coverage Özeti", s["h"]))
    story.append(block_coverage_table(coverage, s))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Bu tablo pratikte şunu gösteriyor: modelin günlük işleyen omurgası için <b>core</b> blok yeterli; "
            "gelişmiş deneyler ve ek açıklanabilirlik için <b>extended</b> blok kullanılmalı.",
            s["body"],
        )
    )

    story.append(Paragraph("5. Kaynak Bazlı Güncel Görünüm", s["h"]))
    story.append(
        Paragraph(
            "Kaynak-bazlı tablo, toplam doluluğun arkasındaki heterojen yapıyı gösteriyor. "
            "Aynı toplam yüzdeye sahip iki sistem bile farklı depolama ve verim ilişkileri nedeniyle farklı risk profiline sahip olabilir.",
            s["body"],
        )
    )
    story.append(source_table(source_current, s))

    story.append(Paragraph("6. Hazır Çıktılar", s["h"]))
    for text in [
        "Core eğitim matrisi: output/model_useful_data_bundle/tables/istanbul_model_core_monthly.csv",
        "Extended eğitim matrisi: output/model_useful_data_bundle/tables/istanbul_model_extended_monthly.csv",
        "Kaynak-bazlı güncel bağlam: output/model_useful_data_bundle/tables/istanbul_source_current_context.csv",
        "Feature coverage özeti: output/model_useful_data_bundle/tables/istanbul_model_feature_block_coverage.csv",
    ]:
        story.append(Paragraph(text, s["bullet"], bulletText="-"))

    story.append(Paragraph("7. Sonuç", s["h"]))
    story.append(
        Paragraph(
            "Proje artık yalnızca doluluk serisi üzerine kurulu basit bir tahmin fikri değil. "
            "Resmi su arzı, resmi baraj durumu, iklim sürücüleri, ET0 kontrol katmanı, rejim verisi ve kaynak-bazlı açıklama katmanı olan "
            "tekrar üretilebilir bir model altyapısına dönüştü.",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            f"Oluşturulma zamanı: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            s["small"],
        )
    )

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=14 * mm,
        title="İstanbul Baraj Modeli Veri Özeti",
        author="Codex",
    )
    doc.build(story)


if __name__ == "__main__":
    build_pdf()
