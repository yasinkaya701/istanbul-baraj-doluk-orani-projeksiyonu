#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/Users/yasinkaya/Hackhaton")
OUT_REPORT_DIR = ROOT / "output" / "report"
OUT_PDF_DIR = ROOT / "output" / "pdf"
OUT_MD = OUT_REPORT_DIR / "istanbul_baraj_proje_detayli_rapor.md"
OUT_PDF = OUT_PDF_DIR / "istanbul_baraj_proje_detayli_rapor.pdf"

FONT_DIR = Path("/opt/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf")
REGULAR_FONT = FONT_DIR / "DejaVuSans.ttf"
BOLD_FONT = FONT_DIR / "DejaVuSans-Bold.ttf"


@dataclass(frozen=True)
class Reference:
    code: str
    title: str
    url: str
    note: str


REFERENCES = [
    Reference("K1", "FAO-56 Crop Evapotranspiration Chapter 2", "https://www.fao.org/4/X0490E/x0490e06.htm", "Penman-Monteith ET0 method base"),
    Reference("K2", "USACE HEC-HMS Penman-Monteith Method", "https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/evaporation-and-transpiration/penman-monteith-method", "Operational ET implementation reference"),
    Reference("K3", "Reservoir evaporation and water availability under climate stress", "https://hess.copernicus.org/articles/28/3243/2024/index.html", "Reservoir evaporation and water-availability link"),
    Reference("K4", "Open-water evaporation and water-surface response", "https://hess.copernicus.org/articles/30/67/2026/", "Distinction between ET0 and open-water evaporation"),
    Reference("K5", "Reservoir volume forecasting with AI", "https://doi.org/10.1016/j.jhydrol.2022.128766", "Forecasting benchmark from Journal of Hydrology"),
    Reference("K6", "Conditioned LSTM reservoir releases", "https://doi.org/10.1016/j.jhydrol.2025.133750", "Operational conditioning benchmark for reservoir models"),
    Reference("K7", "Urban water demand restriction review", "https://doi.org/10.1111/j.1936-704X.2024.3402.x", "Demand restriction evidence review"),
    Reference("K8", "Mandatory vs voluntary water restriction range", "https://doi.org/10.22004/ag.econ.19327", "Restriction effect comparison"),
    Reference("K9", "Los Angeles water conservation restriction effect", "https://doi.org/10.1016/j.resconrec.2014.10.005", "Observed demand reduction evidence"),
    Reference("K10", "Guide to Instruments and Methods of Observation (WMO-No. 8)", "https://wmo.int/guide-instruments-and-methods-of-observation-wmo-no-8-0", "Radiation and sunshine measurement reference for actinograph integration"),
    Reference("K11", "ISKI Water Loss Annual Reports page", "https://iski.istanbul/kurumsal/stratejik-yonetim/su-kayiplari-yillik-raporlari/", "Landing page for annual water-loss forms"),
    Reference("K12", "ISKI 2020 Standard Water Balance Form", "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2020_9a984f0ba7.pdf", "Official 2020 NRW and loss table"),
    Reference("K13", "ISKI 2021 Standard Water Balance Form", "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2021_9e4b97ee29.pdf", "Official 2021 NRW and loss table"),
    Reference("K14", "ISKI 2022 Standard Water Balance Form", "https://cdn.iski.istanbul/uploads/Su_Denge_Tablosu_2022_46b2a9477c.pdf", "Official 2022 NRW and loss table"),
    Reference("K15", "ISKI 2023 Standard Water Balance Form", "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2023_4c07821536.pdf", "Official 2023 NRW and loss table"),
    Reference("K16", "ISKI 2020 Activity Report", "https://cdn.iski.istanbul/uploads/2020_FAALIYET_RAPORU_903efe0267.pdf", "Official annual subscribers, supply and reclaimed-water metrics"),
    Reference("K17", "ISKI 2021 Activity Report", "https://cdn.iski.istanbul/uploads/2021_FAALIYET_RAPORU_64bf206f27.pdf", "Official annual subscribers, supply and reclaimed-water metrics"),
    Reference("K18", "ISKI 2022 Activity Report", "https://cdn.iski.istanbul/uploads/2022_Faaliyet_Raporu_c65c8a733d.pdf", "Official annual subscribers, supply and reclaimed-water metrics"),
    Reference("K19", "ISKI 2023 Activity Report", "https://iskiapi.iski.gov.tr/uploads/2023_Yili_Faaliyet_Raporu_24309dd9dd.pdf", "Official annual subscribers, supply and reclaimed-water metrics"),
    Reference("K20", "ISKI 2014 Activity Report", "https://www.iski.gov.tr/web/assets/SayfalarDocs/faaliyetraporlari/faaliyetraporu2008/faaliyet_raporu2014.pdf", "Official monthly city-supply averages covering 2010-2014"),
    Reference("K21", "ISKI 2015 Activity Report", "https://www.iski.gov.tr/web/assets/SayfalarDocs/faaliyetraporlari/faaliyetraporu2008/faaliyet_raporu2015.pdf", "Official monthly city-supply averages covering 2011-2015"),
    Reference("K22", "ISKI 2017 Activity Report", "https://www.iski.gov.tr/web/assets/SayfalarDocs/faaliyetraporlari/faaliyetraporu2008/2017_Faaliyet_Raporu..pdf", "Official monthly city-supply averages covering 2013-2017"),
    Reference("K23", "ISKI Baraj Doluluk Oranlari page", "https://iski.istanbul/baraj-doluluk", "Official current occupancy dashboard"),
    Reference("K24", "ISKI Water Sources page", "https://iski.istanbul/kurumsal/hakkimizda/su-kaynaklari", "Official source storage, yield, year and basin context"),
    Reference("K25", "Open-Meteo Historical Weather API documentation", "https://open-meteo.com/en/docs/historical-weather-api", "Historical proxy weather and ET0 documentation"),
    Reference("K26", "NOAA CPC monthly NAO index", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii", "Monthly NAO regime series"),
    Reference("K27", "ISKI 2026 Water Unit Tariffs", "https://iski.istanbul/abone-hizmetleri/abone-rehberi/su-birim-fiyatlari/", "Sector and tariff grouping context"),
    Reference("K28", "Water Loss Control Regulation official PDF", "https://www.tarimorman.gov.tr/SYGM/Belgeler/%C4%B0%C3%A7me%20Suyu%20Temin%20Ve%20Da%C4%9F%C4%B1t%C4%B1m%2002.09.2019/i%C3%A7me%20suyu%20temin%20ve%20da%C4%9F%C4%B1t%C4%B1m%20sistemlerindeki%20su%20kay%C4%B1plar%C4%B1n%C4%B1n%20kontrol%C3%BC%20y%C3%B6netmeli%C4%9Fi.pdf", "Official annual water-loss reduction context"),
]

GLOSSARY = [
    ("Kapasite agirlikli toplam doluluk", "Her baraji esit saymak yerine buyuk hacimli barajlara daha fazla agirlik veren toplam doluluk olcusu."),
    ("Referans evapotranspirasyon (ET0)", "Havanin standart bir yuzeyden ne kadar su cekme egilimi gosterdigini anlatan olcu."),
    ("Evapotranspirasyon", "Toprak ve bitki yuzeyinden havaya giden toplam su kaybi."),
    ("Acik su buharlasmasi", "Baraj yuzeyinden dogrudan olan su kaybi. ET0 ile ilgili ama ayni sey degil."),
    ("Vekil degisken (proxy variable)", "Dogrudan olcemedigimiz bir seyi dolayli olarak temsil eden degisken."),
    ("Yeniden analiz verisi (reanalysis)", "Olcumlerle fizik tabanli hava modellerinin birlestirilmesiyle uretilen tarihsel hava veri seti."),
    ("Buhar basinci acigi (vapor pressure deficit, VPD)", "Havanin ne kadar 'su istemeye' egilimli oldugunu gosteren nem-aciklik olcusu."),
    ("Gelir getirmeyen su (non-revenue water, NRW)", "Sisteme verilen ama faturalanamayan veya kayip olan suyun toplami."),
    ("Yetkili tuketim", "Sistemde resmen izinli ve tanimli kullanilan su miktari."),
    ("Geri kazanilmis su", "Aritilarak tekrar kullanima kazandirilan su."),
    ("Kok ortalama kare hata (root mean square error, RMSE)", "Tahmin hatasinin buyuklugunu ozetleyen performans olcusu. Dusuk olmasi daha iyidir."),
    ("Belirleme katsayisi (R-squared, R2)", "Modelin degisimin ne kadarini acikladigini gosteren olcu. Yuksek olmasi daha iyidir."),
    ("Kisa vadeli guncel durum tahmini (nowcasting)", "Cok kisa vadeli guncel durum tahmini veya anlik izleme."),
    ("Giris akimi (inflow)", "Baraja giren su miktari."),
    ("Senaryo testi (scenario test)", "Bir parametreyi degistirip sistemin nasil tepki verdigini gormek icin yapilan deney."),
]


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


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="DejaVuSans-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=10.2,
            leading=14,
            textColor=colors.HexColor("#334155"),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="DejaVuSans-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="DejaVuSans-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#111827"),
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        ),
        "body_left": ParagraphStyle(
            "body_left",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#111827"),
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=9.3,
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
            fontSize=8.1,
            leading=10.5,
            textColor=colors.HexColor("#475569"),
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=8.3,
            leading=10.5,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=8,
        ),
        "metric_num": ParagraphStyle(
            "metric_num",
            parent=base["BodyText"],
            fontName="DejaVuSans-Bold",
            fontSize=17,
            leading=18,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "metric_label": ParagraphStyle(
            "metric_label",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=8.4,
            leading=10.2,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "ref": ParagraphStyle(
            "ref",
            parent=base["BodyText"],
            fontName="DejaVuSans",
            fontSize=8.4,
            leading=10.8,
            textColor=colors.HexColor("#111827"),
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> dict[str, object]:
    output = ROOT / "output"
    data = {
        "bundle_summary": load_json(output / "model_useful_data_bundle" / "model_useful_data_summary.json"),
        "supply_summary": load_json(output / "newdata_feature_store" / "official_monthly_supply_context_summary.json"),
        "reanalysis_summary": load_json(output / "newdata_feature_store" / "kandilli_openmeteo_reanalysis_summary.json"),
        "nao_summary": load_json(output / "newdata_feature_store" / "noaa_nao_context_summary.json"),
        "source_summary": load_json(output / "newdata_feature_store" / "official_iski_source_context_summary.json"),
        "oper_summary": load_json(output / "newdata_feature_store" / "official_operational_context_summary.json"),
        "policy_summary": load_json(output / "newdata_feature_store" / "official_policy_leverage_summary.json"),
        "deep_summary": load_json(output / "newdata_feature_store" / "deepened_feature_summary.json"),
        "annual_ctx_summary": load_json(output / "newdata_feature_store" / "annual_context_monthly_model_summary.json"),
        "hub_status": load_json(ROOT / "research" / "baraj_doluluk_hub" / "admin" / "HUB_STATUS.json"),
        "api_manifest": load_json(output / "iski_baraj_api_snapshot" / "api_manifest.json"),
        "coverage": pd.read_csv(output / "model_useful_data_bundle" / "tables" / "istanbul_model_feature_block_coverage.csv"),
        "source_current": pd.read_csv(output / "model_useful_data_bundle" / "tables" / "istanbul_source_current_context.csv"),
        "deep_metrics": pd.read_csv(output / "newdata_feature_store" / "tables" / "deepened_feature_model_metrics.csv"),
        "annual_ctx_metrics": pd.read_csv(output / "newdata_feature_store" / "tables" / "annual_context_monthly_model_metrics.csv"),
        "policy_annual": pd.read_csv(output / "newdata_feature_store" / "tables" / "official_policy_leverage_annual.csv"),
        "api_current": pd.read_csv(output / "iski_baraj_api_snapshot" / "tables" / "genel_oran.csv"),
        "api_14d": pd.read_csv(output / "iski_baraj_api_snapshot" / "tables" / "son_14_gun_toplam_doluluk.csv"),
        "scenario_summary": pd.read_csv(output / "istanbul_dam_deep_research" / "scenario_summary.csv"),
        "deep_research_summary": load_json(output / "istanbul_dam_deep_research" / "summary.json"),
    }
    return data


def ensure_dirs() -> None:
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PDF_DIR.mkdir(parents=True, exist_ok=True)


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def escape(text: str) -> str:
    return html.escape(text, quote=True)


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def format_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def add_bullets(story: list, items: list[str], style: ParagraphStyle) -> None:
    for item in items:
        story.append(Paragraph(item, style, bulletText="-"))


def add_glossary_story(story: list, s: dict[str, ParagraphStyle]) -> None:
    story.append(para("2. Teknik Terimler Icin Kisa Okuma Rehberi", s["h1"]))
    story.append(
        para(
            "Bu bolum domain bilgisi olmayan okuyucu icin eklendi. Raporun geri kalaninda gecen ana teknik terimler asagida tek cumle ile aciklanmistir.",
            s["body"],
        )
    )
    for term, definition in GLOSSARY:
        story.append(Paragraph(f"<b>{escape(term)}:</b> {escape(definition)}", s["bullet"], bulletText="-"))


def make_metric_table(s: dict[str, ParagraphStyle], data: dict[str, object]) -> Table:
    api_current = data["api_current"]
    bundle_summary = data["bundle_summary"]
    deep_summary = data["deep_summary"]
    supply_summary = data["supply_summary"]
    reanalysis_summary = data["reanalysis_summary"]
    deep_research_summary = data["deep_research_summary"]
    hub_status = data["hub_status"]

    metrics = [
        (f"{float(api_current['oran'].iloc[0]):.1f}%", "11 Mart 2026 resmi toplam doluluk"),
        (f"{deep_summary['best_rmse_pp']:.2f}", "En iyi gözlenen pencere RMSE (yüzde puan)"),
        (f"{supply_summary['model_vs_supply']['corr']:.3f}", "Resmi arz ile tüketim proxy korelasyonu"),
        (f"{reanalysis_summary['compare_to_local_et0']['et0_corr']:.3f}", "Yeniden analiz (reanalysis) ET0 ile yerel ET0 korelasyonu"),
        (f"{deep_research_summary['weighted_vs_mean_avg_diff_pp']:.2f}", "Ağırlıklı toplam ile eşit ortalama farkı"),
        (str(hub_status["external_sources_count"]), "Kayıtlı dış kaynak sayısı"),
    ]

    rows: list[list[Paragraph]] = []
    for i in range(0, len(metrics), 3):
        row_metrics = metrics[i : i + 3]
        row_a = [para(m[0], s["metric_num"]) for m in row_metrics]
        row_b = [para(m[1], s["metric_label"]) for m in row_metrics]
        while len(row_a) < 3:
            row_a.append(para("", s["metric_num"]))
            row_b.append(para("", s["metric_label"]))
        rows.extend([row_a, row_b])

    table = Table(rows, colWidths=[57 * mm, 57 * mm, 57 * mm], rowHeights=[10 * mm, 15 * mm, 10 * mm, 15 * mm])
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


def make_table(rows: list[list[str]], widths_mm: list[float], s: dict[str, ParagraphStyle], font_style: str = "small") -> Table:
    wrapped: list[list[Paragraph]] = []
    for idx, row in enumerate(rows):
        style = s["small"] if idx == 0 else s[font_style]
        wrapped.append([para(cell, style) for cell in row])
    table = Table(wrapped, colWidths=[w * mm for w in widths_mm])
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


def fit_image(path: Path, max_width_mm: float, max_height_mm: float) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    scale = min((max_width_mm * mm) / width, (max_height_mm * mm) / height)
    return Image(str(path), width=width * scale, height=height * scale)


def add_figure(story: list, path: Path, caption: str, s: dict[str, ParagraphStyle], max_height_mm: float = 85) -> None:
    story.append(fit_image(path, 176, max_height_mm))
    story.append(para(caption, s["caption"]))


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DejaVuSans", 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(18 * mm, 289 * mm, "Istanbul Baraj Doluluk Projesi - Detayli Rapor")
    canvas.drawRightString(192 * mm, 10 * mm, f"Sayfa {doc.page}")
    canvas.restoreState()


def build_markdown(data: dict[str, object]) -> str:
    bundle_summary = data["bundle_summary"]
    supply_summary = data["supply_summary"]
    reanalysis_summary = data["reanalysis_summary"]
    nao_summary = data["nao_summary"]
    source_summary = data["source_summary"]
    oper_summary = data["oper_summary"]
    policy_summary = data["policy_summary"]
    deep_summary = data["deep_summary"]
    annual_ctx_summary = data["annual_ctx_summary"]
    api_manifest = data["api_manifest"]
    api_current = data["api_current"]
    api_14d = data["api_14d"]
    coverage = data["coverage"]
    scenario_summary = data["scenario_summary"]
    deep_research_summary = data["deep_research_summary"]
    source_current = data["source_current"]

    current_fill = float(api_current["oran"].iloc[0])
    snapshot_time = str(api_current["snapshot_updated_at"].iloc[0])
    start_14d = float(api_14d["dolulukOrani"].iloc[0])
    end_14d = float(api_14d["dolulukOrani"].iloc[-1])

    coverage_lines = []
    for _, row in coverage.iterrows():
        coverage_lines.append(
            f"- `{row['block_name']}`: %{row['coverage_pct']:.1f} tam blok kapsama, "
            f"`{str(row['start_with_full_block']).split(' ')[0]}` -> `{str(row['end_with_full_block']).split(' ')[0]}`"
        )

    selected_scenarios = {
        "rain_plus10_3m": "3 ay boyunca yagis +%10",
        "et0_plus10_3m": "3 ay boyunca ET0 +%10",
        "cons_plus10_3m": "3 ay boyunca tuketim +%10",
        "restriction_minus15_3m": "3 ay boyunca talep -%15 kisit",
        "hot_dry_high_demand": "sicak-kurak-yuksek talep bilesik senaryo",
    }
    scenario_lines = []
    for key, label in selected_scenarios.items():
        sub = scenario_summary[scenario_summary["scenario"] == key].set_index("horizon_month")
        if 3 in sub.index:
            scenario_lines.append(
                f"- {label}: 3. ay etkisi `{sub.loc[3, 'delta_pp']:+.2f}` yp"
                + (f", 6. ay etkisi `{sub.loc[6, 'delta_pp']:+.2f}` yp" if 6 in sub.index else "")
            )

    top_sources = source_current.sort_values("dolulukOrani", ascending=False).head(5)
    top_source_lines = []
    for _, row in top_sources.iterrows():
        top_source_lines.append(
            f"- `{row['baslikAdi']}`: doluluk `%{row['dolulukOrani']:.2f}`, "
            f"mevcut su / yillik verim `{row['current_storage_to_yield_ratio']:.2f}`, "
            f"mevcut su / azami depolama `{row['current_storage_to_max_storage_ratio']:.2f}`"
        )

    refs_md = "\n".join(
        f"- **[{r.code}]** {r.title}. {r.note}. {r.url}" for r in REFERENCES
    )

    return f"""# Istanbul Baraj Doluluk Projesi - Detayli Kaynakcali Rapor

Olusturulma tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}

## 1. Yonetici Ozeti

Bu proje, Istanbul'un toplam baraj doluluk oranini yalnizca gecmis doluluk degerlerinden tahmin etmeye calisan dar bir seri-modeli olarak kalmadi. Calisma, iklim suruculerini, referans evapotranspirasyon (ET0) temelli atmosferik talebi, resmi su arzi ve operasyon verilerini, baraj bazli kaynak baglamini ve guncel resmi uygulama anlik kesit (API snapshot) katmanini bir araya getiren bir karar-destek altyapisina donustu. Bu omurganin merkezinde aylik zaman adiminda **kapasite agirlikli toplam doluluk** hedefi yer aliyor. Bu secim kritik; cunku es agirlikli ortalama ile kapasite agirlikli toplam arasindaki ortalama fark mevcut derin testlerde **{deep_research_summary['weighted_vs_mean_avg_diff_pp']:.2f} yuzde puan** bulundu.

11 Mart 2026 tarihli resmi ISKI anlik kesitine (snapshot) gore toplam doluluk **%{current_fill:.1f}** olup kesit zamani **{snapshot_time}** olarak kaydedildi. Son 14 gunluk resmi seri ayni kesite gore **%{start_14d:.2f} -> %{end_14d:.2f}** araliginda yukari yonlu hareket etti. Bu gunluk katman, uzun donem aylik model ile birlikte proje icin dogrudan operasyonel guncellik sagliyor. [K23]

Teknik olarak bugun gelinen nokta su:

- Aylik cekirdek model matrisi hazir: `{bundle_summary['core_rows']}` satir, `{bundle_summary['core_start']}` -> `{bundle_summary['core_end']}`
- Gelismis genisletilmis model matrisi hazir: resmi arz, yeniden analiz (reanalysis) ve Kuzey Atlantik Salinimi (NAO) bloklari entegre
- Resmi aylik sehir suyu serisi `{supply_summary['official_city_supply_window']['start']}` -> `{supply_summary['official_city_supply_window']['end']}` araligina ulasti
- Kandilli yakininda yeniden analiz vekil serisi (reanalysis proxy) `{reanalysis_summary['request_window']['start']}` -> `{reanalysis_summary['request_window']['end']}` araliginda mevcut
- Kaynak bazli guncel baglam tablosu 10 ana aktif kaynak icin guncel durum veriyor

## 2. Teknik Terimler Icin Kisa Okuma Rehberi

Bu bolum domain bilgisi olmayan okuyucu icin eklendi. Raporun geri kalaninda gecen ana teknik terimler asagida tek cumle ile aciklanmistir:

{chr(10).join(f"- **{term}**: {definition}" for term, definition in GLOSSARY)}

## 3. Problem Tanimi ve Hedef

Ana soru sudur: Istanbul'un toplam baraj dolulugu hangi fiziksel, operasyonel ve talep tarafli degiskenlerle daha guvenilir bicimde izah ve tahmin edilebilir?

Bu soruya cevap verirken kapsam bilerek daraltildi:

- hedef degisken: Istanbul toplam baraj doluluk orani
- tercih edilen tanim: kapasite agirlikli toplam doluluk
- zaman adimi: aylik
- ana amac: gecmisin yeniden kurulmasi ve bunun uzerinden guvenilir 15 yillik projeksiyon motorunun hazirlanmasi

Bu nedenle proje, uc farkli fikir arasinda dagilmak yerine tek omurgaya cekildi: **Istanbul toplam baraj doluluk tahmini + risk/mudahale senaryolari + karar destegi**.

## 4. Metodolojik Cerceve

Modelleme tarafi uc katmanda kuruldu:

1. **Fiziksel iklim katmani**: yagis, referans evapotranspirasyon (ET0), sicaklik, bagil nem, basinc vekilleri (proxy), buhar basinci acigi (vapor pressure deficit, VPD) ve yagis-ET0 su dengesi vekilleri. Buradaki fikir su: hava ne kadar yagis getiriyor ve ayni anda atmosfer ne kadar su cekmek istiyor? ET0 tarafi FAO-56 Penman-Monteith omurgasina dayandirildi. [K1][K2]
2. **Insan kullanim ve operasyon katmani**: aylik tuketim vekili (proxy), resmi sehir arzi, yillik gelir getirmeyen su (non-revenue water, NRW), yetkili tuketim, aktif abone ve geri kazanilmis su bloklari. Bu katman, sehri yoneten insan faaliyetlerinin baraj sistemi uzerindeki baskisini temsil ediyor. [K11][K12][K13][K14][K15][K16][K17][K18][K19]
3. **Kaynak ve rejim katmani**: baraj bazli kapasite, verim, havza alani, guncel resmi anlik kesit (snapshot) ve kis rejimini temsil eden Kuzey Atlantik Salinimi (North Atlantic Oscillation, NAO) degiskeni. Bu katman, hem tek tek kaynaklarin farkini hem de buyuk olcekli hava rejimlerini modele bagliyor. [K23][K24][K26]

Ek olarak, acik su buharlasmasi ile ET0'nun ayni sey gibi sunulmamasi icin yontemsel ayrim korunmustur. ET0, referans yuzey atmosferik talebi temsil ederken; acik su buharlasmasi rezervuar yuzeyi, enerji dengesi ve alan tepkisi ile ilgilidir. Bu ayrim reservoir evaporation literaturunde acik bicimde vurgulanir. [K3][K4]

## 5. Ne Insa Edildi

Proje boyunca yalnizca veri toplanmadi; yeniden kullanilabilir bir arastirma ve model altyapisi kuruldu:

- Profesyonel arastirma hub'i: kaynak registry, dataset envanteri, karar log'u, worklog ve artifact katalogu
- `new data` merkezli ozellik deposu (feature store) katmani
- Resmi ISKI raporlarindan cekilen aylik ve yillik operasyon serileri
- Resmi baraj sayfasindan cikarilan guncel uygulama anlik kesit (API snapshot) paketi
- Model icin tek giris noktasi olan `cekirdek aylik tablo (core monthly)` ve `genisletilmis aylik tablo (extended monthly)` matrisleri
- Kaynak bazli guncel durum ve aciklanabilirlik tablosu

Bu altyapi sayesinde veri ve denemeler artik daginik dosyalar halinde degil; tekrar uretilebilir pipeline olarak duruyor.

## 6. Veri Katmanlari ve Kapsam

### 6.1 Cekirdek aylik model matrisi

`istanbul_model_core_monthly.csv` bugun icin varsayilan egitim tablosudur. Icerigi:

- hedef doluluk ve 1-2 aylik gecikmeleri
- yagis ve roll-3 ozetleri
- referans evapotranspirasyon (ET0) ve roll-3 ozetleri
- tuketim vekili (proxy) ve roll-3 ozetleri
- sicaklik, bagil nem, basinç proxy'leri
- buhar basinci acigi (VPD)
- su dengesi vekili (proxy)
- mevsimsellik bileşenleri

Tam blok kapsama bilgisi:

{chr(10).join(coverage_lines)}

### 6.2 Resmi sehir suyu serisi

2010-2023 araliginda resmi ISKI faaliyet raporlarindan aylik sehir suyu verisi yeniden kuruldu. En onemli bulgu, mevcut tuketim vekilinin (proxy) faturalanmis veya kayitli suyu degil; neredeyse dogrudan **sisteme verilen suyu** izledigidir. Resmi arz ile mevcut vekil arasindaki korelasyon **{supply_summary['model_vs_supply']['corr']:.3f}**, ortalama oran ise **%{supply_summary['model_vs_supply']['mean_ratio_pct']:.2f}** bulundu. Buna karsilik kayitli su serisi ile uyum daha zayif kaldi. [K16][K17][K18][K19][K20][K21][K22]

Bu sonuc model acisindan onemli; cunku mevcut talep serisinin "toplam sistem cekisi" tarafini tasidigi, ancak sektor ayrimi veya faturalanmis tuketim anlatisi icin tek basina yetmedigi anlasildi.

### 6.3 Resmi operasyon ve kayip katmani

Yillik operasyon baglaminda su an kamuya acik ve makine-islenebilir bloklar sunlar:

- aktif abone
- yetkili tuketim
- sehre verilen su
- geri kazanilmis su
- gelir getirmeyen su (NRW), idari kayip ve fiziki kayip

2020 -> 2023 arasinda:

- aktif abone artisi: %{oper_summary['active_subscriber_growth_2020_2023_pct']:.2f}
- yetkili tuketim yogunlugu: {oper_summary['authorized_consumption_l_per_active_subscriber_day_2020']:.2f} -> {oper_summary['authorized_consumption_l_per_active_subscriber_day_2023']:.2f} L/abone-gun
- geri kazanilmis su payi: %{oper_summary['reclaimed_share_pct_2020']:.2f} -> %{oper_summary['reclaimed_share_pct_2023']:.2f}

Gelir getirmeyen su (NRW) yillik olarak anlamli bir yonetisim degiskeni sagliyor; ancak su anki acik veride aylik yogun bir seri olmadigi icin bu blok ay bazli tahminde sinirli katkida kaliyor. [K11][K12][K13][K14][K15][K28]

### 6.4 Kaynak bazli baglam

Resmi kaynak sayfasi uzerinden 17 satirlik bir kaynak baglam tablosu kuruldu. Bunun icinde yillik verim, azami depolama, devreye alma yili ve varsa havza alani bulunuyor. Bu tablo iki nedenle cok yararli:

- yagis -> inflow proxy tasarimi icin havza alani veriyor
- toplam doluluk icindeki heterojenligi acikliyor

Ozet:

- baraj sayisi: {source_summary['baraj_count']}
- regule edici kaynak sayisi: {source_summary['regulator_count']}
- havza alani bilgisi olan kaynak: {source_summary['with_basin_area_count']}
- en buyuk yillik verim: `{source_summary['max_yield_source']}`
- en buyuk azami depolama: `{source_summary['max_storage_source']}` [K24]

Guncel ilk bes kaynak ornegi:

{chr(10).join(top_source_lines)}

### 6.5 Kandilli yakininda yeniden analiz vekili (reanalysis proxy)

Aktinograf ve dogrudan radyasyon olcumleri gelene kadar ET0 hattini kontrol edecek dis katman olarak Open-Meteo historical API uzerinden ERA5-seamless temelli yeniden analiz vekili (reanalysis proxy) cekildi. Kullanilan koordinat ankari Kandilli yakininda **41.0615, 29.0592** olup API'nin döndürdüğü grid noktasi yaklasik **41.100006, 29.100006** olmustur. [K25]

Bu seri:

- gunluk pencere: `{reanalysis_summary['request_window']['start']}` -> `{reanalysis_summary['request_window']['end']}`
- gunluk satir: `{reanalysis_summary['daily_rows']}`
- aylik satir: `{reanalysis_summary['monthly_rows']}`

Yerel ET0 tarihi ile ustuste bindiginde:

- ET0 korelasyonu: `{reanalysis_summary['compare_to_local_et0']['et0_corr']:.3f}`
- ET0 aylik MAE: `{reanalysis_summary['compare_to_local_et0']['et0_mae_mm_month']:.2f}` mm/ay
- radyasyon korelasyonu: `{reanalysis_summary['compare_to_local_et0']['radiation_month_corr']:.3f}`

Bu nedenle yeniden analiz katmani (reanalysis) yer istasyonu yerine gecen bir kaynak degil; **karsilastirma cipasi (benchmark), eksik tamamlama (gap-fill) ve makulluk kontrolu (sanity-check) katmani** olarak degerlidir. Daha sade okumayla bu sunu soyluyor: eldeki ET0 serimiz yalniz degil, dis bir hava veri seti tarafindan da buyuk olcude destekleniyor. Aktinograf geldikten sonra WMO-uyumlu radyasyon entegrasyonu ile bu belirsizlik daha da azaltilabilir. [K10][K25]

### 6.6 Kuzey Atlantik Salinimi (NAO) rejim katmani

NOAA CPC aylik Kuzey Atlantik Salinimi (North Atlantic Oscillation, NAO) serisi kis rejimi baglami icin eklendi. Bu degisken yerel hidrolojik olcum degil; buyuk olcekli dolasim rejimini temsil eder. Istanbul verisiyle uzun pencere testinde DJF toplam yagis ile korelasyon **{nao_summary['long_climate_summary']['seasonal_djf_rain_corr']:.3f}** bulundu. Bu, degiskenin dogrudan hedef tahmincisi degil; **kis risk baglami** olarak yararli bir dis ozellik (feature) oldugunu gosteriyor. [K26]

## 7. Modelleme Sonuclari

### 7.1 Agirlikli toplam hedef neden secildi?

Es agirlikli `overall_mean` yaklasimi ile kapasite agirlikli toplam doluluk arasindaki ortalama fark **{deep_research_summary['weighted_vs_mean_avg_diff_pp']:.2f} yuzde puan** bulundu. Bu fark, Istanbul gibi kapasitesi asimetrik bir sistemde kritik. Buyuk kaynaklarda olan degisimlerin sehir olcegindeki gercek su bulunurlugu uzerindeki etkisini esit ortalama maskeleyebiliyor. Bu nedenle projenin hedef degiskeni kapasite agirlikli toplam doluluk olarak sabitlendi.

### 7.2 Gozlenen pencere derin feature testi

Sikı gozlenen `new data` penceresinde (`{deep_summary['window']['start']}` -> `{deep_summary['window']['end']}`) en iyi model **{deep_summary['best_model']}** oldu ve RMSE **{deep_summary['best_rmse_pp']:.2f} yuzde puan** seviyesine indi. Temel model (baseline) olan `rain + ET0 + demand + memory` kurgusuna sicaklik ve nem bloklari eklendiginde somut iyilesme olustu. Daha sade soylemle: hava kosullarini biraz daha ayrintili anlattigimizda model biraz daha iyi calisiyor.

Model siralamasi:

- `plus_temp_humidity`: RMSE {deep_summary['models'][0]['rmse_pp']:.2f}, R2 {deep_summary['models'][0]['r2']:.3f}
- `deep_all`: RMSE {deep_summary['models'][1]['rmse_pp']:.2f}, R2 {deep_summary['models'][1]['r2']:.3f}
- `baseline_full` (temel model): RMSE {deep_summary['models'][2]['rmse_pp']:.2f}, R2 {deep_summary['models'][2]['r2']:.3f}
- `plus_vpd_balance`: RMSE {deep_summary['models'][3]['rmse_pp']:.2f}, R2 {deep_summary['models'][3]['r2']:.3f}

Yorum:

- depolama hafizasi hala en guclu blok
- yagis talebe gore daha net aciklayicilik tasiyor
- sicaklik ve nem anlamli ek sinyal veriyor
- buhar basinci acigi (VPD) ve yagis-ET0 farki yorumlayici ama tek basina kazanan blok degil

### 7.3 Yillik operasyon baglami kisa pencere testi

Kamuya acik yillik resmi baglam su an 2020-2023 araliginda yogunlastigi icin bu test daha kisa pencereye sahip. Buna ragmen `plus_reuse_intensity` modeli RMSE'yi **{annual_ctx_summary['models'][0]['rmse_pp']:.2f}** seviyesine getirerek temel modelin (baseline) onune gecti. Bu, geri kullanim ve tuketim yogunlugu bloklarinin aylik modele az da olsa yararli baglam tasidigini gosteriyor.

Ancak `plus_nrw` modelinin RMSE'si **{annual_ctx_summary['models'][3]['rmse_pp']:.2f}** olup daha kotu kaldi. Dolayisiyla gelir getirmeyen su (NRW) su an icin aylik tahminde ana surucu degil; daha cok yonetisim, verimlilik ve politika kaldiraci gibi davranmaktadir.

## 8. Senaryo ve Politika Bulgu Seti

Yillik operasyon verisinden turetilen kaba ama yararli politika kaldiraci analizi 2023 yili icin sunu gosteriyor:

- gelir getirmeyen suda (NRW) 1 yuzde puan azalis: `+{policy_summary['latest_year_levers']['delta_fill_1pp_nrw_reduction_pp']:.2f}` yp esit-deger doluluk
- yetkili tuketimde %1 azalis: `+{policy_summary['latest_year_levers']['delta_fill_1pct_authorized_demand_reduction_pp']:.2f}` yp esit-deger doluluk
- geri kazanilmis suda %10 artis: `+{policy_summary['latest_year_levers']['delta_fill_10pct_reclaimed_increase_pp']:.2f}` yp esit-deger doluluk
- 100 bin aktif abone artisi: `{policy_summary['latest_year_levers']['delta_fill_100k_subscriber_growth_pp']:.2f}` yp esit-deger baski
- geri kazanilmis su payinin %5 seviyesine cikmasi: `+{policy_summary['latest_year_levers']['delta_fill_5pct_reclaimed_share_target_pp']:.2f}` yp esit-deger doluluk

Bu hesaplar dogrudan aylik nedensel tahmin degil; toplam aktif depolama kapasitesi uzerinden yillik esit-deger hacim analizi olarak okunmalidir. [K12][K13][K14][K15][K16][K17][K18][K19]

Ic senaryo simulasyonlari, sistemin hangi tur sarsintilara hassas oldugunu gosteriyor:

{chr(10).join(scenario_lines)}

Bu tablo sunum acisindan guclu; cunku sistemin yalnizca "tahmin eden" degil, ayni zamanda "hangi parametre degisirse ne olur" sorusuna cevap veren bir yapiya donustugunu gosteriyor.

## 9. Mevcut Resmi Durumun Operasyonel Okumasi

Resmi uygulama anlik kesit (public API snapshot) paketi su an 10 ana kaynak icin guncel durum, son 14 gun, son 1 yil ay sonu serisi, ayni gun 10 yillik karsilastirma, yillik yagis ve verilen su katmanlarini uretiyor. API ailesi dogrudan dokumante edilmis bir acik API degil; resmi baraj sayfasinin arayuz paketi (frontend bundle) icinden cikarilip yerel olarak paketlendi. Bu nedenle veri resmi kaynaga bagli, ancak "resmi acik API dokumani" gibi sunulmuyor. [K23]

Bu katmanin pratik faydasi:

- guncel durum tahmini (nowcasting)
- ayni gun yil-yil karsilastirma
- sunumda anlik operasyonel iliski kurma
- kaynak bazli guncel stres gosterimi

## 10. Sinirlar ve Veri Bosluklari

Projenin teknik omurgasi guclu; ancak halen acik kalan bosluklar var:

- dogrudan rezervuar giris akimi (inflow) gozlemi yok
- dogrudan acik su buharlasmasi gozlemi yok
- su kesintisi / ariza / kisit takvimi yogun ve acik bir tarih serisi olarak yok
- sektor bazli cekis ayirimi henuz aylik hacim serisine donusmedi
- aktinograf verisi henuz entegre edilmedi
- 2018 ve 2019 su kaybi formlari var ama image-first PDF oldugu icin OCR gerekiyor

Bu nedenle proje bugun icin en iyi sekilde **aciklanabilir aylik toplam doluluk modeli** ve **senaryo/politika altyapisi** sunuyor; tam fiziksel su butcesi icin bir sonraki turda daha yogun operasyon verisi gerekli.

## 11. Bir Sonraki Teknik Asama

En dogru sira su:

1. Aktinograf verisini ET0 radyasyon terimine entegre etmek
2. Havza alani + yagis birlesimiyle kaynak-bazli giris akimi vekili (inflow proxy) kurmak
3. Su kesintisi, ariza ve zorunlu kisit olay takvimi cikarmak
4. Tarife siniflari ve resmi sektor yapisiyla sektor-bazli talep vekili (demand proxy) kurmak
5. Ardindan 15 yillik projeksiyon icin iklim senaryosu / forcing secmek

Literatur tarafi, reservoir forecasting icin daha karma sequence modellerinin mumkun oldugunu gosteriyor; ancak mevcut kamuya acik veri derinligi dusunuldugunde bugunku yorumlanabilir aylik omurga daha savunulabilir bir baslangic noktasidir. [K5][K6]

## 12. Uretilen Ana Ciktilar

- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_core_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_extended_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_source_current_context.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store`
- `/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub`

## 13. Kaynakca

{refs_md}
"""


def build_pdf(data: dict[str, object]) -> None:
    register_fonts()
    ensure_dirs()
    s = styles()

    bundle_summary = data["bundle_summary"]
    supply_summary = data["supply_summary"]
    reanalysis_summary = data["reanalysis_summary"]
    nao_summary = data["nao_summary"]
    source_summary = data["source_summary"]
    oper_summary = data["oper_summary"]
    policy_summary = data["policy_summary"]
    deep_summary = data["deep_summary"]
    annual_ctx_summary = data["annual_ctx_summary"]
    api_current = data["api_current"]
    api_14d = data["api_14d"]
    coverage = data["coverage"]
    source_current = data["source_current"]
    scenario_summary = data["scenario_summary"]
    deep_research_summary = data["deep_research_summary"]
    hub_status = data["hub_status"]

    current_fill = float(api_current["oran"].iloc[0])
    snapshot_time = str(api_current["snapshot_updated_at"].iloc[0])
    start_14d = float(api_14d["dolulukOrani"].iloc[0])
    end_14d = float(api_14d["dolulukOrani"].iloc[-1])

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=14 * mm,
        title="Istanbul Baraj Doluluk Projesi - Detayli Kaynakcali Rapor",
        author="Codex",
    )

    story: list = []
    story.append(para("Istanbul Baraj Doluluk Projesi - Detayli Kaynakcali Rapor", s["title"]))
    story.append(
        para(
            "Bu rapor, proje boyunca insa edilen veri katmanlarini, modelleme mimarisini, test sonuclarini, senaryo ciktilarini ve mevcut veri bosluklarini tek belgede toplar. Metin 11 Mart 2026 itibariyla mevcut yerel artefaktlar ve resmi kaynaklar uzerinden uretilmistir.",
            s["subtitle"],
        )
    )
    story.append(make_metric_table(s, data))
    story.append(Spacer(1, 8))
    story.append(
        para(
            f"Yonetici ozeti: Proje bugun itibariyla aylik zaman adiminda kapasite agirlikli toplam doluluk hedefi kullanan, resmi ISKI verileri ile dis iklim baglamini birlestiren ve sunumda savunulabilir sayisal cikti uretebilen bir karar-destek omurgasina donusmustur. 11 Mart 2026 tarihli resmi anlik kesite (snapshot) gore toplam doluluk %{current_fill:.1f} ve son 14 gunluk hareket %{start_14d:.2f} -> %{end_14d:.2f} araligindadir. [K23]",
            s["body"],
        )
    )

    story.append(para("1. Projenin Sorusu ve Kurulan Omurga", s["h1"]))
    story.append(
        para(
            "Ana soru, Istanbul toplam baraj dolulugunun yalnizca gecmis doluluk serisi ile degil; yagis, ET0, iklim kosullari, insan kullanim baskisi, operasyonel verimlilik ve kaynak bazli baglam ile birlikte ne kadar daha iyi aciklanip tahmin edilebilecegidir. Daha sade ifadeyle, 'barajlari sadece gecmis doluluk oranina bakarak mi anlamaliyiz, yoksa yagis, hava ve kullanim gibi etkenleri de ayni anda modele katmali miyiz?' sorusuna cevap ariyoruz. Bu nedenle proje tek ana hatta indirildi: toplam doluluk tahmini, risk/mudahale senaryolari ve karar destegi.",
            s["body"],
        )
    )
    add_bullets(
        story,
        [
            "Hedef degisken olarak es ortalama yerine kapasite agirlikli toplam doluluk secildi. Derin testte iki tanim arasindaki ortalama fark 3.25 yuzde puan bulundu.",
            f"Cekirdek model matrisi {bundle_summary['core_start']} -> {bundle_summary['core_end']} araliginda {bundle_summary['core_rows']} aylik gozlem iceriyor.",
            "Ayni omurgaya resmi sehir suyu, yillik operasyon metrikleri, guncel resmi uygulama anlik kesiti (API snapshot), Kandilli yakininda yeniden analiz vekili (reanalysis proxy) ve Kuzey Atlantik Salinimi (NAO) rejim katmani eklendi.",
            f"Arastirma hub'i bugun {hub_status['external_sources_count']} dis kaynak, {hub_status['dataset_inventory_count']} dataset ve {hub_status['artifact_count']} artifact ile izlenebilir durumda.",
        ],
        s["bullet"],
    )

    add_glossary_story(story, s)

    story.append(para("3. Yontem ve Veri Felsefesi", s["h1"]))
    story.append(
        para(
            "ET0 hattinda FAO-56 Penman-Monteith mantigi esas alinmistir; operasyonel yorumlama icin USACE HEC-HMS dokumani ile uyum korunmustur. Sade anlatimla: yagis sisteme su ekleyen taraf, referans evapotranspirasyon (ET0) ise havanin sistemden su cekme baskisini temsil ediyor. ET0 ile acik su buharlasmasi bilerek ayrilmistir; cunku reservoir evaporation literaturu bu iki kavramin farkli fiziksel sistemleri temsil ettigini gosteriyor. [K1][K2][K3][K4]",
            s["body"],
        )
    )
    add_bullets(
        story,
        [
            "Aylik zaman adimi secildi; cunku kamuya acik veri yogunlugu ve karar seviyesindeki sinyal su an en guclu bu skala üzerinde.",
            "Ilk model ailesi yorumlanabilir tutuldu; amac hangi ozellik blogunun (feature block) gercek katkı verdigini acik bicimde gostermekti.",
            "Talep, operasyon ve iklim bloklari ayni tabloda tutuldu ama birbirinin yerine gecen kavramlar olarak sunulmadi.",
            "Kamuya acik veri derinligi yetersiz olan bloklar simdilik senaryo ve baglam degiskeni olarak kullanildi.",
        ],
        s["bullet"],
    )

    story.append(para("4. Toplanan ve Islenen Veri Katmanlari", s["h1"]))
    data_rows = [
        ["Blok", "Kapsam", "Neden onemli"],
        ["Cekirdek aylik tablo (core monthly)", f"{bundle_summary['core_start']} -> {bundle_summary['core_end']}", "Hedef doluluk + yagis + ET0 + tuketim + iklim vekil bloklari"],
        ["Resmi sehir suyu", f"{supply_summary['official_city_supply_window']['start']} -> {supply_summary['official_city_supply_window']['end']}", "Mevcut tuketim vekilinin (proxy) neyi temsil ettigini dogrular"],
        ["Yillik operasyon", "2020 -> 2023 yogun pencere", "Gelir getirmeyen su (NRW), aktif abone, yetkili tuketim ve geri kullanim baglami"],
        ["Kaynak baglami", "17 resmi satir", "Verim, depolama ve havza boyutu ile kaynak-duyarli (source-aware) yorum"],
        ["Yeniden analiz vekili (reanalysis proxy)", f"{reanalysis_summary['request_window']['start']} -> {reanalysis_summary['request_window']['end']}", "Radyasyon, ET0, ruzgar ve guneslenme icin kontrol katmani"],
        ["Kuzey Atlantik Salinimi (NAO) rejimi", f"{nao_summary['nao_start']} -> {nao_summary['nao_end']}", "Kis yagis riski icin dis rejim sinyali"],
        ["Guncel resmi uygulama anlik kesiti", "11 Mart 2026 anlik kesit", "Kisa vadeli guncel durum tahmini (nowcasting) ve operasyonel anlatim"],
    ]
    story.append(make_table(data_rows, [31, 40, 105], s))
    story.append(Spacer(1, 6))
    story.append(
        para(
            f"Resmi sehir suyu serisinin mevcut tuketim vekili (proxy) ile korelasyonu {supply_summary['model_vs_supply']['corr']:.3f} bulundu. Bu, vekilin faturalanmis suyu degil, sisteme verilen toplam suyu daha iyi temsil ettigini gosteriyor. [K16][K17][K18][K19][K20][K21][K22]",
            s["body"],
        )
    )
    story.append(
        para(
            f"Kandilli yakininda yeniden analiz vekili (reanalysis proxy) ile yerel ET0 tarihi arasindaki korelasyon {reanalysis_summary['compare_to_local_et0']['et0_corr']:.3f}; radyasyon korelasyonu ise {reanalysis_summary['compare_to_local_et0']['radiation_month_corr']:.3f}. Sade okumayla bu sunu soyluyor: eldeki ET0 serimiz yalniz degil, dis bir hava veri seti tarafindan da buyuk olcude destekleniyor. Bu nedenle aktinograf gelene kadar bu blok kuvvetli bir kontrol katmani olarak kullanilabilir. [K10][K25]",
            s["body"],
        )
    )
    story.append(
        para(
            f"Kuzey Atlantik Salinimi (NAO) serisinin DJF toplam yagis ile korelasyonu {nao_summary['long_climate_summary']['seasonal_djf_rain_corr']:.3f}. Degisken ana tahminci degil; ama kis risk baglami icin anlamli dis ozellik (feature) adayi. [K26]",
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(para("5. Kapsama ve Hedef Degisken Secimi", s["h1"]))
    coverage_rows = [["Blok", "Tam kapsama", "Aralik"]]
    for _, row in coverage.iterrows():
        coverage_rows.append(
            [
                str(row["block_name"]),
                f"%{row['coverage_pct']:.1f}",
                f"{str(row['start_with_full_block']).split(' ')[0]} -> {str(row['end_with_full_block']).split(' ')[0]}",
            ]
        )
    story.append(make_table(coverage_rows, [35, 25, 115], s))
    story.append(Spacer(1, 6))
    story.append(
        para(
            f"Agirlikli toplam hedefe gecis bu projenin teknik olarak dogru donuslerinden biri oldu. Derin ozet dosyasina gore kapasite agirlikli toplam ile es ortalama arasindaki ortalama fark {deep_research_summary['weighted_vs_mean_avg_diff_pp']:.2f} yuzde puan. Bu boyutta bir fark, Istanbul gibi asimetrik depolama yapisinda toplam su bulunurlugunun es ortalama ile eksik temsil edilebilecegini gosteriyor.",
            s["body"],
        )
    )
    add_figure(
        story,
        ROOT / "output" / "istanbul_dam_deep_research" / "figures" / "weighted_total_vs_mean.png",
        "Sekil 1. Kapasite agirlikli toplam doluluk ile es ortalama doluluk serisinin farki. Bu grafik proje hedef degiskeninin neden agirlikli toplam olarak sabitlendiginin gorsel kanitidir.",
        s,
    )

    story.append(para("6. Modelleme Sonuclari", s["h1"]))
    deep_rows = [["Model", "RMSE", "MAE", "R2", "N"]]
    for _, row in data["deep_metrics"].iterrows():
        deep_rows.append([str(row["model"]), f"{row['rmse_pp']:.2f}", f"{row['mae_pp']:.2f}", f"{row['r2']:.3f}", str(int(row["n_predictions"]))])
    story.append(para("6.1 Gozlenen pencere derin ozellik testi (feature test)", s["h2"]))
    story.append(make_table(deep_rows, [48, 23, 23, 20, 15], s))
    story.append(
        para(
            f"Sik gozlenen pencerede en iyi model `{deep_summary['best_model']}` oldu ve RMSE {deep_summary['best_rmse_pp']:.2f} yuzde puana indi. Buradaki ana ders, depolama hafizasinin hala baskin oldugu; ancak sicaklik ve nem blogunun mevcut `yagis + ET0 + talep + hafiza` yapisina gercek sinyal ekledigi yonundedir. Daha sade soylemle: hava kosullarini biraz daha ayrintili anlattigimizda model biraz daha iyi calisiyor.",
            s["body"],
        )
    )
    add_figure(
        story,
        ROOT / "output" / "newdata_feature_store" / "figures" / "deepened_feature_model_rmse.png",
        "Sekil 2. Gozlenen pencere uzerinde derinlestirilmis ozellik blogu (feature block) karsilastirmasi. Sicaklik ve nem eklemesi temel modele (baseline) kucuk ama kalici iyilesme sagliyor.",
        s,
    )

    annual_rows = [["Model", "RMSE", "MAE", "R2", "N"]]
    for _, row in data["annual_ctx_metrics"].iterrows():
        annual_rows.append([str(row["model"]), f"{row['rmse_pp']:.2f}", f"{row['mae_pp']:.2f}", f"{row['r2']:.3f}", str(int(row["n_predictions"]))])
    story.append(para("6.2 Yillik operasyon baglami testi", s["h2"]))
    story.append(make_table(annual_rows, [52, 23, 23, 20, 15], s))
    story.append(
        para(
            f"Kisa resmi pencere icin en iyi model `plus_reuse_intensity` oldu ve RMSE {annual_ctx_summary['models'][0]['rmse_pp']:.2f} seviyesine geldi. Buna karsilik `plus_nrw` modelinin performansi daha zayif kaldi. Yorum nettir: geri kullanim ve tuketim yogunlugu aylik modelde sinirli ama olumlu baglam saglarken, gelir getirmeyen su (NRW) su an icin daha cok politika ve verimlilik degiskenidir.",
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(para("7. Operasyonel ve Politika Bulgulari", s["h1"]))
    story.append(
        para(
            f"11 Mart 2026 tarihli resmi ISKI anlik kesiti (snapshot) toplam dolulugu %{current_fill:.1f} olarak veriyor. Son 14 gunluk seri ayni kesite gore %{start_14d:.2f} seviyesinden %{end_14d:.2f} seviyesine yukselmis durumda. Bu guncel katman, uzun donem aylik model ile birlikte kisa ufuklu guncel durum tahmini (nowcasting) konusmasini mumkun kiliyor. [K23]",
            s["body"],
        )
    )
    add_figure(
        story,
        ROOT / "output" / "newdata_feature_store" / "figures" / "official_supply_vs_model_consumption.png",
        "Sekil 3. Resmi sehir suyu serisi ile proje tuketim proxy'sinin karsilastirmasi. Yakin uyum, mevcut talep serisinin sistem cekisini izledigini gosteriyor.",
        s,
    )

    policy_levers = policy_summary["latest_year_levers"]
    lever_rows = [
        ["Kaldirac", "2023 esit-deger etki (yp)"],
        ["Gelir getirmeyen su (NRW) -1 yuzde puan", f"{policy_levers['delta_fill_1pp_nrw_reduction_pp']:+.2f}"],
        ["Yetkili tuketim -%1", f"{policy_levers['delta_fill_1pct_authorized_demand_reduction_pp']:+.2f}"],
        ["Geri kazanilmis su +%10", f"{policy_levers['delta_fill_10pct_reclaimed_increase_pp']:+.2f}"],
        ["+100 bin aktif abone", f"{policy_levers['delta_fill_100k_subscriber_growth_pp']:+.2f}"],
        ["Geri kullanim payi %5 hedefi", f"{policy_levers['delta_fill_5pct_reclaimed_share_target_pp']:+.2f}"],
    ]
    story.append(make_table(lever_rows, [95, 80], s))
    story.append(
        para(
            "Bu tablo yillik esit-deger hacim mantigiyla okunmali. Dogrudan aylik nedensel katsayi degildir; ancak hangi operasyon kaldiracinin sehir olceginde daha buyuk etkisi olabilecegini hizli siralamak icin kuvvetlidir. [K12][K13][K14][K15][K16][K17][K18][K19]",
            s["body"],
        )
    )
    add_figure(
        story,
        ROOT / "output" / "newdata_feature_store" / "figures" / "official_policy_leverage_latest_year.png",
        "Sekil 4. 2023 icin resmi politika kaldiraclarinin toplam doluluk esit-deger etkileri. Amaç aylik nedensellik kurmak degil, etkileri boyut olarak siralamaktir.",
        s,
    )

    story.append(para("7.1 Ic senaryo deneyleri", s["h2"]))
    scenario_rows = [["Senaryo", "3. ay etki", "6. ay etki"]]
    selected = {
        "rain_plus10_3m": "Yagis +%10",
        "et0_plus10_3m": "ET0 +%10",
        "cons_plus10_3m": "Tuketim +%10",
        "restriction_minus15_3m": "Talep -%15 kisit",
        "hot_dry_high_demand": "Sicak-kurak-yuksek talep",
    }
    for key, label in selected.items():
        sub = scenario_summary[scenario_summary["scenario"] == key].set_index("horizon_month")
        scenario_rows.append([label, f"{sub.loc[3, 'delta_pp']:+.2f}", f"{sub.loc[6, 'delta_pp']:+.2f}"])
    story.append(make_table(scenario_rows, [80, 35, 35], s))
    story.append(
        para(
            "Ic senaryo denemeleri, sistemin yagis artisi ve talep kisiti ile rahatladigini; sicak-kurak-yuksek talep kombinasyonunda ise hizli baski altina girdigini gosteriyor. Bu katman projeyi yalnizca tahmin ureten degil, parametre degisimi karsisinda sistem davranisi tartisan bir araca donusturuyor.",
            s["body"],
        )
    )

    story.append(para("8. Kaynak Bazli Guncel Gorunum", s["h1"]))
    src_rows = [["Kaynak", "Doluluk", "Yillik verim", "Azami depolama", "Mevcut/yillik verim"]]
    view = source_current.sort_values("dolulukOrani", ascending=False).head(7)
    for _, row in view.iterrows():
        src_rows.append(
            [
                str(row["baslikAdi"]),
                f"%{row['dolulukOrani']:.2f}",
                f"{row['annual_yield_million_m3']:.0f} milyon m3",
                f"{row['max_storage_million_m3']:.1f} milyon m3",
                f"{row['current_storage_to_yield_ratio']:.2f}",
            ]
        )
    story.append(make_table(src_rows, [37, 20, 34, 34, 40], s))
    story.append(
        para(
            f"Kaynak baglam tablosu su an {source_summary['row_count']} resmi satirdan uretiliyor. Bu katman, toplam doluluk yuzdesinin arkasinda birbirinden cok farkli depolama-verim oranlari oldugunu gosteriyor. Ornegin Elmalı guncel snapshot'ta yuksek dolulukta gorunurken, Omurga acisindan Omerli gibi buyuk hacimli kaynaklarin davranisi sehir capinda daha kritik agirlik tasiyor. [K24]",
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(para("9. Sinirlar, Riskler ve Sonraki Asama", s["h1"]))
    add_bullets(
        story,
        [
            "Dogrudan rezervuar giris akimi (inflow) gozlemi yok; bu nedenle yagis -> giris akimi gecisi henuz vekil (proxy) mantiginda.",
            "Acik su buharlasmasi gozlemi yok; ET0 ile fiziksel rezervuar yuzey buharlasmasi ayri tutuluyor ama rezervuar-alan bazli dogrudan seri henuz uretilmedi.",
            "Su kesintisi, ariza ve zorunlu kisit olay takvimi henuz yogun ve makine-islenebilir bir tarih serisine donusmedi.",
            "Sektor bazli aylik cekis serisi yok; resmi tarife yapisi biliniyor ama hacimsel ayrim kamuya acik veriyle henuz tamam degil. [K27]",
            "Aktinograf verisi geldigi anda ET0 radyasyon terimine dogrudan baglanacak; bu entegrasyon WMO olcum prensipleri ile uyumlu yurutilmeli. [K10]",
            "2018 ve 2019 su kaybi formlari var fakat OCR gerektiriyor; dolayisiyla uzun donem yillik kayip serisi henuz eksik.",
        ],
        s["bullet"],
    )
    story.append(
        para(
            "Bugun icin en mantikli teknik sira su: aktinografi ET0 hattina baglamak, havza alaniyla yagisi birlestirip kaynak-bazli giris akimi vekili (inflow proxy) kurmak, ariza-kisit olay takvimi cikarmak ve ancak bundan sonra 15 yillik iklim senaryolari ile ileri projeksiyon motoruna gecmek. Reservoir forecasting literaturu sequence modellerini destekliyor, ancak mevcut kamuya acik veri derinligi dusunuldugunde yorumlanabilir aylik omurga daha savunulabilir ilk adimdir. [K5][K6][K7][K8][K9]",
            s["body"],
        )
    )

    story.append(para("10. Ana Uretilen Dosyalar", s["h1"]))
    add_bullets(
        story,
        [
            "/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_core_monthly.csv",
            "/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_extended_monthly.csv",
            "/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_source_current_context.csv",
            "/Users/yasinkaya/Hackhaton/output/newdata_feature_store",
            "/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot",
            "/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub",
        ],
        s["bullet"],
    )

    story.append(para("11. Kaynakca", s["h1"]))
    for ref in REFERENCES:
        title = escape(ref.title)
        url = escape(ref.url)
        note = escape(ref.note)
        story.append(para(f"<b>[{ref.code}]</b> {title}. {note}. {url}", s["ref"]))

    story.append(Spacer(1, 4))
    story.append(
        para(
            f"Rapor olusturma zamani: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            s["small"],
        )
    )

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def update_artifact_catalog() -> None:
    artifact_csv = ROOT / "research" / "baraj_doluluk_hub" / "artifacts" / "artifact_catalog.csv"
    with artifact_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = f.readline().strip().split(",")

    existing_paths = {row["path"] for row in rows}
    additions = [
        {
            "artifact_id": "ART-043",
            "artifact_type": "report",
            "path": str(OUT_MD),
            "status": "active",
            "owner_domain": "deliverable",
            "description": "Detailed markdown report covering project scope data modeling findings and bibliography",
        },
        {
            "artifact_id": "ART-044",
            "artifact_type": "report",
            "path": str(OUT_PDF),
            "status": "active",
            "owner_domain": "deliverable",
            "description": "Detailed PDF report covering project scope data modeling findings and bibliography",
        },
    ]
    changed = False
    for addition in additions:
        if addition["path"] not in existing_paths:
            rows.append(addition)
            changed = True
    if changed:
        with artifact_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["artifact_id", "artifact_type", "path", "status", "owner_domain", "description"])
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    data = load_inputs()
    OUT_MD.write_text(build_markdown(data), encoding="utf-8")
    build_pdf(data)
    update_artifact_catalog()
    print(OUT_MD)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
