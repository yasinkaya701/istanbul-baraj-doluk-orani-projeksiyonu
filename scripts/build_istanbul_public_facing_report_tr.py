#!/usr/bin/env python3
from __future__ import annotations

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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path("/Users/yasinkaya/Hackhaton")
OUT_PDF = ROOT / "output" / "pdf" / "istanbul_baraj_proje_detayli_rapor.pdf"
OUT_MD = ROOT / "output" / "report" / "istanbul_baraj_proje_detayli_rapor.md"
FIG_DIR = ROOT / "output" / "report" / "figures_tr"
FONT_DIR = Path("/opt/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf")


@dataclass(frozen=True)
class Ref:
    code: str
    title: str
    url: str


REFS = [
    Ref("K1", "FAO-56 Crop Evapotranspiration Chapter 2", "https://www.fao.org/4/X0490E/x0490e06.htm"),
    Ref("K2", "USACE HEC-HMS Penman-Monteith Method", "https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/evaporation-and-transpiration/penman-monteith-method"),
    Ref("K3", "Reservoir evaporation and water availability under climate stress", "https://hess.copernicus.org/articles/28/3243/2024/index.html"),
    Ref("K4", "Open-water evaporation and water-surface response", "https://hess.copernicus.org/articles/30/67/2026/"),
    Ref("K5", "İSKİ Baraj Doluluk Oranları", "https://iski.istanbul/baraj-doluluk"),
    Ref("K6", "İSKİ Su Kaynakları", "https://iski.istanbul/kurumsal/hakkimizda/su-kaynaklari"),
    Ref("K7", "İSKİ 2020 Faaliyet Raporu", "https://cdn.iski.istanbul/uploads/2020_FAALIYET_RAPORU_903efe0267.pdf"),
    Ref("K8", "İSKİ 2021 Faaliyet Raporu", "https://cdn.iski.istanbul/uploads/2021_FAALIYET_RAPORU_64bf206f27.pdf"),
    Ref("K9", "İSKİ 2022 Faaliyet Raporu", "https://cdn.iski.istanbul/uploads/2022_Faaliyet_Raporu_c65c8a733d.pdf"),
    Ref("K10", "İSKİ 2023 Faaliyet Raporu", "https://iskiapi.iski.gov.tr/uploads/2023_Yili_Faaliyet_Raporu_24309dd9dd.pdf"),
    Ref("K11", "İSKİ 2020 Standart Su Dengesi Formu", "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2020_9a984f0ba7.pdf"),
    Ref("K12", "İSKİ 2021 Standart Su Dengesi Formu", "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2021_9e4b97ee29.pdf"),
    Ref("K13", "İSKİ 2022 Standart Su Dengesi Formu", "https://cdn.iski.istanbul/uploads/Su_Denge_Tablosu_2022_46b2a9477c.pdf"),
    Ref("K14", "İSKİ 2023 Standart Su Dengesi Formu", "https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2023_4c07821536.pdf"),
    Ref("K15", "Open-Meteo Historical Weather API", "https://open-meteo.com/en/docs/historical-weather-api"),
    Ref("K16", "NOAA CPC Monthly NAO Index", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"),
]


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="DejaVuSans-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#0f172a"), spaceAfter=8),
        "subtitle": ParagraphStyle("subtitle", parent=base["BodyText"], fontName="DejaVuSans", fontSize=10.3, leading=14, textColor=colors.HexColor("#334155"), spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="DejaVuSans-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#0f172a"), spaceBefore=8, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="DejaVuSans-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#0f172a"), spaceBefore=7, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="DejaVuSans", fontSize=9.6, leading=13.5, alignment=TA_JUSTIFY, textColor=colors.HexColor("#111827"), spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName="DejaVuSans", fontSize=9.3, leading=13, leftIndent=12, bulletIndent=0, textColor=colors.HexColor("#111827"), spaceAfter=2),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="DejaVuSans", fontSize=8.2, leading=10.5, textColor=colors.HexColor("#475569"), spaceAfter=2),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontName="DejaVuSans", fontSize=8.3, leading=10.5, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceBefore=2, spaceAfter=8),
        "ref": ParagraphStyle("ref", parent=base["BodyText"], fontName="DejaVuSans", fontSize=8.4, leading=10.6, textColor=colors.HexColor("#111827"), alignment=TA_LEFT, spaceAfter=4),
        "metric": ParagraphStyle("metric", parent=base["BodyText"], fontName="DejaVuSans-Bold", fontSize=17, leading=18, textColor=colors.white),
        "metric_label": ParagraphStyle("metric_label", parent=base["BodyText"], fontName="DejaVuSans", fontSize=8.4, leading=10, textColor=colors.white),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullets(story: list, items: list[str], style: ParagraphStyle) -> None:
    for item in items:
        story.append(Paragraph(item, style, bulletText="-"))


def fit_image(path: Path, max_width_mm: float, max_height_mm: float) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    scale = min((max_width_mm * mm) / width, (max_height_mm * mm) / height)
    return Image(str(path), width=width * scale, height=height * scale)


def add_figure(story: list, filename: str, caption: str, styles: dict[str, ParagraphStyle], height_mm: float = 80) -> None:
    story.append(fit_image(FIG_DIR / filename, 176, height_mm))
    story.append(p(caption, styles["caption"]))


def table(rows: list[list[str]], widths_mm: list[float], styles: dict[str, ParagraphStyle]) -> Table:
    wrapped = []
    for i, row in enumerate(rows):
        style = styles["small"] if i == 0 else styles["small"]
        wrapped.append([p(cell, style) for cell in row])
    t = Table(wrapped, colWidths=[w * mm for w in widths_mm])
    t.setStyle(
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
    return t


def metric_table(styles: dict[str, ParagraphStyle], current_fill: float, deep_rmse: float, supply_corr: float, reanalysis_corr: float, diff_pp: float, source_count: int) -> Table:
    nums = [f"%{current_fill:.1f}", f"{deep_rmse:.2f}", f"{supply_corr:.3f}", f"{reanalysis_corr:.3f}", f"{diff_pp:.2f}", str(source_count)]
    labels = [
        "11 Mart 2026 resmî toplam doluluk",
        "En iyi model hatası (RMSE)",
        "Resmî arz ile tüketim vekili uyumu",
        "Yeniden analiz ET0 ile yerel ET0 uyumu",
        "Ağırlıklı toplam ile eşit ortalama farkı",
        "Kayıtlı dış kaynak sayısı",
    ]
    rows = []
    for i in range(0, len(nums), 3):
        row_nums = [p(x, styles["metric"]) for x in nums[i : i + 3]]
        row_labels = [p(x, styles["metric_label"]) for x in labels[i : i + 3]]
        while len(row_nums) < 3:
            row_nums.append(p("", styles["metric"]))
            row_labels.append(p("", styles["metric_label"]))
        rows.extend([row_nums, row_labels])
    t = Table(rows, colWidths=[57 * mm, 57 * mm, 57 * mm], rowHeights=[10 * mm, 15 * mm, 10 * mm, 15 * mm])
    t.setStyle(
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
    return t


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DejaVuSans", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(16 * mm, 287 * mm, "İstanbul Baraj Doluluk Projesi - Kamuya Anlaşılır Rapor")
    canvas.drawRightString(194 * mm, 10 * mm, f"Sayfa {doc.page}")
    canvas.restoreState()


def build_markdown(data: dict) -> str:
    policy = data["policy_summary"]["latest_year_levers"]
    return f"""# İstanbul Baraj Doluluk Projesi - Kamuya Anlaşılır Rapor

Oluşturulma tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}

## 1. Bu proje neyi amaçlıyor?

Bu çalışma, İstanbul'un toplam baraj doluluk oranını sadece geçmiş doluluk değerlerine bakarak değil; yağış, referans evapotranspirasyon (ET0), sıcaklık, nem, su kullanımı ve resmî işletme verileriyle birlikte anlamayı hedefliyor.

## 2. Şu an elimizde ne var?

- Güncel resmî doluluk verisi
- Uzun dönem aylık doluluk serisi
- Yağış ve ET0 verileri
- Resmî şehir suyu verisi
- Su kaybı ve geri kazanılmış su verileri
- Baraj bazlı kapasite ve verim bilgileri
- Kısa vadeli senaryo deneyleri

## 3. En önemli bulgular

- 11 Mart 2026 resmî toplam doluluk: %{data['current_fill']:.1f}
- En iyi model hatası: {data['deep_rmse']:.2f} yüzde puan
- Resmî arz ile tüketim vekili uyumu: {data['supply_corr']:.3f}
- Ağırlıklı toplam ile eşit ortalama farkı: {data['diff_pp']:.2f} yüzde puan

## 4. Politika açısından ne anlama geliyor?

- Gelir getirmeyen su (NRW) 1 yüzde puan azalırsa yaklaşık +{policy['delta_fill_1pp_nrw_reduction_pp']:.2f} yüzde puan eşdeğer rahatlama görülebilir.
- Yetkili tüketim %1 azalırsa yaklaşık +{policy['delta_fill_1pct_authorized_demand_reduction_pp']:.2f} yüzde puan eşdeğer rahatlama görülebilir.
- Geri kazanılmış su %10 artarsa yaklaşık +{policy['delta_fill_10pct_reclaimed_increase_pp']:.2f} yüzde puan eşdeğer katkı oluşabilir.

## 5. Sonraki adımlar

1. Aktinograf verisini ET0 hattına bağlamak
2. Havza alanı ile yağışı birleştirip giriş akımı vekili kurmak
3. Su kesintisi ve arıza takvimini toplamak
4. Sektör bazlı talep ayrımını güçlendirmek

## 6. Kaynakça

""" + "\n".join(f"- **[{r.code}]** {r.title}: {r.url}" for r in REFS) + "\n"


def main() -> None:
    register_fonts()
    styles = build_styles()

    deep_summary = json.loads((ROOT / "output" / "newdata_feature_store" / "deepened_feature_summary.json").read_text(encoding="utf-8"))
    supply_summary = json.loads((ROOT / "output" / "newdata_feature_store" / "official_monthly_supply_context_summary.json").read_text(encoding="utf-8"))
    reanalysis_summary = json.loads((ROOT / "output" / "newdata_feature_store" / "kandilli_openmeteo_reanalysis_summary.json").read_text(encoding="utf-8"))
    policy_summary = json.loads((ROOT / "output" / "newdata_feature_store" / "official_policy_leverage_summary.json").read_text(encoding="utf-8"))
    hub_status = json.loads((ROOT / "research" / "baraj_doluluk_hub" / "admin" / "HUB_STATUS.json").read_text(encoding="utf-8"))
    deep_research_summary = json.loads((ROOT / "output" / "istanbul_dam_deep_research" / "summary.json").read_text(encoding="utf-8"))

    current_df = pd.read_csv(ROOT / "output" / "iski_baraj_api_snapshot" / "tables" / "genel_oran.csv")
    supply_df = pd.read_csv(ROOT / "output" / "newdata_feature_store" / "tables" / "official_supply_vs_model_consumption_monthly.csv")
    source_df = pd.read_csv(ROOT / "output" / "model_useful_data_bundle" / "tables" / "istanbul_source_current_context.csv")
    scenario_df = pd.read_csv(ROOT / "output" / "istanbul_dam_deep_research" / "scenario_summary.csv")
    coverage_df = pd.read_csv(ROOT / "output" / "model_useful_data_bundle" / "tables" / "istanbul_model_feature_block_coverage.csv")
    current_fill = float(current_df["oran"].iloc[0])
    snapshot_time = str(current_df["snapshot_updated_at"].iloc[0])

    supply_sub = supply_df.dropna(subset=["city_supply_m3_month_official", "model_consumption_m3_month"])
    start_supply = supply_sub["date"].iloc[0][:10]
    end_supply = supply_sub["date"].iloc[-1][:10]

    top_sources = source_df.sort_values("dolulukOrani", ascending=False).head(5)
    top_source_lines = [
        f"{row['baslikAdi']}: %{row['dolulukOrani']:.2f} doluluk, {row['max_storage_million_m3']:.1f} milyon m³ azami depolama"
        for _, row in top_sources.iterrows()
    ]

    selected = {
        "rain_plus10_3m": "Yağış %10 artarsa",
        "et0_plus10_3m": "ET0 %10 artarsa",
        "cons_plus10_3m": "Tüketim %10 artarsa",
        "restriction_minus15_3m": "Talep %15 kısılırsa",
        "hot_dry_high_demand": "Sıcak-kurak-yüksek talep birlikte olursa",
    }
    scenario_lines = []
    for key, label in selected.items():
        sub = scenario_df[scenario_df["scenario"] == key].set_index("horizon_month")
        scenario_lines.append(f"{label}: 3. ayda {sub.loc[3, 'delta_pp']:+.2f}, 6. ayda {sub.loc[6, 'delta_pp']:+.2f} yüzde puan etki.")

    coverage_rows = [["Blok", "Tam kapsama", "Aralık"]]
    for _, row in coverage_df.iterrows():
        coverage_rows.append([str(row["block_name"]), f"%{row['coverage_pct']:.1f}", f"{str(row['start_with_full_block']).split(' ')[0]} - {str(row['end_with_full_block']).split(' ')[0]}"])

    story: list = []
    story.append(p("İstanbul Baraj Doluluk Projesi - Daha Anlaşılır ve Görsel Rapor", styles["title"]))
    story.append(
        p(
            "Bu sürüm, teknik terimleri sadeleştirerek, grafik başlıklarını Türkçeleştirerek ve görselliği artırarak hazırlandı. Amaç, alan dışındaki bir kişinin de raporu okuyup sunabilecek düzeyde anlayabilmesidir.",
            styles["subtitle"],
        )
    )
    story.append(
        metric_table(
            styles,
            current_fill=current_fill,
            deep_rmse=deep_summary["best_rmse_pp"],
            supply_corr=supply_summary["model_vs_supply"]["corr"],
            reanalysis_corr=reanalysis_summary["compare_to_local_et0"]["et0_corr"],
            diff_pp=deep_research_summary["weighted_vs_mean_avg_diff_pp"],
            source_count=hub_status["external_sources_count"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(p("1. Proje Ne Yapıyor?", styles["h1"]))
    story.append(
        p(
            "Bu proje, İstanbul'daki toplam baraj doluluğunu sadece geçmiş doluluk serisine bakarak tahmin etmeye çalışmıyor. Bunun yerine şu soruyu soruyor: Yağış, sıcaklık, buharlaşma baskısı, şehirdeki su kullanımı ve işletme kararları birlikte ele alındığında baraj sistemi daha doğru anlaşılabilir mi?",
            styles["body"],
        )
    )
    bullets(
        story,
        [
            "Hedef değişken olarak kapasite ağırlıklı toplam doluluk kullanılıyor. Böylece büyük barajlar küçük barajlarla aynı ağırlığa sahipmiş gibi davranılmıyor.",
            "Aylık zaman adımı seçildi. Çünkü kamuya açık verilerin en güçlü ve en düzenli olduğu ölçek bu.",
            "Model yalnızca tahmin üretmek için değil; hangi değişkenin sistemi nasıl etkilediğini göstermek için kuruldu.",
        ],
        styles["bullet"],
    )

    story.append(p("2. Kısa Terimler Rehberi", styles["h1"]))
    bullets(
        story,
        [
            "Referans evapotranspirasyon (ET0): Havanın standart bir yüzeyden ne kadar su çekme eğilimi gösterdiğini anlatır.",
            "Gelir getirmeyen su (NRW): Sisteme verilen ama faturalanamayan veya kayıp olan sudur.",
            "Yeniden analiz verisi (reanalysis): Ölçüm ve hava modelinin birleşmesiyle elde edilen tarihsel hava verisidir.",
            "Vekil değişken (proxy): Doğrudan ölçemediğimiz bir şeyi dolaylı temsil eden değişkendir.",
            "Kısa vadeli güncel durum tahmini (nowcasting): Çok kısa vadeli güncel durum okumasıdır.",
        ],
        styles["bullet"],
    )

    story.append(p("3. Elimizde Hangi Veri Katmanları Var?", styles["h1"]))
    data_rows = [
        ["Veri bloğu", "Kapsam", "Ne işe yarıyor?"],
        ["Çekirdek aylık tablo", "2000-10 - 2024-02", "Doluluk, yağış, ET0, tüketim ve temel iklim bilgilerini birlikte tutar."],
        ["Resmî şehir suyu verisi", "2010-01 - 2023-12", "Tüketim vekilinin gerçekten neyi temsil ettiğini kontrol eder."],
        ["Yıllık işletme verileri", "2020 - 2023", "Su kaybı, geri kazanım ve abone baskısını gösterir."],
        ["Baraj bazlı kaynak tablosu", "Güncel", "Hangi barajın kapasite ve verim açısından daha kritik olduğunu gösterir."],
        ["Yeniden analiz hava verisi", "1940 - 2026", "ET0 ve radyasyon tarafında kontrol katmanı sağlar."],
        ["Kuzey Atlantik Salınımı (NAO)", "1950 - 2026", "Kış rejimini anlamaya yardım eder."],
    ]
    story.append(table(data_rows, [33, 36, 107], styles))
    story.append(Spacer(1, 6))
    story.append(
        p(
            f"Resmî şehir suyu ile model tüketim vekili arasındaki korelasyon {supply_summary['model_vs_supply']['corr']:.3f}. Bu, mevcut tüketim serisinin büyük ölçüde sisteme verilen suyu izlediğini gösteriyor. Resmî arz verisinin aylık kapsama aralığı {start_supply} ile {end_supply} arasında.",
            styles["body"],
        )
    )
    story.append(table(coverage_rows, [35, 26, 113], styles))

    story.append(PageBreak())
    story.append(p("4. Şu An Sistem Ne Söylüyor?", styles["h1"]))
    story.append(
        p(
            f"11 Mart 2026 tarihli resmî güncellemede İstanbul toplam baraj doluluğu %{current_fill:.1f}. Bu değer {snapshot_time} zaman damgasıyla kaydedildi. Aşağıdaki grafik, son 14 gün içindeki hareketi doğrudan gösteriyor.",
            styles["body"],
        )
    )
    add_figure(
        story,
        "son_14_gun_doluluk_tr.png",
        "Şekil 1. Son 14 günde toplam doluluk artış eğiliminde. Bu grafik, sunumda güncel durumu tek cümlede anlatmak için kullanılabilir.",
        styles,
        72,
    )
    story.append(
        p(
            "Aşağıdaki baraj bazlı görsel ise aynı gün itibarıyla hangi kaynağın daha dolu, hangisinin daha zayıf durumda olduğunu gösteriyor. Sağdaki küçük etiketler, yalnızca doluluk oranını değil, o kaynağın toplam kapasitesini de akılda tutmak için eklendi.",
            styles["body"],
        )
    )
    add_figure(
        story,
        "kaynak_bazli_guncel_durum_tr.png",
        "Şekil 2. Kaynak bazında güncel doluluk durumu. Yalnızca yüzdeye değil, barajın toplam kapasitesine de bakmak gerekir.",
        styles,
        82,
    )
    bullets(story, top_source_lines, styles["bullet"])

    story.append(p("5. Neden Ağırlıklı Toplam Kullanıyoruz?", styles["h1"]))
    story.append(
        p(
            f"Eğer tüm barajları eşit ağırlıklı ortalama ile toplarsak, küçük hacimli bir baraj ile çok büyük hacimli bir barajı aynı etkiye sahipmiş gibi saymış oluruz. Bu projede yapılan karşılaştırmada eşit ortalama ile kapasite ağırlıklı toplam arasında ortalama {deep_research_summary['weighted_vs_mean_avg_diff_pp']:.2f} yüzde puan fark görüldü. Bu nedenle şehir ölçeğinde daha doğru temsil için kapasite ağırlıklı toplam kullanıldı.",
            styles["body"],
        )
    )
    add_figure(
        story,
        "agirlikli_vs_esit_ortalama_tr.png",
        "Şekil 3. Aynı sistemi iki farklı toplama yöntemiyle görmek farklı sonuç veriyor. Şehir ölçeğinde daha doğru olan yaklaşım kapasite ağırlıklı toplamdır.",
        styles,
    )

    story.append(PageBreak())
    story.append(p("6. Model Ne Kadar İşe Yarıyor?", styles["h1"]))
    story.append(
        p(
            f"Sık gözlenen veri penceresinde en iyi model hatası {deep_summary['best_rmse_pp']:.2f} yüzde puan seviyesine kadar indi. Bu, kaba bir seri takibinden daha güçlü bir yapı kurulduğunu gösteriyor. Aşağıdaki grafik model seçeneklerini karşılaştırıyor.",
            styles["body"],
        )
    )
    add_figure(
        story,
        "model_karsilastirma_tr.png",
        "Şekil 4. Sıcaklık ve nem bilgisi eklendiğinde temel modele göre küçük ama kalıcı bir iyileşme görülüyor.",
        styles,
        72,
    )
    bullets(
        story,
        [
            "Depolama hafızası hâlâ en güçlü bilgi bloğu.",
            "Yağış, toplam tüketime göre daha net açıklayıcı değer taşıyor.",
            "Sıcaklık ve nem eklenince hata biraz daha düşüyor.",
            "Bu sonuç, sistemin sadece yağışla değil atmosferin kurutucu etkisiyle de açıklanması gerektiğini gösteriyor.",
        ],
        styles["bullet"],
    )

    story.append(p("7. Tüketim ve İşletme Tarafı Neden Önemli?", styles["h1"]))
    story.append(
        p(
            "Resmî şehir suyu ile model tüketim vekilinin yakın gitmesi, insan kullanım baskısının modele gerçek bir sinyal taşıdığını gösteriyor. Yani bu çalışma sadece yağmura bakan bir model değil; şehirdeki kullanım baskısını da izliyor.",
            styles["body"],
        )
    )
    add_figure(
        story,
        "resmi_arz_ve_tuketim_tr.png",
        "Şekil 5. Model tüketim vekili, resmî şehir suyu serisini büyük ölçüde izliyor. Bu, talep tarafının modelde boşa durmadığını gösteriyor.",
        styles,
    )
    lever = policy_summary["latest_year_levers"]
    lever_rows = [
        ["Politika veya işletme kolu", "2023 için yaklaşık eşdeğer etki"],
        ["Gelir getirmeyen su 1 yüzde puan azalırsa", f"+{lever['delta_fill_1pp_nrw_reduction_pp']:.2f} yüzde puan"],
        ["Yetkili tüketim %1 azalırsa", f"+{lever['delta_fill_1pct_authorized_demand_reduction_pp']:.2f} yüzde puan"],
        ["Geri kazanılmış su %10 artarsa", f"+{lever['delta_fill_10pct_reclaimed_increase_pp']:.2f} yüzde puan"],
        ["100 bin aktif abone eklenirse", f"{lever['delta_fill_100k_subscriber_growth_pp']:.2f} yüzde puan"],
        ["Geri kullanım payı %5 hedefine çıkarsa", f"+{lever['delta_fill_5pct_reclaimed_share_target_pp']:.2f} yüzde puan"],
    ]
    story.append(table(lever_rows, [100, 74], styles))
    story.append(
        p(
            "Bu tablo doğrudan aylık nedensellik iddiası taşımıyor. Ama hangi yönetim kaldıraçlarının sistem üzerinde daha büyük etki yaratabileceğini hızlıca sıralamak için çok değerli.",
            styles["body"],
        )
    )

    story.append(p("8. Senaryo Deneyleri Bize Ne Anlatıyor?", styles["h1"]))
    story.append(
        p(
            "Bu bölümde tek tek şunu soruyoruz: Yağış artsa ne olur, ET0 artsa ne olur, tüketim artsa ne olur, talep kısılsa ne olur? Böylece model sadece tahmin eden değil, aynı zamanda açıklayan bir araç hâline geliyor.",
            styles["body"],
        )
    )
    add_figure(
        story,
        "senaryo_etkileri_tr.png",
        "Şekil 6. Senaryo denemeleri, hangi şokun sistemi rahatlatıp hangisinin baskıladığını doğrudan gösteriyor.",
        styles,
        78,
    )
    bullets(story, scenario_lines, styles["bullet"])

    story.append(p("9. Bu Çalışmanın Şu Anki Sınırları", styles["h1"]))
    bullets(
        story,
        [
            "Barajlara doğrudan giren su miktarı için kamuya açık yoğun bir ölçüm serisi yok.",
            "Baraj yüzeyinden olan açık su buharlaşması için doğrudan gözlem yok.",
            "Su kesintisi, arıza ve zorunlu kısıt olayları henüz yoğun bir tarih serisine dönüştürülmedi.",
            "Sektör bazlı aylık çekiş ayrımı henüz tam kurulmadı.",
            "Aktinograf verisi henüz ET0 hattına eklenmedi.",
        ],
        styles["bullet"],
    )

    story.append(p("10. Bundan Sonra Ne Yapılmalı?", styles["h1"]))
    bullets(
        story,
        [
            "Aktinograf verisini doğrudan ET0 hesaplarına bağlamak",
            "Havza alanı ile yağışı birleştirip kaynak bazlı giriş akımı vekili kurmak",
            "Su kesintisi ve arıza takvimini olay serisi hâline getirmek",
            "Tarife ve sektör bilgisiyle daha güçlü bir talep ayrımı kurmak",
            "Bu temelden sonra 15 yıllık projeksiyon motoruna geçmek",
        ],
        styles["bullet"],
    )
    story.append(
        p(
            "Kısacası bu proje artık sadece 'baraj doluluğunu tahmin etmeye çalışan bir fikir' değil. Veri temeli kurulmuş, resmî kaynaklarla desteklenmiş, güncel durumu gösterebilen ve senaryo konuşabilen bir model omurgası hâline geldi.",
            styles["body"],
        )
    )

    story.append(p("11. Kaynakça", styles["h1"]))
    for ref in REFS:
        story.append(p(f"<b>[{ref.code}]</b> {ref.title}: {ref.url}", styles["ref"]))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=14 * mm,
        title="İstanbul Baraj Doluluk Projesi - Daha Anlaşılır ve Görsel Rapor",
        author="Codex",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    OUT_MD.write_text(
        build_markdown(
            {
                "current_fill": current_fill,
                "deep_rmse": deep_summary["best_rmse_pp"],
                "supply_corr": supply_summary["model_vs_supply"]["corr"],
                "diff_pp": deep_research_summary["weighted_vs_mean_avg_diff_pp"],
                "policy_summary": policy_summary,
            }
        ),
        encoding="utf-8",
    )
    print(OUT_PDF)
    print(OUT_MD)


if __name__ == "__main__":
    main()
