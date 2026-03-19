#!/usr/bin/env python3
from __future__ import annotations

import math
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

ROOT = Path('/Users/yasinkaya/Hackhaton')
PROJ_DIR = ROOT / 'output' / 'istanbul_dam_forward_projection_2040'
FIG_DIR = PROJ_DIR / 'figures'
OUT_MD = ROOT / 'output' / 'report' / 'istanbul_baraj_2026_2040_projeksiyon_raporu.md'
OUT_PDF = ROOT / 'output' / 'pdf' / 'istanbul_baraj_2026_2040_projeksiyon_raporu.pdf'
FONT_DIR = Path('/opt/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf')
SOURCE_REGISTRY = ROOT / 'research' / 'baraj_doluluk_hub' / 'registry' / 'sources' / 'external_sources.csv'
HUB_STATUS = ROOT / 'research' / 'baraj_doluluk_hub' / 'admin' / 'HUB_STATUS.json'

USED_SOURCE_IDS = [
    'SRC-014', 'SRC-057', 'SRC-059', 'SRC-064', 'SRC-066', 'SRC-067', 'SRC-068',
    'SRC-069', 'SRC-073', 'SRC-074'
]


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont('DejaVuSans', str(FONT_DIR / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', str(FONT_DIR / 'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DejaVuSans', normal='DejaVuSans', bold='DejaVuSans-Bold')


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('title', parent=base['Title'], fontName='DejaVuSans-Bold', fontSize=20, leading=24, textColor=colors.HexColor('#0f172a'), spaceAfter=7),
        'subtitle': ParagraphStyle('subtitle', parent=base['BodyText'], fontName='DejaVuSans', fontSize=10, leading=14, textColor=colors.HexColor('#334155'), spaceAfter=12),
        'h1': ParagraphStyle('h1', parent=base['Heading1'], fontName='DejaVuSans-Bold', fontSize=13.5, leading=18, textColor=colors.HexColor('#0f172a'), spaceBefore=8, spaceAfter=5),
        'h2': ParagraphStyle('h2', parent=base['Heading2'], fontName='DejaVuSans-Bold', fontSize=10.8, leading=14, textColor=colors.HexColor('#0f172a'), spaceBefore=5, spaceAfter=4),
        'body': ParagraphStyle('body', parent=base['BodyText'], fontName='DejaVuSans', fontSize=9.4, leading=13.2, alignment=TA_JUSTIFY, textColor=colors.HexColor('#111827'), spaceAfter=4),
        'bullet': ParagraphStyle('bullet', parent=base['BodyText'], fontName='DejaVuSans', fontSize=9.2, leading=12.8, leftIndent=12, bulletIndent=0, textColor=colors.HexColor('#111827'), spaceAfter=2),
        'small': ParagraphStyle('small', parent=base['BodyText'], fontName='DejaVuSans', fontSize=8.2, leading=10.2, textColor=colors.HexColor('#475569'), spaceAfter=2),
        'caption': ParagraphStyle('caption', parent=base['BodyText'], fontName='DejaVuSans', fontSize=8.2, leading=10.4, alignment=TA_CENTER, textColor=colors.HexColor('#475569'), spaceBefore=2, spaceAfter=7),
        'ref': ParagraphStyle('ref', parent=base['BodyText'], fontName='DejaVuSans', fontSize=8.4, leading=10.4, alignment=TA_LEFT, textColor=colors.HexColor('#111827'), spaceAfter=3),
        'metric': ParagraphStyle('metric', parent=base['BodyText'], fontName='DejaVuSans-Bold', fontSize=17, leading=18, textColor=colors.white, alignment=TA_CENTER),
        'metric_label': ParagraphStyle('metric_label', parent=base['BodyText'], fontName='DejaVuSans', fontSize=8.1, leading=9.7, textColor=colors.white, alignment=TA_CENTER),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullets(story: list, items: list[str], style: ParagraphStyle) -> None:
    for item in items:
        story.append(Paragraph(item, style, bulletText='-'))


def fit_image(path: Path, max_width_mm: float, max_height_mm: float) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    scale = min((max_width_mm * mm) / width, (max_height_mm * mm) / height)
    return Image(str(path), width=width * scale, height=height * scale)


def add_figure(story: list, path: Path, caption: str, styles: dict[str, ParagraphStyle], height_mm: float = 82) -> None:
    story.append(fit_image(path, 178, height_mm))
    story.append(p(caption, styles['caption']))


def table_from_rows(rows: list[list[str]], widths_mm: list[float], styles: dict[str, ParagraphStyle], header_bg: str = '#e2e8f0') -> Table:
    wrapped = []
    for i, row in enumerate(rows):
        row_style = styles['small']
        wrapped.append([p(str(cell), row_style) for cell in row])
    t = Table(wrapped, colWidths=[w * mm for w in widths_mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ('GRID', (0, 0), (-1, -1), 0.45, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont('DejaVuSans', 8)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(16 * mm, 287 * mm, 'İstanbul baraj doluluğu - 2026-2040 projeksiyon raporu')
    canvas.drawRightString(194 * mm, 10 * mm, f'Sayfa {doc.page}')
    canvas.restoreState()


def fmt_pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return '-'
    return f'%{x:.{digits}f}'


def fmt_pp(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return '-'
    return f'{x:.{digits}f} yp'


def fmt_date(x) -> str:
    if pd.isna(x) or x == '':
        return '-'
    return str(x)[:10]


def scenario_label(s: str) -> str:
    return {
        'wet_mild': 'Ilık-ıslak',
        'management_improvement': 'Yönetim iyileşme',
        'base': 'Temel',
        'hot_dry_high_demand': 'Sıcak-kurak-yüksek talep',
        'base_transfer_relief': 'Temel + transfer rahatlama',
        'base_transfer_stress': 'Temel + transfer stresi',
        'hot_dry_transfer_stress': 'Sıcak-kurak + transfer stresi',
    }.get(s, s)


def load_data() -> dict[str, object]:
    summary = pd.read_csv(PROJ_DIR / 'scenario_projection_summary_2026_2040.csv')
    risk = pd.read_csv(PROJ_DIR / 'scenario_threshold_risk_summary_2026_2040.csv')
    transfer = pd.read_csv(PROJ_DIR / 'transfer_sensitivity_summary_2026_2040.csv')
    metrics = pd.read_csv(PROJ_DIR / 'model_selection_metrics.csv')
    sensitivity_rd = pd.read_csv(PROJ_DIR / 'sensitivity_rain_demand_grid_2040.csv')
    sensitivity_et = pd.read_csv(PROJ_DIR / 'sensitivity_et0_transfer_grid_2040.csv')
    transfer_anchor = pd.read_csv(PROJ_DIR / 'official_transfer_dependency_annual_2021_2025.csv')
    source_registry = pd.read_csv(SOURCE_REGISTRY)
    selected = metrics.sort_values('rmse_pp').iloc[0]
    refs = source_registry[source_registry['source_id'].isin(USED_SOURCE_IDS)].copy()
    refs['sort_key'] = refs['source_id'].str.extract(r'(\d+)').astype(int)
    refs = refs.sort_values('sort_key')
    return {
        'summary': summary,
        'risk': risk,
        'transfer': transfer,
        'metrics': metrics,
        'selected': selected,
        'sensitivity_rd': sensitivity_rd,
        'sensitivity_et': sensitivity_et,
        'transfer_anchor': transfer_anchor,
        'transfer_anchor_share': float(transfer_anchor['transfer_share_pct'].mean()),
        'refs': refs,
    }


def build_markdown(data: dict[str, object]) -> str:
    summary: pd.DataFrame = data['summary']
    risk: pd.DataFrame = data['risk']
    transfer: pd.DataFrame = data['transfer']
    selected = data['selected']
    refs: pd.DataFrame = data['refs']
    rd: pd.DataFrame = data['sensitivity_rd']

    best = rd.loc[rd['end_fill_2040_12_pct'].idxmax()]
    worst = rd.loc[rd['end_fill_2040_12_pct'].idxmin()]
    lines = [
        '# İstanbul Baraj Doluluğu 2026-2040 Projeksiyon Raporu',
        '',
        '## 1. Amaç',
        '',
        'Bu rapor, İstanbul toplam baraj doluluğu için kurulan `2026-2040` aylık projeksiyon motorunun güncel durumunu özetler. Çalışma artık sadece geçmiş seriyi uzatan bir tahmin değil; iklim, talep, işletme ve dış transfer katmanlarını birlikte kullanan bir karar-destek çerçevesidir.',
        '',
        '## 2. Mevcut model durumu',
        '',
        f"- Seçilen model: `{selected['model']}`",
        f"- Yürüyen test hatası: `{selected['rmse_pp']:.2f}` yüzde puan",
        '- `history_only_ridge` yerine `hybrid_ridge` seçildi; yani yalnız geçmiş doluluk değil, yağış, ET0, talep ve iklim blokları da modele giriyor.',
        '',
        '## 3. Ana senaryolar',
        '',
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"- `{scenario_label(row.scenario)}`: 2040 sonu `{row.end_fill_2040_12_pct:.2f}%`, 2026-2040 ortalama `{row.mean_fill_2026_2040_pct:.2f}%`")
    lines += [
        '',
        '## 4. Risk okuması',
        '',
    ]
    for row in risk[risk['threshold_pct'] == 30].itertuples(index=False):
        lines.append(
            f"- `{scenario_label(row.scenario)}`: %30 altı ay sayısı `{int(row.months_point_below_threshold)}`, ilk geçiş `{fmt_date(row.first_cross_date)}`, kalıcı geçiş `{fmt_date(row.permanent_cross_date)}`"
        )
    lines += [
        '',
        '## 5. Dış transfer etkisi',
        '',
        f"- 2021-2025 resmî ortalama Melen + Yeşilçay payı: `{data['transfer_anchor_share']:.2f}%`",
    ]
    for row in transfer.itertuples(index=False):
        lines.append(
            f"- `{scenario_label(row.scenario)}`: eşlenik temele göre 2040 sonunda `{row.delta_vs_paired_baseline_2040_12_pp:.2f}` yüzde puan fark"
        )
    lines += [
        '',
        '## 6. Parametre duyarlılığı',
        '',
        f"- En iyi yağış-talep noktası: yağış `{best['rain_end_pct_2040']:+.0f}%`, talep `{best['direct_demand_end_pct_2040']:+.0f}%`, 2040 sonu `{best['end_fill_2040_12_pct']:.2f}%`",
        f"- En kötü yağış-talep noktası: yağış `{worst['rain_end_pct_2040']:+.0f}%`, talep `{worst['direct_demand_end_pct_2040']:+.0f}%`, 2040 sonu `{worst['end_fill_2040_12_pct']:.2f}%`",
        '- `yağış × talep` yüzeyi sunumda kullanılabilir durumda.',
        '- `ET0 × transfer` yüzeyinde transfer tarafı anlamlı olsa da ET0 tek başına izole edildiğinde işaret kararlılığı sorunu var. Bu nedenle bu yüzey şu aşamada karar çıktısı değil, model tanısı olarak okunmalıdır.',
        '',
        '## 7. Teknik sınırlar',
        '',
        '- Aktinograf henüz bağlanmadı; ET0 bloğu hâlâ vekil seriye dayanıyor.',
        '- Dış transfer katmanı aylık fiziksel akım modeli değil, talep eşdeğeri duyarlılık olarak kodlandı.',
        '- Sektörel talep ayrımı ve su kesintisi olay takvimi hâlâ eksik.',
        '',
        '## 8. Kaynaklar',
        '',
    ]
    for row in refs.itertuples(index=False):
        lines.append(f"- **[{row.source_id}]** {row.title}: {row.url}")
    return '\n'.join(lines) + '\n'


def build_pdf(data: dict[str, object]) -> None:
    styles = build_styles()
    summary: pd.DataFrame = data['summary']
    risk: pd.DataFrame = data['risk']
    transfer: pd.DataFrame = data['transfer']
    metrics: pd.DataFrame = data['metrics']
    refs: pd.DataFrame = data['refs']
    transfer_anchor: pd.DataFrame = data['transfer_anchor']
    sensitivity_rd: pd.DataFrame = data['sensitivity_rd']

    best_rd = sensitivity_rd.loc[sensitivity_rd['end_fill_2040_12_pct'].idxmax()]
    worst_rd = sensitivity_rd.loc[sensitivity_rd['end_fill_2040_12_pct'].idxmin()]
    selected = data['selected']

    story = []
    story.append(p('İstanbul Baraj Doluluğu 2026-2040 Projeksiyon Raporu', styles['title']))
    story.append(p('Bu rapor, mevcut projeksiyon motorunun geldiği noktayı anlaşılır dille özetler. Odak, tek bir sayı vermek değil; hangi koşulun sistemi nereye ittiğini, hangi bulguya ne kadar güvenebildiğimizi ve hangi parçanın hâlâ geliştirilmesi gerektiğini açıkça göstermektir.', styles['subtitle']))

    metric_rows = [
        [p('%30.38', styles['metric']), p('%6.11', styles['metric']), p('%47.16', styles['metric'])],
        [p('Temel senaryoda 2040 sonu doluluk', styles['metric_label']), p('Sıcak-kurak-yüksek talepte 2040 sonu doluluk', styles['metric_label']), p('2021-2025 ortalama dış transfer payı', styles['metric_label'])],
        [p(f"{selected['rmse_pp']:.2f}", styles['metric']), p('4.03 yp', styles['metric']), p('-8.06 yp', styles['metric'])],
        [p('Seçilen model hatası (RMSE)', styles['metric_label']), p('Transfer rahatlamasının temel senaryoya etkisi', styles['metric_label']), p('Transfer stresinin temel senaryoya etkisi', styles['metric_label'])],
    ]
    mt = Table(metric_rows, colWidths=[57 * mm, 57 * mm, 57 * mm], rowHeights=[11 * mm, 15 * mm, 11 * mm, 15 * mm])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f766e')),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#115e59')),
        ('INNERGRID', (0, 0), (-1, -1), 0.8, colors.HexColor('#115e59')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(mt)
    story.append(Spacer(1, 6 * mm))

    story.append(p('1. Neyi kurduk?', styles['h1']))
    bullets(story, [
        'İstanbul toplam baraj doluluğu için aylık çalışan bir ileri projeksiyon motoru kuruldu.',
        'Bu motor yalnız geçmiş doluluğu değil; yağış, referans evapotranspirasyon (ET0), talep ve işletme baskılarını birlikte kullanıyor.',
        'Çıktı artık tek çizgi değil; senaryolar, eşik riskleri, dış transfer duyarlılığı ve parametre yüzeyleri ile birlikte okunuyor.',
    ], styles['bullet'])

    story.append(p('2. Model performansı', styles['h1']))
    metric_rows_2 = [['Model', 'RMSE (yp)', 'Yorum']]
    for row in metrics.itertuples(index=False):
        comment = 'Seçilen model' if row.model == selected['model'] else 'Karşılaştırma modeli'
        if row.model == 'history_only_ridge':
            label = 'Yalnız tarihsel'
        elif row.model == 'hybrid_ridge':
            label = 'Hibrit Ridge'
        else:
            label = 'Extra Trees'
        metric_rows_2.append([label, f'{row.rmse_pp:.2f}', comment])
    story.append(table_from_rows(metric_rows_2, [55, 30, 90], styles))
    story.append(p('Sonuç net: yalnız geçmiş doluluğa dayanan model yerine hibrit yapı seçildi. Bu, iklim ve talep bloklarının gerçek katkı verdiğini gösteriyor.', styles['body']))

    add_figure(story, FIG_DIR / 'benchmark_history_vs_hybrid.png', 'Şekil 1. Yalnız tarihsel model ile hibrit model karşılaştırması.', styles, height_mm=60)
    add_figure(story, FIG_DIR / 'scenario_paths_2026_2040.png', 'Şekil 2. Dört ana senaryo için 2026-2040 toplam doluluk yolları.', styles, height_mm=70)

    story.append(p('3. Ana senaryoların anlamı', styles['h1']))
    rows = [['Senaryo', '2040 sonu', '2026-2040 ort.', 'İlk %30 altı']] 
    for row in summary.itertuples(index=False):
        rows.append([
            scenario_label(row.scenario),
            fmt_pct(row.end_fill_2040_12_pct),
            fmt_pct(row.mean_fill_2026_2040_pct),
            fmt_date(row.first_below_30_date),
        ])
    story.append(table_from_rows(rows, [58, 28, 30, 44], styles))
    story.append(p('Temel senaryoda sistem 2040 sonunda %30 civarına kadar çekiliyor. Sıcak-kurak-yüksek talep senaryosu ise kritik baskı oluşturuyor ve 2040 sonunda yaklaşık %6 seviyesine kadar iniyor.', styles['body']))

    add_figure(story, FIG_DIR / 'future_driver_paths_2026_2040.png', 'Şekil 3. Gelecek sürücülerin birlikte okunması: yağış, ET0, talep, sıcaklık, bağıl nem, VPD, su dengesi ve toplam doluluk.', styles, height_mm=135)

    story.append(PageBreak())
    story.append(p('4. Risk katmanı', styles['h1']))
    risk30 = risk[risk['threshold_pct'] == 30].copy()
    rows = [['Senaryo', '%30 altı ay', 'İlk geçiş', 'Kalıcı geçiş', 'Ortalama olasılık']]
    for row in risk30.itertuples(index=False):
        rows.append([
            scenario_label(row.scenario),
            str(int(row.months_point_below_threshold)),
            fmt_date(row.first_cross_date),
            fmt_date(row.permanent_cross_date),
            fmt_pct(row.mean_prob_below_threshold_pct),
        ])
    story.append(table_from_rows(rows, [55, 24, 30, 30, 35], styles))
    story.append(p('Risk tarafında en sert tablo sıcak-kurak-yüksek talep senaryosunda görülüyor. Temel senaryoda da risk sıfır değil; ancak zaman içinde daha geç ortaya çıkıyor.', styles['body']))
    add_figure(story, FIG_DIR / 'threshold_risk_below_30.png', 'Şekil 4. Senaryolara göre %30 altı risk seviyesi.', styles, height_mm=68)

    story.append(p('5. Dış transfer neden kritik?', styles['h1']))
    story.append(p(f"Resmî İSKİ verisine göre `2021-2025` döneminde Melen + Yeşilçay kaynaklarının şehre verilen su içindeki ortalama payı %{data['transfer_anchor_share']:.2f}. Bu nedenle dış transfer değişimi, İstanbul iç baraj sistemi için ikincil değil birincil baskı değişkenlerinden biridir.", styles['body']))
    rows = [['Senaryo', '2040 sonu', 'Temele fark', 'En düşük seviye']]
    for row in transfer.itertuples(index=False):
        rows.append([
            scenario_label(row.scenario),
            fmt_pct(row.end_fill_2040_12_pct),
            fmt_pp(row.delta_vs_paired_baseline_2040_12_pp),
            fmt_pct(row.min_fill_2026_2040_pct),
        ])
    story.append(table_from_rows(rows, [65, 25, 35, 30], styles))
    add_figure(story, FIG_DIR / 'transfer_dependency_history_2021_2025.png', 'Şekil 5. Resmî Melen-Yeşilçay hacmi ve toplam verilen su içindeki payı.', styles, height_mm=62)
    add_figure(story, FIG_DIR / 'transfer_sensitivity_paths_2026_2040.png', 'Şekil 6. Dış transfer rahatlama ve stres senaryolarının doluluk yoluna etkisi.', styles, height_mm=68)

    story.append(p('6. Parametrelerle oynadığımızda ne oluyor?', styles['h1']))
    story.append(p('Karar-destek açısından en kullanışlı yeni çıktı, parametre ızgaraları oldu. Böylece kullanıcı yalnız dört senaryo arasında seçim yapmıyor; yağış, talep, ET0 ve dış transfer gibi sürücüler değiştiğinde 2040 sonunun nereye gittiğini doğrudan okuyabiliyor.', styles['body']))
    story.append(p(f"Yağış-talep yüzeyinde en iyi test noktası `yağış +%10 / talep -%10` ve `2040 sonu % {best_rd['end_fill_2040_12_pct']:.2f}`; en kötü nokta ise `yağış -%10 / talep +%10` ve `2040 sonu % {worst_rd['end_fill_2040_12_pct']:.2f}` oldu.", styles['body']))
    add_figure(story, FIG_DIR / 'sensitivity_heatmap_rain_demand_2040.png', 'Şekil 7. Yağış ve talep değişince 2040 sonu doluluk nereye gidiyor?', styles, height_mm=78)
    add_figure(story, FIG_DIR / 'sensitivity_heatmap_et0_transfer_2040.png', 'Şekil 8. ET0 ve dış transfer yüzeyi. Not: ET0 tarafı şu aşamada tanısal amaçlıdır.', styles, height_mm=78)

    story.append(p('7. Şu an neye güveniyoruz, neye temkinli bakıyoruz?', styles['h1']))
    bullets(story, [
        'Güvenilen katmanlar: ana senaryo yolları, yağış-talep yüzeyi, dış transfer stresi/rahatlaması, eşik riski.',
        'Temkinli okunan katman: ET0 tek başına izole edildiğinde işaret kararlılığı bozulduğu için ET0 yüzeyi.',
        'Bunun ana nedeni aktinograf verisinin henüz bağlanmamış olması ve ET0 bloğunun hâlâ vekil seriye dayanmasıdır.',
    ], styles['bullet'])

    story.append(p('8. Sonuç', styles['h1']))
    story.append(p('Bu proje artık sadece doluluk tahmini üreten bir çalışma değil. İstanbul baraj sistemini iklim, talep, işletme ve dış transfer açısından birlikte okuyan; riski zaman içinde gösteren; müdahale alanlarını karşılaştıran ve kullanıcıya parametrelerle oynama imkânı verebilecek noktaya gelmiş bir karar-destek altyapısıdır.', styles['body']))
    story.append(p('Bir sonraki teknik sıçrama, aktinograf geldiğinde ET0 katmanını yeniden kurmak ve bugün tanısal olan ET0 yüzeyini güvenilir karar yüzeyine çevirmektir.', styles['body']))

    story.append(PageBreak())
    story.append(p('Kaynakça', styles['h1']))
    for row in refs.itertuples(index=False):
        story.append(p(f"<b>[{row.source_id}]</b> {row.title}<br/>{row.url}", styles['ref']))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=14 * mm)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    register_fonts()
    data = load_data()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_markdown(data), encoding='utf-8')
    build_pdf(data)
    print(OUT_MD)
    print(OUT_PDF)


if __name__ == '__main__':
    main()
