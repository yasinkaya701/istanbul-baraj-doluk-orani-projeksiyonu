#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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
PREFERRED_DIR = ROOT / 'output' / 'istanbul_preferred_projection_2040'
RISK_DIR = PREFERRED_DIR / 'operational_risk_2030'
BENCH_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_sourceaware_ensemble_2040' / 'ensemble_vs_benchmark_models.csv'
ITER_PATH = ROOT / 'output' / 'model_iteration_comparison_2026_03_12.csv'
BOTTOM_UP_SUMMARY_PATH = ROOT / 'output' / 'nearterm_bottom_up_reconciliation' / 'nearterm_bottom_up_reconciliation_summary.json'
SOURCE_REGISTRY = ROOT / 'research' / 'baraj_doluluk_hub' / 'registry' / 'sources' / 'external_sources.csv'
OUT_MD = ROOT / 'output' / 'report' / 'istanbul_baraj_tercih_edilen_projeksiyon_raporu.md'
OUT_PDF = ROOT / 'output' / 'pdf' / 'istanbul_baraj_tercih_edilen_projeksiyon_raporu.pdf'
FONT_DIR = Path('/opt/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf')
USED_SOURCE_IDS = ['SRC-001', 'SRC-002', 'SRC-057', 'SRC-059', 'SRC-064', 'SRC-065', 'SRC-066', 'SRC-067', 'SRC-068', 'SRC-072', 'SRC-073']

SCENARIO_LABELS = {
    'wet_mild': 'Ilık-ıslak',
    'management_improvement': 'Yönetim iyileşme',
    'base': 'Temel',
    'hot_dry_high_demand': 'Sıcak-kurak-yüksek talep',
}


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
    for row in rows:
        wrapped.append([p(str(cell), styles['small']) for cell in row])
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
    canvas.drawString(16 * mm, 287 * mm, 'İstanbul baraj doluluğu - tercih edilen projeksiyon raporu')
    canvas.drawRightString(194 * mm, 10 * mm, f'Sayfa {doc.page}')
    canvas.restoreState()


def fmt_pct(x: float | int | str | None, digits: int = 2) -> str:
    if x is None or x == '' or pd.isna(x):
        return '-'
    return f'%{float(x):.{digits}f}'


def fmt_num(x: float | int | str | None, digits: int = 2) -> str:
    if x is None or x == '' or pd.isna(x):
        return '-'
    return f'{float(x):.{digits}f}'


def load_data() -> dict[str, object]:
    det = pd.read_csv(PREFERRED_DIR / 'deterministic_summary.csv')
    prob = pd.read_csv(PREFERRED_DIR / 'probabilistic_summary.csv')
    compare = pd.read_csv(PREFERRED_DIR / 'compare_with_baseprob.csv')
    nearterm = pd.read_csv(RISK_DIR / 'preferred_nearterm_risk_summary_2026_2030.csv')
    yearly = pd.read_csv(RISK_DIR / 'preferred_yearly_consecutive_risk_2026_2040.csv')
    bench = pd.read_csv(BENCH_PATH)
    iterations = pd.read_csv(ITER_PATH)
    bottom_up = json.loads(BOTTOM_UP_SUMMARY_PATH.read_text(encoding='utf-8'))
    sources = pd.read_csv(SOURCE_REGISTRY)
    refs = sources[sources['source_id'].isin(USED_SOURCE_IDS)].copy()
    refs['sort_key'] = refs['source_id'].str.extract(r'(\d+)').astype(int)
    refs = refs.sort_values('sort_key')
    return {
        'det': det,
        'prob': prob,
        'compare': compare,
        'nearterm': nearterm,
        'yearly': yearly,
        'bench': bench,
        'iterations': iterations,
        'bottom_up': bottom_up,
        'refs': refs,
    }


def build_markdown(data: dict[str, object]) -> str:
    det: pd.DataFrame = data['det']
    prob: pd.DataFrame = data['prob']
    nearterm: pd.DataFrame = data['nearterm']
    bench: pd.DataFrame = data['bench']
    iterations: pd.DataFrame = data['iterations']
    refs: pd.DataFrame = data['refs']
    bottom_up = data['bottom_up']

    chosen = bench[bench['model'] == 'hybrid_physics_ensemble_phys'].iloc[0]
    preferred_iter = iterations[iterations['preferred'] == True].iloc[0]

    lines = [
        '# İstanbul Baraj Doluluğu Tercih Edilen Projeksiyon Raporu',
        '',
        '## 1. Sonuç',
        '',
        'Bu raporda tercih edilen çıktı paketi, kısa vadede baraj bazlı tahminlerle uzlaştırılmış, uzun vadede ise fizik-kısıtlı ve kaynak duyarlı hibrit ensemble omurgasını koruyan projeksiyon setidir.',
        '',
        '## 2. Seçilen model neden bu?',
        '',
        f"- Tercih edilen ana model: `hybrid_physics_ensemble_phys`",
        f"- Tek adım hata: `{chosen['one_step_rmse_pp']:.2f}` yüzde puan",
        f"- Çok adımlı ortalama hata: `{chosen['mean_recursive_rmse_pp']:.2f}` yüzde puan",
        f"- Fiziksel işaret testi: `{int(chosen['physics_pass_count'])}/4`",
        f"- Tercih edilen sürüm ailesi: `{preferred_iter['model']}`",
        '',
        '## 3. Neleri denedik ve neden elettik?',
        '',
        '- `water_balance_v5_sourcerain`: kısa vadede iyileşti ama uzun ufukta fazla bozuldu.',
        '- `water_balance_v6_regionrain`: hem kısa hem uzun ufukta zayıf kaldı.',
        '- Kaynak-bazlı yağış fikri doğru, fakat gelecek için uzaysal forcing zayıf olduğunda model aşırı sertleşiyor.',
        '',
        '## 4. Yakın vade uzlaştırması',
        '',
        f"- Bottom-up baraj tahminleri ile top-down temel senaryo arasındaki ortalama fark: `{bottom_up['mean_gap_bottom_up_minus_topdown_pp']:.2f}` yüzde puan",
        '- Bu fark doğrudan rapora taşınmadı; 2026-2029 arasında azalan bir düzeltme eğrisi olarak uygulandı.',
        '',
        '## 5. Deterministik 2040 sonuçları',
        '',
    ]
    for row in det.itertuples(index=False):
        lines.append(f"- `{SCENARIO_LABELS.get(row.scenario,row.scenario)}`: 2040 sonu `{row.end_fill_2040_12_pct:.2f}%`, dönem ortalaması `{row.mean_fill_2026_2040_pct:.2f}%`")
    lines += ['', '## 6. Olasılıksal 2040 sonuçları', '']
    for row in prob.itertuples(index=False):
        lines.append(
            f"- `{SCENARIO_LABELS.get(row.scenario,row.scenario)}`: P10 `{row.p10_endpoint_2040_12_pct:.2f}%`, P50 `{row.p50_endpoint_2040_12_pct:.2f}%`, P90 `{row.p90_endpoint_2040_12_pct:.2f}%`"
        )
    lines += ['', '## 7. 2026-2030 operasyonel risk', '']
    for row in nearterm.itertuples(index=False):
        lines.append(
            f"- `{SCENARIO_LABELS.get(row.scenario,row.scenario)}`: en yüksek `%40 altı` aylık risk `{row.max_monthly_prob_below_40_2026_2030_pct:.2f}%`, tepe ay `{row.peak_month_prob_below_40_date}`"
        )
    lines += ['', '## 8. Kaynaklar', '']
    for row in refs.itertuples(index=False):
        lines.append(f"- **[{row.source_id}]** {row.title}: {row.url}")
    return '\n'.join(lines) + '\n'


def build_pdf(data: dict[str, object]) -> None:
    styles = build_styles()
    det: pd.DataFrame = data['det']
    prob: pd.DataFrame = data['prob']
    compare: pd.DataFrame = data['compare']
    nearterm: pd.DataFrame = data['nearterm']
    yearly: pd.DataFrame = data['yearly']
    bench: pd.DataFrame = data['bench']
    iterations: pd.DataFrame = data['iterations']
    refs: pd.DataFrame = data['refs']
    bottom_up = data['bottom_up']

    chosen = bench[bench['model'] == 'hybrid_physics_ensemble_phys'].iloc[0]
    v5 = iterations[iterations['model'] == 'ensemble_v5_sourcerain_phys'].iloc[0]

    story: list = []
    story.append(p('İstanbul Baraj Doluluğu Tercih Edilen Projeksiyon Raporu', styles['title']))
    story.append(p('Bu rapor, şu anda kullanılması önerilen projeksiyon paketini açıklar. Seçilen çözüm, kısa vadede baraj bazlı tahminlerle uzlaştırılmış, uzun vadede ise fizik-kısıtlı ve kaynak duyarlı hibrit ensemble omurgasını koruyan modeldir.', styles['subtitle']))

    metric_rows = [
        [p(fmt_num(chosen['one_step_rmse_pp']), styles['metric']), p(fmt_num(chosen['mean_recursive_rmse_pp']), styles['metric']), p('4/4', styles['metric'])],
        [p('Tek adım hata', styles['metric_label']), p('Çok adımlı ortalama hata', styles['metric_label']), p('Fiziksel işaret testi', styles['metric_label'])],
        [p(f"{bottom_up['mean_gap_bottom_up_minus_topdown_pp']:.2f}", styles['metric']), p(fmt_pct(prob[prob['scenario']=='base']['p50_endpoint_2040_12_pct'].iloc[0]), styles['metric']), p(fmt_pct(prob[prob['scenario']=='hot_dry_high_demand']['p50_endpoint_2040_12_pct'].iloc[0]), styles['metric'])],
        [p('Yakın vade bottom-up farkı (yp)', styles['metric_label']), p('Temel senaryo 2040 P50', styles['metric_label']), p('Sıcak-kurak 2040 P50', styles['metric_label'])],
    ]
    mt = Table(metric_rows, colWidths=[57 * mm, 57 * mm, 57 * mm], rowHeights=[11 * mm, 15 * mm, 11 * mm, 15 * mm])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f766e')),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#115e59')),
        ('INNERGRID', (0, 0), (-1, -1), 0.8, colors.HexColor('#115e59')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(mt)
    story.append(Spacer(1, 6 * mm))

    story.append(p('1. Seçilen model ve gerekçe', styles['h1']))
    bullets(story, [
        'Ana omurga `hybrid_physics_ensemble_phys` olarak seçildi.',
        'Bu model, istatistiksel hata ile fiziksel tutarlılık arasında en dengeli sonucu veriyor.',
        'Tek adımda `3.91` yüzde puan hata üretirken, fiziksel işaret testlerinin tamamını geçiyor.',
        'Kısa vade için ayrıca baraj bazlı tahminlerle uzlaştırma uygulanıyor; böylece yakın dönem yol fazla iyimser kalmıyor.',
    ], styles['bullet'])

    bench_rows = [['Model', 'Tek adım hata', 'Çok adımlı hata', 'Fizik testi']]
    for row in bench.sort_values(['one_step_rmse_pp']).head(6).itertuples(index=False):
        bench_rows.append([
            row.model,
            fmt_num(row.one_step_rmse_pp),
            fmt_num(row.mean_recursive_rmse_pp),
            f"{int(row.physics_pass_count)}/4",
        ])
    story.append(table_from_rows(bench_rows, [63, 34, 38, 25], styles))
    story.append(Spacer(1, 4 * mm))
    bullets(story, [
        f"`ensemble_v5_sourcerain_phys` kısa vadede `{v5['one_step_rmse_pp']:.2f}` ile daha düşük hata verdi; ancak çok adımlı hata `{v5['mean_recursive_rmse_pp']:.2f}` seviyesine çıktı ve ana hatta alınmadı.",
        'Bu sonuç, uzaysal yağış fikrinin doğru ama gelecek forcing zayıfken kırılgan olduğunu gösteriyor.',
    ], styles['bullet'])

    story.append(p('2. Yakın vade uzlaştırması', styles['h1']))
    bullets(story, [
        f"Baraj bazlı bottom-up tahminler ile top-down temel yol arasında ortalama `{bottom_up['mean_gap_bottom_up_minus_topdown_pp']:.2f}` yüzde puan fark bulundu.",
        'Bu fark 2026-2029 döneminde azalarak uygulanan bir düzeltme eğrisine çevrildi.',
        'Amaç, yakın vade operasyonel görünümde aşırı iyimserliği kırmak; 2030 sonrası ise ana fizik-kısıtlı omurgayı korumak.',
    ], styles['bullet'])
    add_figure(story, ROOT / 'output' / 'nearterm_bottom_up_reconciliation' / 'nearterm_bottom_up_vs_topdown.png', 'Şekil 1. Baraj bazlı bottom-up tahminlerin kapasite ağırlıklı toplamı ile top-down temel yol arasındaki fark.', styles, height_mm=64)
    add_figure(story, PREFERRED_DIR / 'reconciled_base_projection.png', 'Şekil 2. Uzlaştırılmış temel senaryo yolu. Düzeltme 2026-2029 döneminde etkili, sonrasında sıfıra yaklaşıyor.', styles, height_mm=62)

    story.append(PageBreak())
    story.append(p('3. Deterministik ve olasılıksal senaryo çıktıları', styles['h1']))

    merged = det.merge(prob, on='scenario', how='left')
    scen_rows = [['Senaryo', '2040 sonu', '2040 P10', '2040 P50', '2040 P90', 'İlk %40>=50 olasılık yılı']]
    for row in merged.itertuples(index=False):
        scen_rows.append([
            SCENARIO_LABELS.get(row.scenario, row.scenario),
            fmt_pct(row.end_fill_2040_12_pct),
            fmt_pct(row.p10_endpoint_2040_12_pct),
            fmt_pct(row.p50_endpoint_2040_12_pct),
            fmt_pct(row.p90_endpoint_2040_12_pct),
            '-' if pd.isna(row.first_year_prob_below_40_ge_50pct) else str(int(row.first_year_prob_below_40_ge_50pct)),
        ])
    story.append(table_from_rows(scen_rows, [41, 23, 23, 23, 23, 36], styles))
    story.append(Spacer(1, 4 * mm))

    bullets(story, [
        'Temel senaryoda 2040 medyan doluluk yaklaşık `%34.69` seviyesinde kalıyor.',
        'Yönetim iyileşme senaryosu, iklim aynı kalsa bile sistemi yukarı taşıyor.',
        'Sıcak-kurak-yüksek talep senaryosu tek agresif kırılma senaryosu olarak öne çıkıyor.',
        'Rekonsiliasyon sonrası 2040 uç değerleri neredeyse değişmiyor; fark esas olarak yakın vadede oluşuyor.',
    ], styles['bullet'])

    add_figure(story, PREFERRED_DIR / 'reconciled_probabilistic_fan_paths_2026_2040.png', 'Şekil 3. Tercih edilen paketin olasılıksal senaryo yolları. Koyu çizgi medyan, bantlar belirsizlik aralığıdır.', styles, height_mm=80)
    add_figure(story, PREFERRED_DIR / 'reconciled_vs_baseprob_base.png', 'Şekil 4. Rekonsiliasyon öncesi ve sonrası temel senaryo medyan yolu. Etki kısa vadede görülür, uzun vadede sönümlenir.', styles, height_mm=60)

    story.append(PageBreak())
    story.append(p('4. 2026-2030 operasyonel risk görünümü', styles['h1']))
    near_rows = [['Senaryo', 'Tepe %40 altı risk', 'Tepe ay', 'Tepe %30 altı risk', 'Tepe ay', 'En zayıf yaz P50']]
    for row in nearterm.itertuples(index=False):
        near_rows.append([
            SCENARIO_LABELS.get(row.scenario, row.scenario),
            fmt_pct(row.max_monthly_prob_below_40_2026_2030_pct),
            row.peak_month_prob_below_40_date,
            fmt_pct(row.max_monthly_prob_below_30_2026_2030_pct),
            row.peak_month_prob_below_30_date,
            f"{int(row.lowest_p50_summer_year_2026_2030)} / {fmt_pct(row.lowest_p50_summer_fill_pct_2026_2030)}",
        ])
    story.append(table_from_rows(near_rows, [34, 26, 23, 26, 23, 36], styles))
    story.append(Spacer(1, 4 * mm))

    base_y = yearly[(yearly['scenario'] == 'base') & (yearly['year'] <= 2030)].copy()
    hot_y = yearly[(yearly['scenario'] == 'hot_dry_high_demand') & (yearly['year'] <= 2030)].copy()
    bullets(story, [
        'Temel senaryoda 2030 sonuna kadar `%40 altında üç ay üst üste` olasılığı `%50`yi geçmiyor.',
        'Yönetim iyileşme senaryosu da yakın vadede benzer şekilde kalıcı düşük seviye üretmiyor.',
        f"Sıcak-kurak-yüksek talep senaryosunda `%40 altında üç ay üst üste` olasılığı 2026 yılında bile `%{hot_y.loc[hot_y['year']==2026, 'prob_3consec_below_40_pct'].iloc[0]:.2f}` düzeyinde.",
        'Bu nedenle operasyonel alarmı erken veren tek senaryo sıcak-kurak-yüksek talep senaryosu.',
    ], styles['bullet'])
    add_figure(story, RISK_DIR / 'preferred_monthly_threshold_risk_2026_2030.png', 'Şekil 5. 2026-2030 döneminde aylık `%40`, `%30` ve `%20` altı olasılık yolları.', styles, height_mm=74)
    add_figure(story, RISK_DIR / 'preferred_threshold_heatmap_40_2026_2030.png', 'Şekil 6. 2026-2030 aylık `%40 altı` risk ısı haritası. Kırmızı alanlar yoğun risk kümelerini gösterir.', styles, height_mm=44)

    story.append(PageBreak())
    story.append(p('5. Teknik yorum ve sınırlar', styles['h1']))
    bullets(story, [
        'Bu pakette referans evapotranspirasyon (ET0) hâlâ vekil radyasyon serileriyle çalışıyor. Aktinograf geldiğinde bu blok yeniden kurulmalı.',
        'Açık su yüzeyi buharlaşması, ET0 ile aynı şey gibi ele alınmıyor; modüler su bütçesi mantığı korunuyor.',
        'Kaynak-bazlı yağış forcing denendi; fakat geleceğe taşınan uzaysal forcing zayıf olduğu için ana projeksiyon hattına alınmadı.',
        'Bugünkü en güvenilir kullanım şekli şudur: yakın vadede rekonsiliasyonlu yol, orta-uzun vadede fizik-kısıtlı ensemble omurgası ve her ikisi için olasılıksal risk okuması.',
    ], styles['bullet'])

    diff = compare[['scenario', 'delta_p50_endpoint_pp']].copy()
    diff_rows = [['Senaryo', 'Rekonsiliasyonun 2040 P50 etkisi (yp)']]
    for row in diff.itertuples(index=False):
        diff_rows.append([SCENARIO_LABELS.get(row.scenario, row.scenario), fmt_num(row.delta_p50_endpoint_pp)])
    story.append(table_from_rows(diff_rows, [72, 85], styles, header_bg='#dcfce7'))
    story.append(Spacer(1, 4 * mm))
    story.append(p('Tablo yorumu: rekonsiliasyonun 2040 sonu etkisi çok küçük. Bu beklenen bir sonuç, çünkü düzeltme tasarımı gereği yakın vadede etki ediyor ve daha sonra sönümleniyor.', styles['body']))

    story.append(p('6. Kaynakça', styles['h1']))
    for row in refs.itertuples(index=False):
        txt = f"[{row.source_id}] {row.title}. {row.organization_or_journal}. {row.year}. {row.url}"
        story.append(p(txt, styles['ref']))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=14 * mm,
        title='İstanbul Baraj Doluluğu Tercih Edilen Projeksiyon Raporu',
        author='OpenAI Codex',
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    register_fonts()
    data = load_data()
    OUT_MD.write_text(build_markdown(data), encoding='utf-8')
    build_pdf(data)


if __name__ == '__main__':
    main()
