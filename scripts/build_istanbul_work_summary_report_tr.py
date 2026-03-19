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
OUT_MD = ROOT / 'output' / 'report' / 'istanbul_baraj_yapilanlar_ozet_raporu.md'
OUT_PDF = ROOT / 'output' / 'pdf' / 'istanbul_baraj_yapilanlar_ozet_raporu.pdf'

PREFERRED_DIR = ROOT / 'output' / 'istanbul_preferred_projection_2040'
RISK_DIR = PREFERRED_DIR / 'operational_risk_2030'
BENCH_PATH = ROOT / 'output' / 'istanbul_hybrid_physics_sourceaware_ensemble_2040' / 'ensemble_vs_benchmark_models.csv'
BOTTOMUP_SUMMARY_PATH = ROOT / 'output' / 'nearterm_bottom_up_reconciliation' / 'nearterm_bottom_up_reconciliation_summary.json'
DIRREC_DIR = ROOT / 'output' / 'istanbul_bestmethod_dirrec_conformal_2040'
RECT_DIR = ROOT / 'output' / 'istanbul_bestmethod_rectify_conformal_2040'
FONT_DIR = Path('/opt/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf')

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
        'title': ParagraphStyle('title', parent=base['Title'], fontName='DejaVuSans-Bold', fontSize=19, leading=23, textColor=colors.HexColor('#0f172a'), spaceAfter=7),
        'subtitle': ParagraphStyle('subtitle', parent=base['BodyText'], fontName='DejaVuSans', fontSize=10, leading=13.5, textColor=colors.HexColor('#334155'), spaceAfter=10),
        'h1': ParagraphStyle('h1', parent=base['Heading1'], fontName='DejaVuSans-Bold', fontSize=13.2, leading=17, textColor=colors.HexColor('#0f172a'), spaceBefore=8, spaceAfter=4),
        'body': ParagraphStyle('body', parent=base['BodyText'], fontName='DejaVuSans', fontSize=9.3, leading=13, alignment=TA_JUSTIFY, textColor=colors.HexColor('#111827'), spaceAfter=4),
        'bullet': ParagraphStyle('bullet', parent=base['BodyText'], fontName='DejaVuSans', fontSize=9.1, leading=12.4, leftIndent=12, bulletIndent=0, textColor=colors.HexColor('#111827'), spaceAfter=2),
        'small': ParagraphStyle('small', parent=base['BodyText'], fontName='DejaVuSans', fontSize=8.2, leading=10.2, textColor=colors.HexColor('#475569'), spaceAfter=2),
        'caption': ParagraphStyle('caption', parent=base['BodyText'], fontName='DejaVuSans', fontSize=8.0, leading=10.0, alignment=TA_CENTER, textColor=colors.HexColor('#475569'), spaceBefore=2, spaceAfter=7),
        'ref': ParagraphStyle('ref', parent=base['BodyText'], fontName='DejaVuSans', fontSize=8.2, leading=10.1, alignment=TA_LEFT, textColor=colors.HexColor('#111827'), spaceAfter=3),
        'metric': ParagraphStyle('metric', parent=base['BodyText'], fontName='DejaVuSans-Bold', fontSize=16, leading=18, textColor=colors.white, alignment=TA_CENTER),
        'metric_label': ParagraphStyle('metric_label', parent=base['BodyText'], fontName='DejaVuSans', fontSize=8, leading=9.4, textColor=colors.white, alignment=TA_CENTER),
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


def add_figure(story: list, path: Path, caption: str, styles: dict[str, ParagraphStyle], height_mm: float = 72) -> None:
    story.append(fit_image(path, 178, height_mm))
    story.append(p(caption, styles['caption']))


def table_from_rows(rows: list[list[str]], widths_mm: list[float], styles: dict[str, ParagraphStyle], header_bg: str = '#e2e8f0') -> Table:
    wrapped = [[p(str(cell), styles['small']) for cell in row] for row in rows]
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
    canvas.drawString(16 * mm, 287 * mm, 'İstanbul baraj doluluğu - yapılanlar özet raporu')
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
    nearterm = pd.read_csv(RISK_DIR / 'preferred_nearterm_risk_summary_2026_2030.csv')
    bench = pd.read_csv(BENCH_PATH)
    bottom_up = json.loads(BOTTOMUP_SUMMARY_PATH.read_text(encoding='utf-8'))
    dirrec_summary = json.loads((DIRREC_DIR / 'dirrec_summary.json').read_text(encoding='utf-8'))
    dirrec_compare = pd.read_csv(DIRREC_DIR / 'dirrec_compare_vs_current_preferred.csv')
    rectify_summary = json.loads((RECT_DIR / 'rectify_summary.json').read_text(encoding='utf-8'))
    rectify_compare = pd.read_csv(RECT_DIR / 'rectify_compare_vs_current_preferred.csv')
    rectify_weights = pd.read_csv(RECT_DIR / 'rectify_correction_weight_grid.csv')
    return {
        'det': det,
        'prob': prob,
        'nearterm': nearterm,
        'bench': bench,
        'bottom_up': bottom_up,
        'dirrec_summary': dirrec_summary,
        'dirrec_compare': dirrec_compare,
        'rectify_summary': rectify_summary,
        'rectify_compare': rectify_compare,
        'rectify_weights': rectify_weights,
    }


def build_markdown(data: dict[str, object]) -> str:
    det: pd.DataFrame = data['det']
    prob: pd.DataFrame = data['prob']
    nearterm: pd.DataFrame = data['nearterm']
    bench: pd.DataFrame = data['bench']
    bottom_up = data['bottom_up']
    dirrec_summary = data['dirrec_summary']
    dirrec_compare: pd.DataFrame = data['dirrec_compare']
    rectify_summary = data['rectify_summary']
    rectify_compare: pd.DataFrame = data['rectify_compare']

    chosen = bench[bench['model'] == 'hybrid_physics_ensemble_phys'].iloc[0]
    lines = [
        '# İstanbul Baraj Doluluğu - Yapılanlar Özet Raporu',
        '',
        '## 1. Bu projede ne yaptık?',
        '',
        'Bu çalışmada İstanbul toplam baraj doluluğu için veri temeli kuruldu, fizik-kısıtlı hibrit bir ana model geliştirildi, kısa vade için bottom-up uzlaştırma eklendi, olasılıksal risk katmanı üretildi ve ardından akademik olarak güçlü challenger yöntemler test edildi.',
        '',
        '## 2. Şu an kullanılan ana hat',
        '',
        f"- Ana model: `hybrid_physics_ensemble_phys`",
        f"- Tek adım hata: `{chosen['one_step_rmse_pp']:.2f}` yp",
        f"- Çok adımlı ortalama hata: `{chosen['mean_recursive_rmse_pp']:.2f}` yp",
        f"- Fiziksel işaret testi: `{int(chosen['physics_pass_count'])}/4`",
        '',
        '## 3. Yakın vade uzlaştırması',
        '',
        f"- Bottom-up ve top-down temel yol arasındaki ortalama fark: `{bottom_up['mean_gap_bottom_up_minus_topdown_pp']:.2f}` yp",
        '- Bu fark 2026-2029 dönemine yayılan bir düzeltme eğrisi olarak ana yola eklendi.',
        '',
        '## 4. Tercih edilen senaryo sonuçları',
        '',
    ]
    for row in det.itertuples(index=False):
        lines.append(f"- `{SCENARIO_LABELS.get(row.scenario,row.scenario)}`: 2040 sonu `{row.end_fill_2040_12_pct:.2f}%`, dönem ortalaması `{row.mean_fill_2026_2040_pct:.2f}%`")
    lines += ['', '## 5. En iyi challenger yöntemler', '']
    lines += [
        f"- `direct-recursive + conformal`: anchor ortalama RMSE `{dirrec_summary['mean_blend_rmse_anchor_pp']:.2f}` yp",
        f"- `rectify + conformal`: anchor ortalama RMSE `{rectify_summary['mean_blended_rmse_anchor_pp']:.2f}` yp",
        '- İkisi de backtestte iyileşme verdi, fakat uzun vadede fazla iyimser kaldı.',
    ]
    base_dirrec = dirrec_compare[dirrec_compare['scenario'] == 'base'].iloc[0]
    hot_dirrec = dirrec_compare[dirrec_compare['scenario'] == 'hot_dry_high_demand'].iloc[0]
    base_rect = rectify_compare[rectify_compare['scenario'] == 'base'].iloc[0]
    hot_rect = rectify_compare[rectify_compare['scenario'] == 'hot_dry_high_demand'].iloc[0]
    lines += [
        f"- `direct-recursive` 2040 sonunda temel senaryoda mevcut hatta göre `{base_dirrec['delta_vs_current_preferred_p50_pp']:.2f}` yp, kötü senaryoda `{hot_dirrec['delta_vs_current_preferred_p50_pp']:.2f}` yp yukarı çıktı.",
        f"- `rectify` 2040 sonunda temel senaryoda mevcut hatta göre `{base_rect['delta_vs_current_prob_p50_pp']:.2f}` yp, kötü senaryoda `{hot_rect['delta_vs_current_prob_p50_pp']:.2f}` yp yukarı çıktı.",
        '',
        '## 6. Son karar',
        '',
        '- En iyi backtest challenger: `rectify + conformal`',
        '- En güvenli üretim hattı: mevcut tercih edilen paket',
        '- Yani yeni yöntemler challenger olarak tutuldu, ana hat değiştirilmedi.',
        '',
        '## 7. Darboğaz',
        '',
        '- Bu aşamada ana sorun algoritma değil, veri/forcing kalitesi.',
        '- En kritik eksikler: aktinograf, kaynak bazlı gelecek yağış forcing, aylık transfer güvenilirliği, aylık NRW/operasyon verisi.',
    ]
    for row in prob.itertuples(index=False):
        if row.scenario in SCENARIO_LABELS:
            lines.append(f"- `{SCENARIO_LABELS[row.scenario]}` 2040 P50: `{row.p50_endpoint_2040_12_pct:.2f}%`")
    lines += ['', '## 8. Ana dosyalar', '']
    lines += [
        f"- Ana üretim hattı: `{PREFERRED_DIR}`",
        f"- Direct-recursive challenger: `{DIRREC_DIR}`",
        f"- Rectify challenger: `{RECT_DIR}`",
    ]
    lines += ['', '## 9. Yöntem dayanakları', '']
    lines += [
        '- Rectify strategy: https://robjhyndman.com/papers/rectify.pdf',
        '- Sequential Predictive Conformal Inference: https://proceedings.mlr.press/v202/xu23r.html',
        '- İstanbul hibrit yaklaşımı için referans: https://www.mdpi.com/2071-1050/16/17/7696',
    ]
    return '\n'.join(lines) + '\n'


def build_pdf(data: dict[str, object]) -> None:
    styles = build_styles()
    det: pd.DataFrame = data['det']
    prob: pd.DataFrame = data['prob']
    nearterm: pd.DataFrame = data['nearterm']
    bench: pd.DataFrame = data['bench']
    bottom_up = data['bottom_up']
    dirrec_summary = data['dirrec_summary']
    dirrec_compare: pd.DataFrame = data['dirrec_compare']
    rectify_summary = data['rectify_summary']
    rectify_compare: pd.DataFrame = data['rectify_compare']
    rectify_weights: pd.DataFrame = data['rectify_weights']

    chosen = bench[bench['model'] == 'hybrid_physics_ensemble_phys'].iloc[0]
    base_prob = prob[prob['scenario'] == 'base'].iloc[0]
    hot_prob = prob[prob['scenario'] == 'hot_dry_high_demand'].iloc[0]

    story: list = []
    story.append(p('İstanbul Baraj Doluluğu - Yapılanlar Özet Raporu', styles['title']))
    story.append(p('Bu rapor, bu projede hangi veri ve model adımlarının uygulandığını, hangi yöntemlerin denendiğini ve neden mevcut ana hatta karar verildiğini tek yerde özetler.', styles['subtitle']))

    metric_rows = [
        [p(fmt_num(chosen['one_step_rmse_pp']), styles['metric']), p(fmt_num(chosen['mean_recursive_rmse_pp']), styles['metric']), p(fmt_num(bottom_up['mean_gap_bottom_up_minus_topdown_pp']), styles['metric'])],
        [p('Tek adım hata', styles['metric_label']), p('Çok adımlı ortalama hata', styles['metric_label']), p('Yakın vade bottom-up farkı', styles['metric_label'])],
        [p(fmt_pct(base_prob['p50_endpoint_2040_12_pct']), styles['metric']), p(fmt_pct(hot_prob['p50_endpoint_2040_12_pct']), styles['metric']), p('4/4', styles['metric'])],
        [p('Temel 2040 P50', styles['metric_label']), p('Sıcak-kurak 2040 P50', styles['metric_label']), p('Fizik testi', styles['metric_label'])],
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

    story.append(p('1. Bu projede ne yapıldı?', styles['h1']))
    bullets(story, [
        'İstanbul toplam baraj doluluğu için veri paketi kuruldu; yağış, ET0, talep, sıcaklık, nem ve operasyon bağlamı tek tabloda toplandı.',
        'Kaynak duyarlı fizik-kısıtlı hibrit ensemble geliştirildi ve ana hat olarak seçildi.',
        'Yakın vade için baraj bazlı bottom-up tahminlerle top-down toplam yol uzlaştırıldı.',
        'Olasılıksal senaryo ve operasyonel risk katmanı üretildi.',
        'Son aşamada akademik olarak güçlü challenger yöntemler test edildi.',
    ], styles['bullet'])

    story.append(p('2. Neden bu ana model seçildi?', styles['h1']))
    bench_rows = [['Model', 'Tek adım hata', 'Çok adımlı hata', 'Fizik testi']]
    for row in bench.sort_values(['one_step_rmse_pp']).head(6).itertuples(index=False):
        bench_rows.append([row.model, fmt_num(row.one_step_rmse_pp), fmt_num(row.mean_recursive_rmse_pp), f"{int(row.physics_pass_count)}/4"])
    story.append(table_from_rows(bench_rows, [63, 34, 38, 25], styles))
    story.append(Spacer(1, 4 * mm))
    bullets(story, [
        'Tercih edilen omurga `hybrid_physics_ensemble_phys` oldu.',
        'Bu model sadece düşük hata vermedi; fiziksel işaret testlerinin tamamını da geçti.',
        'Bu nedenle üretim hattında güvenlik ve açıklanabilirlik açısından en dengeli çözüm bu model oldu.',
    ], styles['bullet'])

    add_figure(story, ROOT / 'output' / 'nearterm_bottom_up_reconciliation' / 'nearterm_bottom_up_vs_topdown.png', 'Şekil 1. Yakın vadede bottom-up baraj toplamı ile top-down temel yol arasındaki fark.', styles, height_mm=60)
    add_figure(story, PREFERRED_DIR / 'reconciled_base_projection.png', 'Şekil 2. Bottom-up farkı kullanılarak üretilen uzlaştırılmış temel yol.', styles, height_mm=58)

    story.append(PageBreak())
    story.append(p('3. Tercih edilen çıktı paketi ne veriyor?', styles['h1']))
    scen_rows = [['Senaryo', '2040 sonu', '2040 P10', '2040 P50', '2040 P90']]
    merged = det.merge(prob, on='scenario', how='left')
    for row in merged.itertuples(index=False):
        if row.scenario in SCENARIO_LABELS:
            scen_rows.append([
                SCENARIO_LABELS[row.scenario],
                fmt_pct(row.end_fill_2040_12_pct),
                fmt_pct(row.p10_endpoint_2040_12_pct),
                fmt_pct(row.p50_endpoint_2040_12_pct),
                fmt_pct(row.p90_endpoint_2040_12_pct),
            ])
    story.append(table_from_rows(scen_rows, [52, 28, 28, 28, 28], styles))
    story.append(Spacer(1, 4 * mm))
    bullets(story, [
        'Temel senaryoda 2040 medyanı yaklaşık `%34.69` seviyesinde.',
        'Yönetim iyileşme senaryosu iklim aynı kalsa bile sistemi anlamlı biçimde yukarı taşıyor.',
        'Asıl sert kırılma sıcak-kurak-yüksek talep senaryosunda görülüyor.',
    ], styles['bullet'])
    add_figure(story, PREFERRED_DIR / 'reconciled_probabilistic_fan_paths_2026_2040.png', 'Şekil 3. Tercih edilen paketin olasılıksal senaryo yolları.', styles, height_mm=72)
    add_figure(story, RISK_DIR / 'preferred_threshold_heatmap_40_2026_2030.png', 'Şekil 4. 2026-2030 döneminde `%40 altı` risk ısı haritası.', styles, height_mm=42)

    story.append(PageBreak())
    story.append(p('4. Sonradan hangi güçlü yöntemleri denedik?', styles['h1']))
    best_rows = [['Yöntem', 'Anchor ort. RMSE', '2040 temel farkı', '2040 kötü senaryo farkı', 'Karar']]
    dirrec_base = dirrec_compare[dirrec_compare['scenario'] == 'base'].iloc[0]
    dirrec_hot = dirrec_compare[dirrec_compare['scenario'] == 'hot_dry_high_demand'].iloc[0]
    rectify_base = rectify_compare[rectify_compare['scenario'] == 'base'].iloc[0]
    rectify_hot = rectify_compare[rectify_compare['scenario'] == 'hot_dry_high_demand'].iloc[0]
    best_rows.append([
        'Direct-recursive + conformal',
        fmt_num(dirrec_summary['mean_blend_rmse_anchor_pp']),
        fmt_num(dirrec_base['delta_vs_current_preferred_p50_pp']),
        fmt_num(dirrec_hot['delta_vs_current_preferred_p50_pp']),
        'Challenger',
    ])
    best_rows.append([
        'Rectify + conformal',
        fmt_num(rectify_summary['mean_blended_rmse_anchor_pp']),
        fmt_num(rectify_base['delta_vs_current_prob_p50_pp']),
        fmt_num(rectify_hot['delta_vs_current_prob_p50_pp']),
        'Challenger',
    ])
    story.append(table_from_rows(best_rows, [56, 28, 28, 32, 24], styles, header_bg='#dcfce7'))
    story.append(Spacer(1, 4 * mm))
    bullets(story, [
        'Direct-recursive ve rectify iki güçlü challenger olarak uygulandı.',
        'İkisi de backtestte iyileşme verdi.',
        'Fakat ikisi de uzun vadede, özellikle kötü senaryoda, mevcut ana hatta göre fazla iyimser kaldı.',
        'Bu nedenle ana üretim hattı değiştirilmedi.',
    ], styles['bullet'])
    add_figure(story, DIRREC_DIR / 'figures' / 'dirrec_anchor_method_comparison.png', 'Şekil 5. Direct-recursive challenger için anchor horizon karşılaştırması.', styles, height_mm=58)
    add_figure(story, RECT_DIR / 'figures' / 'rectify_anchor_method_comparison.png', 'Şekil 6. Rectify challenger için anchor horizon karşılaştırması.', styles, height_mm=58)

    story.append(PageBreak())
    story.append(p('5. Son karar ve teknik anlamı', styles['h1']))
    bullets(story, [
        'En iyi backtest challenger: `rectify + conformal`.',
        'En güvenli üretim hattı: mevcut tercih edilen paket.',
        'Bu şu anlama geliyor: yöntem tarafında ciddi arama yapıldı ve güçlü challengerlar gerçekten uygulandı.',
        'Buna rağmen ana kalite darboğazı artık algoritma değil, veri/forcing kalitesi.',
        'Aktinograf, kaynak bazlı gelecek yağış forcing, aylık transfer güvenilirliği ve aylık NRW verisi gelmeden yeni algoritmalardan alınacak ek kazanç sınırlı kalacak.',
    ], styles['bullet'])

    weight_rows = [['Ufuk (ay)', 'Rectify düzeltme ağırlığı']]
    for row in rectify_weights.itertuples(index=False):
        weight_rows.append([str(int(row.horizon_months)), fmt_num(row.correction_weight)])
    story.append(table_from_rows(weight_rows, [42, 52], styles))
    story.append(Spacer(1, 4 * mm))

    story.append(p('6. Yöntem dayanakları', styles['h1']))
    refs = [
        '[1] Rectify strategy - Ben Taieb & Hyndman. https://robjhyndman.com/papers/rectify.pdf',
        '[2] Sequential Predictive Conformal Inference. https://proceedings.mlr.press/v202/xu23r.html',
        '[3] İstanbul barajları için hibrit yaklaşım - Sustainability 2024. https://www.mdpi.com/2071-1050/16/17/7696',
    ]
    for ref in refs:
        story.append(p(ref, styles['ref']))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=14 * mm,
        title='İstanbul Baraj Doluluğu - Yapılanlar Özet Raporu',
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
