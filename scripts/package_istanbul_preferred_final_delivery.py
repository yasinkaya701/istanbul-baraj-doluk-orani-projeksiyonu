#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path('/Users/yasinkaya/Hackhaton')
OUT_DIR = ROOT / 'output' / 'final_delivery' / 'istanbul_baraj_tercih_edilen_paket_2026_03_12'
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    'pdf_report': ROOT / 'output' / 'pdf' / 'istanbul_baraj_tercih_edilen_projeksiyon_raporu.pdf',
    'docx_report_academic': ROOT / 'output' / 'doc' / 'istanbul_baraj_tercih_edilen_projeksiyon_raporu_akademik.docx',
    'pptx_deck_v2': ROOT / 'output' / 'slides' / 'istanbul_baraj_tercih_edilen_projeksiyon_deck' / 'istanbul_baraj_tercih_edilen_projeksiyon_sunumu_v2.pptx',
    'pptx_source_js_v2': ROOT / 'output' / 'slides' / 'istanbul_baraj_tercih_edilen_projeksiyon_deck' / 'build_deck_v2.js',
    'preferred_manifest': ROOT / 'output' / 'istanbul_preferred_projection_2040' / 'PREFERRED_PACKAGE.json',
    'risk_summary_csv': ROOT / 'output' / 'istanbul_preferred_projection_2040' / 'operational_risk_2030' / 'preferred_nearterm_risk_summary_2026_2030.csv',
}

copied = {}
for key, src in files.items():
    dst = OUT_DIR / src.name
    shutil.copy2(src, dst)
    copied[key] = str(dst)

readme = OUT_DIR / 'README.md'
readme.write_text(
    '\n'.join([
        '# İstanbul Baraj Doluluğu Tercih Edilen Teslim Paketi',
        '',
        'Bu klasör, tercih edilen projeksiyon çıktılarının teslime hazır sürümünü içerir.',
        '',
        'İçerik:',
        '- PDF rapor',
        '- Akademik dille düzenlenmiş DOCX rapor',
        '- Güçlendirilmiş sunum diliyle PPTX deck',
        '- Deck kaynak JavaScript dosyası',
        '- Tercih edilen paket manifesti',
        '- Yakın vade operasyonel risk özeti CSV',
        '',
        'Not:',
        '- PDF render kontrolü yapılmıştır.',
        '- DOCX ve PPTX için bu ortamda `soffice` bulunmadığı için yapısal doğrulama yapılmıştır.',
    ]),
    encoding='utf-8'
)

manifest = {
    'package_dir': str(OUT_DIR),
    'files': copied,
    'validation': {
        'pdf_render_checked': True,
        'docx_structural_check_only': True,
        'pptx_structural_check_only': True,
    }
}
(OUT_DIR / 'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(OUT_DIR)
