#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path('/Users/yasinkaya/Hackhaton')
FINAL_DIR = ROOT / 'output' / 'final_delivery' / 'istanbul_baraj_tercih_edilen_paket_2026_03_12'
PREFERRED_DIR = ROOT / 'output' / 'istanbul_preferred_projection_2040'
RISK_DIR = PREFERRED_DIR / 'operational_risk_2030'
OUT_SPEECH = FINAL_DIR / 'juri_konusma_metni.md'
OUT_MAIL = FINAL_DIR / 'hoca_mail_govdesi.md'
OUT_ORDER = FINAL_DIR / 'dosya_kullanim_sirasi.md'
MANIFEST = FINAL_DIR / 'DELIVERY_MANIFEST.json'


def main() -> None:
    det = pd.read_csv(PREFERRED_DIR / 'deterministic_summary.csv')
    prob = pd.read_csv(PREFERRED_DIR / 'probabilistic_summary.csv')
    risk = pd.read_csv(RISK_DIR / 'preferred_nearterm_risk_summary_2026_2030.csv')

    def p50(scenario: str) -> float:
        return float(prob.loc[prob['scenario'] == scenario, 'p50_endpoint_2040_12_pct'].iloc[0])

    def risk40(scenario: str) -> float:
        return float(risk.loc[risk['scenario'] == scenario, 'max_monthly_prob_below_40_2026_2030_pct'].iloc[0])

    hot_peak_month = str(risk.loc[risk['scenario'] == 'hot_dry_high_demand', 'peak_month_prob_below_30_date'].iloc[0])
    hot_peak30 = float(risk.loc[risk['scenario'] == 'hot_dry_high_demand', 'max_monthly_prob_below_30_2026_2030_pct'].iloc[0])

    speech = f"""# Jüri Konuşma Metni

## Kısa Açılış
Merhaba. Bu çalışmada İstanbul toplam baraj doluluğu için yalnızca geçmiş doluluk serisini uzatan bir tahmin kurmadık. Bunun yerine, yakın vadede baraj bazlı tahminlerle uzlaştırılmış, orta ve uzun vadede ise fizik-kısıtlı ve kaynak duyarlı hibrit bir projeksiyon çerçevesi geliştirdik.

## Slayt 1
Ana iddiamız şu: kısa vadede temkinli, uzun vadede savunulabilir bir doluluk projeksiyonu kurduk. Seçilen model tek adımda yaklaşık 3.91 yüzde puan hata veriyor, çok adımlı ortalama hata 7.72 yüzde puan ve fiziksel işaret testlerinin tamamını geçiyor.

## Slayt 2
Model seçimini sadece en düşük hata üzerinden yapmadık. Daha agresif bazı sürümler kısa vadede daha iyi görünse de uzun vadede kararsızlaştı veya fiziksel tutarlılık kaybetti. Bu nedenle ana hatta en dengeli modeli aldık.

## Slayt 3
Yakın vade düzeltmesini eklememizin nedeni, baraj bazlı bottom-up tahminlerin temel top-down yolun kısa vadede fazla iyimser kaldığını göstermesiydi. Ortalama fark yaklaşık -7.46 yüzde puan. Bu yüzden 2026-2029 dönemi için düzeltme uyguladık, 2030 sonrası ise ana omurgayı koruduk.

## Slayt 4
2040 görünümünde senaryolar net biçimde ayrışıyor. Temel senaryoda 2040 medyan doluluk yaklaşık %{p50('base'):.2f}. Yönetim iyileşme senaryosunda bu değer yaklaşık %{p50('management_improvement'):.2f}. Ilık-ıslak senaryoda yaklaşık %{p50('wet_mild'):.2f}. Sıcak-kurak-yüksek talep senaryosunda ise yaklaşık %{p50('hot_dry_high_demand'):.2f}. Yani en sert kırılma, yüksek talep ile sıcak-kurak koşullar birleştiğinde ortaya çıkıyor.

## Slayt 5
Yakın vade operasyonel riskte temel senaryonun tepe yüzde 40 altı riski yaklaşık %{risk40('base'):.0f}. Yönetim iyileşme de benzer biçimde kalıcı bir alarm üretmiyor. Ama sıcak-kurak-yüksek talep senaryosunda tepe yüzde 40 altı risk %{risk40('hot_dry_high_demand'):.0f} seviyesine çıkıyor. Yüzde 30 altı tepe risk ise %{hot_peak30:.2f} ve en kritik ay {hot_peak_month}. Bu nedenle yakın vade alarmı esas olarak bu senaryoda yoğunlaşıyor.

## Slayt 6
Bu paketi bugün nasıl kullanmak gerektiği net: yakın vadede uzlaştırılmış yol ve eşik riskleri kullanılmalı; 2040 için ise tek sayı yerine P10-P50-P90 aralığıyla konuşulmalı. Bu, nihai işletme modeli değildir; fakat bugün savunulabilecek en sağlam karar çerçevesidir.

## Kapanış
Özetle, bu çalışma İstanbul baraj doluluğu için yalnızca bir tahmin üretmiyor; kısa vade operasyonel gerçekçiliği ile uzun vade senaryo okumasını aynı yapıda birleştiriyor.
"""

    mail = f"""Konu: İstanbul baraj doluluğu projeksiyonu - tercih edilen çerçeve ve güncel çıktılar

Merhaba hocam,

İstanbul toplam baraj doluluğu için geliştirdiğimiz projeksiyon çalışmasında tercih edilen yöntemi ve güncel çıktıları toparladım.

Çalışmada yalnızca geçmiş doluluk serisini ileri taşıyan bir yaklaşım yerine, yakın vadede baraj bazlı tahminlerle uzlaştırılmış, orta ve uzun vadede ise fizik-kısıtlı ve kaynak duyarlı hibrit ensemble omurgasına dayanan bir çerçeve kullandık.

Model seçimini yaparken tek adımlı hata ile çok adımlı kararlılığı birlikte değerlendirdik. Tercih edilen modelin tek adım hatası yaklaşık 3.91 yüzde puan, çok adımlı ortalama hatası 7.72 yüzde puan ve fiziksel işaret testlerinin tamamını geçiyor.

Ayrıca yakın vadede top-down temel yolun bir miktar iyimser kaldığını gördüğümüz için, baraj bazlı bottom-up tahminlerle uzlaştırılmış bir düzeltme katmanı ekledik. Böylece 2026-2029 dönemindeki görünüm daha temkinli hale geldi; 2030 sonrasında ise ana fizik-kısıtlı omurga korunmuş oldu.

2040 görünümünde temel senaryo için medyan doluluk yaklaşık %{p50('base'):.2f}, yönetim iyileşme senaryosu için yaklaşık %{p50('management_improvement'):.2f}, ılık-ıslak senaryo için yaklaşık %{p50('wet_mild'):.2f} ve sıcak-kurak-yüksek talep senaryosu için yaklaşık %{p50('hot_dry_high_demand'):.2f} düzeyinde görünüyor.

Yakın vade risk analizinde temel ve yönetim iyileşme senaryoları 2030 sonuna kadar kalıcı yüzde 40 altı alarm üretmezken, sıcak-kurak-yüksek talep senaryosu erken dönemde belirgin risk üretiyor.

Hazırladığım güncel dosyalar şunlar:
- PDF rapor
- daha resmî dille yazılmış DOCX rapor
- jüri için kısa ve doğrudan PPTX sunumu

İsterseniz bir sonraki adımda metni daha akademik bir makale diliyle sıkılaştırabilir veya sunum dilini daha kısa hale getirebilirim.

İyi çalışmalar dilerim.
"""

    order = """# Dosya Kullanım Sırası

## Sunum günü için önerilen sıra
1. `istanbul_baraj_tercih_edilen_projeksiyon_sunumu_v3.pptx`
2. `istanbul_baraj_tercih_edilen_projeksiyon_raporu.pdf`
3. `istanbul_baraj_tercih_edilen_projeksiyon_raporu_resmi.docx`
4. `preferred_nearterm_risk_summary_2026_2030.csv`

## Dosya ne işe yarıyor
- `istanbul_baraj_tercih_edilen_projeksiyon_sunumu_v3.pptx`: jüri anlatımı için kısa ve net sürüm
- `istanbul_baraj_tercih_edilen_projeksiyon_raporu.pdf`: görsel olarak en stabil rapor sürümü
- `istanbul_baraj_tercih_edilen_projeksiyon_raporu_resmi.docx`: gerektiğinde metin düzenlemesi yapılabilecek resmî sürüm
- `preferred_nearterm_risk_summary_2026_2030.csv`: yakın vade risk sayıları için ham özet tablo

## Not
- Sunumda ana referans olarak `v3` deck kullanılmalı.
- Yazılı teslim veya mail eki için öncelik `pdf` ve `resmi docx` olmalı.
"""

    OUT_SPEECH.write_text(speech, encoding='utf-8')
    OUT_MAIL.write_text(mail, encoding='utf-8')
    OUT_ORDER.write_text(order, encoding='utf-8')

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    manifest['files'].update({
        'jury_talk_track': str(OUT_SPEECH),
        'mail_body': str(OUT_MAIL),
        'file_usage_order': str(OUT_ORDER),
    })
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    print(OUT_SPEECH)
    print(OUT_MAIL)
    print(OUT_ORDER)


if __name__ == '__main__':
    main()
