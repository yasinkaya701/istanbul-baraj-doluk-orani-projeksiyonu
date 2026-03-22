# Baraj ET0 - İklim Paneli Aylık Seri Paketi

## Veri Kaynağı

- Dosya: `assets/data/climate_baseline.js`
- Seri: `et0_mm_month` (aylık)
- Gözlem kapsamı: `2010-01` -> `2024-12`
- Projeksiyon kapsamı: `2026-01` -> `2040-12`
- Not: 2025 yılı panelde yok; projeksiyon 2026’dan başlar.

## Kabuller

1. ET0 değerleri iklim panelinden hazır alındı; bu rapor ET0’yu yeniden hesaplamaz.
2. Yıllık ET0, ilgili yıldaki 12 ayın toplamı olarak hesaplandı.
3. Trend, 2010-2024 yıllık toplamlar üzerinde basit doğrusal regresyonla hesaplandı.

## Temel Bulgular

- Ortalama yıllık ET0 (2010-2024): `1097.6 mm/yıl`
- Yıllık ET0 trend eğimi (2010-2024): `+24.5 mm / 10 yıl`
- Min/Max yıllık ET0 (2010-2024): `1044.1` - `1160.9 mm/yıl`
- Baz dönem (2015-2024) ortalama yıllık ET0: `1099.6 mm/yıl`
- 2031-2035 tahmin ortalaması: `1155.7 mm/yıl`
- Seviye farkı (2031-2035 ort. - 2015-2024 ort.): `+56.1 mm/yıl`

Not: `+56.1 mm/yıl` değeri yıllık artış hızı değildir; iki dönem ortalaması arasındaki seviye farkıdır.

## Üretilen Çıktılar

- Özet: `assets/et0/reports/baraj_et0_real_radiation_summary.json`
- Trend istatistikleri: `assets/et0/reports/baraj_et0_yearly_trend_stats.json`
- Grafikler: `assets/et0/*.png`
