# Baraj ET0 - Gerçek Radyasyon Girdili Paket

## Kullandığımız radyasyon dosyası

- Dosya: `daily_solar_radiation_complete.csv` (ham veri yolu repoda paylaşılmıyor)
- Günlük veri kapsamı: `1975-01-01` → `2004-12-31`
- Model kapsamı: `1975-01-01` → `2004-12-01`
- `real_extracted` gün: `0`
- `synthetic` gün: `10957`

Bu dosya kullanıcı tarafından verilen radyasyon girdisi olarak doğrudan ET0
hesabına sokuldu. Not: Dosya içindeki `data_source` kolonu korunmuştur; yani
hangi günün gerçek çıkarım, hangisinin sentetik doldurma olduğu tabloda
görülmektedir.

## Kabuller

1. `Tmean = (Tmax + Tmin) / 2` kullanıldı.
2. `Delta`, Tmean üzerinden FAO-56 eğri eğimiyle hesaplandı.
3. `G = 0` alındı.
4. `u2 = 2.0 m/s` sabit rüzgar kullanıldı.
5. Basınç rakımdan sabit türetildi.
6. Radyasyon olarak kullanıcının verdiği günlük seri kullanıldı.
7. Aylık modelde sadece en az %80 gün kapsamasına sahip aylar kullanıldı.
8. Gelecek öngörüsü ET0 serisinin kendisi üzerinden quant model ile yapıldı.

## Temel bulgular

- Ortalama yıllık ET0: `945.9 mm/yıl`
- Yıllık ET0 trendi: `+25.1 mm/10y`
- Min yıllık ET0: `882.6 mm/yıl`
- Max yıllık ET0: `1071.1 mm/yıl`
- Baz dönem (1995-2004) ortalama yıllık ET0: `970.6 mm/yıl`
- 2031-2035 quant öngörü ortalama yıllık ET0: `1030.2 mm/yıl`
- Beklenen fark: `+59.6 mm/yıl`

## Üretilen dosyalar

- Günlük ET0: `baraj_et0_daily_radiation_complete.csv`
- Aylık ET0: `baraj_et0_monthly_radiation_complete.csv`
- Yıllık ET0: `baraj_et0_yearly_radiation_complete.csv`
- Quant forecast: `baraj_et0_quant_forecast_to_2035.csv`
- Grafikler: ET0 paket grafikleri
