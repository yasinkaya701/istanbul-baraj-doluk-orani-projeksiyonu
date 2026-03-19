# Tarimsal ET0 - Gercek Radyasyon Girdili Paket

## Kullandigim Radyasyon Dosyasi

- Dosya: `daily_solar_radiation_complete.csv` (ham veri yolu repoda paylasilmiyor)
- Gunluk veri kapsami: `1975-01-01` -> `2004-12-31`
- Model kapsami: `1975-01-01` -> `2004-12-01`
- `real_extracted` gun: `0`
- `synthetic` gun: `10957`

Bu dosya kullanicinin verdigi radyasyon girdisi olarak dogrudan ET0 hesabina sokuldu.
Not: dosya icindeki `data_source` kolonu korunmustur; yani hangi gunun gercek cikarim, hangisinin sentetik doldurma oldugu tabloda goruluyor.

## Kabuller

1. `Tmean = (Tmax + Tmin) / 2` kullanildi.
2. `Delta`, Tmean uzerinden FAO-56 egri egimiyle hesaplandi.
3. `G = 0` alindi.
4. `u2 = 2.0 m/s` sabit ruzgar kullanildi.
5. Basinc rakimdan sabit turetildi.
6. Radyasyon olarak kullanicinin verdigi gunluk seri kullanildi.
7. Aylik modelde sadece en az %80 gun kapsamasina sahip aylar kullanildi.
8. Gelecek oengorusu ET0 serisinin kendisi uzerinden quant model ile yapildi.

## Temel Bulgular

- Ortalama yillik ET0: `945.9 mm/yil`
- Yillik ET0 trendi: `+25.1 mm/10y`
- Min yillik ET0: `882.6 mm/yil`
- Max yillik ET0: `1071.1 mm/yil`
- Baz donem (1995-2004) ortalama yillik ET0: `970.6 mm/yil`
- 2031-2035 quant oengoru ortalama yillik ET0: `1030.2 mm/yil`
- Beklenen fark: `+59.6 mm/yil`

## Uretilen Dosyalar

- Gunluk ET0: `tarim_et0_daily_radiation_complete.csv`
- Aylik ET0: `tarim_et0_monthly_radiation_complete.csv`
- Yillik ET0: `tarim_et0_yearly_radiation_complete.csv`
- Quant forecast: `tarim_et0_quant_forecast_to_2035.csv`
- Grafikler: ET0 paket grafikleri
