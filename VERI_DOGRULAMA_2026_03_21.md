# Veri ve Kaynak Doğrulama Notu (21 Mart 2026)

Bu not, web sitesindeki kritik sayıların ve kaynakların güncel/doğru olmasını sağlamak için yapılan kontrolleri özetler.

## 1) Kaynak erişim kontrolü

Aşağıdaki ana kaynaklar 21 Mart 2026 tarihinde tekrar kontrol edildi:

- İSKİ Baraj Doluluk — `200`
- İSKİ Su Kaynakları — `200`
- İSKİ Su Kayıpları Yıllık Raporları — `200`
- İBB Açık Veri CKAN API — `409` (parametre olmadan normal cevap)
- İBB/İSKİ temiz su XLSX — `200`
- Open‑Meteo docs / archive — `200`
- NOAA NAO serisi — `200`
- FAO/HEC/WMO referansları — `200`
- İSKİ API `sonOnyildaVerilenToplamSu` endpointi — `403` (bazı ağlarda erişim kısıtı)

## 2) 2023 toplam su arzı düzeltmesi

İBB Açık Veri’deki resmi XLSX dosyasından (İSKİ yayımlayıcı) 2023 toplamı kontrol edildi:

- 2023 toplam verilen su: `1,117,064,116 m³`

Bu nedenle webdeki baz veri dosyasında şu alan güncellendi:

- `total_supply_2023_m3 = 1117064116`

## 3) Simülasyon tabanı tutarlılığı

- ET0 simülasyonu `assets/data/climate_baseline.js` içindeki `et0_mm_month` serisinden besleniyor.
- `SIM_COEFFS.mean_et0` değeri (2010–2024 ortalaması) ile iklim serisi tutarlı.
- Buharlaşma hesabı `E = Kc × ET0 × A_lake` üzerinden ve `evap_usage_baseline.js` ile uyumlu.

## 4) Kaynakça temizliği

- `references.html` ve `baraj_web/references.html` bozuk/teknik gürültü linklerinden temizlendi.
- Artık yalnızca projede aktif kullanılan, doğrulanmış kaynaklar listeleniyor.
- `REFERENCES.md` de aynı doğrulanmış set ile güncellendi.

## 5) Güncellenen dosyalar

- `index.html`
- `baraj_web/index.html`
- `references.html`
- `baraj_web/references.html`
- `REFERENCES.md`
- `assets/data/evap_usage_baseline.js`
- `baraj_web/assets/data/evap_usage_baseline.js`
