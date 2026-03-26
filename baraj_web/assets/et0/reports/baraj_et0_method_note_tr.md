# Baraj ET0 Çalışması - Yöntem ve Veri Kaynağı

## 1. Bu çalışmada ne yaptık?

Bu çalışmada baraj sistemi için `referans evapotranspirasyon (ET0)` serisini iklim panelinden alıp özetledik. Bu rapor ET0 değerlerini yeniden hesaplamaz; amacımız, simülasyonlarda tek ve tutarlı bir ET0 kaynağı kullanmaktır.

## 2. ET0 nedir?

ET0, iyi sulanmış referans bir yüzey için atmosferik buharlaşma talebini temsil eder. Açık su yüzeyi buharlaşmayla aynı değildir; baraj su dengesi için ET0 genellikle bir katsayı ile ölçeklenir.

## 3. Veri kaynağı ve kapsama

- Veri dosyası: `assets/data/climate_baseline.js`
- Gözlem dönemi: `2010-01` -> `2024-12`
- Projeksiyon dönemi: `2026-01` -> `2040-12`
- Not: 2025 yılı panelde bulunmuyor; projeksiyon 2026’dan başlıyor.

## 4. Hesaplama adımları

1. Aylık ET0 değerleri panelden doğrudan alındı.
2. Yıllık toplam ET0, 12 aylık değerlerin toplamı ile hesaplandı.
3. Trend, 2010-2024 yıllık toplamlar üzerinde basit doğrusal regresyonla hesaplandı.
4. Baz dönem ortalaması `2015-2024`, karşılaştırma ortalaması `2031-2035` olarak alındı.

## 5. Neden bu yaklaşım?

- Repoda sürekli, günlük meteoroloji ve radyasyon serisi bulunmuyor.
- İklim paneli aylık ET0’yı zaten tutarlı bir şekilde sağlıyor.
- Simülasyon ve raporlama için tek kaynak kullanmak modeli daha savunulabilir yapıyor.

## 6. Sınırlar

- Aylık zaman adımı kullanılıyor; günlük ET0 ayrıntı seviyesi yok.
- Projeksiyonlar panelin varsayımlarına bağlı.
- ET0, açık su buharlaşma miktarını doğrudan vermez.

## 7. Bu paketin sayısal özeti

- Gözlem yılları: `2010-2024` (15 yıl)
- Ortalama yıllık ET0: `1097.6 mm/yıl`
- Trend: `+24.5 mm / 10 yıl`
- Baz dönem (2015-2024) ortalaması: `1099.6 mm/yıl`
- 2031-2035 ortalaması: `1155.7 mm/yıl`
- Seviye farkı (2031-2035 ort. - 2015-2024 ort.): `+56.1 mm/yıl`

Not: `+24.5 mm / 10 yıl` ve `+56.1 mm/yıl` farklı metriklerdir. İlki trend eğimidir, ikincisi dönem ortalaması farkıdır.

## 8. Bir sonraki mantıklı adımlar

1. Ham meteoroloji verisi gelirse FAO-56 ile ET0’yu yeniden hesaplamak
2. Açık su buharlaşma katsayısını (E = K * ET0) kalibre etmek
3. Belirsizlik ve duyarlılık analizini rapora eklemek
