# 2026-03-11 Makale Yöntem Notu

Bu not, 11 Mart 2026 tarihinde ayrıca incelenen üç makalenin yöntem tarafını proje hafızasına kaydetmek için hazırlanmıştır.

## 1. Estimating the Volume of Evaporation from the Main Dams of Iran
Kaynak: Results in Engineering (2025)
Bağlantı: https://www.sciencedirect.com/science/article/pii/S2590123025031135

### Yöntem özeti
- Çalışma, İran genelinde 117 büyük rezervuarı kapsıyor.
- Rezervuar buharlaşmasını hesaplamak için altı ampirik yöntem ile dört uydu/yeniden analiz veri seti birlikte deneniyor.
- Metinde açık geçen uydu/yeniden analiz setleri: `MODIS`, `GLDAS`, `SEBAL`, `ERA5`.
- Doğrulama için `339` sinoptik istasyon gözlemi kullanılıyor.
- Performans değerlendirmesi `R²` ve `RMSE` ile yapılıyor.
- Sonuçta klasik yöntemler içinde `Penman-Monteith`, uzaktan algılama/yeniden analiz ürünleri içinde `SEBAL` ve `ERA5` en iyi grupta yer alıyor.

### Bizim proje için metodolojik anlamı
- `ET0` ile `açık su buharlaşması` mutlaka ayrı tutulmalı.
- Doğrudan ölçüm yoksa yeniden analiz tabanlı açık su buharlaşması katmanı kurulması savunulabilir.
- Baraj alanı bilgisiyle çarpılan kaynak-bazlı buharlaşma vekili artık sadece fikir değil, literatürce destekli bir sonraki adım.

## 2. Advanced Predictive Modeling for Dam Occupancy Using Historical and Meteorological Data
Kaynak: Sustainability (2024)
Bağlantı: https://www.mdpi.com/2071-1050/16/17/7696

### Yöntem özeti
- Çalışma doğrudan İstanbul için ve `7` barajı kapsıyor: Ömerli, Darlık, Elmalı, Terkos, Alibey, Büyükçekmece, Sazlıdere.
- Veri aralığı yaklaşık `5 yıl` ve günlük ölçekte `1777` gün olarak veriliyor (`2019-2024`).
- İki ayrı veri seti kuruluyor:
  - `hibrit veri seti`: evapotranspirasyon + hava verisi + tüketim + tarihsel baraj verisi
  - `yalnız tarihsel veri seti`
- ET0 tarafı `Penman-Monteith` ile kuruluyor.
- Hesapta rüzgar hızı, net radyasyon ve sıcaklık kullanılıyor; toprak ısı akısı `0` kabul ediliyor.
- Hava verisi tarafında sıcaklık, hissedilen sıcaklık, nem, çiğ noktası, bulutluluk, yağış, kar derinliği ve gün uzunluğu inceleniyor.
- Korelasyon analizi sonrası özellikle `güneş radyasyonu`, `çiğ noktası`, `gün uzunluğu` ve `yağış` seçiliyor.
- Tüketim ve yağış zayıf korelasyon gösterse de fiziksel olarak anlamlı oldukları için veri setinde tutuluyor.
- Tarihsel dolulukta otokorelasyon olduğu gösteriliyor; bu nedenle zaman serisi mantığı korunuyor.
- Geçmiş veri pencereleri `1` ile `7` arası gecikmelerle deneniyor.
- Tahmin ufukları: `günlük`, `haftalık`, `15 günlük`, `30 günlük`.
- Veri ön işleme tarafında:
  - normal dağılım için `KS testi`
  - aykırı değer için `Z-score`, eşik `3`
  - eğitim/test bölmesi `80/20`
- Denenen modeller:
  - `Orthogonal Matching Pursuit CV`
  - `Lasso Lars CV`
  - `Extra Trees`
  - `Random Forest`
  - `Ridge CV`
  - `Transformed Target Regressor`
  - `LSTM`
- Karşılaştırmada ayrıca `ANOVA`, `Z testi` ve `eşleştirilmiş t-testi` kullanılıyor.
- Makalenin sonucunda en iyi yöntem `Extra Trees` olarak raporlanıyor.

### Bizim proje için metodolojik anlamı
- Hibrit çerçevemiz doğru: `tarihsel hafıza + fiziksel sürücüler + tüketim`.
- Bir sonraki model turunda `Extra Trees` zorunlu benchmark olmalı.
- `hibrit veri seti` ve `yalnız tarihsel veri seti` ayrımı bizde de birebir uygulanmalı.
- Farklı ufuklar için ayrı test yapmak gerekli; hava verisinin katkısı kısa ve uzun ufukta aynı değil.

## 3. İstanbul Baraj Doluluk Oranlarının Zamansal İncelenmesi ve Çözüm Önerileri
Kaynak: Journal of Research in Atmospheric Science (2020)
Bağlantı: https://resatmsci.com/?mod=makale_tr_ozet&makale_id=49617
Tam metin erişimi: https://resatmsci.com/sayi/55082bc5-a28a-483c-a79d-7370dd2db586.pdf

### Yöntem özeti
- Çalışmada kullanılan baraj doluluk oranları doğrudan `İSKİ`'den alınmış.
- Veri seti `2005` yılından itibaren günlük doluluk verisini kapsıyor.
- Yöntem, bu verilerin:
  - `aylık`
  - `mevsimsel`
  - `yıllık`
  ölçekte zamansal analizine dayanıyor.
- Baraj dolulukları, uzun yıllar `yağış` ve `sıcaklık` verileri ile birlikte yorumlanıyor.
- Aylık ortalama doluluk oranları `Python` ile yıllara göre görselleştiriliyor.
- Metodoloji bölümünde ayrıca İstanbul su kaynaklarının yıllık verim, azami biriktirme hacmi ve hizmete giriş yılları tablo halinde kullanılıyor.
- Sonuç bölümünde kurak yıllar, toparlanma gecikmesi ve dönemsel düşüş örüntüsü tarifleniyor.

### Bizim proje için metodolojik anlamı
- Basit ama güçlü bir yerel temel çizgi sunuyor.
- `toparlanma gecikmesi` bizim senaryo çıktılarında ayrıca yorumlanmalı.
- `aylık`, `mevsimsel`, `yıllık` üçlü okuma düzeni sunum için de uygun.
- Kurak yıl analizi ve tasarruf önerileri model dışı karar katmanı için doğrudan kullanılabilir.

## Bu üç kaynaktan çıkan net aksiyonlar
1. `Extra Trees` yeni benchmark modeli olarak eklenecek.
2. `yalnız tarihsel veri` ile `hibrit veri seti` ayrımı test setine formalize edilecek.
3. `baraj alanı x açık su buharlaşması` vekili kurulacak.
4. `toparlanma gecikmesi` metriği ve yorum katmanı eklenecek.
5. Sunum dilinde `kurak yıl`, `toparlanma`, `talep baskısı`, `işletme kaldıraçları` çerçevesi korunacak.
