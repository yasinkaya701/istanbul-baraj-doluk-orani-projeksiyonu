# İstanbul Baraj Doluluk Tahmini
## Kandilli Rasathanesi & Boğaziçi Üniversitesi Bilgisayar Mühendisliği Hackathon Projesi

Bu depo, İstanbul baraj sisteminde doluluk dinamiklerini iklim sürücüleri (yağış, ET0), kullanım baskısı ve çoklu model yaklaşımıyla analiz etmek için hazırlanmış hackathon odaklı bir karar-destek çalışmasıdır. İçerik, teknik jüri ve alan uzmanları tarafından incelenebilecek akademik/teknik sunum standardında düzenlenmiştir.

## Canlı Yayın
- Ana sayfa: [İstanbul Baraj Projeksiyonu](https://yasinkaya701.github.io/istanbul-baraj-doluluk-orani-projeksiyonu/)
- Elmalı alt modülü: [Elmalı Barajı Sayfası](https://yasinkaya701.github.io/istanbul-baraj-doluluk-orani-projeksiyonu/baraj_web/elmali.html)

## Hackathon Bağlamı
- Proje türü: Uygulamalı hidro-iklim modelleme ve karar desteği
- Kurumsal bağlam: Kandilli Rasathanesi ve Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü hackathonu
- Hedef: 2040 ufkunda belirsizlik bandı ile birlikte açıklanabilir doluluk projeksiyonu üretmek

## Kapsam
- 2000–2040 zaman ufkunda baraj doluluk projeksiyonları
- ET0 tabanlı buharlaşma etkisi (FAO-56 Penman-Monteith referans çerçevesi)
- Kullanım etkisi ve kullanım artışı senaryolarının simülasyon motoruna entegrasyonu
- Temel model ailesi (Ridge, GBR, HGB, RF, ETR) + gelişmiş karşılaştırma modelleri
- Elmalı Barajı için ayrı modelleme ve simülasyon katmanı

## Öne Çıkan Tahmin Sonuçları
- Çapraz doğrulama (Purged Walk-Forward, 3 yıl) sonuçlarında en yüksek korelasyon: **XGBoost (r=0.839)**.
- Aynı tabloda en düşük RMSE değeri: **XGBoost (RMSE=39.33 puan)**.
- Düşük varyanslı lineer referans: **RidgeCV (r=0.685, RMSE=40.12 puan)**.
- Toplulaştırma yaklaşımı (**Stacking** ve **Ensemble Median**) tekil model yanlılığını azaltmak için karar notlarında temel referans olarak kullanılır.

Not: Bu metrikler model karşılaştırma panelindeki güncel sonuçlardan türetilmiştir ve yeni veri güncellemesiyle birlikte yenilenir.

## Kurumsal Veri Kaynakları
- Operasyonel baraj/doluluk ve su arzı verileri: **İSKİ** ve **İBB Açık Veri Portalı**
- İklim/ET0 sürücüleri: Proje akışında standardize edilen iklim serileri (Kandilli odaklı çalışma çerçevesi)

Not: Kaynak izlenebilirliği için web kaynakçası ve detaylı kaynakça birlikte tutulur.

## Veri Politikası
Bu repoda ham veri setleri ve büyük ara çıktılar tutulmaz.

Git dışında bırakılan klasörler:
- `DATA/`
- `new data/`
- `output/`
- `tmp/`

Repoda tutulan veri tipleri:
- Arayüzün çalışması için gerekli türetilmiş JS payload dosyaları
- `assets/data/*.js`
- `baraj_web/assets/data/*.js`

## Proje Yapısı
```text
.
|-- index.html
|-- baraj_web/
|-- assets/
|-- scripts/
|-- dashboard/
|-- research/
|-- references.html
|-- REFERENCES.md
```

## Hızlı Başlangıç (Lokal)
```bash
python3 -m http.server 8000 --directory baraj_web
```
Ardından tarayıcıda `http://localhost:8000` adresini açın.

## Güncelleme Akışı
```bash
python3 scripts/update_sim_inputs.py
```
Bu adım, simülasyon girişleri (kullanım profili, katsayılar, denge girdileri) için güncel hesapları üretir.

## Branch ve Yayın
- `main`: geliştirme ve içerik güncellemeleri
- `gh-pages`: canlı yayın branch'i

## Kaynakça
- Web kaynakça: [`references.html`](references.html)
- Detaylı kaynakça: [`REFERENCES.md`](REFERENCES.md)

## Kullanım Notu
Bu çalışma eğitim, araştırma ve hackathon değerlendirmesi amaçlıdır. Operasyonel kararlar için resmi kurum yayınları ve güncel doğrulama çıktıları ile birlikte değerlendirilmelidir.
