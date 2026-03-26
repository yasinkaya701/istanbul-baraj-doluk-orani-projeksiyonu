# İstanbul Baraj Doluluk ve İklim Etki Analizi

Bu depo, İstanbul baraj doluluk dinamiklerini iklim sürücüleri, buharlaşma (ET0), kullanım baskısı ve model projeksiyonları ile birlikte inceleyen hackathon/proje çalışma alanıdır.

Bu çalışma, Kandilli Rasathanesi ve Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü hackathon bağlamında geliştirilmiştir.

## Canlı Site
- Ana yayın: [https://yasinkaya701.github.io/istanbul-baraj-web/](https://yasinkaya701.github.io/istanbul-baraj-web/)
- Elmalı sayfası: [https://yasinkaya701.github.io/istanbul-baraj-web/baraj_web/elmali.html](https://yasinkaya701.github.io/istanbul-baraj-web/baraj_web/elmali.html)

## Kurumsal Kaynak Ayrımı
- Operasyonel baraj/doluluk ve su arzı verileri: **İSKİ** ve **İBB Açık Veri Portalı**.
- İklim ve ET0 sürücü serileri: Kandilli odağında derlenmiş ve proje akışında standardize edilmiş iklim zaman serileri.
- Bu ayrım, model çıktılarının kaynak güvenilirliği ve akademik izlenebilirliği için korunur.

## Proje Kapsamı
- Baraj doluluk projeksiyonları (2000–2040 zaman ufku)
- ET0 tabanlı buharlaşma etkisi (FAO-56 Penman-Monteith referans çerçevesi)
- Kullanım/kayıp etkilerinin senaryo tabanlı simülasyonu
- Çoklu model karşılaştırmaları (temel modeller + gelişmiş modeller)
- Elmalı özelinde ayrı model ve simülasyon ekranı

## Dizin Yapısı
```text
Hackhaton/
|-- index.html                      # Ana arayüz (kök)
|-- baraj_web/                      # Yayınlanan web dosyaları
|-- assets/                         # Kök arayüz için JS/CSS/görsel varlıklar
|-- scripts/                        # Veri hazırlama, kalibrasyon, üretim scriptleri
|-- dashboard/                      # Dashboard bileşenleri
|-- research/                       # Araştırma notları ve çalışma dokümanları
|-- references.html                 # Web kaynakça sayfası
|-- REFERENCES.md                   # Detaylı kaynak listesi
|-- output/                         # Yerel üretim çıktıları (git dışı)
|-- DATA/                           # Ham veri klasörü (git dışı)
|-- new data/                       # Ham/ara veri klasörü (git dışı)
```

## Temel Model Seti
- Ridge (simülasyonda varsayılan referans model)
- GBR (Gradient Boosting Regressor)
- HGB (HistGradientBoosting Regressor)
- RF (Random Forest Regressor)
- ETR (Extra Trees Regressor)

Not: Model ailesi genişletmeleri ve ensemble çıktıları arayüzde ayrıca sunulur.

## Simülasyon Çekirdeği (Özet)
- Aylık su dengesi mantığı kullanılır.
- Senaryo girdileri: yağış değişimi, ET0 değişimi, yıllık kullanım ve kullanım trendi.
- Etkiler kapasite ve kalibrasyon katsayıları ile sınırlandırılarak uygulanır.
- Amaç tek nokta tahmin değil; karşılaştırmalı karar desteğidir.

## Veri Politikası (Kritik)
Bu repoda ham veri setleri tutulmaz.

Git dışında tutulanlar:
- `DATA/`
- `new data/`
- `output/`
- `tmp/`
- büyük dosya tipleri (`*.csv`, `*.xlsx`, `*.parquet`, `*.pkl`, vb.)

Repoda tutulanlar:
- Arayüzde kullanılan türetilmiş JS veri payload dosyaları (`assets/data/*.js`, `baraj_web/assets/data/*.js`)

## Lokal Çalıştırma
```bash
cd /Users/yasinkaya/Hackhaton/baraj_web
python3 -m http.server 8000
```
Sonra tarayıcıdan `http://localhost:8000` adresini açın.

## Sık Kullanılan Güncelleme Komutları
```bash
cd /Users/yasinkaya/Hackhaton
python3 scripts/update_sim_inputs.py
```
Bu akış, kullanım trendi/profili, simülasyon katsayıları ve temel denge verilerini günceller.

## Dağıtım Notu
- `main`: geliştirme ve içerik güncellemeleri
- `gh-pages`: canlı yayın branch'i

Sadece dokümantasyon (`README.md`) değişiklikleri, tek başına web uygulamasının çalışma mantığını değiştirmez.

## Kaynaklar
- Web kaynakça: [`references.html`](references.html)
- Detaylı kaynakça: [`REFERENCES.md`](REFERENCES.md)

## Lisans ve Kullanım
Bu depo eğitim, araştırma ve hackathon sunum amaçlıdır. Kurumsal/operasyonel kararlar için resmi kurum yayınları ve güncel doğrulama ile birlikte değerlendirilmelidir.
