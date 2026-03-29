# İstanbul Baraj Doluluk ve İklim Etki Analizi

Bu proje, İstanbul baraj doluluk oranlarının iklim değişkenleri ve kullanım baskısı ile birlikte değerlendirilmesini amaçlayan akademik nitelikli bir karar-destek çalışmasıdır. Çalışma, Kandilli Rasathanesi ve Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü hackathon bağlamında geliştirilmiştir.

## Canlı Yayın
- Ana sayfa: [https://yasinkaya701.github.io/istanbul-baraj-doluluk-orani-projeksiyonu/](https://yasinkaya701.github.io/istanbul-baraj-doluluk-orani-projeksiyonu/)
- Elmalı sayfası: [https://yasinkaya701.github.io/istanbul-baraj-doluluk-orani-projeksiyonu/baraj_web/elmali.html](https://yasinkaya701.github.io/istanbul-baraj-doluluk-orani-projeksiyonu/baraj_web/elmali.html)

## Bilimsel Çerçeve
- Amaç: Doluluk oranı dinamiklerini çok etmenli bir yapıda analiz etmek ve 2040 ufkunda senaryo duyarlılığı üretmek.
- Yaklaşım: ET0 (FAO-56 Penman-Monteith referans çerçevesi), yağış ve kullanım bileşenlerinin birlikte değerlendirildiği model-temelli projeksiyon.
- Model ailesi: Ridge, GBR, HGB, RF ve ETR başta olmak üzere karşılaştırmalı modelleme.
- Değerlendirme mantığı: Tek bir sayı yerine dağılım, belirsizlik bandı ve senaryo etkisi üzerinden yorum.

## Kurumsal Veri Kaynakları
- İSKİ (İstanbul Su ve Kanalizasyon İdaresi)
- İBB Açık Veri Portalı
- Kandilli Rasathanesi bağlamında kullanılan iklim serileri

## Proje Çıktıları
- `hackhaton_model_kartlari_2026_03_18/`: 5 yıl/10 yıl model kartları ve performans görselleri
- `hackhaton_projection_2040_2026_03_18/`: 2040 projeksiyon grafik seti
- `Özetler/`: proje özeti ve rapor PDF paketleri
- `references.html` ve `REFERENCES.md`: kaynakça ve izlenebilirlik dokümantasyonu

## Depo Yapısı
```text
istanbul-baraj-doluluk-orani-projeksiyonu/
|-- index.html
|-- baraj_web/
|-- assets/
|-- styles.css
|-- references.html
|-- REFERENCES.md
|-- hackhaton_model_kartlari_2026_03_18/
|-- hackhaton_projection_2040_2026_03_18/
|-- Özetler/
```

## Veri Yönetişimi
- Bu depoda ham veri setleri paylaşılmaz.
- Yayınlanan içerik, arayüzde kullanılan türetilmiş veri dosyaları ve raporlanmış çıktı paketleriyle sınırlandırılmıştır.
- Kurumsal karar süreçlerinde, ilgili kurumların güncel resmi yayınları ile birlikte değerlendirme yapılmalıdır.

## English Summary
This repository hosts an academic decision-support web project for Istanbul reservoir occupancy projections. The framework integrates climate drivers (including ET0), water-use pressure, and comparative model families to assess medium-term behavior up to 2040.

### Scope
- Web-based projection dashboard (Istanbul + Elmali subpage)
- Model comparison outputs and projection figure packs
- Structured references for transparency and traceability

### Data Governance
- Raw datasets are intentionally excluded from this repository.
- Only derived, publication-ready artifacts are maintained for reproducible presentation.

## Kaynaklar
- Web kaynakça: [`references.html`](references.html)
- Detaylı kaynakça: [`REFERENCES.md`](REFERENCES.md)

## Kullanım Çerçevesi
Bu içerik eğitim, araştırma ve hackathon sunum amaçlıdır; operasyonel uygulamalarda güncel resmi veri doğrulaması esastır.
