<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00C6FF,50:3F51B5,100:F7971E&height=180&section=header&text=Hackhaton&fontSize=54&fontColor=FFFFFF&animation=twinkling" />
  <h3>Su, iklim ve baraj analitiği çalışma alanı</h3>
  <p>Araştırma, modelleme ve karar destek panelleri bir arada.</p>
  <p><strong>Not:</strong> Bu çalışma Kandilli Rasathanesi ve Boğaziçi Üniversitesi Bilgisayar Mühendisliği Bölümü Hackhatonu için yapılmıştır.</p>
  <p>
    <img src="https://img.shields.io/badge/Odak-Climate%20%26%20Water-00BCD4" />
    <img src="https://img.shields.io/badge/Stack-Python%20%2B%20JS-1E88E5" />
    <img src="https://img.shields.io/badge/Çıktılar-Reports%20%26%20Charts-F4511E" />
    <img src="https://img.shields.io/badge/Data-Ignored%20in%20Git-8E24AA" />
  </p>
</div>

---

## Hızlı harita
```text
Hackhaton/
|-- dashboard/                      # Paneller için frontend + server
|-- scripts/                        # Modelleme, veri hazırlama ve analiz akışları
|-- research/                       # Araştırma hub, loglar ve template'ler
|-- baraj_web/                      # Statik web sunumu
|-- external/ArtikongrafConverter/  # Harici araç (entegre)
|-- index.html                      # Legacy web sunumu (kök)
|-- assets/                         # Legacy web sunumu asset'leri
|-- styles.css                      # Legacy web sunumu stil dosyası
|-- netlify.toml                    # Legacy web sunumu deploy ayarı
|-- hackhaton_model_kartlari_2026_03_18/  # Model kartları (görsel)
|-- hackhaton_projection_2040_2026_03_18/ # Projeksiyon görselleri
|-- ADVERSARIAL_TEST/               # Test görüntüleri (git dışı)
|-- DESKTOP_PROCESSING/             # Yerel işleme çıktıları (git dışı)
|-- DATA/                           # Yerel veri setleri (git dışı)
|-- new data/                       # Yerel veri setleri (git dışı)
|-- output/                         # Üretilen çıktılar (git dışı)
|-- tmp/                            # Geçici dosyalar (git dışı)
|-- baraj_web/assets/data/          # Üretilen veri (git dışı)
```

## İçerik özeti
- Baraj doluluk, ET0 ve iklim sinyalleri için uçtan uca modelleme scriptleri
- Karar destek dashboard'ları
- Log, template ve sentez notları ile araştırma hub yapısı
- Model kartları ve projeksiyon görselleri

## Modeller ve yöntemler (özet)
- ET0 hesaplama: FAO-56 Penman-Monteith yaklaşımı, günlük/aylık/yıllık paketler
- Gözetimli ML modelleri: Ridge, Random Forest (RF), Gradient Boosting (GBR), HistGradientBoosting (HGB), ExtraTrees (ETR)
- Ensemble ve konsensus paketleri: farklı model ailelerinden stabil birleşimler
- Su dengesi ve senaryo simülasyonları: iklim, kullanım ve buharlaşma etkileri ile hacim tabanlı güncelleme
- Olasılıksal yaklaşım: quantile tabanlı tahminler ve belirsizlik bantları
- Konformal/kalibrasyon adımları: doğruluk ve güven aralığı iyileştirme
- Validasyon ve stres testleri: holdout, walk-forward, senaryo testleri

## Web sunumu
- Tercih edilen sunum: `baraj_web/index.html`
- Legacy sunum (kök dizin): `index.html`
- Basit statik sunucu için:
```bash
cd baraj_web
python -m http.server 8000
```
Sonra `http://localhost:8000`.

## Rapor ve ET0 yeniden üretim
### PDF raporlar
```bash
python scripts/build_istanbul_current_status_pdf.py
python scripts/build_istanbul_current_status_detailed_pdf.py
python scripts/build_hackathon_final_pdf_report.py
```

### ET0 paket (gerçek radyasyon)
```bash
python scripts/build_es_ea_newdata_csv.py \
  --temp-xlsx "<TEMP_XLSX>" \
  --humidity-xlsx "<HUMIDITY_XLSX>" \
  --auto-table1 "<AUTO_TABLE1>" \
  --auto-table2 "<AUTO_TABLE2>"

python scripts/build_complete_solar_dataset.py
python scripts/build_tarim_et0_real_radiation_package.py
python scripts/build_et0_formula_card.py
python scripts/build_one_year_explained_et0_charts.py --year 2004
python scripts/build_et0_trend_robust_chart.py
python scripts/build_et0_multiscale_charts.py
```

## Veri notu
Veri setleri bilerek git dışında tutulur. Yerel verileri şu dizinlere koy:
- `DATA/`
- `new data/`
- `baraj_web/assets/data/`

Yaygın veri formatları (ör. `*.csv`, `*.parquet`, `*.xlsx`, `*.pkl`) `.gitignore` ile dışarıda tutulur.

## Kaynakça (özet)
- İBB Açık Veri API (CKAN): [data.ibb.gov.tr](https://data.ibb.gov.tr/api/3/action/datastore_search)
- İBB yetki alanı (havza alanı hesabı): [ibb.istanbul](https://www.ibb.istanbul/ibb/belediye-hakkinda/yetki-alani/)
- İSKİ baraj doluluk: [iski.istanbul/baraj-doluluk](https://iski.istanbul/baraj-doluluk)
- İSKİ su kaynakları: [iski.istanbul/su-kaynaklari](https://iski.istanbul/kurumsal/hakkimizda/su-kaynaklari)
- İSKİ su birim fiyatları: [iski.istanbul/su-birim-fiyatlari](https://iski.istanbul/abone-hizmetleri/abone-rehberi/su-birim-fiyatlari/)
- İSKİ su kayıpları raporları: [iski.istanbul/su-kayiplari](https://iski.istanbul/kurumsal/stratejik-yonetim/su-kayiplari-yillik-raporlari/)
- FAO-56 Penman-Monteith (PDF): [fao56.pdf](https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/fao56.pdf)
- FAO Penman-Monteith notu: [fao.org](https://www.fao.org/4/X0490E/x0490e06.htm)
- WMO Guide (WMO-No.8): [wmo.int](https://wmo.int/guide-instruments-and-methods-of-observation-wmo-no-8-0)
- Open-Meteo Historical Weather API: [open-meteo.com](https://open-meteo.com/en/docs/historical-weather-api)
- NOAA NAO indeks serisi: [cpc.ncep.noaa.gov](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii)
- HEC USACE Penman-Monteith yöntemi: [hec.usace.army.mil](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/evaporation-and-transpiration/penman-monteith-method)
- 2023 su kullanımı haber kaynağı: [aa.com.tr](https://www.aa.com.tr/tr/gundem/istanbulda-gecen-yil-1-milyar-117-milyon-64-bin-116-metrekup-su-kullanildi/3104905)
- Forecasting: Principles and Practice (FPP3): [otexts.com/fpp3](https://otexts.com/fpp3/)
- Detaylı ve tam bağlantı listesi: `REFERENCES.md`

## Repo notu
- Önceki alt repolar birleştirildi. Eski git metadata yedekleri yerel olarak `.git_backup/` altında saklanır ve git dışındadır.

## Konvansiyonlar
- Büyük binary ve üretilen çıktıları git dışında tut
- Yeni pipeline'lar için `scripts/` altına kısa not ekle

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=0:3F51B5,100:00C6FF&height=16&section=footer" />
</div>
