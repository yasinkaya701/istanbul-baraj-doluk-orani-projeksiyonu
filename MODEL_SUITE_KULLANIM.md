# Model Suite Kullanım Rehberi

Bu rehber, hackathon içindeki **tüm ana modelleri** (`quant`, `prophet`, `prophet_ultra`, `strong`, `analog`, `walkforward`, `best_meta`, `stable_consensus`, `literature`)  
**tek komutla** farklı datasetlerde çalıştırmak için hazırlanmıştır.

## 1) Tek Komut Çalıştırıcı

- Script: `scripts/model_suite_runner.py`
- Kısa yol: `scripts/run_model_suite.sh`
- Entegre kısa yol (model + sağlık): `scripts/run_integrated_pipeline.sh`

Genel kullanım:

```bash
cd /Users/yasinkaya/Hackhaton
./scripts/run_model_suite.sh --dataset <dosya_veya_klasor> --models quant,prophet,strong,analog,prophet_ultra,walkforward,best_meta,stable_consensus,literature

# veya model + sağlık entegrasyonu için:
./scripts/run_integrated_pipeline.sh --dataset <dosya_veya_klasor> --models quant,prophet,strong,analog,prophet_ultra,walkforward,best_meta,stable_consensus,literature
```

Not:
- `model_suite_runner.py` artık varsayılan olarak sağlık suite adımını da çalıştırır (`--run-health-suite true`).
- İstenmezse kapat: `--run-health-suite false`
- `model_suite_runner.py` artık varsayılan olarak gözlem stabilizasyonu uygular (`--stabilize-observations true`):
  - basınç ölçek kalibrasyonu
  - büyük zaman boşluklarında recent regime seçimi
  - recent regime çok dar kalırsa full calibrated fallback
- `model_suite_runner.py` varsayılan olarak robust model seçimi de üretir (`--run-robust-selection true`):
  - model indekslerinden hata/stabilite sinyali toplar
  - mümkün olan modellerde gözlem seti ile overlap hata ölçümü (RMSE/bias/skill) yapar
  - değişken bazında en güvenilir modeli otomatik seçer
  - birleşik seçili tahmin dosyaları üretir (`robust_selection/`)

## 1.1) Data Factory (Kayıpsız Hazırlık + Opsiyonel Quant)

Eğer amaç önce **ham veriyi kayıpsız şekilde tek formata çevirmek** ise bu scripti kullan:

- Script: `scripts/run_data_factory.py`

Örnek:

```bash
python3 scripts/run_data_factory.py \
  --dataset-root /Users/yasinkaya/Hackhaton/DATA \
  --output-root /Users/yasinkaya/Hackhaton/output/data_factory \
  --run-quant false
```

Quant ile birlikte:

```bash
python3 scripts/run_data_factory.py \
  --dataset-root /Users/yasinkaya/Hackhaton/DATA \
  --output-root /Users/yasinkaya/Hackhaton/output/data_factory \
  --run-quant true \
  --target-year 2035 \
  --variables temp,humidity,pressure,precip
```

Data factory çıktıları:

1. `bronze` (kayıpsız görsel manifest):
   - `visual_process_report.csv`
   - Her görsel dosya için infer edilen tarih/değişken + extraction kalite metrikleri
2. `silver` (kayıpsız birleşik gözlem):
   - `observations_lossless_silver.parquet/csv`
   - Numeric + görselden çıkarılan tüm ölçümler (model değişkeni filtresi olmadan)
3. `gold` (model-eğitime hazır):
   - `observations_with_all_visuals_for_quant.parquet/csv`
   - `temp/humidity/pressure/precip` + aynı gün/değişkende numeric öncelikli deduplikasyon

## 1.2) Kalibrasyon + Rejim Temizleme (Önerilen)

Grafik ve tablo kaynakları farklı ölçeklerde olabildiği için, model eğitiminden önce
kalibrasyon adımı önerilir:

```bash
python3 scripts/calibrate_observations_for_forecast.py \
  --input-observations /Users/yasinkaya/Hackhaton/output/data_factory/<run>/prepared/observations_with_all_visuals_for_quant.parquet \
  --output-dir /Users/yasinkaya/Hackhaton/output/data_factory/<run>/calibrated \
  --gap-years 5 \
  --pressure-offset 900
```

Bu adım:
1. Basınçta düşük ölçekli numeric değerleri (`~40..120`) otomatik olarak `+900` ofsetiyle hPa bandına taşır.
2. Büyük zaman boşluklarını (ör. 1918 -> 1980 gibi) tespit eder ve her değişkende son sürekli rejimi çıkarır.
3. Model için daha stabil bir giriş üretir:
   - `observations_calibrated_recent_regime.parquet`

## 1.3) v4 Otomatik Model Seçimi (Arbitration)

Kalibre edilmiş birden fazla model çıktısı varsa, değişken bazında otomatik seçim:

```bash
python3 scripts/build_v4_arbitrated_forecast.py \
  --run-dir /Users/yasinkaya/Hackhaton/output/data_factory/<run>
```

Çıktılar (`quant/reports` altında):
1. `v4_candidate_scores.csv`
   - Tüm aday modellerin metrik + ceza + toplam skor tablosu
2. `v4_final_arbitrated_ozet.csv`
   - Her değişken için seçilen nihai model
3. `v4_final_arbitrated_dashboard.png`
   - Seçilen modellerle tek dashboard
4. `v4_final_arbitrated_yorum.md`
   - Türkçe kısa yorum

## 1.4) v5 Robust Arbitration (Onerilen)

v4'e ek olarak `v5` su iyilestirmeleri getirir:
1. Daha genis aday havuzu (`quant + strong + prophet + best_meta_v41`)
2. Gozlem ortusumune dayali ek hata metrikleri (`overlap_rmse`, `rolling_rmse`)
3. Dusuk ortusum / zayif interval / fizik disi patern cezalandirma
4. `identity-fit` tespiti (`yhat` gecmiste `actual` kopyasiysa ceza)

Calistirma:

```bash
python3 scripts/build_v5_robust_arbitrated_forecast.py \
  --run-dir /Users/yasinkaya/Hackhaton/output/data_factory/<run>
```

Ciktilar (`quant/reports`):
1. `v5_candidate_scores.csv`
2. `v5_final_arbitrated_ozet.csv`
3. `v5_final_arbitrated_dashboard.png`
4. `v5_confidence_panel.png`
5. `v5_final_arbitrated_yorum.md`
6. `v5_vs_v4_comparison.csv`
7. `v5_vs_v4_comparison.md`

## 1.5) v6 Stable Consensus (Stabil + Daha Dayanikli)

`v6`, `v5` aday skorlarini kullanip tek modele bagli kalmadan consensus tahmin uretir:
1. Degisken bazinda en iyi 2-3 adayi agirlikli birlestirir.
2. Aday agirliklarini gecmis performansa gore otomatik optimize eder.
2. Forecast baslangic sicrama duzeltmesi uygular.
3. Trend ve aylik salinim (amplitude) stabilizasyonu yapar.
4. Gecmis-ortusum residual'i ile lineer kalibrasyon (bias + slope) ve guven bandi kalibrasyonu uygular.

Calistirma:

```bash
python3 scripts/build_v6_stable_consensus_forecast.py \
  --run-dir /Users/yasinkaya/Hackhaton/output/data_factory/<run>
```

Ciktilar (`quant/reports`):
1. `v6_stable_consensus_ozet.csv`
2. `v6_stable_consensus_dashboard.png`
3. `v6_stable_consensus_yorum.md`
4. `v6_vs_v5_comparison.csv`
5. `v6_vs_v5_comparison.md`
6. `v6_vs_v5_stability_metrics.csv`

## 2) Desteklenen Veri Tipleri

`model_suite_runner.py` aşağıdaki kaynaklarla çalışır:

1. Hazır gözlem dosyası: `--prepared-observations`
   - Beklenen şema: `timestamp`, `variable`, `value` (opsiyonel: `qc_flag`)
   - Formatlar: `parquet`, `csv`, `tsv`, `xlsx`, `xls`, `ods`
2. Ham dataset klasörü: `--dataset <klasor>`
   - İçindeki tablo dosyaları `universal_climate_forecast_pipeline.py` ile parse edilir.
3. TIFF görseller: `--include-visuals true --graph-root <klasor>`
   - Tüm `.tif/.tiff` dosyaları işlenir ve gözlem setine eklenir.

## 3) Model Açıklamaları (Ne zaman kullanılır?)

1. `quant`
   - Rejim + volatilite tabanlı yaklaşım.
   - Anomali analizi ve belirsizlik bandı güçlüdür.
   - Varsayılan ana model olarak önerilir.

2. `prophet`
   - Hızlı, okunabilir trend/sezonsallık bazlı tahmin.
   - Kısa sürede baseline üretmek için uygundur.

3. `prophet_ultra`
   - Prophet odaklı daha kapsamlı CV/ayar süreci.
   - Prophet ailesinde daha agresif optimizasyon isterken kullanılır.

4. `strong`
   - Çoklu model ensemble (Prophet + ETS + SARIMA + Ridge + seasonal naive).
   - Tutarlılık odaklı hibrit tahmin için kullanılır.

5. `analog`
   - Geçmiş patern benzerliği + trend birleşimi.
   - “Geçmişte benzer dönem olmuş mu?” sorusu için uygundur.

6. `walkforward`
   - Her adımda yeniden eğitim (recursive retrain).
   - Yıllık/aylık/haftalık/günlük ileri yürüyen senaryolar için kullanılır.

7. `literature`
   - Literatür-temelli robust ensemble: rolling-origin CV + MASE/sMAPE + conformal belirsizlik.
   - Veri sızıntısına dayanıklı doğrulama ve kalibre hata bandı gerektiğinde önerilir.

## 4) Sık Kullanılan Komutlar

1. Hazır gözlem parquet’i ile tüm modeller:

```bash
./scripts/run_model_suite.sh \
  --prepared-observations /Users/yasinkaya/Hackhaton/output/quant_all_visuals_input/observations_with_all_visuals_for_quant.parquet \
  --models quant,prophet,strong,analog,prophet_ultra,walkforward,literature \
  --variables temp,humidity,pressure,precip \
  --target-year 2035
```

2. Ham klasörü parse ederek tüm modeller:

```bash
./scripts/run_model_suite.sh \
  --dataset /Users/yasinkaya/Hackhaton/DATA \
  --models quant,prophet,strong,analog,prophet_ultra,walkforward,literature \
  --variables temp,humidity,pressure,precip
```

3. Ham klasör + tüm TIFF görseller:

```bash
./scripts/run_model_suite.sh \
  --dataset /Users/yasinkaya/Hackhaton/DATA \
  --include-visuals true \
  --graph-root "/Users/yasinkaya/Hackhaton/DATA/Graf Kağıtları Tarama " \
  --models quant,prophet,strong,analog,prophet_ultra,walkforward,literature \
  --variables temp,humidity,pressure,precip
```

4. Sadece quant + walkforward:

```bash
./scripts/run_model_suite.sh \
  --prepared-observations /Users/yasinkaya/Hackhaton/output/quant_all_visuals_input/observations_with_all_visuals_for_quant.parquet \
  --models quant,walkforward \
  --walkforward-freqs YS,MS,W,D \
  --start-year 2026 \
  --target-year 2035
```

## 5) Çıktı Yapısı

Varsayılan ana çıktı klasörü:

- `/Users/yasinkaya/Hackhaton/output/model_suite`

Bu klasörde:

1. Model bazlı alt klasörler:
   - `quant/`
   - `prophet/`
   - `prophet_ultra/`
   - `strong/`
   - `analog/`
   - `walkforward/`
   - `literature/`
2. Koşu özeti:
   - `model_suite_summary.json`
   - `model_suite_summary.md`
3. (Gerekirse) hazırlık çıktıları:
   - `prepare_universal/`
   - `prepare_visuals/`

## 6) Önemli Parametreler

1. `--variables`
   - `*` veya `temp,humidity,pressure,precip`
2. `--target-year`
   - Tahmin ufku son yılı
3. `--models`
   - Virgülle model listesi
4. `--include-visuals true`
   - TIFF görselleri de sayısallaştırıp modele kat
5. `--fail-fast true`
   - Bir model hata verirse koşuyu hemen durdur

## 7) Bağımlılık Notu

Bazı modeller ek paket ister:

```bash
pip3 install pyarrow statsmodels prophet arch
```

`quant` için `arch`, Prophet tabanlı modeller için `prophet` gerekir.

## 8) Script İşlevleri (Hangi dosya ne yapar?)

1. `scripts/model_suite_runner.py`
   - Tüm modelleri tek komutta orkestre eder.
   - Girdi hazırlar, modelleri çalıştırır, merkezi özet (`model_suite_summary.*`) yazar.

2. `scripts/universal_climate_forecast_pipeline.py`
   - Ham tablo datasetlerini (`csv/xlsx/ods/parquet`) ortak gözlem formatına çevirir.
   - Çıktı: `observations_universal.*`, `parse_report.csv`, `source_summary.csv`.

3. `scripts/process_all_visuals_to_quant.py`
   - Tüm TIFF graf kağıtlarını sayısallaştırır.
   - Görselden ölçüm çıkarır ve modele uygun gözlem tablosu üretir.

4. `scripts/quant_regime_projection.py`
   - Rejim + volatilite + anomali analiziyle tahmin üretir.
   - Çıktı: forecast/anomaly/chart/report ve `quant_index`.

5. `scripts/prophet_climate_forecast.py`
   - Prophet tabanlı üretim pipeline’ı (dengeli, okunabilir baseline).

6. `scripts/prophet_ultra_500.py`
   - Prophet için daha kapsamlı CV/ayar taraması.
   - Daha ağır ama Prophet ailesinde daha agresif optimizasyon.

7. `scripts/train_strong_consistent_model.py`
   - Hibrit ensemble (Prophet + ETS + SARIMA + Ridge + Naive).
   - Amaç: stabilite ve genel hata azaltımı.

8. `scripts/analog_pattern_forecast.py`
   - Geçmiş patern benzerliği (analog) + trend birleşimi.
   - “Benzer geçmiş desen” analizi için kullanılır.

9. `scripts/walkforward_retrain_multifreq.py`
   - Her adımda yeniden eğitim yapar (recursive walk-forward).
   - `YS/MS/W/D` frekanslarında adım adım ileri tahmin verir.

10. `scripts/literature_robust_forecast.py`
   - Rolling-origin CV, MASE/sMAPE model ağırlıkları ve split-conformal aralık üretir.
   - Uygun olduğunda ENSO/NAO dışsal indekslerini de feature olarak kullanır.

## 9) Adım Adım Kullanım Akışı

1. Veri kaynağını seç:
   - Hazır gözlem: `--prepared-observations`
   - Ham klasör: `--dataset`

2. Görselleri dahil edip etmeyeceğini belirle:
   - Dahil: `--include-visuals true --graph-root ...`
   - Dahil değil: `--include-visuals false`

3. Model listesini seç:
   - Hepsi: `quant,prophet,strong,analog,prophet_ultra,walkforward,literature`
   - Hafif koşu: `quant,prophet`

4. Çalıştır:
   - `./scripts/run_model_suite.sh ...`

5. Sonuçları incele:
   - Ana özet: `output/model_suite/model_suite_summary.json`
   - Model alt klasörleri: `output/model_suite/<model_adi>/...`

## 10) Hata Durumları ve Çözüm

1. `ModuleNotFoundError`
   - Eksik paketi yükle: `pip3 install pyarrow statsmodels prophet arch`

2. `parquet engine` hatası
   - `pyarrow` kurulu değilse: `pip3 install pyarrow`

3. OMP/SHM veya thread hataları
   - Script zaten tek thread env set eder; yine olursa komutu `bash -lc` ile çalıştır.

4. Veri parse edilemedi
   - Önce sadece hazırlık adımını test et:
   - `python3 scripts/universal_climate_forecast_pipeline.py --input-dir <klasor> --output-dir <cikti>`

5. Model çok uzun sürüyor
   - `--models quant,prophet` ile daralt.
   - `--variables temp,humidity` gibi daha az değişken seç.
