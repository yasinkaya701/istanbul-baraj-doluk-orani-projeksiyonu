# YENI MODEL (newdata ayrik)

Bu klasor, Hackhaton icinde `newdata` kaynagi ile calisan ayri model kodlarini tutar.

## Kodlar
- `quant_regime_projection_yeni_model.py`: Yeni modele ayrilmis quant kod kopyasi
- `run_yeni_model_newdata.sh`: Yeni modeli calistiran script
- `../run_quant_extreme_event_pipeline.py`: Quant output'u asiri olay + anomaly-day tablosuna baglayan post-process runner

## Varsayilan veri (newdata)
- `/Users/yasinkaya/Hackhaton/output/model_suite_retrain_newdata_20260306_234718/prepare_calibrated/observations_calibrated_full.parquet`

## Calistirma
```bash
/Users/yasinkaya/Hackhaton/scripts/yeni_model_newdata/run_yeni_model_newdata.sh
```

Opsiyonel:
```bash
/Users/yasinkaya/Hackhaton/scripts/yeni_model_newdata/run_yeni_model_newdata.sh <input_parquet> <target_year> <analysis_mode> <news_catalog_csv> <news_window_days> <history_start> <history_end> <dense_window_mode> <run_event_pipeline> <daily_csv>
```

`analysis_mode` secenekleri:
- `anomalies_only` (varsayilan): gelecek projeksiyonu devre disi, tarihsel tum anomali noktalarini yazar
- `full`: projeksiyon + anomali birlikte calisir (gelecek kismi silinmedi, sadece modla acilip kapanir)

Haber entegrasyonu:
- Varsayilan haber dosyasi: `/Users/yasinkaya/Hackhaton/output/extreme_events/news_expanded_v3_relaxed/meteoroloji_haber_baslik_katalogu.csv`
- Anomali kayitlarina haber basligi, kaynak, URL ve haber-eslesme skoru eklenir.

Extreme-event entegrasyonu:
- Varsayilan: `run_event_pipeline=true`
- Quant run tamamlaninca ayni output klasorunde `extreme_events/` altina su zincir otomatik yazilir:
  - `tum_asiri_olay_noktalari.csv`
  - `tum_asiri_olaylar.csv`
  - `tum_asiri_olaylar_zengin.csv`
  - `tum_asiri_olaylar_internet_nedenleri.csv`
  - `tum_asiri_olaylar_bilimsel_filtreli.csv`
  - `anomaly_day_data/anomaly_unique_days_with_daily_climate.csv`
- Varsayilan daily climate kaynak dosyasi:
  `/Users/yasinkaya/Hackhaton/output/spreadsheet/es_ea_newdata_daily.csv`
- Kapatmak icin 9. arguman olarak `false` verilebilir.

Tarih penceresi:
- Varsayilan: `1900-01-01` -> `2020-12-31`
- Veri bu pencereye gore filtrelenir; veri olmayan donemler otomatik bos kalir.

Yogunluk filtresi:
- Her degisken icin otomatik olarak en yogun ve kesintisiz yil blogu secilir.
- Amaç: 1918 gibi seyrek tekil kayitlarin modeli bozmasini engellemek.
- `dense_window_mode=off` verilirse bu filtre kapanir ve degiskenin tum satirlari kullanilir.
