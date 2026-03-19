# Prophet Kullanım Kılavuzu

Bu dosya, `scripts/prophet_climate_forecast.py` scriptini hackathon verisiyle hızlıca çalıştırmak için hazırlanmıştır.

## 1) Kurulum

```bash
pip install prophet pandas pyarrow matplotlib openpyxl
```

## 2) Birleşik gözlem tablosu (önerilen)

`timestamp + variable + value (+ qc_flag)` yapısı için:

```bash
python3 scripts/prophet_climate_forecast.py \
  --observations output/forecast_package/observations_with_graph.parquet \
  --input-kind auto \
  --target-year 2035 \
  --auto-tune true \
  --backtest-splits 4 \
  --holdout-months 18 \
  --min-train-months 36 \
  --output-dir output/prophet_package
```

## 3) Tek seri veri (ör: sadece sıcaklık CSV)

```bash
python3 scripts/prophet_climate_forecast.py \
  --observations output/1987_hourly_temp.csv \
  --input-kind single \
  --timestamp-col timestamp \
  --value-col temp_c \
  --single-variable temp \
  --variables temp \
  --target-year 2030 \
  --auto-tune false \
  --output-dir output/prophet_single_temp
```

## 4) Üretilen dosyalar

- `output/.../forecasts/*.csv`: Tarihsel + gelecek tahmin serisi (`is_forecast` kolonu ile ayrılır)
- `output/.../charts/*.png`: Kesiksiz tarihsel+tahmin grafik
- `output/.../components/*.png`: Trend/sezonsallık bileşen grafikleri
- `output/.../leaderboards/*.csv`: Denenen hiperparametrelerin backtest sıralaması
- `output/.../prophet_index_to_<year>.csv`: Tüm çıktıları tek indeks dosyasında toplar

## 5) Kritik parametreler

- `--auto-tune true`: Prophet hiperparametre taraması açılır (daha yavaş, daha güçlü)
- `--holdout-months`: Her fold için kaç ay test yapılacağı
- `--backtest-splits`: Kaç fold yapılacağı
- `--min-train-months`: Prophet eğitimi için minimum ay sayısı
- `--winsor-quantile`: Aykırı değer kırpma seviyesi (0.90-1.0)

## 6) Model stratejisi alanı

`prophet_index_to_*.csv` dosyasında `model_strategy`:

- `prophet_tuned`: Grid + backtest ile seçilen Prophet
- `prophet_default`: Tek konfigürasyonla Prophet
- `seasonal_naive_fallback`: Veri kısa ise emniyetli fallback

