# ET0 ML Kullanim Rehberi

Bu dokuman, `scripts/et_ml.py` ile ET0 (referans evapotranspirasyon) tahmininin
nasıl egitildigini, hangi ciktilarin uretildigini ve 1987 kosusunda elde edilen
dogruluk sonuclarini ozetler.

## 1) Ne Yapar?

`et_ml.py` su akisi calistirir:

1. `et0_inputs_completed_1987.csv` verisini okur.
2. `water_balance_partial_1987.csv` ile gunluk yagis proxy (`P`) olusturur.
3. Ozellik muhendisligi yapar (`lag`, `moving average`, mevsimsellik, anomali).
4. Asagidaki modelleri egitir:
   - Random Forest
   - Gradient Boosting
   - LSTM (yalnizca TensorFlow varsa ve `--run-lstm` verilirse)
5. Performans metriklerini ve grafikleri yazar.

## 2) Calistirma

Proje kokunden:

```bash
cd /Users/yasinkaya/Hackhaton
python3 scripts/et_ml.py --year 1987 --out-dir output/spreadsheet
```

LSTM denemek icin (TensorFlow kuruluysa):

```bash
python3 scripts/et_ml.py --year 1987 --run-lstm --lstm-window 30 --out-dir output/spreadsheet
```

Yeni/genis veriyle egitmek icin:

```bash
python3 scripts/et_ml.py \
  --data-source wide \
  --wide-csv output/spreadsheet/meteoroloji_model_egitim_wide_genisletilmis_filled.csv \
  --out-dir output/spreadsheet
```

## 3) Girdi Dosyalari

- `output/spreadsheet/et0_inputs_completed_1987.csv`
- `output/spreadsheet/water_balance_partial_1987.csv`

Beklenen ana kolonlar:

- ET0 seti: `date, t_mean_c, t_min_c, t_max_c, rh_mean_pct, u2_m_s, rs_mj_m2_day, et0_completed_mm_day`
- Su butcesi seti: `month, precip_obs_mm`

## 4) Cikti Dosyalari

- Metrikler: `output/spreadsheet/et0_ml_metrics_1987.csv`
- Tahmin satirlari: `output/spreadsheet/et0_ml_predictions_1987.csv`
- Dogruluk ozeti: `output/spreadsheet/et0_ml_accuracy_1987.csv`
- Yeni/genis veri metrikleri: `output/spreadsheet/et0_ml_metrics_wide_1918_2019.csv`
- Yeni/genis veri tahminleri: `output/spreadsheet/et0_ml_predictions_wide_1918_2019.csv`
- Yeni/genis veri dogruluk ozeti: `output/spreadsheet/et0_ml_accuracy_wide_1918_2019.csv`
- Ozellik onemi grafigi: `output/spreadsheet/rf_ozellik_onem.png`
- Performans karsilastirma grafigi: `output/spreadsheet/ml_performans.png`

## 5) 1987 Sonuclari (Mevcut Kosu)

### Model performansi

| Model | RMSE (mm/gun) | MAE (mm/gun) | R2 |
|---|---:|---:|---:|
| Gradient Boosting | 0.268 | 0.214 | 0.7609 |
| Random Forest | 0.454 | 0.398 | 0.3160 |

### Dogruluk orani tablosu

| Model | R2 (%) | 100-MAPE (%) | 100-sMAPE (%) | ±0.25 mm isabet (%) | ±0.50 mm isabet (%) | Yon dogrulugu (%) |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 76.09 | 73.56 | 78.53 | 66.18 | 94.12 | 79.10 |
| Random Forest | 31.60 | 41.72 | 60.44 | 26.47 | 70.59 | 76.12 |

En iyi model: **Gradient Boosting**.

## 6) Dogruluk Metrikleri Nasil Okunur?

- `R2 (%)`: Aciklanan varyans orani (`r2 * 100`).
- `100-MAPE (%)`: Yuzdesel hata yaklasimindan turetilen sezgisel dogruluk.
- `100-sMAPE (%)`: Simetrik MAPE tabanli dogruluk.
- `±0.25 / ±0.50 mm isabet`: Tahminin mutlak hatasi belirli esiklerin altinda kalma orani.
- `Yon dogrulugu`: ET0'in bir onceki gune gore artis/azalis yonunu dogru tahmin etme orani.

## 7) Notlar

- Bu ortamda TensorFlow yoksa LSTM otomatik atlanir.
- Tek yil (1987) ile egitim yapildigi icin model genelleme yetenegi sinirlidir.
- Daha guvenilir modeller icin cok yillik ET0 + meteoroloji serisi ile yeniden egitim onerilir.
- `wide` modunda ET0 etiketi genis meteoroloji verisinden FAO-56 PM ile turetilir; bu nedenle skorlar cok yuksek cikabilir.
