# Strong Ensemble Kullanımı

Bu pipeline, tek model yerine hibrit bir ensemble kurar:

- Prophet
- ETS (Exponential Smoothing)
- SARIMA
- Lag-Feature Ridge
- Seasonal Naive

Her model rolling CV ile ölçülür; en iyi ve en stabil modeller ağırlıklandırılarak final tahmin üretilir.

## Çalıştırma

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/train_strong_consistent_model.py \
  --observations output/forecast_package/observations_with_graph.parquet \
  --target-year 2035 \
  --holdout-steps 12 \
  --backtest-splits 3 \
  --min-train-steps 36 \
  --max-ensemble-models 3 \
  --output-dir output/strong_ensemble_package
```

## Çıktılar

- `output/strong_ensemble_package/strong_ensemble_index_to_2035.csv`
- `output/strong_ensemble_package/leaderboards/*.csv`
- `output/strong_ensemble_package/forecasts/*.csv`
- `output/strong_ensemble_package/charts/*.png`
- `output/strong_ensemble_package/reports/*.json`

## Notlar

- Veri seyrekse pipeline otomatik olarak yıllık frekansa geçebilir.
- Yetersiz veri olan serilerde (ör. kısa sıcaklık geçmişi) daha basit modeller öne çıkabilir.
- Ağırlıklar `index` dosyasındaki `ensemble_weights_json` alanında yazılır.
