# Istanbul Baraj Karar Destek Akisi (v2)

Bu akıs, Istanbul baraj doluluk tahminini karar-destek formatında üretir:

1. Temel karar-destek tahmini (seri bazlı strateji seçimi + olasılıklar)
2. Senaryo üretimi (`baseline`, `dry_stress`, `wet_relief`)
3. Senaryo seçmeli HTML panel
4. JSON erken uyarı çıktıları
5. Slack webhook payload çıktısı
6. Halk dili sonuç özeti + literatür uyum kontrolü + ek görseller
7. Dinamik eşik (mevsimsel kuantil) risk analizi
8. Kalibrasyon değerlendirmesi (CV reliability + interval coverage)
9. Aralık kapsama kalibrasyonu (coverage tuning)
10. Senaryo olasılık ağırlıklı beklenen risk özeti
11. Hikaye odaklı görsel paket (halk dili)

## 1) Temel karar-destek tahmini

```bash
python3 scripts/forecast_istanbul_dam_decision_support.py \
  --output-dir output/istanbul_dam_forecast_decision \
  --horizon-months 60
```

Stabil/dogru konfigürasyon (onerilen):

```bash
python3 scripts/forecast_istanbul_dam_decision_support.py \
  --output-dir output/istanbul_dam_forecast_decision \
  --horizon-months 60 \
  --cv-splits 5 \
  --cv-test-months 12 \
  --min-train-months 72 \
  --alpha 0.10 \
  --ensemble-max-models 3 \
  --ensemble-tie-margin 0.01 \
  --ensemble-shrink 0.00 \
  --enable-stacked-ensemble \
  --stack-l2 1.0 \
  --stack-blend-invscore 0.35 \
  --stack-min-weight 0.0 \
  --recent-split-weight 0.80 \
  --stability-penalty 0.10 \
  --bias-penalty 0.00 \
  --horizon-damping-start 18 \
  --horizon-damping-strength 0.35 \
  --interval-smoothing 0.35 \
  --no-auto-tune-selection
```

Ana çıktılar:

- `output/istanbul_dam_forecast_decision/istanbul_dam_forecasts_decision.csv`
- `output/istanbul_dam_forecast_decision/risk_summary_2026_03_to_2027_02.csv`
- `output/istanbul_dam_forecast_decision/strategy_summary.csv`

## 2) Senaryo üretimi

```bash
python3 scripts/build_istanbul_dam_scenarios.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-dir output/istanbul_dam_forecast_decision \
  --shift-mode series_month \
  --expand-severity-grid \
  --severity-levels mild:0.6,base:1.0,severe:1.4 \
  --dry-shift-k 0.8 \
  --wet-shift-k 0.8 \
  --max-shift-abs 0.12
```

Ana çıktılar:

- `output/istanbul_dam_forecast_decision/scenario_forecasts.csv`
- `output/istanbul_dam_forecast_decision/scenario_risk_summary.csv`
- `output/istanbul_dam_forecast_decision/scenario_catalog.csv`
- `output/istanbul_dam_forecast_decision/scenario_shift_factors.csv`
- `output/istanbul_dam_forecast_decision/scenario_seasonal_ratios.csv`
- `output/istanbul_dam_forecast_decision/scenario_shift_factors_heatmap.png`

## 3) Senaryo seçmeli dashboard (v2)

```bash
python3 scripts/build_istanbul_dam_decision_dashboard_v2.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-html output/istanbul_dam_forecast_decision/dashboard_v2.html
```

Çıktı:

- `output/istanbul_dam_forecast_decision/dashboard_v2.html`

## 4) JSON erken uyarı

Baseline tek senaryo:

```bash
python3 scripts/build_istanbul_dam_alerts.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-json output/istanbul_dam_forecast_decision/alerts_2026_03_2027_02.json
```

Çoklu senaryo:

```bash
python3 scripts/build_istanbul_dam_alerts_multi_scenario.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-json output/istanbul_dam_forecast_decision/alerts_multi_scenario.json
```

## 5) Slack webhook payload

```bash
python3 scripts/build_istanbul_dam_slack_payload.py \
  --alerts-json output/istanbul_dam_forecast_decision/alerts_multi_scenario.json \
  --output-json output/istanbul_dam_forecast_decision/slack_payload.json \
  --top-n 5
```

Çıktı:

- `output/istanbul_dam_forecast_decision/slack_payload.json`

## 6) Sonuç özeti + literatür uyum kontrolü

```bash
python3 scripts/build_istanbul_dam_public_report.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-dir output/istanbul_dam_forecast_decision \
  --window-start 2026-03-01 \
  --window-end 2027-02-01 \
  --top-n 6
```

Ana çıktılar:

- `output/istanbul_dam_forecast_decision/SONUC_OZETI_VE_LITERATUR_KONTROLU.md`
- `output/istanbul_dam_forecast_decision/risk_heatmap_prob_below_40.png`
- `output/istanbul_dam_forecast_decision/top_risk_compare_by_scenario.png`

## 7) Dinamik eşik risk analizi

```bash
python3 scripts/build_istanbul_dam_dynamic_threshold_risk.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-dir output/istanbul_dam_forecast_decision \
  --window-start 2026-03-01 \
  --window-end 2027-02-01 \
  --warn-quantile 0.25 \
  --critical-quantile 0.10
```

Ana çıktılar:

- `output/istanbul_dam_forecast_decision/scenario_dynamic_threshold_forecasts.csv`
- `output/istanbul_dam_forecast_decision/scenario_dynamic_risk_summary.csv`
- `output/istanbul_dam_forecast_decision/dynamic_thresholds_by_series_month.csv`
- `output/istanbul_dam_forecast_decision/dynamic_threshold_risk_counts.png`

## 8) Kalibrasyon değerlendirmesi

```bash
python3 scripts/evaluate_istanbul_dam_calibration.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-dir output/istanbul_dam_forecast_decision \
  --threshold-1 0.40 \
  --threshold-2 0.30 \
  --alpha 0.10
```

Ana çıktılar:

- `output/istanbul_dam_forecast_decision/calibration_metrics.csv`
- `output/istanbul_dam_forecast_decision/calibration_reliability.csv`
- `output/istanbul_dam_forecast_decision/calibration_reliability_overall.png`
- `output/istanbul_dam_forecast_decision/calibration_interval_coverage.png`

## 9) Aralık kapsama kalibrasyonu

```bash
python3 scripts/calibrate_istanbul_dam_intervals.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-dir output/istanbul_dam_forecast_decision \
  --alpha 0.10 \
  --scale-mode series_month
```

Ana çıktılar:

- `output/istanbul_dam_forecast_decision/istanbul_dam_forecasts_decision_calibrated.csv`
- `output/istanbul_dam_forecast_decision/interval_calibration_factors.csv`
- `output/istanbul_dam_forecast_decision/interval_calibration_factors_monthly.csv`
- `output/istanbul_dam_forecast_decision/interval_scale_factors.png`
- `output/istanbul_dam_forecast_decision/interval_coverage_before_after.png`
- `output/istanbul_dam_forecast_decision/interval_scale_factors_monthly_heatmap.png`

## 10) Senaryo olasilik agirlikli beklenen risk

```bash
python3 scripts/build_istanbul_dam_weighted_scenario_risk.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-dir output/istanbul_dam_forecast_decision \
  --weight-series overall_mean \
  --laplace-alpha 1.0
```

Ana çıktılar:

- `output/istanbul_dam_forecast_decision/scenario_weights.csv`
- `output/istanbul_dam_forecast_decision/expected_risk_summary.csv`
- `output/istanbul_dam_forecast_decision/expected_risk_weighted.png`
- `output/istanbul_dam_forecast_decision/expected_risk_summary.json`

## 11) Hikaye odakli gorsel paket

```bash
python3 scripts/build_istanbul_dam_story_visuals.py \
  --input-dir output/istanbul_dam_forecast_decision \
  --output-dir output/istanbul_dam_forecast_decision \
  --top-n 8
```

Ana çıktılar:

- `output/istanbul_dam_forecast_decision/story_visual_pack.png`
- `output/istanbul_dam_forecast_decision/story_overall_timeline_weighted.png`
- `output/istanbul_dam_forecast_decision/story_expected_gap_heatmap.png`
- `output/istanbul_dam_forecast_decision/HIKAYE_OZETI.md`
- `output/istanbul_dam_forecast_decision/story_key_metrics.csv`
- `output/istanbul_dam_forecast_decision/story_overall_timeline_weighted.csv`
- `output/istanbul_dam_forecast_decision/story_expected_gap_heatmap.csv`

## Not

- Veri son gözlem tarihi `2024-02-01` olduğu için `2026` sonrası değerler projeksiyondur.
- `dry_stress` ve `wet_relief` senaryoları model belirsizliği (`interval_q_abs`) tabanlı strestest amaçlıdır.
- Kalibre edilmiş senaryo üretmek için önerilen sıra: `1 -> 9 -> 2 -> 10 -> 7 -> 8 -> 11 -> 3/4/5 -> 6`.
- `build_istanbul_dam_scenarios.py`, kalibre edilmiş tahmin dosyası varsa (`..._calibrated.csv`) onu otomatik kullanır.
- Adım 6 raporu, adım 7-11 çıktıları varsa onları otomatik olarak rapora ekler.
- `forecast_istanbul_dam_decision_support.py` içinde `--auto-tune-selection` modu vardır; ancak mevcut veri setinde en iyi RMSE/MAE dengesi `--no-auto-tune-selection` ile elde edilmiştir.
- `ensemble_stacked` stratejisi aktifken bazı serilerde (`Kazandere`, `Pabucdere` gibi) CV doğruluğu iyileşmiştir.
