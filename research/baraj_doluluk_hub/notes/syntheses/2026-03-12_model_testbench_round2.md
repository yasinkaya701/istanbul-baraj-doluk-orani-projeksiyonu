# 2026-03-12 Model Testbench Round 2

Bu not, Istanbul toplam baraj doluluk modeli icin kurulan yeni test
paketinin kaydidir.

## Paket

- `/Users/yasinkaya/Hackhaton/output/istanbul_forward_model_benchmark_round2/one_step_metrics.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_forward_model_benchmark_round2/recursive_horizon_metrics.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_forward_model_benchmark_round2/physical_sanity_checks.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_forward_model_benchmark_round2/model_selection_scorecard.csv`
- `/Users/yasinkaya/Hackhaton/output/istanbul_forward_model_benchmark_round2/benchmark_summary.json`

## Test protokolu

Bu turda yalniz tek-adimli RMSE degil, bes ayri test uygulandi:

- tek adimli yuruyen test
- cok adimli recursive geri test (`1`, `3`, `6`, `12` ay)
- yon / dogrultu dogrulugu
- `%40` ve `%30` esik alti siniflama dogrulugu
- fiziksel isaret testi

## Denenen modeller

- `persistence`
- `history_only_ridge`
- `hybrid_ridge`
- `hybrid_elastic_net`
- `extra_trees_full`
- `random_forest_full`
- `hist_gbm_full`
- `hist_gbm_monotonic`

## Ana sonuclar

Tek adimli yuruyen test:

- `hybrid_ridge`: `RMSE = 4.27` yp
- `extra_trees_full`: `RMSE = 4.36` yp
- `hybrid_elastic_net`: `RMSE = 4.42` yp

Recursive ortalama hata:

- `hybrid_ridge`: `7.31` yp
- `hybrid_elastic_net`: `9.12` yp
- `extra_trees_full`: `10.13` yp

`12` aylik hata:

- `hybrid_ridge`: `7.65` yp
- `hybrid_elastic_net`: `11.93` yp
- `extra_trees_full`: `13.68` yp

## Fiziksel isaret testi

Beklenen yonler:

- `yagis +%10` -> doluluk artmali
- `talep +%10` -> doluluk dusmeli
- `ET0 +%10` -> doluluk dusmeli
- `transfer stresi` -> doluluk dusmeli

Sonuc:

- `extra_trees_full`: `4/4` gecti
- `random_forest_full`: `4/4` gecti
- `hybrid_ridge`: `3/4`
- `hybrid_elastic_net`: `3/4`

Kritik bulgu:

- `hybrid_ridge` ve `hybrid_elastic_net` ET0 tekil isaret testini
  gecemedi
- yani istatistiksel olarak en guclu model ile fiziksel isaret testinde
  en temiz model ayni model degil

## Karar

Bu tur sonunda:

- `hybrid_ridge` ana operasyonel model olarak tutuldu
- `extra_trees_full` challenger / denetleyici model olarak tutuldu

Gerekce:

- `hybrid_ridge` tek adimli, cok adimli ve `12` aylik testlerde genel
  olarak en iyi model
- `extra_trees_full` fiziksel isaret testlerinde daha guven verici
- bu nedenle ileri turlarda sayisal iyilestirme yaparken yalniz RMSE'ye
  bakilmamali; `extra_trees_full` ile capraz kontrol surmeli

## Acik gelistirme noktasi

- `ET0` blogu fiziksel olarak daha guvenilir hale getirilmeli
- aktinograf geldiginde ayni test paketi yeniden calistirilmali
- fiziksel kisitli veya monotonik modeller tekrar denenecek, fakat bu turdaki
  `hist_gbm_monotonic` varyanti performans olarak zayif kaldi
