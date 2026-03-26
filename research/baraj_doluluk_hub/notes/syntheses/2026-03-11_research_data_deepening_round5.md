# Research and Data Deepening Round 5

Date: 2026-03-11

Scope of this round:

- translate official annual context into intervention leverage metrics
- test whether annual official context helps the monthly model in the short public window

## 1. Official policy leverage table is now explicit

Produced table:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_policy_leverage_annual.csv`

Produced figure:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/figures/official_policy_leverage_latest_year.png`

Method:

- annual official volumes were converted into occupancy-equivalent percentage points
- storage basis:
  total active storage capacity of Istanbul reservoirs

Important caution:

- these are annual upper-bound equivalents
- they are useful for ranking interventions
- they are not month-by-month causal estimates

## 2. 2023 official leverage values

Using the 2023 official context:

- `NRW -1` percentage point:
  about `+1.29` percentage points equivalent total occupancy
- `authorized demand -1%`:
  about `+1.04` percentage points equivalent total occupancy
- `reclaimed water +10%`:
  about `+0.34` percentage points equivalent total occupancy
- `100,000` active-subscriber growth at current intensity:
  about `-1.51` percentage points equivalent total occupancy
- moving reclaimed-water share toward `5%`:
  about `+3.06` percentage points equivalent total occupancy

Interpretation:

- NRW control and demand reduction are first-tier levers
- reclaimed water is not negligible, but on current scale it is a smaller near-term lever
- subscriber growth is large enough to erase a meaningful part of operational gains if it is ignored

## 3. Annual official context was also tested in the monthly model

Produced metrics table:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/annual_context_monthly_model_metrics.csv`

Produced figure:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/figures/annual_context_monthly_model_rmse.png`

Window:

- `2020-03` to `2023-12`
- predictions:
  `28`

Results:

- `baseline_temp_humidity`:
  `RMSE = 5.41`
- `plus_reuse_intensity`:
  `RMSE = 5.29`
- `plus_all_annual_context`:
  `RMSE = 5.44`
- `plus_nrw`:
  `RMSE = 6.24`

Interpretation:

- official annual `reuse + demand-intensity` context gives a small but real improvement on this short window
- annual `NRW` by itself does not help the monthly forecast on this sample
- this is consistent with scale:
  NRW is operationally important but annual and coarse,
  while reuse/intensity variables better describe annual system pressure

## 4. Practical conclusion after this round

Current project structure is now stronger in two distinct ways:

- monthly forecasting core:
  memory + climate + aggregate demand + selected deep climate features
- annual decision layer:
  NRW + reclaimed water + subscriber growth + intensity proxies

This is a better buyer-facing story because it separates:

- what helps forecast next months
- what helps rank annual interventions

## 5. Source links used in this round

- ISKI 2023 activity report:
  https://iskiapi.iski.gov.tr/uploads/2023_Yili_Faaliyet_Raporu_24309dd9dd.pdf
- ISKI 2022 activity report:
  https://cdn.iski.istanbul/uploads/2022_Faaliyet_Raporu_c65c8a733d.pdf
- ISKI 2021 activity report:
  https://cdn.iski.istanbul/uploads/2021_FAALIYET_RAPORU_64bf206f27.pdf
- ISKI 2020 activity report:
  https://cdn.iski.istanbul/uploads/2020_FAALIYET_RAPORU_903efe0267.pdf
- ISKI water-loss forms page:
  https://iski.istanbul/kurumsal/stratejik-yonetim/su-kayiplari-yillik-raporlari/
- ISKI water sources page:
  https://iski.istanbul/kurumsal/hakkimizda/su-kaynaklari
