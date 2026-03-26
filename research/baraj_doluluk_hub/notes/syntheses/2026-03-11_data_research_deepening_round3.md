# Data and Research Deepening Round 3

Date: 2026-03-11

Scope of this round:

- build a durable monthly feature store from `new data`
- move official water-loss data from concept stage to usable proxy stage
- test whether deeper climate variables improve the monthly occupancy model

## 1. New feature store is now in place

Produced tables:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/istanbul_newdata_monthly_climate_panel.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/istanbul_dam_driver_panel.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_water_loss_annual.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/annual_climate_water_balance_1912_2021.csv`

Coverage summary:

- climate panel:
  `1911-07` to `2026-03`
- dam driver panel:
  `2000-10` to `2024-02`
- full dam-core monthly window:
  `2011-01` to `2024-02`
- deep-climate monthly window with target + demand + rain + ET0 + VPD + temp proxy:
  `2011-01` to `2024-02`

What is now explicitly available:

- monthly rain
- monthly ET0
- monthly VPD / vapor-pressure block
- monthly temperature and humidity proxies
- monthly pressure where observed or automatic-station based
- annual official NRW / physical-loss proxy
- annual rain-minus-ET0 climate water balance

## 2. Official water-loss feature moved forward materially

Primary change:

- `physical water losses / NRW` is no longer only a backlog idea.
- We now have an official annual proxy table based on ISKI standard water-balance forms.

Years currently captured:

- `2014`
- `2016`
- `2017`
- `2020`
- `2021`
- `2022`
- `2023`
- `2024`

Interpretation:

- This is strong enough for annual scenario framing, governance framing, and buyer-facing credibility.
- It is not yet strong enough for direct monthly forecasting without either:
  more annual forms,
  monthly KPI proxies,
  or operational loss sub-series.

## 3. Deepened climate features were tested, not just archived

Produced model-comparison files:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/deepened_feature_model_metrics.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/deepened_feature_summary.json`

Observed evaluation window:

- `2011-03` to `2021-12`
- walk-forward predictions: `70`

Results:

- `baseline_full`:
  `RMSE = 4.62` percentage points
- `plus_temp_humidity`:
  `RMSE = 4.45` percentage points
- `deep_all`:
  `RMSE = 4.45` percentage points
- `plus_vpd_balance`:
  `RMSE = 4.65` percentage points

Implication:

- temperature and humidity add measurable information beyond the current
  `rain + ET0 + demand + memory` structure
- VPD and rain-minus-ET0 by themselves do not beat the baseline here, which suggests
  the current `rain + ET0` block already absorbs much of that dryness signal

This is useful because it tells us where to spend effort:

- keep temperature and humidity as model-ready climate blocks
- keep VPD and water balance for interpretation, stress testing, and alerts
- do not oversell VPD-only improvement

## 4. New visuals now support the data story

Produced figures:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/figures/feature_coverage_timeline.png`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/figures/annual_water_balance_1912_2021.png`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/figures/official_water_loss_trend.png`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/figures/deepened_feature_model_rmse.png`

What they give us:

- a direct coverage map for what is observed, proxied, and partial
- a long-run rain-minus-ET0 dryness view
- an official loss trend that experts can audit
- a clean before/after model comparison for deeper climate features

## 5. New external research added in this round

Official / formal:

- ISKI water-loss reports page:
  https://iski.istanbul/kurumsal/stratejik-yonetim/su-kayiplari-yillik-raporlari/
- ISKI 2024 standard water-balance form:
  https://cdn.iski.istanbul/uploads/Su_Kayiplari_2024_cdd097df1e.pdf
- ISKI 2023 standard water-balance form:
  https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2023_4c07821536.pdf
- ISKI 2022 standard water-balance form:
  https://cdn.iski.istanbul/uploads/Su_Denge_Tablosu_2022_46b2a9477c.pdf
- ISKI 2021 standard water-balance form:
  https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2021_9e4b97ee29.pdf
- ISKI 2020 standard water-balance form:
  https://cdn.iski.istanbul/uploads/Su_denge_tablosu_2020_9a984f0ba7.pdf

Academic:

- Water Research 2024, urban consumption-pattern clustering:
  https://doi.org/10.1016/j.watres.2024.122085

Interpretation:

- The official side now supports annual loss modeling and governance framing.
- The academic side supports future sectoral or cluster-based demand segmentation once
  better usage breakdown data arrives.

## 6. Priority backlog after this round

Highest-value next steps:

- extend the official loss series by finding additional archived annual forms
- obtain sector or tariff-category water-use time series
- ingest actinograph and recompute ET0 with observed radiation
- build a usable intervention-event calendar from outage or archival sources

Current evidence-based position:

- We can already defend a serious monthly occupancy model with climate and demand structure.
- We can now also defend an annual loss block with official utility documents.
- The next quality jump will come from:
  observed radiation,
  sectoral demand split,
  and real intervention-event data.
