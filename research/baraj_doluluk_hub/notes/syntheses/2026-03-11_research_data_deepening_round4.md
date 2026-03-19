# Research and Data Deepening Round 4

Date: 2026-03-11

Scope of this round:

- bring official annual operations context into the project data layer
- tighten the sectoral-demand evidence boundary
- identify extractable vs OCR-blocked official years

## 1. Official annual operations context is now a real dataset

Produced table:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_operational_context_annual.csv`

Produced figure:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/figures/official_operational_context_trends.png`

This table joins:

- active subscribers
- annual city water supplied
- reclaimed-water volumes
- NRW
- physical losses
- authorized consumption

Years with full annual operations context:

- `2020`
- `2021`
- `2022`
- `2023`

Years with annual loss-only context:

- `2014`
- `2016`
- `2017`
- `2024`

## 2. Official trend signals extracted in this round

From the official annual context table:

- active subscribers increased from `6,607,981` in `2020` to `6,891,231` in `2023`
- this is about `4.29%` growth over three years
- authorized consumption per active subscriber was about `353.26` L/day in `2020`
- the same proxy was about `360.01` L/day in `2023`
- reclaimed-water share of total system input was about `2.32%` in `2020`
- reclaimed-water share of total system input was about `2.62%` in `2023`
- NRW fell from `20.68%` in `2020` to `18.94%` in `2023`

Interpretation:

- public data now supports a stronger annual demand-and-operations narrative
- reclaimed water is still small relative to total system input, but it is large enough to treat as a non-zero substitution pathway
- subscriber growth and lower NRW can move in opposite directions, so demand pressure and loss control should stay separate in the model

## 3. Sectoral demand split is now better grounded but still incomplete

What became stronger:

- ISKI tariff documents define categories clearly enough to defend a future split
- activity reports provide annual system context
- the 2022 activity report includes industrial wastewater facilities and debi tables by sector

What is still missing:

- monthly billed water by tariff class
- monthly raw-water use by customer class
- monthly reclaimed-water deliveries by class

Implication:

- sectoral demand split should stay `partial_active`
- it is now backed by official structure and annual context, but not yet by a public monthly series

## 4. Missing official loss years are no longer hypothetical

New official sources confirmed:

- `2018` water-loss form exists
- `2019` water-loss form exists

Constraint:

- the currently available files behave like image-first PDFs in this environment
- they require OCR before structured extraction

Implication:

- the gap between `2017` and `2020` is now a technical extraction problem, not a source-discovery problem

## 5. Practical modeling conclusion after this round

Current monthly production story remains:

- storage memory
- rainfall
- ET0
- aggregate demand
- temperature and humidity support

New annual governance layer is now explicit:

- NRW
- physical loss
- active subscriber growth
- reclaimed-water share

This makes the project stronger in front of experts because it separates:

- physical climate stress
- human demand pressure
- operational efficiency
- substitution / reuse capacity

## 6. Source links used in this round

- ISKI water-loss reports page:
  https://iski.istanbul/kurumsal/stratejik-yonetim/su-kayiplari-yillik-raporlari/
- ISKI 2023 activity report:
  https://iskiapi.iski.gov.tr/uploads/2023_Yili_Faaliyet_Raporu_24309dd9dd.pdf
- ISKI 2022 activity report:
  https://cdn.iski.istanbul/uploads/2022_Faaliyet_Raporu_c65c8a733d.pdf
- ISKI 2021 activity report:
  https://cdn.iski.istanbul/uploads/2021_FAALIYET_RAPORU_64bf206f27.pdf
- ISKI 2020 activity report:
  https://cdn.iski.istanbul/uploads/2020_FAALIYET_RAPORU_903efe0267.pdf
- ISKI 2019 water-loss form:
  https://cdn.iski.istanbul/uploads/Su_Kayiplari_2019_b9cb04a599.pdf
- ISKI 2018 water-loss form:
  https://cdn.iski.istanbul/uploads/Sukayiplari_4a59e691c2.pdf
- Water Research 2024 urban consumption-pattern paper:
  https://doi.org/10.1016/j.watres.2024.122085
