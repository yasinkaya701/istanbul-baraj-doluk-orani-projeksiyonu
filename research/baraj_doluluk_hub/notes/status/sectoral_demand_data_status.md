# Sectoral Demand Data Status

Date: 2026-03-11

Current status:

- Public official material is sufficient to defend demand segmentation as a model need.
- Public official material is not yet sufficient to build a reliable monthly sector-level demand time series.

What is officially available now:

- tariff categories from ISKI tariff pages and tariff regulation
- annual active subscriber counts from activity reports
- annual reclaimed-water volumes from activity reports
- annual NRW / physical-loss values from water-loss forms
- industrial wastewater facilities and sector-distribution tables in the 2022 activity report

What is still missing for a full sector-demand model:

- monthly billed water by tariff category
- monthly raw-water withdrawals by user group
- monthly reclaimed-water deliveries by user group
- district or treatment-source linkage by sector

What this means for the current model:

- keep aggregate monthly consumption in production
- keep sectoral demand split as `partial_active`, not `fully_active`
- use annual official context to bound and explain demand behavior
- treat industrial wastewater sector tables as a future industrial proxy, not as direct water-withdrawal truth

Specific new evidence from this round:

- active subscribers increased from `6,607,981` in `2020` to `6,891,231` in `2023`
- reclaimed-water share of total system input was about `2.32%` in `2020` and about `2.62%` in `2023`
- official 2018 and 2019 water-loss forms exist, but the currently accessible PDFs are image-like and need OCR before structured extraction

Operational conclusion:

- sectoral demand is now a defensible roadmap item with partial official grounding
- but the monthly forecasting core should still rely on:
  aggregate demand,
  climate forcing,
  storage memory,
  and explicit annual operational context proxies
