# 2026-03-11 Useful Data Discovery Round 6

## What was added

### 1. Official monthly city-supply series

A longer official monthly city-supply series was recovered from İSKİ activity reports and rebuilt into a machine-readable table:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_city_supply_monthly_2010_2023.csv`

Coverage now spans `2010-01` to `2023-12` (`168` monthly rows).

Main source reports used in this extension:

- 2014 activity report: 2010-2014 monthly daily averages
- 2015 activity report: 2011-2015 monthly daily averages
- 2017 activity report: 2013-2017 monthly daily averages
- later activity reports already used for 2018-2023

This matters because it gives an official monthly supply series that overlaps the model window much more deeply than the short 2018-2023 reconstruction.

### 2. Official monthly supply vs model-consumption validation table

The comparison table was regenerated on the longer window:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_supply_vs_model_consumption_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/official_monthly_supply_context_summary.json`

Current summary:

- official city-supply window: `2010-01-01` to `2023-12-01`
- overlap with model consumption is almost one-to-one at aggregate level
- `model_vs_supply corr ≈ 0.9991`
- `model_vs_supply mean ratio ≈ 99.97%`
- recorded-water comparison remains weaker, supporting the interpretation that current project consumption proxy behaves more like supplied water than billed water

### 3. Public İSKİ baraj frontend API package

A new reproducible snapshot package was built from the official public dashboard frontend API:

- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/README.md`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/api_manifest.json`

The fetch script is:

- `/Users/yasinkaya/Hackhaton/scripts/fetch_iski_baraj_public_api_snapshot.py`

Useful tables now locally stored:

- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/genel_oran.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/gunluk_ozet.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/son_14_gun_toplam_doluluk.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/son_1_yil_ay_sonu_doluluk.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/yillik_yagis.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/son_14_gun_verilen_su.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/son_10_yil_toplam_verilen_su.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/melen_yesilcay.csv`
- `/Users/yasinkaya/Hackhaton/output/iski_baraj_api_snapshot/tables/baraj_bazli_son_10_yil_doluluk.csv`

Snapshot metadata at fetch time:

- snapshot time: `2026-03-11T22:32:14`
- dam count: `10`
- global endpoints harvested: `14`
- per-dam endpoint families harvested: `4`

## Why this is useful for the project

### Presentation layer

The API package is strong for the presentation because it is visibly official, current, and operational.
It supports statements like:

- current Istanbul total occupancy
- current per-dam stress distribution
- how this date compares with the same date in the previous 10 years
- recent 14-day recovery or decline
- annual rainfall context
- treated-water and Melen/Yeşilçay support flows

### Modeling layer

The longer monthly city-supply series is stronger for training and validation than the dashboard API because it spans a wider historical window.
The API package is stronger for nowcasting and operational scenario framing.

So the clean split is now:

- `2010-2023 official monthly city-supply`: monthly validation / calibration layer
- `2026 official API snapshot`: operational dashboard / short-horizon / demo layer

## Immediate next uses

1. Join `yillik_yagis.csv` with existing annual rainfall and occupancy-equivalent sensitivity tables.
2. Use `baraj_bazli_son_10_yil_doluluk.csv` in the presentation as the strongest official same-day comparison view.
3. Refresh the API snapshot close to presentation time for a live, official current-state slide.
