# 2026-03-11 Useful Data Discovery Round 9

## Newly added useful data

### NOAA monthly NAO regime index

A new outside-data feature was added:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/noaa_cpc_nao_monthly_1950_present.csv`

Join tables:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/noaa_nao_vs_istanbul_driver_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/noaa_nao_vs_istanbul_climate_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/noaa_nao_vs_istanbul_djf_seasonal.csv`

Summary:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/noaa_nao_context_summary.json`

Source:

- NOAA CPC monthly NAO index ASCII feed

## Why this is useful

NAO is not a local measurement, but it is a useful regime-scale climate feature.
For Istanbul, the practical value is winter precipitation context.

This means it can help answer a different question than raw rainfall:

- not just `what happened locally this month?`
- but `what large-scale circulation regime was the region under?`

## Local empirical screening result

The new joins show the following descriptive relationships:

### Long climate window

Using the longer overlap between NOAA NAO and the Istanbul monthly climate panel:

- overlap: `1950-01` to `2021-12`
- wet-season monthly NAO vs rainfall correlation: about `-0.215`
- winter monthly NAO vs rainfall correlation: about `-0.209`
- DJF seasonal mean NAO vs DJF seasonal rainfall correlation: about `-0.293`

### Driver-panel window

Using the direct model driver panel:

- overlap: `2000-10` to `2024-02`
- wet-season monthly NAO vs rainfall correlation: about `-0.142`
- winter monthly NAO vs rainfall correlation: about `-0.215`
- wet-season monthly NAO vs weighted total fill: about `-0.143`

## Interpretation

This is not strong enough to replace local rainfall.
But it is strong enough to justify NAO as a feature candidate for:

- wet-season risk tagging
- seasonal scenario framing
- early-warning classification
- regime-conditioned model branches

In short:

- `rainfall` stays the direct forcing
- `NAO` becomes a regime context feature

That is a technically cleaner use than trying to force NAO into a direct volume predictor without screening.
