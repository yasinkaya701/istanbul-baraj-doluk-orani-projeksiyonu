# 2026-03-11 Useful Data Discovery Round 8

## Newly added useful data

### Kandilli-area reanalysis proxy for ET0 forcing

A new historical proxy layer was fetched for Kandilli coordinates and stored locally:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/kandilli_openmeteo_daily_1940_present.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/kandilli_openmeteo_monthly_1940_present.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/kandilli_openmeteo_vs_local_et0_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/kandilli_openmeteo_reanalysis_summary.json`

Fetch script:

- `/Users/yasinkaya/Hackhaton/scripts/fetch_kandilli_openmeteo_reanalysis.py`

## Variables now available

Daily series now include:

- mean / max / min temperature
- precipitation
- shortwave radiation sum
- FAO ET0 proxy from the archive API
- max wind speed at 10 m
- sunshine duration

Coverage:

- daily: `1940-01-01` to `2026-03-10`
- monthly aggregate rows: `1035`

## Why this is useful

This is one of the most directly usable outside-data additions because the current project still has radiation uncertainty in ET0.

The new layer gives a defensible external proxy for:

- radiation sanity-checking
- ET0 sanity-checking
- wind backfill support
- presentation-ready historical context when actinograph has not yet been ingested

## Comparison against the local FAO-56 ET0 history

Monthly overlap with the existing local ET0 history now has a direct comparison table.

Current overlap summary:

- overlap window: `1940-01` to `2026-02`
- overlap rows: `1034`
- monthly ET0 correlation: about `0.988`
- monthly ET0 MAE: about `10.05 mm/ay`
- monthly ET0 bias: about `+9.20 mm/ay`
- monthly radiation correlation: about `0.986`

Interpretation:

- the proxy is not a replacement for actinograph or local station physics
- but it is strong enough to act as a fallback / benchmark layer
- especially for checking whether our current radiation proxy is directionally right through time

## Modeling implication

This means the ET0 block is no longer limited to only one internally reconstructed series.
The project now has:

- local FAO-56 ET0 history
- official source context for basin-aware interpretation
- official current İSKİ API state
- external reanalysis proxy for radiation and ET0 benchmarking

That combination is materially stronger than a single-series ET0 story.
