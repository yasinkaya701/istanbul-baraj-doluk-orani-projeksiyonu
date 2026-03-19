# 2026-03-11 Useful Data Discovery Round 10

## Consolidated model-ready package

All currently useful data blocks were pulled into a dedicated model bundle:

- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/README.md`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/model_useful_data_summary.json`

Main tables:

- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_core_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_extended_monthly.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_model_feature_block_coverage.csv`
- `/Users/yasinkaya/Hackhaton/output/model_useful_data_bundle/tables/istanbul_source_current_context.csv`

Builder script:

- `/Users/yasinkaya/Hackhaton/scripts/build_istanbul_model_useful_data_bundle.py`

## What is inside

### Core monthly matrix

The core matrix is the immediate training table.
It keeps the strongest existing monthly blocks together:

- target: `weighted_total_fill`
- lags: target lag 1-2
- rainfall
- ET0
- monthly demand
- temperature / humidity / pressure proxies
- VPD and water-balance proxy
- seasonal sine/cosine
- observation quality score

Coverage:

- rows: `281`
- window: `2000-10` -> `2024-02`
- full-core usable block: roughly `2011-02` -> `2024-02`

### Extended monthly matrix

The extended matrix adds the newly found useful blocks:

- official city-supply and recorded-water comparisons
- annual operations context
- Kandilli-area reanalysis proxy
- NOAA NAO regime context

This is now the correct entry point for richer ablation and feature-screening tests.

### Source-aware current context

A separate table now joins:

- latest official İSKİ source snapshot
- source storage and yield context
- basin-aware metrics

This is useful for:

- source-aware explanation
- stress ranking
- buyer-facing scenario discussion

## Why this matters

Before this step, useful data existed but was spread across many files.
Now the project has a clean split:

- `core monthly`: default training
- `extended monthly`: advanced experiments
- `source current context`: source-aware nowcast and scenario work

That makes the next modeling step much faster and less error-prone.
