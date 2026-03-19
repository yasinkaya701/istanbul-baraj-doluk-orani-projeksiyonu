# 2026-03-11 Useful Data Discovery Round 7

## Newly extracted useful data

### Official source-level basin context

A new official table was built from the İSKİ water-sources page:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/tables/official_iski_source_context.csv`

It contains, at source level:

- annual safe yield (`milyon m3/yıl`)
- maximum storage (`milyon m3`)
- commissioning year
- basin area (`km2`) where available
- normal lake area where explicitly reported on the source page

Summary file:

- `/Users/yasinkaya/Hackhaton/output/newdata_feature_store/official_iski_source_context_summary.json`

## Why this matters

This is one of the most useful new context layers because it directly supports model structure, not just presentation.

### 1. Rainfall-to-inflow proxy design

Basin area is now available for most major sources. That means the project can move beyond raw citywide rainfall and create source-aware rainfall forcing blocks.

Examples:

- `yield_per_basin_mm = annual_yield / basin_area`
- `storage_per_basin_mm = max_storage / basin_area`

These ratios help compare which reservoirs are hydrologically more responsive and which ones act more like large storage buffers.

### 2. Evaporation prioritization

Open-water evaporation is still limited by missing reservoir surface-area time series, but the source-context table now helps prioritize where that effort matters most.

A practical rule is now possible:

- first estimate open-water evaporation explicitly for high-storage sources
- prioritize sources where storage is large and basin response is structurally weak

### 3. Source-specific scenario framing

The project can now explain why equal treatment of all reservoirs is weak.
Different reservoirs differ not only by current occupancy, but also by:

- storage size
- annual yield
- catchment scale
- commissioning era
- regulator support versus pure reservoir behavior

## Immediate next use

The clean next engineering step is to join this source-context table with:

- capacity-weighted occupancy
- source-level API snapshot data
- source-level ET0 or evaporation assumptions

That would create a first source-aware stress map instead of a single citywide aggregate view.
