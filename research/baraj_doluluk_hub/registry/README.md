# Registry Guide

Two registries are mandatory:

- `sources/external_sources.csv`
- `datasets/local_data_inventory.csv`

Additional structured registries:

- `features/feature_candidate_matrix.csv`
- `events/intervention_event_registry.csv`

## `sources/external_sources.csv` fields

- `source_id`
  Stable id such as `SRC-016`
- `category`
  `official`, `literature`, `method`, `news`, `technical_doc`
- `title`
  Human-readable source title
- `organization_or_journal`
  Publisher, institution, or journal
- `year`
  Publication year or page year
- `url`
  Canonical link
- `access_date`
  Date checked
- `relevance`
  Why the source matters for the project
- `status`
  `active`, `pending_capture`, `deprecated`
- `note`
  Short implementation or caveat note

## `datasets/local_data_inventory.csv` fields

- `dataset_id`
  Stable id such as `DATA-015`
- `kind`
  `raw_local`, `derived`, `official_export`, `pending_incoming`
- `path`
  Absolute path to the dataset or reserved landing zone
- `coverage`
  Date span or content span
- `granularity`
  `daily`, `monthly`, `10min`, `scenario_table`, etc.
- `status`
  `active`, `pending`, `deprecated`
- `note`
  One-line purpose or caveat

## `features/feature_candidate_matrix.csv` fields

- `feature_id`
  Stable id such as `FEAT-010`
- `feature_name`
  Human-readable feature title
- `domain`
  `hydroclimate`, `demand`, `losses`, `operations`, `network`, `evaporation`
- `current_state`
  `active`, `partial`, `pending`
- `data_path_or_source`
  Main local path or upstream source
- `official_support`
  Official or primary-source basis
- `model_role`
  Why this feature matters in the model
- `priority`
  `P0`, `P1`, `P2`
- `next_action`
  Concrete next implementation step

## `events/intervention_event_registry.csv` fields

- `event_id`
  Stable event identifier
- `event_type`
  `outage`, `restriction`, `maintenance`, `policy_change`
- `start_ts`, `end_ts`
  Event timestamps
- `duration_hours`
  Standardized event duration
- `geography_level`, `district`, `neighborhood`
  Spatial scope
- `source_name`, `source_url`, `source_kind`
  Traceability fields
- `status`
  `confirmed`, `partial`, `pending`
- `effect_direction`
  Expected occupancy-demand direction
- `note`
  Caveat or interpretation note
