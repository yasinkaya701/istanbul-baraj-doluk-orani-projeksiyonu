# Operating Protocol

This is the default operating mode for the Istanbul dam occupancy project.

## Non-negotiable rules

1. `new data` is the primary local source whenever coverage exists.
2. External sources are allowed and expected, but every source must be logged.
3. No ad hoc outside material stays unregistered.
4. No material modeling change is left undocumented.
5. All project outputs must be traceable back to sources, data files, and notes.

## Source handling

When a new external source is used:

1. Put the raw material under `incoming/` if it is a file.
2. Register the source in `registry/sources/external_sources.csv`.
3. Add why it matters in `notes/decisions/decision_log.md` if it changes the model or argument.
4. Reference it in any report or artifact that uses it.

## Local data handling

When a new local or transformed dataset is created:

1. Save it in a stable project path.
2. Register it in `registry/datasets/local_data_inventory.csv`.
3. State coverage, granularity, and role.
4. If it changes modeling, update `notes/status/current_model_status.md`.

## Modeling workflow

Every meaningful modeling pass should leave:

- a reproducible script in `scripts/`
- a stable output directory under `output/`
- a trace in `logs/`
- an update to `artifacts/ARTIFACT_INDEX.md` if the output matters for delivery

## Incoming actinograph workflow

When actinograph files arrive:

1. Land them in `incoming/actinograph/`
2. Do not overwrite raw files
3. Create parsed outputs under `output/actinograph/`
4. Register both raw and processed datasets
5. Update `notes/operations/actinograph_integration_plan.md`

## Naming conventions

- Reports:
  `subject_scope_status.ext`
- Derived tables:
  `domain_metric_granularity.csv`
- Logs:
  `YYYY-MM-DD_session.md`
- Scripts:
  verb-first and project-specific, for example:
  `build_...`, `update_...`, `analyze_...`, `forecast_...`

## Quality bar

Anything that goes into a presentation or academic note must satisfy:

- source traceability
- date coverage clarity
- assumption disclosure
- reproducibility
- professional file placement
