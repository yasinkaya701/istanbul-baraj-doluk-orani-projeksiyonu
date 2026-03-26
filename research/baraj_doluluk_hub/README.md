# Baraj Doluluk Research Hub

Purpose:

Single professional storage layer for all external research, local data inventory,
decision notes, incoming materials, and generated artifacts used by the
Istanbul dam occupancy project.

Core project question:

`How reliably can we forecast Istanbul total dam occupancy with new data, ET0,
demand signals, and intervention scenarios, and turn that into a decision-support product?`

Directory structure:

- `admin/`
  Operating protocol and governance documents.
- `registry/`
  Structured registries for sources, datasets, features, and event logs.
- `notes/`
  Decision logs, status notes, syntheses, and operational plans.
- `incoming/`
  Raw outside material to be added later: PDFs, screenshots, source exports,
  incoming actinograph files, official notices, sector notes.
- `artifacts/`
  Index files that point to generated outputs used in presentations and reports.
- `logs/`
  Worklog and dated session records.
- `templates/`
  Intake templates for new sources, datasets, and event collections.

Operating rule:

Any outside source used in the project must be registered in
`registry/sources/external_sources.csv`.

Any new local dataset or transformed dataset that affects modeling must be added
to `registry/datasets/local_data_inventory.csv`.

Any material methodological change must be written into
`notes/decisions/decision_log.md`.

Current primary outputs:

- `/Users/yasinkaya/Hackhaton/output/istanbul_dam_deep_research`
- `/Users/yasinkaya/Hackhaton/output/doc/istanbul_baraj_durum_ozeti_akademik_derinlesmis.docx`

Current primary raw data root:

- `/Users/yasinkaya/Hackhaton/new data`
