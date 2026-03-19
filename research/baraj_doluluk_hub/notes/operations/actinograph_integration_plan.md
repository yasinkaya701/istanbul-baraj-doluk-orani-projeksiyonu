# Actinograph Integration Plan

Goal:

Replace radiation proxy logic with observed radiation so ET0 and summer water
stress estimates become more defensible.

Current state:

- ET0 exists and is usable.
- Radiation is not yet fully observed across the long series.
- Present ET0 package documents radiation as a proxy/fill component.

Planned ingestion path:

1. Land raw actinograph files under:
   `/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/incoming/actinograph`
2. Parse timestamp, instrument value, unit, station id, QC flag.
3. Convert to a standard radiation unit:
   `MJ/m2/day` for daily ET0 or `MJ/m2/hour` for sub-daily validation.
4. Run QC:
   negative values, nighttime signal, spikes, flatline detection, missing spans.
5. Create canonical dataset:
   `output/actinograph/actinograph_radiation_daily.csv`
6. Recompute ET0 using observed `Rs`.
7. Compare:
   proxy-radiation ET0 vs observed-radiation ET0
8. Re-run occupancy model:
   old ET0 block vs new ET0 block
9. Register every new file in:
   `registry/datasets/local_data_inventory.csv`

Expected gain:

- Better summer stress estimation
- Better physical credibility in front of experts
- Better answer to how much evaporation demand contributes to occupancy loss
