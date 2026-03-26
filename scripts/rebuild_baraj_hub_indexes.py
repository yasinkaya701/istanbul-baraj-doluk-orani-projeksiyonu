#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


HUB_ROOT = Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub")
ARTIFACT_CSV = HUB_ROOT / "artifacts" / "artifact_catalog.csv"
ARTIFACT_MD = HUB_ROOT / "artifacts" / "ARTIFACT_INDEX.md"
HUB_STATUS_JSON = HUB_ROOT / "admin" / "HUB_STATUS.json"

EXTERNAL_SOURCES = HUB_ROOT / "registry" / "sources" / "external_sources.csv"
LOCAL_DATA = HUB_ROOT / "registry" / "datasets" / "local_data_inventory.csv"
FEATURES = HUB_ROOT / "registry" / "features" / "feature_candidate_matrix.csv"
EVENTS = HUB_ROOT / "registry" / "events" / "intervention_event_registry.csv"


GROUP_TITLES = {
    "report": "Primary Deliverables",
    "package": "Primary Deliverables",
    "note": "Primary Deliverables",
    "figure": "Key Figures",
    "table": "Model Tables",
    "probe": "Operational Research Artifacts",
    "reference_table": "Operational Research Artifacts",
    "registry": "Operational Research Artifacts",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_artifact_index(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        title = GROUP_TITLES.get(row["artifact_type"], "Other Artifacts")
        grouped.setdefault(title, []).append(row)

    order = [
        "Primary Deliverables",
        "Key Figures",
        "Model Tables",
        "Operational Research Artifacts",
        "Other Artifacts",
    ]

    lines = ["# Artifact Index", ""]
    for section in order:
        items = grouped.get(section, [])
        if not items:
            continue
        lines.append(f"{section}:")
        lines.append("")
        for item in items:
            lines.append(f"- {item['description']}:")
            lines.append(f"  `{item['path']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_status(rows: list[dict[str, str]]) -> dict[str, object]:
    paths_exist = sum(Path(r["path"]).exists() for r in rows if r["status"] != "deprecated")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_count": len(rows),
        "artifact_paths_existing": paths_exist,
        "external_sources_count": len(read_csv(EXTERNAL_SOURCES)),
        "dataset_inventory_count": len(read_csv(LOCAL_DATA)),
        "feature_matrix_count": len(read_csv(FEATURES)),
        "event_registry_count": len(read_csv(EVENTS)),
    }


def main() -> None:
    rows = read_csv(ARTIFACT_CSV)
    ARTIFACT_MD.write_text(build_artifact_index(rows), encoding="utf-8")
    HUB_STATUS_JSON.write_text(
        json.dumps(build_status(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(ARTIFACT_MD)
    print(HUB_STATUS_JSON)


if __name__ == "__main__":
    main()
