#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


MANIFEST_PATH = Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/admin/HUB_MANIFEST.json")
REPORT_DIR = Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/logs/validation")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_path_rows(rows: list[dict[str, str]], key: str, status_key: str | None = None) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for row in rows:
        path = Path(row[key])
        status = row.get(status_key, "") if status_key else ""
        if status in {"deprecated"}:
            continue
        if status == "pending":
            continue
        if not path.exists():
            issues.append(
                {
                    "kind": "missing_path",
                    "path": str(path),
                    "id": row.get("dataset_id") or row.get("artifact_id") or row.get("feature_id") or "",
                }
            )
    return issues


def build_markdown_report(summary: dict[str, object], issues: list[dict[str, str]]) -> str:
    lines = [
        "# Hub Validation Report",
        "",
        f"- Timestamp: `{summary['timestamp']}`",
        f"- Required directories checked: `{summary['required_directories_checked']}`",
        f"- Required files checked: `{summary['required_files_checked']}`",
        f"- External sources: `{summary['external_sources_count']}`",
        f"- Dataset inventory rows: `{summary['dataset_inventory_count']}`",
        f"- Feature matrix rows: `{summary['feature_matrix_count']}`",
        f"- Artifact catalog rows: `{summary['artifact_catalog_count']}`",
        f"- Issue count: `{len(issues)}`",
        "",
    ]
    if not issues:
        lines.append("No validation issues found.")
    else:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue['kind']}`: `{issue['path']}` ({issue.get('id', '')})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    issues: list[dict[str, str]] = []

    required_dirs = [Path(p) for p in manifest["required_directories"]]
    required_files = [Path(p) for p in manifest["required_files"]]

    for path in required_dirs:
        if not path.exists() or not path.is_dir():
            issues.append({"kind": "missing_directory", "path": str(path), "id": ""})
    for path in required_files:
        if not path.exists() or not path.is_file():
            issues.append({"kind": "missing_file", "path": str(path), "id": ""})

    external_sources = load_csv_rows(Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/registry/sources/external_sources.csv"))
    datasets = load_csv_rows(Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/registry/datasets/local_data_inventory.csv"))
    features = load_csv_rows(Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/registry/features/feature_candidate_matrix.csv"))
    artifacts = load_csv_rows(Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/artifacts/artifact_catalog.csv"))

    issues.extend(validate_path_rows(datasets, "path", "status"))
    issues.extend(validate_path_rows(artifacts, "path", "status"))

    # Feature paths may be sources, registry tables, or conceptual placeholders.
    for row in features:
        path_str = row["data_path_or_source"]
        if path_str.startswith("/Users/"):
            path = Path(path_str)
            if row["current_state"] != "pending" and not path.exists():
                issues.append({"kind": "missing_feature_path", "path": str(path), "id": row["feature_id"]})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "required_directories_checked": len(required_dirs),
        "required_files_checked": len(required_files),
        "external_sources_count": len(external_sources),
        "dataset_inventory_count": len(datasets),
        "feature_matrix_count": len(features),
        "artifact_catalog_count": len(artifacts),
        "issue_count": len(issues),
    }

    json_path = REPORT_DIR / f"hub_validation_{timestamp}.json"
    md_path = REPORT_DIR / f"hub_validation_{timestamp}.md"
    latest_json = REPORT_DIR / "hub_validation_latest.json"
    latest_md = REPORT_DIR / "hub_validation_latest.md"

    json_path.write_text(
        json.dumps({"summary": summary, "issues": issues}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(build_markdown_report(summary, issues), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(latest_json)
    print(latest_md)


if __name__ == "__main__":
    main()
