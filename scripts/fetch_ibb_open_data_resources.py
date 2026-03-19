#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
import urllib.parse
import urllib.request


DEFAULT_RESOURCES = [
    # Istanbul Barajları Günlük Doluluk Oranları (XLSX)
    "af0b3902-cfd9-4096-85f7-e2c3017e4f21",
    # Istanbul General Dam Occupancy Rates (CSV)
    "b68cbdb0-9bf5-474c-91c4-9256c07c4bdf",
    # İstanbul Barajlarında Yağış ve Günlük Tüketim Verileri (XLSX)
    "762b802e-c5f9-4175-a5c1-78b892d9764b",
    # İstanbul'a Verilen Temiz Su Miktarları (XLSX)
    "27bdb043-0051-49df-bd7c-b68f60f31247",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download IBB Open Data CKAN resources by resource id.")
    p.add_argument(
        "--resource-id",
        action="append",
        dest="resource_ids",
        default=[],
        help="CKAN resource id (repeatable).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/external/raw/ibb"),
        help="Output directory for downloaded files + metadata.",
    )
    return p.parse_args()


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        out_path.write_bytes(resp.read())


def resolve_resource(resource_id: str) -> dict:
    api = "https://data.ibb.gov.tr/api/3/action/resource_show"
    url = api + "?" + urllib.parse.urlencode({"id": resource_id})
    payload = fetch_json(url)
    if not payload.get("success"):
        raise RuntimeError(f"resource_show failed for {resource_id}")
    return payload["result"]


def sanitize(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def main() -> None:
    args = parse_args()
    resource_ids: Iterable[str] = args.resource_ids or DEFAULT_RESOURCES
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for rid in resource_ids:
        res = resolve_resource(rid)
        name = res.get("name") or res.get("id")
        fmt = (res.get("format") or "").lower()
        url = res.get("url")
        if not url:
            raise RuntimeError(f"No download URL for {rid}")

        filename = sanitize(f"{name}_{rid}.{fmt or 'data'}")
        out_path = out_dir / filename
        download(url, out_path)

        meta = {
            "resource_id": rid,
            "name": res.get("name"),
            "title": res.get("title"),
            "format": res.get("format"),
            "url": url,
            "download_path": str(out_path),
            "last_modified": res.get("last_modified"),
        }
        meta_rows.append(meta)
        print(f"Downloaded {rid} -> {out_path.name}")

    meta_path = out_dir / "ibb_resource_metadata.json"
    meta_path.write_text(json.dumps(meta_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(meta_path)


if __name__ == "__main__":
    main()
