#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import re

import pandas as pd


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def fetch_html(url: str, timeout: int = 25) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def first_image_url(page_url: str, html: str) -> str:
    # Priority: official gallery static images.
    patterns = [
        r'src="([^"]*_images/[^"]+\.png)"',
        r'data-src="([^"]*_images/[^"]+\.png)"',
        r'src="([^"]+\.png)"',
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            raw = m.group(1)
            return urljoin(page_url, raw)
    return ""


def main() -> int:
    base = Path("/Users/yasinkaya/Hackhaton")
    ref_dir = base / "output" / "analysis" / "internet_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ref_dir / "internet_chart_manifest.tsv"
    candidates_path = ref_dir / "chart_page_candidates.tsv"
    rejected_path = ref_dir / "chart_page_candidates_rejected.tsv"

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, sep="\t")
    else:
        manifest = pd.DataFrame(
            columns=["ref_id", "provider", "chart_type", "title", "page_url", "image_url", "local_file"]
        )

    if not candidates_path.exists():
        raise SystemExit(f"Candidate list not found: {candidates_path}")

    cands = pd.read_csv(candidates_path, sep="\t")
    needed = {"provider", "chart_type", "title", "page_url"}
    miss = needed - set(cands.columns)
    if miss:
        raise SystemExit(f"Candidate list missing columns: {sorted(miss)}")

    existing_pages = set(manifest["page_url"].astype(str))
    new_rows = []
    bad_rows = []

    for _, r in cands.iterrows():
        provider = str(r["provider"]).strip()
        ctype = str(r["chart_type"]).strip()
        title = str(r["title"]).strip()
        page_url = str(r["page_url"]).strip()
        if not page_url or page_url in existing_pages:
            continue
        try:
            html = fetch_html(page_url)
            img = first_image_url(page_url, html)
            if not img:
                bad_rows.append({"page_url": page_url, "reason": "image_not_found"})
                continue

            host = urlparse(page_url).netloc
            stem = slug(Path(urlparse(page_url).path).stem or title)
            ref_id = f"{'mpl' if 'matplotlib' in host else 'sns' if 'seaborn' in host else 'ref'}_{stem}"
            local_file = f"{ref_id}.png"

            # Guarantee unique ref_id/local_file.
            i = 2
            while (manifest["ref_id"].astype(str) == ref_id).any() or any(x["ref_id"] == ref_id for x in new_rows):
                ref_id = f"{ref_id}_{i}"
                local_file = f"{ref_id}.png"
                i += 1

            new_rows.append(
                {
                    "ref_id": ref_id,
                    "provider": provider,
                    "chart_type": ctype,
                    "title": title,
                    "page_url": page_url,
                    "image_url": img,
                    "local_file": local_file,
                }
            )
            existing_pages.add(page_url)
        except Exception as e:
            bad_rows.append({"page_url": page_url, "reason": f"fetch_error: {type(e).__name__}"})

    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
        manifest = manifest.drop_duplicates(subset=["page_url"], keep="first")
        manifest.to_csv(manifest_path, sep="\t", index=False)
    elif not manifest_path.exists():
        manifest.to_csv(manifest_path, sep="\t", index=False)

    pd.DataFrame(bad_rows).to_csv(rejected_path, sep="\t", index=False)

    print(f"Added {len(new_rows)} new references")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Saved: {manifest_path}")
    print(f"Saved: {rejected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
