#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path


PAGE_URL = "https://iski.istanbul/abone-hizmetleri/ariza-kesinti/"
PAYLOAD_URL = "https://iski.istanbul/_nuxt/static/1773145496/abone-hizmetleri/ariza-kesinti/payload.js"
OUT_JSON = Path("/Users/yasinkaya/Hackhaton/output/official_api_probes/iski_outage_probe.json")


def fetch_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    page_html = fetch_text(PAGE_URL)
    payload_js = fetch_text(PAYLOAD_URL)

    endpoint_match = re.search(
        r'endpoint:"(https:\\u002F\\u002Fiskiapi\.iski\.gov\.tr\\u002Fapi\\u002Fiski\\u002FbolgeselAriza\\u002Flistesi)"',
        payload_js,
    )
    endpoint = None
    if endpoint_match:
        endpoint = endpoint_match.group(1).replace("\\u002F", "/")

    probe = {
        "page_url": PAGE_URL,
        "payload_url": PAYLOAD_URL,
        "page_has_payload_reference": "payload.js" in page_html,
        "endpoint_found_in_payload": bool(endpoint),
        "endpoint_url": endpoint,
        "endpoint_status": None,
        "endpoint_response_excerpt": None,
    }

    if endpoint:
        req = urllib.request.Request(
            endpoint,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": PAGE_URL,
                "Origin": "https://iski.istanbul",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                probe["endpoint_status"] = getattr(resp, "status", 200)
                probe["endpoint_response_excerpt"] = body[:500]
        except Exception as exc:  # pragma: no cover - network dependent
            probe["endpoint_status"] = "error"
            probe["endpoint_response_excerpt"] = str(exc)

    OUT_JSON.write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_JSON)


if __name__ == "__main__":
    main()
