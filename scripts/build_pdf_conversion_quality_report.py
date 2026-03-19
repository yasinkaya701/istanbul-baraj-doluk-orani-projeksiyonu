#!/usr/bin/env python3
"""Build a human-readable quality report from PPTX->PDF conversion JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PDF conversion quality report")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--date-label", type=str, required=True)
    p.add_argument("--output-md", type=Path, default=None)
    return p.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e), "path": str(path)}


def mode_quality(mode: str) -> tuple[str, str]:
    if mode == "soffice":
        return "yuksek", "gorsel_fidelity"
    if mode == "keynote":
        return "yuksek", "gorsel_fidelity"
    if mode == "fallback_text":
        return "dusuk", "metin_fallback"
    return "bilinmiyor", "unknown"


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    out_md = args.output_md or (root / f"pdf_donusum_kalite_raporu_{args.date_label}_v1.md")

    report_files = [
        root / f"sunum_6_slayt_{args.date_label}_v3_pdf_conversion.json",
        root / f"sunum_10_slayt_{args.date_label}_v4_detailed_pdf_conversion.json",
        root / f"sunum_13_slayt_{args.date_label}_v6_board_pdf_conversion.json",
    ]

    rows: list[dict] = []
    for p in report_files:
        d = load_json(p)
        if d.get("missing"):
            rows.append(
                {
                    "file": p.name,
                    "mode": "missing",
                    "quality": "yok",
                    "tag": "missing",
                    "slides": 0,
                    "size": 0,
                    "attempts": "rapor dosyasi yok",
                }
            )
            continue
        if "error" in d:
            rows.append(
                {
                    "file": p.name,
                    "mode": "error",
                    "quality": "yok",
                    "tag": "error",
                    "slides": 0,
                    "size": 0,
                    "attempts": f"json error: {d['error']}",
                }
            )
            continue
        mode = str(d.get("mode", "unknown"))
        quality, tag = mode_quality(mode)
        attempts = "; ".join(d.get("attempts", [])) if isinstance(d.get("attempts", []), list) else str(d.get("attempts"))
        rows.append(
            {
                "file": p.name,
                "mode": mode,
                "quality": quality,
                "tag": tag,
                "slides": int(d.get("slide_count", 0)),
                "size": int(d.get("output_size_bytes", 0)),
                "attempts": attempts,
            }
        )

    n_total = len(rows)
    n_visual = sum(1 for r in rows if r["mode"] in {"soffice", "keynote"})
    n_fallback = sum(1 for r in rows if r["mode"] == "fallback_text")
    n_problem = sum(1 for r in rows if r["mode"] in {"missing", "error", "unknown"})

    overall = "YESIL"
    if n_problem > 0:
        overall = "KIRMIZI"
    elif n_fallback > 0:
        overall = "SARI"

    lines = [
        f"# PDF Donusum Kalite Raporu ({args.date_label})",
        "",
        f"- Genel durum: **{overall}**",
        f"- Visual fidelity (soffice/keynote): {n_visual}/{n_total}",
        f"- Text fallback: {n_fallback}/{n_total}",
        f"- Problemli/missing: {n_problem}/{n_total}",
        "",
        "## Dosya Bazli Sonuc",
        "",
        "| Donusum JSON | Mode | Kalite | Slayt | PDF Boyut (bytes) |",
        "|---|---|---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['file']} | {r['mode']} | {r['quality']} | {r['slides']} | {r['size']} |")

    lines += [
        "",
        "## Teknik Not",
        "- `fallback_text` modu, slayt tasarimini birebir render etmez; metin tabanli PDF uretir.",
        "- Gorsel fidelite gerekiyorsa `soffice/libreoffice` veya calisan `keynote` otomasyonu gerekir.",
        "",
        "## Ayrintili Girisim Gecmisi",
        "",
    ]
    for r in rows:
        lines.append(f"- {r['file']}: {r['attempts']}")

    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
