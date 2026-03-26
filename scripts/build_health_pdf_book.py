#!/usr/bin/env python3
"""Build a single combined PDF handbook from release PDF artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine health PDFs into a single handbook.")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--date-label", type=str, required=True)
    p.add_argument("--output-pdf", type=Path, default=None)
    return p.parse_args()


def append_pdf(writer: PdfWriter, path: Path) -> int:
    reader = PdfReader(str(path))
    n = len(reader.pages)
    for i in range(n):
        writer.add_page(reader.pages[i])
    return n


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    out_pdf = args.output_pdf or (root / f"sunum_pdf_kitabi_{args.date_label}_v1.pdf")

    inputs = [
        root / f"sunum_6_slayt_{args.date_label}_v3.pdf",
        root / f"sunum_10_slayt_{args.date_label}_v4_detailed.pdf",
        root / f"sunum_13_slayt_{args.date_label}_v6_board.pdf",
        root / f"yonetici_brif_tek_sayfa_{args.date_label}.pdf",
    ]
    existing = [p for p in inputs if p.exists()]
    if not existing:
        raise SystemExit("No source PDFs found for handbook.")

    writer = PdfWriter()
    total_pages = 0
    for p in existing:
        total_pages += append_pdf(writer, p)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as f:
        writer.write(f)

    print(f"Wrote: {out_pdf}")
    print(f"Source count: {len(existing)}")
    print(f"Total pages: {total_pages}")


if __name__ == "__main__":
    main()
