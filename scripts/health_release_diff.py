#!/usr/bin/env python3
"""Compare current health release with previous release and write a delta report."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diff two health release snapshots")
    p.add_argument("--root-dir", type=Path, default=Path("output/health_impact"))
    p.add_argument("--date-label", type=str, required=True)
    p.add_argument("--prev-date-label", type=str, default=None)
    p.add_argument("--version-tag", type=str, default="v6")
    return p.parse_args()


def parse_manifest(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^- (.+) \((\d+) bytes\)$", line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def parse_readout_metrics(path: Path) -> dict[str, float]:
    keys = {
        "strong_future_rr": r"Strong future RR:\s*([0-9.]+)",
        "strong_delta_rr": r"Strong delta RR:\s*([+-]?[0-9.]+)",
        "quant_future_rr": r"Quant future RR:\s*([0-9.]+)",
        "quant_delta_rr": r"Quant delta RR:\s*([+-]?[0-9.]+)",
        "strong_threshold": r"Strong threshold exceed:\s*([0-9.]+)%",
        "quant_threshold": r"Quant threshold exceed:\s*([0-9.]+)%",
    }
    out: dict[str, float] = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for k, pat in keys.items():
        m = re.search(pat, text)
        if m:
            out[k] = float(m.group(1))
    return out


def find_prev_date(root: Path, date_label: str, version_tag: str) -> str | None:
    patterns = [
        rf"release_manifest_(\d{{4}}-\d{{2}}-\d{{2}})_{re.escape(version_tag)}\.txt$",
        r"sunum_12_slayt_(\d{4}-\d{2}-\d{2})_v\d+_board\.pptx$",
        r"health_slide_pack_(\d{4}-\d{2}-\d{2})_v\d+\.zip$",
    ]
    dates: list[str] = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        name = p.name
        for pat in patterns:
            m = re.match(pat, name)
            if m:
                dates.append(m.group(1))
                break
    dates = sorted(set(dates))
    prev = [d for d in dates if d < date_label]
    return prev[-1] if prev else None


def pick_latest_for_date(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def gather_release_sizes(root: Path, date_label: str) -> dict[str, int]:
    candidates = {
        "exec_pptx": f"sunum_6_slayt_{date_label}_*.pptx",
        "detailed_pptx": f"sunum_10_slayt_{date_label}_*.pptx",
        "board_pptx": f"sunum_12_slayt_{date_label}_*.pptx",
        "brief_pdf": f"yonetici_brif_tek_sayfa_{date_label}.pdf",
        "bundle_zip": f"health_slide_pack_{date_label}_*.zip",
    }
    out: dict[str, int] = {}
    for key, pat in candidates.items():
        p = pick_latest_for_date(root, pat)
        if p is not None and p.exists():
            out[key] = p.stat().st_size
    return out


def fmt_delta(curr: float | None, prev: float | None) -> str:
    if curr is None or prev is None:
        return "n/a"
    return f"{curr - prev:+.4f}"


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()

    curr_date = args.date_label
    prev_date = args.prev_date_label or find_prev_date(root, curr_date, args.version_tag)

    out = root / f"release_diff_{curr_date}_{args.version_tag}.md"
    if prev_date is None:
        out.write_text(
            "\n".join([f"# Release Diff ({curr_date})", "", "- Onceki release bulunamadi; karsilastirma yapilamadi.", ""])
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote: {out}")
        return

    curr_manifest = root / f"release_manifest_{curr_date}_{args.version_tag}.txt"
    prev_manifest = root / f"release_manifest_{prev_date}_{args.version_tag}.txt"
    curr_readout = root / f"release_readout_{curr_date}_{args.version_tag}.md"
    prev_readout = root / f"release_readout_{prev_date}_{args.version_tag}.md"

    c_files = parse_manifest(curr_manifest)
    p_files = parse_manifest(prev_manifest)
    c_met = parse_readout_metrics(curr_readout)
    p_met = parse_readout_metrics(prev_readout)
    c_sizes = gather_release_sizes(root, curr_date)
    p_sizes = gather_release_sizes(root, prev_date)

    common = sorted(set(c_files).intersection(p_files))
    added = sorted(set(c_files) - set(p_files))
    removed = sorted(set(p_files) - set(c_files))

    prev_manifest_exists = prev_manifest.exists()

    lines = [
        f"# Release Diff ({curr_date} vs {prev_date})",
        "",
        f"- Current manifest: `{curr_manifest.name}`",
        f"- Previous manifest: `{prev_manifest.name}`" + ("" if prev_manifest_exists else " (bulunamadi)"),
        "",
        "## Kilit Metrik Farklari",
        "",
        f"- Strong future RR: {c_met.get('strong_future_rr', float('nan')):.4f} (delta vs prev {fmt_delta(c_met.get('strong_future_rr'), p_met.get('strong_future_rr'))})"
        if "strong_future_rr" in c_met
        else "- Strong future RR: n/a",
        f"- Strong delta RR: {c_met.get('strong_delta_rr', float('nan')):+.4f} (delta vs prev {fmt_delta(c_met.get('strong_delta_rr'), p_met.get('strong_delta_rr'))})"
        if "strong_delta_rr" in c_met
        else "- Strong delta RR: n/a",
        f"- Quant future RR: {c_met.get('quant_future_rr', float('nan')):.4f} (delta vs prev {fmt_delta(c_met.get('quant_future_rr'), p_met.get('quant_future_rr'))})"
        if "quant_future_rr" in c_met
        else "- Quant future RR: n/a",
        f"- Quant delta RR: {c_met.get('quant_delta_rr', float('nan')):+.4f} (delta vs prev {fmt_delta(c_met.get('quant_delta_rr'), p_met.get('quant_delta_rr'))})"
        if "quant_delta_rr" in c_met
        else "- Quant delta RR: n/a",
        f"- Strong threshold exceed (%): {c_met.get('strong_threshold', float('nan')):.1f} (delta {fmt_delta(c_met.get('strong_threshold'), p_met.get('strong_threshold'))})"
        if "strong_threshold" in c_met
        else "- Strong threshold exceed (%): n/a",
        f"- Quant threshold exceed (%): {c_met.get('quant_threshold', float('nan')):.1f} (delta {fmt_delta(c_met.get('quant_threshold'), p_met.get('quant_threshold'))})"
        if "quant_threshold" in c_met
        else "- Quant threshold exceed (%): n/a",
        "",
        "## Dosya Boyut Farklari (ortak dosyalar)",
        "",
    ]

    if common:
        for name in common:
            d = c_files[name] - p_files[name]
            lines.append(f"- {name}: {c_files[name]} bytes (delta {d:+d})")
    else:
        lines.append("- Ortak dosya bulunamadi.")

    lines += ["", "## Yeni / Kaldirilan Dosyalar", ""]
    if not prev_manifest_exists:
        lines.append("- Onceki manifest olmadigi icin dosya listesi farki hesaplanamadi.")
    elif added:
        lines.append("- Added:")
        for n in added:
            lines.append(f"  - {n}")
    else:
        lines.append("- Added: yok")
    if not prev_manifest_exists:
        lines.append("- Removed: n/a")
    elif removed:
        lines.append("- Removed:")
        for n in removed:
            lines.append(f"  - {n}")
    else:
        lines.append("- Removed: yok")

    lines += ["", "## Temel Artefakt Boyut Farklari", ""]
    keys = ["exec_pptx", "detailed_pptx", "board_pptx", "brief_pdf", "bundle_zip"]
    for k in keys:
        cv = c_sizes.get(k)
        pv = p_sizes.get(k)
        if cv is None and pv is None:
            lines.append(f"- {k}: n/a")
            continue
        if cv is None:
            lines.append(f"- {k}: current n/a, previous {pv} bytes")
            continue
        if pv is None:
            lines.append(f"- {k}: current {cv} bytes, previous n/a")
            continue
        lines.append(f"- {k}: current {cv} bytes, previous {pv} bytes (delta {cv - pv:+d})")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
