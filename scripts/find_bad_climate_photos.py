#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class ImageMetrics:
    path: str
    ok: bool
    width: int
    height: int
    megapixels: float
    mean_brightness: float
    std_contrast: float
    dynamic_range: float
    laplacian_var: float
    edge_density: float
    ink_ratio: float
    white_ratio: float
    dark_ratio: float
    flags: str = ""
    score: float = 0.0
    severity: str = "OK"


def load_image_bgr(path: Path) -> np.ndarray | None:
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            raise ValueError("empty_file")
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    except Exception:
        pass

    try:
        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            arr = np.array(rgb)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def resize_for_speed(image: np.ndarray, max_side: int = 1800) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = float(max_side) / float(side)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def compute_metrics(path: Path) -> ImageMetrics:
    image = load_image_bgr(path)
    if image is None:
        return ImageMetrics(
            path=str(path),
            ok=False,
            width=0,
            height=0,
            megapixels=0.0,
            mean_brightness=0.0,
            std_contrast=0.0,
            dynamic_range=0.0,
            laplacian_var=0.0,
            edge_density=0.0,
            ink_ratio=0.0,
            white_ratio=0.0,
            dark_ratio=0.0,
            flags="load_failed",
            score=10.0,
            severity="UNUSABLE",
        )

    h, w = image.shape[:2]
    image_small = resize_for_speed(image)
    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    mean_brightness = float(gray_f.mean())
    std_contrast = float(gray_f.std())
    p05 = float(np.percentile(gray_f, 5))
    p95 = float(np.percentile(gray_f, 95))
    dynamic_range = p95 - p05
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    med = float(np.median(gray_f))
    lower = max(0, int(0.66 * med))
    upper = min(255, int(1.33 * med))
    edges = cv2.Canny(gray, lower, upper)
    edge_density = float(edges.mean() / 255.0)

    ink_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        10,
    )
    ink_ratio = float(ink_mask.mean() / 255.0)
    white_ratio = float((gray_f >= 245).mean())
    dark_ratio = float((gray_f <= 40).mean())

    return ImageMetrics(
        path=str(path),
        ok=True,
        width=w,
        height=h,
        megapixels=float((w * h) / 1_000_000.0),
        mean_brightness=mean_brightness,
        std_contrast=std_contrast,
        dynamic_range=dynamic_range,
        laplacian_var=laplacian_var,
        edge_density=edge_density,
        ink_ratio=ink_ratio,
        white_ratio=white_ratio,
        dark_ratio=dark_ratio,
    )


def q(values: Iterable[float], p: float, default: float) -> float:
    arr = np.array(list(values), dtype=np.float32)
    if arr.size == 0:
        return default
    return float(np.quantile(arr, p))


def classify(metrics: List[ImageMetrics]) -> Dict[str, float]:
    good = [m for m in metrics if m.ok]
    thresholds = {
        "mp_q05": q((m.megapixels for m in good), 0.05, 0.8),
        "blur_q05": q((m.laplacian_var for m in good), 0.05, 25.0),
        "contrast_q05": q((m.std_contrast for m in good), 0.05, 20.0),
        "dyn_q05": q((m.dynamic_range for m in good), 0.05, 60.0),
        "edge_q05": q((m.edge_density for m in good), 0.05, 0.01),
        "ink_q05": q((m.ink_ratio for m in good), 0.05, 0.02),
        "bright_q02": q((m.mean_brightness for m in good), 0.02, 70.0),
        "bright_q98": q((m.mean_brightness for m in good), 0.98, 230.0),
        "white_q98": q((m.white_ratio for m in good), 0.98, 0.95),
        "dark_q02": q((m.dark_ratio for m in good), 0.02, 0.002),
    }

    for m in metrics:
        if not m.ok:
            continue
        flags: List[str] = []
        score = 0.0
        mp_floor = max(0.35, thresholds["mp_q05"] * 0.8)

        if m.megapixels < mp_floor:
            flags.append("very_low_resolution")
            score += 2.0
        if m.laplacian_var < thresholds["blur_q05"]:
            flags.append("blurry")
            score += 1.6
        if m.std_contrast < thresholds["contrast_q05"]:
            flags.append("low_contrast")
            score += 1.3
        if m.dynamic_range < thresholds["dyn_q05"]:
            flags.append("narrow_dynamic_range")
            score += 1.0
        if m.edge_density < thresholds["edge_q05"]:
            flags.append("low_detail")
            score += 1.0
        if m.ink_ratio < thresholds["ink_q05"]:
            flags.append("low_ink")
            score += 1.5
        if m.white_ratio > thresholds["white_q98"] and m.dark_ratio < thresholds["dark_q02"]:
            flags.append("too_blank_or_overexposed")
            score += 2.3
        if m.mean_brightness > thresholds["bright_q98"]:
            flags.append("overexposed")
            score += 1.5
        if m.mean_brightness < thresholds["bright_q02"]:
            flags.append("underexposed")
            score += 1.5

        if "low_ink" in flags and ("low_detail" in flags or "too_blank_or_overexposed" in flags):
            flags.append("possible_missing_writing")
            score += 1.2

        if any(f in flags for f in ("very_low_resolution", "too_blank_or_overexposed")):
            severity = "UNUSABLE"
        elif score >= 3.0 or len(flags) >= 2:
            severity = "REVIEW"
        else:
            severity = "OK"

        m.flags = "|".join(flags)
        m.score = float(score)
        m.severity = severity

    return thresholds


def write_csv(path: Path, rows: List[ImageMetrics]) -> None:
    fields = list(asdict(rows[0]).keys()) if rows else list(asdict(ImageMetrics("", False, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def copy_flagged(rows: List[ImageMetrics], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()
    for row in rows:
        src = Path(row.path)
        if not src.exists():
            continue
        safe_name = src.name
        dst = out_dir / safe_name
        idx = 1
        while dst.exists() and dst.resolve() != src.resolve():
            dst = out_dir / f"{src.stem}_{idx}{src.suffix}"
            idx += 1
        if dst.exists():
            continue
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            dst.write_bytes(src.read_bytes())


def build_summary(rows: List[ImageMetrics], out_path: Path, thresholds: Dict[str, float]) -> None:
    total = len(rows)
    unusable = [r for r in rows if r.severity == "UNUSABLE"]
    review = [r for r in rows if r.severity == "REVIEW"]
    failed = [r for r in rows if not r.ok]

    by_flag: Dict[str, int] = {}
    for r in rows:
        if not r.flags:
            continue
        for flag in r.flags.split("|"):
            by_flag[flag] = by_flag.get(flag, 0) + 1

    top = sorted([r for r in rows if r.severity != "OK"], key=lambda x: (-x.score, x.path))[:30]

    lines = [
        "# Tarama Kalite Raporu",
        "",
        f"- Toplam dosya: **{total}**",
        f"- UNUSABLE: **{len(unusable)}**",
        f"- REVIEW: **{len(review)}**",
        f"- Okunamayan dosya: **{len(failed)}**",
        "",
        "## Eşikler",
        "",
    ]
    for k in sorted(thresholds.keys()):
        lines.append(f"- `{k}`: {thresholds[k]:.6f}")

    lines.extend(["", "## Flag Dağılımı", ""])
    for k, v in sorted(by_flag.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{k}`: {v}")

    lines.extend(["", "## İlk 30 Şüpheli", ""])
    for r in top:
        lines.append(f"- `{r.severity}` score={r.score:.2f} `{r.path}` flags={r.flags}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graf kagidi taramalarinda eksik yazi / ise yaramaz foto adaylarini tespit eder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("DATA/Graf Kağıtları Tarama "),
        help="Taramalarin oldugu klasor",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".tif", ".tiff"],
        help="Taranacak uzantilar",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/photo_quality_scan"),
        help="Rapor cikti klasoru",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Sadece ilk N dosyayi tara (0 = tumu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions}

    files = [p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]
    files.sort()
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"Dosya bulunamadi: {args.input_dir}")

    metrics = [compute_metrics(p) for p in files]
    thresholds = classify(metrics)

    all_sorted = sorted(metrics, key=lambda x: x.path)
    flagged = sorted((m for m in metrics if m.severity != "OK"), key=lambda x: (-x.score, x.path))
    unusable = [m for m in flagged if m.severity == "UNUSABLE"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_metrics.csv", all_sorted)
    write_csv(args.output_dir / "flagged_review.csv", flagged)
    write_csv(args.output_dir / "unusable.csv", unusable)
    (args.output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    build_summary(all_sorted, args.output_dir / "summary.md", thresholds)
    copy_flagged(flagged, args.output_dir / "flagged_links")

    print(f"Scanned: {len(all_sorted)}")
    print(f"UNUSABLE: {len(unusable)}")
    print(f"REVIEW: {len(flagged) - len(unusable)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
