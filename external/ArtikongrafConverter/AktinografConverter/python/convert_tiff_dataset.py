#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .tif/.tiff files to .png or .jpg for Java/Python pipelines."
    )
    parser.add_argument("--input-dir", default="data", help="Folder containing tif/tiff files.")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output folder for converted files. Default writes next to originals in data/.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output image format.",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (only for --format jpg).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Convert all pages of multi-page TIFF. Default: first page only.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input folder recursively.",
    )
    return parser.parse_args()


def find_tiff_files(input_dir: Path, recursive: bool) -> List[Path]:
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(input_dir.rglob(pattern))
        else:
            files.extend(input_dir.glob(pattern))
    return sorted(set(files))


def normalize_for_format(img: Image.Image, out_fmt: str) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if out_fmt == "jpg":
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.getchannel("A")
            bg.paste(img.convert("RGB"), mask=alpha)
            return bg
        return img.convert("RGB")
    # PNG
    if img.mode == "P":
        return img.convert("RGBA")
    if img.mode in ("1", "L", "LA", "RGB", "RGBA"):
        return img
    return img.convert("RGBA")


def convert_file(
    src: Path,
    out_dir: Path,
    out_fmt: str,
    quality: int,
    overwrite: bool,
    all_pages: bool,
) -> int:
    converted = 0
    with Image.open(src) as im:
        page_count = getattr(im, "n_frames", 1)
        pages = range(page_count) if all_pages else range(1)

        for i in pages:
            if page_count > 1:
                im.seek(i)

            img_page = im.copy()
            img_out = normalize_for_format(img_page, out_fmt)

            if all_pages and page_count > 1:
                out_name = f"{src.stem}_p{i + 1:03d}.{out_fmt}"
            else:
                out_name = f"{src.stem}.{out_fmt}"

            dst = out_dir / out_name
            if dst.exists() and not overwrite:
                continue

            save_kwargs = {}
            if out_fmt == "jpg":
                save_kwargs["quality"] = int(max(1, min(100, quality)))
                save_kwargs["optimize"] = True

            img_out.save(dst, **save_kwargs)
            converted += 1
    return converted


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_dir = (project_root / args.input_dir).resolve()
    out_dir = (project_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_tiff_files(input_dir, recursive=args.recursive)
    if not files:
        print(f"No TIFF files found in: {input_dir}")
        return

    total = 0
    for src in files:
        try:
            n = convert_file(
                src=src,
                out_dir=out_dir,
                out_fmt=args.format,
                quality=args.quality,
                overwrite=args.overwrite,
                all_pages=args.all_pages,
            )
            total += n
            print(f"[OK] {src.name} -> {n} file(s)")
        except Exception as e:
            print(f"[ERR] {src.name}: {e}")

    print(f"Converted files: {total}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()

