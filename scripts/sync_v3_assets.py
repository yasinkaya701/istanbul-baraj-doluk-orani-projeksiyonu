#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/Users/yasinkaya/Hackhaton")
SRC_DIR = ROOT / "output/istanbul_v3/figures"
TARGETS = [
    ROOT / "assets/v3",
    ROOT / "baraj_web/assets/v3",
]

FILES = [
    "lgb_dart_v3_projection.png",
    "xgb_v3_projection.png",
    "etr_v3_projection.png",
    "stack_v3_projection.png",
    "ensemble_median_v3_projection.png",
    "model_comparison_cv.png",
]


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"Source directory not found: {SRC_DIR}")

    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            src = SRC_DIR / name
            if not src.exists():
                raise SystemExit(f"Missing figure: {src}")
            shutil.copy2(src, target / name)
            print(f"Copied {src} -> {target / name}")


if __name__ == "__main__":
    main()
