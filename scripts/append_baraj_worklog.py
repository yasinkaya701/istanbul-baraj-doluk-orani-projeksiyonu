#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


WORKLOG = Path("/Users/yasinkaya/Hackhaton/research/baraj_doluluk_hub/logs/WORKLOG.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a timestamped line to the baraj hub worklog.")
    parser.add_argument("message", help="Short worklog message.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"- `{stamp}` - {args.message}\n"
    with WORKLOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(WORKLOG)


if __name__ == "__main__":
    main()
