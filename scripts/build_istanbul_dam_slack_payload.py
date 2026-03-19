#!/usr/bin/env python3
"""Build Slack webhook payload from multi-scenario alert JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Slack payload for Istanbul dam alerts")
    p.add_argument(
        "--alerts-json",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/alerts_multi_scenario.json"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/slack_payload.json"),
    )
    p.add_argument("--top-n", type=int, default=5)
    return p.parse_args()


def line_for_alert(alert: dict) -> str:
    return (
        f"- {alert['series']}: level={alert['risk_level']}, "
        f"<40 ay={alert['months_below_40']}, <30 ay={alert['months_below_30']}, "
        f"ort P(<40)={alert['mean_probability_below_40_pct']:.1f}%, "
        f"en kotu={alert['worst_month']} ({alert['worst_forecast_pct']:.1f}%)"
    )


def scenario_sort_key(name: str) -> tuple[int, int, str]:
    s = str(name)
    if s == "baseline":
        return (0, 0, s)
    if s.startswith("dry_"):
        order = {"mild": 1, "base": 2, "stress": 2, "severe": 3, "extreme": 4}
        label = s.split("_", 1)[1]
        return (1, order.get(label, 9), s)
    if s.startswith("wet_"):
        order = {"mild": 1, "base": 2, "relief": 2, "severe": 3, "extreme": 4}
        label = s.split("_", 1)[1]
        return (2, order.get(label, 9), s)
    return (3, 99, s)


def build_payload(data: dict, top_n: int) -> dict:
    counts = data.get("counts_by_scenario", {})
    alerts_by = data.get("alerts_by_scenario", {})
    blocks = []
    blocks.append(
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Istanbul Baraj Risk Ozeti", "emoji": False},
        }
    )

    scenarios = sorted(alerts_by.keys(), key=scenario_sort_key)
    for scenario in scenarios:
        if scenario not in alerts_by:
            continue
        c = counts.get(scenario, {})
        alerts = alerts_by.get(scenario, [])[: max(1, int(top_n))]
        lines = [line_for_alert(a) for a in alerts]
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Senaryo:* `{scenario}`\n"
                        f"high={c.get('high',0)}, medium={c.get('medium',0)}, low={c.get('low',0)}\n"
                        + ("\n".join(lines) if lines else "- alert yok")
                    ),
                },
            }
        )
        blocks.append({"type": "divider"})

    # Remove trailing divider if any.
    if blocks and blocks[-1].get("type") == "divider":
        blocks = blocks[:-1]

    text_lines = ["Istanbul Baraj Risk Ozeti"]
    for scenario in scenarios:
        if scenario in counts:
            c = counts[scenario]
            text_lines.append(f"{scenario}: high={c.get('high',0)}, medium={c.get('medium',0)}, low={c.get('low',0)}")
    return {"text": " | ".join(text_lines), "blocks": blocks}


def main() -> None:
    args = parse_args()
    if not args.alerts_json.exists():
        raise SystemExit("alerts_multi_scenario.json not found.")
    data = json.loads(args.alerts_json.read_text(encoding="utf-8"))
    payload = build_payload(data, top_n=args.top_n)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
