#!/usr/bin/env python3
"""Build a continuous Istanbul overall dam chart with side event notes."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Istanbul baraj kesintisiz olayli grafik")
    p.add_argument(
        "--history-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/istanbul_dam_monthly_history.csv"),
    )
    p.add_argument(
        "--forecast-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/istanbul_dam_forecasts_decision.csv"),
    )
    p.add_argument(
        "--events-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar_bilimsel_filtreli.csv"),
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/istanbul_baraj_kesintisiz_olayli_grafik.png"),
    )
    p.add_argument(
        "--output-notes-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/istanbul_baraj_ani_dusus_olay_notlari.csv"),
    )
    p.add_argument(
        "--output-notes-md",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/istanbul_dam_forecast_decision/istanbul_baraj_ani_dusus_olay_notlari.md"),
    )
    p.add_argument("--top-drops", type=int, default=8)
    p.add_argument("--min-gap-months", type=int, default=2)
    p.add_argument("--match-window-days", type=int, default=90)
    return p.parse_args()


def load_series(history_csv: Path, forecast_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist = pd.read_csv(history_csv, parse_dates=["ds"])
    if "overall_mean" not in hist.columns:
        raise SystemExit("history csv missing overall_mean")
    hist = hist[["ds", "overall_mean"]].dropna().sort_values("ds").reset_index(drop=True)
    hist = hist.rename(columns={"overall_mean": "value"})
    hist["source"] = "gerceklesen"

    fc = pd.read_csv(forecast_csv, parse_dates=["ds"])
    fc = fc[fc["series"] == "overall_mean"][["ds", "yhat"]].dropna().sort_values("ds").reset_index(drop=True)
    fc = fc.rename(columns={"yhat": "value"})
    last_hist = pd.Timestamp(hist["ds"].max())
    fc = fc[fc["ds"] > last_hist].copy()
    fc["source"] = "tahmin"

    combined = pd.concat([hist, fc], ignore_index=True)
    combined = combined.sort_values("ds").reset_index(drop=True)
    combined["value"] = combined["value"].clip(0.0, 1.0)
    return hist, fc, combined


def pick_sharp_drops(hist: pd.DataFrame, top_n: int, min_gap_months: int) -> pd.DataFrame:
    d = hist.copy()
    d["diff"] = d["value"].diff()
    cand = d[d["diff"] < 0].copy()
    cand = cand.sort_values("diff")  # most negative first

    selected = []
    for _, r in cand.iterrows():
        cur = pd.Timestamp(r["ds"])
        too_close = False
        for sr in selected:
            delta_m = abs((cur.year - sr.year) * 12 + (cur.month - sr.month))
            if delta_m < max(0, int(min_gap_months)):
                too_close = True
                break
        if too_close:
            continue
        selected.append(cur)
        if len(selected) >= max(1, int(top_n)):
            break

    out = d[d["ds"].isin(selected)].copy().sort_values("ds")
    out["drop_pct_point"] = out["diff"] * 100.0
    return out


def load_events(events_csv: Path) -> pd.DataFrame:
    ev = pd.read_csv(events_csv)
    # Mode A: news matched dataset
    if ("headline_date" in ev.columns or "top_headline_date" in ev.columns) and (
        "headline" in ev.columns or "top_headline" in ev.columns
    ):
        date_col = "headline_date" if "headline_date" in ev.columns else "top_headline_date"
        title_col = "headline" if "headline" in ev.columns else "top_headline"
        source_col = "source" if "source" in ev.columns else ("top_headline_source" if "top_headline_source" in ev.columns else None)
        url_col = "url" if "url" in ev.columns else ("top_headline_url" if "top_headline_url" in ev.columns else None)
        hazard_col = "hazard_type" if "hazard_type" in ev.columns else ("top_headline_hazard" if "top_headline_hazard" in ev.columns else None)
        out = pd.DataFrame(
            {
                "event_date": pd.to_datetime(ev[date_col], errors="coerce"),
                "headline": ev[title_col].astype(str),
                "source": ev[source_col].astype(str) if source_col else "",
                "url": ev[url_col].astype(str) if url_col else "",
                "hazard_type": ev[hazard_col].astype(str) if hazard_col else "",
            }
        )
    # Mode B: scientific enriched events dataset
    elif "center_time" in ev.columns or "start_time" in ev.columns:
        date_col = "center_time" if "center_time" in ev.columns else "start_time"
        variable = ev["variable"] if "variable" in ev.columns else pd.Series([""] * len(ev))
        cause = ev["context_cause_primary"] if "context_cause_primary" in ev.columns else pd.Series([""] * len(ev))
        known = ev["known_window_title"] if "known_window_title" in ev.columns else pd.Series([""] * len(ev))
        anomaly = ev["context_anomaly_type_tr"] if "context_anomaly_type_tr" in ev.columns else pd.Series([""] * len(ev))
        severity = ev["severity_level"] if "severity_level" in ev.columns else pd.Series([""] * len(ev))

        def _clean(x: object) -> str:
            s = "" if x is None else str(x).strip()
            return "" if s.lower() in {"", "nan", "none"} else s

        headline_vals = []
        for v, c, k, a, s in zip(variable, cause, known, anomaly, severity, strict=False):
            parts = []
            vv = _clean(v)
            cc = _clean(c)
            kk = _clean(k)
            aa = _clean(a)
            ss = _clean(s)
            if vv:
                parts.append(f"Degisken={vv}")
            if cc:
                parts.append(f"Neden={cc}")
            if aa:
                parts.append(f"Tip={aa}")
            if kk:
                parts.append(f"Pencere={kk}")
            if ss:
                parts.append(f"Siddet={ss}")
            if not parts:
                parts = ["Bilimsel olay kaydi"]
            headline_vals.append(" | ".join(parts))

        links = ev["internet_source_links"].astype(str) if "internet_source_links" in ev.columns else ""
        first_link = links.astype(str).str.split(";").str[0].str.strip()
        out = pd.DataFrame(
            {
                "event_date": pd.to_datetime(ev[date_col], errors="coerce"),
                "headline": pd.Series(headline_vals).astype(str),
                "source": "bilimsel_olay",
                "url": first_link.astype(str),
                "hazard_type": variable.astype(str),
            }
        )
    else:
        raise SystemExit("events csv format not recognized")

    out = out.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
    return out


def match_event(drop_date: pd.Timestamp, events: pd.DataFrame, window_days: int) -> pd.Series:
    if events.empty:
        return pd.Series({"event_date": pd.NaT, "headline": "Veri yok", "source": "", "url": "", "hazard_type": ""})
    w = max(1, int(window_days))
    cand = events[(events["event_date"] >= drop_date - pd.Timedelta(days=w)) & (events["event_date"] <= drop_date + pd.Timedelta(days=w))].copy()
    if cand.empty:
        return pd.Series(
            {
                "event_date": pd.NaT,
                "headline": "Bu tarihe yakin olay kaydi bulunamadi",
                "source": "",
                "url": "",
                "hazard_type": "",
            }
        )

    txt = (cand["headline"].fillna("") + " " + cand["source"].fillna("")).str.lower()
    bonus = (
        txt.str.contains("istanbul").astype(int) * 3
        + txt.str.contains("baraj|su|kurak|yagis|sel|flood|drought", regex=True).astype(int) * 2
        + txt.str.contains("turkiye|turkey").astype(int)
    )
    day_diff = np.abs((cand["event_date"] - drop_date).dt.days.to_numpy(dtype=float))
    score = -day_diff + bonus.to_numpy(dtype=float)
    idx = int(np.argmax(score))
    return cand.iloc[idx]


def build_notes(drop_df: pd.DataFrame, events: pd.DataFrame, window_days: int) -> pd.DataFrame:
    rows = []
    for i, (_, r) in enumerate(drop_df.sort_values("ds").iterrows(), start=1):
        e = match_event(pd.Timestamp(r["ds"]), events, window_days=window_days)
        rows.append(
            {
                "sira": i,
                "dusus_tarihi": pd.Timestamp(r["ds"]).strftime("%Y-%m-%d"),
                "dusus_puan": float(r["drop_pct_point"]),
                "olay_tarihi": "" if pd.isna(e["event_date"]) else pd.Timestamp(e["event_date"]).strftime("%Y-%m-%d"),
                "olay_basligi": str(e["headline"]),
                "kaynak": str(e["source"]),
                "url": str(e["url"]),
                "tehlike_tipi": str(e["hazard_type"]),
            }
        )
    return pd.DataFrame(rows)


def maybe_load_direct_notes(events_csv: Path, drop_df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        raw = pd.read_csv(events_csv)
    except Exception:
        return None

    if "dusus_tarihi" not in raw.columns or "olay_basligi" not in raw.columns:
        return None

    out = raw.copy()
    out["dusus_tarihi"] = pd.to_datetime(out["dusus_tarihi"], errors="coerce")
    out = out.dropna(subset=["dusus_tarihi"]).copy()

    drop_point_map = {
        pd.Timestamp(r["ds"]).strftime("%Y-%m-%d"): float(r["drop_pct_point"])
        for _, r in drop_df.iterrows()
    }
    if "dusus_puan" not in out.columns:
        out["dusus_puan"] = out["dusus_tarihi"].dt.strftime("%Y-%m-%d").map(drop_point_map)

    if "sira" not in out.columns:
        out = out.sort_values("dusus_tarihi").reset_index(drop=True)
        out["sira"] = np.arange(1, len(out) + 1)

    for col, default_val in {
        "olay_tarihi": "",
        "kaynak": "",
        "url": "",
        "tehlike_tipi": "",
    }.items():
        if col not in out.columns:
            out[col] = default_val

    out["dusus_tarihi"] = out["dusus_tarihi"].dt.strftime("%Y-%m-%d")
    if "olay_tarihi" in out.columns:
        out["olay_tarihi"] = pd.to_datetime(out["olay_tarihi"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

    keep = ["sira", "dusus_tarihi", "dusus_puan", "olay_tarihi", "olay_basligi", "kaynak", "url", "tehlike_tipi"]
    out = out[keep].sort_values("dusus_tarihi").reset_index(drop=True)
    out["sira"] = np.arange(1, len(out) + 1)
    return out


def plot_chart(
    hist: pd.DataFrame,
    fc: pd.DataFrame,
    combined: pd.DataFrame,
    drop_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.8], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    # continuous backbone
    ax.plot(combined["ds"], combined["value"] * 100.0, color="#0f172a", linewidth=2.0, alpha=0.85, label="Kesintisiz cizgi")
    ax.plot(hist["ds"], hist["value"] * 100.0, color="#1d4ed8", linewidth=2.2, label="Gerceklesen")
    if not fc.empty:
        ax.plot(fc["ds"], fc["value"] * 100.0, color="#dc2626", linewidth=2.0, linestyle="--", label="Tahmin")
        ax.axvline(hist["ds"].max(), color="#6b7280", linestyle=":", linewidth=1.0)
        ax.text(hist["ds"].max(), 99, "  Tahmin baslangici", fontsize=9, color="#374151", va="top")

    # highlight sharp drops
    drop_sorted = drop_df.sort_values("ds").reset_index(drop=True)
    ax.scatter(drop_sorted["ds"], drop_sorted["value"] * 100.0, color="#b91c1c", s=44, zorder=5)
    for i, (_, r) in enumerate(drop_sorted.iterrows(), start=1):
        ax.text(
            pd.Timestamp(r["ds"]),
            float(r["value"] * 100.0) + 1.2,
            str(i),
            color="#7f1d1d",
            fontsize=9,
            weight="bold",
            ha="center",
            va="bottom",
        )

    ax.set_title("Istanbul Baraj Doluluk (Kesintisiz) + Ani Dusus Notlari", fontsize=14, weight="bold")
    ax.set_ylabel("Doluluk (%)")
    ax.set_xlabel("Tarih")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    ax_text.axis("off")
    ax_text.set_title("Ani Dusus Tarihleri ve Olaylar", fontsize=12, weight="bold", pad=8)

    y = 0.98
    step = 0.115
    for _, r in notes_df.iterrows():
        title = str(r["olay_basligi"]).strip()
        if len(title) > 88:
            title = title[:85] + "..."
        wrapped = textwrap.fill(title, width=46)
        note = (
            f"{int(r['sira'])}) {r['dusus_tarihi']} | {float(r['dusus_puan']):.1f} puan\n"
            f"Olay: {wrapped}\n"
            f"Kaynak: {r['kaynak']}"
        )
        ax_text.text(0.01, y, note, ha="left", va="top", fontsize=9, family="sans-serif")
        y -= step
        if y < 0.04:
            break

    fig.suptitle("Istanbul Genel Baraj Analizi (Turkce, Kesintisiz)", fontsize=15, weight="bold", y=0.99)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_notes_md(notes_df: pd.DataFrame, out_md: Path, events_csv: Path) -> None:
    lines = []
    lines.append("# Istanbul Baraj Ani Dusus Notlari")
    lines.append("")
    lines.append(f"- Olay veri seti: `{events_csv}`")
    lines.append("")
    for _, r in notes_df.iterrows():
        lines.append(
            f"- {int(r['sira'])}) {r['dusus_tarihi']} ({float(r['dusus_puan']):.1f} puan) -> "
            f"{r['olay_basligi']} [{r['kaynak']}]"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    hist, fc, combined = load_series(args.history_csv, args.forecast_csv)
    drop_df = pick_sharp_drops(hist, top_n=args.top_drops, min_gap_months=args.min_gap_months)
    notes_df = maybe_load_direct_notes(args.events_csv, drop_df)
    if notes_df is None:
        events = load_events(args.events_csv)
        notes_df = build_notes(drop_df, events, window_days=args.match_window_days)

    notes_df.to_csv(args.output_notes_csv, index=False)
    write_notes_md(notes_df, args.output_notes_md, args.events_csv)
    plot_chart(hist, fc, combined, drop_df, notes_df, args.output_png)

    print(args.output_png)
    print(args.output_notes_csv)
    print(args.output_notes_md)
    print(args.events_csv)


if __name__ == "__main__":
    main()
