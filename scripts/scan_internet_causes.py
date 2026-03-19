#!/usr/bin/env python3
"""Scan internet-backed climate drivers and enrich extreme events with causes.

Internet sources (downloaded during run):
- NOAA CPC ONI page (ENSO index table)
- NOAA CPC NAO monthly index ascii
- NOAA CPC NAO description page (impact notes)
- NOAA Climate.gov ENSO overview page
- NASA Earth Observatory 2010 blocking event page
- NASA/NOAA Pinatubo impact references
"""

from __future__ import annotations

import argparse
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ONI_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
NAO_ASCII_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
NAO_DOC_URL = "https://www.cpc.ncep.noaa.gov/data/teledoc/nao.shtml"
ENSO_OVERVIEW_URL = "https://content-drupal.climate.gov/enso"
NASA_2010_BLOCKING_URL = "https://earthobservatory.nasa.gov/images/44815/russian-fires-and-pakistan-floods-linked"
PINATUBO_NASA_URL = "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19990021520.pdf"
PINATUBO_NOAA_URL = "https://gml.noaa.gov/grad/abstracts.html"


@dataclass
class KnownWindow:
    title: str
    start: str
    end: str
    note: str
    source: str


KNOWN_WINDOWS = [
    KnownWindow(
        title="1982-1983 El Nino + El Chichon",
        start="1982-04-01",
        end="1983-12-31",
        note="Guclu El Nino ve volkanik aerosol etkisi birlikte.",
        source=f"{ONI_URL}; {PINATUBO_NOAA_URL}",
    ),
    KnownWindow(
        title="1991-1993 Pinatubo aerosol cooling",
        start="1991-06-01",
        end="1993-12-31",
        note="Pinatubo sonrasi gecici soguma ve atmosferik bozulma etkisi.",
        source=f"{PINATUBO_NASA_URL}; {PINATUBO_NOAA_URL}",
    ),
    KnownWindow(
        title="1997-1998 Super El Nino",
        start="1997-05-01",
        end="1998-06-30",
        note="Cok guclu El Nino donemi.",
        source=ONI_URL,
    ),
    KnownWindow(
        title="1998-2001 Multi-year La Nina",
        start="1998-07-01",
        end="2001-03-31",
        note="Cok yilli La Nina donemi.",
        source=ONI_URL,
    ),
    KnownWindow(
        title="2010 blocking + Russia fires/Pakistan floods",
        start="2010-07-01",
        end="2010-09-30",
        note="Kuresel olcekte dalga/engelleme baglantili asiri olaylar.",
        source=NASA_2010_BLOCKING_URL,
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asiri olaylar icin internet tabanli neden taramasi.")
    p.add_argument(
        "--events-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar_zengin.csv"),
    )
    p.add_argument(
        "--quant-context-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/quant_all_visuals_package/reports/top_anomalies_global_context_input.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events"),
    )
    p.add_argument("--context-day-window", type=int, default=45)
    return p.parse_args()


def download(url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sS", url, "-o", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def parse_oni_monthly(oni_html: Path) -> pd.DataFrame:
    if not oni_html.exists() or oni_html.stat().st_size == 0:
        raise SystemExit(f"ONI kaynagi bulunamadi: {oni_html}")
    html_text = oni_html.read_text(encoding="utf-8", errors="ignore")
    try:
        tables = pd.read_html(io.StringIO(html_text), flavor="lxml")
    except Exception:
        tables = pd.read_html(io.StringIO(html_text))
    target = None
    for t in tables:
        if t.empty:
            continue
        tt = t.copy()
        # Header can live in first row.
        if str(tt.iloc[0, 0]).strip().lower() == "year":
            tt.columns = tt.iloc[0]
            tt = tt.iloc[1:].copy()
        cols = [str(c).strip().upper() for c in tt.columns]
        if "YEAR" in cols and "DJF" in cols and "NDJ" in cols:
            target = tt.copy()
            target.columns = [str(c).strip().upper() for c in target.columns]
            break
    if target is None:
        raise SystemExit("ONI tablosu parse edilemedi.")

    season_to_month = {
        "DJF": 1,
        "JFM": 2,
        "FMA": 3,
        "MAM": 4,
        "AMJ": 5,
        "MJJ": 6,
        "JJA": 7,
        "JAS": 8,
        "ASO": 9,
        "SON": 10,
        "OND": 11,
        "NDJ": 12,
    }
    rows: list[dict[str, Any]] = []
    for _, r in target.iterrows():
        y = pd.to_numeric(r.get("YEAR"), errors="coerce")
        if pd.isna(y):
            continue
        yy = int(y)
        for s, m in season_to_month.items():
            v = pd.to_numeric(r.get(s), errors="coerce")
            if pd.isna(v):
                continue
            rows.append({"year": yy, "month": m, "oni": float(v)})
    out = pd.DataFrame(rows).drop_duplicates(["year", "month"]).sort_values(["year", "month"])
    return out.reset_index(drop=True)


def parse_nao_monthly(nao_ascii: Path) -> pd.DataFrame:
    recs = []
    for line in nao_ascii.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            y = int(parts[0])
            m = int(parts[1])
            v = float(parts[2])
        except Exception:
            continue
        recs.append({"year": y, "month": m, "nao": v})
    return pd.DataFrame(recs).drop_duplicates(["year", "month"]).sort_values(["year", "month"]).reset_index(drop=True)


def enso_phase(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "unknown"
    if v >= 1.5:
        return "el_nino_strong"
    if v >= 0.5:
        return "el_nino"
    if v <= -1.5:
        return "la_nina_strong"
    if v <= -0.5:
        return "la_nina"
    return "enso_neutral"


def nao_phase(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "unknown"
    if v >= 1.0:
        return "nao_strong_positive"
    if v >= 0.5:
        return "nao_positive"
    if v <= -1.0:
        return "nao_strong_negative"
    if v <= -0.5:
        return "nao_negative"
    return "nao_neutral"


def seasonal_nao_note(var: str, month: int, nao_v: float | None) -> str:
    if nao_v is None or not np.isfinite(nao_v):
        return ""
    if month not in {11, 12, 1, 2, 3}:
        return ""
    if nao_v >= 0.5:
        if var in {"precip", "humidity"}:
            return "NAO+ kis paterni: Akdeniz/Turkiye tarafinda daha kuru egilim."
        if var == "temp":
            return "NAO+ kis paterni: bolgesel olarak daha ilik kosullara kayma olasi."
        return "NAO+ paterni: basinç gradyani ve firtina rotasi kaymasi etkisi olasi."
    if nao_v <= -0.5:
        if var in {"precip", "humidity"}:
            return "NAO- kis paterni: Akdeniz/Turkiye tarafinda daha islak/yagisli egilim."
        if var == "temp":
            return "NAO- kis paterni: daha serin/soguk kosullara kayma olasi."
        return "NAO- paterni: basinç ve siklonik aktivite degisimi etkisi olasi."
    return ""


def overlap_window(center: pd.Timestamp) -> tuple[str, str, str]:
    for w in KNOWN_WINDOWS:
        s = pd.Timestamp(w.start)
        e = pd.Timestamp(w.end)
        if s <= center <= e:
            return w.title, w.note, w.source
    return "", "", ""


def match_quant_context(
    ev_var: str,
    center: pd.Timestamp,
    qctx: pd.DataFrame,
    day_window: int,
) -> pd.Series | None:
    sub = qctx[qctx["variable"] == ev_var]
    if sub.empty:
        return None
    # Try same month first.
    m = sub[sub["ds"].dt.strftime("%Y-%m") == center.strftime("%Y-%m")]
    if not m.empty:
        return m.iloc[0]
    # Then same year.
    y = sub[sub["ds"].dt.year == center.year]
    if not y.empty:
        dd = (y["ds"] - center).abs().dt.days
        return y.iloc[int(dd.argmin())]
    # Then nearest within window.
    dd = (sub["ds"] - center).abs().dt.days
    i = int(dd.argmin())
    if int(dd.iloc[i]) <= int(day_window):
        return sub.iloc[i]
    return None


def confidence_label(score: int) -> str:
    if score >= 3:
        return "cok_yuksek"
    if score == 2:
        return "yuksek"
    if score == 1:
        return "orta"
    return "dusuk"


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    src_dir = out / "internet_sources"
    src_dir.mkdir(parents=True, exist_ok=True)

    oni_html = src_dir / "oni_v5.html"
    nao_ascii = src_dir / "nao_monthly.ascii"
    nao_doc = src_dir / "nao_doc.html"
    enso_doc = src_dir / "enso_overview.html"

    ok_oni = download(ONI_URL, oni_html)
    ok_nao = download(NAO_ASCII_URL, nao_ascii)
    ok_nao_doc = download(NAO_DOC_URL, nao_doc)
    ok_enso_doc = download(ENSO_OVERVIEW_URL, enso_doc)
    if not all([ok_oni, ok_nao, ok_nao_doc, ok_enso_doc]):
        raise SystemExit(
            "Bazi internet kaynaklari indirilemedi. "
            f"ONI={ok_oni}, NAO={ok_nao}, NAO_DOC={ok_nao_doc}, ENSO_DOC={ok_enso_doc}"
        )

    oni = parse_oni_monthly(oni_html)
    nao = parse_nao_monthly(nao_ascii)

    events = pd.read_csv(args.events_csv).copy()
    events["variable"] = events["variable"].astype(str).str.lower().str.strip()
    events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce")
    events["end_time"] = pd.to_datetime(events["end_time"], errors="coerce")
    events = events.dropna(subset=["start_time", "end_time", "variable"]).copy()

    qctx = pd.read_csv(args.quant_context_csv).copy()
    qctx["variable"] = qctx["variable"].astype(str).str.lower().str.strip()
    qctx["ds"] = pd.to_datetime(qctx["ds"], errors="coerce")
    qctx = qctx.dropna(subset=["variable", "ds"]).copy()

    rows = []
    oni_ix = oni.set_index(["year", "month"])["oni"].to_dict()
    nao_ix = nao.set_index(["year", "month"])["nao"].to_dict()

    for _, r in events.iterrows():
        center = r["start_time"] + (r["end_time"] - r["start_time"]) / 2
        y, m = int(center.year), int(center.month)
        oni_v = oni_ix.get((y, m), np.nan)
        nao_v = nao_ix.get((y, m), np.nan)
        ep = enso_phase(float(oni_v) if np.isfinite(oni_v) else None)
        npf = nao_phase(float(nao_v) if np.isfinite(nao_v) else None)

        tele_note = seasonal_nao_note(str(r["variable"]), m, float(nao_v) if np.isfinite(nao_v) else None)
        win_title, win_note, win_src = overlap_window(center)
        qhit = match_quant_context(str(r["variable"]), center, qctx, day_window=int(args.context_day_window))

        cause_parts: list[str] = []
        source_parts = [ONI_URL, NAO_ASCII_URL, NAO_DOC_URL, ENSO_OVERVIEW_URL]
        conf_score = 0

        if ep != "unknown" and ep != "enso_neutral":
            cause_parts.append(f"ENSO fazi: {ep} (ONI={oni_v:.2f})")
            conf_score += 1
        elif ep == "enso_neutral":
            cause_parts.append(f"ENSO neutral (ONI={oni_v:.2f})")

        if npf not in {"unknown", "nao_neutral"}:
            cause_parts.append(f"NAO fazi: {npf} (NAO={nao_v:.2f})")
            conf_score += 1
        elif npf == "nao_neutral" and np.isfinite(nao_v):
            cause_parts.append(f"NAO neutral (NAO={nao_v:.2f})")

        if tele_note:
            cause_parts.append(tele_note)

        if win_title:
            cause_parts.append(f"Kuresel pencere eslesmesi: {win_title} - {win_note}")
            source_parts.append(win_src)
            conf_score += 1

        quant_cause_primary = ""
        quant_local = ""
        quant_global = ""
        quant_source = ""
        if qhit is not None:
            quant_cause_primary = str(qhit.get("cause_primary", "")).strip()
            quant_local = str(qhit.get("local_event_match", "")).strip()
            quant_global = str(qhit.get("global_event_match", "")).strip()
            quant_source = str(qhit.get("global_event_source", "")).strip()
            if quant_cause_primary:
                cause_parts.append(f"Quant baglami: {quant_cause_primary}")
                conf_score += 1
            if quant_local and quant_local != "doğrudan eşleşme yok":
                cause_parts.append(f"Yerel olay eslesmesi: {quant_local}")
            if quant_global and quant_global != "doğrudan eşleşme yok":
                cause_parts.append(f"Global olay eslesmesi: {quant_global}")
            if quant_source:
                source_parts.append(quant_source)

        if not cause_parts:
            cause_parts.append("Indeks tabanli acik bir dis surucu eslesmesi bulunamadi; yerel/sinoptik etki olasi.")

        rows.append(
            {
                **r.to_dict(),
                "center_time": center,
                "oni_value": float(oni_v) if np.isfinite(oni_v) else np.nan,
                "enso_phase": ep,
                "nao_value": float(nao_v) if np.isfinite(nao_v) else np.nan,
                "nao_phase": npf,
                "known_window_title": win_title,
                "known_window_note": win_note,
                "quant_cause_primary": quant_cause_primary,
                "quant_local_event_match": quant_local,
                "quant_global_event_match": quant_global,
                "internet_cause_summary": " | ".join(cause_parts),
                "internet_confidence": confidence_label(conf_score),
                "internet_source_links": " ; ".join(sorted({x for x in source_parts if str(x).strip()})),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        ["peak_severity_score", "duration_points"], ascending=[False, False]
    ).reset_index(drop=True)
    out_df.insert(0, "internet_rank", np.arange(1, len(out_df) + 1))

    full_csv = out / "tum_asiri_olaylar_internet_nedenleri.csv"
    out_df.to_csv(full_csv, index=False)

    summary = (
        out_df.groupby("variable", as_index=False)
        .agg(
            event_count=("event_id", "size"),
            strong_enso=("enso_phase", lambda s: int(s.astype(str).str.contains("strong").sum())),
            nonneutral_enso=("enso_phase", lambda s: int((~s.astype(str).isin(["unknown", "enso_neutral"])).sum())),
            nonneutral_nao=("nao_phase", lambda s: int((~s.astype(str).isin(["unknown", "nao_neutral"])).sum())),
            known_window_hits=("known_window_title", lambda s: int(s.astype(str).str.len().gt(0).sum())),
        )
        .sort_values("event_count", ascending=False)
    )
    sum_csv = out / "tum_asiri_olaylar_internet_ozet.csv"
    summary.to_csv(sum_csv, index=False)

    top = out_df.head(60)
    md = out / "tum_asiri_olaylar_internet_rapor.md"
    lines = [
        "# Tum Asiri Olaylar - Internet Taramasi Neden Raporu",
        "",
        f"- Toplam olay: **{len(out_df)}**",
        f"- Kaynaklar: ONI, NAO, CPC teleconnection dokumani, ENSO overview, quant baglam dosyasi.",
        "",
        "## Kaynak Baglantilari",
        f"- ONI: {ONI_URL}",
        f"- NAO monthly index: {NAO_ASCII_URL}",
        f"- NAO dokuman: {NAO_DOC_URL}",
        f"- ENSO overview: {ENSO_OVERVIEW_URL}",
        f"- NASA 2010 blocking olayi: {NASA_2010_BLOCKING_URL}",
        f"- Pinatubo (NASA): {PINATUBO_NASA_URL}",
        f"- Pinatubo (NOAA/GML): {PINATUBO_NOAA_URL}",
        "",
        "## Degisken Ozet",
    ]
    for _, rr in summary.iterrows():
        lines.append(
            f"- {rr['variable']}: olay={int(rr['event_count'])}, ENSO_nonneutral={int(rr['nonneutral_enso'])}, "
            f"NAO_nonneutral={int(rr['nonneutral_nao'])}, pencere_eslesme={int(rr['known_window_hits'])}"
        )
    lines += [
        "",
        "## En Siddetli 60 Olay (internet neden ozeti)",
        "",
        "|Sira|Event|Degisken|Baslangic|Bitis|Siddet|ENSO|NAO|Guven|Neden Ozeti|",
        "|---:|---|---|---|---|---:|---|---|---|---|",
    ]
    for _, rr in top.iterrows():
        lines.append(
            f"|{int(rr['internet_rank'])}|{rr['event_id']}|{rr['variable']}|{pd.Timestamp(rr['start_time']).date()}|"
            f"{pd.Timestamp(rr['end_time']).date()}|{float(rr['peak_severity_score']):.2f}|{rr['enso_phase']}|"
            f"{rr['nao_phase']}|{rr['internet_confidence']}|{str(rr['internet_cause_summary']).replace('|',' / ')}|"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {full_csv}")
    print(f"Wrote: {sum_csv}")
    print(f"Wrote: {md}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
