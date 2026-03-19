#!/usr/bin/env python3
"""Attach historical meteorology news headlines to extreme-event records.

This script uses a curated headline catalog (from newspaper/news archive scans)
and matches each event by:
- time proximity
- hazard compatibility (flood/heatwave/drought/hail/etc.)
- source reliability prior
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HEADLINE_CATALOG: list[dict[str, Any]] = [
    {
        "headline_date": "2009-09-09",
        "headline": "Istanbul'da sel felaketi: 31 olu",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/istanbulda-sel-felaketi-31-olu-12441642",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-09",
        "headline": "Istanbul'da sel felaketi",
        "source": "Hurriyet Bigpara",
        "url": "https://bigpara.hurriyet.com.tr/haberler/genel-haberler/istanbul-da-sel-felaketi_ID689305/",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-13",
        "headline": "Sel, 3 kilometrede 300 milyon TL'yi goturdu",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/sel-3-kilometrede-300-milyon-tlyi-goturdu-12460670",
        "hazard_type": "flood_damage",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-10",
        "headline": "Selin maliyeti simdilik 150 milyon euro",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/selin-maliyeti-simdilik-150-milyon-euro-12451976",
        "hazard_type": "flood_damage",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-14",
        "headline": "Seldeki hasar 150 milyon dolari gececek",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/seldeki-hasar-150-milyon-dolari-gececek-12470447",
        "hazard_type": "flood_damage",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-07-20",
        "headline": "Savsat'ta sel: 5 kurban",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/savsatta-sel-5-kurban-12107044",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2012-07-04",
        "headline": "Samsun'da sel felaketi: 9 olu",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/samsunda-sel-felaketi-9-olu-20920568",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-09",
        "headline": "Felakette olu sayisi 31'e yukseldi",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/felakette-olu-sayisi-31e-yukseldi-81713",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-10",
        "headline": "Topbas kotu haber verdi",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/topbas-kotu-haber-verdi-82490",
        "hazard_type": "flood_warning",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-12",
        "headline": "Istanbul yagmura teslim oldu",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/istanbul-yagmura-teslim-oldu-83588",
        "hazard_type": "heavy_rain",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-11",
        "headline": "Istanbul afet bolgesi ilan edilsin",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/istanbul-afet-bolgesi-ilan-edilsin-83126",
        "hazard_type": "flood_damage",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-12",
        "headline": "Istanbul sel alarminda",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/istanbul-sel-alarminda-83431",
        "hazard_type": "flood_warning",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-09-14",
        "headline": "Sel felaketi Meclis gundeminde",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/sel-felaketi-meclis-gundeminde-84425",
        "hazard_type": "flood_damage",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2017-07-27",
        "headline": "Istanbul'da siddetli dolu yagisi",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/istanbulda-siddetli-dolu-yagisi-789787",
        "hazard_type": "hail",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2017-09-21",
        "headline": "Aksam saatlerine dikkat! Istanbul'da dolu yagisi olabilir",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/aksam-saatlerine-dikkat-istanbulda-dolu-yagisi-olabilir-830134",
        "hazard_type": "hail_warning",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2017-08-16",
        "headline": "Istanbul'da otoparklara dolu zammi",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/istanbulda-otoparklara-dolu-zammi-805102",
        "hazard_type": "hail_damage",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2014-08-28",
        "headline": "Istanbul icin geriye sayim basladi",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/istanbul-icin-geriye-sayim-basladi-113404",
        "hazard_type": "drought",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2014-01-22",
        "headline": "DSI'den korkutan aciklama: 110 gunluk su kaldi",
        "source": "Cumhuriyet",
        "url": "https://www.cumhuriyet.com.tr/haber/dsiden-korkutan-aciklama-110-gunluk-su-kaldi-27606",
        "hazard_type": "drought",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2017-07-27",
        "headline": "Heavy hailstorm and rainfall hit Istanbul",
        "source": "Daily Sabah",
        "url": "https://www.dailysabah.com/turkey/2017/07/27/heavy-hailstorm-and-rainfall-hit-istanbul",
        "hazard_type": "hail",
        "scope": "istanbul",
        "language": "en",
    },
    {
        "headline_date": "2010-02-18",
        "headline": "Istanbul'da 2 ay sonra su sikintisi olacak",
        "source": "Milliyet",
        "url": "https://www.milliyet.com.tr/gundem/istanbul-da-2-ay-sonra-2010-da-su-sikintisi-olacak-1201854",
        "hazard_type": "drought",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2010-08-31",
        "headline": "Saat 17'de yagis geliyor",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/saat-17de-yagis-geliyor-15685330",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2002-04-05",
        "headline": "Sel alarmi",
        "source": "Milliyet",
        "url": "https://www.milliyet.com.tr/gundem/sel-alarmi-5228371",
        "hazard_type": "flood_warning",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2002-07-25",
        "headline": "Death toll of Turkish floods hits 30",
        "source": "The Irish Times",
        "url": "https://www.irishtimes.com/news/death-toll-of-turkish-floods-hits-30-1.430985",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "en",
    },
    {
        "headline_date": "2002-07-26",
        "headline": "Turkish flood death toll reaches 27",
        "source": "UPI",
        "url": "https://www.upi.com/Defense-News/2002/07/26/Turkish-flood-death-toll-reaches-27/51101027688569/",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "en",
    },
    {
        "headline_date": "2003-09-03",
        "headline": "Selden hasar buyuk",
        "source": "Milliyet",
        "url": "https://www.milliyet.com.tr/gundem/selden-hasar-buyuk-5094752",
        "hazard_type": "flood_damage",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2005-07-08",
        "headline": "Sel ilk kurbanini aldi",
        "source": "Milliyet",
        "url": "https://www.milliyet.com.tr/gundem/sel-ilk-kurbanini-aldi-5085657",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-07-15",
        "headline": "Savsat'ta sel felaketi: 2 olu 4 kayip",
        "source": "Timeturk",
        "url": "https://www.timeturk.com/tr/2009/07/15/savsat-ta-sel-felaketi-2-olu-4-kayip.html",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2009-07-16",
        "headline": "Karadeniz'i sel vurdu",
        "source": "Timeturk",
        "url": "https://www.timeturk.com/tr/2009/07/16/karadeniz-i-sel-vurdu.html",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2014-11-07",
        "headline": "Buyuk tehlike! Turkiye'yi kuraklik vuracak",
        "source": "Sabah",
        "url": "https://www.sabah.com.tr/haberler/2014/11/07/buyuk-tehlike",
        "hazard_type": "drought",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2014-08-31",
        "headline": "Buyuksehirlerde su tukeniyor",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/ekonomi/buyuksehirlerde-su-tukeniyor-27119433",
        "hazard_type": "drought",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2010-02-13",
        "headline": "Istanbul'da saganak yagis",
        "source": "Timeturk",
        "url": "https://www.timeturk.com/tr/2010/02/13/istanbul-da-saganak-yagis.html",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2010-03-04",
        "headline": "Yagmur ve kar geliyor-HARITALI",
        "source": "Takvim",
        "url": "https://www.takvim.com.tr/guncel/2010/03/04/kuvvetli_yagislara_dikkat",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2010-04-21",
        "headline": "Turkiye genelinde yagis bekleniyor",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/turkiye-genelinde-yagis-bekleniyor-14489316",
        "hazard_type": "heavy_rain",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2010-06-06",
        "headline": "Yagmur geliyor, sicakliklar dusecek!",
        "source": "Timeturk",
        "url": "https://www.timeturk.com/tr/2010/06/06/yagmur-geliyor-sicakliklar-dusecek.html",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2010-06-06",
        "headline": "Dikkat: Yagmur geliyor",
        "source": "Takvim",
        "url": "https://www.takvim.com.tr/aktuel/2010/06/06/dikkat_yagmur_geliyor",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2010-10-01",
        "headline": "Meteoroloji'den kuvvetli yagis uyarisi",
        "source": "Timeturk",
        "url": "https://www.timeturk.com/tr/2010/10/01/meteoroloji-den-kuvvetli-yagis-uyarisi.html",
        "hazard_type": "heavy_rain",
        "scope": "turkiye",
        "language": "tr",
    },
    {
        "headline_date": "2018-07-24",
        "headline": "Istanbul'da saganak yagis",
        "source": "AHaber",
        "url": "https://www.ahaber.com.tr/galeri/yasam/istanbulda-saganak-yagis-24-temmuz-hava-durumu",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2018-07-24",
        "headline": "Istanbul icin meteorolojiden yagis uyarisi",
        "source": "TRT Haber",
        "url": "https://www.trthaber.com/haber/turkiye/istanbul-icin-meteorolojiden-yagis-uyarisi-371824.html",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2019-03-14",
        "headline": "Meteoroloji'den saganak yagis uyarisi! Istanbul'da bugun hava nasil olacak? 15 Mart 2019 hava durumu",
        "source": "Takvim",
        "url": "https://www.takvim.com.tr/yasam/2019/03/14/meteorolojiden-saganak-yagis-uyarisi-istanbulda-bugun-hava-nasil-olacak-15-mart-2019-hava-durumu",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2019-03-15",
        "headline": "Meteoroloji'den son dakika hava durumu uyarisi! Bugun Istanbul'da hava nasil olacak? 15 Mart 2019 hava durumu",
        "source": "AHaber",
        "url": "https://www.ahaber.com.tr/yasam/2019/03/15/meteorolojiden-son-dakika-hava-durumu-uyarisi-bugun-istanbulda-hava-nasil-olacak-15-mart-2019-hava-durumu",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2019-06-09",
        "headline": "Istanbul'da gok gurultulu saganak alarmi",
        "source": "Hurriyet",
        "url": "https://www.hurriyet.com.tr/gundem/istanbulda-gok-gurultulu-saganak-alarmi-41261187",
        "hazard_type": "heavy_rain",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2019-08-18",
        "headline": "Four killed as floods and landslides hit northwest Turkey",
        "source": "Euronews",
        "url": "https://www.euronews.com/2019/08/18/four-killed-as-floods-and-landslides-hit-northwest-turkey",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "en",
    },
    {
        "headline_date": "2019-11-15",
        "headline": "Istanbul icin kuraklik tehlikesi artiyor",
        "source": "Anadolu Agency",
        "url": "https://www.aa.com.tr/tr/turkiye/istanbul-icin-kuraklik-tehlikesi-artiyor/1646261",
        "hazard_type": "drought",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2019-12-03",
        "headline": "Istanbul'un su kaynaklari son 10 yilin en dusuk seviyesinde",
        "source": "TRT Haber",
        "url": "https://www.trthaber.com/haber/turkiye/istanbulun-su-kaynaklari-son-10-yilin-en-dusuk-seviyesinde-444325.html",
        "hazard_type": "drought",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2019-07-17",
        "headline": "Istanbul'da sel: alti kisi hayatini kaybetti",
        "source": "Bianet",
        "url": "https://bianet.org/haber/istanbul-da-sel-alti-kisi-hayatini-kaybetti-212226",
        "hazard_type": "flood",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2019-07-18",
        "headline": "Istanbul'da sel felaketi: en az bes olu",
        "source": "VOA Turkce",
        "url": "https://www.voaturkce.com/a/istanbulda-sel-felaketi-en-az-bes-olu/4996527.html",
        "hazard_type": "flood",
        "scope": "istanbul",
        "language": "tr",
    },
    {
        "headline_date": "2021-08-14",
        "headline": "Turkey combats Black Sea floods, death toll rises",
        "source": "Reuters",
        "url": "https://www.reuters.com/world/middle-east/turkey-combats-black-sea-floods-death-toll-rises-27-2021-08-14/",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "en",
    },
    {
        "headline_date": "2021-08-15",
        "headline": "At least 44 killed in Turkey flood as search continues",
        "source": "Reuters",
        "url": "https://www.reuters.com/world/middle-east/least-44-killed-turkey-flood-search-missing-continues-2021-08-15/",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "en",
    },
    {
        "headline_date": "2021-08-16",
        "headline": "Death toll from northern Turkey floods rises to 62",
        "source": "Euronews (Reuters)",
        "url": "https://www.euronews.com/2021/08/16/death-toll-from-northern-turkey-floods-rises-to-62",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "en",
    },
    {
        "headline_date": "2021-08-14",
        "headline": "Death toll from floods in Turkey's Black Sea region rises to 57",
        "source": "Anadolu Agency",
        "url": "https://www.aa.com.tr/en/turkiye/death-toll-from-floods-in-turkeys-black-sea-region-rises-to-57/2334639",
        "hazard_type": "flood",
        "scope": "turkiye",
        "language": "en",
    },
    {
        "headline_date": "2022-08-18",
        "headline": "Climate change led increase in Turkey's 2021 floods, study says",
        "source": "Reuters",
        "url": "https://www.reuters.com/business/cop/climate-change-led-increase-turkeys-2021-floods-study-2022-08-18/",
        "hazard_type": "attribution",
        "scope": "turkiye",
        "language": "en",
    },
]


SOURCE_PRIOR = {
    "Reuters": 1.00,
    "Anadolu Agency": 0.96,
    "Euronews (Reuters)": 0.92,
    "The Irish Times": 0.94,
    "UPI": 0.90,
    "Hurriyet": 0.88,
    "Hurriyet Bigpara": 0.84,
    "Cumhuriyet": 0.86,
    "Daily Sabah": 0.82,
    "Milliyet": 0.86,
    "Sabah": 0.84,
    "Takvim": 0.82,
    "Timeturk": 0.80,
    "TRT Haber": 0.90,
    "AHaber": 0.78,
    "Euronews": 0.88,
    "Bianet": 0.80,
    "VOA Turkce": 0.88,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich extreme-event table with historical weather-news headlines.")
    p.add_argument(
        "--events-csv",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/tum_asiri_olaylar_bilimsel_filtreli.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yasinkaya/Hackhaton/output/extreme_events/news"),
    )
    p.add_argument("--max-day-diff", type=int, default=90)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-match-score", type=float, default=0.50)
    return p.parse_args()


def hazard_preferences(variable: str, direction: str) -> list[str]:
    var = str(variable).strip().lower()
    d = str(direction).strip().lower()
    if var == "precip":
        if d == "dusuk":
            return ["drought", "water_shortage", "dry_spell"]
        return ["flood", "heavy_rain", "hail", "storm", "flood_warning", "flood_damage"]
    if var == "humidity":
        if d == "dusuk":
            return ["drought", "heatwave", "water_shortage"]
        return ["heavy_rain", "flood", "hail", "storm"]
    if var == "temp":
        if d == "dusuk":
            return ["coldwave", "snowstorm", "frost"]
        return ["heatwave", "drought", "wildfire"]
    if var == "pressure":
        return ["storm", "cyclone", "flood", "heavy_rain"]
    return ["flood", "storm", "heatwave", "drought"]


def hazard_compatibility(hazard_type: str, preferred: list[str]) -> float:
    hz = str(hazard_type).strip().lower()
    if hz in preferred:
        return 1.0
    # Partial overlap by token.
    for p in preferred:
        if p in hz or hz in p:
            return 0.8
    if hz in {"flood_damage", "hail_damage", "hail_warning", "flood_warning"} and any(
        k in preferred for k in ["flood", "hail", "heavy_rain", "storm"]
    ):
        return 0.75
    if hz == "attribution":
        return 0.55
    return 0.20


def time_proximity_score(day_diff: int) -> float:
    # Exponential decay: 0d->1.0, 30d->~0.55, 90d->~0.22
    return float(math.exp(-abs(day_diff) / 50.0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_csv).copy()
    events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce")
    events["end_time"] = pd.to_datetime(events["end_time"], errors="coerce")
    events = events.dropna(subset=["start_time", "end_time"]).copy()
    events["center_time"] = events["start_time"] + (events["end_time"] - events["start_time"]) / 2

    catalog = pd.DataFrame(HEADLINE_CATALOG).copy()
    catalog["headline_date"] = pd.to_datetime(catalog["headline_date"], errors="coerce")
    catalog["source_prior"] = catalog["source"].map(SOURCE_PRIOR).fillna(0.75)
    catalog = catalog.dropna(subset=["headline_date"]).copy()
    catalog = catalog.sort_values("headline_date").reset_index(drop=True)

    match_rows: list[dict[str, Any]] = []
    for _, ev in events.iterrows():
        pref = hazard_preferences(ev.get("variable", ""), ev.get("dominant_direction", ""))
        center = pd.Timestamp(ev["center_time"])
        tmp = catalog.copy()
        tmp["day_diff"] = (tmp["headline_date"] - center).dt.days
        tmp = tmp[tmp["day_diff"].abs() <= int(args.max_day_diff)].copy()
        if tmp.empty:
            continue
        tmp["hazard_score"] = tmp["hazard_type"].map(lambda hz: hazard_compatibility(hz, pref))
        tmp["time_score"] = tmp["day_diff"].map(time_proximity_score)
        tmp["total_match_score"] = (
            0.55 * tmp["hazard_score"] + 0.35 * tmp["time_score"] + 0.10 * tmp["source_prior"]
        )
        tmp = tmp[tmp["total_match_score"] >= float(args.min_match_score)].copy()
        if tmp.empty:
            continue
        tmp = tmp.sort_values("total_match_score", ascending=False).head(int(args.top_k))

        for rank, (_, r) in enumerate(tmp.iterrows(), start=1):
            match_rows.append(
                {
                    "event_id": ev.get("event_id"),
                    "variable": ev.get("variable"),
                    "event_start": ev.get("start_time"),
                    "event_end": ev.get("end_time"),
                    "event_center": center,
                    "event_peak_severity_score": ev.get("peak_severity_score"),
                    "event_scientific_tier": ev.get("scientific_tier"),
                    "event_direction": ev.get("dominant_direction"),
                    "match_rank": rank,
                    "headline_date": r["headline_date"],
                    "headline": r["headline"],
                    "source": r["source"],
                    "url": r["url"],
                    "hazard_type": r["hazard_type"],
                    "scope": r["scope"],
                    "day_diff": int(r["day_diff"]),
                    "hazard_score": float(r["hazard_score"]),
                    "time_score": float(r["time_score"]),
                    "source_prior": float(r["source_prior"]),
                    "total_match_score": float(r["total_match_score"]),
                }
            )

    matches = pd.DataFrame(match_rows)
    if matches.empty:
        raise SystemExit("No headline matches produced. Increase --max-day-diff or broaden catalog.")

    matches = matches.sort_values(["event_id", "match_rank"]).reset_index(drop=True)
    top1 = matches[matches["match_rank"] == 1].copy()
    top1 = top1.rename(
        columns={
            "headline_date": "top_headline_date",
            "headline": "top_headline",
            "source": "top_headline_source",
            "url": "top_headline_url",
            "hazard_type": "top_headline_hazard",
            "day_diff": "top_headline_day_diff",
            "total_match_score": "top_headline_match_score",
        }
    )
    top1 = top1[
        [
            "event_id",
            "top_headline_date",
            "top_headline",
            "top_headline_source",
            "top_headline_url",
            "top_headline_hazard",
            "top_headline_day_diff",
            "top_headline_match_score",
        ]
    ]
    enriched = events.merge(top1, on="event_id", how="left")

    # Summary tables.
    coverage = (
        enriched.groupby("variable", as_index=False)
        .agg(
            event_count=("event_id", "size"),
            with_headline=("top_headline", lambda s: int(s.notna().sum())),
            median_abs_day_diff=(
                "top_headline_day_diff",
                lambda s: float(np.nanmedian(np.abs(pd.to_numeric(s, errors="coerce"))))
                if s.notna().any()
                else np.nan,
            ),
            median_match_score=("top_headline_match_score", lambda s: float(np.nanmedian(s)) if s.notna().any() else np.nan),
        )
        .sort_values("event_count", ascending=False)
    )

    out_catalog = args.output_dir / "meteoroloji_haber_baslik_katalogu.csv"
    out_matches = args.output_dir / "tum_asiri_olaylar_haber_eslesmeleri.csv"
    out_enriched = args.output_dir / "tum_asiri_olaylar_haber_enriched.csv"
    out_coverage = args.output_dir / "tum_asiri_olaylar_haber_kapsam_ozet.csv"
    out_report = args.output_dir / "tum_asiri_olaylar_haber_raporu.md"

    catalog.to_csv(out_catalog, index=False)
    matches.to_csv(out_matches, index=False)
    enriched.to_csv(out_enriched, index=False)
    coverage.to_csv(out_coverage, index=False)

    top_show = matches.sort_values("total_match_score", ascending=False).head(40)
    lines = [
        "# Asiri Olaylar Icin Gazete/Haber Baslik Eslestirme Raporu",
        "",
        f"- Olay dosyasi: `{args.events_csv}`",
        f"- Katalog baslik sayisi: **{len(catalog)}**",
        f"- Eslesen olay sayisi (top-1): **{int(enriched['top_headline'].notna().sum())}/{len(enriched)}**",
        f"- Eslesme kurali: |gun_farki| <= {int(args.max_day_diff)} ve skor >= {float(args.min_match_score):.2f}",
        "",
        "## Kaynak Turu",
        "- Gazete/haber portali basliklari (Hurriyet, Cumhuriyet, Daily Sabah, Reuters, AA, Euronews).",
        "- Bu katalog manuel web taramasindan olusturuldu; script otomatik eslestirme yapar.",
        "",
        "## Degisken Kapsam Ozeti",
    ]
    for _, r in coverage.iterrows():
        lines.append(
            f"- {r['variable']}: olay={int(r['event_count'])}, eslesme={int(r['with_headline'])}, "
            f"medyan_gun_farki={r['median_abs_day_diff']:.1f}, medyan_match_skoru={r['median_match_score']:.3f}"
        )
    lines += [
        "",
        "## En Yuksek Skorlu 40 Eslesme",
        "",
        "|Event|Degisken|Siddet|Merkez Tarih|Baslik Tarihi|Gun Farki|Kaynak|Baslik|Skor|",
        "|---|---|---:|---|---|---:|---|---|---:|",
    ]
    for _, r in top_show.iterrows():
        lines.append(
            f"|{r['event_id']}|{r['variable']}|{float(r['event_peak_severity_score']):.2f}|"
            f"{pd.Timestamp(r['event_center']).date()}|{pd.Timestamp(r['headline_date']).date()}|"
            f"{int(r['day_diff'])}|{r['source']}|{r['headline']}|{float(r['total_match_score']):.3f}|"
        )
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_catalog}")
    print(f"Wrote: {out_matches}")
    print(f"Wrote: {out_enriched}")
    print(f"Wrote: {out_coverage}")
    print(f"Wrote: {out_report}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
