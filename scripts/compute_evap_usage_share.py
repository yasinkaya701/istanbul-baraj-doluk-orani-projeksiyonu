#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/yasinkaya/Hackhaton")

PANEL_PATH = ROOT / "output/newdata_feature_store/tables/istanbul_dam_driver_panel_2000_2026_extended.csv"
OUT_JSON = ROOT / "baraj_web/assets/data/evap_usage_baseline.json"

# Open-water surface areas used in ET0->evaporation conversion (km^2).
# Sazlıdere (11.81 km^2) is from İSKİ "Su Kaynakları" text (normal lake area).
# Other lake areas are project calibration constants because the same official
# source does not provide normal lake area for each reservoir.
AREA_KM2 = {
    "Terkos": 36.10,
    "Büyükçekmece": 24.17,
    "Sazlıdere": 11.81,
    "Alibey": 1.66,
    "Ömerli": 21.07,
    "Darlık": 5.93,
    "Elmalı": 0.91,
}

# FAO-56 suggests Kc ≈ 1.05 for open water (<2 m depth or subhumid climates)
K_OPEN_WATER = 1.05

# Annual water supply from reservoirs (barajlar) for 2023
# (secondary statement source that quotes İSKİ values)
BARAJ_SUPPLY_2023_M3 = 275_104_161
# Total city supply in 2023 from İBB Open Data (İSKİ publisher) monthly table sum
TOTAL_SUPPLY_2023_M3 = 1_117_064_116


def main():
    df = pd.read_csv(PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])

    year = 2023
    et0 = df[df["date"].dt.year == year].copy()
    if et0.empty:
        raise SystemExit(f"No ET0 data for {year} in panel")

    area_km2_total = sum(AREA_KM2.values())
    area_m2_total = area_km2_total * 1_000_000

    # ET0 monthly mm -> m, open-water evaporation via Kc
    et0["et0_m"] = et0["et0_mm_month"] / 1000.0
    et0["evap_m3"] = et0["et0_m"] * K_OPEN_WATER * area_m2_total
    evap_total_m3 = float(et0["evap_m3"].sum())

    usage_m3 = float(BARAJ_SUPPLY_2023_M3)
    total_loss = evap_total_m3 + usage_m3
    evap_share = evap_total_m3 / total_loss if total_loss else 0.0
    usage_share = usage_m3 / total_loss if total_loss else 0.0

    payload = {
        "year": year,
        "area_km2_total": area_km2_total,
        "area_breakdown_km2": AREA_KM2,
        "kc_open_water": K_OPEN_WATER,
        "evap_total_m3": evap_total_m3,
        "usage_baraj_m3": usage_m3,
        "total_supply_2023_m3": TOTAL_SUPPLY_2023_M3,
        "evap_share": evap_share,
        "usage_share": usage_share,
        "sources": {
            "lake_surface_area_sazlidere": "https://iski.istanbul/kurumsal/hakkimizda/su-kaynaklari",
            "total_supply_2023": "https://data.ibb.gov.tr/dataset/96fde959-3d0b-46d6-8b1d-78a7ba879fc6/resource/27bdb043-0051-49df-bd7c-b68f60f31247/download/istanbula-verilen-temiz-su-miktarlar-tr-en.xlsx",
            "total_supply_2023_api_snapshot": "https://iskiapi.iski.istanbul/api/iski/baraj/icmeSuyuAritma/sonOnyildaVerilenToplamSu/v2",
            "baraj_supply_2023_secondary": "https://www.aa.com.tr/tr/gundem/istanbulda-gecen-yil-1-milyar-117-milyon-64-bin-116-metrekup-su-kullanildi/3104905",
            "kc_open_water": "https://www.fao.org/4/X0490E/X0490E00.htm",
        },
        "notes": "Sazlıdere normal su kotu göl alanı İSKİ metninden alınmıştır. Diğer barajların göl yüzey alanları resmi sayfada tek tek verilmediği için kalibre edilmiş sabitler korunmuştur. Pabuçdere/Kazandere/Istrancalar için açık yüzey alanı bulunamadı; hesap ana İstanbul baraj alanlarıyla yapılmıştır. Kc=1.05 FAO-56 açık su katsayısı (sığ su / sub-humid varsayımı).",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
