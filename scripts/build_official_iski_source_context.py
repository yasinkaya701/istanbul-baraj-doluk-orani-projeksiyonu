#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT_DIR = Path("/Users/yasinkaya/Hackhaton/output/newdata_feature_store")


SOURCE_ROWS = [
    {
        "source_name": "Ömerli Barajı",
        "source_group": "baraj",
        "annual_yield_million_m3": 220.0,
        "max_storage_million_m3": 235.371,
        "commissioning_year": "1972",
        "basin_area_km2": 621.0,
        "normal_lake_area_km2": None,
        "note": "Also supports transfer to Avrupa Yakası via Salacak-Sarayburnu system.",
    },
    {
        "source_name": "Darlık Barajı",
        "source_group": "baraj",
        "annual_yield_million_m3": 97.0,
        "max_storage_million_m3": 107.500,
        "commissioning_year": "1989",
        "basin_area_km2": 207.0,
        "normal_lake_area_km2": None,
        "note": "Şile source feeding Emirli treatment system.",
    },
    {
        "source_name": "Elmalı 1 ve 2 Barajları",
        "source_group": "baraj",
        "annual_yield_million_m3": 15.0,
        "max_storage_million_m3": 9.600,
        "commissioning_year": "1893-1950",
        "basin_area_km2": 81.0,
        "normal_lake_area_km2": None,
        "note": "Historic urban source on Anadolu Yakası.",
    },
    {
        "source_name": "Terkos Barajı",
        "source_group": "baraj",
        "annual_yield_million_m3": 142.0,
        "max_storage_million_m3": 162.241,
        "commissioning_year": "1883",
        "basin_area_km2": 619.0,
        "normal_lake_area_km2": None,
        "note": "Historic Terkos system source.",
    },
    {
        "source_name": "Alibey Barajı",
        "source_group": "baraj",
        "annual_yield_million_m3": 36.0,
        "max_storage_million_m3": 34.143,
        "commissioning_year": "1972",
        "basin_area_km2": 160.0,
        "normal_lake_area_km2": None,
        "note": "Built on Alibey Deresi.",
    },
    {
        "source_name": "Büyükçekmece Barajı",
        "source_group": "baraj",
        "annual_yield_million_m3": 100.0,
        "max_storage_million_m3": 148.943,
        "commissioning_year": "1989",
        "basin_area_km2": 620.0,
        "normal_lake_area_km2": None,
        "note": "Lagoon-to-reservoir conversion; Avrupa Yakası source.",
    },
    {
        "source_name": "Sazlıdere Barajı",
        "source_group": "baraj",
        "annual_yield_million_m3": 55.0,
        "max_storage_million_m3": 88.730,
        "commissioning_year": "1998",
        "basin_area_km2": 165.0,
        "normal_lake_area_km2": 11.81,
        "note": "Official page reports normal water-level lake area.",
    },
    {
        "source_name": "Istrancalar",
        "source_group": "baraj_grubu",
        "annual_yield_million_m3": 75.0,
        "max_storage_million_m3": 6.231,
        "commissioning_year": "1995-1997",
        "basin_area_km2": None,
        "normal_lake_area_km2": None,
        "note": "Aggregate group of Düzdere, Kuzuludere, Büyükdere, Sultanbahçedere, Elmalıdere.",
    },
    {
        "source_name": "Düzdere Barajı",
        "source_group": "istranca_alt",
        "annual_yield_million_m3": 4.5,
        "max_storage_million_m3": None,
        "commissioning_year": "1995",
        "basin_area_km2": None,
        "normal_lake_area_km2": None,
        "note": "Official page gives annual yield but basin-area value is missing in visible text.",
    },
    {
        "source_name": "Kuzuludere Barajı",
        "source_group": "istranca_alt",
        "annual_yield_million_m3": 11.3,
        "max_storage_million_m3": None,
        "commissioning_year": "1995",
        "basin_area_km2": 34.0,
        "normal_lake_area_km2": None,
        "note": "Istranca sub-reservoir.",
    },
    {
        "source_name": "Büyükdere Barajı",
        "source_group": "istranca_alt",
        "annual_yield_million_m3": 28.4,
        "max_storage_million_m3": None,
        "commissioning_year": "1995",
        "basin_area_km2": 81.0,
        "normal_lake_area_km2": None,
        "note": "Istranca sub-reservoir.",
    },
    {
        "source_name": "Sultanbahçedere Barajı",
        "source_group": "istranca_alt",
        "annual_yield_million_m3": 19.4,
        "max_storage_million_m3": None,
        "commissioning_year": "1997",
        "basin_area_km2": 46.5,
        "normal_lake_area_km2": None,
        "note": "Istranca sub-reservoir.",
    },
    {
        "source_name": "Elmalıdere Barajı",
        "source_group": "istranca_alt",
        "annual_yield_million_m3": 11.6,
        "max_storage_million_m3": None,
        "commissioning_year": "1997",
        "basin_area_km2": 24.0,
        "normal_lake_area_km2": None,
        "note": "Istranca sub-reservoir feeding FSM treatment plant.",
    },
    {
        "source_name": "Kazandere Barajı",
        "source_group": "baraj",
        "annual_yield_million_m3": 100.0,
        "max_storage_million_m3": 17.424,
        "commissioning_year": "1997",
        "basin_area_km2": 313.0,
        "normal_lake_area_km2": None,
        "note": "Northwest source linked with Pabuçdere.",
    },
    {
        "source_name": "Pabuçdere Barajı",
        "source_group": "baraj",
        "annual_yield_million_m3": 60.0,
        "max_storage_million_m3": 58.500,
        "commissioning_year": "2000",
        "basin_area_km2": 178.5,
        "normal_lake_area_km2": None,
        "note": "Linked to Kazandere by tunnel.",
    },
    {
        "source_name": "Yeşilçay Regülatörü",
        "source_group": "regulator",
        "annual_yield_million_m3": 145.0,
        "max_storage_million_m3": None,
        "commissioning_year": "2004",
        "basin_area_km2": None,
        "normal_lake_area_km2": None,
        "note": "Transfer source into Emirli system.",
    },
    {
        "source_name": "Melen Regülatörü",
        "source_group": "regulator",
        "annual_yield_million_m3": 650.0,
        "max_storage_million_m3": None,
        "commissioning_year": "2007-2014-2023",
        "basin_area_km2": None,
        "normal_lake_area_km2": None,
        "note": "Multi-stage interbasin transfer source.",
    },
]


def main() -> None:
    out_tables = OUT_DIR / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(SOURCE_ROWS)
    df["yield_per_basin_mm"] = 1000.0 * df["annual_yield_million_m3"] / df["basin_area_km2"]
    df["storage_per_basin_mm"] = 1000.0 * df["max_storage_million_m3"] / df["basin_area_km2"]

    out_csv = out_tables / "official_iski_source_context.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "row_count": int(len(df)),
        "baraj_count": int((df["source_group"] == "baraj").sum()),
        "regulator_count": int((df["source_group"] == "regulator").sum()),
        "with_basin_area_count": int(df["basin_area_km2"].notna().sum()),
        "with_normal_lake_area_count": int(df["normal_lake_area_km2"].notna().sum()),
        "max_yield_source": df.sort_values("annual_yield_million_m3", ascending=False).iloc[0]["source_name"],
        "max_storage_source": df.dropna(subset=["max_storage_million_m3"]).sort_values("max_storage_million_m3", ascending=False).iloc[0]["source_name"],
        "notes": [
            "Values are compiled from the official ISKI Su Kaynaklari page.",
            "Basin-area fields are directly useful for rainfall-to-inflow proxy design.",
            "Normal lake-area information is only explicitly available for Sazlidere on the source page used here.",
        ],
    }
    (OUT_DIR / "official_iski_source_context_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(out_csv)
    print(OUT_DIR / "official_iski_source_context_summary.json")


if __name__ == "__main__":
    main()
