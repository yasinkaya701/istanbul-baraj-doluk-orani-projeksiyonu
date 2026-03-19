# Research Synthesis Round 2

Date: 2026-03-11

Scope of this round:

- external literature expansion
- official ISKI policy and tariff context
- actinograph and radiation measurement grounding
- model feature implications

## 1. ET0 and radiation side became sharper

Primary findings:

- WMO No. 8 explicitly separates:
  `radiation`, `sunshine duration`, and `evaporation` measurement chapters.
- ASCE standardized ET provides a second formal reference for ET quality control,
  wind adjustment, vapor pressure handling, and radiation treatment.
- Hobbins (2016) shows ET variability is not explained by temperature alone;
  depending on season and place, humidity, wind, and downward shortwave radiation
  can dominate.

Implication for our project:

- We should not defend a temperature-only or Tmax-Tmin-only ET story in front of experts.
- The actinograph is a first-order improvement, not a cosmetic add-on.
- If actinograph data arrives, ET0 should be recomputed and compared against the
  current proxy-radiation ET0 before reuse in occupancy modeling.

## 2. Open-water evaporation must stay distinct from ET0

Primary findings:

- USGS open-water evaporation review (2012) and reservoir evaporation studies
  emphasize that calibrated energy-budget or mass-transfer approaches are more
  reliable than simple pan-based transfer.
- Reservoir surface area matters.
- Calibration matters.

Implication for our project:

- ET0 can be used as an atmospheric demand proxy, but not as a direct one-to-one
  reservoir evaporation estimate.
- Medium-term roadmap must include:
  water-surface area proxy,
  level-area relation,
  or a calibrated ET0 -> open-water-loss conversion.

## 3. Inflow and level forecasting literature supports a hybrid path

Primary findings:

- Review work in Environmental Modelling and Software (2023) shows most lake and
  reservoir level studies still rely on past level, precipitation, and temperature.
- Journal of Hydrology 2024 and EJRH 2023 support hybrid and explainable inflow
  forecasting rather than black-box-only forecasting.

Implication for our project:

- Our current structure is directionally correct:
  memory + climate + demand + scenario.
- The next technical lift should be a decomposed model:
  `storage memory`,
  `hydro-climate forcing`,
  `human demand`,
  `loss/operations`.

## 4. Official ISKI material directly helps demand modeling

Primary findings:

- ISKI tariff page separates categories such as:
  housing,
  workplace,
  organized industry,
  raw water,
  green-area irrigation,
  recycled water.
- ISKI tariff regulation states tariff logic is based on produced water minus
  physical water losses and expected measurable sales.
- Older official standard water balance forms provide a formal split between:
  billed use,
  non-revenue water,
  physical losses,
  meter error,
  storage losses.

Implication for our project:

- Aggregate consumption should eventually be decomposed into at least:
  household,
  business/industry,
  irrigation-green space,
  raw water,
  recycled water,
  physical losses/non-revenue water.
- We now have an official conceptual basis to defend this split.

## 5. Buyer-facing implication became stronger

Official ISKI 2024 financial report shows the scale of operating and capital
spending is large enough that even modest improvements in forecasting and
intervention timing can be presented as economically relevant.

Implication for product framing:

- This is not just an academic climate dashboard.
- It can be positioned as a planning and intervention support layer for a utility-scale operator.

## 6. Priority feature backlog after this round

Priority A:

- observed radiation ingestion from actinograph
- physical water-loss block
- sectoral demand segmentation

Priority B:

- restriction event calendar
- source-to-treatment-to-demand linkage map
- level-area proxy for open-water evaporation

Priority C:

- inflow surrogate or rainfall-runoff decomposition
- dynamic scenario engine by season and tariff regime

## 7. Source links used in this round

- WMO No. 8 2024 edition:
  https://wmo.int/guide-instruments-and-methods-of-observation-wmo-no-8-0
- ASCE standardized ET:
  https://epic.awi.de/id/eprint/42362/
- Hobbins 2016:
  https://doi.org/10.13031/trans.59.10975
- ISKI tariffs:
  https://iski.istanbul/abone-hizmetleri/abone-rehberi/su-birim-fiyatlari/
- ISKI tariff regulation:
  https://cdn.iski.istanbul/uploads/ISKI_ABONE_HIZMETLERI_TARIFE_VE_UYGULAMA_Y_Oe_NETMELIGI_2024_150ea718d2.pdf
- ISKI 2024 financial report:
  https://cdn.iski.istanbul/uploads/2024_MDB_Raporu_60ca356312.pdf
- ISKI standard water balance form:
  https://cdn.iski.istanbul/uploads/Icmesuyu_Temini_ve_Dagitim_Sistemlerindeki_Su_Kayiplari_Yillik_Raporlari_2016_f790d8c849.pdf
- Systematic review:
  https://doi.org/10.1016/j.envsoft.2023.105684
- Hybrid inflow model:
  https://doi.org/10.1016/j.jhydrol.2024.130623
- Explainable inflow forecasting:
  https://doi.org/10.1016/j.ejrh.2023.101584
- USGS open-water evaporation review:
  https://doi.org/10.3133/sir20125202
- Denver reservoir evaporation:
  https://doi.org/10.3133/wri76114
