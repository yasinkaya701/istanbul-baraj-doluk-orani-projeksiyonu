# Official Policy And Model Implications

Date: 2026-03-11

This note translates official ISKI and ministry material into concrete model and
product implications for the Istanbul dam occupancy project.

## 1. ISKI strategy and product alignment

Official pages and performance programs show that ISKI already frames its work
around:

- integrated water management
- big data / analytic data management
- digital water and wastewater management
- strategic water-loss reduction

Evidence:

- 2021-2025 strategic plan page states the plan contains `7 strategic aims`,
  `28 strategic targets`, `128 activities`, and `97 performance indicators`.
- 2022 performance program explicitly uses the language:
  `Entegre Su Yönetimi`, `Büyük Veri/Analitik Veri Yönetimi`, and
  `7.3 Su Kayıplarını Azaltmak`.
- 2024 performance program includes activities such as:
  `Dijital Su ve Atık Su Yönetim Sisteminin Geliştirilmesi`,
  `Akıllı Su Yönetimi Uygulamaları`,
  `Su Kayıplarının Azaltılması`,
  and `Sınırlandırılmış İzole Alanların (DMA) Oluşturulması`.

Model implication:

- Our product should be framed as an operational decision-support layer that is
  directly compatible with ISKI's own strategic language, not as a detached
  academic dashboard.

## 2. Water-loss regulation creates concrete modeling targets

The official regulation states metropolitan utilities are required to reduce
water losses to:

- at most `30%` by `2023`
- at most `25%` by `2028`

The same regulation also requires annual reporting and internet publication of
water-loss reports for a period after submission.

Model implication:

- `physical losses / NRW` is not optional background context; it is an official
  operational target and should become an explicit feature block in the model.

## 3. Official performance documents define measurable loss variables

The 2017 ISKI performance program defines:

- `Su Kayıp Oranı`
- `Temiz Su İşletme Kayıpları Oranı - Depo ve İsale Hatları`

It also states these indicators use operational records such as:

- ISKI data warehouse records
- SCADA reports
- regional transmission directorate records

Model implication:

- We have an official basis to split demand-side withdrawal from system-side
  operational losses.
- In other words, low occupancy can come from:
  user demand,
  physical network losses,
  or operational discharge/maintenance losses.

## 4. DMA and pressure management are not theoretical extras

The ministry handbook shows:

- DMA creation allows separate loss calculation and strategy design per sub-zone
- pressure management reduces leakage and also reduces failure frequency and
  interruption frequency
- hydraulic models can be used to test operating pressure scenarios before
  implementation

Model implication:

- If neighborhood or district-level data becomes available later, the project can
  evolve from citywide forecasting into localized intervention planning.
- Even before that, pressure management gives a defensible causal link between
  water-loss control and outage reduction.

## 5. Water cuts should be modeled as operational events, not guesses

Official outage infrastructure exists:

- public outage page
- hidden official API endpoint discovered in page payload

Current constraint:

- direct endpoint access returns `403 Forbidden`

Model implication:

- event modeling remains valid, but the acquisition pipeline must be documented
  as a real data-access problem.
- no synthetic event history should be invented.

## 6. Reference implication for open-water evaporation

Official and technical sources support using ET0 as atmospheric demand, but they
do not justify treating ET0 as direct reservoir evaporation one-to-one.

Model implication:

- keep `ET0` and `reservoir evaporation` separate
- push `reservoir surface area / level-area proxy` into the P0 backlog

## 7. Immediate feature priorities after this official-source round

P0:

- observed radiation from actinograph
- physical water-loss / NRW block
- sectoral demand split
- reservoir surface area / open-water proxy

P1:

- event registry for cuts and interruptions
- source-to-treatment mapping
- rainfall-runoff / inflow surrogate

## Official sources used

- ISKI strategic plan page:
  https://iski.istanbul/kurumsal/stratejik-yonetim/stratejik-plan/
- ISKI 2022 performance program:
  https://cdn.iski.istanbul/uploads/2022_Performans_Programi_52134981ad.pdf
- ISKI 2024 performance program:
  https://cdn.iski.istanbul/uploads/2024_PERFORMANS_PROGRAMI_82da4531bf.pdf
- ISKI 2017 performance program:
  https://cdn.iski.istanbul/uploads/2017_performans_programi_8bc4215793.pdf
- Water Loss Control Handbook:
  https://www.tarimorman.gov.tr/SYGM/Belgeler/SU%20VER%C4%B0ML%C4%B0L%C4%B0%C4%9E%C4%B0/%C4%B0%C3%A7me%20Suyu%20Temin%20ve%20Da%C4%9F%C4%B1t%C4%B1m%20Sistemlerindeki%20Su%20Kay%C4%B1plar%C4%B1n%C4%B1n%20Kontrol%C3%BC%20El%20Kitab%C4%B1%20.pdf
- Water Loss Control Regulation:
  https://www.tarimorman.gov.tr/SYGM/Belgeler/%C4%B0%C3%A7me%20Suyu%20Temin%20Ve%20Da%C4%9F%C4%B1t%C4%B1m%2002.09.2019/i%C3%A7me%20suyu%20temin%20ve%20da%C4%9F%C4%B1t%C4%B1m%20sistemlerindeki%20su%20kay%C4%B1plar%C4%B1n%C4%B1n%20kontrol%C3%BC%20y%C3%B6netmeli%C4%9Fi.pdf
- ISKI water sources:
  https://iski.istanbul/kurumsal/hakkimizda/su-kaynaklari
- DSI operated dams page:
  https://bolge14.dsi.gov.tr/Sayfa/Detay/1188
