# 2026-03-11 Source Reading Addendum - 3 Papers

## 1) Results in Engineering (2025)
Title: Estimating the volume of evaporation from the main dams of Iran
Link: https://www.sciencedirect.com/science/article/pii/S2590123025031135

Key points:
- 117 major reservoirs across Iran were assessed.
- Six empirical models and four satellite products were compared.
- Among empirical methods, Penman-Monteith was the best-performing classical approach in the paper's evaluation.
- Among satellite-based products, SEBAL and ERA5 had the strongest agreement with observations.
- MODIS systematically underestimated evaporation.
- Estimated annual evaporation volume across the studied reservoirs was about 2.2-2.8 billion m3/year.

Immediate relevance to our project:
- Supports keeping ET0 and open-water evaporation conceptually separate.
- Supports using reanalysis / satellite forcing as an operational evaporation layer when direct measurements are missing.
- Strengthens the case for adding a source-area-aware open-water evaporation proxy for reservoirs.

## 2) Sustainability (2024)
Title: Advanced Predictive Modeling for Dam Occupancy Using Historical and Meteorological Data
Link: https://www.mdpi.com/2071-1050/16/17/7696

Key points:
- Study is directly about Istanbul dams.
- Uses seven dams over the past five years.
- Recommends combining physical-model inputs (Penman-Monteith evapotranspiration, rainfall, solar radiation, consumption) with historical occupancy data.
- Tested LSTM, RF, Extra Trees, OMPCV, LLCV and related models.
- Extra Trees was reported as the most accurate method across all horizons in that study.
- Historical occupancy was strongly correlated with future occupancy; consumption and rainfall alone were not strongly correlated in simple correlation analysis.
- Weather and consumption features improved performance more clearly as forecast horizon increased.
- LSTM monthly MAPE was reported around 1%-3.5%; Extra Trees was reported around 0.3%-1.4% across intervals.

Immediate relevance to our project:
- Strong external support for our hybrid framing: history + physical drivers + consumption.
- Suggests Extra Trees should be a formal benchmark in our next model round.
- Suggests weather signal may matter more at longer horizons than in immediate short-horizon memory-dominated forecasts.

## 3) JRAS (2020)
Title: İstanbul Baraj Doluluk Oranlarının Zamansal İncelenmesi ve Çözüm Önerileri
Link: https://resatmsci.com/?mod=makale_tr_ozet&makale_id=49617

Key points:
- Daily Istanbul dam occupancy data since 2005 were analysed.
- The paper highlights 2007, 2008, 2014 and 2020 as dry years in the temporal analysis.
- It notes that recovery in reservoir occupancy could be delayed by up to two months.
- It also frames demand-side conservation actions as meaningful in drought periods.
- Future rainwater use and domestic/industrial water-use analysis are suggested.

Immediate relevance to our project:
- Useful local descriptive baseline for identifying dry years and lagged recovery.
- Supports event-style or demand-management variables in the model.
- Supports adding recovery-lag interpretation to scenario outputs.
