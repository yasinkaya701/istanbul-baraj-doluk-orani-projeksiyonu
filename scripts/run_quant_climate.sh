#!/usr/bin/env bash
set -euo pipefail

cd /Users/yasinkaya/Hackhaton

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/quant_regime_projection.py \
  --observations output/forecast_package/observations_with_graph.parquet \
  --output-dir output/quant_climate_package \
  --target-year 2035 \
  --backtest-splits 3 \
  --holdout-steps 12 \
  --min-train-steps 36 \
  --vol-model egarch \
  --egarch-p 1 \
  --egarch-o 1 \
  --egarch-q 1 \
  --egarch-dist t \
  --regime-k 2 \
  --regime-maxiter 200 \
  --interval-alpha 0.10 \
  --anomaly-z 2.5 \
  --anomaly-top 20
