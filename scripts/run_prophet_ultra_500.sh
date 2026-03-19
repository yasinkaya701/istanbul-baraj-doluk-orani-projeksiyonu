#!/usr/bin/env bash
set -euo pipefail

cd /Users/yasinkaya/Hackhaton

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/prophet_ultra_500.py \
  --observations output/forecast_package/observations_with_graph.parquet \
  --output-dir output/prophet_ultra_500 \
  --target-year 2035 \
  --fast-grid true \
  --backtest-splits 3 \
  --holdout-steps 12 \
  --min-train-steps 36 \
  --max-ensemble-models 2 \
  --bias-weight 0.1 \
  --stability-weight 0.2
