#!/usr/bin/env bash
set -euo pipefail

cd /Users/yasinkaya/Hackhaton

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/walkforward_retrain_multifreq.py \
  --observations output/forecast_package/observations_with_graph.parquet \
  --output-dir output/walkforward_retrain_package \
  --start-year 2026 \
  --target-year 2027 \
  --frequencies YS,MS,W,D \
  --vol-model egarch \
  --regime-k 2 \
  --regime-maxiter 200 \
  --interval-alpha 0.10 \
  --max-train-points 2000 \
  "$@"

