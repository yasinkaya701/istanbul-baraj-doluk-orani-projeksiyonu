#!/usr/bin/env bash
set -euo pipefail

cd /Users/yasinkaya/Hackhaton

echo "[1/3] Numeric tablolari ingest et"
python3 scripts/ingest_numeric_and_plot.py \
  --data-dir "/Users/yasinkaya/Hackhaton/DATA/Sayısallaştırılmış Veri" \
  --output-dir "/Users/yasinkaya/Hackhaton/output/sample"

echo "[2/3] Tum TIFF gorsellerini sayisallastir + quant input olustur"
python3 scripts/process_all_visuals_to_quant.py \
  --graph-root "/Users/yasinkaya/Hackhaton/DATA/Graf Kağıtları Tarama " \
  --numeric-parquet "/Users/yasinkaya/Hackhaton/output/sample/observations_numeric.parquet" \
  --output-dir "/Users/yasinkaya/Hackhaton/output/quant_all_visuals_input"

echo "[3/3] Quant modeli calistir"
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/quant_regime_projection.py \
  --observations "/Users/yasinkaya/Hackhaton/output/quant_all_visuals_input/observations_with_all_visuals_for_quant.parquet" \
  --output-dir "/Users/yasinkaya/Hackhaton/output/quant_all_visuals_package" \
  --variables "temp,humidity,pressure,precip" \
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

echo "Bitti: output/quant_all_visuals_package"
