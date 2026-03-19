#!/usr/bin/env bash
set -euo pipefail

cd /Users/yasinkaya/Hackhaton

INPUT_DEFAULT="/Users/yasinkaya/Hackhaton/output/model_suite_retrain_newdata_20260306_234718/prepare_calibrated/observations_calibrated_full.parquet"
INPUT_PATH="${1:-$INPUT_DEFAULT}"
TARGET_YEAR="${2:-2035}"
ANALYSIS_MODE="${3:-anomalies_only}"
NEWS_CATALOG_DEFAULT="/Users/yasinkaya/Hackhaton/output/extreme_events/news_expanded_v3_relaxed/meteoroloji_haber_baslik_katalogu.csv"
NEWS_CATALOG_PATH="${4:-$NEWS_CATALOG_DEFAULT}"
NEWS_WINDOW_DAYS="${5:-75}"
HISTORY_START="${6:-1900-01-01}"
HISTORY_END="${7:-2020-12-31}"
DENSE_WINDOW_MODE="${8:-auto}"
RUN_EVENT_PIPELINE="${9:-true}"
DAILY_CSV_DEFAULT="/Users/yasinkaya/Hackhaton/output/spreadsheet/es_ea_newdata_daily.csv"
DAILY_CSV_PATH="${10:-$DAILY_CSV_DEFAULT}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/Users/yasinkaya/Hackhaton/output/yeni_model_newdata_${RUN_TAG}"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input bulunamadi: $INPUT_PATH" >&2
  exit 1
fi

echo "[YENI MODEL] Input: $INPUT_PATH"
echo "[YENI MODEL] Output: $OUT_DIR"
echo "[YENI MODEL] History window: $HISTORY_START -> $HISTORY_END"
echo "[YENI MODEL] Dense window mode: $DENSE_WINDOW_MODE"
echo "[YENI MODEL] Extreme-event pipeline: $RUN_EVENT_PIPELINE"
if [[ -f "$NEWS_CATALOG_PATH" ]]; then
  echo "[YENI MODEL] News catalog: $NEWS_CATALOG_PATH"
  NEWS_ARGS=(--news-catalog "$NEWS_CATALOG_PATH" --news-window-days "$NEWS_WINDOW_DAYS")
else
  echo "[YENI MODEL] News catalog bulunamadi: $NEWS_CATALOG_PATH (auto-fallback denenecek)"
  NEWS_ARGS=(--news-window-days "$NEWS_WINDOW_DAYS")
fi

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python3 /Users/yasinkaya/Hackhaton/scripts/yeni_model_newdata/quant_regime_projection_yeni_model.py \
  --observations "$INPUT_PATH" \
  --output-dir "$OUT_DIR" \
  --variables "temp,humidity,pressure,precip" \
  --target-year "$TARGET_YEAR" \
  --history-start "$HISTORY_START" \
  --history-end "$HISTORY_END" \
  --dense-window-mode "$DENSE_WINDOW_MODE" \
  --analysis-mode "$ANALYSIS_MODE" \
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
  --anomaly-top 0 \
  "${NEWS_ARGS[@]}"

RUN_EVENT_PIPELINE_NORMALIZED="$(printf '%s' "$RUN_EVENT_PIPELINE" | tr '[:upper:]' '[:lower:]')"
if [[ "$RUN_EVENT_PIPELINE_NORMALIZED" == "true" ]]; then
  if [[ ! -f "$DAILY_CSV_PATH" ]]; then
    echo "Daily CSV bulunamadi: $DAILY_CSV_PATH" >&2
    exit 1
  fi
  echo "[YENI MODEL] Extreme-event pipeline basliyor"
  python3 /Users/yasinkaya/Hackhaton/scripts/run_quant_extreme_event_pipeline.py \
    --observations "$INPUT_PATH" \
    --quant-output-dir "$OUT_DIR" \
    --daily-csv "$DAILY_CSV_PATH" \
    --output-dir "$OUT_DIR/extreme_events"
fi

cat > "$OUT_DIR/YENI_MODEL.txt" <<EOF
YENI MODEL
run_tag=$RUN_TAG
input=$INPUT_PATH
target_year=$TARGET_YEAR
analysis_mode=$ANALYSIS_MODE
news_catalog=$NEWS_CATALOG_PATH
news_window_days=$NEWS_WINDOW_DAYS
history_start=$HISTORY_START
history_end=$HISTORY_END
dense_window_mode=$DENSE_WINDOW_MODE
run_event_pipeline=$RUN_EVENT_PIPELINE
daily_csv=$DAILY_CSV_PATH
extreme_events_dir=$OUT_DIR/extreme_events
code=/Users/yasinkaya/Hackhaton/scripts/yeni_model_newdata/quant_regime_projection_yeni_model.py
runner=/Users/yasinkaya/Hackhaton/scripts/yeni_model_newdata/run_yeni_model_newdata.sh
EOF

ln -sfn "$OUT_DIR" /Users/yasinkaya/Hackhaton/output/yeni_model_newdata_latest

echo "[YENI MODEL] Bitti: $OUT_DIR"
