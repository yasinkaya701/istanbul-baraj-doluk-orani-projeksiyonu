#!/usr/bin/env bash
set -euo pipefail

BASE="/Users/yasinkaya/Hackhaton"
MANIFEST="$BASE/output/analysis/internet_refs/internet_chart_manifest.tsv"
OUTDIR="$BASE/output/analysis/internet_refs/images"

mkdir -p "$OUTDIR"
TAB=$(printf '\t')

tail -n +2 "$MANIFEST" | while IFS="$TAB" read -r ref_id provider chart_type title page_url image_url local_file; do
  dest="$OUTDIR/$local_file"
  curl -sL "$image_url" -o "$dest"
  if [ ! -s "$dest" ]; then
    echo "FAILED $ref_id $image_url"
    continue
  fi
  echo "OK $ref_id -> $local_file"
done
