#!/usr/bin/env bash
set -euo pipefail

BASE="/Users/yasinkaya/Hackhaton"
export MPLCONFIGDIR=/tmp/mpl

cd "$BASE"

echo "[1/14] Expand internet manifest"
python3 "$BASE/scripts/expand_internet_chart_manifest.py"

echo "[2/14] Download internet reference charts"
bash "$BASE/scripts/download_internet_chart_refs.sh"

echo "[3/14] Core internet graph similarity"
python3 "$BASE/scripts/internet_graph_similarity_research.py"

echo "[4/14] Family recommendation layer"
python3 "$BASE/scripts/final_similarity_family_analysis.py"

echo "[5/14] Weight sensitivity robustness"
python3 "$BASE/scripts/internet_similarity_weight_sensitivity.py"

echo "[6/14] Perturbation robustness"
python3 "$BASE/scripts/internet_similarity_augmentation_robustness.py"

echo "[7/14] Consensus v2"
python3 "$BASE/scripts/build_consensus_recommendation.py"

echo "[8/14] Consensus explainability"
python3 "$BASE/scripts/build_consensus_v2_explainability.py"

echo "[9/14] Decision pack"
python3 "$BASE/scripts/build_consensus_v2_decision_pack.py"

echo "[10/14] Top3 pair visuals"
python3 "$BASE/scripts/build_internet_similarity_top3_pairs.py"

echo "[11/14] Archetype map"
python3 "$BASE/scripts/build_internet_similarity_archetype_map.py"

echo "[12/14] Consensus v3 stable"
python3 "$BASE/scripts/build_consensus_v3_stable.py"

echo "[13/14] Consensus v4 calibrated"
python3 "$BASE/scripts/build_consensus_v4_calibrated.py"

echo "[14/14] Consensus v5 hardened"
python3 "$BASE/scripts/build_consensus_v5_hardened.py"

echo "Done: output/analysis/internet_graph_similarity"
