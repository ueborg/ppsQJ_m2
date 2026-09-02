#!/bin/bash
# Analyse THIS ARM under the frozen rules in ../analysis_spec.yaml.
# Contains no scheduler call. Submits nothing and cannot.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON:-python3}
mkdir -p results
# find, not ls: under `set -o pipefail` a glob matching nothing makes ls exit 1
# and abort the script before it can print anything at all.
n=$(find results -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
echo "analysing $n result file(s) in $(pwd)/results"
if [ "$n" -eq 0 ]; then
  echo "nothing to analyse yet." >&2; exit 1
fi
exec "$PY" analyse_arm.py results
