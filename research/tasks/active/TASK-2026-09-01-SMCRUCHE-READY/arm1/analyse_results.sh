#!/bin/bash
# Analyse this arm's results under the FROZEN rules in analysis_spec.yaml.
# Pools with the completed local blocks when PPSQJ_REPO is set, because ARM1's
# seeds continue the SMCSTAT A-P96 stream at an identical cell.
#
# The ${arr[@]+"${arr[@]}"} idiom is deliberate: under `set -u`, bash 3.2 (which
# is what macOS ships) treats "${arr[@]}" on an EMPTY array as an unbound
# variable and aborts. This form expands to nothing when the array is empty.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON:-python3}
EXTRA=()
if [ -n "${PPSQJ_REPO:-}" ]; then
  for f in A-P96 A-BUD; do
    p="$PPSQJ_REPO/research/tasks/active/TASK-2026-08-30-SMCSTAT/scratch/$f.jsonl"
    if [ -f "$p" ]; then EXTRA+=("$p"); fi
  done
fi
# find, not ls: with `set -o pipefail`, a glob that matches nothing makes ls
# exit 1 and aborts the script before it can print anything at all.
mkdir -p results
n=$(find results -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
echo "analysing $n result file(s) in $(pwd)/results"
if [ ${#EXTRA[@]} -gt 0 ]; then
  echo "pooling with the completed local blocks:"
  printf '  %s\n' "${EXTRA[@]}"
else
  echo "not pooling with local blocks (PPSQJ_REPO unset, or the JSONLs are absent)"
fi
if [ "$n" -eq 0 ] && [ ${#EXTRA[@]} -eq 0 ]; then
  echo "nothing to analyse yet." >&2; exit 1
fi
exec "$PY" analyse_ruche.py results ${EXTRA[@]+"${EXTRA[@]}"}
