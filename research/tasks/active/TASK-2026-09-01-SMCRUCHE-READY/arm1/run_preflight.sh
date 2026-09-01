#!/bin/bash
# Preflight for this arm. Prints what would be requested. Submits NOTHING.
# There is deliberately no scheduler-submission command in this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON:-python3}
mkdir -p results
exec "$PY" preflight.py
