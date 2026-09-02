#!/bin/bash
# Preflight for this arm. Prints what WOULD be requested and validates the
# package. There is deliberately no scheduler-submission command in this file
# or in preflight.py, and preflight.py asserts that fact about this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON:-python3}
mkdir -p results
exec "$PY" preflight.py
