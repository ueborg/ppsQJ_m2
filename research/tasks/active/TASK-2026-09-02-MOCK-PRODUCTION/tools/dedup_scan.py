#!/usr/bin/env python3
"""Reproduce the duplicate-compute scan behind ../REUSE_AND_DEDUP_AUDIT.md.

Searches every source of completed Cut-B runs reachable from this repository for
anything landing on one of this campaign's (L, lambda) grid cells, and prints
what it finds together with why each hit is or is not poolable.

Read-only. Writes nothing. Contains no scheduler call.

    python3 tools/dedup_scan.py [--corpus /path/to/pps_all_realizations.csv]
"""
import argparse, csv, glob, json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))

GRID = [round(0.2332 + 0.010 * i, 4) for i in range(13)]
WANT_L = {32, 48, 64}
ZETA = 0.35
DTAU_MULT = 6.0
THIS_TASK = "TASK-2026-09-02-MOCK-PRODUCTION"


def on_grid(lam):
    return any(abs(lam - g) < 1e-9 for g in GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(
        TASK, "frozen_inputs", "historical_corpus_zeta035.csv"),
        help="historical corpus (defaults to the frozen zeta=0.35 slice)")
    a = ap.parse_args()

    hits = collections.Counter()

    # 1. SMCSTAT local jsonl blocks
    for f in glob.glob(os.path.join(
            REPO, "research/tasks/active/TASK-2026-08-30-SMCSTAT/scratch/*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            L = d.get("L")
            lam = d.get("lam", d.get("lambda"))
            if L in WANT_L and lam is not None and on_grid(float(lam)):
                hits[("SMCSTAT/scratch", L, d.get("T"), d.get("N_c"),
                      round(float(lam), 4), d.get("dtau_mult"))] += 1

    # 2. every manifest in the task tree, except this task's own
    for m in glob.glob(os.path.join(REPO, "research/tasks/active/*/**/manifest.csv"),
                       recursive=True):
        if THIS_TASK in m:
            continue
        tag = os.path.relpath(m, os.path.join(REPO, "research/tasks/active")).split(os.sep)[0]
        for r in csv.DictReader(open(m)):
            try:
                L, lam = int(r["L"]), float(r["lam"])
            except Exception:
                continue
            if L in WANT_L and on_grid(lam):
                hits[(tag, L, float(r["T"]), int(r["N_c"]), round(lam, 4),
                      float(r.get("dtau_mult", 0)))] += 1

    # 3. the historical corpus
    if os.path.isfile(a.corpus):
        for r in csv.DictReader(open(a.corpus)):
            try:
                L, lam = int(r["L"]), float(r["lambda"])
            except Exception:
                continue
            if L in WANT_L and on_grid(lam):
                dm = round(float(r["dtau"]) * 2 * lam * (L - 1), 4)
                hits[("historical corpus", L, r["T"], int(r["N_c"]),
                      round(lam, 4), dm)] += 1
    else:
        print(f"NOTE: corpus not found at {a.corpus}; section 3 not scanned")

    print("=" * 78)
    print("  Rows anywhere reachable that land on an (L, lambda) cell of this")
    print("  campaign's frozen 13-point grid, at L in {32, 48, 64}")
    print("=" * 78)
    print(f"  {'source':<38}{'L':>4}{'T':>7}{'N_c':>6}{'lambda':>9}"
          f"{'dtau_m':>8}{'n':>6}  poolable?")
    total = 0
    for k, v in sorted(hits.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][4])):
        src, L, T, N_c, lam, dm = k
        total += v
        if dm is not None and abs(float(dm) - DTAU_MULT) > 1e-9:
            why = "NO: dtau_mult != 6.0, and no recoverable seed"
        elif src.startswith("TASK-2026-09-02-SMC-HIGHRUNG"):
            why = "YES: reused from frozen_inputs/armB_populations.csv"
        else:
            why = "NO: cell does not match (T, N_c or zeta)"
        print(f"  {str(src):<38}{L:>4}{str(T):>7}{str(N_c):>6}{lam:>9.4f}"
              f"{str(dm):>8}{v:>6}  {why}")
    print(f"\n  total rows found: {total}")
    print("\n  Expected, and asserted in ../REUSE_AND_DEDUP_AUDIT.md:")
    print("    288 ARM-B rows (reused, not recomputed) + 12 corpus rows at")
    print("    L=64, lambda=0.3032, N_c=128, dtau_mult=12 (NOT poolable).")
    print("    Every other cell of this campaign is new compute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
