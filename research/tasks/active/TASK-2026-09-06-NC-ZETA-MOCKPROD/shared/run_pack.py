#!/usr/bin/env python3
"""Execute one PACK of manifest rows. Called once per SLURM array task.

Why packing exists: at the cheapest cell in this campaign one population runs in
a few seconds. An array of 1 680 such tasks is scheduler abuse and its queue
overhead exceeds its compute. `packs.csv` (written by tools/build_arms.py)
records the (start, count) of every pack, sized so no array task is shorter than
ten minutes.

Why this wrapper and not a modified run_cell.py: `run_cell.py` is the CERTIFIED
per-row executor and is byte-identical to the predecessor's. It is invoked here
once per row in a fresh process, so the sampler, its arguments, the RNG seeding,
the discretisation, the resampling scheme and the observable are bit-for-bit
what they were -- packing changes the scheduling, never the computation. A row
produced by a packed task is EXACT-COMPATIBLE with one produced unpacked.

Idempotent: run_cell.py skips a row whose output already exists, so a requeued
or timed-out pack costs only what it has left to do, and a pack may be re-run
freely.

THIS FILE CONTAINS NO SCHEDULER CALL AND CANNOT SUBMIT ANYTHING.
"""
import csv, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.getcwd()
RUN_CELL = os.path.join(HERE, "run_cell.py")

pack_id = int(sys.argv[1])
packs = list(csv.DictReader(open(os.path.join(ARM, "packs.csv"))))
try:
    p = next(x for x in packs if int(x["pack"]) == pack_id)
except StopIteration:
    sys.exit(f"pack {pack_id} is not in packs.csv ({len(packs)} packs)")

start, count = int(p["start"]), int(p["count"])
print(f"[pack {pack_id}] rows {start}..{start + count - 1} "
      f"(est {float(p['est_sec']):.0f}s)", flush=True)

t0 = time.time()
failed = []
for idx in range(start, start + count):
    r = subprocess.run([sys.executable, RUN_CELL, str(idx)], cwd=ARM)
    if r.returncode != 0:
        # Record and continue: one bad row must not discard the pack's other
        # completed rows, which are already durable on disk. The analysis counts
        # missing rows explicitly and never treats an absent row as a value.
        failed.append((idx, r.returncode))
        print(f"[pack {pack_id}] row {idx} FAILED rc={r.returncode}", flush=True)

el = time.time() - t0
print(f"[pack {pack_id}] done in {el:.0f}s, {count - len(failed)}/{count} rows ok",
      flush=True)
if failed:
    print(f"[pack {pack_id}] FAILED ROWS: {failed}", flush=True)
    sys.exit(1)
