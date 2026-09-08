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

WHY THE EXECUTOR IS INVOKED FROM THE ARM AND NOT FROM shared/
-------------------------------------------------------------
run_cell.py computes HERE from its own __file__ and reads `HERE/manifest.csv`,
writing to `HERE/results` by default. It takes NO manifest argument and NO
output argument from this wrapper, so the directory the executor FILE sits in --
not the working directory -- decides which manifest is run.

This wrapper previously invoked `shared/run_cell.py` while chdir'ed into the
arm. Every row therefore looked for `shared/manifest.csv`, which does not exist,
and died with FileNotFoundError before the sampler was reached. That is not a
hypothetical: it destroyed the first Ruche submission of this task (job IDs
1694328-1694621, zero result JSONs). See ../RUCHE_INCIDENT_2026-09-08.md.

The repair is packaging-only and run_cell.py is untouched, byte for byte. Every
arm now carries its OWN copy of the certified executor, written by
tools/build_arms.py, and this wrapper invokes ARM/run_cell.py. The copy is
verified below to be byte-identical to shared/run_cell.py before any row runs,
so "arm-local" can never quietly become "arm-modified".

Idempotent: run_cell.py skips a row whose output already exists, so a requeued
or timed-out pack costs only what it has left to do, and a pack may be re-run
freely.

THIS FILE CONTAINS NO SCHEDULER CALL AND CANNOT SUBMIT ANYTHING.
"""
import csv, hashlib, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.getcwd()
RUN_CELL = os.path.join(ARM, "run_cell.py")          # ARM, not HERE. See above.
FROZEN_CELL = os.path.join(HERE, "run_cell.py")      # the shared frozen executor


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def check_executor():
    """The arm-local executor must exist and must be the frozen bytes.

    Read strictly: an arm without an executor is a packaging fault and must not
    silently fall back to the shared copy, because a shared copy reads the wrong
    manifest -- that fallback IS the bug this file was repaired for.
    """
    if not os.path.isfile(RUN_CELL):
        sys.exit(
            "run_pack.py cannot start: no arm-local executor.\n"
            "    expected %s\n"
            "    cwd      %s\n"
            "  Every arm must carry its own byte-identical run_cell.py, written\n"
            "  by tools/build_arms.py. Falling back to %s would read that\n"
            "  directory's manifest.csv instead of this arm's and is refused."
            % (RUN_CELL, ARM, FROZEN_CELL))
    if os.path.isfile(FROZEN_CELL):
        a, b = sha256(RUN_CELL), sha256(FROZEN_CELL)
        if a != b:
            sys.exit("INTEGRITY FAILURE: the arm-local executor is not the frozen "
                     "executor.\n    %s  %s\n    %s  %s\n"
                     "  Refusing to run: the sampler would not be the certified one."
                     % (a, RUN_CELL, b, FROZEN_CELL))
    else:
        print("[warn] %s is absent; arm-local executor not cross-checked "
              "against the frozen copy" % FROZEN_CELL, flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "--resolve":
    # Self-report for shared/preflight.py P19. Resolves paths and exits; runs
    # nothing, reads no manifest, writes nothing.
    print("ARM       %s" % ARM)
    print("RUN_CELL  %s" % RUN_CELL)
    print("EXISTS    %s" % os.path.isfile(RUN_CELL))
    print("FROZEN    %s" % FROZEN_CELL)
    print("SHA256    %s" % (sha256(RUN_CELL) if os.path.isfile(RUN_CELL) else "-"))
    print("FROZEN_SHA256 %s" % (sha256(FROZEN_CELL)
                                if os.path.isfile(FROZEN_CELL) else "-"))
    sys.exit(0)

check_executor()

pack_id = int(sys.argv[1])
packs = list(csv.DictReader(open(os.path.join(ARM, "packs.csv"))))
try:
    p = next(x for x in packs if int(x["pack"]) == pack_id)
except StopIteration:
    sys.exit(f"pack {pack_id} is not in packs.csv ({len(packs)} packs)")

start, count = int(p["start"]), int(p["count"])
print(f"[pack {pack_id}] rows {start}..{start + count - 1} "
      f"(est {float(p['est_sec']):.0f}s)", flush=True)
print(f"[pack {pack_id}] executor {RUN_CELL}", flush=True)

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
