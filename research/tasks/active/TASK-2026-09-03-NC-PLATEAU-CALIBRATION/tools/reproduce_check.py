#!/usr/bin/env python3
"""Prove that this task's run_cell.py is EXACT-COMPATIBLE with the predecessors'.

The whole reuse ledger rests on one claim: a population produced by THIS
package at a given (L, T, zeta, lambda, N_c, dtau_mult, resample_scheme, seed)
is the same population a predecessor produced at that tuple. shared/run_cell.py
was changed -- it records ten per-window histories, the final weight vector,
delta_tau, n_resampling_events and git_commit that the predecessor discarded --
so the claim needs demonstrating rather than asserting.

Method. Take a completed predecessor population, run THIS package's run_cell.py
against a manifest holding exactly that row, and compare the observable to the
stored value BIT FOR BIT. Extra output fields cannot perturb a trajectory, but
a reordered RNG draw or a changed keyword would, and this is what would catch
that.

    tools/reproduce_check.py [n_cells]

Picks the cheapest completed populations available, so this is minutes of local
T0 analysis compute, not a simulation campaign. Submits nothing.
"""
import os, sys, csv, json, glob, math, subprocess, tempfile, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
PY = os.environ.get("PYTHON", os.path.join(REPO, ".venv", "bin", "python3"))
FIELDS = ["arm", "L", "T", "N_c", "zeta", "lam", "dtau_mult",
          "resample_scheme", "seed"]


def candidates():
    out = []
    for p in glob.glob(os.path.join(REPO, "research", "tasks", "**", "results",
                                    "*.json"), recursive=True):
        if os.path.abspath(p).startswith(TASK + os.sep):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "cmi_weighted_mean" not in d:
            continue
        if d.get("status") != "ok":
            continue
        out.append((float(d.get("wall_s", 1e9)), p, d))
    out.sort(key=lambda t: t[0])
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cands = candidates()
    if not cands:
        sys.exit("no completed predecessor population found to reproduce")
    print("=" * 78)
    print("EXACT-COMPATIBILITY REPRODUCTION — this package's run_cell.py against")
    print("populations produced by the predecessor campaigns")
    print("=" * 78)
    bad = 0
    for wall, path, d in cands[:n]:
        work = tempfile.mkdtemp(prefix="ncplat_repro_")
        try:
            shutil.copy2(os.path.join(TASK, "shared", "run_cell.py"),
                         os.path.join(work, "run_cell.py"))
            # run_cell resolves ../support and five-levels-up as the repo, so
            # place the scratch arm where a real arm sits.
            arm = os.path.join(TASK, ".repro_arm")
            if os.path.isdir(arm):
                shutil.rmtree(arm)
            os.makedirs(os.path.join(arm, "results"))
            shutil.copy2(os.path.join(TASK, "shared", "run_cell.py"),
                         os.path.join(arm, "run_cell.py"))
            with open(os.path.join(arm, "manifest.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
                w.writeheader()
                w.writerow({k: d[k] for k in FIELDS})
            t0 = time.time()
            r = subprocess.run([PY, "run_cell.py", "0"], cwd=arm,
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:])
                print(r.stderr[-2000:])
                sys.exit(f"run_cell.py failed: {r.returncode}")
            got = json.load(open(glob.glob(os.path.join(arm, "results",
                                                        "*.json"))[0]))
            # THE CRITERION, and why it is not "every float is bit-equal".
            #
            # The TRAJECTORY must be bit-identical: per_clone_CMI is the
            # per-clone observable the sampler produced, and any change to the
            # RNG stream, the keyword set or the order of primitive calls moves
            # it. Integer diagnostics (n_steps, founder counts, fallbacks) must
            # be exactly equal for the same reason.
            #
            # The DERIVED REDUCTIONS are a different matter. The stored values
            # were reduced on Ruche's x86 CPUs and are recomputed here on
            # arm64; numpy's pairwise summation and the available FMA differ, so
            # a mean over 1024 identical values can differ in the last bit.
            # Demanding bit equality there would be testing the two machines'
            # floating-point units, not the sampler. The tolerance below is
            # 1e-12 relative -- about four orders of magnitude tighter than
            # anything that could hide a real change, and about three orders
            # looser than the ~5e-16 the architectures actually differ by.
            RTOL = 1e-12
            same = []
            for k in ("n_steps", "n_nonfinite", "n_distinct_anc_final",
                      "brentq_fallbacks"):
                a, b = d.get(k), got.get(k)
                same.append((k, a, b, a == b, "exact"))
            for k in ("cmi_weighted_mean", "cmi_unweighted_mean",
                      "cmi_within_var", "gess_final", "ess_cum_final",
                      "ess_frac_mean"):
                a, b = float(d.get(k)), float(got.get(k))
                rel = abs(a - b) / max(abs(a), 1e-300)
                same.append((k, a, b, rel <= RTOL, f"rel {rel:.2e}"))
            pc_eq = d.get("per_clone_CMI") == got.get("per_clone_CMI")
            allok = all(s[3] for s in same) and pc_eq
            bad += 0 if allok else 1
            print(f"\n  L={d['L']} T={d['T']} zeta={d['zeta']} lam={d['lam']} "
                  f"N_c={d['N_c']} dtau_mult={d['dtau_mult']} seed={d['seed']}")
            print(f"  source {os.path.relpath(path, REPO)}")
            print(f"  reran in {time.time() - t0:.1f} s "
                  f"(original Ruche wall_s {wall:.1f})")
            print(f"    {'OK  ' if pc_eq else 'DIFF'}  "
                  f"per_clone_CMI            "
                  f"{len(d.get('per_clone_CMI') or [])} values, "
                  f"{'BIT-IDENTICAL' if pc_eq else 'DIFFERENT'}   "
                  f"<- the trajectory itself")
            for k, a, b, eq, how in same:
                print(f"    {'OK  ' if eq else 'DIFF'}  {k:<24} "
                      f"stored {a!r}  reran {b!r}   [{how}]")
            print(f"    new fields recorded here and absent from the stored "
                  f"row: "
                  f"{sorted(set(got) - set(d))}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(os.path.join(TASK, ".repro_arm"), ignore_errors=True)
    if not bad:
        print("\n  Reductions differ from the stored values only at the last "
              "bit or two.\n  That is x86-vs-arm64 summation order in numpy, "
              "not a code change: the\n  per-clone trajectory is bit-identical. "
              "It is worth recording that a\n  stored AGGREGATE in this corpus "
              "is not bit-reproducible on a different\n  architecture, even "
              "though the physics is.")
    print("\n" + ("REPRODUCTION EXACT — the trajectory is bit-identical, the "
                  "sampler is\nunchanged, and the reuse ledger is sound"
                  if not bad else
                  f"REPRODUCTION FAILED on {bad} cell(s). The reuse ledger is "
                  f"NOT sound and no arm may be submitted."))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
