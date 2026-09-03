#!/usr/bin/env python3
"""Negative controls for shared/preflight.py.

    python3 tools/negative_controls.py [staging_dir]

A preflight that passes everything is not evidence. Each control below copies
the package to a staging tree, injects exactly ONE fault, and requires the
preflight to EXIT NON-ZERO and to name that fault in its problem list. A
control that the preflight lets through is a failure of this script.

It never modifies the package. It contains no scheduler call and cannot submit.
"""
import os, sys, csv, json, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
ARM = "lowlamL64"
DEFAULT_STAGE = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                             "lowlam_negctl")
# The staging tree is deliberately OUTSIDE the repository, so run_cell.py's
# "five levels up" rule cannot find pps_qj there. PPSQJ_REPO is the override
# the preflight documents for exactly this case. Setting it is what makes the
# clean control N00 a real control rather than a layout artefact.
REPO = os.path.abspath(os.path.join(TASK, os.pardir, os.pardir, os.pardir,
                                    os.pardir))
ENV = dict(os.environ, PPSQJ_REPO=REPO)
COPY = ["analysis_spec.yaml", "support", "frozen_inputs", "tools", ARM]


def stage(root, name):
    d = os.path.join(root, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for item in COPY:
        src = os.path.join(TASK, item)
        dst = os.path.join(d, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst,
                            ignore=shutil.ignore_patterns("__pycache__",
                                                          "*.pyc"))
        else:
            shutil.copy2(src, dst)
    return d


def rows_of(d):
    p = os.path.join(d, ARM, "manifest.csv")
    with open(p) as fh:
        rd = csv.DictReader(fh)
        return list(rd), rd.fieldnames


def write_rows(d, rows, fields):
    with open(os.path.join(d, ARM, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sub(d, old, new, path=None):
    p = path or os.path.join(d, ARM, "submit.slurm")
    s = open(p).read()
    assert old in s, "cannot inject: %r not found in %s" % (old, p)
    open(p, "w").write(s.replace(old, new, 1))


# --- the faults -----------------------------------------------------------
def f_duplicate_lambda(d):
    rows, fields = rows_of(d)
    for r in rows[:24]:
        r["lam"] = "0.2332"                 # already measured by the parent
    write_rows(d, rows, fields)


def f_missing_new_lambda(d):
    rows, fields = rows_of(d)
    write_rows(d, [r for r in rows if r["lam"] != "0.1932"], fields)
    sub(d, "--array=0-95%64", "--array=0-71%64")


def f_off_grid_lambda(d):
    rows, fields = rows_of(d)
    rows[0]["lam"] = "0.1982"
    write_rows(d, rows, fields)


def f_stale_seed(d):
    rows, fields = rows_of(d)
    rows[0]["seed"] = "31200000"            # a predecessor seed, exactly
    write_rows(d, rows, fields)


def f_duplicate_seed(d):
    rows, fields = rows_of(d)
    rows[1]["seed"] = rows[0]["seed"]
    write_rows(d, rows, fields)


def f_unequal_R(d):
    rows, fields = rows_of(d)
    write_rows(d, rows[:95], fields)
    sub(d, "--array=0-95%64", "--array=0-94%64")


def f_wrong_dtau(d):
    rows, fields = rows_of(d)
    for r in rows:
        r["dtau_mult"] = "12.0"             # the historical corpus value
    write_rows(d, rows, fields)


def f_wrong_nc(d):
    rows, fields = rows_of(d)
    for r in rows:
        r["N_c"] = "128"
    write_rows(d, rows, fields)


def f_wrong_L(d):
    rows, fields = rows_of(d)
    for r in rows:
        r["L"] = "80"
    write_rows(d, rows, fields)


def f_partition_cpu_short(d):
    sub(d, "--partition=cpu_med", "--partition=cpu_short")
    sub(d, "--time=02:00:00", "--time=00:55:00")


def f_no_partition(d):
    sub(d, "#SBATCH --partition=cpu_med\n", "")


def f_time_too_short(d):
    sub(d, "--time=02:00:00", "--time=00:20:00")


def f_time_over_maxtime(d):
    sub(d, "--time=02:00:00", "--time=05:00:00")


def f_mem_too_small(d):
    # 600M is an ordinary Slurm request and is genuinely below the 665 MB peak
    # x 1.5. NOTE: --mem=1G would NOT be a fault here -- 1 GiB clears the
    # 0.97 GiB requirement -- and using it was what exposed the size-parser
    # defect recorded in ../VALIDATION.md.
    sub(d, "--mem=2G", "--mem=600M")


def f_array_mismatch(d):
    sub(d, "--array=0-95%64", "--array=0-99%64")


def f_corrupt_bundle(d):
    p = os.path.join(d, "support", "instrumented.py")
    open(p, "a").write("\n# an unrecorded edit to the certified sampler\n")


def f_missing_frozen_inputs(d):
    os.remove(os.path.join(d, "frozen_inputs",
                           "predecessor_nc1024_populations.csv"))


def f_cost_model_drift(d):
    p = os.path.join(d, "tools", "cost_model.py")
    s = open(p).read()
    s = s.replace("64: (2.723572, 850.23)", "64: (2.723572, 600.00)")
    open(p, "w").write(s)


def f_scheduler_in_preflight(d):
    p = os.path.join(d, ARM, "run_preflight.sh")
    s = open(p).read()
    open(p, "w").write(s + "\n# sbatch submit.slurm\n")


CONTROLS = [
    ("N01 duplicates an already-measured lambda", f_duplicate_lambda,
     "no predecessor duplication"),
    ("N02 drops one of the four new lambdas", f_missing_new_lambda,
     "all four new lambdas present"),
    ("N03 lambda off the frozen 17-point grid", f_off_grid_lambda,
     "lambda on frozen grid"),
    ("N04 reuses a predecessor seed", f_stale_seed,
     "seeds in the fresh block"),
    ("N05 duplicates a seed within the arm", f_duplicate_seed,
     "seeds unique within arm"),
    ("N06 unequal R across lambdas", f_unequal_R,
     "R equal across lambdas"),
    ("N07 dtau_mult 12.0, the non-poolable corpus value", f_wrong_dtau,
     "dtau_mult"),
    ("N08 N_c = 128 instead of 1024", f_wrong_nc,
     "N_c"),
    ("N09 an L with no measured cost model", f_wrong_L,
     "L"),
    ("N10 partition cpu_short", f_partition_cpu_short,
     "partition"),
    ("N11 no --partition at all", f_no_partition,
     "partition"),
    ("N12 --time below the pessimistic slowest task", f_time_too_short,
     "--time"),
    ("N13 --time above the partition MaxTime", f_time_over_maxtime,
     "MaxTime"),
    ("N14 --mem below 1.5x the estimated peak", f_mem_too_small,
     "--mem"),
    ("N15 --array range not matching the manifest", f_array_mismatch,
     "--array"),
    ("N16 an unrecorded edit to the certified sampler", f_corrupt_bundle,
     "sha256"),
    ("N17 the frozen predecessor data removed", f_missing_frozen_inputs,
     "frozen predecessor data"),
    ("N18 cost-model literal drifted from the frozen data", f_cost_model_drift,
     "cost model"),
    ("N19 a scheduler call added to run_preflight.sh", f_scheduler_in_preflight,
     "run_preflight.sh"),
]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STAGE
    os.makedirs(root, exist_ok=True)

    print("=" * 78)
    print("  NEGATIVE CONTROLS — shared/preflight.py")
    print("  Each injects ONE fault and requires a NON-ZERO exit.")
    print("  staging: %s" % root)
    print("=" * 78)

    # the unmodified control must PASS, or every failure below is meaningless
    d0 = stage(root, "N00_clean")
    r0 = subprocess.run([sys.executable, "preflight.py"],
                        cwd=os.path.join(d0, ARM), env=ENV,
                        capture_output=True, text=True)
    print("\n  N00 unmodified copy                             exit=%d  %s"
          % (r0.returncode, "PASS (as required)" if r0.returncode == 0
             else "** THE CLEAN CONTROL FAILED **"))
    bad = []
    if r0.returncode != 0:
        bad.append("N00: the unmodified staged copy does not pass; every other "
                   "control below is uninterpretable")
        print(r0.stdout[-3000:])

    for name, fn, expect in CONTROLS:
        tag = name.split()[0]
        d = stage(root, tag)
        fn(d)
        r = subprocess.run([sys.executable, "preflight.py"],
                           cwd=os.path.join(d, ARM), env=ENV,
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        caught = (r.returncode != 0)
        named = expect.lower() in out.lower()
        ok = caught and named
        print("  %-48s exit=%d  %s" % (name, r.returncode,
                                       "caught" if ok else "** NOT CAUGHT **"))
        if not ok:
            bad.append("%s: exit=%d, expected non-zero mentioning %r"
                       % (tag, r.returncode, expect))

    print("\n" + "=" * 78)
    if bad:
        print("  NEGATIVE CONTROLS FAILED")
        for b in bad:
            print("    * %s" % b)
        print("=" * 78)
        return 1
    print("  ALL %d NEGATIVE CONTROLS CAUGHT, and the unmodified copy passes."
          % len(CONTROLS))
    print("  The preflight fails closed on every fault this package can carry")
    print("  into a queue.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
