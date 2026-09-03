#!/usr/bin/env python3
"""Injected-fault negative controls.

A preflight that has never been shown to FAIL is a preflight nobody has tested.
Each control below breaks one thing in a throwaway copy of a real arm and
asserts the preflight exits non-zero for THAT reason. A control that passes
because the preflight failed for an unrelated reason is not a control, so each
one also checks the reported failure text.

Also unit-tests the Slurm memory parser directly, because
TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING asked for it to be re-verified after
the unit bug the predecessor found, and a parser that fails OPEN on --mem=2048
is exactly the kind of defect that survives a green preflight.

Writes only into a temporary copy. Contains no scheduler call.
"""
import os, re, sys, csv, shutil, tempfile, subprocess, json

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
PY = os.environ.get("PYTHON", os.path.join(REPO, ".venv", "bin", "python3"))
BASE = "B_L64_cross_nc1024"          # a real arm with reuse, top-ups and 7 cells

results = []


def run_preflight(armdir):
    env = dict(os.environ, NCPLAT_ARM_NAME=BASE)
    r = subprocess.run([PY, "preflight.py"], cwd=armdir, env=env,
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def control(name, mutate, expect_substr):
    """Copy the arm INTO the task tree (preflight resolves ../tools and
    ../support relatively), break it, expect a non-zero exit naming the fault."""
    tmp = os.path.join(TASK, ".nc_" + name)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(os.path.join(TASK, BASE), tmp)
    try:
        mutate(tmp)
        rc, out = run_preflight(tmp)
        hit = expect_substr.lower() in out.lower()
        ok = rc != 0 and hit
        results.append((name, ok, rc, expect_substr,
                        "" if ok else out[-700:]))
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<34} exit={rc} "
              f"expected text {'found' if hit else 'NOT FOUND'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sed(path, pat, rep):
    s = open(path).read()
    s2 = re.sub(pat, rep, s, count=1)
    assert s2 != s, f"injection did not change {path}: {pat}"
    open(path, "w").write(s2)


def main():
    print("=" * 78)
    print("NEGATIVE CONTROLS — every one of these MUST make the preflight fail")
    print("=" * 78)
    print(f"  base arm: {BASE}\n")

    # -- the manifest is the design ------------------------------------------
    def n1(d):
        rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
        rows[0]["lam"] = "0.2500"                      # an off-grid lambda
        write(d, rows)
    control("N01_off_grid_lambda", n1, "manifest == frozen design")

    def n2(d):
        rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
        rows[5]["seed"] = rows[4]["seed"]              # duplicate seed
        write(d, rows)
    control("N02_duplicate_seed", n2, "manifest == frozen design")

    def n3(d):
        rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
        rows[0]["zeta"] = "0.30"
        write(d, rows)
    control("N03_wrong_zeta", n3, "manifest == frozen design")

    def n4(d):
        rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
        rows[0]["dtau_mult"] = "12.0"                  # a different discretisation
        write(d, rows)
    control("N04_wrong_dtau_mult", n4, "manifest == frozen design")

    def n5(d):
        rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
        rows[0]["T"] = "32.0"                          # T != L
        write(d, rows)
    control("N05_T_not_equal_L", n5, "manifest == frozen design")

    def n6(d):
        rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
        rows[0]["resample_scheme"] = "multinomial"
        write(d, rows)
    control("N06_wrong_resampler", n6, "manifest == frozen design")

    def n7(d):
        rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
        rows[0]["seed"] = "31200000"                   # a PREDECESSOR's seed
        write(d, rows)
    control("N07_predecessor_seed", n7, "manifest == frozen design")

    def n8(d):
        rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
        write(d, rows[:-1])                            # one row short
    control("N08_row_count_mismatch", n8, "manifest == frozen design")

    # -- the job script ------------------------------------------------------
    control("N09_array_range_mismatch",
            lambda d: sed(os.path.join(d, "submit.slurm"),
                          r"--array=0-\d+", "--array=0-9"),
            "--array matches manifest")
    control("N10_time_too_short",
            lambda d: sed(os.path.join(d, "submit.slurm"),
                          r"--time=\d\d:00:00", "--time=00:05:00"),
            "--time")
    control("N11_mem_under_request",
            lambda d: sed(os.path.join(d, "submit.slurm"),
                          r"--mem=\d+G", "--mem=100M"),
            "--mem")
    control("N12_mem_unit_trap",
            # --mem=2048 means 2048 MEGABYTES to Slurm. A parser that reads it
            # as 2048 GiB waves through an under-request. This arm needs ~2 GB,
            # so 2048 MB is genuinely marginal and the parser must not read it
            # as 2 TiB. Paired with the unit table below.
            lambda d: sed(os.path.join(d, "submit.slurm"),
                          r"--mem=\d+G", "--mem=200"),
            "--mem")
    control("N13_wrong_partition",
            lambda d: sed(os.path.join(d, "submit.slurm"),
                          r"--partition=\w+", "--partition=cpu_short"),
            "--partition")
    control("N14_multicore_request",
            lambda d: sed(os.path.join(d, "submit.slurm"),
                          r"--cpus-per-task=1", "--cpus-per-task=4"),
            "one core per task")

    # -- the runtime ---------------------------------------------------------
    def n15(d):
        p = os.path.join(d, "run_cell.py")
        open(p, "a").write("\n# drift\n")
    control("N15_run_cell_drift", n15, "shared/run_cell.py")

    def n16(d):
        p = os.path.join(d, "analyse_results.sh")
        open(p, "a").write("\n" + "s" + "batch submit.slurm\n")
    control("N16_scheduler_call_injected", n16, "scheduler call")

    # -- the parser unit table -----------------------------------------------
    print("\n  Slurm --mem unit table (the predecessor's N14 defect, re-verified)")
    sys.path.insert(0, os.path.join(TASK, BASE))
    import preflight as PF
    table = [("2G", 2.0), ("2g", 2.0), ("512M", 0.5), ("600M", 600 / 1024),
             ("2048", 2.0), ("1024K", 1 / 1024), ("1T", 1024.0), ("", 0.0),
             ("nonsense", 0.0)]
    allok = True
    for raw, want in table:
        got = PF.gib(raw)
        good = abs(got - want) < 1e-9
        allok &= good
        print(f"    {'OK  ' if good else 'FAIL'}  --mem={raw!r:<12} -> "
              f"{got:.6f} GiB (expected {want:.6f})")
    results.append(("N17_mem_unit_table", allok, 0, "unit table", ""))

    nfail = [r for r in results if not r[1]]
    print("\n" + ("ALL NEGATIVE CONTROLS PASSED — the preflight fails when it "
                  "should" if not nfail else
                  f"{len(nfail)} CONTROL(S) DID NOT FIRE"))
    for r in nfail:
        print(f"   * {r[0]}: exit={r[2]}, expected {r[3]!r}\n{r[4]}")
    json.dump([dict(control=r[0], fired=r[1], exit=r[2], expected=r[3])
               for r in results],
              open(os.path.join(HERE, "negative_controls.json"), "w"), indent=1)
    return 1 if nfail else 0


def write(d, rows):
    with open(os.path.join(d, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "L", "T", "N_c", "zeta", "lam",
                                           "dtau_mult", "resample_scheme", "seed"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
