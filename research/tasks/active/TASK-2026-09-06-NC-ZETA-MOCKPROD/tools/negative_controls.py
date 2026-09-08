#!/usr/bin/env python3
"""Negative controls for TASK-2026-09-06-NC-ZETA-MOCKPROD.

A check that has never been observed to fail is not known to be a check. Each
control below injects one specific fault into a COPY of a real arm, runs the
real preflight against the copy, and requires the preflight to REJECT it with
the expected code. Nothing under the task directory is modified: every
injection happens inside a temporary directory that is removed afterwards.

    .venv/bin/python3 tools/negative_controls.py

Exit 0 only if every control both (a) passes clean before injection and
(b) fails with the expected code after it.
"""
from __future__ import annotations
import csv, json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
ROOT = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
PY = sys.executable
SHARED = os.path.join(TASK, "shared")
PREFLIGHT = os.path.join(SHARED, "preflight.py")
BASE_ARM = os.path.join(TASK, "M_z010_nc512")


def stage_arm(tmp, src_arm=BASE_ARM):
    """Copy an arm into a skeleton that REPRODUCES ITS DEPTH below the task
    directory.

    A bare copy in /tmp was enough while every path a preflight check cared
    about was resolved from preflight.py's own location. It is not enough any
    more: P17 resolves ../shared from the arm, and P18-P20 resolve the arm-local
    executor, its ../support bundle and PPSQJ_REPO. A copy at the wrong depth
    would fail those three for a reason that has nothing to do with the injected
    fault, and a control that fires for the wrong reason proves nothing.

    So the copy is placed at <tmp>/research/tasks/active/<TASK>/<rel>, with
    shared/ and support/ symlinked to the real ones and pps_qj symlinked at the
    skeleton root. Nothing under the task directory is written.
    """
    rel = os.path.relpath(src_arm, TASK)
    skel = os.path.join(tmp, "research", "tasks", "active", os.path.basename(TASK))
    arm = os.path.join(skel, rel)
    os.makedirs(os.path.dirname(arm), exist_ok=True)
    shutil.copytree(src_arm, arm,
                    ignore=shutil.ignore_patterns("results", "logs", "__pycache__"))
    for link, target in (("shared", SHARED),
                         ("support", os.path.join(TASK, "support"))):
        dst = os.path.join(skel, link)
        if not os.path.exists(dst):
            os.symlink(target, dst)
    os.symlink(os.path.join(ROOT, "pps_qj"), os.path.join(tmp, "pps_qj"))
    return arm


def run_preflight(arm):
    r = subprocess.run([PY, PREFLIGHT, arm], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def edit_slurm(arm, pattern, repl):
    p = os.path.join(arm, "submit.slurm")
    s = open(p).read()
    s2 = re.sub(pattern, repl, s, count=1)
    assert s2 != s, "injection %r did not match" % pattern
    open(p, "w").write(s2)


def edit_manifest(arm, fn):
    p = os.path.join(arm, "manifest.csv")
    rows = list(csv.DictReader(open(p)))
    flds = rows[0].keys()
    fn(rows)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flds))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- the controls
def n1(arm):
    """--time cut below the pessimistic slowest pack."""
    edit_slurm(arm, r"--time=\S+", "--time=00:01:00")
    return "P4"


def n2(arm):
    """--mem written with NO SUFFIX, which Slurm reads as MEGABYTES. The
    predecessor parser read a bare integer as GiB and failed OPEN."""
    edit_slurm(arm, r"--mem=\S+", "--mem=200")
    return "P5"


def n3(arm):
    """cpu_short, which is never used in this programme at any --time."""
    edit_slurm(arm, r"--partition=\S+", "--partition=cpu_short")
    return "P6"


def n4(arm):
    """--array range no longer matches packs.csv."""
    edit_slurm(arm, r"--array=0-\d+", "--array=0-9")
    return "P3"


def n5(arm):
    """A seed collides with another manifest in the repository."""
    edit_manifest(arm, lambda rows: rows[0].__setitem__("seed", "31000000"))
    return "P7"


def n6(arm):
    """R is no longer matched: one cell loses a population."""
    edit_manifest(arm, lambda rows: rows.pop(0))
    return "P1"


def n7(arm):
    """A zeta = 0.35 row appears, i.e. the anchor would be recomputed."""
    edit_manifest(arm, lambda rows: rows[0].__setitem__("zeta", "0.35"))
    return "P8"


def n8(arm):
    """The lambda grid is NARROWED at the top, so the phi=1/2 prediction ends up
    too close to the grid end. This is the failure that killed
    TASK-2026-09-05-NC-ZETA-CALIBRATION and that its own P9 could not see,
    because it tested a stencil CENTRE against a law instead of the SPAN."""
    def f(rows):
        for r in rows:
            if float(r["lam"]) > 0.115:
                r["lam"] = "0.115"
    edit_manifest(arm, f)
    return "P9"


def n9(arm):
    """dtau_mult != 6 leaks into a production arm."""
    edit_manifest(arm, lambda rows: rows[0].__setitem__("dtau_mult", "12.0"))
    return "P13"


def n10(arm):
    """packs.csv est_sec no longer reproduces from the cost model."""
    p = os.path.join(arm, "packs.csv")
    rows = list(csv.DictReader(open(p)))
    rows[0]["est_sec"] = "1.0"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return "P4b"


def n11(arm):
    """The packs no longer tile the manifest: a pack is dropped."""
    p = os.path.join(arm, "packs.csv")
    rows = list(csv.DictReader(open(p)))
    del rows[3]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return "P3b"


def n12(arm):
    """A row duplicates a population already stored in the repository. The
    injected cell is a real completed zeta=0.35 production cell."""
    def f(rows):
        r = rows[0]
        r["zeta"], r["L"], r["T"] = "0.35", "64", "64.0"
        r["N_c"], r["lam"], r["dtau_mult"] = "1024", "0.2732", "6.0"
    edit_manifest(arm, f)
    return "P14"           # P8 also fires; both are correct rejections


def n13(arm):
    """T no longer equals L, so the row's discretisation, cost and observable
    belong to a different system from the cell it is filed under. This control
    is the reason P2 is not the discretisation identity: against the identity
    version of P2, this injection PASSED."""
    edit_manifest(arm, lambda rows: rows[0].__setitem__("T", "17.0"))
    return "P2"


def n14(arm):
    """The path to run_pack.py no longer resolves from the arm directory. This
    is not hypothetical: the conditional arm lives one directory deeper than the
    rest and the template originally hard-coded ../shared, which does not
    resolve there. The job would have died immediately."""
    edit_slurm(arm, r"(PPSQJ_PYTHON\"\s+)\S+/run_pack\.py",
               r"\1../../../nowhere/run_pack.py")
    return "P17"


def n15(arm):
    """The arm carries NO run_cell.py, so the only executor in the package is
    the shared one -- the exact layout that killed Ruche jobs 1694328-1694621.
    run_cell.py reads manifest.csv from its own directory, so a shared executor
    invoked from an arm reads shared/manifest.csv, which does not exist."""
    os.remove(os.path.join(arm, "run_cell.py"))
    return "P18"


def n16(arm):
    """The arm-local executor exists but is NOT the frozen bytes. An arm-local
    copy that has drifted is a different sampler under an identical manifest,
    which is worse than a missing one because it produces plausible numbers."""
    p = os.path.join(arm, "run_cell.py")
    b = open(p, "rb").read()
    open(p, "wb").write(b + b"\n# drift\n")
    return "P18"


CONTROLS = [(f.__name__.upper(), f) for f in
            (n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14,
             n15, n16)]


def bug_reproduction():
    """Reproduce the OBSERVED Ruche failure end to end, then show the repair.

    The controls above prove the preflight rejects the broken LAYOUT. This
    proves the layout was actually broken -- that shared executor + arm working
    directory really does die the way the cluster logs say it did, rather than
    the preflight having been taught to reject a shape nobody demonstrated was
    fatal.

    Four steps, all inside a temporary directory:
      1  the PRE-FIX run_pack.py, run from an arm, resolves an executor OUTSIDE
         that arm  -- the resolution fault;
      2  that executor, run from the arm, dies with FileNotFoundError on a
         manifest.csv outside the arm, before the sampler, writing no result
         -- the observed failure, verbatim;
      3  the SHIPPED run_pack.py, same arm, resolves the arm-local executor
         -- the repair;
      4  the shipped run_pack.py REFUSES to run at all when the arm-local
         executor is absent, instead of silently falling back to the shared one.
    """
    print()
    print("  BUG REPRODUCTION  the failure of Ruche jobs 1694328-1694621")
    tmp = tempfile.mkdtemp(prefix="negctl_repro_")
    fails = []
    try:
        arm = stage_arm(tmp)
        skel = os.path.dirname(arm)
        dep = os.path.join(skel, "shared_asdeployed")
        os.makedirs(dep)
        # The pre-fix wrapper, reconstructed from the shipped one by undoing the
        # single line that was repaired. Reconstructed rather than kept as a
        # second copy, so it cannot drift away from what was actually fixed.
        src = open(os.path.join(SHARED, "run_pack.py")).read()
        pre = src.replace('RUN_CELL = os.path.join(ARM, "run_cell.py")',
                          'RUN_CELL = os.path.join(HERE, "run_cell.py")')
        pre = pre.replace("\ncheck_executor()\n", "\n")
        if pre == src:
            print("    FAIL  cannot reconstruct the pre-fix wrapper: the repaired "
                  "line is not where this control expects it")
            return 1
        open(os.path.join(dep, "run_pack.py"), "w").write(pre)
        shutil.copyfile(os.path.join(SHARED, "run_cell.py"),
                        os.path.join(dep, "run_cell.py"))

        env = dict(os.environ, PPSQJ_REPO=tmp)

        # 1  the resolution fault
        r = subprocess.run([PY, os.path.join(dep, "run_pack.py"), "--resolve"],
                           cwd=arm, capture_output=True, text=True, env=env)
        got = ""
        for line in r.stdout.split("\n"):
            if line.startswith("RUN_CELL"):
                got = line.split(None, 1)[1].strip()
        step1 = os.path.realpath(got).startswith(os.path.realpath(dep))
        print("    %s  step 1  pre-fix wrapper resolves %s"
              % ("ok  " if step1 else "FAIL",
                 os.path.relpath(os.path.realpath(got), os.path.realpath(skel))
                 if got else "?"))
        fails.append(not step1)

        # 2  the observed failure, verbatim
        before = set(os.listdir(os.path.join(arm, "results"))) \
            if os.path.isdir(os.path.join(arm, "results")) else set()
        r = subprocess.run([PY, os.path.join(dep, "run_cell.py"), "0"], cwd=arm,
                           capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        after = set(os.listdir(os.path.join(arm, "results"))) \
            if os.path.isdir(os.path.join(arm, "results")) else set()
        step2 = (r.returncode != 0 and "FileNotFoundError" in out
                 and "manifest.csv" in out and before == after)
        print("    %s  step 2  shared executor + arm cwd -> rc=%d, %s, %d result "
              "JSONs written"
              % ("ok  " if step2 else "FAIL", r.returncode,
                 "FileNotFoundError on manifest.csv" if "FileNotFoundError" in out
                 else "NO FileNotFoundError", len(after - before)))
        fails.append(not step2)

        # 3  the repair
        r = subprocess.run([PY, os.path.join(SHARED, "run_pack.py"), "--resolve"],
                           cwd=arm, capture_output=True, text=True, env=env)
        got = ""
        for line in r.stdout.split("\n"):
            if line.startswith("RUN_CELL"):
                got = line.split(None, 1)[1].strip()
        step3 = os.path.realpath(got) == os.path.realpath(
            os.path.join(arm, "run_cell.py"))
        print("    %s  step 3  shipped wrapper resolves %s"
              % ("ok  " if step3 else "FAIL",
                 os.path.relpath(os.path.realpath(got), os.path.realpath(skel))
                 if got else "?"))
        fails.append(not step3)

        # 4  no silent fallback
        os.remove(os.path.join(arm, "run_cell.py"))
        r = subprocess.run([PY, os.path.join(SHARED, "run_pack.py"), "0"], cwd=arm,
                           capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        step4 = r.returncode != 0 and "no arm-local executor" in out
        print("    %s  step 4  arm-local executor removed -> shipped wrapper "
              "rc=%d, refuses to fall back to shared/"
              % ("ok  " if step4 else "FAIL", r.returncode))
        fails.append(not step4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if any(fails) else 0


def main():
    print("=" * 76)
    print("  NEGATIVE CONTROLS  TASK-2026-09-06-NC-ZETA-MOCKPROD")
    print("  Each injects one fault into a COPY of %s and requires the real"
          % os.path.basename(BASE_ARM))
    print("  preflight to reject it. The task directory itself is not modified.")
    print("=" * 76)

    rc0, _out0 = run_preflight(BASE_ARM)
    if rc0 != 0:
        print("  ABORT: the unmodified arm does not pass preflight, so a "
              "rejection below would prove nothing.")
        return 1
    # And the UNINJECTED STAGED COPY must pass too. Without this, a control could
    # be "rejected" by a check that the staging itself broke.
    tmp0 = tempfile.mkdtemp(prefix="negctl_base_")
    try:
        rc0b, out0b = run_preflight(stage_arm(tmp0))
    finally:
        shutil.rmtree(tmp0, ignore_errors=True)
    if rc0b != 0:
        print("  ABORT: the staged, UNINJECTED copy does not pass preflight, so "
              "every rejection below would be attributable to the staging:")
        print("\n".join("    " + l for l in out0b.split("\n")
                        if l.strip().startswith("FAIL")))
        return 1
    print("  baseline: %s passes clean, and so does an uninjected staged copy "
          "(exit 0)\n" % os.path.basename(BASE_ARM))

    bad = 0
    for label, fn in CONTROLS:
        tmp = tempfile.mkdtemp(prefix="negctl_")
        try:
            arm = stage_arm(tmp)
            want = fn(arm)
            rc, out = run_preflight(arm)
            codes = sorted(set(re.findall(r"FAIL  (\w+)", out)))
            if rc == 0:
                print("  %-4s FAILED TO FIRE   %-42s (preflight passed a broken arm)"
                      % (label, fn.__doc__.split("\n")[0][:42]))
                bad += 1
            elif want not in codes:
                print("  %-4s WRONG CODE       expected %s, got %s"
                      % (label, want, ", ".join(codes)))
                bad += 1
            else:
                print("  %-4s rejected by %-5s %s"
                      % (label, want, fn.__doc__.split("\n")[0][:46]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 76)
    print("  %d of %d controls behaved as required" % (len(CONTROLS) - bad, len(CONTROLS)))
    if bad:
        print("  A control that does not fire means the corresponding preflight "
              "check is decorative.")
    rep = bug_reproduction()
    print("-" * 76)
    print("  bug reproduction: %s" % ("ALL FOUR STEPS AS REQUIRED" if rep == 0
                                      else "DID NOT BEHAVE AS REQUIRED"))
    return 1 if (bad or rep) else 0


if __name__ == "__main__":
    sys.exit(main())
