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
PY = sys.executable
PREFLIGHT = os.path.join(TASK, "shared", "preflight.py")
BASE_ARM = os.path.join(TASK, "M_z010_nc512")


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


CONTROLS = [(f.__name__.upper(), f) for f in
            (n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14)]


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
    print("  baseline: %s passes clean (exit 0)\n" % os.path.basename(BASE_ARM))

    bad = 0
    for label, fn in CONTROLS:
        tmp = tempfile.mkdtemp(prefix="negctl_")
        try:
            arm = os.path.join(tmp, os.path.basename(BASE_ARM))
            shutil.copytree(BASE_ARM, arm)
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
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
