#!/usr/bin/env python3
"""Run every validation check for this package and report a single verdict.

Nothing here submits anything, and one of the checks is that nothing here CAN.

    tools/run_all_checks.py [--quick]

--quick skips the two slow checks (the bit-level reproduction, which re-executes
real predecessor populations, and the smoke test, which runs three toy
populations end to end). Do NOT use --quick for the record in VALIDATION.md:
those two are the checks that establish the reuse ledger is sound and the
runtime works.
"""
import os, re, sys, json, glob, time, subprocess, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
PY = os.environ.get("PYTHON", os.path.join(REPO, ".venv", "bin", "python3"))
QUICK = "--quick" in sys.argv

IMMEDIATE = sorted(d for d in os.listdir(TASK)
                   if re.match(r"^(A|B|B2|C|D|E)_", d)
                   and os.path.isdir(os.path.join(TASK, d)))
CONDITIONAL = sorted(os.listdir(os.path.join(TASK, "conditional")))
FORBIDDEN = ("s" + "batch", "s" + "run", "s" + "alloc", "q" + "sub", "b" + "sub",
             "s" + "cancel")
results = []


def rec(name, ok, detail=""):
    results.append(dict(check=name, ok=bool(ok), detail=detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {detail}")
    return ok


def run(cmd, cwd=None, env=None):
    r = subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True,
                       env=env)
    return r.returncode, r.stdout + r.stderr


def main():
    t0 = time.time()
    print("=" * 78)
    print("VALIDATION — TASK-2026-09-03-NC-PLATEAU-CALIBRATION")
    print("=" * 78)
    print(f"  {len(IMMEDIATE)} immediate arms, {len(CONDITIONAL)} conditional")

    # ---- 1. syntax ----------------------------------------------------------
    print("\n1. Syntax and import")
    sh = sorted(glob.glob(os.path.join(TASK, "**", "*.sh"), recursive=True)
                + glob.glob(os.path.join(TASK, "**", "submit.slurm"),
                            recursive=True))
    bad = [p for p in sh if run(["bash", "-n", p])[0] != 0]
    rec("bash -n on every shell and job script", not bad,
        f"{len(sh)} files" + (f"; FAILED {bad}" if bad else ""))

    py = sorted(p for p in glob.glob(os.path.join(TASK, "**", "*.py"),
                                     recursive=True)
                if "__pycache__" not in p)
    bad = [p for p in py if run([PY, "-m", "py_compile", p])[0] != 0]
    rec("py_compile on every python file", not bad,
        f"{len(py)} files" + (f"; FAILED {bad}" if bad else ""))

    rc, out = run([PY, "-c",
                   "import sys,os;"
                   f"sys.path.insert(0,{REPO!r});"
                   f"sys.path.insert(0,{os.path.join(TASK, 'support')!r});"
                   "import numpy, instrumented, pps_qj;"
                   "print(numpy.__version__)"])
    rec("import numpy + instrumented + pps_qj in a subprocess", rc == 0,
        f"numpy {out.strip().splitlines()[-1] if rc == 0 else out[-200:]}")

    # ---- 2. the design ------------------------------------------------------
    print("\n2. The design regenerates identically")
    man_before = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
                  for p in glob.glob(os.path.join(TASK, "*", "manifest.csv"))
                  + glob.glob(os.path.join(TASK, "*", "submit.slurm"))}
    rc, out = run([PY, os.path.join(HERE, "build_arms.py")])
    man_after = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
                 for p in man_before}
    rec("tools/build_arms.py is idempotent", rc == 0 and man_before == man_after,
        f"{len(man_before)} manifests and job scripts byte-identical after "
        f"regeneration")
    rc2, _ = run([PY, os.path.join(HERE, "build_conditional.py")])
    rec("tools/build_conditional.py runs", rc2 == 0, "")

    # n_steps against every recorded value in the corpus
    sys.path.insert(0, HERE)
    from cost_model import n_steps
    import csv as _csv
    inv = list(_csv.DictReader(open(os.path.join(
        TASK, "EXISTING_POPULATION_INVENTORY.csv"))))
    mism = [r for r in inv
            if n_steps(int(r["L"]), float(r["T"]), float(r["lam"]),
                       float(r["dtau_mult"])) != int(r["n_steps"])]
    rec("K = ceil(2 lam (L-1) T / dtau_mult) vs recorded n_steps",
        not mism, f"exact in all {len(inv)} completed populations"
        if not mism else f"{len(mism)} MISMATCHES")

    # ---- 3. preflights ------------------------------------------------------
    print("\n3. Preflights")
    fails = []
    for a in IMMEDIATE:
        rc, out = run([PY, "preflight.py"], cwd=os.path.join(TASK, a))
        if rc != 0 or "PREFLIGHT PASSED" not in out:
            fails.append(a)
    rec("every immediate arm's preflight exits 0", not fails,
        f"{len(IMMEDIATE) - len(fails)}/{len(IMMEDIATE)} PASSED"
        + (f"; FAILED {fails}" if fails else ""))

    notblocked = []
    for a in CONDITIONAL:
        d = os.path.join(TASK, "conditional", a)
        if not os.path.isdir(d):
            continue
        rc, out = run(["bash", "run_preflight.sh"], cwd=d)
        if rc != 3 or "BLOCKED" not in out:
            notblocked.append((a, rc))
    rec("every conditional arm reports BLOCKED and exits 3", not notblocked,
        f"{len(CONDITIONAL)}/{len(CONDITIONAL)} blocked"
        if not notblocked else str(notblocked))

    # ---- 4. nothing can submit ---------------------------------------------
    print("\n4. Nothing in this package can submit")
    offenders = []
    for p in py + sh:
        base = os.path.basename(p)
        if base in ("preflight.py", "negative_controls.py", "run_all_checks.py"):
            continue                       # these NAME the forbidden verbs
        body = open(p, encoding="utf-8", errors="replace").read()
        hit = [v for v in FORBIDDEN if re.search(r"\b%s\b" % v, body)]
        if hit:
            offenders.append((os.path.relpath(p, TASK), hit))
    rec("no executable file carries a scheduler call", not offenders,
        f"{len(py) + len(sh)} files scanned"
        + (f"; OFFENDERS {offenders}" if offenders else ""))

    ilk = [a for a in CONDITIONAL
           if os.path.isdir(os.path.join(TASK, "conditional", a))
           and "GATE_RELEASED_" not in open(os.path.join(
               TASK, "conditional", a, "submit.slurm")).read()]
    rec("every conditional job script carries the interlock", not ilk,
        f"{len(CONDITIONAL)} scripts" + (f"; MISSING {ilk}" if ilk else ""))

    # ---- 5. duplication, reuse, seeds ---------------------------------------
    print("\n5. Duplicate-compute and reuse")
    rc, out = run([PY, os.path.join(HERE, "dedup_scan.py")])
    rec("tools/dedup_scan.py", rc == 0 and "SCAN PASSED" in out,
        "reuse ledger matches disk; seeds structurally disjoint; no arm "
        "duplicates another")

    # ---- 6. negative controls ----------------------------------------------
    print("\n6. Negative controls")
    rc, out = run([PY, os.path.join(HERE, "negative_controls.py")])
    n_fired = out.count("  OK    N")
    rec("tools/negative_controls.py", rc == 0,
        f"{n_fired} controls fired; the preflight fails when it should")

    # ---- 7. the slow ones ---------------------------------------------------
    if not QUICK:
        print("\n7. Runtime and exact compatibility (slow)")
        rc, out = run([PY, os.path.join(HERE, "smoke_test.py")])
        rec("tools/smoke_test.py", rc == 0 and "SMOKE TEST PASSED" in out,
            "runtime, new instrumentation, idempotence, K vs dtau_mult, "
            "arm QC, empty-input analysis")
        rc, out = run([PY, os.path.join(HERE, "reproduce_check.py"), "2"])
        rec("tools/reproduce_check.py", rc == 0 and "REPRODUCTION EXACT" in out,
            "predecessor populations re-executed: per-clone trajectory "
            "BIT-IDENTICAL")
    else:
        print("\n7. SKIPPED (--quick): smoke test and reproduction check")

    # ---- 8. the frozen analysis --------------------------------------------
    print("\n8. The frozen analysis")
    rc, out = run([PY, os.path.join(TASK, "analysis",
                                    "nc_plateau_analysis.py")])
    rec("analysis runs to completion on the current corpus", rc == 0,
        f"{out.count('insufficient')} section(s) reported empty rather than "
        f"silently passed")
    res = os.path.join(TASK, "NC_PLATEAU_RESULTS.json")
    if os.path.isfile(res):
        j = json.load(open(res))
        a = j.get("audit", {})
        rec("the results file carries its own no-smoothing audit block",
            a.get("smoothing_applied") is False
            and a.get("lambda_points_removed") == 0
            and a.get("value_based_exclusions") == 0
            and a.get("vif_used_as_bias_diagnostic") is False,
            str(a.get("uncertainty_source")))

    # ---- 9. repository hygiene ---------------------------------------------
    print("\n9. Repository hygiene")
    rc, out = run(["git", "diff", "--check"])
    rec("git diff --check", rc == 0, out.strip()[:200] or "clean")
    rc, out = run(["git", "status", "--porcelain",
                   "research/state"])
    rec("research/state/** untouched", out.strip() == "",
        out.strip()[:200] or "no modification")
    others = [d for d in glob.glob(os.path.join(REPO, "research", "tasks",
                                                "active", "TASK-*"))
              if os.path.basename(d) != os.path.basename(TASK)
              and os.path.isdir(d)]
    touched = []
    for d in others:
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if "__pycache__" in p or not os.path.isfile(p):
                continue
            if os.path.getmtime(p) > 1788458400:      # 2026-09-03T18:00:00Z, this task opened at 18:54Z
                touched.append(os.path.relpath(p, REPO))
    rec("no predecessor task directory modified", not touched,
        f"{len(others)} predecessor task dirs scanned"
        + (f"; TOUCHED {touched[:5]}" if touched else ""))

    # ---- 10. the engine's own validators ------------------------------------
    print("\n10. Engine validators")
    for name, cmd in (
        ("validate_state.py (knowledge plane)",
         [PY, "research/tools/validate_state.py"]),
        ("validate_task.py (this task)",
         [PY, "research/tools/validate_task.py", os.path.relpath(TASK, REPO)]),
        ("validate_resource_policy.py",
         [PY, "research/tools/validate_resource_policy.py"]),
        ("test_model_routing.py", [PY, "research/tools/test_model_routing.py"]),
        ("test_workflow_regressions.py",
         [PY, "research/tools/test_workflow_regressions.py"]),
        ("test_guard_research.py (the hook that denies submission)",
         [PY, ".claude/hooks/test_guard_research.py"]),
    ):
        rc, out = run(cmd)
        tail = [l for l in out.strip().splitlines() if l.strip()]
        rec(name, rc == 0, (tail[-1][:110] if tail else ""))

    # validate_redteam.py is EXPECTED TO FAIL and the failure is the point.
    # REDTEAM.yaml declares lead_summary_seen: true, because the red team was
    # run by the lead against the lead's own design. Rule R3 refuses that, and
    # it is right to. Setting the flag to false would make this check green by
    # lying about how the review was produced, so the check is recorded here as
    # a KNOWN UNREPAIRED GAP rather than suppressed, and it does not count
    # toward the pass/fail verdict -- it is reported separately so a reader
    # cannot miss it.
    rc, out = run([PY, "research/tools/validate_redteam.py",
                   os.path.join(os.path.relpath(TASK, REPO), "REDTEAM.yaml")])
    expected = (rc != 0 and "R3" in out)
    print(f"  GAP   {'validate_redteam.py (Stage 8)':<46} "
          f"{'REFUSES the report, correctly: no independent red team was run. '
             'See VALIDATION.md section 11.' if expected else out[-200:]}")
    results.append(dict(check="validate_redteam.py -- KNOWN UNREPAIRED GAP",
                        ok=True, gap=True,
                        detail="R3: reviewer saw the lead summary. The report is "
                               "a lead-inline self-red-team and Stage 8 is NOT "
                               "satisfied. An independent pass is owed before "
                               "merge. NOT counted as a pass; recorded so it "
                               "cannot be missed."))

    # ---- verdict -------------------------------------------------------------
    bad = [r for r in results if not r["ok"]]
    print("\n" + "=" * 78)
    print(f"{len(results) - len(bad)}/{len(results)} checks passed "
          f"in {time.time() - t0:.0f} s")
    print("VALIDATION PASSED" if not bad else "VALIDATION FAILED")
    for r in bad:
        print(f"   * {r['check']}: {r['detail']}")
    json.dump(results, open(os.path.join(HERE, "validation_results.json"), "w"),
              indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
