#!/usr/bin/env python3
"""test_workflow_regressions.py - the seven defects found in the external review
of TASK-2026-08-10-AMP096, each turned into a test that fails if it comes back.

    .venv/bin/python3 research/tools/test_workflow_regressions.py

Read-only with respect to the repository: every test builds a throwaway task
directory under a temp dir. Runs no agents, no workflows, no simulations, and
never touches research/state/**.

  R1  a Stage-1 artifact modified after investigator dispatch      -> FAIL
  R2  the pre-specified falsification plan modified after candidates -> FAIL
  R2b a plan carrying a results column                             -> FAIL
  R3  a C2-equivalent candidate must find METH-EXTRAP-001          -> rediscovery
  R4  task-verified evidence is consumable by the red team without a merge
  R5  mixed outcomes: 4 killed + 1 survives_scoped, not one global kill
  R6  a crossing at the final grid point / multiple sign changes is invalid
  R7  CB-AMP-096-001's direct provenance is surfaced at orientation
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
TEMPLATE = os.path.join(ROOT, "research", "tasks", "TASK_TEMPLATE")

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail and not ok else ""))


def run(*args, cwd=ROOT):
    return subprocess.run([PY, *args], capture_output=True, text=True, cwd=cwd)


def tool(name):
    return os.path.join(ROOT, "research", "tools", name)


def fresh_task(tmp, task_id="TASK-TEST-0001"):
    """A task directory from the real template, with the ledger initialised."""
    d = os.path.join(tmp, task_id)
    shutil.copytree(TEMPLATE, d)
    os.remove(os.path.join(d, "README.md"))
    run(tool("task_phase.py"), d, "init", task_id)
    return d


def write(d, rel, text):
    p = os.path.join(d, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


# ---------------------------------------------------------------------------
# R1  Stage-1 artifact modified after investigator dispatch
# ---------------------------------------------------------------------------
def test_r1(tmp):
    d = fresh_task(tmp, "TASK-TEST-R1")
    write(d, "PROBLEM_MEMO.md", "# memo\n\n[E] The question is X.\n")
    write(d, "CHARTER.md", "# charter\n\nkill criterion: X\n")
    write(d, "SOURCE_REGISTER.md", "# sources\n")
    run(tool("task_phase.py"), d, "close", "stage_1_problem")
    r = run(tool("task_phase.py"), d, "dispatch", "--worker", "theory=sonnet")
    dispatched = r.returncode == 0

    # the defect: backfill the memo with what the investigators found
    write(d, "PROBLEM_MEMO.md",
          "# memo\n\n[E] The question is X.\n[E] Three independent "
          "investigators found that the recorded reason is contradicted.\n")

    chk = run(tool("task_phase.py"), d, "check")
    v = run(tool("validate_task.py"), d)
    check(dispatched and chk.returncode == 1 and "MODIFIED" in chk.stdout,
          "R1  Stage-1 memo backfilled after dispatch -> phase check FAILS",
          chk.stdout.strip()[:300])
    check("M5" in v.stdout and "PROBLEM_MEMO.md" in v.stdout,
          "R1  ... and validate_task reports M5",
          v.stdout.strip()[:300])

    # the escape hatch works, and is recorded
    am = run(tool("task_phase.py"), d, "amend", "PROBLEM_MEMO.md",
             "--reason", "typo", "--authorised-by", "human")
    chk2 = run(tool("task_phase.py"), d, "check")
    man = open(os.path.join(d, "TASK_MANIFEST.yaml"), encoding="utf-8").read()
    check(am.returncode == 0 and chk2.returncode == 0
          and "authorised_by: human" in man,
          "R1  ... an amendment clears it and is recorded with its authoriser")


# ---------------------------------------------------------------------------
# R2  falsification plan modified after candidates froze / carries results
# ---------------------------------------------------------------------------
def test_r2(tmp):
    d = fresh_task(tmp, "TASK-TEST-R2")
    for rel in ("PROBLEM_MEMO.md", "CHARTER.md", "SOURCE_REGISTER.md"):
        write(d, rel, "# x\n")
    run(tool("task_phase.py"), d, "close", "stage_1_problem")
    run(tool("task_phase.py"), d, "dispatch", "--worker", "theory=sonnet")
    # first passes must be frozen before candidates can close (added with the
    # collaboration barrier; stage_3 now sits behind first_pass_frozen)
    write(d, "agent_reports/theory.json", '{"summary":"x"}')
    run(tool("task_phase.py"), d, "close", "first_pass_frozen")
    write(d, "FALSIFICATION_PLAN.md",
          "# plan\n\n| # | check | what would count as failing |\n"
          "|---|---|---|\n| 1 | endpoint | A != lambda_c(1) |\n")
    write(d, "CANDIDATES.md", "## Candidate C1\n1. **Statement.** x\n")
    run(tool("task_phase.py"), d, "close", "stage_3_candidates")

    # the defect: rewrite the pre-spec plan once the answers are known
    write(d, "FALSIFICATION_PLAN.md",
          "# plan\n\n| # | check | result |\n|---|---|---|\n"
          "| 1 | endpoint | yes, excluded 0.96 |\n")
    chk = run(tool("task_phase.py"), d, "check")
    v = run(tool("validate_task.py"), d)
    check(chk.returncode == 1 and "FALSIFICATION_PLAN.md" in chk.stdout,
          "R2  plan rewritten after candidates froze -> phase check FAILS")
    check("M5" in v.stdout and "FALSIFICATION_PLAN.md" in v.stdout,
          "R2  ... and validate_task reports M5")
    check("F1" in v.stdout,
          "R2b a results column in the PRE-SPECIFIED plan -> F1",
          v.stdout.strip()[:300])


# ---------------------------------------------------------------------------
# R3  the C2 rediscovery must be caught by the novelty gate
# ---------------------------------------------------------------------------
def test_r3(tmp):
    c2 = ("A = 0.96 and phi = 0.502 were co-generated by choosing the "
          "crossing-extrapolation exponent p = 1/2, a choice justified by "
          "assuming nu = 2")
    r = run(tool("find_predecessors.py"), c2, "--top", "5", "--json")
    hits = json.loads(r.stdout) if r.returncode == 0 else []
    ids = [h["id"] for h in hits]
    found = "METH-EXTRAP-001" in ids
    dead_flagged = any(h["id"] == "METH-EXTRAP-001" and h["dead_record"] for h in hits)
    check(found, "R3  C2-equivalent candidate surfaces METH-EXTRAP-001",
          f"top-5 was {ids}")
    check(dead_flagged,
          "R3  ... and it is flagged as a DEAD RECORD (withdrawn), not hidden")

    # the gate must also block novelty language without a classification
    d = fresh_task(tmp, "TASK-TEST-R3")
    write(d, "CANDIDATES.md",
          "## Candidate C1\n1. **Statement.** x\n\n**This is the finding of "
          "the task.**\n")
    write(d, "NOVELTY_GATE.md", "# gate\n\n| candidate | predecessor |\n")
    v = run(tool("validate_task.py"), d)
    check("G3" in v.stdout,
          "R3  novelty language with no gate classification -> G3")
    write(d, "NOVELTY_GATE.md",
          "# gate\n\n| candidate | predecessor | classification |\n"
          "| C1 | METH-EXTRAP-001 | rediscovery |\n")
    v2 = run(tool("validate_task.py"), d)
    check("G3" not in v2.stdout and "G2" not in v2.stdout,
          "R3  ... classifying C1 as rediscovery clears the gate",
          v2.stdout.strip()[:300])


# ---------------------------------------------------------------------------
# R4  task-verified evidence is consumable by the red team without a merge
# ---------------------------------------------------------------------------
def test_r4(tmp):
    d = fresh_task(tmp, "TASK-TEST-R4")
    write(d, "TASK_EVIDENCE.yaml", """schema_version: 1
task_id: TASK-TEST-R4
task_verified:
  - id: TV-1
    tier: task_verified
    kind: source_inspection
    source_or_artifact: SRC-LMR-2025
    locator: "body, zeta convention"
    what_was_verified: "zeta = 1 is the Born endpoint"
    verified_by: literature
    date: 2026-08-10
    canonical: false
    promotion_status: proposed
""")
    write(d, "REDTEAM.yaml", """schema_version: 2
task_id: TASK-TEST-R4
reviewer: red-team
inputs_seen:
  claims: []
  evidence: []
  task_verified: [TV-1]
  lead_summary_seen: false
candidate_reviews: {}
overall_task_assessment:
  surviving_candidates: []
  killed_candidates: []
  recommendation_basis: x
  proposed_status: none
""")
    v = run(tool("validate_task.py"), d)
    check("E1" not in v.stdout,
          "R4  red team consumes TV-1 with no canonical merge -> accepted")

    # ...but an unbacked citation is rejected
    write(d, "TASK_EVIDENCE.yaml", "schema_version: 1\ntask_verified: []\n")
    v2 = run(tool("validate_task.py"), d)
    check("E1" in v2.stdout,
          "R4  ... citing a TV id with no record -> E1")

    # ...and a task may not declare its own verification canonical
    write(d, "TASK_EVIDENCE.yaml", """schema_version: 1
task_verified:
  - id: TV-1
    tier: task_verified
    kind: source_inspection
    source_or_artifact: SRC-LMR-2025
    what_was_verified: x
    verified_by: literature
    date: 2026-08-10
    canonical: true
    promotion_status: promoted
""")
    v3 = run(tool("validate_task.py"), d)
    check("E3" in v3.stdout,
          "R4  ... declaring it canonical/promoted in-task -> E3")


# ---------------------------------------------------------------------------
# R5  mixed candidate outcomes
# ---------------------------------------------------------------------------
ATT = """      A1_already_solved_elsewhere: {attempted: true, finding: f, evidence: [], severity: %(s)s, unresolved: none, effect_on_candidate: %(e)s}
      A2_follows_trivially_from_assumptions: {attempted: true, finding: f, evidence: [], severity: none, unresolved: none, effect_on_candidate: none}
      A3_baseline_disadvantaged: {attempted: true, finding: f, evidence: [], severity: none, unresolved: none, effect_on_candidate: none}
      A4_gain_from_extra_information_or_resources: {attempted: true, finding: f, evidence: [], severity: none, unresolved: none, effect_on_candidate: none}
      A5_fails_under_dependence_causality_or_boundary_cases: {attempted: true, finding: f, evidence: [], severity: none, unresolved: none, effect_on_candidate: none}
      A6_measures_a_proxy_not_the_phenomenon: {attempted: true, finding: f, evidence: [], severity: none, unresolved: none, effect_on_candidate: none}
      A7_disappears_under_realistic_conditions: {attempted: true, finding: f, evidence: [], severity: none, unresolved: none, effect_on_candidate: none}
      A8_statistically_or_practically_negligible: {attempted: true, finding: f, evidence: [], severity: none, unresolved: none, effect_on_candidate: none}
      A9_simpler_explanation_accounts_for_evidence: {attempted: true, finding: f, evidence: [], severity: none, unresolved: none, effect_on_candidate: none}
"""


def _redteam_mixed(surviving, killed):
    out = ["schema_version: 2", "task_id: TASK-TEST-R5", "reviewer: red-team",
           "inputs_seen:", "  claims: []", "  evidence: []",
           "  task_verified: []", "  lead_summary_seen: false",
           "candidate_reviews:"]
    for cid in ("C1", "C2", "C3", "C5"):
        out.append(f"  {cid}:")
        out.append(f"    statement: s{cid}")
        out.append("    verdict: killed")
        out.append("    reason: r")
        out.append("    attacks:")
        out.append(ATT % {"s": "fatal", "e": "kill"})
    out += ["  C4:", "    statement: sC4", "    verdict: survives_scoped",
            "    reason: r", "    surviving_scope: one dataset only",
            "    attacks:", ATT % {"s": "minor", "e": "narrow_scope"}]
    out += ["overall_task_assessment:",
            f"  surviving_candidates: [{', '.join(surviving)}]",
            f"  killed_candidates: [{', '.join(killed)}]",
            "  recommendation_basis: mixed",
            "  proposed_status: provisional"]
    return "\n".join(out) + "\n"


def test_r5(tmp):
    d = fresh_task(tmp, "TASK-TEST-R5")
    p = os.path.join(d, "REDTEAM.yaml")

    open(p, "w").write(_redteam_mixed(["C4"], ["C1", "C2", "C3", "C5"]))
    r = run(tool("validate_redteam.py"), p)
    check(r.returncode == 0 and "schema v2" in r.stdout,
          "R5  4 killed + 1 survives_scoped validates as v2",
          r.stdout.strip()[:400])

    # the v1 failure mode: a global kill that erases the survivor
    open(p, "w").write(_redteam_mixed([], ["C1", "C2", "C3", "C4", "C5"]))
    r2 = run(tool("validate_redteam.py"), p)
    check(r2.returncode == 1 and "R10" in r2.stdout,
          "R5  ... claiming C4 was killed too -> R10 summary/verdict mismatch")

    # a fatal attack must kill its own candidate, not survive
    bad = _redteam_mixed(["C4"], ["C1", "C2", "C3", "C5"]).replace(
        "    verdict: survives_scoped", "    verdict: survives_scoped", 1)
    bad = bad.replace("severity: minor, unresolved: none, effect_on_candidate: narrow_scope",
                      "severity: fatal, unresolved: none, effect_on_candidate: kill")
    open(p, "w").write(bad)
    r3 = run(tool("validate_redteam.py"), p)
    check(r3.returncode == 1 and "R9" in r3.stdout,
          "R5  ... a fatal attack on C4 while C4 survives -> R9")

    # the legacy report still reads, and is labelled
    legacy = os.path.join(ROOT, "research/tasks/completed/"
                                "TASK-2026-08-10-AMP096/REDTEAM.yaml")
    if os.path.isfile(legacy):
        r4 = run(tool("validate_redteam.py"), legacy)
        check(r4.returncode == 0 and "LEGACY-V1" in r4.stdout,
              "R5  ... the historical v1 report still validates, labelled LEGACY-V1")


# ---------------------------------------------------------------------------
# R6  crossing validity
# ---------------------------------------------------------------------------
SPEC = """schema_version: 1
task_id: TASK-TEST-R6
analyses:
  - id: AN-1
    purpose: locate lambda_c
    role: primary
    evidence_id: EV-DATA-BOUNDARYCSV-001
    observable_id: OBS-BLPROD-001
    parameterization: lambda_c
    pair_selection_rule: wide pairs L2 >= 2*L1
    crossing_definition: sign change of B_L(L1) - B_L(L2)
    interpolation: linear
    fitting_window: {variable: zeta, range: [0.05, 0.85]}
    weighting: inverse-variance
    uncertainty_model: statistical only
    finite_size_extrapolation: none
    validity_rule:
      internally_bracketed: required
      not_at_scan_endpoint: required
      unique_crossing: required
      observable_not_collapsed: required
      collapse_factor: 3.0
      on_violation: exclude_and_report
    crossing_classification:
      cells:
        - cell: {zeta: 0.8, pair: [64, 128]}
          status: invalid
          reasons: [at_scan_endpoint, observable_collapsed]
          entered_primary_fit: %(entered)s
        - cell: {zeta: 0.05, pair: [64, 128]}
          status: ambiguous
          reasons: [five_sign_changes]
          entered_primary_fit: false
      n_valid: 10
      n_ambiguous: 1
      n_invalid: 1
      excluded_cells: 2
    result: {value: 0.494, uncertainty: 0.01, uncertainty_type: statistical}
"""


def test_r6(tmp):
    d = fresh_task(tmp, "TASK-TEST-R6")
    os.makedirs(os.path.join(d, "agent_reports"), exist_ok=True)
    write(d, "agent_reports/numerics.json", "{}")

    write(d, "ANALYSIS_SPEC.yaml", SPEC % {"entered": "false"})
    v = run(tool("validate_task.py"), d)
    check("N6" not in v.stdout and "N3" not in v.stdout and "N4" not in v.stdout,
          "R6  endpoint + multi-sign-change crossings marked invalid/ambiguous "
          "and excluded -> accepted", v.stdout.strip()[:300])

    write(d, "ANALYSIS_SPEC.yaml", SPEC % {"entered": "true"})
    v2 = run(tool("validate_task.py"), d)
    check("N6" in v2.stdout,
          "R6  ... an INVALID crossing entering the primary fit -> N6")

    # no spec at all, but numerics ran
    os.remove(os.path.join(d, "ANALYSIS_SPEC.yaml"))
    v3 = run(tool("validate_task.py"), d)
    check("N1" in v3.stdout,
          "R6  ... a numerics report with no ANALYSIS_SPEC -> N1")

    # a spec with no declared validity rule
    write(d, "ANALYSIS_SPEC.yaml",
          SPEC.split("    validity_rule:")[0] + "    result: {value: 1}\n")
    v4 = run(tool("validate_task.py"), d)
    check("N3" in v4.stdout,
          "R6  ... an analysis declaring no validity rule -> N3")


# ---------------------------------------------------------------------------
# R7  direct provenance surfaced at orientation
# ---------------------------------------------------------------------------
def test_r7(_tmp):
    r = run(tool("resolve_provenance.py"), "CB-AMP-096-001", "--json")
    try:
        rep = json.loads(r.stdout)[0]
    except Exception:
        check(False, "R7  provenance resolver runs on CB-AMP-096-001",
              r.stdout[:200] + r.stderr[:200])
        return
    docs = rep.get("documents_named_in_prose") or []
    names = " ".join(d["path"] for d in docs)
    check("Y_ZETA_DERIVATION.md" in names,
          "R7  the June r_c reinterpretation document is surfaced",
          f"surfaced: {names}")
    check("SESSION_2026_05_20.md" in names,
          "R7  ... and so is the May origin document")
    located = [d for d in docs if d["path"].endswith("Y_ZETA_DERIVATION.md")]
    check(bool(located) and (located[0]["exists"] or located[0].get("located_at")),
          "R7  ... a bare filename is resolved to a real path on disk",
          str(located[0].get("located_at")) if located else "not found")
    check("CB-AMP-001" in (rep.get("referenced_ids") or []),
          "R7  ... and the superseding claim is listed as a direct reference")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ppsqj_regress_") as tmp:
        for fn in (test_r1, test_r2, test_r3, test_r4, test_r5, test_r6, test_r7):
            print(f"\n--- {fn.__name__} ---")
            fn(tmp)
    passed = sum(1 for ok, _, _ in results if ok)
    print(f"\n{passed}/{len(results)} passed")
    for ok, name, detail in results:
        if not ok:
            print(f"  FAILED: {name}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
