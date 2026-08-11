#!/usr/bin/env python3
"""test_post_stress_regressions.py - regressions for defects demonstrated by
TASK-2026-08-10-UNIVCLASS.

Every test here encodes a failure that ACTUALLY OCCURRED in that run, or a
guarantee that must not be lost while fixing one. Read-only with respect to
research/state/**; builds fixture tasks in a temp directory and deletes them.

Runs no simulation, launches no agent, submits nothing.

    .venv/bin/python3 research/tools/test_post_stress_regressions.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
TASK_TEMPLATE = os.path.join(REPO, "research", "tasks", "TASK_TEMPLATE")

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}  {detail}")


def run(*args):
    r = subprocess.run([PY, *args], capture_output=True, text=True, cwd=REPO)
    return r.returncode, r.stdout + r.stderr


def validate_task(task):
    return run(os.path.join(HERE, "validate_task.py"), task)


def new_task(tmp, name="TASK-2099-01-01-FIXTURE"):
    d = os.path.join(tmp, name)
    shutil.copytree(TASK_TEMPLATE, d)
    run(os.path.join(HERE, "task_phase.py"), d, "init", name, "--mode", "normal")
    return d


def write(path, doc):
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


def read_text(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1-2. EXT-* evidence resolution  (stress-test finding IF-6)
# ---------------------------------------------------------------------------
def test_ext_evidence(tmp):
    task = new_task(tmp, "TASK-2099-01-01-EXT")
    write(os.path.join(task, "TASK_EVIDENCE.yaml"), {
        "schema_version": 1, "task_id": "TASK-2099-01-01-EXT",
        "task_verified": [{
            "id": "TV-1", "tier": "task_verified", "kind": "source_inspection",
            "source_or_artifact": "SRC-X", "what_was_verified": "a thing",
            "what_it_does_not_establish": "another thing",
            "verified_by": "literature", "date": "2099-01-01",
            "canonical": False, "promotion_status": "proposed"}],
        "external_sources": [{
            "id": "EXT-1", "title": "A real paper", "authors": "Someone",
            "year": 2099, "inspection_level": "relevant_sections",
            "sections_inspected": "Sec. III",
            "what_it_establishes": "a specific result",
            "what_it_does_not_establish": "anything about our model",
            "verified_by": "red-team", "date": "2099-01-01",
            "used_as_support": True, "promotion_status": "proposed"}],
    })
    rt = os.path.join(task, "REDTEAM.yaml")
    write(rt, {"schema_version": 2, "task_id": "TASK-2099-01-01-EXT",
               "inputs_seen": {"claims": [], "evidence": [],
                               "task_verified": ["TV-1"],
                               "external_sources": ["EXT-1"],
                               "lead_summary_seen": False}})
    _c, out = validate_task(task)
    check("1  red-team citation of a valid EXT-* resolves",
          "E1" not in out and "EXT-1" not in out.split("W1")[0],
          out)

    # THE EXACT STRESS-TEST FAILURE: EXT-* cited, present under external_sources.
    # Before the fix this raised E1 because only task_verified was indexed.
    check("1b ... this is the exact TASK-2026-08-10-UNIVCLASS IF-6 case",
          "cites task-verified item 'EXT-1'" not in out, out)

    d = yaml.safe_load(read_text(rt))
    d["inputs_seen"]["external_sources"] = ["EXT-NOPE"]
    write(rt, d)
    _c, out = validate_task(task)
    check("2  unknown EXT-* still fails", "E1" in out and "EXT-NOPE" in out, out)

    # A discovery-level source may be recorded but not used as support.
    te = os.path.join(task, "TASK_EVIDENCE.yaml")
    doc = yaml.safe_load(read_text(te))
    doc["external_sources"][0]["inspection_level"] = "search_result_snippet"
    doc["external_sources"][0]["used_as_support"] = True
    write(te, doc)
    _c, out = validate_task(task)
    check("2b abstract/snippet-level source cannot be used_as_support",
          "E5" in out, out)


# ---------------------------------------------------------------------------
# 3-4. Nested predecessor indexing  (stress-test finding IF-4)
# ---------------------------------------------------------------------------
def test_predecessor_nested(_tmp):
    q = ("Importing the Fulga class DIII exponent as the Born rule MIPT "
         "exponent is a replica limit misuse")
    _c, out = run(os.path.join(HERE, "find_predecessors.py"), q, "--top", "5")
    check("3  nested DEC-CITATION-001.items[] is indexed and surfaces",
          "DEC-CITATION-001" in out, out[:800])
    check("4  the matched nested field path is reported",
          "matched on: DEC-CITATION-001.items[" in out, out[:800])

    # The C2 / METH-EXTRAP-001 regression must keep passing.
    _c, out2 = run(os.path.join(HERE, "find_predecessors.py"),
                   "1/sqrt(L) extrapolation of crossings with a chi2/dof "
                   "comparison across correction forms", "--top", "3")
    check("4b C2 regression: withdrawn METH-EXTRAP-001 still ranks first",
          out2.index("METH-EXTRAP-001") < (out2.index("CB-AMP-096-001")
                                           if "CB-AMP-096-001" in out2 else 10**9),
          out2[:600])

    # Dead records still boosted, and indexing must not explode context.
    check("4c dead records are still marked, not filtered",
          "DEAD RECORD" in out2, out2[:400])


# ---------------------------------------------------------------------------
# 5-6. Frozen source scope vs append-only inspection  (finding IF-1)
# ---------------------------------------------------------------------------
def test_source_scope_split(tmp):
    task = new_task(tmp, "TASK-2099-01-01-SRC")
    for n in ("CHARTER.md", "PROBLEM_MEMO.md", "SOURCE_REGISTER.md"):
        with open(os.path.join(task, n), "a", encoding="utf-8") as fh:
            fh.write("\nfilled in for the fixture\n")
    _c, out = run(os.path.join(HERE, "task_phase.py"), task, "close",
                  "stage_1_problem")
    man = yaml.safe_load(read_text(os.path.join(task, "TASK_MANIFEST.yaml")))
    frozen = {}
    for ph in man["phases"]:
        if ph["stage"] == "stage_1_problem":
            frozen = ph.get("frozen") or {}
    before = frozen.get("SOURCE_REGISTER.md")
    check("5a ledger declares the append-only tier",
          "SOURCE_INSPECTIONS.yaml" in (man.get("append_only_artifacts") or []),
          str(man.get("append_only_artifacts")))
    check("5b ledger names the frozen source scope",
          man.get("frozen_source_scope") == "SOURCE_REGISTER.md", str(man))

    # Normal post-Stage-1 source inspection: append, do not touch the scope.
    ins = os.path.join(task, "SOURCE_INSPECTIONS.yaml")
    write(ins, {"schema_version": 1, "task_id": "TASK-2099-01-01-SRC",
                "inspections": [{
                    "id": "INS-1", "source_id": "SRC-X", "external": False,
                    "title": "A paper", "locator": "J. Foo 1, 1",
                    "sections_read": "Sec. 2, pp. 3-5",
                    "inspection_level": "relevant_sections",
                    "what_it_establishes": "a specific result",
                    "what_it_does_not_establish": "anything else",
                    "verified_by": "literature", "date": "2099-01-02",
                    "derived_task_evidence": ["TV-1"],
                    "promotion_status": "proposed"}],
                "could_not_inspect": []})
    _c, out = validate_task(task)
    man2 = yaml.safe_load(read_text(os.path.join(task, "TASK_MANIFEST.yaml")))
    after = None
    for ph in man2["phases"]:
        if ph["stage"] == "stage_1_problem":
            after = (ph.get("frozen") or {}).get("SOURCE_REGISTER.md")
    check("5c Stage-1 scope hash unchanged by a normal inspection",
          before == after and before is not None, f"{before} vs {after}")
    check("5d no M5 from a normal post-Stage-1 inspection",
          "M5" not in out, out)
    check("5e and no amendment was needed",
          not (man2.get("amendments") or []), str(man2.get("amendments")))

    # Editing the Stage-1 scope itself STILL fails. The freeze is not weakened.
    with open(os.path.join(task, "SOURCE_REGISTER.md"), "a", encoding="utf-8") as fh:
        fh.write("\nsneaking in a new source after the freeze\n")
    _c, out = validate_task(task)
    check("6  editing the Stage-1 source scope still produces M5",
          "M5" in out and "SOURCE_REGISTER.md" in out, out)

    # And the tool refuses to freeze/amend an append-only artifact at all.
    c, out = run(os.path.join(HERE, "task_phase.py"), task, "amend",
                 "SOURCE_INSPECTIONS.yaml", "--reason", "x",
                 "--authorised-by", "y")
    check("6b task_phase refuses to amend an append-only artifact",
          c != 0 and "append-only" in out, out)


# ---------------------------------------------------------------------------
# 7-8. Method-aware independence
# ---------------------------------------------------------------------------
def _indep_task(tmp, name, reps_a, reps_b, varied, cls):
    task = new_task(tmp, name)
    with open(os.path.join(task, "RESEARCH_MEMO.md"), "a", encoding="utf-8") as fh:
        fh.write("\nThe result was independently verified by a second worker.\n")
    write(os.path.join(task, "INDEPENDENCE_LEDGER.yaml"), {
        "schema_version": 1, "task_id": name,
        "verifications": [{
            "id": "IV-1", "claim": "no zeta=0 data exists anywhere",
            "original_check": {"performed_by": "numerics",
                               "method": "directory-name scan",
                               "source_representation": reps_a,
                               "command": "ls ... | sed ..."},
            "independent_check": {"performed_by": "lead",
                                  "method": "content scan of zeta fields",
                                  "source_representation": reps_b,
                                  "command": "python3 scan.py"},
            "shared_assumptions": ["target is stored as cloning output"],
            "varied_assumptions": varied,
            "independence": cls,
            "justification": "fixture",
            "outcome": "confirmed"}]})
    return task


def test_independence(tmp):
    # THE STRESS-TEST CASE: different commands, same representation family.
    task = _indep_task(tmp, "TASK-2099-01-01-IND1",
                       ["csv", "json"], ["csv", "json"], [],
                       "methodologically_independent")
    _c, out = validate_task(task)
    check("7  same source representation cannot be methodologically_independent",
          "V4" in out, out)
    check("7b ... and the reason names the shared representation",
          "SAME source representation" in out, out)

    # Genuinely independent: different representation AND a varied assumption.
    task = _indep_task(tmp, "TASK-2099-01-01-IND2",
                       ["csv", "json"], ["markdown", "prose", "git_history"],
                       ["that the target is stored in a tabular format"],
                       "methodologically_independent")
    _c, out = validate_task(task)
    check("8  differing representation + varied assumption IS independent",
          "V4" not in out, out)

    # Independence language with no ledger at all.
    task = new_task(tmp, "TASK-2099-01-01-IND3")
    with open(os.path.join(task, "RESEARCH_MEMO.md"), "a", encoding="utf-8") as fh:
        fh.write("\nThis was independently confirmed.\n")
    os.remove(os.path.join(task, "INDEPENDENCE_LEDGER.yaml"))
    _c, out = validate_task(task)
    check("8b independence language with no ledger fails", "V1" in out, out)


# ---------------------------------------------------------------------------
# 9-10. Child tasks: no post-hoc insertion, provenance without conclusions
# ---------------------------------------------------------------------------
def test_child_task(tmp):
    parent = new_task(tmp, "TASK-2099-01-01-PARENT")
    for n in ("CHARTER.md", "PROBLEM_MEMO.md", "SOURCE_REGISTER.md"):
        with open(os.path.join(parent, n), "a", encoding="utf-8") as fh:
            fh.write("\nfixture\n")
    run(os.path.join(HERE, "task_phase.py"), parent, "close", "stage_1_problem")
    run(os.path.join(HERE, "task_phase.py"), parent, "dispatch",
        "--worker", "theory=sonnet", "--skip", "numerics", "--skip", "literature")
    with open(os.path.join(parent, "agent_reports", "theory.md"), "w",
              encoding="utf-8") as fh:
        fh.write("# theory\nfindings\n")
    run(os.path.join(HERE, "task_phase.py"), parent, "close", "first_pass_frozen")
    run(os.path.join(HERE, "task_phase.py"), parent, "close", "stage_3_candidates")

    # A post-hoc analysis appended to the now-frozen spec must be caught.
    with open(os.path.join(parent, "ANALYSIS_SPEC.yaml"), "a", encoding="utf-8") as fh:
        fh.write("\n# post-hoc analysis added after the freeze\n")
    _c, out = validate_task(parent)
    check("9  post-hoc analysis cannot enter a frozen ANALYSIS_SPEC",
          "M5" in out and "ANALYSIS_SPEC.yaml" in out, out)

    c, out = run(os.path.join(HERE, "child_task.py"), "propose", parent,
                 "--child-id", "TASK-2099-02-02-CHILD")
    prop = os.path.join(parent, "proposed", "CHILD_TASK_TASK-2099-02-02-CHILD.yaml")
    check("10a a child-task proposal is written into the parent's proposed/",
          c == 0 and os.path.isfile(prop), out)
    check("10b proposing runs no analysis",
          "No analysis has been run" in out, out)

    # Unapproved / incomplete proposal must be refused.
    c, out = run(os.path.join(HERE, "child_task.py"), "init",
                 os.path.join(tmp, "TASK-2099-02-02-CHILD"), "--from", prop)
    check("10c an unapproved proposal cannot create a child task",
          c != 0 and "K7" in out, out)

    d = yaml.safe_load(read_text(prop))
    d.update({
        "originating": {"kind": "redteam_attack", "ref": "REDTEAM.yaml#C5.A4",
                        "what_it_said": "a discriminating follow-up exists"},
        "decision_relevance": "decides which experiment is worth designing",
        "why_not_in_parent": "the parent's ANALYSIS_SPEC was already frozen",
        "discriminating_result": {
            "hypotheses": ["matches value A", "matches value B", "neither"],
            "what_each_outcome_excludes": "one of the two identifications",
            "what_would_make_it_uninformative": "if the estimator is unstable at accessible L"},
        "existing_data_sufficient": True,
        "expected_compute_tier": "T0",
        "new_simulation_required": False,
        "fresh_analysis_plan_required": True,
        "inherits_from_parent": {"provenance": True, "conclusions": False},
        "human_approval": {"approved_by": "utku", "date": "2099-02-02"},
    })
    write(prop, d)
    child = os.path.join(tmp, "TASK-2099-02-02-CHILD")
    c, out = run(os.path.join(HERE, "child_task.py"), "init", child,
                 "--from", prop)
    check("10d an approved proposal creates the child task", c == 0, out)
    check("10e creating the child runs no analysis",
          "NO ANALYSIS HAS BEEN RUN" in out, out)

    prov = yaml.safe_load(read_text(os.path.join(child, "PARENT_PROVENANCE.yaml")))
    check("10f child records parent_task_id",
          prov.get("parent_task_id") == "TASK-2099-01-01-PARENT", str(prov)[:200])
    check("10g child inherits provenance but NOT conclusions",
          prov["inherits_from_parent"]["provenance"] is True
          and prov["inherits_from_parent"]["conclusions"] is False, str(prov)[:200])
    check("10h child did not inherit the parent's frozen ANALYSIS_SPEC",
          read_text(os.path.join(child, "ANALYSIS_SPEC.yaml"))
          != read_text(os.path.join(parent, "ANALYSIS_SPEC.yaml")), "")
    c, out = run(os.path.join(HERE, "child_task.py"), "check", child,
                 "--parent-dir", parent)
    check("10i child_task check passes on a clean child", c == 0, out)

    # ... and FIRES when a child really does carry a parent conclusion forward.
    with open(os.path.join(parent, "RESEARCH_MEMO.md"), "w", encoding="utf-8") as fh:
        fh.write("# parent memo\nThe parent concluded X.\n")
    shutil.copy(os.path.join(parent, "RESEARCH_MEMO.md"),
                os.path.join(child, "RESEARCH_MEMO.md"))
    c, out = run(os.path.join(HERE, "child_task.py"), "check", child,
                 "--parent-dir", parent)
    check("10i2 ... and fails when a filled-in parent conclusion is inherited",
          c != 0 and "K8" in out and "RESEARCH_MEMO.md" in out, out)

    # The parent must be untouched apart from proposed/.
    pman = yaml.safe_load(read_text(os.path.join(parent, "TASK_MANIFEST.yaml")))
    check("10j proposing a child made no amendment to the parent",
          not (pman.get("amendments") or []), str(pman.get("amendments")))


# ---------------------------------------------------------------------------
# 11. Implication-strength guard
# ---------------------------------------------------------------------------
def _strength_task(tmp, name, entry):
    task = new_task(tmp, name)
    write(os.path.join(task, "CLAIM_STRENGTH_AUDIT.yaml"),
          {"schema_version": 1, "task_id": name, "conclusions": [entry]})
    return task


def test_claim_strength(tmp):
    base = {
        "id": "CS-1",
        "conclusion": "the two limits are in different universality classes",
        "evidence_directly_establishes": "the two ensembles are different measures",
        "inference_adds": "that this implies different fixed points",
        "strongest_justified_wording": "direct transfer is not established",
        "stronger_wording_rejected": "they cannot share a universality class",
        "ladder": {"established_level": 1, "claimed_level": 4},
    }
    task = _strength_task(tmp, "TASK-2099-01-01-CS1", dict(base))
    _c, out = validate_task(task)
    check("11  microscopic inequivalence cannot become a class claim for free",
          "L4" in out, out)

    e = dict(base)
    e["ladder"] = dict(base["ladder"],
                       additional_inference_step="an RG argument, stated here",
                       inference_risk="the argument may fail if X")
    task = _strength_task(tmp, "TASK-2099-01-01-CS2", e)
    _c, out = validate_task(task)
    check("11b ... but IS allowed with an explicit declared step",
          "L4" not in out, out)

    # One exponent cannot establish sameness of class.
    e2 = dict(base)
    e2["ladder"] = dict(base["ladder"],
                        additional_inference_step="s", inference_risk="r")
    e2["exponent_check"] = {
        "exponents_compared": ["nu"], "n_exponents_compared": 1,
        "observable_matched": True, "convention_matched": True,
        "scaling_regime_matched": "unknown",
        "uncertainty_comparison_valid": True,
        "wording_class": "does_not_discriminate_with_current_evidence",
        "basis": "our uncertainty exceeds the gap"}
    task = _strength_task(tmp, "TASK-2099-01-01-CS3", e2)
    _c, out = validate_task(task)
    check("11c a level-4 claim on one exponent is rejected", "L6" in out, out)

    # `establishes_difference` needs all four match conditions.
    e3 = dict(base)
    e3["ladder"] = {"established_level": 4, "claimed_level": 4}
    e3["exponent_check"] = dict(e2["exponent_check"],
                                n_exponents_compared=3,
                                wording_class="establishes_difference")
    task = _strength_task(tmp, "TASK-2099-01-01-CS4", e3)
    _c, out = validate_task(task)
    check("11d 'establishes_difference' requires a matched scaling regime",
          "L7" in out, out)


# ---------------------------------------------------------------------------
# 12. Matched-observable checks for comparisons
# ---------------------------------------------------------------------------
BASE_ANALYSIS = {
    "id": "AN-1", "purpose": "compare endpoints", "role": "primary",
    "evidence_id": "TV-1", "observable_id": "OBS-X", "parameterization": "lambda",
    "pair_selection_rule": "wide", "crossing_definition": "sign change",
    "interpolation": "linear", "aggregation": "median", "fitting_window": "all",
    "weighting": "unweighted", "uncertainty_model": "SEM",
    "finite_size_extrapolation": "none",
    "validity_rule": {"unique_crossing": "required"},
    "crossing_classification": {"n_valid": 1, "n_ambiguous": 0, "n_invalid": 0},
    "result": {"value": 1.0},
}


def test_matched_observable(tmp):
    task = new_task(tmp, "TASK-2099-01-01-MO1")
    a = dict(BASE_ANALYSIS, compares=["born_endpoint", "forced_endpoint"])
    write(os.path.join(task, "ANALYSIS_SPEC.yaml"),
          {"schema_version": 1, "task_id": "x", "analyses": [a]})
    _c, out = validate_task(task)
    check("12  a comparison with no matched_observable_check fails",
          "N7" in out, out)

    dims = [{"name": n, "endpoint_a": "S_AB_mean", "endpoint_b": "S_half",
             "equivalent": True, "basis": "read from both scripts"}
            for n in ("subsystem_cut", "entropy_definition", "time_convention",
                      "averaging", "boundary_conditions", "parameterization",
                      "fitting_window", "finite_size_set")]
    a2 = dict(a, matched_observable_check={
        "dimensions": dims, "all_dimensions_equivalent": True})
    write(os.path.join(task, "ANALYSIS_SPEC.yaml"),
          {"schema_version": 1, "task_id": "x", "analyses": [a2]})
    _c, out = validate_task(task)
    check("12b omitting a convention dimension fails",
          "N8" in out and "normalization" in out, out)

    dims_full = dims + [{"name": "normalization", "endpoint_a": "n", "endpoint_b": "n",
                         "equivalent": "unknown", "basis": "not established"}]
    a3 = dict(a, matched_observable_check={
        "dimensions": dims_full, "all_dimensions_equivalent": True})
    write(os.path.join(task, "ANALYSIS_SPEC.yaml"),
          {"schema_version": 1, "task_id": "x", "analyses": [a3]})
    _c, out = validate_task(task)
    check("12c claiming all-equivalent while one is unknown fails",
          "N9" in out, out)

    a4 = dict(a, matched_observable_check={
        "dimensions": dims_full, "all_dimensions_equivalent": False,
        "unresolved_dimensions": ["normalization"],
        "consequence_if_unresolved": "the comparison is indicative only"})
    write(os.path.join(task, "ANALYSIS_SPEC.yaml"),
          {"schema_version": 1, "task_id": "x", "analyses": [a4]})
    _c, out = validate_task(task)
    check("12d an honest unresolved dimension with a stated consequence passes",
          "N7" not in out and "N8" not in out and "N9" not in out, out)


# ---------------------------------------------------------------------------
# 8b. Estimator semantics vs estimator defect
# ---------------------------------------------------------------------------
def test_estimator_semantics(tmp):
    task = new_task(tmp, "TASK-2099-01-01-ESS1")
    with open(os.path.join(task, "RESEARCH_MEMO.md"), "a", encoding="utf-8") as fh:
        fh.write("\nThe reported ESS is wrong at large L.\n")
    _c, out = validate_task(task)
    check("Q1  calling a diagnostic defective without its semantics fails",
          "Q1" in out, out)

    task = new_task(tmp, "TASK-2099-01-01-ESS2")
    with open(os.path.join(task, "RESEARCH_MEMO.md"), "a", encoding="utf-8") as fh:
        fh.write("\nThe existing ESS diagnostic does not detect severe "
                 "genealogical degeneracy at large L.\n")
    _c, out = validate_task(task)
    check("Q1b the preferred coverage wording passes", "Q1" not in out, out)


# ---------------------------------------------------------------------------
# 13-15. The suites that must keep passing
# ---------------------------------------------------------------------------
def test_existing_suites(_tmp):
    for name, path in (
        ("13  workflow regressions", os.path.join(HERE, "test_workflow_regressions.py")),
        ("14  collaboration / external", os.path.join(HERE, "test_collab_and_external.py")),
        ("15a guard hook", os.path.join(REPO, ".claude", "hooks", "test_guard_research.py")),
        # Adaptive model routing (RESOURCE_POLICY 5.4). Static + stubbed; it
        # launches no model call and costs nothing to run here.
        ("15d model routing", os.path.join(HERE, "test_model_routing.py")),
    ):
        c, out = run(path)
        check(f"{name} still pass", c == 0, out[-400:])
    for name, path in (
        ("15b validate_state", os.path.join(HERE, "validate_state.py")),
        ("15c validate_resource_policy", os.path.join(HERE, "validate_resource_policy.py")),
    ):
        c, out = run(path)
        check(f"{name} clean", c == 0, out[-400:])


def main():
    tmp = tempfile.mkdtemp(prefix="pss_regress_")
    try:
        for fn in (test_ext_evidence, test_predecessor_nested,
                   test_source_scope_split, test_independence, test_child_task,
                   test_claim_strength, test_matched_observable,
                   test_estimator_semantics, test_existing_suites):
            print(f"\n--- {fn.__name__} ---")
            fn(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{passed}/{passed + failed} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
