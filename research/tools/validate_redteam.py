#!/usr/bin/env python3
"""Validate a red-team report against Charter section 7 Stage 8.
Every one of the nine mandated attacks must carry a complete verdict.
A missing attack is a HARD FAILURE. Read-only. Exit 0 clean, 1 on error.

TWO SCHEMAS ARE ACCEPTED.

  v2 (current): `candidate_reviews` maps each candidate to its OWN nine attacks
      and its OWN verdict, plus an `overall_task_assessment` that must agree
      with them. A fatal attack kills the candidate it applies to and cannot
      erase an unrelated survivor.

  v1 (legacy): one global `verdict` and one set of nine attacks. Reports written
      before 2026-08-10 use it. They are validated under the old rules and
      labelled LEGACY-V1. New reports must use v2: rule R11 rejects a v1 report
      that declares schema_version 2, and R12 warns on any new v1 report.
"""
from __future__ import annotations
import sys, os, glob
try: import yaml
except ImportError: sys.exit("PyYAML required")

ATTACKS = {
 "A1_already_solved_elsewhere": "problem already solved under another formulation",
 "A2_follows_trivially_from_assumptions": "result follows trivially from assumptions",
 "A3_baseline_disadvantaged": "the baseline is disadvantaged",
 "A4_gain_from_extra_information_or_resources": "gain comes from extra information or resources",
 "A5_fails_under_dependence_causality_or_boundary_cases": "fails under dependence, causality, or boundary cases",
 "A6_measures_a_proxy_not_the_phenomenon": "experiment measures a proxy, not the stated phenomenon",
 "A7_disappears_under_realistic_conditions": "contribution disappears under realistic operating conditions",
 "A8_statistically_or_practically_negligible": "result is statistically or practically negligible",
 "A9_simpler_explanation_accounts_for_evidence": "a simpler explanation accounts for the evidence",
}
SEV = {"none","minor","material","fatal"}
EFF = {"none","narrow_scope","downgrade_status","kill"}
V1_VERDICT = {"survives","survives_with_scope_restriction","killed"}
V2_VERDICT = {"killed","survives","survives_scoped","unresolved"}
REQ = ("attempted","finding","severity","unresolved","effect_on_candidate")


def _seen(d):
    seen = d.get("inputs_seen") or {}
    if isinstance(seen, list):   # v1 wrote a list of one-key mappings
        seen = {k: v for item in seen if isinstance(item, dict) for k, v in item.items()}
    return seen


def _check_attacks(at, e, where):
    """The nine mandated attacks. Shared by both schemas."""
    for k, desc in ATTACKS.items():
        if k not in at:
            e.append(f"R4 MANDATED ATTACK MISSING: {where}{k} ({desc}) [charter Stage 8]")
            continue
        a = at[k] or {}
        for f in REQ:
            if a.get(f) in (None, ""):
                e.append(f"R5 {where}{k}: required field '{f}' empty")
        if a.get("attempted") is False and not a.get("finding"):
            e.append(f"R6 {where}{k}: attempted=false requires a finding explaining why "
                     f"the attack does not apply")
        if a.get("severity") not in SEV and a.get("severity") is not None:
            e.append(f"R7 {where}{k}: severity invalid: {a.get('severity')!r}")
        if a.get("effect_on_candidate") not in EFF and a.get("effect_on_candidate") is not None:
            e.append(f"R8 {where}{k}: effect_on_candidate invalid: {a.get('effect_on_candidate')!r}")


def check_v1(d, e):
    for k in ("task_id","candidate","reviewer","verdict","verdict_reason","proposed_status"):
        if not d.get(k): e.append(f"R1 missing top-level field: {k}")
    if d.get("verdict") not in V1_VERDICT:
        e.append(f"R2 verdict invalid: {d.get('verdict')!r}")
    at = d.get("attacks") or {}
    _check_attacks(at, e, "")
    for k in ATTACKS:
        a = at.get(k) or {}
        if a.get("severity") == "fatal" and d.get("verdict") != "killed":
            e.append(f"R9 {k}: severity 'fatal' but verdict is {d.get('verdict')!r}")


def check_v2(d, e):
    for k in ("task_id","reviewer","candidate_reviews","overall_task_assessment"):
        if not d.get(k): e.append(f"R1 missing top-level field: {k}")

    reviews = d.get("candidate_reviews") or {}
    if not isinstance(reviews, dict) or not reviews:
        e.append("R1 candidate_reviews must be a non-empty mapping of candidate id -> review")
        return

    survived, killed = [], []
    for cid, rv in reviews.items():
        rv = rv or {}
        w = f"{cid}."
        for k in ("statement","verdict","reason"):
            if not rv.get(k): e.append(f"R1 {w}{k} missing")
        v = rv.get("verdict")
        if v not in V2_VERDICT:
            e.append(f"R2 {w}verdict invalid: {v!r} (one of {sorted(V2_VERDICT)})")
        if v == "survives_scoped" and not rv.get("surviving_scope"):
            e.append(f"R13 {w}verdict is survives_scoped but surviving_scope is empty; "
                     f"a scoped survival that does not name its scope is a bare survival")

        at = rv.get("attacks") or {}
        _check_attacks(at, e, w)

        # R9, per candidate: a fatal attack kills THIS candidate only.
        for k in ATTACKS:
            a = at.get(k) or {}
            if a.get("severity") == "fatal" and v != "killed":
                e.append(f"R9 {w}{k}: severity 'fatal' but this candidate's verdict "
                         f"is {v!r}; a fatal attack must kill the candidate it applies to")
            if a.get("effect_on_candidate") == "kill" and v != "killed":
                e.append(f"R9 {w}{k}: effect_on_candidate 'kill' but verdict is {v!r}")

        (killed if v == "killed" else survived).append(cid)

    # R10: the summary must agree with the per-candidate verdicts. This is the
    # v1 failure the schema exists to prevent - a global 'killed' sitting next
    # to prose describing a survivor.
    ot = d.get("overall_task_assessment") or {}
    for k in ("surviving_candidates","killed_candidates","recommendation_basis","proposed_status"):
        if k not in ot:
            e.append(f"R1 overall_task_assessment.{k} missing")
    claimed_s = set(ot.get("surviving_candidates") or [])
    claimed_k = set(ot.get("killed_candidates") or [])
    if claimed_s != set(survived):
        e.append(f"R10 overall_task_assessment.surviving_candidates {sorted(claimed_s)} "
                 f"does not match the per-candidate verdicts {sorted(survived)}")
    if claimed_k != set(killed):
        e.append(f"R10 overall_task_assessment.killed_candidates {sorted(claimed_k)} "
                 f"does not match the per-candidate verdicts {sorted(killed)}")

    unknown = (claimed_s | claimed_k) - set(reviews)
    if unknown:
        e.append(f"R10 overall_task_assessment names candidates with no review: {sorted(unknown)}")


def check(path):
    e = []
    try: d = yaml.safe_load(open(path, encoding="utf-8")) or {}
    except Exception as ex: return [f"R0 YAML parse failure: {ex}"], None
    if not isinstance(d, dict): return ["R0 top level is not a mapping"], None

    seen = _seen(d)
    if seen.get("lead_summary_seen") is True:
        e.append("R3 reviewer saw the lead summary; Stage 8 requires review "
                 "independent of the affirmative reasoning")

    declared = d.get("schema_version")
    is_v2 = "candidate_reviews" in d

    if is_v2:
        check_v2(d, e)
        return e, "v2"

    # no candidate_reviews -> v1 shape
    if declared in (2, "2"):
        e.append("R11 schema_version 2 declared but there is no `candidate_reviews` "
                 "block; a v2 report needs one review per candidate")
    check_v1(d, e)
    return e, "v1"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    paths = args or sorted(glob.glob("research/proposals/*redteam*/REDTEAM.yaml")
                           + glob.glob("research/tasks/*/*/REDTEAM.yaml"))
    if not paths:
        print("no red-team reports found (none expected yet)"); return 0
    bad = 0
    for p in paths:
        errs, ver = check(p)
        label = {"v2": "schema v2", "v1": "LEGACY-V1", None: "unparseable"}[ver]
        print(f"\n=== {p}  [{label}] ===")
        if ver == "v1":
            print("  NOTE: legacy single-verdict schema. Readable, but it cannot "
                  "express mixed outcomes (some candidates killed, others "
                  "surviving scoped). New reports must use v2.")
        for x in errs: print("  ERROR " + x)
        print(f"  {len(errs)} error(s)")
        bad += len(errs)
    print(f"\n{bad} error(s) total")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
