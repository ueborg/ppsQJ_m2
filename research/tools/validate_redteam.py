#!/usr/bin/env python3
"""Validate a red-team report against Charter section 7 Stage 8.
Every one of the nine mandated attacks must carry a complete verdict.
A missing attack is a HARD FAILURE. Read-only. Exit 0 clean, 1 on error."""
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
VERDICT = {"survives","survives_with_scope_restriction","killed"}
REQ = ("attempted","finding","severity","unresolved","effect_on_candidate")

def check(path):
    e=[]
    try: d=yaml.safe_load(open(path,encoding="utf-8")) or {}
    except Exception as ex: return [f"R0 YAML parse failure: {ex}"]
    for k in ("task_id","candidate","reviewer","verdict","verdict_reason","proposed_status"):
        if not d.get(k): e.append(f"R1 missing top-level field: {k}")
    if d.get("verdict") not in VERDICT:
        e.append(f"R2 verdict invalid: {d.get('verdict')!r}")
    seen = d.get("inputs_seen") or {}
    if isinstance(seen, list):
        seen = {k: v for item in seen if isinstance(item, dict) for k, v in item.items()}
    if seen.get("lead_summary_seen") is True:
        e.append("R3 reviewer saw the lead summary; Stage 8 requires review "
                 "independent of the affirmative reasoning")
    at = d.get("attacks") or {}
    for k, desc in ATTACKS.items():
        if k not in at:
            e.append(f"R4 MANDATED ATTACK MISSING: {k} ({desc}) [charter Stage 8]")
            continue
        a = at[k] or {}
        for f in REQ:
            if a.get(f) in (None, ""):
                e.append(f"R5 {k}: required field '{f}' empty")
        if a.get("attempted") is False and not a.get("finding"):
            e.append(f"R6 {k}: attempted=false requires a finding explaining why "
                     f"the attack does not apply")
        if a.get("severity") not in SEV and a.get("severity") is not None:
            e.append(f"R7 {k}: severity invalid: {a.get('severity')!r}")
        if a.get("effect_on_candidate") not in EFF and a.get("effect_on_candidate") is not None:
            e.append(f"R8 {k}: effect_on_candidate invalid: {a.get('effect_on_candidate')!r}")
        if a.get("severity") == "fatal" and d.get("verdict") != "killed":
            e.append(f"R9 {k}: severity 'fatal' but verdict is {d.get('verdict')!r}")
    return e

def main():
    args = sys.argv[1:]
    paths = args or sorted(glob.glob("research/proposals/*redteam*/REDTEAM.yaml"))
    if not paths:
        print("no red-team reports found (none expected yet)"); return 0
    bad = 0
    for p in paths:
        errs = check(p)
        print(f"\n=== {p} ===")
        for x in errs: print("  ERROR " + x)
        print(f"  {len(errs)} error(s)")
        bad += len(errs)
    print(f"\n{bad} error(s) total")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
