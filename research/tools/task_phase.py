#!/usr/bin/env python3
"""task_phase.py - open, close and check phases of a /research task.

The ONLY supported way to write TASK_MANIFEST.yaml. Writes nothing outside the
task directory and never touches research/state/**.

    task_phase.py <TASK_DIR> init            <TASK-ID> [--mode MODE]
    task_phase.py <TASK_DIR> close           <stage> [--note TEXT]
    task_phase.py <TASK_DIR> dispatch        --worker role=model ... [--skip role]
    task_phase.py <TASK_DIR> collaborate open  --dependency TEXT --roles a,b
                                               --question TEXT --value TEXT
    task_phase.py <TASK_DIR> collaborate none  --reason TEXT
    task_phase.py <TASK_DIR> check                      # verify frozen hashes
    task_phase.py <TASK_DIR> amend <path> --reason TEXT --authorised-by WHO

`close` stamps the current UTC time and records the SHA-256 of every artifact
that stage is responsible for freezing. From then on, editing one of those files
is an integrity failure that validate_task.py reports as M5.

Time comes from the clock, once, at close. That is the point: a stage cannot be
back-dated through this tool.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required:  .venv/bin/python3 -m pip install pyyaml")

MANIFEST = "TASK_MANIFEST.yaml"

# Which artifacts each stage is responsible for freezing.
# `first_pass_frozen` freezes agent_reports/* dynamically - see freeze_list().
FREEZES = {
    "task_created": [],
    "stage_1_problem": ["CHARTER.md", "PROBLEM_MEMO.md", "SOURCE_REGISTER.md"],
    "investigators_dispatched": [],
    "first_pass_frozen": [],          # dynamic: every file under agent_reports/
    "collaboration_opened": [],       # optional
    "collaboration_closed": ["COLLABORATION_LOG.yaml"],   # optional
    "stage_3_candidates": ["CANDIDATES.md", "FALSIFICATION_PLAN.md",
                           "ANALYSIS_SPEC.yaml", "NOVELTY_GATE.md",
                           "FIELD_MAP.md", "NOVELTY_MATRIX.md"],
    "redteam_dispatched": [],
    "synthesis_closed": ["REDTEAM.yaml", "RESEARCH_MEMO.md", "RECOMMENDATION.md",
                         "FALSIFICATION_RESULTS.md", "ASSESSMENT_AH.md",
                         "SLOP_WARNINGS.md"],
}
ORDER = list(FREEZES)
# Stages a task may legitimately skip. Everything else must close in order.
OPTIONAL = {"collaboration_opened", "collaboration_closed"}
COLLAB_ROLES = {"literature", "theory", "numerics"}   # red-team is NEVER here


def freeze_list(task, stage):
    """Artifacts this stage freezes, including the dynamic first-pass set."""
    rels = list(FREEZES[stage])
    if stage == "first_pass_frozen":
        ar = os.path.join(task, "agent_reports")
        if os.path.isdir(ar):
            rels += [f"agent_reports/{fn}" for fn in sorted(os.listdir(ar))
                     if fn != "README.md" and os.path.isfile(os.path.join(ar, fn))]
    return rels


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def load(task: str) -> dict:
    p = os.path.join(task, MANIFEST)
    if not os.path.isfile(p):
        sys.exit(f"no {MANIFEST} in {task}; run `init` first")
    return yaml.safe_load(open(p, encoding="utf-8")) or {}


def save(task: str, man: dict) -> None:
    with open(os.path.join(task, MANIFEST), "w", encoding="utf-8") as fh:
        fh.write("# TASK PHASE LEDGER - append-only. Written by "
                 "research/tools/task_phase.py.\n"
                 "# Editing a frozen artifact after its stage closed is "
                 "validate_task.py error M5.\n")
        yaml.safe_dump(man, fh, sort_keys=False, default_flow_style=False,
                       width=100, allow_unicode=True)


def phase(man: dict, stage: str) -> dict | None:
    for p in man.get("phases") or []:
        if p.get("stage") == stage:
            return p
    return None


def cmd_init(task, a):
    p = os.path.join(task, MANIFEST)
    if os.path.exists(p):
        sys.exit(f"{MANIFEST} already exists; refusing to overwrite a phase ledger")
    man = {
        "schema_version": 2,
        "task_id": a.task_id,
        "mode": a.mode,
        "phases": [{"stage": s, "closed": None, "frozen": {}} for s in ORDER],
        "amendments": [],
    }
    phase(man, "investigators_dispatched").update(
        {"workers": [], "workers_skipped": []})
    ph = phase(man, "task_created")
    ph["closed"] = now()
    save(task, man)
    print(f"initialised {MANIFEST} for {a.task_id} (mode={a.mode})")
    print(f"  task_created closed at {ph['closed']}")


def cmd_close(task, a):
    man = load(task)
    stage = a.stage
    if stage not in ORDER:
        sys.exit(f"unknown stage {stage!r}; one of {ORDER}")
    ph = phase(man, stage)
    if ph is None:
        sys.exit(f"stage {stage} not in this manifest")
    if ph.get("closed"):
        sys.exit(f"stage {stage} already closed at {ph['closed']}; "
                 f"use `amend` to correct a frozen artifact")

    # ordering: every earlier NON-OPTIONAL stage must already be closed
    for earlier in ORDER[:ORDER.index(stage)]:
        if earlier in OPTIONAL:
            continue
        if not (phase(man, earlier) or {}).get("closed"):
            sys.exit(f"cannot close {stage}: earlier stage {earlier} is still open")

    # first passes must actually exist before they can be frozen, and every
    # dispatched worker must have produced one. Freezing an empty set would
    # make the collaboration barrier vacuous.
    if stage == "first_pass_frozen":
        disp = phase(man, "investigators_dispatched") or {}
        expected = [w.get("role") for w in (disp.get("workers") or [])]
        have = set()
        ar = os.path.join(task, "agent_reports")
        if os.path.isdir(ar):
            have = {os.path.splitext(fn)[0] for fn in os.listdir(ar)}
        missing_roles = [r for r in expected if r not in have]
        if missing_roles:
            sys.exit(f"cannot close first_pass_frozen: no report in "
                     f"agent_reports/ for dispatched worker(s): "
                     f"{', '.join(missing_roles)}")
        if not expected:
            sys.exit("cannot close first_pass_frozen: no workers were recorded "
                     "as dispatched")

    # collaboration may only open once every first pass is frozen
    if stage == "collaboration_opened":
        if not (phase(man, "first_pass_frozen") or {}).get("closed"):
            sys.exit("cannot open collaboration: first_pass_frozen is still "
                     "open. Independent reports must be written and frozen "
                     "before anyone sees anyone else's work.")
        if (man.get("collaboration") or {}).get("rounds", 0) >= 1:
            sys.exit("collaboration round 1 already ran; /research v1 permits "
                     "exactly ONE round. Open a new task instead.")
    if stage == "collaboration_closed":
        if not (phase(man, "collaboration_opened") or {}).get("closed"):
            sys.exit("cannot close collaboration: it was never opened")

    frozen = {}
    missing = []
    for rel in freeze_list(task, stage):
        fp = os.path.join(task, rel)
        if os.path.isfile(fp):
            frozen[rel] = sha(fp)
        else:
            missing.append(rel)

    ph["closed"] = now()
    ph["frozen"] = frozen
    if a.note:
        ph["note"] = a.note
    save(task, man)

    print(f"closed {stage} at {ph['closed']}")
    for rel, h in frozen.items():
        print(f"  frozen  {rel}  {h[:19]}...")
    for rel in missing:
        print(f"  ABSENT  {rel}  (not frozen; validate_task.py will require it "
              f"if it is mandatory)")


def cmd_dispatch(task, a):
    man = load(task)
    ph = phase(man, "investigators_dispatched")
    if not (phase(man, "stage_1_problem") or {}).get("closed"):
        sys.exit("cannot dispatch investigators: stage_1_problem is still open. "
                 "The problem memo and kill criterion must be frozen BEFORE "
                 "anyone investigates.")
    workers = []
    for spec in a.worker or []:
        if "=" not in spec:
            sys.exit(f"--worker expects role=model, got {spec!r}")
        role, model = spec.split("=", 1)
        workers.append({"role": role, "model": model})
    ph["workers"] = workers
    ph["workers_skipped"] = a.skip or []
    ph["closed"] = now()
    save(task, man)
    print(f"dispatched at {ph['closed']}: "
          + ", ".join(f"{w['role']}={w['model']}" for w in workers))
    if a.skip:
        print(f"  skipped: {', '.join(a.skip)}")


def cmd_collaborate(task, a):
    """Open a bounded collaboration round, or record that none is needed."""
    man = load(task)
    collab = man.setdefault("collaboration", {"rounds": 0, "decision": None})

    if a.action == "none":
        if not a.reason:
            sys.exit("--reason is required: say why no cross-role dependency exists")
        collab["decision"] = "COLLABORATION_NOT_NEEDED"
        collab["reason"] = a.reason
        collab["at"] = now()
        save(task, man)
        print("COLLABORATION_NOT_NEEDED recorded.\n  reason: " + a.reason)
        print("Proceed directly to candidate construction.")
        return 0

    # action == open
    if collab.get("rounds", 0) >= 1:
        sys.exit("collaboration round 1 already ran; v1 permits exactly ONE. "
                 "A second round is not a retry and is not allowed.")
    if not (phase(man, "first_pass_frozen") or {}).get("closed"):
        sys.exit("cannot open collaboration before first_pass_frozen is closed")

    missing = [f for f, v in (("--dependency", a.dependency), ("--roles", a.roles),
                              ("--question", a.question), ("--value", a.value))
               if not v]
    if missing:
        sys.exit("collaboration must be justified BEFORE it opens; missing: "
                 + ", ".join(missing)
                 + "\n  --dependency  the unresolved cross-role dependency"
                   "\n  --roles       which roles must exchange information"
                   "\n  --question    the concrete question being answered"
                   "\n  --value       why another invocation has expected decision value")

    roles = [r.strip() for r in a.roles.split(",") if r.strip()]
    bad = [r for r in roles if r not in COLLAB_ROLES]
    if bad:
        sys.exit(f"not affirmative investigator roles: {bad}. "
                 f"The red team NEVER participates in affirmative collaboration.")

    collab.update({"rounds": collab.get("rounds", 0) + 1,
                   "decision": "OPENED", "dependency": a.dependency,
                   "roles": roles, "question": a.question,
                   "expected_value": a.value, "opened_at": now()})
    ph = phase(man, "collaboration_opened")
    ph["closed"] = now()
    ph["note"] = a.question
    save(task, man)
    print(f"collaboration round 1 opened at {ph['closed']}")
    print(f"  roles:      {', '.join(roles)}")
    print(f"  dependency: {a.dependency}")
    print(f"  question:   {a.question}")
    print("Exactly one round. Write COLLABORATION_LOG.yaml, then "
          "`close collaboration_closed`.")
    return 0


def cmd_check(task, _a):
    man = load(task)
    bad = 0
    for ph in man.get("phases") or []:
        for rel, expect in (ph.get("frozen") or {}).items():
            fp = os.path.join(task, rel)
            if not os.path.isfile(fp):
                print(f"MISSING  {rel} (frozen at {ph['stage']})")
                bad += 1
            elif sha(fp) != expect:
                print(f"MODIFIED {rel} after {ph['stage']} closed "
                      f"{ph.get('closed')}")
                bad += 1
    amended = {a["path"] for a in (man.get("amendments") or []) if a.get("path")}
    if amended:
        print(f"(amendments recorded for: {', '.join(sorted(amended))})")
    print("OK: no frozen artifact was modified" if not bad
          else f"{bad} frozen artifact(s) changed after their stage closed")
    return 1 if bad else 0


def cmd_amend(task, a):
    man = load(task)
    fp = os.path.join(task, a.path)
    if not os.path.isfile(fp):
        sys.exit(f"no such file: {a.path}")
    was = None
    for ph in man.get("phases") or []:
        if a.path in (ph.get("frozen") or {}):
            was = ph["frozen"][a.path]
            ph["frozen"][a.path] = sha(fp)
    if was is None:
        sys.exit(f"{a.path} is not frozen by any closed stage; just edit it")
    man.setdefault("amendments", []).append({
        "path": a.path, "was": was, "now": sha(fp), "at": now(),
        "reason": a.reason, "authorised_by": a.authorised_by,
    })
    save(task, man)
    print(f"amended {a.path}\n  was {was[:19]}...\n  now {sha(fp)[:19]}...\n"
          f"  reason: {a.reason}\n  authorised_by: {a.authorised_by}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init"); p.add_argument("task_id")
    p.add_argument("--mode", default="normal")
    p = sub.add_parser("close"); p.add_argument("stage"); p.add_argument("--note")
    p = sub.add_parser("dispatch")
    p.add_argument("--worker", action="append"); p.add_argument("--skip", action="append")
    p = sub.add_parser("collaborate")
    p.add_argument("action", choices=["open", "none"])
    p.add_argument("--dependency"); p.add_argument("--roles")
    p.add_argument("--question"); p.add_argument("--value")
    p.add_argument("--reason")
    sub.add_parser("check")
    p = sub.add_parser("amend"); p.add_argument("path")
    p.add_argument("--reason", required=True); p.add_argument("--authorised-by", required=True)

    a = ap.parse_args()
    task = os.path.abspath(a.task_dir)
    if not os.path.isdir(task):
        sys.exit(f"not a directory: {task}")
    return {"init": cmd_init, "close": cmd_close, "dispatch": cmd_dispatch,
            "collaborate": cmd_collaborate,
            "check": cmd_check, "amend": cmd_amend}[a.cmd](task, a) or 0


if __name__ == "__main__":
    sys.exit(main())
