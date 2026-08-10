#!/usr/bin/env python3
"""validate_task.py - completeness checks for a /research task directory.

Read-only. Modifies nothing. Exit 0 clean, 1 if any ERROR.

The knowledge plane has validate_state.py. The red-team report has
validate_redteam.py. This is the third gap: nothing checked that a task
directory actually contains the charter's mandated stage artifacts, so a run
could skip Stage 1, 2 or 3 and still look finished. It cannot check QUALITY -
only that the required record exists and was filled in.

Checks:
  T1  mandated stage artifact missing                  (charter Stage 0-4, 9)
  T2  artifact still contains template placeholders    (created, never written)
  T3  RECOMMENDATION has no verdict, or more than one  (decision gate)
  T4  REDTEAM.yaml missing, or fails validate_redteam.py  (Stage 8)
  T5  fewer than 12 slop-warning verdicts              (charter sec 6)
  T6  Meaningful-Contribution dimension A-H missing a verdict  (charter sec 5)
  T6b an aggregate score was produced                  (charter sec 5, forbidden)
  T7  candidate count outside 3-8, or missing fields   (charter Stage 3)
  T8  no statement-class labels [E]/[I]/[C]/[J]        (charter sec 2)
  T9  a file inside the task directory targets research/state/  (single-writer)

  M1  TASK_MANIFEST.yaml missing or unparseable        (phase ledger)
  M2  a mandated phase is absent from the ledger
  M3  phases closed out of order
  M4  investigators dispatched before Stage 1 closed
  M5  A FROZEN ARTIFACT WAS MODIFIED AFTER ITS STAGE CLOSED
  M6  a frozen artifact has since disappeared
  M7  an amendment lacks a reason or an authoriser

  F1  the pre-specified falsification plan contains results
  F2  FALSIFICATION_RESULTS.md missing though the plan closed

  N1  ANALYSIS_SPEC.yaml missing though a numerics report exists
  N2  an analysis lacks a required estimator field
  N3  an analysis declares no crossing-validity rule
  N4  crossings were not classified against the declared rule
  N5  a sensitivity analysis does not name the primary it varies from
  N6  an invalid crossing entered the primary fit

  E1  red team cites task-verified evidence that TASK_EVIDENCE.yaml lacks
  E2  a task-verified item lacks its inspection record
  E3  a task-verified item is described as canonical

  G1  NOVELTY_GATE.md missing
  G2  a candidate has no closest-predecessor record
  G3  novelty language used without a gate classification

  W1  BRIDGE_AUDIT.md absent (warning only; required for cross-field claims)

Usage:  python3 research/tools/validate_task.py research/tasks/active/<TASK-ID>
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required")

REQUIRED = [
    ("TASK_MANIFEST.yaml", "phase ledger"),
    ("CHARTER.md", "task charter: question, hypotheses, kill criterion"),
    ("PROBLEM_MEMO.md", "charter Stage 1"),
    ("SOURCE_REGISTER.md", "charter Stage 0, task-scoped"),
    ("FIELD_MAP.md", "charter Stage 2"),
    ("NOVELTY_MATRIX.md", "charter Stage 2"),
    ("NOVELTY_GATE.md", "duplicate / predecessor gate"),
    ("CANDIDATES.md", "charter Stage 3"),
    ("FALSIFICATION_PLAN.md", "charter Stage 4, PRE-SPECIFIED"),
    ("FALSIFICATION_RESULTS.md", "charter Stage 4, outcomes"),
    ("ASSESSMENT_AH.md", "charter sec 5, Meaningful-Contribution Test"),
    ("SLOP_WARNINGS.md", "charter sec 6, twelve warnings"),
    ("RESEARCH_MEMO.md", "charter Stage 9"),
    ("RECOMMENDATION.md", "decision gate"),
]

# Must stay in sync with task_phase.ORDER. A stage missing from this list is a
# stage whose frozen hashes are never checked, which silently disables the
# phase lock for it.
PHASES = ["task_created", "stage_1_problem", "investigators_dispatched",
          "first_pass_frozen", "collaboration_opened", "collaboration_closed",
          "stage_3_candidates", "redteam_dispatched", "synthesis_closed"]
OPTIONAL_PHASES = {"collaboration_opened", "collaboration_closed"}

ESTIMATOR_FIELDS = ("evidence_id", "observable_id", "parameterization",
                    "pair_selection_rule", "crossing_definition",
                    "interpolation", "fitting_window", "weighting",
                    "uncertainty_model", "finite_size_extrapolation")

NOVELTY_WORDS = re.compile(r"\b(?:novel|novelty|new finding|the finding of the "
                           r"task|first (?:demonstration|observation)|"
                           r"unprecedented|contribution of this task)\b", re.I)
GATE_CLASSES = ("replication", "corroboration", "rediscovery",
                "provenance repair", "no predecessor found")

VERDICTS = ("Pursue", "Reformulate", "Infrastructure first", "Stop")
AH = "ABCDEFGH"
PLACEHOLDER = re.compile(r"<TASK-ID>|<One sentence|<statement>|<What decision")

errors: list[str] = []
warns: list[str] = []


def err(code: str, where: str, msg: str) -> None:
    errors.append(f"ERROR {code} [{where}] {msg}")


def warn(code: str, where: str, msg: str) -> None:
    warns.append(f"WARN  {code} [{where}] {msg}")


def read(task: str, name: str) -> str | None:
    p = os.path.join(task, name)
    if not os.path.isfile(p):
        return None
    return open(p, encoding="utf-8").read()


def check(task: str, repo: str) -> None:
    # T1 / T2 -----------------------------------------------------------------
    texts = {}
    for name, why in REQUIRED:
        body = read(task, name)
        if body is None:
            err("T1", name, f"missing ({why})")
            continue
        texts[name] = body
        if PLACEHOLDER.search(body):
            err("T2", name, "still contains template placeholders; the artifact "
                            "was created but never written")

    # T3 decision gate --------------------------------------------------------
    rec = texts.get("RECOMMENDATION.md")
    if rec:
        # look only at the Verdict section, not the enumeration of options
        section = rec.split("## Verdict", 1)[-1].split("##", 1)[0] if "## Verdict" in rec else rec
        hits = [v for v in VERDICTS if re.search(rf"\b{re.escape(v)}\b", section)]
        if not hits:
            err("T3", "RECOMMENDATION.md",
                f"no verdict found; exactly one of {VERDICTS} is required")
        elif len(hits) > 1:
            err("T3", "RECOMMENDATION.md",
                f"{len(hits)} verdicts found ({hits}); the gate emits exactly one")

    # T4 red team -------------------------------------------------------------
    rt = os.path.join(task, "REDTEAM.yaml")
    if not os.path.isfile(rt):
        err("T4", "REDTEAM.yaml", "missing (charter Stage 8 is mandatory)")
    else:
        tool = os.path.join(repo, "research", "tools", "validate_redteam.py")
        proc = subprocess.run([sys.executable, tool, rt],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            tail = [ln.strip() for ln in proc.stdout.splitlines()
                    if "ERROR" in ln][:6]
            err("T4", "REDTEAM.yaml",
                "fails validate_redteam.py: " + ("; ".join(tail) or "see tool output"))

    # T5 slop warnings --------------------------------------------------------
    slop = texts.get("SLOP_WARNINGS.md")
    if slop:
        rows = re.findall(r"^\|\s*(\d{1,2})\s*\|(.+)\|(.+)\|(.+)\|\s*$",
                          slop, re.MULTILINE)
        filled = [r for r in rows if r[2].strip() and r[2].strip() != "|"]
        if len(filled) < 12:
            err("T5", "SLOP_WARNINGS.md",
                f"{len(filled)}/12 warnings carry a verdict; charter sec 6 "
                f"requires an explicit verdict on all twelve")

    # T6 A-H ------------------------------------------------------------------
    ah = texts.get("ASSESSMENT_AH.md")
    if ah:
        for letter in AH:
            m = re.search(rf"^##\s*{letter}\.\s", ah, re.MULTILINE)
            if not m:
                err("T6", "ASSESSMENT_AH.md", f"dimension {letter} missing")
                continue
            start = m.end()
            nxt = re.search(r"^##\s", ah[start:], re.MULTILINE)
            body = ah[start:start + nxt.start()] if nxt else ah[start:]
            if not re.search(r"\*\*Verdict", body) or len(body.strip()) < 40:
                err("T6", "ASSESSMENT_AH.md",
                    f"dimension {letter} has no filled-in verdict")
        # Only an actual score, not the sentence forbidding one. Requires a
        # number, so "No aggregate score" in the template does not trip it.
        if re.search(r"(?:aggregate|total|overall|weighted|combined)\s+score\s*"
                     r"(?:is|of|[:=])\s*\d|score\s*[:=]\s*\d+\s*/\s*\d+",
                     ah, re.IGNORECASE):
            err("T6b", "ASSESSMENT_AH.md",
                "an aggregate score appears; charter sec 5 forbids collapsing "
                "A-H into one number")

    # T7 candidates -----------------------------------------------------------
    cand = texts.get("CANDIDATES.md")
    if cand:
        heads = re.findall(r"^##\s*Candidate\s+(\S+)", cand, re.MULTILINE)
        if not 3 <= len(heads) <= 8:
            err("T7", "CANDIDATES.md",
                f"{len(heads)} candidates; charter Stage 3 says 3-8 "
                f"(fewer is under-exploration, more is variant-spam)")
        for i in range(1, 12):
            if not re.search(rf"^\s*{i}\.\s*\*\*", cand, re.MULTILINE):
                err("T7", "CANDIDATES.md",
                    f"required field {i} of 11 not found for any candidate")
                break

    # T8 statement classes ----------------------------------------------------
    for name in ("PROBLEM_MEMO.md", "RESEARCH_MEMO.md"):
        body = texts.get(name)
        if body and not re.search(r"\[(?:E|I|C|J)\]", body):
            err("T8", name,
                "no [E]/[I]/[C]/[J] statement-class labels; charter sec 2 "
                "requires evidence, inference, conjecture and judgment to be "
                "distinguished")

    # T9 single-writer --------------------------------------------------------
    for root, _dirs, files in os.walk(task):
        for fn in files:
            p = os.path.join(root, fn)
            if os.path.islink(p):
                target = os.path.realpath(p)
                if os.path.join(repo, "research", "state") in target:
                    err("T9", os.path.relpath(p, task),
                        "symlink points into research/state/; the task plane "
                        "must not alias canonical state")

    # --- M: the phase ledger -------------------------------------------------
    check_manifest(task)

    # --- F: pre-spec plan vs results ----------------------------------------
    plan = texts.get("FALSIFICATION_PLAN.md")
    if plan:
        # A pre-specified plan states what WILL be checked and what would kill
        # the candidate. A results column means it was written, or rewritten,
        # after the answers were known - the exact defect found in
        # TASK-2026-08-10-AMP096.
        header = re.search(r"^\|[^\n]*\|\s*$", plan, re.MULTILINE)
        if header and re.search(r"\|\s*(?:result|outcome|finding|what it showed|"
                                r"verdict|done)\s*\|", header.group(0), re.I):
            err("F1", "FALSIFICATION_PLAN.md",
                "the PRE-SPECIFIED plan carries a results/outcome/done column. "
                "Outcomes belong in FALSIFICATION_RESULTS.md; a plan that "
                "records its own answers is not a pre-specification")
        if re.search(r"^\s*\*\*Result", plan, re.MULTILINE):
            err("F1", "FALSIFICATION_PLAN.md",
                "contains a 'Result' section; move it to FALSIFICATION_RESULTS.md")

    # --- N: analysis specification ------------------------------------------
    check_analysis_spec(task)

    # --- E / X: task-local evidence tier and external sources ---------------
    check_task_evidence(task)
    check_external_sources(task)

    # --- G: novelty / duplicate gate ----------------------------------------
    check_novelty_gate(task, texts)

    # --- C: bounded affirmative collaboration -------------------------------
    check_collaboration(task)

    # W1 bridge audit ---------------------------------------------------------
    if not os.path.isfile(os.path.join(task, "BRIDGE_AUDIT.md")):
        warn("W1", "BRIDGE_AUDIT.md",
             "absent; REQUIRED (charter sec 8) if this task makes a "
             "cross-field claim, optional otherwise")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def check_manifest(task):
    p = os.path.join(task, "TASK_MANIFEST.yaml")
    if not os.path.isfile(p):
        return  # T1 already reported it
    try:
        man = yaml.safe_load(open(p, encoding="utf-8")) or {}
    except Exception as e:
        err("M1", "TASK_MANIFEST.yaml", f"YAML parse failure: {e}")
        return
    phases = {ph.get("stage"): ph for ph in (man.get("phases") or [])
              if isinstance(ph, dict)}

    for stage in PHASES:
        if stage not in phases and stage not in OPTIONAL_PHASES:
            err("M2", "TASK_MANIFEST.yaml", f"mandated phase {stage!r} absent")

    # M3 ordering: closed timestamps must not go backwards along PHASES
    last_t, last_s = None, None
    for stage in PHASES:
        ph = phases.get(stage) or {}
        t = ph.get("closed")
        if not t:
            continue
        t = str(t)
        if last_t and t < last_t:
            err("M3", "TASK_MANIFEST.yaml",
                f"{stage} closed {t} which is BEFORE {last_s} closed {last_t}; "
                f"phases must close in order")
        last_t, last_s = t, stage

    # M4: the ordering rule that matters most
    s1 = (phases.get("stage_1_problem") or {}).get("closed")
    disp = (phases.get("investigators_dispatched") or {}).get("closed")
    if disp and not s1:
        err("M4", "TASK_MANIFEST.yaml",
            "investigators were dispatched while stage_1_problem was still "
            "open; the problem memo and kill criterion must be frozen first")

    # M5 / M6: the integrity check the whole ledger exists for
    for stage in PHASES:
        ph = phases.get(stage) or {}
        for rel, expect in (ph.get("frozen") or {}).items():
            fp = os.path.join(task, rel)
            if not os.path.isfile(fp):
                err("M6", rel, f"frozen at {stage} but no longer on disk")
                continue
            if _sha(fp) != expect:
                err("M5", rel,
                    f"MODIFIED after {stage} closed ({ph.get('closed')}). A "
                    f"later phase rewrote an artifact that was supposed to be "
                    f"frozen. If the change was legitimate, record it with "
                    f"`task_phase.py {os.path.basename(task)} amend {rel} "
                    f"--reason ... --authorised-by ...`")

    for i, am in enumerate(man.get("amendments") or []):
        if not isinstance(am, dict):
            continue
        if not am.get("reason") or not am.get("authorised_by"):
            err("M7", "TASK_MANIFEST.yaml",
                f"amendment {i} lacks a reason or an authoriser")


def check_analysis_spec(task):
    has_numerics = any(fn.startswith("numerics")
                       for fn in os.listdir(os.path.join(task, "agent_reports"))
                       ) if os.path.isdir(os.path.join(task, "agent_reports")) else False
    p = os.path.join(task, "ANALYSIS_SPEC.yaml")
    if not os.path.isfile(p):
        if has_numerics:
            err("N1", "ANALYSIS_SPEC.yaml",
                "a numerics report exists but no analysis was specified. An "
                "estimator that was never declared cannot be audited")
        return
    try:
        spec = yaml.safe_load(open(p, encoding="utf-8")) or {}
    except Exception as e:
        err("N1", "ANALYSIS_SPEC.yaml", f"YAML parse failure: {e}")
        return

    analyses = spec.get("analyses") or []
    ids = {a.get("id") for a in analyses if isinstance(a, dict)}
    for a in analyses:
        if not isinstance(a, dict):
            continue
        aid = a.get("id", "?")
        for f in ESTIMATOR_FIELDS:
            if a.get(f) in (None, ""):
                err("N2", "ANALYSIS_SPEC.yaml", f"{aid}: required field {f!r} empty")

        vr = a.get("validity_rule") or {}
        if not vr:
            err("N3", "ANALYSIS_SPEC.yaml",
                f"{aid}: no validity_rule. The rule may be whatever the science "
                f"requires, but it must be declared BEFORE the fit")
        cc = a.get("crossing_classification") or {}
        if vr and not cc:
            err("N4", "ANALYSIS_SPEC.yaml",
                f"{aid}: validity_rule declared but no crossing_classification; "
                f"nothing was checked against it")
        else:
            counts = [cc.get(k) for k in ("n_valid", "n_ambiguous", "n_invalid")]
            if all(c is None for c in counts):
                err("N4", "ANALYSIS_SPEC.yaml",
                    f"{aid}: crossing_classification records no valid/ambiguous/"
                    f"invalid counts")
            for row in (cc.get("cells") or []):
                if isinstance(row, dict) and row.get("status") == "invalid" \
                        and row.get("entered_primary_fit") is True:
                    err("N6", "ANALYSIS_SPEC.yaml",
                        f"{aid}: a crossing classified INVALID entered the "
                        f"primary fit ({row.get('cell')})")

        if a.get("role") == "sensitivity":
            if not a.get("varies_from"):
                err("N5", "ANALYSIS_SPEC.yaml",
                    f"{aid}: sensitivity analysis does not name the primary it "
                    f"varies from")
            elif a["varies_from"] not in ids:
                err("N5", "ANALYSIS_SPEC.yaml",
                    f"{aid}: varies_from {a['varies_from']!r} is not an analysis "
                    f"in this spec")
            if not a.get("what_is_varied"):
                err("N5", "ANALYSIS_SPEC.yaml",
                    f"{aid}: sensitivity does not say what it varies")


def check_task_evidence(task):
    tep = os.path.join(task, "TASK_EVIDENCE.yaml")
    te = {}
    if os.path.isfile(tep):
        try:
            doc = yaml.safe_load(open(tep, encoding="utf-8")) or {}
            for item in doc.get("task_verified") or []:
                if isinstance(item, dict) and item.get("id"):
                    te[item["id"]] = item
        except Exception as e:
            err("E2", "TASK_EVIDENCE.yaml", f"YAML parse failure: {e}")
            return

    for tid, item in te.items():
        for f in ("tier", "source_or_artifact", "what_was_verified",
                  "verified_by", "date"):
            if not item.get(f):
                err("E2", "TASK_EVIDENCE.yaml",
                    f"{tid}: required field {f!r} missing; a task verification "
                    f"that does not say what was inspected is not a verification")
        if str(item.get("canonical", "")).lower() == "true":
            err("E3", "TASK_EVIDENCE.yaml",
                f"{tid}: marked canonical. Task-verified evidence is admissible "
                f"WITHIN this task only, and becomes canonical solely through "
                f"the human merge gate")
        if item.get("promotion_status") not in (None, "", "proposed", "not_proposed"):
            err("E3", "TASK_EVIDENCE.yaml",
                f"{tid}: promotion_status {item['promotion_status']!r}; a task "
                f"may only PROPOSE promotion")

    rt = os.path.join(task, "REDTEAM.yaml")
    if os.path.isfile(rt):
        try:
            d = yaml.safe_load(open(rt, encoding="utf-8")) or {}
        except Exception:
            return
        seen = d.get("inputs_seen") or {}
        if isinstance(seen, list):
            seen = {k: v for it in seen if isinstance(it, dict) for k, v in it.items()}
        for tid in (seen.get("task_verified") or []):
            if tid not in te:
                err("E1", "REDTEAM.yaml",
                    f"cites task-verified item {tid!r} with no record in "
                    f"TASK_EVIDENCE.yaml")


MSG_TYPES = {"source_check", "theoretical_assumption_check", "derivation_check",
             "numerical_test_request", "estimator_question", "contradiction",
             "prior_art_check", "clarification"}
RESULT_CLASSES = {"confirmed", "contradicted", "narrowed", "unresolved", "no_effect"}
AFFIRMATIVE = {"literature", "theory", "numerics", "lead"}
# A message is an atomic fact or question. Anything this long is a transcript.
MAX_MSG_CHARS = 1500


def check_collaboration(task):
    man_p = os.path.join(task, "TASK_MANIFEST.yaml")
    man = {}
    if os.path.isfile(man_p):
        try:
            man = yaml.safe_load(open(man_p, encoding="utf-8")) or {}
        except Exception:
            return
    phases = {ph.get("stage"): ph for ph in (man.get("phases") or [])
              if isinstance(ph, dict)}
    collab_meta = man.get("collaboration") or {}
    opened = (phases.get("collaboration_opened") or {}).get("closed")
    log_p = os.path.join(task, "COLLABORATION_LOG.yaml")

    # C1: collaboration is OPTIONAL. Absent + recorded-as-unneeded is fine.
    if not opened:
        if os.path.isfile(log_p):
            err("C1", "COLLABORATION_LOG.yaml",
                "a collaboration log exists but collaboration_opened was never "
                "closed in the ledger; the round was never justified")
        return

    # C2: it cannot open before the first passes are frozen
    if not (phases.get("first_pass_frozen") or {}).get("closed"):
        err("C2", "TASK_MANIFEST.yaml",
            "collaboration opened while first_pass_frozen was still open; "
            "independent reports must be frozen before anyone sees a peer's work")

    # C7: exactly one round
    rounds = collab_meta.get("rounds", 0)
    if rounds > 1:
        err("C7", "TASK_MANIFEST.yaml",
            f"{rounds} collaboration rounds; /research v1 permits exactly ONE")

    for f in ("unresolved_dependency", "roles_involved", "concrete_question",
              "expected_decision_value"):
        pass  # justification lives in the ledger; checked below

    for f, k in (("dependency", "unresolved dependency"),
                 ("question", "concrete question"),
                 ("expected_value", "expected decision value")):
        if not collab_meta.get(f):
            err("C3", "TASK_MANIFEST.yaml",
                f"collaboration opened without recording the {k}")
    for r in (collab_meta.get("roles") or []):
        if r == "red-team":
            err("C5", "TASK_MANIFEST.yaml",
                "red-team listed as a collaboration participant; the reviewer "
                "never takes part in affirmative collaboration")
        elif r not in AFFIRMATIVE:
            err("C5", "TASK_MANIFEST.yaml", f"unknown collaboration role {r!r}")

    if not os.path.isfile(log_p):
        err("C1", "COLLABORATION_LOG.yaml",
            "collaboration opened but no log was written")
        return
    try:
        log = yaml.safe_load(open(log_p, encoding="utf-8")) or {}
    except Exception as e:
        err("C1", "COLLABORATION_LOG.yaml", f"YAML parse failure: {e}")
        return

    if log.get("round", 1) != 1:
        err("C7", "COLLABORATION_LOG.yaml",
            f"round is {log.get('round')!r}; v1 permits exactly one")

    seen_ids = set()
    for i, m in enumerate(log.get("messages") or []):
        if not isinstance(m, dict):
            continue
        mid = m.get("message_id") or f"#{i}"
        if mid in seen_ids:
            err("C4", "COLLABORATION_LOG.yaml", f"duplicate message_id {mid}")
        seen_ids.add(mid)

        # C4: sender, recipient, reason, evidential references
        for f in ("from_role", "to_roles", "type", "question_or_finding",
                  "requested_action", "result_class", "candidate_relevance"):
            if not m.get(f):
                err("C4", "COLLABORATION_LOG.yaml",
                    f"{mid}: required field {f!r} empty")
        if m.get("from_role") == "red-team" or "red-team" in (m.get("to_roles") or []):
            err("C5", "COLLABORATION_LOG.yaml",
                f"{mid}: red-team appears as a collaboration participant")
        for r in (m.get("to_roles") or []):
            if r not in AFFIRMATIVE:
                err("C5", "COLLABORATION_LOG.yaml", f"{mid}: unknown recipient {r!r}")
        if m.get("type") and m["type"] not in MSG_TYPES:
            err("C4", "COLLABORATION_LOG.yaml",
                f"{mid}: type {m['type']!r} not in {sorted(MSG_TYPES)}")
        if m.get("result_class") and m["result_class"] not in RESULT_CLASSES:
            err("C4", "COLLABORATION_LOG.yaml",
                f"{mid}: result_class {m['result_class']!r} not in "
                f"{sorted(RESULT_CLASSES)}")

        # C6: a decision-relevant result must trace back to frozen evidence
        refs = ((m.get("supporting_canonical") or []) +
                (m.get("supporting_task_verified") or []) +
                (m.get("external_sources") or []))
        if not refs and not m.get("traces_to"):
            err("C6", "COLLABORATION_LOG.yaml",
                f"{mid}: no canonical/task-verified/external reference and no "
                f"traces_to; a collaboration result that points at nothing is "
                f"not evidence")
        if str(m.get("candidate_relevance", "")).lower() not in ("none", "") \
                and not m.get("traces_to"):
            err("C6", "COLLABORATION_LOG.yaml",
                f"{mid}: bears on {m['candidate_relevance']} but does not trace "
                f"back to a frozen first-pass report or a TV-*/EXT- item")

        # C8: atomic messages, not transcripts
        for f in ("question_or_finding", "response"):
            v = m.get(f)
            if isinstance(v, str) and len(v) > MAX_MSG_CHARS:
                err("C8", "COLLABORATION_LOG.yaml",
                    f"{mid}: {f} is {len(v)} chars. Messages are ATOMIC facts or "
                    f"questions; forwarding a peer's report or reasoning trace "
                    f"defeats the independence the first passes bought")

    # C9: revisions must not have been folded back into the frozen report
    for rev in (log.get("revisions") or []):
        if not isinstance(rev, dict):
            continue
        for f in ("role", "FIRST_PASS", "AFTER_CROSS_EXAMINATION", "WHY_CHANGED"):
            if not rev.get(f):
                err("C9", "COLLABORATION_LOG.yaml",
                    f"revision for {rev.get('role', '?')} missing {f!r}; a changed "
                    f"conclusion is recorded here, never by rewriting the report")


EXT_REQUIRED = ("id", "title", "year", "url_or_identifier", "discovery_method",
                "inspection_level", "what_it_establishes",
                "what_it_does_not_establish", "verified_by", "promotion_status")
# Levels at which a source has NOT actually been read.
NOT_INSPECTED = {"search_result_snippet", "abstract_only", "not_inspected",
                 "metadata_only"}


def check_external_sources(task):
    p = os.path.join(task, "TASK_EVIDENCE.yaml")
    if not os.path.isfile(p):
        return
    try:
        doc = yaml.safe_load(open(p, encoding="utf-8")) or {}
    except Exception:
        return
    ext = {}
    for item in doc.get("external_sources") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        ext[item["id"]] = item
        eid = item["id"]
        for f in EXT_REQUIRED:
            if item.get(f) in (None, ""):
                err("X1", "TASK_EVIDENCE.yaml",
                    f"{eid}: required field {f!r} missing. An external source "
                    f"without provenance cannot be re-checked by anyone")
        if item.get("promotion_status") not in (None, "", "proposed", "not_proposed"):
            err("X2", "TASK_EVIDENCE.yaml",
                f"{eid}: promotion_status {item['promotion_status']!r}; a task "
                f"may only PROPOSE promotion to research/state/sources/**")
        lvl = item.get("inspection_level")
        if lvl in NOT_INSPECTED and item.get("used_as_support") is True:
            err("X3", "TASK_EVIDENCE.yaml",
                f"{eid}: inspection_level={lvl!r} but marked used_as_support. A "
                f"search snippet or abstract is DISCOVERY, not evidence "
                f"(charter 4.1)")
        if lvl not in NOT_INSPECTED and not item.get("sections_inspected"):
            warn("X4", "TASK_EVIDENCE.yaml",
                 f"{eid}: claims inspection but records no sections/pages read")

    # a task-verified item derived from an external source must name it
    for tv in doc.get("task_verified") or []:
        if not isinstance(tv, dict):
            continue
        src = tv.get("external_source")
        if src and src not in ext:
            err("X1", "TASK_EVIDENCE.yaml",
                f"{tv.get('id')}: external_source {src!r} has no record under "
                f"external_sources")


def check_novelty_gate(task, texts):
    gate = read(task, "NOVELTY_GATE.md")
    cand = texts.get("CANDIDATES.md") or ""
    ids = re.findall(r"^##\s*Candidate\s+(\S+)", cand, re.MULTILINE)

    if gate is None:
        if NOVELTY_WORDS.search(cand):
            err("G3", "CANDIDATES.md",
                "novelty language is used but NOVELTY_GATE.md does not exist")
        return

    for cid in ids:
        if not re.search(rf"\b{re.escape(cid)}\b", gate):
            err("G2", "NOVELTY_GATE.md",
                f"candidate {cid} has no closest-predecessor record")

    if not any(c in gate.lower() for c in GATE_CLASSES):
        err("G3", "NOVELTY_GATE.md",
            f"no classification found; each candidate must be classified as one "
            f"of {GATE_CLASSES}")

    # G4: "no predecessor found" is a statement about the SEARCH. It must record
    # what was actually searched - canonical, task-verified external prior art,
    # and any external queries run - and must not be read as "novel".
    if "no predecessor found" in gate.lower():
        low = gate.lower()
        if "external" not in low and "websearch" not in low and "web search" not in low:
            err("G4", "NOVELTY_GATE.md",
                "a candidate is 'no predecessor found' but the gate records no "
                "external prior-art search. The local corpus is not exhaustive; "
                "either record the external search performed, or classify "
                "novelty as unresolved")
        if "searches actually performed" not in low and "searches performed" not in low:
            err("G4", "NOVELTY_GATE.md",
                "'no predecessor found' must be qualified in the file as meaning "
                "'none found under the searches actually performed'. It never "
                "means 'novel in the literature' (charter section 3, 4.2)")

    for name, body in (("CANDIDATES.md", cand),
                       ("RESEARCH_MEMO.md", texts.get("RESEARCH_MEMO.md") or ""),
                       ("RECOMMENDATION.md", texts.get("RECOMMENDATION.md") or "")):
        m = NOVELTY_WORDS.search(body)
        if m and not any(c in gate.lower() for c in GATE_CLASSES):
            err("G3", name,
                f"uses novelty language ({m.group(0)!r}) with no gate "
                f"classification in NOVELTY_GATE.md")


LEGACY_BANNER = """
*** LEGACY TASK (pre-2026-08-10 workflow, no TASK_MANIFEST.yaml) ***

This task predates the phase ledger, the pre-spec/results split, the analysis
spec and the novelty gate. Its findings below are therefore the AUDIT of a
historical run, not a to-do list.

DO NOT "FIX" THEM BY BACKFILLING. Writing a phase ledger for a task that has
already finished would fabricate close times, and adding a pre-specified
falsification plan after the answers are known is precisely the defect the
ledger exists to prevent. A preserved failing example is worth more than a
retrofitted passing one (charter section 4.4).

Exit code is 0 so that a deliberately preserved record does not fail a build.
Use --strict to see it as errors.
"""


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    strict = "--strict" in sys.argv
    if not argv:
        print(__doc__)
        return 2
    task = os.path.abspath(argv[0])
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not os.path.isdir(task):
        print(f"not a directory: {task}")
        return 2

    legacy = not os.path.isfile(os.path.join(task, "TASK_MANIFEST.yaml")) and not strict

    print(f"task: {os.path.relpath(task, repo)}")
    check(task, repo)
    if legacy:
        print(LEGACY_BANNER)
        for w in warns:
            print(w)
        for e in errors:
            print("LEGACY " + e[6:] if e.startswith("ERROR ") else e)
        print(f"\n{len(errors)} legacy finding(s), {len(warns)} warning(s) "
              f"- reported, not failed")
        return 0

    for w in warns:
        print(w)
    for e in errors:
        print(e)
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
