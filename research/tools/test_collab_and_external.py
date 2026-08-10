#!/usr/bin/env python3
"""test_collab_and_external.py - tests for the two capabilities added on top of
the v1 defect fixes: genuine external research, and ONE bounded affirmative
collaboration round after independent first passes.

    .venv/bin/python3 research/tools/test_collab_and_external.py

Read-only against the repo; every case builds a throwaway task under a temp
dir. Launches no agents, no workflows, no simulations. Web tools are checked by
STATIC inspection of the agent definitions and settings - nothing here performs
a network request.

  T1  first-pass reports are immutable once collaboration starts
  T2  collaboration cannot open before every dispatched worker has reported
  T3  collaboration is optional (COLLABORATION_NOT_NEEDED)
  T4  a second collaboration round is rejected
  T5  a message needs sender, recipient, reason and evidential references
  T6  peer transcripts are not forwarded (atomic messages only)
  T7  the red team is not an affirmative collaboration participant
  T8  the red team retains independent WebSearch/WebFetch
  T9  an external task-verified source supports a check without becoming canonical
  T10 the novelty gate consults external prior art
  T11 a search-result snippet cannot be registered as support
  T12 off-scope findings can be parked without further investigation
  T13 every project agent has its intended web tools
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
TEMPLATE = os.path.join(ROOT, "research", "tasks", "TASK_TEMPLATE")
results: list[tuple[bool, str, str]] = []


def check(ok, name, detail=""):
    results.append((ok, name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail and not ok else ""))


def run(*args, cwd=ROOT):
    return subprocess.run([PY, *args], capture_output=True, text=True, cwd=cwd)


def tool(n):
    return os.path.join(ROOT, "research", "tools", n)


def write(d, rel, text):
    p = os.path.join(d, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def task_to_first_pass(tmp, name, workers=("literature", "theory", "numerics")):
    """A task with Stage 1 closed and workers dispatched (reports optional)."""
    d = os.path.join(tmp, name)
    shutil.copytree(TEMPLATE, d)
    os.remove(os.path.join(d, "README.md"))
    run(tool("task_phase.py"), d, "init", name)
    for rel in ("CHARTER.md", "PROBLEM_MEMO.md", "SOURCE_REGISTER.md"):
        write(d, rel, "# x\n[E] y\n")
    run(tool("task_phase.py"), d, "close", "stage_1_problem")
    args = []
    for w in workers:
        args += ["--worker", f"{w}=sonnet"]
    run(tool("task_phase.py"), d, "dispatch", *args)
    return d


def open_collab(d):
    return run(tool("task_phase.py"), d, "collaborate", "open",
               "--dependency", "theory must check an assumption literature found",
               "--roles", "literature,theory", "--question", "does X transfer?",
               "--value", "it separates C1 from C2")


# --- T1 / T2 ---------------------------------------------------------------
def test_first_pass(tmp):
    d = task_to_first_pass(tmp, "TASK-T1")

    # T2: cannot freeze first passes while a dispatched worker has no report
    write(d, "agent_reports/literature.json", '{"summary":"a"}')
    r = run(tool("task_phase.py"), d, "close", "first_pass_frozen")
    check(r.returncode != 0 and "theory" in (r.stdout + r.stderr),
          "T2  first_pass_frozen refuses while a dispatched worker has no report",
          (r.stdout + r.stderr).strip()[:200])

    r2 = open_collab(d)
    check(r2.returncode != 0,
          "T2  collaboration refuses to open before first passes are frozen")

    # now complete and freeze
    write(d, "agent_reports/theory.json", '{"summary":"b"}')
    write(d, "agent_reports/numerics.json", '{"summary":"c"}')
    r3 = run(tool("task_phase.py"), d, "close", "first_pass_frozen")
    check(r3.returncode == 0 and "agent_reports/theory.json" in r3.stdout,
          "T2  ... and closes once every dispatched worker has reported",
          r3.stdout.strip()[:200])

    # T1: the reports are now immutable
    r4 = open_collab(d)
    write(d, "agent_reports/theory.json", '{"summary":"b, revised after seeing literature"}')
    chk = run(tool("task_phase.py"), d, "check")
    v = run(tool("validate_task.py"), d)
    check(r4.returncode == 0, "T1  collaboration opens once first passes are frozen")
    check(chk.returncode == 1 and "agent_reports/theory.json" in chk.stdout,
          "T1  a first-pass report edited after collaboration starts -> phase check FAILS")
    check("M5" in v.stdout and "agent_reports/theory.json" in v.stdout,
          "T1  ... and validate_task reports M5")


# --- T3 / T4 ---------------------------------------------------------------
def test_optional_and_single_round(tmp):
    d = task_to_first_pass(tmp, "TASK-T3")
    for w in ("literature", "theory", "numerics"):
        write(d, f"agent_reports/{w}.json", '{"summary":"x"}')
    run(tool("task_phase.py"), d, "close", "first_pass_frozen")

    # T3: optional
    r = run(tool("task_phase.py"), d, "collaborate", "none",
            "--reason", "no cross-role dependency; all three answered independently")
    man = open(os.path.join(d, "TASK_MANIFEST.yaml"), encoding="utf-8").read()
    v = run(tool("validate_task.py"), d)
    check(r.returncode == 0 and "COLLABORATION_NOT_NEEDED" in man,
          "T3  collaboration is optional: COLLABORATION_NOT_NEEDED is recordable")
    check(not re.search(r"ERROR C\d", v.stdout),
          "T3  ... and skipping it raises no collaboration error",
          "\n".join(l for l in v.stdout.splitlines() if "ERROR C" in l)[:200])

    r2 = run(tool("task_phase.py"), d, "collaborate", "none")
    check(r2.returncode != 0,
          "T3  ... but skipping it still requires a stated reason")

    # T4: one round only
    d2 = task_to_first_pass(tmp, "TASK-T4")
    for w in ("literature", "theory", "numerics"):
        write(d2, f"agent_reports/{w}.json", '{"summary":"x"}')
    run(tool("task_phase.py"), d2, "close", "first_pass_frozen")
    first = open_collab(d2)
    second = open_collab(d2)
    check(first.returncode == 0 and second.returncode != 0
          and "ONE" in (second.stdout + second.stderr),
          "T4  a second collaboration round is rejected",
          (second.stdout + second.stderr).strip()[:200])

    # unjustified opening is refused
    d3 = task_to_first_pass(tmp, "TASK-T4b")
    for w in ("literature", "theory", "numerics"):
        write(d3, f"agent_reports/{w}.json", '{"summary":"x"}')
    run(tool("task_phase.py"), d3, "close", "first_pass_frozen")
    r3 = run(tool("task_phase.py"), d3, "collaborate", "open", "--roles", "theory")
    check(r3.returncode != 0 and "--dependency" in (r3.stdout + r3.stderr),
          "T4  opening without a recorded dependency/question/value is refused")


# --- T5 / T6 / T7 ----------------------------------------------------------
GOOD_MSG = """schema_version: 1
task_id: TASK-T5
round: 1
messages:
  - message_id: M1
    from_role: literature
    to_roles: [theory]
    type: theoretical_assumption_check
    question_or_finding: "SRC-KMR-2023 states B_L as a product of averages."
    supporting_canonical: [SRC-KMR-2023]
    supporting_task_verified: [TV-1]
    external_sources: []
    requested_action: "Do KMR's assumptions transfer to our Cut B?"
    response: "No: their duality argument needs w = 0."
    result_class: narrowed
    candidate_relevance: C1
    traces_to: agent_reports/literature.json
"""


def test_messages(tmp):
    d = task_to_first_pass(tmp, "TASK-T5")
    for w in ("literature", "theory", "numerics"):
        write(d, f"agent_reports/{w}.json", '{"summary":"x"}')
    run(tool("task_phase.py"), d, "close", "first_pass_frozen")
    open_collab(d)

    write(d, "COLLABORATION_LOG.yaml", GOOD_MSG)
    v = run(tool("validate_task.py"), d)
    check(not re.search(r"ERROR C[0-9]", v.stdout),
          "T5  a well-formed atomic message validates", v.stdout[:300])

    # missing sender / recipient / action / references
    bad = GOOD_MSG.replace("    from_role: literature\n", "")
    bad = bad.replace("    requested_action: \"Do KMR's assumptions transfer to our Cut B?\"\n", "")
    write(d, "COLLABORATION_LOG.yaml", bad)
    v2 = run(tool("validate_task.py"), d)
    check("C4" in v2.stdout and "from_role" in v2.stdout,
          "T5  a message without a sender or requested action -> C4")

    noref = GOOD_MSG.replace("    supporting_canonical: [SRC-KMR-2023]\n", "    supporting_canonical: []\n")
    noref = noref.replace("    supporting_task_verified: [TV-1]\n", "    supporting_task_verified: []\n")
    noref = noref.replace("    traces_to: agent_reports/literature.json\n", "")
    write(d, "COLLABORATION_LOG.yaml", noref)
    v3 = run(tool("validate_task.py"), d)
    check("C6" in v3.stdout,
          "T5  a message pointing at no evidence and tracing to nothing -> C6")

    # T6: a forwarded transcript is not an atomic message
    huge = GOOD_MSG.replace('response: "No: their duality argument needs w = 0."',
                            'response: "' + ("peer reasoning trace. " * 120) + '"')
    write(d, "COLLABORATION_LOG.yaml", huge)
    v4 = run(tool("validate_task.py"), d)
    check("C8" in v4.stdout,
          "T6  forwarding a peer transcript instead of an atomic fact -> C8")

    # T7: red team may not participate
    rt = GOOD_MSG.replace("    to_roles: [theory]", "    to_roles: [red-team]")
    write(d, "COLLABORATION_LOG.yaml", rt)
    v5 = run(tool("validate_task.py"), d)
    check("C5" in v5.stdout,
          "T7  red-team as a collaboration recipient -> C5")

    d2 = task_to_first_pass(tmp, "TASK-T7b")
    for w in ("literature", "theory", "numerics"):
        write(d2, f"agent_reports/{w}.json", '{"summary":"x"}')
    run(tool("task_phase.py"), d2, "close", "first_pass_frozen")
    r = run(tool("task_phase.py"), d2, "collaborate", "open",
            "--dependency", "d", "--roles", "theory,red-team",
            "--question", "q", "--value", "v")
    check(r.returncode != 0 and "red team" in (r.stdout + r.stderr).lower(),
          "T7  ... and the ledger refuses to enrol it as a participant",
          (r.stdout + r.stderr).strip()[:200])


# --- T8 / T13 --------------------------------------------------------------
EXPECTED_TOOLS = {
    "literature": {"WebSearch", "WebFetch"},
    "theory": {"WebSearch", "WebFetch"},
    "numerics": {"WebSearch", "WebFetch"},
    "red-team": {"WebSearch", "WebFetch"},
}


def test_web_tools(_tmp):
    settings = open(os.path.join(ROOT, ".claude/settings.json"), encoding="utf-8").read()
    for role, want in EXPECTED_TOOLS.items():
        p = os.path.join(ROOT, ".claude", "agents", f"{role}.md")
        body = open(p, encoding="utf-8").read()
        m = re.search(r"^tools:\s*(.+)$", body, re.MULTILINE)
        have = {t.strip() for t in m.group(1).split(",")} if m else set()
        ok = want <= have
        label = ("T8 " if role == "red-team" else "T13") + f" {role} has {sorted(want)}"
        check(ok, label, f"declared: {sorted(have)}")
    # no project rule blocks the web tools
    blocked = [ln for ln in settings.splitlines()
               if re.search(r'"(?:deny|ask)"', ln)] and re.findall(
        r'"(WebSearch|WebFetch)[^"]*"', settings)
    check(not blocked,
          "T13 no project deny/ask rule blocks WebSearch or WebFetch",
          f"found: {blocked}")
    # the guard hook does not intercept web tools
    hook_settings = re.search(r'"matcher":\s*"([^"]+)"', settings)
    matcher = hook_settings.group(1) if hook_settings else ""
    check("WebSearch" not in matcher and "WebFetch" not in matcher,
          "T13 the PreToolUse guard does not intercept web tools",
          f"matcher: {matcher}")


# --- T9 / T11 --------------------------------------------------------------
EXT_GOOD = """schema_version: 1
task_id: TASK-T9
task_verified:
  - id: TV-1
    tier: task_verified
    kind: source_inspection
    source_or_artifact: EXT-1
    external_source: EXT-1
    what_was_verified: "nu = 2.06 for the 2D class-DIII network model"
    verified_by: literature
    date: 2026-08-10
    canonical: false
    promotion_status: proposed
external_sources:
  - id: EXT-1
    title: "Thermal metal-insulator transition in a helical topological superconductor"
    authors: "Fulga et al."
    year: 2012
    doi_or_arxiv: "arXiv:1205.1441"
    url_or_identifier: "https://arxiv.org/abs/1205.1441"
    discovery_method: "WebSearch: class DIII thermal metal-insulator exponent"
    inspection_level: relevant_sections
    sections_inspected: "Sec. V D, Table I"
    what_it_establishes: "nu = 2.06 [1.89, 2.20]"
    what_it_does_not_establish: "nothing about monitoring or the Born rule"
    verified_by: literature
    derived_task_evidence: [TV-1]
    used_as_support: true
    promotion_status: proposed
"""


def test_external(tmp):
    d = task_to_first_pass(tmp, "TASK-T9")
    for w in ("literature", "theory", "numerics"):
        write(d, f"agent_reports/{w}.json", '{"summary":"x"}')
    run(tool("task_phase.py"), d, "close", "first_pass_frozen")
    open_collab(d)
    write(d, "TASK_EVIDENCE.yaml", EXT_GOOD)
    write(d, "COLLABORATION_LOG.yaml",
          GOOD_MSG.replace("task_id: TASK-T5", "task_id: TASK-T9")
                  .replace("    external_sources: []", "    external_sources: [EXT-1]"))
    v = run(tool("validate_task.py"), d)
    check(not re.search(r"ERROR X[0-9]", v.stdout) and not re.search(r"ERROR C[0-9]", v.stdout),
          "T9  an external task-verified source supports a collaboration check",
          v.stdout[:300])
    ev = open(os.path.join(d, "TASK_EVIDENCE.yaml"), encoding="utf-8").read()
    check("canonical: false" in ev and "promotion_status: proposed" in ev,
          "T9  ... while remaining non-canonical and only PROPOSED for promotion")

    # canonical state untouched by any of this
    st = subprocess.run(["git", "status", "--porcelain", "research/state/"],
                        capture_output=True, text=True, cwd=ROOT)
    check(st.stdout.strip() == "",
          "T9  ... and research/state/** is unchanged")

    # T11: a snippet cannot be support
    snip = EXT_GOOD.replace("inspection_level: relevant_sections",
                            "inspection_level: search_result_snippet")
    write(d, "TASK_EVIDENCE.yaml", snip)
    v2 = run(tool("validate_task.py"), d)
    check("X3" in v2.stdout,
          "T11 a search-result snippet marked used_as_support -> X3")

    # missing provenance
    noprov = EXT_GOOD.replace('    url_or_identifier: "https://arxiv.org/abs/1205.1441"\n', "")
    noprov = noprov.replace('    discovery_method: "WebSearch: class DIII thermal metal-insulator exponent"\n', "")
    write(d, "TASK_EVIDENCE.yaml", noprov)
    v3 = run(tool("validate_task.py"), d)
    check("X1" in v3.stdout,
          "T11 an external source without URL/discovery provenance -> X1")


# --- T10 -------------------------------------------------------------------
def test_novelty_external(tmp):
    d = task_to_first_pass(tmp, "TASK-T10")
    write(d, "CANDIDATES.md", "## Candidate C1\n1. **Statement.** a new finding\n")

    write(d, "NOVELTY_GATE.md",
          "# gate\n| candidate | predecessor | classification |\n"
          "| C1 | none | no predecessor found |\n")
    v = run(tool("validate_task.py"), d)
    check("G4" in v.stdout,
          "T10 'no predecessor found' with no external search recorded -> G4")

    write(d, "NOVELTY_GATE.md",
          "# gate\n| candidate | predecessor | classification |\n"
          "| C1 | none | no predecessor found |\n\n"
          "## External prior-art search\n"
          "| candidate | external queries run | sources inspected | prior art? |\n"
          "| C1 | WebSearch: monitored free fermion boundary amplitude | EXT-1 | no |\n\n"
          "`no predecessor found` means only: none found under the searches "
          "actually performed. It does not mean novel in the literature.\n")
    v2 = run(tool("validate_task.py"), d)
    check("G4" not in v2.stdout,
          "T10 ... recording the external search and the caveat clears it",
          v2.stdout[:300])


# --- T12 -------------------------------------------------------------------
def test_parking(tmp):
    d = task_to_first_pass(tmp, "TASK-T12")
    for w in ("literature", "theory", "numerics"):
        write(d, f"agent_reports/{w}.json", '{"summary":"x"}')
    run(tool("task_phase.py"), d, "close", "first_pass_frozen")
    write(d, "PARKING_LOT.md",
          "# parked\n| what | found by | why off-scope | might bear on | worth a task? |\n"
          "|---|---|---|---|---|\n"
          "| a second attribution error in SRC-FAVA-2023 | literature | not "
          "load-bearing for this question | CB-NLSM-001 | probably |\n")
    v = run(tool("validate_task.py"), d)
    parked_errs = [l for l in v.stdout.splitlines()
                   if re.search(r"ERROR (?:C|X)\d", l) or "PARKING" in l.upper()]
    check(not parked_errs,
          "T12 an off-scope finding can be parked with no further investigation",
          "\n".join(parked_errs)[:300])
    body = open(os.path.join(d, "PARKING_LOT.md"), encoding="utf-8").read()
    check("worth a task?" in body,
          "T12 ... and the parked row records whether it deserves its own task")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ppsqj_collab_") as tmp:
        for fn in (test_first_pass, test_optional_and_single_round, test_messages,
                   test_web_tools, test_external, test_novelty_external, test_parking):
            print(f"\n--- {fn.__name__} ---")
            fn(tmp)
    passed = sum(1 for ok, _, _ in results if ok)
    print(f"\n{passed}/{len(results)} passed")
    for ok, name, _ in results:
        if not ok:
            print(f"  FAILED: {name}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
