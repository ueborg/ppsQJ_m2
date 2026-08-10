#!/usr/bin/env python3
"""guard_research.py - PreToolUse guard for the ppsQJ_m2 research engine.

Mechanically enforces the invariants that the Research Charter and
COWORK_AGENT_SPEC.md otherwise state only in prose. Permission rules in
settings.json match Bash commands by prefix and are therefore evadable by shell
composition (`true; git push`, `cd x && sbatch y`). This hook inspects the WHOLE
command string, so composition does not help.

Enforced:
  G1  no write to research/state/**            (agent spec invariant 1)
  G2  no `git push`                            (nothing leaves the machine)
  G3  no destructive git                       (reset --hard, clean -f,
                                                checkout -f, restore, worktree
                                                remove --force, branch -D, rm -rf)
  G4  no HPC, scheduler or remote execution    (SLURM/PBS/LSF/SGE/Condor/OAR,
                                                MPI launchers, ssh/scp/rsync).
                                                PERMANENT - no gate lifts it.
  G5  no manuscript modification               (*.tex, continuousmeasurementslatex/)
  G6  no invocation of known-wrong analysis    (analysis/anchor_scan.py,
                                                EV-CODE-ANCHORSCAN-001)
  G7  charter edits require confirmation       (ask, not deny - human-owned file)

G1 has a documented human override: set PPSQJ_ALLOW_STATE_WRITE=1 in the
environment. That is the merge step, performed by the researcher, never by an
agent or a workflow. No other rule has an override.

Protocol: PreToolUse hook. Reads the tool call as JSON on stdin, writes a
permission decision as JSON on stdout, exits 0. A crash here must not wedge the
session, so unexpected errors fall through to `allow` and say so on stderr -
EXCEPT for G1, which fails closed (see main()).
"""
from __future__ import annotations

import json
import os
import re
import sys

# --- what counts as a write -------------------------------------------------
WRITE_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit", "Update"}
PATH_KEYS = ("file_path", "notebook_path", "path", "filePath")

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

# --- protected zones --------------------------------------------------------
STATE_PREFIX = "research/state/"
MANUSCRIPT_DIRS = ("continuousmeasurementslatex/",)
CHARTER_FILE = "research/RESEARCH_CHARTER.md"

# --- command-position matching ----------------------------------------------
# G4 must deny EXECUTION of a scheduler and permit every form of PREPARATION,
# because RESOURCE_POLICY section 4 explicitly requires agents to write, read,
# grep and validate SLURM scripts and to prepare the exact submission command
# for the researcher. A rule that fires on the mere appearance of "sbatch"
# makes the permitted work impossible - `grep sbatch`, `sed s/sbatch/.../` and
# `echo "sbatch run.sh" > submit.txt` are all legitimate.
#
# So a scheduler name only counts when it sits in COMMAND POSITION: at the very
# start, or after a separator (; && || | newline backtick $( ), or behind a
# wrapper that would run it (xargs, nohup, time, sudo, env VAR=x).
_CMD_START = (
    r"(?:\A|[;&|(`\n]|\$\(|"
    r"\bxargs\s+(?:-\S+\s+|-\S+\s+\S+\s+)*|"
    r"\bnohup\s+|\btime\s+|\bsudo\s+|\benv\s+(?:\S+=\S+\s+)+)\s*"
)

SLURM = r"sbatch|srun|salloc|sattach|scancel|sinfo|squeue|sacctmgr|scontrol"
OTHER_SCHED = r"qsub|qdel|qalter|qrsh|bsub|bkill|condor_submit|condor_q|oarsub"
LAUNCHERS = r"mpirun|mpiexec|jsrun|aprun"
REMOTE = r"ssh|sftp|scp"


def _in_command_position(names: str) -> str:
    return _CMD_START + rf"(?:{names})\b"


# --- shell patterns ---------------------------------------------------------
# Matched against the full command string, case-insensitively, so that
# composition (`;`, `&&`, `||`, `|`, newline, `$(...)`) cannot slip past.
BASH_RULES: list[tuple[str, str, str]] = [
    # (rule id, regex, human-readable reason)
    # Git subcommands are matched within one command segment (up to ; & | or a
    # newline) so that global options such as `git -C /repo push` are covered.
    # A false positive here costs one denied call; a false negative costs a
    # push or a destroyed working tree.
    ("G2", r"\bgit\b[^;&|\n]*\bpush\b",
     "git push is forbidden. Research runs never publish. "
     "Pushing is a human action taken outside /research."),

    ("G3", r"\bgit\b[^;&|\n]*\breset\b[^;&|\n]*--hard\b",
     "git reset --hard destroys uncommitted work, which may be the only copy "
     "of an analysis (charter section 4.4: never overwrite results)."),
    ("G3", r"\bgit\b[^;&|\n]*\bclean\b[^;&|\n]*-[a-z]*f",
     "git clean -f deletes untracked files. Unregistered analysis outputs are "
     "still evidence candidates; deleting them is not recoverable."),
    ("G3", r"\bgit\b[^;&|\n]*\bcheckout\b[^;&|\n]*(?:--force|\s-f\b)",
     "forced checkout discards working-tree changes."),
    ("G3", r"\bgit\b[^;&|\n]*\brestore\b",
     "git restore overwrites working-tree files."),
    ("G3", r"\bgit\b[^;&|\n]*\bbranch\b[^;&|\n]*\s-D\b",
     "forced branch deletion can orphan the only reference to a commit."),
    ("G3", r"\bgit\b[^;&|\n]*\bworktree\s+remove\b[^;&|\n]*--force",
     "forced worktree removal discards uncommitted work in the worktree."),
    ("G3", r"\brm\s+(?:-[a-z]*\s+)*-[a-z]*r[a-z]*f|\brm\s+(?:-[a-z]*\s+)*-[a-z]*f[a-z]*r",
     "recursive forced delete."),

    # G4 is PERMANENT and has no gate. Agents never submit HPC or remote jobs -
    # not during /research, not after Gate A, not after experiment approval,
    # not after a successful local pilot, not when HPC access returns.
    # Preparing a package is allowed; executing it is not. See
    # research/RESOURCE_POLICY.md section 4.
    #
    # Covered: SLURM, PBS/Torque, LSF, SGE, HTCondor, OAR, plus MPI launchers
    # and container runners that would start work on a remote or oversubscribed
    # target. Matching is on the whole command string, so `cd x && sbatch y`
    # and `xargs sbatch` are caught too.
    ("G4", _in_command_position(SLURM),
     "SLURM submission or control. Agents NEVER submit HPC jobs - not at any "
     "stage, not after any gate, not after a local pilot, not when HPC access "
     "returns. Prepare the package and stop at READY_FOR_HUMAN_SUBMISSION; the "
     "researcher submits manually (research/RESOURCE_POLICY.md section 4). "
     "Reading, writing, grepping and validating SLURM scripts IS allowed - "
     "this rule only fires on execution."),
    ("G4", _in_command_position(OTHER_SCHED),
     "PBS/Torque, LSF, SGE, HTCondor or OAR job submission. Same permanent "
     "rule: agents never submit to a scheduler."),
    ("G4", _in_command_position(LAUNCHERS),
     "parallel job launcher. /research runs single, local, read-only analysis; "
     "a launcher here is either an HPC job or a local oversubscription."),
    ("G4", _in_command_position(REMOTE),
     "remote execution or transfer. Ruche access is a human action."),
    ("G4", r"\brsync\b[^;&|\n]*(?:::|\S+@\S+:)",
     "remote rsync transfer. Local rsync is fine; pushing to or pulling from a "
     "remote host is a human action."),
    # Indirect execution: a scheduler hidden behind an interpreter. Denied
    # regardless of position, because `bash -c "sbatch x"` is execution, not
    # preparation.
    ("G4", rf"\b(?:bash|sh|zsh|eval)\s+-c\b[^\n]*\b(?:{SLURM}|{OTHER_SCHED}|{LAUNCHERS})\b",
     "a scheduler command wrapped in an interpreter. Wrapping does not make it "
     "preparation; agents never submit."),

    # Match INVOCATION, not mention. Writing documentation that names the
    # script - as this project's own docs must - is not running it.
    ("G6", r"(?:python[0-9.]*|ipython|pypy[0-9.]*|/bin/python[0-9.]*|"
           r"bash|sh|zsh|exec|source|\./)\s*[^\n;&|]*anchor_scan"
           r"|\bimport\s+[\w.]*anchor_scan"
           r"|\bfrom\s+[\w.]*anchor_scan\b",
     "analysis/anchor_scan.py is KNOWN WRONG (EV-CODE-ANCHORSCAN-001): its "
     "kernel drops the hopping w from the measured bond and it produces "
     "plausible-looking output anyway. It may not produce evidence."),
]

# Shell constructs that write to a path: redirection, sed -i, and the usual
# file-mutating utilities. Used to catch `echo x > research/state/...`.
SHELL_WRITE_RE = re.compile(
    r"(?:>>?\s*|(?:\bsed\b[^|;&]*-i[^|;&]*\s)|\b(?:cp|mv|rm|install|touch|tee|dd)\b[^|;&]*\s)"
    r"['\"]?([^\s'\";|&>]+)",
    re.IGNORECASE,
)


def _rel(path: str) -> str:
    """Project-relative, forward-slashed, for prefix tests."""
    if not path:
        return ""
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_DIR, p)
    p = os.path.normpath(p)
    try:
        rel = os.path.relpath(p, PROJECT_DIR)
    except ValueError:          # different drive; cannot be inside the project
        return ""
    return rel.replace(os.sep, "/")


def _state_write_allowed() -> bool:
    return os.environ.get("PPSQJ_ALLOW_STATE_WRITE") == "1"


def check_path_write(rel: str) -> tuple[str, str] | None:
    """Return (decision, reason) if a write to `rel` is not permitted."""
    if not rel or rel.startswith("../"):
        return None

    if rel.startswith(STATE_PREFIX):
        if _state_write_allowed():
            return None
        return ("deny",
                f"G1 research/state/** is canonical scientific state and is "
                f"single-writer. No agent and no workflow may modify it "
                f"({rel}). Emit a PROPOSAL under research/tasks/active/<TASK-ID>/ "
                f"or research/proposals/ instead; the researcher merges it at "
                f"the human gate. (Override for the human merge step only: "
                f"PPSQJ_ALLOW_STATE_WRITE=1.)")

    if rel.endswith(".tex") or any(rel.startswith(d) for d in MANUSCRIPT_DIRS):
        return ("deny",
                f"G5 manuscripts are out of scope and are not modified by "
                f"research runs ({rel}). Charter section 4.3: prose generation "
                f"is the final stage, not the research process.")

    if rel == CHARTER_FILE:
        return ("ask",
                "G7 research/RESEARCH_CHARTER.md is human-owned (charter "
                "section 3). Confirm this edit was explicitly authorised.")

    return None


def check_bash(command: str) -> tuple[str, str] | None:
    for rule_id, pattern, reason in BASH_RULES:
        if re.search(pattern, command, re.IGNORECASE):
            return ("deny", f"{rule_id} {reason}")

    # shell-level writes into protected zones
    for m in SHELL_WRITE_RE.finditer(command):
        verdict = check_path_write(_rel(m.group(1)))
        if verdict:
            decision, reason = verdict
            return (decision, reason + "  [reached via a shell command rather "
                                        "than the Edit tool]")
    return None


def evaluate(tool_name: str, tool_input: dict) -> tuple[str, str] | None:
    if tool_name == "Bash":
        return check_bash(tool_input.get("command") or "")

    if tool_name in WRITE_TOOLS:
        for key in PATH_KEYS:
            if tool_input.get(key):
                verdict = check_path_write(_rel(tool_input[key]))
                if verdict:
                    return verdict
        # MultiEdit-style batches
        for edit in tool_input.get("edits") or []:
            if isinstance(edit, dict) and edit.get("file_path"):
                verdict = check_path_write(_rel(edit["file_path"]))
                if verdict:
                    return verdict
    return None


def emit(decision: str, reason: str) -> None:
    json.dump({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }}, sys.stdout)
    sys.stdout.write("\n")


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
        tool_name = payload.get("tool_name", "")
        tool_input = payload.get("tool_input") or {}
    except Exception as exc:                                # noqa: BLE001
        # Malformed payload. Fail CLOSED only for the invariant that matters:
        # if we cannot parse it, we cannot prove it is not a state write.
        print(f"guard_research: unparseable hook payload: {exc}", file=sys.stderr)
        emit("ask", "G0 the research guard could not parse this tool call and "
                    "cannot confirm it does not modify research/state/**.")
        return 0

    try:
        verdict = evaluate(tool_name, tool_input)
    except Exception as exc:                                # noqa: BLE001
        print(f"guard_research: internal error: {exc}", file=sys.stderr)
        emit("ask", f"G0 the research guard errored ({exc}) and could not "
                    f"verify this call against the charter invariants.")
        return 0

    if verdict:
        emit(*verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
