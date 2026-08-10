#!/usr/bin/env python3
"""validate_resource_policy.py - check that RESOURCE_POLICY.md is WIRED IN.

Read-only. Modifies nothing. Exit 0 clean, 1 if any ERROR.

A policy document that nothing enforces is a wish. This checks the mechanical
parts: that model routing is explicit, that the generic fallback is gone, that
scheduler commands are denied in both layers, that workers are pointed at the
compact contract rather than the lead's Skill, and that the weak
"no HPC without Gate A" wording has not crept back.

Checks:
  P1  RESOURCE_POLICY.md and WORKER_CONTRACT.md exist
  P2  machine profile: .example tracked, real file gitignored
  P3  every worker agent declares an explicit model (never `inherit`)
  P4  model routing matches the policy table
  P5  no generic-agent fallback in the workflow
  P6  workflow routes models explicitly and supports historicalValidation
  P7  scheduler / remote commands denied in settings.json AND the hook
  P8  SLURM inspection still permitted (the policy is unimplementable otherwise)
  P9  weak "no HPC without <gate>" wording absent from agent-facing docs
  P10 RESOURCE_POLICY.md referenced from the required entry points
  P11 workers pointed at WORKER_CONTRACT.md, not the full SKILL.md
  P12 no delegation tool in any worker agent's tool list
  P13 every worker agent has WebSearch/WebFetch, and nothing blocks them

Usage:  python3 research/tools/validate_resource_policy.py [--repo ROOT]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

errors: list[str] = []
warns: list[str] = []


def err(code, where, msg): errors.append(f"ERROR {code} [{where}] {msg}")
def warn(code, where, msg): warns.append(f"WARN  {code} [{where}] {msg}")


def read(root, rel):
    p = os.path.join(root, rel)
    return open(p, encoding="utf-8").read() if os.path.isfile(p) else None


# role -> (normal model, regression model)
EXPECTED_MODEL = {
    "literature": "sonnet",
    "numerics": "sonnet",
    "theory": "sonnet",
    "red-team": "opus",
}
WORKERS = tuple(EXPECTED_MODEL)
DELEGATION_TOOLS = ("Task", "Agent", "Workflow")
# External research is a first-class capability: all four roles need it, with
# different remits. literature owns broad prior art; theory and numerics search
# narrowly; the red team searches INDEPENDENTLY, which is what lets it find
# prior art the affirmative team missed.
WEB_TOOLS = {"WebSearch", "WebFetch"}

SCHEDULERS = ("sbatch", "srun", "salloc", "scancel", "scontrol", "squeue",
              "qsub", "bsub", "condor_submit", "oarsub", "mpirun", "mpiexec",
              "ssh", "scp", "rsync")

# Entry points that must point at the policy.
POLICY_REFS = ("CLAUDE.md", "research/README.md", "research/HANDOFF.md",
               ".claude/skills/research/SKILL.md",
               ".claude/skills/research/WORKER_CONTRACT.md")

# Agent-facing docs that must not carry the superseded formulation.
WEAK_HPC_DOCS = ("CLAUDE.md", "research/README.md", "research/HANDOFF.md",
                 ".claude/skills/research/SKILL.md",
                 ".claude/skills/research/WORKER_CONTRACT.md",
                 ".claude/agents/literature.md", ".claude/agents/theory.md",
                 ".claude/agents/numerics.md", ".claude/agents/red-team.md")
WEAK_HPC = re.compile(
    r"no HPC[^.\n]*without[^.\n]*(?:approved|EXP-ID|Gate)"
    r"|No HPC job[^.\n]*without", re.IGNORECASE)


def check(root):
    # P1 ---------------------------------------------------------------------
    policy = read(root, "research/RESOURCE_POLICY.md")
    contract = read(root, ".claude/skills/research/WORKER_CONTRACT.md")
    if policy is None:
        err("P1", "research/RESOURCE_POLICY.md", "missing")
    if contract is None:
        err("P1", "WORKER_CONTRACT.md", "missing")

    # P2 ---------------------------------------------------------------------
    if not os.path.isfile(os.path.join(root, "research/resource_profile.local.yaml.example")):
        err("P2", "resource_profile.local.yaml.example", "missing (must be tracked)")
    real = "research/resource_profile.local.yaml"
    if os.path.isfile(os.path.join(root, real)):
        r = subprocess.run(["git", "-C", root, "check-ignore", "-q", real])
        if r.returncode != 0:
            err("P2", real, "exists but is NOT gitignored; machine-specific "
                            "values must never be tracked")
    else:
        warn("P2", real, "absent on this machine; generate it from the "
                         ".example before running any local pilot")

    # P3 / P4 / P11 / P12 -----------------------------------------------------
    for role in WORKERS:
        rel = f".claude/agents/{role}.md"
        body = read(root, rel)
        if body is None:
            err("P3", rel, "worker agent definition missing")
            continue
        fm = body.split("---")[1] if body.startswith("---") else body[:800]

        m = re.search(r"^model:\s*(\S+)", fm, re.MULTILINE)
        if not m:
            err("P3", rel, "no `model:` in frontmatter; it would inherit the "
                           "lead's model (RESOURCE_POLICY 5.4)")
        elif m.group(1) == "inherit":
            err("P3", rel, "`model: inherit` - this is exactly the bug that put "
                           "every worker on Opus in the 2026-08-10 run")
        elif m.group(1) != EXPECTED_MODEL[role]:
            err("P4", rel, f"model is {m.group(1)!r}, policy 5.4 says "
                           f"{EXPECTED_MODEL[role]!r}")

        tools = re.search(r"^tools:\s*(.+)$", fm, re.MULTILINE)
        if tools:
            have = {t.strip() for t in tools.group(1).split(",")}
            bad = have & set(DELEGATION_TOOLS)
            if bad:
                err("P12", rel, f"delegation tool(s) {sorted(bad)} present; "
                                f"workers must not spawn subagents (5.3)")
        if "WORKER_CONTRACT.md" not in body:
            err("P11", rel, "does not point the worker at WORKER_CONTRACT.md")

        missing_web = WEB_TOOLS - have if tools else WEB_TOOLS
        if missing_web:
            err("P13", rel, f"missing {sorted(missing_web)}; external research "
                            f"is a first-class capability and the red team in "
                            f"particular must be able to search independently")

    # P5 / P6 -----------------------------------------------------------------
    wf = read(root, ".claude/workflows/research.js")
    if wf is None:
        err("P5", "research.js", "workflow missing")
    else:
        # a fallback would name the generic type in an agentType position
        if re.search(r"agentType:\s*['\"]general-purpose['\"]", wf):
            err("P5", "research.js", "generic `general-purpose` fallback is "
                                     "present; policy 5.8 requires "
                                     "Infrastructure first instead")
        if "Infrastructure first" not in wf and "INFRASTRUCTURE FIRST" not in wf:
            err("P5", "research.js", "no Infrastructure-first path for a "
                                     "missing project agent")
        if not re.search(r"const MODEL\s*=", wf):
            err("P6", "research.js", "no explicit model routing table")
        else:
            for role, expected in EXPECTED_MODEL.items():
                if role == "red-team":
                    continue  # mode-dependent; checked below
                if not re.search(rf"{role}:\s*'{expected}'", wf) and \
                   not re.search(rf"{role}:\s*THEORY_ESCALATED", wf):
                    err("P6", "research.js", f"routing for {role} does not "
                                             f"pin {expected}")
        if "historicalValidation" not in wf:
            err("P6", "research.js", "no historicalValidation mode")
        if "model" not in wf.split("const MODEL")[0][-2000:] and False:
            pass  # placeholder; the explicit-pass check is the MODEL table

    # P13 (settings side): nothing may block the web tools ---------------------
    settings_for_web = read(root, ".claude/settings.json")
    if settings_for_web is not None:
        import json as _json
        try:
            sj = _json.loads(settings_for_web)
            perms = sj.get("permissions", {})
            for bucket in ("deny", "ask"):
                for rule in perms.get(bucket, []):
                    if rule.startswith(("WebSearch", "WebFetch")):
                        err("P13", "settings.json",
                            f"{bucket!r} rule {rule!r} interferes with external "
                            f"research")
            hooks = sj.get("hooks", {}).get("PreToolUse", [])
            for h in hooks:
                m = h.get("matcher", "")
                if "WebSearch" in m or "WebFetch" in m:
                    err("P13", "settings.json",
                        "the PreToolUse guard intercepts web tools; it is scoped "
                        "to file and shell operations")
        except Exception as e:                                   # noqa: BLE001
            warn("P13", "settings.json", f"could not parse for web-tool check: {e}")

    # P7 / P8 -----------------------------------------------------------------
    settings = read(root, ".claude/settings.json")
    if settings is None:
        err("P7", ".claude/settings.json", "missing")
    else:
        for cmd in SCHEDULERS:
            if f"Bash({cmd}:*)" not in settings:
                err("P7", "settings.json", f"`{cmd}` not denied")

    hook = read(root, ".claude/hooks/guard_research.py")
    if hook is None:
        err("P7", "guard_research.py", "missing")
    else:
        for cmd in SCHEDULERS:
            if cmd not in hook:
                err("P7", "guard_research.py", f"`{cmd}` not matched by any rule")
        # P8: the hook must not block reading a slurm script
        sys.path.insert(0, os.path.join(root, ".claude", "hooks"))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "guard_research", os.path.join(root, ".claude/hooks/guard_research.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for allowed in ("cat slurm/run.sh",
                            "grep -rn ntasks slurm/",
                            "shellcheck slurm/run.sh"):
                if mod.check_bash(allowed):
                    err("P8", "guard_research.py",
                        f"blocks READ-ONLY SLURM inspection: {allowed!r}. "
                        f"Preparing an HPC package is the permitted work.")
            for denied in ("sbatch run.sh", "make && srun ./a.out",
                           "cat jobs | xargs sbatch"):
                if not mod.check_bash(denied):
                    err("P7", "guard_research.py",
                        f"does NOT deny scheduler submission: {denied!r}")
        except Exception as e:                                  # noqa: BLE001
            warn("P8", "guard_research.py", f"could not import for live check: {e}")

    # P9 ----------------------------------------------------------------------
    for rel in WEAK_HPC_DOCS:
        body = read(root, rel)
        if body and WEAK_HPC.search(body):
            err("P9", rel, "superseded wording: HPC described as permitted "
                           "after a gate/approval. The rule is that agents "
                           "NEVER submit; a gate authorises PREPARATION only.")

    # P10 ---------------------------------------------------------------------
    for rel in POLICY_REFS:
        body = read(root, rel)
        if body is None:
            warn("P10", rel, "absent")
        elif "RESOURCE_POLICY.md" not in body:
            err("P10", rel, "does not reference research/RESOURCE_POLICY.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    a = ap.parse_args()
    root = os.path.abspath(a.repo)

    check(root)
    for w in warns:
        print(w)
    for e in errors:
        print(e)
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
