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
  P14 the tier table agrees across model_routing.yaml, RESOURCE_POLICY.md,
      the four agent definitions and the workflow (four copies, one truth)
  P15 only supported aliases are routed, no pinned version IDs, and Tier 3
      degrades through a fallback chain instead of crashing the run
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


# Adaptive routing (RESOURCE_POLICY 5.4, research/model_routing.yaml).
# Tier -> alias.
TIER_ALIAS = {"tier_1": "sonnet", "tier_2": "opus", "tier_3": "best"}
# Aliases this Claude Code build accepts. Verified against the installed CLI's
# alias list. `best` is real and resolves to the strongest available model,
# degrading to Opus-class where Fable is absent - which is exactly why routing
# uses it rather than pinning `fable`.
SUPPORTED_ALIASES = {"sonnet", "opus", "haiku", "fable", "best"}

# Role defaults per posture. Four copies of this table exist (YAML, policy
# prose, agent frontmatter, workflow) because a workflow script cannot read a
# file at run time. They are CHECKED against each other, not trusted.
POSTURE_TIERS = {
    "economical": {"literature": "tier_1", "theory": "tier_1",
                   "numerics": "tier_1", "red-team": "tier_2"},
    "normal": {"literature": "tier_1", "theory": "tier_2",
               "numerics": "tier_1", "red-team": "tier_2"},
    "deep": {"literature": "tier_1", "theory": "tier_3",
             "numerics": "tier_1", "red-team": "tier_2"},
}
# An agent definition's frontmatter is the `normal` posture default: it is what
# the role gets when nothing routes it explicitly.
EXPECTED_MODEL = {r: TIER_ALIAS[t] for r, t in POSTURE_TIERS["normal"].items()}
WORKERS = tuple(EXPECTED_MODEL)
REGRESSION_TIERS = {r: "tier_1" for r in WORKERS}
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
        elif m.group(1) not in SUPPORTED_ALIASES:
            err("P15", rel, f"model {m.group(1)!r} is not a supported Claude "
                            f"Code alias {sorted(SUPPORTED_ALIASES)}; do not "
                            f"pin version IDs (policy 5.4g)")
        elif m.group(1) != EXPECTED_MODEL[role]:
            err("P4", rel, f"model is {m.group(1)!r}; the `normal` posture "
                           f"default for {role} is {EXPECTED_MODEL[role]!r} "
                           f"(policy 5.4b). Frontmatter is the default, not a "
                           f"ceiling - the lead still routes per subproblem.")

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
        if not re.search(r"const POSTURE_DEFAULTS\s*=", wf):
            err("P6", "research.js", "no explicit per-posture model routing table")
        if not re.search(r"agent\([^)]*\{[^}]*model", wf, re.DOTALL) and \
           "agentType: role, model" not in wf:
            err("P6", "research.js", "agent() is called without an explicit "
                                     "model; a worker would inherit the lead's")
        if "historicalValidation" not in wf:
            err("P6", "research.js", "no historicalValidation mode")

        # P14: the four copies of the tier table must agree ------------------
        for posture, roles in POSTURE_TIERS.items():
            row = re.search(rf"\n  {posture}:\s*\{{([^}}]*)\}}", wf)
            if not row:
                err("P14", "research.js", f"no `{posture}` posture row in "
                                          f"POSTURE_DEFAULTS")
                continue
            for role, tier in roles.items():
                if not re.search(rf"'?{role}'?:\s*'{tier}'", row.group(1)):
                    err("P14", "research.js",
                        f"posture {posture}: {role} is not {tier}. The four "
                        f"copies of this table must agree "
                        f"(research/model_routing.yaml is the source).")
        for role, tier in REGRESSION_TIERS.items():
            if not re.search(rf"REGRESSION_TIERS = \{{[^}}]*'?{role}'?:\s*'{tier}'", wf):
                err("P14", "research.js",
                    f"regression mode does not pin {role} to {tier}; "
                    f"historical validation must stay Tier 1")

        # P15: aliases and safe degradation ---------------------------------
        chains = re.search(r"const FALLBACK = \{(.*?)\n\}", wf, re.DOTALL)
        if not chains:
            err("P15", "research.js", "no FALLBACK chains; a Tier-3 request "
                                      "that the runtime rejects would crash "
                                      "the run (policy 5.4g)")
        else:
            used = set(re.findall(r"'([a-z0-9\[\]-]+)'", chains.group(1)))
            bad = used - SUPPORTED_ALIASES
            if bad:
                err("P15", "research.js", f"routes unsupported alias(es) "
                                          f"{sorted(bad)}; allowed: "
                                          f"{sorted(SUPPORTED_ALIASES)}")
            if not re.search(r"tier_3:\s*\['best'", chains.group(1)):
                err("P15", "research.js", "the tier_3 chain does not start at "
                                          "`best`; pinning `fable` breaks in "
                                          "environments without it (5.4g)")
        if re.search(r"claude-(?:opus|sonnet|fable|haiku)-[0-9]", wf):
            err("P15", "research.js", "a pinned model version ID is routed; "
                                      "use aliases (5.4g)")
        if "escalationsRefused" not in wf or "decision_at_stake" not in wf:
            err("P15", "research.js", "escalations are not recorded with a "
                                      "decision_at_stake (policy 5.4e)")

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

    # P14 (YAML + policy side) -------------------------------------------------
    routing = read(root, "research/model_routing.yaml")
    if routing is None:
        err("P14", "research/model_routing.yaml",
            "missing; the machine-readable routing table is the source the "
            "other three copies are checked against")
    else:
        for tier, alias in TIER_ALIAS.items():
            if not re.search(rf"{tier}:\s*\n\s*alias:\s*{alias}\b", routing):
                err("P14", "model_routing.yaml", f"{tier} does not map to {alias}")
        for role in WORKERS:
            pat = (rf"\n  {role}:\n    default_tier:\n"
                   rf"      economical: {POSTURE_TIERS['economical'][role]}\n"
                   rf"      normal: {POSTURE_TIERS['normal'][role]}\n"
                   rf"      deep: {POSTURE_TIERS['deep'][role]}\n")
            if not re.search(pat, routing):
                err("P14", "model_routing.yaml",
                    f"`{role}` default_tier block missing or disagrees with "
                    f"the policy table")
        if "failure_required_first: false" not in routing:
            err("P14", "model_routing.yaml",
                "the withdrawn 'escalate only after the cheap model failed' "
                "rule is not explicitly recorded as withdrawn (policy 5.4d)")
        for v in ("changed_conclusion", "new_derivation", "caught_error",
                  "confirmed_existing", "no_material_gain"):
            if v not in routing:
                err("P14", "model_routing.yaml",
                    f"material_value value {v!r} missing; stronger-model "
                    f"spending cannot be scored afterwards (policy 5.11)")

    if policy:
        for posture in POSTURE_TIERS:
            if f"**{posture}**" not in policy and f"`{posture}`" not in policy:
                err("P14", "RESOURCE_POLICY.md",
                    f"posture `{posture}` is not documented (5.4i)")
        for role in WORKERS:
            row = re.search(rf"^\|\s*`{role}`\s*\|(.+)$", policy, re.MULTILINE)
            if not row:
                err("P14", "RESOURCE_POLICY.md",
                    f"no routing row for `{role}` in the 5.4b table")
                continue
            cells = [c.strip().lower() for c in row.group(1).split("|")]
            for posture, tiers in POSTURE_TIERS.items():
                want = tiers[role].replace("_", " ")
                if not any(want in c for c in cells):
                    err("P14", "RESOURCE_POLICY.md",
                        f"`{role}` row does not show {want} for {posture}: "
                        f"{row.group(1).strip()}")
        if "model_routing.yaml" not in policy:
            err("P14", "RESOURCE_POLICY.md",
                "does not point at research/model_routing.yaml")
        if "material_value" not in policy:
            err("P14", "RESOURCE_POLICY.md",
                "no material_value tracking; there would be no way to tell "
                "later whether stronger models bought anything (5.11)")

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
