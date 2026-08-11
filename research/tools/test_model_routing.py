#!/usr/bin/env python3
"""test_model_routing.py - regression tests for the ADAPTIVE model routing pass.

Read-only. Launches NO model calls, no agents and no simulation. Every test is
static (config/YAML/markdown cross-reads) or executes the routing logic of
.claude/workflows/research.js in a local JS engine with every agent call
STUBBED. Nothing here costs a token.

The pass being tested replaced

    "use the cheapest model unless a stronger model is clearly necessary"

with

    "use the model with the highest expected scientific value for the
     decision being made"

and the failure mode it must not reintroduce is the mirror image of the old
one: the 2026-08-10 run put EVERY worker on the strongest model and bought
nothing. So the tests check both directions - that difficult work can reach
Tier 2/3 without a staged failure first, AND that mechanical work stays on
Tier 1 even in a `deep` posture.

Covers the fourteen required behaviours:
   1  historical/regression mode stays Sonnet-heavy
   2  routine literature extraction routes to Sonnet
   3  difficult theory starts on Opus with no prior Sonnet failure
   4  deep first-principles theory routes to `best`
   5  mechanical numerical benchmarking never routes to `best`
   6  subtle estimator reasoning may route to Opus
   7  ordinary red team routes to Opus
   8  a high-stakes unresolved red-team attack may route to `best`
   9  `deep` posture does NOT force every worker to `best`
  10  a `normal` task may run sonnet + opus + best simultaneously
  11  escalation records carry decision_at_stake and material_value
  12  first-pass independence survives heterogeneous tiers
  13  unavailable/disallowed Tier-3 routing degrades instead of crashing
  14  the routing table is consistent across all four places it appears

Usage:  .venv/bin/python3 research/tools/test_model_routing.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKFLOW = os.path.join(ROOT, ".claude", "workflows", "research.js")
ROUTING = os.path.join(ROOT, "research", "model_routing.yaml")
POLICY = os.path.join(ROOT, "research", "RESOURCE_POLICY.md")
AGENTS = os.path.join(ROOT, ".claude", "agents")

failures: list[str] = []
skips: list[str] = []
passed = 0


def check(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")


def skip(name, why):
    skips.append(f"{name}: {why}")
    print(f"  skip {name}: {why}")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# JS harness. The workflow script is not a valid ES module on its own - it uses
# top-level `return`, which only works because the runtime wraps it in an async
# function. We reproduce that wrapper exactly, stub every host hook, and read
# back the routing decisions. No agent is ever really invoked.
# ---------------------------------------------------------------------------
JS_ENGINES = [
    shutil.which("node"),
    shutil.which("deno"),
    shutil.which("bun"),
    "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc",
]

HARNESS = r"""
// --- host stubs -----------------------------------------------------------
if (typeof print === 'undefined') { var print = (s) => console.log(s) }
const __log = []
const __calls = []
// Model aliases this fake runtime will REFUSE, to exercise the fallback chain.
const __reject = %(reject)s
function log(m) { __log.push(String(m)) }
function phase(t) { __log.push('PHASE:' + t) }
function parallel(thunks) { return Promise.all(thunks.map(t => t())) }
function pipeline() { throw new Error('pipeline not stubbed') }
const budget = { total: null, spent: () => 0, remaining: () => Infinity }
async function agent(prompt, opts) {
  const o = opts || {}
  if (__reject.includes(o.model)) {
    // Shape mirrors a real runtime refusing an alias.
    throw new Error(`invalid model: '${o.model}' is not allowed by availableModels`)
  }
  __calls.push({ role: o.agentType, model: o.model, effort: o.effort || null,
                 label: o.label || null, prompt_len: String(prompt).length,
                 prompt: String(prompt) })
  // A minimal report that satisfies the schemas the script reads back.
  if (o.agentType === 'red-team') {
    return { candidate_verdicts: [{ candidate: 'C1', verdict: 'survives', reason: 'stub' }],
             surviving_candidates: ['C1'], killed_candidates: [],
             validator_passed: true, report_path: 'stub', recommendation_basis: 'stub' }
  }
  return { summary: 'stub', findings: [], candidates: ['stub candidate'],
           contradictions: [], gaps: [], confidence_note: 'stub',
           independence_note: 'none detected' }
}
const args = %(args)s

// --- the workflow body, wrapped exactly as the runtime wraps it ------------
;(async () => {
  let __result = null, __error = null
  try {
    __result = await (async () => {
%(body)s
    })()
  } catch (e) { __error = String(e && e.stack ? e.stack : e) }
  print('---HARNESS---' + JSON.stringify({ result: __result, error: __error,
                                           log: __log, calls: __calls }))
})()
"""


def workflow_body():
    """Strip `export const meta = {...}` - the runtime reads it separately."""
    src = read(WORKFLOW)
    m = re.search(r"^export const meta = \{", src, re.MULTILINE)
    if not m:
        raise AssertionError("research.js no longer starts with `export const meta`")
    # First line after the meta object that is exactly `}` closes it.
    rest = src[m.end():]
    close = re.search(r"^\}\s*$", rest, re.MULTILINE)
    if not close:
        raise AssertionError("could not find the end of the meta object")
    return rest[close.end():]


_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = next((e for e in JS_ENGINES if e and os.path.exists(e)), False)
    return _ENGINE


def run_workflow(args_obj, reject=()):
    """Execute research.js with stubs. Returns the parsed harness payload."""
    eng = engine()
    if not eng:
        return None
    js = HARNESS % {
        "args": json.dumps(args_obj),
        "reject": json.dumps(list(reject)),
        "body": workflow_body(),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(js)
        path = fh.name
    try:
        cmd = [eng, path]
        if os.path.basename(eng) == "deno":
            cmd = [eng, "run", "--quiet", path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = r.stdout
        if "---HARNESS---" not in out:
            raise AssertionError(
                f"harness produced no payload (exit {r.returncode}).\n"
                f"stdout: {out[-2000:]}\nstderr: {r.stderr[-2000:]}")
        payload = json.loads(out.split("---HARNESS---", 1)[1].strip().splitlines()[0])
        if payload["error"]:
            raise AssertionError(f"workflow threw: {payload['error'][:2000]}")
        return payload
    finally:
        os.unlink(path)


BASE = {
    "taskId": "TASK-TEST-ROUTING",
    "taskDir": "research/tasks/active/TASK-TEST-ROUTING",
    "question": "Does the routing engine assign tiers correctly?",
    "context": "(test context, facts only)",
}


def models_of(payload):
    """role -> alias actually passed to agent()."""
    return {c["role"]: c["model"] for c in payload["calls"]}


def summary(payload):
    return payload["result"]["resource_summary"]


# ===========================================================================
# Behavioural tests (JS harness)
# ===========================================================================
def behavioural():
    if not engine():
        skip("behavioural routing tests",
             "no local JS engine (node/deno/bun/jsc) found; static tests still ran")
        return

    # -- 1  regression mode stays Sonnet-heavy ------------------------------
    p = run_workflow({**BASE, "mode": "historicalValidation", "stage": "both",
                      "candidates": ["a regression candidate"]})
    m = models_of(p)
    check("1 regression mode is Sonnet-heavy",
          m and all(v == "sonnet" for v in m.values()),
          f"got {m}")
    check("1b regression posture is economical",
          summary(p)["posture"] == "economical", summary(p)["posture"])
    # ...and Tier 3 is refused there even when asked for with a full record.
    p = run_workflow({**BASE, "mode": "historicalValidation",
                      "roleTier": {"theory": "tier_3"},
                      "escalations": [{"role": "theory", "from": "sonnet", "to": "best",
                                       "question": "q", "decision_at_stake": "d"}]})
    check("1c regression refuses Tier 3 even with a record",
          models_of(p).get("theory") == "sonnet" and
          any("tier_3 refused" in r for r in summary(p)["escalations_refused"]),
          f"{models_of(p)} / {summary(p)['escalations_refused']}")

    # -- 2  routine literature extraction -> sonnet -------------------------
    p = run_workflow({**BASE, "workers": ["literature"]})
    check("2 routine literature extraction routes to sonnet",
          models_of(p).get("literature") == "sonnet", str(models_of(p)))

    # -- 3  difficult theory starts on Opus, no failed Sonnet pass first ----
    p = run_workflow({**BASE, "workers": ["theory"]})
    calls = p["calls"]
    check("3 difficult theory starts on opus in a normal posture",
          models_of(p).get("theory") == "opus", str(models_of(p)))
    check("3b theory reached opus WITHOUT a prior sonnet attempt",
          len([c for c in calls if c["role"] == "theory"]) == 1 and
          calls[0]["model"] == "opus",
          f"calls: {[(c['role'], c['model']) for c in calls]}")

    # -- 4  deep first-principles theory -> best ----------------------------
    p = run_workflow({**BASE, "posture": "deep", "workers": ["theory"]})
    check("4 deep posture routes theory to best",
          models_of(p).get("theory") == "best", str(models_of(p)))

    # -- 5  mechanical numerical benchmarking never -> best -----------------
    for posture in ("economical", "normal", "deep"):
        p = run_workflow({**BASE, "posture": posture, "workers": ["numerics"]})
        check(f"5 numerics stays off Tier 3 in posture={posture}",
              models_of(p).get("numerics") == "sonnet", str(models_of(p)))
    # Even an explicit Tier-3 request without a record is refused.
    p = run_workflow({**BASE, "posture": "deep", "workers": ["numerics"],
                      "roleTier": {"numerics": "tier_3"}})
    check("5b unrecorded Tier-3 request for numerics is refused",
          models_of(p).get("numerics") == "sonnet" and
          summary(p)["escalations_refused"], str(summary(p)))

    # -- 6  subtle estimator reasoning may route to Opus --------------------
    p = run_workflow({**BASE, "workers": ["numerics"],
                      "roleTier": {"numerics": "tier_2"},
                      "escalations": [{"role": "numerics", "from": "sonnet", "to": "opus",
                                       "question": "is the crossing estimator biased at small L",
                                       "decision_at_stake": "whether CB-AMP-001 survives"}]})
    check("6 recorded estimator escalation reaches opus",
          models_of(p).get("numerics") == "opus", str(models_of(p)))

    # -- 7  ordinary red team -> opus ---------------------------------------
    p = run_workflow({**BASE, "stage": "redteam", "candidates": ["C stub"]})
    check("7 ordinary red team routes to opus",
          models_of(p).get("red-team") == "opus", str(models_of(p)))

    # -- 8  high-stakes unresolved red-team attack -> best ------------------
    p = run_workflow({**BASE, "stage": "redteam", "candidates": ["C stub"],
                      "roleTier": {"red-team": "tier_3"},
                      "escalations": [{"role": "red-team", "from": "opus", "to": "best",
                                       "question": "opus pass left the exactness claim unresolved",
                                       "decision_at_stake": "redirect of a production campaign"}]})
    check("8 recorded high-stakes red-team escalation reaches best",
          models_of(p).get("red-team") == "best", str(models_of(p)))

    # -- 9  deep posture does NOT make everything best ----------------------
    p = run_workflow({**BASE, "posture": "deep", "stage": "both",
                      "candidates": ["C stub"]})
    m = models_of(p)
    check("9 deep posture does not force every worker to best",
          m.get("literature") == "sonnet" and m.get("numerics") == "sonnet"
          and m.get("red-team") == "opus" and m.get("theory") == "best",
          f"got {m}")
    check("9b deep posture still uses at most one Tier-3 worker by default",
          sum(1 for v in m.values() if v == "best") == 1, f"got {m}")

    # -- 10  a normal task may mix all three tiers at once -------------------
    p = run_workflow({**BASE, "stage": "both", "candidates": ["C stub"],
                      "roleTier": {"theory": "tier_3"},
                      "escalations": [{"role": "theory", "from": "opus", "to": "best",
                                       "question": "derive a controlled small-zeta boundary theory",
                                       "decision_at_stake": "viability of candidate mechanism H2"}]})
    m = models_of(p)
    check("10 one normal run carries sonnet + opus + best simultaneously",
          {"sonnet", "opus", "best"} <= set(m.values()), f"got {m}")

    # -- 11  escalation record carries decision_at_stake and material_value --
    s = summary(p)
    esc = s["escalations"]
    check("11 escalation is recorded with all five fields",
          len(esc) == 1 and all(k in esc[0] for k in
                                ("role", "from", "to", "question", "decision_at_stake")),
          str(esc))
    check("11b decision_at_stake is preserved verbatim",
          esc and esc[0]["decision_at_stake"] == "viability of candidate mechanism H2",
          str(esc))
    check("11c material_value starts 'pending' and is flagged for the lead",
          esc and esc[0]["material_value"] == "pending"
          and "material_value" in s["material_value_note"], str(esc))
    check("11d tier counts are reported for later scoring",
          s["tier_counts"]["tier_3_best"] == 1 and s["tier_counts"]["tier_2_opus"] >= 1,
          str(s["tier_counts"]))
    check("11e the escalation is logged where the researcher can see it",
          any("MODEL_ESCALATION applied" in line and "decision_at_stake" in line
              for line in p["log"]), str(p["log"])[:400])
    # An escalation with no decision_at_stake is not a record.
    p2 = run_workflow({**BASE, "workers": ["theory"],
                       "roleTier": {"theory": "tier_3"},
                       "escalations": [{"role": "theory", "from": "opus", "to": "best",
                                        "question": "q", "decision_at_stake": "  "}]})
    check("11f an escalation missing decision_at_stake is refused",
          models_of(p2).get("theory") == "opus"
          and summary(p2)["escalations_refused"], str(summary(p2)))

    # -- 12  first-pass independence across heterogeneous tiers -------------
    prompts = {c["role"]: c["prompt"] for c in p["calls"]}
    inv = [r for r in ("literature", "theory", "numerics") if r in prompts]
    check("12 investigators are dispatched in one parallel first pass",
          len(inv) == 3, str(inv))
    check("12b no investigator prompt contains another role's report",
          all("Raw report -" not in prompts[r] for r in inv), "a peer report leaked")
    check("12c no investigator is told which tier its peers run on",
          all(not re.search(r"\b(sonnet|opus|best|fable)\b", prompts[r], re.I)
              for r in inv),
          "a model/tier name reached an investigator prompt")
    rt = prompts.get("red-team", "")
    check("12d the red team still gets raw reports and no lead summary",
          "Raw report -" in rt and "NOT been given any lead summary" in rt,
          "contamination barrier changed")

    # -- 13  unavailable Tier 3 degrades, never crashes ---------------------
    # (a) the environment declares what it allows, and `best` is not in it
    p = run_workflow({**BASE, "posture": "deep", "workers": ["theory"],
                      "availableModels": ["sonnet", "opus"]})
    check("13 declared-unavailable best degrades to opus",
          models_of(p).get("theory") == "opus", str(models_of(p)))
    # (b) the runtime rejects the alias at call time
    p = run_workflow({**BASE, "posture": "deep", "workers": ["theory"]},
                     reject=["best"])
    check("13b a runtime alias rejection steps down the chain, run survives",
          models_of(p).get("theory") in ("fable", "opus")
          and p["result"]["reports"].get("theory"), str(models_of(p)))
    check("13c the substitution is reported, not hidden",
          summary(p)["model_substitutions"], str(summary(p)))
    # (c) the whole chain is rejected - still no crash
    p = run_workflow({**BASE, "posture": "deep", "workers": ["theory"]},
                     reject=["best", "fable", "opus"])
    check("13d exhausted chain fails the ROLE, not the workflow",
          p["result"] is not None and p["result"]["infrastructure_first"],
          str(p["result"])[:300] if p["result"] else "workflow returned nothing")
    # (d) an unknown posture string falls back to normal rather than throwing
    p = run_workflow({**BASE, "posture": "turbo", "workers": ["theory"]})
    check("13e an unrecognised posture degrades to normal",
          summary(p)["posture"] == "normal", str(summary(p)["posture"]))

    # -- effort is an independent axis (policy 5.4f) ------------------------
    p = run_workflow({**BASE, "posture": "deep", "workers": ["theory"]})
    check("effort: tier_3 is NOT automatically paired with max effort",
          all(c["effort"] is None for c in p["calls"]),
          str([(c["role"], c["effort"]) for c in p["calls"]]))
    p = run_workflow({**BASE, "workers": ["theory"], "effort": {"theory": "high"}})
    check("effort: an explicit effort override is passed through",
          any(c["effort"] == "high" for c in p["calls"]),
          str([(c["role"], c["effort"]) for c in p["calls"]]))

    # -- protections preserved (policy 16) ----------------------------------
    p = run_workflow({**BASE, "stage": "collaborate",
                      "collab": {"question": "q", "dependency": "d",
                                 "asks": [{"to": "red-team", "type": "t",
                                           "fact": "f", "ask": "a"}]}})
    check("preserved: the red team still cannot join collaboration",
          p["result"].get("error") == "red-team cannot participate in collaboration",
          str(p["result"])[:200])
    p = run_workflow({**BASE, "workers": ["theory"]})
    body = p["calls"][0]["prompt"]
    for frag, why in (
        ("research/state/** is READ-ONLY", "canonical-state protection"),
        ("NO HPC or remote compute, EVER", "human-only HPC submission"),
        ("No new simulation campaigns", "local-compute control"),
        ("Do NOT spawn a subagent", "no recursive delegation"),
        ("EVIDENCE TIERS", "task-local evidence tiers"),
        ("WORKER_CONTRACT.md", "compact worker contract"),
    ):
        check(f"preserved: {why}", frag in body, f"missing {frag!r} from the worker prompt")


# ===========================================================================
# Static tests - these run everywhere, with or without a JS engine
# ===========================================================================
def static():
    js = read(WORKFLOW)
    yml = read(ROUTING)
    policy = read(POLICY)

    # -- 14  the table is consistent everywhere it appears ------------------
    # (a) tier -> alias, in the workflow and in the YAML
    check("14 workflow tier aliases are sonnet/opus/best",
          re.search(r"TIER_ALIAS\s*=\s*\{\s*tier_1:\s*'sonnet',\s*tier_2:\s*'opus',"
                    r"\s*tier_3:\s*'best'\s*\}", js) is not None,
          "TIER_ALIAS in research.js does not match the policy")
    for tier, alias in (("tier_1", "sonnet"), ("tier_2", "opus"), ("tier_3", "best")):
        check(f"14b {tier} -> {alias} in model_routing.yaml",
              re.search(rf"{tier}:\s*\n\s*alias:\s*{alias}\b", yml) is not None,
              "alias missing or different in the YAML")

    # (b) posture defaults agree between research.js and the YAML
    expected = {
        "economical": {"literature": "tier_1", "theory": "tier_1",
                       "numerics": "tier_1", "red-team": "tier_2"},
        "normal": {"literature": "tier_1", "theory": "tier_2",
                   "numerics": "tier_1", "red-team": "tier_2"},
        "deep": {"literature": "tier_1", "theory": "tier_3",
                 "numerics": "tier_1", "red-team": "tier_2"},
    }
    for posture, roles in expected.items():
        row = re.search(rf"{posture}:\s*\{{([^}}]*)\}}", js)
        check(f"14c research.js has a {posture} posture row", row is not None, "missing")
        if row:
            for role, tier in roles.items():
                check(f"14d research.js {posture}/{role} = {tier}",
                      re.search(rf"'?{role}'?:\s*'{tier}'", row.group(1)) is not None,
                      f"row was: {row.group(1).strip()}")
        # ...and the same cell in the YAML
        blk = re.search(rf"\n  {re.escape('')}(\w[\w-]*):\n    default_tier:\n"
                        rf"      economical: (\w+)\n      normal: (\w+)\n      deep: (\w+)",
                        yml)
        check(f"14e model_routing.yaml declares default_tier blocks", blk is not None,
              "no role default_tier block parsed")
    for role, tiers in (("literature", ("tier_1", "tier_1", "tier_1")),
                        ("theory", ("tier_1", "tier_2", "tier_3")),
                        ("numerics", ("tier_1", "tier_1", "tier_1")),
                        ("red-team", ("tier_2", "tier_2", "tier_2"))):
        pat = (rf"\n  {role}:\n    default_tier:\n      economical: {tiers[0]}\n"
               rf"      normal: {tiers[1]}\n      deep: {tiers[2]}\n")
        check(f"14f YAML {role} defaults = {tiers}",
              re.search(pat, yml) is not None, "YAML role defaults differ")

    # (c) the policy prose carries the same role defaults
    for role, cells in (("literature", ("tier 1", "tier 1", "tier 1")),
                        ("theory", ("tier 1", "tier 2", "tier 3")),
                        ("numerics", ("tier 1", "tier 1", "tier 1")),
                        ("red-team", ("tier 2", "tier 2", "tier 2"))):
        row = re.search(rf"^\|\s*`{role}`\s*\|(.+)$", policy, re.MULTILINE)
        check(f"14g RESOURCE_POLICY has a {role} routing row", row is not None, "missing")
        if row:
            cellsfound = [c.strip().lower() for c in row.group(1).split("|")]
            ok = all(any(want in c for c in cellsfound) for want in set(cells))
            check(f"14h RESOURCE_POLICY {role} row matches {cells}", ok,
                  f"row: {row.group(1).strip()}")

    # (d) agent frontmatter matches the `normal` posture defaults
    for role, alias in (("literature", "sonnet"), ("theory", "opus"),
                        ("numerics", "sonnet"), ("red-team", "opus")):
        fm = read(os.path.join(AGENTS, f"{role}.md")).split("---")[1]
        m = re.search(r"^model:\s*(\S+)", fm, re.MULTILINE)
        check(f"14i .claude/agents/{role}.md declares model: {alias}",
              m is not None and m.group(1) == alias,
              f"got {m.group(1) if m else 'none'}")
        check(f"14j .claude/agents/{role}.md never inherits",
              m is not None and m.group(1) != "inherit", "model: inherit")

    # -- aliases are the ones this installation actually supports -----------
    supported = {"sonnet", "opus", "haiku", "fable", "best"}
    for alias in re.findall(r"tier_\d:\s*\['([^\]]+)\]", js):
        pass
    chains = re.search(r"const FALLBACK = \{(.*?)\n\}", js, re.DOTALL)
    check("aliases: research.js declares fallback chains", chains is not None, "missing")
    if chains:
        used = set(re.findall(r"'([a-z0-9\[\]]+)'", chains.group(1)))
        check("aliases: every routed alias is a supported Claude Code alias",
              used <= supported, f"unsupported: {sorted(used - supported)}")
        check("aliases: tier_3 prefers `best` over a pinned `fable`",
              re.search(r"tier_3:\s*\['best'", chains.group(1)) is not None,
              "tier_3 chain does not start at `best`")
    check("aliases: no pinned model version IDs in routing",
          not re.search(r"claude-(?:opus|sonnet|fable|haiku)-[0-9]", js),
          "a pinned model ID appeared in research.js")

    # -- the superseded principle must not survive anywhere -----------------
    OLD = re.compile(r"cheapest model unless|only escalate if the cheaper|"
                     r"only\s+with a concrete recorded reason", re.I)
    for rel in ("research/RESOURCE_POLICY.md", ".claude/agents/theory.md",
                ".claude/agents/numerics.md", ".claude/agents/literature.md",
                ".claude/agents/red-team.md", ".claude/skills/research/SKILL.md",
                ".claude/workflows/research.js"):
        body = read(os.path.join(ROOT, rel))
        hit = OLD.search(body)
        # The policy is allowed to NAME the old rule in order to withdraw it.
        withdrawn = hit and re.search(r"withdraw|replaces|retired|superseded|NOT\s+\"use the cheapest",
                                      body[max(0, hit.start() - 400):hit.start() + 400], re.I)
        check(f"no-failure-first: {rel} does not require a failed cheap pass",
              hit is None or bool(withdrawn),
              f"found {hit.group(0)!r} without a withdrawal note" if hit else "")

    # -- escalation vocabulary is defined in exactly one place --------------
    for v in ("changed_conclusion", "new_derivation", "caught_error",
              "confirmed_existing", "no_material_gain"):
        check(f"material_value '{v}' is defined in model_routing.yaml", v in yml, "missing")
        check(f"material_value '{v}' is documented in RESOURCE_POLICY", v in policy, "missing")

    # -- protections that a model change must not touch (policy 16) ---------
    for frag, why in (
        ("never submit", "human-only HPC submission"),
        ("READY_FOR_HUMAN_SUBMISSION", "HPC terminal state"),
        ("local-only", "local-compute control"),
        ("Human Gate A", "the human gate"),
    ):
        check(f"preserved in policy: {why}", frag in policy, f"missing {frag!r}")
    tmpl = read(os.path.join(ROOT, "research/tasks/TASK_TEMPLATE/RESOURCE_USAGE.md"))
    check("RESOURCE_USAGE template records tier counts",
          "tier" in tmpl.lower() and "material_value" in tmpl,
          "the template cannot record whether a strong model bought anything")


def main():
    print("== static routing checks ==")
    static()
    print("== behavioural routing checks (stubbed; no model calls) ==")
    behavioural()

    print()
    for s in skips:
        print(f"SKIP {s}")
    for f in failures:
        print(f"FAIL {f}")
    print(f"\n{passed} passed, {len(failures)} failed, {len(skips)} skipped")
    if engine():
        print(f"JS engine used: {engine()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
