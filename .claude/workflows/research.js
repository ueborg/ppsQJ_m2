export const meta = {
  name: 'research',
  description: 'Charter-compliant research investigation: independent investigators, then an uncontaminated red team. Read-only, local-only, stops at Human Gate A.',
  whenToUse: 'A ppsQJ_m2 research question that needs literature, theory and numerics investigated independently and then adversarially reviewed. The main session is the lead: it does orientation (Phase A) before calling this, and synthesis, the decision gate and experiment design (Phases C narrative, E, F) after it returns.',
  phases: [
    { title: 'Investigate', detail: 'up to three independent investigators; sonnet; no preferred conclusion given' },
    { title: 'Collaborate', detail: 'optional ONE bounded cross-examination round; atomic messages only; red team excluded' },
    { title: 'RedTeam', detail: 'all nine Stage-8 attacks per candidate, with independent web search and no lead summary' },
  ],
}

// ---------------------------------------------------------------------------
// /research v1 - ppsQJ_m2
//
// This script owns the two phases where INDEPENDENCE is the whole point, and
// nothing else. Everything requiring the lead's judgment stays with the main
// session, in the open, where the researcher can see it:
//
//   Phase A  orientation, TASK-ID, task-scoped Stage 0   -> main session (lead)
//   Phase B  investigators, in parallel                  -> THIS SCRIPT
//   Phase B2 ONE bounded collaboration round, OPTIONAL   -> THIS SCRIPT
//   Phase C  candidate construction                      -> main session (lead)
//   Phase D  red team                                    -> THIS SCRIPT
//   Phase E  decision gate (A-H, slop warnings, kill)    -> main session (lead)
//   Phase F  experiment design, if anything survived     -> main session (lead)
//            then STOP AT HUMAN GATE A
//
// At most four worker agents. No lead subagent (the main session is the lead)
// and no implementation agent in v1.
//
// RESOURCE POLICY (research/RESOURCE_POLICY.md) is enforced here as code:
//   - explicit model per role; nothing inherits the lead's model  (sec 5.4)
//   - workers get the lead's compact context, not the repository  (sec 5.1)
//   - workers load WORKER_CONTRACT.md, not the full Skill         (sec 5.2)
//   - one invocation, at most one retry, NO generic fallback      (sec 5.8)
//   - a missing project agent returns Infrastructure first        (sec 5.8)
//   - the lead may skip a role that cannot help                   (sec 5.5)
//   - report length is bounded, hard in regression mode           (sec 5.6)
//   - no simulation, no HPC, ever                                 (sec 2, 4)
//
// The one structural guarantee: the red team never sees a lead summary. It gets
// the question, the canonical evidence, the RAW investigator reports and the
// BARE candidate statements. Setting inputs_seen.lead_summary_seen: true is
// validator error R3 and fails the run, by design.
//
// args: {
//   taskId:      "TASK-YYYY-MM-DD-SHORTNAME"
//   taskDir:     "research/tasks/active/<TASK-ID>"
//   question:    the research question, one sentence
//   context:     canonical state gathered in Phase A - claim/evidence/observable/
//                dispute IDs and what each says. FACTS ONLY, NO PREFERRED ANSWER.
//   candidates:  [] on the first pass. On the Phase-D pass, the bare candidate
//                statements from Phase C.
//   stage:       "investigate" | "collaborate" | "redteam" | "both"
//   collab:      required for stage "collaborate". ONE bounded round:
//                { question, dependency, asks: [ {to, type, fact, ask,
//                  refs: [...] } ] }
//                Each ask is an ATOMIC fact/question the lead extracted. Peer
//                reports are NEVER forwarded.
//   mode:        "normal" | "historicalValidation"      (default "normal")
//   workers:     subset of ["literature","theory","numerics"]. Omit for all
//                three. Role economy is the lead's call (sec 5.5).
//   theoryEscalationReason: non-empty string -> theory runs on opus. Requires a
//                concrete reason; anything falsy keeps theory on sonnet.
// }
// ---------------------------------------------------------------------------

// args may arrive as an object or, depending on how the caller encoded it, as a
// JSON string. Accept both: silently returning "no question" because the caller
// stringified its arguments is a failure mode that wastes a whole run.
let a = args || {}
if (typeof a === 'string') {
  try { a = JSON.parse(a) } catch (e) { log(`args was a string and did not parse as JSON: ${e}`); a = {} }
}

const TASK_ID = a.taskId || 'TASK-UNSPECIFIED'
const TASK_DIR = a.taskDir || `research/tasks/active/${TASK_ID}`
const QUESTION = a.question || ''
const CONTEXT = a.context || '(no canonical state supplied - Phase A was skipped)'
const STAGE = a.stage || 'both'
const MODE = a.mode === 'historicalValidation' ? 'historicalValidation' : 'normal'
const REGRESSION = MODE === 'historicalValidation'

if (!QUESTION) {
  log('ERROR: no research question supplied. Pass args.question.')
  return { error: 'no question' }
}

// --- model routing (RESOURCE_POLICY sec 5.4) -------------------------------
// Aliases, not pinned IDs, so this survives model updates. NOTHING inherits.
// In the 2026-08-10 validation run every worker silently ran on opus because
// the definitions said `model: inherit`. That is the bug this table exists to
// prevent, so the resolved choice is logged on every run.
const THEORY_ESCALATED = !REGRESSION && !!a.theoryEscalationReason
const MODEL = {
  literature: 'sonnet',
  numerics: 'sonnet',
  theory: THEORY_ESCALATED ? 'opus' : 'sonnet',
  'red-team': REGRESSION ? 'sonnet' : 'opus',
}

const LENGTH_RULE = REGRESSION
  ? 'HARD LIMIT: your report must be UNDER 1000 WORDS in total, and normally well under. Findings, IDs, decisive evidence, contradictions, gaps. Nothing else.'
  : 'Keep the report concise and structured: findings, IDs, decisive evidence, contradictions, unresolved objections, falsifiers, recommended next check. No transcripts, no tool logs, no repository summaries, no exploratory scratch reasoning.'

const MODE_RULE = REGRESSION
  ? `
# MODE: historical / regression validation

This is a REGRESSION TEST of the research machinery on a case that is already
settled. It is not new research.

- Investigate ONLY what the assignment names. No broad field search, no
  open-ended mechanism development, no repository-wide archaeology.
- No new simulation of any kind.
- The evidential standard is UNCHANGED. Reduced scope never means a reduced
  standard: sources are still inspected or not cited, contradictions are still
  reported first, and the nine Stage 8 attacks are never reduced.
`
  : ''

// Every worker gets this and nothing more. No preferred answer, no hypothesis
// ranking, no hint about which side the lead favours.
const COMMON = `
# Research question

${QUESTION}

# Canonical state supplied by the lead (facts only)

${CONTEXT}
${MODE_RULE}
# Rules that bind you

Read \`.claude/skills/research/WORKER_CONTRACT.md\` FIRST. It is short and it
carries every invariant. Do NOT load \`.claude/skills/research/SKILL.md\` - that
is the lead's procedure, not yours. Consult
\`research/RESEARCH_CHARTER.md\` only for a specific ambiguity, never as
background reading. Resource rules: \`research/RESOURCE_POLICY.md\`.

- research/state/** is READ-ONLY. You may not modify it by any means.
- Only research/state/** is support. Manuscripts, theory/**, audit/**,
  research/history/**, prior chats and project memory are PROVENANCE ONLY.
- "No result" is a complete and valid answer. Nothing here rewards a finding.
- If you find a contradiction against any claim in scope, REPORT IT FIRST.
- Unresolved disputes stay unresolved. Report which way evidence leans; do not
  close a dispute.
- Label every substantive sentence [E] evidence, [I] inference, [C] conjecture,
  or [J] judgment.
- NO HPC or remote compute, EVER - no sbatch, srun, qsub, bsub, ssh, scp,
  rsync, at any stage of any workflow. Agents prepare HPC packages and stop at
  READY_FOR_HUMAN_SUBMISSION; the researcher submits manually. Reading and
  validating SLURM scripts is allowed.
- No new simulation campaigns. Local read-only (T0) analysis only, inside the
  budget in research/RESOURCE_POLICY.md section 3.
- Do NOT spawn a subagent or a workflow. You do not delegate.
- Do NOT reconstruct the repository. The lead resolved scope and handed you the
  IDs and paths you need. Work the assignment.
- STAY IN YOUR ROLE. literature = external sources and attribution; theory =
  derivations, assumptions, limiting cases (NOT dataset-wide numerical
  analysis); numerics = code, data, estimators, statistics, reproducibility.
  Off-scope discoveries go to ${TASK_DIR}/PARKING_LOT.md as ONE LINE and are not
  investigated further.
- EVIDENCE TIERS. canonical = research/state/**. task-verified = something YOU
  inspected during this task, recorded in ${TASK_DIR}/TASK_EVIDENCE.yaml with
  what you read, what it establishes, what it does NOT, and
  \`promotion_status: proposed\`. Task-verified is usable within this task,
  including by the red team. It NEVER becomes canonical here - only the human
  merge gate promotes. Write "proposed promotion", never "promoted".
- BEFORE contradicting a claim's recorded history, follow its DIRECT provenance:
  \`.venv/bin/python3 research/tools/resolve_provenance.py <CLAIM-ID>\`. One
  hop. A claim's provenance_note often names the document that corrected it.
- VOCABULARY. "task-verified" is not "canonical". "proposed promotion" is not
  "promoted". Read-only analysis is "T0 analysis compute", not "no compute";
  say "no new simulation or production compute".
- ${LENGTH_RULE}

You have NOT been told what the answer should be. If you find yourself inferring
a preferred conclusion from how the question was phrased, say so in
\`independence_note\` and disregard it.

Task directory (for reference only; the lead writes the artifacts):
  ${TASK_DIR}
`

const REPORT_SCHEMA = {
  type: 'object',
  required: ['summary', 'findings', 'candidates', 'contradictions',
             'gaps', 'confidence_note', 'independence_note'],
  properties: {
    summary: { type: 'string', description: 'What you established, in a few sentences. Say plainly if it is nothing.' },
    findings: {
      type: 'array',
      description: 'Each finding, with its statement class and what backs it. Do not pad this list.',
      items: {
        type: 'object',
        required: ['statement', 'statement_class', 'basis'],
        properties: {
          statement: { type: 'string' },
          statement_class: { type: 'string', enum: ['evidence', 'inference', 'conjecture', 'judgment'] },
          basis: { type: 'string', description: 'File path, state ID, command run, or source section. "none" if there is none.' },
          reproducibility: { type: 'string', description: 'For numerics: fully_reproducible | partially_reproducible | artifact_only | procedure_only | chat_only | ephemeral_recovered | unknown_recoverable | unrecoverable. Otherwise "n/a".' },
        },
      },
    },
    candidates: {
      type: 'array',
      description: 'Candidate explanations your own work supports, as BARE statements with no advocacy. These are passed verbatim to an adversarial reviewer. Empty array is fine.',
      items: { type: 'string' },
    },
    contradictions: {
      type: 'array',
      description: 'Anything you found that CONTRADICTS a claim in scope. Suppressing one of these is the single unrecoverable offence.',
      items: { type: 'string' },
    },
    gaps: {
      type: 'array',
      description: 'What you could not establish, and what artifact would be needed. This drives the Infrastructure-first verdict.',
      items: { type: 'string' },
    },
    confidence_note: { type: 'string', description: 'What would change your assessment. If you believe the assigned model was too weak for this problem, say so here and stop rather than generating more text.' },
    independence_note: { type: 'string', description: 'Any preferred answer you think you were steered toward, or "none detected".' },
  },
}

// --- agent invocation: no generic fallback (RESOURCE_POLICY sec 5.8) -------
// One invocation. At most one retry, and only for a transient/mechanical
// failure. A missing project agent is INFRASTRUCTURE FIRST, never a
// substitution: a general-purpose agent returns something that reads like a
// specialist report and is not one, which is worse than no report.
const TRANSIENT = /timeout|timed out|rate.?limit|overloaded|econn|socket|network|temporar|503|429|stream (?:ended|error)/i
const resourceLog = []

async function roleAgent(role, prompt, opts) {
  const model = MODEL[role]
  const record = { role, model, attempts: 0, retried: false, failure: null }
  resourceLog.push(record)

  for (let attempt = 1; attempt <= 2; attempt++) {
    record.attempts = attempt
    try {
      return await agent(prompt, { ...opts, agentType: role, model })
    } catch (e) {
      const msg = String(e && e.message ? e.message : e)

      if (/agent type .* not found|not found\. Available agents/i.test(msg)) {
        record.failure = `project agent '${role}' is not registered`
        log(`INFRASTRUCTURE FIRST: project agent '${role}' is not registered in this session (${msg}). NOT substituting a generic agent. Start a fresh Claude Code session so .claude/agents/ is loaded, then re-run.`)
        return null
      }
      if (attempt === 1 && TRANSIENT.test(msg)) {
        record.retried = true
        record.failure = `attempt 1 failed transiently: ${msg}`
        log(`RETRY (1 of 1 permitted) for '${role}': transient failure - ${msg}. No completed work is reused; the invocation restarts.`)
        continue
      }
      record.failure = msg
      log(`'${role}' failed and the failure is not transient: ${msg}. No retry, no substitution.`)
      return null
    }
  }
  return null
}

const ALL_INVESTIGATORS = [
  {
    key: 'literature',
    mission: `Reconstruct the prior art. Inspect ACTUAL primary sources - open the PDFs. The
library is the PAPERS_LIBRARY root in research/data_roots.local.yaml, plus
DATA_INTERNSHIP/Papers and DESKTOP_INTERNSHIP.

For each source in scope report its inspection_level, exactly what it supports
(the precise statement, not a topic summary), and what it does NOT establish.

Search is HYPOTHESIS-DRIVEN and starts from the registered sources bearing on
the claim. Do not browse broadly because web search exists; widen only if the
registered sources cannot resolve the task-specific question, and say why.

You may NOT certify novelty, and you may NOT infer content from a title or an
abstract. If a load-bearing source cannot be inspected, list it under \`gaps\`.`,
  },
  {
    key: 'theory',
    mission: `Attack this analytically. Derive from first principles where the derivation is
actually available. Enumerate the load-bearing assumptions and say where each
enters. Test limiting and degenerate cases. Attempt counterexamples before
completing any proof.

State explicitly whether your result is a PREDICTION or a POSTDICTION. A
derivation reproducing a number the project already believes is a postdiction
and cannot raise support - and that pattern has recurred here (three
invalidated derivations of the same answer).

Treat theory/** and research/history/legacy/ as HISTORICAL, never as current.`,
  },
  {
    key: 'numerics',
    mission: `Answer from data we already hold, if it can be answered at all. Check
results/boundary_aggregate.csv, results/ruche_pull/,
research/runs/_catalogue/, the pps_aggregates set, and
audit/2026-08-10/recovered_ephemeral/ before doing anything else.

Verify that stored observable definitions match the formulas the claims rely on
(research/state/observables/, pps_qj/parallel/worker_clone_pps.py). Report
window dependence over at least three windows for any exponent, whether any
crossing was L-extrapolated, and the reproducibility level of what you relied on.

READ-ONLY (T0) ANALYSIS ONLY. No new trajectory, cloning, sweep,
population-dynamics, scan, Monte Carlo or benchmark campaign - not even a small
one, and not a broad sweep split into small jobs. If new compute would
materially improve the decision, DESCRIBE the smallest useful calculation and
its cost (wall time, memory, threads, parameter points, output size, and the
decision it could change) and STOP; the researcher approves it at Gate A.

Anything you do run: one CPU-intensive job at a time, no nested
multiprocessing, pin BLAS/OpenMP threads explicitly, stay inside real free
memory. See research/resource_profile.local.yaml. Scratch goes to
${TASK_DIR}/scratch/ and nowhere else.`,
  },
]

// Role economy (RESOURCE_POLICY sec 5.5): the lead may run a subset.
const requested = Array.isArray(a.workers) && a.workers.length
  ? a.workers.filter(w => ALL_INVESTIGATORS.some(i => i.key === w))
  : ALL_INVESTIGATORS.map(i => i.key)
const INVESTIGATORS = ALL_INVESTIGATORS.filter(i => requested.includes(i.key))
const skipped = ALL_INVESTIGATORS.map(i => i.key).filter(k => !requested.includes(k))

// --- Phase B ---------------------------------------------------------------
let reports = {}
let infrastructureFirst = []

if (STAGE === 'investigate' || STAGE === 'both') {
  phase('Investigate')
  log(`Phase B on ${TASK_ID} [mode=${MODE}]: ${INVESTIGATORS.map(i => `${i.key}=${MODEL[i.key]}`).join(', ')}. No preferred conclusion supplied.`)
  if (THEORY_ESCALATED) log(`theory escalated to opus. Recorded reason: ${a.theoryEscalationReason}`)
  if (skipped.length) log(`Roles deliberately NOT spawned: ${skipped.join(', ')}. The lead judged they cannot materially improve this question. Their domains are UNINVESTIGATED, not clear.`)

  const results = await parallel(INVESTIGATORS.map(inv => () =>
    roleAgent(inv.key, `${COMMON}\n\n# Your mission\n\n${inv.mission}`, {
      label: `investigate:${inv.key}`,
      phase: 'Investigate',
      schema: REPORT_SCHEMA,
    }).then(r => ({ key: inv.key, report: r }))
  ))

  for (const r of results.filter(Boolean)) if (r.report) reports[r.key] = r.report

  const dead = INVESTIGATORS.map(i => i.key).filter(k => !reports[k])
  if (dead.length) {
    infrastructureFirst.push(`investigators returned nothing: ${dead.join(', ')}`)
    log(`NOT COVERED - no report from: ${dead.join(', ')}. Treat those domains as uninvestigated, not as clear.`)
  }

  const contradictions = Object.entries(reports)
    .flatMap(([k, r]) => (r.contradictions || []).map(c => `${k}: ${c}`))
  if (contradictions.length) log(`${contradictions.length} contradiction(s) reported by investigators.`)
}

// --- Phase B2: ONE bounded collaboration round -----------------------------
// Runs only when the lead explicitly asks for it, having already justified the
// round in the phase ledger. Each participant receives ONE atomic question and
// the references behind it - never a peer's report, never a reasoning trace.
// The red team is not here and never sees this.
let collaboration = null
if (STAGE === 'collaborate') {
  const c = a.collab || {}
  const asks = (c.asks || []).filter(x => x && x.to && x.fact && x.ask)

  if (!asks.length) {
    log('Phase B2 skipped: no atomic cross-role question supplied. Collaboration is OPTIONAL - record COLLABORATION_NOT_NEEDED and go to candidates.')
  } else if (asks.some(x => x.to === 'red-team')) {
    log('REFUSED: the red team is never an affirmative collaboration participant.')
    return { error: 'red-team cannot participate in collaboration' }
  } else {
    phase('Collaborate')
    log(`Phase B2: ONE round, ${asks.length} atomic exchange(s). Dependency: ${c.dependency || '(unrecorded)'}`)

    const answers = await parallel(asks.map((x, i) => () =>
      roleAgent(x.to, `
You are answering ONE cross-examination question in a bounded collaboration
round. Your first-pass report is already frozen and is NOT being reopened.
${MODE_RULE}
# The question

${x.fact}

**Requested action:** ${x.ask}

# References behind it

${(x.refs || []).join(', ') || '(none supplied)'}

# Rules

- Answer ONLY this question. Do not re-open your first pass, do not review
  anyone else's work, and do not expand scope.
- You have NOT been given the peer's report or reasoning, deliberately. If you
  need something specific from it, say what and stop - the lead will extract it.
- STAY IN ROLE. literature: what a source says, prior art, terminology, stated
  assumptions. theory: whether assumptions transfer, whether an implication
  follows, limits/symmetries/mechanism, derivations prompted by evidence.
  numerics: whether local data or code support the requested discriminating
  check, estimator/robustness questions, tiny T0 analyses within
  research/RESOURCE_POLICY.md.
- Cite canonical IDs, TV-*/EXT- task-verified ids, or a path. A result that
  traces to nothing is not evidence.
- If this CHANGES your first-pass conclusion, say so explicitly as
  FIRST_PASS / AFTER_CROSS_EXAMINATION / WHY_CHANGED. Your frozen report is NOT
  rewritten. Continuing to disagree is a valid outcome.
- Off-scope discoveries: one line, and say they belong in PARKING_LOT.md.
- Be brief. This is one answer, not a second report.
`, {
        label: `collab:${x.to}:${i + 1}`,
        phase: 'Collaborate',
        schema: {
          type: 'object',
          required: ['answer', 'result_class', 'traces_to', 'changes_first_pass'],
          properties: {
            answer: { type: 'string', description: 'The answer. Compact.' },
            result_class: { type: 'string', enum: ['confirmed', 'contradicted', 'narrowed', 'unresolved', 'no_effect'] },
            traces_to: { type: 'string', description: 'Canonical ID, TV-*/EXT- id, or path this rests on.' },
            changes_first_pass: { type: 'boolean' },
            first_pass: { type: 'string', description: 'Prior conclusion, if it changed.' },
            after_cross_examination: { type: 'string' },
            why_changed: { type: 'string' },
            still_disagrees: { type: 'string', description: 'Preserved disagreement, if any.' },
            parking_lot: { type: 'array', items: { type: 'string' }, description: 'Off-scope, one line each. Not investigated.' },
          },
        },
      }).then(r => ({ to: x.to, type: x.type, fact: x.fact, ask: x.ask,
                      refs: x.refs || [], response: r }))
    ))

    collaboration = { round: 1, question: c.question, dependency: c.dependency,
                      exchanges: answers.filter(Boolean) }
    const changed = collaboration.exchanges.filter(e => e.response && e.response.changes_first_pass)
    if (changed.length) log(`${changed.length} participant(s) revised a conclusion. Record it in COLLABORATION_LOG.yaml revisions - do NOT rewrite the frozen report.`)
    log('Collaboration round 1 complete. There is no second round.')
  }
}

// --- Phase D ---------------------------------------------------------------
// Candidates reach the reviewer verbatim: whatever the lead passed in, plus
// whatever the investigators proposed. They are deliberately NOT summarised.
let redteam = null
if (STAGE === 'redteam' || STAGE === 'both') {
  const fromWorkers = Object.values(reports).flatMap(r => r.candidates || [])
  const candidates = [...(a.candidates || []), ...fromWorkers]

  if (!candidates.length) {
    log('Phase D skipped: no candidate survived to review. That is a result, not a failure - the lead should read it as Stop or Infrastructure first.')
  } else {
    phase('RedTeam')
    log(`Phase D: red team (${MODEL['red-team']}) against ${candidates.length} bare candidate statement(s). No lead summary supplied.`)

    const rawReports = Object.entries(reports).map(([k, r]) =>
      `## Raw report - ${k}\n\n${JSON.stringify(r, null, 2)}`
    ).join('\n\n')

    redteam = await roleAgent('red-team', `
You are performing the Charter Stage 8 adversarial review for ${TASK_ID}.
${REGRESSION ? '\nMODE: historical / regression validation. Reduced SCOPE, unchanged STANDARD. All nine attacks are still mandatory. Keep the prose tight.\n' : ''}
# The original research question

${QUESTION}

# Canonical state

${CONTEXT}

# Candidate statements under attack

${candidates.map((c, i) => `C${i + 1}. ${c}`).join('\n')}

# Raw investigator reports (unedited)

${rawReports || '(no investigator reports - the review rests on canonical state alone; say so in verdict_reason)'}

# What you have NOT been given

You have deliberately NOT been given any lead summary, synthesis or
recommendation, any A-H assessment, any recommendation, the affirmative team's
search history, or the transcript of their collaboration round. The candidate
statements above are verbatim from the investigators.

"All three agents agreed" is NOT evidence and is not being offered to you. If
consensus language reaches you as though it were an argument, set
inputs_seen.lead_summary_seen: true.

YOU HAVE WebSearch AND WebFetch. Use them independently. Decide for yourself
what external checks are worth running - prior art the affirmative team missed,
a paper that already contains the supposed novelty, an incompatible external
result, a known methodological flaw. Primary sources only; a snippet is
discovery, not support. Register anything you open as an EXT-* entry in
${TASK_DIR}/TASK_EVIDENCE.yaml marked verified_by: red-team. If anything reaching you reads as a persuasive affirmative case
for a candidate, set inputs_seen.lead_summary_seen: true and say so - that is
validator error R3 and it correctly fails the run.

# Your job

Kill these candidates. Not evaluate them fairly - kill them.

1. Copy research/templates/REDTEAM_TEMPLATE.yaml to ${TASK_DIR}/REDTEAM.yaml.
   It is SCHEMA V2: one review PER CANDIDATE.
2. For EACH candidate above, write its own \`candidate_reviews.<Cn>\` block with
   ALL NINE mandated attacks A1-A9 (attempted, finding, evidence, severity,
   unresolved, effect_on_candidate), its own \`verdict\`
   (killed | survives | survives_scoped | unresolved) and its own \`reason\`.
   A missing attack is error R4. survives_scoped REQUIRES surviving_scope (R13).
3. Fill extensions X1-X7 as ADDITIONAL checks. They never substitute for A1-A9.
4. A fatal attack must kill THE CANDIDATE IT APPLIES TO (R9) and must NOT be
   used to erase an unrelated candidate that survived. Mixed outcomes are
   normal and expected.
5. Fill overall_task_assessment. surviving_candidates and killed_candidates
   must EXACTLY match the per-candidate verdicts (R10).
6. You MAY cross role boundaries - read code, re-run an estimator, open a
   paper - whatever kills a candidate. You are the only worker who may.
7. You MAY use task-verified evidence from ${TASK_DIR}/TASK_EVIDENCE.yaml even
   though it is not canonical; list the TV-* ids under inputs_seen.task_verified.
   Do not call it canonical or promoted.
8. Validate and fix until it passes:
   .venv/bin/python3 research/tools/validate_redteam.py ${TASK_DIR}/REDTEAM.yaml
9. Do not modify research/state/**. Write only inside ${TASK_DIR}.
10. Read-only analysis only. No simulation, no HPC, no remote execution.

A well-argued verdict of "killed" is the most valuable thing you can return.
`, {
      label: 'redteam',
      phase: 'RedTeam',
      schema: {
        type: 'object',
        required: ['candidate_verdicts', 'surviving_candidates', 'killed_candidates',
                   'validator_passed', 'report_path', 'recommendation_basis'],
        properties: {
          candidate_verdicts: {
            type: 'array',
            description: 'One entry per candidate. A fatal attack kills only its own candidate.',
            items: {
              type: 'object',
              required: ['candidate', 'verdict', 'reason'],
              properties: {
                candidate: { type: 'string', description: 'C1, C2, ...' },
                verdict: { type: 'string', enum: ['killed', 'survives', 'survives_scoped', 'unresolved'] },
                reason: { type: 'string' },
                surviving_scope: { type: 'string', description: 'Required when survives_scoped.' },
                fatal_attacks: { type: 'array', items: { type: 'string' } },
              },
            },
          },
          surviving_candidates: { type: 'array', items: { type: 'string' } },
          killed_candidates: { type: 'array', items: { type: 'string' } },
          recommendation_basis: { type: 'string', description: 'What the mixed outcome implies. Input to the lead gate, not the gate verdict.' },
          validator_passed: { type: 'boolean', description: 'Did validate_redteam.py exit 0? Report honestly.' },
          report_path: { type: 'string' },
          task_verified_used: { type: 'array', items: { type: 'string' }, description: 'TV-* ids relied on.' },
          lead_summary_contamination: { type: 'string', description: 'Did any affirmative summary reach you? "none" or what did.' },
        },
      },
    })

    if (!redteam) {
      infrastructureFirst.push('red team returned nothing; Stage 8 is incomplete')
      log('Stage 8 produced no report. The run is NOT reviewed and no candidate may be treated as having survived.')
    } else {
      if (redteam.validator_passed === false) {
        log('WARNING: the red-team report did NOT pass validate_redteam.py. Stage 8 is incomplete; the lead must not treat this run as reviewed.')
      }
      // Cross-check the reviewer's own summary against its per-candidate
      // verdicts. This is the v1 failure - a global kill reported next to a
      // surviving candidate - caught here as well as by validator rule R10.
      const cv = redteam.candidate_verdicts || []
      const derivedSurv = cv.filter(c => c.verdict !== 'killed').map(c => c.candidate).sort()
      const claimedSurv = [...(redteam.surviving_candidates || [])].sort()
      if (JSON.stringify(derivedSurv) !== JSON.stringify(claimedSurv)) {
        log(`INCONSISTENT red-team summary: per-candidate verdicts imply survivors [${derivedSurv}] but the summary claims [${claimedSurv}]. Trust the per-candidate verdicts and require a correction.`)
      }
      const scopedNoScope = cv.filter(c => c.verdict === 'survives_scoped' && !c.surviving_scope)
      if (scopedNoScope.length) log(`survives_scoped without a stated scope: ${scopedNoScope.map(c => c.candidate).join(', ')}`)
      if (derivedSurv.length) log(`Survivors after Stage 8: ${derivedSurv.join(', ')}. A fatal attack kills only its own candidate.`)
    }
  }
}

return {
  task_id: TASK_ID,
  task_dir: TASK_DIR,
  mode: MODE,
  question: QUESTION,
  reports,
  collaboration,
  redteam,
  // RESOURCE_POLICY sec 5.10: a small, non-authoritative usage record for the
  // lead to fold into the task's RESOURCE_USAGE.md. Never scientific evidence.
  resource_summary: {
    models_used: resourceLog.map(r => `${r.role}=${r.model}`),
    retries: resourceLog.filter(r => r.retried).map(r => `${r.role}: ${r.failure}`),
    failures: resourceLog.filter(r => r.failure && !r.retried).map(r => `${r.role}: ${r.failure}`),
    roles_skipped_by_lead: skipped,
    generic_fallback_used: false,
  },
  infrastructure_first: infrastructureFirst.length ? infrastructureFirst : null,
  // Deliberately absent: any recommendation. The decision gate (A-H, the twelve
  // slop warnings, the kill criterion) belongs to the lead in the open, and
  // Phase F stops at Human Gate A. This workflow never decides, never runs a
  // simulation, and never submits an HPC job.
  next: infrastructureFirst.length
    ? 'Infrastructure first: see infrastructure_first. Do not treat missing coverage as clear.'
    : 'Lead: Phase C candidates, Phase E decision gate, Phase F experiment design. STOP AT HUMAN GATE A.',
}
