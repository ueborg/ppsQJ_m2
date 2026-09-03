# RECOMMENDATION — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Labels `[E]` `[I]` `[C]` `[J]`.

## Verdict

**Pursue.**

`[J]` One of the four gate outcomes, not a ranked list. **`Pursue` does not mean
"it has been run"** — it means the campaign is designed, validated, costed and
runnable, and stops at Human Gate A. **No agent will submit it**, at any stage,
gate or approval level (`research/RESOURCE_POLICY.md` §4).

---

## What is ready

| | |
|---|---|
| immediate arms | **17**, all preflights passing |
| fresh tasks | 3 280 |
| populations reused, never recomputed | 240 (worth ~1 880 core-hours) |
| cost | **2 180 core-hours** (3 052 pessimistic) |
| longest single job | **31.4 h** (44.0 pessimistic), `D_L128_nc2048` |
| conditional arms | **7**, blocked behind three independent interlocks |
| validation | 25/26 automated checks pass; the one that does not is Stage 8 and is unrepaired |

Everything the human needs to type is in `RUCHE_RUNBOOK.md`. The per-arm gate
table is `HUMAN_SUBMISSION.md`.

## The three things worth knowing before deciding

### 1. The most decision-relevant result cost nothing to obtain

`[E]` From the **existing** measured variances, the matched `R` needed to certify
P2 at the `L = 128` `512 → 1024` step is **2 675** — about 13 000 core-hours for
one `lambda`. `[I]` **Absolute-level plateau certification at the frozen
`tau_I` is unreachable at `L = 128` at any affordable `R`.**

`[J]` This reframes the whole programme's question from *"how large must `N_c`
be"* to *"which tolerance can we afford to certify against"*. At `L = 64` the
absolute-level route is affordable and campaign A takes it. At `L = 128` it is
not, so the answer must come from the **crossing** tolerance — which is
candidate C3, which is untested, and which campaigns B and B2 exist to measure.

### 2. Two design decisions were killed before submission, at ~386 core-hours

`[E]` **`R = 24` in campaign A** was killed by its own power calculation: at that
`R` the top step could not have satisfied P2 whatever the data did. `[J]` An arm
that cannot pass its own frozen criterion is not a measurement. Raised to
`R = 48` (+166 core-hours); the residual under-power at the *lower* step is
stated in `CAMPAIGN_DESIGN.md`, not hidden.

`[E]` **The cheap version of campaign B2** was killed by the frozen crossing
protocol: on three shared `lambda`, both interior crossings are flagged
`ENDPOINT_INDUCED` **by construction** — the exact defect the immediately
preceding task existed to repair. Rebuilt on seven points (+220 core-hours);
`tools/design.py` records the reason at the changed line.

### 3. The inherited cost and memory models were both wrong, in the unsafe direction

`[E]` The runtime model extrapolated **flat** above `N_c = 256` at `L = 128`; the
measured rate turns back up and the old model is **30 % low** at `N_c = 1024`.
`[I]` Consequence: a conditional `L = 128`, `N_c = 4096` population is a **71.5 h**
job (100 h pessimistic) against `cpu_long`'s **168 h ceiling**, not the ~55 h the
old model implied — and `N_c = 8192` at `L = 128` is **not runnable at all**.

`[E]` The `--mem` model had never been checked against a running process. Direct
measurement shows it under-predicts at half the cells probed; `L = 64`,
`N_c = 2048` needs 1 694 MB against a predicted 1 202 MB, and that arm shipped
`--mem=2G`. `[J]` It never broke and it was closer to breaking than anyone knew.

## What I recommend, concretely

**Submit the immediate group.** `[J]` Queue `D_L128_nc2048` first — it is the
long pole at 31.4 h and everything else finishes inside ~9 h of compute.

**If the budget must be cut**, cut **C and D** (794 core-hours, 36 %), not B2.
`[J]` The reasoning is in `RESEARCH_MEMO.md` §9: `tau_I` is a worst-case
translation, so the absolute-level arms are certifying against a tolerance that
may be stricter than the science needs, while B, B2 and E test whether that is
so. `[E]` The counter-argument — and why D is still recommended — is that D is
the cheapest way to learn whether the absolute-level route at `L = 128` is dead,
and that is worth knowing before anything larger is committed.

**Do not cut E.** `[E]` 42 core-hours, 1.9 % of the campaign, and the only arm
whose **both** outcomes kill a mechanism.

**Please run one extra command.** `RUCHE_RUNBOOK.md` §7:
`sacct -j <D job id> --format=JobID,MaxRSS`. `[E]` It would be the first `MaxRSS`
measurement of this sampler on the cluster in existence, and it settles the
memory question for the whole programme.

## What must NOT be concluded from this campaign, whatever it returns

`[E]` No `lambda_c(zeta)`, no phase-boundary law, no exponent. `[E]` The
0.2182–0.2482 window is an **observed locator region** in `L <= 64` curves at
`N_c = 1024`, at or below the programme's own corpus floor — **not** a critical
window. `[E]` No general `N_c_req(L, zeta, lambda)` rule: the predecessor
established there is no controlled analytic one and this task supplies no
empirical replacement.

`[E]` **The frozen theory result must be preserved exactly as stated**: the
standard useful uniform-mixing Feynman–Kac bounds do not directly transfer to
the production mutation kernel, because the no-click branch is deterministic.
That is the failure of a **proof route**. It is **not** "1/`N_c` convergence is
impossible", and nothing in this package upgrades it.

## The gap this task could not close

`[E]` **Charter Stage 8 is not satisfied.** No independent investigator and no
independent red team ran at any point; every role was executed inline by the
lead. `research/tools/validate_redteam.py` **refuses** the report under rule R3,
and `lead_summary_seen` was **not** set to false to make the check green.

`[J]` The self-red-team did real work — it killed the two design decisions above
— but a self-red-team is a checklist, not a check. **Every "survives" verdict in
`REDTEAM.yaml` should be treated as unreviewed.** `INDEPENDENCE_LEDGER.yaml`
names the three passes that would repair it, the most valuable being an
independent numerics pass that tries to *locate* the four-rung `L = 96` ladder
behind the predecessor's `chi2 = 10.54`.

`[E]` Relatedly: **no external prior-art search was performed anywhere in this
task**, so external novelty for C2 and C3 is `UNRESOLVED`, not favourable.

## Files a human may want to update AFTER adjudication

`[J]` Listed, not touched. `research/state/**` is byte-identical to task open.

1. `[E]` **`research/HANDOFF.md`** — stamped `last_reviewed: 2026-08-10`; §3 still
   says `research/tasks/active/` is empty apart from `TASK_TEMPLATE/`. There are
   now 43 entries. Human-owned. *(Flagged by the two preceding tasks and still
   not done.)*
2. `[E]` **`docs/PRODUCTION_ALGORITHM.md`** — worth recording that the deployed
   `--mem` model under-predicts peak RSS and that `128 + 2 N_c per_clone` is a
   floor, not an estimate.
3. `[E]` **The `L = 96` `1/N` provenance item.** Whoever owns the
   `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` result should name the four-rung
   `L = 96` ladder behind `chi2 = 10.54`, or the statement should be narrowed to
   `L = 128`.
4. `[E]` **The engine.** `validate_task.py` implements no independence check —
   `TASK_EVIDENCE.yaml` and `INDEPENDENCE_LEDGER.yaml` can both be left empty
   with nothing failing. Recorded by the two preceding tasks and recurring here:
   **a pattern, not an incident.** And `task_phase.py close first_pass_frozen`
   refuses a run in which the lead does everything, so such a run cannot record
   that fact without filing a "first pass" for a role that was not independently
   executed.
5. `[E]` **Nothing in `research/state/claims/`** should change on this task's
   output. Nothing here is canonical and nothing is proposed for promotion beyond
   the eight `TV-*` items in `TASK_EVIDENCE.yaml`, all marked
   `promotion_status: proposed`.

## Terminal state

`[E]` **Human Gate A.** No HPC submission, no scheduler call, no remote launch,
no new production simulation, no write to `research/state/**`, no edit to any
predecessor task directory, no manuscript touched. `[E]` The only compute run was
local, read-only T0 analysis: rebuilding ladders, fitting cost models, measuring
peak RSS, and re-executing two small predecessor populations to prove the sampler
is unchanged.
