# POST_FREEZE_EVENTS — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Everything that changed a frozen artifact after its stage closed, why, and who
authorised it. Recorded loudly and attributed, because silent is the failure
mode the phase lock exists to prevent. `TASK_MANIFEST.yaml` carries the old and
new hashes for each.

`[E]` **Three amendments. None changed a criterion, a candidate, a kill
condition, an arm, a tolerance or a scientific number.** Two were forced by
`validate_task.py` rejecting a SCHEMA and made the file weaker or more explicit,
never stronger; the third corrected a single miscounted validation total.

---

## AMEND-1 — `CANDIDATES.md`, restructured into the mandated schema

| | |
|---|---|
| frozen at | `stage_3_candidates`, 2026-09-03T19:38:48Z |
| was | `sha256:c2de79d28527…` |
| now | `sha256:13b0d5effcea…` |
| trigger | `validate_task.py` **T7**: "0 candidates; charter Stage 3 says 3-8" and "required field 1 of 11 not found for any candidate" |
| authorised by | lead, on `validate_task.py` T7 |

`[E]` The original used prose headings (`## C1 — a high-N_c plateau is
reachable`) rather than the mandated `## Candidate C1` plus the eleven numbered
fields. The validator was right to reject it: a Stage-3 file that does not carry
the eleven fields has not asked the eleven questions.

`[E]` **What changed**: the presentation, and the addition of the fields the
original prose did not separately answer — principally fields 4–7 (the four
objections) and field 11 (the stronger version). `[E]` **What did not change**:
the six candidates, the killed C0, every kill criterion, every arm, every
tolerance and every number.

`[J]` This should not have needed an amendment. The Skill says routine work must
never need one, and the correct sequence was to write the file in the mandated
schema first. The validator caught it, which is what validators are for; the
amendment is the honest repair rather than a quiet rewrite.

## AMEND-2 — `NOVELTY_GATE.md`, qualification added

| | |
|---|---|
| frozen at | `stage_3_candidates`, 2026-09-03T19:38:48Z |
| was | `sha256:0cf68e56b576…` |
| now | `sha256:2494193d3fcd…` |
| trigger | `validate_task.py` **G4**, twice: a candidate is classified `no predecessor found` while the gate records no external prior-art search, and `no predecessor found` was not qualified as meaning "none found under the searches actually performed" |
| authorised by | lead, on `validate_task.py` G4 |

`[E]` **What changed**: a new section stating that (a) the only searches
performed were `find_predecessors.py` over `research/state/**` and a hand search
of eight predecessor archives, (b) **no external prior-art search was performed
anywhere in this task**, and (c) external novelty for candidates C2 and C3 is
therefore **UNRESOLVED**, not established.

`[E]` **Direction of the change**: strictly weakening. The file now claims less
than it did. Nothing was added in its favour.

`[J]` The validator's objection was correct and worth more than the rule it
enforces: `no predecessor found` in a local corpus is a statement about that
corpus, and this task's C3 — a common offset cancelling in a difference — is
elementary enough that external prior art very likely exists.
`NOVELTY_MATRIX.md` and `ASSESSMENT_AH.md` §E carry the same limitation, and the
search is recorded as owed.

---

## AMEND-3 — `RECOMMENDATION.md`, one factual cell corrected

| | |
|---|---|
| frozen at | `synthesis_closed`, 2026-09-03T20:01:29Z |
| was | `sha256:a74535340369…` |
| now | `sha256:d1304f957185…` |
| trigger | self-audit of `VALIDATION.md` against the **completed** check suite |
| authorised by | lead |

`[E]` The "What is ready" table read `23/24 automated checks pass`. That count
came from a run with `--quick`, which skips the two slow checks (the smoke test
and the bit-level reproduction). The completed suite is **25/26**.

`[E]` **What did not change**: the verdict, the recommendation, every number,
every cost, every caveat — and the identity of the failing check, which is
Stage 8 in both counts.

## What was NOT amended, and could have been

`[E]` **`REDTEAM.yaml` declares `lead_summary_seen: true`**, which makes
`validate_redteam.py` refuse it under rule R3 and `validate_task.py` report T4.

`[E]` **The flag was not changed.** Setting it to `false` would have made the
task validate clean by misdescribing how the review was produced. `[J]` The
error is left standing, `VALIDATION.md` §11 records it as the campaign's one
unrepaired gap, and `RECOMMENDATION.md` states that every "survives" verdict in
that file should be treated as unreviewed.

`[E]` **No frozen tolerance, criterion or hypothesis was touched at any point.**
`SUCCESS_CRITERIA.yaml`, `ANALYSIS_SPEC.yaml` and `FALSIFICATION_PLAN.md` carry
their original hashes.

## Non-frozen files revised after the stage-3 freeze

`[E]` Recorded for completeness; none of these is under the phase lock.

- `tools/cost_model.py` — peak-RSS measurements were added as they completed, in
  two waves, and **the second wave reversed part of the first**. The first
  probes lowered several requests by replacing a conservative model with a
  measurement (`D_L128_nc2048` 14G → 6G, `C_L96_nc2048` 8G → 4G). Then repeat
  probes of the same cells came in **higher** — `L = 128, N_c = 2048` by a
  factor 1.80 across three probes — so the model switched to the **maximum over
  probes** and
  `D_L128_nc2048` went 6G → **9G**, `A_L64_nc8192` 6G → **7G**. `[J]` A request
  that moved down and then back up is worth recording as such: the first
  measurement was not a bound, and the package would have shipped 6G against a
  cell observed to reach 6.3 GB. **No runtime rate changed at any point.**
- `tools/cost_model.py`, again — a final audit of the probe table against the
  raw probe output found two record errors: `L = 96, N_c = 2048` was listed with
  two identical probes when only one complete probe exists (the second run's
  output line was truncated by a concurrent write), and `L = 128, N_c = 2048`
  was listed with two probes when three exist. `[E]` Corrected. **Neither
  changed a `--mem` request** — both maxima are unchanged — but the first was a
  fabricated agreement inside the very table that argues single probes are not
  bounds, so it is recorded rather than quietly fixed. `VALIDATION.md` §9 states
  it in place.
- `tools/design.py`, `tools/build_arms.py` — campaign B2 was rebuilt from three
  `lambda` to seven, for the reason recorded at the changed line and in
  `FALSIFICATION_RESULTS.md`. This happened **before** the stage-3 freeze; the
  frozen `ANALYSIS_SPEC.yaml` and `SUCCESS_CRITERIA.yaml` describe the
  seven-point design.
- `tools/reproduce_check.py` — its pass criterion was changed from bit-equality
  on every field to bit-equality on the trajectory plus `1e-12` relative on
  derived reductions, after the first run showed the reductions differ at
  `~5e-16` because of x86-versus-arm64 summation order. `[J]` A criterion
  loosened after seeing a result is exactly the move this project distrusts, so:
  the loosening is by three orders of magnitude *tighter* than anything that
  could hide a real change, the trajectory criterion was **not** loosened, and
  the reason is written at the line that changed.
