# SLOP_WARNINGS — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter §6. **Twelve explicit verdicts, not a summary judgement.** A flagged
direction is recorded with why it fails and whether a stronger reformulation
survives — never discarded silently. Labels `[E]` `[I]` `[C]` `[J]`.

Verdicts apply to the task as a whole; where a specific candidate is implicated
it is named.

## Verdict table — all twelve, charter §6

| # | warning | verdict | basis |
|---|---|---|---|
| 1 | established method on a routine new dataset / parameter values | **FLAGGED, accepted as accurate** | the task is a replication at three new `zeta`; `NOVELTY_GATE.md` classifies it as such and no novelty is claimed |
| 2 | two known techniques combined with no nontrivial interaction | **not flagged** | nothing is combined; one sampler, one observable, one ladder, byte-identical |
| 3 | a metric that is a rename or monotone transform | **not flagged** | `OBS-CMI-001` unmodified; the reporting statistics are not locators |
| 4 | another constraint on a familiar optimisation | **not applicable** | no optimisation is performed |
| 5 | architecture swap for a small benchmark gain | **not flagged** | new algorithms are forbidden by the brief; sha256 `0a33c403…` enforced by `P11` |
| 6 | a theorem whose assumptions encode its conclusion | **not flagged, actively guarded** | no theorem; grids span both open positions of `DISP-PHI-001`, enforced by `P9`, `N8` shows it firing |
| 7 | a regime constructed to flatter the method | **FLAGGED, mitigated** | grids, pair and thresholds frozen pre-data; the one threshold change is `amendment_1`, made on synthetic data with a known answer |
| 8 | weak or disadvantaged baseline | **not flagged** | rung-to-rung at matched `R`; the `zeta = 0.35` row is declared NOT MATCHED on `R` and grid and is never pooled |
| 9 | computational scale mistaken for depth | **FLAGGED, mitigated** | the largest rung (42.6 % of the design) is conditional, and its interlock **holds** on two of four outcomes |
| 10 | runnable code treated as evidence a problem exists | **not flagged** | the problem is an absence of data; C4 and C5 were killed for running code that would not answer anything |
| 11 | silo-breaking novelty from terminology | **not flagged** | no cross-field claim, no `BRIDGE_AUDIT.md`, adjacent literature named and explicitly not claimed |
| 12 | a paper drafted around an artifact | **not flagged** | no manuscript exists or is proposed |

---

**1. Applying an established method to a routine new dataset, model, topology or
application.** — **FLAGGED. This is exactly what the task is.**
`[E]` An established `N_c` ladder, an established sampler and an established
observable, applied to three new values of `zeta`.
`[I]` **Why it survives the flag:** the three values are where the programme has
**zero** production data, the ladder's answer at `zeta = 0.35` is explicitly
forbidden from being transferred, and the design's own `NOVELTY_GATE.md` calls
it *replication at new `zeta`* rather than a contribution. `[J]` A routine
application is the right instrument when the gap is an absence of data rather
than an absence of ideas. The flag is not withdrawn — it is the correct
description — and no artifact in this task claims otherwise.

**2. Combining two known techniques without identifying a nontrivial
interaction.** — **not flagged.** `[E]` Nothing is combined. One sampler, one
observable, one ladder, byte-identical to the predecessors'.

**3. A metric that is a monotone transformation, weighted sum, or rename of an
existing quantity.** — **not flagged.** `[E]` `OBS-CMI-001` is used unmodified.
No new metric is defined. `[E]` `Delta_N`, RMS and the SEM-normalised `z` are
reporting statistics over an existing observable, not new quantities, and none
is presented as a locator.

**4. Another constraint on a familiar optimisation problem.** — **not
applicable.** `[E]` No optimisation is performed anywhere.

**5. Swapping one architecture for another for a small benchmark gain.** —
**not flagged.** `[E]` The brief forbids a new algorithm and none is introduced;
`support/instrumented.py` is byte-identical (sha256 `0a33c403…`) and preflight
`P11` refuses to run on any other bytes.

**6. A theorem whose assumptions largely encode the desired conclusion.** —
**not flagged, and actively guarded.** `[E]` No theorem is proved. `[E]` The
project-specific instance of this failure is real and recent — three separate
derivations of `sqrt(zeta)`, each invalidated, each replaced by another
derivation of the same answer — so the `lambda` grids are constructed to span
**both** open positions of `DISP-PHI-001` and are centred on neither. Preflight
`P9` fails if that stops being true, and negative control `N8` shows it failing.

**7. A simulation regime constructed mainly because it makes the method look
good.** — **FLAGGED as a live risk, and mitigated.**
`[E]` The programme's recorded instance is `DISP-WINDOW-001`: a window chosen
after seeing the answer.
`[I]` The analogous failure here would be choosing the `lambda` grid, the `L`
pair or the classification thresholds after seeing which choice gives a tidy
answer. `[E]` Mitigations, all pre-data: the grids are frozen in
`ANALYSIS_SPEC.yaml`; the primary pair is fixed at `L48-L64` *in advance* and
`pair_selection_rule` says so; the four classes and their thresholds are frozen;
`INCONCLUSIVE` is evaluated first; and the one threshold change was made on
**synthetic data with a known answer before any datum existed**, recorded as
`amendment_1` with reason and authoriser through `task_phase.py amend`, which
records the old hash.
`[J]` The mitigation is procedural, not logical: nothing physically prevents a
later reader from re-tuning. The amendment record is what makes that visible.

**8. Comparison against weak, obsolete, incorrectly implemented, or
informationally disadvantaged baselines.** — **not flagged, and one specific
trap avoided.** `[E]` The comparison is rung-to-rung within one sampler at
matched `R`, so no baseline is disadvantaged. `[E]` The `zeta = 0.35` reference
row is quoted at its own `R = 24` on its own grid and is **never pooled or
differenced** against a measured row; `ANALYSIS_SPEC.yaml`'s
`matched_observable_check` lists `R` and `lambda_grid` as **NOT MATCHED** and
draws the consequence.

**9. Treating computational scale as scientific depth.** — **FLAGGED, and it is
the reason for the conditional arm.**
`[E]` 6 512 populations and 2 022 core-hours is a large number that could be
mistaken for a large result.
`[E]` The mitigation is structural: the single largest rung — 861.6 core-hours,
42.6 % of the design — is **held back behind an interlock whose `hold` branch
fires on two of four outcomes**, including the one most likely to be misread as
needing more compute (`INCONCLUSIVE`, where `R` binds and more `N_c` buys
nothing). `[E]` C2 (`R = 24`) and C5 were killed on the same principle.
`[J]` The recorded programme instance — a 5 634-realization campaign that sat
unanalysed for six weeks — is why the analysis script is written, tested and
shipped **before** submission rather than after.

**10. Treating the existence of runnable code as evidence that a research
problem exists.** — **not flagged.** `[E]` The problem is stated from an absence
of data (`PROBLEM_MEMO.md` §1) and the code exists to answer it. `[E]` The task
also declines to run code where it would not answer anything: C5's reuse
shortcut and C4's `L`-reduction were both killed for removing a diagnostic.

**11. Claiming silo-breaking novelty from terminology differences.** — **not
flagged.** `[E]` No cross-field claim is made, no `BRIDGE_AUDIT.md` is written,
and `ASSESSMENT_AH.md` §E records cross-silo value as **NONE CLAIMED**. `[E]`
The obvious adjacent literature — particle-filter population sizing — is named
in `FIELD_MAP.md` and explicitly **not** claimed as a bridge.

**12. Drafting a paper around an artifact before identifying the scientific
claim.** — **not flagged.** `[E]` No manuscript, abstract, introduction or
contribution list exists or is proposed. `[E]` The deliverable is two tables and
a recommendation with a caveat column.

---

## Summary of the three flags

`[E]` **№1** (routine application), **№7** (regime that flatters the method) and
**№9** (scale mistaken for depth) are flagged.

`[J]` №1 is **accepted as an accurate description** and is not argued away: the
task is a replication, the gate says so, and its value rests on A and G of the
Meaningful-Contribution Test, not on novelty. `[E]` №7 and №9 are **mitigated by
pre-registration and by the conditional interlock respectively**, and both
mitigations are mechanical rather than promised — a frozen spec with an
attributed amendment record, and an arm that lives in a directory a
submit-everything loop cannot reach.

`[E]` None of the three is discarded. A stronger reformulation of №1 would be a
mechanism question (does the drift follow the `O(1/N)` particle-filter
structure?), and it is **parked, not pursued**, because the brief commissions a
survey and §14 forbids turning it into something else.
