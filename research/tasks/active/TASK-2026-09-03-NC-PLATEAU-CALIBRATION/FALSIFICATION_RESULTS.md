# FALSIFICATION_RESULTS — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Outcomes against the **frozen** `FALSIFICATION_PLAN.md`. That plan is a separate
file, it is hashed in `TASK_MANIFEST.yaml`, and nothing here is merged back into
it.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## Status of the physics falsifiers Y1–Y8

`[E]` **NOT YET ATTEMPTED. NO HPC DATA EXIST.** This task's terminal state is
`READY_FOR_HUMAN_SUBMISSION`; the campaign has not been submitted, because
`research/RESOURCE_POLICY.md` §4 forbids any agent from submitting it.

| falsifier | target | status |
|---|---|---|
| Y1 plateau at `L = 64` | C1 | **awaiting campaign A** |
| Y2 asymptotic form | any coefficient `B` | **awaiting A, C, D** |
| Y3 shape H1/H2/H3 | C2 | **awaiting campaign B** |
| Y4 locator convergence | C3 | **awaiting B + B2** |
| Y5 `L = 96` | C4 | **awaiting campaign C** |
| Y6 `L = 128` | C5 | **awaiting campaign D** |
| Y7 discretisation E1/E2 | C6 | **awaiting campaign E** |
| Y8 the six pre-registered negative outcomes | — | **awaiting all** |

`[E]` Writing an outcome for any of these now would be fabrication. The analysis
that will evaluate them is written, frozen, and demonstrated to run to
completion on zero new results (`tools/smoke_test.py` check 7), which is the
whole point of writing it before the data arrive.

## Y9 — the falsifiers aimed at this task's own machinery, which WERE attempted

`[E]` These do not need HPC data and all five were executed.

### Y9.1 — "The reuse ledger is fiction." **REFUTED.**

`[E]` `tools/dedup_scan.py` compares all 19 reuse-ledger entries against the
populations actually on disk. Every claimed `R` matches the file count exactly;
every reused population has `zeta = 0.35`, `T = L`, systematic resampling and
the `n_steps` the discretisation formula predicts. **SCAN PASSED.**

### Y9.2 — "The modified `run_cell.py` changed the sampler." **REFUTED, decisively.**

`[E]` Two completed predecessor populations re-executed through this task's
wrapper: **all 1 024 per-clone CMI values bit-identical** in both cases, every
integer diagnostic exactly equal.

`[E]` **And one thing was learned that the falsifier was not aimed at.** The
derived reductions — `cmi_weighted_mean`, `cmi_within_var`, `ess_cum_final` —
differ from the stored values by up to `1.7e-14` relative. That is x86-versus-
arm64 summation order in numpy's pairwise reductions, not a code change: the
trajectory is identical. `[I]` **A stored AGGREGATE in this corpus is not
bit-reproducible on a different architecture even though the physics is.**
Anyone re-deriving a published mean on a different machine needs to know that,
and nothing in the repository said so. The check's criterion was tightened to
match — exact on the trajectory, `1e-12` relative on reductions — and the
reason is recorded at the line that changed.

### Y9.3 — "The preflight passes everything." **REFUTED.**

`[E]` `tools/negative_controls.py`: sixteen injected faults, each of which must
make the preflight exit non-zero **for the injected reason**. All sixteen fired:
off-grid `lambda`, duplicate seed, wrong `zeta`, wrong `dtau_mult`, `T != L`,
wrong resampler, a predecessor's seed, a short row count, a mismatched
`--array`, a short `--time`, an under-requested `--mem`, the `--mem=200`
unit trap, `cpu_short`, a multicore request, a drifted `run_cell.py`, and an
injected scheduler call.

`[J]` **A limitation of that result.** Controls N01–N08 all fire through the
same check — "manifest == frozen design" — because the design-identity
comparison is the first and strongest gate and catches them before the
field-level checks run. The field-level checks are a second line of defence
that would fire only if the builder itself were changed. That is the correct
architecture and it does mean eight controls test one mechanism.

### Y9.4 — "The cost model is fitted to a requested `--time`." **REFUTED by construction.**

`[E]` Every rate in `tools/cost_model.py` carries its rung, its `n`, and the
median/p90/max it came from; the model adopts the **max**. `[E]` The
discretisation `K` is verified exact against the `n_steps` every one of the
1 896 completed runs recorded for itself.

### Y9.5 — "The memory model is a model quoted as a measurement." **CONFIRMED — against the INHERITED model.**

`[E]` This is the falsifier that landed. `TASK-2026-09-01-SMCRUCHE-READY`
described "the measured 732 MB peak"; 732 MB is exactly what
`128 + 2 N_c per_clone` returns for `L = 96, N_c = 512`, and no `MaxRSS` from
any Ruche job appears anywhere in the repository. `[E]` Direct `ru_maxrss`
measurement at 14 cells shows the formula **under**-predicts at seven of them —
`L = 64`, `N_c = 2048` reads 1 694 MB against a predicted 1 202 MB, and that arm
shipped `--mem=2G`. `[J]` It never broke and it was closer to breaking than
anyone knew.

`[E]` Fixed here: every immediate arm's `--mem` is now sized from a direct
measurement of that cell, with the old formula retained as a floor and a 1.35×
margin. `[E]` The limitation is stated rather than argued away: these are macOS
measurements and the cluster is Linux, which is why `RUCHE_RUNBOOK.md` §7 asks
for one `sacct … MaxRSS` line.

## Two design decisions that a falsifier killed before submission

`[J]` Recorded here because a design that changed under attack is a result, and
because both cost real core-hours.

**1. `R = 24` in campaign A was killed by its own power calculation.** `[E]` At
`R = 24` the `4096 → 8192` step's `Delta` half-width would be ~1.2× `tau_I`, so
the arm **could not have satisfied P2 whatever the data did**. `[I]` An arm that
cannot pass its own frozen criterion is not a measurement. `R` was raised to 48
(+166 core-hours), and the residual — the *lower* step still needs `R ≈ 80` —
is stated in `CAMPAIGN_DESIGN.md` §2A rather than hidden.

**2. The cheap version of campaign B2 was killed by the frozen crossing
protocol.** `[E]` B2 originally used only the three `lambda` that campaign B
shares with the measured 0.010 grid, at a sixth of the cost. Running the frozen
protocol on that grid showed **both** interior crossings fall in the first or
last interval, so every crossing is flagged `ENDPOINT_INDUCED` **by
construction**, whatever the data say. `[I]` That is exactly the defect
`TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION` was created to repair. The arm was
rebuilt on the full seven points (+220 core-hours) and `tools/design.py` records
the reason at the line that changed.

## One accepted claim that this task could not reproduce

`[E]` **The `L = 96` half of "a clean `1/N` is rejected at `L = 96` and
`L = 128`" does not reproduce.** `L = 128` reproduces exactly
(`chi2 = 12.58`/3, `p = 0.0056`). The `L = 96`, `lambda = 0.3032` ladder that can
be rebuilt from raw files has three rungs and gives `chi2 = 1.90`/1,
`p = 0.168` — **not rejected**. `[I]` The predecessor's `chi2 = 10.54` on 3 dof
needs a four-rung ladder this reconstruction cannot locate.

`[J]` This is reported as an **open provenance item**, not as a refutation. A
ladder this task cannot find is not a ladder this task has shown to be wrong,
and `CLAUDE.md` requires following a claim's direct provenance before
contradicting it. What it does mean is that the `L = 96` half of the accepted
framing should not be leaned on until someone locates the ladder — and that
`L = 96` is *less* characterised than assumed, which strengthens the case for
campaign C rather than weakening it.

## What is owed before this file can be completed

`[E]` The immediate group must run, `analysis/nc_plateau_analysis.py` must be
executed once unedited on the complete set, and Y1–Y8 must then be written
against the frozen plan. `[E]` `FALSIFICATION_PLAN.md` Y8 pre-registers six
negative outcomes — including the most likely one, `UNRESOLVED_R_LIMITED` at
`L = 96` and `L = 128` — so that reporting them is not a retreat.
