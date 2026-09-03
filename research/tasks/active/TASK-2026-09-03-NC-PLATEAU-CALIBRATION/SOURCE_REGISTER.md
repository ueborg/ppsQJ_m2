# SOURCE_REGISTER — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Charter Stage 0, **task-scoped**. Freezes at `stage_1_problem`. Sources
inspected *during* the run go in `SOURCE_INSPECTIONS.yaml`, which is append-only.

This is a **campaign-preparation** task. Its load-bearing sources are almost
entirely internal — predecessor task artifacts and, decisively, the raw result
files those tasks produced. No external literature is load-bearing here: the
question is what this sampler does at this cell, and no paper answers that.

---

## 1. Load-bearing sources, with inspection level BEFORE any work

| id | what | inspection level at task open | why it is load-bearing |
|---|---|---|---|
| `SRC-RAW-CORPUS` | the 1 896 per-population result JSONs under `research/tasks/**/results/` | **not previously enumerated as one corpus** | every ladder, every rate, every variance and every reuse decision in this task comes from these files and from nothing else |
| `SRC-FNCRS` | `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` final artifacts (`RECOMMENDATION.md`, `HUMAN_SUBMISSION.md`, `NUMERICAL_RESULTS.md`, `ruche_package/README.md`) | post-red-team final | supplies the structural obstruction, the `1/N` rejection, and campaign E's exact design and pre-registered predictions |
| `SRC-MOCKPROD` | `TASK-2026-09-02-MOCK-PRODUCTION` (`COST_MODEL.md`, `RUCHE_RUNBOOK.md`, arms) | final | the 13-point `N_c = 1024` curves and the `L = 64` `N_c = 2048` triple |
| `SRC-LOWLAM` | `TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION` (all) | final | the four low-`lambda` points that make the crossings interior; the package architecture this task inherits; the crossing/endpoint protocol |
| `SRC-HIGHRUNG` | `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA` (`COST_MODEL.md`, `SCHEDULER_DECISION.md`, arms) | final | the `L = 128` `N_c = 512, 1024` rungs, the measured Ruche rates, the partition evidence |
| `SRC-SMCRUCHE` | `TASK-2026-09-01-SMCRUCHE-READY` arms 1 and 2 | final | the `L = 96` and `L = 128` low rungs; the `--mem` formula whose provenance this task had to check |
| `SRC-SMCSTAT` | `TASK-2026-08-30-SMCSTAT` (`AN7_scheme_chunking.json`, `instrumented.py`) | final | the certified sampler, validated bitwise against production; the only prior `dtau_mult ∈ {3,6,12}` comparison anywhere |
| `SRC-SMCCERT` | `TASK-2026-08-31-SMCCERT` | final | the standing production rule this task did not improve on: per-cell calibrated `B`, `N_c` from the conservative end of its CI, `R` afterwards |
| `SRC-POLICY` | `research/RESOURCE_POLICY.md` §4, `research/RESEARCH_CHARTER.md`, `CLAUDE.md` | fully inspected | the submission prohibition and the evidence-tier rules that bind every artifact here |
| `SRC-NUMCHARTER` | `NUMERICAL_CAMPAIGN_CHARTER.md` §0, §1 | relevant sections | the governing constraint that the campaign must not be able to manufacture a `phi` |

## 2. What the sources are used FOR, and what they are not

`[E]` `SRC-RAW-CORPUS` is the only **evidential** source in this task. Every
other predecessor artifact is read for **design and provenance**: what was
decided, why, and what it cost. `[E]` No number in this package is taken from a
predecessor's summary table. `tools/reconstruct_inventory.py` rebuilds every
ladder from the JSON the sampler itself wrote, and `tools/dedup_scan.py` fails
if a reuse-ledger entry disagrees with what is on disk.

`[J]` That was not pedantry. It surfaced two discrepancies that a summary-level
read would have carried forward: the `L = 96` `1/N`-rejection that does not
reproduce from the `lambda = 0.3032` raw ladder (`PROBLEM_MEMO.md` §1), and the
`--mem` model that a predecessor described as measured and was not
(`COST_MODEL.md` §"Memory").

## 3. Canonical state in scope

`[E]` Read, not written:

- `OBS-CMI-001` — the observable and estimator this campaign measures.
- `OBS-BL-001` is **retired** and is not used. `OBS-BLPROD-001` and
  `OBS-BLKMR-001` are different quantities and nothing here is compared against
  either.
- `VR-CLOSE-001` — its stated assumption (that `L = 32` decorrelation behaviour
  extends to production `L`) remains untested at production `L` on its own
  observable. `[J]` This campaign does not test it either, and does not contest
  the claim: the observables differ and the conventions must not be crossed.
- `METH-TREQ-001` — `epistemic_status: unsupported`. `T = L` is used here
  because the entire reuse corpus is at `T = L`, **not** because the claim
  supports it. `[E]` Nothing in this campaign bears on `T(L)`.

`[E]` The six live disputes (`DISP-PHI-001`, `DISP-WINDOW-001`, `DISP-XI-001`,
`DISP-CASEA-UNIV-001`, `DISP-SNAPSHOT-001`, `DISP-YZETA-001`) are **not** in
scope and are not moved by anything here. `[I]` This campaign is upstream of all
of them: it decides whether the measurements that would bear on them can be
trusted at all.

## 4. What is deliberately NOT treated as a source

`[E]` `analysis/anchor_scan.py` and anything derived from it
(`EV-CODE-ANCHORSCAN-001`: its kernel drops the hopping `w`). `[E]` The
untracked `analysis/global_fss*.json`, `analysis/phase_diagram_data.json`,
`analysis/parity_sweep.log` — no provenance record, cited by nothing in state.
`[E]` `theory/**`, `audit/**`, `continuousmeasurementslatex/**`,
`research/history/**` — provenance only, and none of it is cited here as
support. `[E]` Unmerged proposals under `research/proposals/`.

## 5. External literature

`[E]` **None inspected, and none is load-bearing.** `[J]` The question is the
convergence of one implemented estimator on one cell of one model; the relevant
external work (Del Moral, Whiteley, Tadić–Doucet) was inspected by
`TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` and its conclusion — that those
hypotheses do not hold for this kernel — is used here as **provenance for a
design decision** (build an empirical calibration, because no analytic one is
available), not as evidence for any measurement. `[E]` Recording this as a gap:
a genuine prior-art sweep for *empirical* finite-`N` plateau-detection protocols
in interacting-particle Monte Carlo was **not** performed by this task, and
`NOVELTY_MATRIX.md` says so rather than claiming novelty by silence.
