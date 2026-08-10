# Stage 1 findings: source map and staleness audit

Audit 2026-08-10. **Non-canonical.** Nothing outside `audit/2026-08-10/` was
created, modified, moved, staged, committed or pushed.

Companion files: `01_CLAIM_LEDGER.md`, `02_DATA_INVENTORY.md`,
`03_AMPLITUDE_TRACE.md`, `04_MISSING_RESULTS.md`.

---

## 1. Information-source map

### Tier A — read in full during Stage 1

| source | size / date | reliability |
|---|---|---|
| `theory/AGENTS.md` | 141 lines, 2026-06-03 | **procedurally sound, factually dangling.** Protocol is good. Its file pointers predate the archive reorg. |
| `theory/HANDOFF.md` | 1174 lines, 133 KB, mtime 2026-08-10 10:28, **uncommitted +132 lines** | **internally contradictory by construction.** See §2. |
| Claude project memory | current | **least reliable source in the system.** See §3. |
| `theory/Y_ZETA_DERIVATION.md` | 2026-06-10 | reliable *because* it carries an explicit superseding header naming what survives (§0, §2, Δ_B=1/Δ_cross=2) and what does not. Model for how a superseded doc should look. |
| Aggregate pickles (10 current, 8 legacy) | 2026-06-17 | **the most reliable scientific source in the project.** Self-describing, complete metadata. |
| Git history | last commit 2026-07-26 | reliable but **three weeks behind the working tree**. |

### Tier B — located and characterised, not read in full

`theory/CURRENT_THEORY_STATUS.md` (2026-06-10), `theory/NUMERICS_STATUS_AND_PLAN.md`
(2026-06-10), `theory/OPEN_ANALYTIC_PROBLEMS.md` (45 KB, 2026-06-10),
`theory/VARIANCE_REDUCTION.md` (2026-08-09, **untracked**),
`theory/CASE_A_IMPLEMENTATION_SPEC.md` (2026-05-20),
`theory/theta1_first_principles.md` (2026-05-18),
`theory/qj_bosonization_calculation.md` (39 KB, 2026-05-11),
`theory/collaborator/` (4 files), `theory/bosonization_roadmap/`,
`theory/archive/` (21 files, all 2026-05-11 → 06-07),
`docs/architecture_theory_map.md`, `CONTEXT.md`, `README.md`.

### Tier C — code and infrastructure

- Core: `pps_qj/gaussian_backend.py` (+ `_caseA`, `_jit`), `cloning.py`
  (+ `_caseA`, `_jax`), `exact_backend.py` (+ `_caseA`), `doob_wtmc.py`,
  `backward_pass*.py`. **Nine `.bak_*` files sit beside live modules, untracked.**
- `pps_qj/parallel/` — grid + worker families (clone, dense, rescue, ladder,
  phase2, caseA, opdim, areaphase, zeta0, chi2).
- `analysis/` — 115 items, **~60 untracked**, mixing live scripts, dead scripts,
  and ~40 loose PNGs with no provenance record.
- `analysis/var_reduction/` — 20 scripts, **entirely untracked**, newest
  `crossing_prod.py` 2026-08-10 11:06.
- `slurm/` 38 scripts, `scripts/ruche/`, `scripts/habrok/`, `tests/` (10 files,
  3 untracked including `conftest.py`), `notebooks/` (10).
- Manuscripts: `continuousmeasurementslatex/` (in repo, 8 sections) and
  `~/Downloads/01_M1_Internship/Thesis/m1thesislatex/` (7 chapters, the one
  carrying the amplitude error). **There is no `paper/` directory.**
- Data: `~/Downloads/01_M1_Internship/{Data,Code,Figures,Papers,Thesis}`;
  `results/`, `outputs/`, `saturation_output/` in repo; `/tmp` scratch.

---

## 2. Why HANDOFF.md fails structurally

HANDOFF is not a handoff. It is a reverse-chronological session log in which
**every correction was implemented by prepending a block rather than by editing
the superseded statement**. Consequences, all present today:

1. **The bottom half contradicts the top half.** The TL;DR still presents the
   matched-NLSM √ζ derivation (Δ_ζ=1, ξ~λ⁻²) as *the* framework; the top
   declares both inputs invalid. "Numerics (best current estimate)" still gives
   φ = 0.56 ± 0.05, C ≈ 0.91, "ν scattered around ~2 consistent with the
   theory-predicted plateau"; the 2026-06-17 block says collapse cannot resolve ν
   at all. Two inline NOTE patches were added, but the superseded numbers remain
   in place and readable.
2. **The header lies about its own currency.** "Last major update: 2026-07-07"
   sits below blocks dated 2026-07-27, 2026-08-09 and 2026-08-10.
3. **Correctness now depends on reading order.** A 1174-line document read
   top-to-bottom yields the right answer; read by search, grep, or partial load
   it yields whichever layer was hit. Any agent with a context budget reads it
   partially.
4. **It is simultaneously the status document, the results database, the lab
   notebook, the decision log, the file map and the operational runbook.** No
   single-writer discipline is possible.
5. **Its file map and data table are both wrong.** Every `~/Downloads` path is
   dead; several theory files are listed at non-archive paths.

## 3. Why project memory is the worst source

Memory is a snapshot of roughly 2026-05-22 that has not tracked the June–August
corrections, and in one case actively **regressed a corrected value while
presenting the regression as the correction**.

| memory assertion | actual state |
|---|---|
| "Confirmed result: A = 0.96 ± 0.05 (not ~0.5; ~0.5 applies to r_c)" | inverted. λ_c A ≈ 0.49 (reproduced this audit); r_c ≈ 0.87 |
| "φ = 0.502 ± 0.026, matching predicted 1/2" | `[X]`, rests on the above and on a B_L method demoted 2026-06-17 |
| "y_λ=1/2, y_ζ=1 from cross-Choi Δ_ζ=1" | Δ_ζ=1 corrected to Δ≈2 (marginal) |
| "λ_c(1) ≈ 0.5 matches Carollo et al." | attribution `[X]` since 2026-06-06 |
| "ν = 1/y_λ = 2" | ν_B unmeasured, confidence set ≈ [1.5, 3] |
| "Use 1/√L (not 1/L) since ν = 2" | circular on an unmeasured ν |
| "SciPost paper draft: `paper/main.tex`" | **no such path exists** |
| "amplitude conflict … urgent — §7 of Y_ZETA_DERIVATION.md" | fixed 2026-06-10 |
| "Cut A implementation: current code only does Cut B" | Cut A implemented, validated, run, 574 records aggregated |
| pending: submit Run A / B / C per `SESSION_2026_05_20.md` | ~3 months and 6 campaign revisions stale |
| "Main empirical aggregate: `clone_aggregate(1).pkl`" | path dead; superseded 2026-06-17 |
| "König–Brouwer 2014 is hallucinated" | **correct and valuable.** One of memory's genuine assets |
| "Click vertex is marginal (Δ=2), purely chiral" | Δ=2 `[V]`; "purely chiral" is contested (`CB-VERTEX-001`) |

Memory also **omits** the two most important recent developments entirely: the
master metric `t_wall·σ²_λc`, and the `N_eff ≈ N_c` mechanism that closed the
sampler programme.

---

## 4. Staleness and contradiction table

Categories: **CUR** current/reliable · **SURG** mostly current, needs surgical
correction · **STALE** substantially stale · **HIST** historically useful, must
not guide new work · **RED** redundant · **UNCL** unclear.

| document | cat | specific problematic claim or section |
|---|---|---|
| Claude project memory | **STALE** | See §3. Highest-priority artifact in the whole audit. Contains an inverted correction. |
| `theory/HANDOFF.md` | **SURG** (top ~40%) / **STALE** (bottom ~60%) | TL;DR "Theory" §; "Numerics (best current estimate)" (φ=0.56, C=0.91, ν≈2 plateau); "Key result: φ from global FSS"; "Data on disk" table (all paths dead); "File map (theory folder)" (non-archive paths); header "Last major update: 2026-07-07"; Case A "not yet run". |
| `theory/AGENTS.md` | **SURG** | "most common follow-on reads" lists 3 files that exist only under `archive/`. End-of-chat protocol assumes HANDOFF is the sole sink — the thing this redesign removes. |
| `theory/Y_ZETA_DERIVATION.md` | **HIST** (well-marked) | Correctly self-superseded. §7 fix landed 2026-06-10. Retain as the exemplar of good superseding practice. |
| `theory/archive/NLSM_FRAMEWORK.md` | **HIST** | Line 286 `A = 0.96 ± 0.05`. HANDOFF says "do not cite". Memory cites it. |
| `theory/archive/SESSION_2026_05_20.md` | **HIST** | Origin of both `A = 0.96` and `φ = 0.502 ± 0.026`. Memory's pending-actions list still points here. |
| `theory/archive/HANDOFF.md.bak*` (4 files) + `theory/HANDOFF.md.bak_20260617` | **RED** | Five superseded HANDOFF copies. Two carry the boxed `A = 0.96` result. Grep-reachable, indistinguishable from current on a keyword search. |
| `theory/CURRENT_THEORY_STATUS.md` | **UNCL** | HANDOFF (2026-06-03 note) says it is "NOT yet reconciled to 2026-06-10". mtime is 2026-06-10 22:22, so it may have been reconciled after the note was written. Unresolved. Name asserts currency it may not have. |
| `theory/NUMERICS_STATUS_AND_PLAN.md` | **UNCL** | Same note, same ambiguity. Superseded in substance by the 2026-06-17 data plan and the 2026-08-10 next-queue. |
| `analysis/anchor_scan.py` | **STALE / hazardous** | Hardcoded kernel drops the hopping w from the measured bond. Produced the falsified Fermi-step / λ*=4/5 / ν₀=1 results. HANDOFF says "do not trust it". **The file carries no warning header and `delta_B_hook()` raises NotImplementedError.** |
| `theory/VARIANCE_REDUCTION.md` | **SURG**, untracked | §5 "learned Doob h_θ: no headroom" is explicitly **RETRACTED** by the 2026-08-09 block, but the retraction lives only in HANDOFF. Reading this file alone gives the retracted conclusion. |
| `theory/archive/qj_chiral_vertex_result.md` | **HIST**, live tension | "purely chiral ⇒ K=1 all orders ⇒ ν constant" conflicts with the 2026-06-17 Ashkin–Teller mechanism. Reconciliation flagged as "the next THEORY task" and not done. |
| `Chapters/Chapter3.tex` (m1thesislatex) | **STALE, published** | Line 236 asserts `A ≈ 0.96` and infers "prefactor roughly twice the diffusive one". In `main.pdf`. See `03_AMPLITUDE_TRACE.md`. |
| `Chapters/Chapter{5,7}.tex` | **STALE** | Outline comments carry `A = 0.96`, `φ = 0.502 ± 0.026`, "sqrt(zeta) ansatz 28% better". |
| `continuousmeasurementslatex/` | **UNCL** | Second manuscript, `sec5.tex` / `sec5new.tex` / `sec5newnew.tex` — three generations of the PPS section, no marker of which is live. |
| `pps_qj/*.bak_{guided,spawn,prelowrank,prenewton}` (9 files) | **RED** | Pre-refactor module copies beside live modules, untracked. Import-shadowing and grep-confusion hazard. |
| `slurm/submit_run_{A,B,C}.sh` | **HIST** | The runs memory still lists as pending. Superseded. |
| `slurm/submit_clone_{rescue_L128,rescue_L160,dense_L64_backfill}.sh` | **HIST** | HANDOFF lists as "submit needed"; the Ruche migration and guided-cloning rebuild superseded the campaign. |
| `analysis/extract_yzeta.py` | **STALE** | Uses the ζ-as-scaling-field collapse that `Y_ZETA_DERIVATION.md` §7 identifies as anchored at the wrong (singular) endpoint. |
| `/tmp/*` | **CUR but ephemeral** | Sole surviving evidence for the entire 2026-08-09/10 programme. |

### Specific issue types the memo asked to detect, as found

- **Old amplitude mislabelled** — `03_AMPLITUDE_TRACE.md`. Found, in a manuscript, and re-inverted in memory.
- **Heuristic later treated as a derivation** — `PPS-YLAM-001`: `y_λ = 1/2` is an ε-expansion *calibrated to* Jian's ν, described as derived in memory and in the TL;DR.
- **Superseded numerical extrapolation** — `METH-EXTRAP-001`: 1/√L justified by ν=2, where ν=2 is unmeasured.
- **Theory invalidated by later calculation** — `CB-NLSM-001`: the matched-NLSM √ζ derivation, both inputs failed, still the framework in memory.
- **Experiment marked pending that has been run** — `CASEA-IMPL-001` (Cut A); `04_MISSING_RESULTS.md` M1 (T=L certification), M2 (L=48 read).
- **Data that exist but are absent from the handoff** — Case A aggregates; all of `/tmp`; the relocated Downloads tree.
- **Result based only on chat rather than executed analysis** — θ₁/SCGF and the Cut A end-to-end-MI observable claim (`04_MISSING_RESULTS.md` M4). Stage 2 targets.
- **Citation later found incorrect** — Carollo PRA 98 010103 (`CB-BORN-002`); König–Brouwer 2014 (hallucinated, correctly recorded in memory).

---

## 5. Preserved disagreements

Recorded, not resolved. None of these should be collapsed by choosing the newer source.

1. **φ ≈ 0.5 on λ_c vs φ ≈ 0.68 on r_c.** Same data, two parameterizations, different exponents. Reproduced in this audit. Compounded by five statistically indistinguishable functional forms on λ_c.
2. **φ = 1 "excluded at 9σ" vs `linear+intercept` fitting at χ²/dof 0.55.** Both in HANDOFF.
3. **Cut A: NLSM predicts Ising ν=1 for all ζ vs LMR's crossover to 5/3.** Genuine open physics. Data to test it partially exists (ζ up to 0.85).
4. **Cross-vertex purely chiral (archive) vs non-chiral ε_+ε_- at the Ising corner (AT memo).** Possibly complementary corners; not settled.
5. **ξ ~ λ⁻²: refuted (2026-06-10) then reinstated for the no-click band structure (2026-06-15).** Both partly right; the refutation itself rested on a wrong script.
6. **Snapshot averaging: "largest confirmed lever" vs L=48 giving 1.22×.**
7. **Controlled cloning: 1.42× at L=32 vs 0.99× at L=64.** Mechanism identified (`VR-CLOSE-001`) but the L-dependence of the earlier positive is still the only anomaly the mechanism does not fully explain.
