# NUMERICAL_CAMPAIGN_CHARTER

**Status: planning document. This is NOT authorization to launch numerics.**

No simulation was run to write it. No HPC system was accessed. No job array was
prepared, submitted or modified. It defines what the *next* dedicated
numerical-mapping task must answer — not how to run it, and not the grid, which
is deliberately left unfrozen (see "What must arrive before the grid is
frozen").

Sources are the final post-red-team artifacts of `TASK-2026-08-11-ALGRD`,
`-ARCH`, `-MAPS`, `TASK-2026-08-12-LAMC`, `-CUTBTHEORY`, `-THEORYPRED`,
`TASK-2026-08-13-QJCONTINUUM`, `-TWOCLICK`, `-KAPPAENT` and
`TASK-2026-08-14-C2CONV`. Where a lead's finding was corrected by its own red
team, the corrected reading is used.

Marks: `[E]` evidence · `[I]` inference · `[C]` conjecture · `[J]` judgment.

---

## 0. The governing constraint

`[E]` Every theory route in the parent chain terminated **without** a derived
boundary law. `r_c <= C·zeta^(1/3)` is an *inequality* with `C` free, and
`QJCONTINUUM` contests the exponent (`l_osc` gives `zeta^(1/2)` from the same
argument). `r_c >= c·zeta` is dead. `THEORYPRED` produced `phi = 1/3`;
`DISP-PHI-001` holds `phi = 1/2` against `phi = 1`. `KAPPAENT` licensed no
statement about `r_c(zeta)` at all.

`[I]` **The campaign must therefore measure the boundary while agnostic among
`1/3`, `1/2` and `1`.** This is a hard design constraint, not a preference: the
cheapest way to choose a lambda window is to centre it on a predicted
`lambda_c(zeta)`, and doing so imports the exponent the campaign exists to
measure.

`[J]` The single most valuable property of the next campaign is **not**
precision on `phi`. It is that its bracketing procedure *cannot manufacture* a
`phi`.

---

## 1. Ranked scientific questions

Ranked by **information value per unit of compute**, not by scientific
appetite. They are not of equal priority and must not be funded equally.

### Tier 0 — preconditions. Near-zero compute, and they change the budget.

**R1. Does `T/L = 1` bias the observables? What is `T(L)`?** (brief §12.8)

`[E]` `METH-TREQ-001` is recorded `epistemic_status: unsupported` — a
conjecture created deliberately so no agent treats it as canonical. Its
relaxation argument is sound in direction but supplies **no factor**. The bulk
of the historical corpus (`pps`, `refine`, `refine_smallz`) is `T/L = 1.000`.

`[E]` The claim file names its own `cheap_available_test`, and it **has never
been run**: Cut A `caseA_guided` already spans `T/L = 2.000` (L=32,48,64),
`1.333` (L=96) and `1.000` (L=128) *within one campaign*.

`[E]` `ALGRD` attempted a horizon result and **withdrew it** under red team: the
test had no power and no positive control (T=64 vs T=128 gave B_L z = −0.34
with 95% CI [−2.29, +1.81]; even the arm called "clearly unconverged" was not
significant).

`[J]` First, because it is zero-compute, it is owed, and it moves the entire
campaign's `T` budget by a factor of two in either direction. A campaign
designed before this is a campaign designed on a guess.

**R2. Which historical cells are safe to reuse, and for what?** (§12.10)

`[E]` Reusable **for orientation and coarse bracketing only**: the coarse
lambda structure at `zeta in [0.25, 0.5]`, `L <= 128`; Cut A `caseA_guided` for
R1. Reusable as a declared-convention comparison point: `zeta >= 0.6` anchors,
not in any exponent fit.

`[E]` Excluded, each for a stated reason: `zeta = 0.55, 0.65, 0.85` (n = 36, 36,
26 runs — too thin); `zeta = 0.45` and `0.8–1.0` (L >= 64 only, so no wide-pair
extrapolation); anything from `analysis/anchor_scan.py` (`EV-CODE-ANCHORSCAN-001`,
kernel drops `w`) or `analysis/chi2_response.py`; the untracked
`analysis/global_fss*.json`, `phase_diagram_data.json`, `parity_sweep.log`;
anything labelled `OBS-BL-001` (retired, one label over two quantities); and any
cell whose crossing sits at a scan endpoint or whose observable has collapsed
into its own error.

`[J]` Cheap, and it gates R3's coarse grid. Doing it late means re-deriving the
bracket.

### Tier 1 — the campaign's spine.

**R3. Unbiased `lambda_c(zeta)` across the physical line.** (§12.1)

`[E]` The current production centring law `lambda_c = 0.51*sqrt(zeta)` with a
±0.08 half-width **fails to bracket its own crossing** where the cost is.
`MAPS` recomputed all 44 gated crossings: deviation +0.114 at zeta=0.5, +0.064
at 0.6, +0.061 at 0.7. At zeta=0.5 the offset **exceeds the entire window**, and
`zeta >= 0.5` carries **67%** of grid cost.

`[E]` `LAMC` confirmed this independently and sharpened it: of the three
interpolants its frozen `ANALYSIS_SPEC.yaml` declared *before* any crossing was
computed, **two of three fall outside the window edge**. The "it brackets"
reading survives only on two-point linear interpolation — the one interpolant
the spec refuses to treat as unbiased.

`[E]` `LAMC` also found the production grid runs `L in {32,48,64,96,128}` while
**no stored Cut B per-realisation data exists at L=32 or L=48 at all**.

`[J]` Highest-value measurement in the campaign. Everything downstream —
`nu`, `phi`, the FSS form — is a functional of `lambda_c(zeta, L)`. It is also
the question the current grid demonstrably gets wrong.

**R4. Is ordinary power-law FSS adequate, or is the behaviour KT-like /
essential?** (§12.4)

`[E]` A log-corrected transition produces crossings that drift with `L` without
converging, and over a limited `L` range that drift is well fitted by an
ordinary power law with a *shifted* effective exponent. `[E]` The project has a
live instance (`DISP-CASEA-UNIV-001`, "higher-zeta crossing drift") and a
standing finding that fitted exponents drift with the fit window
(`CB-WINDOW-001`, `DISP-WINDOW-001`).

`[I]` This is not a hypothetical failure mode here; it is the project's
**recorded** one. `[J]` Ranked immediately after R3 because it costs no extra
simulation — it is an analysis discipline applied to R3's data — and because
skipping it is what would invalidate the campaign's headline after the fact.

**R5. Finite-`N_c` and genealogy bias on the locator.** (§12.9)

`[E]` `ARCH` measured the across-realisation variance of the locator's
population mean at 2.8–16x the independent-sampling prediction, worsening with
`L` (bootstrap ratio 5.81 [1.85, 15.91] from L=32 to L=64 at matched T,
P(ratio<=1) = 0.0025), while per-step `ESS/N_c` sat at **0.969–0.984 in every
cell**. The standard health diagnostic is blind to it.

`[E]` `ALGRD` found the same blindness from the other side: `n_distinct_ancestors`
falls to **1.00 of 350** at the production horizon while `ess_frac_min` stays
0.983.

`[E]` **And it is not only variance.** `ARCH`'s own red team found, in its saved
data and unreported by the lead, a finite-`N_c` shift in the locator itself:
mean population `B_L` = 1.2015 ± 0.0763 at `N_c = 44` versus 0.9161 ± 0.0348 at
`N_c = 350` (**z = 3.4**).

`[E]` `ARCH`'s N_eff statistic was itself shown biased by `1/(1-rho)`; corrected,
`N_eff` grows **sublinearly** in `N_c` (exponent 0.85 [0.49, 1.34]).

`[J]` Ranked above the exponent questions because a locator **bias** that moves
with `N_c` contaminates R3 directly, and no amount of downstream care repairs
it. This must be certified before any exponent-grade production.

### Tier 2 — the exponents.

**R6. Physical finite-`zeta` `nu`.** (§12.3)

`[E]` Exponent-grade extraction needs at least **four** `L` values with
`L2/L1 >= 2` at *every* `zeta` in the fit, so `lambda_c` is `L`-extrapolated
**before** the `zeta`-fit. `[E]` This is the step the project has never
completed: `CB-WINDOW-001`'s scope is explicitly "wide-pair crossings, **not
L-extrapolated**".

`[E]` `nu_0 = 2` is the **zeta = 0 endpoint** exponent and is never
`nu(zeta > 0)`; quoting it as the finite-`zeta` exponent is a recorded error,
not an approximation.

**R7. Small-`zeta` boundary behaviour, assuming no previous fit.** (§12.2)

`[E]` Discriminating power over a `zeta` interval goes as
`r_c(z1)/r_c(z2) = (z1/z2)^phi`. Across one decade, `phi = 1/3` gives 2.15,
`1/2` gives 3.16, `1` gives 10. `[I]` So `1/3` versus `1/2` differ by only
**1.47 across a full decade**: a campaign spanning less than a decade in `zeta`,
or determining `r_c` to worse than ~5%, cannot separate them. `phi = 1` is easy
to exclude; the hard pair is the one that matters.

`[C]` **The trap, stated in advance:** the informative end (`zeta -> 0`) is
where `r_c` is smallest and the `L` requirement largest. The campaign is
hardest exactly where it is most informative. `[J]` A design that quietly drops
the small-`zeta` points to make the `L` budget work has removed its own
discriminating power. **If small `zeta` cannot be reached at adequate `L`, the
honest output is that `phi` is not measurable by this project — not a `phi`
fitted over the easy half of the range.**

`[J]` Ranked below R6 despite being the project's most-wanted number, because on
current evidence it is the **least likely** of these to come back determined.
A campaign justified solely by it is a campaign likely to return nothing.

### Tier 3 — endpoint structure and anchors.

**R8. Reliable Born endpoint `lambda_B`, `nu_B`.** (§12.5)

`[E]` `CUTBTHEORY` named the per-trajectory joint `(B, N_clicks)` record as the
cheapest discriminating measurement available, and asked whether existing
Born-ensemble runs contain it. `[E]` They do not, at scale: `n_real` survives in
**4.6%** of files. `[J]` The blocker is storage, not compute — and the
production entry point now stores it. Cheap to add to any pass.

**R9. Born-end tangent from joint CMI and `N_T`.** (§12.6)

`[E]` Depends on `kappa_ent`, which **has never been derived**: `TWOCLICK`
established that the two candidate constructions disagree about which property
of `Delta_12` enters, and `KAPPAENT` returned `Pursue` only for a converged-`D`
recomputation of `C_2`, explicitly **not** an endorsement that the endpoint
programme reaches the phase boundary. `[J]` Blocked on theory, not numerics.
Collect the data (R8); do not design a campaign pass around it.

**R10. High-`zeta` structure around historically weakly constrained values.**
(§12.7)

`[E]` `zeta >= 0.5` carries **67%** of grid cost and `zeta = 0.85` alone carries
**24%**. `[E]` But `zeta = 0.55, 0.65, 0.85` have only n = 36, 36, 26 runs, and
0.85 exists only at L = 32, 48.

`[J]` **Lowest priority per core-hour, and the ranking must say so plainly.**
The prize is concentrated where the cost is not: high `zeta` is far from the
`zeta -> 0` asymptotics every candidate law describes. Keep it as a
consistency anchor at modest `L`, where `r_c` is large and `L` requirements are
mild. Do not let it consume the budget by default, which is what the current
grid does.

---

## 2. Transition locator and observables

`[J]` **Primary locator: quarter-system CMI (`OBS-CMI-001`)**, used for
crossings, local crossing drift, slopes and FSS. No completed campaign provides
stronger contradictory evidence.

`[E]` **CMI's poor conditioning in the deep no-click perturbative calculation
does not make it a poor critical-point locator.** These are different
questions and the first must not be cited against the second.

| observable | status | role |
|---|---|---|
| `OBS-CMI-001` quarter-system CMI | active | **primary locator** |
| `OBS-BLKMR-001` (product-of-averages) | active | independent **secondary** cross-check |
| `OBS-BLPROD-001` (average-of-products) | active; the `CB-WINDOW-001` claim on it was **withdrawn under review** | supporting, not primary |
| `OBS-BL-001` | **retired** — one label, two quantities | never |
| `OBS-SHALF-FINAL-001` / `-TAVG-001` | active, distinct | supporting (log-law to area-law), not a crossing estimator |
| `OBS-ACTIVITY-001` | **`needs_audit`** | not until audited |

`[E]` CMI and `OBS-BLKMR-001` must **never be averaged together**, and
`OBS-BLPROD-001` versus `OBS-BLKMR-001` is precisely the pair the project must
never compare across.

`[J]` **Do not call the historical `B_L` a Binder cumulant.** It is
`CMI * S_AB`, a Binder-*like* proxy, and the name imports properties it does
not have.

**Mandatory before any crossing is fitted.** The fit must declare a
crossing-validity rule *first* and classify every crossing against it:
internally bracketed; not at a scan endpoint; unique sign change; observable not
collapsed into its own error. `[E]` `TASK-2026-08-10-AMP096` produced a headline
amplitude from a "crossing" pinned to the last sampled lambda with both `B_L`
values collapsed to numerical zero, and nothing in the pipeline had to declare a
rule first. `[I]` This is why `ANALYSIS_SPEC.yaml` exists; it is not optional
hygiene.

**Storage schedule.** `[E]` For CMI FSS, store the **four subsystem entropies
separately** (`S_AB`, `S_BC`, `S_B`, `S_ABC`), not only the assembled CMI: CMI
is a four-term cancellation, and storing only the difference makes it impossible
to diagnose later whether a drift came from one term or from the cancellation.
Store the full lambda scan **including non-crossing cells** — storing only cells
that produced a crossing is how an endpoint artifact becomes invisible. Store
the per-trajectory `(B, N_clicks)` joint record for R8.

`[I]` The production entry point (`pps_qj.production.run`) already satisfies
this schedule and records the `OBS-*` convention IDs alongside the numbers, so
an `OBS-BLPROD` value can never later be read as `OBS-BLKMR`.

---

## 3. Campaign architecture — adaptive, five passes

`[J]` The passes exist so that a hypothesised global law can never reach the
sampling design. **The global fit happens LAST.**

### PASS 0 — method, time and genealogy certification

Answers R1 and R5. Mostly re-analysis and small pilots.

- The zero-compute `T/L` comparison on existing Cut A `caseA_guided` data.
- The `tau_int` pilot at two or three `L` at one `zeta` near the expected
  boundary, converting `T(L)` from a heuristic margin into a measurement.
- An `N_c` ladder measuring the **correct** statistic — `1 + (N-1)rho` with
  `rho` estimated directly, or the Lee–Whiteley single-run genealogical
  estimator (`Biometrika` 105(3) 609–625, arXiv:1509.00394) — at >= 2 system
  sizes, `R >= 40`, on the `zeta >= 0.5` cells that carry the cost, and
  settling whether the finite-`N_c` locator shift is real.

`[J]` **Pass 0 gates everything.** Its outcomes change `T` by a factor of two
and may change the `N_c`/realisation split at fixed cost.

### PASS 1 — broad, unbiased lambda bracket

Answers R3, coarsely.

- A **fixed lambda grid, uniform, the same at every `zeta`**, over a wide
  interval, at the two smallest `L`.
- **No `zeta`-dependent centring at this stage.** `[J]` That is the step that
  would smuggle in a law, and it is the single design requirement most likely to
  be violated by accident.
- Bracket from the data: find the coarse interval containing the sign change of
  the pair difference. **If there is no unique sign change, record that and do
  not refine.** An ambiguous bracket is a result.

`[E]` **Do NOT let a hypothesised global `lambda_c(zeta)` law determine Pass-1
sampling.** This is exactly how the current grid came to miss its own crossing
at `zeta = 0.5`.

### PASS 2 — local CMI crossing refinement

- Refine **only inside the bracket Pass 1 found**, at every `zeta` by the same
  rule: same number of points, same relative width.
- A `zeta`-dependent refinement *width* is acceptable. A `zeta`-dependent
  refinement *centre taken from a formula* is not.
- Classify every crossing against the pre-declared validity rule.

### PASS 3 — selected exponent-grade large-`L` slices

Answers R6, and supplies R7 if the small-`zeta` end is reachable.

- Only at `zeta` values Pass 2 bracketed cleanly.
- `>= 4` values of `L` with `L2/L1 >= 2` at every `zeta` in the fit.
- `lambda_c` is `L`-extrapolated at fixed `zeta` **first**.
- `[C]` The frozen-edge criterion (`L >> 2/r_c^2`, `T >> 1/(w·r_c^2)`) means the
  `L` requirement grows sharply as `r_c` falls; whether the small-`zeta` end is
  affordable is a Pass-0/Pass-2 output, not an assumption.

### PASS 4 — global boundary analysis, only after measurement is complete

Answers R7 and R4 jointly.

- Fit `phi` on the `L`-extrapolated `lambda_c(zeta)` over **>= 3 windows**, and
  report the drift. `[E]` If windows disagree, the answer is that `phi` is not
  determined — `CB-WINDOW-001` already says this is the expected outcome on
  current-quality data, and it must not be re-discovered as a surprise.
- Report in **both** parameterizations, `lambda` and `r = lambda/(1-lambda)`.
  `[E]` Fitted exponents drift with the window in both. Agreement is a weak
  check; disagreement is informative.
- Run the R4 discriminators on the same data: crossing drift plotted against
  `1/ln(L)` **and** against `L^(-1/nu)` (power-law FSS straightens the second,
  KT-like the first); `nu_eff` from successive `L` pairs (a genuine power law
  plateaus, a log-corrected one drifts monotonically); collapse quality with and
  without a log correction, reported as residuals rather than as a picture.
- `[C]` Two registered artifact traps must be checked before any small-`zeta`
  flattening is read as a finite intercept: `THEORYPRED`'s `P-13`, and the OBC
  edge-mode caveat. Both predict an artifact that *mimics* a finite intercept.

`[J]` **Pre-register Passes 1–4 in a frozen `ANALYSIS_SPEC.yaml` before the
campaign runs.** "The FSS form is not determined at these sizes" must be a
pre-registered acceptable outcome.

---

## 4. Unresolved method / time / genealogy certifications

Open at the time of writing. Each is a precondition for the pass named.

| # | open question | status | gates |
|---|---|---|---|
| C1 | `T(L)` — is `T/L = 1` adequate? | `METH-TREQ-001` **unsupported**; zero-compute test never run; `tau_int` pilot owed since 2026-06-17 | Pass 0 → all |
| C2 | finite-`N_c` locator **bias** | z = 3.4 shift seen in `ARCH`'s own unreported data; N_eff statistic was itself biased | Pass 0 → Pass 3 |
| C3 | genealogical collapse | `n_distinct_ancestors` → 1 of 350 while ESS/N_c = 0.98 | Pass 0 |
| C4 | `newton` waiting-time solver | statistical change (~1e-6); **no production-scale paired-seed artifact exists** | not in production baseline |
| C5 | uniformization | theoretically exact (`THEORY_D.md` D4); implementation **not validated**; historical defect still unexplained | excluded |
| C6 | the master efficiency metric | `DEC-MASTER-METRIC-001` (`t_wall × sigma^2(lambda_c)`) is **gameable** — maximised by a 2-point extremes design at 126,461x carrying 19-sigma bias; replacement needs a hard bias constraint | any efficiency claim |
| C7 | `DISP-SNAPSHOT-001` | largest *claimed* lever (3–15x), formally disputed, resolution rule stated, second cell never run | optional |
| C8 | `VR-CLOSE-001` / `DEC-KILLS-001` tension | the kill "N_eff ~ N_c always" rests on evidence that **never measured N_eff** and had no power to detect the effect | Pass 0 |

`[J]` C8 is recorded as a live tension, not resolved here. `DEC-KILLS-001`
remains in force as canonical state; that its cited evidence is inadequate is a
finding for a proposal, not a licence to reopen the direction unilaterally.

---

## 5. What must arrive before the grid is frozen

`[J]` The grid is deliberately **not** specified in this document. Freezing it
now would mean estimating allocation from guesses, which the task forbids and
the project's record punishes.

Required first:

1. **The manually collected Ruche inventory.** What exists on the cluster, which
   campaigns are actually present, how much is provenance-complete, and what
   storage is available. Procedure: `RUCHE_MANUAL_INSTRUCTIONS.md`.
2. **Benchmark outputs from the production commit, measured on Ruche.**
   `[E]` `ALGRD`'s ~4,560 core-hour projection is an **Apple M3 + Accelerate**
   figure. Its own transferability section says the sub-cubic scaling
   (`A(L) ∝ L^1.95`, `B(L) ∝ L^2.4`) is a small-matrix-efficiency artifact and a
   server CPU will differ, "most likely closer to L³". The recorded Mac→cluster
   ratio of ~2.4x was measured at **one point** and must not be trusted per-`L`.
   `[I]` Any allocation request built on the local number is built on sand.
3. **Pass 0's outcomes** — `T(L)` and the finite-`N_c` certification.
4. `[E]` A re-centred, audited `lambda_c(zeta)` from **L-extrapolated** crossings
   on existing stored data, with per-`zeta` uncertainty — `MAPS`'s
   highest-expected-value next calculation, and **zero new simulation**.

`[E]` Structural results that **should** transfer from the local work, and can
be planned against now: a jump costs 6–12 windows; `zeta >= 0.5` carries ~67% of
grid cost; the waiting-time solve is the dominant kernel at large `zeta` while
the running entropy dominates at small `zeta`; realisation-level parallelism is
embarrassingly parallel with no communication; `lowrank` removes an O(L³) `eigh`
per jump.

`[E]` Results that must **not** be assumed to transfer: the absolute core-hour
projection; the `A(L)`/`B(L)` exponents; and the two architecture-scoped kills
(the R2 restructuring and banded `K@Q`), both of which lost specifically to
Accelerate's unusually strong small-matrix `zgemm` and should be re-run once
under MKL/OpenBLAS before being treated as closed.

---

## 6. The smallest plausible pilot, once that information arrives

`[J]` Deliberately small. Its purpose is **not** a number — it is to check that
the campaign's assumptions survive contact with the machine.

**P0 — zero compute, do it first.** The `T/L` comparison on existing Cut A
`caseA_guided` data (`T/L` = 2.000 / 1.333 / 1.000 at L = 32–128 within one
campaign), plus the re-centring calculation of §5.4. No simulation at all.

**P1 — the `tau_int` pilot.** Integrated autocorrelation time of CMI at two or
three `L` at one `zeta` near the expected boundary.

**P2 — one `L`-ladder at one small `zeta`.** A single `zeta = 0.10`,
`L in {128, 192, 256}`, full lambda scan, full storage schedule. It checks that
(i) the crossing is internally bracketed and not endpoint-pinned at the smallest
`zeta` the campaign wants, (ii) the frozen-edge criterion is actually satisfied
at the `L` reached, and (iii) the storage schedule captures what CMI FSS and the
Born-end analysis need.

`[J]` If P2 cannot produce a clean bracket at `zeta = 0.10`, **re-scope the
campaign to `zeta >= 0.15` before submission**, rather than discovering it
afterwards.

`[E]` For scale, `ARCH` costed a comparable two-arm genealogy experiment at
~80 one-core tasks, ~10 core-hours, no inter-task communication. `[C]` P0–P2 are
of that order, not of the 4,560 core-hour order. `[J]` The decision to commit
Ruche time should be taken **after** P0 and P1, because both can change the
required `T` by a factor of two and therefore the entire budget.

---

## 7. The honest case for the campaign

`[J]` The strongest version is **not** "it will measure `phi`".

It is that R1–R6 and R8 are answerable, cheap-to-moderate, and would leave the
project with a **reproducible, provenance-complete dataset it does not currently
have anywhere**. `[E]` Across the 15,139-run historical corpus, `git_commit`,
`seed`, `burn_in` and `job_id` are absent from **every file**; `seed` is
recoverable in 5.3% and `n_real` in 4.6%. `[I]` The project cannot currently
answer "which version of the code produced this" for any of it.

`[J]` `phi` remains a declared **hope**, not the deliverable.

---

**Status: planning document only.** No simulation run. No HPC or Ruche access.
No job array prepared, submitted or modified. The next human decisions are
(a) whether to authorise pilots P0 and P1, which need no campaign at all, and
(b) whether to open a dedicated numerical-mapping task once the Ruche inventory
and production-commit benchmarks exist.
