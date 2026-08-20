# The production QJ-PPS algorithm

**Frozen by `TASK-2026-08-20-PRODUCTION-READY`.**

This document names the one implementation that counts as production, states
what is certified and on what evidence, and — equally load-bearing — states what
is deliberately excluded and why.

Marks: `[E]` evidence · `[I]` inference · `[J]` judgment.

---

## 1. The production entry point

```
python -m pps_qj.production.run --config configs/production/<cell>.yaml
```

Implemented in `pps_qj/production/`:

| module | role |
|---|---|
| `config.py` | `ProductionConfig` — the explicit, validated parameter surface |
| `provenance.py` | provenance capture and the observable-definition registry |
| `run.py` | the driver and CLI |

It is a **thin wrapper**, not a reimplementation. The sampler is
`pps_qj.cloning.run_cloning`, unchanged; the CMI/B_L reduction is
`pps_qj.parallel.worker_clone_pps._batched_compute_B_L`, unchanged. The only
edit to sampler code was additive genealogy bookkeeping (§6).

### It is configured by file and command line only

`[E]` `TASK-2026-08-11-ALGRD` §2 found **three drivers disagreeing about what
"production" meant** — `submit_clone_guided_prod.sh`, `run_local_boundary.py`
and the recorded campaign logs differed on `dtau_mult` (6 vs 12 vs 12), jump
method, solver, entropy stride and `T`. `[I]` That was possible because
configuration lived in `PPS_*` environment variables, where a submit script and
its driver can silently disagree.

`[J]` The production entry point therefore **does not read `PPS_*` environment
variables at all**. A stale `PPS_SOLVER=newton` in the environment cannot change
a production result. This is enforced by a test
(`test_entry_point_does_not_read_pps_env_vars`).

The legacy grid worker `pps_qj.parallel.worker_clone_pps` still exists and still
reads the environment. It is **not** the production entry point.

---

## 2. The algorithm

Guided Feynman–Kac cloning / sequential Monte Carlo for the partial
post-selection quantum-jump target measure on the Cut B monitored Kitaev chain.

| component | production value |
|---|---|
| family | guided Feynman–Kac cloning / SMC, fixed population |
| target measure | QJ partial-post-selection tilted path measure, click weight `zeta^n` |
| proposal | reduced-rate (guided), intensity `c*lambda` with `c = zeta` |
| compensator | **exact** Radon–Nikodym weight `exp[-(1 - zeta) * dLambda]` per window |
| resampling | systematic, fixed population, weights normalised first, every window |
| waiting-time solver | `brentq` |
| jump update | low-rank active-subspace, `refresh_every = 100` |
| entropy stride | 4 |
| window | `delta_tau = 6 / (2*alpha*(L-1))` (`dtau_mult = 6`) |
| burn-in | `n_burnin_frac = 0.25`, time-averaged diagnostics only |
| parallelism | across realisations; single-threaded BLAS inside |

**Parameter convention (Cut B).** `alpha + w = 1` and
`lambda = alpha/(alpha + w)`, hence `alpha = lam`, `w = 1 - lam`. Identical to
`grid_pps._alpha_w_from_lam`. `alpha` and `w` are derived, never set directly.

`[E]` `DEC-KILLS-001` records "Cloning (SMC) is the production algorithm".
`[E]` `TASK-2026-08-11-MAPS` recommends **stopping the broad algorithm search**:
no fundamentally different method beats, or plausibly beats, the optimised
guided approach. Level III is closed for the locators (B_L and CMI are entropic
functionals of the *quenched* ensemble, requiring the replica limit of annealed
objects), and no finite moment closure of the normalised ensemble exists at any
order.

---

## 3. Certified optimisations — and the evidence for each

### Guided reduced-rate proposal + exact RN compensator — **IN**

`[E]` Validated exact against the standard sampler (`B_L` 0.66 sigma at the
crossing), with ESS ~0.98 versus ~0.37 and `B_L` ~4.5x tighter.

`[E]` Re-verified locally this task (gate D, `test_production_entry.py`): guided
and physical proposals agree on CMI at **z = 1.02** over 12 paired seeds at
L=8, zeta=0.4, and the guided arm's spread is ~2.7x smaller. A wrong
compensator would move the target measure; it does not.

`[E]` Gate D also checks the degenerate case: at `zeta = 1` the compensator
exponent `(1-zeta)*dLambda` vanishes and `theta_hat` is exactly 0.

### Low-rank projective-jump update — **IN**

`[E]` Recorded in-project as numerically identical to the `eigh` path (~1e-13
over full trajectories) at both trajectory and cloning-population level.
`[E]` `ALGRD` measured 1.88x / 2.18x / 1.89x at three production points;
~2.0x production-weighted. `[E]` `ALGRD`'s red team confirmed the arms
reproduce each other to `theta ~ 1e-15`.

`[E]` Re-verified locally (gate B): on the same seed, `theta_hat`, `S_mean`,
`n_T_mean`, CMI and the genealogy all agree to `rtol = 1e-9`.

`[J]` **This is the campaign's zero-risk win, and it was default-OFF in the
production submit script.** Enabling it is a configuration change, not new code.

`[E]` Scope limit worth carrying: the flag probes ran 6–24 windows against
production's ~751, so `refresh_every = 100` was never once reached in those
probes. The combined ratio also degrades to ~1.19x, and even 0.94x, at
jumps/window <= 0.05 — a small regression at the smallest-zeta cells, which are
under 1% of grid cost.

### Entropy stride = 4 — **IN**

`[E]` Recorded in-project as bit-exact; 8–20% of runtime, strongly
zeta-dependent (the running entropy is 19.7% of runtime at zeta=0.15 versus
8.0% at zeta=0.6).

`[E]` **The contract is provable, not merely measured.** In `cloning.py` the
running-entropy recording sits inside `if record_entropy and (_k % stride == 0)`
and reads the current state without drawing from the RNG. `[I]` So the t=T
locators must be *bitwise* identical between stride 1 and stride 4 on the same
seed, and only the time-averaged diagnostics sample fewer windows.

`[E]` Gate C asserts exactly that: bitwise equality of `theta_hat` and of the
per-clone `CMI`, `B_L`, `S_AB`, `S_BC`, `S_B`, `S_ABC` arrays — not a tolerance.

### Fixed-population systematic resampling — **IN**

`[E]` `TASK-2026-08-11-ARCH` established that the incumbent normalises weights
before resampling, so the common mode of the rate cancels exactly and systematic
resampling with equal weights is the identity permutation. `[E]` The proposed
continuous-time Fleming–Viot alternative kills at the *un-centred* rate and is
strictly worse, by a factor that **grows with system size**; its mean-shifted
variant provably buys nothing, being the `delta_tau -> 0` limit of the
incumbent.

`[E]` `ARCH` also **refuted its own premise** that the incumbent carries an
`O(delta_tau)` selection-timing bias: with the mutation sampled exactly and the
potential computed exactly — both true in this code, which integrates the hazard
rather than time-stepping it — the unnormalised particle estimate is unbiased at
every `N_c` and `delta_tau`. `[E]` Carried caveat: the cleanest form of that
theorem is for *multinomial* resampling, and this code uses *systematic*.

### `dtau_mult = 6` — **IN, as the recorded production value**

`[E]` `ALGRD` measured the chunk lever at only **~1.08x production-weighted**:
it acts on the per-window term `A(L)`, while 67% of grid cost sits at
`zeta >= 0.5` where the per-jump term dominates. `[E]` Its red team found a
pre-registered kill criterion had fired unreported at `mult = 192`
(`ess_frac_min = 0.332`, below the stated 0.5 floor; `S_mean` shifted
**-2.96 sigma**, the only p < 0.05 result in that campaign's results), and that
wall time is **non-monotone**, minimising near mult ~48 and getting worse
beyond. `[E]` Only one cell (L=64, zeta=0.05) was ever measured; the queued
zeta=0.30 arm was killed mid-flight.

`[J]` Keep 6. It is the recorded production value, the lever is
campaign-negligible, and the evidence for raising it is one cell with a fired
kill criterion.

---

## 4. Deliberately excluded

### `newton` waiting-time solver — **OPTIONAL CANDIDATE, not baseline**

`[E]` Worth +1.19–1.30x. `[E]` But it is a **statistical** change (~1e-6
perturbation of accepted waiting times), and both `ALGRD` and `MAPS` record that
its production-scale **paired-seed validation artifact does not exist anywhere in
`research/state/**`**.

`[J]` Available via `solver_method: newton`, and every run that uses it is
stamped `deviations_from_certified_baseline`. `[J]` Gate E's statistical
agreement at a tiny cell (z < 3 over 10 paired seeds) is **not** the missing
artifact and does not certify it.

`[E]` Note for anyone reading the source data: `kernel_audit.json`'s field
`newton_speedup` compares a *proposed R2 restructuring* against the existing
Newton iteration, **not** `newton` against `brentq`. It is badly named.

### Uniformization / Poisson thinning — **EXCLUDED**

`[E]` Theoretically exact (`ARCH`'s `THEORY_D.md` D4, verified derivation:
`R(x) <= R_max = 2*alpha*(L-1)` state-independent, so Lewis–Shedler thinning
applies verbatim with no root-find). `[E]` Measured 1.39–1.47x over `newton` in
microbenchmark.

`[E]` **Its implementation is NOT validated.** `ARCH`'s red team found a
variable-binding bug in the analysis harness that swapped the two arms' columns
and inverted the sign of the deviation; corrected, the uniformized sampler shows
an **excess** of clicks of +1.70/+1.42/+1.17% at zeta = 1.0/0.7/0.5. `[E]` The
"exactness" gate G1 was near-tautological (it counts Bernoulli acceptances and
compares them to the mean of the same probabilities). `[E]` The arms were not
matched on numerical hygiene — the uniformized arm re-orthonormalises at every
candidate, the reference only at jumps. `[E]` And the historical
`DO NOT SHIP` prototype's defect had the **opposite** signature, so it remains
unexplained rather than reproduced.

`[J]` Excluded until an implementation passes a validation it can actually fail.

### Approximate Doob / twist / Galerkin control — **EXCLUDED**

`[E]` `DEC-KILLS-001`: "Controlled/Doob sampler in production — methods result
only; no estimator benefit at L=64", and "Doob-Gaussian closure at intermediate
zeta — Gaussian closure errors". `[E]` `MAPS` upgraded the second from one
failed ansatz to a structural result: no finite moment closure of the normalised
ensemble exists at any order.

`[J]` Live tension, recorded and **not** resolved here: `ARCH` measured a 9–17x
reduction in `Var(log W)` from the Galerkin control that produced no estimator
gain, and separately showed that `VR-CLOSE-001`, the evidence behind the
selection-side kill, **never measured the quantity it is named for**. That is a
finding for a proposal, not a licence to reopen the direction. See
`NUMERICAL_CAMPAIGN_CHARTER.md` §4 C8.

### Common-random-number / coupled-lambda production — **CLOSED**

Remains closed. Not implemented in the production path.

### Covariance-free (orbital-only) kernel — **EXCLUDED**

`[E]` Correct, and numerically identical to 1.7e-16 at the primitive level.
`[E]` But worth only ~1.05x, and **0.96x at the cost-dominant corner**, because
the jump path still materialises a covariance for the low-rank update.
`[E]` The memory-footprint hypothesis it was built to test was **not supported**:
cutting the live population from ~230 MB to ~77 MB bought 1.07x.
`[E]` Its "bit-exact" claim was withdrawn — the paired-seed runs were 25 windows
against production's 751.

### Batched backend — **EXCLUDED**

`[E]` Statistically equivalent but not bit-exact, and unconfirmed at production
`N_c`: preliminary profiling shows 0.28x at L=32/N_c=40 and 0.97x at
L=64/N_c=40. The production path forces `backend="scalar"`, which the guided
proposal requires anyway.

### `analysis/anchor_scan.py` — **BLOCKED, KNOWN WRONG**

`[E]` `EV-CODE-ANCHORSCAN-001`: its kernel drops the hopping `w` from the
measured bond, and it produces plausible-looking output. Blocked by
`.claude/hooks/guard_research.py`. Never use it.

---

## 5. Required output metadata

Every run writes `<run_id>.npz` and `<run_id>.json` to `output_dir`. The `.npz`
embeds the full provenance record too, so a detached `.npz` remains
self-describing.

`[E]` The failure this prevents: across the 15,139-run historical corpus,
`git_commit`, `seed`, `burn_in` and `job_id` are absent from **every file**;
`seed` is recoverable in 5.3%, `n_real` in 4.6%. The corpus is not reproducible
at the run level.

Recorded, and asserted by gate H:

| group | fields |
|---|---|
| identity | `algorithm_version`, `code_version`, `output_schema_version`, `provenance_schema_version`, `entry_point`, `status`, `run_id` |
| git | `git_commit`, `git_dirty`, `git_branch`, `git_describe`, `git_dirty_paths` |
| host | `hostname`, `platform`, `cpu_count`, `python_version`, `numpy_version`, `scipy_version`, `blas_name`, thread pinning |
| scheduler | `scheduler_job_id`, `scheduler_task_id`, allow-listed `SLURM_*` |
| physics | `L`, `zeta`, `lam`, `alpha`, `w`, `T`, `N_c`, `realizations`, `seed`, `realisation_seeds` |
| discretisation | `delta_tau`, `dtau_mult`, `n_steps`, `n_burnin_frac`, `n_burnin_steps` |
| algorithm | proposal scheme, `proposal_c`, compensator convention, solver, jump update, `low_rank_enabled`, `refresh_every`, `entropy_stride`, resampling parameters |
| deviations | `deviations_from_certified_baseline` |
| observables | per-observable `obs_id`, definition, partition, log base, aggregation |
| genealogy | `ess_mean`, `gess_mean`, `gess_frac_mean`, `gess_frac_worst`, `n_distinct_ancestors_mean`/`_worst`, `max_family_size_worst`, `resampling_events_per_realisation`, per-realisation rows |
| timing | `runtime_seconds`, `cpu_time_seconds`, per-realisation walls |

**Observable conventions are pinned by canonical ID**, so a stored number can
never later be read under the wrong convention: CMI is `OBS-CMI-001`; `B_L` is
`OBS-BLPROD-001` (average-of-products, ours) and is **never** comparable with
`OBS-BLKMR-001` (product-of-averages, KMR's). `OBS-BL-001` is retired and is
deliberately absent from the registry. `OBS-ACTIVITY-001` is emitted with its
`needs_audit` status attached.

`[E]` The four CMI subsystem entropies `S_AB`, `S_BC`, `S_B`, `S_ABC` are stored
**separately**, not only the assembled four-term difference — storing only the
difference makes it impossible to diagnose later whether a drift came from one
term or from the cancellation.

`[J]` **No general environment dump is taken.** Environment capture is a strict
allow-list of scheduler and thread variables, so a credential in the environment
cannot leak into a result file. Gate H asserts this.

---

## 6. Genealogy diagnostics, and why they were added

`[E]` `ALGRD`: `n_distinct_ancestors` falls to **1 of 350** at the production
horizon while `ess_frac_min` sits at 0.983–0.984. `[E]` `ARCH`: the locator's
across-realisation variance is 2.8–16x the independent-sampling prediction while
per-step `ESS/N_c` is 0.969–0.984 **in every cell**.

`[I]` Per-step ESS is blind to genealogical collapse. Recording it alone is
recording the diagnostic that cannot see the problem.

Two additive fields were therefore added to `CloningResult`:
`ancestor_ids_final` (per-slot founder index at t=T — the full clone-family
structure) and `n_resampling_events`. From the first, the production entry point
computes the **genealogical ESS**

```
GESS = (sum_i n_i)^2 / sum_i n_i^2 = N_c^2 / sum_i n_i^2
```

where `n_i` is founder `i`'s surviving-descendant count. `GESS = N_c` when every
founder survives once; `GESS = 1` when the whole population descends from one
founder. A run with `GESS/N_c < 0.05` emits a warning into the record.

`[J]` These are diagnostics, not a fix. Whether the loss is recoverable is open
(`NUMERICAL_CAMPAIGN_CHARTER.md` R5). The point is that it is now **visible** in
every production output rather than invisible behind a healthy-looking ESS.

---

## 7. Tests

`tests/test_production_entry.py` — 19 tests, ~24 s, gates A–H:

| gate | what it establishes |
|---|---|
| A | exact Born activity anchor `k_bar = alpha*(L-1)/L` at zeta=1 (measured z = 0.97); zeta=1 does no resampling |
| B | low-rank vs `eigh` agree to `rtol = 1e-9` on the same seed, genealogy included |
| C | entropy stride 1 vs 4 **bitwise** identical on all t=T locators |
| D | guided+compensator vs physical proposal agree (z = 1.02); `theta_hat == 0` exactly at zeta=1 |
| E | `brentq` is the default and `newton` is flagged as a deviation; newton agrees statistically at a tiny cell |
| F | GESS bounds and consistency; genealogy present and sane in a real run |
| G | fixed seed is bitwise repeatable, with a different-seed negative control |
| H | provenance schema complete; no credentials in the capture; `.npz` self-describing |

Plus config-surface guards, and a test that stale `PPS_*` environment variables
cannot alter a production result.

**Run them:**

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python3 -m pytest tests/test_production_entry.py -q
```

### Repository-wide status at the freeze

`[E]` `pytest tests/ --ignore=tests/test_exact_benchmark.py` → **62 passed,
8 failed in 130 s**.

`[E]` All 8 failures are **pre-existing and unrelated**: 7 in
`tests/test_doob_wtmc.py` and 1 in `tests/test_topological.py`
(`test_backward_pass_ZT_nontrivial_at_zeta_less_than_1`), mostly
`RuntimeError: Non-finite log-denominator at t=0.0000: log_denom=-inf`.
Verified by restoring `pps_qj/cloning.py` to its `HEAD` state and re-running:
**identical 8 failures**. None touches the cloning or production path.

`[E]` `tests/test_exact_benchmark.py` **does not terminate** — it ran over 75
minutes of CPU without completing. It imports `doob_wtmc` and
`backward_pass_sector`, not `cloning`, so it is unreachable from this change,
and it is presumably a consequence of the same Doob defect. `[J]` Both are
reported here as pre-existing repository defects, outside this task's scope, and
worth a task of their own.

---

## 8. Expected usage

```bash
# resolve and inspect a config without running anything
python -m pps_qj.production.run --config configs/production/TEMPLATE.yaml --print-config

# run a cell
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pps_qj.production.run \
    --config configs/production/TEMPLATE.yaml \
    --output-dir outputs/production/run_2026-08-20

# fully from the command line
python -m pps_qj.production.run \
    --L 64 --zeta 0.30 --lam 0.2793 --T 64 --Nc 350 \
    --realizations 5 --seed 640279300 \
    --output-dir outputs/production
```

Config templates: `configs/production/TEMPLATE.yaml` (documented reference) and
`configs/production/benchmark_L64_z030.yaml` (Ruche timing benchmark).

`[J]` Always check `deviations_from_certified_baseline` in the output. An empty
list means the run is on the certified baseline. A non-empty one is not an
error, but that result is not a baseline result and must never be pooled with
baseline results without saying so.

---

## 9. Known limitations

1. `[E]` `L % 4 == 0` is required for the CMI/B_L Majorana tripartition. The
   config rejects other `L` when those observables are requested.
2. `[E]` **The finite-`N_c` locator shift is not certified.** `ARCH`'s red team
   found, in its own saved data, mean population `B_L` = 1.2015 ± 0.0763 at
   `N_c = 44` versus 0.9161 ± 0.0348 at `N_c = 350` — **z = 3.4**. Choose `N_c`
   consistently within a comparison until Pass 0 settles this.
3. `[E]` **`T(L)` is unresolved.** `METH-TREQ-001` is `epistemic_status:
   unsupported`; the `tau_int` pilot has been owed since 2026-06-17; and
   `ALGRD`'s attempt to settle it was withdrawn for lack of power.
4. `[E]` **`lambda_c = 0.51*sqrt(zeta)` ± 0.08 does not bracket at zeta = 0.5**,
   where 67% of grid cost sits. The production entry point takes `lam`
   explicitly and applies no centring law — deliberately.
5. `[E]` The efficiency figures quoted here are wall-clock and
   production-weighted, **not** `t_wall × sigma^2(lambda_c)`, the metric
   `DEC-MASTER-METRIC-001` fixes. That metric is itself gameable (`MAPS`:
   maximised by a 2-point extremes design at 126,461x carrying 19-sigma bias).
   `[J]` Every speed number in §3 is provisional in exactly the way this project
   has already been burned by.
6. `[E]` Architecture-scoped results that must be re-measured on Ruche before
   being trusted: the absolute core-hour projection, the `A(L)`/`B(L)`
   exponents, and the R2 and banded-`K@Q` kills, which lost specifically to
   Accelerate's unusually strong small-matrix `zgemm`.
7. `[J]` The genealogy diagnostics are *measurements*, not remedies.

---

## 10. Provenance of this document

Sources are the final post-red-team artifacts of `TASK-2026-08-11-ALGRD`,
`TASK-2026-08-11-ARCH`, `TASK-2026-08-11-MAPS`, `TASK-2026-08-12-LAMC` and
`TASK-2026-08-14-C2CONV`, plus `research/state/decisions/DEC-KILLS-001.yaml`.
Where a lead's finding was corrected by its own red team, the corrected reading
is used and the withdrawal is noted.

`[J]` Nothing here was merged into `research/state/**`. This is execution-plane
engineering documentation: it records which code to run and why, not a
scientific claim. Any promotion to canonical state requires a proposal, red-team
review and the human gate.
