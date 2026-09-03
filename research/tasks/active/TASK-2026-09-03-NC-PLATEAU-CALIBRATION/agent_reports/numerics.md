# numerics first pass — data and code reconstruction

**Executed INLINE BY THE LEAD, not by an independent investigator.** This report
is filed in `agent_reports/` because that is where the phase ledger records a
first pass, and it must not be read as an independent one. `INDEPENDENCE_LEDGER.yaml`
records the same thing without softening it: **no independence was obtained in
this run**, and every conclusion below shares the lead's blind spots.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## Assignment

Reconstruct, from raw files only, every exact-compatible population in the
repository; rebuild every `N_c` ladder; re-derive the runtime and memory cost
models from measurements; and establish whether this task's modified
`run_cell.py` is exact-compatible with the populations it proposes to reuse.

## Method

`tools/reconstruct_inventory.py` walks every `research/tasks/**/results/*.json`
and reads the fields the sampler itself wrote. **No predecessor summary table,
results JSON or `COST_MODEL.md` figure was used as an input.** Exact
compatibility is decided on `status`, `resample_scheme`, `T == L`, `zeta`, and
`brentq_fallbacks`, and the sampler identity is enforced by the bundle sha256
gate in `run_cell.py`.

## Findings

**F1 `[E]` The corpus is 1 896 completed populations in 62 cells and 53 ladders,
and every one of them is exact-compatible.** No `status != ok`, no
non-systematic resampler, no `T != L`, no `brentq` fallback, anywhere.
Full listing: `EXISTING_POPULATION_INVENTORY.csv`, `EXISTING_LADDERS.md`.

**F2 `[E]` The `L = 128` `1/N` rejection reproduces exactly.** Refitting from
the raw JSONs over the five rungs `N_c = 64 … 1024` gives `chi2 = 12.58` on
3 dof, `p = 0.0056` — the figures
`TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` reported, reached independently of
its code. `[J]` This is the strongest available check that this task's fitting
machinery is correct, and it passed before any new datum existed.

**F3 `[E]` The `L = 96` `1/N` rejection does NOT reproduce.** The
production-geometry `L = 96` ladder at `lambda = 0.3032` that can be rebuilt
from raw files has three rungs (`N_c = 128, 256, 512`, `R = 32, 32, 48`) and
gives `chi2 = 1.90` on 1 dof, `p = 0.168`. `[I]` The predecessor's `L = 96`
result (`chi2 = 10.54`, 3 dof) must rest on a four-rung ladder from a different
cell or corpus slice, which this reconstruction cannot locate. `[J]` Reported as
a discrepancy to resolve, not as a refutation: a ladder this task cannot find is
not a ladder this task has shown to be wrong.

**F4 `[E]` The `N_c` direction of the runtime rate has reversed, and the
deployed cost model extrapolates the wrong way.** At `L = 128` the measured
per-clone-window rate runs 27.18, 26.81, 21.52, 23.42, 27.90 ms across
`N_c = 64 … 1024`. `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA` read the first three as
small-batch inefficiency ending by `N_c ~ 256` and extrapolated flat from there;
that predicts 21.52 ms at `N_c = 1024` against a measured 27.90 — **30 % low, in
the optimistic direction, in exactly the regime this campaign enters.** `[I]` A
log-log fit over the three rungs at `N_c >= 256` gives `rate ~ N_c^0.187`, which
`tools/cost_model.py` now applies to every upward extrapolation.

**F5 `[E]` The `--mem` model was never a measurement.**
`TASK-2026-09-01-SMCRUCHE-READY` describes "the measured 732 MB peak"; 732 MB is
exactly what `128 + 2*N_c*per_clone` returns for `L = 96, N_c = 512`, and no
`MaxRSS` from any Ruche job appears anywhere in the repository. Direct
`ru_maxrss` measurement of the bundled sampler (`tools/mem_probe.py`, 13 cells)
puts the true peak **above** that formula at seven of them — `L = 64`,
`N_c = 2048` reads 1694 MB against a predicted 1202 MB, and that arm shipped
with `--mem=2G`, i.e. 21 % headroom rather than the 70 % its own comment
claimed. `[J]` It never broke and it was closer to breaking than anyone knew.

**F6 `[E]` This task's `run_cell.py` is exact-compatible with the populations it
reuses.** Two completed predecessor populations were re-executed through the
modified wrapper: **all 1 024 per-clone CMI values bit-identical** in both
cases, every integer diagnostic exactly equal. Derived reductions differ by
`<= 1.7e-14` relative. `[I]` That residual is x86-versus-arm64 summation order
in numpy's pairwise reductions, not a code change — the trajectory is identical.
`[J]` Worth recording on its own account: **a stored AGGREGATE in this corpus is
not bit-reproducible on a different architecture even though the physics is**,
which anyone re-deriving a published mean on a different machine needs to know.
Evidence: `VALIDATION.md` §3.

**F7 `[E]` The two interior crossings sit at `lambda ~ 0.2315` (L32−L64) and
`~ 0.2369` (L48−L64)**, and the local slopes of the cross-`L` difference there
are `2.965` and `4.052` per unit `lambda`. `[I]` The smaller of the two is what
converts a CMI tolerance into a `lambda` tolerance, and it is what
`SUCCESS_CRITERIA.yaml` freezes as `dD_dlambda_min`.

**F8 `[E]` At `L = 64`, `N_c = 1024 -> 2048` gives `Delta = +0.00235 ± 0.00528`:
compatible with zero, with a 95 % interval 1.7× wider than the material
tolerance.** `[I]` The step establishes nothing about convergence; it
establishes that `R = 24` was too small to tell. `[E]` Reaching `P2` at that
step needs matched `R ~ 137`.

**F9 `[E]` Absolute-level plateau certification is unreachable at `L = 128`.**
From the measured per-population spreads, the matched `R` needed to put a
`Delta` interval inside `tau_I = 0.006` is ~2 675 at the `512 -> 1024` step and
grows down the ladder. `[I]` At `N_c = 2048` that is ~13 000 core-hours for one
`lambda`. `[J]` This is a design finding, not a measurement failure, and it is
why campaign D is labelled a screening rung with no power to certify
convergence, and why the locator route is the only affordable one at `L = 128`.

## What this pass could NOT establish

`[E]` Nothing about `N_c > 2048` anywhere, or about `N_c > 512` at `L = 96`:
those populations do not exist. `[E]` Nothing about `dtau_mult != 6` at
production geometry: the only prior comparison
(`TASK-2026-08-30-SMCSTAT/AN7_scheme_chunking.json`) is at a much smaller cell
(`K = 232/116/58`) and is not transferable. `[E]` Nothing about whether the
memory measurements transfer from macOS to the cluster's Linux.
