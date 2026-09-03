# INSTRUMENTATION — what this campaign records that no predecessor did

`shared/run_cell.py` was changed. This file states exactly what changed, what
did **not**, and why the change is safe.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## 1. What did NOT change

`[E]` The sampler, its arguments, the RNG seeding, the discretisation, the
resampling scheme, the observable, and the order in which anything is called.
`support/instrumented.py` is **byte-identical** (sha256 `0a33c403…`) to the file
that produced every reused population, and `run_cell.py` refuses to start if the
bundle hash does not match.

`[E]` The call is the same single line with the same keywords:

```python
r = I.run_instrumented(**kw, record_anc=True)
```

`[E]` **Every field the predecessor wrote is still written, under the same key,
with the same definition.** Every existing analysis script reads these files
unchanged.

`[E]` **Demonstrated, not asserted.** `tools/reproduce_check.py` re-executes
completed predecessor populations through the modified wrapper. Result: all
1 024 per-clone CMI values **bit-identical**, every integer diagnostic exactly
equal, on both tested cells. See `VALIDATION.md` §3.

## 2. What is added, and why

`TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` found two instrumentation gaps that
blocked its own conclusions. Both are closed here, at **zero cost to the
simulation**: adding output fields cannot perturb a trajectory.

### 2a. The accumulated log-weight spread — the gap that mattered

`[E]` That task's largest unresolved bridge was *what actually drives the
`L`-growth of finite-`N_c` drift*. Every candidate was damaged except one: the
across-clone spread of the **accumulated** log weight. `[E]` It is **recorded in
0 % of production-geometry runs.**

`[E]` This campaign records `final_weights` — the normalised cumulative
importance weight vector at `t = T`. Since `log(final_weights) = log_carry -
logsumexp(log_carry)`, the variance of the accumulated log weight is recovered
**exactly**, not by proxy:

```
logw_carry_var_final = Var(log final_weights)
```

`[E]` The smoke test recomputes it independently from `final_weights` and
requires agreement.

`[J]` **Why not modify the sampler to record `Var(log_carry)` directly?** Because
`support/instrumented.py` is the byte-identical certified file that produced
every reused population, and changing it would void the reuse ledger for a
quantity that is exactly recoverable from an output vector. The constraint
turned out not to cost anything.

### 2b. `git_commit`

`[E]` Absent from **100 %** of the 3 784-record corpus, so no existing run can be
tied to the code version that produced it. Recorded here, with a `-dirty` suffix
when the working tree differs, and `unavailable` when the package is unpacked
from a tarball with no `.git`. `[E]` Never fatal.

### 2c. The per-window histories

`[E]` The sampler already computes these and the predecessor threw all of them
away, keeping only the final value. Now persisted in full:

`hist_ess`, `hist_ess_cum`, `hist_logw_var`, `hist_w_max`, `hist_dLambda_mean`,
`hist_dLambda_var`, `hist_n_jumps_mean`, `hist_n_distinct_anc`, `hist_gess`,
`hist_max_family_frac`, `hist_resampled`.

`[E]` These are `O(K)` arrays, **not** `O(K x N_c)`. The largest file this
campaign writes is about 230 kB and the total new output is a few tens of MB.

`[E]` Also persisted: `delta_tau`, `K`, `n_resampling_events`, `resample_mode`,
`sampler_sha256`.

`[E]` `delta_tau` as recorded is the **actual** step `T/K`, not the nominal
`dtau_mult/(2 lambda (L-1))` — the `ceil` in `n_steps` rounds it down. The smoke
test asserts this, because recording the nominal value would trap anyone
reconstructing the window schedule from a result file.

## 3. What this buys, concretely

| gap the predecessor named | closed here | what it now enables |
|---|---|---|
| accumulated log-weight spread in 0 % of runs | `logw_carry_var_final` + `final_weights` | the one surviving candidate for the `L`-growth can be tested against `L`, `N_c` and `K` for the first time |
| `git_commit` absent from 100 % | recorded | every new population is tied to its code version |
| per-window weight history discarded | 11 arrays | `Var(log w)` accumulation can be watched **through** the run rather than inferred from its endpoint — which is precisely what campaign E's E1/E2 distinction is about |
| `n_resampling_events`, `delta_tau` not persisted | recorded | the window schedule is reconstructable from a result file alone |

`[J]` Campaign E is the arm this matters most for: if E returns E2
(discretisation-stable), attention moves to the accumulated-weight route, and
these are the only runs in the programme that will carry the data to follow it.

## 4. What is still NOT recorded

`[E]` The full ancestor matrix (`K x N_c`) and the per-window resampling index
maps. They are computed in memory (`record_anc=True`) and discarded. `[E]` At
`N_c = 8192`, `K = 408` that is 27 MB per population per array, ~2.6 GB across
campaign A alone, for a genealogy analysis this campaign does not perform.
`[E]` `mrca_mean` and the genealogy diagnostics of that task's Design 3 are
therefore **not** obtainable from this campaign's output, and Design 3 remains
unrun. Stated so nobody later assumes otherwise.
