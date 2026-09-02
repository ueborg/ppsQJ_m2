# The production path is unchanged

TASK-2026-09-02-MOCK-PRODUCTION. The brief says, in §2:

> Use exactly the same guided-cloning production algorithm and observable
> definitions as the successfully validated HIGHRUNG-LAMBDA task.
> Do not change the sampler.

## The evidence, not the assertion

**[E]** The sampler this campaign runs is `support/instrumented.py` at

```
sha256 0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d
bytes  11396
```

which is **the same hash, byte for byte**, as:

| task | file | sha256 |
|---|---|---|
| `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA` | `support/instrumented.py` | `0a33c403…` |
| `TASK-2026-09-01-SMCRUCHE-READY` | `support/instrumented.py` | `0a33c403…` |
| `TASK-2026-08-30-SMCSTAT` | `analysis/instrumented.py` (untracked origin) | `0a33c403…` |

`TASK-2026-08-30-SMCSTAT` validated that file **bitwise** against the production
`pps_qj` path. Nothing in this task rewrote, reformatted or re-implemented it;
it was copied with `shutil.copy2` and the hash was verified afterwards by
re-reading it from disk.

## What that identity licenses

**[I]** It is the reason the 288 ARM-B populations can be reused as if they were
this campaign's own. They are not "comparable" runs of a "similar" sampler; they
are runs of *this file*, at the same `(L, T, zeta, lambda, N_c, dtau_mult,
resample_scheme)`, with disjoint seeds. Pooling them with this campaign's
`L = 64, N_c = 1024` cells is a bookkeeping operation, not a scientific
judgement.

Had the hash differed by one byte, the reuse in `REUSE_AND_DEDUP_AUDIT.md` §1
would have to be withdrawn and those three grid points recomputed.

## What is enforced, and where

| enforcement | where | when |
|---|---|---|
| the bundled file matches its recorded sha256 | `run_cell.py`, `_man` block | at the start of **every array task**, before any physics runs |
| the same check, before submission | `shared/preflight.py`, `runtime_checks` | in the preflight, per arm |
| `dtau_mult = 6.0` on every manifest row | `shared/preflight.py`, `design_checks` | preflight, hard failure |
| `resample_scheme = systematic` on every row | same | preflight, hard failure |
| `zeta = 0.35` on every row | same | preflight, hard failure |
| `N_c` in `{128, 1024, 2048}` | same | preflight, hard failure |
| `T == L` on every row | same | preflight, hard failure |

`run_cell.py` itself is **copied verbatim** from
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/shared/run_cell.py` — the same file that
produced the ARM-B data. Its call into the sampler is unchanged:

```python
r = I.run_instrumented(L=..., T=..., N_c=..., zeta=..., lam=...,
                       dtau_mult=..., seed=..., resample_scheme=...,
                       record_anc=True)
```

and the observable it records is unchanged: `cmi_weighted_mean` is the
final-weight-weighted mean of `r.obs["CMI"]` over the finite clones, exactly as
before. `shared/run_preflight.sh` and `shared/analyse_results.sh` are also
byte-identical to the predecessor's.

`shared/analyse_arm.py` is verbatim **in its executable body** — lines 20
onwards `diff` clean against the predecessor's — and differs only in its
docstring and two closing `print` strings, which pointed at the predecessor's
`combined_analysis.py`. Leaving them would have printed a wrong file path to the
human at the end of every arm's analysis. The statistics it computes are
untouched.

## What this task DID change, and why none of it touches the sampler

Three files differ from the predecessor's, and all three are packaging:

| file | change | touches physics? |
|---|---|---|
| `shared/preflight.py` | rate model keyed on `(L, N_c)`; 13-point grid instead of a 3-point stencil; `N_c` and `T == L` checks added; seed block moved | **no** — it validates manifests and never runs the sampler |
| `tools/cost_model.py` | new rates, all anchored on measured Ruche wall times | **no** — cost estimation only |
| `tools/build_arms.py` | this campaign's arms | **no** — writes manifests |

`analysis/mock_production_analysis.py` is new, and is analysis of returned
numbers. It does not run the sampler and cannot.

## Not done here

- No change to `pps_qj`.
- No change to `support/instrumented.py`.
- No change to the observable definition, the CMI definition, the `dtau`
  convention, the proposal dynamics, the compensator, the resampling algorithm
  or the Gaussian evolution.
- No predecessor task archive was modified.
- `research/state/**` was not written.

If a change to any of the above ever appears necessary, `analysis_spec.yaml`
requires stopping and raising it as a separate scientific/software change. It is
out of scope here, and a sampling-budget task is the wrong place for it.
