# FIELD_MAP — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Charter Stage 2. Nodes and relations for the finite-`N_c` calibration problem.
Not a bibliography. `[E]` `[I]` `[C]` `[J]` throughout.

`[J]` **Scope statement, stated once and honoured.** This is a campaign-
preparation task and its field is *this programme's* sampler, corpus and cost
structure. The external-literature half of a Stage 2 map was **not** performed
here — see `NOVELTY_MATRIX.md` §"What was not searched". The relevant external
sweep was done by `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` and terminated
negatively; repeating it would be the "three agents independently rediscovering
the same fact" failure the Skill names.

---

## 1. Nodes

### Methods and software
| node | kind | status |
|---|---|---|
| `pps_qj` production path, guided proposal `proposal_c = zeta`, exact RN compensator, systematic resampling | method | certified |
| `support/instrumented.py`, sha256 `0a33c403…` | software | certified, validated bitwise against production by `TASK-2026-08-30-SMCSTAT`; **byte-identical** across every reuse in this campaign |
| `OBS-CMI-001` | observable | active |
| `OBS-BL-001` | observable | **retired** — one label over two quantities; not used here |
| `analysis/anchor_scan.py` | software | **known wrong** (`EV-CODE-ANCHORSCAN-001`, kernel drops `w`); excluded |
| discretisation `K = ceil(2 lambda (L-1) T / dtau_mult)` | method | exact; `dtau_mult = 6` certified |

### Assumptions the campaign depends on
| node | kind | status |
|---|---|---|
| the Feynman–Kac weight is exact at any window size, so the target measure is invariant under `dtau_mult` | assumption | `[E]` established; it is what makes campaign E clean |
| uncertainty comes from independent populations only | convention | standing programme rule |
| `T = L` | convention | `METH-TREQ-001` is `unsupported`; used here only because the entire reuse corpus is at `T = L` |
| cost is linear in `N_c` and in `K` | assumption | `[E]` confirmed at `L = 64` (rate flat 1024→2048); `[E]` **violated upward** at `L = 128`, rate `~ N_c^0.187` |

### Negative results and open bottlenecks — the load-bearing nodes
| node | kind | status |
|---|---|---|
| uniform-mixing Feynman–Kac bounds do not transfer: the no-click branch is deterministic | negative result | `[E]` CONTROLLED. `M^m(x,·) <= beta M^m(x',·)` fails for every finite `beta`, every `m` |
| no controlled analytic `N_c_req` rule exists | open bottleneck | `[E]` derived, bounded and predicted: none |
| `I_N = I_inf + B/N` rejected at `L = 128` | negative result | `[E]` `chi2 = 12.58`/3, `p = 0.0056`, reproduced here from raw files |
| the same rejection at `L = 96` | **disputed** | `[E]` does not reproduce from the `lambda = 0.3032` raw ladder (`p = 0.168`, 1 dof) |
| no diagnostic is stable out of sample across `eps ∈ {0.03, 0.05, 0.10}` | negative result | `[E]` from the predecessor |
| `L` and `ln K` are collinear to `r = 0.99` in the corpus | bottleneck | `[E]` campaign E is the only axis that breaks it |
| the across-clone spread of the ACCUMULATED log weight is recorded in 0 % of production-geometry runs | instrumentation gap | `[E]` **closed by this task** (`INSTRUMENTATION.md`) |
| `git_commit` absent from 100 % of the 3 784-record corpus | instrumentation gap | `[E]` **closed for new runs** by this task |
| the `--mem` model was never measured | instrumentation gap | `[E]` **closed by this task**, and it was under-conservative |

## 2. Relations

```
uniform-mixing FK theory  --[ASSUMES]-->  minorised mutation kernel
production kernel         --[CONTRADICTS]-->  that assumption   (deterministic no-click branch)
        |
        +--[FORCES]--> empirical calibration            (this task exists because of this edge)

I_N = I_inf + B/N   --[EMPIRICALLY REFUTED AT]--> L = 128
                    --[NOT REPRODUCIBLE AT]-->    L = 96, lambda = 0.3032   (F3)
                    --[UNTESTED AT]-->            L = 64   (only two rungs exist)
                              |
                              +--[BLOCKS]--> quoting any exponent or extrapolating I_inf

finite-Nc drift  --[IS L-DEPENDENT]-->  cross-L differences do NOT fully cancel
                 --[THEREFORE MIMICS]-->  finite-size scaling
                              |
                              +--[THREATENS]--> every downstream quantity: lambda_c(zeta,L), nu, phi, the FSS form
                              +--[IS PARTLY CANCELLED IN]--> D = I_L1 - I_L2   <-- C3, load-bearing, untested

N_c  --[CONTROLS]-->  finite-particle drift and within-population variance
R    --[CONTROLS]-->  uncertainty of the finite-Nc population MEAN
N_c  --[DOES NOT CONTROL]-->  R's job         }  conflated in the corpus's own
R    --[DOES NOT CONTROL]-->  N_c's job       }  design history (R runs 96..16 across rungs)

dtau_mult --[MOVES]--> K, exactly
          --[DOES NOT MOVE]--> the target measure     <-- what makes campaign E clean
          --[IS NOT]--> a physical parameter

SMCCERT per-cell calibrated B   --[REMAINS THE PRODUCTION RULE]-->  unchanged by the predecessor
                                --[THIS TASK MEASURES]-->            whether B exists at all at each L
```

## 3. Where this campaign sits in the dependency chain

`[I]` Upstream of everything. `lambda_c(zeta, L)` is a functional of curves
measured at some `N_c`; `nu`, `phi` and the FSS form are functionals of
`lambda_c(zeta, L)`. `[E]` The six live disputes all live downstream of that
chain, which is why none of them is moved here and why closing any of them on
`N_c = 1024` data would be premature in a way no amount of `R` fixes.

## 4. What this map does NOT contain

`[E]` No node for an external empirical protocol for plateau detection in
interacting-particle Monte Carlo. `[J]` One may well exist and this task did not
look, which is recorded rather than glossed: absence of a phrase is not novelty,
and absence of a search is not absence of a phrase either.
