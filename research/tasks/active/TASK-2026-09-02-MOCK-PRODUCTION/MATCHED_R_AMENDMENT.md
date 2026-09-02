# The matched-R amendment — FROZEN before any new datum exists

TASK-2026-09-02-MOCK-PRODUCTION. Amendment issued by the researcher before
commit, and applied before any campaign result exists.

## The confound it removes

The campaign's headline question is whether raising `N_c` makes
`CMI(lambda)` cleaner. As originally designed, the two sides of that comparison
did not carry the same number of independent populations:

| | as designed | R |
|---|---|---:|
| `N_c = 1024`, ten L=64 lambdas and all of L=32/48 | new compute | 24 |
| `N_c = 1024`, three reused ARM-B lambdas | reused | **96** |
| `N_c = 128` matched comparator | new compute | **48** |
| `N_c = 2048` shape check | new compute | 24 |

**[I]** Every statistic that measures "cleanliness" — roughness, adjacent-
increment significance, second differences, split-half stability, crossing
counts, crossing uniqueness — is a comparison of a curve's point-to-point
scatter against its own error bars, and those error bars shrink as `1/sqrt(R)`.
A curve measured at `R = 48` has error bars 0.71× those of the same curve at
`R = 24`. So

> "the `N_c=1024` curve is cleaner than the `N_c=128` curve"

could have been read off data in which the `N_c=128` side had **twice** the
independent populations, and the sentence would have been partly about `R` and
not about `N_c` at all.

**[E] This is not hypothetical.** On the end-to-end synthetic dataset used to
exercise the analysis, M3 returns **INCONCLUSIVE** (5 raw sign changes at
`N_c=1024` against 7 at `N_c=128`) under the original unequal-R analysis, and
**KILLED** (5 against 5) once both sides are cut to `R = 24`. Two of the seven
"extra" sign changes at `N_c=128` were an artefact of it having `R = 48`. Under
the amendment the verdict is the honest one.

The same exercise shows the effect on roughness at `L = 64, N_c = 1024`:
**8.773 at matched R=24 against 18.024 at full R**, because the three reused
`R = 96` points have error bars half their neighbours', which the second
difference reads as excess structure in the *curve* rather than as a feature of
the *design*.

## The rule

**PRIMARY: every curve-quality, crossing and reproducibility statistic is
computed at a matched `R = 24` per `(L, lambda)` cell.**

Cells holding more than 24 populations are cut into consecutive disjoint blocks
of 24 **in seed order**:

```
reused ARM-B cells   R = 96  ->  blocks A B C D   (24 each)
N_c = 128 comparator R = 48  ->  blocks A B       (24 each)
everything else      R = 24  ->  block A only
```

**Block A is always primary.** It is the first 24 populations in ascending seed
order.

### Why seed order, and why that matters

The rule is **deterministic and observable-blind**: block membership is fixed by
`argsort` over the seeds alone, in `load()`, before any statistic is computed.
Permuting the CMI values within a cell cannot move a population between blocks.

That is the whole point. A "primary subset" chosen by any rule that could see
the observable would be a choice made after seeing the data, which is exactly
the failure this amendment exists to prevent. It is therefore **asserted, not
assumed**: `tools/test_matched_r.py` builds the same cell with five different
CMI assignments — including a sorted one, a reverse-sorted one, and an
adversarial one that puts +99 on the first 24 and −99 on the rest — and requires
byte-identical block membership in every case. It also requires invariance to
the order the result files happen to be read off the filesystem.

Because the seed allocation is `seed_base + 1000*grid_index + replicate_index`
(`SEED_LEDGER.md`), each cell's 24/48/96 seeds are consecutive integers, so the
blocks are contiguous seed ranges — e.g. the reused ARM-B cell at
`lambda = 0.2932` cuts as `A: 30300000–30300023`, `B: …24–47`, `C: …48–71`,
`D: …72–95`.

### A consequence worth stating

The primary analysis is now **uniformly `R = 24` at every cell of every curve**.
The per-point error bars are homoscedastic by construction, and the
heteroscedasticity caveat that the original design had to carry — ten points at
`R = 24` beside three at `R = 96` — **no longer applies to any primary
statistic**. Per-point standard errors are still used throughout; they are now
simply equal.

## The hierarchy

| tier | what | may it support a "cleaner curve" claim? |
|---|---|---|
| **PRIMARY** | `N_c=128 R=24` vs `N_c=1024 R=24`, matched | **yes — this is the authority** |
| SECONDARY | `N_c=128` full `R=48` | no; mean displacement only |
| SECONDARY | reused ARM-B full `R=96` | no; mean displacement only |
| SECONDARY | disjoint replicate blocks B/C/D | no; sensitivity only |
| SECONDARY | `R=12` historical-precision subsets | no; mimics the old corpus's precision only |

**M3, and every statement about whether increasing `N_c` makes the curves
cleaner, is decided by the matched-`R=24` analysis.** The analysis prints
`MATCHED R = 24 on both sides` inside M3's own verdict string, so the qualifier
travels with the number.

**For mean finite-`N_c` displacement, both are reported** — the matched-`R=24`
result and the highest-precision result available — because a mean is not a
cleanliness statistic and the extra precision is real. The secondary line names
its own unequal `R` in the output itself, e.g.
`SECONDARY, highest precision available (R_1024 = 96, R_2048 = 24, UNEQUAL R)`.

**Larger `R` is not smaller finite-`N_c` bias.** The analysis prints
`More R does not remove finite-N_c bias` beside the secondary `Delta_N` line.
More independent populations reduce the *uncertainty* on the displacement; they
do not reduce the displacement. `TASK-2026-08-31-SMCCERT` killed the claim that
the displacement follows a controlled `1/N_c` law, and nothing here revives it.

## Replicate blocks are sensitivity, never selection

Blocks B/C/D at the three reused cells exist to answer one question: **do the
conclusions depend materially on which block of 24 is used?** Section E2 of the
analysis reports, per block: the means and SEMs at all three reused lambdas, the
adjacent increments and second difference over the reused triple, and the effect
on the whole `L = 64` curve's median `r`, roughness, `chi2/dof` and cross-`L`
raw sign-change counts.

**[E]** On the existing ARM-B data the four blocks already show real spread —
`d(0.2932→0.3032)` runs from −0.0328 (block B) to −0.0612 (block A), about
2.3 sigma apart. That is exactly the kind of thing this section exists to
surface, and it is why the primary block is fixed in advance rather than chosen.

**Block A is primary because it is deterministic and observable-blind, not
because it is best.** Choosing among A/B/C/D after seeing which one looks
smoothest would be a value-based selection and is forbidden by
`analysis_spec.yaml`'s exclusion rules. If a conclusion turns out to depend on
the block, that dependence is the finding and it gets reported.

## What did NOT change

The amendment is an analysis rule. It touches no compute:

- the frozen 13-point lambda grid — unchanged;
- `zeta = 0.35`, `T = L`, `dtau_mult = 6.0`, systematic resampling — unchanged;
- `L = 32, 48, 64`; `N_c = 1024 / 2048 / 128`; `R = 24 / 24 / 48` — unchanged;
- every manifest, every seed, every array range, every partition — unchanged
  and re-verified byte-identical after the amendment;
- the sampler and the bundled `instrumented.py` — unchanged
  (`PRODUCTION_PATH_UNCHANGED.md`);
- `L = 80` still rejected; the three reused ARM-B cells still reused.

The `N_c = 128` comparator keeps `R = 48` rather than dropping to 24, because
its second block is a genuine independent replication and its full `R` is a
better mean; only the *primary comparison* is cut to 24. Nothing is recomputed
and nothing is discarded.

## Files changed by the amendment

| file | change |
|---|---|
| `analysis/mock_production_analysis.py` | block machinery, primary/secondary split, sections A0, E2, E3 |
| `analysis_spec.yaml` | `matched_R` block, hierarchy, M1–M5 restated on the primary |
| `SUCCESS_CRITERIA.md` | the same in prose |
| `tools/test_matched_r.py` | **new** — 30 unit checks on the block rule |
| `POWER_AND_R_DECISION.md` | matched-R note on the reused points |
| `README.md`, `DESIGN.md`, `VALIDATION.md`, `RUCHE_RUNBOOK.md`, `INPUTS_LEDGER.md` | pointers and the new spec hash |
