# The lambda grid — FROZEN before any new result exists

TASK-2026-09-02-MOCK-PRODUCTION, brief §3.

## Decision

```
13 points, delta_lambda = 0.010, IDENTICAL at L = 32, 48 and 64:

  0.2332  0.2432  0.2532  0.2632  0.2732  0.2832  0.2932
  0.3032  0.3132  0.3232  0.3332  0.3432  0.3532
```

The three already-computed ARM-B lambdas `0.2932 / 0.3032 / 0.3132` are grid
indices 6, 7 and 8 — contiguous, and inside the grid rather than at an edge.

`shared/preflight.py` refuses any manifest whose lambdas are off this grid, so
the choice cannot drift after the fact.

## What the existing data were used for, and what they were not

The endpoints come from the historical Cut-B corpus at `zeta = 0.35`, frozen
into `frozen_inputs/historical_corpus_zeta035.csv` (1200 rows, sha256
`f7598a121c99506021634fb4a6024ab28fb06550e82cb5332b22b666719e811a`), a slice of
`/Users/catlover1337/Downloads/pps_all_realizations.csv` (sha256
`7066bac7…`).

It was used **only** to locate where `CMI(lambda)` varies with `L` — i.e. where
the cross-`L` curves are ordered one way, where they are ordered the other way,
and roughly where they change over. That is a bracketing question, and it is the
whole of what was taken from it.

It was **not** used, and must not be cited, as evidence for a critical law, a
phase-boundary shape, `lambda_c(zeta)` or any exponent. **[E]** The corpus is
`dtau_mult = 12.0`, `N_c = 128`, `R = 12` throughout; this campaign is the
certified `dtau_mult = 6.0`. **[I]** Its absolute CMI values are therefore not
this campaign's, and no endpoint below depends on them being so — only on the
*sign* of the cross-`L` difference, which is a far weaker thing to borrow.

**The brief's constraint 2 is satisfied structurally**: no `sqrt(zeta)` law, no
presumed `lambda_c`, and no critical form of any kind enters the derivation.
The grid is centred on measured behaviour, not on a theory.

## The measurement that fixed the endpoints

**[E]** The corpus at `zeta = 0.35` holds `L in {64, 80, 96, 112, 128}` and
nothing smaller. There is **no L = 32 and no L = 48 anywhere in it**, at any
`N_c`. So the bracket for the three sizes this campaign actually runs cannot be
read off directly and had to be inferred from the smallest pairs available.

Weighted quadratic fits of `CMI(lambda)` over the 17 corpus points with
`lambda in [0.22, 0.37]`, weighted by each cell's own across-realization SEM:

| L | I(0.30) | dI/dlambda | d2I/dlambda2 | chi2/dof |
|---:|---:|---:|---:|---:|
| 64  | 0.4333 | −5.784 | +25.0 | 0.98 |
| 80  | 0.4182 | −6.194 | +23.2 | 1.04 |
| 96  | 0.4091 | −6.645 | +27.1 | 0.67 |
| 112 | 0.3735 | −7.346 | +48.8 | 1.74 |
| 128 | 0.3862 | −7.887 | +48.0 | 0.85 |

Pairwise crossings of those fits, restricted to the fitted window:

| pair | crossing |
|---|---|
| 64 vs 80 | 0.2534 |
| 64 vs 96 | 0.2737 |
| 64 vs 112 | 0.2729 |
| 64 vs 128 | 0.2814 |
| 80 vs 96 | 0.2825 |
| 80 vs 112 | 0.2750 |
| 80 vs 128 | 0.2846 |
| 96 vs 112 | 0.2725 |
| 96 vs 128 | 0.2852 |
| 112 vs 128 | 0.3228 |

**[E]** The cross-`L` crossing band at `zeta = 0.35` is therefore roughly
`lambda in [0.253, 0.285]`, with the 112-vs-128 pair an outlier at 0.323.

## Why these endpoints

**Lower endpoint 0.2332.** **[E]** At that lambda the corpus curves are ordered
strongly and unambiguously by `L` in the *increasing* direction: interpolating,
`L = 64` gives ≈ 0.92 and `L = 128` gives ≈ 1.21, a 30 % separation that dwarfs
the per-point SEMs. **[I]** Any plausible crossing for the `L = 32/48/64`
triple lies above a lambda at which the larger-`L` curve is 30 % above the
smaller-`L` one, so 0.2332 is comfortably on the far side of it. That leaves two
grid points of margin below the lowest measured crossing (0.2534) and a further
one below that.

That margin is deliberately on the *low* side, because the corpus's own
smallest-`L` pair — 64 vs 80, the closest available analogue to a small-`L`
pair — gives the **lowest** crossing of all of them (0.2534). **[C]** If the
crossing drifts further down at smaller `L`, the drift is toward this endpoint,
so the risk is bracketed on the side it is most likely to appear.

**Upper endpoint 0.3532.** **[E]** There the ordering has cleanly reversed:
`L = 64` gives ≈ 0.196 and `L = 128` ≈ 0.094, a factor of two the other way.
**[I]** A grid that starts where the larger `L` is decisively above and ends
where it is decisively below brackets a sign change of the cross-`L` difference
at both ends, which is exactly what the crossing protocol in
`analysis_spec.yaml` needs in order to say anything about uniqueness or
endpoint-induction.

**Identical across L.** Required by the brief and required by the analysis: the
crossing protocol differences `D(lambda) = I_{L2} − I_{L1}` are only defined
point-by-point on a shared grid, and interpolating one curve onto another's
lambdas would be exactly the "interpolation treated as measured evidence" the
brief forbids.

**Containing the ARM-B points.** 0.2932, 0.3032 and 0.3132 are grid points, so
the 288 existing populations are reused rather than recomputed
(`REUSE_AND_DEDUP_AUDIT.md`). Placing them at indices 6–8 rather than at the
centre was a consequence of choosing the endpoints first; it is not a defect.

## 13 points rather than 11 — the wave arithmetic, stated honestly

The brief prefers 13 **if it does not add a scheduler wave against an 11-point
design**. At the campaign's long pole, `mockL64`:

| design | new lambdas at L=64 | tasks | ceil(tasks/64) | core-h | elapsed at %64 |
|---|---:|---:|---:|---:|---:|
| 11-point | 8 | 192 | **3** | 100.4 | 1.80 h |
| **13-point** | **10** | **240** | **4** | **129.3** | **2.32 h** |

**By the strict wave count, 13 points does add a wave.** That is stated plainly
because the arithmetic is what the brief asked for.

It is nonetheless adopted, for three reasons:

1. **The wave count is not how this array actually schedules.** **[E]** Task
   duration varies by 51 % across the grid (`n_steps` runs 314 to 475), so
   Slurm backfills continuously rather than advancing in waves. The
   throughput-based model — calibrated against ARM B's completed 288-task array,
   which finished in 2.76 h against a 2.84 h prediction, a 3 % error — puts the
   13-point arm at **2.32 h**, inside the brief's §8 requirement of ≈3 h. The
   wave count overstates the cost of the two extra points by about 10 minutes.
2. **The two extra points are the bracket.** They are 0.2332 and 0.3532, the
   two endpoints the whole argument above rests on. An 11-point grid of
   0.2432–0.3432 keeps the crossing band but loses the low-side margin that the
   64-vs-80 evidence says is where the risk is, and it weakens the
   endpoint-induction test at both ends.
3. **The extra cost is 28.9 core-hours**, 7.6 % of the campaign.

**If the human prefers strict wave parity**, dropping to 11 points is a one-line
change (`GRID = [...][1:12]` in `tools/build_arms.py`) followed by a rebuild and
a re-run of every preflight. It costs the two endpoint points and nothing else.

## The risk this grid does not cover, pre-registered

**[C]** If the `L = 32/48/64` crossing lies **below** 0.2332, the crossing
protocol will report either no crossing or an endpoint-induced one. That is a
legitimate INCONCLUSIVE for M3 and it must be reported as such rather than
repaired by extending the grid after the fact. `analysis_spec.yaml`
`stopping_criteria` forbids adding a lambda point in response to a result; the
remedy is a child task with its own frozen grid.
