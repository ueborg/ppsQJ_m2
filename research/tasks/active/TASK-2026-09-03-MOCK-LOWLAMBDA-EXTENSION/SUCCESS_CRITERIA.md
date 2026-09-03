# Success criteria — FROZEN

Frozen before any new population exists. Machine-readable form:
`analysis_spec.yaml`. Evaluated in exactly one place:
`analysis/lowlambda_analysis.py`. The preflight prints the spec's sha256 in
every arm; if it stops matching `INPUTS_LEDGER.md`, the spec has been edited and
the run should stop.

Three verdicts throughout: **SUPPORTED**, **INCONCLUSIVE**, **KILLED**, plus
**NOT EVALUATED** for a criterion whose data has not arrived. *Data has not
arrived* and *data arrived and cannot be used* are different states and are
never conflated — that distinction is what stops an unsubmitted arm from
reading as a scientific verdict.

---

## 1. Standing rules (inherited unchanged; violating one voids the analysis)

- Every uncertainty is **across independent populations**. Within-clone spread
  is a `VIF`/`N_eff` diagnostic and is never a standard error.
- Every primary statistic is at **matched `R = 24`** per `(L, lambda)` cell.
  Cells with more are cut into disjoint blocks of 24 **in seed order**,
  observable-blind; block A is primary. Full-`R` views are secondary and carry
  no curve-quality, join or crossing authority.
- **No smoothing. No interpolation replacing a measured point. No value-based
  exclusion. No imposed monotonicity. No lambda point removed. No special fit
  across the join. No second extension of the grid.**
- The weighted polynomial is a **yardstick**: a comparator for how much of the
  point-to-point structure the error bars already explain. It is never plotted
  in place of the measured points and never replaces a measured value.
- CMI is a **locator**. Nothing here is `lambda_c(zeta)`.

---

## 2. Curve quality on the 17-point grid

Computed at every `L`, on the full grid, unsmoothed:

- mean and across-population SEM at every lambda;
- `VIF = Var_across · N_c / mean within-clone variance` (diagnostic only);
- `N_eff = mean within-clone variance / Var_across` (diagnostic only);
- adjacent increments `d_i = m_{i+1} − m_i` and `SEM(d_i)`;
- standardized increments `r_i = |d_i| / SEM(d_i)`;
- second finite differences `q_i = m_{i+1} − 2m_i + m_{i−1}` and `SEM(q_i)`;
- roughness `= mean over interior i of (q_i / SEM(q_i))²`, with a
  10 000-resample bootstrap 95 % CI;
- split-half stability per cell (fixed permutation, identical at every cell);
- leave-one-out stability per cell;
- maximum standardized population outlier per cell, against the two-sided 1 %
  expected-maximum threshold for `R` draws;
- weighted quadratic `chi²/dof` — **yardstick only**.

The 13 old points are additionally recomputed **on their own** from the frozen
snapshot and reported beside the 17-point figures, so that any drift in the
reuse would be visible rather than absorbed.

---

## 3. The join tests — J1, J2, J3

The join is at `lambda = 0.2332`, between grid index 3 (new) and 4 (reused).

### J1 — local roughness at the join

Exactly **two** second-difference triples straddle the join, i.e. contain both a
newly measured and an already-measured lambda: the triples centred at `0.2232`
and at `0.2332`. Let `z_i = q_i / SEM(q_i)` and

```
T_join = max(|z| over those two triples)
M*     = max(|z| over ALL 15 interior triples of a bootstrap resample)
p_join = fraction of 10 000 resamples with M* >= T_join
```

- **PASS** `p_join >= 0.05` — the join is no rougher than this curve's own
  expected worst roughness.
- **FAIL** `p_join < 0.05`.

The null is the curve's own bootstrap maximum, not zero, so J1 cannot fail
merely because the curve is noisy everywhere.

The triple centred at `0.2132` lies entirely on the new side and tests the new
points' internal smoothness, not the join. It is reported with the rest of the
curve and is not part of J1.

### J2 — out-of-sample extrapolation

Fit a weighted quadratic to the **five lowest already-measured points only**
(`0.2332`–`0.2732`). Extrapolate to each of the four new lambdas. Compare
measured to predicted in units of the combined uncertainty, propagating the
fit's own prediction variance from the weighted least-squares covariance and
adding the measurement SEM in quadrature. The prediction variance is inflated by
`max(1, chi²/dof)` of the fit, which is conservative.

- **PASS** all four `|z| <= 3`.
- **FAIL** any `|z| > 3`.

**The fit never sees a new point.** This is the one genuinely out-of-sample
check in the task. A FAIL licenses saying the join is not smooth. It does not
license dropping a point.

### J3 — no step across the join

Take the increment straddling the join, `d(0.2232 → 0.2332)`. Fit a weighted
line to the three increments on each side (six in total, the join increment
excluded) and predict at the join.

- **PASS** `|residual| <= 3 · SEM_total`, where `SEM_total` combines the fit's
  prediction variance with `SEM(d)` at the join.
- **FAIL** otherwise.

A rung is **CONTINUOUS** only if J1, J2 and J3 all pass.

---

## 4. The crossing protocol and the interiority test

Run for `L32–L48`, `L32–L64`, `L48–L64`, with
`D(lambda) = I_{L_large} − I_{L_small}` and 10 000 bootstrap resamples of
independent populations.

Reported for every pair, whatever the outcome: raw sign changes; resolved sign
changes (both bracketing `|D| >= 2·SEM`); crossing lambda(s) by linear
interpolation; the **full** bootstrap crossing-count histogram (never
discarded); the bootstrap 95 % interval; the fraction with exactly one crossing;
endpoint-induced status; stability to deleting one lambda point (jackknife over
all 17); and split-half crossing stability over two disjoint `R = 12` halves in
seed order.

### Interiority — the pre-registered target

```
I1   the raw crossing is NOT in the first or the last interval
     (interval index i satisfies 1 <= i <= 14)

I2   it SURVIVES DELETING THE FIRST LAMBDA POINT (0.1932): the entire
     crossing machinery is re-run on the 16-point grid without 0.1932, and
     a raw crossing must survive at a location inside the full grid's
     bootstrap 95 % interval

I3   the bootstrap 95 % interval lies inside
     (0.1932 + delta_lambda/2, 0.3532 - delta_lambda/2)
```

**I2 is the criterion the brief pre-registered in words** — *an interior
crossing must not depend on the first lambda point* — and it is implemented
literally rather than by proxy.

Four outcome classes, and only four:

| class | condition |
|---|---|
| `INTERIOR` | a raw crossing exists **and** I1 **and** I2 **and** I3 |
| `STILL_BOUNDARY` | a raw crossing exists but fails I1, I2 or I3 |
| `BELOW_GRID` | no raw crossing on 17 points, **and** the bootstrap 95 % upper limit is at or below `0.1932 + 2·delta_lambda` |
| `NONE` | no raw crossing and no such lower-end accumulation |

`STILL_BOUNDARY` is reachable **only** when a raw crossing exists.

---

## 5. The criteria — X1 to X7

### X1 — the twelve new cells are individually sound

Split-half, maximum-outlier and leave-one-out at each of the 12 new
`(L, lambda)` cells.

- **SUPPORTED** all three pass at all 12.
- **KILLED** any leave-one-out failure, or ≥ 2 split-half failures.
- **INCONCLUSIVE** isolated failures otherwise.

### X2 — the extended curve is still statistically smooth

- **SUPPORTED** `2 <= median r <= 20` at every `L`, and ≥ 12 of the 16
  increments have `r >= 2` at every `L`.
- **KILLED** `median r < 2` at any `L` — the curve is not resolved point to
  point.
- **INCONCLUSIVE** otherwise.

*(The predecessor's threshold was ≥ 9 of 12 increments; 12 of 16 is the same
three-quarters fraction on the longer grid.)*

### X3 — the join is continuous

- **SUPPORTED** J1, J2 and J3 all pass at all three `L`.
- **KILLED** ≥ 2 of the 3 rungs are not CONTINUOUS.
- **INCONCLUSIVE** exactly one rung fails a join test.

### X4 — interiority. **This is the point of the task.**

- **SUPPORTED** all three pairs classify `INTERIOR`.
- **KILLED** no pair classifies `INTERIOR`. Extending the scan did not bracket
  the locator: it remains boundary-driven or has moved below `0.1932`. **This
  is a reportable negative result and does not license extending the grid
  again.**
- **INCONCLUSIVE** one or two pairs interior.

### X5 — any crossing that exists is reproducible

Bootstrap stability (interval width `<= 2·delta_lambda`), stability to deleting
one lambda point, and split-half crossing stability.

- **SUPPORTED** all three hold for all three pairs.
- **NOT EVALUATED** no pair has a raw crossing to reproduce.
- **INCONCLUSIVE** otherwise.

### X6 — the standing prohibitions held

The analysis writes an audit block into its own results file:
`smoothing_applied`, `value_based_exclusions`, `lambda_points_removed`,
`special_join_fit`, `grid_extended_again`.

- **SUPPORTED** all false / zero.
- **KILLED** any otherwise.

### X7 — the campaign was as cheap as it claimed

Measured core-hours from the returned `wall_s` against the predicted 61.31
(85.83 pessimistic).

- **SUPPORTED** measured `<= 85.83` core-hours.
- **INCONCLUSIVE** above it.
- **NOT EVALUATED** the arms have not all returned.

---

## 6. Scope lock on any conclusion

A curve-quality statement may be made **only** in this form:

> For `zeta = 0.35`, `L <= 64` and `N_c = 1024`, the measured unsmoothed CMI
> curves are statistically smooth over `lambda = 0.1932–0.3532`.

bound to `zeta = 0.35`, `L <= 64`, `N_c = 1024` and this guided-cloning
configuration, and only if X1, X2 and X3 support it.

It **may not** imply:

- that `N_c = 1024` is adequate at `L = 96` or `L = 128`;
- that `N_c = 1024` is adequate at lower `zeta`;
- that a global `N_c(L, zeta)` law exists;
- that any crossing found here is the thermodynamic critical point, a
  finite-size estimate of one, or a point on a phase boundary;
- anything about `lambda_c(zeta)` or any exponent.
