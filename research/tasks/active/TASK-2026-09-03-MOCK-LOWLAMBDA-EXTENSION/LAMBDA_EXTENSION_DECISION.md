# Why these four lambdas, and why exactly four

The four new points are

```
0.1932   0.2032   0.2132   0.2232
```

at `delta_lambda = 0.010`, giving the frozen 17-point grid

```
0.1932 0.2032 0.2132 0.2232 | 0.2332 0.2432 0.2532 0.2632 0.2732 0.2832
0.2932 0.3032 0.3132 0.3232 0.3332 0.3432 0.3532
                            ^ the join
```

`GRID[4:]` is **bitwise identical** to the predecessor's own 13-point grid.
`tools/build_arms.py` asserts it, and so does `analysis/lowlambda_analysis.py`
at import. A grid that agreed to four decimal places but not in floating point
would silently split every reused cell into two.

---

## 1. Why extend downward at all

Not from a critical law, and not from a fit. From the measured cross-`L`
differences on the predecessor's own grid:

| lambda | `I48 − I32` | `I64 − I32` | `I64 − I48` |
|---:|---:|---:|---:|
| **0.2332** | −0.0202 | −0.0051 | **+0.0151** |
| 0.2432 | −0.0259 | −0.0514 | −0.0255 |
| 0.2532 | −0.0342 | −0.0690 | −0.0348 |
| … | … | … | … |
| 0.3532 | −0.1086 | −0.1714 | −0.0628 |

Two facts, both measured:

1. `I64 − I48` is **positive at the first scanned point and negative at every
   other point**. Its only sign change is in the first interval, which is why
   the predecessor reported it as `endpoint_induced = True`.
2. `I48 − I32` and `I64 − I32` are negative everywhere on the grid but are
   *rising toward zero* as lambda falls, and their bootstrap crossing mass
   accumulated at `0.2333–0.2378` — jammed against the boundary.

All three differences are heading toward a sign change just below the scan. The
extension goes where the measured differences point, and nowhere else.

---

## 2. Why the lower endpoint is 0.1932

`0.1932` is `0.2332 − 4 · 0.010`, i.e. four steps of the existing spacing. The
choice is bounded from both sides:

**Not fewer than four.** The pre-registered interiority test `I2`
(`SUCCESS_CRITERIA.md` §4) requires a crossing to survive **deleting the first
lambda point**. With `n` new points, a crossing can be interior *and* survive
that deletion only if it falls at least two intervals inside the new region.
With one or two new points there is no room for the test to mean anything: any
crossing would sit against the new boundary and the task would reproduce the
predecessor's failure one grid step lower. Four is the smallest extension that
leaves the interiority test something to test.

**Not more than four.** Cost is not the binding constraint — the whole campaign
is 61 core-hours — but scope is. Each additional point moves further from where
the measured differences say the crossing is, and a longer extrapolation of the
`n_steps` cost fit besides. Four points cover `0.0400` in lambda. That is
8.7 times the width of the predecessor's boundary-hugging bootstrap mass
(`0.2378 − 0.2332 = 0.0046`), and it reaches past `lambda ≈ 0.1982`, where a
linear continuation of the measured `I48 − I32` (rising `+0.0057` per step from
`−0.0202` at the join) reaches zero. If the crossing is not inside that window,
it is not a near miss and one more point would not help.

**And there is no second extension.** `FALSIFICATION_PLAN.md` Y6 pre-registers
that a `BELOW_GRID` outcome is a reportable negative result, not a trigger.
Choosing the endpoint *before* seeing the data, and committing not to move it,
is what stops this from becoming a search for a lambda at which a crossing
appears.

---

## 3. Why `delta_lambda` stays at 0.010

Because the increments, second differences and roughness statistic are all
defined on a uniform spacing, and the join is the one place a spacing change
would be indistinguishable from real structure. A finer grid on the new side
would make `q_i` there mechanically smaller and the curve would look smoother
exactly where this task is trying to detect a discontinuity.

---

## 4. What the four points are NOT chosen to do

- They are not chosen to bracket a **predicted** `lambda_c(zeta = 0.35)`. No
  such prediction is used anywhere in this task, and none is produced.
- They are not chosen from the historical `dtau_mult = 12` corpus. That corpus
  is not poolable with this grid and is not read here at all — not even for
  sign information, which is the one use the predecessor made of it.
- They are not chosen after looking at what the new points would have to be for
  a crossing to appear. The grid is frozen in `analysis_spec.yaml`, in
  `tools/build_arms.py`, in three manifests and in three preflights that reject
  anything else, all before a single new population exists.

---

## 5. The join is a hypothesis, not an assumption

Attaching four points measured today to thirteen measured yesterday is an
assumption about reproducibility, and this task treats it as testable rather
than given:

- the sampler is byte-identical (`INPUTS_LEDGER.md`);
- every design parameter is identical (`tools/dedup_scan.py` D4);
- `R` is identical, so the error bars do not change precision across the join;
- and `J1`, `J2`, `J3` test the join itself three different ways, one of them
  fully out of sample.

If the join fails, the honest report is that the join failed. No point is
dropped and no fit is applied across it.
