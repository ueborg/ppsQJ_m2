# LAMBDA_GRID_DECISION — TASK-2026-09-06-NC-ZETA-MOCKPROD

Brief §5: *inspect existing descriptive data; verify these are reasonably broad;
widen them if there is a serious risk that the CMI crossing lies outside; do NOT
narrow them merely to reduce cost.* Labels `[E]` `[I]` `[C]` `[J]`.

**Verdict: the three proposed nine-point grids are adopted UNCHANGED.** They are
strictly wider than the independently re-derived law-agnostic bracket rule at
every `zeta`, on both sides.

---

## 1. What the grids may not be justified by

`[E]` Not `sqrt(zeta)`, not `phi = 1`, not any critical-line law. `DISP-PHI-001`
is **open** and this task moves it in neither direction. The grids are
**coverage brackets**: their job is to contain the crossing under either open
position plus margin, and to make it visible when they do not.

`[E]` This is the exact failure that stopped `TASK-2026-09-05-NC-ZETA-CALIBRATION`
at Gate A. Its stencils inherited the shape of the historical `0.51*sqrt(zeta)`
grid, and its own preflight check `P9` gave false assurance because it tested a
stencil **centre** against a law rather than testing whether the **span**
brackets the alternative. `[E]` This task's `P9` tests the span, against both
positions, symmetrically, and negative control `N8` demonstrates it failing on a
narrowed grid.

## 2. The anchor, and its caveat

`[E]` Both laws are anchored on the one `zeta = 0.35` crossing in this programme
that passes an interiority test: `lambda_x = 0.23691`, the `L48-L64` sign change
on the 17-point grid at `N_c = 1024`, `R = 24`, outcome class `INTERIOR`
(`TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION`, section D).

`[E]` The `L32-L48` crossing at `0.2005` is **not** used: its outcome class is
`STILL_BOUNDARY` and it is endpoint-induced. `[E]` The `L32-L64` crossing
(`0.23145`, `INTERIOR`) would move both predictions down by 2 %, which changes
no conclusion below.

`[E]` **That number is a locator, not `lambda_c(0.35)`**, and it is not
`L`-extrapolated. Both predictions inherit the caveat. `[J]` This is acceptable
here because the grids are being sized, not measured against.

## 3. Coverage of both open positions

`lambda_pred(zeta) = 0.23691 * (zeta/0.35)^phi`, `tau_lambda = 0.004` used as a
yardstick only.

| `zeta` | `phi=1/2` | `phi=1` | adopted grid | step | pts below/above `phi=1/2` | pts below/above `phi=1` | worst margin to a grid **END** | `phi` bracketed |
|---:|---:|---:|---|---:|:--:|:--:|---:|---|
| 0.10 | 0.1266 | 0.0677 | [0.040, 0.160] | 0.0150 | 6 / 3 | 2 / 7 | **6.9 tau** | [0.31, 1.42] |
| 0.20 | 0.1791 | 0.1354 | [0.080, 0.260] | 0.0225 | 5 / 4 | 3 / 6 | **13.8 tau** | [-0.17, 1.94] |
| 0.70 | 0.3350 | 0.4738 | [0.250, 0.520] | 0.0340 | 3 / 6 | 7 / 2 | **11.5 tau** | [0.08, 1.13] |

`[E]` All six predictions are strictly interior with at least two grid points
beyond each, and the worst margin to a grid end is 6.9 `tau_lambda` — well above
the 4 `tau` floor and above the 2.6 `tau` finite-`N_c` crossing drift measured at
`zeta = 0.35` between `N_c = 512` and `2048`.

`[E]` **Margin to a grid end, not to a grid point.** The investigator's first
pass reported `zeta = 0.10, phi = 1/2` as "0.83 tau from grid point 0.130".
`[I]` That is a different quantity and it is not a coverage risk: proximity to a
grid *point* is desirable — it means the crossing lands where the scan is dense.
Coverage risk is proximity to a grid *end*, which is 8.3 `tau` there.

## 4. Independent re-derivation of the bracket rule

`[E]` `TASK-2026-09-06-NC-ZETA-STAGE1` derived a law-agnostic bracket rule —
span both predictions plus `max(25 % of the law span, 4*tau_lambda)` each side,
snapped outward to a 0.005 grid — and justified the `4*tau = 0.016` floor from
the measured low-rung drift. That rule is **re-derived here from the same two
canonical claims**, not copied, and applied to all three `zeta` including
`zeta = 0.70`, which `STAGE1` does not cover:

| `zeta` | law span | margin | rule bracket | **adopted grid** | wider below? | wider above? |
|---:|---:|---:|---|---|:--:|:--:|
| 0.10 | 0.0589 | 0.0160 | [0.050, 0.145] | **[0.040, 0.160]** | yes | yes |
| 0.20 | 0.0437 | 0.0160 | [0.115, 0.200] | **[0.080, 0.260]** | yes | yes |
| 0.70 | 0.1388 | 0.0347 | [0.300, 0.510] | **[0.250, 0.520]** | yes | yes |

`[E]` **Every adopted grid is strictly wider than the rule bracket on both
sides, at all three `zeta`.** No widening is required and none is applied.

`[E]` `STAGE1` is used here **only as a cross-check on a rule this task
re-derived**, and it is not modified — `tools/check_predecessor.py`, preflight
`P15`.

## 5. Descriptive context from the historical scans

`[E]` The historical corpus is centred at `0.51*sqrt(zeta)`: 0.1613, 0.2281 and
0.4267 at `zeta = 0.10, 0.20, 0.70`. `[E]` All three lie inside the adopted
grids — the first at the very top (0.1613 against an end of 0.160, i.e.
marginally outside by 0.0013), the other two comfortably interior.

`[I]` That the `zeta = 0.10` historical centre sits at the grid's upper end is
**not** an argument to widen. The historical line is a `sqrt(zeta)` construction
and this task may not centre on it; and the measured `zeta = 0.35` crossing is
itself at `0.40*sqrt(zeta)`, i.e. 22 % **below** the historical `0.51` centre.
`[I]` Taking the historical line at face value would push the grid the wrong way.

`[E]` This is descriptive context only. It is **not evidence for a `sqrt(zeta)`
law** and nothing in the design is centred on it.

## 6. The residual exposure, stated rather than hidden

`[E]` The `zeta = 0.70` grid brackets `phi` in `[0.08, 1.13]`. It is the
narrowest window of the three, and `[I]` the asymmetry is structural, not an
oversight: `zeta = 0.70` is an **extrapolation above** the 0.35 anchor, and a
fixed multiplicative grid width compresses the `phi` range it can resolve when
extrapolating up, while interpolating down (0.10, 0.20) expands it.

`[C]` **If the true boundary exponent at `zeta = 0.70` exceeds 1.13, the crossing
lies above this grid and this task will not locate it.** That is 13 % beyond the
larger of the two open positions.

`[J]` Three reasons this is accepted rather than bought off:

1. `[E]` No canonical claim proposes `phi > 1`. `DISP-PHI-001` has exactly two
   positions, 1/2 and 1, and the grid contains both with 11.5 `tau` to spare.
2. `[E]` The exposure is **detected, not silently absorbed**: the pre-registered
   outcome classes in `ANALYSIS_SPEC.yaml` include `ABOVE_GRID` and
   `BELOW_GRID`, and a crossing in an end interval is reported
   `ENDPOINT_INDUCED` and is never interpolated as though interior.
3. `[J]` `zeta = 0.70` is where compute is most expensive — 74 % of the committed
   campaign. Extending its grid upward buys margin against a law nobody has
   proposed, at the highest price per point in the design. `[J]` The curves would
   still be measured and the `N_c` question — which is what this task is for —
   would still be answered; only the crossing column would read `ABOVE_GRID`.

`[E]` **No grid was narrowed.** Cost entered this decision only in §6.3, and only
to decline an extension beyond both open positions, never to trim inside them.
