# Falsification plan — FROZEN

Pre-specified, before any new population exists. **This file is the plan. The
outcomes go in `FALSIFICATION_RESULTS.md`, which does not exist yet and must
not be written until the data return.** They are different files on purpose.

Each entry names what would have to be observed for the claim to be wrong, and
what the honest report is when it is.

---

## Y1 — The join is not continuous

**Attack.** Four points measured on a different day, on different nodes, may not
sit on the curve they are being attached to.

**Test.** J1, J2, J3 at every `L` (`SUCCESS_CRITERIA.md` §3). J2 is fully out of
sample: the quadratic is fitted on the five lowest *already-measured* points and
never sees a new one.

**Falsified if.** ≥ 2 of the 3 rungs are not CONTINUOUS → X3 KILLED.

**Then.** Report that the seventeen points are **not** one curve, and stop. The
crossing analysis is then reported for the 13-point grid only, exactly as the
predecessor already published it, with the four new points shown but not joined.
**No point may be dropped to repair the join, and no fit may be applied across
it.**

---

## Y2 — The new points are individually unsound

**Attack.** A low-lambda cell may behave differently: fewer steps, different
genealogy collapse, a population outlier dominating a mean of 24.

**Test.** X1 — split-half, maximum standardized outlier and leave-one-out at
each of the 12 new cells; plus `VIF`/`N_eff` and the non-finite-clone accounting
reported per cell.

**Falsified if.** Any leave-one-out failure, or ≥ 2 split-half failures → X1
KILLED.

**Then.** Report which cells and why. `R = 24` at those cells is inadequate;
that is a statement about `R`, not about lambda, and it does not license
dropping the cell.

---

## Y3 — The extended curve is no longer statistically smooth

**Attack.** The curve may be smooth over `0.2332–0.3532` and rough below it.

**Test.** X2 on the full grid; the roughness statistic with its bootstrap CI;
and the 13-old-points-alone recomputation reported alongside, so that a change
in roughness can be localised to the new region rather than blamed on the grid
being longer.

**Falsified if.** `median r < 2` at any `L` → X2 KILLED.

**Then.** The permitted curve-quality statement in `SUCCESS_CRITERIA.md` §6 is
**not** made. Report the range over which smoothness does hold, without
extending it by assumption.

---

## Y4 — The crossing is still boundary-driven

**Attack.** Extending the scan may simply move the boundary, and the locator
with it.

**Test.** X4 — the interiority classification I1, I2, I3, with I2 implemented as
a literal re-run on the 16-point grid without `0.1932`.

**Falsified if.** No pair classifies `INTERIOR` → X4 KILLED.

**Then.** Report it plainly: *"extending the scan to 0.1932 did not produce an
interior crossing; the locator remains endpoint-sensitive."* See Y6.

---

## Y5 — There is no crossing on the extended grid at all

**Attack.** The differences may simply not change sign anywhere on 17 points.

**Test.** The classification's `BELOW_GRID` and `NONE` branches, and the full
bootstrap crossing-count histogram, which is reported whatever it says.

**Falsified if.** — this is itself an outcome, not a failure.

**Then.** Report `BELOW_GRID` (bootstrap mass at the lower end) or `NONE` (no
accumulation) per pair. Both are results.

---

## Y6 — **PRE-REGISTERED: no second extension**

Whatever Y4 and Y5 return, **the grid is not extended again by this task or
automatically by any successor.** A `BELOW_GRID` outcome is a reportable
negative result, not a trigger.

This is frozen here because the failure mode is obvious and seductive: extend,
find no crossing, extend again, repeat until one appears. That procedure finds a
crossing with probability approaching one regardless of whether anything is
there. The endpoint was chosen in `LAMBDA_EXTENSION_DECISION.md` §2 from the
measured differences, before the data existed, and it does not move.

If a further extension is ever warranted it needs a fresh task, a fresh
justification that does not consist of "the last one didn't work", and a human
gate.

---

## Y7 — The reuse is not faithful

**Attack.** The frozen snapshot may not be the predecessor's data.

**Test.** `tools/freeze_predecessor.py` rebuilds it from the source JSONs and
asserts 39 cells at the expected `R`; the analysis recomputes the 13 old points
on their own and reports them beside the predecessor's published values; the
preflight checks the bundled sampler is byte-identical to the one that produced
them; `tools/dedup_scan.py` D4 checks the two halves are one design.

**Falsified if.** Any recomputed old-point statistic differs from the published
one.

**Then.** Stop. Every 17-point statistic in the task is void, because the reused
half is 13 of its 17 points.

*(Already run: worst absolute deviation over all 39 means and all 39 SEMs is
**exactly 0.0** — `VALIDATION.md` §3.)*

---

## Y8 — Expected negative results, recorded in advance

Pre-registering these so that reporting them later is not a retreat:

1. **The crossing may move to `lambda <= 0.1932` and stay unbracketed.** The
   measured differences say a sign change is near, but "near" was inferred from
   a linear continuation of four to five points, which is a weak
   extrapolation. If it lands below the new boundary, this task's own
   motivation was optimistic, and that gets reported.

2. **`L32–L48` may never cross.** Its measured difference is `−0.0202` at the
   join and rising by only `+0.0057` per step. A slight flattening below the
   join would leave it negative throughout. That would mean the three pairs do
   not share a locator, which is a real and useful negative statement about
   using CMI as one at these `L`.

3. **The crossing may be interior but not reproducible.** X4 SUPPORTED with X5
   INCONCLUSIVE — a bootstrap interval wider than `2·delta_lambda`, or halves
   that disagree — would mean `R = 24` locates a crossing it cannot pin down.
   That is a statement about `R`, and it must not be repaired by pooling the
   halves and reporting the combined result as if the check had passed.

4. **The whole exercise may confirm the predecessor's own caveat.** If the
   extended curves are smooth and the crossings interior and stable, that is
   still only **locator quality at `L <= 64`**, which the programme's corpus
   floor already says is below where physics may be read off. A clean result
   here does not become a phase boundary by being clean.
