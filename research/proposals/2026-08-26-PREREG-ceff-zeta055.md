# PRE-REGISTRATION — c_eff decision criterion at zeta = 0.55

Written 2026-08-26 while job for `$WORKDIR/pps/omni_z055` is RUNNING and before
any of its output has been inspected. The zeta = 1.00 pilot is complete and its
numbers are quoted below as prior context; the zeta = 0.55 numbers do not exist
yet at time of writing.

## Why zeta = 1.00 could not decide this

At zeta = 1.00 every working locator already produces a clean single crossing.
LAMC's frozen gate gives a unique crossing in ~97 percent of bootstraps for CMI
there, and the direct sign-change count is exactly 1 for every L-pair. So
c_eff's 300/300 bootstrap crossing rate at zeta = 1.00 demonstrates that the
machinery works in a healthy regime. It is NOT evidence that c_eff survives the
failure mode of interest. The pilot was a harness check and is not a ranking.

## The question this run answers, and only this

> Does c_eff retain a unique, robust cross-L trend reversal in the regime where
> CMI's crossing multiplicity fails?

This tests CONDITIONING. It does not establish that c_eff is unbiased, that it
is universal, or that its crossing is the thermodynamic MIPT. Those are separate
questions and are addressed under "Not settled by this run".

## Criterion, fixed in advance

PASS requires ALL FOUR. A stable crossing of the wrong finite-size feature would
satisfy the first alone, which is why the first alone is not the criterion.

1. Bootstrap crossing probability >= 0.90.
2. The central slope profile d c_eff / d ln L has exactly ONE sign change across
   the 11 lambda points.
3. lambda_c stable under bootstrap: 68 percent CI width <= 0.04, and stable
   under modest analysis variation: refitting with the first and last lambda
   point dropped moves lambda_c by <= 0.01.
4. No edge locking: lambda_c lies strictly inside the scanned window, at least
   one full grid spacing (0.02) from either edge, and does not sit at a bound.

For reference, at zeta = 0.55 the incumbent CMI attains 4.3 percent unique
crossings under LAMC's gate and n_valid = 0 of 10 pairs.

## Decision tree, fixed in advance

    c_eff clean at 0.55                  -> develop the c_eff locator
    c_eff weaves                         -> run the coupled-lambda pilot
    c_eff weaves AND coupled-lambda fails -> mid-zeta likely inaccessible at
                                            current L and N_c; scoped result

The middle branch matters. c_eff failing on its own does NOT license the
conclusion that mid-zeta is structurally inaccessible, because the coupled-
lambda pilot attacks a different and plausible common cause: independent noise
between neighbouring lambda points. That pilot is written and unrun.

## Not settled by this run, and must not be claimed from it

**c_eff at the crossing is a BENCHMARK, not a constraint.** The pilot gave
c_eff* = 1.035 at zeta = 1.00, L = 32-64. Treating that as a value later slices
must reproduce assumes all of: that c_eff converges to a universal conformal
quantity in this monitored problem; that the whole boundary flows to the SAME
fixed point rather than merely sharing a symmetry class; that there is no line
of fixed points with continuously varying universal data; that 1.035 at L <= 64
is near its asymptotic value; and that the normalisation is identical across
slices. None is established. Record c_eff* at every slice because it is free and
potentially informative. Do not use it as a pass/fail gate.

**The observable spread is a finite-size locator spread, not a systematic on the
thermodynamic lambda_c.** The zeta = 1.00 pilot gave, on identical trajectories:

    MI_ends 0.4082 | CMI 0.4455 | B_L 0.4897 | c_eff 0.4907

a range of 0.0825 against statistical widths of 0.006-0.021. The correct
statement is that statistical error is NOT the dominant uncertainty across
finite-size locators at accessible sizes. It is NOT correct to attach an
0.05-0.08 systematic to lambda_B, because distinct observables carry distinct
irrelevant corrections lambda_O(L) = lambda_c + a_O L^-omega_O, whose amplitudes
can differ greatly at L = 32-64 while all converge to the same lambda_c. CMI
itself moves 0.4455 (L = 32,48,64) -> 0.4364 (L = 64..128), in the expected
direction.

**A structural pattern worth watching.** The four interior locators split by
KIND, not at random: entropy-magnitude observables sit high (B_L 0.4897, c_eff
0.4907, and S_AB bound-hit the UPPER edge at 0.55), while information-difference
observables sit low (CMI 0.4455, MI_ends 0.4082, and I3 bound-hit the LOWER edge
at 0.35). The near-coincidence of B_L and c_eff to 0.001 suggests they respond
to the same finite-L crossover scale. If so, c_eff is less independent of the
incumbents than "a qualitatively different finite-size signal" implies, which
weakens the case for it as a fresh line of attack. This should be checked
explicitly, not assumed either way.

## The Cut A adversarial test, and why it is not free

Cut A pins lambda_c = 1/2 exactly, and it is what exposed the first-sign-change
estimator as wrong by up to 0.078. c_eff should face the same test before being
promoted. Two obstacles, both real:

1. **It needs new compute.** c_eff requires the S(l) profile, hence the final
   covariance matrices. The Cut A per-realisation export carries S, CMI, B_L and
   S_AB only; the covariances were never written to disk. It cannot be computed
   from existing data.

2. **The existing Cut A design cannot calibrate an L-slope estimator anyway.**
   N_c varies with L (500, 400, 350, 300, 300) and T/L varies (2.00, 2.00, 2.00,
   1.33, 1.00), and no pair of L values holds both fixed. Its <CMI> vs L at
   lambda = 0.5 is non-monotone in 11 of 15 slices, with the minimum at L = 64,
   exactly where N_c stops falling and T/L starts falling. c_eff's locator IS an
   L-slope estimator, so re-running the old grid would not calibrate it.

A valid Cut A calibration therefore needs a small NEW run with N_c and T/L held
fixed across L: L in {32, 48, 64}, N_c = 128, T = L throughout, ~11 lambda
points bracketing 0.5, one or two zeta. Cheap at these sizes. Worth doing before
c_eff becomes the primary locator, not after.
