# Should N_c = 2048 be in tonight's campaign?

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §1 Question A.

## Verdict

**Rejected for tonight. Prepared and submit-ready as
`armA2048_optional/`, to be queued tomorrow if and only if F2 comes back
SUPPORTED.**

It was not included merely because it is larger, and it was not dropped merely
because it is expensive. The brief set three conditions; it passes one, is
borderline on the second, and fails the third *as a tonight decision*.

## Condition 1 — does a single job fit an available Ruche partition safely?

**Yes.** Predicted single task 20.12 h, pessimistic 28.16 h. `cpu_long` has
MaxTime 7 days, so a 48:00:00 request has 1.7× headroom over the pessimistic
figure. Memory is 4423 MB estimated, requesting 9G. The preflight passes.

So this condition is met, and it is the only one that is.

## Condition 2 — is the expected total cost defensible?

**Borderline.** At `R = 16` it is 322 core-hours (451 pessimistic) — more than
`armA512` and about equal to `armA1024`, for a single rung at the weakest `R` in
the campaign. `R = 16` gives a per-rung SEM of 0.0186, *worse* than the
`N_c = 512` and `N_c = 1024` rungs it would be compared against, so it would be
the noisiest point on the ladder while costing as much as the best one.

Raising `R` to fix that scales linearly: `R = 32` would be 644 core-hours, more
than half the entire recommended campaign, for one rung.

## Condition 3 — does it materially help distinguish convergence from continued drift?

**No — not tonight, and this is what decides it.**

Two reasons, and the second is the stronger.

**(a) It cannot finish overnight, which is the campaign's stated constraint.**
A single task is 20.1 h predicted and 28.2 h pessimistic. There is no
concurrency trick that helps: the elapsed time of an embarrassingly parallel
array is the single-task time. Including it converts a ~10 h overnight campaign
into a ~20–28 h one, and the researcher asked explicitly for wall-clock speed.

**(b) Its value is entirely conditional on a result we will have by morning.**
The whole point of ARM A is `Delta_256->512` and `Delta_512->1024`. Two cases:

- If `Delta_512->1024` is consistent with zero inside `tau_step`, the mean has
  stabilised at 1024 and `N_c = 2048` measures a difference already known to be
  small. It buys tightening, not a decision.
- If `Delta_512->1024` is still large, then one further doubling is a poor bet
  on the observed evidence. The drift at L = 128 has run
  `−0.099` (64→128) and `−0.121` (128→256) per doubling with **no sign of
  decay** — the second step is larger than the first. A ladder that is not yet
  decaying is not one where the next rung is likely to be the last, and
  `N_c = 2048` at `R = 16` has an MDE of 0.069, which is smaller than the drift
  observed so far but not by much.

Either way, running it in parallel spends 322 core-hours *before* the question
that would justify it has been answered. Running it tomorrow costs one day and
answers the same question with the budget aimed correctly.

## What is prepared anyway

`armA2048_optional/` is a complete, preflight-passing, submit-ready arm with
fresh seeds `30500000–30500015`, `cpu_long`, `--time=48:00:00`, `--mem=9G`, 16
array tasks. Its `README.md` and `submit.slurm` both carry the NOT RECOMMENDED
TONIGHT banner. It is excluded from the recommended overnight total everywhere
in this task.

## The decision rule for tomorrow, stated now

Queue `armA2048_optional` if **F2 comes back SUPPORTED** — i.e. the L = 128 mean
is still moving materially between `N_c = 512` and `N_c = 1024`. In that case
the ladder has not turned over and one more rung is worth 322 core-hours to see
whether it does.

Do **not** queue it if F2 comes back KILLED (drift bounded inside `tau_step`),
because the question it answers has been answered.

If F2 is INCONCLUSIVE, the cheaper move is more `R` at `N_c = 1024`, not a new
rung: an inconclusive F2 means the *uncertainty* was the limit, and `R` fixes
uncertainty while a new rung does not. Note that this would require a separately
versioned analysis spec written before it runs.
