# L = 80 — REJECTED at the runtime gate. Not prepared, not packaged.

TASK-2026-09-02-MOCK-PRODUCTION, brief §7.

The brief's rule is explicit:

> Consider L=80 only if ACTUAL prior timing data or a conservative measured
> runtime model guarantees that its array can complete in approximately 3 hours
> or less. Do not include L>64 based on hopeful extrapolation. If there is no
> directly defensible timing basis, reject it. L=64 is an acceptable ceiling.

## There IS a defensible timing basis, and it rejects L = 80

This is worth stating clearly, because the brief's default on a missing basis is
rejection and it would have been easy to stop there. **The basis exists.**
`L = 80` is the one size in this programme that can be *interpolated* between
two measured Ruche points rather than extrapolated beyond them:

**[E]** Two same-`N_c = 1024` Ruche measurements now bracket it —
`L = 64` at 4.850 ms per clone-window (ARM B, 288 runs) and `L = 128` at
27.898 ms (ARM A1024, 32 runs). The exponent between them is

```
p = ln(27.898 / 4.850) / ln(128 / 64) = 2.563
rate(80) = 4.850 * (80/64)^2.563 = 8.548 ms
```

Cross-checked against the third measured Ruche point, `L = 96` at 11.510 ms
(ARM 1, `N_c = 512`): `11.510 * (80/96)^2.563 = 7.207 ms`, and against the
conservative `p = 2.0` used elsewhere in this package: `4.850 * (80/64)^2 =
7.578 ms`. The three routes bracket 7.2–8.5 ms.

So L = 80 is rejected on measurement, not on missing evidence.

## The arithmetic

`n_steps` at `L = 80, T = 80` runs 492 (λ = 0.2332) to 745 (λ = 0.3532).
A full 13-point scan at `N_c = 1024, R = 24` is 312 array tasks.

| rate used | basis | core-hours | slowest task | elapsed at %64 |
|---|---|---:|---:|---:|
| 8.550 ms | interpolation between the two measured same-`N_c` points | 469.0 | 1.81 h | **8.43 h** |
| 7.578 ms | the deliberately OPTIMISTIC `p = 2.0` route | 415.7 | 1.61 h | **7.47 h** |

**The rejection holds even on the optimistic rate.** At 7.47 h elapsed, `L = 80`
alone is **2.5× the whole campaign's 3 h budget** and would be the critical path
by a factor of 3.2 over `mockL64`. Its 416–469 core-hours would also be larger
than the entire seven-arm campaign as designed (378.8).

To fit `L = 80` into 3 h would need a concurrency cap of about %180 for that one
array — far beyond the 64 slots per array the allocation was observed to grant
on 2026-09-02, and beyond anything this package has evidence for.

## Could it be trimmed to fit?

Considered and rejected:

- **Fewer lambdas.** `L = 80` on a 5-point sub-grid (every third point) would be
  180.4 core-h and 3.24 h elapsed — still over budget, and it would break the
  brief's
  requirement that the grid be *identical across all L* so the curves can be
  compared directly, and a 5-point curve cannot feed the crossing protocol,
  which is the point of having more than one `L`.
- **Lower R.** `R = 12` at 13 lambdas is 234.5 core-h and 4.21 h elapsed — still
  over budget, and it would halve the resolution of exactly the arm that needs
  it most.
- **Lower `N_c`.** That changes the quantity being measured and defeats the
  purpose of a mock *production* run.

None of these produce an `L = 80` curve that is both affordable and comparable,
so none is prepared.

## Verdict

**L = 80 is REJECTED. No `mockL80` package exists in this task** — not as an
optional arm, not as a disabled arm, not as a commented-out entry in
`tools/build_arms.py`. `L = 64` is the ceiling, which the brief explicitly
allows.

**[J]** The next `L` rung is a separate campaign with its own budget
conversation, and this campaign's M7 criterion is designed to cost it: it
re-derives the `L`-scaling exponent from the four Ruche rates that will exist
once `L = 32, 48, 64` return, and projects the `L = 96` and `L = 128` scans from
measurements rather than from the two-point extrapolation available today. That
is the right way to decide the next rung, and it costs nothing extra.
