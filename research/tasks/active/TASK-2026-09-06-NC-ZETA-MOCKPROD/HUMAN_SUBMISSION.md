# HUMAN_SUBMISSION — TASK-2026-09-06-NC-ZETA-MOCKPROD

Per-arm gates. **The researcher submits; no agent does, ever**
(`research/RESOURCE_POLICY.md` §4). Labels `[E]` `[I]` `[C]` `[J]`.

---

## Committed arms — 15 arms, 6 080 populations, 2 557 array tasks, 1 160.3 core-hours

| # | arm | gate before submitting | pops | tasks | core-h | `--time` | `--mem` | partition |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | `M_z010_nc128` | preflight 20/20 | 432 | 23 | 4.0 | `00:45:00` | `1G` | cpu_med |
| 2 | `M_z010_nc256` | preflight 20/20 | 432 | 41 | 7.3 | `00:45:00` | `1G` | cpu_med |
| 3 | `M_z010_nc512` | preflight 20/20 | 432 | 65 | 12.6 | `00:45:00` | `1G` | cpu_med |
| 4 | `M_z010_nc1024` | preflight 20/20 | 432 | 110 | 25.3 | `00:45:00` | `2G` | cpu_med |
| 5 | `M_z010_nc2048` | preflight 20/20 | 432 | 213 | 50.6 | `00:45:00` | `3G` | cpu_med |
| 6 | `M_z020_nc128` | wave 1 returned and analysed | 432 | 47 | 8.7 | `00:45:00` | `1G` | cpu_med |
| 7 | `M_z020_nc256` | as above | 432 | 84 | 16.1 | `00:45:00` | `1G` | cpu_med |
| 8 | `M_z020_nc512` | as above | 432 | 124 | 28.0 | `00:45:00` | `1G` | cpu_med |
| 9 | `M_z020_nc1024` | as above | 432 | 226 | 56.0 | `01:00:00` | `2G` | cpu_med |
| 10 | `M_z020_nc2048` | as above | 432 | 298 | 111.9 | `02:00:00` | `3G` | cpu_med |
| 11 | `M_z070_nc128` | waves 1–2 returned | 432 | 237 | 67.3 | `01:00:00` | `1G` | cpu_med |
| 12 | `M_z070_nc256` | as above | 432 | 318 | 123.9 | `02:00:00` | `1G` | cpu_med |
| 13 | `M_z070_nc512` | as above | 432 | 352 | 215.4 | `03:00:00` | `1G` | cpu_med |
| 14 | `M_z070_nc1024` | as above | 432 | 408 | 430.8 | `06:00:00` | `2G` | cpu_long |
| 15 | `E_dtau_z010` | any time; **analyse only after arm 3** | 32 | 11 | 2.5 | `00:45:00` | `1G` | cpu_med |

`[J]` The gates on arms 6–14 are **advisory, not interlocks**. Waves 1 and 2
together are 323 core-hours, 16 % of the committed spend, and they answer
whether `R = 16` resolves anything at all. If every rung there comes back
`INCONCLUSIVE`, wave 3 — 837 core-hours — will do the same, and the correct
response is to raise `R`, not to spend it.

## The conditional arm — held back

| arm | pops | tasks | core-h | pessimistic | `--time` | `--mem` | partition |
|---|---:|---:|---:|---:|---|---|---|
| `conditional/M_z070_nc2048` | 432 | 432 | **861.6** | **1 206.3** | `12:00:00` | `3G` | cpu_long |

`[E]` **Holding it saves 861.6 core-hours = 42.6 % of the full design.** The
release condition is in `CONDITIONAL_SUBMISSION.md`.

## Totals

```
committed              15 arms   6 080 populations   2 557 tasks   1 160.3 core-h   (1 624.4 pessimistic)
conditional             1 arm      432 populations     432 tasks     861.6 core-h   (1 206.3 pessimistic)
full design            16 arms   6 512 populations   2 989 tasks   2 021.9 core-h
```

`[E]` Elapsed times exclude queue wait, which is expected to dominate the short
arms. The long pole among committed arms is `M_z070_nc1024` at ~9.2 h of
compute at `%64`.

## Standing conditions on every arm

- `[E]` Preflight must pass **immediately before** submission, not once at
  build time. It refits the cost-model literals from raw data on every run and
  fails if they have drifted.
- `[E]` `--time` is `>= 1.6 x` the pessimistic slowest **pack**, and the
  partition is the smallest that fits the time. `cpu_short` is never used at any
  `--time`.
- `[E]` Every job is idempotent. Requeue freely.
- `[E]` Nothing in this package can submit. Preflight `P12` checks it.
