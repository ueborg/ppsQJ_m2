# Numerics first pass — TASK-2026-09-06-NC-ZETA-MOCKPROD

Scratch: `research/tasks/active/TASK-2026-09-06-NC-ZETA-MOCKPROD/scratch/{scan_rates.py,agg_rates.py,lam_dep.py}`.
All numbers below are `[E]` re-derived directly from stored JSON, not quoted from any predecessor doc.

## Q1 — runtime scaling in N_c

`[E]` Scanning every `research/tasks/**/results/*.json` with `wall_s`,`n_steps`,`N_c`,`L`,`dtau_mult=6.0` (4,984 rows) and computing `rate_ms = wall_s/(N_c*n_steps)*1000`, median rate by `(L,N_c)`:

- L=32: Nc 512/1024/2048 → 1.670/1.626/1.686 ms — **flat**.
- L=48: Nc 512/1024/2048 → 3.000/3.000/3.053 ms — **flat**.
- L=64: Nc 64/256/512/1024/2048/4096/8192 → 4.673/4.780/5.200/4.932/5.050/4.604/4.753 ms — **flat to mildly non-monotone, no rising trend even out to Nc=8192**.
- L=96: Nc 128–2048 → 11.7/10.1/11.5/10.4/10.7 — flat/noisy.
- L=128: Nc 64–2048 → 27.2/26.8/21.5/23.4/27.9/22.2 — noisy, non-monotone, **not** a clean power law.

`[E]` `COST_MODEL.md` (TASK-2026-09-03-NC-PLATEAU-CALIBRATION) fit `rate ~ N_c^0.1871` on exactly the three L=128 points 21.52/23.42/27.90 at Nc=256/512/1024, which I reproduce. `[E]` That same task later completed L=64 Nc=4096 (median 4.604) and Nc=8192 (median 4.753) — both **at or below** the L=64 Nc=1024/2048 rates (4.932/5.050), i.e. the rate at L=64 does **not** rise with N_c out to 8192.
`[C→I]` The data available now **weakens** the 0.1871 law as a general N_c-scaling and **refutes** its extrapolation to L=64 specifically (the adopted cost table predicted 6.568→7.477 ms at Nc=4096/8192, measured 4.60/4.75 — the model over-predicted cost there, not under-predicted). `[I]` At L∈{32,48,64}, the honest reading of the corpus is: rate is flat in N_c within noise (±5–10%) over the whole measured range (up to 8192 at L=64), and G≈0.1871 was an L=128-specific, three-point artifact, not a corpus-wide law.
`[J]` For N_c=128,256 at L=32,48 (unmeasured): I would adopt the **flat plateau value already measured at Nc≥512** for that L (1.67–1.69 ms at L=32; 3.00–3.05 ms at L=48), not an upward extrapolation via G=0.1871. `[I]` The small-N_c penalty visible at L=64 (Nc=64→4.67, Nc=256→4.78, both *below* the Nc=512 value 5.20) argues the low-Nc small-batch penalty, where it exists at all in this L range, is small and not obviously upward — the direction assumed by the existing cost model (monotone-non-decreasing envelope) is not supported at L≤64 and could bias the mock-prod cost estimate high, which is the safe direction for `--time` but wastes queue slot count if used to size R.

## Q2 — zeta dependence, confounded by lambda

`[E]` `TASK-2026-08-11-ALGRD/results/b0_L{32,64,96,128}.json`, field `sec_per_clone_window`, at zeta=0.05/0.15/0.3/0.7, lam=0.51·√zeta exactly (verified numerically at all 16 points). `[I]` Log-log interpolating zeta=0.35 between the 0.3 and 0.7 rows and forming ρ(zeta)=rate(zeta)/rate(0.35):

| zeta | L=32 | L=64 | L=96 | L=128 |
|---|---|---|---|---|
| 0.10 | 0.449 | 0.403 | 0.365 | 0.342 |
| 0.20 | 0.639 | 0.604 | 0.592 | 0.587 |
| 0.70 | 2.326 | 2.371 | 2.193 | 2.085 |

`[I]` ρ is L-dependent (decreasing in L at low zeta); I would adopt **L=64** since it is the largest L this task actually runs and brackets 32/48. `[C]` This whole dataset uses `fit_intercept_s`/tiny `n_steps∈{4,…,80}` extrapolated timing probes at Nc≤500 — a **different measurement regime** from Q1's production rates (e.g. at L=128,zeta=0.3 it gives 19.0 ms vs Q1's production ~21–28 ms at similar lam). Treat ρ as directional, not exact.

`[E]` Pure-lambda dependence at fixed zeta=0.35 from the Q1 corpus (Ruche production, lam 0.19–0.35): at L=64,Nc=1024, rate falls 5.55→4.03 ms from lam=0.1932→0.3532 (log-log slope ≈ −0.53); at L=32,Nc=1024, 1.76→1.32 ms, slope ≈ −0.48. `[I]` So rate ~ lam^(−0.5) at fixed zeta, roughly.
`[I]` Correcting ρ for the mismatch between the ALGRD matched line (lam=0.51√zeta) and this task's actual grid centers: at zeta=0.10 the task's minimum lam=0.040 vs matched 0.1613 (ratio 0.248) implies a rate **~2.0× higher** than the naive ρ(0.10) above — a large correction, and `[C]` an extrapolation of the −0.5 power law far below its calibrated range (0.19–0.35), so treat it as a plausible direction, not a number to build a `--time` budget on without a real probe. At zeta=0.70, lam=0.520 vs matched 0.4267 (ratio 1.22) implies only ~9% **lower** rate than naive ρ(0.70) — a minor, in-range correction.

## Q3 — lambda grid coverage vs DISP-PHI-001

`[E]` `research/state/disputes/DISP-PHI-001.yaml`, `CB-PHI-HALF-001` (phi=0.5), `CB-PHI-LINEAR-001` (phi=1.0), both `epistemic_status: provisional`, `contested: true` — dispute stays open, not adjudicated here.
`[E]` Best available zeta=0.35 anchor: `TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/LOWLAMBDA_ANALYSIS.txt` section D — only the **L48–L64** (0.23691) and L32–L64 (0.23145) crossings pass `OUTCOME CLASS: INTERIOR`; L32–L48 (0.2005) is `STILL_BOUNDARY` (endpoint-induced, invalid). `[I]` I anchor on 0.237 (L48–L64), the pair most relevant to this task's own L set. `[E]` That file explicitly states this is a **locator**, not lambda_c(zeta=0.35) or an L-extrapolated estimate — so both law predictions below inherit that caveat and are **not L-extrapolated**.

Predictions (lambda_c(zeta)=0.237·(zeta/0.35)^phi):

| zeta | HALF (phi=.5) | LINEAR (phi=1) |
|---|---|---|
| 0.10 | 0.127 | 0.068 |
| 0.20 | 0.179 | 0.135 |
| 0.70 | 0.335 | 0.474 |

`[E]` All six predictions are strictly interior to the proposed grids. Margins (tau_lambda=0.004): zeta=0.10 HALF sits only 0.83·tau from grid point 0.130 (3 pts above, 6 below); LINEAR sits 0.57·tau from 0.070 (7 above, 2 below). zeta=0.20: HALF 2.3·tau from 0.170 (4 above/5 below); LINEAR 2.6·tau from 0.125 (6 above/3 below). zeta=0.70: HALF 3.95·tau from 0.351 (6 above/3 below); LINEAR 3.0·tau from 0.486 (2 above/6 below).

`[I]` Phi-range each grid actually resolves (edge-to-edge, same anchor): zeta=0.10 → phi∈[0.31,1.42]; zeta=0.20 → phi∈[−0.17,1.94]; zeta=0.70 → phi∈[0.08,1.13]. `[I]` No grid needs widening to keep both candidate laws interior **today**, but zeta=0.70's window is narrowest (width 1.06 in phi vs 2.1 at zeta=0.20) and LINEAR's prediction (phi=1.0) sits closest to that grid's own edge (1.13) of any cell — `[I]` the asymmetry stated in the prompt is real: extrapolating **above** the 0.35 anchor compresses the phi-range a fixed lambda grid can distinguish, while interpolating below it (0.10, 0.20) expands it. `[J]` If either law's constant-of-proportionality carries meaningful uncertainty (not stated anywhere I found), zeta=0.70 is the cell most likely to need widening upward later, never narrowed.

## Q4 — existing data at zeta=0.10/0.20/0.70

`[E]` Repo-wide scan of every `results/*.json` `zeta` field: only zeta∈{0.05 (1 file), 0.3 (7 files), 0.35 (5,176 files), 0.7 (1 file)} appear; **zero** stored raw populations at zeta=0.10 or 0.20 anywhere in `research/tasks/**`.
`[E]` `results/boundary_aggregate.csv` (`EV-DATA-BOUNDARYCSV-001`) **does** carry zeta=0.1, 0.2, 0.7 rows at L∈{64,80,96,…} (nreal=12), but the file has **no N_c, T, dtau_mult, or resampling-scheme column** — reproducibility is `unknown_recoverable` for that metadata — and its lambda grid (e.g. zeta=0.1: 0.0949…0.253) does **not** coincide with this task's proposed grid (0.040…0.160) at any point. `[I]` Not poolable with a production cell even setting the metadata gap aside. The single zeta=0.7 raw JSON (`TASK-2026-08-11-ARCH/results/neff_L32_z070.json`, N_c=500, R=24) is `L=32` only, not `L=48/64`, and its lam=0.4267 (matched-line value) is off the task's zeta=0.70 grid.
`[I]` Conclusion: **no exact-compatible stored population exists** at any of the three zeta values for any proposed cell; `boundary_aggregate.csv` is directional/legacy context only, artifact_only reproducibility.

## Contradiction found

`[E]` The adopted `COST_MODEL.md` cost table for L=64 (rows Nc=4096→6.568 ms, Nc=8192→7.477 ms) is contradicted by that same task's own later-completed measurements (4.604, 4.753 ms) — a ~40% overestimate, in the safe direction for `--time` but evidence the G=0.1871 exponent should not be carried into this task's L∈{32,48,64} without re-deriving from data in that L range.

## Single biggest risk in the lead's design

`[J]` Using Q2's ρ(zeta) uncorrected for the lambda mismatch would understate zeta=0.10 runtime by roughly 2×, because the task's lambda grid there (0.040–0.160) sits far below the zeta-lambda line the only zeta-resolved timing data was measured on, and the −0.5 lambda-rate slope used for the correction is extrapolated ~4× below its own calibrated range.

## Recommended next check

`[I]` A single short local timing probe at zeta=0.10, L=64, N_c=512, lam=0.040 (smallest, riskiest cell) would directly test whether the ×2.0 correction is real or an artifact of extrapolating the lambda-rate power law outside its calibrated window — this is a design decision for the lead/Gate A, not something to run here.

`confidence_note`: none — sonnet tier was adequate for this read/reproduce task; no inference step here required opus-level judgment beyond what is flagged `[I]`/`[C]` above.
