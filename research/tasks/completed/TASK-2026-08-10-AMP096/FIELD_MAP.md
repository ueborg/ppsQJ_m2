# FIELD_MAP — TASK-2026-08-10-AMP096               (Charter Stage 2)

Scope: the dependency structure of the Cut B amplitude, not the field at large.
This question is internal-provenance, so the map is dominated by our own nodes.

## Nodes

| id | type | statement |
|---|---|---|
| `CB-AMP-096-001` | foundational_claim (withdrawn) | A = 0.96 ± 0.05 |
| `CB-AMP-001` | foundational_claim (provisional) | A ≈ 0.49 |
| `CB-PARAM-001` | theorem | the lambda_c vs r_c exponent gap is exactly the Jacobian |
| `CB-WINDOW-001` | negative_result | fitted exponents drift strongly with window |
| `OBS-BLPROD-001` | evaluation_convention | average-of-products locator |
| `OBS-BLKMR-001` | evaluation_convention | product-of-averages locator (KMR's) |
| `analysis/lambda_c_phi_analysis.md` | method (May 2026) | the generating document for 0.96 |
| `EV-EXEC-AUDITREPRO-001` | empirical_validation | the audit reproduction, not L-extrapolated |
| `EV-DATA-BOUNDARYCSV-001` | dataset | July campaign, 5,634 realizations, carries zeta = 1.0 |
| `SRC-KMR-2023` | foundational_claim (external) | the alleged "diffusive prefactor" comparator |
| p = 1/2 extrapolation exponent | assumption | 1/sqrt(L) crossing extrapolation, chosen by assuming nu = 2 |
| `DISP-PHI-001` | open_bottleneck | phi = 1/2 versus phi = 1 |

## Relations

| from | relation | to | basis |
|---|---|---|---|
| `CB-AMP-001` | supersedes | `CB-AMP-096-001` | claim file |
| `CB-AMP-096-001` | **assumes** | p = 1/2 extrapolation exponent | primary document, table of A vs p |
| p = 1/2 | **assumes** | nu = 2 | primary document's own justification |
| `CB-AMP-096-001` | **contradicts (its own note)** | `analysis/lambda_c_phi_analysis.md` | the document fits lambda_c, never r_c |
| `SRC-KMR-2023` | **does NOT support** | the "twice the diffusive prefactor" reading | full-text inspection: no such statement |
| `OBS-BLPROD-001` | ≈ agrees with (0.3–1%) | `OBS-BLKMR-001` | numerics recomputation |
| p = 1/2 | **co-generates** | `CB-PHI-HALF-001` (phi ≈ 0.502) and A = 0.96 | one choice produced both numbers |

## Terminology across communities

| our term | KMR's term | same object? |
|---|---|---|
| r_c = lambda_c/(1−lambda_c) | r_c = detector readout **threshold**, Eq. 20, values −2…−0.5 | **NO.** Same symbol, unrelated quantities. |
| B_L (average-of-products) | B_L = S̄_L S̄^top_L (product-of-averages) | **NO.** Already split into `OBS-BLPROD-001` / `OBS-BLKMR-001`. |

## Barriers to transfer

One symbol (`r_c`) and one label (`B_L`) each carried two distinct meanings
across the internal/external boundary. Both collisions are now identified; the
`r_c` collision was not previously recorded anywhere.
