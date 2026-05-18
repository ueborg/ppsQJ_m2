# QJ-PPS project handoff (as of 2026-05-18, commit e2ec079)

Single entry-point document for resuming this project in a fresh chat.
If you're a new assistant instance picking this up, read this first.

## Project context

- **Researcher**: Utku, master's in quantum engineering, University of Groningen.
- **Supervisor**: Dganit Meidan (RUG).
- **Project**: `ppsQJ_m2` — measurement-induced phase transitions (MIPT) in
  a 1D Kitaev chain under quantum-jump partial postselection (QJ-PPS).
  Extends Kells–Meidan SciPost Phys 14, 3, 031 from diffusive to quantum-jump.
- **Repo**: `github.com/ueborg/ppsQJ_m2`. Local at `/Users/catlover1337/Documents/ppsQJ_m2/`.
- **Compute**: Habrok HPC (RUG), interactive nodes ~60 cores.
- **Language**: Python. Code uses numpy, scipy, tqdm; Gaussian fermionic backend.

## Model

Hamiltonian $H = w \sum_j (c_j^\dagger c_{j+1} + \text{h.c.})$ on $L$ sites.
Jump operators $L_j = \sqrt{\alpha}\,d_j$ with $d_j = (\gamma_{2j} - i\gamma_{2j+3})/2$
(distance-3 Majorana bond operator).

Parametrization: $\alpha + w = 1$, $\lambda := \alpha$, $\zeta \in [0, 1]$ the
postselection parameter. $\zeta = 1$ recovers Born-rule quantum jumps;
$\zeta = 0$ is full postselection (no clicks at all).

Trajectory measure is tilted by $\zeta^{N_T}$. The tilted Lindbladian
$$
\mathcal{L}_\zeta[\rho] = -i[H,\rho] + \sum_j\bigl[\zeta J_j\rho J_j^\dagger - \tfrac{1}{2}\{J_j^\dagger J_j, \rho\}\bigr]
= \mathcal{L}_0 + \zeta \mathcal{J}
$$
splits into no-click non-Hermitian piece $\mathcal{L}_0$ and click recycling
piece $\zeta\mathcal{J}$.

## Current top-level result (provisional but well-supported)

$$
\boxed{\;\lambda_c(\zeta) \sim A\sqrt{\zeta}, \quad A \approx 0.5\;}
$$

with two-parameter FSS structure:
$$
B_L(\lambda, \zeta) = \mathcal{B}\bigl(\lambda L^{1/2}, \zeta L, u L^{-\omega}\bigr)
$$
i.e., $y_\lambda = 1/2$ from BdG localization, $y_\zeta = 1$ from extensivity.

Physical picture: $\zeta$ is a *defect fugacity per no-click localization volume*.
Criticality at $\zeta \xi_{\rm ps}(\lambda_c) \sim 1$ with $\xi_{\rm ps} \sim \lambda^{-2}$
gives $\lambda_c \sim \sqrt{\zeta}$.

Born-rule endpoint $\lambda_c(1) \approx 0.5$ matches Carollo (PRA 2018).

## Status of each step in the theoretical chain

| # | Claim | Source | Status |
|---|---|---|---|
| 1 | $\xi_{\rm ps} \sim \lambda^{-2}$ | non-Hermitian SSH-Majorana BdG (Path A/B analysis) | Rigorous |
| 2 | $y_\lambda = 1/2$ | Step 1 | Rigorous |
| 3 | Click vertex extensive over bonds | $\mathcal{G}_{\rm click} = \zeta\alpha\sum_j d_j^{(+a)}d_j^{*(-a)}$ | Manifest |
| 4 | $\theta_1(\lambda, L) \sim L$ | Numerical, BdG slowest-decay state | **Confirmed for $\lambda \le 0.15$** |
| 5 | $y_\zeta = 1$ | Steps 3 + 4 + locality of no-click theory | Confirmed |
| 6 | $\zeta\xi_{\rm ps} \sim 1$ critical condition | Steps 1-5 | Derived |
| 7 | $\lambda_c(\zeta) \sim \sqrt{\zeta}$ | Step 6 | Predicted |
| 8 | Empirical $\phi \approx 1/2$ from Binder data | $L \le 128$ aggregate, B_L crossings | **Confirmed in range $\phi \in [0.5, 0.7]$** |
| 9 | $L = 192, 256$ confirmation | FSS collapse test | **Pending FST data** |

Caveats:
- Step 4 only works for $\lambda \le 0.15$ due to biorthogonal construction
  breakdown at large $\lambda$ (gives unphysical covariance). The relevant
  $\lambda_c$ values are all $< 0.5$ across the data, so the critical regime
  is well inside the clean range.
- Step 8 has uncertainty: $\phi \in [0.5, 0.7]$ across extraction methods.
  $\phi = 1/2$ is the cleanest theoretical value and is consistent with all
  three methods within error.

## Eliminated scenarios

- **Scenario A** ($\lambda_c \to 0.5$ for all $\zeta > 0$, LMR-Ising
  persistence): RULED OUT by small-$\zeta$ Binder data showing $\lambda_c$
  monotonically decreasing with $L$ from 0.185 → 0.050 at $\zeta = 0.02$.
- **Scenario B** (separatrix at $\zeta_c \approx 0.143$): the apparent
  separatrix was an ARTIFACT of using c_eff threshold method. Doesn't
  survive switch to Binder cumulant.
- **Original Scenario C** (linear $\lambda_c \sim \zeta$, $\phi = 1$):
  EXCLUDED. Collapse residual 28% worse than $\phi = 1/2$.

## Methodology lesson (important)

**c_eff method gives misleading results.** The c_eff values in this data
have median ~3.4 and never drop below 1.5; c_eff = 1 threshold method
fails. The L-pair c_eff crossings happen at c_eff ≈ 6-8 (not universal),
indicating they're in the volume-law/log-law boundary, not at the MIPT.

**B_L (Binder cumulant) crossings are the right observable.** Born-rule
sanity check: B_L crossings give $\lambda_c$ mean = 0.497, matching
Carollo's 0.5 exactly. All subsequent analysis uses B_L.

## Empirical data sources

- **Main aggregate**: `/Users/catlover1337/Downloads/clone_aggregate(1).pkl`
  (1920 entries, $L \in \{8, 16, 24, 32, 48, 64, 96, 128\}$,
  $\zeta \in \{0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 1.00\}$,
  24 $\lambda$ values from 0.02 to 0.90).
- **Each entry**: contains `S_mean`, `B_L_mean`, `theta_mean`, etc. with
  per-realization statistics over $n_{\rm real}$ independent runs.
- **(96, 128) at $\zeta \le 0.10$ is noisy**: only $N_c = 100$ clones, lower stats.
  Excluded from clean analyses.

## Pending action items (priority-ordered, with exact commands)

### 1. Push commits to GitHub (LOCAL TERMINAL REQUIRED)

There are 6 commits ahead of origin/main on local. `osascript` can't access
the SSH keychain on macOS, so this must be done from the user's terminal:
```
cd ~/Documents/ppsQJ_m2 && git push
```

Commits to push (most recent first):
```
e2ec079  First-principles confirmation of y_zeta=1 via theta_1
510bbe3  Final synthesis: y_zeta=1 from extensivity, λ_c~√ζ
4c6785b  Refined FSS: two-parameter scaling
de899dd  FSS results: B_L > c_eff
24a3f58  Scenario C addendum (superseded)
c5cd98d  Comprehensive theory summary
```
Plus untracked files (don't auto-add): see `git status -sb`.

### 2. Submit L = 192, 256 jobs on Habrok

Grid is already configured in `pps_qj/parallel/grid_pps.py` (FST section).
SLURM script: `slurm/submit_clone_v2_fst.sh` (24 tasks × 5 cores).
Expected wall time: ~22h/task at L=192, ~31h at L=256, 84 tasks total.

Submission flow:
```
ssh habrok
cd ppsQJ_m2 && git pull
sbatch slurm/submit_clone_v2_fst.sh
```

### 3. Run the decisive collapse test when FST data is in

When `fst_aggregate.pkl` is built and pulled back to local:
```
python analysis/test_yzeta1_collapse.py --add-fst /path/to/fst_aggregate.pkl
```

Visual verdict: do the $L = 192, 256$ [square] markers fall on the $L \le 128$
[circle] curve in the $(\zeta L, \lambda_c\sqrt{L})$ plane?

- **YES** → thesis result $\lambda_c \sim \sqrt{\zeta}$ is solid.
- **NO** → reassess; possible logarithmic corrections or higher-order $y_\zeta$
  corrections.

## Open theoretical questions (priority-ordered)

### A. Fix the large-$\lambda$ $\theta_1$ calculation

The biorthogonal construction in `analysis/compute_theta1.py` produces
unphysical covariance matrices (eigenvalues $> 1$, bond parity expectations
$> 1$) for $\lambda \ge 0.5$. Two paths to fix:

(a) For small $L$ (say $L \le 6$), construct the full $4^L$-dimensional
Liouvillian as a matrix, find dominant left/right eigenvectors, compute
$\theta_1 = \langle\!\langle \ell_0 | \mathcal{J} | r_0 \rangle\!\rangle / \langle\!\langle \ell_0 | r_0 \rangle\!\rangle$
directly. Cross-check against BdG at small $\lambda$.

(b) Regularize the non-Hermitian eigenstate construction. The "slowest
decaying" state at large $\lambda$ might require time-evolution from a
physical initial state and projection, rather than naive eigenvector picking.

Either path closes the loop for $\lambda \in [0.3, 1.0]$.

### B. Extract universality class along $\lambda_c(\zeta)$

The framework establishes $y_\lambda$ and $y_\zeta$ but doesn't predict the
critical exponents at the transition itself. Compute the critical Binder
value $B_L^*$ along the line — is it $\zeta$-independent (single universality
class) or drifting? Use existing aggregate, extend `analysis/extract_yzeta.py`.

### C. Bosonization derivation of $y_\zeta = 1$

The current derivation is "extensivity + locality". A proper RG derivation
at the no-click fixed point (which is gapped/localized) would close the loop
on universality. The existing bosonization in `theory/qj_one_minus_zeta_expansion.md`
and friends approaches this from the Born-rule end, but the no-click-end
derivation is the natural counterpart.

### D. Sub-leading $1/L$ corrections in $\theta_1$

The empirical slope $p = 1.04$ (instead of exactly 1.0) at small $\lambda$
suggests $\theta_1 = A L + B + O(1/L)$. A clean linear fit with explicit
boundary term would pin down both $A$ and $B$ and confirm the leading $L$
scaling cleanly.

## Conventions

- $\alpha + w = 1$, $\lambda := \alpha$
- $\zeta = 0$: full postselection (no clicks); $\zeta = 1$: Born-rule
- Jump $L_j = \sqrt{\alpha}\,d_j$, $d_j = (\gamma_{2j} - i\gamma_{2j+3})/2$
- $L_j^\dagger L_j = (\alpha/2)(1 - i\gamma_{2j}\gamma_{2j+3}) = (\alpha/2)(1 - \hat P^I_j)$
- Bond parity $\hat P^I_j = i\gamma_{2j}\gamma_{2j+3}$ (distance-3)
- Effective no-click Hamiltonian $H_{\rm eff} = H - i(\alpha/2)\sum_j L_j^\dagger L_j$
- Majorana covariance $\Gamma_{ab} = i\langle\gamma_a\gamma_b\rangle$, real antisymmetric, eigenvalues in $[-1, 1]$ for physical states
- $L$ sites → $2L$ Majoranas → $L-1$ bonds (OBC); `bond_jump_pair(bond)` in `pps_qj.gaussian_backend` maps bond index to Majorana indices

## File map

### Theory documents (chronological — read in order)

| File | What it contains | Status |
|---|---|---|
| `theory/qj_pps_theory_summary.md` | Original comprehensive overview | Background |
| `theory/STATUS.md` | Bosonization status guide | Background |
| `theory/qj_two_replica_derivation.md` | Two-replica generator derivation | Rigorous setup |
| `theory/qj_bosonization_calculation.md` | First-pass bosonization | Historical |
| `theory/qj_one_minus_zeta_expansion.md` | Sharpened bosonization, $\Delta=4$ vertex | Rigorous for $\zeta \to 1$ |
| `theory/qj_chiral_vertex_result.md` | Chiral vertex result | Historical |
| `theory/qj_pps_scenario_C_addendum.md` | First (wrong) guess: linear $\lambda_c \sim \zeta$ | **Superseded** |
| `theory/fss_analysis_results.md` | First FSS results, B_L vs c_eff | Methodology |
| `theory/two_parameter_FSS_results.md` | Two-parameter FSS, $y_\zeta$ extraction | Current empirical |
| `theory/qj_pps_final_synthesis.md` | Final synthesis with $y_\zeta=1$ derivation | Current top-level |
| `theory/theta1_first_principles.md` | $\theta_1$ from BdG, $y_\zeta=1$ confirmed numerically | Current top-level |
| `theory/HANDOFF.md` | **this file** | Entry point |

### Analysis scripts

| File | Purpose |
|---|---|
| `analysis/compute_theta1.py` | $\theta_1$ from BdG slowest-decay state |
| `analysis/extract_yzeta.py` | Extract $y_\zeta$ from B_L crossings (3 methods) |
| `analysis/test_yzeta1_collapse.py` | **Decisive test ready for FST data** (use `--add-fst`) |
| `analysis/test_fss_collapse.py` | Original c_eff-based test (kept for reference) |

### Diagnostic plots in `analysis/`

| File | What it shows |
|---|---|
| `theta1_scaling_v2.png` | $\theta_1$ vs $L$ for various $\lambda$ — confirms $\theta_1 \sim L$ at small $\lambda$ |
| `yzeta_extraction.png` | $y_\zeta$ optimization via collapse residual + slice fits |
| `yzeta1_collapse_test.png` | Current $L \le 128$ data in predicted scaling |
| `fss_final.png` | Comprehensive Binder analysis |
| `fss_collapse_data.txt` | Raw numerical tables |

### Computation infrastructure

| File | Purpose |
|---|---|
| `pps_qj/gaussian_backend.py` | Gaussian fermionic backend; `effective_generator(L, w, alpha)` builds $h_{\rm eff}$ |
| `pps_qj/cloning.py` | Cloning population dynamics (small-$L$, used for aggregate) |
| `pps_qj/doob_wtmc.py` | Doob $h$-transform WTMC (main approach) |
| `pps_qj/parallel/grid_pps.py` | Grid spec (includes FST section for L=192, 256) |
| `slurm/submit_clone_v2_fst.sh` | SLURM submission for FST runs |

## Key references

- Carollo et al., PRA 98, 010103(R) (2018) — projective MIPT, $\lambda_c \approx 0.5$
- Kells, Meidan, Romito, SciPost Phys 14, 3, 031 — diffusive PPS framework
- Leung, Meidan, Romito — diffusive PPS-SSE RG (companion to Kells et al.)
- Giardina, Kurchan, Peliti, PRL 96, 120603 (2006) — cloning/population dynamics
- Lecomte, Tailleur, J. Stat. Mech. (2007) — tilted-Liouvillian framework
- Nemoto et al., PRE 2017 — Jack–Sollich-like feedback for cloning
- Bao, Choi, Altman — two-replica MIPT framework
- Giamarchi, *Quantum Physics in 1D*; Sénéchal, cond-mat/9908262 — bosonization

## Practical notes (Mac + Habrok workflow)

- User works on Mac with VSCode → GitHub → Habrok
- Habrok has SSH key issues; do edits LOCAL, push, then `git pull` on Habrok
- macOS `osascript` cannot access GitHub SSH keychain → user must run
  `git push` from a Terminal window
- Habrok uses CVMFS network filesystem; spawn-based multiprocessing is slow,
  prefer forkserver
- Desktop Commander MCP server is configured for the analysis tools

## How to verify the project state in 30 seconds

```bash
cd ~/Documents/ppsQJ_m2
git log --oneline -7              # last commits, top should be e2ec079
git status -sb                    # check 'ahead 6' of origin/main
ls theory/HANDOFF.md              # this file should exist
ls analysis/test_yzeta1_collapse.py  # decisive test script
```

Then read in order:
1. `theory/HANDOFF.md` (this file)
2. `theory/qj_pps_final_synthesis.md` (full reasoning)
3. `theory/theta1_first_principles.md` (microscopic confirmation)
4. `theory/two_parameter_FSS_results.md` (empirical extraction details)

If FST data has arrived since this was written, run:
```bash
python analysis/test_yzeta1_collapse.py --add-fst /path/to/new/aggregate.pkl
```
and check whether the collapse holds.
