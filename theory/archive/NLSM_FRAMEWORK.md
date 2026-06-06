# NLSM Framework for QJ-PPS — RETRACTION / STATUS UPDATE 2026-05-22

**⚠️ READ THIS FIRST.** The two NLSM-based predictions for the phase
boundary λ_c(ζ) of Case B (single measurement + hopping) are both
inconsistent with the numerical data from Runs A+C (L up to 256):

1. **Naive NLSM** (gradient expansion without matching): predicted
   λ_c → 1 as ζ → 0. REJECTED — data shows λ_c → 0.

2. **Matched NLSM** (multicritical RG from a to ξ_ps ~ λ⁻²):
   predicted λ_c(ζ) ≈ A√ζ at small ζ with A ≈ 1. REJECTED at >3σ.
   Cleanest test (L = 128 vs L = 256 crossings, asymptotic regime
   ζ·L ≥ 5): at ζ = 0.20 the data gives λ_c = 0.43, the prediction
   gives 0.45 (close), but at ζ = 0.10 the data gives 0.23 vs the
   prediction 0.32 (off by 1.4×). The full data is closer to
   λ_c ~ ζ^{0.85} than √ζ, with a saturation at λ_c ≈ 0.48 for ζ ≥ 0.3.

The rest of this document is preserved below for reference, but the
specific quantitative predictions about λ_c(ζ) should NOT be cited as
confirmed. The qualitative observations that survive intact:

- Symmetry class of Case B remains DIII (no change).
- ν = 1/y_λ = 2 is preferred by the two-parameter collapse scan
  (compared to ν = 1, which fits far worse), so the multicritical
  picture for λ-deformations is **not** ruled out, only the ζ
  scaling.
- λ_c → 1/2 at ζ = 1 (Born rule, Carollo) is confirmed empirically.
- λ_c is a monotonically increasing function of ζ on the range
  ζ ∈ [0.02, 1.0] sampled.

What likely went wrong in the matching argument:

- The "matching scale" ξ_ps ~ λ⁻² was assumed too naively. The
  actual no-click localisation length in the BdG H_eff = -iα·n - w·hopping
  is (1 - λ)/λ at α + w = 1, not λ⁻². The naive λ⁻² was a heuristic.
- The cross-Choi vertex's dimension Δ_ζ may not be 1 in the relevant
  field theory. The exponent scan finds y_ζ in [1.5, 2.0] for best
  collapse, not 1.

A correct theoretical treatment likely needs to:
1. Use the actual λ-dependence of the no-click localisation in
   H_eff, including the SSH-like topological structure.
2. Replace the simple Δ_ζ = 1 vertex with whatever has the correct
   RG eigenvalue. The empirically preferred y_ζ ∈ [1.5, 2] suggests
   Δ_ζ ∈ [0, 0.5] in 1+1d. This is a non-trivial constraint.
3. Account for the saturation at ζ ~ 0.1–0.3 — the empirical
   crossover where post-selection stops shifting the transition.

Updated phase diagram is at `analysis/phase_diagram_v2.png`; tabulated
crossings at `analysis/phase_diagram_data.json`.

---

## Original framework document (preserved for reference; predictions
## about λ_c(ζ) are SUPERSEDED as noted above)

# Replica Keldysh / NLSM framework for QJ–PPS

**Date**: 2026-05-20.
**Status**: Theoretical derivation complete; numerical confirmation pending.

This document records the field-theoretic analysis of QJ-PPS monitoring,
adapting the recent Le Gal–Schirò framework (arXiv:2511.22506) to our
PPS-tilted setting and applying it to two distinct model variants. The
LaTeX writeup of this derivation is at
`continuousmeasurementslatex/sections/sec8_replica_nlsm_pps.tex`
(or, more recently, in `~/Downloads/continuousmeasurements(2)/sections/`).

## 1. Why we need a field theory

The Binder cumulant analysis (see `analysis/lambda_c_phi_analysis.md`)
confirms $\lambda_c \sim A\sqrt{\zeta}$ at small $\zeta$ with $\phi = 0.502
\pm 0.026$. But the multicritical RG argument that gives this scaling
($\xi_{\rm ps}\sim\lambda^{-2}$, $\Delta_\zeta = 1$ from cross-Choi) is
only a small-$(\lambda,\zeta)$ argument. It does not tell us:

1. What symmetry class the critical theory belongs to.
2. What the universality class along the critical line is at intermediate
   and large $\zeta$.
3. Whether $\zeta$ enters in any way other than as the bare coupling.
4. What sub-leading observables (Rényi ratios, log corrections,
   topological order parameters) should look like.

The Le Gal–Schirò replica Keldysh framework answers all four
simultaneously, once we make the PPS modification and pick the right
model variant.

## 2. The single algebraic ingredient that PPS adds

Trajectory measure under PPS:
$$
d\mathbb{Q}_s/d\mathbb{P} \;\propto\; \zeta^{N_T},
$$
with $N_T$ the total click count. In the replicated master equation,
this means **only the cross-replica recycling vertex carries the $\zeta$
factor**; the non-Hermitian replica-diagonal dissipator does not:
$$
\partial_t \tilde{\mathcal{R}}^{(N)}_t \;=\; -\tfrac{\gamma}{2}\sum_j \{L^\dagger_j L_j, \tilde{\mathcal{R}}^{(N)}\}_+
\;+\; \zeta\gamma \sum_j L_j^{(N)}\,\tilde{\mathcal{R}}^{(N)}\,L_j^{\dagger\,(N)}.
$$
This translates in the Keldysh action to **$\zeta$ multiplying only
$S_{\rm QJ}$** (the cross-replica vertex), leaving $S_{\rm kin} + S_{\rm nH}$
unchanged. Sanity check: at $\zeta=1$ we recover Le Gal–Schirò; at
$\zeta\to 0$ the cross-replica vertex drops and the $N$-replicated
state evolves under the non-Hermitian effective Hamiltonian alone — the
strict no-click sector.

This is the only structural modification needed. The rest of the
derivation (Hubbard–Stratonovich, saddle point, gradient expansion)
proceeds as in Le Gal–Schirò, with $\zeta\gamma$ everywhere their $\gamma$
appears in the vertex.

## 3. Case A — two competing measurements, no Hamiltonian

### 3.1 Setup

- $H = 0$
- Two channels: $L_j = n_j$ at rate $\gamma$, $\tilde{L}_j = \tilde{n}_j$
  at rate $\alpha$.
- Parametrize $\lambda_A = \alpha/(\alpha+\gamma) \in [0,1]$.
- This is the QJ version of Kells–Meidan–Romito's setup at vanishing
  hopping; in spin language it's the measurement-only TFIM.

### 3.2 Self-duality

The KMR duality $c_j \leftrightarrow d_j$, $\gamma \leftrightarrow \alpha$
is **exact and preserved by PPS** (the trajectory weight depends only
on the total click count, which is invariant under channel exchange).
Hence:
$$
\boxed{\lambda_c^A(\zeta) = 1/2 \quad \forall\zeta\in(0,1].}
$$
This is the sharpest prediction in the project. It says $\zeta$ shifts
the bare NLSM coupling but does not move the transition.

### 3.3 Symmetry constraints

In the $B_{NK}$ basis:
- $(C_2)$: $[R, \sigma_x\otimes\tau_z] = 0$ from $L$-vertex.
- $(C_2')$: $[R, \sigma_x\otimes\tau_y] = 0$ from $\tilde{L}$-vertex's
  pairing component.
- No $C_3$ or $C_4$ since $H = 0$.

Surviving generators (Method 1 via Le Gal–Schirò Table 1):
$(\sigma_0, \tau_0)$, $(\sigma_x, \tau_0)$, $(\sigma_y, \tau_x)$,
$(\sigma_z, \tau_x)$ — three (A,R) + one (S,I), giving $2N^2 - N =
\dim O(2N)$. With $\det R = +1$: $\mathcal{G} = SO(2N)$.

Method 2 (explicit $R$): $R$ has the form
$\begin{pmatrix} a\tau_0 & b\,i\tau_y \\ c\,i\tau_y & d\tau_0 \end{pmatrix}_{KN}$
with $a,b,c,d \in M_N(\mathbb{R})$ and unitarity reducing to orthogonality
of the $2N\times 2N$ matrix — confirming $\mathcal{G} = O(2N)$.

Stabilizer $\mathcal{H}$ (those that commute with $\Lambda = \sigma_z\otimes\tau_0$):
$(\sigma_0,\tau_0)$ and $(\sigma_z,\tau_x)$, dim $= N^2 = \dim U(N)$.

$$
\boxed{\mathcal{G}/\mathcal{H} = SO(2N)/U(N), \quad \text{class } D.}
$$

### 3.4 Topology and universality class

$\pi_2(SO(2N)/U(N)) = \mathbb{Z}$ for $N\geq 2$, allowing a $\theta$-term.
By the duality argument, $\theta\in\{0,\pi\}$ at the self-dual line, and
the topologically non-trivial value $\theta = \pi$ separates the two
area-law phases. A 2D NLSM on $SO(2N)/U(N)$ at $\theta=\pi$ in the
$N\to 1$ limit is the **D-class WZW theory**, reducing to **Ising CFT**
($c=1/2$, $\nu=1$).

### 3.5 Bare coupling and PPS dependence

From gradient expansion: $g_B^{-1} \propto \zeta\sqrt{\alpha\gamma}$.
$\zeta$ enters multiplicatively but does not break the self-duality,
so the location is unchanged. Only the *bare coupling* (and hence the
finite-size approach to the asymptotic universality class) changes
with $\zeta$.

### 3.6 Numerical predictions for Case A

| Prediction | Value | Test |
|---|---|---|
| Location of critical line | $\lambda_c^A = 1/2$ for all $\zeta$ | Binder of Pfaffian invariant crossings |
| Universality | Ising ($c=1/2$, $\nu=1$) | Slope of $S(L/2)$ vs $\log L$ at $\lambda_A = 1/2$ |
| Finite-$L$ correction approach | Slower for smaller $\zeta$ | $\nu_{\rm eff}(\zeta)$ drift study |

KMR (diffusive) find $\nu\approx 5/3$. Predicted asymptotic value for
QJ is $\nu = 1$; KMR's value may be a finite-size effect or may reflect
a genuine difference between QSD and QJ unravelings.

### 3.7 Status

Theory complete; implementation pending. Spec at
`theory/CASE_A_IMPLEMENTATION_SPEC.md` (a hand-off doc for an
implementation agent).

## 4. Case B — single Kitaev-mode measurement + free hopping (the project's main model)

### 4.1 Setup

- $H = w\sum_j (c^\dagger_{j+1} c_j + \text{h.c.})$.
- One channel: $\tilde{L}_j = \tilde{n}_j = d^\dagger_j d_j$ at rate $\alpha$.
- $\lambda = \alpha/(\alpha+w)$, $\alpha+w = 1$.
- This is the project's actual simulation model.

### 4.2 Symmetry constraints

- $(C_2)$: $[R, \sigma_x\otimes\tau_z]$ from $\tilde{L}$-vertex
  density component.
- $(C_2')$: $[R, \sigma_x\otimes\tau_y]$ from $\tilde{L}$-vertex
  pairing component.
- $(C_3)$: $[R, \sigma_0\otimes\tau_z]$ from $H$ tight-binding.

Surviving generators: $(\sigma_0,\tau_0)$ and $(\sigma_x,\tau_0)$,
both (A,R). Dim $= N(N-1) = \dim O(N)\times O(N)$.

$$
\boxed{\mathcal{G}/\mathcal{H} = \frac{SO(N)\times SO(N)}{SO(N)_{\rm diag}} \cong SO(N), \quad \text{class DIII}.}
$$

This is the *same target manifold* as Le Gal–Schirò's general Ising
chain. The mapping: their constraint $C_4$ (Hamiltonian pairing
$\sigma_0\otimes\tau_y$) is replaced in our model by $C_2'$
(measurement pairing $\sigma_x\otimes\tau_y$). Different generators,
same coset.

### 4.3 NLSM and $\beta$-function

$$
S_{\rm NLSM}[U] = \frac{1}{g_B}\int d^2 r\, \text{Tr}[(\partial_\mu U)^T (\partial_\mu U)],
\qquad U\in SO(N),
$$
with bare coupling $g_B \propto \zeta\alpha/w = \zeta\lambda/(1-\lambda)$
from gradient expansion around the Lindbladian saddle.

One-loop $\beta$-function (Hikami / Le Gal–Schirò Eq. 83):
$$
\frac{dg_R}{d\ln L} = \frac{(N-2) g_R^2}{8\pi}.
$$
At $N\to 1$: $dg_R/d\ln L = -g_R^2/(8\pi) < 0$. **Flow toward weak
coupling**. The stable weak-coupling fixed point is the **free-fermion
Gaussian CFT** — this is the project's log phase.

### 4.4 Naive bare-coupling prediction for $\lambda_c$

Setting $g_B = g^*$ gives $\lambda_c^{\rm naive}(\zeta) = g^*/(g^* + \zeta)$.
At $\zeta = 1$: pinning $\lambda_c \approx 0.5$ requires $g^* \approx 1$.
At $\zeta\to 0$: this predicts $\lambda_c \to 1$. **Opposite of the
empirical $\lambda_c \to 0$.** This is the problem the next subsection
resolves.

### 4.5 The exact assumption that fails at small $\zeta$

The Le Gal–Schirò gradient expansion assumes the **dressed length**
$\xi_{\rm dressed}\sim v_0/(\zeta\alpha)$ sets the UV cutoff of the NLSM.
This is correct around the *Lindbladian* saddle point. But at small
$\zeta$ the trajectory ensemble spends long times under no-click
evolution, and the relevant short-distance scale is the **BdG
localization length**:
$$
\xi_{\rm ps} \;\sim\; \lambda^{-2}\,a,
$$
which is *independent of $\zeta$*. When $\zeta$ is small enough that
$\xi_{\rm dressed} > \xi_{\rm ps}$, the gradient expansion's natural UV
cutoff is $\xi_{\rm ps}$, not $\xi_{\rm dressed}$. The Lindbladian
saddle is the wrong zeroth-order point.

### 4.6 Resolution: matching to multicritical RG at $\xi_{\rm ps}$

Run the multicritical RG from $a$ to $\xi_{\rm ps}$. The cross-Choi
operator has dimension $\Delta_\zeta = 1$, so its bare coupling
renormalises as:
$$
\zeta(\ell) = \zeta\cdot(\ell/a)^{2-\Delta_\zeta} = \zeta\cdot(\ell/a).
$$
Running from $a$ to $\xi_{\rm ps} \sim \lambda^{-2} a$:
$$
\zeta_{\rm eff}(\xi_{\rm ps}) = \zeta\cdot\lambda^{-2}.
$$
This is the matched bare coupling for the NLSM. Transition at
$\zeta_{\rm eff}(\xi_{\rm ps}) \sim 1$:
$$
\boxed{\zeta\,\lambda_c^{-2} \sim 1 \implies \lambda_c \sim \sqrt{\zeta}.}
$$
**Recovered from the NLSM framework**, with prefactor $A\sim 1$ from
the matching condition $\zeta\cdot\xi_{\rm ps}/a \sim 1$. Empirical
$A = 0.96 \pm 0.05$ (this session's analysis) is consistent.

### 4.7 The composite picture

Case B has two regimes:
- **Multicritical regime** ($\zeta\ll 1$): Gaussian fixed point at
  origin, cross-Choi perturbation, $\lambda_c \sim \sqrt{\zeta}$,
  $\nu = 2$.
- **DIII NLSM regime** ($\zeta\sim 1$): Lindbladian saddle dominant,
  naive bare coupling applies, $\lambda_c \to \lambda_{\rm BR} \approx 0.5$
  at $\zeta = 1$.

Crossover at $\xi_{\rm ps} \sim a$, i.e., $\lambda\sim 1$, occurs at
$\zeta\sim 1/4$. Consistent with the observed outlier at $\zeta = 0.5$
in the existing data.

### 4.8 What the NLSM gives us analytically

Beyond the small-$\zeta$ scaling, the framework predicts:

1. **Log corrections in the log phase**: $g_R(L) = g_B/(1 + (g_B/8\pi)\ln(L/L_{\rm match}))$,
   leading to $S(L) = (c_{\rm eff}/6)\ln L + b\ln\ln L + \ldots$ with
   $b$ universal. Drift of $c_{\rm eff}(L)$ at fixed $\zeta$ is a
   testable signature.

2. **Rényi ratio drift**: log phase = weak-coupling fixed point of DIII
   NLSM = Gaussian CFT with $c_n/c_1 = (1+1/n)/2$ (i.e., $0.75$ and
   $0.667$ for $n=2,3$). Current data shows $0.79$ and $0.71$,
   slightly above; theory predicts drift downward as $L\to\infty$.

3. **Two-parameter FSS collapse**: $B_L(\lambda, \zeta) =
   \mathcal{B}(\lambda L^{1/2}, \zeta L)$ near the multicritical
   fixed point.

4. **$\nu$ drift with $\zeta$**: $\nu = 2$ at small $\zeta$
   (multicritical); BKT-like at $\zeta = 1$ (DIII non-perturbative
   transition). $\nu_{\rm eff}(L)$ should depend on $\zeta$ at any
   finite $L$.

## 5. Summary comparison

| Property | Case A | Case B |
|---|---|---|
| Hamiltonian | None | $w\sum c^\dagger c$ |
| Measurement(s) | $n_j$ AND $\tilde{n}_j$ | $\tilde{n}_j$ only |
| AZ class | $D$ | $\mathrm{DIII}$ |
| Target | $SO(2N)/U(N)$ | $SO(N)$ |
| Self-duality | YES ($\alpha\leftrightarrow\gamma$) | NO |
| Critical line | $\lambda_c^A = 1/2$ pinned for all $\zeta$ | $\lambda_c(\zeta)$ varies |
| $\zeta$-dependence | none (only bare coupling) | yes (location and universality drift) |
| Topology ($\pi_2$) | $\mathbb{Z}$, $\theta\in\{0,\pi\}$ | $\mathbb{Z}_2$ |
| Universality at critical | Ising ($c=1/2$, $\nu=1$) | $\nu=2$ small-$\zeta$, BKT-like $\zeta=1$ |
| Code status | Spec only, not implemented | Implemented (current code) |
| Numerical status | Not tested | $\phi = 0.502 \pm 0.026$ confirmed |

## 6. References

- Le Gal & Schirò, arXiv:2511.22506 (the parent paper for the QJ
  replica framework)
- Kells, Meidan, Romito, SciPost Phys 14, 031 (2023) — diffusive
  Case A analog
- Fava, Piroli, Swann, Bernard, Nahum (2023) — replica field theory
  for monitored free fermions
- Evers & Mirlin, Rev. Mod. Phys. 80, 1355 (2008) — Anderson
  transitions and class identification
- Hikami, Phys. Lett. B 98, 208 (1981) — NLSM beta functions on
  symmetric spaces
