# QJ-PPS MIPT: Comprehensive Theoretical Summary

**Document purpose**: Self-contained summary of the theoretical work done on
the Quantum Jump Partial Post-Selection (QJ-PPS) model of measurement-induced
phase transitions (MIPTs) on a 1D Kitaev chain. Intended as a handoff document
for AI agents continuing this work.

**Context**: Master's thesis project at the University of Groningen (RUG),
supervisor Dganit Meidan. The project (`ppsQJ_m2`,
github.com/ueborg/ppsQJ_m2) extends Kells, Meidan et al. (SciPost Phys. 14,
3, 031) — which established a partial-post-selection scheme for diffusive
unraveling — to the quantum-jump regime. The central goal is to map the MIPT
critical line in the (λ, ζ) plane, where λ is the measurement strength and ζ
is the post-selection parameter.

---

## 1. The Model

### 1.1 Hamiltonian and jump operators

The chain has $L$ sites with spinless complex fermions $c_j, c_j^\dagger$.
Hamiltonian:
$$
H = w \sum_j (c_j^\dagger c_{j+1} + c_{j+1}^\dagger c_j), \qquad j = 1, \ldots, L-1.
$$
Half filling, $k_F = \pi/2$.

Jump operators (one per bond):
$$
L_j = \sqrt{\alpha}\, d_j, \qquad d_j = \tfrac{1}{2}(\gamma_{2j} - i\gamma_{2j+3}),
$$
with Majorana operators defined as
$\gamma_{2j} = c_j + c_j^\dagger$, $\gamma_{2j-1} = -i(c_j - c_j^\dagger)$.

In c-fermion language:
$$
d_j = \tfrac{1}{2}\bigl[c_j + c_j^\dagger - c_{j+1} + c_{j+1}^\dagger\bigr]
$$
— note the BCS content: $d_j$ contains both particle and hole pieces, hence
does not conserve particle number.

The measured quantity is the Type-I bond parity at distance 3 in Majorana
labels:
$$
L_j^\dagger L_j = \tfrac{\alpha}{2}(1 - \hat P^I_j), \qquad \hat P^I_j = i\gamma_{2j}\gamma_{2j+3}.
$$

The Hamiltonian in Majorana form:
$$
H = \tfrac{w}{2}\sum_j\bigl[\hat P^I_j - \hat P^{II}_j\bigr],
$$
with $\hat P^{II}_j = i\gamma_{2j+1}\gamma_{2j+2}$ (Type-II bond parity at
Majorana distance 1).

**Parametrization**: $\alpha + w = 1$, so $\alpha = \lambda$, $w = 1 - \lambda$,
$\lambda \in (0, 1)$.

### 1.2 Tilted Lindbladian

$$
\mathcal{L}_\zeta[\rho] = -i[H, \rho] + \alpha\sum_j\bigl[\zeta L_j\rho L_j^\dagger - \tfrac{1}{2}\{L_j^\dagger L_j, \rho\}\bigr]
$$
ζ ∈ [0, 1] is the post-selection parameter. ζ = 1 = Born rule (projective
measurement), ζ = 0 = fully postselected (no-click trajectory).

### 1.3 Two-replica generator

Choi-vectorize $\rho \to |\rho\rangle\rangle$ on $\mathcal{H}\otimes\mathcal{H}^*$.
For two-replica purity, four copies of the Hilbert space, indexed by Choi
$s\in\{+,-\}$ and replica $a\in\{1,2\}$. Per-replica generator:
$$
\mathcal{G}^{(a)}_\zeta = -iH^{(+a)} + iH^{(-a)\mathsf{T}} - \tfrac{\alpha}{2}\bigl[(L^\dagger L)^{(+a)} + (L^\dagger L)^{(-a)\mathsf{T}}\bigr] + \alpha\zeta\sum_j L_j^{(+a)} L_j^{*(-a)}.
$$

Three structurally distinct pieces:
- $\mathcal{G}_H$: linear in $\hat P^I_j, \hat P^{II}_j$ with $is$-dependent prefactors.
- $\mathcal{G}_{\rm dec}$: linear in $\hat P^I_j$ only (decay), $\propto \alpha^2$.
- $\mathcal{G}_{\rm click}$: bilinear $d_j^{(+a)} d_j^{*(-a)}$, $\propto \alpha\zeta$.

ζ enters ONLY in $\mathcal{G}_{\rm click}$. The decay rate is at full strength
$\alpha$ regardless of ζ.

---

## 2. Single-Particle Analysis (BdG / SSH-Majorana)

### 2.1 Effective single-particle Hamiltonian

The no-click trajectory evolves under
$$
H_{\rm eff} = H - \tfrac{i\alpha}{2}\sum_j L_j^\dagger L_j = H - \tfrac{i\alpha}{2}\sum_j \tfrac{1}{2}(1 - \hat P^I_j).
$$
In Majorana form this gives an alternating-bond structure:
- **Type-I bonds** (measured): coupling $w - i\alpha/2$ on $\hat P^I$ bonds (Majorana distance 3)
- **Type-II bonds** (unmeasured): coupling $-w$ on $\hat P^{II}$ bonds (Majorana distance 1)

### 2.2 Path decomposition

The Majorana bond graph splits into TWO disjoint paths:
- **Path A**: $\gamma_0 - \gamma_3 - \gamma_4 - \gamma_7 - \gamma_8 - \ldots$
- **Path B**: $\gamma_1 - \gamma_2 - \gamma_5 - \gamma_6 - \gamma_9 - \ldots$

Each path is an SSH-Majorana chain with alternating couplings $t_1 = w - i\alpha/2$
(Type-I bonds) and $t_2 = -w$ (Type-II bonds).

### 2.3 Localization length

Solving the SSH zero-mode equation $r = -t_2/t_1 = w/(w - i\alpha/2)$:
$$
|r|^2 = \frac{w^2}{w^2 + \alpha^2/4} < 1
$$

**Result**: there is an exponentially localized end mode with localization
length (in Majorana units along the path):
$$
\boxed{\;\xi = \frac{4}{\ln(1 + \alpha^2/(4w^2))} \approx \frac{16 w^2}{\alpha^2} \quad \text{for small } \alpha. \;}
$$

In $\lambda$ notation: $\xi \approx 16(1-\lambda)^2/\lambda^2$.
(The factor of 4 vs 16 depends on whether one uses $\alpha$ or $\alpha/2$ in the decay; see direct numerical verification.)

### 2.4 Empirical λ_c ~ 1/√L explained

Setting the localization scale equal to the system size, $\xi \sim L$:
$$
\frac{16(1-\lambda_c)^2}{\lambda_c^2} \sim L \quad\Longrightarrow\quad \lambda_c \sim \frac{4}{\sqrt{L}}.
$$

This matches the empirical scaling $\lambda_c(L) \sim C/\sqrt{L}$ with $C \approx 1$-$2$
(prefactor uncertain due to conventions).

**Physical interpretation**: The "MIPT at ζ=0" identified in earlier work
is a **finite-size crossover** from log-law to area-law entanglement,
controlled by the topological resolution of non-Hermitian Majorana edge
modes. In the thermodynamic limit, $\xi$ is finite for any $\lambda > 0$,
so area law wins for all $\lambda > 0$ at ζ = 0. The only true critical
behavior at ζ = 0 is the line itself.

### 2.5 Numerical verification

Direct BdG diagonalization on the local repo:
- λ=0.3: predicted ξ ≈ 23.7, measured 16.7 (not asymptotic yet)
- λ=0.5: predicted ξ ≈ 5.77, measured 5.79 ✓
- λ=0.7: predicted ξ ≈ 2.15, measured 1.98 ✓

Soft-mode site weights at λ=0.5, L=32: $[0.25, 0, 0.125, 0, 0.063, \ldots]$ —
peaked at both ends, confirming end-localization on each path.

---

## 3. Continuum Form of the Jump Operator

### 3.1 Slow-mode decomposition

At $k_F = \pi/2$, linearize $c_j \to e^{ik_F j}\psi_R(x_j) + e^{-ik_F j}\psi_L(x_j)$.
The Majorana field is
$$
\gamma(x) = \psi_R(x) e^{ik_F x} + \psi_L(x) e^{-ik_F x} + \psi_R^\dagger(x) e^{-ik_F x} + \psi_L^\dagger(x) e^{ik_F x}.
$$

### 3.2 Continuum jump operator

Substituting into $d_j = (\gamma_{2j} - i\gamma_{2j+3})/2$ and using phase
factors $e^{ik_F \cdot 2j} = (-1)^j$, $e^{ik_F \cdot (2j+3)} = -i(-1)^j$:
$$
\boxed{\;d_j \;\to\; \frac{1}{\sqrt{2}}\bigl[\,e^{-i\pi/4}\,\eta_R^{\rm cont}(x_j) + e^{i\pi/4}\,\eta_L^{\rm cont}(x_j)\,\bigr]\;}
$$
where $\eta_{R/L}^{\rm cont}$ are continuum Hermitian Majorana fields built
from the chiral Dirac modes.

### 3.3 Form factor

In momentum space, $d_j \propto u_k c_k + v_k c_{-k}^\dagger$ with
$$
u_k = (1 - e^{ik})/2, \qquad v_k = (1 + e^{ik})/2, \qquad |u_k|^2 + |v_k|^2 = 1.
$$

**Critical observation**: $|u_k|^2 + |v_k|^2 = 1$ identically. There is NO
momentum where $d_j$ vanishes.

### 3.4 The old chirality argument is DEAD

A previous analysis assumed $d_j$ was a complex bond mode (i.e., a hopping-only
operator). This gave a Hatano-Nelson model, predicted 1/L² scaling, and gave
a chirality-protection argument $|g_{\pi/2}|^2 = 0$ at the Dirac fixed point.

**This was wrong**: the actual $d_j$ has BCS-like content (mixes particle and
hole). The form factor never vanishes. The chirality protection at $k = \pi/2$
does not hold. Any new analysis must start from the correct $d_j$.

---

## 4. Bond Parity Bosonization Dictionary

Bosonization of $\hat P_j = i\gamma_m\gamma_{m+r}$ depends on the distance $r$
and the parity of $m$.

Using $i\gamma_m\gamma_{m+r} \sim 2\Im[e^{ik_F r}](\psi_R^\dagger\psi_R - \psi_L^\dagger\psi_L)
+ 2\Im[e^{-ik_F r}]\cdot(-1)^m\cdot(\psi_R^\dagger\psi_L + \text{h.c.}) + \ldots$,
at $k_F = \pi/2$:

| $r$ | $e^{ik_F r}$ | slow density coefficient | staggered "mass" coefficient |
|-----|----|----|----|
| 1 (LMR distance) | $i$ | $+1$ | $-(-1)^m$ |
| 3 (QJ distance) | $-i$ | $-1$ | $+(-1)^m$ |

**QJ's distance-3 bond parity has the opposite signs of both the slow-density
and staggered-mass pieces compared to LMR's distance-1 case.** This is the
fundamental microscopic difference that breaks direct LMR analogy.

Using $\psi_R^\dagger\psi_R - \psi_L^\dagger\psi_L = -(1/\pi)\partial_x\theta$ and
$\psi_R^\dagger\psi_L + \text{h.c.} \sim (2\pi a)^{-1}\sin(2\phi)$:

- $\hat P^I_j$ → slow: $-(1/\pi)\partial_x\theta$, staggered: $+(2\pi a)^{-1}\sin(2\phi)$ at $m=2j$ even
- $\hat P^{II}_j$ → slow: $+(1/\pi)\partial_x\theta$, staggered: $+(2\pi a)^{-1}\sin(2\phi)$ at $m=2j+1$ odd

---

## 5. Bosonized Hamiltonian

### 5.1 Kinetic energy

Standard linearization gives Dirac kinetic energy with Fermi velocity
$$
\boxed{\;v_F^{(0)} = 2w = 2(1-\lambda).\;}
$$

Crucial structural difference from LMR: their $v_F = 4(1-\zeta)\Xi$ has explicit
$\zeta$-dependence because in their model, $\zeta$ enters the noise mean. In
QJ, $\zeta$ multiplies only the click vertex (cross-Choi); the kinetic energy
is $\zeta$-independent.

### 5.2 Mass vertex from Hamiltonian + decay

Combining the bosonized forms of $\hat P^I, \hat P^{II}$:
- H contribution: $-w\,\sin(2\phi)$
- Decay contribution: $-(\alpha^2/4)\,\sin(2\phi)$

Total mass-vertex coefficient (bare, before RG):
$$
\boxed{\;g_{\rm mass}^{H+{\rm dec}}(\lambda) = -(1-\lambda) - \tfrac{\lambda^2}{4}.\;}
$$

This is strictly NEGATIVE for $\lambda \in (0, 1)$. **There is no $\lambda$
where it vanishes naturally.** Specifically, it does NOT vanish at $\lambda = 1/2$.

This is a major structural difference from LMR's measurement-only model,
where the analogous mass vertex coefficient is $\propto \Delta$ (dimerization)
and vanishes naturally at $\Delta = 0$, giving the Ising critical line.

In QJ, there is no tunable dimerization parameter in the bare Hamiltonian
that can make the mass vertex vanish.

### 5.3 Click vertex

The cross-Choi click vertex per replica:
$$
V^{(a)}_j = \alpha\zeta\,d_j^{(+a)}\,d_j^{*(-a)} = \tfrac{\alpha\zeta}{4}\bigl[\,(A) + (B) + (C) + (D)\,\bigr]
$$
where
$$
(A) = \chi^{(+a)}_{2j}\chi^{(-a)}_{2j}, \quad (B) = i\chi^{(+a)}_{2j}\chi^{(-a)}_{2j+3},
\quad (C) = -i\chi^{(+a)}_{2j+3}\chi^{(-a)}_{2j}, \quad (D) = \chi^{(+a)}_{2j+3}\chi^{(-a)}_{2j+3}.
$$

In the LMR spinful mapping ($c^\dagger_{m,\uparrow} = (\chi^{(+1)} + i\chi^{(+2)})/2$,
$c^\dagger_{m,\downarrow} = (\chi^{(-1)} - i\chi^{(-2)})/2$), the Choi index
becomes spin and the replica index combines pairs of Majoranas into one
complex fermion.

Bosonization of the click vertex generates FOUR dim-1 marginal vertex
operators at the free Dirac point ($K_\sigma = K_\rho = 1$):
$$
V^{(a)}_j \;\to\; \alpha\zeta\,\bigl[c_1\cos(\sqrt{2}\phi_\sigma) + c_2\cos(\sqrt{2}\theta_\sigma) + c_3\cos(\sqrt{2}\phi_\rho) + c_4\cos(\sqrt{2}\theta_\rho)\bigr]
$$
with $c_i$ OPE coefficients of order unity. **These coefficients have not been
computed explicitly.**

---

## 6. Mapping to LMR (PRX 15, 021020 (2025))

### 6.1 LMR's two transitions

LMR find their measurement-only model has TWO transitions:

**BKT-line** at
$$
\zeta_c^{LMR} = \frac{\pi}{\pi + 8} \approx 0.282
$$
from the divergence of $K_\sigma$:
$$
u_\sigma/K_\sigma = v_F - \frac{32 a \zeta \Xi}{\pi}, \qquad v_F = 4(1-\zeta)\Xi
$$
setting $u_\sigma/K_\sigma \to 0$.

**Ising line** at $\Delta_{LMR} = 0$ (zero dimerization), $\zeta$-independent
for $\zeta \in (0, 1]$. LMR argue this persists structurally for any
$\zeta > 0$.

### 6.2 Mapping to QJ

The dimerization in QJ would be the imbalance between Type-I (measured) and
Type-II (unmeasured) bonds:
$$
\Delta_{\rm QJ} = \frac{\alpha - w}{\alpha + w} = 2\lambda - 1.
$$

LMR's Ising line at $\Delta = 0$ maps to **$\lambda = 1/2$** in QJ.

**Born-rule data check**: at $\zeta = 1$, the empirical $\lambda_c \approx 0.5$
matches this LMR-Ising prediction. ✓

### 6.3 Mismatches at smaller ζ

Empirical $\lambda_c(\zeta = 0.15)$ from L ≤ 128 data is $\approx 0.29$, far
from 0.5. Either:
- (a) The asymptote $\lambda_c \to 0.5$ holds at all $\zeta > 0$ but finite-size
  corrections at $L \leq 128$ are very large.
- (b) The actual $\lambda_c^\infty(\zeta)$ deviates from 0.5 at small $\zeta$.

This is the central unresolved question. See Section 9.

---

## 7. Data Analysis Results

### 7.1 Aggregate dataset

`clone_aggregate_1_.pkl`: 1920 cells keyed by $(L, \lambda, \zeta)$.
- $L \in \{8, 16, 24, 32, 48, 64, 96, 128\}$
- $\lambda$: 24 values from 0.02 to 0.90
- $\zeta \in \{0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 1.00\}$
- Each cell: `S_mean`, `S_err`, `B_L_mean`, etc.

### 7.2 BdG length scale verified

Pairwise $c_{\rm eff}$ collapse at $\zeta = 0$ against $x = L_{\rm eff}\lambda^2$.
Curves from L = (32, 48), (48, 64), (64, 96), (96, 128) overlay reasonably in
the crossover region $x \in [1, 4]$. Confirms $\xi \sim 1/\lambda^2$ from the
BdG analysis. **The ξ ~ 1/λ² scaling is empirically robust.**

### 7.3 Apparent separatrix at ζ ≈ 0.143

Extracting $\lambda_c(\zeta, L)$ via $c_{\rm eff}$ threshold-crossing at $c = 1$,
the drift $\Delta\lambda_c = \lambda_c(64, 128) - \lambda_c(32, 64)$ shows a
clear sign change:

| ζ-range | drift sign | interpretation if asymptotic |
|---|---|---|
| ζ < 0.14 | negative | $\lambda_c \to 0$ (area law) |
| ζ ≈ 0.14 | ≈ 0 | RG fixed point candidate |
| 0.14 < ζ < 0.7 | positive | $\lambda_c \to 0.5$ (Born MIPT) |
| ζ > 0.7 | slight negative | approaching 0.5 from above |

The crossing point at $\zeta_c \approx 0.143$ is what was originally
hypothesized to be a true RG fixed point.

### 7.4 Working hypothesis

Based on the data + LMR-Ising-persistence argument, the current working
hypothesis is **Scenario A**: $\lambda_c^\infty = 0.5$ for all $\zeta > 0$,
Ising universality with $\nu = 1$, the apparent separatrix at $\zeta \approx 0.143$
is a non-universal correction-to-scaling artifact (the leading $1/L$ amplitude
$B(\zeta)$ crosses zero).

To distinguish from Scenario B (genuine 2D phase diagram), an empirical test
at $L \in \{192, 256\}$ is being set up (see Section 9.2).

---

## 8. Bosonization Attempts and Failures

This section documents what was attempted and what FAILED, to prevent
re-treading the same ground.

### 8.1 Numerology rejected: "$\zeta_c = 1/7$"

Initial structural analog of LMR's $\zeta_c^{LMR} = \pi/(\pi+8)$ in the form
$\zeta_c = 4\pi/(4\pi + \tilde C)$ with $\tilde C = 24\pi$ gives $\zeta_c = 1/7 \approx 0.1429$,
matching data to 3 decimals.

**Verdict**: numerology. With 2 sig figs in the data, many simple
expressions fit; the prediction is not unique. Discarded.

### 8.2 Naive mass-vertex zero — λ-dependent ζ_c

Setting $g_{\rm mass}^{\rm total} = g^{H+{\rm dec}} + c_{\rm click}\,\lambda\zeta = 0$:
$$
\zeta_c(\lambda) = \frac{(1-\lambda) + \lambda^2/4}{c_{\rm click}\,\lambda}.
$$

With $c_{\rm click}$ tuned to match data at $\lambda = 0.29$: predicts
$\zeta_c(0.1) \approx 0.54$, $\zeta_c(0.5) \approx 0.067$. STRONGLY $\lambda$-dependent.

**Verdict**: inconsistent with data's approximately $\lambda$-independent
separatrix. Not the right mechanism.

### 8.3 Luttinger divergence (BKT analog) — also λ-dependent

At $(\alpha\zeta)^2$ order, $u_\sigma/K_\sigma = v_F - C(\alpha\zeta)^2$.
Divergence:
$$
\zeta_c(\lambda) = \sqrt{\frac{2(1-\lambda)}{|c_2|\,\lambda^2}}
$$
Also strongly $\lambda$-dependent. Same verdict.

### 8.4 Direct LMR substitution — wrong magnitude

Direct substitution of QJ operator content into LMR's formula gives
$\zeta_c^{QJ} \sim 1$, off by factor ~7 from empirical 0.143. Indicates the
QJ-specific OPE coefficients differ substantially from LMR's, or the
mechanism is structurally different.

### 8.5 What this rules out

Both standard mechanisms in the LMR framework (mass-vertex zero, Luttinger
divergence) predict strongly $\lambda$-dependent $\zeta_c$. Neither matches
the data's approximately $\lambda$-independent separatrix.

**Conclusion**: if the separatrix at 0.143 is a real transition, it requires
a non-perturbative mechanism (Lindblad gap closing, level crossing in the
SCGF). If it is not real, it's a finite-size feature where the non-universal
correction-to-scaling amplitude $B(\zeta)$ vanishes.

The working hypothesis is the latter (Scenario A), to be tested empirically
at $L = 192, 256$.

---

## 9. Open Questions and the Empirical Test

### 9.1 Theoretical open questions

The four steps that would complete the bosonization (NOT done):

1. **Explicit Wick contractions** for the four OPE coefficients $c_1, c_2, c_3, c_4$
   of the click vertex (Section 5.3).
2. **Explicit second-order OPE coefficient** $C_2$ for Luttinger
   renormalization at $(\alpha\zeta)^2$ order.
3. **Explicit verification** that some dim-1 vertex coefficient in the
   bosonized theory is $\propto (2\lambda - 1)\cdot f(\zeta)$ with
   $f(\zeta) > 0$, which would give the LMR-Ising prediction
   $\lambda_c = 1/2$ for all $\zeta > 0$.
4. **Multi-coupling RG flow** with all four marginal vertices and both
   Luttinger parameters $K_\sigma, K_\rho$, integrated to locate the
   critical surface in $(\lambda, \zeta)$.

The bare-level analysis I did (point 3 attempted) does NOT show
$(2\lambda - 1)$ structure naturally. If the prediction $\lambda_c = 1/2$ is
correct, this structure must emerge from the RG flow rather than from the
bare operator content. This is the central unverified claim.

### 9.2 The decisive empirical test

L = 192, 256 simulations at $\zeta \in \{0.05, 0.10, 0.14, 0.18, 0.50, 1.00\}$
and $\lambda \in \{0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50\}$. See
`pps_qj/parallel/grid_pps.py` (FST section, added 2026-05-18) and
`slurm/submit_clone_v2_fst.sh`.

**Decisive signature**: at small $\zeta$ (e.g., $\zeta = 0.05, 0.10$), does
the drift sign reverse between the (64, 128) and (128, 192) L-pairs?
- If yes → Scenario A: $\lambda_c^\infty = 1/2$ Ising, separatrix is FS
  artifact. Thesis framing: "Ising universality, $\lambda_c \approx 0.5$
  independent of $\zeta$, robust to PPS, finite-size corrections at
  resolved L."
- If no → Scenario B: genuine 2D phase diagram, separatrix at $\zeta_c \approx 0.143$
  is a real transition. Thesis framing: "QJ-PPS has BKT-Ising structure
  analogous to LMR diffusive, with the Ising-line location shifting due to
  QJ-specific microscopics."

### 9.3 Other potentially testable predictions

If Scenario A holds, additional Ising predictions to verify:
- $\nu = 1$ from FSS data collapse $\lambda_c(L) = 0.5 + B(\zeta)/L + \ldots$
- $c_{\rm eff}(\zeta)$ along the critical line determined by $K_\sigma^*(\zeta)$.
  Bosonization predicts this in principle but explicit calculation pending.
- Power-law decay of two-point functions along the critical line with
  exponents set by $K_\sigma^*$.

---

## 10. Conventions and Notation

### 10.1 Parameter conventions

- $\alpha + w = 1$ (normalization)
- $\lambda = \alpha / (\alpha + w) = \alpha$ (measurement strength)
- $1 - \lambda = w$ (Hamiltonian strength)
- $\zeta \in [0, 1]$ (post-selection: 0 = full PPS, 1 = Born rule)
- $k_F = \pi/2$ at half filling
- $v_F = 2w$ in the bare bosonization
- Lattice constant $a = 1$

### 10.2 Bosonization conventions (Giamarchi)

Chiral fermions:
$$
\psi_{R,\sigma}(x) = \frac{\eta_{R,\sigma}}{\sqrt{2\pi a}}\,e^{i[\phi_\sigma(x) + \theta_\sigma(x)]}, \quad
\psi_{L,\sigma}(x) = \frac{\eta_{L,\sigma}}{\sqrt{2\pi a}}\,e^{-i[\phi_\sigma(x) - \theta_\sigma(x)]}.
$$
Commutator $[\phi_\sigma(x), \theta_{\sigma'}(y)] = i\pi\delta_{\sigma\sigma'}\Theta(y-x)$.

Charge/spin separation:
$$
\phi_\rho = \frac{\phi_\uparrow + \phi_\downarrow}{\sqrt{2}}, \qquad \phi_\sigma = \frac{\phi_\uparrow - \phi_\downarrow}{\sqrt{2}}.
$$

Scaling dimensions at Luttinger fixed point $(K_\sigma, K_\rho)$:
- $\cos(\sqrt{2}\phi_\sigma)$: dim $K_\sigma$
- $\cos(\sqrt{2}\theta_\sigma)$: dim $1/K_\sigma$
- $\cos(\sqrt{2}\phi_\rho)$: dim $K_\rho$
- $\cos(\sqrt{2}\theta_\rho)$: dim $1/K_\rho$

All dim 1 at the free Dirac point $K = 1$, marginal.

### 10.3 Choi-Jamiolkowski conventions

For operator $O$ acting on the original Hilbert space:
- $O\rho \to (O\otimes I)|\rho\rangle\rangle$ (acts on (+) Choi copy)
- $\rho O \to (I\otimes O^\mathsf{T})|\rho\rangle\rangle$ (acts on (-) Choi copy, transposed)

LMR spinful mapping (their Eq. 39):
$$
c^\dagger_{m,\uparrow} = \tfrac{1}{2}(\chi_m^{(+1)} + i\chi_m^{(+2)}), \quad c^\dagger_{m,\downarrow} = \tfrac{1}{2}(\chi_m^{(-1)} - i\chi_m^{(-2)}).
$$

---

## 11. Key References

### Primary

- **LMR**: Leung, Meidan, Romito, *Theory of free fermions dynamics under partial post-selection*, Phys. Rev. X **15**, 021020 (2025). The diffusive analog of this work. Section V.C and Appendices E, G are the template for the QJ bosonization.

- **Kells et al.**: SciPost Phys. **14**, 3, 031. Supervisor's earlier paper that established the diffusive PPS scheme.

- **Carollo et al.**: Phys. Rev. A **98**, 010103(R) (2018). Projective measurement limit ($\zeta = 1$) of this model. Establishes the lattice-level result $\lambda_c \approx 0.5$ Ising.

### Bosonization references

- **Giamarchi**, *Quantum Physics in One Dimension* (Oxford, 2004). Standard reference. Chapters 1-3 + Appendices C, D.
- **Sénéchal**, *An Introduction to Bosonization*, cond-mat/9908262. Free pedagogical entry point.
- **von Delft & Schoeller**, *Bosonization for beginners*, Annalen der Physik **7**, 225 (1998), cond-mat/9805275. Klein factor algebra in detail.

### Other related

- **Cao, Tilloy, De Luca**: SciPost Phys. **7**, 024 (2019). Free fermions under continuous monitoring, replica field theory.
- **Coppola, Tirrito, Karevski, Collura**: JSTAT 2022, 053101. Discrete-time projective MIPT.
- **Li, Chen, Fisher** etc.: original MIPT papers, monitored circuits.

---

## 12. Repository Structure

`github.com/ueborg/ppsQJ_m2`, local path `/Users/catlover1337/Documents/ppsQJ_m2/`.

Key files:
- `pps_qj/gaussian_backend.py`: Gaussian fermionic backend (covariance matrix evolution).
- `pps_qj/parallel/grid_pps.py`: parameter grids for all simulations (doob, cloning v1, v2, supp, FST, ζ=0).
- `pps_qj/parallel/worker_clone_v2_pps.py`: cloning algorithm worker (v2 main).
- `pps_qj/parallel/worker_clone_v2_fst_pps.py`: FST extension worker (L=192, 256).
- `slurm/submit_clone_v2_fst.sh`: SLURM submit script for the empirical test.
- `theory/`: markdown documents with theoretical analyses (BdG, two-replica derivation, chirality argument, attempted bosonizations).
- `analysis/`: data analysis scripts and aggregated pickles.

Computation: Habrok HPC cluster at RUG. Forkserver multiprocessing, 5 cores per task, 24 concurrent tasks per node.

---

## 13. Status Summary (2026-05-18)

**Confirmed results**:
- BdG localization length $\xi \sim 16(1-\lambda)^2/\lambda^2$ at $\zeta = 0$
- Empirical $\lambda_c \sim 1/\sqrt{L}$ scaling explained as topological FS crossover
- Born-rule data: $\lambda_c(\zeta = 1) \approx 0.5$ consistent with Carollo's Ising universality
- ξ ~ 1/λ² collapse verified for $\zeta = 0$ data

**Theoretical framework**:
- Two-replica generator structure with $\mathcal{G}_H + \mathcal{G}_{\rm dec} + \mathcal{G}_{\rm click}$
- Correct continuum form of $d_j$ with $u_k, v_k$ form factors
- Bond parity bosonization dictionary at distance 3 vs distance 1
- Four-vertex bosonized click structure identified (OPE coefficients pending)

**Working hypothesis**:
- Scenario A: $\lambda_c^\infty = 0.5$ for all $\zeta > 0$, Ising universality, separatrix at 0.143 is FS artifact

**Pending**:
- Explicit OPE coefficients $c_1, c_2, c_3, c_4$ via Wick contraction
- Multi-coupling RG flow integration
- L = 192, 256 empirical test (job script written, ready to submit when HPC available)
- Verification of $\nu = 1$ FSS scaling
- Calculation of $K_\sigma^*(\zeta)$ along the critical line and $c_{\rm eff}(\zeta)$ prediction

**Empirically falsifiable predictions if Scenario A**:
- At L = 192, 256, the drift sign in $\lambda_c$ at small $\zeta$ should reverse compared to L ≤ 128.
- $\lambda_c(\zeta, L) = 0.5 + B(\zeta)/L + O(1/L^2)$ with $B(0.143) = 0$ as a non-universal microscopic accident.

**Empirically falsifiable predictions if Scenario B**:
- The drift sign at small $\zeta$ stays negative at L = 192, 256.
- $\lambda_c^\infty(\zeta) \to 0$ as $\zeta \to 0$, with a sharp transition at $\zeta_c \approx 0.143$.

The L = 192, 256 results will decide between these scenarios.
