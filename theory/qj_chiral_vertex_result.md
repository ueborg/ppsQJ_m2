# Chirality of the QJ cross-replica vertex and the all-orders cancellation of $K$-renormalization

This document carries out the explicit OPE-driven RG calculation for the
QJ cross-replica click vertex established in `qj_one_minus_zeta_expansion.md`,
and arrives at a sharper conclusion than the prior $(1-\zeta)^2$-coefficient
estimate.

**Headline result:** the QJ cross-replica click vertex $V_j$, after the
four-Choi-copy bosonization, reduces to a **purely chiral (left-moving)
vertex operator** in the inter-replica $\rho_D$ mode:
$$
V_j \;\sim\; \exp\bigl(-4i\,\phi_L^{\rho_D}(x_j)\bigr).
$$
Its self-OPE produces only chiral derivatives of $\phi_L$, which
renormalize the left-mover velocity $u_L$ but **cannot renormalize the
Luttinger parameter $K$**. Consequently
$$
\boxed{
\;K_\sigma^{\rm QJ}(\zeta) \;=\; K_\sigma^{\rm QJ}(\zeta = 1) \;=\; 1
\quad\text{to all orders in } (1-\zeta)\;}
$$
within the leading bosonization.

Equivalently, the central charge $c_{\rm eff}$ extracted from the
log-law scaling of entanglement is **unchanged** by the cross-replica
click vertex at every order of the perturbative expansion around the
Born-rule fixed point.

This is sharper than the earlier statement that the leading $(1-\zeta)$
slope vanishes: the *entire* perturbative expansion in $(1-\zeta)$ from
the bosonized cross-replica vertex gives zero shift in $c_{\rm eff}$.
The shift in $u_L$ exists but is invisible in the entanglement scaling
(velocity rescales the cutoff in the additive constant of $S_A$, not
the prefactor $c/6$).

---

## 1. Recap and chirality of the bond operator

From `qj_one_minus_zeta_expansion.md` §3, the cross-replica click
vertex in the 4-Choi-copy theory bosonizes (per replica) as
$$
\bigl(d_j^{(-, a)}\bigr)^\dagger d_j^{(+, a)} \;\sim\; \frac{1}{2\pi a}\,
\exp\bigl[i\sqrt{2}\,(\Theta_D^a - \Phi_D^a)(x_j)\bigr]
$$
with $\Theta_D^a, \Phi_D^a$ the bra–ket difference modes within replica
$a$. The product over both replicas then bosonizes, in the inter-replica
$(\rho, \sigma)$ basis, to
$$
V_j \;\sim\; \frac{1}{(2\pi a)^2}\,
\exp\bigl[2i\,(\Theta_D^\rho - \Phi_D^\rho)(x_j)\bigr],
$$
which we now examine in chiral form.

### 1.1 Chiral decomposition

The non-chiral fields $(\phi, \theta)$ decompose into chiral parts
(in Luttinger conventions with $K = 1$):
$$
\phi = \phi_R + \phi_L,
\qquad
\theta = \phi_R - \phi_L,
$$
so that
$$
\phi_R = (\phi + \theta)/2,
\qquad
\phi_L = (\phi - \theta)/2.
$$
A generic vertex $\exp[i(q_\phi \phi + q_\theta \theta)]$ becomes
$$
\exp\!\bigl[\,i(q_\phi + q_\theta)\,\phi_R + i(q_\phi - q_\theta)\,\phi_L\,\bigr],
$$
so $\alpha_R = q_\phi + q_\theta$ and $\alpha_L = q_\phi - q_\theta$.

For our $V_j$ with $q_\phi = -2$, $q_\theta = +2$ (in the $\rho_D$ mode):
$$
\alpha_R \;=\; 0,
\qquad
\alpha_L \;=\; -4.
$$
**$V_j$ is purely left-moving.**

### 1.2 Microscopic check: the bond operator at half-filling

The chirality of $V_j$ has an underlying geometric reason at half-filling.
The bond operator is, in Majorana form,
$$
d_j \;=\; \tfrac{1}{2}\bigl(\chi_j^{(1)} - \chi_{j+1}^{(2)}\bigr),
$$
where $\chi_j^{(1)} = c_j + c_j^\dagger$ and $\chi_j^{(2)} = -i(c_j - c_j^\dagger)$
in the project's convention (verified against `gaussian_backend.py` via
`bond_jump_pair`).

At half-filling $k_F = \pi/2$, the bosonization of each Majorana is
$$
\chi_j^{(1)} \;\sim\; e^{i\pi j/2}\,\psi_R + e^{-i\pi j/2}\,\psi_L + \text{h.c.},
\qquad
\chi_j^{(2)} \;\sim\; -i\bigl[i\, e^{i\pi j/2}(\psi_R - \psi_L^\dagger) - i\,e^{-i\pi j/2}(\psi_L - \psi_R^\dagger)\bigr].
$$

For $d_0 = (\chi_0^{(1)} - \chi_1^{(2)})/2$, the phase factors are
$e^{i\pi \cdot 0/2} = 1$ for site 0 and $e^{i\pi \cdot 1/2} = i$ for site 1.
Working through the algebra (carried out in
`/tmp/qj_verify_chirality.py`):
$$
d_0 \;\sim\; \psi_L(0) + \psi_L^\dagger(0).
$$
**The bond operator at half-filling is a pure left-mover.** This is a
geometric consequence of the half-filling phase relationship between
$\chi^{(1)}$ on site $j$ and $\chi^{(2)}$ on site $j+1$ — these two
Majoranas, despite living on adjacent sites, both carry the same
chirality after bosonization.

This microscopic chirality is consistent with the algebraic finding
$\alpha_R = 0$ in §1.1.

---

## 2. Self-OPE of a chiral vertex

For a purely left-moving vertex $V(\bar z) = \exp(i\alpha\, \phi_L(\bar z))$
(with $\bar z = x + i\tau$ in Euclidean signature, $\phi_L$ depending only
on $\bar z$), the self-OPE is
$$
V(\bar z_1)\, V^\dagger(\bar z_2)
\;=\;
\bar z_{12}^{-\alpha^2/(2K)}\,
:\exp\!\bigl[i\alpha(\phi_L(\bar z_1) - \phi_L(\bar z_2))\bigr]:
$$
where $\bar z_{12} = \bar z_1 - \bar z_2$. The expansion of the
normal-ordered exponential to second order in $\bar z_{12}$ gives
$$
V(\bar z_1) V^\dagger(\bar z_2) \;=\; \bar z_{12}^{-\alpha^2/(2K)}\bigl[1 + i\alpha\,\bar z_{12}\,\bar\partial \phi_L + \tfrac{1}{2}(i\alpha)^2 \bar z_{12}^2 (\bar\partial \phi_L)^2 + \ldots\bigr].
$$

**The only operators appearing in this OPE are derivatives $\bar\partial^n \phi_L$ of
the left-moving field.** No right-moving operators $\partial^n \phi_R$
appear.

This is a consequence of the chiral algebra: products of left-moving
operators are still left-moving.

### 2.1 RG corrections from a chiral perturbation

The bosonized action with the chiral vertex perturbation is
$$
S \;=\; \frac{1}{4\pi K}\!\int d^2x\,\bigl[\,u_R\,(\partial \phi_R)^2 + u_L\,(\bar\partial \phi_L)^2\,\bigr]
\;-\; g\!\int d^2x\,\bigl[V(\bar z) + V^\dagger(\bar z)\bigr],
$$
where I have allowed for independent right- and left-mover velocities
$u_R, u_L$ (initially equal in the absence of the perturbation, with
$u_R = u_L = u$ and the Luttinger parameter $K$ relating velocity and
stiffness in the usual way).

Second-order in $g$, the effective action correction is
$$
\delta S_{\rm eff}
\;=\;
-\frac{g^2}{2}\int d^2 x_1\, d^2 x_2\,\bar z_{12}^{-\alpha^2/(2K)}\,
\sum_n \frac{(i\alpha\,\bar z_{12})^n}{n!}\,(\bar\partial \phi_L)^n(x_2),
$$
plus the (analogous) complex conjugate. **Every term in this sum is
purely left-moving** — only $\bar\partial \phi_L$ appears, never
$\partial \phi_R$.

### 2.2 What gets renormalized

The term at order $n = 2$ in the OPE generates an operator
$(\bar\partial \phi_L)^2$, which has the same form as the bare kinetic
term for $\phi_L$. Matching coefficients in the renormalized action:
$$
\frac{u_L^{\rm eff}}{4\pi K} \;=\; \frac{u_L}{4\pi K} \;+\; \text{(positive coefficient)}\,g^2 + O(g^4),
$$
so the left-mover velocity $u_L$ shifts under RG, but $u_R$ does not.

The Luttinger parameter $K$ is determined by the ratio of stiffnesses
of $\phi$ and $\theta$, which in chiral language is symmetric in
$u_R \leftrightarrow u_L$. Specifically, with
$\phi_{R,L} = (\phi \mp \theta)/2$:
- The $\phi$-stiffness is $u_R + u_L$ (sum).
- The $\theta$-stiffness is $u_R + u_L$ (sum, same form by parity).
- Their **ratio** is 1 regardless of how $u_R, u_L$ shift independently.

The "Luttinger parameter" $K$ in the original $(\phi, \theta)$ basis,
defined by the ratio of stiffnesses, is thus **invariant** under the
chiral perturbation, even though both $u_R$ and $u_L$ shift.

In fact: only the *average* velocity $u = (u_R + u_L)/2$ and the
*chirality breaking* $\Delta u = u_R - u_L$ are renormalized.
$\Delta u$ shifts at order $g^2$ and is what makes the system
"chiral" at finite $(1-\zeta)$. The renormalized $K$ is invariant.

### 2.3 To all orders in $g$

The argument generalizes to all orders. At any order $n$ in $g$, the
$n$-fold OPE of the chiral vertex with itself produces only chiral
operators (no $\partial \phi_R$). These contribute only to $u_L$
renormalization (and possibly to other left-moving relevant or
irrelevant chiral operators). The non-chiral $K$ remains exactly
fixed at its bare value at every order.

This is a non-perturbative statement (about the structure of OPE
algebras, not about convergence of any specific series).

---

## 3. Consequence for $c_{\rm eff}$

The entanglement entropy of a 1+1D CFT with central charge $c$ scales as
$$
S_A \;=\; \frac{c}{6} \ln\!\left(\frac{L}{\pi a}\,\sin\!\frac{\pi x}{L}\right) + \text{const}
$$
on an open chain of length $L$ for a region $A = [0, x]$. The constant
$c$ depends on the universal CFT data — it is *not* the velocity. For
a free Dirac fermion (Luttinger liquid with $K=1$), $c = 1$ regardless
of $u_R, u_L$.

Even when chiral velocities $u_R \neq u_L$ (as in a Galilean-non-invariant
Luttinger liquid), the total central charge $c_R + c_L$ is preserved
under perturbations that don't affect $K$.

**Therefore: $c_{\rm eff}^{\rm QJ}(\zeta) = 1$ for all $\zeta$ in the
log-law phase**, within the leading bosonization, to all orders in
$(1-\zeta)$.

---

## 4. What this does NOT explain

The numerical data shows
$$
\lambda(c=1)\;:\quad
\zeta = 1.00 \to 0.364,\ \zeta = 0.85 \to 0.365,\ \zeta = 0.70 \to 0.370,
$$
which is **flat near $\zeta = 1$**, consistent with the prediction
above. But it also shows
$$
\zeta = 0.50 \to 0.334,\ \zeta = 0.30 \to 0.237,
$$
a **sharp drop** at $\zeta \lesssim 0.5$. The bosonization result above
**cannot account for this drop**, because the cross-replica vertex
doesn't renormalize $K$ at any perturbative order.

The sharp drop must come from physics outside the leading bosonization.
Most likely candidates:

  1. **Operators beyond the leading-order bosonization.** The full
     four-Choi-copy theory has many vertex operators, not just $V_j$.
     A systematic enumeration could identify an operator that becomes
     relevant at strong PPS and drives the area-law transition at
     $\zeta \lesssim 0.5$. The most likely candidates are **higher-product
     vertices** (e.g., products $V_j V_{j+1}$ at coincident points, which
     could have lower scaling dimension after OPE contractions) or
     **density-density operators** in the $\sigma_D$ mode that I have
     not analyzed.

  2. **Non-perturbative crossover from the non-Hermitian dynamics.**
     The diagonal non-Hermitian generator $\mathcal{L}_{\rm diag}^{(2)}$
     contains the deterministic decay $-i \alpha/2 \sum_j L_j^\dagger L_j$.
     At small $\zeta$ (suppressed click rate) this decay dominates and
     drives the system to a specific non-Hermitian steady state. The
     bosonized theory linearises around the Born-rule fixed point and
     misses this strong-coupling regime entirely.

  3. **Finite-$L$ crossover.** The drop in $\lambda(c=1)$ at small $\zeta$
     is the *finite-size* signature of a thermodynamic-limit transition.
     In the $L \to \infty$ limit, the transition is presumably sharp at
     some $\zeta_c$, but at our $L \in [32, 128]$ it appears smeared
     over $\zeta \in [0.3, 0.7]$. A proper finite-size scaling collapse
     would identify $\zeta_c$ unambiguously.

I have not been able to compute the strong-PPS transition analytically.
Some scaling arguments are possible (see §5), but a full RG analysis in
the spirit of LMR's Section V C for the QJ vertex is genuinely
weeks-of-work and beyond the scope of what I can produce here.

---

## 5. Scaling estimate for the strong-PPS crossover

A heuristic estimate: the strong-PPS crossover happens when the click
rate $\zeta\,\alpha\,\langle P_j\rangle$ becomes comparable to the
relaxation rate of the deterministic non-Hermitian dynamics. The latter
is set by the imaginary spectrum of the single-particle Majorana matrix
$\mathcal{H}_{\rm eff} = \mathcal{H} - i \alpha\,\mathcal{P}$.

Numerical evaluation (using
`pps_qj/gaussian_backend.py::effective_generator`):
- At $\lambda = 0.364$ (the Born-rule critical $\lambda$): the smallest
  $|\text{Im}\,E|$ for $L=128$ is $\sim 7 \times 10^{-5}$, essentially zero.
- At smaller $\lambda \sim 0.05$: $|\text{Im}\,E_{\rm min}| \sim 0.013$,
  comparable to $\lambda/4$.

The single-particle gap is too small (essentially $0$ at large $L$ across
the critical phase) to set the relevant scale. The right scale is
presumably set by *many-body* properties — e.g., the rate at which
information is destroyed by the deterministic non-Hermitian dynamics on
collective scales.

A more refined estimate, beyond what I can derive cleanly: the
many-body relaxation rate of the no-click dynamics on length scale $\ell$
is something like $\alpha \cdot \ell^{-\eta}$ for some $\eta$. The
crossover happens when $\zeta\alpha \sim \alpha \ell^{-\eta}$, i.e.,
$\ell \sim \zeta^{-1/\eta}$. For the log-law phase to survive at scale
$\ell$, we need $\zeta \gtrsim \ell^{-\eta}$, which translates to a
$\lambda$-dependent crossover.

This is plausible but not a proof. The data shows $\zeta_c \sim 0.5\text{–}0.7$
for $L = 128$, which would correspond to $\eta \sim 1$ and a length
scale $\ell^{-\eta} \sim 1/128 \approx 0.008$ — much smaller than the
observed $\zeta_c$. So the simple scaling argument doesn't work either.

The honest summary: **the strong-PPS regime is non-perturbative in QJ
and requires a more careful analysis that I have not been able to
complete.** What I have established is:

  - The cross-replica vertex is chiral → $K$ doesn't renormalize from
    perturbative bosonization. This explains the flat region $\zeta \in [0.7, 1]$.
  - The sharp drop at $\zeta \lesssim 0.5$ is non-perturbative and
    requires operators or physics outside the leading bosonization.

---

## 6. Comparison with LMR's diffusive case

LMR's diffusive cross-replica term contains $\cos(2\phi_\sigma)$, which
has $\alpha_R = 2$, $\alpha_L = 2$ — **both chiralities present**. Such
a non-chiral vertex has self-OPE producing both $\partial\phi_R$ and
$\bar\partial\phi_L$, which renormalize $K$ at leading order in $g^2$.

This is the structural reason LMR find a continuously varying $K(\zeta)$
across the entire $\zeta$ range, including the eventual divergence at
$\zeta^* \approx 0.28$ that signals the Ising–to–BKT phase boundary.

Our QJ vertex is chiral, so the analog continuous $K(\zeta)$ doesn't
exist — $K$ stays at 1 over the entire perturbative regime. The
analog of LMR's $\zeta^*$ phase boundary in QJ must come from a
non-perturbative mechanism (and is what we observe as the sharp drop
in $\lambda(c=1)$ at $\zeta \sim 0.5$).

This is a **qualitative difference** between the two unravellings:

| Feature | LMR diffusive | QJ |
|---------|---------------|-----|
| Cross-replica vertex | $\cos(2\phi_\sigma)$ (non-chiral, marginal) | $\exp(-4i\phi_L^{\rho_D})$ (chiral, irrelevant) |
| $K(\zeta)$ from bosonization | Smoothly varies | Stays at $K(\zeta=1)$ |
| Critical line $\lambda(c=1)$ | Smooth $\zeta$-dependence | Flat at $\zeta \gtrsim 0.7$, then sharp drop |
| Phase boundary $\zeta^*$ | $\approx 0.28$, perturbatively accessible | $\approx 0.5\text{–}0.7$, non-perturbative |

---

## 7. Falsifiable predictions for the Renyi reruns

If the chirality result above is correct, the Renyi entropy ratios
should satisfy the free-Dirac CFT predictions
$$
c_2/c_1 = 3/4,
\qquad
c_3/c_1 = 2/3,
$$
at **every test point in the log-law phase**, including the
strong-PPS points $(\lambda = 0.10, \zeta = 0.20)$ and
$(\lambda = 0.325, \zeta = 0.50)$.

Possible outcomes:
  - **All six test points give CFT ratios within statistical error.**
    This confirms the chirality picture: the log-law phase has a
    single conformal description throughout the $(\lambda, \zeta)$
    plane, with $c=1$ and free-Dirac universality.
  - **Strong-PPS points $(\lambda = 0.10, \zeta = 0.20)$ and
    $(\lambda = 0.325, \zeta = 0.50)$ deviate from CFT ratios.** This
    falsifies the chirality picture as the *complete* description of
    the strong-PPS regime, and indicates that the non-perturbative
    operator(s) driving the area-law transition also break conformal
    invariance.
  - **Some test points show critical CFT, others area-law.** This is
    consistent with a phase boundary inside the $(\lambda, \zeta)$
    plane separating the bosonization-described log-law phase from a
    non-perturbative area-law/strong-PPS regime.

I expect the third outcome based on the existing $c_{\rm eff}$ data.
But the Renyi ratios will give a much cleaner diagnostic than the
$c_{\rm eff}$ values alone.

---

## 8. Honest summary

What's been established rigorously in this document:

  1. The bond annihilation operator $d_j$ at half-filling is a pure
     left-mover after bosonization (geometric: phase relationship between
     adjacent Majoranas). ✓
  2. The cross-replica click vertex $V_j$ in the four-Choi-copy theory
     is therefore a pure left-moving exponential of the inter-replica
     $\rho_D$ mode. ✓
  3. Self-OPE of a chiral vertex produces only chiral operators, which
     renormalize chiral velocity but not the Luttinger parameter $K$. ✓
  4. Therefore $c_{\rm eff}^{\rm QJ}(\zeta) = c_{\rm eff}^{\rm QJ}(\zeta = 1)$
     to all orders in $(1-\zeta)$, within the leading bosonization. ✓

What's not been established:

  - The mechanism driving the sharp drop in $\lambda(c=1)$ at
    $\zeta \lesssim 0.5$. The chirality argument explains the absence
    of perturbative shift but not the non-perturbative crossover. ✗
  - Whether the chirality persists if we include higher-derivative
    corrections to the bosonization (away from the strict half-filling
    continuum limit). ✗
  - The Klein-factor structure across Choi copies. I have continued the
    standard LMR assumption that Klein factors are global phases that
    don't affect local scaling dimensions. ✗

The all-orders cancellation result is the most striking new outcome
of this calculation. It implies a much cleaner picture of the
$\zeta \to 1$ regime than I had previously and gives a sharp numerical
test (Renyi ratios across all test points).

The strong-PPS regime remains an open problem.
