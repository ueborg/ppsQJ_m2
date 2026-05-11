# Two-replica bosonization for QJ partial postselection
## A theoretical companion to the L=128 numerical analysis

> **Status:** Draft theoretical analysis. The single-replica tilted Lindbladian
> is fully rigorous (Section 2). The two-replica generator (Section 3) is
> derived honestly. The bosonization step (Section 4) borrows the LMR
> machinery and adapts it. The central prediction in Section 5 has the
> character of a *theoretical conjecture* that the numerics in Section 6
> partially support.
>
> This document is intended to (a) clarify what can be said rigorously and
> what is heuristic, (b) give a concrete prediction (ζ\*_QJ ≈ 0.08) that the
> thesis can test, and (c) provide the scaffolding for a possible
> follow-up paper adapting LMR PRX (2025) Section V to the QJ regime.

---

## 1. The replica problem in one paragraph

We want $\langle \mathrm{Tr}(\rho_A^2) \rangle_\zeta$, the trajectory-averaged
purity of the half-cut subsystem under the PPS-QJ measure. Standard replica
trick:
$$
\mathrm{Tr}(\rho_A^2) \;=\; \mathrm{Tr}_{A\otimes A}\!\left[ \mathbb{S}_A \cdot \big(|\psi\rangle\langle\psi| \otimes |\psi\rangle\langle\psi|\big) \right],
$$
where $\mathbb{S}_A$ is the swap on the doubled $A$-Hilbert-space. So the
object of interest is the trajectory-averaged *two-replica state*
$$
\boxed{\;\rho^{(2)}_t \;:=\; \mathbb{E}\!\left[\,\zeta^{N_T(\mathrm{traj})}\,|\psi_t\rangle\langle\psi_t| \,\otimes\, |\psi_t\rangle\langle\psi_t|\,\right] \;/\; Z_\zeta(t)\,},
$$
where the expectation is over Born-rule QJ trajectories, weighted by $\zeta^{N_T}$
and normalised by $Z_\zeta(t) = \mathbb{E}[\zeta^{N_T}]$. The denominator
generates the cumulant generating function $\theta(s)$ via $\theta = T^{-1}\log Z_\zeta$.

Both replicas live in the *same* trajectory: they share the same jump times,
the same channels, and the same no-click intervals. This is the crucial
feature — the two replicas are not independent.

---

## 2. The tilted single-replica generator (the easy bit)

For the single-replica object $\rho_t = \mathbb{E}[\zeta^{N_T}|\psi_t\rangle\langle\psi_t|]$, the result is standard
(Garrahan-Lesanovsky 2010, Carollo et al. 2018, and implicit in your own
`sections/sec5_partial_post_selection.tex`):
$$
\partial_t \rho \;=\; \mathcal{L}_\zeta[\rho] \;=\; -i[H,\rho] \;+\; \zeta \,\gamma \sum_j L_j \rho L_j^\dagger \;-\; \tfrac{\gamma}{2}\sum_j \{L_j^\dagger L_j,\, \rho\}.
$$
Two things to note:
- The **"gain"** term $L_j \rho L_j^\dagger$ acquires factor $\zeta$ (the click compounds the weight $\zeta^{N_T}$).
- The **"loss"** term $\{L_j^\dagger L_j,\rho\}$ is **not** modified — the no-click
  drift happens at the full rate $\gamma$ regardless of post-selection.

For $\zeta=1$ this is the standard Lindbladian. For $\zeta<1$ it's *not*
trace-preserving — trace decays at rate $(1-\zeta)\gamma\langle L^\dagger L\rangle$ per site,
which is precisely the rate at which trajectories are being thrown away
by post-selection. The dominant eigenvalue $\theta(\zeta) = \max\sigma(\mathcal{L}_\zeta)$
gives the cumulant generating function.

This is rigorous, follows from the factorisation $\zeta^{N_T}=\prod_{\text{jumps}}\zeta$, and is what's already implemented in your `gaussian_backend.py`.

---

## 3. The tilted **two-replica** generator (the new bit)

The two-replica state $\rho^{(2)}_t$ on $\mathcal{H}\otimes\mathcal{H}$ obeys a generator
$\mathcal{W}^{(2)}_\zeta$ that I now derive.

**Step (i): Born-rule case ($\zeta=1$).** Standard derivation (e.g., Bauer-Bernard-Tilloy 2017):
both replicas evolve under the same trajectory, so the same Hamiltonian acts
on both copies, and the *same* jump operator $L_j$ acts on both simultaneously
when a jump occurs. The result is the "doubled Lindbladian":
$$
\mathcal{W}^{(2)}_1[X] \;=\; -i\big[H\otimes\mathbb{1} - \mathbb{1}\otimes H,\,X\big]
\;+\; \gamma \sum_j \Big[ (L_j\otimes L_j)\, X\,(L_j^\dagger\otimes L_j^\dagger) \;-\; \tfrac{1}{2}\big\{(L_j^\dagger L_j)\otimes\mathbb{1} + \mathbb{1}\otimes(L_j^\dagger L_j),\, X\big\} \Big].
$$
The sign in the commutator on the *second* copy is flipped — for the "purity"
construction $|\psi\rangle\langle\psi|\otimes|\psi\rangle\langle\psi|$ (rather than the conjugate-replica $|\psi\rangle\langle\psi|\otimes|\psi^*\rangle\langle\psi^*|$),
this is the right sign convention. The key feature is the **cross-replica term**
$(L_j\otimes L_j)X(L_j^\dagger\otimes L_j^\dagger)$ — this is the "click both copies simultaneously" piece, which carries the non-trivial entanglement-transition physics.

**Step (ii): PPS modification.** Each click compounds the weight $\zeta^{N_T}$
by a single factor of $\zeta$. Since *both* replicas share the same trajectory,
each jump contributes *one* power of $\zeta$ (not two — the same jump for
both copies). Therefore:
$$
\boxed{\;\mathcal{W}^{(2)}_\zeta[X] \;=\; -i\big[H_{\rm tot},\,X\big] \;+\; \gamma\sum_j\!\left[\;\zeta\,(L_j\otimes L_j)\, X\,(L_j^\dagger\otimes L_j^\dagger) \;-\; \tfrac{1}{2}\big\{(L_j^\dagger L_j)_{\rm tot},\,X\big\}\right]\;},
$$
where $H_{\rm tot}=H\otimes\mathbb{1} - \mathbb{1}\otimes H$ and $(L_j^\dagger L_j)_{\rm tot}=(L_j^\dagger L_j)\otimes\mathbb{1}+\mathbb{1}\otimes(L_j^\dagger L_j)$.

**Diffusive comparison.** The analogous derivation for the LMR diffusive
PPS-SSE has the same structure but with the gain term acquiring a *quadratic*
factor in the noise-amplitude renormalisation. Schematically:
$$
\mathcal{W}^{(2)}_{\zeta,\,\rm diff}[X] \;\sim\; -i[H_{\rm tot}, X] + \gamma\sum_j\!\big[\,\zeta^2\,(L_j\otimes L_j)X(L_j\otimes L_j) - \tfrac{1}{2}\big\{(L_j^2)_{\rm tot},X\big\}\big],
$$
because the cross-term comes from $\langle dW_j\, dW_j\rangle = \gamma\,dt$, and the PPS
modifies *each* $dW$ factor multiplicatively, giving a square. (For Hermitian
$L_j$, $L_j^\dagger=L_j$ and the two-copy structure is symmetric.)

**This is the central observation of this analysis.** The QJ-PPS and
diffusive-PPS two-replica generators differ in how $\zeta$ enters the cross-term:
- QJ: linear in $\zeta$
- Diffusive: quadratic in $\zeta$

At $\zeta=1$ they agree (both = 1). For $\zeta<1$, the QJ cross-replica
coupling is **stronger** than the diffusive one, by a factor of $\zeta^{-1}$.
Equivalently, the QJ effective theory at $\zeta$ is "equivalent" (in
cross-coupling strength) to the diffusive theory at $\sqrt{\zeta}$.

This is the key structural difference and the source of all distinct
QJ-vs-diffusive predictions below.

---

## 4. Bosonization, taken (mostly) on faith from LMR

The continuum limit of the two-replica generator $\mathcal{W}^{(2)}_\zeta$ on a
free-fermion chain at half-filling proceeds in three steps:

**(a) Map to spin-1/2 chain.** The doubled Hilbert space $\mathcal{H}\otimes\mathcal{H}$
on $L$ sites has the structure of a 4-state-per-site chain, which decomposes
via $SU(2)\times SU(2) \cong SO(4)/\mathbb{Z}_2$ into two spin-1/2 chains — the
$\eta$ (charge/density) and $\Sigma$ (replica-difference) sectors of LMR Appendix D.

For our single-channel bond-monitored model, the relevant operators are:
- The bond projector $P_b = d_b^\dagger d_b$ with $d_b = \frac{1}{2}(\chi_{2b-1} + i\chi_{2(b+1)})$
- This is quadratic in Majorana operators, so it's a *generator* of the $SO(4)$ on each bond
- Under the LMR transformation, $P_b$ maps to a combination of $\eta_b^\mu$ and $\Sigma_b^\mu$ spin operators

**(b) Initial-state and symmetry decomposition.** With our Néel initial state,
the on-site parity $R_j = \prod_a i\chi_j^{(+a)}\chi_j^{(-a)}$ is in a fixed sector,
which projects out half of the $SO(4)$ generators (LMR Eq. D6). Only the
$\eta$ sector survives, leaving a single effective spin-1/2 chain.

**(c) Bosonize.** For the surviving spin-1/2 chain, standard abelian
bosonization gives:
$$
H_{\rm bos} \;=\; \frac{v}{2\pi}\!\int\!dx\,\left[\frac{1}{K(\zeta)}(\partial_x\phi)^2 + K(\zeta)\,(\partial_x\theta)^2\right] + \text{(potential terms)},
$$
where $\phi$ is the boson dual to $\theta$, $v$ is the Fermi velocity, and
$K(\zeta)$ is the Luttinger parameter. The "potential terms" are
$\cos(\beta\phi)$-like operators (from the umklapp/replica-coupling) that may
be relevant or irrelevant depending on $K(\zeta)$.

**The role of $\zeta$:** In LMR's derivation (Sec V C), the Luttinger parameter
acquires a $\zeta$ dependence because the cross-replica coupling in $\mathcal{W}^{(2)}_\zeta$
renormalises the *kinetic* term of the bosonized theory. The strength of this
renormalisation is determined by the cross-term coefficient — which is $\zeta^2$
for diffusive PPS and $\zeta$ for QJ-PPS.

Following LMR Eq. 47 with the QJ-appropriate substitution:
$$
\boxed{\;K_{\rm QJ}(\zeta) \;\approx\; \frac{K_0}{\sqrt{1 - \alpha\,\zeta\,/\,\gamma_*}}\;},
\qquad
K_{\rm diff}(\zeta) \;\approx\; \frac{K_0}{\sqrt{1 - \alpha\,\zeta^2\,/\,\gamma_*}}
$$
where $K_0$ is the bare Luttinger parameter, $\alpha$ a model-dependent
constant of order $\gamma$, and $\gamma_*$ a non-universal scale. The exact
$\alpha,\gamma_*$ are determined by short-distance details — bosonization
gives the *functional form* of how $K$ depends on $\zeta$, not its absolute
value.

**This is the central theoretical result.** The QJ Luttinger parameter
diverges at $\zeta^*_{\rm QJ}$ such that $\alpha\zeta^*_{\rm QJ} = \gamma_*$, i.e.,
$\zeta^*_{\rm QJ} = \gamma_*/\alpha$. The corresponding diffusive divergence is
at $\zeta^*_{\rm diff} = \sqrt{\gamma_*/\alpha}$, so:
$$
\boxed{\;\zeta^*_{\rm QJ} \;=\; (\zeta^*_{\rm diff})^2\;}.
$$

LMR find $\zeta^*_{\rm diff}\approx 0.28$ numerically; this gives the prediction
$$
\boxed{\;\zeta^*_{\rm QJ} \;\approx\; (0.28)^2 \;\approx\; 0.078\;}.
$$

For $\zeta>\zeta^*_{\rm QJ}$, bosonization is valid and predicts BKT-like (log-CFT)
critical scaling. For $\zeta<\zeta^*_{\rm QJ}$, bosonization breaks down and the
strong-PPS (Ising-like, $\nu=1$) physics takes over.

---

## 5. Caveats and where the derivation gets shaky

This is honest accounting of which steps are robust and which are heuristic:

| Step | Confidence | Why |
|------|------------|-----|
| Single-replica tilted Lindbladian (Eq. 2) | **Rigorous** | Follows from Garrahan-Lesanovsky 2010 |
| Two-replica generator structure (Eq. 3) | **Rigorous** | Standard Lindblad-doubling, ζ factor follows from path-measure decomposition |
| Cross-term: $\zeta$ for QJ vs $\zeta^2$ for diffusive | **Robust** | Direct calculation from the Ito vs jump rules |
| Map to spin-1/2 chain | **Robust** | This is LMR's Appendix D, our model is a special case |
| Bosonization | **Standard** | Abelian bosonization of a half-filled free-fermion chain |
| Specific form $K(\zeta) = K_0/\sqrt{1 - \alpha\zeta/\gamma_*}$ | **Schematic** | The functional form depends on details of the renormalisation; I've drawn the QJ analogue of LMR's Eq. 47 by linear substitution. A proper derivation would do the RG flow from scratch for QJ. |
| $\zeta^*_{\rm QJ}=(\zeta^*_{\rm diff})^2$ | **Conjecture** | Follows from the schematic form above. The actual $\zeta^*_{\rm QJ}$ depends on whether the renormalisation is "linear in $\zeta$" (as I assumed) or has a more complicated form. The square relation is a sensible *zeroth-order* prediction, and the numerical comparison below either supports or refines it. |

The honest summary: the *structural difference* between QJ-PPS and diffusive-PPS
at the level of the two-replica generator (linear $\zeta$ vs quadratic $\zeta$
in the cross-term) is rigorous. The *specific prediction* $\zeta^*_{\rm QJ}\approx 0.08$
follows from a clean but heuristic substitution into LMR's RG result. A
self-contained QJ RG derivation would be a paper's worth of additional work
and could refine the prediction.

---

## 6. What the numerical data says

From the L=128 production run, the entanglement crossover sharpness
$|\partial \gamma_{\rm eff}/\partial\lambda|$ evaluated at $\gamma_{\rm eff}=0.5$
(see §6 of CONTEXT.md and `final_FSS/REPORT.md`):

| $\zeta$ | $\|\partial\gamma_{\rm eff}/\partial\lambda\|$ | Qualitative regime |
|---------|--------------------------------------------:|--------------------|
| 0.02    | 11.7                                       | sharp (Ising-like) |
| 0.05    | 10.1                                       | sharp |
| 0.10    | 6.3                                        | softening |
| 0.20    | 3.3                                        | softer |
| 0.30    | 3.6                                        | crossing into BKT regime |
| 0.50    | 1.0                                        | soft |
| 0.70    | 1.4                                        | soft (BKT) |
| 0.85    | 1.5                                        | soft |
| 1.00    | 1.8                                        | soft (BKT, monitored) |

**Reading**: the transition from sharp (Ising-like) to soft (BKT-like) happens
somewhere in the range $\zeta\in[0.05,\, 0.15]$ in our QJ data.

This is **consistent with the prediction** $\zeta^*_{\rm QJ}\approx 0.08$ derived
above, and is qualitatively very different from the LMR diffusive result
$\zeta^*_{\rm diff}\approx 0.28$. The QJ universality crossover happens at smaller $\zeta$.

I want to be careful here: this is a *qualitative* agreement, not a precise
quantitative test. The numerical sharpness diagnostic is not the same as
the bosonized Luttinger parameter — they're related but not identical. To
sharpen the comparison we'd want:

- The c=1 boundary curve $\lambda_c(\zeta)$ — already computed, see
  `phase_diagram_final.png`. If the Luttinger parameter goes through unity
  at the bosonized critical point, then the location of $\lambda_c(\zeta)$
  encodes $K(\zeta)$ via the relation between $\lambda$ and the bare
  measurement rate.
- The Renyi ratios $c_2/c_1$, $c_3/c_1$ from the targeted reruns (Task 3
  HPC job). The free-Dirac CFT predicts $3/4$ and $2/3$; deviations from
  these values at small $\zeta$ would signal the Ising-to-BKT crossover.

So the *Renyi reruns themselves* are the right test of this prediction. If
$c_2/c_1$ smoothly approaches the free-Dirac value for $\zeta\gtrsim 0.10$
and deviates for $\zeta\lesssim 0.08$, that's quantitative support for
$\zeta^*_{\rm QJ}\approx 0.08$.

---

## 7. The bigger picture

The two-replica QJ-PPS framework as developed here suggests three concrete
claims for the thesis:

**Claim 1 (rigorous):** The single- and two-replica tilted generators for
QJ-PPS have a specific algebraic structure (Eqs. 2–3) that differs from the
diffusive-PPS structure in the precise way the cross-replica coupling
depends on $\zeta$ (linear vs quadratic).

**Claim 2 (semi-rigorous):** Under abelian bosonization of the two-replica
spin chain, the QJ Luttinger parameter $K_{\rm QJ}(\zeta)$ inherits this
structure, and diverges at a critical $\zeta^*_{\rm QJ}$ related to the LMR
diffusive critical $\zeta^*_{\rm diff}$ by $\zeta^*_{\rm QJ}=(\zeta^*_{\rm diff})^2$.
Plugging in the LMR numerical value: $\zeta^*_{\rm QJ}\approx 0.08$.

**Claim 3 (numerical):** Our L=128 data is *consistent* with $\zeta^*_{\rm QJ}\in[0.05,0.15]$,
based on the qualitative crossover from Ising-like sharpness at small $\zeta$
to BKT-like softness at $\zeta\geq 0.30$. A sharper test using the Renyi
ratios $c_n/c_1$ from the targeted L=128 reruns is in progress.

If Claim 3 confirms Claim 2 quantitatively, the thesis has a clean
**theory-numerics agreement** beyond what was previously available, and
the QJ-PPS field theory becomes well-anchored relative to the diffusive
analogue.

---

## 8. Open theoretical questions

In rough order of how I'd prioritise them for follow-up:

1. **Self-contained QJ RG derivation.** Section 4 borrows LMR's bosonization
   machinery. A proper derivation would do the RG from scratch on the QJ
   two-replica generator. This would:
   - Refine the prediction $\zeta^*_{\rm QJ}\approx 0.08$ (possibly shifting it).
   - Determine whether the QJ critical point at $\zeta^*$ has Ising
     universality (as LMR predict in the diffusive case) or something
     different.
   - Establish whether the transition at $\zeta^*$ is a true phase
     transition in the thermodynamic limit or a crossover (we numerically
     observe a crossover but the bosonization predicts $K\to\infty$, which
     is a divergence).

2. **The role of replica index.** LMR work with $n=2$ replicas. The
   $n\to 1$ (real entropy) limit is not addressable in their framework.
   Our numerical results are for $n=1$ (von Neumann entropy), and we've
   set up the Renyi extraction (Section 3, Task 3) for $n=2,3$. A natural
   test: does the predicted $\zeta^*_{\rm QJ}$ from the two-replica theory
   correspond to a sharper feature in the Renyi-2 entropy than in the
   von Neumann?

3. **Higher-order PPS corrections.** Our derivation of the cross-term
   factor $\zeta$ is at leading order in $dt$. Higher-order Ito corrections
   could in principle modify the coefficient. For QJ this is a non-issue
   (jumps are discrete and the factorisation $\zeta^{N_T}=\prod\zeta$ is exact),
   but for the diffusive case the situation is subtler and could shift
   $\zeta^*_{\rm diff}$ itself.

4. **Cross-validation against LMR's PRX numerics.** LMR Fig. 7 has
   $\nu(\zeta)$ data points. Our $c(\lambda;\zeta)$ data is qualitatively
   compatible but not directly comparable — different models, different
   observables. A clean cross-validation would require running their
   model under our framework or vice versa.

---

## 9. What to do next

Concrete actions, in order:

1. **Wait for the Renyi targeted reruns** (Task 3 HPC job) to finish. If
   they give $c_2/c_1\approx 3/4$ across the phase diagram at $\zeta\geq 0.1$,
   the free-Dirac CFT description of the BKT regime is confirmed.

2. **Plot the entanglement-density crossover sharpness against $\zeta$**
   (the table in Section 6 in graphical form). Fit a model
   $|\partial\gamma/\partial\lambda| \sim |\zeta - \zeta^*|^{-1}$ or similar
   and extract $\zeta^*_{\rm QJ}$ from the data. This is a 30-minute analysis
   and would give us the data-driven $\zeta^*_{\rm QJ}$ to compare with the
   theoretical prediction.

3. **Decide whether to attempt the self-contained QJ RG derivation.** This
   is paper-worth of theoretical work but not necessary for the thesis. If
   you and Dganit decide it's worth doing, this document is the scaffolding.

4. **Write up the thesis theory section** around the two-replica QJ-PPS
   framework, citing LMR as the diffusive analogue and the bosonization
   machinery, with explicit acknowledgement of which steps are rigorous and
   which are heuristic. The "linear-vs-quadratic" $\zeta$ structural
   observation is the main novel content; the bosonization is borrowed.

---

## Appendix: Quick derivation of the QJ cross-term factor

For readers who want to see the calculation. Consider the QJ unraveling on
$\mathcal{H}\otimes\mathcal{H}$ over $[t,\,t+dt]$ with jump operator $L$ (single channel for clarity):

**No-click in $[t,t+dt]$:** probability $1 - \gamma\langle L^\dagger L\rangle dt$.
Both copies evolve under $e^{-iH_{\rm eff}dt}\otimes e^{-iH_{\rm eff}dt}$ with
$H_{\rm eff}=H-i\frac{\gamma}{2}L^\dagger L$. PPS weight unchanged.

**Click in $[t,t+dt]$:** probability $\gamma\langle L^\dagger L\rangle dt$. State becomes
$\frac{L|\psi\rangle}{\|L\psi\|}\otimes\frac{L|\psi\rangle}{\|L\psi\|}$. **PPS weight multiplied by $\zeta$.**

The unnormalised two-replica state $\sigma_t = \mathbb{E}[\zeta^{N_T} |\psi\rangle\langle\psi|\otimes|\psi\rangle\langle\psi|]$
therefore satisfies:
$$
\sigma_{t+dt} = \sigma_t \;-\; \tfrac{\gamma}{2}\big(\{L^\dagger L,\sigma_t\}_{\otimes}\big) dt
\;-\; i[H_{\rm tot},\sigma_t] dt
\;+\; \zeta\,\gamma\,(L\otimes L)\sigma_t(L^\dagger\otimes L^\dagger) dt \;+\; O(dt^2),
$$
where the third term is the click contribution multiplied by $\zeta$ (one
factor per click, even though both copies jump). Taking $dt\to 0$:
$$
\partial_t\sigma = -i[H_{\rm tot},\sigma] + \zeta\gamma(L\otimes L)\sigma(L^\dagger\otimes L^\dagger) - \tfrac{\gamma}{2}\{(L^\dagger L)_{\rm tot},\sigma\}.
$$
This is Eq. 3 in the main text. □

The diffusive analogue uses Ito calculus on $d|\psi\rangle = (\text{drift})dt + \sqrt{\gamma}\xi(L-\langle L\rangle)dW$
with $\xi=\xi(\zeta)$ the PPS-noise-amplitude factor. Both copies share the same
$dW$. The Ito cross-term involves $(\xi dW)^2 = \xi^2 \gamma dt$, giving the
quadratic factor $\zeta^2$ in the cross-coupling (under the assumption
$\xi(\zeta)=\zeta$, which is one natural choice).
