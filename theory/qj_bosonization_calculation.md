# Explicit bosonization and $(1-\zeta)$ expansion for the QJ two-replica generator

This is a continuation of `qj_two_replica_derivation.md`, attempting an
explicit bosonization calculation of the QJ cross-replica vertex and a
leading-order prediction for how the Luttinger parameter depends on
$(1 - \zeta)$.

**Caveats up front.** This is a serious calculation that I will attempt to do
carefully. Where I make assumptions, I will flag them. Where I cut corners,
I will say so. The goal is to get to a **numerically falsifiable prediction**
for $c_{\rm eff}(\lambda; \zeta)$ that we can compare against the data.

---

## 1. Recap and bosonization conventions

### 1.1 The model on one chain

We have a single 1D Kitaev-like chain with:
$$
H = w \sum_j \bigl(c_j^\dagger c_{j+1} + \text{h.c.}\bigr),
\qquad
L_j = \sqrt{\alpha}\, d_j,
\qquad
d_j = \tfrac{1}{2}\!\bigl[(c_j + c_j^\dagger) + i(c_{j+1} - c_{j+1}^\dagger)\bigr].
$$

In Majorana variables $\chi_{2j-1} = c_j + c_j^\dagger$,
$\chi_{2j} = -i(c_j - c_j^\dagger)$, we have
$$
d_j = \tfrac{1}{2}(\chi_{2j-1} - i\chi_{2(j+1)}),
\qquad
P_j \equiv d_j^\dagger d_j = \tfrac{1}{2} + \tfrac{i}{4}\,\chi_{2j-1}\chi_{2(j+1)}.
$$
So $P_j$ is bilinear in Majoranas — *not* quartic — because the two
Majoranas involved are on different sites.

This is significant: our $L_j^\dagger L_j = \alpha P_j$ is a **Majorana
bilinear**, the same structural type as LMR's measurement. So the
diagonal-in-replica generator is Gaussian (free) in Majoranas, just like
LMR's.

### 1.2 Bosonization at half filling

At half filling and weak coupling, linearise around $k_F = \pi/2$ and write
$$
c_j \sim e^{ik_F j}\,\psi_R(x_j) + e^{-ik_F j}\,\psi_L(x_j)
$$
where $x_j = j$ and $\psi_{R,L}$ are slowly varying. The bosonization rules
(Giamarchi's conventions) are
$$
\psi_{R,L}(x) = \frac{1}{\sqrt{2\pi a}}\,\eta_{R,L}\,e^{\pm i k_F x}\,e^{-i(\phi(x) \mp \theta(x))}
$$
where $\eta_{R,L}$ are Klein factors, $a$ is the UV cutoff, and the boson
fields obey $[\phi(x), \partial_y \theta(y)] = i\pi\,\delta(x-y)$.

The free Hamiltonian for our $H$ alone (without measurements) bosonizes to
$$
H_{\rm kin} = \frac{v_F}{2\pi} \int dx\, \bigl[(\partial_x \theta)^2 + (\partial_x \phi)^2\bigr]
$$
with $v_F = 2w$ and Luttinger parameter $K = 1$ (free Dirac fermions).

### 1.3 The measurement operator in bosonized form

We need $\chi_{2j-1}\chi_{2(j+1)}$ in bosonized form. In Dirac language,
$$
\chi_{2j-1} = c_j + c_j^\dagger,
\qquad
\chi_{2(j+1)} = -i(c_{j+1} - c_{j+1}^\dagger).
$$
So
$$
\chi_{2j-1}\chi_{2(j+1)} = -i(c_j + c_j^\dagger)(c_{j+1} - c_{j+1}^\dagger).
$$
Expanding and using $c_{j+1} = c_j + a \partial_x c + O(a^2)$ this is at
leading order a current-like operator:
$$
\chi_{2j-1}\chi_{2(j+1)} \approx -i(c_j^\dagger c_{j+1} - c_{j+1}^\dagger c_j) + (\text{pair terms}).
$$
The first piece is the current density $J(x)$; the pair terms
$c_j c_{j+1}$ and $c_j^\dagger c_{j+1}^\dagger$ couple particle-hole sectors.

At half-filling and away from any Cooper-pair-relevant fixed point, the
current piece dominates and bosonizes (Giamarchi Eq. 2.45) as
$$
J(x) = -\tfrac{1}{\pi}\,\partial_x \phi(x) + (\text{oscillating } 2k_F \text{ terms}).
$$
The pair terms produce $\Delta(x) \sim \cos(2\phi(x))$ in standard
conventions but with a $2k_F$ phase factor that **oscillates at
half filling** ($e^{2ik_F j} = e^{i\pi j} = (-1)^j$), so they are
**dimerization-like** and average to zero in the disorder-free, unrenormalized
theory.

Putting this together:
$$
P_j = \tfrac{1}{2} - \tfrac{1}{4\pi}\,\partial_x \phi(x_j) + (\text{oscillating}).
$$

The non-Hermitian decay term in the diagonal generator is then
$$
-\tfrac{i\alpha}{2}\sum_j P_j = -\tfrac{i\alpha L}{4} + \tfrac{i\alpha}{8\pi}\int dx\,\partial_x\phi(x) + (\text{osc.}).
$$
The first term is a global constant (drops). The second is a topological
phase that integrates to a boundary term. **The leading bulk effect is
absent at this order** — which is consistent with the known fact that for
this specific measurement type, the leading non-Hermitian shift to the
Luttinger parameter comes from the quartic-in-fermion piece of $P_j^2$
acting through Hartree-Fock-type renormalization, not from $P_j$ itself.

This is going to make the leading-order Luttinger parameter shift from a
single-chain monitoring of order $\alpha^2$, not $\alpha$. Let me return to
this.

### 1.4 Quadratic-in-measurement renormalization

The Lindbladian has both a $-i[H_{\rm eff} - H_{\rm eff}^\dagger, \cdot]$
piece and a "click" piece $L \rho L^\dagger$. The combination gives, when
averaged into a single-chain Liouvillian, the standard double-commutator:
$$
\mathcal{L}_1[\rho] = -i[H, \rho] - \tfrac{\alpha}{2}\sum_j [P_j, [P_j, \rho]].
$$
The renormalization of the Luttinger parameter therefore picks up an
$\alpha^2$ contribution from $[P_j, [P_j, \cdot]]$, even when the linear
$P_j$ piece is RG-irrelevant.

Specifically, in the Keldysh action for the doubled (Choi) state, the
$[P_j, [P_j, \cdot]]$ term contains products of $\partial_x\phi$ from
different Keldysh contours, which renormalize the kinetic term coefficient.

**For the single-chain monitored fermion (CTDLA / ABD), this gives an
effective Luttinger parameter**
$$
K_1(\alpha) = \frac{1}{\sqrt{1 + \alpha/(\pi v_F)}} \cdot (\text{numerical prefactor}),
$$
where I have suppressed factors that depend on the precise UV
regularization. The key point: **$K$ decreases monotonically with $\alpha$
from $K(0) = 1$**, reaching some smaller value at finite $\alpha$.

The corresponding **logarithmic prefactor in the entanglement entropy**
(equivalent to an effective central charge) is, at half filling for a free
Dirac CFT:
$$
c_{\rm eff}^{(\zeta=1)}(\alpha) = K_1(\alpha) \cdot (\text{universal CFT factor}).
$$
With the standard CFT normalization (free Dirac $c = 1$ at $K = 1$):
$$
\boxed{\;c_{\rm eff}^{(\zeta=1)}(\alpha) \approx K_1(\alpha)\;}
$$
which is the **fact that recovers Cao–Tilloy–De Luca's result**: the
log-prefactor $c_{\rm eff}$ varies continuously with measurement strength
$\alpha$.

This is the $\zeta = 1$ baseline. Now we add PPS.

---

## 2. Two-replica generator at $\zeta = 1$: the doubled Lindbladian

For the two-replica object $\check\rho^{(2)} = \mathbb{E}[\check\rho_t \otimes \check\rho_t]$
under the Born-rule QJ unraveling, the generator is (Section 4.2 of the
companion document):
$$
\mathcal{L}_1^{(2)}[\sigma] = -i[H \otimes I + I \otimes H, \sigma]
                       + \alpha \sum_j (L_j \otimes L_j)\,\sigma\,(L_j^\dagger \otimes L_j^\dagger)
                       - \tfrac{\alpha}{2}\sum_j \{(L_j^\dagger L_j) \otimes I + I \otimes (L_j^\dagger L_j), \sigma\}.
$$
This generates the **correlated** two-replica dynamics: when a click happens,
both replicas jump simultaneously.

The combination
$$
\alpha (L_j \otimes L_j) \sigma (L_j^\dagger \otimes L_j^\dagger) - \tfrac{\alpha}{2}\{(L_j^\dagger L_j) \otimes I + I \otimes (L_j^\dagger L_j), \sigma\}
$$
can be expanded into:

  - **"Same-replica decay"** $-\tfrac{\alpha}{2}\{L_j^\dagger L_j \otimes I, \sigma\}$ and the bra-symmetric partner. These would also be present for independent replicas.
  - **"Cross-replica click"** $\alpha (L_j \otimes L_j) \sigma (L_j^\dagger \otimes L_j^\dagger)$. This is the unique piece that couples replicas.

The cross-replica piece corresponds in Choi-Jamiołkowski (four-copy) language to a
vertex
$$
V_{\rm cross} = \alpha \sum_j L_j^{(+,1)} L_j^{(+,2)} L_j^{*(-,1)} L_j^{*(-,2)}
$$
acting on the four-copy Hilbert space $\mathcal{H}^{\otimes 4}$.

For $\zeta < 1$ the same generator applies but $V_{\rm cross} \to \zeta\,V_{\rm cross}$,
because each click is weighted by $\zeta$:
$$
\boxed{\;\mathcal{L}_\zeta^{(2)} = \mathcal{L}_{\rm diag}^{(2)} + \zeta\,V_{\rm cross}\;}
$$
This is the central equation of the calculation.

---

## 3. Bosonization of the cross-replica vertex

### 3.1 Each $L_j$ as a bosonized operator

From §1, $L_j = \sqrt\alpha \cdot d_j$ where $d_j = \frac{1}{2}(\chi_{2j-1} - i\chi_{2(j+1)})$
in Majorana form. In Dirac form:
$$
d_j = \tfrac{1}{2}(c_j + c_j^\dagger) - \tfrac{1}{2}(c_{j+1} - c_{j+1}^\dagger).
$$
Bosonizing each Dirac field gives:
$$
c_j + c_j^\dagger \sim e^{ik_F j}\bigl[\psi_R + \psi_R^\dagger\bigr] + e^{-ik_F j}\bigl[\psi_L + \psi_L^\dagger\bigr],
$$
which at half filling ($k_F j = \pi j/2$) involves rapidly oscillating
phases. Multiplied across sites in the cross-replica vertex (where the
products are at the **same** $j$ in all four copies), the $j$-dependent
phases cancel:
$$
L_j^{(\sigma_1, a_1)}\,L_j^{(\sigma_2, a_2)}\,\text{etc.} \sim
\text{site-local product of vertex operators with phases that cancel within the product.}
$$
The **leading non-oscillating contribution** to the cross-replica vertex
is (up to overall constants):
$$
L_j^{(+,1)} L_j^{(+,2)} (L_j^*)^{(-,1)} (L_j^*)^{(-,2)} \sim
\bigl[\partial_x \phi^{(+,1)} + \partial_x\phi^{(+,2)} - \partial_x\phi^{(-,1)} - \partial_x\phi^{(-,2)}\bigr]^2 / (something).
$$
This is a quartic-in-$\partial\phi$ expression at low energy.

**Actually let me be more careful.** When you have a product of four vertex
operators at the same point, OPE collapses them to lower-dimension operators.
For two right-movers at the same point:
$$
\psi_R(x)\psi_R(x') \to (x - x')^{-1}\cdot e^{-2i\phi_R(x)} + \ldots
$$
At coincident points, $\psi_R^2 = 0$ by Fermi statistics. So
$L_j^{(+,1)} L_j^{(+,2)}$ does NOT vanish only because the two replicas
involve **independent** Klein factors and field operators.

Let me re-do this more carefully. Treating $\psi^{(a)}$ as independent for
different replicas $a$, and ignoring the bra-copies $\sigma = -$ for a moment:
$$
L_j^{(+,1)} L_j^{(+,2)} \approx d_j^{(+,1)}\,d_j^{(+,2)}
$$
$$
\approx \bigl[\psi_R^{(+,1)} + \psi_L^{(+,1)} + \text{h.c.}\bigr]_{x_j}
        \bigl[\psi_R^{(+,2)} + \psi_L^{(+,2)} + \text{h.c.}\bigr]_{x_j}.
$$
Multiplying and keeping only the non-oscillating (zero-$2k_F$) pieces:
$$
\sim \psi_R^{(+,1)\dagger}\psi_R^{(+,2)} + \psi_L^{(+,1)\dagger}\psi_L^{(+,2)} + (\text{cross R-L terms}).
$$
Each of these is a "spin-flip"-like operator across replicas (with replicas
playing the role of spin). In bosonized form, with $\psi_R^{(a)} \sim e^{-i(\phi^{(a)} - \theta^{(a)})}$:
$$
\psi_R^{(+,1)\dagger}\psi_R^{(+,2)} \sim e^{+i(\phi^{(+,1)} - \theta^{(+,1)})}\,e^{-i(\phi^{(+,2)} - \theta^{(+,2)})}
= e^{i[(\phi^{(+,1)} - \phi^{(+,2)}) - (\theta^{(+,1)} - \theta^{(+,2)})]}.
$$
In the ket-replica $(\rho, \sigma)$ basis $\phi_\sigma^{(+)} = \tfrac{1}{\sqrt 2}(\phi^{(+,1)} - \phi^{(+,2)})$:
$$
\psi_R^{(+,1)\dagger}\psi_R^{(+,2)} \sim e^{i\sqrt 2 (\phi_\sigma^{(+)} - \theta_\sigma^{(+)})}.
$$
This is a **vertex operator carrying $\sigma$ charge but no $\rho$ charge**.
A similar piece comes from the bra-replica $(-,1), (-,2)$ pair.

The full four-copy cross-replica vertex then takes the schematic form:
$$
V_{\rm cross} \sim \alpha \int dx\,\bigl[\psi_R^{(+,1)\dagger}\psi_R^{(+,2)}\,\psi_R^{(-,1)}\psi_R^{(-,2)\dagger} + \ldots\bigr]
$$
$$
\sim \alpha \int dx\,\cos\bigl[\sqrt 2\,(\phi_\sigma^{(+)} - \phi_\sigma^{(-)}) - \sqrt 2\,(\theta_\sigma^{(+)} - \theta_\sigma^{(-)})\bigr]
$$
(after extracting the real part of the vertex operator product).

Now we make one more change of basis: introduce
$\phi_\sigma^{\rm rel} = \tfrac{1}{\sqrt 2}(\phi_\sigma^{(+)} - \phi_\sigma^{(-)})$
(the σ-mode difference between ket and bra Choi-copies). Then:
$$
\boxed{\;V_{\rm cross} \sim \alpha \int dx\,\cos\bigl(2\phi_\sigma^{\rm rel} - 2\theta_\sigma^{\rm rel}\bigr)\;}
$$
where I've identified $\sqrt 2 \cdot \sqrt 2 = 2$.

This is a **dual sine-Gordon vertex** carrying both $\phi_\sigma^{\rm rel}$
and $\theta_\sigma^{\rm rel}$ phases.

### 3.2 Scaling dimension of the cross-replica vertex

For a sine-Gordon vertex $\cos(\beta_\phi \phi + \beta_\theta \theta)$
in a Luttinger liquid with parameter $K$, the scaling dimension is
$$
\Delta = \tfrac{1}{2}\bigl[K^{-1} \beta_\phi^2 + K \beta_\theta^2\bigr].
$$
For our vertex $\cos(2\phi_\sigma^{\rm rel} - 2\theta_\sigma^{\rm rel})$
we have $\beta_\phi = 2$, $\beta_\theta = -2$, giving
$$
\Delta_{\rm cross} = \tfrac{1}{2}\bigl[4K^{-1} + 4K\bigr] = 2(K + K^{-1}).
$$

A vertex is **relevant** if $\Delta < 2$ in 1+1D. Since $K + K^{-1} \ge 2$
with equality at $K = 1$, we have $\Delta_{\rm cross} \ge 4$, so the
cross-replica vertex is **strongly irrelevant** in the free-Dirac Luttinger
phase.

**This is the key analytical result.** At leading order in bosonization
around the free Dirac fixed point, the QJ cross-replica vertex is strongly
irrelevant (scaling dimension $\ge 4$). It does not produce a relevant
perturbation that could drive a phase transition by itself.

[**Caveat**: the scaling dimension depends on whether we use a single
Luttinger parameter $K$ for both $\phi$ and $\theta$ sectors, which is true
only at the strict $K = 1$ free Dirac fixed point. Away from this point
the two have separate scaling dimensions and the result modifies. For
small departures from $K=1$, $\Delta_{\rm cross}$ remains $\ge 4 - O(K-1)^2$.]

### 3.3 Implications

This result is interesting and somewhat at odds with the LMR scenario:

  - **LMR find a relevant vertex** for the diffusive cross-replica term,
    which drives them to an Ising fixed point at strong PPS.
  - **We find an irrelevant vertex** for the QJ cross-replica term.

If correct, this means the QJ critical theory at $\zeta < 1$ is in
**the same universality class as $\zeta = 1$** — a free-Dirac Luttinger
liquid with measurement-renormalized Luttinger parameter, **with no
sharp transition to Ising universality at any $\zeta^*$**.

**This is a falsifiable prediction**: the Renyi entropy reruns should
find $c_2/c_1 \approx 0.75$ and $c_3/c_1 \approx 0.667$ (free-Dirac CFT
values) **at all six** test points, including the strong-PPS ones
$\zeta = 0.20$ and $\zeta = 0.50$. If we find these ratios at every point,
the QJ model has free-Dirac CFT universality throughout the log-law
region. If we find departures from $0.75, 0.667$ at strong PPS, the
prediction is wrong and there is indeed a more interesting universality
crossover.

---

## 4. The $(1-\zeta)$ correction to the Luttinger parameter

Even if the cross-replica vertex is irrelevant in the RG sense, it
contributes at one-loop to the Luttinger parameter $K$ (similar to how
backscattering renormalizes $K$ in the standard Luttinger liquid even when
it's marginally irrelevant).

### 4.1 Leading-order renormalization

The general formula for the one-loop $K$ shift from a sine-Gordon vertex
$g \int \cos(\beta_\phi \phi + \beta_\theta \theta)$, in the regime where
the vertex is irrelevant and we just want the Hartree-Fock-type
correction, is
$$
\delta K \sim g^2 \cdot (\beta_\phi^2 K - \beta_\theta^2/K) \cdot (\text{cutoff-dependent factor}).
$$
For our vertex with $g = \alpha\zeta$ and $\beta_\phi = 2$, $\beta_\theta = -2$:
$$
\delta K \sim (\alpha \zeta)^2 \cdot (4K - 4/K) \cdot (\text{cutoff}) \approx 0
$$
at $K = 1$. So the leading-order shift vanishes at the free-Dirac point.
The next correction is at $O((\alpha\zeta)^2 (K-1))$, which is genuinely small.

### 4.2 The non-Hermitian Hartree-Fock contribution

A second, **more important** correction comes from the **non-Hermitian
shift** in the diagonal generator $\mathcal{L}_{\rm diag}^{(2)}$. The
per-replica term $-\tfrac{\alpha}{2}\sum_j L_j^\dagger L_j$ is, in
bosonized form, a current-like operator $\partial_x \phi$ that
**does not** depend on $\zeta$.

When combined with the cross-replica click term $\zeta V_{\rm cross}$, the
relative weight of "decay" to "click" sets the effective measurement
strength. The Born-rule single-replica Lindblad has these in balance
($\alpha$-coupling for both). The PPS-modified version has the click
weight rescaled by $\zeta$, breaking this balance.

The effective single-replica monitored Hamiltonian in the PPS ensemble
is therefore not the same as the Born-rule one. The decay shifts the
Luttinger parameter by an amount that comes from the $\alpha^2$
contribution of $[P_j, [P_j, \cdot]]$ (the standard Lindblad
double-commutator), and the cross-replica click contributes an additional
shift proportional to $\alpha^2 \zeta^2$ from the renormalized vertex.

Putting these together, at leading order in $\alpha$:
$$
K_\sigma(\alpha, \zeta) = 1 - \alpha^2 \kappa_0 \cdot (1 - \zeta^2) + O(\alpha^4)
$$
where $\kappa_0$ is a model-dependent constant of order unity.

The structure $(1 - \zeta^2)$ captures: at $\zeta = 1$ (Born) the
correction vanishes (since Born-rule preserves the bare $K = 1$ free Dirac
result modulo a different renormalisation); at $\zeta = 0$ (no-click)
the correction is maximal.

**[Heuristic warning]** The exact $\zeta$-dependence here is uncertain.
The structure $(1 - \zeta^2)$ comes from a Hartree-Fock-type
self-energy calculation that I have done schematically. To pin down the
prefactor $\kappa_0$ and verify the $(1-\zeta^2)$ structure precisely, one
would need to:

  1. Compute the one-loop self-energy of the $\phi_\sigma^{\rm rel}$ field
     from $\zeta V_{\rm cross}$ explicitly.
  2. Carry out the matching to the standard Luttinger-parameter renormalization
     equations (Giamarchi Chap. 5).
  3. Re-derive the relation $c_{\rm eff} \sim K$ from the conformal scaling
     of the Renyi entropy.

I have skipped steps 1–3 and used the schematic dependence. The result
should be checked.

---

## 5. Prediction for $c_{\rm eff}(\lambda; \zeta)$

Combining §1.4 (single-replica $K_1(\alpha)$) with §4.2 (cross-replica
correction):
$$
c_{\rm eff}(\alpha; \zeta) = K_1(\alpha)\,\bigl[1 - \alpha^2 \kappa_0\,(1-\zeta^2) + O(\alpha^4)\bigr].
$$

Converting from $\alpha$ to $\lambda$ via $\alpha = \lambda$, $w = 1-\lambda$:
$$
c_{\rm eff}(\lambda; \zeta) = K_1(\lambda) \bigl[1 - \lambda^2 \kappa_0\,(1-\zeta^2)\bigr]
$$
to leading order.

### 5.1 Comparison with numerics

Let's check this against our $c(\lambda; \zeta)$ table from earlier:

| $\lambda$ | $c(\lambda; \zeta=1)$ | $c(\lambda; \zeta=0.5)$ | $c(\lambda; \zeta=0.2)$ |
|-----------|----------------------:|------------------------:|------------------------:|
| 0.05      |                      ~2.5 (?)  |  ~5  | ~3 |
| 0.10      | ~5                   | ~5 | ~2 |
| 0.20      | ~3                   | ~2 | ~1 |
| 0.30      | ~2                   | ~1.5 | ~0.4 |
| 0.50      | ~0.2                 | ~0.09 | ~0.01 |

(From the printed table in our earlier analysis; numbers are rough.)

Pattern from the data:
  - At fixed $\lambda$, $c$ decreases as $\zeta$ decreases. **This is the
    opposite of what my formula predicts.** My formula has $c \propto 1 - (1-\zeta^2)$,
    which means $c$ should increase as $\zeta \to 1$.
  - Wait. At $\lambda = 0.5$: $c(\zeta=1) = 0.2$, $c(\zeta=0.5) = 0.09$,
    $c(\zeta=0.2) = 0.01$. So $c$ **does** decrease as $\zeta$ decreases.
    This means $c$ is smaller at low $\zeta$.
  - The formula predicts $c(\zeta) = K_1(\alpha)[1 - \alpha^2\kappa_0(1-\zeta^2)]$,
    which is larger when $\zeta = 1$ (factor in brackets = 1) and smaller
    when $\zeta < 1$ (factor in brackets < 1, assuming $\kappa_0 > 0$).

**So actually the sign of the prediction matches the data**: $c$ is
**reduced** at low $\zeta$, which the formula captures if $\kappa_0 > 0$.

### 5.2 Quantitative check at $\lambda = 0.5$

From the data, $c(\lambda=0.5; \zeta=1) \approx 0.2$, $c(\lambda=0.5; \zeta=0.5) \approx 0.09$.
The ratio is $\approx 0.45$, so the bracket factor at $\zeta = 0.5$ is
$\approx 0.45$.

Setting $1 - \alpha^2 \kappa_0 (1 - \zeta^2) = 0.45$ with $\alpha = 0.5$ and
$\zeta = 0.5$:
$$
0.25\,\kappa_0\,(1 - 0.25) = 0.55
\implies \kappa_0 \approx 2.9.
$$

Plug back in for $\zeta = 0.2$:
$$
1 - 0.25 \cdot 2.9 \cdot (1 - 0.04) = 1 - 0.70 = 0.30.
$$
Predicted $c(\lambda=0.5; \zeta=0.2) = 0.2 \cdot 0.30 = 0.06$.
Observed: $\approx 0.01$.

**Off by a factor of 6**, in the right direction but too large. The
$(1-\zeta^2)$ functional form is wrong, or the prefactor depends on $\alpha$
more strongly than $\alpha^2$.

### 5.3 Alternative: $(1-\zeta)$ rather than $(1-\zeta^2)$

If I assume instead that the correction goes like $(1-\zeta)$, with
$c(\lambda; \zeta) = K_1(\lambda)[1 - \kappa_1 \alpha (1-\zeta)]$, then at
$\lambda = 0.5, \zeta = 0.5$: bracket = $0.45 \Rightarrow \kappa_1 \alpha (1-\zeta) = 0.55 \Rightarrow \kappa_1 \approx 2.2$.
At $\zeta = 0.2$: bracket $= 1 - 2.2 \cdot 0.5 \cdot 0.8 = 1 - 0.88 = 0.12$,
giving $c = 0.024$. **Much better agreement with observed 0.01.**

So the empirical scaling is closer to $(1-\zeta)$ than $(1-\zeta^2)$.

**[Refined prediction]**
$$
\boxed{\;c_{\rm eff}(\lambda; \zeta) \approx K_1(\lambda) \cdot \bigl[1 - \kappa_1\,\lambda\,(1-\zeta) - O(\lambda^2(1-\zeta)^2)\bigr]\;}
$$
with $\kappa_1 \sim 2-3$ from the fit at $\lambda = 0.5$.

This is **linear in $(1-\zeta)$**, which matches what one expects from a
first-order perturbation in the PPS weight (since $\zeta = e^{-s}$ and the
generator depends linearly on $\zeta$).

### 5.4 Physical interpretation

The factor $\kappa_1 \approx 2-3$ encodes the **strength** of the
non-Hermitian renormalization in the cross-replica sector. The sign of
the correction (positive $\kappa_1$, so $c$ decreases as $\zeta \to 0$)
is consistent with the picture that the PPS-biased ensemble has **less
critical entanglement** than the Born-rule ensemble at the same $\lambda$.

This makes physical sense: increasing PPS strength (decreasing $\zeta$)
biases the ensemble toward fewer clicks, which is closer to the
deterministic no-click trajectory, which has area-law entanglement at
late times. So $c$ decreasing with $\zeta$ corresponds to the system
moving "closer to area-law" as PPS strengthens.

---

## 6. Summary of analytical results

1. **Two-replica generator (Section 2)**: $\mathcal{L}_\zeta^{(2)} = \mathcal{L}_{\rm diag}^{(2)} + \zeta V_{\rm cross}$,
   with $V_{\rm cross}$ the four-copy click vertex.

2. **Cross-replica vertex bosonization (Section 3)**: $V_{\rm cross} \sim \cos(2\phi_\sigma^{\rm rel} - 2\theta_\sigma^{\rm rel})$,
   strongly irrelevant ($\Delta = 4$) at the free-Dirac fixed point.

3. **Universality prediction (Section 3.3)**: free-Dirac CFT universality
   throughout $\zeta \in (0, 1]$, **with no sharp transition**. Testable
   via $c_2/c_1 \approx 3/4$ and $c_3/c_1 \approx 2/3$ at all six Renyi
   test points.

4. **Luttinger parameter and $c_{\rm eff}$ (Sections 4-5)**:
   $c_{\rm eff}(\lambda;\zeta) = K_1(\lambda)[1 - \kappa_1\,\lambda\,(1-\zeta)]$
   linear in $(1-\zeta)$, with prefactor $\kappa_1 \sim 2-3$ extracted
   from the numerics. Functional form derivable from the cross-replica
   click weight scaling linearly with $\zeta$.

5. **Comparison with LMR**: their analysis predicts a Luttinger parameter
   divergence at $\zeta^* \approx 0.28$ for the diffusive case, signaling
   a universality-class change. My QJ analysis finds no such divergence —
   the cross-replica vertex is irrelevant at every $\zeta$, so the
   universality class is fixed throughout. **This is the structural
   difference between QJ and diffusive unravelings.**

---

## 7. Caveats and what I have NOT done

  - The bosonization in §3 uses a fairly standard but not airtight
    procedure for handling the four-copy vertex. The exact coefficient of
    the cosine and the precise form of $\phi_\sigma^{\rm rel}$ depend on
    Klein factor and contour conventions I have glossed over.
  - The scaling dimension calculation in §3.2 assumes a single Luttinger
    parameter $K$ for both $\phi$ and $\theta$. Away from the strict $K = 1$
    free Dirac point, the dimension changes; the qualitative conclusion
    (irrelevance) holds as long as $K$ is not driven to small values.
  - The leading-order correction to $K$ in §4 uses a schematic
    Hartree-Fock argument rather than a careful one-loop calculation.
    The functional form $(1-\zeta)$ is supported by the numerical fit
    but should be rigorously derived.
  - I have not derived the actual scale-by-scale Luttinger flow equations
    for our model. LMR did this for theirs (their Eq. 49) but their model
    has different microscopic operator content, so I cannot directly
    reuse their flow equations.

---

## 8. Concrete next steps

  1. **Wait for the Renyi reruns** to come back. Check whether $c_2/c_1$
     and $c_3/c_1$ are consistent with free-Dirac CFT values $(3/4, 2/3)$
     at all six test points. If yes, supports the universality prediction.
     If no, the cross-replica vertex is not as irrelevant as my §3 analysis
     suggests, and the universality changes with $\zeta$.

  2. **Fit the full $c(\lambda; \zeta)$ data** to the linear-in-$(1-\zeta)$
     prediction at several $\lambda$ values. Check whether the prefactor
     $\kappa_1(\lambda)$ is consistent with a constant times $\lambda$
     (as the leading-order analysis suggests) or has stronger
     $\lambda$-dependence.

  3. **Discuss the Klein-factor and contour conventions** with Dganit
     to make sure my §3.1 bosonization of the cross-replica vertex is
     correct. The conclusion that $\Delta_{\rm cross} = 4$ (irrelevant)
     hinges on the field combination being $\phi_\sigma^{\rm rel} - \theta_\sigma^{\rm rel}$
     rather than just $\phi_\sigma^{\rm rel}$; if the dual field doesn't
     appear, the dimension drops to $2$ and the vertex is marginal,
     which would change the prediction substantially.

  4. **Calibrate $K_1(\lambda)$** by an independent measurement — e.g.,
     extracting it from the correlation function decay $|C(r)| \sim r^{-1/2K}$
     which is in the Renyi reruns. This gives a self-consistency check
     of the bosonized theory independent of the entanglement scaling.


---

## 9. Self-critique and corrections [important, added on review]

On a second pass through the calculation above I realised I have made some
oversights that affect the conclusions. Recording them here for honesty
and so the corrected picture is on record.

### 9.1 I conflated three distinct cross-replica operators

A product of four single-fermion operators at the same point
$L^{(+,1)}\,L^{(+,2)}\,L^{*(-,1)}\,L^{*(-,2)}$
can be split into multiple channels by choosing different
annihilation/creation contractions. The three physically relevant channels
are:

**Channel A — "Cooper-pair" channel.** All four are annihilation operators (or
their creation conjugates). This is the channel I bosonized in §3.1 and
got $\cos(2\phi^{\rm rel}_\rho - 2\theta^{\rm rel}_\rho)$ with dimension 4.
This is strongly irrelevant. ✓ correctly identified.

**Channel B — "Density-density" channel.** Two of the four are creation,
two annihilation, in pairs that form local densities. This gives the operator
$n_j^{(+,1)} n_j^{(+,2)} n_j^{(-,1)*} n_j^{(-,2)*}$, which in bosonized form
is $\partial_x\phi^{(+,1)} \partial_x\phi^{(+,2)} \partial_x\phi^{(-,1)} \partial_x\phi^{(-,2)}$.
This is a higher-derivative operator with dimension 8 — also irrelevant.

But more importantly, when one of the products contracts to give a
$\delta$-function (OPE contraction), this reduces to a two-derivative
operator like $\partial_x\phi^{(+,1)} \partial_x\phi^{(-,1)}$, with dimension 2.
This is **marginal** and contributes to the Luttinger parameter
renormalization at order $\zeta$ (not $\zeta^2$).

**Channel C — "Backscattering" channel.** Two right-movers and two
left-movers paired into back-scattering. This produces an oscillating
phase $e^{4ik_F x}$ which at half-filling is $e^{2i\pi x} = 1$
(non-oscillating). The resulting operator is a marginal cosine like
$\cos(2\phi^{(1)} + 2\phi^{(2)} - 2\phi^{(-,1)} - 2\phi^{(-,2)})$ with
dimension 2 at $K = 1$ — marginal.

### 9.2 The leading shift to K is from Channel B, not Channel A

I had focused on Channel A (Cooper-pair vertex, dimension 4, irrelevant)
and computed the leading $K$-shift as $\sim g^2$. This is wrong; the
Cooper-pair vertex is too irrelevant to dominate.

The correct leading shift comes from **Channel B**, which after the OPE
contraction generates a marginal current-current operator
$\zeta \alpha \cdot (\partial_x \phi^{(+,1)})(\partial_x \phi^{(-,1)})$ etc.
This renormalizes $K$ at **first order in $\zeta\alpha$**:
$$
\delta K(\zeta, \alpha) \sim \zeta \alpha \cdot f(\lambda)
$$
where $f(\lambda)$ is some function determined by the precise UV details
of the OPE contraction.

### 9.3 The correct leading-order prediction

With this correction, the leading shift in $K$ is **linear** in $\zeta$:
$$
K(\zeta) = K(\zeta = 0) + \alpha \zeta \cdot f(\lambda).
$$
At $\zeta = 0$: $K = K(\zeta=0) = K_0$ (the bare value).
At $\zeta = 1$: $K = K_0 + \alpha f(\lambda)$.

If $c_{\rm eff}$ is proportional to $K$ with a $\lambda$-dependent
proportionality, this gives
$$
c_{\rm eff}(\lambda; \zeta) \propto K(\lambda; \zeta) \propto \zeta \cdot \text{(something)}.
$$

Combined with the **physical argument** that at $\zeta = 0$ the no-click
trajectory drives the system to a deterministic area-law state with
$c_{\rm eff} = 0$ in the L → ∞ limit, the natural functional form is
$$
\boxed{\;c_{\rm eff}(\lambda; \zeta) \approx \zeta \cdot c_{\rm eff}(\lambda; \zeta = 1)\;}
$$
in the linear-in-$\zeta$ approximation.

### 9.4 Numerical check of this corrected prediction

Setting $c_{\rm eff}(\lambda; \zeta) = \zeta \cdot c_{\rm eff}(\lambda; \zeta=1)$:

At $\lambda = 0.5$:
  - $c(\zeta = 1) = 0.20$ (data, approx)
  - $c(\zeta = 0.5)$: predicted $0.5 \times 0.20 = 0.10$. Observed: $\approx 0.09$. ✓ excellent agreement.
  - $c(\zeta = 0.2)$: predicted $0.2 \times 0.20 = 0.04$. Observed: $\approx 0.01$. Off by ~4.

At $\lambda = 0.3$:
  - $c(\zeta = 1) \approx 1.5$ (data).
  - $c(\zeta = 0.5)$: predicted $0.75$. Observed: $\approx 1.4$. ✗ predicts too low.
  - $c(\zeta = 0.2)$: predicted $0.3$. Observed: $\approx 0.4$. ✓ approximately.

So the linear-in-$\zeta$ scaling **roughly works at $\lambda = 0.5$** but
not at $\lambda = 0.3$. This suggests:

  - The scaling has additional $\lambda$ dependence beyond just $\zeta$
    (e.g., $c \propto \zeta \cdot g(\lambda)$ with non-monotonic $g$).
  - Or the relation between $c_{\rm eff}$ and $K$ has logarithmic
    corrections that I have not accounted for.
  - Or the linear-in-$\zeta$ approximation breaks down at small $\zeta$ in
    the regime where higher-order $\zeta$ corrections matter.

### 9.5 Physical picture (best-faith summary)

The PPS-QJ ensemble has **fewer clicks per unit time** than the Born-rule
ensemble at the same $\lambda$, by a factor of $\zeta$ (since the click
intensity is rescaled). Each click contributes additively to the
entanglement structure in the steady state. In a steady-state balance
between click creation and non-Hermitian relaxation, the entanglement
"density" should scale roughly linearly with the click rate at fixed
non-Hermitian rate.

Therefore at leading order, $c_{\rm eff}(\zeta) \sim \zeta \cdot c_{\rm eff}(\zeta = 1)$,
modulated by $\lambda$-dependent corrections from the non-trivial
$H_{\rm eff}$ dynamics between clicks.

The data is broadly consistent with this picture (correct sign, correct
overall scale at intermediate $\lambda$), but the quantitative agreement
is imperfect, pointing to higher-order effects I have not computed.

### 9.6 What this all means for the universality question

The Cooper-pair vertex (Channel A) being irrelevant **still suggests**
free-Dirac CFT universality throughout the log-law region — the qualitative
conclusion of §3.3 survives. What I was wrong about was the **leading shift
in $K$**: it comes from a marginal operator (Channel B), not the
irrelevant Cooper-pair vertex.

The Renyi entropy reruns remain the key test: if $c_2/c_1 \approx 0.75$
and $c_3/c_1 \approx 0.667$ at all six test points, free-Dirac universality
holds throughout. If the ratios deviate at strong PPS, the universality
changes — but this would imply that the marginal channel B operator
becomes relevant at some $\zeta^*$, which my analysis does not capture.

### 9.7 Honest summary of what I have produced

**Solid contributions:**

  - The two-replica generator for QJ (Section 2 of the companion document
    and Section 2 here) — this is correct and not previously written down.
  - The structural comparison with LMR (Appendix A of the companion
    document) — identifies the genuine difference between QJ and diffusive
    cross-replica couplings.
  - The empirical observation that $c_{\rm eff}(\lambda; \zeta) \approx \zeta \cdot c_{\rm eff}(\lambda; \zeta=1)$
    approximately, at least at $\lambda = 0.5$ (Section 9.4 above) — this
    is data-driven but consistent with a leading-order PPS scaling argument.

**Conjectures pending verification:**

  - Free-Dirac CFT universality throughout the log-law region (Section 3.3).
    Testable by Renyi reruns.
  - Linear-in-$\zeta$ scaling of $c_{\rm eff}$ (Section 9.5). Testable by
    fitting the existing $c(\lambda; \zeta)$ data more carefully.

**Things I genuinely have not done:**

  - A one-loop calculation of the marginal Channel B coupling and its
    effect on $K$. This is needed to predict $f(\lambda)$ in Section 9.3
    and would tell us why the prediction works better at some $\lambda$
    than others.
  - A proper renormalization group flow analysis for the QJ generator.
    LMR did this for theirs (their Eq. 49) but the QJ flow equations are
    different and have not been derived.
  - The QJ analog of LMR's $\zeta^*$ — whether such a special value
    exists in the QJ case is still open. My analysis suggests "no" (no
    universality change in the log-law region), but the marginal operator
    analysis I did not complete is precisely what could overturn this.


---

## 10. What the data actually tells us [empirical, post-hoc]

I ran the test `R(λ, ζ) = c_eff(λ; ζ) / [ζ · c_eff(λ; 1)]` against the
full v2 production dataset. The linear-in-ζ prediction is $R = 1$
everywhere. **The data is very far from this** — the heatmap of $R(λ, ζ)$
shows large deviations in both directions across the parameter plane.
Concretely:

  - **Bottom-left (small λ, small ζ)**: $R \gg 1$ (red). The measured $c$ is
    much larger than $\zeta \cdot c(\zeta=1)$ would predict.
  - **Top center (moderate λ, large ζ ≈ 0.7–0.85)**: $R \approx 1$ (white).
    Linear scaling works here.
  - **Bottom-right (large λ, small ζ)**: $R \ll 1$ (deep blue). Measured
    $c$ is much smaller than linear prediction (i.e., closer to area-law).

The "white band" where $R \approx 1$ sits approximately along the
diagonal in the $(\lambda, \zeta)$ plane, roughly tracking the $c = 1$
phase boundary curve we identified in `phase_diagram_final.png`.

### 10.1 The (1 - (1-ζ)^p) fit

A more flexible ansatz $c_{\rm eff}(\lambda; \zeta) = c_{\rm eff}(\lambda; 1) \cdot [1 - (1-\zeta)^{p(\lambda)}]$
fits the data with $p(\lambda) \approx 0.85$–$1.03$ across the full
$\lambda$ range, with $p \approx 1$ at intermediate λ (0.30-0.55) and
$p \approx 0.9$ at smaller λ. This is broadly consistent with linear-
in-$(1-\zeta)$ behavior **near the crossover** but not globally.

### 10.2 The honest empirical structure

Looking at the right panel of `linear_zeta_test.png`: at each fixed λ,
$R(\zeta)$ is **non-monotonic** in ζ:

  - For $\lambda \in [0.10, 0.30]$ (deep log-law at ζ=1): R has a
    *minimum* around ζ ≈ 0.15-0.2, where the data crosses into the
    area-law region. Below this minimum R rises again (finite-size
    artifact, see below).
  - For $\lambda \approx 0.40$-$0.50$ (near the crossover at ζ=1): R is
    monotonically decreasing in ζ as we move from ζ=0.85 (R ≈ 1.3) to
    ζ=0.02 (R close to 0).
  - For $\lambda \ge 0.55$ (area-law at ζ=1): c is already tiny at ζ=1
    so the ratio is unstable and not informative.

The non-monotonicity at small λ (R increasing again as ζ → 0) is most
likely a **finite-size artifact**: at strictly ζ = 0 the system follows
the deterministic no-click H_eff dynamics, which at L = 128 still has
substantial entanglement at small λ because the gap of H_eff is
unresolved. In L → ∞, c(λ > 0, ζ → 0) → 0, but at L = 128 we see
finite c that biases the ratio.

### 10.3 The picture this paints

The data is **inconsistent with the simple linear-in-(1-ζ) prediction**
across the full plane. It is more consistent with the LMR-style
scenario:

  - There is a **phase boundary** in the (λ, ζ) plane separating a
    log-law region from an area-law region.
  - On the log-law side of the boundary, $c$ takes its "natural" value
    close to $c(\zeta = 1)$ regardless of ζ.
  - On the area-law side, $c \to 0$.
  - The transition between the two regions is **relatively sharp** for
    a finite system, becoming sharper as L → ∞ (consistent with a phase
    transition in the thermodynamic limit).

This is the **LMR scenario applied to QJ**: the QJ unraveling also
exhibits a log-to-area phase boundary, with the c=1 curve plausibly
tracing the boundary.

### 10.4 What this means for the bosonization prediction

My §3.3 conclusion ("free-Dirac CFT universality throughout, no sharp
transition") **is not supported by the data**. The data favors a more
LMR-like picture with a phase boundary.

This means my bosonization analysis in §3 must be incomplete or wrong.
The most likely culprit (per §9.2) is that I missed the relevant
contribution from the **density-density (Channel B) operator**. That
operator is marginal at K = 1 and can be marginally relevant if the
sign of the coupling is right — driving the system to a different
fixed point at strong measurement.

This is the scenario LMR find for their diffusive case: a marginal
operator becomes relevant at strong PPS, driving the system to the
Ising fixed point. **By analogy, our QJ case should also have a
relevant operator at strong PPS**, producing the phase boundary visible
in the data.

### 10.5 Honest assessment of the calculation

After this empirical check, the analytical work in this document
provides:

  - **A correct setup** for the QJ two-replica generator (§2).
  - **A correct identification** of three operator channels in the
    bosonized form (§9.1).
  - **An incorrect quantitative prediction** ($c \propto \zeta$) that
    the data clearly disfavors (§10.1).
  - **A clear research direction**: a proper analysis of the marginal
    Channel B operator and its RG flow is required, following LMR's
    Section V C template adapted to the QJ case.

The conjecture in §6 of the companion document (free-Dirac CFT
universality throughout) **looks wrong in light of the data**. The
LMR-style scenario (phase boundary in the (λ, ζ) plane) appears to
hold for QJ as well, just with a possibly different boundary location.

This is a useful negative result. The Renyi entropy reruns will provide
a direct test: if c_n/c_1 ratios are universal (i.e., free Dirac) at all
six test points, the universality is fixed; if they differ between
log-law and area-law sides of the boundary, there is a genuine
universality change.

I will need to wait for the Renyi data before saying more about which
scenario is realized.
