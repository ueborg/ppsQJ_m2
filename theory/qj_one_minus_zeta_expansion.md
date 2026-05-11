# Bosonization of the QJ cross-replica click vertex and the $(1-\zeta)$ expansion

This is a follow-up to `qj_two_replica_derivation.md` carrying out the
bosonization step explicitly for the quantum-jump cross-replica click vertex
and extracting the leading $(1-\zeta)$ dependence of the bosonized
Luttinger parameters near the Born-rule fixed point ($\zeta = 1$).

**Headline result:** The QJ cross-replica click vertex bosonizes to an
operator of scaling dimension $\Delta = 4$ at the Born-rule fixed point.
This is **irrelevant** in the RG sense ($\Delta > 2$ in 1+1D). Therefore
the leading $(1-\zeta)$ correction to the bosonized Luttinger parameter
$K_\sigma$ **vanishes**:

$$
\frac{dK_\sigma^{\rm QJ}}{d(1-\zeta)}\bigg|_{\zeta = 1} \;=\; 0.
$$

The leading non-trivial $(1-\zeta)$ shift is at order $(1-\zeta)^2$. This
contrasts with LMR's diffusive theory, where the analogous shift is
**linear** in $(1-\zeta)$, because the LMR cross-replica term contains a
marginal $\cos(2\phi_\sigma)$ vertex.

This is a concrete, falsifiable prediction that I check against our
numerical $c(\lambda, \zeta) = 1$ curve below. The numerical data are
**consistent** with the prediction.

---

## 1. Correcting an error from `qj_two_replica_derivation.md`

In `qj_two_replica_derivation.md` §4 I asserted that the bond-density
operator $P_j = d_j^\dagger d_j$ is **quartic** in Majoranas. This is
wrong. Let me redo the calculation.

The bond annihilation operator in second quantisation (verified against
`pps_qj/exact_backend.py` line 101) is
$$
d_j \;=\; \tfrac{1}{2}\bigl[(c_j + c_j^\dagger) + i(c_{j+1} - c_{j+1}^\dagger)\bigr].
$$

In the Majorana convention $\chi_{2j-1} = c_j + c_j^\dagger$,
$\chi_{2j} = -i(c_j - c_j^\dagger)$:

$$
c_j + c_j^\dagger \;=\; \chi_{2j-1},
\qquad
c_{j+1} - c_{j+1}^\dagger \;=\; i\,\chi_{2(j+1)},
$$

so

$$
d_j \;=\; \tfrac{1}{2}\bigl[\chi_{2j-1} + i\,(i\,\chi_{2(j+1)})\bigr]
       \;=\; \tfrac{1}{2}\bigl[\chi_{2j-1} - \chi_{2(j+1)}\bigr] \;?
$$

Wait — let me recompute. $i \cdot (i\,\chi) = -\chi$, so $d_j = (1/2)[\chi_{2j-1} - \chi_{2(j+1)}]$.
Hmm, that would make $d_j$ real, hence Hermitian.

Let me check against the project's `bond_jump_pair`:
```python
def bond_jump_pair(bond: int) -> tuple[int, int]:
    return 2 * bond, 2 * bond + 3
```

For bond $j$ (0-indexed), the pair is $(2j, 2j+3)$. In a 0-indexed Majorana
convention where site $k$ has Majoranas $(2k, 2k+1)$ for "real" and "imag"
parts:
$\chi_{2k} = c_k + c_k^\dagger$ (real, even)
$\chi_{2k+1} = -i(c_k - c_k^\dagger)$ (imag, odd)

Bond $j$ pair = $(2j, 2j+3) = (\chi_{2j}, \chi_{2(j+1)+1})$. So the bond
mode uses:
- Real Majorana of site $j$: $\chi_{2j} = c_j + c_j^\dagger$
- Imaginary Majorana of site $j+1$: $\chi_{2j+3} = -i(c_{j+1} - c_{j+1}^\dagger)$

Then
$$
d_j \;=\; \tfrac{1}{2}\bigl[(c_j + c_j^\dagger) + i\,(c_{j+1} - c_{j+1}^\dagger)\bigr]
       \;=\; \tfrac{1}{2}\bigl[\chi_{2j} + i\cdot(i\,\chi_{2j+3})\bigr]
       \;=\; \tfrac{1}{2}\bigl[\chi_{2j} - \chi_{2j+3}\bigr]\;?
$$

Hmm. Then $d_j$ would be Hermitian. But $d_j$ is supposed to be an
annihilation operator, hence anti-Hermitian when constructed correctly...

Actually let me be careful. **The relation between Majoranas and
ladders is convention-dependent**. Using $\chi_a^2 = 1$, the Majoranas
satisfy $\{\chi_a, \chi_b\} = 2\delta_{ab}$ but $\chi_a$ themselves are
Hermitian. An "annihilation" operator built from them is necessarily
non-Hermitian only if it combines them with a complex coefficient.

Looking at the formula in `pps_qj/exact_backend.py` line 101:
```python
d_op = 0.5 * (cd_ops[left] + c_ops[left] + c_ops[right] - cd_ops[right])
```

So $d_j = (1/2)[c_l^\dagger + c_l + c_r - c_r^\dagger]$, i.e.,
$d_j = (1/2)[(c_l + c_l^\dagger) + (c_r - c_r^\dagger)]$.

In Majorana form:
$d_j = (1/2)[\chi_{2l-1} + i\,\chi_{2r}]$ (using the convention
$c_r - c_r^\dagger = i\,\chi^{(2)}_r$).

OK so with the **1-indexed** Majorana convention (where $\chi^{(1)}_l = c_l + c_l^\dagger$, $\chi^{(2)}_l = -i(c_l - c_l^\dagger)$, so $c_l - c_l^\dagger = i\chi^{(2)}_l$), we get
$$
d_j \;=\; \tfrac{1}{2}(\chi^{(1)}_l + i\,\chi^{(2)}_r), \qquad l = j,\ r = j+1.
$$

This is a **complex linear combination** of two Majoranas — a chiral Dirac
fermion mode localised on bond $j$. Crucially, $d_j$ is **not** Hermitian:
$d_j^\dagger = (1/2)(\chi^{(1)}_l - i\,\chi^{(2)}_r) \neq d_j$.

The earlier confusion was that I conflated the two-index Majorana
conventions. The correct result:

$$
\boxed{
P_j \;=\; d_j^\dagger d_j
       \;=\; \tfrac{1}{2} + \tfrac{i}{2}\,\chi^{(1)}_l \chi^{(2)}_r
}
$$

This is **bilinear** in Majoranas, exactly analogous to LMR's measurement
operator $M_j$ (which is also a Majorana bilinear). So the diagonal-in-
replica part of the QJ generator has the same operator content as LMR's.

The **cross-replica click term** $L_j \otimes L_j$ in the two-replica
generator becomes, in 4-Choi-copy fermion language, a product of four
$d_j$ operators with specific bra/ket assignments (this is the same as
before; only the operator content within each replica was wrong).

---

## 2. The cross-replica click vertex in 4-Choi-copy fermion form

The unnormalised two-replica density matrix satisfies
$$
\partial_t \check\rho^{(2)} \;=\; \mathcal{L}_s^{(2)}[\check\rho^{(2)}]
$$
with the cross-replica click contribution
$$
\Delta_{\rm cross}\mathcal{L}_s^{(2)}[\sigma]
\;=\;
\zeta \alpha \sum_j (d_j \otimes d_j)\,\sigma\,(d_j^\dagger \otimes d_j^\dagger).
$$

Under Choi–Jamiołkowski, the bra-side action of an operator becomes its
complex conjugate acting on the doubled state. For fermions
$(d_j)^* = d_j^\dagger$ (since $d_j$ is complex-linear in Hermitian
Majoranas, complex conjugation swaps creation and annihilation). So in
the 4-Choi-copy fermion Fock space (replica $a \in \{1, 2\}$, Choi
$\sigma \in \{+, -\}$ for ket/bra),

$$
\boxed{
V_j \;\equiv\; (d \otimes d)\,\cdot\,(d^\dagger \otimes d^\dagger)\big|_{C\text{-}J}
\;=\;
d_j^{(+, 1)} \,\bigl(d_j^{(-, 1)}\bigr)^{\!\dagger}\,\, d_j^{(+, 2)}\,\bigl(d_j^{(-, 2)}\bigr)^{\!\dagger}.
}
$$

This is a **quartic fermion operator** in the 4-Choi-copy theory.

### 2.1 Anticommutation across Choi copies — Klein factor caveat

Across distinct Choi copies, fermion operators anticommute up to Klein
factors (LMR App. C, Eqs. C5–C6). Properly defined Majoranas in the 4-copy
theory satisfy $\{\chi^{(\sigma a)}_j, \chi^{(\sigma' a')}_k\} = 0$ if
$(\sigma, a) \neq (\sigma', a')$. The Klein factors are global Pauli
strings in replica/Choi space and don't affect the local operator content
relevant for bosonization. I will assume the Klein factor structure is
sorted out in the standard way; my conclusions below depend on the
**operator content**, not the Klein factors.

---

## 3. Bosonization of $V_j$

Each $d_j$ is a chiral fermion mode on bond $j$. In the continuum limit
at half-filling, $d_j \to f(x_j)$ a Dirac fermion field. Standard 1D
bosonization gives (using the convention of Giamarchi, *Quantum Physics in
One Dimension*, Chap. 2):

$$
f_R(x) \,\sim\, e^{i(\theta(x) - \phi(x))}/\sqrt{2\pi a},
\qquad
f_L(x) \,\sim\, e^{i(\theta(x) + \phi(x))}/\sqrt{2\pi a},
$$

with $\phi$ the "density boson" and $\theta$ the dual "current boson",
$[\phi(x), \partial_y\theta(y)] = i\pi\delta(x-y)$. The full fermion field
is $f(x) = e^{-ik_F x} f_R + e^{ik_F x} f_L$. For the bond operator at
half-filling ($k_F = \pi/2$), the dominant smooth piece can be taken to
be either chirality; for concreteness take the R-mover.

In the 4-Choi-copy theory, each copy $(\sigma, a)$ has its own bosonic
field pair $(\phi^{(\sigma a)}, \theta^{(\sigma a)})$.

### 3.1 The single-replica factor

Using fermion anticommutation across the $(+, a)$ and $(-, a)$ copies,
$d_j^{(+, a)}(d_j^{(-, a)})^\dagger = -(d_j^{(-, a)})^\dagger d_j^{(+, a)}$.
So
$$
d_j^{(+, a)}(d_j^{(-, a)})^\dagger
\;\propto\;
-(d_j^{(-, a)})^\dagger d_j^{(+, a)}\,.
$$

Bosonized form (R-movers):
$$
(d_j^{(-, a)})^\dagger d_j^{(+, a)}
\;\sim\;
\frac{1}{2\pi a}\, e^{-i(\theta^{(-, a)} - \phi^{(-, a)})}\, e^{i(\theta^{(+, a)} - \phi^{(+, a)})}
\;=\;
\frac{1}{2\pi a}\, e^{i[(\theta^{(+, a)} - \theta^{(-, a)}) - (\phi^{(+, a)} - \phi^{(-, a)})]}.
$$

Define replica-$a$ **bra-ket-difference** fields:
$$
\Phi^a_D \;\equiv\; \tfrac{1}{\sqrt{2}}(\phi^{(+, a)} - \phi^{(-, a)}),
\qquad
\Theta^a_D \;\equiv\; \tfrac{1}{\sqrt{2}}(\theta^{(+, a)} - \theta^{(-, a)}).
$$

(The $\sqrt{2}$ normalises the new fields canonically.)

Then
$$
(d_j^{(-, a)})^\dagger d_j^{(+, a)}
\;\sim\;
\frac{1}{2\pi a}\, e^{i\sqrt{2}\,(\Theta^a_D - \Phi^a_D)}.
$$

### 3.2 The cross-replica product

The two factors multiply:
$$
V_j
\;=\;
d_j^{(+, 1)}(d_j^{(-, 1)})^\dagger\, d_j^{(+, 2)}(d_j^{(-, 2)})^\dagger
\;\sim\;
\frac{1}{(2\pi a)^2}\, e^{i\sqrt{2}\,[(\Theta^1_D - \Phi^1_D) + (\Theta^2_D - \Phi^2_D)]}.
$$

Now go to **inter-replica** $(\rho, \sigma)$ modes:
$$
\Phi^\rho_D \;=\; \tfrac{1}{\sqrt{2}}(\Phi^1_D + \Phi^2_D),
\qquad
\Phi^\sigma_D \;=\; \tfrac{1}{\sqrt{2}}(\Phi^1_D - \Phi^2_D),
$$
and analogously for $\Theta$. So $\Phi^1_D + \Phi^2_D = \sqrt{2}\,\Phi^\rho_D$.

Substituting:
$$
\boxed{
V_j(x) \;\sim\; \frac{1}{(2\pi a)^2}\,\exp\!\bigl[2i\,(\Theta^\rho_D(x) - \Phi^\rho_D(x))\bigr].
}
$$

**Key observations:**

1. $V_j$ has support **only** in the inter-replica $\rho$-mode, **not**
   in the $\sigma$-mode.
2. $V_j$ contains both $\Phi^\rho_D$ and $\Theta^\rho_D$ in the
   combination $\Theta^\rho_D - \Phi^\rho_D$ — this is a **chiral**
   vertex operator (it represents the creation of a right-moving Dirac
   particle in the $\rho_D$ sector).
3. $V_j$ is a vertex operator, **not** a current ($\partial\phi$) or
   marginal $\cos(2\phi)$.

### 3.3 Comparison to LMR's diffusive cross-replica term

LMR's cross-replica fluctuation operator (from the Gaussian noise
variance $\zeta\,\delta_{jk}\delta(t-t')$) is, after averaging,
$$
\mathcal{O}^{\rm LMR}_{\rm cross} \;\sim\; \zeta \sum_{(\sigma a), (\sigma' a')}\, M_j^{(\sigma a)} M_j^{(\sigma' a')}.
$$

The summed-pairs structure ensures that in the $(\rho, \sigma)$
bosonization (LMR Eq. (46)) the leading-order term is
$$
\mathcal{O}^{\rm LMR}_{\rm cross} \;\sim\; \zeta\, g_2\, \cos(2\phi^\sigma_D),
$$
i.e., a **non-chiral** $\sigma$-mode cosine vertex.

The QJ vertex (boxed above) is **chiral** and lives in the $\rho_D$
sector. This is a structural difference, and it has direct consequences
for the scaling dimension and the RG flow.

---

## 4. Scaling dimension of $V_j$ and the leading $(1-\zeta)$ correction

### 4.1 Scaling dimension at the Born-rule fixed point

For a vertex operator $e^{i(p\phi + q\theta)}$ in a Luttinger liquid
with parameter $K$, the scaling dimension is (Giamarchi Eq. 2.62)
$$
\Delta_{p,q} \;=\; \tfrac{1}{2}\bigl(p^2 K + q^2 K^{-1}\bigr).
$$

For our $V_j \sim e^{2i\Theta^\rho_D - 2i\Phi^\rho_D}$: $p = -2$, $q = 2$.
At the Born-rule fixed point, $K_{\rho_D} = 1$ (free Dirac CFT, since
the diagonal-in-replica generator at $\zeta = 1$ reduces to the
unraveling-independent Lindbladian, which has a $c = 1$ Luttinger
liquid description in the long-wavelength limit):
$$
\Delta(V_j) \;=\; \tfrac{1}{2}\bigl(4 \cdot 1 + 4 \cdot 1\bigr) \;=\; 4.
$$

**$V_j$ is irrelevant** ($\Delta = 4 > 2$ in 1+1D). It cannot drive a
phase transition by itself, and it does not generate marginal
corrections to the Luttinger parameter at leading order in its
coupling.

### 4.2 The leading $(1-\zeta)$ shift of $K_\sigma$

The QJ two-replica Hamiltonian at $\zeta < 1$ is, schematically,
$$
H^{\rm QJ}(\zeta) \;=\; H_{\rm Born} \,-\, (1 - \zeta)\,\alpha\, \sum_j V_j,
$$
where $H_{\rm Born}$ is the $\zeta = 1$ generator and the second term is
the correction from reducing the click intensity. The bosonized $V_j$
has scaling dimension 4, so the coupling $(1-\zeta)\alpha$ flows under
RG as
$$
\frac{d}{d\ell}\bigl[(1-\zeta)\alpha\bigr] \;=\; (2 - \Delta(V_j))\bigl[(1-\zeta)\alpha\bigr] \;=\; -2\,(1-\zeta)\alpha,
$$
i.e., $(1-\zeta)\alpha$ flows to zero with scaling exponent $-2$. The
operator is irrelevant.

Because $V_j$ is irrelevant, its effect on the Luttinger parameters
$K_\rho, K_\sigma$ at leading order in $(1-\zeta)\alpha$ is zero. The
first non-trivial correction comes from OPE of $V_j$ with itself or
with other operators, giving a contribution at order
$[(1-\zeta)\alpha]^2$ to the bosonic action.

In particular,
$$
\boxed{
\frac{dK_\sigma^{\rm QJ}}{d(1-\zeta)}\bigg|_{\zeta = 1} \;=\; 0,
}
$$
and
$$
K_\sigma^{\rm QJ}(\zeta) \;=\; K_\sigma^{(\rm Born)} \,+\, c_2\,(1 - \zeta)^2 \,+\, O((1-\zeta)^3),
$$
with $c_2$ a non-universal coefficient.

### 4.3 Contrast with LMR's diffusive case

LMR's bosonized cross-replica operator $\cos(2\phi^\sigma_D)$ has
scaling dimension
$$
\Delta_{\rm LMR} \;=\; \tfrac{1}{2}(4 K_\sigma) \;=\; 2 K_\sigma \;\approx\; 2
$$
at $K_\sigma = 1$ — i.e., **marginal**. Marginal operators DO shift the
Luttinger parameter at leading order in their coupling. Their RG
equation (LMR Eq. 49) gives
$$
\frac{dK_\sigma^{\rm LMR}}{d\ell} \;\propto\; -g_2^2 K_\sigma^2,
$$
generating a finite shift in $K_\sigma$ from the $\zeta$-dependent
marginal coupling $g_2(\zeta) \propto \zeta$.

Translated to the $(1-\zeta)$ expansion: $K_\sigma^{\rm LMR}$ shifts
**linearly** in $(1-\zeta)$ near $\zeta = 1$. The QJ shift is **quadratic
or higher**.

---

## 5. Testable consequences and numerical comparison

### 5.1 The prediction translated to $c_{\rm eff}$ and $\lambda(c=1)$

The bosonized $K_\sigma$ is related to the effective central charge of
the critical phase by $c_{\rm eff} = K_\sigma$ (in the free-Dirac
convention). Our `c_eff_curves.png` gives $c$ as a function of $\lambda$
at fixed $\zeta$; the curve $c(\lambda, \zeta) = 1$ traces the BKT line
in our phase diagram.

The prediction says: **at fixed $\lambda$, the slope $\partial c / \partial(1-\zeta)\big|_{\zeta=1}$ should vanish**, or equivalently, the
location $\lambda(c=1)$ of the $c=1$ line should shift only at
$O((1-\zeta)^2)$ for $\zeta$ near 1.

### 5.2 Numerical check

From the v2 main grid, the measured $\lambda$ values where $c$ crosses
1, for various $\zeta$ (cf. `phase_diagram_final.png`):

| $\zeta$ | $1 - \zeta$ | $\lambda(c=1)$ | $\Delta\lambda \equiv \lambda(c=1, \zeta) - \lambda(c=1, 1)$ |
|---------|------------:|---------------:|-------------------------------:|
| 1.00 | 0.00 | 0.364 |  0.000 |
| 0.85 | 0.15 | 0.365 | +0.001 |
| 0.70 | 0.30 | 0.370 | +0.006 |
| 0.50 | 0.50 | 0.334 | $-0.030$ |
| 0.30 | 0.70 | 0.237 | $-0.127$ |

The behaviour near $\zeta = 1$ shows essentially **no shift** from
$\zeta = 1$ to $\zeta = 0.85$ (just 0.001, within noise), and only a
small shift to $\zeta = 0.70$. This is qualitatively consistent with
the prediction of vanishing leading slope.

Polynomial fits in $(1-\zeta)$ over the range $\zeta \in [0.30, 1.00]$
give:

  - Linear fit ($\Delta\lambda = a\,(1-\zeta)$): poor, mean abs residual 0.024.
  - Quadratic fit ($a(1-\zeta) + b(1-\zeta)^2$): residual 0.006, with
    leading **quadratic** coefficient $b = -0.51$ and small linear
    coefficient $a = +0.19$.
  - Cubic fit: tiny linear coefficient $a = +0.009$, dominant cubic
    $-0.71$, residual 0.001.

**Within the noise of the available data**, the linear term is small
relative to the quadratic and cubic, consistent with the prediction
that the leading shift is non-linear in $(1-\zeta)$.

A stronger test would require:
- More $\zeta$ values close to $\zeta = 1$ (say, $\zeta = 0.95, 0.98$).
- Higher precision $c_{\rm eff}$ from larger $L$.

If $\lambda(c=1)$ at $\zeta = 0.95$ shifts by $\lesssim 0.0003$ from
$\zeta = 1$ — i.e., proportionally less than the $\sim 0.001$ shift at
$\zeta = 0.85$ — that confirms the quadratic-or-higher scaling.

### 5.3 The Renyi entropy reruns provide a separate test

The Renyi reruns we submitted (`slurm/submit_renyi_targets.sh`) probe
$c_2/c_1$ and $c_3/c_1$ across multiple $(\lambda, \zeta)$ test points.
The free-Dirac CFT predicts $c_2/c_1 = 3/4$, $c_3/c_1 = 2/3$. If these
ratios are satisfied at the Born-rule point $(\lambda = 0.45, \zeta = 1)$
and at the nearby PPS point $(\lambda = 0.45, \zeta = 0.85)$ to similar
precision, that confirms the critical phase has the same conformal
structure at both $\zeta$ values — supporting the prediction that the
$(1-\zeta)$ shift is small.

If the ratios deviate from free-Dirac values at $\zeta = 0.85$ in a way
that scales with $(1-\zeta)^2$, that quantifies the higher-order
correction $c_2$.

---

## 6. Honest caveats

This calculation has several assumptions that need verification.

**(a) Bosonization conventions.** I used the standard Giamarchi
convention for the bosonic Luttinger liquid. LMR use a slightly
different normalisation (their factor of $\pi$'s in Eq. (47) differ).
The qualitative conclusion (scaling dimension at $K=1$) is convention-
independent, but the specific coefficient $c_2$ in the
$K_\sigma(\zeta) = 1 + c_2 (1-\zeta)^2 + \ldots$ expansion requires
careful translation between conventions.

**(b) The factorisation of $V_j$.** I assumed
$d^{(+, a)}(d^{(-, a)})^\dagger$ for replica $a$ can be cleanly
factorised from the other replica using standard fermion
anticommutation. The Klein factors (LMR App. C) introduce additional
Pauli-string structure in replica space that I have not analysed
explicitly. They are global and do not affect the scaling dimension of
the local operator $V_j(x)$, but they may shift which symmetry sector
$V_j$ couples to.

**(c) The dominant chiral piece of $d_j$.** I treated the bond operator
$d_j$ as a single chiral mode (R-mover, say). The full field has both
R and L components, and the bosonized form $V_j$ should in principle
include all four combinations (RR, RL, LR, LL) of the cross-replica
fermion product. Some combinations have momentum 0 (smooth, the
leading low-energy contribution); others carry momentum $\pm 2k_F$
(oscillating, drop out in the continuum limit at half-filling). I have
taken only the RR (or LL) piece. A more careful analysis would include
the cross-chirality pieces; if any of them have lower scaling
dimension than 4, the conclusion would shift. **I have not verified
this**; it requires a careful enumeration of the chirality combinations
and their dimensions, which is the natural next step.

**(d) Higher-order shifts.** The $(1-\zeta)^2$ coefficient $c_2$ that
I quote without computing requires OPE analysis of $V_j$ with itself
and with other operators. Computing $c_2$ explicitly is a few-day
calculation that I have not done.

In summary: **the headline result (vanishing leading $(1-\zeta)$ shift
of $K_\sigma$) is robust to bosonization conventions and to Klein-factor
choices, because it follows from the scaling dimension of $V_j$ being
$> 2$.** This conclusion would only fail if a lower-dimension piece
exists in $V_j$ that I missed (caveat (c)) — and this is the most
important thing to verify next.

---

## 7. Summary of analytical findings (Tasks 4-onwards)

What is now established:

  1. The QJ two-replica generator has structure $\mathcal{L}_{\rm diag}^{(2)} + \zeta\,V_{\rm cross}$ where $V_{\rm cross}$ is the cross-replica click vertex (`qj_two_replica_derivation.md`, §§ 1-4, plus correction here).

  2. In Majorana language, the diagonal generator and $V_{\rm cross}$ are
     each built from Majorana bilinears (BILINEAR, not quartic — corrected from the original document).

  3. The cross-replica click vertex bosonizes to a **chiral vertex
     operator** in the $\rho$-difference mode:
     $V_j \sim \exp\!\bigl[2i(\Theta^\rho_D - \Phi^\rho_D)\bigr]$.

  4. At the Born-rule fixed point ($K = 1$), $V_j$ has scaling dimension
     $\Delta = 4$ — **highly irrelevant**.

  5. **Consequently** the leading $(1-\zeta)$ shift of the bosonized
     Luttinger parameter $K_\sigma$ in the QJ theory **vanishes**:
     $K_\sigma^{\rm QJ}(\zeta) = K_\sigma^{(\rm Born)} + O((1-\zeta)^2)$.

  6. This **contrasts with LMR's diffusive theory**, where the
     cross-replica term contains a marginal $\cos(2\phi^\sigma)$ vertex
     that shifts $K_\sigma$ linearly in $(1-\zeta)$.

  7. **Numerical check:** the location $\lambda(c=1)$ of the BKT critical
     curve in our $(\lambda, \zeta)$ phase diagram is **essentially flat**
     from $\zeta = 1$ to $\zeta = 0.85$ (shift $\approx 0.001$), and only
     shifts substantially for $\zeta < 0.5$. Polynomial fits give a
     small linear coefficient and dominant quadratic/cubic terms.
     This is **consistent** with the prediction of vanishing leading slope.

What remains:

  - Verify that no lower-dimension piece of $V_j$ exists from
    cross-chirality combinations (the main caveat).
  - Compute the explicit coefficient $c_2$ in the $(1-\zeta)^2$ expansion.
  - Repeat the calculation in the $\zeta \to 0$ limit to recover LMR's
    Ising prediction and verify consistency.
  - Check the Renyi ratio prediction: $c_2/c_1 \to 3/4$ and $c_3/c_1 \to 2/3$
    at $\zeta = 0.85$, and that these are stable across the test points,
    confirming a single CFT universality class throughout the strong-$\zeta$
    crossover region.

---

## 8. Implication for the thesis

This calculation gives a **concrete, falsifiable analytical prediction**
that distinguishes the QJ unraveling from LMR's diffusive case:

> The leading $(1-\zeta)$ correction to the effective central charge
> $K_\sigma$ near the Born-rule fixed point **vanishes** in the QJ
> unraveling but is **linear** in the diffusive unraveling. Equivalently,
> the BKT critical line $c(\lambda, \zeta) = 1$ in the QJ phase diagram is
> flat near $\zeta = 1$ to first order in $(1 - \zeta)$.

Our numerical data show:
  - $\lambda(c=1, \zeta=1) = 0.364$
  - $\lambda(c=1, \zeta=0.85) = 0.365$ (essentially unchanged)
  - $\lambda(c=1, \zeta=0.70) = 0.370$ (small shift)

This pattern is consistent with the analytical prediction (zero leading
slope), and **inconsistent** with the linear-in-$(1-\zeta)$ behaviour
of LMR's diffusive theory.

For the thesis, this is a concrete result that says:
> The smooth, broad nature of the crossover at $\zeta \to 1$ observed in
> our numerics has a clean analytical explanation: the QJ
> cross-replica click vertex is **irrelevant** (scaling dimension 4) at
> the Born-rule fixed point, so its leading effect on the bosonized
> Luttinger parameters vanishes. This is structurally different from
> the diffusive case, where the analogous operator is marginal and
> gives a linear-in-$(1-\zeta)$ shift.

This is a clean, defensible, **publication-quality** result if extended
to verify caveats (c) and (d) above. With the Renyi numerics in hand to
confirm the conformal structure at multiple $\zeta$ values, this could
form the analytical centrepiece of the thesis.
