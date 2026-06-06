# Response to collaborator on Problems 1 and 2

Thank you for the careful work on both problems. Your analysis significantly tightens
the theoretical story and forces a more honest framing. I want to share one important
piece of additional information about the actual code, and pose a follow-up question.

## 1. Acknowledged: the IR dimension question

Your argument on Problem 1 is convincing. The cross-vertex's $\partial S_{\rm NLSM}/\partial\zeta$
is not naturally a dimension-1 primary in the Le Gal-Schirò NLSM; it most likely
renormalizes the marginal NLSM stiffness coupling. I will not claim $\Delta_\zeta^{\rm IR}=1$
in the thesis. The honest statement is:

- $\Delta_\zeta^{\rm UV}=1$: microscopically established.
- The matched argument runs $\zeta$ with the UV scaling up to the crossover scale
  $\xi_\lambda^{\rm cross}$, then matches to the NLSM.
- The argument works **provided** the critical condition is reached at the
  matching scale, so anomalous IR running affects only $K^*$ and subleading
  corrections.

This is now in the LaTeX section explicitly (see the "Validity of the matched argument"
subsubsection of `sec_matching_revised.tex`).

## 2. One important correction to the model assumption in Problem 2

Your BdG analysis used $H_{\rm hop} = (iw/2)\sum_j B_j(A_{j-1}+A_{j+1})$, i.e., a
**distance-1 nearest-neighbor Majorana hopping**, plus the distance-3 measurement
bond. With that model, you correctly derived $q(k) = 2w\cos k + (i\alpha/2)e^{2ik}$
and $\xi \sim 1/\alpha$.

However, **the actual code has both distance-1 AND distance-3 bonds in the hopping
Hamiltonian itself** — not just in the measurement. Direct inspection of
`pps_qj/gaussian_backend.py:53-65` (the function `majorana_hamiltonian_generator`):

```python
def majorana_hamiltonian_generator(L: int, w: float) -> np.ndarray:
    h = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)   # (2*bond, 2*bond+3) — distance 3
        c = 2 * bond + 1
        d = 2 * bond + 2              # distance 1
        h[a, b] = w; h[b, a] = -w     # distance-3 bond with weight +w
        h[c, d] = -w; h[d, c] = w     # distance-1 bond with weight -w
    return h
```

At $L=6$, this gives 5 distance-1 bonds AND 5 distance-3 bonds. This is the
**Kitaev chain at the topological point** ($\mu = 0$, $\Delta = w$) expressed in
Majorana language, where both terms emerge from $-w(c^\dagger_{j+1}c_j + c^\dagger_{j+1}c^\dagger_j + \mathrm{h.c.})$.

The measurement then only modifies the imaginary part of the distance-3 bond:
$h_{\rm eff}[a,b] = w - i\alpha$ for the distance-3 bond, $h_{\rm eff}[c,d] = -w$ for
the distance-1 bond.

I redid the Bloch analysis with this corrected structure:
$$H_{AB}(k) = (w - i\alpha) e^{+ik} + w\, e^{-ik}, \qquad H_{BA}(k) = -H_{AB}(-k).$$

The eigenvalues $\pm\sqrt{H_{AB}(k)\cdot H_{BA}(k)}$ give, at $k = \pi/2$,
*purely real* dispersion — the imaginary part vanishes exactly there. Direct
numerics confirms: **the actual code's no-click $H_{\rm eff}$ is effectively
gapless** (the imaginary part of the spectrum reaches zero at certain bulk momenta).

| Model | $\min_k |\mathrm{Im}\,E(k)|$ vs $\alpha$ | $\xi$ scaling |
|---|---|---|
| Your simplified (distance-1 hopping only) | $\sim \alpha^{1}$ | $\xi \sim \alpha^{-1}$ |
| Actual code (distance-1 + distance-3 hopping = Kitaev topo point) | $\sim \alpha^{0}$ (gapless) | algebraic decay |

So your $\xi \sim \alpha^{-1}$ is correct for the simplified model, but the actual
code's no-click dynamics is gapless at the single-particle level. There is no
clean single-particle no-click length scale for this exact model.

## 3. The KMR connection clarifies the situation

We are essentially on the $\gamma = 0$ edge of the Kells-Meidan-Romito (SciPost 2023)
phase diagram (Kitaev chain Hamiltonian + single $d_j^\dagger d_j$ measurement),
**but with quantum-jump (QJ) unraveling rather than quantum-state-diffusion (QSD)**.

The Lindbladian is the same as KMR's $\gamma = 0$ edge. KMR's $\xi_{\rm ps} \sim \lambda^{-2}$
result is a genuine property of their **QSD** no-click dynamics. Our **QJ** no-click
dynamics has a different structure (gapless at single-particle level, as above).

Universal critical exponents ($\nu$, $\lambda_c$, $\phi$) are unraveling-independent,
so $\nu = 2$ applies in both QSD and QJ. The matched-NLSM derivation with $\nu = 2$
giving $\phi = 1/2$ remains correct. The microscopic identification of $\xi_\lambda^{\rm cross}$
with a single-particle length scale fails for QJ Case B; the $\lambda^{-2}$ scale
must be treated purely as the universal class-DIII multicritical NLSM crossover length.

## 4. The remaining theoretical question I would value your help with

Given the above, the cleanest framing of the matched-NLSM argument is:

> The bare cross-vertex coupling $\zeta$ runs from the lattice scale $a$ to the
> crossover scale $\xi_\lambda^{\rm cross} = c_\lambda \lambda^{-2} a$ with the
> UV eigenvalue $y_\zeta^{\rm UV} = 1$. The critical condition $\zeta_{\rm eff}(\xi_\lambda^{\rm cross}) = K^*$
> gives $\lambda_c = \sqrt{c_\lambda/K^*}\,\sqrt{\zeta}$. This works at the
> Lindbladian (averaged) level and is independent of the choice of unraveling.

**The question:** Is there a clean way to argue, at the Lindbladian or replica-trick
level, that the relevant matching scale is $\lambda^{-2}$ for the QJ-PPS Case B problem
**without** appealing to a specific unraveling's single-particle no-click length?
Specifically:

- The replica Keldysh action is computed at the Lindbladian level (averaged trajectories),
  not at fixed unraveling.
- The NLSM derivation following Le Gal-Schirò gives DIII class with target $SO(R)$.
- The class-DIII multicritical NLSM has $\nu = 2$ as a property of the field theory.
- Identifying $\xi_\lambda^{\rm cross} = \lambda^{-2}$ with the multicritical correlation
  length is correct **by definition** of $\nu$.

Is this the right way to close the argument, or is there still a hidden dependence
on the no-click structure that we are missing? Concretely, in your derivation of
the running $\zeta(\ell) \sim \zeta\cdot(\ell/a)^{y_\zeta}$ from $a$ to $\xi_\lambda^{\rm cross}$,
which underlying single-particle physics enters and which doesn't?

Any thoughts on whether the $\lambda^{-2}$ scale can be derived purely from
field-theoretic / RG considerations (i.e., from the linearized RG eigenvalues at the
multicritical point $(0,0)$) without microscopic single-particle input, would close
the last conceptual gap in our thesis presentation.

## 5. What I am NOT asking for

I am not asking for the one-loop computation of $\Delta_\zeta^{\rm IR}$ (your Problem 1
analysis already gave a strong answer that this is not 1) or for the precise value of
$K^*$ (non-universal and scheme-dependent). The thesis will present $C \approx 0.9$
empirically and note that the precision computation is out of scope.

Thank you again for the substantive work — it has improved the theory considerably.
