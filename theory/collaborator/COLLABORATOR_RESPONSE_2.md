# Final response to collaborator — argument closed

Thank you for the careful and decisive response. Your derivation closes the loop
cleanly and the thesis section has been updated accordingly. A brief acknowledgment
of what we now consider settled, plus one extra piece of context that may be useful
for any future work on Case~A.

## What is now settled

The matched-NLSM derivation of $\lambda_c\sim\sqrt{\zeta}$ for our QJ Case~B
model rests on exactly two ingredients:

1. **UV scaling dimension of the cross-Choi vertex** at the free-Dirac fixed point:
   $\Delta_\zeta^{\rm UV}=1$, microscopically verified via the lattice cross-Choi
   two-point function $\langle\mathcal V_\zeta(x)\mathcal V_\zeta(0)\rangle\sim x^{-2}$.

2. **Class-DIII multicritical exponent**: $\nu=2$, taken from König--Brouwer
   PRB 90, 165140 (2014) and Le~Gal--Schirò arXiv:2511.22506.

The linearised-RG argument at the multicritical point $(\lambda,\zeta)=(0,0)$ then
gives, with no further microscopic input:
$$\xi_\lambda^{\rm cross}\sim\lambda^{-2}\ \text{(definition of }\nu),\quad
\zeta_{\rm eff}(\xi_\lambda^{\rm cross})\sim\zeta\,\lambda^{-2},\quad
\lambda_c\sim\sqrt{\zeta}.$$

The $\lambda^{-2}$ scale is a class-DIII multicritical RG crossover length, **not**
a microscopic no-click localisation length. The QJ no-click effective Hamiltonian
$H_{\rm eff}=H-\tfrac{i\alpha}{2}\sum_j d_j^\dagger d_j$ for our actual
distance-3 Majorana bond is effectively gapless at the single-particle level, in
contrast to the QSD no-click problem of KMR (SciPost Phys 14, 031, 2023) and
LMR (Leung--Meidan--Romito PRX 15, 021020, 2025) where a BdG inverse gap
$\sim\lambda^{-2}$ emerges from the continuous-monitoring effective dynamics.

The universal critical exponents are unraveling-independent at the Lindbladian
level, so $\phi=1/2$ is unaffected by the different microscopic no-click structure
in our QJ unraveling.

## The residual assumption

The argument additionally assumes that the microscopic $\lambda$ has nonzero linear
overlap with the DIII multicritical relevant scaling field, so that
$\xi_\lambda^{\rm cross}\sim\lambda^{-1/y_\lambda}$. This is the standard generic
expectation; an accidental vanishing of the linear coefficient would force
$\phi=1$ instead of $\phi=1/2$. The empirical value $\phi=0.56\pm0.05$ excludes
$\phi=1$ at $9\sigma$, supporting the linear-overlap assumption.

This assumption is now stated explicitly as a bullet in the thesis section's
"Validity" subsubsection.

## Context: the place in the KMR / LMR / our project lineage

For completeness, the model lineage:
- **KMR 2023** (SciPost Phys 14, 031): deterministic Kitaev chain + two
  on-site Bogoliubov measurements, QSD unraveling, Born rule.
- **LMR 2025** (PRX 15, 021020): random unitaries + two competing Majorana
  bond-parity measurements, QSD unraveling, PPS introduced via the PPS-SSE.
  Finds Ising MIPT ($\nu=1$) for strong PPS, monitored MIPT ($\nu=5/3$) for
  weak PPS, with a sharp universality change at $\zeta\approx 0.28$.
- **Our project**: deterministic Kitaev chain + single Bogoliubov bond
  measurement + PPS, QJ unraveling. The Lindbladian is the $\gamma=0$ edge of
  KMR's diagram; the PPS framework is taken from LMR (their PPS-SSE
  construction adapted to QJ); but our universality is class DIII (not Ising,
  not LMR's strong/weak PPS dichotomy) because of the deterministic Hamiltonian
  and the single-measurement structure.

So our project sits at a non-overlapping point of the KMR-LMR design space, and
the $\phi=1/2$ prediction is genuinely new to the QJ-PPS Case~B regime.

## Open theoretical questions (for the curious)

These are not blocking for the thesis but would be valuable extensions:

1. **Case~A** ($D$-class with self-duality $\alpha\leftrightarrow\gamma$): under
   PPS, is $\lambda_c=1/2$ strictly pinned for all $\zeta\in(0,1]$, or does PPS
   generate a slight shift through a subleading mechanism? Theoretical
   prediction is rigid pinning; numerics not yet available.

2. **Crossover function**: is there a derivation of the closed-form
   $\lambda_c(\zeta)=\sqrt{\zeta}/(1+\sqrt{\zeta})$ from a two-parameter RG
   separatrix analysis, or is this purely a phenomenological interpolation? The
   empirical fit is excellent without any free parameters, which suggests there
   may be a derivation.

3. **Anomalous IR dimension**: even if not needed for $\phi=1/2$, computing
   $\Delta_\zeta^{\rm IR}$ at the class-DIII NLSM fixed point would settle the
   role of subleading corrections and finite-size scaling at the small-$\zeta$
   end.

Thank you again for the substantive analytical work. The theory chapter is now
on solid footing and the matched-NLSM framing is honest about what is derived,
what is borrowed from the literature, and what is assumed.
