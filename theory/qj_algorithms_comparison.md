# Continuous-time waiting-time vs discrete-time stepped algorithms

A clarifying note on the equivalence (and the subtle differences) between
the two standard ways of simulating quantum-jump trajectories. I wrote
this in response to a confusion between Utku's implementation (which
samples waiting times via inversion of the survival probability) and
Dganit's implementation (which propagates state in fixed time steps
$\Delta t$ and rolls a die at each step).

**Short answer:** they target the *same* SSE and produce statistically
identical trajectories in the limit $\Delta t \to 0$. They differ in
operational realization and in their finite-$\Delta t$ bias. Neither is
"more fundamental"; they are two implementations of the same
mathematical object.

---

## 1. The mathematical object both algorithms target

The standard quantum-jump SSE for a single channel $L$ at rate $\gamma$
(Wiseman & Milburn, Carmichael):
$$
d|\psi\rangle \;=\; \left[\,-i\,H_{\rm eff}\,dt
 \;+\; \tfrac{1}{2}\gamma\,\langle L^\dagger L\rangle dt
 \;+\; \bigl(\tfrac{L}{\sqrt{\langle L^\dagger L\rangle}} - \mathbb{1}\bigr) dN_t\,\right]|\psi\rangle,
$$
where $H_{\rm eff} = H - (i\gamma/2) L^\dagger L$ and $dN_t \in \{0, 1\}$ is
a Poisson point process with conditional intensity
$\gamma\,\langle L^\dagger L\rangle_t$.

A trajectory is a sequence of "click" times $0 < t_1 < t_2 < \cdots < t_n < T$
with associated channels (if more than one). Between clicks, the state
evolves deterministically by the non-Hermitian Hamiltonian
$H_{\rm eff}$ and the normalisation drifts. At each click, the state is
mapped through $L/\|L|\psi\rangle\|$ and renormalised.

The *probability distribution* over trajectories is determined by:
1. The non-Hermitian propagator $e^{-i H_{\rm eff}\,\Delta t}$ (no-click
   propagation).
2. The instantaneous jump rate $r(t) = \gamma\,\langle L^\dagger L\rangle_t$.

Both algorithms below sample from this distribution. They differ only in
how they generate the random click times.

---

## 2. Algorithm A — Waiting-time inversion (Utku's pps_qj)

This is the **continuous-time** algorithm. It exploits the fact that, given
the current state $|\psi_t\rangle$, the *probability of no click in the
interval $[t, t+\Delta t]$* is exactly
$$
S(\Delta t) \;=\; \frac{\bigl\|\,e^{-i H_{\rm eff}\,\Delta t}\,|\psi_t\rangle\,\bigr\|^2}{\|\,|\psi_t\rangle\,\|^2},
$$
called the **branch-norm** or **survival probability**.

### 2.1 Sampling step

1. Draw a uniform random number $U \in (0, 1)$.
2. Find the time $\Delta t^*$ such that $S(\Delta t^*) = U$. This is the
   waiting time until the next jump.
3. If $\Delta t^* > T_{\rm rem}$ (the remaining time to the end of the
   trajectory), there is no jump; propagate to $T$ and stop.
4. Otherwise propagate the state to $t + \Delta t^*$ using
   $e^{-i H_{\rm eff}\,\Delta t^*}$, renormalise.
5. Sample the jump channel proportional to
   $p_j \propto \langle\psi | L_j^\dagger L_j | \psi\rangle$.
6. Apply $L_j$ to the state, renormalise. Set $t \to t + \Delta t^*$ and
   repeat.

This is implemented in `pps_qj/gaussian_backend.py::gaussian_born_rule_trajectory`,
using `scipy.optimize.brentq` to find $\Delta t^*$ from
$S(\Delta t) - U = 0$.

### 2.2 Why this is exact

The waiting-time density $w(\Delta t)$ for the first jump after time $t$ is
$$
w(\Delta t) \;=\; -\frac{dS(\Delta t)}{d\Delta t}.
$$
Sampling $\Delta t^*$ by inversion $S(\Delta t^*) = U$ with $U \sim \text{Uniform}(0,1)$
is exactly equivalent to sampling $\Delta t^*$ from $w$. So:
$$
\Delta t^* \;\sim\; w(\Delta t) \quad\Longleftrightarrow\quad S(\Delta t^*) = U,\ U \sim \text{Uniform}.
$$

No discretisation error. The only approximations are numerical:
- The matrix exponential $e^{-i H_{\rm eff}\,\Delta t}$ is computed using
  the cached eigendecomposition (exact up to floating-point error).
- The root-finding is iterative to tolerance $10^{-6}$.

### 2.3 Computational cost per trajectory

Cost per jump:
- One eigendecomposition (cached at model construction, free in the
  hot path).
- A handful of $L \times L$ matrix multiplies for the brentq evaluations.
- One $\mathcal{O}(L)$ jump-channel selection.

Total cost: $\mathcal{O}(L^2 \cdot N_T)$ where $N_T$ is the number of
jumps. Since $\langle N_T\rangle$ scales linearly with $T$ in the
Born-rule regime, total trajectory cost is $\mathcal{O}(L^2 \cdot T)$
*excluding the eigendecomposition*. (The eigendecomp is the dominant
one-time cost.)

---

## 3. Algorithm B — Stepped trajectory (Dganit's implementation)

This is the **discrete-time** algorithm. It chops $[0, T]$ into $N$ steps
of size $\Delta t = T/N$ and at each step does:

### 3.1 Sampling step

1. Compute the click probability for this step: $p_{\rm click} = \gamma\,\langle L^\dagger L\rangle_t\,\Delta t$.
2. Draw $U \sim \text{Uniform}(0,1)$.
3. If $U < p_{\rm click}$: a click happened in this step. Apply the jump
   operator $L$, renormalise. (If multiple channels, choose channel by
   conditional probability proportional to $\langle L_j^\dagger L_j\rangle$.)
4. If $U \geq p_{\rm click}$: no click. Propagate by $e^{-i H_{\rm eff}\,\Delta t}$,
   renormalise.
5. Advance $t \to t + \Delta t$ and repeat.

### 3.2 Approximation error

The stepped algorithm has $\mathcal{O}(\Delta t)$ error in the trajectory
distribution. Specifically:
- Two clicks within the same $\Delta t$ window are *not* resolved — the
  algorithm only registers one click per step.
- The click time is quantised to the step boundary (or to a uniform
  position within the step, depending on implementation), introducing
  $\mathcal{O}(\Delta t)$ jitter in $t_k$.
- The state at the click moment is approximated as $|\psi_t\rangle$
  evaluated at the step boundary, not at the actual (uncomputed) click
  time within the step.

For "thermodynamic" observables averaged over many trajectories,
$\mathcal{O}(\Delta t)$ bias is acceptable as long as $\Delta t$ is small
enough — typically the requirement is $\gamma \langle L^\dagger L\rangle\,\Delta t \ll 1$,
i.e., $\Delta t \ll$ inverse click rate.

### 3.3 Computational cost per trajectory

Cost per step:
- One matrix exponential application: $\mathcal{O}(L^2)$ or
  $\mathcal{O}(L^3)$ depending on precomputation.
- One $\mathcal{O}(L)$ click test.

Total cost: $\mathcal{O}(L^2 \cdot N)$ steps, where $N = T/\Delta t$.

**This is a *worse* scaling than waiting-time** if $\Delta t \ll
1/(\gamma\langle L^\dagger L\rangle)$, because we do work in many steps
where nothing happens. The waiting-time algorithm "skips" the no-click
intervals analytically.

### 3.4 But: stepped algorithms have one operational advantage

If you need to apply control operations or measurements at fixed
*pre-determined* times $\{0, \Delta t, 2\Delta t, \ldots\}$ — e.g., in a
hybrid quantum-classical feedback loop, or for a tilted ensemble where
the tilt is applied stepwise — the stepped algorithm naturally
incorporates this. Waiting-time algorithms require additional
bookkeeping to interleave clicks with pre-scheduled control events.

For our problem this advantage is largely irrelevant: the PPS tilt
$\zeta^{N_T}$ is *per-jump*, not per-time-step. So the waiting-time
algorithm is operationally cleaner.

---

## 4. Are they the same procedure in different "operational realizations"?

**Yes — they are sampling schemes for the same statistical distribution.**

A cleaner way to see this: the waiting-time density $w(\Delta t)$ is
related to the instantaneous click rate by
$$
w(\Delta t) \;=\; \gamma\,\langle L^\dagger L\rangle_t \cdot S(\Delta t).
$$
For a small interval $[t, t + \Delta t]$ with $\Delta t \ll
1/(\gamma\langle L^\dagger L\rangle)$:
$$
S(\Delta t) \approx 1 - \gamma\,\langle L^\dagger L\rangle\,\Delta t,
$$
so
$$
\text{Prob}(\text{click in }[t, t+\Delta t]) \approx \gamma\,\langle L^\dagger L\rangle\,\Delta t.
$$
This is exactly Dganit's click probability in step 1 of §3.1. **The
two algorithms agree at first order in $\Delta t$**.

The waiting-time algorithm is what you get from the stepped algorithm in
the limit $\Delta t \to 0$. It is the "continuous-time limit" version.

### 4.1 Where the algorithms differ in finite $\Delta t$

The stepped algorithm with finite $\Delta t$ commits a small but
systematic error:
- Probability of two clicks per step $\sim (\gamma\langle L^\dagger L\rangle\,\Delta t)^2$.
  The algorithm misses these.
- The Poisson statistics are approximated as Bernoulli.

For typical parameters in our project: $\gamma \langle L^\dagger L\rangle \sim \alpha/2 \sim 0.2$
and $\Delta t = 0.1$ gives a click probability per step of $\sim 0.02$,
with two-click probability per step $\sim 4 \times 10^{-4}$. Negligible
for most purposes. But over a long trajectory with $T = 50$ and $\sim 10$
clicks, the bias accumulates to $\sim 1\%$ in averaged observables.

The waiting-time algorithm has none of this bias.

---

## 5. Why does our pipeline use waiting times?

Three reasons specific to our project:

  1. **Free-fermion / Gaussian structure.** The non-Hermitian
     propagator $e^{-i H_{\rm eff}\Delta t}$ has an analytic form in the
     single-particle orbital basis. The branch-norm $S(\Delta t)$ is
     an $L \times L$ Gram-matrix Cholesky logdet (one matmul + a small
     Cholesky), not a $2^L$-dimensional state-vector calculation. So
     the brentq evaluation is cheap.

  2. **PPS weighting is per-jump.** The factor $\zeta^{N_T}$ is
     determined by the number of jumps. With waiting times, $N_T$ is
     explicit and the weight is just $\zeta^{N_T}$. With a stepped
     algorithm, we'd still need to count clicks separately.

  3. **No fixed dt requirement.** Our observables (entanglement at
     time $T$, $\zeta^{N_T}$-weighted) only depend on the final state.
     There's no need for a fixed temporal grid.

For Dganit's spin-1/2 simulations (small Hilbert space, no Gaussian
structure, control fields applied at fixed times), the stepped algorithm
is the natural choice. For our free-fermion problem, waiting-times are
faster and more accurate.

---

## 6. Sanity check: do they agree on observables?

The "Method 2" investigation earlier in this project (cf.
`outputs/dganit_method2_analysis.md`) ran the stepped algorithm at
$\Delta t = 0.05$ and the waiting-time algorithm at full precision on
matched small systems. The first-jump waiting-time distributions
agreed to within sampling error (KS test passed). The disagreement
that ultimately falsified "Method 2" was about a different point — the
two formulations of the PPS-tilted measure — not about discrete-vs-continuous
time sampling.

So: **yes, the two algorithms are equivalent for sampling the same
SSE**. The choice between them is operational.

---

## 7. Summary table

| Feature | Waiting-time (ours) | Stepped (Dganit's) |
|---------|--------------------:|-------------------:|
| Bias in trajectory distribution | None (exact) | $\mathcal{O}(\Delta t)$ |
| Cost per click | $\mathcal{O}(L^2)$ × brentq iters | — |
| Cost per "no-click" interval | $\mathcal{O}(L^2)$ × 1 | $\mathcal{O}(L^2)$ × $T/\Delta t$ |
| Natural for feedback control | No (extra bookkeeping) | Yes |
| Natural for per-jump tilting (PPS) | Yes | Less so |
| Numerical convergence parameter | brentq tolerance ($10^{-6}$) | $\Delta t$ |
| Computational regime where preferred | Continuous-time dynamics, large $T$ between clicks | Discrete-time control, small constant work per step |

The two are not in competition. They are different operational realizations
of the same physical algorithm, with different trade-offs that make one
preferable for different problem structures.
