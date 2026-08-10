"""
Simple jump-count distribution validation.

Shows that the cloning algorithm samples from the tilted measure
Q_s ∝ ζ^{N_T} P_Born by comparing the distribution of total jump counts
N_T accumulated over [0, T] between:

  - Born rule (unbiased IS, grey)
  - Q_s reference (IS reweighted by ζ^{N_T}, blue)
  - Cloning for N_c = 50, 200, 500 (increasingly dark red)

Since ζ = 0.7 < 1, Q_s suppresses trajectories with many jumps.
As N_c grows the cloning histogram converges to the Q_s reference.

The cumulative jump count per clone is tracked through the ancestry chain:
at each resampling step the jump counts are resampled with the same indices
as the covariance matrices, so each surviving clone carries the total jumps
it (and its ancestors) experienced from t=0 to t=T.

Runtime: ~3-4 minutes (dominated by IS at M=3000).

Usage::
    python scripts/validate_jump_distribution.py
Output: scripts/validation_jump_distribution.pdf + .png
"""
from __future__ import annotations

import time
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from pps_qj.gaussian_backend import (
    build_gaussian_chain_model,
    gaussian_born_rule_trajectory,
)
from pps_qj.cloning import _systematic_resample_pairs

# ---------------------------------------------------------------------------
# Parameters — same system as validate_cloning.py for consistency
# ---------------------------------------------------------------------------
L     = 8
LAM   = 0.40
ALPHA = LAM
W     = 1.0 - LAM
ZETA  = 0.70
T     = 2.0    # Short T keeps IS tractable: mean_jumps * (zeta-1)^2 ~ 0.5 << 1.
               # At T=10 this is ~2.3, causing IS weight collapse (a handful of
               # rare low-N_T trajectories get 5000x the weight of typical ones
               # and dominate the histogram — the IS estimator fails).
               # At T=2 the mean jumps ~5.6 and ESS/M ~ 0.6, giving a reliable
               # reference.  The cloning algorithm is validated here; its
               # correctness then extends to any T by construction.
M_IS  = 5_000

NC_VALS = [200, 5000, 10000]


# ---------------------------------------------------------------------------
# IS reference: N_T distribution under P_Born and Q_s
# ---------------------------------------------------------------------------

def importance_sampling_jump_counts(
    model, zeta: float, T: float, M: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run M Born-rule trajectories, return (n_jumps_array, is_weights).

    is_weights[i] = ζ^{n_i} / Z  gives the Q_s probability weight for
    trajectory i.  Plotting a weighted histogram of n_jumps with these
    weights gives the exact Q_s distribution of N_T.
    """
    cov0, orb0 = model.gamma0.copy(), model.orbitals0.copy()
    sub_rngs = rng.spawn(M)
    n_jumps = np.zeros(M, dtype=np.int64)

    print(f"  IS: {M} trajectories ...", flush=True)
    t0 = time.perf_counter()
    for i in range(M):
        r = gaussian_born_rule_trajectory(
            replace(model, gamma0=cov0, orbitals0=orb0),
            T=T, rng=sub_rngs[i],
        )
        n_jumps[i] = r.n_jumps
    print(f"  IS done ({time.perf_counter()-t0:.0f}s)  "
          f"mean N_T={n_jumps.mean():.1f}  range=[{n_jumps.min()},{n_jumps.max()}]",
          flush=True)

    log_w = n_jumps * np.log(zeta)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()

    ess = float(1.0 / np.sum(w ** 2))
    ess_frac = ess / M
    flag = "OK" if ess_frac > 0.1 else "WARNING: IS reference unreliable"
    print(f"  IS ESS = {ess:.0f} / {M}  ({100*ess_frac:.1f}%)  {flag}", flush=True)
    if ess_frac < 0.1:
        print(f"  --> Reduce T or increase zeta for a reliable IS reference.", flush=True)

    return n_jumps, w


# ---------------------------------------------------------------------------
# Cloning with cumulative jump tracking
# ---------------------------------------------------------------------------

def _resample_indices(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling — return the index array."""
    N = len(weights)
    F = np.cumsum(weights / weights.sum()); F[-1] = 1.0
    U = float(rng.uniform(0.0, 1.0 / N))
    return np.clip(np.searchsorted(F, U + np.arange(N) / N, "left"), 0, N - 1)


def cloning_jump_counts(
    model, zeta: float, T: float, N_c: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Run the cloning algorithm and return the cumulative jump count
    accumulated by each final clone along its full ancestry chain.

    After systematic resampling the N_c final clones have equal weight,
    so their unweighted histogram estimates P_{Q_s}(N_T).
    """
    delta_tau = 1.0 / max(2.0 * model.alpha * (model.L - 1), 1e-6)
    n_steps   = int(np.ceil(T / delta_tau))
    dt_eff    = T / n_steps

    covs     = [model.gamma0.copy() for _ in range(N_c)]
    orbs     = [model.orbitals0.copy() for _ in range(N_c)]
    cum_jumps = np.zeros(N_c, dtype=np.int64)   # ancestry-chain jump totals

    for _ in range(n_steps):
        sub_rngs = rng.spawn(N_c)
        step_jumps = np.zeros(N_c, dtype=np.int64)

        for i in range(N_c):
            r = gaussian_born_rule_trajectory(
                replace(model, gamma0=covs[i], orbitals0=orbs[i]),
                T=dt_eff, rng=sub_rngs[i],
            )
            covs[i]       = np.asarray(r.final_covariance, dtype=np.float64)
            orbs[i]       = r.final_orbitals
            step_jumps[i] = r.n_jumps

        cum_jumps += step_jumps   # each clone accumulates its personal jumps

        # Systematic resampling by weight ζ^{step_jumps}
        log_w = step_jumps * np.log(zeta)
        lw_max = log_w.max()
        w = np.exp(log_w - lw_max)

        idxs      = _resample_indices(w, rng)
        covs      = [covs[int(i)].copy() for i in idxs]
        orbs      = [orbs[int(i)].copy() for i in idxs]
        cum_jumps = cum_jumps[idxs]   # inherited jump history follows ancestry

    return cum_jumps


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    n_born:  np.ndarray,          # IS jump counts, unweighted (Born rule)
    n_qs_w:  np.ndarray,          # IS weights for Q_s
    clone_counts: dict,           # {N_c: np.ndarray of cum_jumps}
    outpath: Path,
):
    fig, axes = plt.subplots(1, len(NC_VALS), figsize=(5 * len(NC_VALS), 4.5),
                             sharey=False)

    n_max = max(n_born.max(),
                max(c.max() for c in clone_counts.values())) + 2
    bins  = np.arange(0, n_max + 1) - 0.5   # integer-centred bins

    # Pre-compute reference histograms (same for every panel)
    h_born, _ = np.histogram(n_born, bins=bins, density=True)
    h_qs, _   = np.histogram(n_born, bins=bins, weights=n_qs_w, density=False)
    h_qs      = h_qs / h_qs.sum()   # normalise to probability mass

    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    REDS = ["#FCBBA1", "#FB6A4A", "#99000D"]   # light → dark for growing N_c

    for ax_idx, (N_c, color) in enumerate(zip(NC_VALS, REDS)):
        ax = axes[ax_idx]
        h_clone, _ = np.histogram(clone_counts[N_c], bins=bins, density=True)

        # Born-rule reference (grey, filled)
        ax.bar(bin_centres, h_born, width=1.0, color="#CCCCCC", alpha=0.6,
               label="Born rule $P_{\\mathrm{Born}}(N_T)$", zorder=1)

        # Q_s reference from IS (blue line + dots)
        ax.step(bins[:-1], h_qs, where="post", color="#2166AC", lw=2.0,
                label=f"$Q_s$ ref. (IS, $M={M_IS}$)", zorder=3)

        # Cloning histogram
        ax.step(bins[:-1], h_clone, where="post", color=color, lw=2.0,
                ls="--", label=f"Cloning ($N_c = {N_c}$)", zorder=4)

        # KS test: draw iid bootstrap samples from IS Q_s vs cloning
        w_norm = n_qs_w / n_qs_w.sum()
        rng_ks = np.random.default_rng(ax_idx)
        s_is = rng_ks.choice(n_born, size=M_IS, replace=True, p=w_norm)
        ks, pval = ks_2samp(s_is, clone_counts[N_c])

        verdict = "not rejected" if pval > 0.05 else "rejected"
        ax.text(0.97, 0.97,
                f"KS = {ks:.3f}\n$p = {pval:.3f}$\n({verdict})",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888",
                          alpha=0.85))

        ax.set_xlabel("Total jump count $N_T$", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Probability", fontsize=11)
        ax.set_title(f"$N_c = {N_c}$", fontsize=12)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(bins[0], min(bins[-1], bin_centres[h_qs > 0.001].max() + 5))

    fig.suptitle(
        rf"Jump-count distribution: Born rule vs $Q_s$ ($\zeta={ZETA}$), "
        rf"$L={L}$, $\lambda={LAM}$, $T={T}$",
        fontsize=11,
    )
    fig.tight_layout()

    for ext in (".pdf", ".png"):
        p = outpath.with_suffix(ext)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model   = build_gaussian_chain_model(L=L, w=W, alpha=ALPHA)
    master  = np.random.default_rng(20250429)
    rng_IS, *rng_clones = master.spawn(1 + len(NC_VALS))

    print(f"Parameters: L={L}, λ={LAM}, ζ={ZETA}, T={T}")
    print(f"δτ = {1.0/max(2*ALPHA*(L-1),1e-6):.4f}")

    n_born, w_qs = importance_sampling_jump_counts(model, ZETA, T, M_IS, rng_IS)

    clone_counts = {}
    for N_c, rng_c in zip(NC_VALS, rng_clones):
        print(f"\nCloning N_c={N_c} ...", flush=True)
        t0 = time.perf_counter()
        clone_counts[N_c] = cloning_jump_counts(model, ZETA, T, N_c, rng_c)
        print(f"  done ({time.perf_counter()-t0:.0f}s)  "
              f"mean={clone_counts[N_c].mean():.1f}  "
              f"range=[{clone_counts[N_c].min()},{clone_counts[N_c].max()}]",
              flush=True)

    make_figure(
        n_born, w_qs, clone_counts,
        Path(__file__).parent / "validation_jump_distribution",
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
