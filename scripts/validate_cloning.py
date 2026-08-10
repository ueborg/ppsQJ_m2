"""
Validation of the cloning algorithm as a sampler of Q_s ∝ ζ^{N_T} P_Born.

Three-panel figure:

  Panel A  --  SCGF convergence in N_c
      theta_hat = (1/T) log Z_hat_T converges to the IS reference as N_c grows.
      This checks that the weight estimator is unbiased in the N_c->inf limit.
      The systematic upward bias at small N_c is theoretically expected:
      by Jensen's inequality, log E[w] >= E[log w], so finite-N_c undersampling
      of rare high-weight trajectories biases theta_hat upward.

      Why we care: theta is the direct output of the cloning algorithm and has
      an exact IS reference. Agreement confirms the partition function estimate
      is correct. This is a standard diagnostic in the large-deviations
      literature (Lecomte & Tailleur 2007, Carollo et al. 2018).

  Panel B  --  SCGF independence of delta_tau
      The resampling interval delta_tau controls how often we resample but
      should not affect the estimate if the algorithm is working correctly.
      Per-step weight variance grows as sigma^2(log w) ~ zeta^2 * lambda *
      delta_tau * L (Poisson fluctuations in jump counts per window).  The
      estimate is flat for small delta_tau and degrades when delta_tau is large
      enough that a significant fraction of clones gets zero weight in one step.

      Why we care: justifies the production choice delta_tau = 1/(2*alpha*(L-1)).

  Panel C  --  Distribution match under Q_s  [THE KEY VALIDATION]
      Compares the distribution of the half-chain entropy S(T) under Q_s:
        (i)  IS (exact): M Born-rule Gaussian trajectories reweighted by
             zeta^{n_i}.  Bootstrapped to get iid samples from Q_s.
        (ii) Cloning: final clone population after systematic resampling.
      We plot empirical CDFs with bootstrap error bands and report the
      two-sample KS statistic.  If the null (both CDFs from the same
      distribution) is not rejected at p > 0.05, the distributions are
      statistically indistinguishable at this sample size.

      Note: both the IS and the Gaussian backend are EXACT for free-fermion
      systems.  This is not an approximation-vs-exact comparison — it is a
      cross-check of two different estimators of the same exact quantity.

Runtime: approximately 5-10 minutes on a laptop (dominated by IS).

Usage::

    cd /path/to/repo
    source .venv/bin/activate
    python scripts/validate_cloning.py

Output: scripts/validation_cloning.pdf + .png
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
L      = 8       # small enough for fast IS, large enough for non-trivial physics
LAM    = 0.40
ALPHA  = LAM
W      = 1.0 - LAM
ZETA   = 0.70
T      = 10.0

# ---------------------------------------------------------------------------
# Panel A: N_c convergence
# ---------------------------------------------------------------------------
NC_VALS  = [20, 50, 100, 200, 500]
NREAL_A  = 12

# ---------------------------------------------------------------------------
# Panel B: delta_tau sensitivity
# ---------------------------------------------------------------------------
DTAU_FACTORS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
NC_B         = 200
NREAL_B      = 10

# ---------------------------------------------------------------------------
# Panel C + IS reference
# ---------------------------------------------------------------------------
M_IS  = 3_000    # Gaussian IS trajectories — exact for free-fermion
NC_C  = 500
N_BOOT = 2_000   # bootstrap resamples for CDF error bands


# ---------------------------------------------------------------------------
# IS reference using the Gaussian (free-fermion) backend
# ---------------------------------------------------------------------------

def importance_sampling_reference(
    L: int, alpha: float, w: float, zeta: float, T: float,
    M: int, rng: np.random.Generator,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run M Born-rule Gaussian trajectories; reweight by zeta^{n_i}.

    The Gaussian backend is exact for free-fermion systems — it is not an
    approximation, it just uses the 2L×2L covariance representation instead
    of the 2^L statevector.  This is orders of magnitude faster than the
    exact statevector backend and produces the same answer.

    Returns
    -------
    theta_IS    : (1/T) log mean(zeta^n_i)
    theta_std   : bootstrap standard error
    n_jumps_arr : (M,) jump counts
    S_final_arr : (M,) half-chain entropy at time T
    """
    from pps_qj.gaussian_backend import (
        build_gaussian_chain_model, gaussian_born_rule_trajectory,
        entanglement_entropy,
    )
    from dataclasses import replace

    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    n_jumps_arr = np.zeros(M, dtype=np.int64)
    S_final_arr = np.zeros(M, dtype=np.float64)

    print(f"  IS: {M} Gaussian Born-rule trajectories (L={L}) ...", flush=True)
    t0 = time.perf_counter()
    cov0, orb0 = model.gamma0.copy(), model.orbitals0.copy()
    sub_rngs = rng.spawn(M)
    for i in range(M):
        r = gaussian_born_rule_trajectory(
            replace(model, gamma0=cov0, orbitals0=orb0),
            T=T, rng=sub_rngs[i],
        )
        n_jumps_arr[i] = r.n_jumps
        S_final_arr[i] = entanglement_entropy(r.final_covariance, L // 2)
    elapsed = time.perf_counter() - t0

    log_w = n_jumps_arr * np.log(zeta)
    lw_max = log_w.max()
    w_rel = np.exp(log_w - lw_max)
    theta_IS = (np.log(w_rel.mean()) + lw_max) / T

    # Bootstrap SE on theta
    boot = rng.integers(0, M, size=(N_BOOT, M))
    boot_thetas = np.array([
        (np.log(w_rel[idx].mean()) + lw_max) / T
        for idx in boot
    ])
    theta_std = float(np.std(boot_thetas))

    print(f"  IS done: theta={theta_IS:.6f} ± {theta_std:.6f}  ({elapsed:.0f}s  "
          f"{elapsed/M*1000:.1f}ms/traj)", flush=True)
    return float(theta_IS), theta_std, n_jumps_arr, S_final_arr


# ---------------------------------------------------------------------------
# Weighted bootstrap: draw iid samples from the importance-weighted distribution
# ---------------------------------------------------------------------------

def weighted_bootstrap_samples(
    values: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples iid approximate samples from the weighted distribution."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    idx = rng.choice(len(values), size=n_samples, replace=True, p=w)
    return values[idx]


# ---------------------------------------------------------------------------
# Cloning helper
# ---------------------------------------------------------------------------

def cloning_theta(
    L: int, alpha: float, w: float, zeta: float, T: float,
    N_c: int, delta_tau: float, n_real: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.cloning import run_cloning, CloningCollapse

    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    thetas = np.full(n_real, np.nan)
    ESSs   = np.full(n_real, np.nan)
    sub_rngs = rng.spawn(n_real)
    for r in range(n_real):
        try:
            res = run_cloning(
                model, zeta=zeta, T_total=T, N_c=N_c,
                rng=sub_rngs[r], delta_tau=delta_tau,
                record_entropy=False,
            )
            thetas[r] = res.theta_hat
            ESSs[r]   = res.eff_sample_size
        except CloningCollapse:
            pass
    return thetas, ESSs


# ---------------------------------------------------------------------------
# Panel data collectors
# ---------------------------------------------------------------------------

def panel_A_data(rng, dt_default):
    print("\n=== Panel A: N_c convergence ===", flush=True)
    res = {}
    for N_c in NC_VALS:
        t0 = time.perf_counter()
        thetas, ESSs = cloning_theta(L, ALPHA, W, ZETA, T, N_c, dt_default, NREAL_A, rng)
        valid = np.isfinite(thetas)
        res[N_c] = (
            float(np.nanmean(thetas)),
            float(np.nanstd(thetas) / np.sqrt(valid.sum())) if valid.sum() > 1 else np.nan,
            float(np.nanmean(ESSs)),
        )
        print(f"  N_c={N_c:>4}: theta={res[N_c][0]:.5f} ± {res[N_c][1]:.5f}  "
              f"ESS={res[N_c][2]:.1f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return res


def panel_B_data(rng, dt_default):
    print("\n=== Panel B: delta_tau sensitivity ===", flush=True)
    res = {}
    for fac in DTAU_FACTORS:
        dt = dt_default * fac
        t0 = time.perf_counter()
        thetas, ESSs = cloning_theta(L, ALPHA, W, ZETA, T, NC_B, dt, NREAL_B, rng)
        valid = np.isfinite(thetas)
        res[fac] = (
            float(np.nanmean(thetas)),
            float(np.nanstd(thetas) / np.sqrt(valid.sum())) if valid.sum() > 1 else np.nan,
            float(np.nanmean(ESSs)),
        )
        print(f"  factor={fac:.3f}: theta={res[fac][0]:.5f} ± {res[fac][1]:.5f}  "
              f"ESS={res[fac][2]:.1f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return res


# ---------------------------------------------------------------------------
# Panel C: empirical CDF comparison with bootstrap error bands
# ---------------------------------------------------------------------------

def panel_C_data(rng_IS, rng_clone, dt_default):
    print("\n=== Panel C: distribution match ===", flush=True)

    # IS
    rng_IS_traj, rng_IS_boot = rng_IS.spawn(2)
    _, _, n_jumps_IS, S_IS = importance_sampling_reference(
        L, ALPHA, W, ZETA, T, M=M_IS, rng=rng_IS_traj,
    )
    log_w_IS = n_jumps_IS * np.log(ZETA)
    lw_max   = log_w_IS.max()
    w_IS     = np.exp(log_w_IS - lw_max)
    w_IS    /= w_IS.sum()

    # Draw M_IS iid samples from IS distribution for KS test
    S_IS_boot = weighted_bootstrap_samples(S_IS, w_IS, M_IS, rng_IS_boot)

    # Bootstrap CDF error bands for IS
    s_grid = np.linspace(S_IS.min() - 0.05, S_IS.max() + 0.05, 200)
    boot_cdfs_IS = np.zeros((N_BOOT, len(s_grid)))
    for b in range(N_BOOT):
        idx = rng_IS_boot.integers(0, M_IS, size=M_IS)
        boot_cdfs_IS[b] = np.mean(S_IS_boot[idx, None] <= s_grid[None, :], axis=0)
    cdf_IS_lo = np.percentile(boot_cdfs_IS, 5,  axis=0)
    cdf_IS_hi = np.percentile(boot_cdfs_IS, 95, axis=0)
    cdf_IS_mean = np.mean(boot_cdfs_IS, axis=0)

    # Cloning: final clone population (equally weighted after systematic resampling)
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.cloning import run_cloning, _batched_entanglement_entropy

    model = build_gaussian_chain_model(L=L, w=W, alpha=ALPHA)
    print(f"  Cloning N_c={NC_C} ...", flush=True)
    t0 = time.perf_counter()
    result = run_cloning(
        model, zeta=ZETA, T_total=T, N_c=NC_C,
        rng=rng_clone, delta_tau=dt_default, record_entropy=False,
    )
    S_clone = _batched_entanglement_entropy(
        [np.asarray(c, dtype=np.float64) for c in result.final_covs], L // 2
    )
    print(f"  Cloning done  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # Bootstrap CDF error bands for cloning
    boot_cdfs_cl = np.zeros((N_BOOT, len(s_grid)))
    for b in range(N_BOOT):
        idx = rng_clone.integers(0, NC_C, size=NC_C)
        boot_cdfs_cl[b] = np.mean(S_clone[idx, None] <= s_grid[None, :], axis=0)
    cdf_cl_lo   = np.percentile(boot_cdfs_cl, 5,  axis=0)
    cdf_cl_hi   = np.percentile(boot_cdfs_cl, 95, axis=0)
    cdf_cl_mean = np.mean(boot_cdfs_cl, axis=0)

    # KS test: iid IS bootstrap samples vs iid clone bootstrap samples
    ks_stat, ks_p = ks_2samp(S_IS_boot, S_clone)
    print(f"  KS statistic = {ks_stat:.4f}   p-value = {ks_p:.4f}", flush=True)

    return (
        s_grid,
        cdf_IS_mean, cdf_IS_lo, cdf_IS_hi,
        cdf_cl_mean, cdf_cl_lo, cdf_cl_hi,
        ks_stat, ks_p,
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    theta_IS, theta_IS_std,
    panel_A, panel_B,
    s_grid, cdf_IS, cdf_IS_lo, cdf_IS_hi,
    cdf_cl,  cdf_cl_lo,  cdf_cl_hi,
    ks_stat, ks_p,
    dt_default, outpath,
):
    fig = plt.figure(figsize=(15, 4.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    BLUE   = "#2166AC"
    RED    = "#D6604D"
    GREY   = "#888888"

    # ---- Panel A ----
    ax = axes[0]
    nc  = np.array(sorted(panel_A), dtype=float)
    mu  = np.array([panel_A[n][0] for n in nc])
    sem = np.array([panel_A[n][1] for n in nc])

    ax.axhspan(theta_IS - 2*theta_IS_std, theta_IS + 2*theta_IS_std,
               color=BLUE, alpha=0.12, zorder=0)
    ax.axhline(theta_IS, color=BLUE, lw=1.5, ls="--",
               label=f"IS: {theta_IS:.4f} ± {theta_IS_std:.4f}")
    ax.errorbar(nc, mu, yerr=2*sem, fmt="o-", color=RED,
                capsize=4, lw=1.5, ms=5, label=r"Cloning $\pm\,2\sigma$")

    # Annotate the expected bias direction
    ax.annotate("Jensen bias\n(finite $N_c$)",
                xy=(nc[0], mu[0]), xytext=(nc[0]*1.3, mu[0] + 0.015),
                fontsize=7, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel(r"Clone population $N_c$", fontsize=11)
    ax.set_ylabel(r"$\hat\theta(\zeta)$", fontsize=11)
    ax.set_title(r"(A) SCGF convergence in $N_c$", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)

    # ---- Panel B ----
    ax = axes[1]
    facs = np.array(sorted(panel_B), dtype=float)
    mu_B = np.array([panel_B[f][0] for f in facs])
    sem_B= np.array([panel_B[f][1] for f in facs])
    dtau_vals = facs * dt_default

    ax.axhspan(theta_IS - 2*theta_IS_std, theta_IS + 2*theta_IS_std,
               color=BLUE, alpha=0.12, zorder=0)
    ax.axhline(theta_IS, color=BLUE, lw=1.5, ls="--", label="IS reference")
    ax.axvline(dt_default, color=GREY, lw=1.0, ls=":", label=r"Default $\delta\tau$")
    ax.errorbar(dtau_vals, mu_B, yerr=2*sem_B, fmt="s-", color="seagreen",
                capsize=4, lw=1.5, ms=5, label=r"Cloning $\pm\,2\sigma$")

    # Mark the weight-collapse region
    ax.fill_betweenx(
        [ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else mu_B.min() - 0.05,
         mu_B.max() + 0.05],
        x1=dtau_vals[-2], x2=dtau_vals[-1],
        color="orange", alpha=0.12, label="ESS collapse region",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Resampling interval $\delta\tau$", fontsize=11)
    ax.set_ylabel(r"$\hat\theta(\zeta)$", fontsize=11)
    ax.set_title(r"(B) $\delta\tau$ independence", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    # ---- Panel C: CDF comparison ----
    ax = axes[2]

    # IS band
    ax.fill_between(s_grid, cdf_IS_lo, cdf_IS_hi,
                    color=BLUE, alpha=0.25, label=f"IS 90% CI ($M={M_IS}$)")
    ax.plot(s_grid, cdf_IS, color=BLUE, lw=2.0, label="IS (exact Q_s)")

    # Cloning band
    ax.fill_between(s_grid, cdf_cl_lo, cdf_cl_hi,
                    color=RED, alpha=0.25, label=f"Cloning 90% CI ($N_c={NC_C}$)")
    ax.plot(s_grid, cdf_cl, color=RED, lw=2.0, ls="--", label="Cloning")

    # KS annotation
    verdict = "not rejected" if ks_p > 0.05 else "REJECTED"
    ax.text(0.04, 0.96,
            f"KS = {ks_stat:.3f}\n$p = {ks_p:.3f}$ ({verdict})",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.8))

    ax.set_xlabel(r"Half-chain entropy $S_{L/2}(T)$", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    ax.set_title(r"(C) $Q_s$ distribution: IS vs cloning", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        rf"$L={L},\ \lambda={LAM},\ \zeta={ZETA},\ T={T}$"
        rf"   —   default $\delta\tau = {dt_default:.3f}$",
        fontsize=11, y=1.01,
    )
    fig.subplots_adjust(top=0.88)

    for ext in (".pdf", ".png"):
        p = outpath.with_suffix(ext)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}", flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    # Verify key imports exist
    from pps_qj.cloning import run_cloning, _batched_entanglement_entropy
    from scipy.stats import ks_2samp

    outpath = Path(__file__).parent / "validation_cloning"

    dt_default = 1.0 / max(2.0 * ALPHA * (L - 1), 1e-6)
    print(f"Parameters: L={L}, λ={LAM}, α={ALPHA}, w={W}, ζ={ZETA}, T={T}")
    print(f"Default δτ = {dt_default:.4f}  (n_steps = {int(np.ceil(T/dt_default))})")

    master = np.random.default_rng(20250428)
    rng_IS, rng_A, rng_B, rng_C_IS, rng_C_clone = master.spawn(5)

    # IS reference (also used for Panel C)
    print("\n=== IS reference ===", flush=True)
    theta_IS, theta_IS_std, n_jumps_IS, S_IS = importance_sampling_reference(
        L, ALPHA, W, ZETA, T, M=M_IS, rng=rng_IS,
    )

    panel_A = panel_A_data(rng_A, dt_default)
    panel_B = panel_B_data(rng_B, dt_default)

    # Panel C: pass the IS data already computed
    print("\n=== Panel C: distribution match ===", flush=True)
    log_w_IS = n_jumps_IS * np.log(ZETA)
    lw_max   = log_w_IS.max()
    w_IS     = np.exp(log_w_IS - lw_max); w_IS /= w_IS.sum()
    (s_grid,
     cdf_IS, cdf_IS_lo, cdf_IS_hi,
     cdf_cl, cdf_cl_lo, cdf_cl_hi,
     ks_stat, ks_p) = panel_C_data(rng_C_IS, rng_C_clone, dt_default)

    # Console summary
    print("\n=== Summary ===")
    print(f"IS reference:  theta = {theta_IS:.6f} ± {theta_IS_std:.6f}")
    print(f"\nPanel A  (N_c convergence, δτ=default):")
    print(f"  {'N_c':>5}  {'theta':>10}  {'2σ':>8}  {'ESS':>7}")
    for N_c in sorted(panel_A):
        m, s, e = panel_A[N_c]
        print(f"  {N_c:>5}  {m:10.6f}  {2*s:8.6f}  {e:7.1f}")
    print(f"\nPanel B  (δτ sensitivity, N_c={NC_B}):")
    print(f"  {'factor':>7}  {'theta':>10}  {'2σ':>8}  {'ESS':>7}")
    for f in sorted(panel_B):
        m, s, e = panel_B[f]
        print(f"  {f:>7.3f}  {m:10.6f}  {2*s:8.6f}  {e:7.1f}")
    print(f"\nPanel C  (KS test):")
    print(f"  KS statistic = {ks_stat:.4f}   p-value = {ks_p:.4f}")
    print(f"  {'PASS' if ks_p > 0.05 else 'FAIL'}: null (same distribution) "
          f"{'not rejected' if ks_p > 0.05 else 'REJECTED'} at p=0.05")

    make_figure(
        theta_IS, theta_IS_std,
        panel_A, panel_B,
        s_grid, cdf_IS, cdf_IS_lo, cdf_IS_hi,
        cdf_cl,  cdf_cl_lo,  cdf_cl_hi,
        ks_stat, ks_p,
        dt_default, outpath,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
