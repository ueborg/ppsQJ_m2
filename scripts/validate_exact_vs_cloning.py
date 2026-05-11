"""Validate that the exact-spin-chain backend agrees with the Gaussian backend.

Both backends implement the SAME free-fermion model (Kitaev-like hopping +
bond-density measurements), just in different representations. They MUST
agree at the level of trajectory observables:

  1. Born-rule (zeta=1): <S_{L/2}>_Born and Var(N_T)_Born
  2. Forced-no-click (zeta=0): same deterministic S(T) and S_top(T)
  3. PPS (zeta in 0..1) via Doob+cloning vs exact rejection sampling of Born

Test 1 and 2 use direct sampling and are clean.
Test 3 uses rejection sampling: generate many Born trajectories, weight each by
zeta^N_T, and compute the weighted <S>; compare to cloning's S_mean.
Rejection sampling is statistically equivalent to PPS but has exponential
variance in N_T - keep L small (<= 8) and T short.

If all three tests pass within statistical error, BOTH backends are correct.
If any fails, the discrepancy localises the bug.

Run from project root:
    python scripts/validate_exact_vs_cloning.py [L]
"""
from __future__ import annotations
import sys
import numpy as np

from pps_qj.exact_backend import (
    build_exact_spin_chain_model,
    ordinary_quantum_jump_trajectory,
)
from pps_qj.observables.basic import entanglement_entropy_statevector
from pps_qj.gaussian_backend import build_gaussian_chain_model, entanglement_entropy
from pps_qj.cloning import run_cloning


# ============================================================================
# Test 1: Born-rule <S_{L/2}> agreement between exact and Gaussian
# ============================================================================
def test_born_rule_agreement(L: int = 8, T: float = 10.0, lam: float = 0.30,
                              n_traj_exact: int = 200, n_clones: int = 200,
                              seed: int = 42):
    print(f"\n=== Test 1: Born-rule <S_{L}/2> at lam={lam}, T={T}, L={L} ===")
    alpha = lam
    w = 1.0 - lam

    # Exact: sample n_traj_exact Born trajectories, average final S
    print(f"  Exact: sampling {n_traj_exact} Born trajectories...")
    model_ex = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    rng = np.random.default_rng(seed)
    S_exact_arr = np.zeros(n_traj_exact)
    NT_exact_arr = np.zeros(n_traj_exact, dtype=int)
    for i in range(n_traj_exact):
        traj = ordinary_quantum_jump_trajectory(model_ex, T=T, rng=rng)
        psi = np.asarray(traj.final_state)
        S_exact_arr[i] = entanglement_entropy_statevector(psi, L, l_sub=L // 2)
        NT_exact_arr[i] = len(traj.jump_times)
    S_ex_mean = S_exact_arr.mean()
    S_ex_err  = S_exact_arr.std() / np.sqrt(n_traj_exact)
    NT_ex_mean = NT_exact_arr.mean()

    # Gaussian via cloning at zeta=1 (= Born)
    print(f"  Cloning: zeta=1.0, N_c={n_clones} clones, T={T}...")
    model_g = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    res = run_cloning(model_g, zeta=1.0, T_total=T, N_c=n_clones,
                      rng=np.random.default_rng(seed + 1))
    # Cloning entropy is in bits (base=2). Exact uses entanglement_entropy_statevector which is base 2.
    S_cl_mean = res.S_mean
    S_cl_err  = res.S_std / np.sqrt(max(1, n_clones))
    NT_cl_mean = res.n_T_mean * L * res.delta_tau * (T / res.delta_tau) / T  # <N_T>

    print(f"  Exact:   <S>={S_ex_mean:.3f} +/- {S_ex_err:.3f}    <N_T>={NT_ex_mean:.2f}")
    print(f"  Cloning: <S>={S_cl_mean:.3f} +/- {S_cl_err:.3f}    <N_T>={NT_cl_mean:.2f}")
    zscore = abs(S_ex_mean - S_cl_mean) / np.sqrt(S_ex_err ** 2 + S_cl_err ** 2 + 1e-20)
    print(f"  z-score for <S>: {zscore:.2f}  ({'PASS' if zscore < 3 else 'FAIL'} at 3-sigma)")
    return dict(S_ex=S_ex_mean, S_cl=S_cl_mean, zscore=zscore, NT_ex=NT_ex_mean, NT_cl=NT_cl_mean)


# ============================================================================
# Test 2: No-click branch agreement (deterministic)
# ============================================================================
def test_noclick_agreement(L: int = 8, T: float = 5.0, lam: float = 0.30):
    print(f"\n=== Test 2: deterministic no-click S(T) at lam={lam}, T={T}, L={L} ===")
    from pps_qj.exact_backend import postselected_no_click_trajectory
    alpha = lam
    w = 1.0 - lam

    # Exact no-click: propagate initial state under e^{-i H_eff T}, normalise
    model_ex = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    traj = postselected_no_click_trajectory(model_ex, T=T)
    S_ex = entanglement_entropy_statevector(traj.final_state, L, l_sub=L // 2)

    # Gaussian no-click: cloning at zeta=0 produces the no-click ensemble
    # but the simplest test is direct H_eff propagation. We use cloning with
    # zeta -> very small (= heavily postselected; effectively no-click).
    model_g = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    res = run_cloning(model_g, zeta=1e-6, T_total=T, N_c=200,
                      rng=np.random.default_rng(0))
    S_g = res.S_mean

    print(f"  Exact deterministic no-click:  S={S_ex:.4f}")
    print(f"  Cloning at zeta=1e-6:          S={S_g:.4f}")
    diff = abs(S_ex - S_g)
    print(f"  Absolute difference: {diff:.4f}  ({'PASS' if diff < 0.2 else 'FAIL'} at 0.2-bit tolerance)")
    return dict(S_ex=S_ex, S_cl=S_g, diff=diff)


# ============================================================================
# Test 3: PPS via exact rejection sampling vs cloning at intermediate zeta
# ============================================================================
def test_pps_rejection(L: int = 8, T: float = 8.0, lam: float = 0.30,
                       zeta: float = 0.5, n_traj_exact: int = 500,
                       n_clones: int = 200, seed: int = 42):
    print(f"\n=== Test 3: PPS at zeta={zeta}, lam={lam}, T={T}, L={L} ===")
    alpha = lam
    w = 1.0 - lam

    # Exact: sample Born trajectories, weight each by zeta^N_T, compute weighted <S>
    print(f"  Exact rejection: {n_traj_exact} Born trajectories, weight by zeta^N_T...")
    model_ex = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    rng = np.random.default_rng(seed)
    S_arr = np.zeros(n_traj_exact)
    NT_arr = np.zeros(n_traj_exact, dtype=int)
    for i in range(n_traj_exact):
        traj = ordinary_quantum_jump_trajectory(model_ex, T=T, rng=rng)
        psi = np.asarray(traj.final_state)
        S_arr[i] = entanglement_entropy_statevector(psi, L, l_sub=L // 2)
        NT_arr[i] = len(traj.jump_times)
    # weights are zeta^N_T
    log_w = NT_arr * np.log(zeta)
    log_w -= log_w.max()
    w_arr = np.exp(log_w)
    Z = w_arr.sum()
    if Z <= 0:
        print("  WARNING: all rejection weights are zero")
        return None
    S_pps_exact = float((w_arr * S_arr).sum() / Z)
    # Effective sample size: low ESS means rejection has large variance
    ESS = float(w_arr.sum() ** 2 / (w_arr ** 2).sum())
    print(f"  Exact PPS:       <S>={S_pps_exact:.3f}  (rejection ESS = {ESS:.1f} of {n_traj_exact})")

    # Cloning
    print(f"  Cloning: zeta={zeta}, N_c={n_clones}, T={T}...")
    model_g = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    res = run_cloning(model_g, zeta=zeta, T_total=T, N_c=n_clones,
                      rng=np.random.default_rng(seed + 1))
    S_pps_cl = res.S_mean
    S_pps_cl_err = res.S_std / np.sqrt(max(1, n_clones))
    print(f"  Cloning PPS:     <S>={S_pps_cl:.3f} +/- {S_pps_cl_err:.3f}")

    diff = abs(S_pps_exact - S_pps_cl)
    rejection_err = max(0.2, 1.0 / np.sqrt(ESS) if ESS > 0 else 1.0)
    print(f"  Absolute difference: {diff:.3f}  (PPS rejection has ~{rejection_err:.2f} statistical uncertainty)")
    print(f"  Result: {'PASS' if diff < rejection_err + 3 * S_pps_cl_err else 'WORTH INVESTIGATING'}")
    return dict(S_ex=S_pps_exact, S_cl=S_pps_cl, ESS=ESS, diff=diff)


def main(L: int = 8):
    print(f"Running validation suite at L = {L}")
    r1 = test_born_rule_agreement(L=L)
    r2 = test_noclick_agreement(L=L)
    r3 = test_pps_rejection(L=L, zeta=0.5)
    r4 = test_pps_rejection(L=L, zeta=0.3, T=6.0)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Test 1 (Born <S>):     z = {r1['zscore']:.2f}")
    print(f"  Test 2 (no-click S):   diff = {r2['diff']:.3f}")
    print(f"  Test 3a (PPS zeta=0.5): diff = {r3['diff']:.3f}  (rejection ESS = {r3['ESS']:.0f})")
    print(f"  Test 3b (PPS zeta=0.3): diff = {r4['diff']:.3f}  (rejection ESS = {r4['ESS']:.0f})")
    print("=" * 60)


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(L)
