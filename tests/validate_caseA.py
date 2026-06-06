from __future__ import annotations

"""Standalone validation for the Case A QJ backend (CASE_A spec Gates 1-3 + the
reduction and self-duality checks). Run from anywhere:

    python tests/validate_caseA.py

Prints PASS/FAIL per gate and a final summary. No pytest needed (kept standalone
so it can be driven and read over Desktop Commander).
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pps_qj.gaussian_backend import (
    effective_generator,
    entanglement_entropy,
    build_gaussian_chain_model,
    gaussian_born_rule_trajectory,
)
from pps_qj.gaussian_backend_caseA import (
    effective_generator_caseA,
    build_caseA_model,
    gaussian_born_rule_trajectory_caseA,
)


def _z(m1, e1, m2, e2):
    return (m1 - m2) / np.sqrt(e1 * e1 + e2 * e2 + 1e-300)


def _gauss_caseA_batch(model, T, rng, n):
    """n independent zeta=1 Case A trajectories -> (S_half[], n_jumps[], site_frac[])."""
    S = np.empty(n)
    nj = np.empty(n)
    sf = np.full(n, np.nan)
    ell = model.L // 2
    for k in range(n):
        r = gaussian_born_rule_trajectory_caseA(model, T, rng)
        S[k] = entanglement_entropy(r.final_covariance, ell)
        nj[k] = r.n_jumps
        if r.n_jumps > 0:
            ns = sum(1 for ch in r.jump_channels if ch < model.n_site)
            sf[k] = ns / r.n_jumps
    return S, nj, sf


def gate1():
    print("=" * 72)
    print("GATE 1  generator correctness")
    print("=" * 72)
    L, alpha = 8, 0.7
    A0 = effective_generator_caseA(L, 0.0, alpha)
    B0 = effective_generator(L, 0.0, alpha)
    d = float(np.max(np.abs(A0 - B0)))
    ok_a = d < 1e-12
    print(f"  (a) caseA(gamma=0) == Case B(w=0):   max|diff| = {d:.2e}   {'PASS' if ok_a else 'FAIL'}")

    M = effective_generator_caseA(L, 0.6, 0.0).copy()
    for j in range(L):
        M[2 * j:2 * j + 2, 2 * j:2 * j + 2] = 0.0
    off = float(np.max(np.abs(M)))
    ok_b = off < 1e-12
    print(f"  (b) caseA(alpha=0) block-diagonal:    max off-block = {off:.2e}   {'PASS' if ok_b else 'FAIL'}")

    Af = effective_generator_caseA(L, 0.5, 0.5)
    asym = float(np.max(np.abs(Af + Af.T)))
    ok_c = asym < 1e-12
    print(f"  (c) generator antisymmetric (h=-h^T): max|h+h^T| = {asym:.2e}   {'PASS' if ok_c else 'FAIL'}")
    return ok_a and ok_b and ok_c


def gate2(seed=0):
    print("=" * 72)
    print("GATE 2  single/multi-trajectory sanity at lambda_A = 0.5")
    print("=" * 72)
    L, T, n = 16, 8.0, 300
    model = build_caseA_model(L, 0.5, 0.5)
    rng = np.random.default_rng(seed)
    site = bond = 0
    max_unphys = 0.0
    S = []
    for _ in range(n):
        r = gaussian_born_rule_trajectory_caseA(model, T, rng)
        for ch in r.jump_channels:
            if ch < model.n_site:
                site += 1
            else:
                bond += 1
        ev = np.linalg.eigvalsh(1j * r.final_covariance)
        max_unphys = max(max_unphys, float(np.max(np.abs(ev)) - 1.0))
        S.append(entanglement_entropy(r.final_covariance, L // 2))
    frac = site / max(site + bond, 1)
    print(f"  clicks: site={site} bond={bond}   site_frac={frac:.3f}  (roughly balanced ~0.5 expected)")
    print(f"  physicality: max(|spec(iC)| - 1) = {max_unphys:.2e}  (expect <~1e-9)")
    print(f"  mean S(L/2) = {np.mean(S):.4f} +/- {np.std(S) / np.sqrt(n):.4f}")
    ok_frac = 0.40 <= frac <= 0.60
    ok_phys = max_unphys < 1e-7
    print(f"  -> site_frac in [0.40,0.60]: {'PASS' if ok_frac else 'FAIL'};  physical: {'PASS' if ok_phys else 'FAIL'}")
    return ok_frac and ok_phys


def _rates_from_lambda(lam):
    """alpha+gamma=1, lambda_A = alpha/(alpha+gamma) -> alpha=lam, gamma=1-lam."""
    return (1.0 - lam), lam   # gamma_rate, alpha_rate


def gate3_exact(seed=1):
    """THE decisive correctness test: Gaussian vs exact Fock space, L=6.

    Compares mean half-chain entanglement entropy. A wrong site-channel
    probability convention (using <1-n_j> where <n_j> is required, or vice
    versa) reweights the trajectory ensemble and shows up here as a large
    z-score on <S(L/2)>. The bond channel is already validated for Case B, so
    a failure points squarely at the site channel.
    """
    from pps_qj.exact_backend_caseA import (
        build_exact_caseA_model,
        caseA_qj_trajectory_exact,
    )
    print("=" * 72)
    print("GATE 3  Gaussian vs exact Fock space  (L=6)")
    print("=" * 72)
    L, T, n = 6, 4.0, 500
    ell = L // 2
    all_ok = True
    for lam in (0.3, 0.5, 0.7):
        gamma_rate, alpha_rate = _rates_from_lambda(lam)
        gmodel = build_caseA_model(L, gamma_rate, alpha_rate)
        emodel = build_exact_caseA_model(L, gamma_rate, alpha_rate)
        rg = np.random.default_rng(seed)
        re = np.random.default_rng(seed + 777)
        Sg = np.empty(n); sfg = np.full(n, np.nan)
        Se = np.empty(n); sfe = np.full(n, np.nan)
        for k in range(n):
            r = gaussian_born_rule_trajectory_caseA(gmodel, T, rg)
            Sg[k] = entanglement_entropy(r.final_covariance, ell)
            if r.n_jumps:
                sfg[k] = sum(1 for c in r.jump_channels if c < gmodel.n_site) / r.n_jumps
            d = caseA_qj_trajectory_exact(emodel, T, re)
            Se[k] = d["S_half"]
            if d["n_jumps"]:
                sfe[k] = d["n_site"] / d["n_jumps"]
        mg, eg = float(np.mean(Sg)), float(np.std(Sg) / np.sqrt(n))
        me, ee = float(np.mean(Se)), float(np.std(Se) / np.sqrt(n))
        z = _z(mg, eg, me, ee)
        sfg_m = float(np.nanmean(sfg)); sfe_m = float(np.nanmean(sfe))
        ok = abs(z) < 4.0
        all_ok = all_ok and ok
        print(f"  lambda_A={lam:.2f}:  S_gauss={mg:.4f}+/-{eg:.4f}  "
              f"S_exact={me:.4f}+/-{ee:.4f}  z={z:+.2f}  {'PASS' if ok else 'FAIL'}")
        print(f"             site_frac gauss={sfg_m:.3f}  exact={sfe_m:.3f}  "
              f"(should agree if convention correct)")
    print(f"  -> Gaussian/exact agreement: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def gate_reduction(seed=2):
    """Case A with gamma_rate=0 must reduce to Case B with w=0 (bond-only,
    H=0). The generators are bit-identical (Gate 1a); here we check the full
    trajectory ensemble agrees distributionally in <S(L/2)> and mean clicks.
    Trajectory-by-trajectory equality is NOT expected (the RNG consumes a
    different number of categories in channel selection), only distributional.
    """
    print("=" * 72)
    print("GATE reduction  caseA(gamma=0) == Case B(w=0)")
    print("=" * 72)
    L, T, n, alpha = 12, 8.0, 400, 1.0
    ell = L // 2
    amodel = build_caseA_model(L, 0.0, alpha)
    bmodel = build_gaussian_chain_model(L, 0.0, alpha)
    ra = np.random.default_rng(seed)
    rb = np.random.default_rng(seed + 99)
    Sa = np.empty(n); na = np.empty(n)
    Sb = np.empty(n); nb = np.empty(n)
    for k in range(n):
        r = gaussian_born_rule_trajectory_caseA(amodel, T, ra)
        Sa[k] = entanglement_entropy(r.final_covariance, ell); na[k] = r.n_jumps
        r2 = gaussian_born_rule_trajectory(bmodel, T, rb)
        Sb[k] = entanglement_entropy(r2.final_covariance, ell); nb[k] = r2.n_jumps
    zS = _z(np.mean(Sa), np.std(Sa) / np.sqrt(n), np.mean(Sb), np.std(Sb) / np.sqrt(n))
    zN = _z(np.mean(na), np.std(na) / np.sqrt(n), np.mean(nb), np.std(nb) / np.sqrt(n))
    okS = abs(zS) < 4.0
    okN = abs(zN) < 4.0
    print(f"  <S> caseA={np.mean(Sa):.4f}  caseB={np.mean(Sb):.4f}  z={zS:+.2f}  {'PASS' if okS else 'FAIL'}")
    print(f"  <clicks> caseA={np.mean(na):.2f}  caseB={np.mean(nb):.2f}  z={zN:+.2f}  {'PASS' if okN else 'FAIL'}")
    return okS and okN


def gate_selfduality(seed=3):
    """INFORMATIONAL ONLY -- deliberately not pass/fail.

    The KMR c<->d duality maps lambda_A -> 1 - lambda_A and pins the critical
    *location* at lambda_A=1/2. It does NOT guarantee the half-chain
    entanglement *distribution* is symmetric under lambda_A <-> 1-lambda_A:
    the duality is a half-translation/relabeling of the Majorana chain, so it
    maps a real-space cut to a shifted cut. <S(L/2)> at 0.3 and 0.7 are in
    fact strongly asymmetric (the Neel state is a site-density eigenstate, so
    site measurement disentangles toward a product state while bond
    measurement does not). We print the gap only as a sanity signal; the
    asymmetry is real and confirmed by the exact backend in Gate 3, and it
    does NOT bear on lambda_c since S is not a duality-invariant observable.
    """
    print("=" * 72)
    print("GATE self-duality  <S(L/2)> at lambda_A vs 1-lambda_A  (informational)")
    print("=" * 72)
    L, T, n = 16, 10.0, 300
    ell = L // 2
    for lam in (0.3, 0.4):
        g1, a1 = _rates_from_lambda(lam)
        g2, a2 = _rates_from_lambda(1.0 - lam)
        m1 = build_caseA_model(L, g1, a1)
        m2 = build_caseA_model(L, g2, a2)
        r1 = np.random.default_rng(seed); r2 = np.random.default_rng(seed + 5)
        S1 = np.array([entanglement_entropy(
            gaussian_born_rule_trajectory_caseA(m1, T, r1).final_covariance, ell)
            for _ in range(n)])
        S2 = np.array([entanglement_entropy(
            gaussian_born_rule_trajectory_caseA(m2, T, r2).final_covariance, ell)
            for _ in range(n)])
        z = _z(np.mean(S1), np.std(S1) / np.sqrt(n), np.mean(S2), np.std(S2) / np.sqrt(n))
        print(f"  lam={lam:.2f} <S>={np.mean(S1):.4f}   1-lam={1-lam:.2f} <S>={np.mean(S2):.4f}"
              f"   z={z:+.2f}  (asymmetric expected; S(L/2) not duality-invariant)")
    return True  # never gates the suite


def main():
    t0 = time.time()
    results = {}
    results["gate1_generator"] = gate1()
    results["gate2_sanity"] = gate2()
    results["gate3_exact"] = gate3_exact()
    results["gate_reduction"] = gate_reduction()
    gate_selfduality()  # informational, not scored
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {name:24s} {'PASS' if ok else 'FAIL'}")
    hard_pass = all(results.values())
    print(f"  {'-' * 40}")
    print(f"  OVERALL (hard gates): {'PASS' if hard_pass else 'FAIL'}   "
          f"[{time.time() - t0:.1f}s]")
    return 0 if hard_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
