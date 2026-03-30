"""
Prop 43 / prop:mcwf_generates_qs validation:
For q0=1 (psi=[0,1]), Procedure A accepted and PPS-MC both generate Q_s,
so their click-count PMFs must agree: TV < 0.01, chi-sq p > 0.5.

Model: single_projector_model, gamma=1.0, T=2.0, exact backend.
N_A = 200_000 (accepted only), N_C = 80_000.
Zeta: 0.5, 0.2, 0.8.
"""

import sys
import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")

from pps_qj import Simulator, SimulationConfig
from pps_qj.models import single_projector_model

GAMMA = 1.0
T = 2.0
PSI = np.array([0.0, 1.0], dtype=np.complex128)   # q0 = 1
Q0 = 1.0
BACKEND = "exact"
N_TRAJ_A = 200_000
N_TRAJ_PPS = 80_000
ZETAS = [0.5, 0.2, 0.8]
SEED = 42

sim = Simulator()
model = single_projector_model(gamma=GAMMA, initial_state=PSI)


def theoretical_accept_rate(q0, zeta, gamma, T):
    return (1.0 - q0) + q0 * np.exp(-(1.0 - zeta) * gamma * T)


def get_click_counts(records, accepted_only=False):
    if accepted_only:
        records = [r for r in records if r.accepted]
    return np.array([r.n_clicks for r in records])


def compare_pmfs(counts_a, counts_pps):
    max_k = max(counts_a.max(), counts_pps.max())
    bins = np.arange(-0.5, max_k + 1.5)
    hist_a, _ = np.histogram(counts_a, bins=bins)
    hist_b, _ = np.histogram(counts_pps, bins=bins)
    pmf_a = hist_a / hist_a.sum()
    pmf_b = hist_b / hist_b.sum()
    tv = 0.5 * np.sum(np.abs(pmf_a - pmf_b))
    contingency = np.vstack([hist_a, hist_b])
    col_mask = contingency.sum(axis=0) > 0
    contingency = contingency[:, col_mask]
    _, p_val, _, _ = chi2_contingency(contingency)
    return tv, p_val


print("=" * 70)
print("Prop 43 (prop:mcwf_generates_qs) validation")
print(f"single_projector_model  gamma={GAMMA}  T={T}  q0={Q0}  backend={BACKEND}")
print(f"psi = [0, 1]")
print(f"N_A = {N_TRAJ_A:,} (Proc A, accepted only)  |  N_C = {N_TRAJ_PPS:,} (PPS-MC)")
print(f"Expected: TV < 0.01 and p-value > 0.5 for all zeta")
print("=" * 70)

results = {}

for zeta in ZETAS:
    theory_rate = theoretical_accept_rate(Q0, zeta, GAMMA, T)
    print(f"\n--- zeta = {zeta} ---")
    print(f"  Theory accept rate   : {theory_rate:.6f}")

    cfg_a = SimulationConfig(
        T=T, zeta=zeta, n_traj=N_TRAJ_A, backend=BACKEND, method="procedure_a", seed=SEED
    )
    cfg_pps = SimulationConfig(
        T=T, zeta=zeta, n_traj=N_TRAJ_PPS, backend=BACKEND, method="pps_mc", seed=SEED
    )

    print(f"  Running Procedure A  (n={N_TRAJ_A:,}) ...", end=" ", flush=True)
    recs_a = sim.run_ensemble(cfg_a, model)
    print("done")

    print(f"  Running PPS-MC       (n={N_TRAJ_PPS:,}) ...", end=" ", flush=True)
    recs_pps = sim.run_ensemble(cfg_pps, model)
    print("done")

    n_accepted = sum(1 for r in recs_a if r.accepted)
    empirical_rate = n_accepted / N_TRAJ_A
    print(f"  Empirical accept rate: {empirical_rate:.6f}  "
          f"(|err| = {abs(empirical_rate - theory_rate):.6f})")

    counts_a = get_click_counts(recs_a, accepted_only=True)
    counts_pps = get_click_counts(recs_pps, accepted_only=False)

    tv, p_val = compare_pmfs(counts_a, counts_pps)
    mean_a = counts_a.mean()
    mean_pps = counts_pps.mean()

    passed = (tv < 0.01) and (p_val > 0.5)
    results[zeta] = dict(
        theory_rate=theory_rate,
        empirical_rate=empirical_rate,
        tv=tv,
        p_val=p_val,
        mean_a=mean_a,
        mean_pps=mean_pps,
        passed=passed,
    )

    print(f"  TV distance          : {tv:.6f}   {'PASS' if tv < 0.01 else 'FAIL'} (< 0.01?)")
    print(f"  Chi-sq p-value       : {p_val:.4e}   {'PASS' if p_val > 0.5 else 'FAIL'} (> 0.5?)")
    print(f"  Mean clicks (Proc A) : {mean_a:.4f}")
    print(f"  Mean clicks (PPS-MC) : {mean_pps:.4f}")
    print(f"  Overall              : {'PASS' if passed else 'FAIL'}")

# Summary
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
all_pass = all(r["passed"] for r in results.values())
print(f"{'zeta':>6} | {'TV dist':>10} | {'p-value':>12} | {'mean(A)':>8} | {'mean(C)':>8} | {'verdict':>7}")
print("-" * 70)
for zeta, r in results.items():
    print(
        f"{zeta:>6.1f} | {r['tv']:>10.6f} | {r['p_val']:>12.4e} | "
        f"{r['mean_a']:>8.4f} | {r['mean_pps']:>8.4f} | "
        f"{'  PASS' if r['passed'] else '  FAIL':>7}"
    )
print("-" * 70)
print(f"All zeta: {'ALL PASS' if all_pass else 'SOME FAIL'}")
print("=" * 70)
