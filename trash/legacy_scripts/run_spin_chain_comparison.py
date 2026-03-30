"""
Compare click-count PMFs (Procedure A vs PPS-MC) for spin_chain_model.
L=2 and L=3, w=0.5, gamma=1.0, T=2.0, zeta=0.5, exact backend.
"""

import sys
import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")

from pps_qj import Simulator, SimulationConfig
from pps_qj.models import spin_chain_model

W = 0.5
GAMMA = 1.0
T = 2.0
ZETA = 0.5
BACKEND = "exact"
N_TRAJ_A = 200_000
N_TRAJ_PPS = 80_000
SEED = 42

sim = Simulator()


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


print("=" * 68)
print(f"spin_chain_model  w={W}  gamma={GAMMA}  T={T}  zeta={ZETA}  backend={BACKEND}")
print(f"Procedure A: n_traj={N_TRAJ_A:,} (accepted only) | PPS-MC: n_traj={N_TRAJ_PPS:,}")
print("=" * 68)

results = {}

for L in [2, 3]:
    print(f"\n--- L = {L} ---")
    model = spin_chain_model(L=L, w=W, gamma=GAMMA)

    cfg_a = SimulationConfig(
        T=T, zeta=ZETA, n_traj=N_TRAJ_A, backend=BACKEND, method="procedure_a", seed=SEED
    )
    cfg_pps = SimulationConfig(
        T=T, zeta=ZETA, n_traj=N_TRAJ_PPS, backend=BACKEND, method="pps_mc", seed=SEED
    )

    print(f"  Running Procedure A  (n={N_TRAJ_A:,}) ...", end=" ", flush=True)
    recs_a = sim.run_ensemble(cfg_a, model)
    print("done")

    print(f"  Running PPS-MC       (n={N_TRAJ_PPS:,}) ...", end=" ", flush=True)
    recs_pps = sim.run_ensemble(cfg_pps, model)
    print("done")

    counts_a = get_click_counts(recs_a, accepted_only=True)
    counts_pps = get_click_counts(recs_pps, accepted_only=False)

    n_accepted = len(counts_a)
    accept_rate = n_accepted / N_TRAJ_A
    print(f"  Proc A accepted      : {n_accepted:,} / {N_TRAJ_A:,} ({accept_rate:.3%})")

    tv, p_val = compare_pmfs(counts_a, counts_pps)
    mean_a = counts_a.mean()
    mean_pps = counts_pps.mean()

    results[L] = dict(tv=tv, p_val=p_val, mean_a=mean_a, mean_pps=mean_pps)

    print(f"  TV distance          : {tv:.6f}")
    print(f"  Chi-sq p-value       : {p_val:.4e}")
    print(f"  Mean clicks (Proc A) : {mean_a:.4f}")
    print(f"  Mean clicks (PPS-MC) : {mean_pps:.4f}")

print("\n")
print("=" * 68)
print(f"{'L':>3} | {'TV dist':>10} | {'p-value':>12} | {'mean(A)':>9} | {'mean(PPS)':>9}")
print("-" * 68)
for L in [2, 3]:
    r = results[L]
    print(
        f"{L:>3} | {r['tv']:>10.6f} | {r['p_val']:>12.4e} |"
        f" {r['mean_a']:>9.4f} | {r['mean_pps']:>9.4f}"
    )
print("=" * 68)
