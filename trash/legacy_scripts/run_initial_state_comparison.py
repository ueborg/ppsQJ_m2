"""
Compare Procedure A vs PPS-MC click-count PMFs for two initial states.
Also verify empirical vs theoretical Procedure A acceptance rate.

Model: single_projector_model, gamma=1.0, T=1.5, zeta=0.5, exact backend.
Case 1: psi = [0, 1]        (q0 = 1.0)
Case 2: psi = [sqrt(0.3), sqrt(0.7)]  (q0 = 0.7)

Theoretical acceptance rate: (1 - q0) + q0 * exp(-(1 - zeta) * gamma * T)
"""

import sys
import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")

from pps_qj import Simulator, SimulationConfig
from pps_qj.models import single_projector_model

GAMMA = 1.0
T = 1.5
ZETA = 0.5
BACKEND = "exact"
N_TRAJ_A = 150_000
N_TRAJ_PPS = 60_000
SEED = 42

sim = Simulator()

CASES = [
    {
        "label": "q0=1.0  psi=[0, 1]",
        "q0": 1.0,
        "psi": np.array([0.0, 1.0], dtype=np.complex128),
    },
    {
        "label": "q0=0.7  psi=[sqrt(0.3), sqrt(0.7)]",
        "q0": 0.7,
        "psi": np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=np.complex128),
    },
]


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
print(f"single_projector_model  gamma={GAMMA}  T={T}  zeta={ZETA}  backend={BACKEND}")
print(f"Procedure A: n_traj={N_TRAJ_A:,} | PPS-MC: n_traj={N_TRAJ_PPS:,}")
print("=" * 70)

results = {}

for case in CASES:
    label = case["label"]
    q0 = case["q0"]
    psi = case["psi"]
    theory_rate = theoretical_accept_rate(q0, ZETA, GAMMA, T)

    print(f"\n--- {label} ---")
    print(f"  Theoretical accept rate : {theory_rate:.6f}")

    model = single_projector_model(gamma=GAMMA, initial_state=psi)

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

    n_accepted = sum(1 for r in recs_a if r.accepted)
    empirical_rate = n_accepted / N_TRAJ_A
    rate_err = abs(empirical_rate - theory_rate)

    print(f"  Empirical accept rate   : {empirical_rate:.6f}  "
          f"(|error| = {rate_err:.6f},  {rate_err / theory_rate * 100:.3f}%)")

    counts_a = get_click_counts(recs_a, accepted_only=True)
    counts_pps = get_click_counts(recs_pps, accepted_only=False)

    tv, p_val = compare_pmfs(counts_a, counts_pps)
    mean_a = counts_a.mean()
    mean_pps = counts_pps.mean()

    results[label] = dict(
        q0=q0,
        theory_rate=theory_rate,
        empirical_rate=empirical_rate,
        tv=tv,
        p_val=p_val,
        mean_a=mean_a,
        mean_pps=mean_pps,
    )

    print(f"  TV distance             : {tv:.6f}")
    print(f"  Chi-sq p-value          : {p_val:.4e}")
    print(f"  Mean clicks (Proc A)    : {mean_a:.4f}")
    print(f"  Mean clicks (PPS-MC)    : {mean_pps:.4f}")

# --- Summary ---
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Case':<30} {'Theory P(acc)':>13} {'Empirical':>11} {'|err|':>8}")
print("-" * 70)
for label, r in results.items():
    print(f"{label:<30} {r['theory_rate']:>13.6f} {r['empirical_rate']:>11.6f} {abs(r['empirical_rate']-r['theory_rate']):>8.6f}")

print()
print(f"{'Case':<30} {'TV dist':>10} {'p-value':>12} {'mean(A)':>8} {'mean(PPS)':>10}")
print("-" * 70)
for label, r in results.items():
    print(f"{label:<30} {r['tv']:>10.6f} {r['p_val']:>12.4e} {r['mean_a']:>8.4f} {r['mean_pps']:>10.4f}")
print("=" * 70)
