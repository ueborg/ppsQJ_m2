"""
Compare click-count PMFs between Procedure A and PPS-MC for three zeta values.
Reports: TV distance, chi-squared p-value, mean clicks.
"""

import numpy as np
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon

import sys
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")

from pps_qj import Simulator, SimulationConfig
from pps_qj.models import single_projector_model

# ---------- Setup ----------
GAMMA = 1.0
PSI0 = np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=np.complex128)
T = 4
BACKEND = "exact"
N_TRAJ_A = 150_000
N_TRAJ_PPS = 60_000
ZETAS = [0.5, 0.2, 0.8]

sim = Simulator()
model = single_projector_model(gamma=GAMMA, initial_state=PSI0)


def get_click_counts(records, accepted_only=False):
    if accepted_only:
        records = [r for r in records if r.accepted]
    return np.array([r.n_clicks for r in records])


def compare_pmfs(counts_a, counts_pps):
    """Compute TV distance and chi-squared p-value between two click-count samples."""
    max_k = max(counts_a.max(), counts_pps.max())
    bins = np.arange(-0.5, max_k + 1.5)

    hist_a, _ = np.histogram(counts_a, bins=bins)
    hist_b, _ = np.histogram(counts_pps, bins=bins)

    # Normalize to PMFs
    pmf_a = hist_a / hist_a.sum()
    pmf_b = hist_b / hist_b.sum()

    # TV distance = 0.5 * L1
    tv = 0.5 * np.sum(np.abs(pmf_a - pmf_b))

    # Chi-squared test on the contingency table (raw counts, trim zero-sum columns)
    contingency = np.vstack([hist_a, hist_b])
    col_mask = contingency.sum(axis=0) > 0
    contingency = contingency[:, col_mask]
    _, p_val, _, _ = chi2_contingency(contingency)

    return tv, p_val, pmf_a, pmf_b


print("=" * 65)
print(f"Model: single_projector_model, gamma={GAMMA}, T={T}, backend={BACKEND}")
print(f"Initial state: [sqrt(0.3), sqrt(0.7)]")
print(f"Procedure A: n_traj={N_TRAJ_A:,}")
print(f"PPS-MC:      n_traj={N_TRAJ_PPS:,}")
print("=" * 65)

results = {}

for zeta in ZETAS:
    print(f"\n--- zeta = {zeta} ---")

    cfg_a = SimulationConfig(
        T=T, zeta=zeta, n_traj=N_TRAJ_A, backend=BACKEND, method="procedure_a", seed=42
    )
    cfg_pps = SimulationConfig(
        T=T, zeta=zeta, n_traj=N_TRAJ_PPS, backend=BACKEND, method="pps_mc", seed=42
    )

    print(f"  Running Procedure A  (n={N_TRAJ_A:,}) ...", end=" ", flush=True)
    recs_a = sim.run_ensemble(cfg_a, model)
    print("done")

    print(f"  Running PPS-MC       (n={N_TRAJ_PPS:,}) ...", end=" ", flush=True)
    recs_pps = sim.run_ensemble(cfg_pps, model)
    print("done")

    # Procedure A: condition on accepted trajectories (post-selected ensemble)
    counts_a = get_click_counts(recs_a, accepted_only=True)
    counts_pps = get_click_counts(recs_pps, accepted_only=False)

    n_accepted_a = len(counts_a)
    accept_rate_a = n_accepted_a / N_TRAJ_A
    print(f"  Proc A accepted      : {n_accepted_a:,} / {N_TRAJ_A:,} ({accept_rate_a:.3%})")

    tv, p_val, pmf_a, pmf_pps = compare_pmfs(counts_a, counts_pps)
    mean_a = counts_a.mean()
    mean_pps = counts_pps.mean()

    results[zeta] = dict(tv=tv, p_val=p_val, mean_a=mean_a, mean_pps=mean_pps)

    print(f"  TV distance          : {tv:.6f}")
    print(f"  Chi-sq p-value       : {p_val:.4e}")
    print(f"  Mean clicks (Proc A) : {mean_a:.4f}")
    print(f"  Mean clicks (PPS-MC) : {mean_pps:.4f}")

# ---------- Summary table ----------
print("\n")
print("=" * 65)
print(f"{'zeta':>6} | {'TV dist':>10} | {'p-value':>12} | {'mean(A)':>9} | {'mean(PPS)':>9}")
print("-" * 65)
for zeta in ZETAS:
    r = results[zeta]
    print(
        f"{zeta:>6.1f} | {r['tv']:>10.6f} | {r['p_val']:>12.4e} | {r['mean_a']:>9.4f} | {r['mean_pps']:>9.4f}"
    )
print("=" * 65)
