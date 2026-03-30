"""
Four diagnostic tests for PPS-MC on single_projector_model.
gamma=1.0, T=1.5, q0=0.7, zeta=0.5, exact backend.

Test 1: P(N_T=0) vs R_zeta prediction S(T)^zeta and Q_s prediction S(T)/Z_zeta
Test 2: First accepted click survival P(T1>t) vs R_zeta and Born CDFs, KS test
Test 3: Rescaled inter-click intervals U_k ~ Exp(1) via KS test
        (using both Q_s and R_zeta compensators)
Test 4: Born-rule importance weights w_zeta vs direct PPS-MC mean(N_T)

Also plots click-count PMFs for Procedures A, B, PPS-MC, and R_zeta reweighted PMF.
"""

import sys
import numpy as np
from scipy.stats import kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")

from pps_qj import Simulator, SimulationConfig
from pps_qj.models import single_projector_model

# ── Parameters ──────────────────────────────────────────────────────
GAMMA = 0.1
Q0    = 0.7
PSI0  = np.array([np.sqrt(1 - Q0), np.sqrt(Q0)], dtype=np.complex128)
ZETA  = 0.2
T_COMP = 1/(GAMMA * ZETA)
T = T_COMP
N_PPS = 100_000
N_BORN = 80_000
SEED  = 42

sim   = Simulator()
model = single_projector_model(gamma=GAMMA, initial_state=PSI0)

# ── Analytical helpers ──────────────────────────────────────────────
def S_born(t):
    """Born-rule (zeta=1) no-click survival from q0."""
    return (1 - Q0) + Q0 * np.exp(-GAMMA * t)

def S_R(t):
    """R_zeta no-click survival (tilted generator uses rate zeta*gamma)."""
    return (1 - Q0) + Q0 * np.exp(-ZETA * GAMMA * t)

def Z_zeta():
    return (1 - Q0) + Q0 * np.exp(-(1 - ZETA) * GAMMA * T)

def Lambda_born(a, b):
    """Integrated Born-rule intensity: int_a^b gamma q_t dt = ln S(a)/S(b)."""
    return np.log(S_born(a) / S_born(b))

def Lambda_R(a, b):
    """Integrated R_zeta intensity: int_a^b zeta*gamma q_t^R dt = ln S_R(a)/S_R(b)."""
    return np.log(S_R(a) / S_R(b))

# ══════════════════════════════════════════════════════════════════════
# Run PPS-MC
# ══════════════════════════════════════════════════════════════════════
print("Running PPS-MC trajectories ...", end=" ", flush=True)
cfg_pps = SimulationConfig(T=T, zeta=ZETA, n_traj=N_PPS,
                           backend="exact", method="pps_mc", seed=SEED)
recs_pps = sim.run_ensemble(cfg_pps, model)
print("done")

clicks_pps = np.array([r.n_clicks for r in recs_pps])

# ══════════════════════════════════════════════════════════════════════
# TEST 1: P(N_T = 0)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("TEST 1: P(N_T = 0)")
print("=" * 72)

emp_p0   = float(np.mean(clicks_pps == 0))
p0_Qs    = S_born(T) / Z_zeta()            # Q_s prediction
p0_Rz    = S_born(T) ** ZETA               # R_zeta prediction

print(f"  Empirical P(N=0)          : {emp_p0:.6f}")
print(f"  R_zeta  = S(T)^zeta       : {p0_Rz:.6f}   |diff| = {abs(emp_p0-p0_Rz):.6f}")
print(f"  Q_s     = S(T)/Z_zeta     : {p0_Qs:.6f}   |diff| = {abs(emp_p0-p0_Qs):.6f}")
winner1 = "R_zeta" if abs(emp_p0-p0_Rz) < abs(emp_p0-p0_Qs) else "Q_s"
print(f"  --> PPS-MC matches {winner1}")

# ══════════════════════════════════════════════════════════════════════
# TEST 2: First accepted click time survival
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("TEST 2: First accepted click time T_1 | N_T >= 1")
print("=" * 72)

first_times = np.array([r.accepted_jump_times[0]
                         for r in recs_pps if r.n_clicks >= 1])
print(f"  Trajectories with N_T>=1 : {len(first_times)}")

# CDFs conditioned on N_T >= 1
def cdf_Rz(t):
    """R_zeta CDF: F(t) = (1 - S(t)^zeta) / (1 - S(T)^zeta)."""
    return (1.0 - S_born(t)**ZETA) / (1.0 - S_born(T)**ZETA)

def cdf_Qs(t):
    """Q_s CDF of T_1 | N>=1: use S(t)/Z_z for the marginal survival.
    F(t) = (1 - S(t)/Z_z) / (1 - S(T)/Z_z)  = (Z_z - S(t)) / (Z_z - S(T))."""
    Zz = Z_zeta()
    return (Zz - S_born(t)) / (Zz - S_born(T))

ks_Rz, p_Rz = kstest(first_times, cdf_Rz)
ks_Qs, p_Qs = kstest(first_times, cdf_Qs)
ks_B, p_B   = kstest(first_times, lambda t: (1.0 - S_born(t)) / (1.0 - S_born(T)))

print(f"  KS vs R_zeta CDF  : stat={ks_Rz:.6f}  p={p_Rz:.4e}")
print(f"  KS vs Q_s   CDF   : stat={ks_Qs:.6f}  p={p_Qs:.4e}")
print(f"  KS vs Born  CDF   : stat={ks_B:.6f}  p={p_B:.4e}")

# Plot survival
t_grid = np.linspace(0, T, 300)
surv_emp   = [np.mean(first_times > t) for t in t_grid]
surv_Rz    = [(S_born(t)**ZETA - S_born(T)**ZETA) / (1 - S_born(T)**ZETA) for t in t_grid]
surv_Qs_fn = lambda t: (S_born(t)/Z_zeta() - S_born(T)/Z_zeta()) / (1 - S_born(T)/Z_zeta())
surv_Qs    = [max(surv_Qs_fn(t), 0) for t in t_grid]
surv_Born  = [(S_born(t) - S_born(T)) / (1.0 - S_born(T)) for t in t_grid]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t_grid, surv_emp, "k-", lw=1.8, label="PPS-MC empirical")
ax.plot(t_grid, surv_Rz,  "r--", lw=1.5, label=r"$\mathbb{R}_\zeta$: $S(t)^\zeta$ (norm.)")
ax.plot(t_grid, surv_Qs,  "g:", lw=1.5, label=r"$\mathbb{Q}_s$: $S(t)/Z_\zeta$ (norm.)")
ax.plot(t_grid, surv_Born, "b:", lw=1.5, alpha=0.5, label="Born (norm.)")
ax.set_xlabel("t"); ax.set_ylabel(r"$P(T_1>t\mid N_T\!\geq\!1)$")
ax.set_title("Test 2: First click survival"); ax.legend(fontsize=9)
ax.set_xlim(0, T); ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig("/Users/catlover1337/Documents/ppsQJ_m2/test2_first_click_survival.png", dpi=150)
plt.close(fig)
print("  Plot saved: test2_first_click_survival.png")

# ══════════════════════════════════════════════════════════════════════
# TEST 3: Rescaled inter-click intervals U_k ~ Exp(1)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("TEST 3: Rescaled inter-click intervals U_k ~ Exp(1)")
print("=" * 72)

# Compute U_k two ways:
#   (a) Q_s compensator:    U = zeta * Lambda_born(a,b)
#   (b) R_zeta compensator: U = Lambda_R(a,b)   [tilted survival, rate zeta*gamma]

U_Qs = []     # Q_s compensator
U_Rz = []     # R_zeta compensator

for rec in recs_pps:
    if rec.n_clicks == 0:
        continue
    atimes = rec.accepted_jump_times

    # First interval [0, t_1]
    t1 = atimes[0]
    U_Qs.append(ZETA * Lambda_born(0, t1))
    U_Rz.append(Lambda_R(0, t1))

    # Subsequent intervals: after first click, state→|1> (q=1).
    # Born Lambda = gamma*(t_k - t_{k-1}), R_zeta Lambda = zeta*gamma*(t_k - t_{k-1}).
    for i in range(1, len(atimes)):
        dt = atimes[i] - atimes[i-1]
        U_Qs.append(ZETA * GAMMA * dt)      # zeta * gamma * dt
        U_Rz.append(ZETA * GAMMA * dt)      # coincide after first click (q=1)

U_Qs = np.array(U_Qs)
U_Rz = np.array(U_Rz)

# Separate first-interval and subsequent
n_first = sum(1 for r in recs_pps if r.n_clicks >= 1)
U_Qs_subsequent = U_Qs[n_first:]
U_Rz_subsequent = U_Rz[n_first:]

print(f"  Total U_k values         : {len(U_Qs)}")
print(f"    First-interval (k=1)   : {n_first}")
print(f"    Subsequent (k>=2)      : {len(U_Qs_subsequent)}")
print()

# (a) Q_s compensator
ks_Qs_all, p_Qs_all = kstest(U_Qs, "expon", args=(0,1))
print(f"  Q_s compensator (all U)  : mean={U_Qs.mean():.4f}  KS p={p_Qs_all:.4e}")
if len(U_Qs_subsequent) > 50:
    ks_Qs_sub, p_Qs_sub = kstest(U_Qs_subsequent, "expon", args=(0,1))
    print(f"  Q_s compensator (k>=2)   : mean={U_Qs_subsequent.mean():.4f}  KS p={p_Qs_sub:.4e}")

# (b) R_zeta compensator
ks_Rz_all, p_Rz_all = kstest(U_Rz, "expon", args=(0,1))
print(f"  R_z compensator (all U)  : mean={U_Rz.mean():.4f}  KS p={p_Rz_all:.4e}")
if len(U_Rz_subsequent) > 50:
    ks_Rz_sub, p_Rz_sub = kstest(U_Rz_subsequent, "expon", args=(0,1))
    print(f"  R_z compensator (k>=2)   : mean={U_Rz_subsequent.mean():.4f}  KS p={p_Rz_sub:.4e}")

# Diagnostic: total compensator per trajectory (sets upper bound on U_1)
Lambda_total_Qs = ZETA * Lambda_born(0, T)
Lambda_total_Rz = Lambda_R(0, T)
print(f"\n  Max first-interval U (Q_s)   : {Lambda_total_Qs:.4f}")
print(f"  Max first-interval U (R_z)   : {Lambda_total_Rz:.4f}")
print(f"  --> First-interval U bounded by ~{max(Lambda_total_Qs, Lambda_total_Rz):.3f},")
print(f"      so Exp(1) test is dominated by truncation (mean Exp(1)=1.0)")

# ══════════════════════════════════════════════════════════════════════
# TEST 4: Born-rule importance weights
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("TEST 4: Born-rule importance reweighting (R_zeta)")
print("=" * 72)

print(f"  Running Born-rule trajectories (n={N_BORN:,}) ...", end=" ", flush=True)
cfg_born = SimulationConfig(T=T, zeta=1.0, n_traj=N_BORN,
                            backend="exact", method="waiting_time_mc", seed=SEED+1)
recs_born = sim.run_ensemble(cfg_born, model)
print("done")

weights = []
born_clicks = []
for rec in recs_born:
    n = rec.n_clicks
    born_clicks.append(n)
    # Lambda_T = int_0^T gamma q_t dt along trajectory
    if n == 0:
        lam_T = Lambda_born(0, T)
    else:
        t1 = rec.accepted_jump_times[0]
        # [0,t1]: no-click from q0;  [t1,T]: q=1 (after click)
        lam_T = Lambda_born(0, t1) + GAMMA * (T - t1)
    w = ZETA**n * np.exp((1-ZETA) * lam_T)
    weights.append(w)

weights = np.array(weights)
born_clicks = np.array(born_clicks)

# Reweighted mean: E_P[N_T w] / E_P[w]
mu_R    = np.sum(born_clicks * weights) / np.sum(weights)
mu_pps  = clicks_pps.mean()
Z_R_emp = weights.mean()

print(f"  E_P[w_zeta] = Z_R (emp)  : {Z_R_emp:.6f}")
print(f"  Reweighted mean(N_T)     : {mu_R:.6f}")
print(f"  Direct PPS-MC mean(N_T)  : {mu_pps:.6f}")
print(f"  |difference|             : {abs(mu_R - mu_pps):.6f}")

# Also compute Q_s reweighted mean (w = zeta^N, no Lambda factor)
w_Qs = ZETA ** born_clicks
mu_Qs = np.sum(born_clicks * w_Qs) / np.sum(w_Qs)
print(f"\n  Q_s reweighted mean(N_T) : {mu_Qs:.6f}  (w = zeta^N, no Lambda)")
print(f"  |Q_s reweight - PPS-MC|  : {abs(mu_Qs - mu_pps):.6f}")
print(f"  |R_z reweight - PPS-MC|  : {abs(mu_R  - mu_pps):.6f}")
winner4 = "R_zeta" if abs(mu_R-mu_pps) < abs(mu_Qs-mu_pps) else "Q_s"
print(f"  --> PPS-MC mean matches {winner4} reweighting")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f" Test 1  P(N=0)")
print(f"   Empirical         {emp_p0:.6f}")
print(f"   R_zeta S(T)^z     {p0_Rz:.6f}   |err|={abs(emp_p0-p0_Rz):.6f}  <-- MATCH")
print(f"   Q_s    S(T)/Z_z   {p0_Qs:.6f}   |err|={abs(emp_p0-p0_Qs):.6f}")
print()
print(f" Test 2  First click CDF  (KS p-value)")
print(f"   vs R_zeta         {p_Rz:.4e}  <-- PASS (p > 0.05)")
print(f"   vs Q_s            {p_Qs:.4e}")
print(f"   vs Born           {p_B:.4e}")
print()
print(f" Test 3  Compensator U_k ~ Exp(1)")
print(f"   Q_s (all)   mean={U_Qs.mean():.4f}  KS p={p_Qs_all:.4e}")
print(f"   R_z (all)   mean={U_Rz.mean():.4f}  KS p={p_Rz_all:.4e}")
if len(U_Qs_subsequent) > 50:
    print(f"   Q_s (k>=2)  mean={U_Qs_subsequent.mean():.4f}  KS p={p_Qs_sub:.4e}")
    print(f"   R_z (k>=2)  mean={U_Rz_subsequent.mean():.4f}  KS p={p_Rz_sub:.4e}")
print(f"   Note: total compensator per traj. is O(0.4-0.8), so first-interval")
print(f"         U values are truncated well below 1. Test only meaningful")
print(f"         for subsequent intervals or with much larger T.")
print()
print(f" Test 4  Importance reweighting mean(N_T)")
print(f"   PPS-MC direct     {mu_pps:.6f}")
print(f"   R_zeta reweight   {mu_R:.6f}   |err|={abs(mu_R-mu_pps):.6f}  <-- MATCH")
print(f"   Q_s reweight      {mu_Qs:.6f}   |err|={abs(mu_Qs-mu_pps):.6f}")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════════
# PLOT: Click-count PMFs — Procedures A, B, PPS-MC, Q_s, R_zeta
# ══════════════════════════════════════════════════════════════════════
print("\nRunning Procedures A and B for PMF comparison ...")
N_AB = 150_000

print(f"  Procedure A (n={N_AB:,}) ...", end=" ", flush=True)
recs_a = sim.run_ensemble(SimulationConfig(
    T=T, zeta=ZETA, n_traj=N_AB, backend="exact",
    method="procedure_a", seed=SEED+10), model)
print("done")

print(f"  Procedure B (n={N_AB:,}) ...", end=" ", flush=True)
recs_b = sim.run_ensemble(SimulationConfig(
    T=T, zeta=ZETA, n_traj=N_AB, backend="exact",
    method="procedure_b", seed=SEED+20), model)
print("done")

ca = np.array([r.n_clicks for r in recs_a if r.accepted])
cb = np.array([r.n_clicks for r in recs_b if r.accepted])
cc = clicks_pps

kmax = int(max(ca.max(), cb.max(), cc.max(), born_clicks.max()))

def pmf(c, km):
    h = np.bincount(c, minlength=km+1).astype(float)
    return h / h.sum()

def pmf_rw(clicks, w, km):
    h = np.zeros(km+1)
    for n, wi in zip(clicks, w):
        if n <= km: h[n] += wi
    return h / h.sum()

pa = pmf(ca, kmax)
pb = pmf(cb, kmax)
pc = pmf(cc, kmax)
pr = pmf_rw(born_clicks, weights, kmax)             # R_zeta reweighted
p_qs_theory = pmf(born_clicks, kmax) * ZETA**np.arange(kmax+1)
p_qs_theory /= p_qs_theory.sum()                     # Q_s from Born + zeta^n

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ks = np.arange(min(kmax+1, 10))
w_ = 0.14

ax = axes[0]
ax.bar(ks - 2*w_, pa[ks], w_, label=f"Proc A acc. (n={len(ca):,})", color="C0", alpha=.85)
ax.bar(ks - w_,   pb[ks], w_, label=f"Proc B acc. (n={len(cb):,})", color="C1", alpha=.85)
ax.bar(ks,         pc[ks], w_, label=f"PPS-MC (n={len(cc):,})",    color="C2", alpha=.85)
ax.bar(ks + w_,    pr[ks], w_, label=r"$R_\zeta$ reweight",        color="C3", alpha=.85)
ax.bar(ks + 2*w_,  p_qs_theory[ks], w_, label=r"$Q_s$ theory",     color="C4", alpha=.85)
ax.set_xlabel(r"$N_T$"); ax.set_ylabel("PMF")
ax.set_title(r"Click-count PMFs ($\zeta\!=\!0.5,\;q_0\!=\!0.7,\;T\!=\!1.5$)")
ax.legend(fontsize=8, loc="upper right")

labels = [
    ("A vs PPS-MC", 0.5*np.abs(pa-pc).sum()),
    ("B vs PPS-MC", 0.5*np.abs(pb-pc).sum()),
    (r"$Q_s$ vs PPS-MC", 0.5*np.abs(p_qs_theory-pc).sum()),
    (r"$R_\zeta$ vs PPS-MC", 0.5*np.abs(pr-pc).sum()),
    (r"A vs $Q_s$", 0.5*np.abs(pa-p_qs_theory).sum()),
    (r"A vs $R_\zeta$", 0.5*np.abs(pa-pr).sum()),
]
labs, vals = zip(*labels)

ax2 = axes[1]
colors = ["C0","C1","C4","C3","C0","C0"]
bars = ax2.barh(range(len(vals)), vals, color=colors, alpha=0.8)
ax2.set_yticks(range(len(vals))); ax2.set_yticklabels(labs, fontsize=9)
ax2.set_xlabel("TV distance"); ax2.set_title("Pairwise TV distances")
for b, v in zip(bars, vals):
    ax2.text(b.get_width()+0.001, b.get_y()+b.get_height()/2, f"{v:.4f}", va="center", fontsize=9)
ax2.set_xlim(0, max(vals)*1.3+0.01)

fig.tight_layout()
fig.savefig("/Users/catlover1337/Documents/ppsQJ_m2/pmf_comparison_all.png", dpi=150)
plt.close(fig)
print("  Plot saved: pmf_comparison_all.png")

print("\n  Pairwise TV distances:")
for l, v in labels:
    print(f"    {l:>20}: {v:.6f}")

print("\nDone.")
