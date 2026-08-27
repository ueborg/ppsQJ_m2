#!/usr/bin/env python3
"""EXPERIMENTAL adaptive-resampling fork. NOT production. Nothing here writes
to pps_qj/. Production cloning.py is untouched.

WHY A FORK. run_cloning hard-wires resample-every-window: per-window weights
are rebuilt each step and never accumulate, which is only consistent because a
resample fires every step. Evidence in hand says that schedule is the low-zeta
problem: per-window pre-resampling ESS has median 0.972*N_c on the very rows
whose ROOT genealogy is fully collapsed (Cut A export, 4011 rows) -- near-flat
weights, systematically resampled 40-950 times per run. Classic SMC over-
resampling. The fix is textbook (Del Moral/Doucet: trigger on ESS), but it
requires accumulated weights and WEIGHTED READOUT, which is exactly what a toy
that "keeps unweighted averages and skips resampling" gets wrong -- at low zeta
the bias is invisible because weights are near-flat, at mid zeta it is not.

MODES
  always : reproduce production exactly (gate below proves bit-identity)
  ess    : accumulate log-weights; resample only when ESS(acc) < tau*N_c
  never  : no interaction at all; N_c independent guided trajectories combined
           by self-normalised importance weights. UNBIASED for the tilted
           measure by construction (same proposal, exact compensator), so it is
           the small-size reference every other mode is scored against. Its own
           validity is its final ESS, which is reported.

LOG-Z BOOKKEEPING (telescoped): at each resample event and at final time,
log_Z += logmeanexp(log_w_acc); reset acc on resample. In always-mode this
reduces to production's per-window sum, which the gate checks numerically.

READOUT: weighted by the ACCUMULATED weights of the final population, for every
observable. In always-mode acc=0 after the last resample -> equal weights,
matching production final_covs semantics.

GENEALOGY: root ancestors (from t=0) AND recent ancestors (lookback reset at
T/2), to test the claim that root GESS exaggerates the damage.

Scored later by DEC-MASTER-METRIC-001's t_wall*var(lambda_c); pilot metrics here
are bias-vs-reference and variance-at-matched-cost, which decide whether the
metric run is worth staging.
"""
import os, sys, json, time, argparse, itertools

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pps_qj.cloning import (_spawn_rngs, _systematic_resample_idxs,
                            _batched_entanglement_entropy, CloningCollapse)
from pps_qj.gaussian_backend import (build_gaussian_chain_model,
                                     gaussian_born_rule_trajectory)
from omnibus_observables import _ent


def _cmi_vals(covs, L):
    """Per-clone CMI, identical block convention to omnibus observables()."""
    hL = L // 2
    out = np.empty(len(covs))
    for i, G in enumerate(covs):
        G = np.asarray(G, float)
        S_AB = _ent(G[:L, :L])
        S_BC = _ent(G[hL:hL + L, hL:hL + L])
        S_B = _ent(G[hL:L, hL:L])
        S_ABC = _ent(G[:3 * L // 2, :3 * L // 2])
        out[i] = S_AB + S_BC - S_B - S_ABC
    return out


def _logmeanexp(x):
    m = float(np.max(x))
    if not np.isfinite(m):
        return -np.inf
    return m + float(np.log(np.mean(np.exp(x - m))))


def run_adaptive(model, zeta, T_total, N_c, rng, delta_tau, mode="always",
                 ess_tau=0.5, entropy_stride=1, jump_update_method="lowrank",
                 refresh_every=100, solver_method="newton", eps_hazard=1e-9):
    L = model.L
    n_steps = max(1, int(np.ceil(T_total / delta_tau)))
    dte = T_total / n_steps
    covs = [model.gamma0.copy() for _ in range(N_c)]
    orbs = [model.orbitals0.copy() for _ in range(N_c)]
    _jp = model.jump_pairs
    _ja = np.array([p[0] for p in _jp], dtype=np.intp)
    _jb = np.array([p[1] for p in _jp], dtype=np.intp)
    sub_rngs = _spawn_rngs(rng, N_c)          # identical consumption order to prod

    log_w_acc = np.zeros(N_c)
    log_Z = 0.0
    anc_root = np.arange(N_c, dtype=np.intp)
    anc_recent = np.arange(N_c, dtype=np.intp)
    k_half = n_steps // 2
    n_events = 0
    S_hist, ess_hist, cmi_hist = [], [], []
    pc = zeta                                  # production proposal

    for k in range(n_steps):
        nj = np.zeros(N_c, dtype=np.int64)
        dL = np.zeros(N_c)
        for i in range(N_c):
            r = gaussian_born_rule_trajectory(
                model, T=dte, rng=sub_rngs[i],
                gamma0_override=covs[i], orbitals0_override=orbs[i],
                ja_cached=_ja, jb_cached=_jb, proposal_c=pc,
                jump_update_method=jump_update_method,
                refresh_every=refresh_every, solver_method=solver_method,
                eps_hazard=eps_hazard)
            covs[i], orbs[i] = r.final_covariance, r.final_orbitals
            nj[i], dL[i] = r.n_jumps, r.Lambda
        if zeta == 1.0:
            step_lw = np.zeros(N_c)
        else:
            step_lw = nj * np.log(zeta / pc) - (1.0 - pc) * dL
        log_w_acc = log_w_acc + step_lw

        m = float(np.max(log_w_acc))
        w = np.exp(log_w_acc - m)
        sw, sq = float(w.sum()), float(np.sum(w ** 2))
        if sw <= 0.0 or not np.isfinite(sw):
            raise CloningCollapse("weights collapsed at step %d" % k)
        ess = sw * sw / sq
        ess_hist.append(ess / N_c)

        if entropy_stride and (k % max(1, entropy_stride) == 0):
            S_vals = _batched_entanglement_entropy(covs, L // 2)
            S_hist.append(float(np.dot(w / sw, S_vals)))
            cmi_hist.append(float(np.dot(w / sw, _cmi_vals(covs, L))))

        if k == k_half:
            anc_recent = np.arange(N_c, dtype=np.intp)

        trigger = (zeta != 1.0) and (
            mode == "always" or (mode == "ess" and ess < ess_tau * N_c))
        if trigger:
            log_Z += _logmeanexp(log_w_acc)
            idxs = _systematic_resample_idxs(w, rng)
            covs = [covs[int(i)].copy() for i in idxs]
            orbs = [orbs[int(i)].copy() for i in idxs]
            anc_root = anc_root[idxs]
            anc_recent = anc_recent[idxs]
            log_w_acc = np.zeros(N_c)
            n_events += 1

    log_Z += _logmeanexp(log_w_acc)
    m = float(np.max(log_w_acc))
    w_fin = np.exp(log_w_acc - m); w_fin /= w_fin.sum()

    def gess(a):
        c = np.bincount(a, minlength=N_c).astype(float); c = c[c > 0]
        return float(c.sum() ** 2 / np.sum(c ** 2))

    nb = int(np.ceil((n_steps // 4) / max(1, entropy_stride)))
    ch = np.asarray(cmi_hist)
    n_rec = len(ch)
    cmi_t50 = float(np.mean(ch[n_rec // 2:])) if n_rec >= 4 else float("nan")
    cmi_t75 = float(np.mean(ch[(3 * n_rec) // 4:])) if n_rec >= 4 else float("nan")
    return dict(theta_hat=log_Z / T_total,
                CMI_tavg50=cmi_t50, CMI_tavg75=cmi_t75,
                S_mean=float(np.mean(S_hist[nb:])) if len(S_hist) > nb else float("nan"),
                final_covs=covs, final_weights=w_fin,
                n_resampling_events=n_events,
                ess_final_frac=float(1.0 / np.sum(w_fin ** 2) / N_c),
                min_ess_frac=float(np.min(ess_hist[n_steps // 4:])) if n_steps > 4 else float("nan"),
                gess_root=gess(anc_root), gess_recent=gess(anc_recent),
                n_anc_root=int(len(np.unique(anc_root))),
                n_anc_recent=int(len(np.unique(anc_recent))))


# ---------------------------------------------------------------- gate & study
def gate(args):
    """Bit-identity of always-mode against production run_cloning."""
    from pps_qj.cloning import run_cloning
    L, zeta, lam, Nc, T = 16, 0.55, 0.35, 24, 8.0
    dtau = 12.0 / (2.0 * lam * (L - 1))
    model = build_gaussian_chain_model(L, 1.0 - lam, lam)
    ref = run_cloning(model, zeta, T, Nc, np.random.default_rng(7), delta_tau=dtau,
                      record_entropy=True, entropy_stride=1, proposal_c=zeta,
                      jump_update_method="lowrank", refresh_every=100,
                      solver_method="newton")
    mine = run_adaptive(model, zeta, T, Nc, np.random.default_rng(7), dtau,
                        mode="always", entropy_stride=1)
    dth = abs(ref.theta_hat - mine["theta_hat"])
    dcv = max(float(np.max(np.abs(a - b)))
              for a, b in zip(ref.final_covs, mine["final_covs"]))
    print("GATE always-mode vs production:")
    print("  |dtheta| = %.3e   max|dcov| = %.3e   n_events %d vs %d"
          % (dth, dcv, ref.n_resampling_events, mine["n_resampling_events"]))
    ok = dth < 1e-10 and dcv < 1e-10 and ref.n_resampling_events == mine["n_resampling_events"]
    print("  ->", "BIT-IDENTICAL, fork is valid" if ok else "MISMATCH, fork INVALID, stop")
    return ok


def study(args):
    if not gate(args):
        sys.exit(1)
    from omnibus_observables import observables
    rng_master = np.random.default_rng(20260827)
    cells = []
    for tok in args.cells.split(";"):
        z, lam = tok.split(":")
        cells.append((float(z), float(lam)))
    modes = [("always", 0.0), ("ess", 0.9), ("ess", 0.5), ("never", 0.0)]
    out = []
    for (zeta, lam) in cells:
        L, Nc, T = args.L, args.Nc, float(args.Tmult) * float(args.L)
        dtau = 12.0 / (2.0 * lam * (L - 1))
        model = build_gaussian_chain_model(L, 1.0 - lam, lam)
        for mode, tau in modes:
            tag = mode if mode != "ess" else "ess%.1f" % tau
            for r in range(args.nreal):
                sd = 1_000_003 * r + int(1e4 * lam) + int(1e3 * zeta)
                t0 = time.time()
                try:
                    res = run_adaptive(model, zeta, T, Nc,
                                       np.random.default_rng(sd), dtau,
                                       mode=mode, ess_tau=tau,
                                       entropy_stride=4)
                except CloningCollapse as e:
                    out.append(dict(zeta=zeta, lam=lam, mode=tag, real=r,
                                    status="collapse")); continue
                wall = time.time() - t0
                per = [observables(np.asarray(G, float), L) for G in res["final_covs"]]
                wv = res["final_weights"]
                rec = dict(zeta=zeta, lam=lam, mode=tag, real=r, status="ok",
                           wall=wall, theta=res["theta_hat"],
                           n_events=res["n_resampling_events"],
                           ess_final=res["ess_final_frac"],
                           min_ess=res["min_ess_frac"],
                           gess_root=res["gess_root"],
                           gess_recent=res["gess_recent"],
                           n_anc_root=res["n_anc_root"],
                           n_anc_recent=res["n_anc_recent"])
                for k in ("CMI", "B_L", "c_eff", "S_AB"):
                    v = np.array([p[k] for p in per], float)
                    rec[k] = float(np.dot(wv, v))
                rec["CMI_tavg50"] = res["CMI_tavg50"]
                rec["CMI_tavg75"] = res["CMI_tavg75"]
                out.append(rec)
            done = [o for o in out if o["mode"] == tag and o["zeta"] == zeta
                    and o["status"] == "ok"]
            c = np.array([o["CMI"] for o in done])
            print("z=%.2f lam=%.3f %-7s  CMI %.4f+-%.4f  events %5.1f  "
                  "gess_root %5.1f  gess_recent %5.1f  essF %.2f  wall %.2fs"
                  % (zeta, lam, tag, c.mean(), c.std(ddof=1)/np.sqrt(len(c)),
                     np.mean([o["n_events"] for o in done]),
                     np.mean([o["gess_root"] for o in done]),
                     np.mean([o["gess_recent"] for o in done]),
                     np.mean([o["ess_final"] for o in done]),
                     np.mean([o["wall"] for o in done])), flush=True)
        json.dump(out, open(args.out, "w"))
    print("\nwrote", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gate", "study"], default="study")
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--Nc", type=int, default=48)
    p.add_argument("--nreal", type=int, default=20)
    p.add_argument("--cells", default="0.10:0.14;0.20:0.21;0.55:0.35")
    p.add_argument("--Tmult", type=float, default=1.0)
    p.add_argument("--out", default="/tmp/adaptive_study.json")
    a = p.parse_args()
    gate(a) if a.mode == "gate" else study(a)
