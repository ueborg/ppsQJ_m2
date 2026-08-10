"""Pilot for memo 1: Guided Correlated Suffix MCMC (GCS-MCMC).

Target       pi_zeta[omega] propto P_Born[omega] zeta^{N[omega]}
Proposal     regrow the suffix under the guided c=zeta process
Acceptance   A = min(1, exp[-(1-zeta)(Lambda'_suffix - Lambda_suffix)])

Derivation check (this is the memo's algebra, verified independently):
  P_Born/Q_c = c^{-N} exp[-(1-c)Lambda]  =>  pi_zeta/Q_c = (zeta/c)^N e^{-(1-zeta)Lambda}
  at c = zeta the click factor cancels and the MH ratio is the Lambda term only.
  The backend's `Lambda` field is the integrated PHYSICAL hazard (verified:
  lambda_acc += -log(U_eff), U_eff = U^{1/c}, proposal survival = exp(-c Lambda)).

MODE 'accept'  fixed-suffix probe chains -> acceptance(f), sd(Delta Lambda)(f)
MODE 'mix'     full chain, random checkpoint -> tau_int of S, CMI, N_T
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from traj_common import build, KW  # noqa
from pps_qj.gaussian_backend import (gaussian_born_rule_trajectory,
                                     entanglement_entropy,
                                     topological_entanglement_entropy)


class PathState:
    """Trajectory stored as checkpoint states on a uniform time grid."""

    def __init__(self, model, ja, jb, T, n_cp, rng, zeta):
        self.model, self.ja, self.jb = model, ja, jb
        self.zeta, self.T, self.n_cp = zeta, T, n_cp
        self.dt = T / n_cp
        self.cov = [np.asarray(model.gamma0, float).copy()]
        self.orb = [np.asarray(model.orbitals0, complex).copy()]
        self.lam = []
        self.njp = []
        self._grow(0, rng)

    def _grow(self, s, rng):
        """Regenerate segments s..n_cp-1 from checkpoint s. Returns new lists."""
        cov, orb = self.cov[s].copy(), self.orb[s].copy()
        covs, orbs, lams, njs = [], [], [], []
        for _ in range(s, self.n_cp):
            res = gaussian_born_rule_trajectory(
                self.model, self.dt, rng, gamma0_override=cov,
                orbitals0_override=orb, ja_cached=self.ja, jb_cached=self.jb,
                proposal_c=self.zeta, **KW)
            cov, orb = res.final_covariance, res.final_orbitals
            covs.append(cov); orbs.append(orb)
            lams.append(res.Lambda); njs.append(res.n_jumps)
        if s == 0 and not self.lam:      # initial build
            self.cov += covs; self.orb += orbs
            self.lam = lams; self.njp = njs
            return None
        return covs, orbs, lams, njs

    def propose(self, s, rng):
        return self._grow(s, rng)

    def accept(self, s, prop):
        covs, orbs, lams, njs = prop
        self.cov[s + 1:] = covs
        self.orb[s + 1:] = orbs
        self.lam[s:] = lams
        self.njp[s:] = njs

    def suffix_lambda(self, s):
        return float(np.sum(self.lam[s:]))

    def obs(self):
        cov = self.cov[-1]
        L = cov.shape[0] // 2
        return (entanglement_entropy(cov, L // 2),
                topological_entanglement_entropy(cov),
                float(np.sum(self.njp)))


def tau_int(x, maxlag=None):
    x = np.asarray(x, float)
    n = len(x)
    if maxlag is None:
        maxlag = min(n // 4, 200)
    x = x - x.mean()
    v = (x * x).mean()
    if v <= 0:
        return np.nan
    s = 0.5
    for k in range(1, maxlag - 1, 2):
        p = ((x[:n-k] * x[k:]).mean() + (x[:n-k-1] * x[k+1:]).mean()) / v
        if p <= 0:
            break
        s += p
    return float(s)


def run_accept_scan(L, lam, zeta, T, n_cp, n_moves, seed):
    model, ja, jb = build(L, lam)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    st = PathState(model, ja, jb, T, n_cp, rng, zeta)
    build_s = time.time() - t0
    rows = []
    for s in range(n_cp):
        f = (n_cp - s) / n_cp
        acc = 0
        dls = []
        tA = time.time()
        for _ in range(n_moves):
            old = st.suffix_lambda(s)
            prop = st.propose(s, rng)
            new = float(np.sum(prop[2]))
            dls.append(new - old)
            logR = -(1.0 - zeta) * (new - old)
            if np.log(rng.uniform()) < logR:
                st.accept(s, prop); acc += 1
        rows.append(dict(s=s, f=f, acc=acc / n_moves,
                         sd_dLambda=float(np.std(dls, ddof=1)),
                         sd_logR=float((1 - zeta) * np.std(dls, ddof=1)),
                         wall_per_move=(time.time() - tA) / n_moves))
        print(f"  f={f:.3f}  acc={acc/n_moves:.3f}  "
              f"sd(logR)={(1-zeta)*np.std(dls,ddof=1):.2f}  "
              f"{(time.time()-tA)/n_moves:.2f}s/move", flush=True)
    return rows, build_s


def run_mix(L, lam, zeta, T, n_cp, n_moves, f_fixed, seed):
    """Chain with suffix fraction f_fixed, measuring observable autocorrelation."""
    model, ja, jb = build(L, lam)
    rng = np.random.default_rng(seed)
    st = PathState(model, ja, jb, T, n_cp, rng, zeta)
    s = int(round(n_cp * (1.0 - f_fixed)))
    S, C, N, A = [], [], [], []
    t0 = time.time()
    for _ in range(n_moves):
        old = st.suffix_lambda(s)
        prop = st.propose(s, rng)
        new = float(np.sum(prop[2]))
        a = float(np.log(rng.uniform()) < -(1 - zeta) * (new - old))
        if a:
            st.accept(s, prop)
        o = st.obs()
        S.append(o[0]); C.append(o[1]); N.append(o[2]); A.append(a)
    return dict(f=f_fixed, acc=float(np.mean(A)),
                tau_S=tau_int(S), tau_CMI=tau_int(C), tau_N=tau_int(N),
                mean_S=float(np.mean(S)), sd_S=float(np.std(S, ddof=1)),
                mean_CMI=float(np.mean(C)),
                wall_per_move=(time.time() - t0) / n_moves,
                S=S, CMI=C)


def main():
    L = int(os.environ.get("L", 64))
    zeta = float(os.environ.get("ZETA", 0.9))
    lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
    T = float(os.environ.get("T", L))
    n_cp = int(os.environ.get("NCP", 16))
    n_moves = int(os.environ.get("NMOVES", 30))
    mode = os.environ.get("MODE", "accept")
    tag = f"L{L}_z{zeta}_{mode}"
    if mode == "accept":
        rows, bs = run_accept_scan(L, lam, zeta, T, n_cp, n_moves, 77000)
        out = dict(L=L, lam=lam, zeta=zeta, T=T, n_cp=n_cp, n_moves=n_moves,
                   build_wall_s=bs, rows=rows)
    else:
        f = float(os.environ.get("FFIX", 0.25))
        out = run_mix(L, lam, zeta, T, n_cp, n_moves, f, 78000)
        out.update(L=L, lam=lam, zeta=zeta, T=T, n_cp=n_cp)
        tag += f"_f{f}"
    with open(f"/tmp/mcmc_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("S", "CMI")}, indent=1))


if __name__ == "__main__":
    main()
