# ========================================================================
# ppsQJ_m2 variance-reduction prototype (saved 2026-06-17 from /tmp scratch)
# ------------------------------------------------------------------------
# END-TO-END SCGF control-variate test (the NEGATIVE result, kept as evidence).
# theta=(1/T)sum log E[G] with/without M. VARRED=1.03x -> CV does NOT help the SCGF.
# rho^2(G,M)=0.05 in-population (vs 0.78 fixed-start). Shows the earlier 4.2x on E[G]
# was a fixed-start artifact. See theory/VARIANCE_REDUCTION.md sec 4.
# ========================================================================

import numpy as np
from pps_qj.gaussian_backend import build_gaussian_chain_model, gaussian_born_rule_trajectory
from pps_qj.cloning import _systematic_resample_idxs
def run_scgf(model, N_c, rng, dtau, zeta, n_steps, burn_frac, beta, collect=False):
    covs=[model.gamma0.copy() for _ in range(N_c)]; orbs=[model.orbitals0.copy() for _ in range(N_c)]
    rngs=rng.spawn(N_c+1); sub_rngs=rngs[:N_c]; res_rng=rngs[N_c]
    nb=int(burn_frac*n_steps); logR=0.0; logR_cv=0.0; nrec=0; n_neg=0; Gp=[]; Mp=[]
    for step in range(n_steps):
        G=np.empty(N_c); Mv=np.empty(N_c)
        for i in range(N_c):
            r=gaussian_born_rule_trajectory(model,T=dtau,rng=sub_rngs[i],gamma0_override=covs[i],orbitals0_override=orbs[i],proposal_c=zeta)
            covs[i]=r.final_covariance; orbs[i]=r.final_orbitals
            G[i]=np.exp(-(1-zeta)*r.Lambda); Mv[i]=r.n_jumps-zeta*r.Lambda
        Rk=G.mean(); Rk_cv=(G-beta*Mv).mean()
        if step>=nb:
            logR+=np.log(Rk); nrec+=1
            if Rk_cv<=0: n_neg+=1; logR_cv+=np.log(max(Rk,1e-300))
            else: logR_cv+=np.log(Rk_cv)
            if collect: Gp.append(G.copy()); Mp.append(Mv.copy())
        idx=_systematic_resample_idxs(G, res_rng)
        covs=[covs[int(j)].copy() for j in idx]; orbs=[orbs[int(j)].copy() for j in idx]
    T=nrec*dtau; out=(logR/T, logR_cv/T, n_neg)
    if collect: out=out+((np.concatenate(Gp), np.concatenate(Mp)),)
    return out
L=48; zeta=0.3; lam=0.51*np.sqrt(zeta); alpha=lam; w=1-lam; Nc=100; NR=8
model=build_gaussian_chain_model(L=L,w=w,alpha=alpha)
T=max(40.0,min(2.0*L,128.0)); dt0=1.0/(2*alpha*(L-1)); dtau=8*dt0; n_steps=int(round(T/dtau)); burn=0.25
print(f"SCGF end-to-end CV: L={L} zeta={zeta} N_c={Nc} mult=8 n_steps={n_steps} NR={NR}",flush=True)
_,_,_,(Gp,Mp)=run_scgf(model,Nc,np.random.default_rng(0),dtau,zeta,n_steps,burn,0.0,collect=True)
beta=np.cov(Gp,Mp)[0,1]/np.var(Mp); r2=np.corrcoef(Gp,Mp)[0,1]**2
print(f"  pilot beta={beta:.4f} rho2(G,M)={r2:.3f} (one-window R_k var-red ~ {1/(1-r2):.2f}x)",flush=True)
th_p=[];th_c=[];negs=0
for r in range(NR):
    tp,tc,nn=run_scgf(model,Nc,np.random.default_rng(100+r),dtau,zeta,n_steps,burn,beta); th_p.append(tp);th_c.append(tc);negs+=nn
th_p=np.array(th_p);th_c=np.array(th_c)
print(f"  theta_plain: mean={th_p.mean():.5f} std={th_p.std(ddof=1):.5f}",flush=True)
print(f"  theta_CV   : mean={th_c.mean():.5f} std={th_c.std(ddof=1):.5f}",flush=True)
print(f"  END-TO-END VARRED(theta)={(th_p.std(ddof=1)/th_c.std(ddof=1))**2:.2f}x | bias_shift={th_c.mean()-th_p.mean():+.5f} | neg-Rk windows={negs}",flush=True)
print("DONE",flush=True)
