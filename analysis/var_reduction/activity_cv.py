# ========================================================================
# ppsQJ_m2 variance-reduction prototype (saved 2026-06-17 from /tmp scratch)
# ------------------------------------------------------------------------
# END-TO-END tilted-activity martingale control variate M=n-zeta*Delta_Lambda.
# VALIDATED WIN ~3.26x on <n>_Q (full run, NR=8). rho^2(Y_n,M)=0.96 in-population.
# This is the real end-to-end number (the fixed-pool ~400x was an artifact).
# Activity pipeline only. See theory/VARIANCE_REDUCTION.md sec 4.
# ========================================================================

import numpy as np
from pps_qj.gaussian_backend import build_gaussian_chain_model, gaussian_born_rule_trajectory
from pps_qj.cloning import _systematic_resample_idxs
def run_act(model,N_c,rng,dtau,zeta,n_steps,burn,beta,gamma,collect=False):
    covs=[model.gamma0.copy() for _ in range(N_c)];orbs=[model.orbitals0.copy() for _ in range(N_c)]
    rngs=rng.spawn(N_c+1);sub_rngs=rngs[:N_c];res_rng=rngs[N_c];nb=int(burn*n_steps)
    ap=[];ac=[];Gp=[];Mp=[];Np=[]
    for step in range(n_steps):
        G=np.empty(N_c);M=np.empty(N_c);NN=np.empty(N_c)
        for i in range(N_c):
            r=gaussian_born_rule_trajectory(model,T=dtau,rng=sub_rngs[i],gamma0_override=covs[i],orbitals0_override=orbs[i],proposal_c=zeta)
            covs[i]=r.final_covariance;orbs[i]=r.final_orbitals
            G[i]=np.exp(-(1-zeta)*r.Lambda);M[i]=r.n_jumps-zeta*r.Lambda;NN[i]=r.n_jumps
        if step>=nb:
            ap.append((G*NN).sum()/G.sum()); ac.append((G*NN-beta*M).sum()/(G-gamma*M).sum())
            if collect:Gp.append(G.copy());Mp.append(M.copy());Np.append(NN.copy())
        idx=_systematic_resample_idxs(G,res_rng);covs=[covs[int(j)].copy() for j in idx];orbs=[orbs[int(j)].copy() for j in idx]
    out=(float(np.mean(ap)),float(np.mean(ac)))
    if collect:out=out+((np.concatenate(Gp),np.concatenate(Mp),np.concatenate(Np)),)
    return out
L=48;zeta=0.3;lam=0.51*np.sqrt(zeta);alpha=lam;w=1-lam;Nc=100;NR=8
model=build_gaussian_chain_model(L=L,w=w,alpha=alpha)
T=max(40.0,min(2.0*L,128.0));dt0=1.0/(2*alpha*(L-1));dtau=8*dt0;n_steps=int(round(T/dtau));burn=0.25
print(f"END-TO-END tilted activity <n>_Q CV: L={L} zeta={zeta} N_c={Nc} mult=8 NR={NR}",flush=True)
_,_,(Gp,Mp,Np)=run_act(model,Nc,np.random.default_rng(0),dtau,zeta,n_steps,burn,0.0,0.0,collect=True)
beta=np.cov(Gp*Np,Mp)[0,1]/np.var(Mp);gamma=np.cov(Gp,Mp)[0,1]/np.var(Mp)
mu=(Gp*Np).sum()/Gp.sum();Yn=Gp*(Np-mu)
print(f"  pilot per-window-population: rho2(Y_n=G(n-mu), M)={np.corrcoef(Yn,Mp)[0,1]**2:.3f}  (compare hmcv fixed-pool 0.997)",flush=True)
ap=[];ac=[]
for r in range(NR):
    a,c=run_act(model,Nc,np.random.default_rng(100+r),dtau,zeta,n_steps,burn,beta,gamma);ap.append(a);ac.append(c)
ap=np.array(ap);ac=np.array(ac)
print(f"  <n>_Q plain: mean={ap.mean():.4f} std={ap.std(ddof=1):.5f}",flush=True)
print(f"  <n>_Q CV   : mean={ac.mean():.4f} std={ac.std(ddof=1):.5f}",flush=True)
print(f"  END-TO-END VARRED(<n>_Q)={(ap.std(ddof=1)/ac.std(ddof=1))**2:.2f}x | bias_shift={ac.mean()-ap.mean():+.5f}",flush=True)
print("DONE",flush=True)
