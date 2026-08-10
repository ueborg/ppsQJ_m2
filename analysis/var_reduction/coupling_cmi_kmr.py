# ========================================================================
# ppsQJ_m2 variance-reduction prototype (saved 2026-06-17 from /tmp scratch)
# ------------------------------------------------------------------------
# Coupling on the CLEAN observables: time-averaged <CMI> and KMR product <CMI><S>.
# VALIDATED: <CMI>-diff 1.76x, KMR-diff 1.98x at L=32 zeta=0.3 delta=0.04.
# Confirms the coupling works on the observables that should drive FSS (not the
# noisy trajectory product <CMI*S>). See theory/VARIANCE_REDUCTION.md sec 3.
# ========================================================================

import numpy as np
from pps_qj.gaussian_backend import build_gaussian_chain_model, gaussian_born_rule_trajectory
from pps_qj.cloning import _systematic_resample_idxs, _batched_entanglement_entropy
from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
def run_cmi(model,N_c,rng,dtau,zeta,meas0,n_steps,cmi_stride=3):
    L=model.L;ell=L//2
    covs=[model.gamma0.copy() for _ in range(N_c)];orbs=[model.orbitals0.copy() for _ in range(N_c)]
    rngs=rng.spawn(N_c+1);sub_rngs=rngs[:N_c];res_rng=rngs[N_c];Svals=[];Cvals=[]
    for step in range(n_steps):
        dL=np.zeros(N_c);nj=np.zeros(N_c,dtype=np.int64)
        for i in range(N_c):
            r=gaussian_born_rule_trajectory(model,T=dtau,rng=sub_rngs[i],gamma0_override=covs[i],orbitals0_override=orbs[i],proposal_c=zeta)
            covs[i]=r.final_covariance;orbs[i]=r.final_orbitals;nj[i]=r.n_jumps;dL[i]=r.Lambda
        logw=-(1.0-zeta)*dL;logw-=logw.max();wv=np.exp(logw);wn=wv/wv.sum()
        if step>=meas0 and (step-meas0)%cmi_stride==0:
            Sv=_batched_entanglement_entropy(covs,ell);Svals.append(float(np.dot(wn,Sv)))
            cmi=np.array(_batched_compute_B_L([cc for cc in covs],L)["CMI"],float);cmi=np.nan_to_num(cmi,nan=np.nanmean(cmi));Cvals.append(float(np.dot(wn,cmi)))
        idx=_systematic_resample_idxs(wv,res_rng);covs=[covs[int(j)].copy() for j in idx];orbs=[orbs[int(j)].copy() for j in idx]
    return float(np.mean(Cvals)), float(np.mean(Svals))
def test(L,zeta,delta,NR=8,Nc=80):
    lamc=0.51*np.sqrt(zeta)
    mp=build_gaussian_chain_model(L=L,w=1-(lamc+delta),alpha=lamc+delta);mm=build_gaussian_chain_model(L=L,w=1-(lamc-delta),alpha=lamc-delta)
    alpha=lamc;T=max(30.0,min(2.0*L,128.0));dt0=1.0/(2*alpha*(L-1));dtau=8*dt0;n_steps=int(round(T/dtau));meas0=int(0.45*n_steps)
    dC_i=[];dK_i=[];dC_c=[];dK_c=[]
    for r in range(NR):
        Cp,Sp=run_cmi(mp,Nc,np.random.default_rng(3000+2*r),dtau,zeta,meas0,n_steps);Cm,Sm=run_cmi(mm,Nc,np.random.default_rng(3000+2*r+1),dtau,zeta,meas0,n_steps);dC_i.append(Cp-Cm);dK_i.append(Cp*Sp-Cm*Sm)
    for r in range(NR):
        Cp,Sp=run_cmi(mp,Nc,np.random.default_rng(5000+r),dtau,zeta,meas0,n_steps);Cm,Sm=run_cmi(mm,Nc,np.random.default_rng(5000+r),dtau,zeta,meas0,n_steps);dC_c.append(Cp-Cm);dK_c.append(Cp*Sp-Cm*Sm)
    dC_i=np.array(dC_i);dK_i=np.array(dK_i);dC_c=np.array(dC_c);dK_c=np.array(dK_c)
    def vr(i,c):return (i.std(ddof=1)/c.std(ddof=1))**2
    print(f"L={L} zeta={zeta} delta={delta} NR={NR}: <CMI>-diff VARRED={vr(dC_i,dC_c):.2f}x  KMR <CMI><S>-diff VARRED={vr(dK_i,dK_c):.2f}x",flush=True)
print("#9 coupling on clean observables <CMI> and KMR product:",flush=True)
test(32,0.3,0.04)
print("DONE",flush=True)
