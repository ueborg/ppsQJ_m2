# ========================================================================
# ppsQJ_m2 variance-reduction prototype (saved 2026-06-17 from /tmp scratch)
# ------------------------------------------------------------------------
# Coupled neighbouring lambda-points (common random numbers): delta-scan + L-scan.
# VALIDATED WIN ~2x on entropy <S> and B_L DIFFERENCES at delta<=0.04 (BREAKS at 0.06).
# Prototype for the entanglement-FSS pipeline. See theory/VARIANCE_REDUCTION.md sec 3.
# Production needs: split coupling + maximally coupled resampling + paired-covariance GLS.
# ========================================================================

import numpy as np
from pps_qj.gaussian_backend import build_gaussian_chain_model, gaussian_born_rule_trajectory
from pps_qj.cloning import _systematic_resample_idxs, _batched_entanglement_entropy
from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
def run_sched(model,N_c,rng,dtau,zeta,meas0,n_steps):
    L=model.L;ell=L//2
    covs=[model.gamma0.copy() for _ in range(N_c)];orbs=[model.orbitals0.copy() for _ in range(N_c)]
    rngs=rng.spawn(N_c+1);sub_rngs=rngs[:N_c];res_rng=rngs[N_c];S=[]
    for step in range(n_steps):
        dL=np.zeros(N_c);nj=np.zeros(N_c,dtype=np.int64)
        for i in range(N_c):
            r=gaussian_born_rule_trajectory(model,T=dtau,rng=sub_rngs[i],gamma0_override=covs[i],orbitals0_override=orbs[i],proposal_c=zeta)
            covs[i]=r.final_covariance;orbs[i]=r.final_orbitals;nj[i]=r.n_jumps;dL[i]=r.Lambda
        logw=-(1.0-zeta)*dL;logw-=logw.max();wv=np.exp(logw);wsum=wv.sum()
        if step>=meas0:
            Sv=_batched_entanglement_entropy(covs,ell);S.append(float(np.dot(wv/wsum,Sv)))
        idx=_systematic_resample_idxs(wv,res_rng);covs=[covs[int(j)].copy() for j in idx];orbs=[orbs[int(j)].copy() for j in idx]
    return float(np.mean(S)), float(np.nanmean(_batched_compute_B_L([cc.copy() for cc in covs],L)["B_L"]))
def test(L,zeta,delta,NR=10,Nc=80):
    lamc=0.51*np.sqrt(zeta)
    mp=build_gaussian_chain_model(L=L,w=1-(lamc+delta),alpha=lamc+delta)
    mm=build_gaussian_chain_model(L=L,w=1-(lamc-delta),alpha=lamc-delta)
    alpha=lamc;T=max(30.0,min(2.0*L,128.0));dt0=1.0/(2*alpha*(L-1));dtau=8*dt0
    n_steps=int(round(T/dtau));meas0=int(0.45*n_steps)
    dS_i=[];dB_i=[];dS_c=[];dB_c=[]
    for r in range(NR):
        Sp,Bp=run_sched(mp,Nc,np.random.default_rng(3000+2*r),dtau,zeta,meas0,n_steps);Sm,Bm=run_sched(mm,Nc,np.random.default_rng(3000+2*r+1),dtau,zeta,meas0,n_steps);dS_i.append(Sp-Sm);dB_i.append(Bp-Bm)
    for r in range(NR):
        Sp,Bp=run_sched(mp,Nc,np.random.default_rng(5000+r),dtau,zeta,meas0,n_steps);Sm,Bm=run_sched(mm,Nc,np.random.default_rng(5000+r),dtau,zeta,meas0,n_steps);dS_c.append(Sp-Sm);dB_c.append(Bp-Bm)
    dS_i=np.array(dS_i);dB_i=np.array(dB_i);dS_c=np.array(dS_c);dB_c=np.array(dB_c)
    def vr(i,c):return (i.std(ddof=1)/c.std(ddof=1))**2
    print(f"L={L} zeta={zeta} delta={delta}: dS_VARRED={vr(dS_i,dS_c):.2f}x  dB_L_VARRED={vr(dB_i,dB_c):.2f}x  slopeS_std(ind={ (dS_i/(2*delta)).std(ddof=1):.3f},cou={ (dS_c/(2*delta)).std(ddof=1):.3f})",flush=True)
print("#9 coupling refinement: delta-scan (L=32) + L=64 check",flush=True)
for d in (0.02,0.04,0.06): test(32,0.3,d)
test(64,0.3,0.04,NR=8)
print("DONE",flush=True)
