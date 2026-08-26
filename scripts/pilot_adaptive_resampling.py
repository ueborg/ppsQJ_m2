#!/usr/bin/env python3
"""Adaptive-resampling QJ-PPS pilot.  NO HPC submission from agents.

Compares current every-window resampling against exact-target cumulative-weight
SMC with ESS-triggered resampling.  The matched guide c=zeta is held fixed so
only the resampling schedule changes.
"""
import os, sys, json, time, argparse, traceback, subprocess
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np

ARMS={
 "baseline":dict(resampling_mode="always",ess_threshold=1.0,resampler="systematic"),
 "adaptive97":dict(resampling_mode="adaptive",ess_threshold=.97,resampler="systematic"),
 "adaptive90":dict(resampling_mode="adaptive",ess_threshold=.90,resampler="systematic"),
 "adaptive75":dict(resampling_mode="adaptive",ess_threshold=.75,resampler="systematic"),
 "residual90":dict(resampling_mode="adaptive",ess_threshold=.90,resampler="residual_stratified"),
}
DEFAULT_GRIDS={.55:[.31,.335,.36,.385,.41],.30:[.19,.215,.24,.265,.29],
               .10:[.08,.12,.16],.05:[.05,.08,.11]}

def parse_grids(s):
    if not s:return DEFAULT_GRIDS.copy()
    out={}
    for b in s.split("|"):
        z,x=b.split(":",1); out[float(z)]=[float(v) for v in x.split(",")]
    return out

def seed_of(L,lam,zeta,real):
    base=int(L*10_000_000+round(lam*1e4)*1_000+round(zeta*1_000))
    return base*101+real

def ckpt(outdir,arm,L,lam,zeta,real):
    d=os.path.join(outdir,arm,"L%d_z%.3f_lam%.4f"%(L,zeta,lam))
    return d,os.path.join(d,"real%03d.json"%real)

def git_commit():
    try:
        root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        h=subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip()
        d=subprocess.check_output(["git","-C",root,"status","--porcelain","--untracked-files=no"],text=True).strip()
        return h,bool(d)
    except Exception:return "unknown",None

def weighted_summary(per,w,key):
    v=np.asarray([x[key] for x in per],float); p=np.asarray(w,float)
    good=np.isfinite(v)&np.isfinite(p)
    if not good.any():return np.nan,np.nan
    v=v[good];p=p[good];p/=p.sum();mu=float(p@v)
    return mu,float(np.sqrt(max(float(p@((v-mu)**2)),0.0)))

def run_one(t):
    from omnibus_observables import observables
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.cloning_adaptive import run_cloning_adaptive
    arm,L,lam,zeta,real=t["arm"],t["L"],t["lam"],t["zeta"],t["real"]
    d,path=ckpt(t["outdir"],arm,L,lam,zeta,real)
    if os.path.exists(path):return "skip"
    os.makedirs(d,exist_ok=True)
    try:
        alpha,w=float(lam),float(1-lam);T=float(t["Tmult"]*L)
        dt=t["dtau_mult"]/max(2*alpha*(L-1),1e-6)
        model=build_gaussian_chain_model(L,w,alpha);sd=seed_of(L,lam,zeta,real)
        commit,dirty=git_commit();t0=time.time()
        res=run_cloning_adaptive(model,zeta,T,t["Nc"],np.random.default_rng(sd),
            delta_tau=dt,proposal_c=zeta,jump_update_method="lowrank",
            refresh_every=100,solver_method=t["solver"],**ARMS[arm])
        wall=time.time()-t0
        per=[observables(np.asarray(G,float),L) for G in res.final_covs]
        weights=np.asarray(res.final_weights,float)
        rec={"arm":arm,"L":L,"lambda":lam,"zeta":zeta,"real":real,"seed":sd,
             "git_commit":commit,"git_dirty":dirty,"T":T,"N_c":t["Nc"],
             "dtau_requested":dt,"delta_tau_eff":res.delta_tau,"n_steps":res.n_steps,
             "theta_hat":res.theta_hat,"eff_sample_size":res.eff_sample_size,
             "min_ess_frac":res.min_ess_frac_postburnin,"mean_ess_frac":res.mean_ess_frac_postburnin,
             "n_resampling_events":res.n_resampling_events,"coalescence_burden":res.coalescence_burden,
             "genealogical_ess":res.root_genealogical_ess,
             "gen_ess_frac":res.root_genealogical_ess/t["Nc"],
             "n_distinct_ancestors":res.n_distinct_root_ancestors,
             "mean_jumps_per_clone_window":res.mean_jumps_per_clone_window,
             "wall_traj_s":wall,"status":"ok"}
        for lag,g in res.lagged_gess.items():
            rec[f"lagged_gess_{lag}w"]=g;rec[f"lagged_gess_frac_{lag}w"]=g/t["Nc"]
        for key in ("CMI","B_L","S_AB","I3","MI_ends","varN","c_eff"):
            mu,s=weighted_summary(per,weights,key);rec[key+"_mean"]=mu;rec[key+"_std"]=s
        if per and "S_prof_S" in per[0]:
            P=np.asarray([x["S_prof_S"] for x in per],float);p=weights/weights.sum()
            rec["S_prof_l"]=per[0]["S_prof_l"];rec["S_prof_S_mean"]=(p@P).tolist()
        json.dump(rec,open(path,"w"));return "ok"
    except Exception as e:
        json.dump({"arm":arm,"L":L,"lambda":lam,"zeta":zeta,"real":real,
                   "status":"error","error":str(e),"traceback":traceback.format_exc()},open(path,"w"))
        return "error"

def main():
    p=argparse.ArgumentParser();p.add_argument("--outdir",required=True);p.add_argument("--grids",default="")
    p.add_argument("--Ls",default="32,64");p.add_argument("--arms",default="baseline,adaptive97,adaptive90,adaptive75")
    p.add_argument("--nreal",type=int,default=12);p.add_argument("--Nc",type=int,default=128)
    p.add_argument("--Tmult",type=float,default=1.0);p.add_argument("--dtau-mult",type=float,default=12.,dest="dtau_mult")
    p.add_argument("--solver",default="newton");p.add_argument("--shard",type=int,default=0);p.add_argument("--nshards",type=int,default=1)
    p.add_argument("--nworkers",type=int,default=1);p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    grids=parse_grids(a.grids);Ls=[int(x) for x in a.Ls.split(",")];arms=[x for x in a.arms.split(",") if x]
    bad=set(arms)-set(ARMS)
    if bad:raise ValueError("unknown arms %s"%sorted(bad))
    tasks=[]
    for arm in arms:
      for z,lams in grids.items():
       for L in Ls:
        for lam in lams:
         for r in range(a.nreal):tasks.append(dict(arm=arm,L=L,lam=lam,zeta=z,real=r,outdir=a.outdir,Nc=a.Nc,Tmult=a.Tmult,dtau_mult=a.dtau_mult,solver=a.solver))
    tasks.sort(key=lambda t:-(t["L"]**4));mine=[t for i,t in enumerate(tasks) if i%a.nshards==a.shard]
    todo=[t for t in mine if not os.path.exists(ckpt(t["outdir"],t["arm"],t["L"],t["lam"],t["zeta"],t["real"])[1])]
    print("[adaptive] shard %d/%d: %d/%d tasks, %d remaining"%(a.shard,a.nshards,len(mine),len(tasks),len(todo)),flush=True)
    if a.dry_run:return
    if a.nworkers<=1:out=[run_one(t) for t in todo]
    else:
      import multiprocessing as mp
      with mp.Pool(a.nworkers) as pool:out=pool.map(run_one,todo,chunksize=1)
    print("[adaptive] done",{s:out.count(s) for s in set(out)},flush=True)
if __name__=="__main__":main()
