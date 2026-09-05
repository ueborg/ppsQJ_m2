"""Red-team synthetic injection: realistic noise (taken from the zeta=0.35
corpus at R=16) plus a LARGE, monotone, real N_c drift. Question: what does the
frozen classifier and the recommendation table say?"""
import json,os,math,numpy as np
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fake","arm","results")
os.makedirs(OUT,exist_ok=True)
for f in os.listdir(OUT): os.remove(os.path.join(OUT,f))
rng=np.random.default_rng(7)
GRID={0.10:[0.040,0.055,0.070,0.085,0.100,0.115,0.130,0.145,0.160]}
NCS=[128,256,512,1024,2048]
# realistic relative SEM at R=16, from the measured zeta=0.35 ladder
REL={128:0.050,256:0.030,512:0.024,1024:0.019,2048:0.016}
DRIFT=float(os.environ.get("DRIFT","0.10"))   # 10 % CMI change PER RUNG, real
seed=0
for z in GRID:
  for L in (32,48,64):
    for nc in NCS:
      k=NCS.index(nc)
      for lam in GRID[z]:
        # a peaked CMI curve, crossing between L=48 and L=64 near lam=0.10
        base=0.9*math.exp(-((lam-0.105)/0.075)**2)*(1.0+0.02*(L-48)/16.0)
        base*= (1.0-DRIFT)**k                 # monotone real drift, 10 % per rung
        sd=REL[nc]*base*math.sqrt(16)
        for r in range(16):
          seed+=1
          v=float(base+rng.normal(0,sd))
          json.dump(dict(L=L,T=float(L),N_c=nc,zeta=z,lam=lam,dtau_mult=6.0,
                         resample_scheme="systematic",seed=seed,status="ok",
                         cmi_weighted_mean=v,cmi_within_var=(sd*math.sqrt(16))**2*nc,
                         n_steps=10,wall_s=1.0),
                    open(os.path.join(OUT,"p%06d.json"%seed),"w"))
print("wrote",seed,"populations, drift per rung =",DRIFT)
