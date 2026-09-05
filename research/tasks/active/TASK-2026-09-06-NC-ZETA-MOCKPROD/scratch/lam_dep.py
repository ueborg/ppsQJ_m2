import json, glob
from collections import defaultdict
import statistics as st

files = glob.glob("research/tasks/**/results/*.json", recursive=True)
rows=[]
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict): continue
    keys=["wall_s","n_steps","N_c","L","lam","dtau_mult","zeta"]
    if all(k in d for k in ["wall_s","n_steps","N_c","L","lam","dtau_mult"]):
        try:
            Nc=float(d["N_c"]); L=float(d["L"]); n_steps=float(d["n_steps"]); wall=float(d["wall_s"])
            dtau=float(d["dtau_mult"]); lam=float(d["lam"]); zeta=float(d.get("zeta", -1))
        except: continue
        if dtau!=6.0: continue
        if n_steps<=0 or Nc<=0: continue
        rate=wall/(Nc*n_steps)*1000.0
        rows.append((L,Nc,lam,zeta,rate,f))

# fixed zeta=0.35, fixed L and Nc bucket, look at rate vs lam
by = defaultdict(list)
for L,Nc,lam,zeta,rate,f in rows:
    if abs(zeta-0.35)<1e-6:
        by[(L,Nc)].append((lam,rate))

for k in sorted(by.keys()):
    v = by[k]
    lam_set = sorted(set(round(l,4) for l,_ in v))
    print(k, "lam values:", lam_set)
    # median rate per lam
    d2=defaultdict(list)
    for l,r in v: d2[round(l,4)].append(r)
    for l in sorted(d2):
        print("   lam=",l,"n=",len(d2[l]),"median_rate=",round(st.median(d2[l]),4))
