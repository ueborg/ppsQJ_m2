import glob,json,collections,math,numpy as np
ROOT="/Users/catlover1337/Documents/ppsQJ_m2"
cells=collections.defaultdict(list)
for p in glob.glob(ROOT+"/research/tasks/**/results/*.json",recursive=True):
    try: d=json.load(open(p))
    except Exception: continue
    if not isinstance(d,dict) or d.get("status")!="ok": continue
    if float(d.get("dtau_mult",0))!=6.0: continue
    cells[(float(d["zeta"]),int(d["L"]),int(d["N_c"]),round(float(d["lam"]),4))].append(d)
print("cells:",len(cells))
def stat(pops,R=16):
    pops=sorted(pops,key=lambda d:d["seed"])[:R]
    m=np.array([p["cmi_weighted_mean"] for p in pops],float)
    return len(m), float(m.mean()), float(m.std(ddof=1)/math.sqrt(len(m)))
# relative SEM at R=16, L=48,64
rows=[]
for k in sorted(cells):
    z,L,nc,lam=k
    if L not in (48,64) or len(cells[k])<16: continue
    n,mu,se=stat(cells[k])
    rows.append((z,L,nc,lam,mu,se,se/abs(mu) if mu else float('nan'),len(cells[k])))
import statistics as st
print("\n rel SEM at R=16 by (z,L,Nc): median over lambda, min, max, nlam")
by=collections.defaultdict(list)
for z,L,nc,lam,mu,se,r,N in rows: by[(z,L,nc)].append((lam,r,mu,se))
for k in sorted(by):
    v=by[k]; rs=[x[1] for x in v]
    print("  z=%.2f L=%d Nc=%-5d  medrel=%.4f  min=%.4f max=%.4f  nlam=%d  CMIrange %.4f..%.4f"
          %(k[0],k[1],k[2],st.median(rs),min(rs),max(rs),len(v),min(x[2] for x in v),max(x[2] for x in v)))
