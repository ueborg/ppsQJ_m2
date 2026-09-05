import glob,json,collections,math,numpy as np
from statistics import NormalDist
ROOT="/Users/catlover1337/Documents/ppsQJ_m2"
cells=collections.defaultdict(list)
for p in glob.glob(ROOT+"/research/tasks/**/results/*.json",recursive=True):
    try: d=json.load(open(p))
    except Exception: continue
    if not isinstance(d,dict) or d.get("status")!="ok": continue
    if float(d.get("dtau_mult",0))!=6.0 or float(d["zeta"])!=0.35: continue
    cells[(int(d["L"]),int(d["N_c"]),round(float(d["lam"]),4))].append(d)
def stat(k,R=16):
    ps=sorted(cells[k],key=lambda d:d["seed"])[:R]
    if len(ps)<2: return None
    m=np.array([p["cmi_weighted_mean"] for p in ps],float)
    return len(m),float(m.mean()),float(m.std(ddof=1)/math.sqrt(len(m)))
print("L=64 lam=0.3032 ladder (R=16):")
prev=None
for nc in (64,256,1024,2048,4096,8192):
    s=stat((64,nc,0.3032))
    if not s: print("  Nc=%d  --"%nc); continue
    n,mu,se=s
    line="  Nc=%-5d n=%d CMI=%.5f SEM=%.5f rel=%.4f"%(nc,n,mu,se,se/mu)
    if prev:
        d=mu-prev[1]; sc=math.sqrt(se**2+prev[2]**2)
        line+="   Delta=%+.5f (%.1f%%)  z=%.2f"%(d,100*d/prev[1],abs(d)/sc)
    print(line); prev=(nc,mu,se)
print()
print("L=48/64 lam sweep at Nc=512 vs 1024 vs 2048 (7 shared lambdas), z per lambda, R=16")
lams=[0.2182,0.2232,0.2282,0.2332,0.2382,0.2432,0.2482]
for (a,b) in [(512,1024),(1024,2048)]:
    zs=[]
    for L in (48,64):
        for lam in lams:
            s1,s2=stat((L,a,lam)),stat((L,b,lam))
            if not s1 or not s2: continue
            sc=math.sqrt(s1[2]**2+s2[2]**2)
            zs.append(abs(s2[1]-s1[1])/sc)
    n=len(zs); zc=NormalDist().inv_cdf(0.5*(1+0.99**(1.0/n)))
    print("  rung %d->%d : n=%d max|z|=%.2f  z_crit=%.3f -> %s"%(a,b,n,max(zs),zc,
        "CLEARLY_TOO_SMALL" if max(zs)>=2*zc else "STILL_CHANGING" if max(zs)>=zc else "ROUGHLY_STABLE"))
    print("    all z:", " ".join("%.2f"%x for x in zs))
