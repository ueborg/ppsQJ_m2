import glob,json,collections,statistics as st
ROOT="/Users/catlover1337/Documents/ppsQJ_m2"
rows=[]
keys=None
for p in glob.glob(ROOT+"/research/tasks/**/results/*.json",recursive=True):
    try: d=json.load(open(p))
    except Exception: continue
    if not isinstance(d,dict) or d.get("status")!="ok": continue
    if float(d.get("dtau_mult",0))!=6.0 or float(d["zeta"])!=0.35: continue
    if keys is None: keys=sorted(d.keys()); print("JSON KEYS:",keys)
    rows.append((int(d["L"]),int(d["N_c"]),round(float(d["lam"]),4),
                 float(d["wall_s"])/(int(d["N_c"])*int(d["n_steps"]))*1000.0, p))
print()
for L in (32,48,64,128):
    print("=== L=%d, lambda-matched rate (median [max], n) ==="%L)
    lams=sorted(set(r[2] for r in rows if r[0]==L))
    ncs=sorted(set(r[1] for r in rows if r[0]==L))
    for lam in lams:
        s=[]
        for nc in ncs:
            v=[r[3] for r in rows if r[0]==L and r[1]==nc and r[2]==lam]
            s.append("%d:%s"%(nc,"--" if not v else "%.2f[%.2f]n%d"%(st.median(v),max(v),len(v))))
        print("  lam=%.4f  "%lam+"  ".join(s))
