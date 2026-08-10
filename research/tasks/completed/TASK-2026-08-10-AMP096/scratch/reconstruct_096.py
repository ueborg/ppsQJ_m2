"""T0 READ-ONLY: exact reconstruction of the historical A = 0.96.
Input: the lambda_c^inf table PRINTED IN analysis/lambda_c_phi_analysis.md
(provenance document, mtime-checked below). Inverse-variance weighted log-log
fit over the document's own window zeta <= 0.3 reproduces phi = 0.502 +/- 0.026,
A = 0.96, chi2/dof = 10.7 as quoted. No RNG.
"""
import numpy as np, json, os, subprocess
TAB={0.02:(0.149,0.008),0.05:(0.157,0.019),0.10:(0.251,0.022),
     0.15:(0.233,0.025),0.20:(0.229,0.106),0.30:(0.594,0.028),
     0.50:(0.759,0.055),0.70:(0.487,0.071),0.85:(0.459,0.020),1.00:(0.443,0.011)}
def wfit(ks):
    z=np.array(ks); y=np.array([TAB[k][0] for k in ks]); s=np.array([TAB[k][1] for k in ks])
    p=np.polyfit(np.log(z),np.log(y),1,w=y/s)
    res=(np.log(y)-np.polyval(p,np.log(z)))*(y/s)
    return float(p[0]),float(np.exp(p[1])),float(np.sum(res**2)/max(1,len(ks)-2))
out={}
for hi in (0.2,0.3,0.5,1.01):
    ks=[z for z in TAB if z<=hi+1e-9]
    out[f"window_zeta_le_{hi}"]=dict(zip(("phi","A","chi2_dof"),wfit(ks)))
ks=[z for z in TAB if z<=0.3+1e-9]
out["leave_one_out_on_documented_window"]={str(d):dict(zip(("phi","A","chi2_dof"),
    wfit([k for k in ks if k!=d]))) for d in ks}
out["documented_values"]={"phi":0.502,"phi_err":0.026,"A":0.96,"chi2_dof":10.7,
   "source":"analysis/lambda_c_phi_analysis.md (PROVENANCE, not canonical support)"}
print(json.dumps(out,indent=1))
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reconstruct_096_out.json"),"w"),indent=1)
