import os, glob, pickle, numpy as np, pandas as pd

base = '/Users/catlover1337/Downloads'
SETS = {
    'B_prod':  'pps_clone_guided_prod',
    'A':       'pps_caseA_guided',
    'B_highL': 'pps_clone_guided_highL',
}

# scalar fields we always want if present
SCALARS = ['task_id','L','lam','zeta','T','N_c','n_real',
           'alpha','w','gamma_rate','alpha_rate',
           'S_mean','S_err','theta_mean','theta_err','ESS_mean','min_ess_frac_mean',
           'B_L_mean','B_L_err','CMI_mean','CMI_err',
           'S_AB_mean','S_BC_mean','S_B_mean','S_ABC_mean',
           'S_renyi_2_mean','S_renyi_3_mean',
           'n_T_mean','n_collapses','n_js_fallbacks','wall_time']
# per-realisation arrays (for bootstrap)
ARRAYS = ['B_L_means_all','S_means_all','CMI_means_all','S_AB_means_all','thetas_all','ESSs_all',
          'n_ancestors_all','min_ess_fracs_all']

def load_set(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(base, folder, '*.npz'))):
        z = np.load(f, allow_pickle=True)
        keys = set(z.files)
        r = {}
        for k in SCALARS:
            r[k] = float(z[k]) if k in keys else np.nan
        for k in ARRAYS:
            r[k] = z[k].astype(float) if k in keys else None
        rows.append(r)
    return rows

all_rows = {}
for tag, folder in SETS.items():
    rows = load_set(folder)
    for r in rows:
        r['dset'] = tag
    all_rows[tag] = rows
    print(f'{tag:8s} loaded {len(rows)} tasks')

# flatten to DataFrame (scalars only for the table; arrays kept in a parallel dict)
flat = []
arrs = []
for tag in SETS:
    for r in all_rows[tag]:
        d = {k: r[k] for k in r if k not in ARRAYS}
        flat.append(d)
        arrs.append({k: r[k] for k in ARRAYS})
df = pd.DataFrame(flat)

with open('/Users/catlover1337/Downloads/_guided_master.pkl','wb') as fh:
    pickle.dump({'df': df, 'arrs': arrs}, fh)

print('\n==== GRID COVERAGE ====')
for tag in SETS:
    sub = df[df.dset == tag]
    print('\n---', tag, '--- (', len(sub), 'tasks )')
    print('  N_c values:', sorted(sub.N_c.dropna().unique().astype(int).tolist()))
    print('  T values  :', sorted(np.round(sub['T'].dropna().unique(),1).tolist())[:8])
    Ls = sorted(sub.L.dropna().unique().astype(int).tolist())
    zs = sorted(np.round(sub.zeta.dropna().unique(),3).tolist())
    print('  L     :', Ls)
    print('  zeta  :', zs)
    # per (L,zeta): how many lambda points and the lambda range
    print('  (L,zeta) -> n_lam, lam_min..lam_max  [first 40]')
    g = sub.groupby(['L','zeta'])
    cnt = 0
    for (L,z), gg in g:
        lams = np.sort(gg.lam.unique())
        print(f'    L={int(L):3d} z={z:5.3f}: n={len(lams):2d}  lam {lams.min():.3f}..{lams.max():.3f}')
        cnt += 1
        if cnt >= 40:
            print('    ... (truncated)')
            break

print('\nstatus / health (Case B prod):')
sub = df[df.dset=='B_prod']
print('  ESS/N_c median:', np.round((sub.ESS_mean/sub.N_c).median(),3),
      ' min:', np.round((sub.ESS_mean/sub.N_c).min(),3))
print('  n_collapses>0:', int((sub.n_collapses>0).sum()), '/', len(sub))
