import pickle, numpy as np

D = pickle.load(open('/Users/catlover1337/Downloads/_guided_master.pkl','rb'))
df, arrs = D['df'], D['arrs']

# attach array dict index to df rows
df = df.reset_index(drop=True)

def get_curve(dset, L, zeta, obs):
    """Return (lams, mean, per-real matrix [n_lam x n_real]) for an observable.
    obs in {'CMI','S_AB','B_L','KMR'}."""
    m = (df.dset==dset) & (df.L==L) & (np.abs(df.zeta-zeta)<1e-9)
    idx = np.where(m.values)[0]
    if len(idx)==0: return None
    rows = [(df.lam.values[i], i) for i in idx]
    rows.sort()
    lams = np.array([r[0] for r in rows])
    mats = []
    for _,i in rows:
        a = arrs[i]
        if obs=='CMI':   v = a['CMI_means_all']
        elif obs=='S_AB':v = a['S_AB_means_all']
        elif obs=='B_L': v = a['B_L_means_all']
        elif obs=='KMR': v = a['CMI_means_all']*a['S_AB_means_all']
        mats.append(np.asarray(v,float))
    M = np.vstack(mats)               # [n_lam, n_real]
    return lams, np.nanmean(M,axis=1), M

# ---- sanity: CMI(lam) per L at a few zeta (Case B) ----
print('=== Case B: <CMI>(lam) per L  (sanity / crossing visibility) ===')
for zeta in [0.1, 0.3, 0.5]:
    print(f'\n zeta={zeta}')
    Ls = [32,48,64,96,128,160]
    # collect on union grid
    for L in Ls:
        dset = 'B_highL' if L==160 else 'B_prod'
        c = get_curve(dset, L, zeta, 'CMI')
        if c is None: 
            print(f'   L={L:3d}: (none)'); continue
        lams, mean, _ = c
        s = '  '.join(f'{x:.2f}:{y:.3f}' for x,y in zip(lams,mean))
        print(f'   L={L:3d}: {s}')
