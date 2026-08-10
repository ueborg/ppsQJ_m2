import os, json, glob, numpy as np

base = '/Users/catlover1337/Downloads'
dirs = ['pps_clone_guided_prod', 'pps_caseA_guided', 'pps_clone_guided_highL']

for d in dirs:
    p = os.path.join(base, d)
    npz = sorted(glob.glob(os.path.join(p, '*.npz')))
    js = sorted(glob.glob(os.path.join(p, 'summary_*.json')))
    print('=' * 70)
    print(d, '| npz:', len(npz), '| summaries:', len(js))
    if js:
        s = json.load(open(js[0]))
        print('  summary keys:', list(s.keys()))
    if npz:
        z = np.load(npz[0], allow_pickle=True)
        print('  npz keys + shapes:')
        for k in z.files:
            a = z[k]
            print('   ', k, getattr(a, 'shape', None), getattr(a, 'dtype', None))
