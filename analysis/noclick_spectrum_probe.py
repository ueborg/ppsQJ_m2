"""
No-click correlation length of the QJ-PPS Case B effective generator.

KEY RESULT (this session): the no-click steady-state correlation length scales as
    xi_nc ~ lambda^{-1}   (prefactor ~ 4-5),
NOT lambda^{-2}.  This is established by stable stepped no-click evolution with
L >> xi and a clean exponential fit of the center-bond two-point function.

Why it matters: the "matched-NLSM" derivation of lambda_c ~ sqrt(zeta) requires a
no-click length xi ~ lambda^{-2} (inherited from the QSD/KMR analysis).  The QJ
model does NOT have that length.  Two distinct scales are present:
  - the SSH dimerization gap of each decoupled sublattice -> xi_dim ~ lambda^{-2},
  - the decay-rate minimum momentum k* ~ lambda -> xi_decay ~ lambda^{-1}.
The steady state is selected by the slowest-decaying modes (near k*), so the
correlation length governing entanglement is xi_nc ~ lambda^{-1}, NOT the gap.

The Hamiltonian splits exactly into two decoupled Majorana chains (E = indices
0,3 mod 4; O = 1,2 mod 4); the measurement attaches -i*alpha only to the +w
bonds.  Each sublattice is a non-Hermitian SSH Majorana chain with bonds
t1 = w - i*alpha (measured) and t2 = -w (unmeasured).
"""
import numpy as np
from scipy.linalg import expm


def h_eff_majorana(L, w, alpha):
    h = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    for b in range(L - 1):
        a, c = 2 * b, 2 * b + 3
        e, f = 2 * b + 1, 2 * b + 2
        h[a, c] = w; h[c, a] = -w
        h[e, f] = -w; h[f, e] = w
        h[a, c] -= 1j * alpha; h[c, a] += 1j * alpha
    return h


def _neel_cov(L):
    g = np.zeros((2 * L, 2 * L))
    for s in range(L):
        a = 2 * s; sg = 1.0 if s % 2 == 0 else -1.0
        g[a, a + 1] = sg; g[a + 1, a] = -sg
    return g


def _cov_from_orb(orb):
    n = orb.shape[0]
    g = 1j * (2 * (orb @ orb.conj().T) - np.eye(n))
    return 0.5 * (g.real - g.real.T)


def _orb_from_cov(G):
    n = G.shape[0]
    v, V = np.linalg.eigh(1j * G)
    o = np.argsort(v.real)
    Q = V[:, o[:n // 2]]
    Q, _ = np.linalg.qr(Q, mode="reduced")
    return Q


def noclick_xi(L, lam, n_steps=400):
    """Stable stepped no-click evolution; fit xi from |Gamma_{c,c+r}| ~ exp(-r/xi)."""
    w, alpha = 1.0 - lam, lam
    h = h_eff_majorana(L, w, alpha)
    sp = np.linalg.eigvals(h)
    dt = 2.0 / max(np.max(np.abs(sp.imag)), 1.0)
    M = expm(h * dt)
    Q = _orb_from_cov(_neel_cov(L))
    for _ in range(n_steps):
        Q, _ = np.linalg.qr(M @ Q, mode="reduced")
    G = _cov_from_orb(Q)
    c0 = 2 * (L // 2)
    rmax = min(2 * L - c0 - 4, 6 * int(1 / lam ** 2) + 40)
    rr = np.arange(4, rmax)
    vals = np.array([abs(G[c0, c0 + r]) for r in rr])
    m = vals > 1e-11
    if m.sum() < 12:
        return np.nan
    x, y = rr[m], np.log(vals[m])
    k = min(len(x), 80)
    A = np.vstack([x[:k], np.ones(k)]).T
    slope, _ = np.linalg.lstsq(A, y[:k], rcond=None)[0]
    return (-1.0 / slope) if slope < -1e-6 else np.inf


if __name__ == "__main__":
    print(f"{'lam':>6} {'L':>6} {'xi_nc':>10} {'xi*lam':>9} {'xi*lam^2':>10}")
    for lam, L in [(0.40, 128), (0.30, 128), (0.25, 160), (0.20, 200),
                   (0.15, 300), (0.12, 400), (0.10, 500)]:
        xi = noclick_xi(L, lam)
        print(f"{lam:6.2f} {L:6d} {xi:10.2f} {xi*lam:9.3f} {xi*lam**2:10.3f}")
    print("\nFlat xi*lam  => xi_nc ~ lambda^{-1}  (NOT lambda^{-2}).")
