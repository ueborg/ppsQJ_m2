"""
PPS-Doob Lindbladian diagonalization for QJ-PPS Case B.

Restricts to FIXED FILLING SECTOR (since [N, H]=0 and [N, L_j]=0).
"""
import numpy as np
from scipy import linalg
from itertools import combinations

def basis_at_filling(L, N):
    """List of basis states (as bitstrings) with exactly N particles."""
    states = []
    for positions in combinations(range(L), N):
        bs = 0
        for p in positions:
            bs |= (1 << p)
        states.append(bs)
    return states

def hopping_matrix_elements(L, N, w):
    """Build H = -w sum_j (c_{j+1}^dag c_j + h.c.) in the N-particle sector."""
    states = basis_at_filling(L, N)
    d = len(states)
    idx = {s: i for i, s in enumerate(states)}
    H = np.zeros((d, d), dtype=complex)
    for s in states:
        i = idx[s]
        for j in range(L-1):
            # c_{j+1}^dag c_j: site j has 1, site j+1 has 0; move particle
            if (s >> j) & 1 and not ((s >> (j+1)) & 1):
                t = s ^ (1 << j) ^ (1 << (j+1))
                # Jordan-Wigner sign: count fermions between j and j+1 — none, so sign is +1
                # (since they're adjacent sites and there's nothing between)
                H[idx[t], i] += -w
                H[i, idx[t]] += -w  # h.c. (real w)
    return H, states

def density_op_diagonal(L, N, j):
    """n_j is diagonal in the number basis. Returns its diagonal in N-sector."""
    states = basis_at_filling(L, N)
    return np.array([(s >> j) & 1 for s in states], dtype=complex)

def vectorize_lindbladian(L, N, alpha, w, zeta):
    """Build vectorized Lindbladian in the N-particle sector.
    
    Vec convention: rho is d x d, vec(rho)_{ab} = rho_{ab} (row-stacked or col-stacked).
    We use col-stacked: vec(rho)[a + d*b] = rho[a,b].
    Then A rho B = (B^T x A) vec(rho).
    """
    H, states = hopping_matrix_elements(L, N, w)
    d = len(states)
    I = np.eye(d, dtype=complex)
    Lvec = -1j * np.kron(I, H) + 1j * np.kron(H.T, I)
    for j in range(L):
        n_diag = density_op_diagonal(L, N, j)
        # n_j is diagonal in N-sector: as matrix it's diag(n_diag)
        # L_j = L_j^dag = n_j (Hermitian)
        # L_j^dag L_j = n_j^2 = n_j (since n_j is projector)
        # Recycling: zeta * L rho L^dag = zeta * n_j rho n_j
        #   In matrix elements: (n_j rho n_j)_{ab} = n_a * rho_{ab} * n_b
        #   So as superoperator: diag(n_a * n_b) on vec
        N_diag = np.outer(n_diag, n_diag).flatten('F')  # col-major: vec[a + d*b]
        Lvec += alpha * zeta * np.diag(N_diag)
        # Anti-commutator: -(1/2) {n_j, rho} = -(1/2)(n_j rho + rho n_j)
        #   (n_j rho + rho n_j)_{ab} = (n_a + n_b) rho_{ab}
        anti = np.add.outer(n_diag, n_diag).flatten('F')
        Lvec += -0.5 * alpha * np.diag(anti)
    return Lvec, states

def compute_gap(L, alpha, w, zeta, N=None):
    """Compute the spectral structure of the PPS-Doob Lindbladian at half-filling."""
    if N is None:
        N = L // 2
    Lvec, states = vectorize_lindbladian(L, N, alpha, w, zeta)
    eigs = linalg.eigvals(Lvec)
    eigs = sorted(eigs, key=lambda z: -z.real)
    scgf = eigs[0].real
    # Find next eigenvalue below the (possibly degenerate) top
    tol = 1e-9 * max(abs(scgf), 1)
    second = next((e for e in eigs[1:] if (scgf - e.real) > tol), eigs[1])
    gap = scgf - second.real
    return scgf, gap, np.array(eigs[:12])

if __name__ == "__main__":
    print("=== Sanity check: Born rule (zeta=1) ===")
    for L in [4, 6]:
        scgf, gap, eigs = compute_gap(L, alpha=0.5, w=0.5, zeta=1.0)
        print(f"L={L}, alpha=w=0.5, zeta=1.0:  SCGF={scgf:+.6f}, gap={gap:.6f}")
        print(f"  top 5 eigs: " + ", ".join(f"{e.real:+.4f}{e.imag:+.4f}j" for e in eigs[:5]))
    
    print("\n=== Test: zeta dependence at fixed alpha=w=0.5 ===")
    for zeta in [1.0, 0.5, 0.2, 0.1, 0.05]:
        scgf, gap, eigs = compute_gap(L=4, alpha=0.5, w=0.5, zeta=zeta)
        print(f"  zeta={zeta:.2f}: SCGF={scgf:+.5f}, gap={gap:.5f}")
