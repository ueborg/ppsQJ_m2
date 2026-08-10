"""Two-replica Lindbladian for QJ-PPS Case B.

The trajectory-averaged ⟨ρ ⊗ ρ⟩_PPS evolves under a master equation with:
- Independent unitary evolution on each replica (H, -H^T structure)
- Independent decay anti-commutator (-α/2 {L_j^dag L_j, ·} on each replica)
- CROSS recycling vertex: α·ζ (L_j ⊗ L_j) M (L_j^dag ⊗ L_j^dag)
  (Note: replicas share clicks, so the recycling COUPLES them)

Setup uses two copies of the half-filling sector. For L=6, dim = 20*20 = 400.
The two-replica Liouville space (vec(M)) has dim = 400^2 = 160000.
That's borderline. We use sparse / iterative methods to find the top eigenvalues.

Restricted version: we work in the SAME-FILLING-EACH-REPLICA sector,
which is what's relevant for the MIPT problem at fixed total particle number.
"""
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import eigs as sparse_eigs
from itertools import combinations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pps_lindbladian import basis_at_filling, hopping_matrix_elements, density_op_diagonal

def build_two_replica_lindbladian(L, alpha, w, zeta, N=None):
    """Build the linearized two-replica Lindbladian acting on vec(M),
    where M is a (d×d) operator on each replica's N-particle sector (d = binom(L,N)).
    
    The full superoperator acts on a (d²)² = d⁴ dim space.
    """
    if N is None:
        N = L // 2
    H, states = hopping_matrix_elements(L, N, w)
    d = len(states)
    I = np.eye(d, dtype=complex)
    
    # Single-replica vectorized Lindbladian (no recycling — we'll add cross instead)
    # L_single_diag[M] = -i(H rho - rho H) - (alpha/2)({L^dag L, rho}) for each replica
    # In vec form: -i(I x H) + i(H^T x I) - (alpha/2)(I x LdL + LdL^T x I)
    
    # We want to build L^(2) acting on M (d x d) for each replica separately, plus cross vertex
    # M lives in d^2 x d^2 space (replica A and replica B as separate vectors)
    # vec(M)_{a1a2,b1b2} = M[a1 a2, b1 b2]
    
    # The 4-index object has dimension d^4. Indexing: I = a1*d^3 + a2*d^2 + b1*d + b2
    d4 = d**4
    print(f"  d = {d}, d^4 = {d4}, building sparse Liouvillian...")
    
    # Build using vectorized operations / sparse construction
    # Let's use Kronecker structure
    
    # Single-replica Lindbladian WITHOUT recycling (so just unitary + decay)
    Lvec_no_recyc = -1j * np.kron(I, H) + 1j * np.kron(H.T, I)
    for j in range(L):
        n_diag = density_op_diagonal(L, N, j)
        anti = np.add.outer(n_diag, n_diag).flatten('F')
        Lvec_no_recyc += -0.5 * alpha * np.diag(anti)
    
    # Two-replica part: act on each replica independently
    Id2 = np.eye(d**2, dtype=complex)
    L2_intra = np.kron(Id2, Lvec_no_recyc) + np.kron(Lvec_no_recyc, Id2)
    
    # Cross recycling vertex: alpha * zeta * sum_j (n_j ⊗ n_j) on M (n_j ⊗ n_j)
    # In vec form on M (which is d^2 x d^2 operator), this becomes:
    # (n_j ⊗ n_j) ⊗ (n_j ⊗ n_j)^T = diag(n_a1[j] n_a2[j] n_b1[j] n_b2[j])
    L2_cross = np.zeros((d4, d4), dtype=complex)
    for j in range(L):
        n_diag = density_op_diagonal(L, N, j)
        # 4-index diagonal: n_a1 * n_a2 * n_b1 * n_b2
        # Indexing in vec(M_{a1a2, b1b2}): order (a1, a2, b1, b2) flattened
        # We use vec convention: vec[i] where i = a1*d^3 + a2*d^2 + b1*d + b2
        Lj_diag_4 = np.zeros(d4, dtype=complex)
        for a1 in range(d):
            for a2 in range(d):
                for b1 in range(d):
                    for b2 in range(d):
                        idx = a1*d**3 + a2*d**2 + b1*d + b2
                        Lj_diag_4[idx] = n_diag[a1] * n_diag[a2] * n_diag[b1] * n_diag[b2]
        L2_cross += alpha * zeta * np.diag(Lj_diag_4)
    
    return L2_intra + L2_cross

def compute_two_replica_gap(L, alpha, w, zeta, N=None, n_eigs=8):
    """Compute the leading eigenvalues of the two-replica Lindbladian."""
    L2 = build_two_replica_lindbladian(L, alpha, w, zeta, N)
    # Find top eigenvalues by real part
    eigs = linalg.eigvals(L2)
    eigs = sorted(eigs, key=lambda z: -z.real)[:n_eigs]
    return np.array(eigs)

if __name__ == "__main__":
    print("=== Two-replica Lindbladian: gap structure ===\n")
    # Quick test at L=4 first (d=6 for half-filling, d^4 = 1296)
    L = 4
    print(f"Test at L={L}:")
    for zeta in [1.0, 0.5, 0.2, 0.1]:
        eigs = compute_two_replica_gap(L, alpha=0.5, w=0.5, zeta=zeta)
        # Filter out essentially-zero eigenvalues (numerical noise)
        nonzero = [e for e in eigs if abs(e.real) > 1e-8]
        top = max(e.real for e in eigs)
        # Find first eigenvalue strictly below top
        second_top = max((e.real for e in eigs if e.real < top - 1e-6), default=top)
        gap = top - second_top
        print(f"  zeta={zeta:.2f}:  top_eig={top:+.5f}, second={second_top:+.5f}, gap={gap:.5f}")
        print(f"    first 4 eigs: " + ", ".join(f"{e.real:+.4f}" for e in eigs[:4]))
