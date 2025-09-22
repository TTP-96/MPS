"""
Demo of MPS construction from a dense state vector and reconstruction back to state vector.

Convention:
  Gamma[i] shape = (chi_left, 2, chi_right)  # reshaped U, but made 'pure Gamma' by dividing out left λ
  Lambda[k] shape = (chi_{k+1},)              # bond lambdas on bond (k, k+1), k = [0,L-2]
Reconstruction inserts Lambda[k] on the right virtual leg between sites k and k+1.
"""

import numpy as np

def split(M, bond_dim):
    """
    One SVD step 
    Input:
      M: 2D array
      bond_dim: max Schmidt rank to keep
    Returns:
      U-> reshaped to (left_dim, 2, chi)
      S  -> (chi,)
      V -> reshaped to (chi, 2, remainder)
    """
    U, S, V = np.linalg.svd(M, full_matrices=False)
    bonds = len(S)
    chi = min(bonds, bond_dim)

    # make sure to reshape BEFORE slicing so that slicing is done consistently
    V = V.reshape(bonds, 2, -1)[:chi]          # (chi, 2, rest)
    U  = U.reshape(-1, 2, bonds)[:, :, :chi]     # (left_dim, 2, chi)
    S  = S[:chi]

    return U, S, V

def dense_to_mps(psi, bond_dim):
    """
    Build an MPS from a normalized state psi with shape (2,)*n
    Produces Gammas with shapes (chiL, 2, chiR) and Lambda (list of length n-1).
    Uses Gamma–λ gauge. We divide U by previous λ on the left index to keep tensors as pure Gamma
    The final site is V_last^{\dagger} ONLY (no lambda absorbed): shape (chi_{L-1}, 2, 1).
    """
    
    n = psi.ndim # number of sites

    Gamma, Lambda = [], []
    lambda_prev = np.ones(1, dtype=float)  # λ on the left of site 0

    # First split: psi[2, 2^(n-1)]
    psi_p = psi.reshape(2, -1)
    U, S, V = split(psi_p, bond_dim)  # U: (1, 2, chi1) S (chi1,) V (chi1, 2, rest)
    U = U / lambda_prev[:, None, None]     # pure Gamma. divide by previous λ on left index
    Gamma.append(U)
    Lambda.append(S)
    lambda_prev = S.copy()
    psi_p = np.tensordot(np.diag(S), V, axes=1)   # (chi1, 2, rest) = lambda V^{\dagger}

    # Middle splits
    for _ in range(n - 2):
        chi_left = psi_p.shape[0]
        psi_p = psi_p.reshape(2 * chi_left, -1)
        U, S, V = split(psi_p, bond_dim)      # U: (chi_left, 2, chi_right)
        U = U / lambda_prev[:, None, None]       # pure Gamma: divide by previous λ on left index
        Gamma.append(U)
        Lambda.append(S)
        lambda_prev = S.copy()
        psi_p = np.tensordot(np.diag(S), V, axes=1)  # (chi_right, 2, rest)

    # Final site: psi_p == lambda_last V_last^{dagger}, shape (chi_{L-1}, 2, 1)
    assert psi_p.shape[2] == 1, f"Expected trailing rem=1, got {psi_p.shape}"
    psi_p2 = psi_p[:, :, 0]      # (chi_{L-1}, 2)
    S_last = Lambda[-1]
    S_mod = S_last.copy()
    S_mod[S_mod == 0] = 1.0
    Vbare = psi_p2 / S_mod[:, None]              # (chi_{L-1}, 2) = V_last^{\dagger} 
    M_last = Vbare[:, :, None]                 # (chi_{L-1}, 2, 1)
    Gamma.append(M_last)

    # Sanity checks 
    # assert len(Gamma) == n and len(Lambda) == n - 1, "MPS should have n sites and n-1 lambdas"
    # assert Gamma[0].shape[0] == 1 and Gamma[-1].shape[2] == 1, "Boundary virtual dims must be 1"
    # for i in range(n - 1):
    #     chi_r = Gamma[i].shape[2]
    #     chi_lp1 = Gamma[i + 1].shape[0]
    #     assert chi_r == chi_lp1 == Lambda[i].shape[0], \
    #         f"Mismatch at bond {i}: {chi_r} vs {chi_lp1} vs {Lambda[i].shape[0]}"

    return Gamma, Lambda

def mps_to_dense(Gamma, Lambda):
    """
    Reconstruct dense state from Gamma (chiL, 2, chiR) and Lambda (bond lambdas).
    Insert λ on the RIGHT virtual leg between consecutive sites.
    psi =  Γ[0] λ[0] Γ[1] λ[1] ... λ[L-2] Γ[L-1]
    """
    L = len(Gamma)
    d = 2
    assert len(Lambda) == L - 1, "Require exactly one lambda vector per internal bond"

    # Start with site 0: (1, 2, chi1) -> (2, chi1)
    T = np.squeeze(Gamma[0], axis=0)          # (2, chi1)
    

    for i in range(L - 1):
        lam = Lambda[i]                           # (chi_{i+1},)
        T = T * lam[np.newaxis, :]   # insert λ on right virtual bond. (2,chi_i) -> (2, chi_i) x (,chi_i,) = (2, chi_i)
        # (..., chi_{i+1}) x (chi_{i+1}, 2, chi_{i+2}) ->        
        T = np.tensordot(T, Gamma[i + 1], axes=([-1], [0]))
        #print(f"After site {i+1}, T.shape={T.shape}")

    T = np.squeeze(T, axis=-1)                # (..., 2) 
    return T.reshape(d ** L) # (2, 2, 2, ...) -> (2^L,)

if __name__ == "__main__":
    n = 12
    rng = np.random.default_rng(1)
    psi_original = rng.normal(size=(2,) * n) + 1j * rng.normal(size=(2,) * n)
    psi = psi_original / np.linalg.norm(psi_original)

    bond_dim_list = [2, 4, 8, 16, 32, 64, 128, 512, 2048, 10000]
    print(f"Normalized random state with {2**n} amplitudes")
    print("Fidelity vs χ:\n")

    for chi in bond_dim_list:
        Gamma, Lambda = dense_to_mps(psi, chi)
        psi_rec = mps_to_dense(Gamma, Lambda)
        psi_rec /= np.linalg.norm(psi_rec)           # numerical drift
        F = np.abs(np.vdot(psi.reshape(-1), psi_rec))**2
        print(f"chi={chi:5d} | Fidelity: {F:.12f}")

    chi_req = 2 ** (n // 2)
    print(f"Theoretical exact χ for n={n}: {chi_req}")
