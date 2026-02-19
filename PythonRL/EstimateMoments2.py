import numpy as np

def pade_from_series(s, L, M, reg=1e-12):
    """
    Build Pade [L/M] approximant from series coefficients s[0..L+M]
    Returns numerator coeffs p (len L+1) and denominator coeffs q (len M+1, q[0]=1)
    """
    N = L + M
    assert len(s) >= N+1, "Need at least L+M+1 series coefficients"

    # Build linear system A q = b to find q1..qM
    # For i = 1..M: sum_{j=1..M} q_j * s[L + i - j] = - s[L + i]
    A = np.zeros((M, M), dtype=float)
    b = np.zeros(M, dtype=float)
    for i in range(M):
        for j in range(M):
            A[i, j] = s[L + i - j]      # s index safe if len(s) >= L+M+1 and L>=M-1
        b[i] = -s[L + i + 1]            # right-hand side: -s[L+1], -s[L+2], ..., -s[L+M]

    # Solve (regularize if necessary)
    try:
        q_tail = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        q_tail, *_ = np.linalg.lstsq(A + reg*np.eye(M), b, rcond=None)

    q = np.concatenate(([1.0], q_tail))   # q_0 = 1

    # Compute numerator coefficients p_k for k=0..L:
    p = np.zeros(L+1, dtype=float)
    for k in range(L+1):
        # p_k = sum_{j=0..min(k,M)} q_j * s[k-j]
        acc = 0.0
        for j in range(min(k, M) + 1):
            acc += q[j] * s[k - j]
        p[k] = acc

    return p, q

def estimate_E_R_oil_from_X(X_samples, phi=0.1, FSens=0.2, Nseries=30, L=None, M=None):
    """
    X_samples: 1D array of price samples (pooled across paths/time).
    Returns: (E_R_oil_pade, E_invX_pade, details dict)
    """
    X = np.asarray(X_samples).ravel()
    mu = X.mean()

    # central moments m_k for k=1..Nseries
    m = [1.0]  # m0 = 1
    for k in range(1, Nseries+1):
        m_k = np.mean((X - mu)**k)
        m.append(m_k)

    # Build series coefficients s_k = (-1)^k * E[Y^k], with Y=(X-mu)/mu
    # E[Y^k] = m_k / mu^k for k>=1; E[Y^0] = 1
    s = np.zeros(Nseries+1, dtype=float)
    for k in range(Nseries+1):
        if k == 0:
            s[k] = 1.0
        else:
            s[k] = ((-1.0)**k) * (m[k] / (mu**k))

    # choose Pade orders
    if L is None or M is None:
        M = Nseries // 2
        L = Nseries - M

    # Need at least L+M = Nseries, so require len(s) >= Nseries+1
    if len(s) < L + M + 1:
        raise ValueError("Increase Nseries or adjust L/M")

    # Build Pade [L/M] from s[0..L+M]
    p, q = pade_from_series(s, L, M)

    # Evaluate P(1)/Q(1)
    num = np.sum(p)  # sum_{k=0..L} p_k * 1^k
    den = np.sum(q)  # sum_{j=0..M} q_j * 1^j
    f_at_1 = num / den

    E_invX_pade = (1.0 / mu) * f_at_1
    E_R_oil_pade = phi * (mu * E_invX_pade - 1.0)
    E_R_pade = FSens * E_R_oil_pade

    details = {
        "mu": mu,
        "m_k": m,
        "s_k": s,
        "p_coeff": p,
        "q_coeff": q,
        "f_at_1": f_at_1,
        "E_invX_pade": E_invX_pade,
        "E_R_oil_pade": E_R_oil_pade,
        "E_R_pade": E_R_pade,
        "L": L, "M": M, "Nseries": Nseries
    }
    return E_R_oil_pade, E_R_pade, details

# ----------------------
# Example usage:
# If you have Xall (shape npaths x nt) as in your simulation,
# flatten it and pass X_flat = Xall[:, burn:].ravel()
#
# Example:
# X_flat = Xall[:, burn:].ravel()
# E_R_oil_pade, E_R_pade, info = estimate_E_R_oil_from_X(X_flat, phi=0.1, FSens=0.2, Nseries=30)
# print("Padé estimate E[Roil]=", E_R_oade, " Padé E[R]=", E_R_pade)