import numpy as np
import math
from numpy.linalg import cond

# ---------------- helper functions ----------------

def pade_from_series(s, L, M, reg=1e-12):
    N = L + M
    assert len(s) >= N+1, "Need at least L+M+1 series coefficients"
    A = np.zeros((M, M), dtype=float)
    b = np.zeros(M, dtype=float)
    for i in range(M):
        for j in range(M):
            A[i, j] = s[L + i - j]
        b[i] = -s[L + i + 1]
    try:
        cnum = cond(A)
    except Exception:
        cnum = np.nan
    try:
        q_tail = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        q_tail, *_ = np.linalg.lstsq(A + reg*np.eye(M), b, rcond=None)
    q = np.concatenate(([1.0], q_tail))
    p = np.zeros(L+1, dtype=float)
    for k in range(L+1):
        acc = 0.0
        for j in range(min(k, M) + 1):
            acc += q[j] * s[k - j]
        p[k] = acc
    return p, q, cnum


def estimate_pade(X_flat, phi=0.1, Nseries=30, L=None, M=None):
    """Estimate E[1/X] and E[Roil] via Padé from centered moments of X."""
    X = np.asarray(X_flat)
    mu = X.mean()

    # central moments
    m = [1.0]
    for k in range(1, Nseries + 1):
        m.append(np.mean((X - mu) ** k))

    # series for E[1/(mu + (X - mu))] = (1/mu) * sum (-1)^k * m_k / mu^k
    s = np.zeros(Nseries + 1)
    for k in range(Nseries + 1):
        s[k] = 1.0 if k == 0 else ((-1.

