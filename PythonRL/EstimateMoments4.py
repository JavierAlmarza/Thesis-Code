import numpy as np
import math
from numpy.linalg import cond

# ---------------- helper functions ----------------

def pade_from_series(s, L, M, reg=1e-12):
    N = L + M
    assert len(s) >= N + 1, "Need at least L+M+1 series coefficients"
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
        q_tail, *_ = np.linalg.lstsq(A + reg * np.eye(M), b, rcond=None)
    q = np.concatenate(([1.0], q_tail))
    p = np.zeros(L + 1, dtype=float)
    for k in range(L + 1):
        acc = 0.0
        for j in range(min(k, M) + 1):
            acc += q[j] * s[k - j]
        p[k] = acc
    return p, q, cnum


def estimate_pade(X_flat, phi=0.1, Nseries=30, L=None, M=None):
    """Estimate E[1/X] and E[Roil] via Padé from centered moments of X."""
    X = np.asarray(X_flat)
    mu = X.mean()

    # Central moments
    m = [1.0]
    for k in range(1, Nseries + 1):
        m.append(np.mean((X - mu) ** k))

    # Series for E[1/(mu + (X - mu))] = (1/mu) * sum (-1)^k * m_k / mu^k
    s = np.zeros(Nseries + 1)
    for k in range(Nseries + 1):
        if k == 0:
            s[k] = 1.0
        else:
            s[k] = ((-1.0) ** k) * (m[k] / (mu ** k))

    if L is None or M is None:
        M = Nseries // 2
        L = Nseries - M

    p, q, condA = pade_from_series(s, L, M)

    f_at_1 = np.sum(p) / np.sum(q)
    E_invX_pade = (1.0 / mu) * f_at_1
    E_Roil_pade = phi * (mu * E_invX_pade - 1.0)

    return {
        'Nseries': Nseries,
        'L': L,
        'M': M,
        'condA': condA,
        'E_invX_pade': E_invX_pade,
        'E_Roil_pade': E_Roil_pade,
        'p_coeff': p,
        'q_coeff': q,
        'mu': mu,
        'm_k': m,
        's_k': s,
    }


def diagnostics_and_estimates(X_flat, Roil_flat, phi=0.1):
    """Compute empirical diagnostics and analytical approximations for Roil."""
    X = np.asarray(X_flat)
    mu = float(X.mean())
    varX = float(X.var())
    sdX = math.sqrt(varX)

    # Empirical expectations
    E_invX_emp = float(np.mean(1.0 / X))
    E_Roil_emp = float(np.mean(Roil_flat))

    out = {
        'mu': mu,
        'varX': varX,
        'sdX': sdX,
        'E_invX_emp': E_invX_emp,
        'E_Roil_emp': E_Roil_emp,
    }

    # Sweep Padé estimates for stability
    Nchoices = [6, 8, 10, 12, 14, 16, 18, 20, 24]
    pade_results = []
    for N in Nchoices:
        for tryLM in [(N // 2, N - N // 2), (N - (N // 2 + 1), N // 2 + 1)]:
            L, M = tryLM
            try:
                info = estimate_pade(X, phi=phi, Nseries=N, L=L, M=M)
                pade_results.append(info)
            except Exception as e:
                pade_results.append({'Nseries': N, 'L': L, 'M': M, 'error': str(e)})
    out['pade_results'] = pade_results

    # Lognormal-fit approximation
    s2 = math.log(1.0 + varX / (mu ** 2))
    E_invX_logn = (1.0 / mu) * math.exp(s2)
    E_Roil_logn = phi * (mu * E_invX_logn - 1.0)
    out['lognormal'] = {'E_invX_logn': E_invX_logn, 'E_Roil_logn': E_Roil_logn}

    # Gamma-fit approximation
    if mu**2 > varX:
        E_invX_gamma = mu / (mu**2 - varX)
        E_Roil_gamma = phi * (mu * E_invX_gamma - 1.0)
    else:
        E_invX_gamma = float('nan')
        E_Roil_gamma = float('nan')
    out['gamma'] = {'E_invX_gamma': E_invX_gamma, 'E_Roil_gamma': E_Roil_gamma}

    return out
