import numpy as np

def simulate_multiplicative_OU(T, mu=1.0, phi=0.95, sigma=0.05, x0=None, seed=None):
    """
    Simulate the modified multiplicative OU process:
        X_{t+1} = (phi + eps_t) * X_t + (1 - phi) * mu
    where eps_t ~ N(0, sigma^2).

    Parameters
    ----------
    T : int
        Number of timesteps to simulate.
    mu : float
        Long-run mean level (positive).
    phi : float
        Mean reversion parameter (typically in (0,1)).
    sigma : float
        Std deviation of the multiplicative noise term.
    x0 : float or None
        Initial value. If None, starts at mu.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (T+1,)
        Simulated process values.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=T)

    X = np.empty(T + 1)
    X[0] = mu if x0 is None else x0

    for t in range(T):
        X[t + 1] = (phi + eps[t]) * X[t] + (1 - phi) * mu

    return X
