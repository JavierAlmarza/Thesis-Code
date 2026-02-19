import numpy as np

def generate_arch1(N, T, alpha0, alpha1):
    """
    Generate N simulations of an ARCH(1) process of length T.

    Parameters:
    - N: int, number of simulation runs
    - T: int, number of time steps per run
    - alpha0: float, ARCH(1) alpha_0 parameter
    - alpha1: float, ARCH(1) alpha_1 parameter

    Returns:
    - r: np.ndarray of shape (N, T), simulated returns
    """
    r = np.zeros((N, T))
    sigma2 = np.zeros((N, T))

    # Initial conditional variance
    sigma2[:, 0] = alpha0 / (1 - alpha1)

    # Generate standard normal shocks
    z = np.random.randn(N, T)

    # First return
    r[:, 0] = np.sqrt(sigma2[:, 0]) * z[:, 0]

    for t in range(1, T):
        sigma2[:, t] = alpha0 + alpha1 * r[:, t-1]**2
        r[:, t] = np.sqrt(sigma2[:, t]) * z[:, t]

    return r
