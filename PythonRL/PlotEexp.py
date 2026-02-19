import numpy as np
import matplotlib.pyplot as plt

def EexpDeltaX_minus1(a1=0.3, b=0.4, rho=0.6, sigma_eta=0.2, sigma_vals=None):
    """
    Compute E[e^{ΔX} - 1] for the AR(1)-driven model:
        x_t = a1 * x_{t-1} + b * c_t + σ_η * η_t
        c_t = ρ * c_{t-1} + σ * e_t
    with η_t, e_t ~ N(0,1) iid.
    """
    if sigma_vals is None:
        sigma_vals = np.linspace(0.05, 1.0, 200)
    vals = []
    for sigma in sigma_vals:
        # Var(c)
        Var_c = sigma**2 / (1 - rho**2)
        # Cov(X_{t-1}, c_t)
        Cov_Xc = b * Var_c * rho / (1 - a1 * rho)
        # Var(X)
        Var_X = (b**2 * Var_c * (1 + a1 * rho) / (1 - a1 * rho) + sigma_eta**2) / (1 - a1**2)
        # Var(ΔX)
        Var_dX = ((1 - a1)**2 * Var_X +
                  b**2 * Var_c + sigma_eta**2 +
                  2 * (a1 - 1) * b * Cov_Xc)
        vals.append(np.exp(0.5 * Var_dX) - 1)
    return sigma_vals, np.array(vals)

# Example: reproduce your parameter values and plot
a1, b, rho, sigma_eta = 0.3, 0.4, 0.6, 0.2
sigma_vals, Evals = EexpDeltaX_minus1(a1, b, rho, sigma_eta)

plt.figure(figsize=(6,4))
plt.plot(sigma_vals, Evals, lw=2)
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$\mathbb{E}[e^{\Delta X}-1]$")
plt.title(rf"$a_1={a1},\, b={b},\, \rho={rho},\, \sigma_\eta={sigma_eta}$")
plt.grid(True)
plt.show()
