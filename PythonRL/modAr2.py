import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# assume run_recursion is already defined as in previous answer

def compute_rho2_surface(a_vals, b_vals, Lag = 2, L=5, K_max=5000, eta=1e-6):
    rho2 = np.full((len(a_vals), len(b_vals)), np.nan)
    assert Lag<L+1
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            try:
                res = run_recursion(a, b, L=L, K_max=K_max, eta=eta)
                rho2[i, j] = res['rhos_by_lag'][Lag][-1]
            except Exception:
                # not stationary or failed numerically
                rho2[i, j] = np.nan
    return rho2

def plot_rho2_surface(Lag = 2):
    # parameter grid
    a_vals = np.linspace(-1.5, 1.5, 40)
    b_vals = np.linspace(-0.9, 0.9, 40)
    A, B = np.meshgrid(a_vals, b_vals, indexing='ij')

    # compute surface
    rho2_surface = compute_rho2_surface(a_vals, b_vals, Lag= Lag)

    # plot
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(A, B, rho2_surface, cmap='viridis', edgecolor='none')

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel(r'$\rho^{(2)}_{k_{final}}$')
    ax.set_title('Final lag-'+str(Lag)+' autocorrelation vs (a,b)')
    fig.colorbar(surf, shrink=0.6, aspect=10)
    plt.show()


def check_ar2_stationary(phi1, phi2):
    # roots of 1 - phi1 z - phi2 z^2 = 0  -> phi2 z^2 + phi1 z - 1 = 0
    coeffs = [phi2, phi1, -1]
    roots = np.roots(coeffs)
    return np.all(np.abs(roots) > 1), roots

def initial_autocov_ar2(phi1, phi2, L=7):
    # Solve 2x2 system:
    A = np.array([
        [1 - phi2**2, -phi1*(1+phi2)],
        [-phi1,       1 - phi2]
    ], dtype=float)
    b = np.array([1.0, 0.0], dtype=float)  # noise var = 1
    gamma0, gamma1 = np.linalg.solve(A, b)
    gammas = np.zeros(L+1, dtype=float)
    gammas[0] = gamma0
    gammas[1] = gamma1
    for k in range(2, L+1):
        gammas[k] = phi1*gammas[k-1] + phi2*gammas[k-2]
    return gammas  # gammas[h] = gamma_h, for h=0..L

def run_recursion(a, b, L=5, K_max=100000, eta=1e-6, verbose=False):
    phi1 = a
    phi2 = b
    stationary, roots = check_ar2_stationary(phi1, phi2)
    if not stationary:
        raise ValueError(f"AR(2) not stationary: roots = {roots}")

    # compute initial gammas up to L+2 (so we can update up to L+1 safely)
    gammas = initial_autocov_ar2(phi1, phi2, L=L+2)  # length L+3 elements: 0..L+2

    rhos = []
    rhos_by_lag = {i: [] for i in range(1, L+1)}

    for k in range(int(K_max)):
        gamma0 = gammas[0]
        if not (gamma0 > 0 and np.isfinite(gamma0)):
            return {
                'phi': (phi1, phi2),
                'gamma_last': gammas.copy(),
                'rhos': rhos,
                'rhos_by_lag': rhos_by_lag,
                'k_final': k,
                'stopped_by': 'gamma0_nonpos_or_nan'
            }

        rho_k = gammas[1] / gamma0
        rhos.append(rho_k)
        for i in range(1, L+1):
            rhos_by_lag[i].append(gammas[i] / gamma0)

        if abs(rho_k) < eta:
            if verbose:
                print(f"Stopped at k={k}, rho_k={rho_k:.3e} < eta")
            return {
                'phi': (phi1, phi2),
                'gamma_last': gammas.copy(),
                'rhos': rhos,
                'rhos_by_lag': rhos_by_lag,
                'k_final': k,
                'stopped_by': 'eta'
            }

        # update gammas up to index L+2 using transform formula
        new_g = np.zeros_like(gammas)
        for h in range(len(gammas)):
            gamma_h = gammas[h]
            gamma_hp1 = gammas[h+1] if (h+1) < len(gammas) else (phi1*gammas[-1] + phi2*gammas[-2])
            gamma_hm1 = gammas[h-1] if h-1 >= 0 else gammas[1]  # gamma_{-1}=gamma_1
            new_g[h] = (1.0 + rho_k**2)*gamma_h - rho_k*(gamma_hp1 + gamma_hm1)

        gammas = new_g

    # reached K_max
    return {
        'phi': (phi1, phi2),
        'gamma_last': gammas.copy(),
        'rhos': rhos,
        'rhos_by_lag': rhos_by_lag,
        'k_final': int(K_max),
        'stopped_by': 'K_max'
    }

# Example:
# res = run_recursion(a=0.3, b=0.1, L=5, K_max=100000, eta=1e-6, verbose=True)
# print(res['k_final'], res['rhos'][:10])
