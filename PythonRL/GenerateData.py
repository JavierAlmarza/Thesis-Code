import numpy as np
import torch as torch

def kalman_stats(alpha, beta, tau_eta, tau_eps):
    R = tau_eta**2       # observation noise variance
    Q = tau_eps**2       # process noise variance

    # Riccati equation:
    # beta^2 P^2 + (R*(1-alpha^2) - Q*beta^2) P - Q*R = 0
    A = beta**2
    B = R*(1 - alpha**2) - Q*beta**2
    C = -Q*R

    P = (-B + np.sqrt(B*B - 4*A*C)) / (2*A)   # positive root

    # stationary variance of Z_t
    sigma2_Z = Q / (1 - alpha**2)

    # quantities of interest
    pred_power = beta**2 * (sigma2_Z - P)          # E|X_{t|t-1}|^2
    var_X = beta**2 * sigma2_Z + R                 # Var(X_t)
    K = (P * beta) / (beta**2 * P + R)             # Kalman gain
    coeff = (1-K*beta)*alpha

    return pred_power, var_X, K, coeff


def kalman_predictors(X, alpha, beta, tau_eta, tau_eps):
    X = np.asarray(X, dtype=float).reshape(-1)
    T = len(X)

    R = float(tau_eta)**2
    Q = float(tau_eps)**2
    alpha = float(alpha)
    beta = float(beta)

    # initialize with stationary mean and variance
    Z_hat = 0.0
    P = Q / (1 - alpha**2)

    preds = np.zeros(T, dtype=float)

    for t in range(T):
        # scalar, safe
        preds[t] = beta * Z_hat

        y = X[t] - preds[t]

        S = beta*beta * P + R
        K = (P * beta) / S

        Z_hat = Z_hat + K*y
        P = (1 - K*beta) * P

        # time update
        Z_hat = alpha * Z_hat
        P = alpha*alpha * P + Q

    return preds

def generate_kalman_data(
        T: int,
        alpha: float,
        beta: float,
        tau_eta: float,
        tau_epsilon: float,
        burn_in: int = 500
    ):

    # stationary variance of Z_t when |alpha| < 1
    if abs(alpha) < 1:
        var_Z = tau_epsilon**2 / (1 - alpha**2)
        Z0 = np.random.randn() * np.sqrt(var_Z)
    else:
        # non-stationary case: just start at zero
        Z0 = 0.0

    total = T + burn_in
    Z = np.zeros(total)
    X = np.zeros(total)

    Z[0] = Z0

    for t in range(total - 1):
        eps = tau_epsilon * np.random.randn()
        Z[t+1] = alpha * Z[t] + eps
        X[t]   = beta * Z[t] + tau_eta * np.random.randn()

    X[total-1] = beta * Z[total-1] + tau_eta * np.random.randn()
    Z = Z[burn_in:]
    X = X[burn_in:]

    # shape as requested: X is (1, T)
    return X.reshape(1, T), Z


def generate_garch_data(T, omega, alpha, beta, burn_in=500):
    
    total_T = T + burn_in

    X = np.zeros(total_T)
    h = np.zeros(total_T)
    epsi = np.zeros(total_T)

    # unconditional variance for initialization
    h[0] = omega / (1 - alpha - beta)

    for t in range(1, total_T):
        h[t] = omega + alpha * X[t-1]**2 + beta * h[t-1]
        eps = np.random.randn()
        X[t] = np.sqrt(h[t]) * eps
        epsi[t] = eps

    # drop burn-in and reshape
    return X[burn_in:].reshape(1, T), h[burn_in:], epsi[burn_in:]

def simulate_stochastic_volatility(T, mu, phi, sigma_eta, obs_noise=1):
    burn_in = 200
    T = burn_in + T
    h = np.zeros(T)
    y = np.zeros(T)
    x = np.zeros(T)
    sigma_eps = 1
    eta = np.random.normal(0, sigma_eta, T)
    eps = np.random.normal(0, sigma_eps, T)
    
    h[0] = mu + eta[0]
    y[0] = np.exp(h[0] / 2) * eps[0]
    x[0] = np.log(y[0]**2)+1.27036
    # Simulate the process forward
    for t in range(1, T):
        # Latent AR(1) process for log-volatility
        h[t] = mu + phi * (h[t-1] - mu) + eta[t]
        
        # Observed process (returns)
        y[t] = np.exp(h[t] / 2) * eps[t]

    
    x = h + obs_noise * (np.log(eps**2)- np.log(eps**2).mean())/np.log(eps**2).std()    
    y = y[burn_in:]
    h = h[burn_in:]
    x = x[burn_in:]
        
    return x, h
    

