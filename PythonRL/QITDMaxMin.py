import copy
import numpy as np
import torch
import torch.nn as nn
import GenerateData as gd

torch.set_default_dtype(torch.float32)

# ============================================================
# Networks
# ============================================================

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FTheta(nn.Module):
    def __init__(self, dx=1, dy=1):
        super().__init__()
        self.model = MLP([dx + dy, 13, 13, dy])

    def forward(self, x):
        return self.model(x)


class Gphi(nn.Module):
    def __init__(self, zdim=1, xdim=1):
        super().__init__()
        self.net = MLP([zdim + xdim, 14, 14, zdim])

    def forward(self, z_prev, x_prev):
        inp = torch.cat([z_prev, x_prev], dim=-1)
        return self.net(inp)


# ============================================================
# Time recursions
# ============================================================

def compute_predictions_theta(model, X, k):
    T = X.shape[0]
    y = torch.zeros(T, device=X.device)
    y_prev = torch.zeros(1, device=X.device)

    for t in range(k, T):
        inp = torch.stack([X[t - k], y_prev.squeeze()])
        y_t = model(inp).squeeze()
        y[t] = y_t
        y_prev = y_t

    return y


def compute_Z_sequence(G, X):
    T = X.shape[0]
    Z = torch.zeros(T, device=X.device)

    # z_prev must be (1,1), not (1,)
    z_prev = torch.zeros(1, 1, device=X.device)

    for t in range(1, T):
        x_prev = X[t - 1].view(1, 1)     # (1,1)
        z_prev = G(z_prev, x_prev)       # (1,1)
        Z[t] = z_prev.squeeze()          # scalar

    return Z

# ============================================================
# Loss
# ============================================================

def kalman_loss(theta_models, phi_models, X):
    Ftheta = theta_models[0]
    Gphi   = phi_models[0]
    tb = 500

    k = 0
    y = compute_predictions_theta(Ftheta, X, k)
    Z = compute_Z_sequence(Gphi, X)

    Z_tilde = Z - Z.mean()
   

    mse = ((y - X) ** 2).mean()
    coupling = (Z_tilde * y).mean()

    return mse + lambda_coupling * coupling


# ============================================================
# Parameter vector utilities
# ============================================================

def flatten_params(params):
    return torch.cat([p.reshape(-1) for p in params])


def unflatten_params(vec, params_template):
    out = []
    idx = 0
    for p in params_template:
        n = p.numel()
        out.append(vec[idx:idx+n].view_as(p))
        idx += n
    return out


def make_signature(theta_params, phi_params, device):
    n_theta = sum(p.numel() for p in theta_params)
    n_phi   = sum(p.numel() for p in phi_params)
    J = torch.eye(n_theta + n_phi, device=device)
    J[n_theta:, n_theta:] *= -1
    return J


def compute_G(loss, theta_params, phi_params):
    grads = torch.autograd.grad(
        loss,
        theta_params + phi_params,
        retain_graph=False,
        create_graph=False
    )
    return flatten_params(grads)


# ============================================================
# QITD step
# ============================================================

def qitd_step(
    loss_fn,
    theta_models,
    phi_models,
    theta_params,
    phi_params,
    B,
    J,
    X,
    eta,
    eta_prev,
    gamma=0.8,
    eps=1e-3,
    beta=0.1,
    eta_max=3e-2
):
    # ---- current loss and gradient ----
    L_n = loss_fn(theta_models, phi_models, X)
    G_n = compute_G(L_n, theta_params, phi_params)
    u_n = flatten_params(theta_params + phi_params)

    # --------------------------------------------------------
    # double-inequality line search
    # L(theta_{n+1},phi_n) <= L(theta_{n+1},phi_{n+1}) <= L(theta_n,phi_{n+1})
    # --------------------------------------------------------

    def eval_losses(u_candidate):
        new_params = unflatten_params(u_candidate, theta_params + phi_params)
        split = len(theta_params)
        theta_new = new_params[:split]
        phi_new   = new_params[split:]

        # θ_{n+1}, φ_n
        for p, q in zip(theta_params, theta_new):
            p_tmp = p.data.clone()
            p.data.copy_(q)
        L_theta_new_phi_n = loss_fn(theta_models, phi_models, X)
        for p, q in zip(theta_params, theta_new):
            p.data.copy_(p_tmp)

        # θ_n, φ_{n+1}
        for p, q in zip(phi_params, phi_new):
            p_tmp2 = p.data.clone()
            p.data.copy_(q)
        L_theta_n_phi_new = loss_fn(theta_models, phi_models, X)
        for p, q in zip(phi_params, phi_new):
            p.data.copy_(p_tmp2)

        # θ_{n+1}, φ_{n+1}
        for p, q in zip(theta_params + phi_params, new_params):
            p.data.copy_(q)
        L_both_new = loss_fn(theta_models, phi_models, X)

        return L_theta_new_phi_n, L_both_new, L_theta_n_phi_new

    eta_try = eta

    while True:
        u_try = u_n - eta_try * (B @ G_n)
        L1, Lmid, L2 = eval_losses(u_try)

        if (L1 <= Lmid <= L2) or (eta_try <= eps * eta_prev):
            break

        eta_try *= gamma
    eta_try = max(eta_try, 1e-4)
    eta = min((1 + beta) * eta_try, eta_max) if (L1 <= Lmid <= L2) else eta_try

    # ---- commit step ----
    u_np1 = u_n - eta * (B @ G_n)
    new_params = unflatten_params(u_np1, theta_params + phi_params)
    for p, q in zip(theta_params + phi_params, new_params):
        p.data.copy_(q)

    # ---- secant update ----
    L_np1 = loss_fn(theta_models, phi_models, X)
    G_np1 = compute_G(L_np1, theta_params, phi_params)

    s = J @ G_np1 - B @ G_n
    denom = torch.dot(G_n, s)

    if torch.abs(denom) > 1e-12:
        alpha = torch.dot(s, s) / denom
        alpha = torch.sign(alpha) * min(torch.abs(alpha), torch.tensor(1.0))
        B = B + (alpha / torch.dot(s, s)) * torch.outer(s, s)

    return B, eta


# ============================================================
# Experiment
# ============================================================

T = 2000
tb = 500
lambda_coupling = 30.0
n_updates = 800
M = 500
k=0

Xsim, _ = gd.generate_kalman_data(T, 0.8, 1, 0.4, 0.4)
Xsim = Xsim.reshape(-1)
X = torch.tensor(Xsim, dtype=torch.float32)

X_train = X[tb:]   # burn-in

model_theta = FTheta()
model_phi = Gphi()

theta_params = list(model_theta.parameters())
phi_params   = list(model_phi.parameters())

device = X.device
J = make_signature(theta_params, phi_params, device)
B = J.clone()

eta = 5e-3
eta_prev = eta

for n in range(n_updates):
    t0 = torch.randint(0, len(X_train) - M, (1,)).item()
    batch = X_train[t0:t0 + M]

    B, eta = qitd_step(
        kalman_loss,
        [model_theta],
        [model_phi],
        theta_params,
        phi_params,
        B,
        J,
        batch,
        eta,
        eta_prev
    )

    eta_prev = eta
    P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.8,1,0.4,0.4)
    y_pred = compute_predictions_theta(model_theta, X, k)   # shape (T,)
    Z = compute_Z_sequence(model_phi, X)      # shape (T,)
    Z_tilde = Z - Z.mean() 

    if n % 20 == 0:
        with torch.no_grad():
            L = kalman_loss([model_theta], [model_phi], X_train)
            print(f"{n:4d}  loss={L.item():.6f}  |Y-D|^2={((y_pred[tb:].detach().numpy().reshape(-1) - P[tb:])**2).mean():6f}  eta={eta:6f}  coupling={ (Z_tilde[tb:] * y_pred[tb:]).mean():6f}")
