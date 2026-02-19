import copy
import torch
import torch.nn as nn
import torch.optim as optim
import GenerateData as gd
import numpy as np

# ---- your existing nets ----
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
    def __init__(self, dx: int = 1, dy: int = 1):
        super().__init__()
        # if FTheta should also take Z, change the input size accordingly
        self.model = MLP([dx + dy, 10, 8, 10, dy])
    def forward(self, x_in):
        return self.model(x_in)

# G_phi: recursion network for Z_{t} = G_phi(Z_{t-1}, X_{t-1})
class Gphi(nn.Module):
    def __init__(self, zdim=1, xdim=1):
        super().__init__()
        self.net = MLP([zdim + xdim, 10, 8, 10, zdim])
    def forward(self, z_prev, x_prev):
        inp = torch.cat([z_prev, x_prev], dim=-1)
        return self.net(inp)

# Prediction routine (variant A: y does NOT depend on Z)
def compute_predictions_theta(model, X, k):
    T = X.shape[0]
    device = X.device
    y = torch.zeros(T, dtype=X.dtype, device=device)

    # keep y_prev as differentiable tensor
    y_prev = torch.tensor(0.0, dtype=X.dtype, device=device)

    for t in range(k, T):
        inp = torch.stack([X[t - k], y_prev])  # <-- NO .item(), NO new leaf
        y_t = model(inp)
        y_val = y_t.squeeze()
        y[t] = y_val

        # keep gradient flow
        y_prev = y_val
    
    return X-y


# Compute Z sequence given G_phi and X.
# Z_0 = 0, for t >= 1: Z_t = G_phi(Z_{t-1}, X_{t-1})
def compute_Z_sequence(G, X):
    T = X.shape[0]
    device = X.device
    zdim = 1
    Z = torch.zeros(T, zdim, dtype=X.dtype, device=device)  # shape (T, zdim)
    z_prev = torch.zeros(1, dtype=X.dtype, device=device).reshape(1)  # shape (zdim,)
    for t in range(1, T):
        x_prev = X[t-1].reshape(1)
        z_prev = G(z_prev.unsqueeze(0), x_prev.unsqueeze(0))  # make batch dim
        z_prev = z_prev.squeeze(0)  # shape zdim
        Z[t] = z_prev
    return Z.squeeze(-1)  # return shape (T,)

def prox_penalty_sq(model, old_params):
    s = 0.0
    for p, p0 in zip(model.parameters(), old_params):
        s = s + (p - p0).pow(2).sum()
    return s


# -------- example training loop with alternation ----------
T = 2000
k = 0
Xsim, Ztrue = gd.generate_kalman_data(T,0.8,1,0.4,0.4)
X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)

model_theta = FTheta()
model_phi = Gphi()


opt_theta = optim.Adam(model_theta.parameters(), lr=2e-3)
opt_phi = optim.Adam(model_phi.parameters(), lr=1e-3)

# Mirror–Prox / implicit parameters
eta_theta = opt_theta.param_groups[0]['lr']   # descent step
tau_phi   = opt_phi.param_groups[0]['lr']     # ascent step

n_phi_prox_steps = 20        # implicit φ solve accuracy
phi_prox_lr = 5e-4           # Adam lr inside prox
clip_val = 5.0


lambda_coupling = 50.0
n_outer = 3000
n_inner = 0   
n_maxim = 2

# Extragradient replacement training loop
lr_theta = opt_theta.param_groups[0]['lr']
lr_phi   = opt_phi.param_groups[0]['lr']
clip_val = 5.0

T = X.shape[0]
t0=500
mask = torch.arange(T) >= t0
P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.8,1,0.4,0.4)

for outer in range(n_outer):

    # ==========================================================
    # 0) save current φ for proximal center
    # ==========================================================
    phi_old = [p.detach().clone() for p in model_phi.parameters()]

    # ==========================================================
    # 1) implicit / proximal φ half-step
    # ==========================================================
    opt_phi_prox = optim.Adam(model_phi.parameters(), lr=phi_prox_lr)

    y_fixed = compute_predictions_theta(model_theta, X, k).detach()

    for _ in range(n_phi_prox_steps):
        opt_phi_prox.zero_grad()

        Z = compute_Z_sequence(model_phi, X)
        Z_tilde = Z - Z.mean()

        coupling = (Z_tilde[mask] * y_fixed[mask]).mean()
        prox = prox_penalty_sq(model_phi, phi_old)

        # maximize coupling  <=> minimize negative + prox
        loss_phi_imp = -lambda_coupling * coupling + (0.5 / tau_phi) * prox
        loss_phi_imp.backward()
        torch.nn.utils.clip_grad_norm_(model_phi.parameters(), clip_val)
        opt_phi_prox.step()

    # φ_half
    phi_half = copy.deepcopy(model_phi)

    # ==========================================================
    # 2) θ gradient at (θ, φ_half)
    # ==========================================================
    Z_half = compute_Z_sequence(phi_half, X)
    Z_half_tilde = (Z_half - Z_half.mean()).detach()

    y = compute_predictions_theta(model_theta, X, k)
    mse = ((y[t0:] - X[t0:])**2).mean()
    coupling = (Z_half_tilde[mask] * y[mask]).mean()
    loss_theta = mse + lambda_coupling * coupling

    for p in model_theta.parameters():
        p.grad = None
    loss_theta.backward()
    grads_theta = [p.grad.clone() for p in model_theta.parameters()]

    # ==========================================================
    # 3) extrapolated θ
    # ==========================================================
    theta_ex = copy.deepcopy(model_theta)
    with torch.no_grad():
        for p, g in zip(theta_ex.parameters(), grads_theta):
            p -= eta_theta * g

    # ==========================================================
    # 4) gradients at extrapolated point
    # ==========================================================
    # θ-gradient
    y_ex = compute_predictions_theta(theta_ex, X, k)
    mse_ex = ((y_ex[t0:] - X[t0:])**2).mean()
    coupling_ex = (Z_half_tilde[mask] * y_ex[mask]).mean()
    loss_theta_ex = mse_ex + lambda_coupling * coupling_ex

    for p in theta_ex.parameters():
        p.grad = None
    loss_theta_ex.backward()
    grads_theta_ex = [p.grad.clone() for p in theta_ex.parameters()]

    # φ-gradient (ascent) at (θ_ex, φ_half)
    for p in phi_half.parameters():
        p.grad = None

    Z_ex = compute_Z_sequence(phi_half, X)
    Z_ex_tilde = Z_ex - Z_ex.mean()
    coupling_phi = (Z_ex_tilde[mask] * y_ex.detach()[mask]).mean()
    loss_phi_ex = -lambda_coupling * coupling_phi
    loss_phi_ex.backward()
    grads_phi_ex = [p.grad.clone() for p in phi_half.parameters()]

    # ==========================================================
    # 5) final Mirror–Prox updates
    # ==========================================================
    with torch.no_grad():
        for p, g in zip(model_theta.parameters(), grads_theta_ex):
            p -= eta_theta * g

        for p, g in zip(model_phi.parameters(), grads_phi_ex):
            p += tau_phi * g    # ascent

    # ==========================================================
    # diagnostics
    # ==========================================================
    Z = compute_Z_sequence(model_phi, X)
    Z_tilde = Z - Z.mean()
    y = compute_predictions_theta(model_theta, X, k)
    mse = ((y[t0:] - X[t0:])**2).mean()
    coupling = (Z_tilde[mask] * y.detach()[mask]).mean()

    if outer % 5 == 0:
        Diff = np.sqrt(((X[t0:].detach().numpy().reshape(-1) - y[t0:].detach().numpy().reshape(-1) - P[t0:])**2).mean())
        print(f"outer {outer:4d}  mse={mse.item():.6f}  coupling={coupling.item():.6f} |Y-Pred|={Diff.item():.6f}")
        print("Z sd:", Z_tilde[t0:].std().item(),
              "Y sd:", y[t0:].std().item(),
              "Y mean:", y[t0:].mean().item())

# Done


