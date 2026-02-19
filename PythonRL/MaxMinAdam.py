import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import GenerateData as gd

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
        self.model = MLP([dx + dy, 13, 13, dy])
    def forward(self, x_in):
        return self.model(x_in)

# G_phi: recursion network for Z_{t} = G_phi(Z_{t-1}, X_{t-1})
class Gphi(nn.Module):
    def __init__(self, zdim=1, xdim=1):
        super().__init__()
        self.net = MLP([zdim + xdim, 14, 14, zdim])
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
    
    return y


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

# -------- example training loop with alternation ----------
T = 2000
k = 0
t0 = 500
Xsim, Ztrue = gd.generate_kalman_data(T,0.8,1,0.4,0.4)
X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)

model_theta = FTheta()
model_phi = Gphi()

opt_theta = optim.Adam(model_theta.parameters(), lr=3e-5)
opt_phi = optim.Adam(model_phi.parameters(), lr=3e-5)

lambda_coupling = 20.0
n_outer = 100
n_inner = 15  
n_maximizer = 15
P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.8,1,0.4,0.4)

for outer in range(n_outer):
    # 1) compute current Z sequence from current phi
    Z = compute_Z_sequence(model_phi, X)      # shape (T,)
    Z_tilde = Z - Z.mean()                    # center
    # mask out first k indices if you want coupling only for t>=k
    mask = torch.arange(T) >= t0

    # --- inner loop: update theta (minimize MSE + lambda * coupling) ---
    for inner in range(n_inner):
        opt_theta.zero_grad()

        # WHEN UPDATING THETA: treat Z as a constant (no grad into phi)
        Z_for_theta = Z_tilde.detach()

        y_pred = compute_predictions_theta(model_theta, X, k)   # shape (T,)

        mse_loss = ((y_pred[t0:] - X[t0:])**2).mean()
        coupling_term = ((Z_for_theta[mask] * y_pred[mask]).mean())  # average coupling
        loss_theta = mse_loss + lambda_coupling * coupling_term

        loss_theta.backward()
        torch.nn.utils.clip_grad_norm_(model_theta.parameters(), 5.0)
        opt_theta.step()

    # --- outer step: update phi to MAXIMIZE coupling ---
    # Strategy A (simple): treat y_pred as constant and update phi to increase coupling
    for maxim in range(n_maximizer):
        opt_phi.zero_grad()

        # recompute Z (or reuse) - we should recompute so phi grads flow through the Z generation
        Z = compute_Z_sequence(model_phi, X)    # now Z depends on phi
        Z_tilde = Z - Z.mean()

        # Use y_pred as fixed target: detach so gradient doesn't flow into theta
        y_pred_fixed = compute_predictions_theta(model_theta, X, k).detach()

        coupling_mean = ((Z_tilde[mask] * y_pred_fixed[mask]).mean())
        loss_phi = - lambda_coupling * coupling_mean   # minimize negative => maximize coupling

        loss_phi.backward()
        torch.nn.utils.clip_grad_norm_(model_phi.parameters(), 5.0)
        opt_phi.step()

    if outer % 5 == 0:
        print(f"outer {outer:4d}  mse={mse_loss.item():.6f}  coupling={coupling_term.item():.6f}")
        Diff = np.sqrt(((X[t0:].detach().numpy().reshape(-1) - y_pred[t0:].detach().numpy().reshape(-1) - P[t0:])**2).mean())
        print('Z sd is ', Z_tilde[t0:].detach().cpu().numpy().std(),', |Y-Pred| is ',Diff.item())
        print('Y mean is ',y_pred_fixed[t0:].numpy().mean(),', Y sd is ',y_pred_fixed[t0:].numpy().std())

# Done
