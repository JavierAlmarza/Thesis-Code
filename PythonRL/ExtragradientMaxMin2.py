import copy
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
T = 1500
k = 0
Xsim, Ztrue = gd.generate_kalman_data(T,0.8,1,0.4,0.4)
X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)

model_theta = FTheta()
model_phi = Gphi()

opt_theta = optim.Adam(model_theta.parameters(), lr=1e-3)
opt_phi = optim.Adam(model_phi.parameters(), lr=5e-4)

lambda_coupling = 8.0
n_outer = 70  
n_maximizer = 3

# extragradient hyperparams (tune these)
lr_theta = 1e-3    # try 1e-3 -> 1e-4
lr_phi   = 5e-4    # try smaller than lr_theta or similar
n_eg_steps = 5     # number of extragradient steps per outer loop (repeat if desired)
mask = torch.arange(T) >= k

# utility: compute gradients w.r.t. parameters and collect them
def compute_grads_from_loss(model, loss):
    # zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    loss.backward()
    return [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

# utility: set params from a list of tensors
def set_params_from_list(model, tensor_list):
    for p, new_data in zip(model.parameters(), tensor_list):
        p.data = new_data.data.clone()

# main extragradient loop
for outer in range(n_outer):
    # Step A: compute current Z from phi and center it
    Z = compute_Z_sequence(model_phi, X)
    Z_tilde = Z - Z.mean()

    # (perform n_eg_steps extragradient "sub-iterations" if desired)
    for _eg in range(n_eg_steps):
        # 1) compute gradient of theta-loss at current (theta, phi)
        #    treat Z as constant (detach)
        Z_for_theta = Z_tilde.detach()
        y_pred = compute_predictions_theta(model_theta, X, k)
        mse_loss = ((y_pred[k:] - X[k:])**2).mean()
        coupling_term = (Z_for_theta[mask] * y_pred[mask]).mean()
        loss_theta = mse_loss + lambda_coupling * coupling_term

        # collect grads for theta
        for p in model_theta.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        loss_theta.backward()
        grads_theta = [p.grad.clone() for p in model_theta.parameters()]

        # 2) compute gradient of phi-loss at current (theta, phi)
        #    we will use y_pred.detach() so phi does not prop into theta
        for p in model_phi.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        y_pred_fixed = y_pred.detach()
        Z_recomputed = compute_Z_sequence(model_phi, X)  # depends on phi
        Z_tilde_re = Z_recomputed - Z_recomputed.mean()
        coupling_mean = (Z_tilde_re[mask] * y_pred_fixed[mask]).mean()
        loss_phi = - lambda_coupling * coupling_mean
        loss_phi.backward()
        grads_phi = [p.grad.clone() for p in model_phi.parameters()]

        # 3) build extrapolated models (theta_ex, phi_ex)
        theta_ex = copy.deepcopy(model_theta)
        phi_ex = copy.deepcopy(model_phi)

        # apply half-step to the copies: theta_ex = theta - lr_theta * grad_theta
        for p_ex, g in zip(theta_ex.parameters(), grads_theta):
            if g is None:
                continue
            p_ex.data = p_ex.data - lr_theta * g

        # phi is a maximizer: phi_ex = phi + lr_phi * grad_phi (ascent half-step)
        for p_ex, g in zip(phi_ex.parameters(), grads_phi):
            if g is None:
                continue
            p_ex.data = p_ex.data + lr_phi * g

        # 4) compute gradients at the extrapolated point
        # Compute Z_ex from phi_ex
        Z_ex = compute_Z_sequence(phi_ex, X)
        Z_ex_tilde = Z_ex - Z_ex.mean()

        # theta gradient at theta_ex (Z_ex detached)
        y_pred_ex = compute_predictions_theta(theta_ex, X, k)
        mse_loss_ex = ((y_pred_ex[k:] - X[k:])**2).mean()
        coupling_ex = (Z_ex_tilde.detach()[mask] * y_pred_ex[mask]).mean()
        loss_theta_ex = mse_loss_ex + lambda_coupling * coupling_ex

        # compute grads at extrapolated theta
        for p in theta_ex.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        loss_theta_ex.backward()
        grads_theta_ex = [p.grad.clone() for p in theta_ex.parameters()]

        # phi gradient at phi_ex (use y_pred_ex detached)
        for p in phi_ex.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        y_pred_ex_fixed = y_pred_ex.detach()
        Z_ex_re = compute_Z_sequence(phi_ex, X)
        Z_ex_re_tilde = Z_ex_re - Z_ex_re.mean()
        coupling_phi_ex = (Z_ex_re_tilde[mask] * y_pred_ex_fixed[mask]).mean()
        loss_phi_ex = - lambda_coupling * coupling_phi_ex
        loss_phi_ex.backward()
        grads_phi_ex = [p.grad.clone() for p in phi_ex.parameters()]

        # 5) update original models using the extrapolated gradients
        # theta: descent
        for p, g_ex in zip(model_theta.parameters(), grads_theta_ex):
            if g_ex is None:
                continue
            p.data = p.data - lr_theta * g_ex
        # phi: ascent
        for p, g_ex in zip(model_phi.parameters(), grads_phi_ex):
            if g_ex is None:
                continue
            p.data = p.data + lr_phi * g_ex

        # recompute Z (for next EG substep or outer logging)
        Z = compute_Z_sequence(model_phi, X)
        Z_tilde = Z - Z.mean()
        y_pred = compute_predictions_theta(model_theta, X, k)
        mse_loss = ((y_pred[k:] - X[k:])**2).mean()
        coupling_mean = (Z_tilde[mask] * y_pred.detach()[mask]).mean()

    # logging
    if outer % 10 == 0:
        print(f"outer {outer:4d}  mse={mse_loss.item():.6f}  coupling={coupling_mean.item():.6f}")
