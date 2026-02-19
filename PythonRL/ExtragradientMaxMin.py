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

# -------- example training loop with alternation ----------
T = 2000
k = 0
Xsim, Ztrue = gd.generate_kalman_data(T,0.8,1,0.4,0.4)
X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)

model_theta = FTheta()
model_phi = Gphi()

opt_theta = optim.Adam(model_theta.parameters(), lr=2e-3)
opt_phi = optim.Adam(model_phi.parameters(), lr=1e-3)

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

for outer in range(n_outer):

    # compute current Z and center it
    Z = compute_Z_sequence(model_phi, X)
    Z_tilde = Z - Z.mean()

    # (optional) do several inner theta descent steps before EG if you want:
    # I kept n_inner behavior: perform n_inner plain theta descent steps (with Z fixed)
    # before performing the extragradient steps. If you'd rather do EG every iter,
    # set n_inner = 0 and rely solely on the EG update below.
    
    for inner in range(n_inner):
        # plain theta descent (Z held constant)
        opt_phi.zero_grad()
        Z_for_theta = Z_tilde.detach()
        y_pred = compute_predictions_theta(model_theta, X, k)
        mse_loss = ((y_pred[t0:] - X[t0:])**2).mean()
        coupling_term = (Z_for_theta[mask] * y_pred[mask]).mean()
        #coupling_term = ((Z_for_theta[mask] * y_pred[mask])**2).mean()
        loss_theta = mse_loss + lambda_coupling * coupling_term
        loss_theta.backward()
        torch.nn.utils.clip_grad_norm_(model_theta.parameters(), clip_val)
        # apply manual step using lr_theta
        with torch.no_grad():
            for p in model_theta.parameters():
                if p.grad is None:
                    continue
                p.data = p.data - lr_theta * p.grad.data
        # zero grads after manual step
        for p in model_theta.parameters():
            p.grad = None

    for winner in range(n_maxim):
        # plain theta descent (Z held constant)
        opt_phi.zero_grad()
        Z = compute_Z_sequence(model_phi, X)    # now Z depends on phi
        Z_tilde = Z - Z.mean()
        y_pred_fixed = compute_predictions_theta(model_theta, X, k).detach()
        
        coupling_mean = (Z_tilde[mask] * y_pred_fixed[mask]).mean()
        loss_phi = - lambda_coupling * coupling_mean   # minimize negative => maximize coupling

        
        loss_phi.backward()
        torch.nn.utils.clip_grad_norm_(model_phi.parameters(), clip_val)
        # apply manual step using lr_phi
        with torch.no_grad():
            for p in model_phi.parameters():
                if p.grad is None:
                    continue
                p.data = p.data - lr_phi * p.grad.data
        # zero grads after manual step
        for p in model_phi.parameters():
            p.grad = None


    # ---------- Extragradient main step ----------
    # 1) compute gradients at current point (theta, phi)

    # theta grad (Z treated as constant)
    Z_for_theta = Z_tilde.detach()
    y_pred = compute_predictions_theta(model_theta, X, k)
    mse_loss = ((y_pred[t0:] - X[t0:])**2).mean()
    #coupling_term = ((Z_for_theta[mask] * y_pred[mask])**2).mean()
    coupling_term = (Z_for_theta[mask] * y_pred[mask]).mean()
    loss_theta = mse_loss + lambda_coupling * coupling_term

    # zero grads for theta
    for p in model_theta.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    loss_theta.backward()
    grads_theta = [p.grad.clone() if p.grad is not None else None for p in model_theta.parameters()]

    #2) phi-loss grad at current (phi,theta) (use y_pred detached so phi does not backprop into theta)

    y_pred_fixed = y_pred.detach()
    Z_recomputed = compute_Z_sequence(model_phi, X)   # depends on phi
    Z_tilde_re = Z_recomputed - Z_recomputed.mean()
    #coupling_mean = ((Z_tilde_re[mask] * y_pred_fixed[mask])**2).mean()
    coupling_mean = (Z_tilde_re[mask] * y_pred_fixed[mask]).mean()
    loss_phi = - lambda_coupling * coupling_mean
    for p in model_phi.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    loss_phi.backward()
    grads_phi = [p.grad.clone() if p.grad is not None else None for p in model_phi.parameters()]

    # 3) build extrapolated (half-step) models by deepcopy + param updates
    theta_ex = copy.deepcopy(model_theta)
    phi_ex = copy.deepcopy(model_phi)

    # apply half-step: theta_ex = theta - lr_theta * grads_theta
    with torch.no_grad():
        for p_ex, g in zip(theta_ex.parameters(), grads_theta):
            if g is None:
                continue
            p_ex.data = p_ex.data - lr_theta * g

        
        for p_ex, g in zip(phi_ex.parameters(), grads_phi):
            if g is None:
                continue
            p_ex.data = p_ex.data - lr_phi * g

    # 4) compute gradients at extrapolated point
    # compute Z_ex using phi_ex
    Z_ex = compute_Z_sequence(phi_ex, X)
    Z_ex_tilde = Z_ex - Z_ex.mean()

    # theta gradient at theta_ex (treat Z_ex as constant)
    y_pred_ex = compute_predictions_theta(theta_ex, X, k)
    mse_loss_ex = ((y_pred_ex[t0:] - X[t0:])**2).mean()
    #coupling_ex = ((Z_ex_tilde.detach()[mask] * y_pred_ex[mask])**2).mean()
    coupling_ex = (Z_ex_tilde.detach()[mask] * y_pred_ex[mask]).mean()
    loss_theta_ex = mse_loss_ex + lambda_coupling * coupling_ex

    # zero grads on theta_ex and backprop on extrapolated theta
    for p in theta_ex.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    loss_theta_ex.backward()
    grads_theta_ex = [p.grad.clone() if p.grad is not None else None for p in theta_ex.parameters()]

    # phi gradient at phi_ex (use y_pred_ex detached)
    for p in phi_ex.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    y_pred_ex_fixed = y_pred_ex.detach()
    Z_ex_re = compute_Z_sequence(phi_ex, X)
    Z_ex_re_tilde = Z_ex_re - Z_ex_re.mean()
    #coupling_phi_ex = ((Z_ex_re_tilde[mask] * y_pred_ex_fixed[mask])**2).mean()
    coupling_phi_ex = (Z_ex_re_tilde[mask] * y_pred_ex_fixed[mask]).mean()
    loss_phi_ex = - lambda_coupling * coupling_phi_ex
    loss_phi_ex.backward()
    grads_phi_ex = [p.grad.clone() if p.grad is not None else None for p in phi_ex.parameters()]

    # 5) update original models using extrapolated gradients (full EG step)
    # theta: descent
    with torch.no_grad():
        # apply clipping on grads before update (clip elementwise norm via global norm)
        # compute global norm for theta grads
        total_sq = 0.0
        for g in grads_theta_ex:
            if g is None:
                continue
            total_sq += (g.detach()**2).sum().item()
        total_norm = total_sq**0.5 if total_sq > 0 else 0.0
        clip_coef = min(1.0, clip_val / (total_norm + 1e-12)) if total_norm > 0 else 1.0

        for p, g in zip(model_theta.parameters(), grads_theta_ex):
            if g is None:
                continue
            p.data = p.data - lr_theta * (g.detach() * clip_coef)

        # phi: ascent
        total_sq = 0.0
        for g in grads_phi_ex:
            if g is None:
                continue
            total_sq += (g.detach()**2).sum().item()
        total_norm = total_sq**0.5 if total_sq > 0 else 0.0
        clip_coef = min(1.0, clip_val / (total_norm + 1e-12)) if total_norm > 0 else 1.0

        for p, g in zip(model_phi.parameters(), grads_phi_ex):
            if g is None:
                continue
            p.data = p.data - lr_phi * (g.detach() * clip_coef)

    # zero grads on originals
    for p in model_theta.parameters():
        p.grad = None
    for p in model_phi.parameters():
        p.grad = None

    # recompute diagnostics for logging
    Z = compute_Z_sequence(model_phi, X)
    P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.8,1,0.4,0.4)
    Z_tilde = Z - Z.mean()
    y_pred = compute_predictions_theta(model_theta, X, k)
    mse_loss = ((y_pred[t0:] - X[t0:])**2).mean()
    #coupling_term = ((Z_tilde[mask] * y_pred[mask])**2).mean()
    coupling_term = (Z_tilde[mask] * y_pred[mask]).mean()
    Diff = ((X[t0:].detach().numpy().reshape(-1) - y_pred[t0:].detach().numpy().reshape(-1) - P[t0:])**2).mean()

    # the prints you used
    if outer % 5 == 0:
        print(f"outer {outer:4d}  mse={mse_loss.item():.6f}  coupling={coupling_term.item():.6f}")
        print('Z sd is ', Z_tilde[t0:].detach().cpu().numpy().std(),', |Y-Pred|^2 is ',Diff.item())
        print('Y mean is ', y_pred[t0:].detach().cpu().numpy().mean(), ', Y sd is ', y_pred[t0:].detach().cpu().numpy().std())

# Done
