"""

Behavior changes requested by user:
- introduce Y_t with recurrence Y_{t+1}=G(X_t, Y_t). Gnet takes (x_t,y_t) of size d_x+d_y and outputs d_y. So T_out = Y_t and Gnet parameters play the role of T's parameters (they are maximized).

Implementation notes:
- All parameter updates use `with torch.no_grad()` to avoid using `.data`.
- The optimistic mirror descent waiting/actual update logic is preserved.
"""

import copy
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import GenerateData as gd
import matplotlib.pyplot as plt



# ---------------------- Model definitions ----------------------
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


class FNet(nn.Module):
    """F: (d_x + d_z) -> d_z with architecture: d_x+d_z -> 7 -> 7 -> d_z"""

    def __init__(self, dx: int, dz: int):
        super().__init__()
        self.model = MLP([dx + dz, 11, 11, dz])

    def forward(self, x, z):
        return self.model(torch.cat([x, z], dim=-1))


class LambdaNet(nn.Module):
    """lambda: d_z -> d_x with architecture: d_z -> 7 -> 7 -> d_x"""

    def __init__(self, dx: int, dz: int):
        super().__init__()
        self.model = MLP([dz, 11, 11,dx])

    def forward(self, z):
        return self.model(z)


class GNet(nn.Module):
    """G: (d_x + d_y) -> d_y with same hidden layers as FNet (7->7)."""

    def __init__(self, dx: int, dy: int):
        super().__init__()
        self.model = MLP([dx + dy, 10, 8, 10, dy])

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=-1))

class GNet2(nn.Module):
    def __init__(self, dx, dy, gamma=0.05):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.gamma = gamma

        # GRUCell: input is x_{t+1} (dim dx), hidden is y_t (dim dy)
        self.gru = nn.GRUCell(input_size=dx, hidden_size=dy)

        # Optional small linear map after GRU output (helps expressivity)
        self.out = nn.Linear(dy, dy)

        # Initialize for stability
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            else:
                nn.init.zeros_(p)

    def forward(self, x_next, y_t):
        """
        Inputs:
            x_next : (batch, dx)
            y_t    : (batch, dy)
        Output:
            y_{t+1} : (batch, dy)
        """

        h = self.gru(x_next, y_t)         # GRU hidden = nonlinear summary
        h = self.out(h)                   # small linear transformation
        y_next = y_t + self.gamma * h     # gated residual update

        return y_next


# ---------------------- Utilities ----------------------

def compute_z_sequence(F: nn.Module, X: torch.Tensor, z0: Optional[torch.Tensor] = None) -> torch.Tensor:
    dx, T = X.shape
    device = X.device
    if z0 is None:
        in_features = F.model.net[0].in_features
        dz = in_features - dx
        z_prev = torch.zeros(1, dz, device=device)
    else:
        z_prev = z0.unsqueeze(0)
        dz = z_prev.shape[-1]

    Z = torch.zeros((T, dz), device=device)

    # init x_{-1} from empirical multivariate Gaussian
    X_mean = X.mean(dim=1)
    X_centered = (X - X_mean[:, None]).t()
    Cov = (X_centered.t() @ X_centered) / max(1, X_centered.shape[0] - 1)
    Cov = Cov + 1e-6 * torch.eye(dx, device=device)
    x_prev = torch.distributions.MultivariateNormal(X_mean, Cov).sample()

    for t in range(T):
        z_next = F(x_prev.unsqueeze(0), z_prev)
        Z[t] = z_next.squeeze(0)
        x_prev = X[:, t]
        z_prev = z_next
    return Z

def compute_y_sequence(G, X: torch.Tensor, d_y: int, y0: Optional[torch.Tensor] = None):
    """
    Deterministic, fully-differentiable recursion:
        Y_{t+1} = G(X_{t+1}, Y_t)
    X: (d_x, T)
    Returns Y: (T, d_y)
    """
    dx, T = X.shape
    device = X.device

    # init hidden
    if y0 is None:
        y_prev = torch.zeros(1, d_y, device=device)
    else:
        y_prev = y0.unsqueeze(0).to(device)

    Y = torch.zeros((T, d_y), device=device)

    # first step: use X[:,0]
    x_next = X[:, 0].unsqueeze(0)   # shape (1, dx), differentiable
    y_prev = G(x_next, y_prev)      # returns (1, d_y)
    Y[0] = y_prev.squeeze(0)

    # remaining steps
    for t in range(1, T):
        x_next = X[:, t].unsqueeze(0)    # (1, dx)
        y_prev = G(x_next, y_prev)       # (1, d_y), uses previous y_prev (keeps graph)
        Y[t] = y_prev.squeeze(0)

    return Y


def compute_y_sequence2(G, X, d_y: int, y0: Optional[torch.Tensor] = None):
    """
    Compute Y_t sequence under recursion:
        Y_{t+1} = G(X_{t+1}, Y_t)
    X: (d_x, T)
    Returns Y: (T, d_y)
    """
    dx, T = X.shape
    device = X.device

    # initialize y0 if needed
    if y0 is None:
        y_prev = torch.zeros(1, d_y, device=device)
    else:
        y_prev = y0.unsqueeze(0)

    Y = torch.zeros((T, d_y), device=device)
    """

    # FIRST STEP uses X[ :, 0 ]  (i.e. X_1)
    X_mean = X.mean(dim=1)
    X_centered = (X - X_mean[:, None]).t()
    Cov = (X_centered.t() @ X_centered) / max(1, X_centered.shape[0] - 1)
    Cov = Cov + 1e-6 * torch.eye(dx, device=device)
    eps = torch.randn_like(X_mean)
    x_next = X_mean + (Cov @ eps)
    y_prev = G(x_next.unsqueeze(0), y_prev)
    """
    
    x_next = X[:, 0]                          # (dx,)
    Y_next = G(x_next.unsqueeze(0), y_prev)
    Y[0] = Y_next.squeeze(0)
    y_prev = Y_next

    # REMAINING STEPS
    for t in range(1, T):
        x_next = X[:, t]                     # (dx,)
        Y_next = G(x_next.unsqueeze(0), y_prev) 
        Y[t] = Y_next.squeeze(0)
        y_prev = Y_next

    return Y



# ---------------------- Loss and grads ----------------------

def loss_and_attach_grads(F: nn.Module,
                          Lamb: nn.Module,
                          Gnet: Optional[torch.nn.Module],
                          X: torch.Tensor,
                          batch_range: Tuple[int, int],
                          dy: Optional[int] = None,
                          do_backward: bool = True):
    """Compute loss (scalar tensor) and call backward(), leaving gradients attached to parameters.
    Returns detached loss tensor and the pure reconstruction loss (ret) as a Python float."""
    # zero grads
    modules = [F, Lamb] + ([] if Gnet is None else [Gnet])
    for m in modules:
        for p in m.parameters():
            if p.grad is not None:
                p.grad = None

    # compute sequences
    Z = compute_z_sequence(F, X)
    #Z = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.8,1,0.4,0.4)
    assert Gnet is not None and dy is not None
    Y = compute_y_sequence(Gnet, X, dy)
    
    #START HERE

#    with torch.no_grad():
#        Z_full = compute_z_sequence(F, X)        # (T, d_z)
#        Y_full = compute_y_sequence(Gnet, X, dy)   # (T, d_y)
#        X_np_t = X.t()                              # (T, d_x)


#    for t in range(5, 12):
#        print("t=", t)
#        print("  X[t]        :", X_np_t[t].cpu().numpy())
#        print("  Y[t]        :", Y_full[t].cpu().numpy())
#        if t+1 < X_np_t.shape[0]:
#            print("  X[t+1]      :", X_np_t[t+1].cpu().numpy())
#            print("  Y[t+1]      :", Y_full[t+1].cpu().numpy())
    #END HERE

    t0, t1 = batch_range
    Xb = X[:, t0:t1].t()          # (batch, d_x)                 # (batch, d_z)
    Tb = Y[t0:t1]    
    #Znum = torch.tensor(Z, dtype=torch.float32, device=X.device).view(-1, 1)
    #Zb = Znum[t0:t1]
    Zb = Z[t0:t1]
    #Lambda_all = Lamb(Znum)          # (T, d_x)
    Lambda_all = Lamb(Z)
    Lambda_mean = Lambda_all.mean(dim=0, keepdim=True)
    Lambda_tilde = Lambda_all[t0:t1] - Lambda_mean

    coeff = 5.0
    gamma = 0.0
    # separate reconstruction and coupling terms so we can return recon loss (ret)
    recon_loss_vec = ((Xb - (Tb)) ** 2).sum(dim=1)   # (batch,)
    coupling_vec = (coeff * Lambda_tilde * (Tb)).sum(dim=1)
    #coupling_vec = (0.000000 * (Tb)).sum(dim=1)
    lambda_reg_term = 0.5 * gamma * (Lambda_all**2).mean()   # scalar
    loss_vec = recon_loss_vec + coupling_vec
    #loss = loss_vec.mean()
    loss = loss_vec.mean()- lambda_reg_term   
    
    print('Y mean is ',Tb.detach().numpy().mean(),', Y std is ',Tb.detach().numpy().std())
    print('Lambda std is ',Lambda_tilde.detach().numpy().std())    
    print('recon loss is ',recon_loss_vec.detach().numpy().mean(),', coupling loss is ',coupling_vec.detach().numpy().mean())

    # pure reconstruction loss (no coupling term)
    ret = recon_loss_vec.mean().item()
    
    if do_backward:
        loss.backward()
        return loss.detach(), ret
    else:
        # do not call backward; return loss (with grad_fn) so caller can call backward()
        return loss, ret



    #loss.backward()
    return loss, ret


# ---------------------- Optimistic Mirror Descent ----------------------

def run_optimistic_md(X_np: np.ndarray,
                      d_z: int,
                      N: int = 2000,
                      eta: float = 1e-4,
                      batch_len: Optional[int] = None,
                      burn_in: int = 101,
                      tol: float = 1e-8,
                      device: str = 'cpu',
                      verbose: bool = True,
                      seed: Optional[int] = 0,
                      d_y: Optional[int] = None) -> Tuple[Optional[FNet], Optional[GNet], float, Optional[LambdaNet]]:


    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    d_x, T = X.shape
    d_y = d_x
    if batch_len is None:
        batch_len = int(min(256, max(10, T - burn_in - 1)))

    # instantiate
    
    Fnet = FNet(d_x, d_z).to(device)
    #Fnet = nn.Identity()
    Lamb = LambdaNet(d_x, d_z).to(device)
    Gnet = GNet(d_x, d_y).to(device)
    
    opt_G = torch.optim.Adam(Gnet.parameters(), lr=eta)
    opt_L = torch.optim.Adam(Lamb.parameters(), lr=eta)
    if any(p.requires_grad for p in Fnet.parameters()):
        opt_F = torch.optim.Adam(Fnet.parameters(), lr=eta)
    else:
        opt_F = None

    
    # build ordered parameter list and signs
    T_params = [] if Gnet is None else list(Gnet.parameters())
    F_params = list(Fnet.parameters())
    L_params = list(Lamb.parameters())
    all_params = T_params + F_params + L_params
    signs = [1.0] * len(T_params) + [-1.0] * (len(F_params) + len(L_params))
        

    # main loop
    for n in range(N):
        t0 = np.random.randint(burn_in, T - batch_len + 1)
        t1 = t0 + batch_len

        # compute loss at current params and attach grads
        loss_curr, ret_curr = loss_and_attach_grads(Fnet, Lamb, Gnet, X, (t0, t1), d_y, do_backward=False)
        ret = ret_curr


        # ----- zero old grads -----
        opt_G.zero_grad()
        if opt_F is not None:
            opt_F.zero_grad()
        opt_L.zero_grad()

        loss_curr.backward() # now backprop once on the non-detached loss
        

        # flip sign because F and lambda maximize/ascent => equivalent to minimizing (-loss)
        opt_G.step()

        if opt_F is not None:
            for p in Fnet.parameters():
                if p.grad is not None:
                    p.grad = -p.grad
            opt_F.step()
            
        for p in Lamb.parameters():
            if p.grad is not None:
                p.grad = -p.grad

        opt_L.step()

        
        """
        #STARTS HERE
        

        # --- compute grads at current params (already done) ---
        # loss_curr = loss_and_attach_grads(...)  # you already have this
        # all_params is the ordered list: [G_params , F_params, L_params]
        # signs is the list of +1/-1 same length as all_params

        # snapshot current gradients (cloned) so we can modify params safely
        grads_curr = [ (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p.data)) for p in all_params ]

        # save originals
        orig_params = [p.data.clone() for p in all_params]

        # apply waiting update in-place (no autograd history)
        with torch.no_grad():
            for p, g, s in zip(all_params, grads_curr, signs):
                p -= eta * s * g   # waiting param = orig - eta * sign * grad_curr

        # zero grads before second backward
        for p in all_params:
            if p.grad is not None:
                p.grad = None

        # compute loss_wait and attach gradients to current (waiting) params
        loss_wait, _ = loss_and_attach_grads(Fnet, Lamb, Gnet, X, (150, T), d_y)

        # collect grads_wait (cloned)
        grads_wait = [ (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p.data)) for p in all_params ]

        # optional: guard for NaN/Inf in grads_wait or loss_wait
        if (torch.isnan(loss_wait) or torch.isinf(loss_wait)) or any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads_wait):
            # revert to originals, reduce eta and skip this iteration (safe fallback)
            with torch.no_grad():
                for p, orig in zip(all_params, orig_params):
                    p.copy_(orig)
            eta *= 0.5
            print(f"Numerical issue detected; halving eta -> {eta:.2e} and skipping update")
            continue

        # optional gradient clipping on grads_wait (tune max_norm)
        max_norm = None  # e.g. 1.0 or None to disable
        if max_norm is not None:
            total_norm = 0.0
            for g in grads_wait:
                total_norm += float((g**2).sum().item())
            total_norm = total_norm**0.5
            clip_coef = max_norm / (total_norm + 1e-12)
            if clip_coef < 1.0:
                grads_wait = [g * clip_coef for g in grads_wait]

        # apply final update using original params and waiting grads:
        update_norm_sq = 0.0
        with torch.no_grad():
            for p, orig, g_wait, s in zip(all_params, orig_params, grads_wait, signs):
                new_p = orig - eta * s * g_wait
                update = new_p - orig
                p.copy_(new_p)
                update_norm_sq += float((update**2).sum().item())

        last_update_norm = update_norm_sq ** 0.5

        # zero grads so next iter starts clean
        for p in all_params:
            if p.grad is not None:
                p.grad = None
                
        #ENDS HERE
        """
        
        """
        # ===== REPLACE FROM HERE (entire waiting/actual update section) =====
        import numpy as _np, torch as _torch

        # Helper: full-dataset pure reconstruction loss (no coupling)
        def _full_recon_loss(G_obj):
            with _torch.no_grad():
                Y_full = compute_y_sequence(G_obj, X, d_y)   # (T, d_y)
                return float(((X.t() - Y_full)**2).mean().item())

        # --- snapshot originals
        orig_params = [p.data.clone() for p in all_params]

        # Record original recon loss
        L_orig = _full_recon_loss(Gnet)
        print("L_orig (full recon loss) =", L_orig)

        # --- compute current grads at theta
        loss_curr, ret_curr = loss_and_attach_grads(Fnet, Lamb, Gnet, X, (t0, t1), d_y)
        grads_curr = [ (p.grad.detach().clone() if p.grad is not None else _torch.zeros_like(p.data)) for p in all_params ]

        # --- apply waiting update in-place: theta_tilde = theta - eta * J * grad_curr
        with _torch.no_grad():
            for p, g, s in zip(all_params, grads_curr, signs):
                p -= eta * s * g

        # recon loss at waiting params
        L_wait = _full_recon_loss(Gnet)
        print("L_wait (at waiting params) =", L_wait)

        # compute grads at waiting params (grads_wait)
        for p in all_params:
            if p.grad is not None: p.grad = None
        loss_wait, _ = loss_and_attach_grads(Fnet, Lamb, Gnet, X, (t0, t1), d_y)
        grads_wait = [ (p.grad.detach().clone() if p.grad is not None else _torch.zeros_like(p.data)) for p in all_params ]

        # expected final params from originals: theta' = theta - eta * J * grad_wait
        expected_params = [ orig - eta * s * gw for orig, gw, s in zip(orig_params, grads_wait, signs) ]

        # temporarily apply expected params and measure recon loss
        with _torch.no_grad():
            for p, newp in zip(all_params, expected_params):
                p.copy_(newp)
        L_expected = _full_recon_loss(Gnet)
        print("L_expected (after applying expected final params) =", L_expected)

        # restore originals so training loop is unaffected
        with _torch.no_grad():
            for p, orig in zip(all_params, orig_params):
                p.copy_(orig)

        # ===== END REPLACEMENT =====
        
        """


        """
        # snapshot current gradients (detach copies)
        grads_curr = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in all_params]

        # build waiting models (deep copies)
        Fw = None if Fnet is None else copy.deepcopy(Fnet)
        Lw = copy.deepcopy(Lamb)
        Gw = None if Gnet is None else copy.deepcopy(Gnet)

        # build waiting parameter list in same order as all_params
        wait_params = []
        for p in (Gw.parameters() if Gw is not None else []): wait_params.append(p)
        for p in Fw.parameters(): wait_params.append(p)
        for p in Lw.parameters(): wait_params.append(p)

        # apply waiting update on waiting models WITHOUT tracking grad history
        with torch.no_grad():
            for p_wait, g, s in zip(wait_params, grads_curr, signs):
                p_wait -= eta * s * g

        # compute grads at waiting state
        loss_wait, _ = loss_and_attach_grads(Fw, Lw, Gw, X, (t0, t1), d_y)
                    

        # snapshot waiting gradients
        wait_params_list = []
        wait_params_list = list(Gw.parameters()) if Gw is not None else []
        wait_params_list += list(Fw.parameters()) + list(Lw.parameters())
        grads_wait = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in wait_params_list]

        # apply actual update to original params
        update_norm_sq = 0.0
        with torch.no_grad():
            for p, g, s in zip(all_params, grads_wait, signs):
                update = -eta * s * g
                p += update
                update_norm_sq += float((update ** 2).sum().item())

        last_update_norm = update_norm_sq ** 0.5
        
        """

        if verbose and (n % max(1, N // 10) == 0 or n < 5):
            print(f"Iter {n+1}/{N}: loss_curr={loss_curr.item():.6e}")

    return Fnet, Gnet, ret, Lamb



# ---------------------- Example usage ----------------------
if __name__ == "__main__":
    dx = 1
    T = 2500
    dz = 1
    #Xsim = np.random.randn(dx, T).astype(np.float32)
    Xsim, Z = gd.generate_kalman_data(T,0.8,1,0.4,0.4)   #(dx,T)
    X = torch.tensor(Xsim, dtype=torch.float32, device='cpu')
    
    #TEST STARTS
    """
    # supervised pretrain for G (teacher forcing)
    device = 'cpu'
    Gnet = GNet(dx, dx).to(device)
    Fnet = FNet(dx, dz).to(device)
    Lamb = LambdaNet(dx, dz).to(device)
    Gnet.train(); 
    for p in Fnet.parameters(): p.requires_grad=False
    for p in Lamb.parameters(): p.requires_grad=False

    opt = torch.optim.Adam(Gnet.parameters(), lr=1e-3)
    for it in range(500):
        opt.zero_grad()
        Y = compute_y_sequence(Gnet, X, dx)   # returns (T, d_y)
        loss = ((X.t() - Y)**2).mean()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            print("sup it", it, "loss", loss.item())

    # After training, print final full-dataset MSE
    with torch.no_grad():
        Y = compute_y_sequence(Gnet, X, dx)
        print("supervised final MSE:", ((X.t()-Y)**2).mean().item())
        
    """
    #TEST ENDS
   
    F_trained2, G_trained, ret2, L_trained2 = run_optimistic_md(Xsim, dz, N=700, eta=1e-3, batch_len = 400, verbose=True, d_y=dx)
    #START HERE
    
    with torch.no_grad():
        #Z_full = compute_z_sequence(F_trained, X)
        Xfull = X.t()   # (T, d_x)
        
        # recurrent
        ResY = compute_y_sequence(G_trained, X, dx)
        Y_full = ResY #Add Xfull if residual
        recurrent_mse = ((Xfull - Y_full)**2).mean().item()
        mse_t = ((Xfull - Y_full)**2).sum(dim=1).cpu().numpy()   # shape (T,)
        Z_full = compute_z_sequence(F_trained2, X)
        #Z_full = gd.kalman_predictors(Xsim.reshape(-1),0.8,1,0.4,0.4)
        #Lambda_all = L_trained2(torch.from_numpy(Z_full).float().view(-1,1))   # (T, d_x)
        Lambda_all = L_trained2(Z_full)   # (T, d_x)
        Lambda_mean = Lambda_all.mean(dim=0, keepdim=True)
        Lam_tilde = Lambda_all - Lambda_mean
        recon_vec = ((Xfull - Y_full)**2).sum(dim=1)        # (T,)
        coupling_vec = (Lam_tilde * Y_full).sum(dim=1)      # (T,)

        

    print("recurrent  full MSE:", recurrent_mse)


    print("mse_t mean, std:", np.mean(mse_t), np.std(mse_t))
    print("mse_t [first 20]:", mse_t[:20])
    print("mse_t [last 20]:", mse_t[-20:])

    print("recon mean, coupling mean:", recon_vec.mean().item(), coupling_vec.mean().item())
    print("|coupling| / recon ratio:", abs(coupling_vec.mean().item()) / (recon_vec.mean().item()+1e-12))
    Ynum2 = Y_full.numpy().reshape(-1)
    D = Xfull.detach().numpy().reshape(-1) - gd.kalman_predictors(Xsim,0.8,1,0.4,0.4)
    print("(X - Kalmanpred) variance: ",D.std()**2)
    Key = (Xfull.detach().numpy()-Y_full.detach().numpy())-5 * Lam_tilde.detach().numpy()
    print('Key shape is ',np.shape(Key))
    print('2 times y mean is:', 2*ResY.numpy().mean())
    print('Lambda(z) std is:', Lam_tilde.detach().numpy().std())
    plt.plot(Key,color='red',label='2(X-Y)-Lambda')
    plt.show()


    
    #END HERE

    print('Done')
    print(f"Recurrent loss is {ret2}")
