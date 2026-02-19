"""
Implementation of Optimistic Mirror Descent saddle-point solver for the BaryNet-like
objective described by the user.

Usage: import this file and call `run_optimistic_md(X, dz, ...)` where X is a numpy array
of shape (d_x, T).

Returns trained PyTorch modules T_net (decoder) and F_net (transition).

Notes & choices made (defaults):
- PyTorch implementation, manual gradient steps (no torch.optim) to implement J sign matrix.
- Default learning rate eta=1e-3, default iterations N=2000.
- Default batch length: min(512, T-200) with burn-in t0=101 by default.
- Stop criterion: early stop if parameter updates are extremely small (norm of all updates < tol).
- For the centered lambda, the mean is computed across the full time-series z_t for the given theta
  at each gradient evaluation (cost O(T) but T<=5000 by spec so acceptable).
- Networks use ReLU and the exact architectures requested.

Author: ChatGPT (generated for the user's request). 
"""

import copy
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import GenerateData as gd


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
        self.model = MLP([dx + dz, 7, 7, dz], activation=nn.ReLU())

    def forward(self, x, z):
        # x: (..., d_x), z: (..., d_z)
        inp = torch.cat([x, z], dim=-1)
        return self.model(inp)


class TNet(nn.Module):
    """T: (d_x + d_z) -> d_x with architecture: d_x+d_z -> 12 -> 7 -> 12 -> d_x"""

    def __init__(self, dx: int, dz: int):
        super().__init__()
        self.model = MLP([dx + dz, 12, 7, 12, dx], activation=nn.ReLU())

    def forward(self, x, z):
        inp = torch.cat([x, z], dim=-1)
        return self.model(inp)


class LambdaNet(nn.Module):
    """lambda: d_z -> d_x with architecture: d_z -> 7 -> 7 -> d_x"""

    def __init__(self, dx: int, dz: int):
        super().__init__()
        self.model = MLP([dz, 7, 7, dx], activation=nn.ReLU())

    def forward(self, z):
        return self.model(z)


# ---------------------- Utility helpers ----------------------

def compute_z_sequence(F: nn.Module, X: torch.Tensor, z0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute z_t sequence for t=1..T given F and inputs X (shape d_x x T as torch tensor).
    Returns Z of shape (T, d_z) where Z[t] corresponds to z_{t+1} in the user's indexing (i.e. z_t for x_{<t}).
    Clarification: We'll follow a convention: input X is shape (d_x, T) and we will produce Z of shape (T, d_z)
    with z[0] = z_1 (after seeing x_0 in recursion). To keep indexing simple, we shift so that
    z_t depends on x_{t-1} as requested.
    """
    dx, T = X.shape
    device = X.device
    # infer d_z from F by running a dummy pass if z0 is None
    if z0 is None:
        # initialize z0 as zeros with appropriate dim by doing a dummy forward
        dummy_x = X[:, 0].unsqueeze(0)  # (1, d_x)
        # try multiple z sizes: we assume F model is registered; attempt a small z of zeros
        # to infer out dimension by using a zero vector of size (1, F.model.net[0].in_features - dx)
        in_features = F.model.net[0].in_features
        dz = in_features - dx
        z_prev = torch.zeros(1, dz, device=device)
    else:
        z_prev = z0.unsqueeze(0)
        dz = z_prev.shape[-1]

    Z = torch.zeros((T, dz), device=device)
    # We'll use x_{t-1} to compute z_t. For t=0 (first z) we use x_{-1} which we take as zeros.
    #x_prev = torch.zeros(dx, device=device)
    # We'll use x_{t-1} to compute z_t. For t=0 (first z) we use x_{-1} which we take as zeros.
    # initialize x_prev from multivariate Gaussian with empirical mean and covariance from X
    # X has shape (d_x, T). Compute mean over columns and covariance over columns.
    X_mean = X.mean(dim=1) # (d_x,)
    X_centered = (X - X_mean[:, None]).t() # (T, d_x)
    Cov = (X_centered.t() @ X_centered) / (X_centered.shape[0] - 1) # (d_x, d_x)
    # Regularize covariance to avoid singularity
    Cov = Cov + 1e-6 * torch.eye(dx, device=device)
    x_prev = torch.distributions.MultivariateNormal(X_mean, Cov).sample()
    for t in range(T):
        x_prev_batch = x_prev.unsqueeze(0)  # (1, dx)
        z_prev_batch = z_prev  # (1, dz)
        z_next = F(x_prev_batch, z_prev_batch)  # (1, dz)
        z_next = z_next.squeeze(0)
        Z[t] = z_next
        # update for next step: x_{t} becomes x_prev for next iteration
        x_prev = X[:, t]
        z_prev = z_next.unsqueeze(0)

    return Z  # (T, dz)


def loss_and_grads(F: nn.Module, Tnet: nn.Module, Lamb: nn.Module, X: torch.Tensor, batch_range: Tuple[int, int], device='cpu'):
    """
    Compute scalar loss on batch and return gradients for all parameters.
    - X: torch tensor shape (d_x, T)
    - batch_range: (t0, T0) inclusive start, exclusive end indices in python indexing over 0..T-1 corresponding to t in the user's notation
    Returns: loss (float tensor), grads dict mapping parameter tensors -> gradient tensors (cloned)
    Also returns the parameter-to-gradient mapping in the same order as list(parameters).
    """
    # Ensure grads are zeroed
    for p in list(F.parameters()) + list(Tnet.parameters()) + list(Lamb.parameters()):
        if p.grad is not None:
            p.grad = None

    device = X.device
    d_x, T = X.shape

    # compute z sequence under current F
    Z = compute_z_sequence(F, X)  # (T, d_z)

    t0, t1 = batch_range
    assert 0 <= t0 < t1 <= T
    batch_len = t1 - t0

    # get batch x_t (for t in [t0,t1)) note indexing: X is (d_x,T) and Z[t] corresponds to z_t
    X_batch = X[:, t0:t1].t()  # (batch_len, d_x)
    Z_batch = Z[t0:t1]  # (batch_len, d_z)

    # compute T outputs: shape (batch_len, d_x)
    T_out = Tnet(X_batch, Z_batch)

    # compute lambda for all times to get mean
    Lambda_all = Lamb(Z)  # (T, d_x)
    Lambda_mean = Lambda_all.mean(dim=0, keepdim=True)  # (1, d_x)

    Lambda_batch = Lambda_all[t0:t1]  # (batch_len, d_x)
    Lambda_tilde = Lambda_batch - Lambda_mean  # broadcast (batch_len, d_x)

    # loss = ||x - T||^2 + lambda_tilde' * T  (summed over d_x then mean over batch)
    recon_loss = ((X_batch - T_out) ** 2).sum(dim=1)  # (batch_len,)
    coupling = (Lambda_tilde * T_out).sum(dim=1)  # (batch_len,)
    loss_vec = recon_loss + coupling
    loss = loss_vec.mean()
    ret = recon_loss.mean().item()

    # backward
    loss.backward()

    # collect grads
    params = list(Tnet.parameters()) + list(F.parameters()) + list(Lamb.parameters())
    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p.data) for p in params]

    return loss.detach(), params, grads, ret


def _params_signs(Tnet: nn.Module, F: nn.Module, Lamb: nn.Module):
    """Return a list of +1/-1 signs corresponding to J matrix ordering used in our code: params order is
    [Tnet.params (dim_tau -> +1), F.params (dim_theta -> -1), Lamb.params (dim_xi -> -1)]"""
    signs = []
    for _ in Tnet.parameters():
        signs.append(1.0)
    for _ in F.parameters():
        signs.append(-1.0)
    for _ in Lamb.parameters():
        signs.append(-1.0)
    return signs


# ---------------------- Optimistic Mirror Descent Loop ----------------------

def run_optimistic_md(X_np: np.ndarray,
                      d_z: int,
                      N: int = 2000,
                      eta: float = 1e-3,
                      batch_len: Optional[int] = None,
                      burn_in: int = 101,
                      tol: float = 1e-8,
                      device: str = 'cpu',
                      verbose: bool = True,
                      seed: Optional[int] = 0) -> Tuple[TNet, FNet, float]:
    """
    Main entrypoint.
    - X_np: numpy array shape (d_x, T)
    - d_z: latent dimension
    - N: number of optimistic mirror descent iterations
    - eta: step size
    - batch_len: length of contiguous batch; default = min(512, T-burn_in)
    - burn_in: first t index to allow (1-based in user's text). We will convert to 0-based.

    Returns Tnet (decoder) and F (transition) trained.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    d_x, T = X.shape
    if batch_len is None:
        batch_len = int(min(256, max(10, T - burn_in - 1)))

    # instantiate models
    Fnet = FNet(d_x, d_z).to(device)
    Tnet = TNet(d_x, d_z).to(device)
    Lamb = LambdaNet(d_x, d_z).to(device)

    signs = _params_signs(Tnet, Fnet, Lamb)

    # helper to flatten param lists and operate
    def get_param_list(mods):
        lst = []
        for m in mods:
            for p in m.parameters():
                lst.append(p)
        return lst

    all_params = get_param_list([Tnet, Fnet, Lamb])

    # main loop
    last_update_norm = float('inf')
    for n in range(N):
        # sample contiguous batch with burn-in
        t0 = np.random.randint(burn_in, T - batch_len + 1)
        t1 = t0 + batch_len

        # compute current gradients
        loss_curr, params_curr, grads_curr, ret = loss_and_grads(Fnet, Tnet, Lamb, X, (t0, t1), device=device)

        # construct waiting state params: param_wait = param - eta * sign * grad_curr
        # We'll create clones of models and overwrite their parameter data
        F_wait = copy.deepcopy(Fnet)
        T_wait = copy.deepcopy(Tnet)
        Lamb_wait = copy.deepcopy(Lamb)

        wait_params = get_param_list([T_wait, F_wait, Lamb_wait])

        # apply waiting update
        for p_wait, g_curr, sign in zip(wait_params, grads_curr, signs):
            p_wait.data = p_wait.data - eta * sign * g_curr

        # compute gradient at waiting state
        loss_wait, _, grads_wait, ret = loss_and_grads(F_wait, T_wait, Lamb_wait, X, (t0, t1), device=device)

        # apply actual update to original params: param = param - eta * sign * grad_wait
        update_norm_sq = 0.0
        for p, g_wait, sign in zip(all_params, grads_wait, signs):
            update = -eta * sign * g_wait
            p.data = p.data + update  # since update already contains sign
            update_norm_sq += float((update ** 2).sum().item())

        last_update_norm = update_norm_sq ** 0.5

        if verbose and (n % max(1, N // 10) == 0 or n < 5):
            print(f"Iter {n+1}/{N}: loss_curr={loss_curr.item():.6e}, loss_wait={loss_wait.item():.6e}, update_norm={last_update_norm:.3e}")

        # stopping criterion
        if last_update_norm < tol:            
            if verbose:
                print(f"Stopping at iter {n+1} due to small updates (norm={last_update_norm:.3e} < tol={tol})")
            break

    # return trained Tnet and Fnet (user requested these maps)
    return Tnet, Fnet, ret


# ---------------------- Example usage ----------------------
if __name__ == "__main__":
    # small sanity check with random data
    dx = 3
    T = 2000
    dz = 2
    #Xsim = np.random.randn(dx, T).astype(np.float32)
    Xsim, Z = gd.generate_kalman_data(T,0.8,1,0.4,0.4)
    T_trained, F_trained, ret = run_optimistic_md(Xsim, dz, N=200, eta=1e-3, verbose=True)
    print("Done. Returned trained T and F networks.")
    print(f"Loss is {ret}")
