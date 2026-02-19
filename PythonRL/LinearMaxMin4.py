import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import time
import GenerateData as gd

# --- JIT Compiled Recurrence ---
@torch.jit.script
def compute_Z_jit(X: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    T = X.shape[0]
    Z_list = torch.zeros(T, device=X.device, dtype=X.dtype)
    current_Z = torch.tensor(0.0, device=X.device, dtype=X.dtype)
    
    # We perform the recurrence
    for t in range(T - 1):
        val = a * X[t] + b * current_Z
        Z_list[t+1] = val
        current_Z = val
        
    return Z_list

class RobustMaxMinSolver:
    def __init__(self, X, burn_in=50, lr_ab=5e-3, device='cpu'):
        """
        Args:
            lr_ab: Lowered slightly (1e-2 -> 5e-3) to handle the high-variance 
                   regime of alpha=0.9 smoother.
        """
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.t0 = burn_in
        self.device = device
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.9,0.8,0.6,0.6)
        self.P = P
    
        # Initialization for Y
        #self.Y = torch.zeros_like(self.X, requires_grad=True, device=device) #with zero
        self.Y = self.X.clone().detach().requires_grad_(True) #with X
        #self.Y = (torch.randn_like(self.X, device=device) * 0.1).requires_grad_(True) #with noise
        
        # Initialization for outer variables (a, b random, lambda slightly nonzero for coupling awareness
        self.raw_a = torch.randn((), requires_grad=True, device=device) 
        self.raw_b = torch.randn((), requires_grad=True, device=device)
        self.raw_lambda = torch.tensor(0.1, requires_grad=True, device=device)

        # --- Optimizers ---
        # L-BFGS: We allow more 'max_iter' per step, but only call step() once.
        # This is more efficient than a python loop (n_inner).
        # 'tolerance_change' ensures we exit early if we hit the solution.
        self.opt_Y = torch.optim.LBFGS([self.Y], 
                                       lr=1, 
                                       max_iter=50,       # High ceiling
                                       history_size=50,   # Better curvature approx
                                       tolerance_change=1e-7,
                                       line_search_fn="strong_wolfe")
        
        self.opt_outer = torch.optim.Adam([self.raw_a, self.raw_b, self.raw_lambda], lr=lr_ab)

    def get_params(self):
        a = 0.5 * torch.tanh(self.raw_a)
        b = torch.tanh(self.raw_b)
        lmbda = 50.0 * self.raw_lambda  # Scaling factor
        return a, b, lmbda

    def loss_function(self, Y, Z, lmbda):
        #Statistics computed on valid range
        X_valid = self.X[self.t0:]
        Y_valid = Y[self.t0:]
        Z_valid = Z[self.t0:]
        
        # PRECONDITIONING (critical for stability): we normalize Z statistics. This prevents the variance explosion 
        # at alpha=0.9 from destabilizing the lambda/Y relationship.
        Z_mean = Z_valid.mean()
        Z_std = Z_valid.std(unbiased=False) + 1e-6 # Avoid div by zero        
        # The constraint is now on corr, not cov, this is mathematically cleaner for optimization.
        Z_norm = (Z_valid - Z_mean) / Z_std        
        
        mse = ((X_valid - Y_valid)**2).mean()
        constraint = (Z_norm * Y_valid).mean()
        
        # Total Lagrangian
        loss = mse + lmbda * constraint
        return loss, mse, constraint

    def step(self):
        # 1. Update inner loop (Y), we assume Z is fixed for this step.
        with torch.no_grad():
            a, b, lmbda_fixed = self.get_params()
            Z_fixed = compute_Z_jit(self.X, a, b)
        
        def closure():
            self.opt_Y.zero_grad()
            # We must re-evaluate loss with the current Y
            loss, _, _ = self.loss_function(self.Y, Z_fixed, lmbda_fixed)
            loss.backward()
            return loss
        
        # One call with max_iter=50 is faster & better than loop of 10
        self.opt_Y.step(closure)
        
        # 2. Outer Loop: maximize L (minimize -L) wrt a, b, lambda
        self.opt_outer.zero_grad()
        
        a, b, lmbda = self.get_params() # Recompute params to track gradients   
        Z = compute_Z_jit(self.X, a, b)    # Recompute Z to track gradients back to a,b       
        
        loss, mse, coupling = self.loss_function(self.Y, Z, lmbda) # Compute loss (using Y optimized in step 2)        
        (-loss).backward() # max = minimize negative loss       
        
        # Optional: Gradient Clipping for stability with high alpha
        torch.nn.utils.clip_grad_norm_([self.raw_a, self.raw_b, self.raw_lambda], 1.0)
        
        self.opt_outer.step()
        
        return loss.item(), mse.item(), coupling.item()

    def train(self, steps=2000, verbose=True):
        start_time = time.time()
        for i in range(steps):
            loss, mse, coupling = self.step()
            
            if verbose and i % 50 == 0:
                a, b, l = self.get_params()
                with torch.no_grad():
                    Z = compute_Z_jit(self.X, a, b)
                    Z_valid = Z[self.t0:]
                    X_valid = self.X[self.t0:]
                    Y_valid = self.Y[self.t0:]
                    P_valid = self.P[self.t0:]
                    Zc = Z_valid - Z_valid.mean()
                    corr = torch.dot(Zc, X_valid) / (Zc.norm() * X_valid.norm())
                Diff =  ((X_valid.detach().numpy().reshape(-1) - Y_valid.detach().numpy().reshape(-1) - P_valid)**2).mean()                    
                print(f"Step {i:4d} | Loss: {loss:.5f} | MSE: {mse:.5f} | "
                      f"|Y-(X-Pred)|^2 ={Diff.item():.4f}, "
                      f"Coup: {coupling:.1e} | a: {a.item():.3f} | b: {b.item():.3f} | lam: {l.item():.3f} | ")
                print(f"corr(Z,X) ={corr:.4f}, "
                      f"|X-Pred|^2={((X_valid.detach().numpy().reshape(-1) - P_valid)**2).mean():.4f}, "
                      f"|Pred|^2={(P_valid**2).mean():.4f}, "
                      f"Var(X)={(X_valid.detach().numpy().reshape(-1).std()**2).mean():.4f}")        
        
        print(f"Total time: {time.time() - start_time:.2f}s")
        return self.get_params()

if __name__ == "__main__":
    T = 3000
    burn_in = 500

    Xsim, Ztrue = gd.generate_kalman_data(T,0.9,0.8,0.6,0.6)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)

    solver = RobustMaxMinSolver(X, burn_in=burn_in, lr_ab=5e-3)
    solver.train(steps=500)