import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import time
import GenerateData as gd


@torch.jit.script # JIT compiled recurrence for speed
def compute_Z_jit(X: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    T = X.shape[0]
    current_Z = torch.tensor(0.0, device=X.device, dtype=X.dtype)    
    Z_list = torch.zeros(T, device=X.device, dtype=X.dtype)
    
    for t in range(T - 1):
        val = a * X[t] + b * current_Z
        Z_list[t+1] = val
        current_Z = val
        
    return Z_list

class FastMaxMinMaxSolver:
    def __init__(self, X, burn_in=50, lr_ab=1e-2, lr_lambda=1e-2, n_inner=1, device='cpu'):
            self.X = X.to(device).float()
            self.T = X.shape[0]
            self.t0 = burn_in
            self.device = device
            self.n_inner = n_inner
            P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.9,0.8,0.6,0.6)
            self.P = P
            
            # Initialization for Y
            #self.Y = torch.zeros_like(self.X, requires_grad=True, device=device)
            self.Y = self.X.clone().detach().requires_grad_(True)
            
            # Initialization for outer variables (a, b random, lambda slightly nonzero for coupling awareness
            self.raw_a = torch.randn((), requires_grad=True, device=device) 
            self.raw_b = torch.randn((), requires_grad=True, device=device)
            self.raw_lambda = torch.tensor(0.0, requires_grad=True, device=device)

            # Optimizers, L-BFGS for Y
            self.opt_Y = torch.optim.LBFGS([self.Y], 
                                           lr=1, 
                                           max_iter=20, 
                                           history_size=10, 
                                           line_search_fn="strong_wolfe")
            
            self.opt_outer = torch.optim.Adam([self.raw_a, self.raw_b, self.raw_lambda], lr=lr_ab)

    def get_params(self):
        a = 0.5 * torch.tanh(self.raw_a)
        b = torch.tanh(self.raw_b)
        # Lambda shouldn't necessarily be bounded, so large scaling factor allows it to grow
        lmbda = 100.0 * self.raw_lambda 
        return a, b, lmbda

    def loss_function(self, Y, Z, lmbda):
        # Statistics computed on valid range
        X_valid = self.X[self.t0:]
        Y_valid = Y[self.t0:]
        Z_valid = Z[self.t0:]

        Z_mean = Z_valid.mean()        
        mse = ((X_valid - Y_valid)**2).mean()

        constraint = ((Z_valid - Z_mean) * Y_valid).mean()
        
        return mse + lmbda * constraint, mse, constraint

    def step(self):
        # 1) Outer Loop setup: get current outer variables
        a, b, lmbda = self.get_params()        
        # Compute Z (differentiable wrt a,b)
        Z = compute_Z_jit(self.X, a, b)
        
        # 2) Inner Loop: Minimize L wrt Y using L-BFGS      
        for _ in range(self.n_inner):
            def closure():  # We define a closure that re-evaluates the loss for L-BFGS
                self.opt_Y.zero_grad()
                # we treat Z and lmbda as constants (ie detach so no diff thru them)
                loss, _, _ = self.loss_function(self.Y, Z.detach(), lmbda.detach())
                loss.backward()
                return loss
            
            self.opt_Y.step(closure)
        
        # 3) Outer Loop: maximize L (minimize -L) wrt a, b, lambda
        self.opt_outer.zero_grad()
        
        a, b, lmbda = self.get_params() # Recompute params to track gradients   
        Z = compute_Z_jit(self.X, a, b)    # Recompute Z to track gradients back to a,b       
        loss, mse, coupling = self.loss_function(self.Y, Z, lmbda) # Compute loss (using Y optimized in step 2)        
        (-loss).backward() # max = minimize negative loss
        
        self.opt_outer.step()
        
        return loss.item(), mse.item(), coupling.item()

    def train(self, steps=500):
        start_time = time.time()
        for i in range(steps):
            loss, mse, coupling = self.step()
            
            if i % 50 == 0:
                a, b, l = self.get_params()
                with torch.no_grad():                    
                    Z = compute_Z_jit(self.X,a, b)
                    Z_valid = Z[self.t0:]
                    X_valid = self.X[self.t0:]
                    Y_valid = self.Y[self.t0:]
                    P_valid = self.P[self.t0:]
                    Zc = Z_valid - Z_valid.mean()
                    corr = torch.dot(Zc, X_valid) /(Zc.norm() * X_valid.norm())
                Diff =  ((X_valid.detach().numpy().reshape(-1) - Y_valid.detach().numpy().reshape(-1) - P_valid)**2).mean()
                print(f"Step {i:4d} | Loss: {loss:.5f} | MSE: {mse:.5f} | "
                      f"|Y-(X-Pred)|^2 ={Diff.item():.4f}, "
                      f"Coup: {coupling:.1e} | a: {a.item():.3f} | b: {b.item():.3f} | lam: {l.item():.3f}")

                print(f"corr(Z,X) ={corr:.4f}, "
                      f"|X-Pred|^2={((X_valid.detach().numpy().reshape(-1) )**2).mean():.4f}, "
                      f"|Pred|^2={(self.P[self.t0:]**2).mean():.4f}, "
                      f"Var(X)={(self.X[self.t0:].detach().numpy().reshape(-1).std()**2).mean():.4f}")        
        print(f"Total time: {time.time() - start_time:.2f}s")
        return self.get_params()


    
if __name__ == "__main__":
    T = 3000
    burn_in = 500

    Xsim, Ztrue = gd.generate_kalman_data(T,0.9,0.8,0.6,0.6)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
    
    solver = FastMaxMinMaxSolver(X, burn_in=500, lr_ab=0.01, n_inner=5)
    solver.train(steps=2000)
    