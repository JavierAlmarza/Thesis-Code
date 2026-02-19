import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import time
import GenerateData as gd


# ==========================================
# 1. The Generic JIT Runner (Unchanged)
# ==========================================
class FastRecurrentRunner(nn.Module):
    def __init__(self, cell: nn.Module, hidden_dim: int):
        super().__init__()
        self.cell = cell
        self.hidden_dim = hidden_dim

    def forward(self, X_embeddings: torch.Tensor) -> torch.Tensor:
        T = X_embeddings.shape[0]
        # Allocate output [T, Hidden]
        H_list = torch.zeros(T, self.hidden_dim, device=X_embeddings.device, dtype=X_embeddings.dtype)
        current_h = torch.zeros(self.hidden_dim, device=X_embeddings.device, dtype=X_embeddings.dtype)
        
        for t in range(T - 1):
            next_h = self.cell(X_embeddings[t], current_h)
            H_list[t+1] = next_h
            current_h = next_h
            
        return H_list

# ==========================================
# 2. Vectorized Linear Cell (Adapted)
# ==========================================
class VectorizedLinearCell(nn.Module):
    def __init__(self):
        super().__init__()
        # Same parameters as your original LinearCell
        self.raw_a = nn.Parameter(torch.randn(()))
        self.raw_b = nn.Parameter(torch.randn(()))
        
        # Architecture constants for the Solver
        self.input_dim = 1
        self.hidden_dim = 1 

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes (a * X) in parallel for all T.
        Mathematical equivalent: The 'input' to the loop is a*x_t
        """
        a = 0.5 * torch.tanh(self.raw_a)
        # X is [T], make it [T, 1] to match runner expectation
        return (X * a).unsqueeze(1)

    def forward(self, x_proj_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        Step: z_new = x_proj + b * z_prev
        Since x_proj = a * x_original, this equals a*x + b*z
        """
        b = torch.tanh(self.raw_b)
        return x_proj_t + b * z_prev

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        """
        Identity readout (Z is the state itself)
        """
        return H.squeeze()

# ==========================================
# 3. Vectorized GRU Cell (Updated Interface)
# ==========================================
class VectorizedGRUCell(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Internal Weights
        self.weight_hh = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.weight_ih = nn.Linear(input_dim, 3 * hidden_dim, bias=True)
        
        # Readout Layer (Hidden -> Z)
        self.fc_readout = nn.Linear(hidden_dim, 1)
        
        # Init
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes W_ih * X in parallel
        """
        # X is [T], make it [T, 1]
        return self.weight_ih(X.unsqueeze(1))

    def forward(self, x_projected_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Standard Vectorized GRU Logic
        gate_h = self.weight_hh(h_prev) 
        x_r, x_z, x_n = x_projected_t.chunk(3, dim=0)
        h_r, h_z, h_n = gate_h.chunk(3, dim=0)
        
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n)
        
        return (1 - z) * n + z * h_prev

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        """
        Linear projection from Hidden State to Scalar Z
        """
        return self.fc_readout(H).squeeze()

# ==========================================
# 4. The Unified Solver
# ==========================================
class UnifiedMaxMinSolver:
    def __init__(self, X, cell_instance, runner_script, burn_in=50, lr_model=1e-2, lr_lambda=1e-3, verbose_b=False, use_tanh_lambda=False,device='cpu'):
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.t0 = burn_in
        self.device = device
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.9,0.8,0.6,0.6)
        self.P = P
        self.verbose_b = verbose_b
        self.use_tanh_lambda = use_tanh_lambda
        
        # --- Modular Components ---
        self.cell = cell_instance.to(device)
        self.runner = runner_script # The JIT compiled runner
        
        # --- Optimization Variables ---
        # Optimization Variable Y (Initialized with noise)
        #self.Y = torch.randn_like(self.X, requires_grad=True, device=device) * 0.1

        self.Y = self.X.clone().detach().requires_grad_(True) #with X
        
        self.raw_lambda = torch.tensor(0.0, requires_grad=True, device=device)

        # --- Optimizers ---
        self.opt_Y = torch.optim.LBFGS([self.Y], lr=1, max_iter=50, 
                                       history_size=50, tolerance_change=1e-8, 
                                       line_search_fn="strong_wolfe")
        
        # Group all model parameters (whether a/b or Weights)
        self.opt_outer = torch.optim.Adam([
            {'params': self.cell.parameters(), 'lr': lr_model},  # Fast Physics
            {'params': [self.raw_lambda],      'lr': lr_lambda}  # Slow/Damped Penalty
        ])

    def get_lambda(self):
        scale = 150.0
        if self.use_tanh_lambda:
            # Tanh squashing: we scale output to range [-scale, scale]
            return scale * torch.tanh(self.raw_lambda)
        else:
            # Linear scaling
            return scale * self.raw_lambda

    def forward_Z(self):
        # 1. Project (Polymorphic call: works for Linear or GRU)
        X_emb = self.cell.project(self.X)
        
        # 2. Run (JIT Optimized)
        H = self.runner(X_emb)
        
        # 3. Readout (Polymorphic)
        Z = self.cell.readout(H)
        return Z

    def loss_function(self, Y, Z, lmbda):
        X_valid = self.X[self.t0:]
        Y_valid = Y[self.t0:]
        Z_valid = Z[self.t0:]
        
        Z_mean = Z_valid.mean()
        Z_std = Z_valid.std(unbiased=False) + 1e-6
        Z_norm = (Z_valid - Z_mean) / Z_std
        
        mse = ((X_valid - Y_valid)**2).mean()
        constraint = (Z_norm * Y_valid).mean()
        return mse + lmbda * constraint, mse, constraint

    def step(self):
        # Inner Loop
        with torch.no_grad():
            Z_fixed = self.forward_Z()
            lmbda_fixed = self.get_lambda()
        
        def closure():
            self.opt_Y.zero_grad()
            loss, _, _ = self.loss_function(self.Y, Z_fixed, lmbda_fixed)
            loss.backward()
            return loss
        self.opt_Y.step(closure)
        
        # Outer Loop
        self.opt_outer.zero_grad()
        Z = self.forward_Z()
        lmbda = self.get_lambda()
        loss, mse, coupling = self.loss_function(self.Y, Z, lmbda)
        (-loss).backward()
        
        # Clip grads (Safety for both)
        torch.nn.utils.clip_grad_norm_(self.cell.parameters(), 1.0)
        self.opt_outer.step()
        
        return loss.item(), mse.item(), coupling.item()
    
    def train(self, steps=1000, verbose = True):
        for i in range(steps):
            loss, mse, coupling = self.step()
            if i % 25 == 0:
                with torch.no_grad():
                    Z = self.forward_Z()
                    Z_valid = Z[self.t0:]
                    X_valid = self.X[self.t0:]
                    Y_valid = self.Y[self.t0:]
                    P_valid = self.P[self.t0:]
                    Zc = Z_valid - Z_valid.mean()
                    corr = torch.dot(Zc, X_valid) / (Zc.norm() * X_valid.norm())
                    
                    lmbda = self.get_lambda()
                Diff =  ((X_valid.detach().numpy().reshape(-1) - Y_valid.detach().numpy().reshape(-1) - P_valid)**2).mean() 
                msg = f"Step {i:4d} | Loss: {loss:.5f} | MSE: {mse:.5f} | Coupling: {coupling:.5f} | "
                msg += f"lam: {lmbda.item():.3f} | |Y-(X-Pred)|^2 ={Diff.item():.4f}, "
                if self.verbose_b and hasattr(self.cell, 'raw_b'):
                    b_val = torch.tanh(self.cell.raw_b).item()
                    msg += f" | b: {b_val:.4f}"
                print(msg)                          
                print(f"corr(Z,X) ={corr:.4f}, "
                      f"|X-Pred|^2={((X_valid.detach().numpy().reshape(-1) - P_valid)**2).mean():.4f}, "
                      f"|Pred|^2={(P_valid**2).mean():.4f}, "
                      f"Var(X)={(X_valid.detach().numpy().reshape(-1).std()**2).mean():.4f}")


# 5. Main
if __name__ == "__main__":
    import GenerateData as gd
    
    T = 3000
    burn_in = 500
    Xsim, Ztrue = gd.generate_kalman_data(T, 0.9, 0.8, 0.6, 0.6)
    Xsim, Ztrue, epsi = gd.generate_garch_data(T,0.05,0.12,0.86)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)

    # Select G type
    mode = "GRU" 
    
    if mode == "Linear":
        print("Running with Linear Cell...")
        cell = VectorizedLinearCell()
        # Linear cell uses hidden_dim=1
        runner = FastRecurrentRunner(cell, hidden_dim=1)
        show_b = True
        
    else: 
        print("Running with GRU Cell...")
        hidden_dim = 2
        cell = VectorizedGRUCell(input_dim=1, hidden_dim=hidden_dim)
        runner = FastRecurrentRunner(cell, hidden_dim=hidden_dim)
        show_b = False
    
    scripted_runner = torch.jit.script(runner)
    solver = UnifiedMaxMinSolver(X, cell, scripted_runner, burn_in=burn_in, verbose_b=show_b)
    solver.train(steps=500)