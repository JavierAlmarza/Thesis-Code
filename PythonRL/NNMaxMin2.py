import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import time
import GenerateData as gd


# This was a (failed) attempt to optimize solver with split cell

# 1. The modular cells G(x, z): Linear and Neural
class LinearCell(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize raw parameters
        self.raw_a = nn.Parameter(torch.randn(()))
        self.raw_b = nn.Parameter(torch.randn(()))
        
    def forward(self, x_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        # Constrain parameters (tanh) inside the forward pass
        a = 0.5 * torch.tanh(self.raw_a)
        b = torch.tanh(self.raw_b)
        return a * x_t + b * z_prev

class NeuralCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),  # Tanh is more stable for recurrence than ReLU
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
            # No final activation (Linear output) to allow arbitrary range for Z
        )
        
        # Optional: Initialize weights to be small to ensure stability at start
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        # 1. Construct input vector [x_t, z_prev]
        inp = torch.stack([x_t, z_prev], dim=0)  # torch.stack handles 0-d tensors correctly by creating a 1-d tensor of size [2]
        
        # 2. Forward pass
        out = self.net(inp)
        
        return out[0] #(out is shape [1], we want 0-d scalar)

class SplitNeuralCell(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        # We separate the input processing for X and Z
        # fc_x is NOT here; it will be applied outside the loop!
        
        # This layer processes ONLY the Z recurrence
        self.fc_z = nn.Linear(1, hidden_dim, bias=False) 
        
        # Bias for the first layer (shared)
        self.bias_1 = nn.Parameter(torch.zeros(hidden_dim))
        
        # The rest of the network
        self.net_rest = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialization
        with torch.no_grad():
            self.fc_z.weight.mul_(0.5)

    def forward(self, x_embedded_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        x_embedded_t: The pre-calculated W_x * X[t] (Shape: [Hidden])
        z_prev: Scalar state (Shape: [])
        """
        # 1. Manual Linear Layer for Z (No stacking needed!)
        # We treat scalar z_prev as a vector of size 1 for the multiplication
        # z_prev * weight_col is faster than matmul for scalar
        z_part = self.fc_z.weight.squeeze() * z_prev 
        
        # 2. Combine (Add pre-computed X part, Z part, and bias)
        h1 = x_embedded_t + z_part + self.bias_1
        
        # 3. Rest of network
        return self.net_rest(h1).squeeze()

# 2. The JIT-Scriptable recurrence unroller
class RecurrentRunner(nn.Module):
    # Takes any 'cell' module and unrolls it over time, thiss entire module will be JIT compiled.
    def __init__(self, cell: nn.Module):
        super().__init__()
        self.cell = cell

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        T = X.shape[0]
        Z_list = torch.zeros(T, device=X.device, dtype=X.dtype)
        current_z = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        
        # PyTorch JIT will compile this loop efficiently
        for t in range(T - 1):
            next_z = self.cell(X[t], current_z)
            Z_list[t+1] = next_z
            current_z = next_z
            
        return Z_list

class FastRecurrentRunner(nn.Module):
    def __init__(self, cell: nn.Module):
        super().__init__()
        self.cell = cell

    def forward(self, X_embeddings: torch.Tensor) -> torch.Tensor:
        # X_embeddings shape: [T, Hidden]
        T = X_embeddings.shape[0]
        Z_list = torch.zeros(T, device=X_embeddings.device, dtype=X_embeddings.dtype)
        current_z = torch.tensor(0.0, device=X_embeddings.device, dtype=X_embeddings.dtype)
        
        for t in range(T - 1):
            # Pass the pre-computed row corresponding to time t
            next_z = self.cell(X_embeddings[t], current_z)
            Z_list[t+1] = next_z
            current_z = next_z
            
        return Z_list

# 3. The Modular Solver
class ModularMaxMinSolver:
    def __init__(self, X, model_runner, burn_in=50, lr_model=5e-3, device='cpu'):
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.t0 = burn_in
        self.device = device
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.9,0.8,0.6,0.6)
        self.P = P
        
        # The Recurrent Model (G)
        self.model = model_runner.to(device)
        
        # Optimization Variable Y (Initialized with noise)
        #self.Y = torch.randn_like(self.X, requires_grad=True, device=device) * 0.1
        # Create the data first (detached from graph), then enable gradients
        #self.Y = (torch.randn_like(self.X, device=device) * 0.1).requires_grad_(True)
        #self.Y = torch.zeros_like(self.X, requires_grad=True, device=device) #with zero
        self.Y = self.X.clone().detach().requires_grad_(True) #with X
        
        # Optimization Variable Lambda (Lagrange Multiplier)
        self.raw_lambda = torch.tensor(0.0, requires_grad=True, device=device)

        # Optimizers
        # 1. Inner: L-BFGS for Y
        self.opt_Y = torch.optim.LBFGS([self.Y], 
                                       lr=1, 
                                       max_iter=50, 
                                       history_size=50, 
                                       tolerance_change=1e-7,
                                       line_search_fn="strong_wolfe")
        
        # 2. Outer: Adam for model parameters (G) and lambda
        # Note: We group them together, but we maximize wrt all of them
        self.opt_outer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': [self.raw_lambda]}
        ], lr=lr_model)

    def get_lambda(self):
        return 50.0 * self.raw_lambda

    def loss_function(self, Y, Z, lmbda):
        X_valid = self.X[self.t0:]
        Y_valid = Y[self.t0:]
        Z_valid = Z[self.t0:]
        
        # Robust preconditioning (crucial for stability) 
        Z_mean = Z_valid.mean()
        Z_std = Z_valid.std(unbiased=False) + 1e-6
        Z_norm = (Z_valid - Z_mean) / Z_std
        
        mse = ((X_valid - Y_valid)**2).mean()
        constraint = (Z_norm * Y_valid).mean()
        
        loss = mse + lmbda * constraint
        return loss, mse, constraint

    def step(self):
        # 1. Update inner loop (Y), we need to run the model once to get the "current" fixed Z for the inner loop
        with torch.no_grad():
            Z_fixed = self.model(self.X)
            lmbda_fixed = self.get_lambda()
        
        def closure():
            self.opt_Y.zero_grad()
            loss, _, _ = self.loss_function(self.Y, Z_fixed, lmbda_fixed)
            loss.backward()
            return loss
        
        self.opt_Y.step(closure)
        
        # 2. Outer Loop: maximize L (minimize -L) wrt a, b, lambda
        self.opt_outer.zero_grad()
        
        # Re-run model to build the computation graph from Params -> Z -> Loss
        Z = self.model(self.X)
        lmbda = self.get_lambda()
        
        loss, mse, coupling = self.loss_function(self.Y, Z, lmbda)
        (-loss).backward()     # Maximize loss => Minimize negative loss
        
        # Clip gradients (Safe guard for recurrent models)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.opt_outer.step()
        
        return loss.item(), mse.item(), coupling.item()

    def train(self, steps=2000, verbose=True):
        start_time = time.time()
        for i in range(steps):
            loss, mse, coupling = self.step()
            
            if verbose and i % 25 == 0:
                with torch.no_grad():
                    Z = self.model(self.X)
                    Z_valid = Z[self.t0:]
                    X_valid = self.X[self.t0:]
                    Y_valid = self.Y[self.t0:]
                    P_valid = self.P[self.t0:]
                    Zc = Z_valid - Z_valid.mean()
                    corr = torch.dot(Zc, X_valid) / (Zc.norm() * X_valid.norm())
                    
                    # Access parameters from the cell for logging (specific to LinearCell)
                    # Note: In a generic NN, you might log norm of weights instead
                    #a = 0.5 * torch.tanh(self.model.cell.raw_a)
                    #b = torch.tanh(self.model.cell.raw_b)
                    lmbda = self.get_lambda()
                Diff =  ((X_valid.detach().numpy().reshape(-1) - Y_valid.detach().numpy().reshape(-1) - P_valid)**2).mean() 
                print(f"Step {i:4d} | Loss: {loss:.5f} | MSE: {mse:.5f} | Coupling: {coupling:.5f} | "
                      #f"Coup: {coupling:.1e} | a: {a.item():.3f} | b: {b.item():.3f} | "
                      f"lam: {lmbda.item():.3f} | |Y-(X-Pred)|^2 ={Diff.item():.4f}, ")
                print(f"corr(Z,X) ={corr:.4f}, "
                      f"|X-Pred|^2={((X_valid.detach().numpy().reshape(-1) - P_valid)**2).mean():.4f}, "
                      f"|Pred|^2={(P_valid**2).mean():.4f}, "
                      f"Var(X)={(X_valid.detach().numpy().reshape(-1).std()**2).mean():.4f}")    
        
        print(f"Total time: {time.time() - start_time:.2f}s")

class OptimizedModularSolver:
    def __init__(self, X, cell_class=SplitNeuralCell, burn_in=50, lr_model=5e-3, device='cpu'):
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.t0 = burn_in
        self.device = device
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.9,0.8,0.6,0.6)
        self.P = P

        
        # --- Model Components ---
        self.cell = cell_class().to(device)
        self.runner = torch.jit.script(FastRecurrentRunner(self.cell))
        
        # The Input Projector (Computes W_x * X for all T at once)
        self.fc_x = nn.Linear(1, 8, bias=False).to(device) # Matches hidden_dim=8

        
        # Optimization Variable Y (Initialized with noise)
        #self.Y = torch.randn_like(self.X, requires_grad=True, device=device) * 0.1
        # Create the data first (detached from graph), then enable gradients
        #self.Y = (torch.randn_like(self.X, device=device) * 0.1).requires_grad_(True)
        #self.Y = torch.zeros_like(self.X, requires_grad=True, device=device) #with zero
        self.Y = self.X.clone().detach().requires_grad_(True) #with X
        
        # Optimization Variable Lambda (Lagrange Multiplier)
        self.raw_lambda = torch.tensor(0.0, requires_grad=True, device=device)

        # --- Optimizers ---
        self.opt_Y = torch.optim.LBFGS([self.Y], lr=1, max_iter=50, 
                                       history_size=50, tolerance_change=1e-7, 
                                       line_search_fn="strong_wolfe")
        
        # Group all "Outer" parameters
        self.opt_outer = torch.optim.Adam([
            {'params': self.cell.parameters()},
            {'params': self.fc_x.parameters()}, # Don't forget the projector!
            {'params': [self.raw_lambda]}
        ], lr=lr_model)

    def get_lambda(self):
        return 50.0 * self.raw_lambda

    def forward_Z(self):
        """Helper to run the full model efficiently"""
        # 1. Vectorized Projection (Parallel over T!)
        # Input X: [T] -> [T, 1] -> [T, 8]
        X_emb = self.fc_x(self.X.unsqueeze(1))
        
        # 2. JIT Recurrence (Sequential over T, but simplified)
        Z = self.runner(X_emb)
        return Z

    def loss_function(self, Y, Z, lmbda):
        X_valid = self.X[self.t0:]
        Y_valid = Y[self.t0:]
        Z_valid = Z[self.t0:]
        
        # Preconditioning
        Z_mean = Z_valid.mean()
        Z_std = Z_valid.std(unbiased=False) + 1e-6
        Z_norm = (Z_valid - Z_mean) / Z_std
        
        mse = ((X_valid - Y_valid)**2).mean()
        constraint = (Z_norm * Y_valid).mean()
        
        return mse + lmbda * constraint, mse, constraint

    def step(self):
        # 1. Inner Loop (Y)
        with torch.no_grad():
            Z_fixed = self.forward_Z()
            lmbda_fixed = self.get_lambda()
        
        def closure():
            self.opt_Y.zero_grad()
            loss, _, _ = self.loss_function(self.Y, Z_fixed, lmbda_fixed)
            loss.backward()
            return loss
        
        self.opt_Y.step(closure)
        
        # 2. Outer Loop (Model)
        self.opt_outer.zero_grad()
        
        Z = self.forward_Z() # Re-run to build graph
        lmbda = self.get_lambda()
        loss, mse, coupling = self.loss_function(self.Y, Z, lmbda)
        
        (-loss).backward()
        torch.nn.utils.clip_grad_norm_(self.cell.parameters(), 1.0)
        self.opt_outer.step()
        
        return loss.item(), mse.item(), coupling.item()

    def train(self, steps=2000, verbose=True):
        start = time.time()
        for i in range(steps):
            loss, mse, coupling = self.step()
            if verbose and i % 25 == 0:
                with torch.no_grad():
                    Z = self.forward_Z()
                    Z_valid = Z[self.t0:]
                    X_valid = self.X[self.t0:]
                    Y_valid = self.Y[self.t0:]
                    P_valid = self.P[self.t0:]
                    Zc = Z_valid - Z_valid.mean()
                    corr = torch.dot(Zc, X_valid) / (Zc.norm() * X_valid.norm())
                    
                    # Access parameters from the cell for logging (specific to LinearCell)
                    # Note: In a generic NN, you might log norm of weights instead
                    #a = 0.5 * torch.tanh(self.model.cell.raw_a)
                    #b = torch.tanh(self.model.cell.raw_b)
                    lmbda = self.get_lambda()
                Diff =  ((X_valid.detach().numpy().reshape(-1) - Y_valid.detach().numpy().reshape(-1) - P_valid)**2).mean() 
                print(f"Step {i:4d} | Loss: {loss:.5f} | MSE: {mse:.5f} | Coupling: {coupling:.5f} | "
                      #f"Coup: {coupling:.1e} | a: {a.item():.3f} | b: {b.item():.3f} | "
                      f"lam: {lmbda.item():.3f} | |Y-(X-Pred)|^2 ={Diff.item():.4f}, ")
                print(f"corr(Z,X) ={corr:.4f}, "
                      f"|X-Pred|^2={((X_valid.detach().numpy().reshape(-1) - P_valid)**2).mean():.4f}, "
                      f"|Pred|^2={(P_valid**2).mean():.4f}, "
                      f"Var(X)={(X_valid.detach().numpy().reshape(-1).std()**2).mean():.4f}")    
        print(f"Total time: {time.time() - start:.2f}s")

# 4. MAIN
if __name__ == "__main__":

    T = 3000
    burn_in = 500
   
    Xsim, Ztrue = gd.generate_kalman_data(T,0.9,0.8,0.6,0.6)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
    
    # A. Instantiate the specific cell (G)
    #cell = LinearCell()
    #cell = NeuralCell()
    
    # B. Wrap it in the Runner and JIT Script it (Option B)
    # This compiles the loop and the cell into optimized code
    
    #runner = RecurrentRunner(cell)
    #scripted_runner = torch.jit.script(runner)
    
    # C. Pass to Solver and Train
    #solver = ModularMaxMinSolver(X, scripted_runner, burn_in=burn_in, lr_model=1e-3)
    solver = OptimizedModularSolver(X, cell_class=SplitNeuralCell)
    
    solver.train(steps=1000)