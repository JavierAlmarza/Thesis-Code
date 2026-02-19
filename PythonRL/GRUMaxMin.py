import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import time
import GenerateData as gd

# This one does Vectorized GRU and Linear-GRU with GRU-adapted Solver

# 1. The modular cells G(x, z) are GRU      
class LinearRecurrentCell(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=2):
        super().__init__()
        # Pure Linear Recurrence: h_t = W_h * h_{t-1} + W_x * x_t
        self.weight_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.weight_ih = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Initialization near identity for stability
        with torch.no_grad():
            nn.init.eye_(self.weight_hh.weight)
            self.weight_hh.weight.mul_(0.9) # Contractive start
            nn.init.xavier_uniform_(self.weight_ih.weight)

    def forward(self, x_projected_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Note: x_projected_t is pre-computed W_ih * x_t
        # Linear update
        h_next = self.weight_hh(h_prev) + x_projected_t
        return h_next
        
class VectorizedGRUCell(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # --- 1. The Recurrent Weights (Sequential Part) ---
        # [h] -> [3*h] (Reset, Update, New)
        self.weight_hh = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        
        # --- 2. The Input Weights (Parallel Part) ---
        # [x] -> [3*h]
        self.weight_ih = nn.Linear(input_dim, 3 * hidden_dim, bias=True)
        
        # --- Corrected Initialization ---
        # Iterate over modules, not parameters, to target weights specifically
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.orthogonal_(param)
                else:
                    # Fallback for 1D weights (rare in Linear) or safety check
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x_projected_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x_projected_t: Pre-computed (Wx * x_t + b_ih). Shape: [3*hidden_dim]
        h_prev: Previous state. Shape: [hidden_dim] (or scalar expanded)
        """
        # 1. Compute all recurrent parts at once
        # gate_h = Wh * h_prev + b_hh
        gate_h = self.weight_hh(h_prev) 
        
        # 2. Combine Input (pre-computed) and Recurrent
        # The raw gate values
        gates = x_projected_t + gate_h
        
        # 3. Split into Reset (r), Update (z), and New (n) parts
        r_gate, z_gate, n_gate = gates.chunk(3, dim=0)
        
        # 4. Apply Activations
        r = torch.sigmoid(r_gate)
        z = torch.sigmoid(z_gate)
        
        # For the 'new' memory, the reset gate affects the recurrent part only
        # We must re-separate the logic slightly for the 'n' gate standard GRU formulation:
        # n = tanh(W_in*x + b_in + r * (W_hn*h + b_hn))
        # To keep it vectorized efficiently, we use a slight approximation or 
        # perform the specific split math. Let's do the standard implementation:
        
        # Re-calc the recurrent part for 'n' specifically to apply 'r' correctly
        # We extract the specific chunks from the pre-computed layers
        
        # Efficient Implementation trick:
        # The previous 'gates' calculation assumed simple addition. 
        # Standard GRU requires: n = tanh(x_n + r * h_n)
        
        # Let's unpack the pre-computed X parts
        x_r, x_z, x_n = x_projected_t.chunk(3, dim=0)
        
        # Let's unpack the computed H parts (before adding to X)
        h_r, h_z, h_n = gate_h.chunk(3, dim=0)
        
        # Now apply standard GRU logic
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n) # Reset gate acts here
        
        # 5. Update State
        h_next = (1 - z) * n + z * h_prev
        
        return h_next

# 2. The JIT-Scriptable GRU-adapted recurrence unroller        
class FastGRURunner(nn.Module):
    def __init__(self, cell: nn.Module, hidden_dim: int):
        super().__init__()
        self.cell = cell
        self.hidden_dim = hidden_dim

    def forward(self, X_embeddings: torch.Tensor) -> torch.Tensor:
        T = X_embeddings.shape[0]
        # Allocate output [T, Hidden]
        H_list = torch.zeros(T, self.hidden_dim, device=X_embeddings.device, dtype=X_embeddings.dtype)
        
        # Initial state h_0
        current_h = torch.zeros(self.hidden_dim, device=X_embeddings.device, dtype=X_embeddings.dtype)
        
        for t in range(T - 1):
            next_h = self.cell(X_embeddings[t], current_h)
            H_list[t+1] = next_h
            current_h = next_h
            
        return H_list
        
# 3. The Modular Solver
class GRUMaxMinSolver:
    def __init__(self, X, cell, runner, hidden_dim=4, burn_in=50, lr_model=5e-3, device='cpu'):
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.t0 = burn_in
        self.device = device
        self.hidden_dim = hidden_dim
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.9,0.8,0.6,0.6)
        self.P = P
        
        # --- Model Components ---
        # 1. The Cell
        self.cell = cell
        
        # 2. The Runner (JIT Compiled)
        # Note: We need a specialized runner that handles hidden_dim sized states
        self.runner = runner
        
        # 3. The Input Projector (Held in the cell for organization, but used here)
        # We use the cell's own layer to ensure dimensions match
        self.projector = self.cell.weight_ih
        
        # 4. Readout Layer (Hidden -> Scalar Z)
        self.readout = nn.Linear(hidden_dim, 1).to(device)

        # --- Optimization Variables ---
        # Optimization Variable Y (Initialized with noise)
        #self.Y = torch.randn_like(self.X, requires_grad=True, device=device) * 0.1
        # Create the data first (detached from graph), then enable gradients
        #self.Y = (torch.randn_like(self.X, device=device) * 0.1).requires_grad_(True)
        #self.Y = torch.zeros_like(self.X, requires_grad=True, device=device) #with zero
        self.Y = self.X.clone().detach().requires_grad_(True) #with X
        # Optimization variable lambda
        self.raw_lambda = torch.tensor(0.0, requires_grad=True, device=device)

        # --- Optimizers ---
        # Increased max_iter for Inner loop as Neural landscapes are trickier
        self.opt_Y = torch.optim.LBFGS([self.Y], lr=1, max_iter=100, 
                                       history_size=100, tolerance_change=1e-9, 
                                       line_search_fn="strong_wolfe")
        
        self.opt_outer = torch.optim.Adam([
            {'params': self.cell.parameters()},
            {'params': self.readout.parameters()},
            {'params': [self.raw_lambda]}
        ], lr=lr_model)

    def get_lambda(self):
        return 200.0 * self.raw_lambda

    def forward_Z(self):
        # 1. Vectorized Projection (Input X -> GRU Gates)
        # X: [T] -> [T, 1] -> [T, 3*Hidden]
        X_emb = self.projector(self.X.unsqueeze(1))
        
        # 2. Run GRU Recurrence
        H = self.runner(X_emb) # Returns [T, Hidden]
        
        # 3. Project Hidden State to Scalar Z
        Z = self.readout(H).squeeze()
        return Z

    # ... [loss_function and step are identical to previous ModularSolver] ...
    # (Just copy them from the previous response)
    
    # Copying essential parts for completeness:
    def loss_function(self, Y, Z, lmbda):
        X_valid = self.X[self.t0:]
        Y_valid = Y[self.t0:]
        Z_valid = Z[self.t0:]
        
        # Robust Preconditioning
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
        torch.nn.utils.clip_grad_norm_(self.cell.parameters(), 1.0)
        self.opt_outer.step()
        return loss.item(), mse.item(), coupling.item()
    
    def train(self, steps=2000, verbose = True):
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
    hidden_dim = 2
   
    Xsim, Ztrue = gd.generate_kalman_data(T,0.9,0.8,0.6,0.6)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
    
    # A. Instantiate the specific cell (G)
    #cell = LinearRecurrentCell(input_dim=1, hidden_dim=hidden_dim).to('cpu')
    cell = VectorizedGRUCell(input_dim=1, hidden_dim=hidden_dim).to('cpu')

    
    # B. Wrap in Runner and JIT script it so loop and cell are compiled into optimized code
    #runner = RecurrentRunner(cell)
    runner = FastGRURunner(cell, hidden_dim=hidden_dim)
    scripted_runner = torch.jit.script(runner)
    
    # C. Pass to Solver and Train
    
    #solver = ModularMaxMinSolver(X, scripted_runner, burn_in=burn_in, lr_model=1e-3)
    solver = GRUMaxMinSolver(X,cell, scripted_runner,hidden_dim=hidden_dim)
    solver.train(steps=500)