import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import GenerateData as gd
import time
import matplotlib.pyplot as plt


class IdentityPsi(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters to optimize        
    def forward(self, Y):
        return Y

class PolyPsi(nn.Module):
    def __init__(self, degree=5):
        super().__init__()
        self.degree = degree
        # Initialize close to identity (c1=1, others=0) to start stable
        self.coeffs = nn.Parameter(torch.zeros(degree + 1))
        with torch.no_grad():
            self.coeffs[1] = 1.0

    def forward(self, Y):
        res = torch.zeros_like(Y)
        for d in range(self.degree + 1):
            res = res + self.coeffs[d] * (Y ** d)   # we use Horner's method or direct summation
        return res

class NeuralPsi(nn.Module):
    def __init__(self, hidden_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Initialize output layer small to start near-zero influence
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)

    def forward(self, Y):
        # Y is [T], needs to be [T, 1] for Linear
        return self.net(Y.unsqueeze(1)).squeeze()

# 1. The Generic JIT Runner
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

# 2. Vectorized linear cell
class VectorizedLinearCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_a = nn.Parameter(torch.randn(()))
        self.raw_b = nn.Parameter(torch.randn(()))
        self.input_dim = 1
        self.hidden_dim = 1 

    def project(self, X: torch.Tensor) -> torch.Tensor:
        #Computes (a * X) in parallel for all T.      
        a = 0.5 * torch.tanh(self.raw_a)
        return (X * a).unsqueeze(1) # X is [T], make it [T, 1] to match runner expectation

    def forward(self, x_proj_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        #Step: z_new = x_proj + b * z_prev
        b = torch.tanh(self.raw_b)
        return x_proj_t + b * z_prev

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        return H.squeeze()

# 3. Vectorized GRU cell 
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
        #Computes W_ih * X in parallel
        # X is [T], make it [T, 1]
        return self.weight_ih(X.unsqueeze(1))

    def forward(self, x_projected_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
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


# 4. The Unified Solver with Psi
class UnifiedMaxMinSolver:
    def __init__(self, X, cell_instance, runner_script,
                 P, epsi,    
                 psi_mode='identity',
                 data_mode="Kalman", 
                 burn_in=50, lr_Y=1, lr_model=1e-2, lr_lambda=1e-3, lr_psi=1e-2,
                 verbose_b=False, use_tanh_lambda=False, device='cpu'):
        
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.t0 = burn_in
        self.device = device
        self.verbose_b = verbose_b
        self.use_tanh_lambda = use_tanh_lambda
        self.data_mode = data_mode
        self.P = P
        self.epsi = epsi
        self.runner = runner_script 

        # Initialize functions (G and Psi)
        
        self.cell = cell_instance.to(device)
        if psi_mode == 'identity':
            self.psi = IdentityPsi().to(device)
        elif psi_mode == 'poly':
            self.psi = PolyPsi(degree=2).to(device)
        elif psi_mode == 'neural':
            self.psi = NeuralPsi().to(device)
        else:
            raise ValueError(f"Unknown psi_mode: {psi_mode}")

        # Initialize variables (Y and Lambda)
        
        #self.Y = (torch.randn_like(self.X, device=device) * 0.1).requires_grad_(True) # Create rnd data first (detached from graph), then enable gradients
        #self.Y = torch.zeros_like(self.X, requires_grad=True, device=device) #with zero
        self.Y = self.X.clone().detach().requires_grad_(True)
        self.raw_lambda = torch.tensor(0.0, requires_grad=True, device=device)

        # Optimizers: LBFGS for Y and Adam for G, Psi and Lambda
        self.opt_Y = torch.optim.LBFGS([self.Y], lr=lr_Y, max_iter=50, 
                                       history_size=50, tolerance_change=1e-8, 
                                       line_search_fn="strong_wolfe")
        
        self.opt_outer = torch.optim.Adam([
            {'params': self.cell.parameters(), 'lr': lr_model}, 
            {'params': self.psi.parameters(),  'lr': lr_psi}, 
            {'params': [self.raw_lambda],      'lr': lr_lambda} 
        ])
        
        # Scheduler: decays LR by factor of 0.5 every 100 steps
        # This prevents the maximizer from "wiggling" too much late in the game
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_outer, step_size=200, gamma=0.5)
        # After 150 steps, 0.995^150 approx 0.47 (similar to halving)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_outer, gamma=0.9975)

    def get_lambda(self):
        scale = 150.0
        if self.use_tanh_lambda:
            return scale * torch.tanh(self.raw_lambda)
        else:
            return scale * self.raw_lambda

    def forward_Z(self):
        X_emb = self.cell.project(self.X)
        H = self.runner(X_emb)
        Z = self.cell.readout(H)
        return Z

    def loss_function(self, Y, Z, lmbda):
        X_valid = self.X[self.t0:]
        Y_valid = Y[self.t0:]
        Z_valid = Z[self.t0:]
        
        # 1. Normalize Z 
        Z_mean = Z_valid.mean()
        Z_std = Z_valid.std(unbiased=False) + 1e-6
        Z_norm = (Z_valid - Z_mean) / Z_std
        
        # 2. Apply Psi to Y 
        # Note: We do NOT normalize Psi(Y) here. The maximizer learns the scale of Psi to maximize the dot product.
        # Lambda will naturally balance this out.
        psi_Y = self.psi(Y_valid)
        # Alternative: we force the output of Psi to have unit variance and zero mean
        # This prevents the maximizer from cheating by exploding the scale, but it didn't work
        #raw_psi_Y = self.psi(Y_valid) 
        #psi_mean = raw_psi_Y.mean()
        #psi_std = raw_psi_Y.std(unbiased=False) + 1e-6
        #psi_Y = (raw_psi_Y - psi_mean) / psi_std
        
        mse = ((X_valid - Y_valid)**2).mean()
        
        # 3. Coupling term: Correlation between Z_norm and Psi(Y)
        constraint = (Z_norm * psi_Y).mean()
                
        # 4. Optional: Ridge reg to stabilize flat minima
        reg = 0
        #reg = 1e-4 * (Y_valid**2).mean()
        
        
        return mse + lmbda * constraint + reg, mse, constraint

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
        # We can also clip Psi gradients if needed
        if hasattr(self.psi, 'parameters'):
             torch.nn.utils.clip_grad_norm_(self.psi.parameters(), 1.0)
             
        self.opt_outer.step()
        self.scheduler.step() # Update LR
        
        return loss.item(), mse.item(), coupling.item()
    
    def train(self, steps=1000):
        X_valid = self.X[self.t0:]
        P_valid = self.P[self.t0:]
        Diffv = np.zeros(steps)
        MSEv = np.zeros(steps)
        Lossv = np.zeros(steps)
        EqVal = (P_valid**2).mean()
        if self.data_mode == "Kalman":
            Ystar = X_valid.detach().numpy().reshape(-1) - P_valid
        elif self.data_mode == "GARCH":
            E_sd = (X_valid.detach().numpy().reshape(-1)/self.epsi[self.t0:]).mean()
            Ystar = self.epsi[self.t0:] * E_sd
        print("Data metrics")
        print(f"Var(X) = {(X_valid.detach().numpy().reshape(-1).std()**2).mean():.4f}")
        print(f"Equilibrium V = |Pred|^2 = {EqVal:.4f}")
        print(f"|Y*|^2 = {(Ystar**2).mean():.4f} ")
        print(" ")

        for i in range(steps):
            loss, mse, coupling = self.step()
            Y_valid = self.Y[self.t0:]
            Diff =  ((Ystar - Y_valid.detach().numpy().reshape(-1))**2).mean()
            Diffv[i] = Diff.item()/(Ystar**2).mean()
            MSEv[i] = mse
            Lossv[i] = loss
            if i % 25 == 0:
                with torch.no_grad():
                    Z = self.forward_Z()
                    Z_valid = Z[self.t0:]
                    Zc = Z_valid - Z_valid.mean()
                    corr = torch.dot(Zc, X_valid) / (Zc.norm() * X_valid.norm())
                    lmbda = self.get_lambda()
                msg = f"Step {i:4d}  |  Loss: {loss:.5f}  |  MSE: {mse:.5f}  |  Pnlty: {coupling:.5f}  |  "
                msg += f"lmbd: {lmbda.item():.3f} "
                if self.verbose_b and hasattr(self.cell, 'raw_b'):
                    b_val = torch.tanh(self.cell.raw_b).item()
                    msg += f" | b: {b_val:.4f}" 
                print(msg)
                print(f"| corr(Z,X) ={corr:.4f} | |Y-Y*|^2/|Y*|^2 ={Diffv[i]:.4f} | ")
                print("-" * 90)
                     
        Y_valid = self.Y[self.t0:]
        Diff =  ((Ystar - Y_valid.detach().numpy().reshape(-1))**2).mean()
        print(" ")
        print("---Final Analysis---")
        print(f"Equilibrium V = |Pred|^2={EqVal:.4f} ")
        print(f"Total Loss: {loss:.5f} | MSE: {mse:.5f} | Penalty: {coupling:.5f}")
        print(f"Squared Relative Error |Y-Y*|^2/|Y*|^2 ={Diff.item()/(Ystar**2).mean():.4f} ")
        
        x_range = range(steps)
        Value = np.full(steps, EqVal)

        fig, axs = plt.subplots(2, 1, sharex=True) 

        axs[0].plot(x_range, MSEv, label='MSE')
        axs[0].plot(x_range, Lossv, label='MSE+coupling')
        axs[0].plot(x_range, Value, label='Actual value')
        axs[0].set_ylabel('Value')
        axs[0].set_title(f"MSE and total loss during training with {self.data_mode} data")
        axs[0].legend()

        axs[1].plot(x_range, Diffv, label='Squared Relative Error for Y')
        axs[1].set_xlabel('Training steps')
        axs[1].set_ylabel('|Y-Y*|^2/|Y*|^2')
        axs[1].legend()
        plt.show()


# 5. Main
if __name__ == "__main__":
    
    T = 3000
    burn_in = 500
    
    # 1. Select data mode and extract it with additional data for printouts
    mode_data = "GARCH"
    print(" ")
    print(f"Input data: {mode_data}")
    if mode_data == "Kalman":
        #Xsim, Ztrue = gd.generate_kalman_data(T, 0.8, 1, 0.4, 0.4)
        Xsim, Ztrue = gd.simulate_stochastic_volatility(T, 0, 0.8, 0.4, obs_noise = 0.4)
        X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),  0.8, 1, 0.4, 0.4)
        epsi = X.detach().numpy().reshape(-1) - P
    elif mode_data == "GARCH":
        Xsim, Ztrue, epsi = gd.generate_garch_data(T,0.05,0.20,0.79)
        X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
        P = np.sqrt(Ztrue) - np.sqrt(Ztrue).mean()
    else:
        raise ValueError(f"Unknown data mode: {mode_data}")
    

    # 2. Select G mode. Options: "LINEAR" or "GRU"
    mode_G = "GRU"  
    if mode_G == "LINEAR":
        cell = VectorizedLinearCell()
        runner = FastRecurrentRunner(cell, hidden_dim=1)
        show_b = True
    else: 
        hidden_dim = 2
        cell = VectorizedGRUCell(input_dim=1, hidden_dim=hidden_dim)
        runner = FastRecurrentRunner(cell, hidden_dim=hidden_dim)
        show_b = False
    print(f"G type: {mode_G}")
    
    # 3. Select Psi function space. Options: 'identity', 'poly', 'neural'
    mode_Psi = "poly" 
    print(f"Test function type: {mode_Psi}")
    print(" ")
    
    scripted_runner = torch.jit.script(runner)
    
    solver = UnifiedMaxMinSolver(X, cell, scripted_runner, 
                                 P, epsi,
                                 psi_mode=mode_Psi,
                                 data_mode=mode_data,
                                 burn_in=burn_in, 
                                 verbose_b=show_b)
    
    solver.train(steps=300)