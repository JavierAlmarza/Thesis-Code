import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. The Non-Gaussian Stochastic Process
# ==========================================
class NonGaussianProcess:
    def __init__(self, phi=0.8, sigma_eta=1.0, batch_size=64, seq_len=50):
        self.phi = phi
        self.sigma_eta = sigma_eta
        self.batch_size = batch_size
        self.seq_len = seq_len
        # Expected value of log(chi_sq_1) is approx -1.2704
        self.euler_gamma = 0.57721
        self.log_chi_sq_mean = -self.euler_gamma - np.log(2)

    def generate_batch(self):
        # State h_t
        h = torch.zeros(self.batch_size, 1) 
        # Container for x_t
        x_seq = []
        
        # Burn-in to reach stationarity
        with torch.no_grad():
            for _ in range(40):
                eta = torch.randn(self.batch_size, 1)
                h = self.phi * h + self.sigma_eta * eta

            # Generate actual sequence
            for _ in range(self.seq_len):
                # Update State: h_{t+1} = phi * h_t + eta_t
                eta = torch.randn(self.batch_size, 1)
                h = self.phi * h + self.sigma_eta * eta
                
                # Observation Noise: zeta_t = log(eps^2) - E[...]
                # eps ~ N(0,1)
                eps = torch.randn(self.batch_size, 1)
                # Avoid log(0) by adding epsilon
                log_eps_sq = torch.log(eps**2 + 1e-10)
                zeta = log_eps_sq - self.log_chi_sq_mean
                
                x_t = h + zeta
                x_seq.append(x_t)
        
        # Shape: (Batch, Seq, 1)
        return torch.stack(x_seq, dim=1)

# ==========================================
# 2. The Models (RNNs & MLPs)
# ==========================================

# Generator y_t: Causal RNN
# Input: x_{0:t}, Output: y_t
class Generator(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, 1)
        # out: (Batch, Seq, Hidden)
        out, _ = self.rnn(x)
        # y: (Batch, Seq, 1)
        y = self.head(out) 
        return y

# Adversary f_1: Causal RNN (Strictly Past)
# It must be measurable wrt x_{<= t-1}.
# We implement this by shifting the input or hidden states.
class AdversaryPast(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, 1)
        # We want output at index t to depend on x_{0...t-1}.
        # Standard RNN output at t depends on x_{0...t}.
        # So we shift inputs to the right: [0, x_0, x_1, ... x_{T-1}]
        
        batch_size, seq_len, _ = x.shape
        # Pad with zero at the start (t=0 has no past)
        padding = torch.zeros(batch_size, 1, 1, device=x.device)
        x_shifted = torch.cat([padding, x[:, :-1, :]], dim=1)
        
        out, _ = self.rnn(x_shifted)
        f1 = self.head(out)
        return f1

# Adversary f_2: Smooth MLP
# Input: y_t (current prediction), Output: Scalar
class AdversaryPresent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(), # Smooth, bounded-ish
            nn.Linear(16, 1)
        )

    def forward(self, y):
        return self.net(y)

# ==========================================
# 3. The Objective Functions
# ==========================================

def soft_hgr_loss(f1, f2):
    """
    Computes the Soft-HGR maximization objective.
    Goal for Max Player: Maximize this.
    Value converges to 0.5 * rho^2
    """
    # f1, f2 shape: (Batch, Seq, 1) -> Flatten to (Batch*Seq, 1)
    f1_flat = f1.reshape(-1, 1)
    f2_flat = f2.reshape(-1, 1)
    
    term_cov = torch.mean(f1_flat * f2_flat)
    term_var1 = 0.5 * torch.mean(f1_flat**2)
    term_var2 = 0.5 * torch.mean(f2_flat**2)
    
    # Objective to MAXIMIZE
    return term_cov - term_var1 - term_var2

def total_loss(x, y, f1, f2, lambda_reg):
    mse = torch.mean((x - y)**2)
    hgr = soft_hgr_loss(f1, f2)
    
    # For Generator (Minimizer): Min MSE + lambda * HGR
    # Note: Generator wants HGR to be LOW (indep).
    # But HGR is defined as a maximization problem for f.
    # This is the saddle point: Min_y ( MSE + lambda * Max_f (HGR_obj) )
    
    return mse, hgr

# ==========================================
# 4. The Experiment Logic
# ==========================================

def run_experiment(mode="primal", steps=2000, lambda_reg=10.0, k_loops=5):
    """
    mode 'primal': MinMax (Inner loop maximizes f). Checks Independence.
    mode 'dual': MaxMin (Inner loop minimizes y). Checks Gap.
    """
    process = NonGaussianProcess()
    
    gen = Generator()
    adv_past = AdversaryPast()
    adv_pres = AdversaryPresent()
    
    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=1e-3)
    opt_adv = optim.Adam(list(adv_past.parameters()) + list(adv_pres.parameters()), lr=1e-3)
    
    history_mse = []
    history_hgr = []
    
    print(f"--- Running {mode.upper()} Optimization (Lambda={lambda_reg}) ---")
    
    for step in range(steps):
        
        # --- PRIMAL MODE (MinMax) ---
        # "Strict Independence": We ensure f is optimal (detects all correlations)
        # before y steps.
        if mode == "primal":
            # 1. Inner Loop: Maximize Adversary (f)
            for _ in range(k_loops):
                x = process.generate_batch()
                y = gen(x).detach() # y is fixed for f
                f1 = adv_past(x)
                f2 = adv_pres(y)
                
                loss_hgr = soft_hgr_loss(f1, f2)
                # We want to MAXIMIZE hgr, so minimize negative
                loss_adv = -loss_hgr 
                
                opt_adv.zero_grad()
                loss_adv.backward()
                opt_adv.step()
            
            # 2. Outer Loop: Minimize Generator (y)
            x = process.generate_batch()
            y = gen(x)
            f1 = adv_past(x).detach() # f is fixed for y
            f2 = adv_pres(y) # f2 gradients flow through y
            
            mse = torch.mean((x - y)**2)
            hgr_val = soft_hgr_loss(f1, f2)
            
            # Generator fights the adversary
            loss_gen = mse + lambda_reg * hgr_val
            
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # --- DUAL MODE (MaxMin) ---
        # "Weak Constraint": We ensure y is optimal (finds best MSE for current f)
        # before f moves.
        elif mode == "dual":
            # 1. Inner Loop: Minimize Generator (y)
            for _ in range(k_loops):
                x = process.generate_batch()
                # f is fixed
                f1 = adv_past(x).detach()
                
                # In Dual, y sees f as a static penalty field
                y = gen(x)
                f2 = adv_pres(y) # f2 is fixed function, but input is y
                
                mse = torch.mean((x - y)**2)
                hgr_val = soft_hgr_loss(f1, f2)
                
                loss_gen = mse + lambda_reg * hgr_val
                
                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
                
            # 2. Outer Loop: Maximize Adversary (f)
            x = process.generate_batch()
            y = gen(x).detach()
            f1 = adv_past(x)
            f2 = adv_pres(y)
            
            loss_hgr = soft_hgr_loss(f1, f2)
            loss_adv = -loss_hgr
            
            opt_adv.zero_grad()
            loss_adv.backward()
            opt_adv.step()

        # Logging
        if step % 100 == 0:
            history_mse.append(mse.item())
            history_hgr.append(hgr_val.item())
            print(f"Step {step}: MSE={mse.item():.4f}, SoftHGR={hgr_val.item():.4f}")

    return np.mean(history_mse[-10:]), np.mean(history_hgr[-10:])

# ==========================================
# 5. Execute Comparison
# ==========================================
if __name__ == "__main__":
    # Parameters
    # Lambda should be high enough to force the constraint
    LAMBDA = 15.0 
    STEPS = 1500
    
    print("Starting Duality Gap Check...")
    
    # 1. Run Primal (MinMax) - The Hard Problem
    mse_primal, hgr_primal = run_experiment("primal", steps=STEPS, lambda_reg=LAMBDA, k_loops=10)
    
    # 2. Run Dual (MaxMin) - The Easier Problem
    mse_dual, hgr_dual = run_experiment("dual", steps=STEPS, lambda_reg=LAMBDA, k_loops=10)
    
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Primal MSE (MinMax): {mse_primal:.5f} | HGR Residual: {hgr_primal:.5f}")
    print(f"Dual MSE   (MaxMin): {mse_dual:.5f} | HGR Residual: {hgr_dual:.5f}")
    
    gap = mse_primal - mse_dual
    print(f"Estimated Duality Gap: {gap:.5f}")
    
    if gap > 0.05: # Threshold depends on noise magnitude
        print(">> Positive Gap Detected: Constraint is Non-Convex.")
    else:
        print(">> No significant gap (or lambda too small).")