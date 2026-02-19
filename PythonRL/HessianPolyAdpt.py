import numpy as np
import torch
import torch.nn as nn
import GenerateData as gd

# ==========================================
# 1. Polynomial Model G (Strictly Causal & Symmetric)
# ==========================================

class PolynomialG(nn.Module):
    def __init__(self):
        super().__init__()
        # h_t = tanh(w1)*x_{t-1} + tanh(w2)*h_{t-1} + w3*x_{t-1}^2 + w4*h_{t-1}^2
        # Initialize near a stable linear regime
        self.params = nn.Parameter(torch.tensor([0.2, 0.2, 0.0, 0.0]))
        self.num_params = 4

    def forward_trajectory_only(self, X, params=None):
        if params is None: params = self.params
        w1_raw, w2_raw, w3, w4 = params[0], params[1], params[2], params[3]
        
        # Apply tanh constraints
        w1 = torch.tanh(w1_raw)
        w2 = torch.tanh(w2_raw) 
        
        T = X.shape[0]
        h = torch.zeros((), device=X.device) 
        
        Z_list = [h] # Causal padding
        
        for t in range(T - 1):
            x = X[t]
            h = w1*x + w2*h + w3*(x**2) + w4*(h**2)
            Z_list.append(h)
            
        return torch.stack(Z_list)

    def forward_second_order(self, X, params=None, threshold=1e-4):
        if params is None: params = self.params
        w1_raw, w2_raw, w3, w4 = params
        
        # Chain Rule components
        w1 = torch.tanh(w1_raw)
        dtanh1 = 1.0 - w1**2
        d2tanh1 = -2.0 * w1 * dtanh1
        
        w2 = torch.tanh(w2_raw)
        dtanh2 = 1.0 - w2**2
        d2tanh2 = -2.0 * w2 * dtanh2
        
        T = X.shape[0]
        N = self.num_params
        
        h = torch.zeros((), device=X.device)      
        Jh = torch.zeros(N, device=X.device)      
        Hh = torch.zeros(N, N, device=X.device)   
        
        Z_list = [h]
        J_list = [Jh.clone()]
        H_list = [Hh.clone()]
        
        accumulated_decay = 1.0

        for t in range(T - 1):
            x = X[t]
            h_prev = h
            Jh_prev = Jh.clone()
            Hh_prev = Hh.clone()

            # --- 1. Update Value ---
            h = w1*x + w2*h_prev + w3*(x**2) + w4*(h_prev**2)
            Z_list.append(h)

            # --- 2. Update Jacobian ---
            df_dw = torch.stack([x * dtanh1, h_prev * dtanh2, x**2, h_prev**2])
            df_dh = w2 + 2*w4*h_prev
            
            Jh = df_dw + df_dh * Jh_prev
            J_list.append(Jh)

            # --- 3. Update Hessian (NOW MATHEMATICALLY SYMMETRIC) ---
            
            # A. Pure partials w.r.t parameters (Diagonal matrix)
            H_ww_partial = torch.zeros(N, N, device=X.device)
            H_ww_partial[0, 0] = x * d2tanh1
            H_ww_partial[1, 1] = h_prev * d2tanh2
            
            # B. Gradient of the scalar multiplier (df_dh) w.r.t parameters
            grad_df_dh = torch.zeros(N, device=X.device)
            grad_df_dh[1] = dtanh2
            grad_df_dh[3] = 2.0 * h_prev
            
            # C. The Missing Outer Product Symmetry
            outer_term = torch.outer(grad_df_dh, Jh_prev)
            
            # The flawless, perfectly symmetric rank-2 update:
            Hh = H_ww_partial + outer_term + outer_term.t() + (df_dh * Hh_prev)
            
            H_list.append(Hh)

            # --- Truncation ---
            accumulated_decay *= abs(df_dh.item())
            if accumulated_decay < threshold:
                h = h.detach()
                Jh = Jh.detach(); Jh.zero_()
                Hh = Hh.detach(); Hh.zero_()
                accumulated_decay = 1.0

        return (torch.stack(Z_list), 
                torch.stack(J_list), 
                torch.stack(H_list))

# ==========================================
# 2. Solver (Identity Psi)
# ==========================================

class HessianSolver:
    def __init__(self, X, G_net, lmbda=1.0, 
                 eta_z_m=0.01, eta_y_m=0.01, 
                 eta_z_i=0.1, eta_y_i=0.1, 
                 device='cpu'):
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.device = device
        self.G = G_net.to(device)
        
        self.Y = (torch.randn_like(self.X, device=device) * 0.2 + 0.3).requires_grad_(True)
        
        self.lmbda = lmbda
        self.eta_z_min = eta_z_m 
        self.eta_y_min = eta_y_m 
        self.eta_z = eta_z_i
        self.eta_y = eta_y_i

    def get_w_vector(self):
        return self.G.params
    
    def compute_L_at(self, w_vector, y_val):
        Z = self.G.forward_trajectory_only(self.X, w_vector)
        psi_val = y_val 
        
        mse_val = ((self.X - y_val)**2).mean()
        Z_norm_val = Z - Z.mean()
        corr_val = (Z_norm_val * psi_val).mean()
        return mse_val + self.lmbda * corr_val

    def update_step(self, eta_z, eta_y, grad_min):
        w_current = self.get_w_vector().detach().requires_grad_(True)
        w_g = w_current
        
        # --- 1. Forward Pass G ---
        Z, J_Z, H_Z = self.G.forward_second_order(self.X, w_g)
        Z_centered = Z - Z.mean()
        J_Z_centered = J_Z - J_Z.mean(dim=0, keepdim=True)
        H_Z_centered = H_Z - H_Z.mean(dim=0, keepdim=True)
        
        # --- 2. Identity Psi Derivatives ---
        Y_detached = self.Y.detach().requires_grad_(True)
        psi_vals = Y_detached
        d_psi_dy = torch.ones_like(Y_detached)
        d2_psi_dy2 = torch.zeros_like(Y_detached)
        
        # --- 3. Construct Gradient ---
        grad_w = self.lmbda * (psi_vals.detach().unsqueeze(1) * J_Z_centered).mean(dim=0)
        grad_y = -(2.0/self.T)*(self.X - self.Y) + (self.lmbda/self.T)*Z_centered*d_psi_dy.detach()
        
        # --- 4. Construct Hessian L_ww ---
        L_ww = self.lmbda * (psi_vals.detach().view(-1, 1, 1) * H_Z_centered).mean(dim=0)

        # --- 5. L_yw ---
        diag_correction = (self.lmbda / self.T) * Z_centered.detach() * d2_psi_dy2.detach()
        alpha_vec = (2.0 / self.T) + diag_correction
        
        vec_G = (self.lmbda / self.T) * d_psi_dy.detach() 
        L_yw = J_Z_centered * vec_G.unsqueeze(1)
        
        # ==========================================================
        # 6. Solver Loop
        # ==========================================================
        gw = grad_w
        gy = grad_y.detach()
        
        inv_Lyy_diag = 1.0 / (alpha_vec + 1e-8)
        
        accept = False
        dw = torch.zeros_like(gw)
        dy = torch.zeros_like(gy)
        
        mse = ((self.X - self.Y)**2).mean()
        corr = (Z_centered * psi_vals.detach()).mean()
        L_curr = mse + self.lmbda * corr
        
        norm_dw = torch.tensor(0.0)
        norm_dy = torch.tensor(0.0)

        while (not accept) and (eta_y + eta_z) > (self.eta_y_min + self.eta_z_min):
            eta_z = max(eta_z, self.eta_z_min)
            eta_y = max(eta_y, self.eta_y_min)
            
            scaled_gy = gy * inv_Lyy_diag
            term_correction = L_yw.t() @ scaled_gy
            
            D_w = -eta_z * (gw - term_correction)
            D_y = eta_y * gy
            
            scaled_Lyw = L_yw.t() * inv_Lyy_diag.unsqueeze(0)
            term_cov = scaled_Lyw @ L_yw
            block_inner = L_ww - term_cov
            
            H_11 = torch.eye(len(w_current), device=self.device) - eta_z * block_inner
            
            damping = 1e-7 * torch.eye(len(w_current), device=self.device)
            try:
                dw = torch.linalg.solve(H_11 + damping, -D_w)
            except RuntimeError:
                dw = torch.linalg.pinv(H_11 + damping) @ (-D_w)
            
            rhs_y = -D_y - (eta_y * L_yw @ dw)
            lhs_diag = 1.0 + eta_y * alpha_vec
            dy = rhs_y / lhs_diag
            
            max_norm = 0.5
            norm_dw = torch.norm(dw)
            if norm_dw > max_norm: dw *= (max_norm / (norm_dw + 1e-8))
            norm_dy = torch.norm(dy)
            if norm_dy > max_norm: dy *= (max_norm / (norm_dy + 1e-8))
            
            w_new = w_current.detach() + dw
            y_new = self.Y.detach() + dy
            
            L_new_w_old = self.compute_L_at(w_new, self.Y.detach())
            L_new_w_new = self.compute_L_at(w_new, y_new)
            
            accept_y = (L_new_w_new <= L_new_w_old)
            accept_z = (L_new_w_new >= L_curr) or (torch.norm(gy) > grad_min)
            accept = (accept_z and accept_y)
            
            if not accept:
                if not accept_z: eta_z *= 0.5
                else: eta_z *= 0.9
                if not accept_y: eta_y *= 0.5
                else: eta_y *= 0.9
        
        if accept:
            eta_z *= 1.1
            eta_y *= 1.1
            
            with torch.no_grad():
                self.G.params.copy_(w_new)
                self.Y.copy_(y_new)
                
        return L_curr.item(), mse.item(), corr.item(), gw.norm().item(), gy.norm().item(), norm_dw.item(), norm_dy.item(), accept, eta_z, eta_y

    def train(self, steps=100):
        print(f"{'Step':<5} | {'Loss':<10} | {'MSE':<10} | {'Corr':<10} | {'|GradW|':<10} | {'|GradY|':<10} | {'|NormDW|':<10} | {'|NormDY|':<10}")
        print("-" * 105)
        grad_min = 0.0001
        eta_z, eta_y = self.eta_z, self.eta_y
        for i in range(1, steps+1):
            L, mse, corr, nGw, nGy, ndw, ndy, accept, eta_z, eta_y = self.update_step(eta_z, eta_y, grad_min)
            if not accept: break
            if i % 5 == 0 or i == 1:
                 print(f"{i:<5d} | {L:.6f}    | {mse:.6f}    | {corr:.6f}    | {nGw:.4f}      | {nGy:.4f}      | {ndw:.4f}     | {ndy:.4f}")

if __name__ == "__main__":
    T = 3000
    alpha = 0.8
    beta = 1.0
    tau_x = 0.4
    tau_z = 0.4
    Xsim, Ztrue = gd.generate_kalman_data(T, alpha, beta, tau_x, tau_z)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
    
    G = PolynomialG()     
    
    lmbda = 1.0
    eta_z_m = T * 0.001
    eta_y_m = T * 0.003
    eta_z_i = T * 0.05
    eta_y_i = T * 0.1
    
    solver = HessianSolver(X, G, lmbda=lmbda, 
                           eta_z_m=eta_z_m, eta_y_m=eta_y_m,
                           eta_z_i=eta_z_i, eta_y_i=eta_y_i)
                           
    print("\nStarting Training with Mathematically Symmetric Manual AD...")
    solver.train(steps=100)
    
    with torch.no_grad():
        Y_final = solver.Y
        
        P = gd.kalman_predictors(X.detach().numpy(), alpha, beta, tau_x, tau_z)
        True_Noise = X.numpy() - P
        
        err = ((Y_final.numpy() - True_Noise)**2).mean()
        true_var = (True_Noise**2).mean()
        
        print("\n--- Final Results ---")
        print(f"Equilibrium value |Pred|^2: {(P**2).mean():.6f}")
        print(f"Squared Relative Error (|Y - Y*|^2 / |Y*|^2): {err/true_var:.6f}")
        print(f"|Y|^2: {(Y_final**2).mean():.6f}")
        print(f"Var(X): {X.numpy().var():.6f}")
        
        w1_raw, w2_raw, w3, w4 = solver.G.params
        w1 = torch.tanh(w1_raw)
        w2 = torch.tanh(w2_raw)
        print(f"\nLearned Params: w1 (Input)={w1.item():.4f}, w2 (Recurrent)={w2.item():.4f}, w3={w3.item():.4f}, w4={w4.item():.4f}")