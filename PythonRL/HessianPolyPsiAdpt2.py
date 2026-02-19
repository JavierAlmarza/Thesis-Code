import numpy as np
import torch
import torch.nn as nn
import GenerateData as gd
import math

# ==========================================
# 1. Taylor-Scaled Polynomial Models (With Interactions)
# ==========================================

class PolynomialG(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        
        # Build valid interaction pairs (i, j) such that i >= 1, j >= 1, and i + j <= degree
        self.cross_indices = [(i, j) for i in range(1, degree) for j in range(1, degree) if i + j <= degree]
        
        self.num_w_x = degree
        self.num_w_h = degree
        self.num_w_c = len(self.cross_indices)
        self.num_params = self.num_w_x + self.num_w_h + self.num_w_c
        
        # Layout: [w_x_1..D, w_h_1..D, w_c_1..C]
        init_vals = torch.zeros(self.num_params)
        init_vals[0] = 0.2          
        init_vals[self.num_w_x] = 0.2     
        self.params = nn.Parameter(init_vals)

    def forward_trajectory_only(self, X, params=None):
        if params is None: params = self.params
        D = self.degree
        
        w = torch.tanh(params)
        w_x = w[:self.num_w_x]
        w_h = w[self.num_w_x : self.num_w_x + self.num_w_h]
        w_c = w[self.num_w_x + self.num_w_h :]
        
        T = X.shape[0]
        h = torch.zeros((), device=X.device) 
        Z_list = [h] 
        
        for t in range(T - 1):
            x = X[t]
            # 0-indexed power arrays: index d corresponds to (val^d / d!)
            x_pow = [torch.tensor(1.0, device=X.device)] + [(x**d)/math.factorial(d) for d in range(1, D+1)]
            h_pow = [torch.tensor(1.0, device=X.device)] + [(h**d)/math.factorial(d) for d in range(1, D+1)]
            
            h_new = sum(w_x[d-1] * x_pow[d] for d in range(1, D+1)) + \
                    sum(w_h[d-1] * h_pow[d] for d in range(1, D+1))
                    
            if self.num_w_c > 0:
                h_new += sum(w_c[k] * x_pow[i] * h_pow[j] for k, (i, j) in enumerate(self.cross_indices))
                
            h = h_new
            Z_list.append(h)
            
        return torch.stack(Z_list)

    def forward_second_order(self, X, params=None, threshold=1e-4):
        if params is None: params = self.params
        D = self.degree
        N = self.num_params
        
        w_raw = params
        w = torch.tanh(w_raw)
        dtanh = 1.0 - w**2
        d2tanh = -2.0 * w * dtanh
        
        w_x, w_h, w_c = w[:D], w[D:2*D], w[2*D:]
        dtanh_x, dtanh_h, dtanh_c = dtanh[:D], dtanh[D:2*D], dtanh[2*D:]
        d2tanh_x, d2tanh_h, d2tanh_c = d2tanh[:D], d2tanh[D:2*D], d2tanh[2*D:]
        
        T = X.shape[0]
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

            x_pow = [torch.tensor(1.0, device=X.device)] + [(x**d)/math.factorial(d) for d in range(1, D+1)]
            h_pow = [torch.tensor(1.0, device=X.device)] + [(h_prev**d)/math.factorial(d) for d in range(1, D+1)]

            # --- 1. Update Value ---
            h_val = sum(w_x[d-1] * x_pow[d] for d in range(1, D+1)) + \
                    sum(w_h[d-1] * h_pow[d] for d in range(1, D+1))
            if self.num_w_c > 0:
                h_val += sum(w_c[k] * x_pow[i] * h_pow[j] for k, (i, j) in enumerate(self.cross_indices))
            h = h_val
            Z_list.append(h)

            # --- 2. Update Jacobian ---
            df_dw = torch.zeros(N, device=X.device)
            for d in range(1, D+1): df_dw[d-1] = x_pow[d] * dtanh_x[d-1]
            for d in range(1, D+1): df_dw[D + d-1] = h_pow[d] * dtanh_h[d-1]
            for k, (i, j) in enumerate(self.cross_indices):
                df_dw[2*D + k] = x_pow[i] * h_pow[j] * dtanh_c[k]
            
            df_dh = sum(w_h[d-1] * h_pow[d-1] for d in range(1, D+1))
            if self.num_w_c > 0:
                df_dh += sum(w_c[k] * x_pow[i] * h_pow[j-1] for k, (i, j) in enumerate(self.cross_indices))
                
            Jh = df_dw + df_dh * Jh_prev
            J_list.append(Jh)

            # --- 3. Update Hessian ---
            V = torch.zeros(N, device=X.device)
            for d in range(1, D+1): V[D + d-1] = h_pow[d-1] * dtanh_h[d-1]
            for k, (i, j) in enumerate(self.cross_indices):
                V[2*D + k] = x_pow[i] * h_pow[j-1] * dtanh_c[k]
            
            H_ww = torch.zeros(N, N, device=X.device)
            for d in range(1, D+1): H_ww[d-1, d-1] = x_pow[d] * d2tanh_x[d-1]
            for d in range(1, D+1): H_ww[D + d-1, D + d-1] = h_pow[d] * d2tanh_h[d-1]
            for k, (i, j) in enumerate(self.cross_indices):
                H_ww[2*D + k, 2*D + k] = x_pow[i] * h_pow[j] * d2tanh_c[k]
            
            f_hh = torch.tensor(0.0, device=X.device)
            if D >= 2:
                f_hh += sum(w_h[d-1] * h_pow[d-2] for d in range(2, D+1))
                if self.num_w_c > 0:
                    f_hh += sum(w_c[k] * x_pow[i] * h_pow[j-2] for k, (i, j) in enumerate(self.cross_indices) if j >= 2)
            
            outer_V_J = torch.outer(V, Jh_prev)
            outer_J_J = torch.outer(Jh_prev, Jh_prev)
            
            Hh = H_ww + outer_V_J + outer_V_J.t() + f_hh * outer_J_J + (df_dh * Hh_prev)
            H_list.append(Hh)

        return (torch.stack(Z_list), 
                torch.stack(J_list), 
                torch.stack(H_list))

class PolynomialPsi(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.num_params = degree
        
        init_vals = torch.zeros(degree)
        init_vals[0] = 1.47 
        self.params = nn.Parameter(init_vals)
        
    def forward(self, Y, params=None):
        if params is None: params = self.params
        w = torch.tanh(params)
        # Taylor Scaled
        Y_pow = torch.stack([(Y**d)/math.factorial(d) for d in range(1, self.degree+1)], dim=1) 
        return (Y_pow * w).sum(dim=1)
        
    def get_derivatives(self, Y, params=None):
        if params is None: params = self.params
        D = self.degree
        w = torch.tanh(params)
        dtanh = 1.0 - w**2
        d2tanh = -2.0 * w * dtanh  # Second derivative of tanh
        
        Y_pow = torch.stack([(Y**d)/math.factorial(d) for d in range(1, D+1)], dim=1)
        psi_vals = (Y_pow * w).sum(dim=1)
        
        d_psi_dy = torch.zeros_like(Y)
        for d in range(1, D+1):
            d_psi_dy += w[d-1] * (Y**(d-1))/math.factorial(d-1)
            
        d2_psi_dy2 = torch.zeros_like(Y)
        for d in range(2, D+1):
            d2_psi_dy2 += w[d-1] * (Y**(d-2))/math.factorial(d-2)
            
        J_Psi_w = Y_pow * dtanh 
        
        # New: Pure second derivative of Psi w.r.t parameters
        H_Psi_ww = Y_pow * d2tanh 
        
        J_PsiPrime_w = torch.zeros_like(J_Psi_w)
        for d in range(1, D+1):
            J_PsiPrime_w[:, d-1] = ((Y**(d-1))/math.factorial(d-1)) * dtanh[d-1]
            
        return psi_vals, d_psi_dy, d2_psi_dy2, J_Psi_w, J_PsiPrime_w, H_Psi_ww

# ==========================================
# 2. Generalized Solver 
# ==========================================

class HessianSolver:
    def __init__(self, X, G_net, Psi_net, lmbda=1.0, 
                 eta_z_m=0.01, eta_y_m=0.01, 
                 eta_z_i=0.1, eta_y_i=0.1, 
                 device='cpu'):
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.device = device
        self.G = G_net.to(device)
        self.Psi = Psi_net.to(device)
        
        self.Y = (torch.randn_like(self.X, device=device) * 0.2 + 0.3)
        
        self.lmbda = lmbda
        self.eta_z_min = eta_z_m 
        self.eta_y_min = eta_y_m 
        self.eta_z = eta_z_i
        self.eta_y = eta_y_i

    def get_w_vector(self):
        return torch.cat([self.G.params, self.Psi.params])

    def split_w(self, w_vector):
        len_g = self.G.num_params
        w_g = w_vector[:len_g]
        w_psi = w_vector[len_g:]
        return w_g, w_psi
    
    def compute_L_at(self, w_vector, y_val):
        w_g, w_psi = self.split_w(w_vector)
        Z = self.G.forward_trajectory_only(self.X, w_g)
        psi_val = self.Psi(y_val, w_psi)
        
        mse_val = ((self.X - y_val)**2).mean()
        Z_norm_val = Z - Z.mean()
        corr_val = (Z_norm_val * psi_val).mean()
        return mse_val + self.lmbda * corr_val

    def update_step(self, eta_z, eta_y, grad_min):
        w_current = self.get_w_vector().detach()
        w_g, w_psi = self.split_w(w_current)
        
        Z, J_Z, H_Z = self.G.forward_second_order(self.X, w_g)
        Z_centered = Z - Z.mean()
        J_Z_centered = J_Z - J_Z.mean(dim=0, keepdim=True)
        H_Z_centered = H_Z - H_Z.mean(dim=0, keepdim=True)
        
        Y_detached = self.Y.detach()
        psi_vals, d_psi_dy, d2_psi_dy2, J_Psi_w, J_PsiPrime_w, H_Psi_ww = self.Psi.get_derivatives(Y_detached, w_psi)
        
        grad_w_G = self.lmbda * (psi_vals.detach().unsqueeze(1) * J_Z_centered).mean(dim=0)
        grad_w_Psi = self.lmbda * (Z_centered.unsqueeze(1) * J_Psi_w).mean(dim=0)
        
        grad_w = torch.cat([grad_w_G, grad_w_Psi])
        grad_y = -(2.0/self.T)*(self.X - self.Y) + (self.lmbda/self.T)*Z_centered*d_psi_dy.detach()
        
        L_GG = self.lmbda * (psi_vals.detach().view(-1, 1, 1) * H_Z_centered).mean(dim=0)
        L_GPsi = (self.lmbda / self.T) * (J_Z_centered.t() @ J_Psi_w)
        
            
        L_PsiPsi_diag = self.lmbda * (Z_centered.unsqueeze(1) * H_Psi_ww).mean(dim=0)
        L_PsiPsi = torch.diag(L_PsiPsi_diag)
        
        row1 = torch.cat([L_GG, L_GPsi], dim=1)
        row2 = torch.cat([L_GPsi.t(), L_PsiPsi], dim=1)
        L_ww = torch.cat([row1, row2], dim=0)

        diag_correction = (self.lmbda / self.T) * Z_centered.detach() * d2_psi_dy2.detach()
        alpha_vec = (2.0 / self.T) + diag_correction
        
        vec_G = (self.lmbda / self.T) * d_psi_dy.detach() 
        L_yw_G = J_Z_centered * vec_G.unsqueeze(1)
        
        # THE FIX: L_yw correctly uses J_PsiPrime_w
        vec_Z = (self.lmbda / self.T) * Z_centered 
        L_yw_Psi = J_PsiPrime_w * vec_Z.unsqueeze(1) 
        
        L_yw = torch.cat([L_yw_G, L_yw_Psi], dim=1) 
        
        # ==========================================================
        # 6. Solver Loop
        # ==========================================================
        gw = grad_w
        gy = grad_y.detach()
        inv_Lyy_diag = 1.0 / alpha_vec 
        
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
            
            damping = 1e-4 * torch.eye(len(w_current), device=self.device)
            try:
                dw = torch.linalg.solve(H_11 + damping, -D_w)
            except RuntimeError:
                dw = torch.linalg.pinv(H_11 + damping) @ (-D_w)
            
            rhs_y = -D_y - (eta_y * L_yw @ dw)
            lhs_diag = 1.0 + eta_y * alpha_vec
            dy = rhs_y / lhs_diag
            
            # Using 0.4 clipping as observed in your logs
            max_norm = 0.4
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
                len_g = self.G.num_params
                self.G.params.copy_(w_new[:len_g])
                self.Psi.params.copy_(w_new[len_g:])
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
    
    mode_data = "GARCH"
    print(" ")
    print(f"Input data: {mode_data}")
    if mode_data == "Kalman":
        Xsim, Ztrue = gd.generate_kalman_data(T, alpha, beta, tau_x, tau_z)
    elif mode_data == "GARCH":
        Xsim, Ztrue, epsi = gd.generate_garch_data(T,0.05,0.20,0.79)
    else:
        raise ValueError(f"Unknown data mode: {mode_data}")

    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)    

    if mode_data == "Kalman":
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),  alpha, beta, tau_x, tau_z)
        Ystar = X.numpy() - P
    elif mode_data == "GARCH":
        P = np.sqrt(Ztrue) - np.sqrt(Ztrue).mean()
        Ystar = (X.numpy() * np.sqrt(Ztrue).mean()) / np.sqrt(Ztrue)

    degree_G = 2
    degree_Psi = 2

    print("Data metrics")
    print(f"Var(X) = {(X.detach().numpy().reshape(-1).std()**2).mean():.4f}")
    print(f"Equilibrium V = |Pred|^2 = {(P**2).mean():.4f}")
    print(f"|Y*|^2 = {(Ystar**2).mean():.4f} ")
    print(" ")

    G = PolynomialG(degree=degree_G)     
    Psi = PolynomialPsi(degree=degree_Psi)
    
    lmbda = 1.0
    eta_z_m = 0.05
    eta_y_m = 0.05
    eta_z_i = T * 0.05
    eta_y_i = T * 0.1
    
    solver = HessianSolver(X, G, Psi, lmbda=lmbda, 
                           eta_z_m=eta_z_m, eta_y_m=eta_y_m,
                           eta_z_i=eta_z_i, eta_y_i=eta_y_i)
                           
    print(f"\nStarting Training (deg_G={degree_G}, deg_Psi={degree_Psi})...")
    print("")
    solver.train(steps=100)
    
    with torch.no_grad():
        Y_final = solver.Y        
        err = ((Y_final.numpy() - Ystar)**2).mean()
        true_var = (Ystar**2).mean()
        
        print("\n--- Final Results ---")
        print(f"Equilibrium value |Pred|^2: {(P**2).mean():.6f}")
        print(f"Squared Relative Error (|Y - Y*|^2 / |Y*|^2): {err/true_var:.6f}")
        print(f"|Y|^2: {(Y_final**2).mean():.6f}")
        print(f"Var(X): {X.numpy().var():.6f}")
        
        # Adjusted unrolling to handle the new cross parameters dynamically
        w_x = solver.G.params[:degree_G].clone()
        w_h = solver.G.params[degree_G:2*degree_G].clone()
        w_c = solver.G.params[2*degree_G:].clone()
        
        w_x_tanh = torch.tanh(w_x)
        w_h_tanh = torch.tanh(w_h)
        
        print(f"\nLearned G Params (w_x): {[round(p.item(), 4) for p in w_x_tanh]}")
        print(f"Learned G Params (w_h): {[round(p.item(), 4) for p in w_h_tanh]}")
        
        if solver.G.num_w_c > 0:
            w_c_tanh = torch.tanh(w_c)
            print(f"Learned G Params (w_c): {[round(p.item(), 4) for p in w_c_tanh]} for pairs {solver.G.cross_indices}")
            
        print(f"Learned Psi Params: {[round(torch.tanh(p).item(), 4) for p in solver.Psi.params]}")