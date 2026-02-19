import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import GenerateData as gd

# ==========================================
# 1. Models (With Manual Second-Order AD)
# ==========================================

class NeuralG(nn.Module):
    def __init__(self, hidden_dim=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dims = {
            'w_ih': (hidden_dim, 1), 'w_hh': (hidden_dim, hidden_dim),
            'b_h':  (hidden_dim,), 'w_out': (1, hidden_dim)
        }
        self.offsets = {}
        curr = 0
        self.offsets['w_ih'] = (curr, curr + hidden_dim); curr += hidden_dim
        self.offsets['w_hh'] = (curr, curr + hidden_dim**2); curr += hidden_dim**2
        self.offsets['b_h']  = (curr, curr + hidden_dim); curr += hidden_dim
        self.offsets['w_out']= (curr, curr + hidden_dim); curr += hidden_dim
        self.total_params = curr
        self.flat_params = nn.Parameter(torch.randn(self.total_params) * 0.2)

    def unpack_params(self, flat_params):
        s = self.offsets; d = self.dims
        w_ih = flat_params[s['w_ih'][0]:s['w_ih'][1]].view(d['w_ih'])
        w_hh = flat_params[s['w_hh'][0]:s['w_hh'][1]].view(d['w_hh'])
        b_h  = flat_params[s['b_h'][0]:s['b_h'][1]].view(d['b_h'])
        w_out= flat_params[s['w_out'][0]:s['w_out'][1]].view(d['w_out'])
        return w_ih, w_hh, b_h, w_out

    def forward_trajectory_only(self, X, params=None):
        """
        LIGHTWEIGHT Forward Pass.
        Computes ONLY Z. No Jacobian, No Hessian.
        Used for Line Search checks.
        """
        if params is None: params = self.flat_params
        w_ih, w_hh, b_h, w_out = self.unpack_params(params)
        T = X.shape[0]
        h = torch.zeros(self.hidden_dim, device=X.device)
        Z_list = []
        
        for t in range(T):
            # Readout
            z_t = F.linear(h.unsqueeze(0), w_out).squeeze(0)
            Z_list.append(z_t)
            
            # Update
            x_t = X[t].view(1, 1)
            lin_ih = F.linear(x_t, w_ih)             
            lin_hh = F.linear(h.unsqueeze(0), w_hh)  
            a = lin_ih + lin_hh + b_h                
            h = torch.tanh(a).squeeze(0)
            
        return torch.stack(Z_list).squeeze()

    def forward_second_order(self, X, params=None, threshold=1e-4):
        """
        HEAVY Forward Pass.
        Computes Z, Jacobian (J_Z), and Hessian (H_Z).
        Used for Update Step calculation.
        """
        if params is None: params = self.flat_params
        w_ih, w_hh, b_h, w_out = self.unpack_params(params)
        T = X.shape[0]; H = self.hidden_dim; N = self.total_params
        
        # States
        h = torch.zeros(H, device=X.device)
        Jh = torch.zeros(H, N, device=X.device)
        Hh = torch.zeros(H, N, N, device=X.device)
        
        Z_list, J_list, H_list = [], [], []
        
        # Indices
        s = self.offsets
        idx_ih = s['w_ih']; idx_hh = s['w_hh']; idx_bh = s['b_h']; idx_out = s['w_out']
        
        accumulated_decay = 1.0
        
        for t in range(T):
            # --- Dynamic Truncation ---
            if accumulated_decay < threshold:
                h = h.detach()
                Jh = Jh.detach(); Jh.zero_()
                Hh = Hh.detach(); Hh.zero_()
                accumulated_decay = 1.0

            # --- 1. READOUT Z ---
            z_t = F.linear(h.unsqueeze(0), w_out).squeeze(0)
            Z_list.append(z_t)
            
            # Jacobian J_z
            Jz_t = w_out @ Jh # [1, N]
            Jz_t[0, idx_out[0]:idx_out[1]] += h
            J_list.append(Jz_t)
            
            # Hessian H_z
            # d^2z/dw^2 = w_out * Hh + CrossTerms
            Hz_t = torch.einsum('ij,jkl->ikl', w_out, Hh) # [1, N, N]
            
            # Add Cross Terms (Symmetric)
            # Row p_idx gets Jh[j], Col p_idx gets Jh[j]
            # Manual loop is faster than scatter for small H
            for j in range(H):
                p_idx = idx_out[0] + j
                Hz_t[0, p_idx, :] += Jh[j, :]
                Hz_t[0, :, p_idx] += Jh[j, :]
                
            H_list.append(Hz_t)
            
            # --- 2. UPDATE STATE ---
            x_t = X[t].view(1, 1)
            lin_ih = F.linear(x_t, w_ih)             
            lin_hh = F.linear(h.unsqueeze(0), w_hh)  
            a = lin_ih + lin_hh + b_h                
            h_new = torch.tanh(a).squeeze(0)
            
            # Derivatives of nonlinearity
            sig_prime  = (1 - h_new**2)       # [H]
            sig_double = -2 * h_new * sig_prime # [H]
            
            local_contraction = sig_prime.abs().max() * w_hh.norm()
            accumulated_decay *= local_contraction.item()
            
            # --- 3. JACOBIAN UPDATE ---
            Ja = w_hh @ Jh
            Ja[:, idx_ih[0]:idx_ih[1]] += X[t] * torch.eye(H, device=X.device)
            for i in range(H):
                start = idx_hh[0] + i * H
                Ja[i, start : start + H] += h
            Ja[:, idx_bh[0]:idx_bh[1]] += torch.eye(H, device=X.device)
            
            Jh_new = sig_prime.view(H, 1) * Ja
            
            # --- 4. HESSIAN UPDATE ---
            # Linear part: w_hh * Hh
            Ha = torch.einsum('ij,jkl->ikl', w_hh, Hh)
            
            # Cross Terms for W_hh
            for i in range(H):
                for j in range(H):
                    p_idx = idx_hh[0] + i*H + j
                    Ha[i, p_idx, :] += Jh[j, :]
                    Ha[i, :, p_idx] += Jh[j, :]

            # Combine: H_h = sig' * Ha  +  sig'' * (Ja outer Ja)
            term1 = sig_prime.view(H, 1, 1) * Ha
            term2 = sig_double.view(H, 1, 1) * torch.einsum('bi,bj->bij', Ja, Ja)
            
            Hh = term1 + term2
            
            h = h_new
            Jh = Jh_new
            
        return (torch.stack(Z_list).squeeze(), 
                torch.stack(J_list).squeeze(), 
                torch.stack(H_list).squeeze())

class NeuralPsi(nn.Module):
    def __init__(self, hidden_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.total_params = (hidden_dim * 1) + hidden_dim + (hidden_dim * 1) + 1
        self.flat_params = nn.Parameter(torch.randn(self.total_params) * 0.2)
        
    def unpack_params(self, flat_params):
        idx = 0
        w1 = flat_params[idx : idx + self.hidden_dim].view(self.hidden_dim, 1)
        idx += self.hidden_dim
        b1 = flat_params[idx : idx + self.hidden_dim].view(self.hidden_dim)
        idx += self.hidden_dim
        w2 = flat_params[idx : idx + self.hidden_dim].view(1, self.hidden_dim)
        idx += self.hidden_dim
        b2 = flat_params[idx : idx + 1].view(1)
        return w1, b1, w2, b2

    def forward(self, Y, params=None):
        if params is None: params = self.flat_params
        w1, b1, w2, b2 = self.unpack_params(params)
        h = F.linear(Y.unsqueeze(1), w1, b1)
        h = torch.tanh(h)
        out = F.linear(h, w2, b2).squeeze()
        return Y + out
        


# ==========================================
# 2. Solver (Optimized)
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
        self.Y = (self.X.clone() + torch.randn_like(self.X)*0.3).detach().requires_grad_(True)
        self.lmbda = lmbda
        self.eta_z_min = eta_z_m 
        self.eta_y_min = eta_y_m 
        self.eta_z = eta_z_i
        self.eta_y = eta_y_i

    def get_w_vector(self):
        return torch.cat([self.G.flat_params, self.Psi.flat_params])

    def split_w(self, w_vector):
        len_g = self.G.flat_params.numel()
        w_g = w_vector[:len_g]
        w_psi = w_vector[len_g:]
        return w_g, w_psi
    
    def compute_L_at(self, w_vector, y_val):
        w_g, w_psi = self.split_w(w_vector)
        # OPTIMIZED CALL: Only compute Z
        Z = self.G.forward_trajectory_only(self.X, w_g)
        psi_val = self.Psi(y_val, w_psi)
        
        mse_val = ((self.X - y_val)**2).mean()
        Z_norm_val = Z - Z.mean()
        corr_val = (Z_norm_val * psi_val).mean()
        return mse_val + self.lmbda * corr_val

    def update_step(self, eta_z, eta_y, grad_min):
        w_current = self.get_w_vector().detach()
        w_g, w_psi = self.split_w(w_current)
        
        # 1. Forward Pass (Second Order)
        # Returns [T], [T, Ng], [T, Ng, Ng]
        Z, J_Z, H_Z = self.G.forward_second_order(self.X, w_g, threshold=1e-3)
        Z_centered = Z - Z.mean()
        J_Z_centered = J_Z - J_Z.mean(dim=0, keepdim=True)
        H_Z_centered = H_Z - H_Z.mean(dim=0, keepdim=True)
        
        # 2. Psi & Derivatives (Cheap Autograd)
        Y_detached = self.Y.detach().requires_grad_(True)
        # We need Psi values and gradients w.r.t Y
        psi_vals = self.Psi(Y_detached, w_psi)
        d_psi_dy = torch.autograd.grad(psi_vals.sum(), Y_detached, create_graph=True)[0]
        d2_psi_dy2 = torch.autograd.grad(d_psi_dy.sum(), Y_detached, create_graph=False)[0]
        
        # Jacobian of Psi w.r.t w_psi
        def get_d_psi_dy_grad(p):
            # 1. Forward pass with the specific parameters 'p'
            psi_val = self.Psi(Y_detached, p)
            
            # 2. Compute first derivative w.r.t Y (create_graph=True to allow higher derivatives)
            # We sum psi_val because autograd.grad requires a scalar output
            d_psi = torch.autograd.grad(psi_val.sum(), Y_detached, create_graph=True)[0]
            return d_psi #It's a 1xT vector

        # Compute Jacobian of the derivative w.r.t parameters
        J_Psi_w = torch.autograd.functional.jacobian(get_d_psi_dy_grad, w_psi) #it's a TxNpsi vector
        
        # 3. Construct Gradient (algebraic): grad_G = lambda * E[Psi * J_Z]
        grad_w_G = self.lmbda * (psi_vals.detach().unsqueeze(1) * J_Z_centered).mean(dim=0)
        
        # grad_Psi = lambda * mean( Z * J_Psi )
        grad_w_Psi = self.lmbda * (Z_centered.unsqueeze(1) * J_Psi_w).mean(dim=0)
        
        grad_w = torch.cat([grad_w_G, grad_w_Psi])
        grad_y = -(2.0/self.T)*(self.X - self.Y) + (self.lmbda/self.T)*Z_centered*d_psi_dy.detach()
        
        # 4. Construct Hessian L_ww (algebraic)
        # Block L_GG: lambda * mean( Psi * H_Z )
        L_GG = self.lmbda * (psi_vals.detach().view(-1, 1, 1) * H_Z_centered).mean(dim=0)
        
        # Block L_GPsi: lambda * mean( J_Z^T * J_Psi )
        L_GPsi = (self.lmbda / self.T) * (J_Z_centered.t() @ J_Psi_w)

        # Define the scalar correlation objective just for Psi
        def psi_corr_obj(p):
            vals = self.Psi(Y_detached, p)
            return self.lmbda * (Z_centered.detach() * vals).mean()
        
        # Block L_PsiPsi: compute exact Hessian of this scalar objective w.r.t w_psi
        L_PsiPsi = torch.autograd.functional.hessian(psi_corr_obj, w_psi)
        
        # Assemble L_ww
        row1 = torch.cat([L_GG, L_GPsi], dim=1)
        row2 = torch.cat([L_GPsi.t(), L_PsiPsi], dim=1)
        L_ww = torch.cat([row1, row2], dim=0)

        # 5. L_yw (algebraic)
        diag_correction = (self.lmbda / self.T) * Z_centered.detach() * d2_psi_dy2.detach()
        alpha_vec = (2.0 / self.T) + diag_correction
        
        vec_G = (self.lmbda / self.T) * d_psi_dy.detach() 
        L_yw_G = J_Z_centered * vec_G.unsqueeze(1)
        vec_Z = (self.lmbda / self.T) * Z_centered 
        L_yw_Psi = J_Psi_w * vec_Z.unsqueeze(1) 
        L_yw = torch.cat([L_yw_G, L_yw_Psi], dim=1) 
        
        # ==========================================================
        # 6. Solver Loop
        # ==========================================================
        gw = grad_w
        gy = grad_y.detach()
        Lww = L_ww
        Lyw = L_yw
        inv_Lyy_diag = 1.0 / (alpha_vec + 1e-6)
        
        accept = False;
        dw = torch.zeros_like(gw) 
        dy = torch.zeros_like(gy)
        
        mse = ((self.X - self.Y)**2).mean()
        corr = (Z_centered * psi_vals.detach()).mean()
        L_curr = mse + self.lmbda * corr
        
        while (not accept) and (eta_y + eta_z) > (self.eta_y_min + self.eta_z_min):
            eta_z = max(eta_z, self.eta_z_min)
            eta_y = max(eta_y, self.eta_y_min)
            
            scaled_gy = gy * inv_Lyy_diag
            term_correction = Lyw.t() @ scaled_gy
            D_w = -eta_z * (gw - term_correction)
            D_y = eta_y * gy
            
            scaled_Lyw = Lyw.t() * inv_Lyy_diag.unsqueeze(0)
            term_cov = scaled_Lyw @ Lyw
            block_inner = Lww - term_cov
            H_11 = torch.eye(len(w_current), device=self.device) - eta_z * block_inner
            
            damping = 1e-6 * torch.eye(len(w_current), device=self.device)
            try:
                dw = torch.linalg.solve(H_11 + damping, -D_w)
            except RuntimeError:
                dw = torch.linalg.pinv(H_11 + damping) @ (-D_w)
            
            rhs_y = -D_y - (eta_y * Lyw @ dw)
            lhs_diag = 1.0 + eta_y * alpha_vec
            dy = rhs_y / lhs_diag
            
            
            if torch.norm(dw) > 0.05: dw *= (0.05 / (torch.norm(dw)+1e-8))
            if torch.norm(dy) > 0.05: dy *= (0.05 / (torch.norm(dy)+1e-8))
            
            w_new = w_current + dw
            y_new = self.Y.detach() + dy
           
            
            L_new_w_old = self.compute_L_at(w_new, self.Y.detach())
            L_new_w_new = self.compute_L_at(w_new, y_new)
            
            accept_y = (L_new_w_new <= L_new_w_old)
            accept_z = (L_new_w_new >= L_curr) or (torch.norm(gy) > grad_min)
            
            # Total Accept
            accept = (accept_z and accept_y) 
            
            if not accept:
                if not accept_z:
                    eta_z = 0.5 * eta_z
                else:
                    eta_z = 0.9 * eta_z
                if not accept_y:
                    eta_y = 0.5 * eta_y
                else:
                    eta_y = 0.9 * eta_y
                    
         
        if accept:
            # Slight growth on success
            eta_z *= 1.1 
            eta_y *= 1.1

            with torch.no_grad():
                len_g = self.G.flat_params.numel()
                self.G.flat_params.copy_(w_new[:len_g])
                self.Psi.flat_params.copy_(w_new[len_g:])
                self.Y.copy_(y_new)
        
        return L_curr.item(), mse.item(), corr.item(), gw.norm().item(), gy.norm().item(), dw.norm().item(), dy.norm().item(), accept, eta_z, eta_y

    def train(self, steps=100):
        print(f"{'Step':<5} | {'Loss':<10} | {'MSE':<10} | {'Corr':<10} | {'|GradW|':<10} | {'|GradY|':<10} | {'Accept'}")
        print("-" * 95)
        grad_min = 0.0001
        eta_z, eta_y = self.eta_z, self.eta_y
        for i in range(1, steps+1):
            L, mse, corr, nGw, nGy, ndw, ndy, accept, eta_z, eta_y = self.update_step(eta_z, eta_y, grad_min)
            if i % 1 == 0:
                print(f"{i:<5d} | {L:.6f}   | {mse:.6f}   | {corr:.6f}   | {nGw:.4f}     | {nGy:.4f}     | {accept}")

if __name__ == "__main__":
    T = 2000
    eta_z_m = T * 0.005
    eta_y_m = T * 0.005
    eta_z_i = T * 0.01
    eta_y_i = T * 0.01
    lmbda = 1.5
    Xsim, Ztrue = gd.generate_kalman_data(T, 0.8, 1.0, 0.4, 0.4)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
    G = NeuralG(hidden_dim=5)   
    Psi = NeuralPsi(hidden_dim=10) 
    solver = HessianSolver(X, G, Psi, lmbda=lmbda, eta_z_m=eta_z_m, eta_y_m=eta_y_m,eta_z_i=eta_z_i, eta_y_i=eta_y_i)
    print("\nStarting Training with Manual Second-Order AD...")
    solver.train(steps=100)