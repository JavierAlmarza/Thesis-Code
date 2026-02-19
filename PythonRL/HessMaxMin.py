import numpy as np
import torch
import torch.nn as nn
import GenerateData as gd


# 1. Linear Cell (contains G)
class LinearCell(nn.Module):
    def __init__(self):
        super().__init__()
        # We optimize raw logits to ensure constraints via tanh
        self.raw_a = nn.Parameter(torch.randn(())) 
        self.raw_b = nn.Parameter(torch.randn(()))
        self.input_dim = 1
        self.hidden_dim = 1 

    def get_params_vector(self):
        return torch.stack([self.raw_a, self.raw_b])

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        return H.squeeze()

# 2. Solver(solves the Maxmin problem)
class HessianSolver:
    def __init__(self, X, cell, lmbda=10.0, eta_z=0.01, eta_y=0.01, device='cpu',split=False):
        self.X = X.to(device).float()
        self.T = X.shape[0]
        self.device = device
        self.split = split
        self.cell = cell.to(device)
        
        # Y is initialized to X
        #self.Y = self.X.clone().detach().requires_grad_(True)
        self.Y = (torch.randn_like(self.X, device=device) * 0.2 + 0.3).requires_grad_(True) #random 
        #self.Y = torch.zeros_like(self.X, requires_grad=True, device=device) #with zero
        
        # Hyperparameters
        self.lmbda = lmbda
        self.eta_z = eta_z 
        self.eta_y = eta_y 

    def forward_Z(self, w_vector=None): 
        # 1) Select source of parameters (None for inference, w_vector for training)
        if w_vector is not None:
            raw_a = w_vector[0]
            raw_b = w_vector[1]
        else:
            raw_a = self.cell.raw_a
            raw_b = self.cell.raw_b
            
        # 2) Apply tanh
        tan_a = torch.tanh(raw_a)
        tan_b = torch.tanh(raw_b)
        a = 1.0 * tan_a 
        b = tan_b 
        
        # Derivatives of transformation functions
        d_a_draw = 1.0 * (1 - tan_a**2)
        d_b_draw = (1 - tan_b**2)

        # 3) Run recurrent loop
        X_in = (self.X * a).unsqueeze(1)
        H_vals = [torch.zeros(1, device=self.device)]
        
        # Jacobian columns tracking dz/da and dz/db
        J_a_vals = [torch.zeros(1, device=self.device)]
        J_b_vals = [torch.zeros(1, device=self.device)]
        curr = torch.zeros(1, device=self.device)
        curr_ja = torch.zeros(1, device=self.device)
        curr_jb = torch.zeros(1, device=self.device)
        
        for t in range(self.T - 1):
            # dz[t]/da = X[t] + b * dz[t-1]/da
            # dz[t]/db = z[t-1] + b * dz[t-1]/db
            z_prev = curr
            curr = X_in[t] + b * z_prev
            curr_ja = self.X[t] + b * curr_ja
            curr_jb = z_prev    + b * curr_jb
            H_vals.append(curr)
            J_a_vals.append(curr_ja)
            J_b_vals.append(curr_jb)
        
        H = torch.cat(H_vals).squeeze()        
        J_a = torch.cat(J_a_vals).squeeze()
        J_b = torch.cat(J_b_vals).squeeze()
        J_raw_a = J_a * d_a_draw
        J_raw_b = J_b * d_b_draw
        
        # Stack to create Tx2 matrix
        J = torch.stack([J_raw_a, J_raw_b], dim=1)

        return H, J

    def update_step(self):
        # 1) Prepare "ghost" parameters for Hessian calculation
        w_current = self.cell.get_params_vector().detach().requires_grad_(True)
        
        # 2) Compute Jacobian and loss L
        Z, J = self.forward_Z(w_current)
        mse = ((self.X - self.Y)**2).mean()
        Z_norm = (Z - Z.mean())
        corr = (Z_norm * self.Y).mean()
        L = mse + self.lmbda * (corr**1) 
        
        # 3) First order gradients (we only need graph for w to compute L_ww)
        grad_w = torch.autograd.grad(L, w_current, create_graph=True, retain_graph=True)[0]
        grad_y = torch.autograd.grad(L, self.Y, create_graph=False, retain_graph=True)[0]
        
        # 4) Second order derivatives        
        # L_ww (2x2)
        L_ww = torch.zeros(2, 2, device=self.device)
        if split:
            L_ww = L_ww
        else:
            for i in range(2):
                grads_i = torch.autograd.grad(grad_w[i], w_current, retain_graph=True)[0]
                L_ww[i] = grads_i

        # L_wy (2xT) and L_yw (Tx2)
        # we use the Jacobian we computed manually, because L_yw = lambda * J (Z_mean=a/(1-b) X_mean=0)
        J_raw= J.detach()
        J_centered = J_raw - J_raw.mean(dim=0, keepdim=True)
        #corrd = corr.detach()
        corrd = 1
        L_yw = (self.lmbda/self.T) *  corrd * J_centered # Detach because we use it as numbers in the update                    

        # L_yy (TxT) is scalar*Id
        alpha = 2.0 / self.T
        
        # 5) Construct gradients and matrices (detach graphs now)
        gw = grad_w.detach() 
        gy = grad_y.detach() 
        Lww = L_ww.detach()  
        Lyw = L_yw #already detached 
        
        # D vector
        term_correction = (1.0 / alpha) * (Lyw.t() @ gy) 
        D_w = -self.eta_z * (gw - term_correction)
        D_y = self.eta_y * gy 
        
        # H blocks
        if split:
            H_11 = 0
            H_12 = self.eta_z * Lyw.t()
        else:
            term_cov = (1.0 / alpha) * (Lyw.t() @ Lyw) 
            block_inner = Lww - term_cov
            H_11 = torch.eye(2, device=self.device) - self.eta_z * block_inner
            H_12 = 0
        H_21 = self.eta_y * Lyw
        scale_H22 = 1.0 + self.eta_y * alpha

        # 6) Solve system H * dx = -D        
        # solve for dw
        if split:
            M = H_12 @ H_21
            # Add a small damping value to the diagonal to prevent explosion
            damping = 1e-7 * torch.eye(2, device=self.device)
            M_damped = M + damping            
            # RHS vector
            rhs_w = scale_H22 * D_w - (H_12 @ D_y)            
            # Solve M * dw = rhs_w
            try:
                dw = torch.linalg.solve(M_damped, rhs_w)
            except RuntimeError:
                dw = torch.linalg.pinv(M_damped) @ rhs_w

        else:
            try:
                dw = torch.linalg.solve(H_11, -D_w)
            except RuntimeError:
                dw = torch.linalg.pinv(H_11) @ (-D_w)
                
            
        # Back-substitute for dy = (-Dy - H21*dw) / s
        rhs_y = -D_y - (H_21 @ dw)
        dy = rhs_y / scale_H22
       
    
        # (After solving for dw and dy, but BEFORE applying them) safety Clipping 
        
        # Calculate norms
        norm_dw = torch.norm(dw)
        norm_dy = torch.norm(dy)
        
        # Clip dw: this prevents the "Singular Matrix" explosion from killing the params
        max_dw_norm = 0.5 
        if norm_dw > max_dw_norm:
            scale_factor = max_dw_norm / (norm_dw + 1e-8)
            dw = dw * scale_factor
            
        # Clip dy: Prevent Y from getting kicked into orbit
        max_dy_norm = 0.5 
        if norm_dy > max_dy_norm:
            scale_factor = max_dy_norm / (norm_dy + 1e-8)
            dy = dy * scale_factor

        # 7) Apply update
        with torch.no_grad():
            self.cell.raw_a.add_(dw[0])
            self.cell.raw_b.add_(dw[1])
            self.Y.add_(dy)
            
        return L.item(), mse.item(), corr.item(), gw.norm().item(), gy.norm().item(), norm_dw.item(), norm_dy.item()

    def train(self, steps=200):
        print(f"{'Step':<5} | {'Loss':<10} | {'MSE':<10} | {'Corr':<10} | {'|GradW|':<10} | {'|GradY|':<10}| {'|normDW|':<10}| {'|normDY|':<10}")
        print("-" * 95)
        
        for i in range(steps):
            L, mse, corr, nGw, nGy, ndw, ndy = self.update_step()
            
            if i % 10 == 0:
                print(f"{i:<5d} | {L:.6f}   | {mse:.6f}   | {corr:.6f}   | {nGw:.4f}     | {nGy:.4f}     | {ndw:.4f}     | {ndy:.7f}")
        print("\n--- Final Results ---")
        print(f"Final MSE:{mse:.5f}")

# 3. Main
if __name__ == "__main__":
    
    # 1) Generate Kalman data
    T = 3000
    alpha = 0.9
    beta = 1.0
    tau_x = 0.4
    tau_z = 0.4
    Xsim, Ztrue = gd.generate_kalman_data(T, alpha, beta, tau_x, tau_z)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
    
    # 2) Set up solver
    cell = LinearCell()
    lmbda = 1.0
    eta_z = T * 0.005
    eta_y = T * 0.01
    split = False
    solver = HessianSolver(X, cell, lmbda=lmbda, eta_z=eta_z, eta_y=eta_y, split=split)
    print(" ")
    print(f"Starting training with parameters Lambda: {lmbda:.4f}, eta_z: {eta_z:.2f}, eta_y: {eta_y:.2f}")
    print(" ")
    solver.train(steps=100)
    
    # 3) Final analysis
    with torch.no_grad():

        Y_final = solver.Y
        
        P = gd.kalman_predictors(X.detach().numpy(), alpha, beta, tau_x, tau_z)
        True_Noise = X.numpy() - P
        
        err = ((Y_final.numpy() - True_Noise)**2).mean()
        true_var = (True_Noise**2).mean()
        

        print(f"Equilibrium value |Pred|^2: {(P**2).mean():.6f}")
        print(f"Squared Relative Error (|Y - Y*|^2 / |Y*|^2): {err/true_var:.6f}")
        
        final_a = 1.0 * torch.tanh(solver.cell.raw_a)
        final_b = torch.tanh(solver.cell.raw_b)
        print(f"Learned Params: a={final_a.item():.4f}, b={final_b.item():.4f}")