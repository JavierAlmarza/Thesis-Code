import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import GenerateData as gd

class MaxMinMaxSolver:
    def __init__(self, X, burn_in=50, lr_Y=1e-3, lr_ab=1e-3, n_inner_Y = 50, lr_lambda=1e-4, device='cpu'):
        """
        Args:
            X: torch.Tensor of shape [T], the observed process
            burn_in: int, number of steps to ignore for Z_t convergence
            lr_Y, lr_ab, lr_lambda: learning rates for inner min, outer max variables
            device: 'cpu' or 'cuda'
        """
        self.X = X.to(device)
        self.T = X.shape[0]
        self.t0 = burn_in
        self.device = device
        self.n_inner_Y = n_inner_Y
        self.n_inner_Z = 2

        
        # Initialize variables
        self.Y = torch.zeros(self.T, requires_grad=True, device=device)
        self.raw_b = torch.randn((), requires_grad=True, device=device)
        self.raw_a = torch.randn((), requires_grad=True, device=device)
        self.raw_lambda = torch.randn((), requires_grad=True, device=device)

                    
        # Separate optimizers
        self.opt_Y = torch.optim.Adam([self.Y], lr=lr_Y)
        self.opt_ab = torch.optim.Adam([self.raw_a, self.raw_b], lr=lr_ab)
        self.opt_lambda = torch.optim.Adam([self.raw_lambda], lr=lr_lambda)
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.7,0.7,0.7,0.7)
        self.P = P
        self.dim = self.T + 3
        self.B = torch.diag(
            torch.cat([
                torch.ones(self.T),
                -torch.ones(3)
            ])
        ).to(self.device)

        self.eta = 1e-2
        self.eta_max = 3e-2
        self.gamma = 0.8
        self.eps = 1e-3
        self.beta = 0.1


    def get_a(self):
        return 0.5 * torch.tanh(self.raw_a)

    def get_b(self):
        return torch.tanh(self.raw_b)

    def get_lambda(self):
        return 125.0 * torch.tanh(self.raw_lambda)

        
    def compute_Z(self, a, b):
        """Compute Z_t recursively without in-place operations."""
        Z_list = [torch.tensor(0.0, device=self.device)]  # initial Z_0
        for t in range(self.T - 1):
            Z_next = a * self.X[t] + b * Z_list[-1]
            Z_list.append(Z_next)
        Z = torch.stack(Z_list)  # shape [T]
        return Z
        
    def loss_and_grad(self, u):
        Y = u[:self.T]
        a, b, lam = u[self.T:]

        Z = self.compute_Z(a, b)
        Zm = Z[self.t0:].mean()

        loss = ((self.X[self.t0:] - Y[self.t0:])**2).mean() \
               + lam * ((Z[self.t0:] - Zm) * Y[self.t0:]).mean()

        G = torch.autograd.grad(loss, u, create_graph=False)[0]
        return loss.detach(), G.detach()

    def hessian_matrix(self, u):
        """
        Explicit Hessian of loss w.r.t. packed variable u.
        Size: (T+3) x (T+3)
        """
        u = u.detach().requires_grad_(True)
        loss = self.loss_full(u[:self.T], *u[self.T:])
        grad = torch.autograd.grad(loss, u, create_graph=True)[0]

        H_rows = []
        for i in range(len(u)):
            Hi = torch.autograd.grad(grad[i], u, retain_graph=True)[0]
            H_rows.append(Hi)

        H = torch.stack(H_rows)
        return H.detach()


    def qitd_step(self):
        T0 = self.T - self.t0
        c = 2.0 / T0

        # pack
        Y = self.Y
        a = self.get_a()
        b = self.get_b()
        lam = self.get_lambda()

        # Z and sensitivities
        Z = self.compute_Z(a, b)
        Zc = Z[self.t0:] - Z[self.t0:].mean()

        dZa = torch.zeros_like(Z)
        dZb = torch.zeros_like(Z)
        for t in range(1, self.T):
            dZa[t] = self.X[t-1] + b * dZa[t-1]
            dZb[t] = Z[t-1] + b * dZb[t-1]

        dZa = dZa[self.t0:]
        dZb = dZb[self.t0:]

        # gradients
        GY = (-2 * (self.X[self.t0:] - Y[self.t0:]) + lam * Zc) / T0
        Ga = lam * torch.dot(dZa, Y[self.t0:]) / T0
        Gb = lam * torch.dot(dZb, Y[self.t0:]) / T0
        Glam = torch.dot(Zc, Y[self.t0:]) / T0

        Gz = torch.stack([Ga, Gb, Glam])

        # cross Hessian K
        Ka = lam * dZa / T0
        Kb = lam * dZb / T0
        Klam = Zc / T0
        K = torch.stack([Ka, Kb, Klam], dim=1)  # T0 x 3

        # small Hessian
        M = torch.zeros(3, 3, device=self.device)
        M[0,2] = torch.dot(dZa, Y[self.t0:]) / T0
        M[1,2] = torch.dot(dZb, Y[self.t0:]) / T0
        M[2,0] = M[0,2]
        M[2,1] = M[1,2]

        # Schur complement
        A = (-torch.eye(3, device=self.device)
             + self.eta * M
             - (self.eta**2 / (1 + self.eta * c)) * (K.T @ K))

        rhs = (Gz
               - (self.eta / (1 + self.eta * c)) * (K.T @ GY))

        Dz = torch.linalg.solve(A, rhs)

        DY = (GY - self.eta * (K @ Dz)) / (1 + self.eta * c)

        # update
        with torch.no_grad():
            self.Y[self.t0:] -= DY
            self.raw_a.copy_(torch.atanh(2 * (a - Dz[0])))
            self.raw_b.copy_(torch.atanh(b - Dz[1]))
            self.raw_lambda.copy_(torch.atanh((lam - Dz[2]) / 125.0))
    
    def step(self):
        self.qitd_step()

    def apply_J(self, v):
        vJ = v.clone()
        vJ[:self.T] *= +1      # Y minimizes
        vJ[self.T:] *= -1      # a,b,lambda maximize
        return vJ
        
    def loss_full(self, Y, a, b, lam):
        Z = self.compute_Z(a, b)
        Zm = Z[self.t0:].mean()
        return ((self.X[self.t0:] - Y[self.t0:])**2).mean() \
               + lam * ((Z[self.t0:] - Zm) * Y[self.t0:]).mean()


    
    def train(self, num_steps=1000, verbose=False):
        """Run optimization for a number of steps."""
        for step in range(num_steps):
            self.step()
            if verbose and (step % 50 == 0):
                a = self.get_a()
                b = self.get_b()
                lambda_ = self.get_lambda()
                Z = self.compute_Z(a, b)
                Z_mean = Z[self.t0:].mean()
                current_loss = ((self.X[self.t0:] - self.Y[self.t0:]) ** 2).mean() + \
                               lambda_ * ((Z[self.t0:] - Z_mean) * self.Y[self.t0:]).mean()
                Diff =  ((self.X[self.t0:].detach().numpy().reshape(-1) - self.Y[self.t0:].detach().numpy().reshape(-1) - self.P[self.t0:])**2).mean()
                with torch.no_grad():                    
                    Z = self.compute_Z(a, b)
                    Zc = Z[self.t0:] - Z[self.t0:].mean()
                    print("corr(Z,X) =", torch.dot(Zc, self.X[self.t0:]) /
                                         (Zc.norm() * self.X[self.t0:].norm()))

                print(f"Step {step:4d}: Loss={current_loss.item():.6f}, "
                      f"a={a.item():.4f}, b={b.item():.4f}, "
                      f"lambda={lambda_.item():.4f}, "
                      f"|Y-Pred|^2 ={Diff.item():.4f}, "
                      f"coupling={lambda_ * ((Z[self.t0:] - Z_mean) * self.Y[self.t0:]).mean().item():.4f}")
    
    def get_solution(self):
        """Return current solution."""
        return {
            'Y': self.Y.detach().clone(),
            'a': self.get_a().item(),
            'b': self.get_b().item(),
            'lambda': self. get_lambda().item(),
            'mse loss': ((self.X[self.t0:] - self.Y[self.t0:]) ** 2).mean().item()
        }

# ===== Example usage =====
if __name__ == "__main__":
    T = 2000
    burn_in = 500
    # Simulate a stationary Gaussian process
    #X = torch.randn(T)
    Xsim, Ztrue = gd.generate_kalman_data(T,0.7,0.7,0.7,0.7)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
    
    solver = MaxMinMaxSolver(X, burn_in=burn_in, lr_Y=1e-2, lr_ab=1e-2, lr_lambda=1e-3,n_inner_Y = 50)
    solver.train(num_steps=10000, verbose=True)
    
    solution = solver.get_solution()
    print(solution)
